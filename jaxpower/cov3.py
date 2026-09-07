import functools
import itertools
import os
import warnings

import numpy as np
import jax
from jax import numpy as jnp

from .mesh import MeshAttrs, split_particles
from .mesh2 import get_smooth2_window_bin_attrs
from .mesh3 import BinMesh3CorrelationPoles, compute_mesh3, FKPField, get_sugiyama_window_convolution_coeffs, get_smooth3_window_bin_attrs
from .types import CovarianceMatrix, ObservableTree
from .utils import wigner_3j, wigner_9j, get_legendre, legendre_product
from .pt import integration, IntegralND, get_S
from .cov2 import Correlation2Spectrum, compute_spectrum2_covariance_window_block, matrix_rebin


@functools.lru_cache(maxsize=None)
def get_sugiyama_covariance_window_convolution_coeffs(ell, ellin):
    r"""Window-multipole coefficients for the *covariance* 4-point kernel.

    ``ell`` = (L1, L2, J) indexes the unprimed-side S-basis channel and
    ``ellin`` = (L1', L2', J') the primed-side one (both z3, M = 0). Returns
    the list of (q, coeff) such that the 4-point angular window kernel is

    .. math::
        Q_W(k_1, k_1', k_2, k_2')
        = \sum_{\ell,\ell'} \Big[\sum_q c_q\, \mathrm{Hankel}_{L_1 L_1' L_2 L_2'}[Q_{W,q}]\Big]
          S_\ell(\hat k_1, \hat k_2)\, S_{\ell'}(\hat k_1', \hat k_2'),

    following the :math:`\mathcal C^{\lambda_1\lambda_2\Lambda}_{L_1L_1'L_2L_2'}`
    kernel of ``_cov3_math.tex``, keeping only its N = 0 term (the N != 0
    azimuthal channels vanish identically under the estimators' independent
    per-side orientation averages). This differs from
    :func:`get_sugiyama_window_convolution_coeffs` (the *mean* bispectrum
    window convolution, eq. 63 of arXiv:1803.02132): here the monopole
    window feeds each diagonal channel with the Parseval weight
    :math:`(2L_1+1)(2L_2+1)(2J+1) H_{L_1L_2J}^2 = 1 / \| S_\ell \|^2`.

    The relative phase :math:`(-i)^{L_1+L_2} i^{L_1'+L_2'}` is real
    (:math:`\pm 1`) for every allowed q: the two triangle conditions with
    even-sum q force :math:`(L_1'-L_1)+(L_2'-L_2)` even. It is included here
    because the covariance Hankel matrices drop the transforms'
    :math:`i^\ell` prefactors. Normalization is anchored so that the
    ((0,0,0), (0,0,0)) channel has coeff((0,0,0)) = 1, matching the box
    limit.
    """
    L1, L2, J = ell
    L1p, L2p, Jp = ellin
    HJ = wigner_3j(L1, L2, J, 0, 0, 0)
    HJp = wigner_3j(L1p, L2p, Jp, 0, 0, 0)
    if abs(HJ) < 1e-12 or abs(HJp) < 1e-12:
        return []
    coeffs = []
    for q in itertools.product(range(L1 + L1p + 1), range(L2 + L2p + 1), range(abs(J - Jp), J + Jp + 1)):
        if sum(q) % 2 or q[2] % 2:
            continue
        Hq = wigner_3j(*q, 0, 0, 0)
        if abs(Hq) < 1e-12:
            continue
        coeff = (2 * L1 + 1) * (2 * L1p + 1) * (2 * L2 + 1) * (2 * L2p + 1)
        coeff *= wigner_3j(q[0], L1, L1p, 0, 0, 0) * wigner_3j(q[1], L2, L2p, 0, 0, 0) / Hq
        coeff *= (2 * J + 1) * (2 * Jp + 1) * HJ * HJp
        coeff *= wigner_9j(L1, L2, J, L1p, L2p, Jp, *q)
        coeff *= wigner_3j(J, Jp, q[2], 0, 0, 0)
        if abs(coeff) < 1e-10:
            continue
        coeff *= (-1) ** (((L1p - L1 + L2p - L2) // 2) % 2)
        coeffs.append((tuple(q), coeff))
    return coeffs


def unitvec(mu, phi):
    s = jnp.sqrt(jnp.clip(1. - mu**2, 0., None))
    return jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), mu], axis=-1)


def get_kvec1(knorm, mu):
    khat = unitvec(mu, jnp.zeros_like(mu))
    kvec = knorm[..., None] * khat
    return knorm, khat, kvec


def get_kvec3(k1norm, k2norm, mu1, mu2, phi2):
    k1hat = unitvec(mu1, jnp.zeros_like(mu1))
    k2hat = unitvec(mu2, phi2)
    k1vec = k1norm[..., None] * k1hat
    k2vec = k2norm[..., None] * k2hat
    k3vec = -k1vec - k2vec
    k3norm = jnp.sqrt(jnp.sum(k3vec**2, axis=-1))
    # Guard the exactly-folded configuration (k2 = -k1, e.g. equal-k
    # sugiyama-diagonal bins when a quadrature node lands on mu1 = mu2 = 0,
    # phi2 = pi, as odd-size Gauss-Legendre grids do): k3 = 0 there, a valid
    # (measure-zero) triangle, not an error -- avoid 0/0 -> NaN in k3hat.
    k3hat = k3vec / jnp.where(k3norm == 0., 1., k3norm)[..., None]
    return (k1norm, k2norm, k3norm), (k1hat, k2hat, k3hat), (k1vec, k2vec, k3vec)


def get_kvec3_x(k1norm, k2norm, mu1, x, phi):
    """Triangle from (mu1, x, phi) instead of (mu1, mu2, phi2).

    `x = k1hat . k2hat` is the triangle shape, carried here as a direct integration variable;
    `phi` is the azimuth of k2hat about k1hat. The solid-angle element is unchanged,
    `dmu2 dphi2 = dx dphi`, so the quadrature weights and the normalization are identical to
    :func:`get_kvec3` -- only the node placement differs.

    Why it matters: the integrand's sharp structure lives in `k3 = |k1 + k2|`, which depends on
    the shape *only* through `x` (`k3^2 = k1^2 + k2^2 + 2 k1 k2 x`), and is most singular at the
    folded configuration `k1 ~ -k2`. Sampling `x` directly resolves it; deriving it as
    `x = mu1 mu2 + sqrt(1-mu1^2) sqrt(1-mu2^2) cos(phi2)` smears that structure across all three
    variables and is what limits convergence. This is the parameterization
    :class:`jaxpower.pt.ProjectToSell` already uses (and FOLPS's `Sugiyama_Bell`), where the
    same substitution was measured to be worth 10% on B000 and 11% on B202 at size 6.
    """
    smu = jnp.sqrt(jnp.clip(1. - mu1**2, 0., None))
    zero, one = jnp.zeros_like(mu1), jnp.ones_like(mu1)
    k1hat = jnp.stack([smu, zero, mu1], axis=-1)
    # Orthonormal frame about k1hat, with e3 perpendicular to the line of sight.
    e2 = jnp.stack([-mu1, zero, smu], axis=-1)
    e3 = jnp.stack([zero, -one, zero], axis=-1)
    sx = jnp.sqrt(jnp.clip(1. - x**2, 0., None))
    # Explicit trailing axes: under vmap (the grid path) x and phi are scalars and plain
    # products broadcast, but called directly on (..., ) arrays against (..., 3) vectors they
    # must be lifted -- the exact-x path does exactly that.
    k2hat = (x[..., None] * k1hat
             + sx[..., None] * (jnp.cos(phi)[..., None] * e2 + jnp.sin(phi)[..., None] * e3))
    k1vec = k1norm[..., None] * k1hat
    k2vec = k2norm[..., None] * k2hat
    k3vec = -k1vec - k2vec
    k3norm = jnp.sqrt(jnp.sum(k3vec**2, axis=-1))
    k3hat = k3vec / jnp.where(k3norm == 0., 1., k3norm)[..., None]
    return (k1norm, k2norm, k3norm), (k1hat, k2hat, k3hat), (k1vec, k2vec, k3vec)


def get_kvec5(k1norm, k2norm, k2pnorm, mu1, mu2, phi2, mu2p, phi2p):
    k1hat = unitvec(mu1, jnp.zeros_like(mu1))
    k2hat = unitvec(mu2, phi2)
    k2phat = unitvec(mu2p, phi2p)

    k1vec = k1norm[..., None] * k1hat
    k2vec = k2norm[..., None] * k2hat
    k2pvec = k2pnorm[..., None] * k2phat

    k3vec = -k1vec - k2vec
    k3pvec = -k1vec - k2pvec

    k3norm = jnp.sqrt(jnp.sum(k3vec**2, axis=-1))
    k3pnorm = jnp.sqrt(jnp.sum(k3pvec**2, axis=-1))

    k3hat = k3vec / k3norm[..., None]
    k3phat = k3pvec / k3pnorm[..., None]

    return (k1norm, k2norm, k3norm, k2pnorm, k3pnorm), \
           (k1hat, k2hat, k3hat, k2phat, k3phat), \
           (k1vec, k2vec, k3vec, k2pvec, k3pvec)


def compute_fkp2_covariance_window(fkps, bin=None, los="local", fields=None, split=None,
                                   group_sizes=(2, 3, 4), max_total_size=6,
                                   group_pairs=None, **kwargs):
    r"""
    Compute the two-anchor covariance-window multipoles used by ``compute_QW_AB``.

    Unlike a plain power-spectrum window, this routine keeps the two window
    factors as grouped field labels.  This is required by the mixed covariance
    terms in the LaTeX formulae, e.g.

        Q_W^{(ac)(bde)}       for PB,
        Q_W^{(bcap)(bpcpa)}   for BB,
        Q_W^{(aap)(bcbpcp)}   for PT.

    Parameters
    ----------
    group_sizes : tuple, default=(2, 3, 4)
        Field multiplicities allowed for each of the two window factors.
    max_total_size : int, default=6
        Maximum total number of elementary FKP fields in the two factors.
    group_pairs : list, optional
        Explicit list of pairs of grouped field labels.  Each element is
        ``(fields1, fields2)``.  When omitted, all combinations with replacement
        of the requested ``group_sizes`` are generated, restricted by
        ``max_total_size``.

    Notes
    -----
    The returned ``ObservableTree`` stores grouped field labels
    ``((...), (...))``.  ``compute_QW_AB`` also understands older flat labels,
    but grouped labels avoid ambiguities such as distinguishing
    ``(ab,cde)`` from ``(abc,de)``.
    """
    if not isinstance(fkps, (tuple, list)):
        fkps = [fkps]

    if fields is None:
        fields = list(range(len(fkps)))

    fkps = {field: fkp for field, fkp in zip(fields, fkps, strict=True)}
    field_values = tuple(fields)

    try:
        from .mesh2 import BinMesh2CorrelationPoles, compute_mesh2
    except ImportError:
        from .mesh import BinMesh2CorrelationPoles, compute_mesh2

    if bin is None:
        mattrs = next(iter(fkps.values())).attrs
        kw = {"edges": None, "ells": None, "basis": None, "klimit": None, "batch_size": None}
        kw = kw | get_smooth2_window_bin_attrs([0, 2, 4], ellsin=2)

        for name in kw:
            kw[name] = kwargs.pop(name, kw[name])

        edges = kw.pop("edges")
        if edges is None:
            edges = {}

        bin = BinMesh2CorrelationPoles(mattrs, edges=edges, **kw)

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    def get_W(fkp, mask=None):
        randoms = get_randoms(fkp)

        if mask is not None:
            randoms = randoms.clone(weights=randoms.weights * mask)

        alpha = fkp.data.weights.sum() / randoms.weights.sum() if isinstance(fkp, FKPField) else 1.0
        mesh = randoms.paint(**kwargs, out="real")

        return alpha * mesh / mesh.cellsize.prod()

    def canonical_group(group):
        return tuple(sorted(tuple(group)))

    if group_pairs is None:
        groups_by_size = {
            size: tuple(itertools.combinations_with_replacement(field_values, size))
            for size in group_sizes
        }
        group_pairs = []
        for size1 in group_sizes:
            for size2 in group_sizes:
                if size1 > size2 or size1 + size2 > max_total_size:
                    continue
                for group1 in groups_by_size[size1]:
                    for group2 in groups_by_size[size2]:
                        group_pairs.append((group1, group2))
    else:
        group_pairs = [(tuple(group1), tuple(group2)) for group1, group2 in group_pairs]

    splits = None
    if split is not None:
        if not isinstance(split, list):
            split = [split]
        splits = {field: split for field, split in zip(fields, split, strict=True)}

    windows = {}
    compute_mesh2_window = jax.jit(compute_mesh2, static_argnames=["los"])

    for group1, group2 in group_pairs:
        group1, group2 = canonical_group(group1), canonical_group(group2)
        wfield = (group1, group2)

        if wfield in windows:
            continue

        flat_field = group1 + group2
        _fkps = [fkps[field] for field in flat_field]
        masks = [None] * len(_fkps)

        if split is not None:
            seed = list({field: splits[field] for field in flat_field}.values())
            masks = split_particles(*[get_randoms(fkp) for fkp in _fkps], seed=seed, fields=list(flat_field), return_masks=True)

        n1 = len(group1)
        WA = get_W(_fkps[0], mask=masks[0])
        for fkp, mask in zip(_fkps[1:n1], masks[1:n1], strict=True):
            WA = WA * get_W(fkp, mask=mask)

        WB = get_W(_fkps[n1], mask=masks[n1])
        for fkp, mask in zip(_fkps[n1 + 1:], masks[n1 + 1:], strict=True):
            WB = WB * get_W(fkp, mask=mask)

        # Normalization: cross-pair group1's anchor field (fkps[0]) with
        # group1's remaining n1-1 fields with the LAST fields of group2, not
        # the naive product of each group's own fields (WA.sum()*WB.sum()).
        Ws = [get_W(fkp, mask=mask) for fkp, mask in zip(_fkps, masks, strict=True)]
        normA = Ws[0]
        for w in Ws[n1:2 * n1 - 1]:
            normA = normA * w
        normB = None
        for w in Ws[1:n1] + Ws[2 * n1 - 1:]:
            normB = w if normB is None else normB * w
        if normB is None:
            normB = jnp.ones_like(normA)
        norm = normA.sum() * normB.sum()
        update = dict(norm=[norm * jnp.ones_like(bin.xavg)] * len(bin.ells))
        windows[wfield] = compute_mesh2_window(WA, WB, bin=bin, los=los).clone(**update)
        del WA, WB

    return ObservableTree(
        list(windows.values()),
        fields1=[field_groups[0] for field_groups in windows],
        fields2=[field_groups[1] for field_groups in windows],
    )


def compute_fkp3_covariance_window(fkps, bin=None, los="local", fields=None, split=None, **kwargs):
    r"""
    Compute the WWW 3-point covariance-window multipoles

        Q^{ABC}_{W,lambda1 lambda2 Lambda}(s1, s2)

    in the Sugiyama / TripoSH basis. Here A, B, C are field-pair labels, e.g.

        A = (a, a')
        B = (b, b')
        C = (c, c')

    and the configuration-space window is schematically

        Q_W^{ABC}(s1, s2) ~ < W_A(x) W_B(x + s1) W_C(x + s2) >.

    This function only computes the WWW piece.
    """
    if not isinstance(fkps, (tuple, list)):
        fkps = [fkps]

    if fields is None:
        fields = list(range(len(fkps)))

    fkps = {field: fkp for field, fkp in zip(fields, fkps, strict=True)}

    if bin is None:
        mattrs = next(iter(fkps.values())).attrs
        kw = {"edges": None, "ells": None, "basis": 'sugiyama', "klimit": None, "batch_size": None, 'buffer_size': 0}
        kw = kw | get_smooth3_window_bin_attrs([(0, 0, 0), (2, 0, 2)], ellsin=2)

        for name in kw:
            kw[name] = kwargs.pop(name, kw[name])

        edges = kw.pop("edges")
        if edges is None:
            edges = {}

        bin = BinMesh3CorrelationPoles(mattrs, edges=edges, **kw)

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    def get_W(fkp, mask=None):
        randoms = get_randoms(fkp)

        if mask is not None:
            randoms = randoms.clone(weights=randoms.weights * mask)

        alpha = fkp.data.weights.sum() / randoms.weights.sum() if isinstance(fkp, FKPField) else 1.0
        mesh = randoms.paint(**kwargs, out="real")

        return alpha * mesh / mesh.cellsize.prod()

    splits = None
    if split is not None:
        if not isinstance(split, list):
            split = [split]
        splits = {field: split for field, split in zip(fields, split, strict=True)}

    pairs = tuple(itertools.combinations_with_replacement(tuple(fields), 2))
    # Q_W^{ABC}(s1, s2) ~ <W_A(x) W_B(x+s1) W_C(x+s2)>: positions 1 and 2 (the
    # two "arm" separations s1, s2) are interchangeable (Q_W^{ABC} = Q_W^{BAC}),
    # but position 3 is not (Q_W^{ABC} != Q_W^{ACB} in general) -- so only the
    # first two groups are drawn from an unordered (sorted) pair-of-pairs;
    # the third is chosen independently, not from a fully symmetric 3-way
    # combinations_with_replacement.
    pairs_ab = tuple(itertools.combinations_with_replacement(pairs, 2))
    triplets = tuple((pab[0], pab[1], pc) for pab in pairs_ab for pc in pairs)
    windows = {}
    compute_mesh3_window = jax.jit(compute_mesh3, static_argnames=["los"])

    for triple in triplets:
        wfield = tuple(tuple(sorted(pair)) for pair in triple)
        flat_field = sum(wfield, start=tuple())

        if wfield in windows:
            continue

        _fkps = [fkps[field] for field in flat_field]
        masks = [None] * len(_fkps)

        if split is not None:
            seed = list({field: splits[field] for field in flat_field}.values())
            masks = split_particles(*[get_randoms(fkp) for fkp in _fkps], seed=seed, fields=list(flat_field), return_masks=True)

        Ws = [get_W(_fkps[i], mask=masks[i]) for i in range(6)]
        WA, WB, WC = Ws[0] * Ws[1], Ws[2] * Ws[3], Ws[4] * Ws[5]

        # Normalization: the product of the two bispectrum-estimator
        # normalizations, int(n_a n_b n_c) x int(n_a' n_b' n_c') (NOT the
        # product of the three pair integrals int(n n') as for the 2-point
        # covariance window): the window's periodic limit is then
        # (2 pi)^6 delta_D delta_D / V, matching the Sugiyama PPP covariance
        # with no extra volume factor in the assembly. The 1/cellsize makes
        # the norm's total cell-volume power (one per anchor, three) match
        # the numerator (compute_mesh3) conventions: two mesh sums only
        # carry two.
        norm = (Ws[0] * Ws[2] * Ws[4]).sum() * (Ws[1] * Ws[3] * Ws[5]).sum() / WA.cellsize.prod()
        del Ws
        update = dict(norm=[norm * jnp.ones_like(bin.xavg[..., 0])] * len(bin.ells))
        windows[wfield] = compute_mesh3_window(WA, WB, WC, bin=bin, los=los).clone(**update)
        del WA, WB, WC

    return ObservableTree(
        list(windows.values()),
        fields1=[field_groups[0] for field_groups in windows],
        fields2=[field_groups[1] for field_groups in windows],
        fields3=[field_groups[2] for field_groups in windows],
    )


def _hankel_matrix(s, ell, cache=None):
    """Explicit ``(n_k, n_s)`` matrix of ``CorrelationToSpectrum(s, ell=ell)``,
    extracted via ``jax.jacfwd`` (same technique ``CorrelationToSpectrum``
    uses, including its prefactor/postfactor handling: the Jacobian is taken
    of the *raw* FFTlog transform with ``ignore_prepostfactor=True``, and the
    physical normalization is re-applied explicitly afterward as a
    per-row scale, since it factorizes as ``scale(k) * scale(k')`` and must
    be split symmetrically between the two sides of a bilinear sandwich).

    Unlike composing two independently-built FFTlog round-trip objects (one
    ``SpectrumToCorrelation``, one ``CorrelationToSpectrum``, which are only
    approximate *inverses* of each other, not *transposes*), sandwiching a
    window between two explicit forward-transform matrices built this way --
    one per side (``ell``/``ellin``) -- gives a bilinear form that is
    symmetric under side-swap by construction.

    Returns ``(k, matrix)``.
    """
    from .fftlog import CorrelationToSpectrum

    cache = {} if cache is None else cache
    key = (tuple(np.ravel(s)), ell)
    if key not in cache:
        fftlog = CorrelationToSpectrum(s=s, ell=ell, lowring=False, minfolds=0, check_level=1).fftlog

        def fwd(fun):
            return fftlog(fun, extrap=False, ignore_prepostfactor=True)[1]

        raw_matrix = jax.jacfwd(fwd)(jnp.zeros_like(jnp.asarray(s)))
        k = fftlog.y
        dlnk = jnp.diff(jnp.log(k)).mean()
        scale = jnp.sqrt(2 * np.pi**2 / dlnk) / k**1.5
        cache[key] = (k, scale[:, None] * raw_matrix)
    return cache[key]


def compute_spectrum3_covariance_window_block(window3, kedges, kpedges,
                                              ell, ellin,
                                              fields1=None, fields2=None, fields3=None,
                                              cache=None, batch_size=None,
                                              paxes=(0, 1), kp_points=None, kp_measure=None):
    """
    Compute one smooth covariance-window block W_{ell,ellin}(k1,k2;k1',k2').

    The unprimed radial axes always carry the (binned) k1, k2 legs. The
    primed radial axes are selected by ``paxes``: entry 0 or 1 rebins over
    the corresponding ``kpedges`` column (which primed *binned* leg pairs
    with each separation axis depends on the Wick permutation), while entry
    2 point-evaluates that axis at the primed *closure*-leg magnitudes
    ``kp_points`` (shape ``(nnodes, nbinsp)``, node- and bin-dependent, as
    in the box path's exact substitution) -- or, when ``kp_measure`` (a
    callable ``measure(fk) -> (nnodes * nbinsp, n_fk)`` exact-phi-measure
    weight matrix, see ``_closure_measure_matrix``) is given,
    *cell-averages* the transform over each node's own closure sweep, so
    window structure narrower than the node spacing cannot slip between nodes.
    With a points axis the result has shape ``(n, npx, nnodes)`` instead of
    ``(n, npx)``.
    """
    if cache is None:
        cache = {}
    rebin_cache = cache.setdefault("rebin_matrix", [])
    hankel_cache = cache.setdefault("hankel_matrix", {})

    def get_window_field(window, fields1, fields2, fields3):
        if fields1 is None:
            return window
        return window.get(fields1=fields1, fields2=fields2, fields3=fields3)

    def get_w_rect(q):
        transpose = False
        if q not in window3.ells:
            qswap = q[1::-1] + q[2:]
            if qswap in window3.ells:
                q = qswap
                transpose = True
            else:
                return jnp.zeros(())
        value = window3.get(q).value().real

        if transpose:
            value = jnp.swapaxes(value, 0, 1)

        return value

    def normalize_edges(edges):
        edges = np.asarray(edges)

        if edges.ndim == 1:
            edges = np.column_stack([edges[:-1], edges[1:]])

        return edges

    def unravel_edges(edges):
        """
        Convert input edge specification into a PAIRED-bin array of shape
        (nbins, 2, 2): one (k1, k2) edge pair per bispectrum bin. This
        supports both genuinely paired binnings (e.g. the sugiyama-diagonal
        basis, where bins are a *list* of (k1, k2) pairs, NOT a product
        grid -- reinterpreting them as a sqrt(n) x sqrt(n) product grid
        scrambles the bins) and product grids (tuple of 1D edges, or an
        explicit (N1, N2, 2, 2) array, flattened row-major).
        """
        if isinstance(edges, (tuple, list)) and len(edges) == 2:
            k1edges, k2edges = map(normalize_edges, edges)
            out = np.empty((len(k1edges), len(k2edges), 2, 2), dtype=float)
            out[:, :, 0, :] = k1edges[:, None, :]
            out[:, :, 1, :] = k2edges[None, :, :]
            return out.reshape(-1, 2, 2)

        edges = np.asarray(edges)

        if edges.ndim == 4 and edges.shape[-2:] == (2, 2):
            return edges.reshape(-1, 2, 2)

        if edges.ndim == 3 and edges.shape[-2:] == (2, 2):
            return edges

        raise ValueError("Invalid edge specification.")

    edges = unravel_edges(kedges)
    edgesp = unravel_edges(kpedges)

    n, npx = edges.shape[0], edgesp.shape[0]

    window3 = get_window_field(window3, fields1, fields2, fields3)

    # Covariance-kernel coefficients (the C^{lambda}_{L1L1'L2L2'} pairing of
    # _cov3_math.tex, N = 0) -- NOT the mean-convolution coefficients of
    # get_sugiyama_window_convolution_coeffs: the monopole window must feed
    # each diagonal (ell, ell) channel with the Parseval weight 1/||S_ell||^2.
    wcoeffs = get_sugiyama_covariance_window_convolution_coeffs(ell, ellin)

    if not wcoeffs:
        return np.zeros((n, npx))

    Qs = sum(coeff * get_w_rect(q) for q, coeff in wcoeffs)

    if np.ndim(Qs) == 0:
        return np.zeros((n, npx))

    s = tuple(next(iter(window3)).coords().values())

    k1_fftlog, H1 = _hankel_matrix(s[0], ell[0], cache=hankel_cache)
    k2_fftlog, H2 = _hankel_matrix(s[1], ell[1], cache=hankel_cache)
    k1p_fftlog, H1p = _hankel_matrix(s[0], ellin[0], cache=hankel_cache)
    k2p_fftlog, H2p = _hankel_matrix(s[1], ellin[1], cache=hankel_cache)

    interp_order = 3
    Mk1 = matrix_rebin(edges[:, 0, :], k1_fftlog, wt=k1_fftlog**2, interp_order=interp_order, cache=rebin_cache)
    Mk2 = matrix_rebin(edges[:, 1, :], k2_fftlog, wt=k2_fftlog**2, interp_order=interp_order, cache=rebin_cache)

    # Fuse the bin-rebinning into the forward-transform matrices before
    # contracting against Qs, so the (n_s1, n_s2) ~ 1000x1000 grid is never
    # expanded into a dense 4D tensor. Rows/columns are paired bins: bin a
    # applies its own k1-rebin on the s1 axis AND its own k2-rebin on the s2
    # axis.
    R1, R2 = Mk1 @ H1, Mk2 @ H2

    def primed_R(axis, kp_fftlog, Hp):
        spec = paxes[axis]
        if spec == 2:
            if kp_measure is not None:
                # Closure leg, cell-averaged with the exact phi measure over
                # each node's own closure sweep instead of sampled at the
                # node magnitude (see _closure_measure_matrix).
                avg = jnp.asarray(kp_measure(np.asarray(kp_fftlog))) @ Hp
                return jnp.reshape(avg, (np.shape(kp_points)[0], npx, -1))  # (n_nodes, npx, n_s)
            # Closure leg: point-evaluate the transform at the node- and
            # bin-dependent magnitudes (n_nodes, npx), as in the box path.
            pts = jnp.reshape(jnp.asarray(kp_points), (-1,))
            # Clamp to the FFTlog grid: k3' -> 0 at antiparallel quadrature
            # nodes, and polynomial spline extrapolation there is unsafe.
            pts = jnp.clip(pts, jnp.min(kp_fftlog), jnp.max(kp_fftlog))
            interp = matrix_spline_interp(kp_fftlog, pts, interp_order, cache=hankel_cache)
            return jnp.reshape(interp @ Hp, (np.shape(kp_points)[0], npx, -1))  # (n_nodes, npx, n_s)
        Mkp = matrix_rebin(edgesp[:, spec, :], kp_fftlog, wt=kp_fftlog**2, interp_order=interp_order, cache=rebin_cache)
        return Mkp @ Hp

    R1p = primed_R(0, k1p_fftlog, H1p)
    R2p = primed_R(1, k2p_fftlog, H2p)

    if np.ndim(R1p) == 3:
        return jnp.einsum('as,at,st,nbs,bt->abn', R1, R2, Qs, R1p, R2p, optimize=True)
    if np.ndim(R2p) == 3:
        return jnp.einsum('as,at,st,bs,nbt->abn', R1, R2, Qs, R1p, R2p, optimize=True)
    return jnp.einsum('as,at,st,bs,bt->ab', R1, R2, Qs, R1p, R2p, optimize=True)



def matrix_spline_interp(xt, xo, interp_order, cache=None):
    """Like ``cov2.matrix_spline_interp(xt, xo, ...)``, but ``xo`` may be a
    traced (e.g. vmapped) array; ``xt`` must be concrete.

    If ``interpax`` is installed, this runs natively in JAX: no host
    callback, transparently batched under ``vmap`` (no special
    ``vmap_method`` needed), and differentiable. ``interpax``'s
    ``method='cubic2'`` (C2, natural-spline-like) is used because it matches
    scipy's ``make_interp_spline(k=3)`` (used by ``matrix_rebin`` elsewhere
    in this module) to ~1e-7 relative -- its plain ``'cubic'`` (C1, local
    splines) is a *different* scheme and differs by ~10%, so is not a
    drop-in match.

    Otherwise, falls back to scipy via ``jax.pure_callback``. The
    cubic-spline basis only depends on ``xt`` (not ``xo``), so it is built
    once and cached -- otherwise every vmapped call would refit it from
    scratch on a ``len(xt)``-sized identity matrix.
    """
    xt = jnp.asarray(xt)
    xo = jnp.asarray(xo)

    try:
        import interpax
    except ImportError:
        interpax = None

    if interpax is not None:
        method = 'linear' if interp_order == 1 else 'cubic2'
        return interpax.interp1d(xo, xt, jnp.eye(xt.shape[-1], dtype=xo.dtype), method=method)

    from scipy.interpolate import make_interp_spline

    spline_cache = {} if cache is None else cache
    key = (tuple(np.ravel(xt)), interp_order)
    if key not in spline_cache:
        xt_arr = np.asarray(xt, dtype=float)
        spline_cache[key] = make_interp_spline(xt_arr, np.eye(len(xt_arr)), k=interp_order, axis=0)
    spl = spline_cache[key]

    def host_fn(xo):
        return jnp.asarray(spl(np.asarray(xo)))

    out_shape = jax.ShapeDtypeStruct((xo.shape[-1], len(xt)), xo.dtype)
    # 'broadcast_all', not 'sequential': scipy's spline evaluation already
    # broadcasts over an extra leading batch axis, so under vmap (e.g. over
    # quadrature points) this calls the host once for the whole batch
    # instead of once per point -- ~50x faster in practice, same result.
    return jax.pure_callback(host_fn, out_shape, xo, vmap_method='broadcast_all')


def _cumtrapz0(x, table, axis=0):
    """Trapezoid cumulative integral of ``table`` along ``axis`` (sampled at
    ``x``), prepended with a zero slice so C[0] = 0 and C has the same length
    as ``table`` along ``axis``."""
    t = jnp.moveaxis(jnp.asarray(table), axis, 0)
    dx = jnp.diff(jnp.asarray(x)).reshape((-1,) + (1,) * (t.ndim - 1))
    mid = 0.5 * (t[1:] + t[:-1]) * dx
    C = jnp.concatenate([jnp.zeros_like(t[:1]), jnp.cumsum(mid, axis=0)], axis=0)
    return jnp.moveaxis(C, 0, axis)


def _interp_linear(x, table, xq, axis=0):
    """Linear interpolation of ``table`` (sampled at ``x`` along ``axis``) at
    query points ``xq`` (1D). Queries are clipped to the grid. Applied to a
    cumulative table (see :func:`_cumtrapz0`), this evaluates the exact
    integral of the piecewise-linear interpolant."""
    x = jnp.asarray(x)
    t = jnp.moveaxis(jnp.asarray(table), axis, 0)
    xq = jnp.clip(jnp.ravel(jnp.asarray(xq)), x[0], x[-1])
    idx = jnp.clip(jnp.searchsorted(x, xq, side='right') - 1, 0, x.shape[0] - 2)
    w = (xq - x[idx]) / (x[idx + 1] - x[idx])
    out = t[idx] + w.reshape((-1,) + (1,) * (t.ndim - 1)) * (t[idx + 1] - t[idx])
    return jnp.moveaxis(out, 0, axis)


def _closure_measure_matrix(fk, lo, hi, ksq, k1k2, c, w, chunk=2048):
    """Exact phi-measure weight matrix for closure-leg cell averages.

    For each output row (one quadrature node's phi cell x one bispectrum
    bin), the closure magnitude q3 sweeps [lo, hi]; the phi-measure of any
    q3 sub-segment is analytic: with mu12(q) = (q^2 - ksq) / (2 k1k2) and
    x = (mu12 - c) / w (c = mu1 mu2, w = s1 s2), dphi = -dmu12 / (w sin phi)
    integrates to |arccos(x_hi) - arccos(x_lo)|. Each row distributes the
    per-grid-segment measures onto the two segment-end grid nodes
    (trapezoid split) and is normalized to unit sum, so ``M @ table`` is the
    exact phi-measure average of the (piecewise-linear) table over each
    node's own cell -- the sharp direction is integrated with its true
    Jacobian weighting, never point-sampled. Degenerate rows (zero sweep,
    e.g. a closure turning point) fall back to linear interpolation at the
    midpoint, the consistent limit.

    Returns a dense ``(n_rows, n_fk)`` numpy array.
    """
    fk = np.asarray(fk)
    nfk = len(fk)
    lo, hi, ksq, k1k2, c, w = map(np.asarray, (lo, hi, ksq, k1k2, c, w))
    n = len(lo)
    out = np.zeros((n, nfk))

    def x_of(q, sl):
        mu12 = (q**2 - ksq[sl, None]) / (2. * k1k2[sl, None])
        return np.clip((mu12 - c[sl, None]) / np.maximum(w[sl, None], 1e-300), -1., 1.)

    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        m = out[sl].shape[0]
        # Segment [fk_j, fk_j+1] clipped to the row's sweep [lo, hi].
        seg_lo = np.maximum(fk[None, :-1], lo[sl, None])
        seg_hi = np.minimum(fk[None, 1:], hi[sl, None])
        valid = seg_hi > seg_lo
        seg_lo = np.where(valid, seg_lo, fk[None, :-1])
        seg_hi = np.where(valid, seg_hi, fk[None, :-1])
        dphi = np.abs(np.arccos(x_of(seg_hi, sl)) - np.arccos(x_of(seg_lo, sl))) * valid
        out[sl, :-1] += 0.5 * dphi
        out[sl, 1:] += 0.5 * dphi
        rowsum = out[sl].sum(axis=1)
        good = rowsum > 1e-12
        out[sl][good] /= rowsum[good, None]
        # Degenerate rows: linear-interpolation weights at the midpoint.
        bad = np.flatnonzero(~good) + start
        if bad.size:
            mid = np.clip(0.5 * (lo[bad] + hi[bad]), fk[0], fk[-1])
            idx = np.clip(np.searchsorted(fk, mid, side='right') - 1, 0, nfk - 2)
            frac = (mid - fk[idx]) / (fk[idx + 1] - fk[idx])
            out[bad, :] = 0.
            out[bad, idx] = 1. - frac
            out[bad, idx + 1] = frac
    return out


def _cell_average(x, table, intervals, axis=0):
    """Average ``table`` over per-output cells [lo, hi] along ``axis``:
    (C(hi) - C(lo)) / (hi - lo) with C the trapezoid cumulative integral --
    the sharp direction is *integrated*, never point-sampled, so structure
    narrower than the output cells cannot alias. Degenerate cells (hi = lo,
    e.g. a closure-leg turning point) fall back to the interpolated point
    value at the midpoint, the consistent limit."""
    intervals = jnp.asarray(intervals)
    lo, hi = intervals[..., 0], intervals[..., 1]
    mid = 0.5 * (lo + hi)
    C = _cumtrapz0(x, table, axis=axis)
    Ihi = _interp_linear(x, C, hi, axis=axis)
    Ilo = _interp_linear(x, C, lo, axis=axis)
    x = jnp.asarray(x)
    # Effective cell width after clipping to the grid (a cell partly outside
    # the tabulated range must be normalized by its covered part only).
    delta_cov = jnp.clip(hi, x[0], x[-1]) - jnp.clip(lo, x[0], x[-1])
    small = delta_cov <= 1e-10 * (jnp.abs(hi) + jnp.abs(lo) + 1e-30)
    point = _interp_linear(x, table, mid, axis=axis)
    davg = jnp.where(small, 1., delta_cov)
    shape = [1] * jnp.asarray(table).ndim
    shape[axis] = -1
    return jnp.where(small.reshape(shape), point, (Ihi - Ilo) / davg.reshape(shape))


class _QWSpectrum(object):
    """The (k1, k2) covariance-window table of one (ell1, ell2) pair, kept lazy.

    Every consumer of this table either sandwiches it between two matrices
    (``A @ T @ B.T`` -- bin rebinning, closure-cell measures) or interpolates it at traced
    points. The first form never needs the dense ``T``: both FFTlog transforms are linear, so
    the left matrix can be pushed through them (:meth:`Correlation2Spectrum.contracted`),
    touching only ``(nrow, n_s)`` arrays. At the ``n_s = 8192`` the window must be resampled to
    for the FFTlog to be accurate, dense is 537 MB against 27 MB contracted -- and it is what
    made the Zel'dovich P+B covariance OOM an 80 GB device. Only the interpax path, which
    genuinely interpolates the table, materializes ``.dense`` (once, cached).
    """
    def __init__(self, fftlog, w):
        self._fftlog, self._w, self._dense = fftlog, w, None

    @property
    def k(self):
        return self._fftlog.k

    @property
    def dense(self):
        if self._dense is None:
            self._dense = self._fftlog(self._w)[1]
        return self._dense

    def contract(self, A, B):
        if self._dense is not None:  # already paid for; reuse
            return jnp.asarray(A) @ self._dense @ jnp.asarray(B).T
        # The contraction is a win only when the left operand has FEW rows: it costs
        # O(nrow * n_s log n_s) against the dense table's one-off O(n_s^2 log n_s) plus a
        # matmul. Binned rows (~400) against a resampled n_s = 8192 grid: contract. Closure
        # measures (nside * nbins rows, ~9k at q = 6 and ~21k at q = 8) against the stored
        # n_s = 1024 grid: the dense table is far cheaper and only 8 MB.
        nrow = np.shape(A)[0]
        if int(os.environ.get('COV3_WINDOW_DENSE', '0')) or nrow > len(self.k) // 4:
            return jnp.asarray(A) @ self.dense @ jnp.asarray(B).T
        return self._fftlog.contracted(self._w, A, B)


def compute_spectrum2_covariance_window_block(window2, k1edges, k2edges, ell1, ell2,
                                              fields1=None, fields2=None, cache=None,
                                              k1_is_points=False, k2_is_points=False,
                                              k1_measure=None, k2_measure=None):
    r"""Return one :math:`(k_1, k_2)` covariance-window block.

    If ``k1_is_points`` (``k2_is_points``) is True, ``k1edges`` (``k2edges``) is
    treated as literal k-values (e.g. a derived/closure leg with no native bin
    edges) at which Q_W is interpolated, rather than bin edges Q_W is rebinned
    into. If additionally ``k1_measure`` (``k2_measure``) is given -- a
    callable ``measure(fk) -> (n, n_fk)`` row-normalized weight matrix (see
    ``_closure_measure_matrix``) -- Q_W is *cell-averaged* with the exact
    closure phi measure instead of point-sampled, so window structure
    narrower than the node spacing cannot slip between nodes.
    """
    if cache is None:
        cache = {}
    rebin_cache = cache.setdefault("rebin_matrix", [])
    spectrum_cache = cache.setdefault("QW_spectrum", {})
    spline_cache = cache.setdefault("spline_basis", {})

    def normalize_edges(edges):
        edges = np.asarray(edges)
        if edges.ndim == 1:
            return np.column_stack([edges[:-1], edges[1:]])
        if edges.ndim >= 2 and edges.shape[-1] == 2:
            return edges.reshape(-1, 2)
        raise ValueError("k edges must be 1D bin edges or explicit per-bin edges with last dimension 2.")

    def get_window_field(window, fields1, fields2):
        if fields1 is None:
            return window
        if isinstance(window, tuple):
            # Symmetrize Q_W^{A,B} with the second window evaluated as Q_W^{B,A}.
            w1 = get_window_field(window[0], fields1, fields2)
            w2 = get_window_field(window[1], fields2, fields1)
            return w1.clone(value=(w1.value() + w2.value()) / 2.)
        return window.get(fields1=fields1, fields2=fields2)

    if k1_is_points:
        # jnp, not np: k1edges may be a traced (e.g. vmapped) array here.
        k1points = jnp.ravel(jnp.asarray(k1edges))
    else:
        k1edges = normalize_edges(k1edges)
    if k2_is_points:
        k2points = jnp.ravel(jnp.asarray(k2edges))
    else:
        k2edges = normalize_edges(k2edges)

    # The FFTlog-transformed window grid only depends on (window2, fields,
    # ell1, ell2) -- not on k1edges/k2edges -- so cache it by that key alone.
    # This keeps it cached even when k1edges/k2edges are traced (e.g. a
    # closure leg varying under jax.vmap), where the outer block_cache in
    # compute_QW_AB can't be used.
    # Key by the id(s) of the *underlying* window object(s): callers like
    # compute_QW_AB pass a freshly-built (window2, window2) symmetrization
    # tuple each call, whose own id changes every time -- keying on that
    # would defeat the cache entirely, re-running the FFTlog + jacfwd
    # construction and appending another dense spectrum grid to the cache
    # for every W2 call (tens of GB leaked over a 6D-quadrature run).
    window2_ids = tuple(id(w) for w in window2) if isinstance(window2, tuple) else id(window2)
    spectrum_key = (window2_ids, fields1, fields2, ell1, ell2)
    if spectrum_key not in spectrum_cache:
        window2_field = get_window_field(window2, fields1, fields2) if fields1 is not None else window2
        w = sum(legendre_product(ell1, ell2, q) * window2_field.get(q).value().real if q in window2_field.ells else jnp.zeros(())
                for q in range(abs(ell1 - ell2), ell1 + ell2 + 1))
        if w.size <= 1:
            spectrum_cache[spectrum_key] = None
        else:
            tmpw = next(iter(window2_field))
            s = tmpw.coords('s')
            fftlog = Correlation2Spectrum(s, (ell1, ell2), check_level=1)
            spectrum_cache[spectrum_key] = (fftlog.k, _QWSpectrum(fftlog, w))

    cached = spectrum_cache[spectrum_key]
    if cached is None:
        n1 = len(k1points) if k1_is_points else len(k1edges)
        n2 = len(k2points) if k2_is_points else len(k2edges)
        return np.zeros((n1, n2))
    fk, spectrum = cached

    interp_order = 3

    if k1_measure is not None or k2_measure is not None:
        # Cell-averaged closure leg(s): average the (concrete, cached)
        # spectrum table with the exact closure phi measure over each node's
        # own cell instead of sampling it at the node value -- see the
        # docstring. The other axis is either rebinned (concrete bin edges)
        # or cell-averaged too.
        if k1_measure is not None and k2_measure is not None:
            W1 = jnp.asarray(k1_measure(fk))
            W2 = jnp.asarray(k2_measure(fk))
            return spectrum.contract(W1, W2)
        if k1_measure is not None:
            My = matrix_rebin(k2edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
            return spectrum.contract(jnp.asarray(k1_measure(fk)), My)
        Mx = matrix_rebin(k1edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
        return spectrum.contract(Mx, jnp.asarray(k2_measure(fk)))

    try:
        import interpax
    except ImportError:
        interpax = None

    if interpax is None or not (k1_is_points or k2_is_points):
        # Both legs binned (concrete): the plain matrix sandwich is cheap and
        # cached upstream by compute_QW_AB. Without interpax, also use the
        # (memory-hungrier) matrix path for traced-points legs.
        if k1_is_points:
            Mx = matrix_spline_interp(fk, k1points, interp_order=interp_order, cache=spline_cache)
        else:
            Mx = matrix_rebin(k1edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
        if k2_is_points:
            My = matrix_spline_interp(fk, k2points, interp_order=interp_order, cache=spline_cache)
        else:
            My = matrix_rebin(k2edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
        return spectrum.contract(Mx, My)

    # At least one traced-points leg: interpolate *values* of the cached
    # spectrum directly instead of building an (npoints, n_fftlog)
    # interpolation matrix from an identity and sandwiching the dense
    # (n_fftlog, n_fftlog) spectrum. Interpolation is linear in the table, so
    # this is mathematically identical to the matrix form, but per vmapped
    # quadrature point it allocates only (npoints, nbins) outputs instead of
    # (npoints, n_fftlog) operators and batched (npoints, n_fftlog) @
    # (n_fftlog, n_fftlog) matmuls -- the difference between a few MB and
    # tens of GB over a 6D-quadrature vmap chunk. 'cubic2' matches the
    # scipy/matrix_rebin spline convention (see matrix_spline_interp).
    method = 'linear' if interp_order == 1 else 'cubic2'
    contracted_cache = cache.setdefault("QW_contracted", {})

    if k1_is_points and k2_is_points:
        # 2D interpolation at the (k1, k2) outer grid; spline coefficients
        # of the (concrete, cached) spectrum are precomputed once.
        key = spectrum_key + ('interp2d',)
        if key not in contracted_cache:
            contracted_cache[key] = interpax.Interpolator2D(fk, fk, spectrum.dense, method=method)
        interp = contracted_cache[key]
        n1, n2 = k1points.shape[0], k2points.shape[0]
        xq = jnp.repeat(k1points, n2)
        yq = jnp.tile(k2points, n1)
        return interp(xq, yq).reshape(n1, n2)

    if k1_is_points:
        # k2 binned (concrete): contract the binned side into the cached
        # spectrum once, then a single 1D interpolation along the traced leg.
        key = spectrum_key + ('right', tuple(np.ravel(k2edges)))
        if key not in contracted_cache:
            My = matrix_rebin(k2edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
            contracted_cache[key] = spectrum.dense @ My.T  # (n_fftlog, n2)
        table = contracted_cache[key]
        return interpax.interp1d(k1points, fk, table, method=method)  # (n1, n2)

    # k2 traced, k1 binned: mirror case.
    key = spectrum_key + ('left', tuple(np.ravel(k1edges)))
    if key not in contracted_cache:
        Mx = matrix_rebin(k1edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
        contracted_cache[key] = (Mx @ spectrum.dense).T  # (n_fftlog, n1)
    table = contracted_cache[key]
    return interpax.interp1d(k2points, fk, table, method=method).T  # (n1, n2)


def compute_QW_AB(window2, k1edges, k2edges, khat_dot_n, khatp_dot_n, fields1=None, fields2=None, cache=None, ells=None,
                  k1_is_points=False, k2_is_points=False):
    """
    Reconstruct

        Q_W^{A,B}(k - k') = sum_{ell1,ell2} Q^W_{ell1 ell2}(k,k') L_{ell1}(khat . n) L_{ell2}(khat' . n)

    using compute_spectrum2_covariance_window_block.  If ``k1_is_points``
    (``k2_is_points``) is True, ``k1edges`` (``k2edges``) is literal k-values
    (e.g. a derived/closure leg with no native bin edges) at which Q_W is
    interpolated rather than rebinned.
    """
    if cache is None:
        cache = {}
    if ells is None:
        ells = [0, 2, 4]
    block_cache = cache.setdefault("QW_ell_blocks", {})
    # Pass window2 as a (window2, window2) pair, not a single pre-resolved
    # window2.get(fields1=fields1, fields2=fields2): compute_spectrum2_covariance_window_block's
    # own get_window_field only symmetrizes Q_W^{A,B} with Q_W^{B,A} when its
    # window argument is a tuple -- a single resolved field block is used
    # as-is, un-symmetrized. compute_spectrum2_covariance (cov2.py) already
    # relies on this same tuple form for its own WW/WS/SS lookups; compute_QW_AB
    # needs it too, for the PP block where fields1/fields2 are same-size
    # (e.g. (a,a')/(b,b')) groups and the swap is a genuinely equivalent
    # relabeling. For PB/BP, fields1/fields2 are *mixed*-size groups (a
    # 2-field spectrum group and a 3-field bispectrum-derived group) with
    # fixed, non-interchangeable roles -- window2 only stores that one
    # canonical ordering, so swapping is not a valid lookup there (and
    # raises). Only symmetrize when the two groups are the same size.
    window2_pair = window2 if (fields1 is None or len(fields1) != len(fields2)) else (window2, window2)
    # k1edges/k2edges may be traced (e.g. a closure leg under jax.vmap), in
    # which case they cannot be used as a cache key: recompute, uncached.
    cacheable = jax.core.is_concrete(k1edges) and jax.core.is_concrete(k2edges)
    out = None
    for ell1 in ells:
        L1 = get_legendre(ell1)(khat_dot_n)
        for ell2 in ells:
            L2 = get_legendre(ell2)(khatp_dot_n)
            prefactor = (2 * ell1 + 1) * (2 * ell2 + 1) * (-1)**(ell1 // 2) * (-1)**(ell2 // 2)
            if cacheable:
                key = (id(window2), tuple(np.ravel(k1edges)), tuple(np.ravel(k2edges)), ell1, ell2, (fields1, fields2), k1_is_points, k2_is_points)
                if key not in block_cache:
                    block_cache[key] = prefactor * compute_spectrum2_covariance_window_block(window2_pair, k1edges, k2edges, ell1, ell2, fields1=fields1, fields2=fields2, cache=cache, k1_is_points=k1_is_points, k2_is_points=k2_is_points)
                block = block_cache[key]
            else:
                block = prefactor * compute_spectrum2_covariance_window_block(window2_pair, k1edges, k2edges, ell1, ell2, fields1=fields1, fields2=fields2, cache=cache, k1_is_points=k1_is_points, k2_is_points=k2_is_points)

            term = block * L1[:, None] * L2[None, :]
            out = term if out is None else out + term
    return out


def compute_QW_ABC(window3, kedges, kpedges,
                   khat1, khat2, khat1p, khat2p,
                   fields1=None, fields2=None, fields3=None,
                   ells=None, cache=None, batch_size=None):
    """
    Reconstruct

        Q_W^{ABC}(k1,k1',k2,k2')
        =
        sum_{ell,ellin}
        Q^W_{ell,ellin}(k1,k2;k1',k2')
        S_ell(khat1,khat2,n)
        S_ellin(khat1',khat2',n)

    using compute_spectrum3_covariance_window_block.
    """
    if cache is None:
        cache = {}
    if ells is None:
        ells = [(0, 0, 0)]

    block_cache = cache.setdefault("QW_ABC_ell_blocks", {})
    basis_cache = cache.setdefault("QW_ABC_S_basis", {})

    def basis(ell, xhat1, xhat2):
        key = ("S", tuple(ell))
        if key not in basis_cache:
            basis_cache[key] = get_S(ell, z3=True)
        return jnp.ravel(basis_cache[key](xhat1, xhat2))

    Sell = {tuple(ell): basis(ell, khat1, khat2) for ell in ells}
    Sellp = {tuple(ell): basis(ell, khat1p, khat2p) for ell in ells}

    out = None

    for ell in ells:
        ell = tuple(ell)
        S_ell = Sell[ell]
        for ellp in ells:
            ellp = tuple(ellp)
            S_ellp = Sellp[ellp]
            key = (id(window3), tuple(np.ravel(kedges)), tuple(np.ravel(kpedges)), ell, ellp, (fields1, fields2, fields3))
            if key not in block_cache:
                block_cache[key] = compute_spectrum3_covariance_window_block(
                    window3, kedges, kpedges, ell, ellp,
                    fields1=fields1, fields2=fields2, fields3=fields3,
                    cache=cache, batch_size=batch_size,
                )
            term = block_cache[key] * S_ell[:, None] * S_ellp[None, :]
            out = term if out is None else out + term

    return out


def compute_spectrum3_covariance(window2, window3, observable, theory=None, shotnoise: float=0.,
                                 shotnoise_p=None, cache=None, batch_size=None):
    r"""
    Parameters
    ----------
    shotnoise : float, dict, callable
        The **coincidence** amplitude: what a pair of points landing on top of one another
        contributes. A scalar is sn2 and asserts the Poisson relation sn_m = sn2^(m-1) for the
        higher coincidences; ``{2: sn2, 3: sn3, 4: sn4}`` supplies measured moments instead.
        This is what ``B^(N)``, ``T^(N)`` and the Cov[B, B] families are built from.

    shotnoise_p : float, dict, callable, optional
        The constant added to the theory to make ``P^(N) = P + shotnoise_p``. Defaults to
        ``shotnoise``, which is right whenever the tracer's discreteness is purely Poisson.

        It is NOT right when the theory P already carries part of the discreteness. That
        happens whenever P is a fitted EFT model of a halo-occupation tracer: the one-halo
        power is absorbed by the broadband and by whatever constant the fit calls ``sn_res``,
        so adding it again here double-counts, while the *coincidence* structures in ``B^(N)``
        and ``T^(N)`` are absorbed nowhere and do need the full amplitude. Measured on 500
        AbacusSummit LRG HOD mocks (``cosmodesi/claude_abacus_analytic_cov``), where the
        Poisson amplitude is 1995 and the one-halo excess 1830: using 1995 everywhere gives
        sigma_analytic/sigma_mock = 0.98 on P0 and 0.72 on B000, using 3825 everywhere gives
        1.17 and 0.90. Neither is right; the two amplitudes belong to different terms.

        Splitting the amplitude between this function and the theory's own stochastic
        parameters is fine, and is in fact the *right* thing for a halo-occupation tracer: the
        self-coincidences belong here, where the i != j estimator conventions apply to them,
        and the one-halo part belongs in the theory, where it gets the full leg structure
        because an i != j estimator does not exclude two distinct galaxies in one halo. The
        cross terms are not lost by doing that -- ``T^(N)``'s ``sn_ij * B(...)`` term uses the
        theory's B *including* its stochastic part, which is exactly ``sn_P x sn_1h x P``.
    """

    if cache is None:
        cache = {}

    # Theory should be a function that takes fields and returns a callable that takes len(fields) - 1 wavenumbers
    if isinstance(window2, MeshAttrs):
        mattrs = window2
        volume = mattrs.boxsize.prod()
        use_window_kernels = False
    else:
        volume = None
        use_window_kernels = True

    cov = [[None for _ in observable.items(level=None)] for _ in observable.items(level=None)]
    # Angular quadrature order (per axis); overridable for convergence
    # checks via the COV3_QUAD_SIZE environment variable.
    #
    # Default 8 (was 6, briefly 10). Measured against 1000 periodic-box Gaussian mocks
    # (2.2% noise on sigma), sigma_analytic / sigma_mock at k < 0.2:
    #
    #   q     P0     P2     P4    B000   B202
    #   6   1.001  0.998  0.979  0.965  0.976
    #   8   1.001  0.998  0.998  0.967  0.980     <- default
    #  10   1.001  0.998  0.998  0.980  0.985
    #  14   1.001  0.998  0.998  0.995  0.991
    #  16   1.001  0.998  0.998  0.996  0.995
    #  20   1.001  0.998  0.998  0.995  0.994
    #
    # P0/P2 are converged at 6; P4 needs >= 8; the bispectrum is still moving at
    # 12 and only settles by ~14.
    #
    # The same scan on the *windowed* (cutsky) path, vs 100 zero-shot-noise mesh
    # mocks (7.1% noise on sigma), for reference -- it converges by 16 too:
    #
    #   q     P0     P2     P4    B000   B202
    #   8   1.011  1.007  1.068  0.976  1.062
    #  10   1.011  1.007  1.068  0.992  1.064
    #  14   1.011  1.007  1.068  0.995  1.072
    #  16   1.011  1.007  1.068  0.999  1.072
    #  20   1.011  1.007  1.068  0.999  1.072
    #
    # (P4's 1.068 and B202's 1.072 are quadrature-INDEPENDENT: a genuine
    # window-path excess on the anisotropic blocks, tracked separately.)
    #
    # THE q^6 MEMORY CEILING IS GONE (both of them).
    #
    # There used to be two. The first was in the Q_W leg tables, which built a
    # (q^3 nbins)^2 node x node array for the (2, 2) tie that the assembly then
    # discarded; _tie_pair_used now stops them being built at all. The second was
    # the P x T term's pt_F, shape (q^3, q^3, nbins, nbinsp) -- 13 GB of float64 at
    # q = 10 and 101 GB at q = 14 -- built whenever the theory supplies a
    # trispectrum, directly or via the shot-noise renormalized T^(N), i.e. any run
    # with shotnoise != 0. At q = 14 XLA aborted with "byte size of input/output
    # arguments exceeds the base limit"; worse, once a *connected* T was supplied
    # (several surviving T groups rather than one) even q = 10 stopped fitting, and
    # it did not raise -- it sat in the GPU allocator's retry loop at 0%
    # utilisation indefinitely.
    #
    # That array is now never materialized: the (ell, ellp) blocks sharing a given
    # precompute are collected up front and their weights contracted inside the
    # scan carry, so the stored object is (npairs, nbins, nbinsp) instead of
    # (q^3, q^3, nbins, nbinsp) -- under a megabyte instead of 13 GB -- while the
    # expensive part, the T evaluations, is still paid once for all of them. See
    # the note at the `pt_pairs` construction in the (3)(3) windowed precompute.
    #
    # The default is 8: q^6 scaling makes 10 about 2.5x the cost of 8, and the
    # accuracy tables above price that at ~1.3% on B000 in the box and ~1.6% on the
    # windowed path -- small against the residuals these covariances are compared to.
    # P0/P2 are converged by 6 and P4 by 8, so only the bispectrum blocks pay.
    # Raise it with COV3_QUAD_SIZE when the B covariance is the limiting term.
    _qsize = int(os.environ.get('COV3_QUAD_SIZE', 8))
    # Angular parameterization of the triangle quadrature; see get_kvec3_x.
    _quad_param = os.environ.get('COV3_QUAD_PARAM', 'mu1mu2phi2')
    assert _quad_param in ('mu1mu2phi2', 'mu1xphi', 'mu1xphi_exact'), _quad_param
    # 'mu1xphi_exact': the shared grid is the mu1xphi one, and in addition the box PPP term's
    # closure-tied permutations integrate x PIECEWISE-EXACTLY over the k3 bin windows -- see
    # the block guarded by _x_exact in the box (3, 3) branch.
    _x_exact = _quad_param == 'mu1xphi_exact'
    if _qsize % 2:
        # Odd sizes are catastrophically broken (diagonals ~1e36): an odd phi grid
        # (uniform midpoint and Gauss-Legendre alike) places a node exactly at
        # phi2 = pi, where the closure leg k3 = |k1 + k2| -> 0 for equal-magnitude
        # legs (e.g. sugiyama-diagonal bins at a mu1 = mu2 = 0 node) -- a
        # measure-zero folded triangle that then carries finite quadrature weight
        # through 1 / k3-singular factors.
        raise ValueError(
            f'COV3_QUAD_SIZE = {_qsize} is odd: odd phi grids put a node at '
            'phi2 = pi where the closure leg k3 -> 0 for equal-magnitude legs, '
            'blowing up the covariance; use an even size.')
    integ_mu = integration(-1., 1., size=_qsize)
    # Separate order for the triangle-shape axis x = k1hat.k2hat, used ONLY by the (3)(3)
    # triangle grid under COV3_QUAD_PARAM='mu1xphi' (every other quadrature here keeps
    # integ_mu, so this cannot perturb them). The point of an anisotropic order: the
    # integrand's hard structure -- the k3 bin-edge step, since
    # k3^2 = k1^2 + k2^2 + 2 k1 k2 x -- lives on the x axis ALONE, while mu1 and phi carry only
    # smooth Legendre/azimuth dependence. Refining all three together costs q^3; refining x
    # alone costs q^2 * qx. FOLPS exposes the same per-axis choice (precision=[Nphi, Nx, Nmu]).
    _qxsize = int(os.environ.get('COV3_QUAD_SIZE_X', _qsize))
    if _qxsize % 2:
        raise ValueError(f'COV3_QUAD_SIZE_X = {_qxsize} is odd; use an even order')
    integ_x = integration(-1., 1., size=_qxsize)
    # Azimuthal axes: uniform midpoint rule (spectrally accurate for the
    # smooth 2 pi-periodic integrands here, and each node owns the cell
    # [phi_i - h/2, phi_i + h/2] exactly -- the closure-leg interval
    # construction below relies on that). COV3_PHI_RULE=leggauss restores
    # the legacy Gauss-Legendre nodes for A/B checks.
    _phi_rule = os.environ.get('COV3_PHI_RULE', 'midpoint')
    integ_phi = integration(0., 2. * np.pi, size=_qsize, method=_phi_rule)

    def closure_measure(coords_tri):
        """Exact closure-leg cell-average operator for the (mu1, mu2, phi2)
        triangle grid: for each quadrature node, the closure magnitude
        k3 = |k1 + k2| sweeps a range as phi2 crosses the node's own cell;
        sharp window factors are averaged over that sweep with the *exact*
        analytic phi measure (see _closure_measure_matrix) rather than
        point-sampled at the node's k3 -- nothing narrower than a cell can
        slip between nodes, with the true Jacobian weighting.

        Cell boundaries are exact for the midpoint phi rule; under
        COV3_PHI_RULE=leggauss they are approximated by mid-gaps (weights no
        longer match cell widths exactly -- legacy mode is for A/B only).
        Since k3(phi2) is monotonic on [0, pi] and [pi, 2 pi] and both 0 and
        pi land on cell boundaries for even midpoint grids, interval ends at
        the cell edges bracket the sweep; the node value is folded in too,
        covering interior extrema of odd/legacy grids at O(cell) accuracy.

        Returns a callable ``measure(fk) -> (n_rows, n_fk)`` weight matrix
        (cached per fk grid), rows raveled in the same (mu1, mu2, phi2, bin)
        C-order as the IntegralND node grids.
        """
        mu_nodes = np.asarray(integ_mu.x())
        phi_nodes = np.asarray(integ_phi.x())
        nphi = len(phi_nodes)
        if _phi_rule == 'midpoint':
            h = 2. * np.pi / nphi
            phi_bounds = np.arange(nphi + 1) * h
        else:
            phi_bounds = np.concatenate([[0.], 0.5 * (phi_nodes[:-1] + phi_nodes[1:]), [2. * np.pi]])
        nmu = len(mu_nodes)
        mu1g, mu2g = [g.ravel() for g in np.meshgrid(mu_nodes, mu_nodes, indexing='ij', sparse=False)]

        def k3_at(phis):
            m1 = np.repeat(mu1g, len(phis))
            m2 = np.repeat(mu2g, len(phis))
            p2 = np.tile(phis, nmu * nmu)
            (k1n, k2n, k3n), _, _ = jax.vmap(lambda a, b, c: get_kvec3(coords_tri[0], coords_tri[1], a, b, c))(
                jnp.asarray(m1), jnp.asarray(m2), jnp.asarray(p2))
            return np.asarray(k3n).reshape(nmu * nmu, len(phis), -1)

        k3b = k3_at(phi_bounds)      # (nmu^2, nphi + 1, nbins)
        k3c = k3_at(phi_nodes)       # (nmu^2, nphi, nbins)
        stack = np.stack([k3b[:, :-1], k3b[:, 1:], k3c], axis=0)
        lo, hi = stack.min(axis=0), stack.max(axis=0)     # (nmu^2, nphi, nbins)
        nbins = lo.shape[-1]
        k1, k2 = np.asarray(coords_tri[0]), np.asarray(coords_tri[1])          # (nbins,)
        shape = (nmu * nmu, nphi, nbins)
        ksq = np.broadcast_to((k1**2 + k2**2)[None, None, :], shape)
        k1k2 = np.broadcast_to((k1 * k2)[None, None, :], shape)
        cpar = np.broadcast_to((mu1g * mu2g)[:, None, None], shape)
        wpar = np.broadcast_to((np.sqrt(1. - mu1g**2) * np.sqrt(1. - mu2g**2))[:, None, None], shape)
        args = tuple(a.reshape(-1) for a in (lo, hi, ksq, k1k2, cpar, wpar))

        _cache = {}

        def measure(fk):
            fk = np.asarray(fk)
            key = (fk.shape[0], float(fk[0]), float(fk[-1]))
            if key not in _cache:
                _cache[key] = _closure_measure_matrix(fk, *args)
            return _cache[key]

        return measure

    def d_inverse_nmodes(edges, k):
        # edges[..., 0]/[..., 1], not edges[0]/edges[1]: edges has shape
        # (nbins, 2) (lower/upper per bin) -- edges[0]/edges[1] would
        # instead pick out bins 0 and 1 themselves.
        lo, hi = edges[..., 0], edges[..., 1]
        mask = (k >= lo) & (k <= hi)
        invnmodes = 1. / (4. / 3. * np.pi) * mask / (hi**3 - lo**3)
        invnmodes *= (2. * np.pi)**3 / volume
        return invnmodes

    def _bc0(v, n1, n2):
        # Broadcast an unprimed-triangle (axis-0) vector to (n1, n2, 3).
        # Needed -- not just v[:, None, :] -- whenever this feeds into a
        # get_theory(...) callable: that wrapper's _flatten reshapes each
        # argument to (-1, 3) independently, which would silently collapse
        # a bare leading-1 broadcast dim (e.g. kpvec[None, :]) before any
        # outer-product broadcasting against the other arguments happens,
        # turning an intended (n1, n2) block into wrong, element-wise pairing.
        return jnp.broadcast_to(v[:, None, :], (n1, n2, 3))

    def _bc1(v, n1, n2):
        # Broadcast a primed-triangle (axis-1) vector to (n1, n2, 3). See _bc0.
        return jnp.broadcast_to(v[None, :, :], (n1, n2, 3))

    def _norm(kvec):
        return jnp.sqrt(jnp.sum(kvec ** 2, axis=-1))

    def _mu(kvec):
        k = _norm(kvec)
        return jnp.where(k == 0, 0., kvec[..., 2] / k)

    def _hat(kvec):
        k = _norm(kvec)
        return jnp.where(k[..., None] == 0, 0., kvec / k[..., None])

    def inverse_V2(fields1, fields2):
        if not use_window_kernels:
            return 1. / volume
        # The 1/V^(4) factor is the monopole of the two-anchor window.
        # New convention: window2 stores the two factors separately.
        return window2.get(fields1=fields1, fields2=fields2, ells=0).value()[0]

    def inverse_V3(fields1, fields2, fields3):
        if not use_window_kernels:
            return 1. / volume
        # The 1/V^(6) factor is the monopole of the three-anchor window.
        return window3.get(fields1=fields1, fields2=fields2, fields3=fields3, ells=(0, 0, 0)).value()[0]

    def _oriented_kvec3(knorms, order, mu1, mu2, phi2):
        """Return a triangle parametrized by the first two fixed sides in ``order``.

        ``knorms`` is side-major, shape ``(2, nbins)`` (one row per fixed
        bispectrum leg).  The returned tuple follows ``order``.  For instance
        ``order=(2, 0, 1)`` integrates at fixed (k3, k1), builds k2 = -k3 - k1,
        and returns ``(k3, k1, k2)``.  Only ``order[0]`` and ``order[1]`` (always
        0 or 1) are used here; the side==2 case is handled separately by callers.
        """
        kA, kB = knorms[order[0]], knorms[order[1]]
        return get_kvec3(kA, kB, mu1, mu2, phi2)

    def _pb_order(side):
        if side == 0:
            return (0, 1, 2)
        if side == 1:
            return (1, 0, 2)
        if side == 2:
            return (2, 0, 1)
        raise ValueError(f"Invalid triangle side {side}.")

    def W2(kvec, kpvec, fields1=None, fields2=None, edges=None, edgesp=None, edges_is_points=False, edgesp_is_points=False):
        if edges is None or edgesp is None:
            raise ValueError("W2 requires fixed k-bin edges for both covariance anchors.")
        if not use_window_kernels:
            if edges_is_points or edgesp_is_points:
                raise NotImplementedError("Box-limit (no-window) covariance does not support a literal-points leg.")
            k = jnp.sqrt(jnp.sum(kvec**2, axis=-1))
            invnmodes = d_inverse_nmodes(edges, k)
            # Box-limit Gaussian covariance is diagonal in k-bins: distinct
            # bins are statistically independent with no window to mix them
            # (cf. compute_spectrum2_covariance's box-limit np.diag(...)).
            # Match by literal bin edges, not just index, so this is correct
            # even if edges != edgesp.
            same_bin = jnp.all(jnp.asarray(edges)[:, None, :] == jnp.asarray(edgesp)[None, :, :], axis=-1)
            return invnmodes[:, None] * same_bin
        return compute_QW_AB(
            window2, edges, edgesp,
            _mu(kvec), _mu(kpvec), fields1=fields1, fields2=fields2,
            cache=cache, k1_is_points=edges_is_points, k2_is_points=edgesp_is_points,
        ).real

    def W3(k1vec, k1pvec, k2vec, k2pvec,
           fields1=None, fields2=None, fields3=None,
           edges=None, edgesp=None):
        if edges is None or edgesp is None:
            raise ValueError("W3 requires fixed triangle-bin edges for both covariance anchors.")
        if not use_window_kernels:
            return 1. / inverse_V3(fields1, fields2, fields3)
        return compute_QW_ABC(
            window3, edges, edgesp,
            _hat(k1vec), _hat(k2vec), _hat(k1pvec), _hat(k2pvec),
            fields1=fields1, fields2=fields2, fields3=fields3,
            cache=cache, batch_size=batch_size,
        ).real

    def get_N(ell1, ell2, ell3):
        return (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1)

    def get_H(ell1, ell2, ell3):
        return wigner_3j(ell1, ell2, ell3, 0, 0, 0)

    qw_ells = [0, 2, 4]        # compute_QW_AB's default window multipoles
    w3_ells = [(0, 0, 0)]      # PPP kernel S-basis expansion channels
    if use_window_kernels and window3 is not None:
        # S-basis channels for the angular reconstruction of the 4-point PPP
        # kernel. These are INDEPENDENT of which multipoles the 3-point
        # window stores (get_w_rect zero-fills missing ones, and the stored
        # monopole alone feeds every diagonal channel with weight
        # 1/||S_ell||^2): tying this list to the stored multipoles nearly
        # cancels the Gaussian variance of anisotropic bispectrum poles such
        # as (2, 0, 2). The list must also be closed under (ell1, ell2)
        # exchange -- the primed-side S is evaluated at permuted legs, so a
        # mirror-asymmetric list (e.g. sorted storage representatives)
        # unbalances the 6-permutation Wick sum. The default lmax = 2 covers
        # the standard poles ((0,0,0), (2,0,2), ..., (2,2,4)) and is the
        # largest alias-free order at the default COV3_QUAD_SIZE = 6: on that
        # quadrature, ell >= 3 channels are NOT numerically orthogonal to
        # lower ones (e.g. <S_000 S_442> ~ 0.11), so raise
        # COV3_PPP_CHANNEL_LMAX and COV3_QUAD_SIZE together.
        _lmax = int(os.environ.get('COV3_PPP_CHANNEL_LMAX', 2))
        w3_ells = []
        for _l1, _l2 in itertools.product(range(_lmax + 1), repeat=2):
            for _L in range(abs(_l1 - _l2), _l1 + _l2 + 1, 2):
                if (_l1 + _l2 + _L) % 2 or _L % 2:
                    continue
                if abs(wigner_3j(_l1, _l2, _L, 0, 0, 0)) < 1e-12:
                    continue
                w3_ells.append((_l1, _l2, _L))

    def _qw_tables(win, rowspec, colspec, fields1, fields2, rows_shape=None, cols_shape=None, F_u=None, F_p=None,
                   rows_measure=None, cols_measure=None):
        """Q_W(k, k') multipole tables for one pair of covariance legs, with
        the compute_QW_AB conventions absorbed (ells = qw_ells and the
        (2l1+1)(2l2+1)(-1)^(l1//2)(-1)^(l2//2) prefactor).

        A side is binned when its ``*_shape`` is None; else it is a per-node
        interpolated (points) leg whose raveled axis is reshaped to
        ``(n_nodes, n_bins)``. Per-node factors ``F_u``/``F_p`` (dicts
        multipole -> (n_nodes, n_bins) array, e.g. Legendre or Legendre x
        theory) and the corresponding multipole sums are absorbed into the
        table along whichever node axes it has; whatever could not be
        absorbed (the factor of a side the table has no node axis for) stays
        keyed by its multipole ('e1'/'e2'/'e12' keys), applied per node by
        the caller.
        """
        row_is_pts, col_is_pts = rows_shape is not None, cols_shape is not None
        out = {}
        for e1 in qw_ells:
            for e2 in qw_ells:
                pref = (2 * e1 + 1) * (2 * e2 + 1) * (-1)**(e1 // 2) * (-1)**(e2 // 2)
                blk = pref * jnp.asarray(compute_spectrum2_covariance_window_block(
                    win, rowspec, colspec, e1, e2,
                    fields1=fields1, fields2=fields2, cache=cache,
                    k1_is_points=row_is_pts, k2_is_points=col_is_pts,
                    k1_measure=rows_measure, k2_measure=cols_measure)).real
                if row_is_pts and col_is_pts:
                    blk = blk.reshape(rows_shape + cols_shape) * F_u[e1][:, :, None, None] * F_p[e2][None, None, :, :]
                    out[None] = out.get(None, 0.) + blk
                elif row_is_pts:
                    blk = blk.reshape(rows_shape + (blk.shape[-1],)) * F_u[e1][:, :, None]
                    out['e2', e2] = out.get(('e2', e2), 0.) + blk
                elif col_is_pts:
                    blk = blk.reshape((blk.shape[0],) + cols_shape) * F_p[e2][None, :, :]
                    out['e1', e1] = out.get(('e1', e1), 0.) + blk
                else:
                    out['e12', e1, e2] = blk
        return out

    def make_pt_qmask(qmin):
        # Trispectrum evaluations here are generically OFF-SHELL: the window
        # (or the tie approximations) smears momentum conservation, so
        # internal kernel momenta that are bounded on-shell can vanish:
        # pair sums q_ij = k_i + k_j (alpha/beta/Z2 denominators, squeezed
        # T ~ P(q)/q^2) and triple sums q_ijk (the F3/G3 recursion's
        # 1/|q1+q2+q3|^2 -- on-shell equal to the fourth leg). Quadrature
        # nodes can land exactly on these degeneracies (e.g. symmetric
        # Gauss-Legendre phi nodes summing to 2 pi make a rotated primed leg
        # coincide with an unprimed leg), letting one node dominate by
        # orders of magnitude. Mask every pair and triple sum below qmin.
        def _pt_qmask(a1, a2, b1, b2):
            _q = lambda v: jnp.sqrt(jnp.sum(v**2, axis=-1))
            ok = _q(a1 + a2) >= qmin
            for u, v in ((a1, b1), (a1, b2), (a2, b1), (a2, b2), (b1, b2)):
                ok = ok & (_q(u + v) >= qmin)
            for tri in (a1 + a2 + b1, a1 + a2 + b2, a1 + b1 + b2, a2 + b1 + b2):
                ok = ok & (_q(tri) >= qmin)
            return ok
        return _pt_qmask

    def _tie_pair_used(li, lj, env='COV3_BB_TERMS'):
        """Whether the (unprimed leg ``li``, primed leg ``lj``) window tie enters
        the windowed (3, 3) assembly.

        Single source of truth for the BB and P x T skip rules, applied at
        *precompute* time as well as in the assembly loops. That is what lifts
        the q^6 memory ceiling: (2, 2) is the only pair whose Q_W table carries
        quadrature-node axes on BOTH sides, so its size is (q^3 nbins)^2 --
        40 GB of float64 at COV3_QUAD_SIZE = 12, 225 GB at 16 -- and it is
        discarded unconditionally by both assembly loops unless
        COV3_WIN_DOUBLE_CLOSURE is set. Building the tables the assembly throws
        away was the whole ceiling; nothing here changes what is summed.
        """
        if li == 2 and lj == 2 and not os.environ.get('COV3_WIN_DOUBLE_CLOSURE'):
            # The doubly-derived tie cannot be represented by the m = 0,
            # ell <= 4 double-Legendre channels (they enforce only
            # |k3| = |k3'|, not the vector tie); handled instead by the exact
            # substitution _c22_closure_tie. See the BB assembly loop.
            return False
        # Debug knobs: restrict to arm-only (li, lj < 2) or closure-involving
        # (2 in (li, lj)) tie pairs, or explicit pairs 'pair:li,lj[;li,lj...]'.
        # NOTE the precomputed tables are cached per (fields, binning): use a
        # fresh cache when changing these.
        _sel = os.environ.get(env)
        if _sel:
            if _sel.startswith('pair:'):
                return (li, lj) in {tuple(map(int, t.split(','))) for t in _sel[5:].split(';')}
            return ('closure' in _sel) == (2 in (li, lj))
        return True

    # --- Shot-noise MOMENTS -------------------------------------------------------------
    # For a weighted point set the coincidence ("contact") terms of the discretized
    # correlators carry the weight moments
    #     sn2 = V sum(w^2) / (sum w)^2      (= P_shot; pair coincidences)
    #     sn3 = V^2 sum(w^3) / (sum w)^3    (triple coincidences)
    #     sn4 = V^3 sum(w^4) / (sum w)^4    (quadruple coincidences)
    # and these are NOT powers of one another unless the weights are trivial: a scalar
    # `shotnoise` implicitly asserts the Poisson relation sn3 = sn2^2, sn4 = sn2^3, which a
    # Gaussian weight distribution violates by at most ~25% but a skewed one violates freely
    # (DESI LRG FKP x completeness weights: sn3 / sn2^2 = 1.28, sn4 / sn2^3 = 2.25; a
    # lognormal test weight: e and e^3, which was measured to collapse the B covariance to
    # 0.46 while leaving P untouched). Pass a dict {2: sn2, 3: sn3, 4: sn4} to supply the
    # measured moments; a scalar keeps the Poisson relation and is exactly backward
    # compatible for unweighted tracers.
    _sn_moments = None
    if isinstance(shotnoise, dict) and shotnoise and all(isinstance(k, (int, np.integer)) for k in shotnoise):
        _sn_moments = {int(k): float(v) for k, v in shotnoise.items()}
        shotnoise = _sn_moments.get(2, 0.)
    if isinstance(shotnoise_p, dict) and shotnoise_p and all(isinstance(k, (int, np.integer)) for k in shotnoise_p):
        shotnoise_p = {int(k): float(v) for k, v in shotnoise_p.items()}.get(2, 0.)
    # The contact terms below exist for FFT estimators, which do not exclude self-pairs /
    # self-triples WITHIN an estimator; Sugiyama's i != j != k estimators do exclude them, and
    # the original formulas implement exactly that convention.
    #
    # DEFAULT 0, i.e. the original formulas. What decides it is not whether the estimator is an
    # FFT one, but whether the bispectrum has already had its contact terms SUBTRACTED per
    # realization -- and the production pipeline does exactly that (`compute_fkp3_shotnoise`
    # evaluated on each catalog, stored as `num_shotnoise` and removed by `.value()`; for DESI
    # LRG z0.4-0.6 it removes 2.4e8 against a 7.3e7 signal at k ~ 0.25). Subtracting per
    # realization removes not just the mean shot noise but its realization-to-realization
    # fluctuation, which makes the FFT estimator equivalent to the i != j != k one, so the
    # covariance must revert to the original formulas. That equivalence was verified directly
    # on a box mock set with deliberately extreme (lognormal) weights: subtracted mocks +
    # original formulas gave B000 = 1.05 / 1.04 / 1.01 / 1.03 across four k bands, against 0.46
    # for the same mocks left unsubtracted with the same formulas.
    #
    # Set COV3_FFT_CONTACT=1 for the other convention -- a bispectrum whose shot noise has NOT
    # been subtracted from the estimator. Then also pass the measured weight moments as
    # `shotnoise={2: sn2, 3: sn3, 4: sn4}`: with skewed weights a scalar is not enough (see the
    # note just above).
    _fft_contact = bool(int(os.environ.get('COV3_FFT_CONTACT', '0')))

    def _get_sn(sn, a, b):
        if callable(sn):
            return sn(a, b)
        if isinstance(sn, dict):
            return sn.get((a, b), sn.get((b, a), 0.))
        return sn if a == b else 0.

    def get_shotnoise(a, b):
        # The COINCIDENCE amplitude: B^(N), T^(N) and the Cov[B, B] families.
        return _get_sn(shotnoise, a, b)

    def get_shotnoise_p(a, b):
        # The constant added to P^(N). Same thing unless the caller says otherwise -- see the
        # docstring for when it is not (a fitted EFT P already carries the one-halo power).
        return _get_sn(shotnoise if shotnoise_p is None else shotnoise_p, a, b)

    def get_sn_moment(fields, order):
        # Coincidence of `order` points, all of which must be the SAME tracer (a contact
        # term across distinct tracers vanishes). Falls back to the Poisson relation
        # sn_m = sn2^(m-1) when explicit moments were not supplied.
        a = fields[0]
        if any(f != a for f in fields[1:]):
            return 0.
        if _sn_moments is not None and order in _sn_moments:
            return _sn_moments[order]
        return get_shotnoise(a, a) ** (order - 1)

    def get_zero(kvec):
        return jnp.zeros_like(jnp.asarray(kvec)[..., 0])

    def get_base(fields):
        return theory(tuple(fields))

    def get_theory(fields):
        fields = tuple(fields)
        ndim = len(fields)

        def _flatten(*ks):
            k0 = jnp.asarray(ks[0])
            orig_shape = k0.shape[:-1]
            return orig_shape, tuple(jnp.asarray(k).reshape(-1, 3) for k in ks)

        # P^(N)_ab(k) = P_ab(k) + delta_ab / nbar
        if ndim == 2:
            a, b = fields
            P = get_base(fields)
            sn = get_shotnoise_p(a, b)      # P^(N) only; every other sn below is a coincidence

            if P is None and sn == 0:
                return None

            def P_N(k):
                orig_shape, (k,) = _flatten(k)
                out = get_zero(k)
                if P is not None:
                    out = out + P(k)
                if sn != 0:
                    out = out + sn
                return out.reshape(orig_shape)

            return P_N

        # B^(N)_abc(k1,k2,k3)
        # Eq. (24): B(k1,k2,k3) + 1/nbar [P(k2) + P(k3)]
        # generalized to fields: contractions of leg 1 with legs 2 and 3.
        # CONVENTION (why only two pair terms): in the covariance skeletons this trio is one
        # CROSS leg (slot 1) plus a SAME-ESTIMATOR pair (slots 2, 3). Sugiyama's i != j
        # estimators exclude coincidences within an estimator, so only the cross pairs
        # (1,2) -> P(k3) and (1,3) -> P(k2) survive -- that is Eq. (24), and it is why the
        # "normal" bispectrum shot noise (all three pairs + a constant) does NOT appear.
        # FFT estimators keep the within-pair terms too: the (2,3) pair -> P(k1), and the
        # full triple coincidence -> the constant sn3.
        if ndim == 3:
            a, b, c = fields
            B = get_base(fields)
            sn_ab = get_shotnoise(a, b)
            sn_ac = get_shotnoise(a, c)
            sn_bc = get_shotnoise(b, c) if _fft_contact else 0.
            sn3 = get_sn_moment((a, b, c), 3) if _fft_contact else 0.
            P_ac = get_base((a, c))
            P_ab = get_base((a, b))

            if (B is None and (sn_ab == 0 or P_ac is None) and (sn_ac == 0 or P_ab is None)
                    and (sn_bc == 0 or P_ab is None) and sn3 == 0):
                return None

            def B_N(k1, k2, k3):
                orig_shape, (k1, k2, k3) = _flatten(k1, k2, k3)
                out = get_zero(k1)
                if B is not None:
                    out = out + B(k1, k2, k3)
                if sn_ab != 0 and P_ac is not None:
                    out = out + sn_ab * P_ac(k3)
                if sn_ac != 0 and P_ab is not None:
                    out = out + sn_ac * P_ab(k2)
                if sn_bc != 0 and P_ab is not None:
                    # within-pair (2,3) contact correlated with the cross leg (FFT only)
                    out = out + sn_bc * P_ab(k1)
                if sn3 != 0:
                    # full triple coincidence (FFT only); the moment, NOT sn2^2
                    out = out + sn3
                return out.reshape(orig_shape)

            return B_N

        # T^(N), Eq. (14), generalized to cross-field shot-noise.
        # CONVENTION: (k1, k2) are the two legs of ONE estimator's pair and (k3, k4) the
        # other's -- exactly how the PT assembly calls this. Sugiyama's i != j estimators
        # allow only CROSS coincidences, giving the sn2 B terms (single cross pair) and the
        # sn2^2 P(k+k') terms (two disjoint cross pairs); any triple would contain a
        # within-estimator pair and is excluded -- which is why Eq. (14) carries no sn3 or
        # sn4. FFT estimators keep the within-pair coincidences, adding:
        #   * sn2 * B(k1+k2, k3, k4)          one within-pair contact + two free legs;
        #   * sn3 * P(k_leftover)  (x4)       a within pair + one cross leg coincide;
        #   * sn2 * sn2 * P(k1+k2)            both within pairs contact separately;
        #   * sn4                             all four coincide -- the moment, NOT sn2^3.
        # These are the pieces measured missing on the lognormal-weight box set (B000 -> 0.46
        # with everything else exact) and the sn4 one is the most moment-sensitive of all.
        if ndim == 4:
            a, b, c, d = fields
            T = get_base(fields)

            pairs = [(0, 2, get_shotnoise(a, c)), (0, 3, get_shotnoise(a, d)),
                     (1, 2, get_shotnoise(b, c)), (1, 3, get_shotnoise(b, d))]
            sn_ab_w = get_shotnoise(a, b) if _fft_contact else 0.
            sn_cd_w = get_shotnoise(c, d) if _fft_contact else 0.
            trios = ([(0, 1, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1), (2, 3, 1, 0)]
                     if _fft_contact else [])
            sn4 = get_sn_moment((a, b, c, d), 4) if _fft_contact else 0.

            if (T is None and all(sn == 0 for (_, _, sn) in pairs)
                    and sn_ab_w == 0 and sn_cd_w == 0 and sn4 == 0):
                return None

            def T_N(k1, k2, k3, k4):
                orig_shape, (k1, k2, k3, k4) = _flatten(k1, k2, k3, k4)
                ks = (k1, k2, k3, k4)
                fs = (a, b, c, d)
                out = get_zero(k1)

                if T is not None:
                    out = out + T(k1, k2, k3, k4)

                for i, j, sn_ij in pairs:
                    if sn_ij == 0:
                        continue
                    r, s = [m for m in range(4) if m not in (i, j)]
                    Bij = get_base((fs[r], fs[s], fs[i]))
                    if Bij is not None:
                        out = out + sn_ij * Bij(ks[r], ks[s], -ks[r] - ks[s])

                sn_ac = get_shotnoise(a, c)
                sn_bd = get_shotnoise(b, d)
                P_ac = get_base((a, c))
                if sn_ac != 0 and sn_bd != 0 and P_ac is not None:
                    out = out + sn_ac * sn_bd * P_ac(k1 + k3)

                sn_ad = get_shotnoise(a, d)
                sn_bc = get_shotnoise(b, c)
                P_ad = get_base((a, d))
                if sn_ad != 0 and sn_bc != 0 and P_ad is not None:
                    out = out + sn_ad * sn_bc * P_ad(k1 + k4)

                # ---- FFT-estimator (within-pair) contact terms ----
                # GUARD: these exist only where the within-pair momentum sum is NONZERO. In
                # the PP block T^(N) is called at (k1, -k1, k1', -k1'), where the pair sum
                # vanishes identically -- and there the contact Sum_i w_i^2 e^{-i(k1+k2) x_i}
                # is a deterministic constant (no x dependence), contributing nothing to the
                # covariance; the FFT P estimator's subtracted mean removes it exactly.
                # Without this guard the sn3 P and sn4 constants leak into Cov[P, P] and were
                # measured to inflate the P0 prediction by 14% on the lognormal set. The
                # relative threshold separates the exact arithmetic zero of the PP context
                # from genuinely small (squeezed) B-context sums, which are physical.
                def _alive(ka, kb):
                    q2 = jnp.sum((ka + kb)**2, axis=-1)
                    ref = jnp.sum(ka**2, axis=-1) + jnp.sum(kb**2, axis=-1)
                    return (q2 > 1e-12 * ref).astype(out.dtype)

                alive12 = _alive(k1, k2)
                alive34 = _alive(k3, k4)
                if sn_ab_w != 0:
                    Bw = get_base((a, c, d))
                    if Bw is not None:
                        out = out + sn_ab_w * alive12 * Bw(k1 + k2, k3, k4)
                if sn_cd_w != 0:
                    Bw = get_base((c, a, b))
                    if Bw is not None:
                        out = out + sn_cd_w * alive34 * Bw(k3 + k4, k1, k2)
                for i, j, l, r in trios:
                    sn3 = get_sn_moment((fs[i], fs[j], fs[l]), 3)
                    Pr = get_base((fs[r], fs[l]))
                    if sn3 != 0 and Pr is not None:
                        out = out + sn3 * (alive12 if i == 0 else alive34) * Pr(ks[r])
                if sn_ab_w != 0 and sn_cd_w != 0 and P_ac is not None:
                    out = out + sn_ab_w * sn_cd_w * alive12 * alive34 * P_ac(k1 + k2)
                if sn4 != 0:
                    out = out + sn4 * alive12 * alive34

                return out.reshape(orig_shape)

            return T_N

        return get_base(fields)

    def _c22_closure_tie(Sell, Sellp, f, fp, coords, coordsp, edges, edgesp,
                         kv_u, kv_p, k3n_u, k3n_p, hat_u, hat_p, w_u, volume_tie,
                         norm_tie, debug_label=''):
        """Double-closure (li = lj = 2) BB / PT tie via the exact local-frame
        angular substitution (shared by the periodic-box and windowed paths):
        k'_3 = s k_3 is consumed vector-exactly; the partner triangle's
        remaining freedom (its leg-2 direction) is targeted at its leg-1 bin
        through an exact mu23 / mu13 sub-interval substitution with the
        residual azimuth integrated; unprimed- and primed-anchored variants
        are averaged (see the anchoring-ambiguity note below).

        ``volume_tie`` sets the integrated tie strength: the box volume in
        the periodic limit, or 1 / Q_W^{(abc)(a'b'c')}(s -> 0) (the
        effective volume of the two-anchor triple-triple window) in the
        windowed path -- exact in the periodic limit; the finite window
        ridge width (a +-20-50%-level effect on single-closure ties) is
        neglected. The double-Legendre channel representation CANNOT be
        used for this tie: with two derived directions it only enforces
        |k3| = |k3'|, overcounting by the directions per shell (x10-30,
        refuted by the discrete vector-tie mode-sum).

        Returns ``{family: (nbins, nbinsp) block}``, ``norm_tie``
        (= M / (32 pi^2)) applied.
        """
        nu = len(np.asarray(w_u))
        nbins, nbinsp = coords.shape[-1], coordsp.shape[-1]
        nchunk = 8
        xi_c, w_xi_c = np.asarray(integ_mu.x()), np.asarray(integ_mu.w)
        nxc = len(xi_c)
        _pt_qmin = max(0.5 * min(np.min(edges), np.min(edgesp)),
                       0.5 * min(np.min(np.asarray(edges)[..., 1] - np.asarray(edges)[..., 0]),
                                 np.min(np.asarray(edgesp)[..., 1] - np.asarray(edgesp)[..., 0])))
        _pt_qmask = make_pt_qmask(_pt_qmin)

        def _keep(family):
            _sel = os.environ.get('COV3_BB_TERMS' if family == 'bb' else 'COV3_PT_TERMS')
            if not _sel:
                return True
            if _sel.startswith('pair:'):
                return (2, 2) in {tuple(map(int, t.split(','))) for t in _sel[5:].split(';')}
            return 'closure' in _sel

        S_u = Sell(hat_u[0], hat_u[1])                                               # (nu, nbins)
        S_p_out = Sellp(hat_p[0], hat_p[1])                                          # (nu, nbinsp)

        zhat_c = hat_u[2]                                                            # (nu, nbins, 3)
        ref_c = jnp.where(jnp.abs(zhat_c[..., 0:1]) < 0.9,
                           jnp.asarray([1., 0., 0.]), jnp.asarray([0., 1., 0.]))
        e1c = ref_c - zhat_c * jnp.sum(ref_c * zhat_c, axis=-1, keepdims=True)
        e1c = e1c / jnp.linalg.norm(e1c, axis=-1, keepdims=True)
        e2c = jnp.cross(zhat_c, e1c)                                                  # (nu, nbins, 3)

        k3mag_c = jnp.asarray(k3n_u)                                                  # (nu, nbins)
        k2pmag_c = np.asarray(coordsp[1])                                             # (nbinsp,)
        lo_b, hi_b = np.asarray(edgesp)[:, 0, 0], np.asarray(edgesp)[:, 0, 1]          # (nbinsp,)
        dk_b = hi_b - lo_b
        denom_c = 2. * k3mag_c[..., None] * k2pmag_c[None, None, :]
        mu23_lo = jnp.clip((lo_b[None, None, :]**2 - k3mag_c[..., None]**2 - k2pmag_c[None, None, :]**2) / denom_c, -1., 1.)
        mu23_hi = jnp.clip((hi_b[None, None, :]**2 - k3mag_c[..., None]**2 - k2pmag_c[None, None, :]**2) / denom_c, -1., 1.)
        valid_c = mu23_hi > mu23_lo                                                    # (nu, nbins, nbinsp)
        half_c = 0.5 * (mu23_hi - mu23_lo)
        mu23_nodes = mu23_lo[..., None] + (jnp.asarray(xi_c)[None, None, None, :] + 1.) * half_c[..., None]   # (nu, nbins, nbinsp, nxc)

        phi_free, w_phi_free = np.asarray(integ_phi.x()), np.asarray(integ_phi.w)      # raw sum 2 pi
        nphif = len(phi_free)
        cosphi_f, sinphi_f = jnp.cos(jnp.asarray(phi_free)), jnp.sin(jnp.asarray(phi_free))
        wcphi = (half_c[..., None, None] * jnp.asarray(w_xi_c)[None, None, None, :, None]
                  * jnp.asarray(w_phi_free)[None, None, None, None, :])                # (nu, nbins, nbinsp, nxc, nphif)
        ok_c = valid_c[..., None, None] & jnp.ones((nxc, nphif), dtype=bool)

        # Primed-anchored mirror: the (2, 2) tie has a genuine "which side
        # supplies the free shape integral" ambiguity (both tied legs are
        # derived); average the two anchorings.
        zhat_m = hat_p[2]                                                             # (nu, nbinsp, 3)
        ref_m = jnp.where(jnp.abs(zhat_m[..., 0:1]) < 0.9,
                           jnp.asarray([1., 0., 0.]), jnp.asarray([0., 1., 0.]))
        e1m = ref_m - zhat_m * jnp.sum(ref_m * zhat_m, axis=-1, keepdims=True)
        e1m = e1m / jnp.linalg.norm(e1m, axis=-1, keepdims=True)
        e2m = jnp.cross(zhat_m, e1m)                                                   # (nu, nbinsp, 3)

        k3mag_m = jnp.asarray(k3n_p)                                                   # (nu, nbinsp)
        k2mag_m = np.asarray(coords[1])                                                # (nbins,)
        lo_a, hi_a = np.asarray(edges)[:, 0, 0], np.asarray(edges)[:, 0, 1]             # (nbins,)
        dk_a = hi_a - lo_a
        denom_m = 2. * k3mag_m[..., None] * k2mag_m[None, None, :]
        mu13_lo = jnp.clip((lo_a[None, None, :]**2 - k3mag_m[..., None]**2 - k2mag_m[None, None, :]**2) / denom_m, -1., 1.)
        mu13_hi = jnp.clip((hi_a[None, None, :]**2 - k3mag_m[..., None]**2 - k2mag_m[None, None, :]**2) / denom_m, -1., 1.)
        valid_m = mu13_hi > mu13_lo                                                     # (nu, nbinsp, nbins)
        half_m = 0.5 * (mu13_hi - mu13_lo)
        mu13_nodes = mu13_lo[..., None] + (jnp.asarray(xi_c)[None, None, None, :] + 1.) * half_m[..., None]   # (nu, nbinsp, nbins, nxc)

        wcphi_m = (half_m[..., None, None] * jnp.asarray(w_xi_c)[None, None, None, :, None]
                    * jnp.asarray(w_phi_free)[None, None, None, None, :])               # (nu, nbinsp, nbins, nxc, nphif)
        ok_m = valid_m[..., None, None] & jnp.ones((nxc, nphif), dtype=bool)

        out = {}
        for (s_, family) in ((1., 'bb'), (-1., 'pt')):
            if not _keep(family):
                continue
            if family == 'bb':
                Bu_f = get_theory((f[2], f[0], f[1]))
                Bp_f = get_theory((fp[2], fp[0], fp[1]))
                if Bu_f is None or Bp_f is None:
                    continue
                Au = Bu_f(kv_u[2], kv_u[0], kv_u[1])                                # (nu, nbins)
                Ap_m = Bp_f(kv_p[2], kv_p[0], kv_p[1])                              # (nu, nbinsp)
            else:
                Pu_f = get_theory((f[2], fp[2]))
                Pp_f = get_theory((fp[2], f[2]))
                T_f = get_theory((f[0], f[1], fp[0], fp[1]))
                T_f_m = get_theory((fp[0], fp[1], f[0], f[1]))
                if Pu_f is None or Pp_f is None or T_f is None or T_f_m is None:
                    continue
                Au = Pu_f(kv_u[2])                                                  # (nu, nbins)
                Ap_m = Pp_f(kv_p[2])                                                # (nu, nbinsp)
            wA = jnp.asarray(w_u)[:, None] * S_u * Au                               # (nu, nbins)
            wA_m = jnp.asarray(w_u)[:, None] * S_p_out * Ap_m                       # (nu, nbinsp)
            block_t = 0.
            for sl in [slice(c, c + (nu + nchunk - 1) // nchunk) for c in range(0, nu, (nu + nchunk - 1) // nchunk)]:
                nc = len(range(*sl.indices(nu)))
                shp = (nc, nbins, nbinsp, nxc, nphif)
                mu23_s, s23_s = mu23_nodes[sl], jnp.sqrt(jnp.clip(1. - mu23_nodes[sl]**2, 0., None))
                zhat_s = jnp.broadcast_to(zhat_c[sl][:, :, None, None, None, :], shp + (3,))
                e1_s = jnp.broadcast_to(e1c[sl][:, :, None, None, None, :], shp + (3,))
                e2_s = jnp.broadcast_to(e2c[sl][:, :, None, None, None, :], shp + (3,))
                mu23_b = jnp.broadcast_to(mu23_s[..., None], shp)
                cph_b = jnp.broadcast_to(cosphi_f[None, None, None, None, :], shp)
                sph_b = jnp.broadcast_to(sinphi_f[None, None, None, None, :], shp)
                s23_b = jnp.broadcast_to(s23_s[..., None], shp)
                k2phat = mu23_b[..., None] * zhat_s + (s23_b * cph_b)[..., None] * e1_s + (s23_b * sph_b)[..., None] * e2_s
                k2pvec = k2pmag_c[None, None, :, None, None, None] * k2phat            # (nc, nbins, nbinsp, nxc, nphif, 3)
                k3pvec_b = jnp.broadcast_to((s_ * kv_u[2][sl])[:, :, None, None, None, :], shp + (3,))
                k1pvec = -(k3pvec_b + k2pvec)
                kamag = jnp.sqrt(jnp.sum(k1pvec**2, axis=-1))
                kasafe = jnp.where(kamag == 0., 1., kamag)
                ntilde = 4. * np.pi * kasafe * coordsp[0][None, None, :, None, None] * dk_b[None, None, :, None, None] * volume_tie / (2. * np.pi)**3
                D = jnp.asarray(ok_c[sl]) / ntilde
                hal = k1pvec / kasafe[..., None]
                Sp_g = Sellp(hal, k2phat)                                            # (nc, nbins, nbinsp, nxc, nphif)
                if family == 'bb':
                    Bp = Bp_f(k3pvec_b, k1pvec, k2pvec)
                else:
                    Ta1 = jnp.broadcast_to(kv_u[0][sl][:, :, None, None, None, :], shp + (3,))
                    Ta2 = jnp.broadcast_to(kv_u[1][sl][:, :, None, None, None, :], shp + (3,))
                    Bp = T_f(Ta1, Ta2, k1pvec, k2pvec) * _pt_qmask(Ta1, Ta2, k1pvec, k2pvec)
                integrand = wcphi[sl] * Sp_g * D * Bp
                block_t = block_t + jnp.einsum('ua,uabnp->ab', wA[sl], integrand)

            block_t_m = 0.
            for sl in [slice(c, c + (nu + nchunk - 1) // nchunk) for c in range(0, nu, (nu + nchunk - 1) // nchunk)]:
                nc = len(range(*sl.indices(nu)))
                shpm = (nc, nbinsp, nbins, nxc, nphif)
                mu13_s, s13_s = mu13_nodes[sl], jnp.sqrt(jnp.clip(1. - mu13_nodes[sl]**2, 0., None))
                zhat_sm = jnp.broadcast_to(zhat_m[sl][:, :, None, None, None, :], shpm + (3,))
                e1_sm = jnp.broadcast_to(e1m[sl][:, :, None, None, None, :], shpm + (3,))
                e2_sm = jnp.broadcast_to(e2m[sl][:, :, None, None, None, :], shpm + (3,))
                mu13_b = jnp.broadcast_to(mu13_s[..., None], shpm)
                cph_bm = jnp.broadcast_to(cosphi_f[None, None, None, None, :], shpm)
                sph_bm = jnp.broadcast_to(sinphi_f[None, None, None, None, :], shpm)
                s13_b = jnp.broadcast_to(s13_s[..., None], shpm)
                k2hat_m = mu13_b[..., None] * zhat_sm + (s13_b * cph_bm)[..., None] * e1_sm + (s13_b * sph_bm)[..., None] * e2_sm
                k2vec_m = k2mag_m[None, None, :, None, None, None] * k2hat_m           # (nc, nbinsp, nbins, nxc, nphif, 3)
                k3vec_tied_m = jnp.broadcast_to((s_ * kv_p[2][sl])[:, :, None, None, None, :], shpm + (3,))
                k1vec_m = -(k3vec_tied_m + k2vec_m)
                kmag_m = jnp.sqrt(jnp.sum(k1vec_m**2, axis=-1))
                ksafe_m = jnp.where(kmag_m == 0., 1., kmag_m)
                ntilde_m = 4. * np.pi * ksafe_m * coords[0][None, None, :, None, None] * dk_a[None, None, :, None, None] * volume_tie / (2. * np.pi)**3
                Dm = jnp.asarray(ok_m[sl]) / ntilde_m
                hal_m = k1vec_m / ksafe_m[..., None]
                S_here_m = Sell(hal_m, k2hat_m)                                       # (nc, nbinsp, nbins, nxc, nphif)
                if family == 'bb':
                    Au_m = Bu_f(k3vec_tied_m, k1vec_m, k2vec_m)
                else:
                    Tb1 = jnp.broadcast_to(kv_p[0][sl][:, :, None, None, None, :], shpm + (3,))
                    Tb2 = jnp.broadcast_to(kv_p[1][sl][:, :, None, None, None, :], shpm + (3,))
                    Au_m = T_f_m(Tb1, Tb2, k1vec_m, k2vec_m) * _pt_qmask(Tb1, Tb2, k1vec_m, k2vec_m)
                integrand_m = wcphi_m[sl] * S_here_m * Dm * Au_m
                block_t_m = block_t_m + jnp.einsum('ub,ubanp->ab', wA_m[sl], integrand_m)

            out[family] = norm_tie * 0.5 * (block_t + block_t_m)
            if os.environ.get('COV3_BOX_DEBUG'):
                print(f"{debug_label} (c) mu23 {family}: diag = {np.array2string(np.diag(np.asarray(out[family])), precision=3)}")
        return out

    _observable = observable
    # COV3_TIMING=1: per-(i, ip) block wall time. The blocks differ by orders of magnitude
    # (PP is a 2D quadrature, BB a 6D one), and knowing which dominates is the difference
    # between optimizing the window machinery -- 4.6% of a Zel'dovich run, measured -- and
    # optimizing what actually costs.
    _timing = int(os.environ.get('COV3_TIMING', '0'))
    _tblocks = []
    _tmark = [0.]

    def _mark(label, obj=None):
        # Phase timer for the (3,3) precompute. Blocks on the phase's output, without which
        # JAX's async dispatch would attribute every phase's cost to whichever one is forced
        # first. Only active under COV3_TIMING.
        if not _timing:
            return
        import time as _t
        if obj is not None:
            try: jax.block_until_ready(obj)
            except Exception: pass
        now = _t.time()
        if _tmark[0]:
            print('[cov3]   precompute %s: %.1fs' % (label, now - _tmark[0]), flush=True)
        _tmark[0] = now

    for i, (label, observable) in enumerate(_observable.items(level=None)):
        for ip, (labelp, observablep) in enumerate(_observable.items(level=None)):
            if ip < i:
                continue
            if _timing:
                import time as _time
                _tblock = _time.time()

            fields, fieldsp = tuple(label['fields']), tuple(labelp['fields'])
            nfields, nfieldsp = len(fields), len(fieldsp)
            ell, ellp = label['ells'], labelp['ells']
            # edges, edgesp are of shape (nbins, 2) for spectrum, (nbins, 2, 2) for bispectrum
            edges, edgesp = [np.asarray(obs.edges('k')) for obs in [observable, observablep]]
            center = 'mid_if_edges_and_nan'
            # coords, coordsp are of shape (nbins,) for spectrum, (2, nbins) for bispectrum
            coords, coordsp = [obs.coords('k', center=center).T for obs in [observable, observablep]]

            # PP block
            if nfields == 2 and nfieldsp == 2:
                a, b = fields
                ap, bp = fieldsp
                leg, legp = get_legendre(ell), get_legendre(ellp)

                if not use_window_kernels:
                    # Reference periodic form (_cov3_math_periodic.tex,
                    # Eq. CovPP_G): the angular delta ties khat' = +-khat, so
                    # a single *shared*-mu integral remains,
                    #   (2l+1)(2l'+1) delta_bins / N_mode(k)
                    #     x int dmu/2 L_l(mu) L_l'(mu) [P-route products],
                    # with N_mode(k) = 4 pi k^2 dk V / (2 pi)^3. The two Wick
                    # routes (a-a')(b-b') and (a-b')(b-a') are enumerated
                    # explicitly (the reference's leading 2 in the
                    # single-tracer limit).
                    P_a_ap, P_b_bp, P_a_bp, P_b_ap = get_theory((a, ap)), get_theory((b, bp)), get_theory((a, bp)), get_theory((b, ap))
                    T_abapbp = get_theory((a, b, ap, bp))

                    mu_s, w_mu = np.asarray(integ_mu.x()), np.asarray(integ_mu.w)
                    kvec_u = jax.vmap(lambda m: get_kvec1(coords, m)[2])(jnp.asarray(mu_s))          # (nq, nbins, 3)
                    pair = P_a_ap(kvec_u) * P_b_bp(-kvec_u) + P_a_bp(kvec_u) * P_b_ap(-kvec_u)       # (nq, nbins)
                    same_bin = np.all(np.asarray(edges)[:, None, :] == np.asarray(edgesp)[None, :, :], axis=-1)
                    nmode = 4. * np.pi * np.asarray(coords)**2 * (np.asarray(edges)[:, 1] - np.asarray(edges)[:, 0]) * volume / (2. * np.pi)**3
                    # Raw w_mu sums to 2: /2 is the dmu/2 measure.
                    wmu = jnp.asarray(w_mu * leg(mu_s) * legp(mu_s)) * (2 * ell + 1) * (2 * ellp + 1) / 2.
                    block = jnp.einsum('u,ua,ab->ab', wmu, pair, jnp.asarray(same_bin / nmode[:, None]))

                    if T_abapbp is not None:
                        def _t0_block_fn(mu, mup, phi, w):
                            kvec = coords[..., None] * unitvec(mu, jnp.zeros_like(mu))
                            kpvec = coordsp[..., None] * unitvec(mup, phi)
                            n1, n2 = coords.shape[0], coordsp.shape[0]
                            T_val = T_abapbp(_bc0(kvec, n1, n2), _bc0(-kvec, n1, n2), _bc1(kpvec, n1, n2), _bc1(-kpvec, n1, n2))
                            pref = (2 * ell + 1) * (2 * ellp + 1) / (8. * np.pi)
                            return pref * inverse_V2((a, b), (ap, bp)) * T_val * leg(mu) * legp(mup) * w

                        integ_t0 = IntegralND(mu=integ_mu, mup=integ_mu, phi=integ_phi)
                        mu_t, mup_t, phi_t = integ_t0.x(['mu', 'mup', 'phi'], sparse=False)
                        mu_t, mup_t, phi_t, w_t = (np.ravel(arr) for arr in (mu_t, mup_t, phi_t, integ_t0.w))
                        block = block + jax.vmap(_t0_block_fn)(mu_t, mup_t, phi_t, w_t).sum(axis=0)

                else:
                    # Everything below except the final assembly is
                    # independent of the observable multipoles (ell, ellp):
                    # cache it keyed by fields and binning, so all multipole
                    # blocks of this observable pair pay it once.
                    pre_cache = cache.setdefault('pp22_ell_independent', {})
                    pre_key = (fields, fieldsp,
                               np.asarray(coords).tobytes(), np.asarray(coordsp).tobytes(),
                               np.asarray(edges).tobytes(), np.asarray(edgesp).tobytes())

                    if pre_key not in pre_cache:
                        pre = {}
                        P_a_ap, P_b_bp, P_a_bp, P_b_ap = get_theory((a, ap)), get_theory((b, bp)), get_theory((a, bp)), get_theory((b, ap))
                        T_abapbp = get_theory((a, b, ap, bp))

                        # mu (k's own orientation) and mup (k''s) are
                        # independent: the window breaks rotational
                        # invariance, so k and k' are generally oriented
                        # differently relative to the LOS.
                        mu_s, w_mu = np.asarray(integ_mu.x()), np.asarray(integ_mu.w)
                        pre['mu_s'], pre['w_mu'] = mu_s, w_mu
                        kvec_u = jax.vmap(lambda m: get_kvec1(coords, m)[2])(jnp.asarray(mu_s))    # (nq, nbins, 3)
                        kvec_p = jax.vmap(lambda m: get_kvec1(coordsp, m)[2])(jnp.asarray(mu_s))   # (nq, nbinsp, 3)

                        # Gaussian terms:
                        #   Q_W^{(a ap)(b bp)}(k, +k') P_aa'(k) P_bb'(-k')
                        # + Q_W^{(a bp)(b ap)}(k, -k') P_ab'(k) P_ba'(+k').
                        # The window's Legendre reconstruction is insensitive
                        # to the +-k' sign (even multipoles); the sign enters
                        # only through the P arguments.
                        wpair = (window2, window2)
                        pre['gauss'] = []
                        # Sort each group independently (window2 stores each
                        # same-size group's own fields sorted, e.g. (a,ap) ->
                        # (min,max)); the (window2, window2) tuple already
                        # handles swapping flds1 <-> flds2 as a whole.
                        for (flds1, flds2, PL, PR) in [
                                ((a, ap), (b, bp), P_a_ap(kvec_u), P_b_bp(-kvec_p)),
                                ((a, bp), (b, ap), P_a_bp(kvec_u), P_b_ap(kvec_p))]:
                            tab = _qw_tables(wpair, edges, edgesp, tuple(sorted(flds1)), tuple(sorted(flds2)))
                            pre['gauss'].append({'tab': tab, 'PL': PL, 'PR': PR})

                        # T0 term: the trispectrum T(k, -k, k', -k')
                        # genuinely depends on the *relative* azimuthal angle
                        # between k and k' through q+- = k +- k' (k's azimuth
                        # stays 0 WLOG by rotational symmetry about the LOS),
                        # so a joint (mu, mup, phi) quadrature remains; store
                        # the unweighted per-node values, the multipole
                        # weights are applied per block.
                        pre['t0_F'] = None
                        if T_abapbp is not None:
                            integ_t0 = IntegralND(mu=integ_mu, mup=integ_mu, phi=integ_phi)
                            mu_t, mup_t, phi_t = integ_t0.x(['mu', 'mup', 'phi'], sparse=False)
                            mu_t, mup_t, phi_t, w_t = (np.ravel(arr) for arr in (mu_t, mup_t, phi_t, integ_t0.w))
                            iV2 = inverse_V2((a, b), (ap, bp))

                            def _t0_point(mu, mup, phi):
                                kvec = coords[..., None] * unitvec(mu, jnp.zeros_like(mu))
                                kpvec = coordsp[..., None] * unitvec(mup, phi)
                                n1, n2 = coords.shape[0], coordsp.shape[0]
                                # Explicit (n1, n2, 3) broadcast for all four
                                # legs, not kpvec[None, :]: get_theory(...)'s
                                # _flatten reshapes each argument to (-1, 3)
                                # independently, which would collapse a bare
                                # leading-1 dim into wrong element-wise
                                # pairing.
                                return iV2 * T_abapbp(_bc0(kvec, n1, n2), _bc0(-kvec, n1, n2), _bc1(kpvec, n1, n2), _bc1(-kpvec, n1, n2))

                            pre['t0_F'] = jax.vmap(_t0_point)(mu_t, mup_t, phi_t)   # (nq^3, nbins, nbinsp)
                            pre['t0_nodes'] = (mu_t, mup_t, w_t)

                        pre_cache[pre_key] = pre

                    pre = pre_cache[pre_key]

                    # ---- Per-(ell, ellp) assembly (cheap) ----
                    mu_s, w_mu = pre['mu_s'], pre['w_mu']
                    # /4, not /2: two independent mu, mup integrals, each its
                    # own (2ell+1)/2 multipole-extraction normalization.
                    pref22 = (2 * ell + 1) * (2 * ellp + 1) / 4.
                    block = 0.
                    for entry in pre['gauss']:
                        tab, PL, PR = entry['tab'], entry['PL'], entry['PR']
                        for e1 in qw_ells:
                            lvec = jnp.asarray(w_mu * leg(mu_s) * get_legendre(e1)(mu_s)) @ PL     # (nbins,)
                            for e2 in qw_ells:
                                rvec = jnp.asarray(w_mu * legp(mu_s) * get_legendre(e2)(mu_s)) @ PR  # (nbinsp,)
                                block = block + pref22 * tab['e12', e1, e2] * lvec[:, None] * rvec[None, :]

                    if pre['t0_F'] is not None:
                        mu_t, mup_t, w_t = pre['t0_nodes']
                        # /(8 pi): (1/2)(1/2) mu, mup multipole-extraction
                        # normalizations times the 1/(2 pi) relative-azimuth
                        # average.
                        wt = jnp.asarray(w_t * leg(mu_t) * legp(mup_t)) * (2 * ell + 1) * (2 * ellp + 1) / (8. * np.pi)
                        block = block + jnp.einsum('n,nab->ab', wt, pre['t0_F'])

            # PB block
            elif nfields == 2 and nfieldsp == 3:
                a, b = fields
                c, d, e = fieldsp
                leg = get_legendre(ell)
                Sp = get_S(ellp, z3=True)

                if not use_window_kernels:
                    # Periodic (box) approximation: Q_W^{(A)(B)}(p, q) ->
                    # (2 pi)^3 delta_D(p - q) / V. The angular delta collapses
                    # the spectrum-side integral, evaluating L_ell at the
                    # contracted leg's own orientation (with L_ell(-mu) for
                    # the -k terms); the radial delta is shell-averaged over
                    # the spectrum bins, mask / V_shell.
                    pb_terms_spec = [
                        (0, (a, c), (b, d, e), +1),
                        (1, (a, d), (b, c, e), +1),
                        (2, (a, e), (b, c, d), +1),
                        (0, (b, c), (a, d, e), -1),
                        (1, (b, d), (a, c, e), -1),
                        (2, (b, e), (a, c, d), -1),
                    ]

                    integ_tri = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                    _tri = integ_tri.x(['mu1', 'mu2', 'phi2'], sparse=False)
                    mu1_s, mu2_s, phi2_s = (np.ravel(arr) for arr in _tri)
                    w_tri = np.ravel(integ_tri.w)

                    def _pb_oriented(side, mu1, mu2, phi2):
                        # Return (q, r1, r2): the contracted bispectrum leg
                        # and the two fixed legs; for side == 2, q = k3 is
                        # the closure of coordsp[0], coordsp[1].
                        if side == 2:
                            (k1n, k2n, k3n), (k1h, k2h, k3h), (k1v, k2v, k3v) = get_kvec3(coordsp[0], coordsp[1], mu1, mu2, phi2)
                            return (k3n, k1n, k2n), (k3h, k1h, k2h), (k3v, k1v, k2v)
                        order = _pb_order(side)
                        return _oriented_kvec3(coordsp, order, mu1, mu2, phi2)

                    def _pb_side(side, fieldsP, fieldsB):
                        P, B = get_theory(fieldsP), get_theory(fieldsB)
                        if P is None or B is None:
                            return None
                        _fn = lambda m1, m2, p2: _pb_oriented(side, m1, m2, p2)
                        qnorms, qhats, kvecs = jax.vmap(_fn)(jnp.asarray(mu1_s), jnp.asarray(mu2_s), jnp.asarray(phi2_s))
                        qn, qh = qnorms[0], qhats[0]
                        qvec, r1vec, r2vec = kvecs
                        PB_u = P(qvec) * B(qvec, r1vec, r2vec)              # (ntri, nbinsp)
                        if side == 2:
                            muq = qh[..., 2]                                # (ntri, nbinsp)
                        else:
                            muq = jnp.broadcast_to(jnp.asarray(mu1_s)[:, None], PB_u.shape)
                        return qn, muq, PB_u

                    hat1_tri = jax.vmap(lambda m: unitvec(m, jnp.zeros_like(m)))(jnp.asarray(mu1_s))   # (ntri, 3)
                    hat2_tri = jax.vmap(unitvec)(jnp.asarray(mu2_s), jnp.asarray(phi2_s))              # (ntri, 3)

                    lo, hi = np.asarray(edges)[:, 0], np.asarray(edges)[:, 1]
                    kk, dk = np.asarray(coords), hi - lo
                    wS_tri = jnp.asarray(w_tri) * Sp(hat1_tri, hat2_tri)
                    # side == 1 parametrizes the LITERAL k2 leg at (mu1, 0)
                    # and k1 at (mu2, phi2) (see _oriented_kvec3's order
                    # (1, 0, 2)): S_{l1 l2 L}(k1hat, k2hat) then takes the
                    # swapped arguments.
                    wS_tri_swap = jnp.asarray(w_tri) * Sp(hat2_tri, hat1_tri)
                    # Reference (Eq. covPB_multipole): (2l+1) N H^2 per Wick
                    # route on the normalized triangle measure (raw weights
                    # sum to 8 pi); the reference's leading 2 is the two
                    # routes, already enumerated explicitly in pb_terms_spec
                    # (the spectrum-side normalized measure is consumed by
                    # the angular delta). Radial delta: W(k, q) /
                    # Ntilde_mode(k, q), Ntilde_mode = 4 pi k q dk V/(2 pi)^3.
                    pref_box = (2 * ell + 1) * get_N(*ellp) * get_H(*ellp)**2 / (8. * np.pi)
                    block = jnp.zeros((coords.shape[-1], coordsp.shape[-1]))
                    for (side, fieldsP, fieldsB, sign) in pb_terms_spec:
                        _side_out = _pb_side(side, fieldsP, fieldsB)
                        if _side_out is None:
                            continue
                        qn, muq, PB_u = _side_out
                        wS = wS_tri_swap if side == 1 else wS_tri
                        Lq = get_legendre(ell)(sign * muq)              # (ntri, nbinsp)
                        mask = (qn[None, ...] >= lo[:, None, None]) & (qn[None, ...] <= hi[:, None, None])
                        qn_safe = jnp.where(qn == 0., 1., qn)
                        ntilde = 4. * np.pi * kk[:, None, None] * qn_safe[None, ...] * dk[:, None, None] * volume / (2. * np.pi)**3
                        invn = mask / ntilde                            # (nbins, ntri, nbinsp)
                        block = block + pref_box * jnp.einsum('u,ub,aub->ab', wS, Lq * PB_u, invn)

                    # ---- The P5 term, arXiv:1908.06234 Eq. (26)-(27) -----------------
                    #
                    #   Cov[P(k), B(k1,k2,k3)]_P5 = (1/V) P5^(N)(k, -k, k1, k2, k3)
                    #
                    #   P5^(N) = P5
                    #     + (1/nbar)  [ T(k+k1, -k, k2, k3) + T(k+k2, -k, k1, k3)
                    #                 + T(k+k3, -k, k1, k2) + T(-k+k1, k, k2, k3)
                    #                 + T(-k+k2, k, k1, k3) + T(-k+k3, k, k1, k2) ]
                    #     + (1/nbar^2)[ B(k+k1, k2-k, k3) + B(k+k1, k3-k, k2)
                    #                 + B(k+k2, k3-k, k1) + B(-k+k1, k2+k, k3)
                    #                 + B(-k+k1, k3+k, k2) + B(-k+k2, k3+k, k1) ]
                    #
                    # Unlike the PB family above this carries NO radial delta: the power
                    # spectrum leg's direction and the triangle's orientation are integrated
                    # independently. That is exactly why it, and not PB, populates the
                    # OFF-DIAGONAL of the block -- as the reference says, "while the PB term
                    # provides small contributions to the off-diagonal elements of the
                    # covariance matrix, the P5 term dominates the off-diagonal elements".
                    #
                    # Measured against 500 AbacusSummit-small LRG HOD mocks
                    # (cosmodesi/claude_abacus_analytic_cov): the PB family alone supplies a
                    # median 0.138 of the mocks' off-diagonal Cov[P0, B000], and is EXACTLY
                    # ZERO wherever the tie mask cannot fire; adding the two shot-noise lines
                    # here takes that to 0.609, and from zero to 0.20-0.52 in the corner.
                    # On the diagonal they are a small correction growing with k (P5/PB =
                    # 0.016, 0.092, 0.196, 0.36 at k = 0.052, 0.111, 0.171, 0.250).
                    #
                    # The connected P5 is NOT built here -- its tree expression runs to
                    # hundreds of permutations and needs a Z4 kernel (reference Appendix A).
                    # It is picked up automatically if `theory` supplies a 5-point callable.
                    # The two shot-noise lines need only T and B, which jaxpower.pt has.
                    #
                    # COV3_NO_P5=1 restores the previous behaviour (no P5 term at all).
                    _p5_qk = int(os.environ.get('COV3_P5_QK', '6'))
                    if not int(os.environ.get('COV3_NO_P5', '0')) and _p5_qk > 0:
                        _T5 = get_base(fields + fieldsp[:2])       # connected T (4 legs)
                        _B5 = get_base((a,) + fieldsp[:2])         # connected B (3 legs)
                        _P5 = get_base(fields + fieldsp)           # connected P5, if supplied
                        _sn5 = get_shotnoise(a, c)
                        # One shared galaxy between the P and B estimators merges one leg of
                        # each: a pair coincidence, order 2. Two shared galaxies -> order 3.
                        # For a Poisson tracer get_sn_moment(m) = sn^(m-1) exactly, so this is
                        # inert unless the caller supplies measured moments; for an HOD the
                        # higher ones are far from Poisson (A_3 = 3.4, A_4 = 23).
                        _sn5_1 = get_sn_moment((a, a), 2)
                        _sn5_2 = get_sn_moment((a, a, a), 3)
                        if not (_T5 is None and _B5 is None and _P5 is None):
                            # The canonical primed triangle, on the same (mu1, mu2, phi2) grid
                            # the PB family uses, so wS_tri (weights x S_ellp) is reused.
                            _fnc = lambda m1, m2, p2: get_kvec3(coordsp[0], coordsp[1], m1, m2, p2)
                            _, _, (t1, t2, t3) = jax.vmap(_fnc)(
                                jnp.asarray(mu1_s), jnp.asarray(mu2_s), jnp.asarray(phi2_s))
                            # Power-spectrum leg: its own 2-sphere grid, independent of the
                            # triangle. Azimuth offset as in the BB branch, so that k + k_i
                            # cannot vanish to machine zero against a triangle node.
                            _mk, _wmk = np.asarray(integration(-1., 1., size=_p5_qk).x()), \
                                        np.asarray(integration(-1., 1., size=_p5_qk).w)
                            _pk = (np.arange(_p5_qk) + 0.5) * 2. * np.pi / _p5_qk + np.pi / 17.
                            _wpk = np.full(_p5_qk, 2. * np.pi / _p5_qk)
                            _MK, _PK = (g.ravel() for g in np.meshgrid(_mk, _pk, indexing='ij'))
                            _WK = jnp.asarray(np.einsum('i,j->ij', _wmk, _wpk).ravel())
                            _kdir = jax.vmap(unitvec)(jnp.asarray(_MK), jnp.asarray(_PK))
                            _Lk = get_legendre(ell)(jnp.asarray(_MK))
                            _kmag = jnp.asarray(coords)                       # (nbins,)

                            # One (sphere node, unprimed bin) per scan step. Holding the
                            # full (ntri, nbins, nbinsp) grid live instead costs nbins times
                            # more, and a trispectrum evaluation carries a lot of
                            # intermediates: at COV3_QUAD_SIZE = 10 that ran the 80 GB device
                            # out of memory on a 14 x 14 block, on top of the q^6 quadrature
                            # tables the rest of the function already holds. Stepping over the
                            # bins keeps each evaluation at (ntri, nbinsp).
                            _nba, _nbb, _ntri = coords.shape[-1], coordsp.shape[-1], len(w_tri)

                            def _p5_step(carry, n):
                                iv, ia = n // _nba, n % _nba
                                kv = jnp.broadcast_to(_kmag[ia] * _kdir[iv], (_ntri, _nbb, 3))
                                L = [t1, t2, t3]                       # each (ntri, nbinsp, 3)
                                acc = jnp.zeros((_ntri, _nbb))
                                # T legs (s k + k_i, -s k, k_j, k_l) sum to k_i+k_j+k_l = 0;
                                # B legs (s k + k_i, k_j - s k, k_l) likewise. All callables
                                # take every leg explicitly, as elsewhere in this function.
                                if _T5 is not None and _sn5 != 0:
                                    for i in range(3):
                                        jj, ll = [m for m in range(3) if m != i]
                                        for sg in (1., -1.):
                                            acc = acc + _sn5_1 * _T5(sg * kv + L[i], -sg * kv,
                                                                     L[jj], L[ll])
                                if _B5 is not None and _sn5 != 0:
                                    for (i, jj) in ((0, 1), (0, 2), (1, 2)):
                                        ll = 3 - i - jj
                                        for sg in (1., -1.):
                                            acc = acc + _sn5_2 * _B5(sg * kv + L[i],
                                                                     L[jj] - sg * kv, L[ll])
                                if _P5 is not None:
                                    acc = acc + _P5(kv, -kv, L[0], L[1], L[2])
                                # Emit `acc` rather than contracting here: it depends only on
                                # the two grids and the theory, NOT on (ell, ellp) -- only
                                # `_Lk` (Legendre in ell) and `wS_tri` (S_ellp) do, and both are
                                # cheap. Contracting inside would redo every pt call for each of
                                # the 6 P x B pairs.
                                return carry, acc

                            _p5_cache = cache.setdefault('p5_ell_independent', {})
                            _p5_key = (fields, fieldsp, _p5_qk,
                                       float(_sn5_1), float(_sn5_2),
                                       np.asarray(coords).tobytes(),
                                       np.asarray(coordsp).tobytes())
                            if _p5_key not in _p5_cache:
                                # (nWK * nba, ntri, nbb); the scan keeps each pt evaluation at
                                # (ntri, nbb), as before -- only the stacked output is new.
                                _, _p5_cache[_p5_key] = jax.lax.scan(
                                    _p5_step, 0., jnp.arange(len(_WK) * _nba))
                            # wS_tri is the triangle measure x S_ellp already built for the PB
                            # family above; the canonical (side 0) orientation.
                            _acc5 = _p5_cache[_p5_key].reshape(len(_WK), _nba, _ntri, _nbb)
                            _p5 = jnp.einsum('v,u,vaub->ab', _WK * _Lk, wS_tri, _acc5)
                            norm_p5 = ((2 * ell + 1) * get_N(*ellp) * get_H(*ellp)**2
                                       / (8. * np.pi) / (4. * np.pi) / volume)
                            if os.environ.get('COV3_BOX_DEBUG'):
                                print(f'boxPB p5: max = {np.abs(np.asarray(norm_p5 * _p5)).max():.3e}'
                                      f'  vs PB max {np.abs(np.asarray(block)).max():.3e}')
                            block = block + norm_p5 * _p5

                    # To host as soon as it exists. Leaving blocks as device arrays makes
                    # the final np.block do every device->host copy at once, at the moment
                    # the GPU is fullest (precompute caches still resident) -- measured: a
                    # box q = 10 run completed the whole computation and then died in
                    # np.block asking for 168 MB. Converting here also frees each block's
                    # device buffer immediately.
                    block = np.asarray(block)
                    cov[i][ip] = block
                    cov[ip][i] = block.T
                    if _timing:
                        dt = _time.time() - _tblock
                        _tblocks.append((dt, i, ip, tuple(label['fields']), label['ells'],
                                         tuple(labelp['fields']), labelp['ells'], block.shape))
                        print(f'[cov3] block ({i},{ip}) '
                              f'{len(label["fields"])}x{len(labelp["fields"])} ells '
                              f'{label["ells"]}/{labelp["ells"]} {block.shape}: {dt:.1f}s', flush=True)
                    continue

                # Everything below except the final assembly is independent
                # of the observable multipoles (ell, ellp): cache it keyed by
                # fields and binning, so all multipole blocks of this
                # observable pair pay it once.
                pre_cache = cache.setdefault('pb23_ell_independent', {})
                pre_key = (fields, fieldsp,
                           np.asarray(coords).tobytes(), np.asarray(coordsp).tobytes(),
                           np.asarray(edges).tobytes(), np.asarray(edgesp).tobytes())

                if pre_key not in pre_cache:
                    pre = {}
                    P_ac, P_ad, P_ae, P_bc, P_bd, P_be = get_theory((a, c)), get_theory((a, d)), get_theory((a, e)), get_theory((b, c)), get_theory((b, d)), get_theory((b, e))
                    B_bde, B_bce, B_bcd, B_ade, B_ace, B_acd = get_theory((b, d, e)), get_theory((b, c, e)), get_theory((b, c, d)), get_theory((a, d, e)), get_theory((a, c, e)), get_theory((a, c, d))

                    # Angle factorization: the spectrum side enters only
                    # through Legendre factors of its own mu; the theory
                    # (P x B), the S basis and the contracted leg live on the
                    # bispectrum side's own (mu1, mu2, phi2) grid.
                    mu_s, w_mu = np.asarray(integ_mu.x()), np.asarray(integ_mu.w)
                    pre['mu_s'], pre['w_mu'] = mu_s, w_mu
                    integ_tri = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                    _tri = integ_tri.x(['mu1', 'mu2', 'phi2'], sparse=False)
                    mu1_s, mu2_s, phi2_s = (np.ravel(arr) for arr in _tri)
                    w_tri = np.ravel(integ_tri.w)
                    ntri = len(w_tri)
                    nbinsp_bisp = coordsp.shape[-1]
                    pre['w_tri'] = w_tri

                    # Closure-leg (side == 2) tie: the survey window Q_W(k, k3)
                    # is much narrower than the k3 = |k1 + k2| sweep across the
                    # angular grid, so point-sampling it at node values imprints
                    # spurious ridges. Instead, cell-average Q_W over each
                    # node's own closure sweep (see closure_measure /
                    # _cell_average): the sharp direction is integrated, never
                    # sampled, on the shared coarse grid.
                    k3msr_c = closure_measure(coordsp)     # rows: (ntri * nbinsp)

                    def _pb_oriented(side, mu1, mu2, phi2):
                        # Return (q, r1, r2), where q is the contracted
                        # bispectrum side (the leg replaced by Q_W(p, q)) and
                        # r1, r2 are the bispectrum's own two fixed legs
                        # (used directly as B's arguments). For side == 2,
                        # q = k3 = -(k1 + k2) is the closure of the
                        # bispectrum's own k1 = coordsp[0], k2 = coordsp[1]
                        # (per the PB covariance formula: B(-p, k1, k2), with
                        # only k3 contracted via the window).
                        if side == 2:
                            (k1n, k2n, k3n), (k1h, k2h, k3h), (k1v, k2v, k3v) = get_kvec3(coordsp[0], coordsp[1], mu1, mu2, phi2)
                            return (k3n, k1n, k2n), (k3h, k1h, k2h), (k3v, k1v, k2v)
                        order = _pb_order(side)
                        return _oriented_kvec3(coordsp, order, mu1, mu2, phi2)

                    # S_{ell1 ell2 L}(k1hat, k2hat, n): always the
                    # bispectrum's own literal first two legs (mu1,
                    # (mu2, phi2)) and the line of sight (z3=True) -- never
                    # permuted by which leg is contracted.
                    pre['hat1'] = jax.vmap(lambda m: unitvec(m, jnp.zeros_like(m)))(jnp.asarray(mu1_s))       # (ntri, 3)
                    pre['hat2'] = jax.vmap(unitvec)(jnp.asarray(mu2_s), jnp.asarray(phi2_s))                  # (ntri, 3)

                    # Contract p = sign * k (the spectrum's own k, +/-) with
                    # the bispectrum's own leg q_i through Q_W(p, q_i); P and
                    # B are evaluated directly on the bispectrum's own
                    # triangle (q_i, r1, r2), independent of p. The window's
                    # Legendre reconstruction is insensitive to the sign
                    # (even multipoles); it only distinguishes the field
                    # pairings. fields sizes differ (pair vs triple), so
                    # bare window2 (no symmetrization pair), matching
                    # compute_QW_AB.
                    pb_terms = [
                        (0, P_ac, B_bde, (a, c), (b, d, e), +1),
                        (1, P_ad, B_bce, (a, d), (b, c, e), +1),
                        (2, P_ae, B_bcd, (a, e), (b, c, d), +1),
                        (0, P_bc, B_ade, (b, c), (a, d, e), -1),
                        (1, P_bd, B_ace, (b, d), (a, c, e), -1),
                        (2, P_be, B_acd, (b, e), (a, c, d), -1),
                    ]
                    pre['terms'] = []
                    for (side, P, B, fieldsP, fieldsB, sign) in pb_terms:
                        # Skip terms with no bispectrum (or power spectrum) theory, e.g. a
                        # field with no bispectrum defined (mirrors the periodic-box
                        # _pb_side guard and the BB block's None-safety).
                        if P is None or B is None:
                            continue
                        # window2 stores each (mixed-size, non-interchangeable) group's
                        # own fields sorted -- sort here to match regardless of numeric
                        # field-label order (does not affect P/B, already resolved above).
                        fieldsP, fieldsB = tuple(sorted(fieldsP)), tuple(sorted(fieldsB))
                        _fn = lambda m1, m2, p2: _pb_oriented(side, m1, m2, p2)
                        qnorms, qhats, kvecs = jax.vmap(_fn)(jnp.asarray(mu1_s), jnp.asarray(mu2_s), jnp.asarray(phi2_s))
                        qn, qh = qnorms[0], qhats[0]                    # (ntri, nbinsp[, 3]) or (ntri[, 3]) for side < 2
                        qvec, r1vec, r2vec = kvecs
                        PB_u = P(qvec) * B(qvec, r1vec, r2vec)          # (ntri, nbinsp)
                        entry = {'side': side, 'sign': sign}
                        if side == 2:
                            # k3 has no native bin edges: cell-average Q_W
                            # over each node's own closure sweep (see above);
                            # its khat . n is bin-dependent. Absorb
                            # L_e2(mu_q) x P x B into the table's node axis.
                            muq = qh[..., 2]                            # (ntri, nbinsp)
                            F_p = {e2: get_legendre(e2)(muq) * PB_u for e2 in qw_ells}
                            entry['tab'] = _qw_tables(window2, edges, np.asarray(jnp.ravel(qn)), fieldsP, fieldsB,
                                                      cols_shape=(ntri, nbinsp_bisp), F_p=F_p,
                                                      cols_measure=k3msr_c)
                        else:
                            # edgesp is bin-major, shape (nbins, 2, 2):
                            # axis 1 selects the leg. The contracted leg's
                            # khat . n is its own polar angle, bin-independent.
                            entry['tab'] = _qw_tables(window2, edges, edgesp[:, side, :], fieldsP, fieldsB)
                            entry['PB_u'] = PB_u
                            entry['muq'] = np.asarray(mu1_s)            # (ntri,)
                        pre['terms'].append(entry)

                    pre_cache[pre_key] = pre

                pre = pre_cache[pre_key]

                # ---- Per-(ell, ellp) assembly (cheap) ----
                mu_s, w_mu = pre['mu_s'], pre['w_mu']
                wS_tri = jnp.asarray(pre['w_tri']) * Sp(pre['hat1'], pre['hat2'])
                # side == 1 places the literal k2 leg at (mu1, 0) and k1 at
                # (mu2, phi2); S's literal-leg arguments are then swapped.
                wS_tri_swap = jnp.asarray(pre['w_tri']) * Sp(pre['hat2'], pre['hat1'])
                # (2l+1) N H^2 acting on normalized measures (see
                # _cov3_math.tex): /2 for the spectrum side's dmu/2 and
                # /(8 pi) for the triangle side's (dmu1/2)(dmu2 dphi2/4pi) --
                # the quadrature weights below are raw (summing to 2 and
                # 8 pi respectively). As in the periodic-box branch, the
                # reference's leading 2 counts the two Wick routes, which
                # pb_terms already enumerates explicitly (+/- signs) --
                # keeping it double-counted the whole PB block.
                prefPB = (2 * ell + 1) * get_N(*ellp) * get_H(*ellp)**2 / (2. * 8. * np.pi)
                block = 0.
                for entry in pre['terms']:
                    tab, sign = entry['tab'], entry['sign']
                    wS = wS_tri_swap if entry['side'] == 1 else wS_tri
                    for e1 in qw_ells:
                        # Spectrum-side scalar: sum_mu w L_ell(mu) L_e1(sign mu).
                        lsc = np.sum(w_mu * leg(mu_s) * get_legendre(e1)(sign * mu_s))
                        if entry['side'] == 2:
                            block = block + prefPB * lsc * jnp.einsum('aub,u->ab', tab['e1', e1], wS)
                        else:
                            for e2 in qw_ells:
                                cvec = (wS * get_legendre(e2)(jnp.asarray(entry['muq']))) @ entry['PB_u']   # (nbinsp,)
                                block = block + prefPB * lsc * tab['e12', e1, e2] * cvec[None, :]

            # BP block
            elif nfields == 3 and nfieldsp == 2:
                a, b, c = fields
                dp, ep = fieldsp
                S = get_S(ell, z3=True)
                legp = get_legendre(ellp)

                if not use_window_kernels:
                    # Periodic (box) approximation, mirror of the PB one:
                    # Q_W(q, p) -> (2 pi)^3 delta_D(q - p) / V, the radial
                    # delta shell-averaged over the spectrum (column) bins.
                    bp_terms_spec = [
                        (0, (a, dp), (b, c, ep), +1),
                        (1, (b, dp), (a, c, ep), +1),
                        (2, (c, dp), (a, b, ep), +1),
                        (0, (a, ep), (b, c, dp), -1),
                        (1, (b, ep), (a, c, dp), -1),
                        (2, (c, ep), (a, b, dp), -1),
                    ]

                    integ_tri = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                    _tri = integ_tri.x(['mu1', 'mu2', 'phi2'], sparse=False)
                    mu1_s, mu2_s, phi2_s = (np.ravel(arr) for arr in _tri)
                    w_tri = np.ravel(integ_tri.w)

                    def _bp_oriented(side, mu1, mu2, phi2):
                        # Contracted bispectrum leg + fixed legs; for
                        # side == 2, q = k3 is the closure of coords[0],
                        # coords[1].
                        if side == 2:
                            (k1n, k2n, k3n), (k1h, k2h, k3h), (k1v, k2v, k3v) = get_kvec3(coords[0], coords[1], mu1, mu2, phi2)
                            return (k3n, k1n, k2n), (k3h, k1h, k2h), (k3v, k1v, k2v)
                        order = _pb_order(side)
                        return _oriented_kvec3(coords, order, mu1, mu2, phi2)

                    def _bp_side(side, fieldsP, fieldsB):
                        P, B = get_theory(fieldsP), get_theory(fieldsB)
                        if P is None or B is None:
                            return None
                        _fn = lambda m1, m2, p2: _bp_oriented(side, m1, m2, p2)
                        qnorms, qhats, kvecs = jax.vmap(_fn)(jnp.asarray(mu1_s), jnp.asarray(mu2_s), jnp.asarray(phi2_s))
                        qn, qh = qnorms[0], qhats[0]
                        qvec, r1vec, r2vec = kvecs
                        PB_u = P(qvec) * B(qvec, r1vec, r2vec)              # (ntri, nbins)
                        if side == 2:
                            muq = qh[..., 2]                                # (ntri, nbins)
                        else:
                            muq = jnp.broadcast_to(jnp.asarray(mu1_s)[:, None], PB_u.shape)
                        return qn, muq, PB_u

                    hat1_tri = jax.vmap(lambda m: unitvec(m, jnp.zeros_like(m)))(jnp.asarray(mu1_s))   # (ntri, 3)
                    hat2_tri = jax.vmap(unitvec)(jnp.asarray(mu2_s), jnp.asarray(phi2_s))              # (ntri, 3)

                    lo, hi = np.asarray(edgesp)[:, 0], np.asarray(edgesp)[:, 1]
                    kk, dk = np.asarray(coordsp), hi - lo
                    wS_tri = jnp.asarray(w_tri) * S(hat1_tri, hat2_tri)
                    # side == 1: literal k2 leg at (mu1, 0), k1 at
                    # (mu2, phi2) -- S's literal-leg arguments swapped (see
                    # the PB box branch).
                    wS_tri_swap = jnp.asarray(w_tri) * S(hat2_tri, hat1_tri)
                    # See the PB box branch: (2l'+1) N H^2 per Wick route (no
                    # extra 2, the six bp_terms_spec entries are the routes);
                    # radial delta W(k, q) / Ntilde_mode(k, q).
                    pref_box = (2 * ellp + 1) * get_N(*ell) * get_H(*ell)**2 / (8. * np.pi)
                    block = jnp.zeros((coords.shape[-1], coordsp.shape[-1]))
                    for (side, fieldsP, fieldsB, sign) in bp_terms_spec:
                        _side_out = _bp_side(side, fieldsP, fieldsB)
                        if _side_out is None:
                            continue
                        qn, muq, PB_u = _side_out
                        wS = wS_tri_swap if side == 1 else wS_tri
                        Lq = get_legendre(ellp)(sign * muq)             # (ntri, nbins)
                        mask = (qn[..., None] >= lo[None, None, :]) & (qn[..., None] <= hi[None, None, :])
                        qn_safe = jnp.where(qn == 0., 1., qn)
                        ntilde = 4. * np.pi * kk[None, None, :] * qn_safe[..., None] * dk[None, None, :] * volume / (2. * np.pi)**3
                        invn = mask / ntilde                            # (ntri, nbins, nbinsp)
                        block = block + pref_box * jnp.einsum('u,ua,uab->ab', wS, Lq * PB_u, invn)
                    # To host as soon as it exists. Leaving blocks as device arrays makes
                    # the final np.block do every device->host copy at once, at the moment
                    # the GPU is fullest (precompute caches still resident) -- measured: a
                    # box q = 10 run completed the whole computation and then died in
                    # np.block asking for 168 MB. Converting here also frees each block's
                    # device buffer immediately.
                    block = np.asarray(block)
                    cov[i][ip] = block
                    cov[ip][i] = block.T
                    if _timing:
                        dt = _time.time() - _tblock
                        _tblocks.append((dt, i, ip, tuple(label['fields']), label['ells'],
                                         tuple(labelp['fields']), labelp['ells'], block.shape))
                        print(f'[cov3] block ({i},{ip}) '
                              f'{len(label["fields"])}x{len(labelp["fields"])} ells '
                              f'{label["ells"]}/{labelp["ells"]} {block.shape}: {dt:.1f}s', flush=True)
                    continue

                # Symmetric PB case: the bispectrum is the FIRST observable
                # (rows), the spectrum the second (columns); Q_W is evaluated
                # as Q_W(q_i, p). See the PB block for the shared structure.
                pre_cache = cache.setdefault('bp32_ell_independent', {})
                pre_key = (fields, fieldsp,
                           np.asarray(coords).tobytes(), np.asarray(coordsp).tobytes(),
                           np.asarray(edges).tobytes(), np.asarray(edgesp).tobytes())

                if pre_key not in pre_cache:
                    pre = {}
                    P_ad, P_bd, P_cd, P_ae, P_be, P_ce = get_theory((a, dp)), get_theory((b, dp)), get_theory((c, dp)), get_theory((a, ep)), get_theory((b, ep)), get_theory((c, ep))
                    B_bce, B_ace, B_abe, B_bcd, B_acd, B_abd = get_theory((b, c, ep)), get_theory((a, c, ep)), get_theory((a, b, ep)), get_theory((b, c, dp)), get_theory((a, c, dp)), get_theory((a, b, dp))

                    mu_s, w_mu = np.asarray(integ_mu.x()), np.asarray(integ_mu.w)
                    pre['mu_s'], pre['w_mu'] = mu_s, w_mu
                    integ_tri = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                    _tri = integ_tri.x(['mu1', 'mu2', 'phi2'], sparse=False)
                    mu1_s, mu2_s, phi2_s = (np.ravel(arr) for arr in _tri)
                    w_tri = np.ravel(integ_tri.w)
                    ntri = len(w_tri)
                    nbins_bisp = coords.shape[-1]
                    pre['w_tri'] = w_tri

                    # Closure-leg cell-averaged window tables; see the PB block.
                    k3msr_c = closure_measure(coords)     # rows: (ntri * nbins)

                    def _bp_oriented(side, mu1, mu2, phi2):
                        # Return (q, r1, r2), where q is the contracted side
                        # of the first/bispectrum observable and r1, r2 are
                        # the bispectrum's own two fixed legs (used directly
                        # as B's arguments). For side == 2, q = k3 =
                        # -(k1 + k2) is the closure of the bispectrum's own
                        # k1 = coords[0], k2 = coords[1] (per the PB
                        # covariance formula: B(k1, k2, -p), with only k3
                        # contracted via the window).
                        if side == 2:
                            (k1n, k2n, k3n), (k1h, k2h, k3h), (k1v, k2v, k3v) = get_kvec3(coords[0], coords[1], mu1, mu2, phi2)
                            return (k3n, k1n, k2n), (k3h, k1h, k2h), (k3v, k1v, k2v)
                        order = _pb_order(side)
                        return _oriented_kvec3(coords, order, mu1, mu2, phi2)

                    # S_{ell1 ell2 L}(k1hat, k2hat, n): always the
                    # bispectrum's own literal first two legs and the line of
                    # sight (z3=True) -- never permuted by which leg is
                    # contracted.
                    pre['hat1'] = jax.vmap(lambda m: unitvec(m, jnp.zeros_like(m)))(jnp.asarray(mu1_s))       # (ntri, 3)
                    pre['hat2'] = jax.vmap(unitvec)(jnp.asarray(mu2_s), jnp.asarray(phi2_s))                  # (ntri, 3)

                    bp_terms = [
                        (0, P_ad, B_bce, (a, dp), (b, c, ep), +1),
                        (1, P_bd, B_ace, (b, dp), (a, c, ep), +1),
                        (2, P_cd, B_abe, (c, dp), (a, b, ep), +1),
                        (0, P_ae, B_bcd, (a, ep), (b, c, dp), -1),
                        (1, P_be, B_acd, (b, ep), (a, c, dp), -1),
                        (2, P_ce, B_abd, (c, ep), (a, b, dp), -1),
                    ]
                    pre['terms'] = []
                    for (side, P, B, fieldsP, fieldsB, sign) in bp_terms:
                        # Skip terms with no bispectrum (or power spectrum) theory, e.g. a
                        # field with no bispectrum defined (mirrors the periodic-box
                        # _bp_side guard and the BB block's None-safety).
                        if P is None or B is None:
                            continue
                        # window2 stores each (mixed-size, non-interchangeable) group's
                        # own fields sorted -- sort here to match regardless of numeric
                        # field-label order (does not affect P/B, already resolved above).
                        fieldsP, fieldsB = tuple(sorted(fieldsP)), tuple(sorted(fieldsB))
                        _fn = lambda m1, m2, p2: _bp_oriented(side, m1, m2, p2)
                        qnorms, qhats, kvecs = jax.vmap(_fn)(jnp.asarray(mu1_s), jnp.asarray(mu2_s), jnp.asarray(phi2_s))
                        qn, qh = qnorms[0], qhats[0]
                        qvec, r1vec, r2vec = kvecs
                        PB_u = P(qvec) * B(qvec, r1vec, r2vec)          # (ntri, nbins)
                        entry = {'side': side, 'sign': sign}
                        if side == 2:
                            # k3 has no native bin edges: cell-average Q_W
                            # over each node's own closure sweep (see the PB
                            # block); its khat . n is bin-dependent. Absorb
                            # L_e1(mu_q) x P x B into the table's node axis.
                            muq = qh[..., 2]                            # (ntri, nbins)
                            F_u = {e1: get_legendre(e1)(muq) * PB_u for e1 in qw_ells}
                            entry['tab'] = _qw_tables(window2, np.asarray(jnp.ravel(qn)), edgesp, fieldsP, fieldsB,
                                                      rows_shape=(ntri, nbins_bisp), F_u=F_u,
                                                      rows_measure=k3msr_c)
                        else:
                            # edges is bin-major, shape (nbins, 2, 2):
                            # axis 1 selects the leg. The contracted leg's
                            # khat . n is its own polar angle, bin-independent.
                            entry['tab'] = _qw_tables(window2, edges[:, side, :], edgesp, fieldsP, fieldsB)
                            entry['PB_u'] = PB_u
                            entry['muq'] = np.asarray(mu1_s)            # (ntri,)
                        pre['terms'].append(entry)

                    pre_cache[pre_key] = pre

                pre = pre_cache[pre_key]

                # ---- Per-(ell, ellp) assembly (cheap) ----
                mu_s, w_mu = pre['mu_s'], pre['w_mu']
                wS_tri = jnp.asarray(pre['w_tri']) * S(pre['hat1'], pre['hat2'])
                # side == 1: literal k2 leg at (mu1, 0), k1 at (mu2, phi2) --
                # S's literal-leg arguments swapped (see the PB assembly).
                wS_tri_swap = jnp.asarray(pre['w_tri']) * S(pre['hat2'], pre['hat1'])
                # Mirror of the PB normalization (Cov^BP is the PB
                # transpose): (2l'+1) N H^2 on normalized measures, /2 for
                # the spectrum side and /(8 pi) for the triangle side; the
                # reference's leading 2 (the two Wick routes) is already
                # enumerated explicitly in the terms.
                prefBP = (2 * ellp + 1) * get_N(*ell) * get_H(*ell)**2 / (2. * 8. * np.pi)
                block = 0.
                for entry in pre['terms']:
                    tab, sign = entry['tab'], entry['sign']
                    wS = wS_tri_swap if entry['side'] == 1 else wS_tri
                    for e2 in qw_ells:
                        # Spectrum-side scalar: sum_mu w L_ellp(mu) L_e2(sign mu).
                        rsc = np.sum(w_mu * legp(mu_s) * get_legendre(e2)(sign * mu_s))
                        if entry['side'] == 2:
                            block = block + prefBP * rsc * jnp.einsum('uab,u->ab', tab['e2', e2], wS)
                        else:
                            for e1 in qw_ells:
                                rvec = (wS * get_legendre(e1)(jnp.asarray(entry['muq']))) @ entry['PB_u']   # (nbins,)
                                block = block + prefBP * rsc * tab['e12', e1, e2] * rvec[:, None]

            # BB block
            elif nfields == 3 and nfieldsp == 3:

                a, b, c = fields
                ap, bp, cp = fieldsp

                S, Sp = get_S(ell, z3=True), get_S(ellp, z3=True)
                M = get_N(*ell) * get_N(*ellp) * get_H(*ell)**2 * get_H(*ellp)**2

                if not use_window_kernels:
                    # Periodic (box) reference, following
                    # _cov3_math_periodic.tex ("Further simplification of the
                    # unconnected parts" + the appendix with the full
                    # permutations): the Gaussian PPP term (6 perms, two
                    # radial deltas each), and the single-delta BB and PT
                    # terms (9 perms each, one radial delta tying unprimed
                    # leg i to primed leg j: k_i = k'_j for BB, k_i = -k'_j
                    # for PT). Only the P6 term is omitted (as in the
                    # windowed path).
                    P_aap, P_abp, P_acp = get_theory((a, ap)), get_theory((a, bp)), get_theory((a, cp))
                    P_bap, P_bbp, P_bcp = get_theory((b, ap)), get_theory((b, bp)), get_theory((b, cp))
                    P_cap, P_cbp, P_ccp = get_theory((c, ap)), get_theory((c, bp)), get_theory((c, cp))

                    # Angular quadrature parameterization. COV3_QUAD_PARAM='mu1xphi' carries the triangle
                    # shape x = k1hat.k2hat as a DIRECT integration variable (see get_kvec3_x); the default
                    # 'mu1mu2phi2' derives it from all three angles, which smears the sharp 1/k3 structure at
                    # the folded configuration and is what limits B convergence in q.
                    if _quad_param in ('mu1xphi', 'mu1xphi_exact'):
                        integ_tri = IntegralND(mu1=integ_mu, x=integ_x, phi=integ_phi)
                        _tri = integ_tri.x(['mu1', 'x', 'phi'], sparse=False)
                        _kvec3 = get_kvec3_x
                    else:
                        integ_tri = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                        _tri = integ_tri.x(['mu1', 'mu2', 'phi2'], sparse=False)
                        _kvec3 = get_kvec3
                    mu1_s, mu2_s, phi2_s = (np.ravel(arr) for arr in _tri)
                    w_side = np.ravel(integ_tri.w)

                    _fn = lambda m1, m2, p2: _kvec3(coords[0], coords[1], m1, m2, p2)
                    (k1n_u, k2n_u, k3n_u), (k1h_u, k2h_u, k3h_u), kv_u = jax.vmap(_fn)(jnp.asarray(mu1_s), jnp.asarray(mu2_s), jnp.asarray(phi2_s))
                    # Primed triangle on its own grid: closure magnitudes for
                    # the deltas that tie an unprimed leg to the primed
                    # closure leg k'_3, and the full legs/hats for the BB/PT
                    # terms anchored on the primed side.
                    _fnp = lambda m1, m2, p2: _kvec3(coordsp[0], coordsp[1], m1, m2, p2)
                    (k1n_p, k2n_p, k3n_p), (k1h_p, k2h_p, k3h_p), kv_p = jax.vmap(_fnp)(jnp.asarray(mu1_s), jnp.asarray(mu2_s), jnp.asarray(phi2_s))
                    hat_p = (jnp.broadcast_to(k1h_p[:, None, :], k3h_p.shape),
                             jnp.broadcast_to(k2h_p[:, None, :], k3h_p.shape),
                             k3h_p)

                    hat_u = (jnp.broadcast_to(k1h_u[:, None, :], k3h_u.shape),
                             jnp.broadcast_to(k2h_u[:, None, :], k3h_u.shape),
                             k3h_u)
                    S_u = S(hat_u[0], hat_u[1])                            # (nside, nbins)

                    ppp_terms = [
                        ((P_aap, P_bbp, P_ccp), (0, 1)),
                        ((P_aap, P_bcp, P_cbp), (0, 2)),
                        ((P_abp, P_bap, P_ccp), (1, 0)),
                        ((P_abp, P_bcp, P_cap), (1, 2)),
                        ((P_acp, P_bap, P_cbp), (2, 0)),
                        ((P_acp, P_bbp, P_cap), (2, 1)),
                    ]

                    lo_u = [np.asarray(edges)[:, m, :] for m in range(2)]  # unprimed leg bins
                    dk_u = [l[:, 1] - l[:, 0] for l in lo_u]

                    # Reference (periodic) conventions, cf.
                    # _cov3_math_periodic.tex, "Further simplification of the
                    # unconnected parts": Cov_PPP = N N' H^2 H'^2 V x
                    # [normalized triangle measure] x
                    # prod_m W(k_m, k'_m) / Ntilde_mode(k_m, k'_m) x P P P,
                    # with Ntilde_mode(k, k') = 4 pi k k' dk V / (2 pi)^3.
                    # 1/(8 pi): the unprimed normalized triangle measure (raw
                    # weights sum to 8 pi); the primed measure is consumed by
                    # the angular deltas.
                    norm_box = get_N(*ell) * get_N(*ellp) * get_H(*ell)**2 * get_H(*ellp)**2 * volume / (8. * np.pi)
                    # Per-term accumulators (PPP / BB / PT), summed at the
                    # end; kept separate for the COV3_BOX_DEBUG breakdown.
                    parts = {'ppp': 0., 'bb': 0., 'pt': 0.}
                    block = 0.
                    _xsub_cache = {}
                    for (Ps, (s1, s2)) in ppp_terms:
                        # As for the BB / PT families: skip when the theory
                        # provides no power spectrum (e.g. B-only theory).
                        if any(P is None for P in Ps):
                            continue
                        # Debug knob: restrict to arm-only ((0,1)/(1,0)) or
                        # closure-only (a primed/unprimed closure leg is tied)
                        # pairings, to compare against the windowed path.
                        _sel = os.environ.get('COV3_PPP_TERMS')
                        if _sel and (('closure' in _sel) != (2 in (s1, s2))):
                            continue
                        # sigma[m]: which primed leg is tied to unprimed leg
                        # m by the deltas (m = 0, 1 explicitly; leg 2 by
                        # closure); siginv: the unprimed leg each primed leg
                        # is tied to.
                        sigma = {0: s1, 1: s2, 2: 3 - s1 - s2}
                        siginv = {v: m for m, v in sigma.items()}
                        if _x_exact and 2 in (s1, s2):
                            # ---- PIECEWISE-EXACT x integration for the closure-tied perms ----
                            # The x-dependence of this perm's radial delta is a STEP: the tied
                            # primed closure magnitude k3'(x) = sqrt(k1'^2 + k2'^2 + 2 k1' k2' x)
                            # is masked against the unprimed leg-m0 bin edges, and sampling that
                            # step with a global Gauss-Legendre x grid is what makes B000
                            # converge non-monotonically (measured: 0.982, 0.980, 1.015, 0.989
                            # at qx = 16, 24, 32, 48). Since k3'(x) is monotone in x, the edges
                            # invert to explicit breakpoints
                            #     x* = (k3_edge^2 - k1'^2 - k2'^2) / (2 k1' k2'),
                            # and for a FIXED primed bin b those breakpoints partition [-1, 1]
                            # into sub-intervals on each of which exactly one unprimed bin a is
                            # selected -- the mask is constant there, i.e. EXACT. Gauss-Legendre
                            # is then applied per sub-interval to the remaining smooth integrand
                            # (S weights, P's, 1/Ntilde), and the result scattered into (a, b).
                            # Everything else (mu1, phi axes; the two non-closure perms; the
                            # BB / PT tie families) keeps the shared grid.
                            m0 = 0 if s1 == 2 else 1
                            m1 = 1 - m0
                            sprim = sigma[m1]
                            if m0 not in _xsub_cache:
                                _edg = np.asarray(edges)[:, m0, :]
                                _k1p = np.asarray(coordsp[0]); _k2p = np.asarray(coordsp[1])
                                _den = (2. * _k1p * _k2p)[:, None, None]
                                _xed = (_edg[None, :, :]**2 - _k1p[:, None, None]**2 - _k2p[:, None, None]**2) / _den
                                _xlo = np.clip(_xed[..., 0], -1., 1.); _xhi = np.clip(_xed[..., 1], -1., 1.)
                                _ok = _xhi > _xlo + 1e-12
                                _nbp = len(_k1p)
                                _smax = max(int(_ok.sum(axis=1).max()), 1)
                                _ai = np.zeros((_nbp, _smax), int)
                                _lo = np.zeros((_nbp, _smax)); _hi = np.zeros((_nbp, _smax)); _vv = np.zeros((_nbp, _smax))
                                for _b in range(_nbp):
                                    _aa = np.where(_ok[_b])[0]
                                    _ai[_b, :len(_aa)] = _aa; _lo[_b, :len(_aa)] = _xlo[_b, _aa]
                                    _hi[_b, :len(_aa)] = _xhi[_b, _aa]; _vv[_b, :len(_aa)] = 1.
                                _uj, _wj = np.polynomial.legendre.leggauss(int(os.environ.get('COV3_X_SUB', 6)))
                                _xn = 0.5 * (_hi - _lo)[..., None] * _uj + 0.5 * (_hi + _lo)[..., None]
                                _wx = 0.5 * (_hi - _lo)[..., None] * _wj * _vv[..., None]
                                _xsub_cache[m0] = (jnp.asarray(_ai), jnp.asarray(_xn), jnp.asarray(_wx))
                            a_i, xn, wx = _xsub_cache[m0]
                            nbp, smax, jx = xn.shape
                            nbins_u = coords.shape[-1]
                            mu1g, wmu1g = jnp.asarray(integ_mu.x()), jnp.asarray(integ_mu.w)
                            phig, wphig = jnp.asarray(integ_phi.x()), jnp.asarray(integ_phi.w)
                            k1u_a = jnp.asarray(coords[0])[a_i]; k2u_a = jnp.asarray(coords[1])[a_i]
                            shp = (len(mu1g), len(phig), nbp, smax, jx)
                            mu1B = jnp.broadcast_to(mu1g[:, None, None, None, None], shp)
                            phiB = jnp.broadcast_to(phig[None, :, None, None, None], shp)
                            xB = jnp.broadcast_to(xn[None, None], shp)
                            k1B = jnp.broadcast_to(k1u_a[None, None, :, :, None], shp)
                            k2B = jnp.broadcast_to(k2u_a[None, None, :, :, None], shp)
                            _, hh, vh = get_kvec3_x(k1B, k2B, mu1B, xB, phiB)
                            F = S(hh[0], hh[1]) * get_S(ellp, z3=True)(hh[siginv[0]], hh[siginv[1]])
                            for m in range(3):
                                F = F * Ps[m](vh[m])
                            k1pj = jnp.asarray(coordsp[0])[:, None, None]; k2pj = jnp.asarray(coordsp[1])[:, None, None]
                            k3p = jnp.sqrt(k1pj**2 + k2pj**2 + 2. * k1pj * k2pj * xn)                 # (nbp, smax, jx)
                            nt0 = (4. * np.pi * jnp.asarray(coords[m0])[a_i][..., None] * k3p
                                   * jnp.asarray(dk_u[m0])[a_i][..., None] * volume / (2. * np.pi)**3)
                            core = jnp.einsum('i,k,bsj,ikbsj->bs', wmu1g, wphig, wx, F / nt0[None, None])
                            kpv1 = jnp.asarray(coordsp[sprim])[:, None]
                            m1ed = jnp.asarray(lo_u[m1])[a_i]                                          # (nbp, smax, 2)
                            mask1 = (kpv1 >= m1ed[..., 0]) & (kpv1 <= m1ed[..., 1])
                            nt1 = (4. * np.pi * jnp.asarray(coords[m1])[a_i] * kpv1
                                   * jnp.asarray(dk_u[m1])[a_i] * volume / (2. * np.pi)**3)
                            core = core * mask1 / nt1
                            flat = (a_i * nbp + jnp.arange(nbp)[:, None]).ravel()
                            contrib = jnp.zeros(nbins_u * nbp).at[flat].add(core.ravel()).reshape(nbins_u, nbp)
                            if os.environ.get('COV3_BOX_DEBUG'):
                                print(f"box33 ppp exact-x perm ({s1},{s2}): max = {np.abs(np.asarray(norm_box * contrib)).max():.3e}")
                            parts['ppp'] = parts['ppp'] + norm_box * contrib
                            continue
                        Sp_u = get_S(ellp, z3=True)(hat_u[siginv[0]], hat_u[siginv[1]])   # (nside, nbins)
                        # Reference: P^N(k1) P^N(k2) P^N(k3), all evaluated
                        # on the unprimed triangle's own legs (the deltas tie
                        # the primed legs to these exactly).
                        prod = jnp.asarray(w_side)[:, None] * S_u * Sp_u
                        for m in range(3):
                            prod = prod * Ps[m](kv_u[m])
                        # Radial deltas on the two explicit legs: unprimed
                        # leg m (row bins) against the tied primed leg
                        # sigma(m) (columns): primed leg < 2 -> its bin
                        # value; primed leg 2 -> its closure magnitude at the
                        # tied orientation (node-dependent). Each delta:
                        # W(k_m, k') / Ntilde_mode(k_m, k').
                        D = 1.
                        for m in range(2):
                            sigm = sigma[m]
                            if sigm < 2:
                                kpv = jnp.broadcast_to(jnp.asarray(coordsp[sigm])[None, None, :], (len(w_side), lo_u[m].shape[0], coordsp.shape[-1]))
                            else:
                                kpv = jnp.broadcast_to(k3n_p[:, None, :], (len(w_side), lo_u[m].shape[0], coordsp.shape[-1]))
                            mask = (kpv >= lo_u[m][None, :, 0, None]) & (kpv <= lo_u[m][None, :, 1, None])
                            kpv_safe = jnp.where(kpv == 0., 1., kpv)
                            ntilde = 4. * np.pi * jnp.asarray(coords[m])[None, :, None] * kpv_safe * dk_u[m][None, :, None] * volume / (2. * np.pi)**3
                            D = D * mask / ntilde   # (nside, nbins, nbinsp)
                        parts['ppp'] = parts['ppp'] + norm_box * jnp.einsum('ua,uab->ab', prod, D)

                    # ---- Single-delta BB and PT terms (appendix, 9 perms
                    # each) ----
                    # Tie: unprimed leg li = s k'_lj (s = +1 BB, -1 PT). The
                    # anchored triangle runs the full (mu1, mu2, phi2) grid;
                    # the other triangle's tied leg is the shared vector and
                    # its remaining explicit leg runs its own 2D orientation
                    # grid, its closure following. Radial delta:
                    # W(k, k') / Ntilde_mode(k, k'). Terms with lj < 2 (and
                    # the double-closure (2, 2) term, rewritten on the primed
                    # leg 1 via k'_1 = -s k_3 - k'_2) anchor on the unprimed
                    # side; ties to the primed closure (li < 2, lj = 2)
                    # anchor on the primed side, mirrored.
                    f, fp = fields, fieldsp
                    r1, r2 = (1, 2, 0), (2, 0, 1)
                    nu = len(w_side)
                    nbins, nbinsp = coords.shape[-1], coordsp.shape[-1]

                    integ_2d = IntegralND(mu=integ_mu, phi=integ_phi)
                    _g2 = integ_2d.x(['mu', 'phi'], sparse=False)
                    muq_s, phiq_s = (np.ravel(arr) for arr in _g2)
                    w_2d = np.ravel(integ_2d.w)                                          # raw sum 4 pi
                    # Offset the free-leg azimuth grid (valid for a
                    # 2 pi-periodic integrand): with the same (mu, phi) nodes
                    # as the anchored triangle grid, nodes with mu' = -mu,
                    # phi' = phi + pi and equal bin magnitudes make internal
                    # trispectrum pair-sum momenta (e.g. k + k') cancel to
                    # machine zero -- an unguarded squeezed-T configuration
                    # that then dominates the sum by ~1e13.
                    phiq_s = phiq_s + np.pi / 17.
                    dir2 = jax.vmap(unitvec)(jnp.asarray(muq_s), jnp.asarray(phiq_s))    # (n2, 3)
                    n2 = len(w_2d)

                    Sell, Sellp = get_S(ell, z3=True), get_S(ellp, z3=True)
                    # M x [w_side / (8 pi)] x [w_2d / (4 pi)]: the appendix
                    # measure d cos(theta_1)/2 x dOmega_2/(4 pi) on the
                    # anchored side and dOmega'/(4 pi) on the free leg.
                    norm_tie = get_N(*ell) * get_N(*ellp) * get_H(*ell)**2 * get_H(*ellp)**2 / (32. * np.pi**2)
                    w_u, w_q = jnp.asarray(w_side), jnp.asarray(w_2d)
                    nchunk = 8

                    # IR cutoff on the trispectrum's off-shell internal
                    # momenta (see make_pt_qmask): half the smallest k-bin
                    # edge.
                    # Floored at half the smallest bin width: binnings extending to k ~ 0
                    # make the smallest edge (hence the cutoff) collapse, leaving the
                    # squeezed-T degeneracies unmasked.
                    _pt_qmin = max(0.5 * min(np.min(edges), np.min(edgesp)),
                                   0.5 * min(np.min(np.asarray(edges)[..., 1] - np.asarray(edges)[..., 0]),
                                             np.min(np.asarray(edgesp)[..., 1] - np.asarray(edgesp)[..., 0])))
                    _pt_qmask = make_pt_qmask(_pt_qmin)

                    def _delta_D(vmag, spec_edges, spec_coords, spec_dk):
                        # W(vmag, k'_bin) / Ntilde_mode on the paired bins of
                        # the non-anchored observable; vmag (..., ) ->
                        # (..., nbins_other).
                        mask = (vmag[..., None] >= spec_edges[:, 0]) & (vmag[..., None] <= spec_edges[:, 1])
                        vsafe = jnp.where(vmag == 0., 1., vmag)
                        ntilde = 4. * np.pi * vsafe[..., None] * spec_coords * spec_dk * volume / (2. * np.pi)**3
                        return mask / ntilde

                    def _bcast(v, shape):
                        return jnp.broadcast_to(v, shape + (3,))

                    # (a) lj < 2, anchored on the unprimed side. li < 2
                    # (bin-native legs) use the standard mask/Ntilde radial
                    # delta below (constant across nodes, no discontinuity).
                    # li = 2 (closure leg) is handled separately further
                    # down via an exact angular substitution (see
                    # _cov3_math.tex, "Periodic (box) closure-leg tie"):
                    # k3 is a nonlinear function of the shape angles, and a
                    # post-hoc mask on it was found to be badly
                    # under-resolved by the shared (mu1, mu2, phi2) grid
                    # (order-of-magnitude, sign-flipping disagreement
                    # between the two ways of computing an i != ip
                    # cross-multipole block, non-convergent with
                    # quadrature order).
                    def _tie_keep(family, is_closure):
                        # Debug knob (see the windowed branch): COV3_BB_TERMS /
                        # COV3_PT_TERMS = 'arm' | 'closure' restrict the tie
                        # families; unset keeps everything.
                        _sel = os.environ.get('COV3_BB_TERMS' if family == 'bb' else 'COV3_PT_TERMS')
                        return (not _sel) or (('closure' in _sel) == bool(is_closure))

                    for li in range(2):
                        for lj in range(2):
                            ljf = 1 - lj
                            kf = jnp.asarray(coordsp[ljf])[None, :, None] * dir2[:, None, :]   # (n2, nbinsp, 3)
                            vmag = jnp.broadcast_to(jnp.asarray(coords[li])[None, :], (nu, nbins))
                            D = _delta_D(vmag, np.asarray(edgesp)[:, lj, :], jnp.asarray(coordsp[lj]),
                                         jnp.asarray(edgesp[:, lj, 1] - edgesp[:, lj, 0]))     # (nu, nbins, nbinsp)
                            for (s, family) in ((1., 'bb'), (-1., 'pt')):
                                if not _tie_keep(family, False):
                                    continue
                                if family == 'bb':
                                    Bu_f = get_theory((f[li], f[r1[li]], f[r2[li]]))
                                    Bp_f = get_theory((fp[lj], fp[r1[lj]], fp[r2[lj]]))
                                    if Bu_f is None or Bp_f is None:
                                        continue
                                    Au = Bu_f(kv_u[li], kv_u[r1[li]], kv_u[r2[li]])            # (nu, nbins)
                                else:
                                    Pu_f = get_theory((f[li], fp[lj]))
                                    T_f = get_theory((f[r1[li]], f[r2[li]], fp[r1[lj]], fp[r2[lj]]))
                                    if Pu_f is None or T_f is None:
                                        continue
                                    Au = Pu_f(kv_u[li])                                        # (nu, nbins)
                                hj = _bcast((s * hat_u[li])[:, :, None, :], (nu, nbins, n2))
                                hjf = _bcast(dir2[None, None, :, :], (nu, nbins, n2))
                                Sp_g = Sellp(hj, hjf) if lj == 0 else Sellp(hjf, hj)           # (nu, nbins, n2)
                                wA = w_u[:, None] * S_u * Au                                   # (nu, nbins)
                                block_t = 0.
                                for sl in [slice(c, c + (nu + nchunk - 1) // nchunk) for c in range(0, nu, (nu + nchunk - 1) // nchunk)]:
                                    nc = len(range(*sl.indices(nu)))
                                    shp = (nc, nbins, n2, nbinsp)
                                    tied = _bcast((s * kv_u[li][sl])[:, :, None, None, :], shp)
                                    free = _bcast(kf[None, None, :, :, :], shp)
                                    clos = -(s * kv_u[li][sl])[:, :, None, None, :] - kf[None, None, :, :, :]
                                    legs = {lj: tied, ljf: free, 2: _bcast(clos, shp)}
                                    if family == 'bb':
                                        Bp = Bp_f(legs[lj], legs[r1[lj]], legs[r2[lj]])        # (nc, nbins, n2, nbinsp)
                                    else:
                                        Ta1 = _bcast(kv_u[r1[li]][sl][:, :, None, None, :], shp)
                                        Ta2 = _bcast(kv_u[r2[li]][sl][:, :, None, None, :], shp)
                                        Bp = T_f(Ta1, Ta2, legs[r1[lj]], legs[r2[lj]]) * _pt_qmask(Ta1, Ta2, legs[r1[lj]], legs[r2[lj]])
                                    block_t = block_t + jnp.einsum('q,ua,uaq,uab,uaqb->ab', w_q, wA[sl], Sp_g[sl], D[sl], Bp)
                                if os.environ.get('COV3_BOX_DEBUG'):
                                    print(f"box33 (a) {family} li={li} lj={lj}: max|term| = {np.abs(np.asarray(norm_tie * block_t)).max():.3e}")
                                parts[family] = parts[family] + norm_tie * block_t

                    # (a, li = 2) unprimed closure leg tied to primed lj < 2:
                    # exact mu12 substitution (_cov3_math.tex, "Periodic
                    # (box) closure-leg tie"). mu1, mu2 keep their original,
                    # shared quadrature; only the phi2 direction (along
                    # which k3's threshold crossing was sampled
                    # discontinuously) is replaced by a Gauss-Legendre grid
                    # in mu12 = khat1.khat2, built exactly over the
                    # sub-interval that lands k3 in each (row, column) bin
                    # pair, summed over the two phi2 = +-arccos(...) branches.
                    mu1_c, w_mu1_c = np.asarray(integ_mu.x()), np.asarray(integ_mu.w)
                    mu2_c, w_mu2_c = mu1_c, w_mu1_c
                    xi_c, w_xi_c = mu1_c, w_mu1_c   # reuse the same base [-1, 1] rule for mu12
                    nxc = len(xi_c)
                    k1v, k2v = np.asarray(coords[0]), np.asarray(coords[1])   # (nbins,)

                    for lj in range(2):
                        ljf = 1 - lj
                        lo_p, hi_p = np.asarray(edgesp)[:, lj, 0], np.asarray(edgesp)[:, lj, 1]   # (nbinsp,)
                        dkp_lj = hi_p - lo_p
                        coordp_lj = np.asarray(coordsp[lj])                                       # (nbinsp,)
                        coordp_ljf = np.asarray(coordsp[ljf])                                     # (nbinsp,)
                        k1v2, k2v2 = k1v[:, None], k2v[:, None]
                        denom = 2. * k1v2 * k2v2
                        mu12_lo = np.clip((lo_p[None, :]**2 - k1v2**2 - k2v2**2) / denom, -1., 1.)
                        mu12_hi = np.clip((hi_p[None, :]**2 - k1v2**2 - k2v2**2) / denom, -1., 1.)
                        valid_ab = mu12_hi > mu12_lo                                  # (nbins, nbinsp)
                        half = 0.5 * (mu12_hi - mu12_lo)                              # (nbins, nbinsp)
                        mu12_nodes = mu12_lo[..., None] + (xi_c[None, None, :] + 1.) * half[..., None]   # (nbins, nbinsp, nxc)

                        for (s, family) in ((1., 'bb'), (-1., 'pt')):
                            if not _tie_keep(family, True):
                                continue
                            if family == 'bb':
                                Bu_f = get_theory((f[2], f[0], f[1]))
                                Bp_f = get_theory((fp[lj], fp[r1[lj]], fp[r2[lj]]))
                                if Bu_f is None or Bp_f is None:
                                    continue
                            else:
                                Pu_f = get_theory((f[2], fp[lj]))
                                T_f = get_theory((f[0], f[1], fp[r1[lj]], fp[r2[lj]]))
                                if Pu_f is None or T_f is None:
                                    continue

                            # Vectorized over (m1, m2, branch) via jax.vmap: replaces a
                            # 6x6x2 (mu1_c x mu2_c x branch) nested Python loop -- each
                            # iteration previously dispatching its own small JAX ops --
                            # with a single vmapped call. A < 1e-9 (measure-zero mu1 or
                            # mu2 = +-1) is now a multiplicative mask instead of a
                            # `continue`, mathematically equivalent since those nodes'
                            # contribution is discarded either way.
                            def _node_fn(m1, wm1, m2, wm2, branch):
                                s1 = jnp.sqrt(jnp.clip(1. - m1**2, 0., None))
                                s2 = jnp.sqrt(jnp.clip(1. - m2**2, 0., None))
                                k1hat = jnp.stack([s1, jnp.zeros_like(s1), m1])
                                k1vec_row = k1v[:, None] * k1hat[None, :]                  # (nbins, 3)
                                A = s1 * s2
                                ok_A = A > 1e-9
                                A_safe = jnp.where(ok_A, A, 1.)

                                delta = A_safe**2 - (mu12_nodes - m1 * m2)**2              # (nbins, nbinsp, nxc)
                                ok = valid_ab[..., None] & (delta > 0.) & ok_A
                                delta_safe = jnp.where(ok, delta, 1.)
                                cosphi = (mu12_nodes - m1 * m2) / A_safe
                                jac = jnp.where(ok, (w_xi_c[None, None, :] * half[..., None]) / jnp.sqrt(delta_safe), 0.)   # (nbins, nbinsp, nxc)
                                wmu = wm1 * wm2

                                sinphi = branch * jnp.sqrt(delta_safe) / A_safe
                                k2hat = jnp.stack([s2 * cosphi, s2 * sinphi, jnp.full_like(cosphi, m2)], axis=-1)   # (nbins, nbinsp, nxc, 3)
                                k2vec = k2v[:, None, None, None] * k2hat                                          # (nbins, nbinsp, nxc, 3)
                                k1vec_b = jnp.broadcast_to(k1vec_row[:, None, None, :], k2vec.shape)
                                k3vec = -(k1vec_b + k2vec)
                                k3mag = jnp.sqrt(jnp.sum(k3vec**2, axis=-1))
                                k3mag_safe = jnp.where(k3mag == 0., 1., k3mag)
                                khat3 = k3vec / k3mag_safe[..., None]

                                # Radial delta on the tied leg, W(k3, k'_lj) / Ntilde_mode(k3, k'_lj),
                                # evaluated at each node's own exact k3 (mirrors _delta_D); bin
                                # membership is already exact by construction (mu12 clipping), so
                                # only the 1/Ntilde_mode weighting is new relative to that helper.
                                ntilde = 4. * np.pi * k3mag_safe * coordp_lj[None, :, None] * dkp_lj[None, :, None] * volume / (2. * np.pi)**3
                                Dtie = jnp.asarray(ok) / ntilde                                                    # (nbins, nbinsp, nxc)

                                k1hat_b = jnp.broadcast_to(k1hat[None, None, None, :], k2vec.shape)
                                S_here = Sell(k1hat_b, k2hat)                                                      # (nbins, nbinsp, nxc)

                                if family == 'bb':
                                    Au = Bu_f(k3vec, k1vec_b, k2vec)                                              # (nbins, nbinsp, nxc)
                                else:
                                    Au = Pu_f(k3vec)

                                wA = wmu * S_here * Au * jac * Dtie                                                # (nbins, nbinsp, nxc)

                                block_node = 0.
                                for q0 in range(0, n2, max(n2 // nchunk, 1)):
                                    qsl = slice(q0, min(q0 + max(n2 // nchunk, 1), n2))
                                    nqc = qsl.stop - qsl.start
                                    shp = (nbins, nbinsp, nxc, nqc)
                                    # FULL primed-leg vectors (magnitude x direction), not
                                    # bare unit directions: the tied leg is exactly
                                    # s k3vec, the free leg its own bin-center magnitude
                                    # (unit vectors left B'/T evaluated at |k| ~ 1,
                                    # suppressing these ties by orders of magnitude for
                                    # any scale-dependent theory).
                                    tied_v = (s * k3vec)[..., None, :]
                                    free_v = jnp.asarray(coordp_ljf)[None, :, None, None, None] * dir2[None, None, None, qsl, :]
                                    tied = _bcast(tied_v, shp)
                                    free = jnp.broadcast_to(free_v, shp + (3,))
                                    clos = -tied - free
                                    legs = {lj: tied, ljf: free, 2: clos}
                                    if family == 'bb':
                                        Bp = Bp_f(legs[lj], legs[r1[lj]], legs[r2[lj]])                          # (nbins, nbinsp, nxc, nqc)
                                    else:
                                        Ta1 = _bcast(k1vec_row[:, None, None, None, :], shp)
                                        Ta2 = _bcast(k2vec[..., None, :], shp)
                                        Bp = T_f(Ta1, Ta2, legs[r1[lj]], legs[r2[lj]]) * _pt_qmask(Ta1, Ta2, legs[r1[lj]], legs[r2[lj]])
                                    block_node = block_node + jnp.einsum('q,abn,abnq->ab', w_q[qsl], wA, Bp)
                                return block_node                                                                  # (nbins, nbinsp)

                            branch_c = np.array([1., -1.])
                            m1_flat, m2_flat, branch_flat = (jnp.asarray(v.ravel()) for v in
                                                             np.meshgrid(mu1_c, mu2_c, branch_c, indexing='ij'))
                            wm1_flat, wm2_flat, _ = (jnp.asarray(v.ravel()) for v in
                                                     np.meshgrid(w_mu1_c, w_mu2_c, branch_c, indexing='ij'))

                            block_t = jax.vmap(_node_fn)(m1_flat, wm1_flat, m2_flat, wm2_flat, branch_flat).sum(axis=0)

                            if os.environ.get('COV3_BOX_DEBUG'):
                                print(f"box33 (a) mu12 {family} lj={lj}: diag = {np.array2string(np.diag(np.asarray(norm_tie * block_t)), precision=3)}")
                            parts[family] = parts[family] + norm_tie * block_t

                    # (b) li < 2 tied to the primed closure (lj = 2),
                    # anchored on the primed side: exact mirror of
                    # (a, li = 2) -- primed legs 0, 1 keep their shared
                    # (mu1', mu2') quadrature, and the primed closure's
                    # direction is targeted directly at each unprimed bin li
                    # via mu12' = khat'1.khat'2 (_cov3_math.tex, "Periodic
                    # (box) closure-leg tie").
                    k1pv, k2pv = np.asarray(coordsp[0]), np.asarray(coordsp[1])   # (nbinsp,)
                    for li in range(2):
                        lif = 1 - li
                        lo_u, hi_u = np.asarray(edges)[:, li, 0], np.asarray(edges)[:, li, 1]   # (nbins,)
                        dku_li = hi_u - lo_u
                        coordu_li = np.asarray(coords[li])                                       # (nbins,)
                        coordu_lif = np.asarray(coords[lif])                                     # (nbins,)
                        k1pv2, k2pv2 = k1pv[:, None], k2pv[:, None]
                        denomp = 2. * k1pv2 * k2pv2
                        mu12p_lo = np.clip((lo_u[None, :]**2 - k1pv2**2 - k2pv2**2) / denomp, -1., 1.)
                        mu12p_hi = np.clip((hi_u[None, :]**2 - k1pv2**2 - k2pv2**2) / denomp, -1., 1.)
                        valid_ba = mu12p_hi > mu12p_lo                                  # (nbinsp, nbins)
                        halfp = 0.5 * (mu12p_hi - mu12p_lo)                             # (nbinsp, nbins)
                        mu12p_nodes = mu12p_lo[..., None] + (xi_c[None, None, :] + 1.) * halfp[..., None]   # (nbinsp, nbins, nxc)

                        for (s, family) in ((1., 'bb'), (-1., 'pt')):
                            if not _tie_keep(family, True):
                                continue
                            if family == 'bb':
                                Bp_f = get_theory((fp[2], fp[0], fp[1]))
                                Bu_f = get_theory((f[li], f[r1[li]], f[r2[li]]))
                                if Bu_f is None or Bp_f is None:
                                    continue
                            else:
                                Pu_f = get_theory((f[li], fp[2]))
                                T_f = get_theory((f[r1[li]], f[r2[li]], fp[0], fp[1]))
                                if Pu_f is None or T_f is None:
                                    continue

                            # Vectorized over (m1, m2, branch) via jax.vmap; see the mirror
                            # comment in section (a) above.
                            def _node_fn(m1, wm1, m2, wm2, branch):
                                s1 = jnp.sqrt(jnp.clip(1. - m1**2, 0., None))
                                s2 = jnp.sqrt(jnp.clip(1. - m2**2, 0., None))
                                k1phat = jnp.stack([s1, jnp.zeros_like(s1), m1])
                                k1pvec_row = k1pv[:, None] * k1phat[None, :]                  # (nbinsp, 3)
                                A = s1 * s2
                                ok_A = A > 1e-9
                                A_safe = jnp.where(ok_A, A, 1.)

                                delta = A_safe**2 - (mu12p_nodes - m1 * m2)**2               # (nbinsp, nbins, nxc)
                                ok = valid_ba[..., None] & (delta > 0.) & ok_A
                                delta_safe = jnp.where(ok, delta, 1.)
                                cosphi = (mu12p_nodes - m1 * m2) / A_safe
                                jac = jnp.where(ok, (w_xi_c[None, None, :] * halfp[..., None]) / jnp.sqrt(delta_safe), 0.)   # (nbinsp, nbins, nxc)
                                wmu = wm1 * wm2

                                sinphi = branch * jnp.sqrt(delta_safe) / A_safe
                                k2phat = jnp.stack([s2 * cosphi, s2 * sinphi, jnp.full_like(cosphi, m2)], axis=-1)   # (nbinsp, nbins, nxc, 3)
                                k2pvec = k2pv[:, None, None, None] * k2phat                                          # (nbinsp, nbins, nxc, 3)
                                k1pvec_b = jnp.broadcast_to(k1pvec_row[:, None, None, :], k2pvec.shape)
                                k3pvec = -(k1pvec_b + k2pvec)
                                k3pmag = jnp.sqrt(jnp.sum(k3pvec**2, axis=-1))
                                k3pmag_safe = jnp.where(k3pmag == 0., 1., k3pmag)
                                khat3p = k3pvec / k3pmag_safe[..., None]

                                # Radial delta on the tied leg (mirrors (a, li=2)):
                                # W(k3', k_li) / Ntilde_mode(k3', k_li), evaluated at
                                # each node's own exact k3'; bin membership is already
                                # exact by construction (mu12' clipping).
                                ntilde = 4. * np.pi * k3pmag_safe * coordu_li[None, :, None] * dku_li[None, :, None] * volume / (2. * np.pi)**3
                                Dtie = jnp.asarray(ok) / ntilde                                                    # (nbinsp, nbins, nxc)

                                k1phat_b = jnp.broadcast_to(k1phat[None, None, None, :], k2pvec.shape)
                                Sp_here = Sellp(k1phat_b, k2phat)                                                  # (nbinsp, nbins, nxc)

                                if family == 'bb':
                                    Ap = Bp_f(k3pvec, k1pvec_b, k2pvec)                                            # (nbinsp, nbins, nxc)
                                else:
                                    Ap = Pu_f(k3pvec)

                                wA = wmu * Sp_here * Ap * jac * Dtie                                               # (nbinsp, nbins, nxc)

                                block_node = 0.
                                for q0 in range(0, n2, max(n2 // nchunk, 1)):
                                    qsl = slice(q0, min(q0 + max(n2 // nchunk, 1), n2))
                                    nqc = qsl.stop - qsl.start
                                    shp = (nbinsp, nbins, nxc, nqc)
                                    # FULL unprimed-leg vectors (see the mirror comment in
                                    # (a, li = 2)): tied = s k3pvec exactly, free at its
                                    # own bin-center magnitude.
                                    tied_v = (s * k3pvec)[..., None, :]
                                    free_v = jnp.asarray(coordu_lif)[None, :, None, None, None] * dir2[None, None, None, qsl, :]
                                    tied = _bcast(tied_v, shp)
                                    free = jnp.broadcast_to(free_v, shp + (3,))
                                    clos = -tied - free
                                    legs = {li: tied, lif: free, 2: clos}
                                    if family == 'bb':
                                        Bu = Bu_f(legs[li], legs[r1[li]], legs[r2[li]])                          # (nbinsp, nbins, nxc, nqc)
                                    else:
                                        Tb1 = _bcast(k1pvec_row[:, None, None, None, :], shp)
                                        Tb2 = _bcast(k2pvec[..., None, :], shp)
                                        Bu = T_f(legs[r1[li]], legs[r2[li]], Tb1, Tb2) * _pt_qmask(legs[r1[li]], legs[r2[li]], Tb1, Tb2)
                                    block_node = block_node + jnp.einsum('q,ban,banq->ab', w_q[qsl], wA, Bu)
                                return block_node                                                                  # (nbinsp, nbins)

                            branch_c = np.array([1., -1.])
                            m1_flat, m2_flat, branch_flat = (jnp.asarray(v.ravel()) for v in
                                                             np.meshgrid(mu1_c, mu2_c, branch_c, indexing='ij'))
                            wm1_flat, wm2_flat, _ = (jnp.asarray(v.ravel()) for v in
                                                     np.meshgrid(w_mu1_c, w_mu2_c, branch_c, indexing='ij'))

                            block_t = jax.vmap(_node_fn)(m1_flat, wm1_flat, m2_flat, wm2_flat, branch_flat).sum(axis=0)

                            if os.environ.get('COV3_BOX_DEBUG'):
                                print(f"box33 (b) mu12 {family} li={li}: diag = {np.array2string(np.diag(np.asarray(norm_tie * block_t)), precision=3)}")
                            parts[family] = parts[family] + norm_tie * block_t

                    # (c) double-closure tie (li = lj = 2): exact local-frame
                    # angular substitution, shared with the windowed path --
                    # see _c22_closure_tie.
                    _c22 = _c22_closure_tie(Sell, Sellp, f, fp, coords, coordsp, edges, edgesp,
                                            kv_u, kv_p, k3n_u, k3n_p, hat_u, hat_p, w_u, volume,
                                            norm_tie=norm_tie, debug_label='box33')
                    for _fam, _bl in _c22.items():
                        parts[_fam] = parts[_fam] + _bl

                    if os.environ.get('COV3_BOX_DEBUG'):
                        for _nm in ('ppp', 'bb', 'pt'):
                            _v = np.atleast_2d(np.asarray(parts[_nm]))
                            print(f"box33 {_nm.upper()} ell={ell} ellp={ellp}: max|.| = {np.abs(_v).max():.3e} diag head = {np.diag(_v)[:4]}")
                    # ---- The P6 term, arXiv:1908.06234 Eq. (33) with B4-B6 ----------
                    #
                    #  Cov[B,B]_P6 = (1/V) { P6
                    #      + (1/nbar)  [ P5(k1+k1', k2, k3, k2', k3') + 8 perms ]
                    #      + (1/nbar^2)[ T(k1+k1', k2+k2', k3, k3')   + 17 perms ]
                    #      + (1/nbar^3)[ B(k1+k1', k2+k2', k3+k3')    + 5 perms ] }
                    #
                    # The T and B lines (Eq. B5, B6) need only what jaxpower.pt already has.
                    # The 1/nbar P5 line is still omitted. The connected P6 does now get picked up if
                    # `theory` supplies a 6-point callable, i.e. responds to a 6-field key --
                    # before 2026-08-29 this comment claimed that but no 6-field `get_base` was
                    # ever issued, so the term was silently absent; see _P6 just below. As for the PB
                    # block's P5 term, there is no radial delta -- both triangles' orientations
                    # are integrated independently -- which is why this is the family that
                    # populates the OFF-DIAGONAL. The reference: "for the off-diagonal
                    # elements, the P6 term becomes dominant, and the PP, PT and BB terms are
                    # small so that they can be ignored."
                    #
                    # COST. Two independent triangle grids make this (ntri x ntri' x nbins x
                    # nbinsp) with 18 T and 6 B per node -- ~40x the PB block's P5 term at the
                    # shared quadrature. It therefore uses its OWN order, COV3_P6_QTRI
                    # (default 4), and scans over the primed nodes to bound memory. Raise it
                    # if the P6 contribution matters at the per-cent level for your case.
                    # COV3_NO_P6=1 switches the term off entirely.
                    # There are two node sets, because the two families of Eq. (33) have opposite needs
                    # (both measured 2026-08-30 against paired Monte Carlo on the same integrand):
                    #
                    #   shot lines -- converge on the tensor grid, but not by q = 4. The T-line is
                    #     98-99% of the shot total and q = 4 recovers only 0.68 of it at k = 0.09
                    #     and 0.44 at k = 0.21 (q = 10: 0.91 / 0.73). They are ~34x cheaper per
                    #     configuration than the connected term, so a high q is affordable here.
                    #     COV3_P6_QTRI (default 4, kept for reproducibility -- raise it in production).
                    #
                    #   connected P6 -- out of reach of a tensor grid at all. It is a
                    #     sign-changing cancellation residue (individual topologies 3-38x their
                    #     sum) whose sharp structure lives on the joint diagonal |k_i + k'_j| ~ 0,
                    #     which a q-level product grid aliases: q = 4 gets even the sign wrong below
                    #     k ~ 0.2 and 0.14-0.15 of the answer above it; q = 10 costs 244x more and
                    #     is still 50-100% short, the deficit falling only as q^-0.8..-1.8 (5%
                    #     would need q ~ 65-190, i.e. 1e11-1e13 node pairs). Paired Monte Carlo --
                    #     one random orientation per triangle per sample, N evaluations per bin
                    #     pair instead of N^2 -- reaches 1% with N ~ 6.5e4.
                    #     COV3_P6_MC (default 65536; 0 falls back to the tensor grid).
                    _p6_q = int(os.environ.get('COV3_P6_QTRI', '4'))
                    _p6_mc = int(os.environ.get('COV3_P6_MC', '65536'))
                    if not int(os.environ.get('COV3_NO_P6', '0')) and _p6_q > 0:
                        _T6 = get_base(fields + fieldsp[:1])            # connected T (4 legs)
                        _B6 = get_base(fields)                          # connected B (3 legs)
                        # The connected P6, the leading term of Eq. (33). It shares the (1/V)
                        # prefactor of the whole brace, so it is accumulated alongside the shot
                        # lines and picked up by norm_p6 below. Unlike them it carries no
                        # permutation sum (the connected 6-point is already symmetric under
                        # relabelling of its legs) and no shot factor.
                        _P6 = get_base(fields + fieldsp)                # connected P6 (6 legs)
                        _sn6 = get_shotnoise(a, ap)
                        # Two galaxies shared between the two B estimators -> order 3, three
                        # shared -> order 4. Poisson fallback sn^(m-1) keeps this inert for a
                        # scalar `shotnoise`.
                        _sn6_2 = get_sn_moment((a, a, a), 3)
                        _sn6_3 = get_sn_moment((a, a, a, a), 4)
                        if _P6 is not None or (_sn6 != 0 and not (_T6 is None and _B6 is None)):
                            _ig = IntegralND(mu1=integration(-1., 1., size=_p6_q),
                                             mu2=integration(-1., 1., size=_p6_q),
                                             phi2=integration(0., 2. * np.pi, size=_p6_q,
                                                              method='midpoint'))
                            _m1, _m2, _f2 = (np.ravel(x) for x in
                                             _ig.x(['mu1', 'mu2', 'phi2'], sparse=False))
                            _w6 = np.ravel(_ig.w)                        # raw sum 8 pi
                            # A relative azimuth offset between the two grids: with identical
                            # nodes and equal bin magnitudes, sums like k_i + k_j' collapse to
                            # machine zero on a measure-zero set of node pairs and the T there
                            # is an unguarded squeezed configuration (same trap the BB tie
                            # family documents above).
                            _fu = lambda x, y, z: get_kvec3(coords[0], coords[1], x, y, z)
                            _fp = lambda x, y, z: get_kvec3(coordsp[0], coordsp[1], x, y, z)
                            _, (uh1, uh2, _uh3), (u1, u2, u3) = jax.vmap(_fu)(
                                jnp.asarray(_m1), jnp.asarray(_m2), jnp.asarray(_f2))
                            _, (ph1, ph2, _ph3), (q1, q2, q3) = jax.vmap(_fp)(
                                jnp.asarray(_m1), jnp.asarray(_m2),
                                jnp.asarray(_f2 + np.pi / 17.))
                            _wSu = jnp.asarray(_w6) * S(uh1, uh2)        # (nu,)
                            _wSp = jnp.asarray(_w6) * Sp(ph1, ph2)       # (np,)
                            U = [u1, u2, u3]                             # (nu, nbins, 3)
                            Q = [q1, q2, q3]                             # (np, nbinsp, 3)
                            _shp = (len(_w6), coords.shape[-1], coordsp.shape[-1])
                            # Eq. (B5): T(u_i + q_j, u_k + q_l, u_m, q_n)
                            _TP = [(0, 0, 1, 1, 2, 2), (0, 0, 1, 2, 2, 1), (0, 1, 1, 2, 2, 0),
                                   (0, 1, 1, 0, 2, 2), (0, 2, 1, 0, 2, 1), (0, 2, 1, 1, 2, 0),
                                   (0, 0, 2, 1, 1, 2), (0, 0, 2, 2, 1, 1), (0, 1, 2, 2, 1, 0),
                                   (0, 1, 2, 0, 1, 2), (0, 2, 2, 0, 1, 1), (0, 2, 2, 1, 1, 0),
                                   (1, 0, 2, 1, 0, 2), (1, 0, 2, 2, 0, 1), (1, 1, 2, 2, 0, 0),
                                   (1, 1, 2, 0, 0, 2), (1, 2, 2, 0, 0, 1), (1, 2, 2, 1, 0, 0)]
                            # Eq. (B6): B(u_i + q_j, u_k + q_l, u_m + q_n)
                            _BP = [(0, 0, 1, 1, 2, 2), (0, 0, 1, 2, 2, 1), (0, 1, 1, 0, 2, 2),
                                   (0, 1, 1, 2, 2, 0), (0, 2, 1, 0, 2, 1), (0, 2, 1, 1, 2, 0)]

                            def _p6_node(carry, iv):
                                def _u(x):    # (nu, nbins, 3) -> (nu, nbins, nbinsp, 3)
                                    return jnp.broadcast_to(x[:, :, None, :], _shp + (3,))
                                def _q(x):    # (nbinsp, 3) at node iv -> same
                                    return jnp.broadcast_to(x[iv][None, None, :, :], _shp + (3,))
                                acc = jnp.zeros(_shp)
                                if _P6 is not None and not _p6_mc:
                                    acc = acc + _P6(_u(U[0]), _u(U[1]), _u(U[2]),
                                                    _q(Q[0]), _q(Q[1]), _q(Q[2]))
                                if _T6 is not None:
                                    for (i, j, k, l, m, n) in _TP:
                                        acc = acc + _sn6_2 * _T6(
                                            _u(U[i]) + _q(Q[j]), _u(U[k]) + _q(Q[l]),
                                            _u(U[m]), _q(Q[n]))
                                if _B6 is not None:
                                    for (i, j, k, l, m, n) in _BP:
                                        acc = acc + _sn6_3 * _B6(
                                            _u(U[i]) + _q(Q[j]), _u(U[k]) + _q(Q[l]),
                                            _u(U[m]) + _q(Q[n]))
                                # Emit `acc` instead of contracting here: it depends only on
                                # the two triangle grids, NOT on (ell, ellp) -- only the S
                                # weights do. Contracting inside would redo all 24 pt calls for
                                # every (ell, ellp) pair, i.e. 3x for the B blocks.
                                return carry, acc

                            _p6_cache = cache.setdefault('p6_ell_independent', {})
                            _p6_key = (fields, fieldsp, _p6_q,
                                       _P6 is not None and not _p6_mc,
                                       float(_sn6_2), float(_sn6_3),
                                       np.asarray(coords).tobytes(),
                                       np.asarray(coordsp).tobytes())
                            if _p6_key not in _p6_cache:
                                # (np, nu, nbins, nbinsp); the scan keeps the per-step pt
                                # intermediates at (nu, nbins, nbinsp), as before.
                                _, _p6_cache[_p6_key] = jax.lax.scan(
                                    _p6_node, 0., jnp.arange(len(_w6)))
                            _acc_all = _p6_cache[_p6_key]
                            # The grid piece: sum_(u,p) w_u S_u w_p S_p acc / (8 pi)^2 is the
                            # normalised double angular average.
                            _p6 = jnp.einsum('p,u,puab->ab', _wSp, _wSu, _acc_all) \
                                / (8. * np.pi)**2

                            # The connected piece, on paired Monte-Carlo nodes: one random
                            # orientation for each triangle per sample. Eq. (33) integrates the
                            # two orientations independently (there is no radial delta in this
                            # family), so the draws are independent; the estimator is the plain
                            # mean (1/N) sum_n S(u_n) S'(q_n) P6_n, already normalised, which is
                            # why it is added to `_p6` after the (8 pi)^2 division and not before.
                            if _P6 is not None and _p6_mc:
                                _mcc = cache.setdefault('p6_mc_ell_independent', {})
                                _mck = (fields, fieldsp, _p6_mc,
                                        np.asarray(coords).tobytes(),
                                        np.asarray(coordsp).tobytes())
                                if _mck not in _mcc:
                                    _rng = np.random.default_rng(
                                        int(os.environ.get('COV3_P6_MC_SEED', '42')))
                                    _am = _rng.uniform(-1., 1., (2, _p6_mc))
                                    _ap = _rng.uniform(-1., 1., (2, _p6_mc))
                                    _pm = _rng.uniform(0., 2. * np.pi, _p6_mc)
                                    _pp = _rng.uniform(0., 2. * np.pi, _p6_mc)
                                    _, (_muh1, _muh2, _), (_mu1, _mu2, _mu3) = jax.vmap(_fu)(
                                        jnp.asarray(_am[0]), jnp.asarray(_am[1]),
                                        jnp.asarray(_pm))
                                    _, (_mph1, _mph2, _), (_mq1, _mq2, _mq3) = jax.vmap(_fp)(
                                        jnp.asarray(_ap[0]), jnp.asarray(_ap[1]),
                                        jnp.asarray(_pp))
                                    _mshp = (_p6_mc, coords.shape[-1], coordsp.shape[-1])

                                    def _mc_chunk(_lo, _hi):
                                        def _e(x, ax):    # (n, nb, 3) -> (n, nbins, nbinsp, 3)
                                            x = x[_lo:_hi]
                                            return jnp.broadcast_to(
                                                x[:, :, None, :] if ax == 0
                                                else x[:, None, :, :],
                                                (_hi - _lo,) + _mshp[1:] + (3,))
                                        return _P6(_e(_mu1, 0), _e(_mu2, 0), _e(_mu3, 0),
                                                   _e(_mq1, 1), _e(_mq2, 1), _e(_mq3, 1))

                                    _step = max(1, int(os.environ.get('COV3_P6_MC_CHUNK', '4096')))
                                    _mc_acc = jnp.concatenate(
                                        [_mc_chunk(_lo, min(_lo + _step, _p6_mc))
                                         for _lo in range(0, _p6_mc, _step)], axis=0)
                                    _mcc[_mck] = (_mc_acc, _muh1, _muh2, _mph1, _mph2)
                                _mc_acc, _muh1, _muh2, _mph1, _mph2 = _mcc[_mck]
                                _Sn = S(_muh1, _muh2) * Sp(_mph1, _mph2)          # (N,)
                                _p6 = _p6 + jnp.einsum('n,nab->ab', _Sn, _mc_acc) / _p6_mc

                            norm_p6 = M / volume
                            parts['p6'] = norm_p6 * _p6
                            if os.environ.get('COV3_BOX_DEBUG'):
                                print(f"box33 P6 ell={ell} ellp={ellp}: max|.| = "
                                      f"{np.abs(np.asarray(parts['p6'])).max():.3e}")
                            if os.environ.get('COV3_DUMP_P6'):
                                np.savez(os.environ['COV3_DUMP_P6']
                                         + f"_i{i}_ip{ip}_ell{''.join(map(str, ell))}"
                                         + f"_ellp{''.join(map(str, ellp))}.npz",
                                         p6=np.asarray(parts['p6']), ell=ell, ellp=ellp,
                                         M=M, volume=volume, wSu=np.asarray(_wSu),
                                         wSp=np.asarray(_wSp), coords=np.asarray(coords),
                                         coordsp=np.asarray(coordsp))

                    block = parts['ppp'] + parts['bb'] + parts['pt'] + parts.get('p6', 0.)

                    if ip == i:
                        # Transpose-partner tie terms (e.g. (li=0, lj=2) vs
                        # (li=2, lj=0)) are the same integral evaluated with
                        # differently-anchored quadratures; symmetrize away
                        # the quadrature-level mismatch on diagonal blocks.
                        block = (block + block.T) / 2.
                    # To host as soon as it exists. Leaving blocks as device arrays makes
                    # the final np.block do every device->host copy at once, at the moment
                    # the GPU is fullest (precompute caches still resident) -- measured: a
                    # box q = 10 run completed the whole computation and then died in
                    # np.block asking for 168 MB. Converting here also frees each block's
                    # device buffer immediately.
                    block = np.asarray(block)

                    # VALIDITY GUARD for the P6 term. Its shot lines evaluate the theory T and B
                    # at shifted momenta |k_i + k'_j|, reaching ~2 k_max -- outside the range the
                    # EFT parameters were fitted over. With a large counterterm the model diverges
                    # there and the term can swamp the block: on a Zel'dovich fit with c1 = -30 it
                    # drove max|diag| from 5.7e21 to 2.4e25 and produced 37 NEGATIVE variances,
                    # silently. Checked here, where `block` is already on the host, so it is free.
                    if 'p6' in parts and ip == i and tuple(ell) == tuple(ellp):
                        _bad = int(np.sum(np.diagonal(block) < 0.))
                        if _bad:
                            warnings.warn(
                                f'cov3: {_bad} of {len(block)} variances are NEGATIVE in the '
                                f'BB block ell={tuple(ell)} after adding the P6 term. Its shot '
                                f"lines evaluate T and B at |k_i + k'_j| up to ~2 k_max, where "
                                f'a large EFT counterterm (check c1, c2, X_FoG) diverges. '
                                f'Re-run with COV3_NO_P6=1 to confirm, and restrict k_max or '
                                f'refit the counterterms before trusting this block.',
                                RuntimeWarning, stacklevel=2)
                    cov[i][ip] = block
                    cov[ip][i] = block.T
                    if _timing:
                        dt = _time.time() - _tblock
                        _tblocks.append((dt, i, ip, tuple(label['fields']), label['ells'],
                                         tuple(labelp['fields']), labelp['ells'], block.shape))
                        print(f'[cov3] block ({i},{ip}) '
                              f'{len(label["fields"])}x{len(labelp["fields"])} ells '
                              f'{label["ells"]}/{labelp["ells"]} {block.shape}: {dt:.1f}s', flush=True)
                    continue

                # Everything below except the final assembly is independent
                # of the observable multipoles (ell, ellp): those enter only
                # through the scalar per-node weights w * S_ell(k1hat, k2hat)
                # and the normalization M. Cache the expensive part -- window
                # tables, theory evaluations and the joint-quadrature
                # trispectrum scan -- keyed by fields and binning, so with
                # several multipole blocks per observable pair (e.g.
                # (000)x(000), (000)x(202), (202)x(202)) it is paid ONCE.
                pre_cache = cache.setdefault('bb33_ell_independent', {})
                pre_key = (fields, fieldsp,
                           np.asarray(coords).tobytes(), np.asarray(coordsp).tobytes(),
                           np.asarray(edges).tobytes(), np.asarray(edgesp).tobytes())

                if pre_key not in pre_cache:
                    pre = {}

                    P_aap, P_abp, P_acp = get_theory((a, ap)), get_theory((a, bp)), get_theory((a, cp))
                    P_bap, P_bbp, P_bcp = get_theory((b, ap)), get_theory((b, bp)), get_theory((b, cp))
                    P_cap, P_cbp, P_ccp = get_theory((c, ap)), get_theory((c, bp)), get_theory((c, cp))

                    # ---- Angle factorization ----
                    # The joint 6D quadrature has nside^2 (6^6 ~ 5e4) points,
                    # but every per-point ingredient of the PPP and BB terms
                    # depends on only ONE side's angles: the closure legs
                    # k3(u), k3'(p), the bispectrum/power theory factors, the
                    # S / Sell basis values, and the per-leg Legendre
                    # factors. The windows enter as sums over (ell1, ell2) of
                    # [block x L_ell1(u-side) x L_ell2(p-side)] -- including
                    # the closure-leg interpolation, which is linear in the
                    # (concrete, cached) spectrum table. So both terms reduce
                    # exactly (same nodes/weights, summation reordered) to
                    # per-side weighted sums of O(nside) work instead of
                    # O(nside^2), all eager/concrete -- no vmap. Only the PT
                    # term's trispectrum genuinely couples the two sides and
                    # keeps the joint quadrature (see below).
                    integ_tri = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                    _tri = integ_tri.x(['mu1', 'mu2', 'phi2'], sparse=False)
                    mu1_s, mu2_s, phi2_s = (np.ravel(arr) for arr in _tri)
                    w_side = np.ravel(integ_tri.w)
                    nside = len(w_side)
                    nbins, nbinsp = coords.shape[-1], coordsp.shape[-1]
                    pre['w_side'], pre['nside'] = w_side, nside
                    pre['nbins'], pre['nbinsp'] = nbins, nbinsp

                    def _side(coords_side):
                        fn = lambda m1, m2, p2: get_kvec3(coords_side[0], coords_side[1], m1, m2, p2)
                        return jax.vmap(fn)(jnp.asarray(mu1_s), jnp.asarray(mu2_s), jnp.asarray(phi2_s))

                    (k1n_u, k2n_u, k3n_u), (k1h_u, k2h_u, k3h_u), kv_u = _side(coords)
                    (k1n_p, k2n_p, k3n_p), (k1h_p, k2h_p, k3h_p), kv_p = _side(coordsp)
                    # k1hat/k2hat per node, for the per-block S weights.
                    pre['k1h_u'], pre['k2h_u'] = k1h_u, k2h_u
                    pre['k1h_p'], pre['k2h_p'] = k1h_p, k2h_p
                    # Per-node closure-sweep measure operators: window
                    # tables are cell-averaged with the exact phi measure
                    # instead of point-sampled at the node k3 magnitudes
                    # (see closure_measure) -- used by both the PPP closure
                    # axis and the BB/PT leg tables below.
                    k3msr_u = closure_measure(coords)      # rows: (nside * nbins)
                    k3msr_p = closure_measure(coordsp)     # rows: (nside * nbinsp)

                    # Per-leg khat . n, shape (nside, nbins): legs 1, 2 are
                    # bin-independent, the closure leg 3 is not.
                    mu_u = (jnp.broadcast_to(jnp.asarray(mu1_s)[:, None], k3n_u.shape),
                            jnp.broadcast_to(jnp.asarray(mu2_s)[:, None], k3n_u.shape),
                            k3h_u[..., 2])
                    mu_p = (jnp.broadcast_to(jnp.asarray(mu1_s)[:, None], k3n_p.shape),
                            jnp.broadcast_to(jnp.asarray(mu2_s)[:, None], k3n_p.shape),
                            k3h_p[..., 2])
                    # Unit vectors per leg, (nside, nbins, 3), for the Sell basis.
                    hat_u = (jnp.broadcast_to(k1h_u[:, None, :], k3h_u.shape),
                             jnp.broadcast_to(k2h_u[:, None, :], k3h_u.shape),
                             k3h_u)
                    hat_p = (jnp.broadcast_to(k1h_p[:, None, :], k3h_p.shape),
                             jnp.broadcast_to(k2h_p[:, None, :], k3h_p.shape),
                             k3h_p)

                    # (1) Gaussian PPP term pieces: two *independent*
                    # triangles, each with its own bins and orientation. Each
                    # P-factor is symmetrized over its leg's unprimed and
                    # primed momentum, (P(k_m) + P(k'_m)) / 2 (the documented
                    # convention, see _cov3_math.tex); expanding the product
                    # of the three symmetrized factors yields 8 terms, each a
                    # product of a u-side-only and a p-side-only factor.
                    # Per term: theory factors for legs 1..3, W3 field
                    # groups, and which primed legs enter the W3 (khat1p,
                    # khat2p) basis -- mirroring the former
                    # W3(k1vec, k?pvec, k2vec, k?pvec, ...) calls.
                    ppp_terms = [
                        ((P_aap, P_bbp, P_ccp), ((a, ap), (b, bp), (c, cp)), (0, 1)),
                        ((P_aap, P_bcp, P_cbp), ((a, ap), (b, cp), (c, bp)), (0, 2)),
                        ((P_abp, P_bap, P_ccp), ((a, bp), (b, ap), (c, cp)), (1, 0)),
                        ((P_abp, P_bcp, P_cap), ((a, bp), (b, cp), (c, ap)), (1, 2)),
                        ((P_acp, P_bap, P_cbp), ((a, cp), (b, ap), (c, bp)), (2, 0)),
                        ((P_acp, P_bbp, P_cap), ((a, cp), (b, bp), (c, ap)), (2, 1)),
                    ]
                    _mark('start')
                    pre['ppp'] = []
                    # Blocks depend only on (edges, channel pair, sorted
                    # fields), not on the permutation entry: cache across the
                    # 6 entries (and across observable blocks). Zero blocks
                    # (channel pairs the stored multipoles do not feed) are
                    # dropped from the assembly. The S closures are cached
                    # too: get_S lambdifies spherical harmonics on each call.
                    ppp_block_cache = cache.setdefault('ppp_blocks', {})
                    _S_closures = {}

                    def _get_S_cached(ell):
                        if ell not in _S_closures:
                            _S_closures[ell] = get_S(ell, z3=True)
                        return _S_closures[ell]

                    _Su_cache, _Sp_cache = {}, {}
                    for (Ps, w3_fields, (s1, s2)) in ppp_terms:
                        # As for the BB / PT families: skip when the theory
                        # provides no power spectrum (e.g. B-only theory).
                        if any(P is None for P in Ps):
                            continue
                        # Debug knob, see the box-path ppp_terms loop.
                        _sel = os.environ.get('COV3_PPP_TERMS')
                        if _sel and (('closure' in _sel) != (2 in (s1, s2))):
                            continue
                        # No extra volume factor: window3 is normalized by
                        # the product of the two bispectrum-estimator
                        # normalizations int(n_a n_b n_c) int(n_a' n_b' n_c')
                        # (see compute_fkp3_covariance_window), so Q_W^{ABC}'s
                        # periodic limit is (2 pi)^6 delta delta / V --
                        # exactly the Sugiyama PPP covariance
                        # (_cov3_math_periodic.tex appendix, where
                        # delta^3(0) = V / (2 pi)^3 from the third pair
                        # delta).
                        entry = {'A_u': [Ps[m](kv_u[m]) for m in range(3)],   # (nside, nbins)
                                 'B_p': [Ps[m](kv_p[m]) for m in range(3)],   # (nside, nbinsp)
                                 'blocks': []}
                        # window3 stores Q_W^{ABC} with each leg's own field pair
                        # sorted (a per-pair swap leaves Q_W unchanged: even
                        # multipoles are parity-symmetric under s -> -s), AND with
                        # positions 1, 2 (A, B) sorted relative to each other
                        # (Q_W^{ABC} = Q_W^{BAC}, an exchange of the two "arm"
                        # separations) while position 3 (C) is kept independent
                        # (Q_W^{ABC} != Q_W^{ACB} in general -- see
                        # compute_fkp3_covariance_window) -- match that here.
                        p1, p2, p3 = tuple(sorted(w3_fields[0])), tuple(sorted(w3_fields[1])), tuple(sorted(w3_fields[2]))
                        p1, p2 = sorted((p1, p2))
                        # Radial spec of the two primed separation axes: the
                        # Wick permutation ties separation axis i to primed
                        # leg s_i -- a binned leg (0, 1: the matching kpedges
                        # column) or the closure leg (2: point-evaluated at
                        # the node-dependent k3' magnitudes, mirroring the
                        # box path's exact substitution).
                        _paxes = (s1, s2)
                        _kp_points = k3n_p if 2 in _paxes else None
                        _kp_measure = k3msr_p if 2 in _paxes else None
                        for ell_w3 in w3_ells:
                            for ellp_w3 in w3_ells:
                                bkey = (tuple(np.ravel(edges)), tuple(np.ravel(edgesp)), ell_w3, ellp_w3, p1, p2, p3, _paxes)
                                if bkey not in ppp_block_cache:
                                    ppp_block_cache[bkey] = np.asarray(compute_spectrum3_covariance_window_block(
                                        window3, edges, edgesp, ell_w3, ellp_w3,
                                        fields1=p1, fields2=p2, fields3=p3,
                                        cache=cache, batch_size=batch_size,
                                        paxes=_paxes, kp_points=_kp_points,
                                        kp_measure=_kp_measure)).real
                                blk = ppp_block_cache[bkey]
                                if not np.any(blk):
                                    continue
                                if os.environ.get('COV3_BOX_DEBUG'):
                                    print(f"win33 ppp blk {w3_fields} {ell_w3}x{ellp_w3}: max|blk| = {np.abs(blk).max():.3e}")
                                if ell_w3 not in _Su_cache:
                                    _Su_cache[ell_w3] = _get_S_cached(ell_w3)(hat_u[0], hat_u[1])           # (nside, nbins)
                                if (ellp_w3, s1, s2) not in _Sp_cache:
                                    _Sp_cache[ellp_w3, s1, s2] = _get_S_cached(ellp_w3)(hat_p[s1], hat_p[s2])   # (nside, nbinsp)
                                entry['blocks'].append((jnp.asarray(blk), _Su_cache[ell_w3], _Sp_cache[ellp_w3, s1, s2]))
                        pre['ppp'].append(entry)

                    # (2) Connected BB term pieces: each bispectrum lives
                    # purely on its own triangle; the only primed/unprimed
                    # coupling is the window Q_W(k_i, k'_j) tying leg i of
                    # the unprimed triangle (group1 = the full unprimed field
                    # triple) to leg j of the primed one (group2 = the full
                    # primed field triple).
                    f, fp = (a, b, c), (ap, bp, cp)
                    # r1(i)/r2(i): the other two legs of triangle i, cyclic --
                    # same definition for unprimed and primed.
                    r1, r2 = (1, 2, 0), (2, 0, 1)

                    # Bispectrum theory on the per-side grids, (nside, nbins);
                    # None when the theory provides no bispectrum (e.g.
                    # P-only theory with no shot noise) -- the BB term then
                    # vanishes and its tables are skipped.
                    def _B_side(th, kv, li):
                        fn = get_theory((th[li], th[r1[li]], th[r2[li]]))
                        return None if fn is None else fn(kv[li], kv[r1[li]], kv[r2[li]])
                    B_u = [_B_side(f, kv_u, li) for li in range(3)]
                    Bp_p = [_B_side(fp, kv_p, lj) for lj in range(3)]

                    wpair = (window2, window2)
                    k3pts_u = np.asarray(jnp.ravel(k3n_u))   # (nside * nbins,)
                    k3pts_p = np.asarray(jnp.ravel(k3n_p))   # (nside * nbinsp,)

                    Lu = {e: [get_legendre(e)(mu_u[li]) for li in range(3)] for e in qw_ells}   # (nside, nbins)
                    Lp = {e: [get_legendre(e)(mu_p[lj]) for lj in range(3)] for e in qw_ells}   # (nside, nbinsp)
                    # Legendre x bispectrum per-side factors for BB.
                    LBu = {e: [None if B_u[li] is None else Lu[e][li] * B_u[li] for li in range(3)] for e in qw_ells}
                    LBp = {e: [None if Bp_p[lj] is None else Lp[e][lj] * Bp_p[lj] for lj in range(3)] for e in qw_ells}
                    _mark('ppp', pre['ppp'])
                    _mark('bb_LB (B theory)', (LBu, LBp))
                    pre['bb_LBu'], pre['bb_LBp'] = LBu, LBp
                    pre['pt_Lu'], pre['pt_Lp'] = Lu, Lp

                    def _leg_tables(win, li, lj, fields1, fields2, F_u, F_p):
                        # Shared-_qw_tables adapter for this block's (li, lj)
                        # leg pairs: leg 2 is the closure leg (per-node
                        # interpolated points), legs 0/1 are binned. Sort each
                        # group independently -- window2 stores each group's
                        # own fields sorted, regardless of numeric field-label
                        # order (does not affect the S/Legendre/theory factors
                        # already baked into F_u/F_p or the caller's own kv_u/kv_p
                        # usage, which keep their physical field<->leg mapping).
                        fields1, fields2 = tuple(sorted(fields1)), tuple(sorted(fields2))
                        return _qw_tables(
                            win,
                            k3pts_u if li == 2 else edges[:, li, :],
                            k3pts_p if lj == 2 else edgesp[:, lj, :],
                            fields1, fields2,
                            rows_shape=(nside, nbins) if li == 2 else None,
                            cols_shape=(nside, nbinsp) if lj == 2 else None,
                            F_u={e: F_u[e][li] for e in qw_ells} if li == 2 else None,
                            F_p={e: F_p[e][lj] for e in qw_ells} if lj == 2 else None,
                            rows_measure=k3msr_u if li == 2 else None,
                            cols_measure=k3msr_p if lj == 2 else None)

                    # BB: (window2, window2) symmetrization pair (same-size
                    # field groups), Legendre x B factors absorbed; None when
                    # either bispectrum factor is unavailable.
                    # _tie_pair_used: do not build the tables the assembly
                    # loop below discards -- in particular the (2, 2)
                    # doubly-derived tie, the only node x node table.
                    def _leg_tables_t(win, li, lj, *a, **kw):
                        if not _timing:
                            return _leg_tables(win, li, lj, *a, **kw)
                        import time as _t
                        _t0 = _t.time()
                        _o = _leg_tables(win, li, lj, *a, **kw)
                        try: jax.block_until_ready(_o)
                        except Exception: pass
                        print('[cov3]     bb_K leg pair (%d,%d): %.1fs' % (li, lj, _t.time() - _t0),
                              flush=True)
                        return _o

                    pre['bb_K'] = [[None if (B_u[li] is None or Bp_p[lj] is None or not _tie_pair_used(li, lj))
                                    else _leg_tables_t(wpair, li, lj,
                                                     (f[li], f[r1[li]], f[r2[li]]), (fp[lj], fp[r1[lj]], fp[r2[lj]]),
                                                     LBu, LBp) for lj in range(3)] for li in range(3)]

                    # (3) P x T term pieces: the trispectrum T(k_r1(u),
                    # k_r2(u), k'_r1(p), k'_r2(p)) genuinely couples the two
                    # triangles' orientations, so a joint quadrature over the
                    # nside^2 angle pairs remains. Every OTHER factor
                    # (window, power spectrum, Legendre) is one-sided:
                    # precompute them concretely OUTSIDE the scanned body,
                    # leaving a pure-JAX body (gathers + the trispectrum
                    # kernel). That makes it safely stageable by
                    # jax.lax.scan -- earlier scan attempts failed on
                    # host-side FFTlog/scipy/cache machinery inside the body
                    # (none left now) -- and small enough to compile, unlike
                    # the earlier jit of the whole PPP+BB+PT body with the
                    # window-interpolation machinery inlined.
                    # Bare window2 here (fields1 is a pair, fields2 a
                    # quadruple -- different sizes, so no symmetrization
                    # pair); plain Legendre factors absorbed.
                    P_lu = [[get_theory((f[li], fp[lj])) for lj in range(3)] for li in range(3)]
                    P_lu = [[None if P is None else P(kv_u[li]) for lj, P in enumerate(row)] for li, row in enumerate(P_lu)]  # (nside, nbins)
                    T_terms = [[get_theory((f[r1[li]], f[r2[li]], fp[r1[lj]], fp[r2[lj]])) for lj in range(3)] for li in range(3)]

                    # Group the (li, lj) pairs by their trispectrum field
                    # tuple so each *distinct* kernel is staged ONCE,
                    # evaluated on inputs stacked over its pairs: unrolling 9
                    # copies of the (very large) trispectrum kernel in the
                    # scanned body made XLA compilation pathologically slow
                    # (hours, near-OOM). For a single tracer this collapses
                    # to one instance.
                    T_groups = {}
                    for li in range(3):
                        for lj in range(3):
                            if P_lu[li][lj] is None or T_terms[li][lj] is None:
                                continue
                            # Skips the doubly-derived (2, 2) tie (see the BB
                            # loop) and honours COV3_PT_TERMS.
                            if not _tie_pair_used(li, lj, env='COV3_PT_TERMS'):
                                continue
                            key = (f[r1[li]], f[r2[li]], fp[r1[lj]], fp[r2[lj]])
                            T_groups.setdefault(key, []).append((li, lj))

                    # Window tables for the P x T ties, built ONLY for the
                    # pairs that survived above -- _pt_point never touches any
                    # other entry. Building all 9 unconditionally was the
                    # actual q^6 ceiling on the windowed path: with a theory
                    # that has no trispectrum (T_groups empty, e.g. a Gaussian
                    # field with no shot noise) every one of them was built and
                    # then discarded, and the (2, 2) entry alone is a
                    # (q^3 nbins)^2 float64 array -- 101 GB at
                    # COV3_QUAD_SIZE = 14, where XLA aborts with "byte size of
                    # input/output arguments exceeds the base limit".
                    QTs = [[None] * 3 for _ in range(3)]
                    for _pairs in T_groups.values():
                        for li, lj in _pairs:
                            QTs[li][lj] = _leg_tables(window2, li, lj,
                                                      (f[li], fp[lj]), (f[r1[li]], f[r2[li]], fp[r1[lj]], fp[r2[lj]]),
                                                      Lu, Lp)

                    # Relative azimuth between the two triangles: only ONE
                    # overall azimuth is a symmetry -- the trispectrum
                    # genuinely depends on the relative azimuth phi between
                    # the primed and unprimed triangle planes (same class of
                    # dependence as the PP block's T0 term). Everything else
                    # is invariant under a common rotation about the LOS:
                    # the window tables (m = 0 Legendre reconstruction), P
                    # (unprimed only), and the S / S' basis values (their
                    # sum_m Ylm Yl-m structure cancels the e^{i m phi}
                    # phases). So the phi average happens entirely inside
                    # the scanned body, applied to T only, and the stored F
                    # table keeps its (u, p) shape.
                    # Relative-azimuth grid. It is a SEPARATE axis from the triangle
                    # quadrature: the integrand's dependence on it comes only through the
                    # trispectrum's dependence on the angle between the two triangle planes,
                    # which is smooth and low-order, whereas q is set by the much harder
                    # triangle-shape structure. Tying it to q makes the dominant cost of the
                    # whole covariance (the P x T scan: 95 s of a 144 s (3,3) block at q = 6)
                    # scale with an order it does not need. COV3_PT_PHI sets it independently.
                    _ptphi = int(os.environ.get('COV3_PT_PHI', '0')) or _qsize
                    _integ_ptphi = (integ_phi if _ptphi == _qsize
                                    else integration(0., 2. * np.pi, size=_ptphi, method=_phi_rule))
                    phi_rel, wphi_rel = np.asarray(_integ_ptphi.x()), np.asarray(_integ_ptphi.w)
                    # Offset the relative-azimuth grid (valid for a 2 pi-periodic integrand):
                    # as in the box-limit tie terms, unoffset nodes rotate primed legs exactly
                    # (anti-)parallel to unprimed ones at equal bin magnitudes, making internal
                    # trispectrum pair sums cancel to machine zero -- unguarded squeezed-T
                    # configurations that dominate the sum.
                    phi_rel = phi_rel + np.pi / 17.
                    nphi = len(phi_rel)
                    cosp, sinp = jnp.asarray(np.cos(phi_rel)), jnp.asarray(np.sin(phi_rel))
                    # Rotated primed legs, (3 legs, nphi, nside, nbinsp, 3).
                    kv_p_rot = [jnp.stack([jnp.stack([kv_p[m][..., 0] * cp - kv_p[m][..., 1] * sp,
                                                      kv_p[m][..., 0] * sp + kv_p[m][..., 1] * cp,
                                                      kv_p[m][..., 2]], axis=-1)
                                           for cp, sp in zip(cosp, sinp)]) for m in range(3)]

                    # IR cutoff on the trispectrum's off-shell internal
                    # momenta (see make_pt_qmask): half the smallest k-bin
                    # edge.
                    # Floored at half the smallest bin width: binnings extending to k ~ 0
                    # make the smallest edge (hence the cutoff) collapse, leaving the
                    # squeezed-T degeneracies unmasked.
                    _pt_qmin = max(0.5 * min(np.min(edges), np.min(edgesp)),
                                   0.5 * min(np.min(np.asarray(edges)[..., 1] - np.asarray(edges)[..., 0]),
                                             np.min(np.asarray(edgesp)[..., 1] - np.asarray(edgesp)[..., 0])))
                    _pt_qmask = make_pt_qmask(_pt_qmin)

                    def _pt_point(u, p):
                        # The (ell, ellp)-independent integrand
                        # F(u, p)[a, b] = sum_{li,lj} Q_lilj(u, p) P_lilj(u)
                        # <T_lilj(u, p, phi)>_phi: the observable-multipole
                        # weights w S_ell / w S_ellp are applied afterwards,
                        # per block, so F is shared across all multipole
                        # blocks of this observable pair.
                        out = 0.
                        for key, pairs in T_groups.items():
                            T_fn = T_terms[pairs[0][0]][pairs[0][1]]
                            # Stack over (pair, phi) so each distinct kernel
                            # is staged once.
                            k_r1 = jnp.stack([jnp.broadcast_to(kv_u[r1[li]][u][:, None, :], (nbins, nbinsp, 3)) for li, lj in pairs for _ in range(nphi)])
                            k_r2 = jnp.stack([jnp.broadcast_to(kv_u[r2[li]][u][:, None, :], (nbins, nbinsp, 3)) for li, lj in pairs for _ in range(nphi)])
                            kp_s1 = jnp.stack([jnp.broadcast_to(kv_p_rot[r1[lj]][iphi][p][None, :, :], (nbins, nbinsp, 3)) for li, lj in pairs for iphi in range(nphi)])
                            kp_s2 = jnp.stack([jnp.broadcast_to(kv_p_rot[r2[lj]][iphi][p][None, :, :], (nbins, nbinsp, 3)) for li, lj in pairs for iphi in range(nphi)])
                            qmask = _pt_qmask(k_r1, k_r2, kp_s1, kp_s2)
                            Tv = (T_fn(k_r1, k_r2, kp_s1, kp_s2) * qmask).reshape(len(pairs), nphi, nbins, nbinsp)
                            # 1/(2 pi): relative-azimuth average (raw weights
                            # sum to 2 pi).
                            Tv = jnp.einsum('gfab,f->gab', Tv, jnp.asarray(wphi_rel)) / (2. * np.pi)
                            for ipair, (li, lj) in enumerate(pairs):
                                tab = QTs[li][lj]
                                if li == 2 and lj == 2:
                                    Q = tab[None][u, :, p, :]
                                elif li == 2:
                                    Q = sum(tab['e2', e2][u] * Lp[e2][lj][p][None, :] for e2 in qw_ells)
                                elif lj == 2:
                                    Q = sum(tab['e1', e1][:, p] * Lu[e1][li][u][:, None] for e1 in qw_ells)
                                else:
                                    Q = sum(tab['e12', e1, e2] * Lu[e1][li][u][:, None] * Lp[e2][lj][p][None, :]
                                            for e1 in qw_ells for e2 in qw_ells)
                                out = out + Q * P_lu[li][lj][u][:, None] * Tv[ipair]
                        return out

                    # Joint quadrature as index pairs into the per-side
                    # grids, chunked for the scan; the tail is padded (with
                    # clamped indices) and the padded per-point values are
                    # zeroed by `valid` before they reach the accumulator.
                    npts = nside * nside
                    # Scan chunk over the joint (u, p) quadrature. This is the dominant cost
                    # of a windowed P+B covariance -- measured at q = 6: 100 s of a 147 s
                    # (3,3) block, against 3.6 s for all eight BB leg tables -- so it is worth
                    # tuning per device. COV3_PT_CHUNK overrides.
                    chunk = int(os.environ.get('COV3_PT_CHUNK', '256'))
                    npad = (-npts) % chunk
                    joint = np.arange(npts + npad)
                    u_idx, p_idx = np.divmod(joint, nside)
                    u_idx, p_idx = np.minimum(u_idx, nside - 1), np.minimum(p_idx, nside - 1)
                    valid = (joint < npts).astype(float)
                    xs = (jnp.asarray(u_idx.reshape(-1, chunk), dtype=jnp.int32),
                          jnp.asarray(p_idx.reshape(-1, chunk), dtype=jnp.int32),
                          jnp.asarray(valid.reshape(-1, chunk)))

                    # THE (ell, ellp) CONTRACTION HAPPENS INSIDE THE SCAN.
                    #
                    # The integrand F(u, p)[a, b] is (ell, ellp)-independent, so the obvious
                    # thing is to store it and let each multipole block contract it later:
                    # `einsum('u,p,upab->ab', wS_u, wSp_p, F)`. But that array is
                    # (q^3, q^3, nbins, nbinsp) -- 13.4 GB at q = 10 for 41 bins, 101 GB at
                    # q = 14 -- and with a *connected* trispectrum (rather than only the
                    # shot-noise renormalized one) several T groups survive, so q = 10 stopped
                    # fitting at all: the run did not raise, it sat in the GPU allocator's
                    # retry loop at 0% utilisation indefinitely.
                    #
                    # Nothing downstream ever wants F itself, only its contractions against a
                    # handful of weight pairs. So the (ell, ellp) blocks that share this
                    # precompute are collected here and accumulated in the scan carry: memory
                    # drops from q^6 * nbins * nbinsp to npairs * nbins * nbinsp, i.e. from
                    # 13.4 GB to under a megabyte, while the expensive part -- the T
                    # evaluations -- is still paid exactly once for all of them.
                    #
                    # The pairs are recovered by re-deriving each observable pair's precompute
                    # key and keeping those that match this one, so this list is by
                    # construction exactly the set of blocks that would have used `pre`.
                    pt_pairs = []
                    for _i, (_lab, _obs) in enumerate(_observable.items(level=None)):
                        for _ip, (_labp, _obsp) in enumerate(_observable.items(level=None)):
                            if _ip < _i:
                                continue
                            if (tuple(_lab['fields']), tuple(_labp['fields'])) != (fields, fieldsp):
                                continue
                            _e = [np.asarray(o.edges('k')) for o in (_obs, _obsp)]
                            _c = [o.coords('k', center=center).T for o in (_obs, _obsp)]
                            if (np.asarray(_c[0]).tobytes(), np.asarray(_c[1]).tobytes(),
                                    _e[0].tobytes(), _e[1].tobytes()) != pre_key[2:]:
                                continue
                            pt_pairs.append((tuple(_lab['ells']), tuple(_labp['ells'])))

                    _wsd = jnp.asarray(w_side)
                    wS_u_g = jnp.stack([_wsd * get_S(_e, z3=True)(k1h_u, k2h_u)
                                        for _e, _ep in pt_pairs]) if pt_pairs else None
                    wS_p_g = jnp.stack([_wsd * get_S(_ep, z3=True)(k1h_p, k2h_p)
                                        for _e, _ep in pt_pairs]) if pt_pairs else None

                    def _pt_scan(xs):
                        def body(carry, x):
                            u, p, v = x
                            Fc = jax.vmap(_pt_point)(u, p)                 # (chunk, nbins, nbinsp)
                            wu = wS_u_g[:, u] * v[None, :]                 # (npairs, chunk)
                            wp = wS_p_g[:, p]                              # (npairs, chunk)
                            return carry, jnp.einsum('gc,gc,cab->gab', wu, wp, Fc)
                        # The partial sums are RETURNED, not accumulated in the carry.
                        # Accumulating is the obvious thing, and is what the first version of
                        # this did -- but it makes every scan iteration depend on the previous
                        # one, and XLA then serializes a loop whose iterations are otherwise
                        # independent: measured 1119 s against 616 s for the same q = 8 cutsky
                        # covariance. Stacking (nchunks, npairs, nbins, nbinsp) partials and
                        # summing at the end keeps the original dependency structure at
                        # negligible cost -- 43 MB at q = 8, against the 3.5 GB of the (u, p)
                        # grid it replaces.
                        _mark('pt setup')
                        _out = jax.lax.scan(body, 0., xs)[1].sum(axis=0)
                        _mark('PT SCAN', _out)
                        return _out

                    if T_groups and pt_pairs:
                        _PT = jax.jit(_pt_scan)(xs)                        # (npairs, nbins, nbinsp)
                        _mark('bb_K', pre['bb_K'])
                        pre['pt_PT'] = {pair: _PT[g] for g, pair in enumerate(pt_pairs)}
                    else:
                        # No trispectrum contribution (e.g. P-only theory
                        # with no shot noise).
                        pre['pt_PT'] = None

                    if pre['pt_PT'] is not None and os.environ.get('COV3_BOX_DEBUG'):
                        for _pair, _v in pre['pt_PT'].items():
                            _v = np.asarray(_v)
                            print(f"win33 pt_PT {_pair}: max|.| = {np.abs(_v).max():.3e}, "
                                  f"median|.| = {np.median(np.abs(_v)):.3e}")

                    # (2, 2) double-closure tie: the double-Legendre channel
                    # representation is invalid for two derived directions (it
                    # only enforces |k3| = |k3'|; mode-sum-refuted) and is
                    # skipped in the loops below. Instead the exact
                    # local-frame substitution (_c22_closure_tie, shared with
                    # the box path) is used, with the tie strength set by the
                    # window's zero-lag (triple)(triple) monopole,
                    # V_eff = 1 / Q_W^{(abc)(a'b'c')}(s -> 0).
                    _mark('pt_PT', pre.get('pt_PT'))
                    pre['c22'] = None
                    if not os.environ.get('COV3_WIN_DOUBLE_CLOSURE'):
                        try:
                            _t_u, _t_p = tuple(sorted((a, b, c))), tuple(sorted((ap, bp, cp)))
                            _inv = 0.5 * (np.real(window2.get(fields1=_t_u, fields2=_t_p, ells=0).value()[0])
                                          + np.real(window2.get(fields1=_t_p, fields2=_t_u, ells=0).value()[0]))
                            pre['c22'] = (kv_u, kv_p, jnp.asarray(k3n_u), jnp.asarray(k3n_p),
                                          hat_u, hat_p, float(1. / _inv))
                        except Exception:
                            # No (3)(3) window group available: leave the
                            # (2, 2) tie out entirely (its true size is ~ one
                            # single-closure tie).
                            pre['c22'] = None

                    pre_cache[pre_key] = pre

                pre = pre_cache[pre_key]

                # ---- Per-(ell, ellp) assembly (cheap) ----
                nbins, nbinsp = pre['nbins'], pre['nbinsp']
                wS_u = jnp.asarray(pre['w_side']) * S(pre['k1h_u'], pre['k2h_u'])
                wSp_p = jnp.asarray(pre['w_side']) * Sp(pre['k1h_p'], pre['k2h_p'])
                # 1/(8 pi)^2: each triangle's own (dcos theta_1 / 2)(dOmega_2 / 4 pi) normalization.
                norm33 = M / (8. * np.pi)**2

                block_PPP = 0.
                for entry in pre['ppp']:
                    A_u, B_p = entry['A_u'], entry['B_p']
                    for blk, Sell_u, Sellp_p in entry['blocks']:
                        for mask in range(8):
                            left = wS_u[:, None] * Sell_u
                            right = wSp_p[:, None] * Sellp_p
                            for m in range(3):
                                if (mask >> m) & 1:
                                    left = left * A_u[m]
                                else:
                                    right = right * B_p[m]
                            if blk.ndim == 3:
                                # Closure-leg entry: the block keeps a primed
                                # node axis (a, b, n) -- contract it against
                                # the per-node primed factors instead of
                                # pre-summing them.
                                term = jnp.einsum('a,nb,abn->ab', left.sum(axis=0), right, blk)
                            else:
                                term = blk * left.sum(axis=0)[:, None] * right.sum(axis=0)[None, :]
                            block_PPP = block_PPP + 0.125 * term
                block_PPP = norm33 * block_PPP

                LBu, LBp = pre['bb_LBu'], pre['bb_LBp']
                block_BB = 0.
                # li/lj: triangle-leg indices (NOT the enclosing observable-
                # pair indices i/ip -- do not shadow those).
                for li in range(3):
                    for lj in range(3):
                        tab = pre['bb_K'][li][lj]
                        if tab is None:
                            continue
                        # The doubly-derived (li = lj = 2) tie cannot be
                        # represented by the m = 0, ell <= 4 double-Legendre
                        # channels: they enforce only |k3| = |k3'|, not the
                        # vector tie, overcounting by the number of
                        # directions per shell (x10-30, mode-sum-refuted).
                        # Skip it (its true size is ~ one single-closure tie,
                        # exactly computed by the box-limit (c) family) until
                        # a (c)-style exact angular substitution is ported to
                        # the windowed path. COV3_WIN_DOUBLE_CLOSURE=1
                        # re-enables it for A/B checks. Also honours the
                        # COV3_BB_TERMS debug knob (arm-only / closure-only /
                        # explicit pairs); the same predicate already kept the
                        # corresponding table from being built at all.
                        if not _tie_pair_used(li, lj):
                            continue
                        if li == 2 and lj == 2:
                            term = jnp.einsum('uapb,u,p->ab', tab[None], wS_u, wSp_p)
                        elif li == 2:
                            term = sum(jnp.einsum('uab,u->ab', tab['e2', e2], wS_u) * (wSp_p @ LBp[e2][lj])[None, :] for e2 in qw_ells)
                        elif lj == 2:
                            term = sum(jnp.einsum('apb,p->ab', tab['e1', e1], wSp_p) * (wS_u @ LBu[e1][li])[:, None] for e1 in qw_ells)
                        else:
                            term = sum(tab['e12', e1, e2] * (wS_u @ LBu[e1][li])[:, None] * (wSp_p @ LBp[e2][lj])[None, :]
                                       for e1 in qw_ells for e2 in qw_ells)
                        block_BB = block_BB + term
                block_BB = norm33 * block_BB

                # Already contracted against this block's (ell, ellp) weights inside the
                # precompute's scan -- see the note there on why F is never materialized.
                block_PT = (0. if pre['pt_PT'] is None
                            else norm33 * pre['pt_PT'][tuple(ell), tuple(ellp)])
                if pre.get('c22') is not None:
                    # Exact-substitution (2, 2) double-closure tie (see the
                    # precompute note): shared machinery with the box path,
                    # windowed integrated tie strength.
                    _g = pre['c22']
                    _c22 = _c22_closure_tie(S, Sp, fields, fieldsp, coords, coordsp, edges, edgesp,
                                            _g[0], _g[1], _g[2], _g[3], _g[4], _g[5],
                                            jnp.asarray(pre['w_side']), _g[6],
                                            norm_tie=M / (32. * np.pi**2), debug_label='win33')
                    block_BB = block_BB + _c22.get('bb', 0.)
                    block_PT = block_PT + _c22.get('pt', 0.)
                if os.environ.get('COV3_BOX_DEBUG'):
                    for _nm, _bl in (('PPP', block_PPP), ('BB', block_BB), ('PT', block_PT)):
                        _v = np.atleast_2d(np.asarray(_bl))
                        print(f"win33 {_nm} ell={ell} ellp={ellp}: max|.| = {np.abs(_v).max():.3e} diag = {np.array2string(np.diag(_v), precision=3, max_line_width=10000)}")
                block = block_PPP + block_BB + block_PT
                if ip == i:
                    # Same fix as the box-limit path's (c) double-closure term: the
                    # closure-leg quadrature anchors its free-shape integral on one
                    # of the two (here identical) triangles, so the diagonal block
                    # is not exactly self-transpose-symmetric on its own -- average
                    # away the quadrature-level mismatch.
                    block = (block + block.T) / 2.

            else:
                continue

            if np.ndim(block) == 0:
                # Every contribution to this pair vanished, leaving the bare scalar the
                # accumulators start from. This is not a degenerate case: the P-B cross
                # block of a Gaussian field with no shot noise is *exactly* zero (the
                # connected 5-point correlator vanishes), and every term there is
                # proportional to B or to the shot noise. Give it its shape back so the
                # transpose below and the np.block assembly work.
                block = jnp.full((len(observable.value()), len(observablep.value())), block)

            # To host as soon as it exists -- see the note at the other assignment sites.
            block = np.asarray(block)
            cov[i][ip] = block
            cov[ip][i] = block.T
            if _timing:
                dt = _time.time() - _tblock
                _tblocks.append((dt, i, ip, tuple(label['fields']), label['ells'],
                                 tuple(labelp['fields']), labelp['ells'], block.shape))
                print(f'[cov3] block ({i},{ip}) fields {len(label["fields"])}x{len(labelp["fields"])} '
                      f'ells {label["ells"]}/{labelp["ells"]} shape {block.shape}: {dt:.1f}s', flush=True)

    if _timing and _tblocks:
        tot = sum(t[0] for t in _tblocks)
        print(f'[cov3] {len(_tblocks)} blocks, {tot:.0f}s total; slowest:', flush=True)
        for dt, i, ip, f1, e1, f2, e2, shape in sorted(_tblocks, reverse=True)[:8]:
            print(f'[cov3]   ({i},{ip}) {len(f1)}x{len(f2)} ells {e1}/{e2} {shape}: '
                  f'{dt:.1f}s ({100 * dt / tot:.0f}%)', flush=True)

    # Assemble into one 2D array (as compute_spectrum2_covariance's finalize
    # does with np.block): CovarianceMatrix consumers (plot_diag, etc.) index
    # _value as a single matrix, not a nested list of per-pair blocks.
    # Unfilled pairs (no implemented term) become zero blocks.
    sizes = [len(obs.value()) for _, obs in _observable.items(level=None)]
    cov = [[np.zeros((sizes[i], sizes[ip])) if block is None else block for ip, block in enumerate(row)]
           for i, row in enumerate(cov)]
    return CovarianceMatrix(observable=_observable, value=np.block(cov))