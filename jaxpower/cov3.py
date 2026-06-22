import itertools

import numpy as np
import jax
from jax import numpy as jnp

from .mesh import MeshAttrs, split_particles
from .mesh2 import get_smooth2_window_bin_attrs
from .mesh3 import BinMesh3CorrelationPoles, compute_mesh3, FKPField, get_sugiyama_window_convolution_coeffs, get_smooth3_window_bin_attrs
from .types import CovarianceMatrix, ObservableTree
from .utils import wigner_3j, get_legendre, legendre_product
from .pt import integration, IntegralND, get_S
from .cov2 import Correlation2Spectrum, compute_spectrum2_covariance_window_block, matrix_rebin


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
    k3hat = k3vec / k3norm[..., None]
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

        norm = WA.sum() * WB.sum()
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
    triplets = tuple(itertools.combinations_with_replacement(pairs, 3))
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

        WA = get_W(_fkps[0], mask=masks[0]) * get_W(_fkps[1], mask=masks[1])
        WB = get_W(_fkps[2], mask=masks[2]) * get_W(_fkps[3], mask=masks[3])
        WC = get_W(_fkps[4], mask=masks[4]) * get_W(_fkps[5], mask=masks[5])

        norm = WA.sum() * WB.sum() * WC.sum()
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
    extracted via ``jax.jacfwd`` (same technique ``Correlation2Spectrum``
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
                                              cache=None, batch_size=None):
    """
    Compute one smooth covariance-window block W_{ell,ellin}(k1,k2;k1',k2').
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
        Convert input edge specification into an array of shape
        (N1, N2, 2, 2).
        """
        if isinstance(edges, (tuple, list)) and len(edges) == 2:
            k1edges, k2edges = map(normalize_edges, edges)
            out = np.empty((len(k1edges), len(k2edges), 2, 2), dtype=float)
            out[:, :, 0, :] = k1edges[:, None, :]
            out[:, :, 1, :] = k2edges[None, :, :]
            return out

        edges = np.asarray(edges)

        if edges.ndim == 4 and edges.shape[-2:] == (2, 2):
            return edges

        if edges.ndim == 3 and edges.shape[-2:] == (2, 2):
            n = int(round(edges.shape[0] ** 0.5))

            if n * n != edges.shape[0]:
                raise ValueError("Cannot infer square unraveling from flattened edges.")

            return edges.reshape(n, n, 2, 2)

        raise ValueError("Invalid edge specification.")

    edges = unravel_edges(kedges)
    edgesp = unravel_edges(kpedges)

    n1, n2 = edges.shape[:2]
    n1p, n2p = edgesp.shape[:2]

    window3 = get_window_field(window3, fields1, fields2, fields3)

    wcoeffs = get_sugiyama_window_convolution_coeffs(ell, ellin)

    if not wcoeffs:
        return np.zeros((n1 * n2, n1p * n2p))

    Qs = sum(coeff * get_w_rect(q) for q, coeff in wcoeffs)

    if np.ndim(Qs) == 0:
        return np.zeros((n1 * n2, n1p * n2p))

    s = tuple(next(iter(window3)).coords().values())

    k1_fftlog, H1 = _hankel_matrix(s[0], ell[0], cache=hankel_cache)
    k2_fftlog, H2 = _hankel_matrix(s[1], ell[1], cache=hankel_cache)
    k1p_fftlog, H1p = _hankel_matrix(s[0], ellin[0], cache=hankel_cache)
    k2p_fftlog, H2p = _hankel_matrix(s[1], ellin[1], cache=hankel_cache)

    k1edges = edges[:, 0, 0, :]
    k2edges = edges[0, :, 1, :]
    k1pedges = edgesp[:, 0, 0, :]
    k2pedges = edgesp[0, :, 1, :]

    interp_order = 3
    Mk1 = matrix_rebin(k1edges, k1_fftlog, wt=k1_fftlog**2, interp_order=interp_order, cache=rebin_cache)
    Mk2 = matrix_rebin(k2edges, k2_fftlog, wt=k2_fftlog**2, interp_order=interp_order, cache=rebin_cache)
    Mk1p = matrix_rebin(k1pedges, k1p_fftlog, wt=k1p_fftlog**2, interp_order=interp_order, cache=rebin_cache)
    Mk2p = matrix_rebin(k2pedges, k2p_fftlog, wt=k2p_fftlog**2, interp_order=interp_order, cache=rebin_cache)

    # Fuse the bin-rebinning into the forward-transform matrices before
    # contracting against Qs, so the (n_s1, n_s2) ~ 1000x1000 grid is never
    # expanded into a dense 4D tensor -- only the small bin-count axes do.
    R1, R2, R1p, R2p = Mk1 @ H1, Mk2 @ H2, Mk1p @ H1p, Mk2p @ H2p

    tmp = jnp.einsum('is,js,st->ijt', R1, R1p, Qs)
    block4d = jnp.einsum('ijt,kt,lt->ijkl', tmp, R2, R2p)

    return jnp.transpose(block4d, (0, 2, 1, 3)).reshape(n1 * n2, n1p * n2p)



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


def compute_spectrum2_covariance_window_block(window2, k1edges, k2edges, ell1, ell2,
                                              fields1=None, fields2=None, cache=None,
                                              k1_is_points=False, k2_is_points=False):
    r"""Return one :math:`(k_1, k_2)` covariance-window block.

    If ``k1_is_points`` (``k2_is_points``) is True, ``k1edges`` (``k2edges``) is
    treated as literal k-values (e.g. a derived/closure leg with no native bin
    edges) at which Q_W is interpolated, rather than bin edges Q_W is rebinned
    into.
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
    spectrum_key = (id(window2), fields1, fields2, ell1, ell2)
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
            spectrum_cache[spectrum_key] = (fftlog.k, fftlog(w)[1])

    cached = spectrum_cache[spectrum_key]
    if cached is None:
        n1 = len(k1points) if k1_is_points else len(k1edges)
        n2 = len(k2points) if k2_is_points else len(k2edges)
        return np.zeros((n1, n2))
    fk, spectrum = cached

    interp_order = 3
    if k1_is_points:
        Mx = matrix_spline_interp(fk, k1points, interp_order=interp_order, cache=spline_cache)
    else:
        Mx = matrix_rebin(k1edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
    if k2_is_points:
        My = matrix_spline_interp(fk, k2points, interp_order=interp_order, cache=spline_cache)
    else:
        My = matrix_rebin(k2edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
    return Mx @ spectrum @ My.T


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
                                 cache=None, batch_size=None):

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
    integ_mu = integration(-1., 1., size=6)
    integ_phi = integration(0., 2. * np.pi, size=6)

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

    def get_shotnoise(a, b):
        if callable(shotnoise):
            return shotnoise(a, b)
        if isinstance(shotnoise, dict):
            return shotnoise.get((a, b), shotnoise.get((b, a), 0.))
        return shotnoise if a == b else 0.

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
            sn = get_shotnoise(a, b)

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
        # generalized to fields: contractions of leg 1 with legs 2 and 3
        if ndim == 3:
            a, b, c = fields
            B = get_base(fields)
            sn_ab = get_shotnoise(a, b)
            sn_ac = get_shotnoise(a, c)
            P_ac = get_base((a, c))
            P_ab = get_base((a, b))

            if B is None and (sn_ab == 0 or P_ac is None) and (sn_ac == 0 or P_ab is None):
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
                return out.reshape(orig_shape)

            return B_N

        # T^(N), Eq. (14), generalized to cross-field shot-noise
        # Only contractions between the two estimator pairs are included.
        if ndim == 4:
            a, b, c, d = fields
            T = get_base(fields)

            pairs = [(0, 2, get_shotnoise(a, c)), (0, 3, get_shotnoise(a, d)),
                     (1, 2, get_shotnoise(b, c)), (1, 3, get_shotnoise(b, d))]

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

                return out.reshape(orig_shape)

            return T_N

        return get_base(fields)

    _observable = observable
    for i, (label, observable) in enumerate(_observable.items(level=None)):
        for ip, (labelp, observablep) in enumerate(_observable.items(level=None)):
            if ip < i:
                continue

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
                P_a_ap, P_b_bp, P_a_bp, P_b_ap = get_theory((a, ap)), get_theory((b, bp)), get_theory((a, bp)), get_theory((b, ap))
                T_abapbp = None  # get_theory((a, b, ap, bp))  # disabled while debugging the mu/mup angular fix

                # mu (k's own orientation) and mup (k''s) are independent:
                # the window breaks rotational invariance, so k and k' are
                # generally oriented differently relative to the LOS. A
                # single shared mu only samples the mu=mup diagonal of the
                # true 2D integral -- exact at ell=ellp=0 (mu-independent),
                # but wrong (and increasingly so) for ell, ellp >= 2.
                integ = IntegralND(mu=integ_mu, mup=integ_mu)
                leg, legp = get_legendre(ell), get_legendre(ellp)

                def _block_fn(mu, mup, w):
                    (knorm, khat, kvec) = get_kvec1(coords, mu)
                    (kpnorm, kphat, kpvec) = get_kvec1(coordsp, mup)
                    # P_{a ap}(k)/P_{a bp}(k) are unprimed-leg power spectra,
                    # shape (n_unprimed,) -- must broadcast along the *row*
                    # axis of W2's (n_unprimed, n_primed) block, hence the
                    # explicit [:, None] (bare broadcasting would silently
                    # align to the last/column axis instead).
                    # Q_W^{(a ap)(b bp)}(k,+k') P_{a ap}(k) P_{b bp}(-k')
                    term = W2(kvec, kpvec, fields1=(a, ap), fields2=(b, bp), edges=edges, edgesp=edgesp) * P_a_ap(kvec)[:, None] * P_b_bp(-kpvec[None, :])
                    # Q_W^{(a bp)(b ap)}(k,-k') P_{a bp}(k) P_{b ap}(+k')
                    term += W2(kvec, -kpvec, fields1=(a, bp), fields2=(b, ap), edges=edges, edgesp=edgesp) * P_a_bp(kvec)[:, None] * P_b_ap(kpvec[None, :])
                    # /4, not /2: two independent mu, mup integrals, each its
                    # own (2ell+1)/2 multipole-extraction normalization.
                    pref = (2 * ell + 1) * (2 * ellp + 1) / 4.
                    block = pref * term * leg(mu) * legp(mup) * w
                    if T_abapbp is not None:  # this is T0 (add beat-coupling + integral constraint?)
                        n1, n2 = knorm.shape[0], kpnorm.shape[0]
                        # Explicit (n1, n2, 3) broadcast for all four legs, not
                        # kpvec[None, :]: get_theory(...)'s _flatten reshapes
                        # each argument to (-1, 3) independently, which would
                        # collapse a bare leading-1 dim before any outer-product
                        # broadcasting against kvec/-kvec happens, silently
                        # turning the intended (n1, n2) block into element-wise
                        # (diagonal-like) pairing instead.
                        T_val = T_abapbp(_bc0(kvec, n1, n2), _bc0(-kvec, n1, n2), _bc1(kpvec, n1, n2), _bc1(-kpvec, n1, n2))
                        block = block + pref * inverse_V2((a, b), (ap, bp)) * T_val * leg(mu) * legp(mup)
                    return block

                mu, mup = integ.x(['mu', 'mup'], sparse=False)
                mu, mup, w = (np.ravel(arr) for arr in (mu, mup, integ.w))
                block = jax.vmap(_block_fn)(mu, mup, w).sum(axis=0)

            # PB block
            elif nfields == 2 and nfieldsp == 3:
                a, b = fields
                c, d, e = fieldsp
                P_ac, P_ad, P_ae, P_bc, P_bd, P_be = get_theory((a, c)), get_theory((a, d)), get_theory((a, e)), get_theory((b, c)), get_theory((b, d)), get_theory((b, e))
                B_bde, B_bce, B_bcd, B_ade, B_ace, B_acd = get_theory((b, d, e)), get_theory((b, c, e)), get_theory((b, c, d)), get_theory((a, d, e)), get_theory((a, c, e)), get_theory((a, c, d))

                integ = IntegralND(mu=integ_mu, mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)

                leg = get_legendre(ell)
                Sp = get_S(ellp, z3=True)

                pref = (2 * ell + 1) * get_N(*ellp) * get_H(*ellp)**2

                def _pb_oriented(side, mu1, mu2, phi2):
                    # Return (q, r1, r2), where q is the contracted bispectrum side
                    # (the leg replaced by Q_W(p, q)) and r1, r2 are the bispectrum's
                    # own two fixed legs (used directly as B's arguments).  For
                    # side == 2, q = k3 = -(k1 + k2) is the closure of the
                    # bispectrum's own k1 = coordsp[0], k2 = coordsp[1] (per the PB
                    # covariance formula: B(-p, k1, k2), with only k3 contracted via
                    # the window).
                    if side == 2:
                        (k1n, k2n, k3n), (k1h, k2h, k3h), (k1v, k2v, k3v) = get_kvec3(coordsp[0], coordsp[1], mu1, mu2, phi2)
                        return (k3n, k1n, k2n), (k3h, k1h, k2h), (k3v, k1v, k2v)
                    order = _pb_order(side)
                    return _oriented_kvec3(coordsp, order, mu1, mu2, phi2)

                def pb_term(side, P, B, fieldsP, fieldsB, sign, mu, mu1, mu2, phi2):
                    # Contract p = sign * k (the spectrum's own k, +/-) with the
                    # bispectrum's own leg q_i through Q_W(p, q_i).  P and B are
                    # evaluated directly on the bispectrum's own triangle
                    # (q_i, r1, r2), independent of p.
                    _, khat, kvec = get_kvec1(coords, mu)
                    qnorms, _, kvecs = _pb_oriented(side, mu1, mu2, phi2)
                    qvec, r1vec, r2vec = kvecs
                    pvec = sign * kvec
                    if side == 2:
                        # k3 has no native bin edges: interpolate Q_W at its literal value.
                        qedges, qedges_is_points = qnorms[0], True
                    else:
                        # edgesp is bin-major, shape (nbins, 2, 2): axis 1 selects the leg.
                        qedges, qedges_is_points = edgesp[:, side, :], False
                    # S_{ell1 ell2 L}(k1hat, k2hat, n): always the bispectrum's own
                    # literal first two legs (mu1, (mu2, phi2)) and the line of
                    # sight (z3=True) -- never permuted by which leg is contracted.
                    Sval = Sp(unitvec(mu1, jnp.zeros_like(mu1)), unitvec(mu2, phi2))
                    return (
                        W2(pvec, qvec, fields1=fieldsP, fields2=fieldsB,
                           edges=edges, edgesp=qedges, edgesp_is_points=qedges_is_points)
                        * leg(khat[..., 2])
                        * Sval
                        * P(qvec)
                        * B(qvec, r1vec, r2vec)
                    )

                def _block_fn(mu, mu1, mu2, phi2, w):
                    term = pb_term(0, P_ac, B_bde, (a, c), (b, d, e), sign=+1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += pb_term(1, P_ad, B_bce, (a, d), (b, c, e), sign=+1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += pb_term(2, P_ae, B_bcd, (a, e), (b, c, d), sign=+1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += pb_term(0, P_bc, B_ade, (b, c), (a, d, e), sign=-1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += pb_term(1, P_bd, B_ace, (b, d), (a, c, e), sign=-1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += pb_term(2, P_be, B_acd, (b, e), (a, c, d), sign=-1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    return pref * term * w

                mu, mu1, mu2, phi2 = integ.x(['mu', 'mu1', 'mu2', 'phi2'], sparse=False)
                mu, mu1, mu2, phi2, w = (np.ravel(arr) for arr in (mu, mu1, mu2, phi2, integ.w))
                block = jax.vmap(_block_fn)(mu, mu1, mu2, phi2, w).sum(axis=0)

            # BP block
            elif nfields == 3 and nfieldsp == 2:
                a, b, c = fields
                dp, ep = fieldsp
                P_ad, P_bd, P_cd, P_ae, P_be, P_ce = get_theory((a, dp)), get_theory((b, dp)), get_theory((c, dp)), get_theory((a, ep)), get_theory((b, ep)), get_theory((c, ep))
                B_bce, B_ace, B_abe, B_bcd, B_acd, B_abd = get_theory((b, c, ep)), get_theory((a, c, ep)), get_theory((a, b, ep)), get_theory((b, c, dp)), get_theory((a, c, dp)), get_theory((a, b, dp))

                integ = IntegralND(mu=integ_mu, mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)

                S = get_S(ell, z3=True)
                legp = get_legendre(ellp)

                pref = (2 * ellp + 1) * get_N(*ell) * get_H(*ell)**2

                def _bp_oriented(side, mu1, mu2, phi2):
                    # Return (q, r1, r2), where q is the contracted side of the
                    # first/bispectrum observable and r1, r2 are the bispectrum's
                    # own two fixed legs (used directly as B's arguments).  For
                    # side == 2, q = k3 = -(k1 + k2) is the closure of the
                    # bispectrum's own k1 = coords[0], k2 = coords[1] (per the PB
                    # covariance formula: B(k1, k2, -p), with only k3 contracted via
                    # the window).
                    if side == 2:
                        (k1n, k2n, k3n), (k1h, k2h, k3h), (k1v, k2v, k3v) = get_kvec3(coords[0], coords[1], mu1, mu2, phi2)
                        return (k3n, k1n, k2n), (k3h, k1h, k2h), (k3v, k1v, k2v)
                    order = _pb_order(side)
                    return _oriented_kvec3(coords, order, mu1, mu2, phi2)

                def bp_term(side, P, B, fieldsP, fieldsB, sign, mu, mu1, mu2, phi2):
                    # Symmetric PB case: q_i, r1, r2 belong to the bispectrum's own
                    # triangle and p = sign * k' belongs to the spectrum observable;
                    # Q_W is evaluated as Q_W(q_i, p).  P and B are evaluated
                    # directly on (q_i, r1, r2), independent of p.
                    _, khatp, kpvec = get_kvec1(coordsp, mu)
                    qnorms, _, kvecs = _bp_oriented(side, mu1, mu2, phi2)
                    qvec, r1vec, r2vec = kvecs
                    pvec = sign * kpvec
                    if side == 2:
                        # k3 has no native bin edges: interpolate Q_W at its literal value.
                        qedges, qedges_is_points = qnorms[0], True
                    else:
                        # edges is bin-major, shape (nbins, 2, 2): axis 1 selects the leg.
                        qedges, qedges_is_points = edges[:, side, :], False
                    # S_{ell1 ell2 L}(k1hat, k2hat, n): always the bispectrum's own
                    # literal first two legs (mu1, (mu2, phi2)) and the line of
                    # sight (z3=True) -- never permuted by which leg is contracted.
                    Sval = S(unitvec(mu1, jnp.zeros_like(mu1)), unitvec(mu2, phi2))
                    # qvec/r1vec/r2vec/P/B vary along the bispectrum's own bins
                    # (axis 0, size n_q); pvec varies along the spectrum bins
                    # (axis 1, size n_p): reshape the q-indexed factors to align.
                    return (
                        W2(qvec, pvec, fields1=fieldsP, fields2=fieldsB,
                           edges=qedges, edgesp=edgesp, edges_is_points=qedges_is_points)
                        * legp(khatp[..., 2])
                        * Sval
                        * P(qvec)[:, None]
                        * B(qvec, r1vec, r2vec)[:, None]
                    )

                def _block_fn(mu, mu1, mu2, phi2, w):
                    term = bp_term(0, P_ad, B_bce, (a, dp), (b, c, ep), sign=+1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += bp_term(1, P_bd, B_ace, (b, dp), (a, c, ep), sign=+1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += bp_term(2, P_cd, B_abe, (c, dp), (a, b, ep), sign=+1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += bp_term(0, P_ae, B_bcd, (a, ep), (b, c, dp), sign=-1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += bp_term(1, P_be, B_acd, (b, ep), (a, c, dp), sign=-1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    term += bp_term(2, P_ce, B_abd, (c, ep), (a, b, dp), sign=-1, mu=mu, mu1=mu1, mu2=mu2, phi2=phi2)
                    return pref * term * w

                mu, mu1, mu2, phi2 = integ.x(['mu', 'mu1', 'mu2', 'phi2'], sparse=False)
                mu, mu1, mu2, phi2, w = (np.ravel(arr) for arr in (mu, mu1, mu2, phi2, integ.w))
                block = jax.vmap(_block_fn)(mu, mu1, mu2, phi2, w).sum(axis=0)

            # BB block
            elif nfields == 3 and nfieldsp == 3:

                a, b, c = fields
                ap, bp, cp = fieldsp

                S, Sp = get_S(ell, z3=True), get_S(ellp, z3=True)
                M = get_N(*ell) * get_N(*ellp) * get_H(*ell)**2 * get_H(*ellp)**2

                # (1) Gaussian PPP term: two *independent* triangles, each with
                # its own bins (coords / coordsp) and own orientation -- S, Sp
                # are always the bispectrum's own literal (k1hat, k2hat), never
                # permuted by which field pairing a given term represents.
                P_aap, P_abp, P_acp = get_theory((a, ap)), get_theory((a, bp)), get_theory((a, cp))
                P_bap, P_bbp, P_bcp = get_theory((b, ap)), get_theory((b, bp)), get_theory((b, cp))
                P_cap, P_cbp, P_ccp = get_theory((c, ap)), get_theory((c, bp)), get_theory((c, cp))

                def _ppp_block_fn(mu1, mu2, phi2, mu1p, mu2p, phi2p, w):
                    (_, _, _), (k1hat, k2hat, _), (k1vec, k2vec, k3vec) = get_kvec3(coords[0], coords[1], mu1, mu2, phi2)
                    (_, _, _), (k1phat, k2phat, _), (k1pvec, k2pvec, k3pvec) = get_kvec3(coordsp[0], coordsp[1], mu1p, mu2p, phi2p)

                    # Each P-factor in the tex is written using only the
                    # unprimed leg (e.g. P_cc'(k3)), with no separate primed
                    # counterpart -- unlike the PP term's P_aa'(k)*P_bb'(k'),
                    # which splits unprimed/primed across its two factors.
                    # Symmetrizing each P-factor over its leg's unprimed and
                    # primed momentum, (P(k_n) + P(k_n')) / 2, restores the
                    # required row<->column (unprimed<->primed) symmetry of
                    # block_PPP and is the documented convention (see
                    # _cov3_math.tex).
                    def _symP(P, kvec, kpvec):
                        return (P(kvec)[:, None] + P(kpvec)[None, :]) / 2.

                    term = W3(k1vec, k1pvec, k2vec, k2pvec, fields1=(a, ap), fields2=(b, bp), fields3=(c, cp), edges=edges, edgesp=edgesp) * _symP(P_aap, k1vec, k1pvec) * _symP(P_bbp, k2vec, k2pvec) * _symP(P_ccp, k3vec, k3pvec)
                    term += W3(k1vec, k1pvec, k2vec, k3pvec, fields1=(a, ap), fields2=(b, cp), fields3=(c, bp), edges=edges, edgesp=edgesp) * _symP(P_aap, k1vec, k1pvec) * _symP(P_bcp, k2vec, k2pvec) * _symP(P_cbp, k3vec, k3pvec)
                    term += W3(k1vec, k2pvec, k2vec, k1pvec, fields1=(a, bp), fields2=(b, ap), fields3=(c, cp), edges=edges, edgesp=edgesp) * _symP(P_abp, k1vec, k1pvec) * _symP(P_bap, k2vec, k2pvec) * _symP(P_ccp, k3vec, k3pvec)
                    term += W3(k1vec, k2pvec, k2vec, k3pvec, fields1=(a, bp), fields2=(b, cp), fields3=(c, ap), edges=edges, edgesp=edgesp) * _symP(P_abp, k1vec, k1pvec) * _symP(P_bcp, k2vec, k2pvec) * _symP(P_cap, k3vec, k3pvec)
                    term += W3(k1vec, k3pvec, k2vec, k1pvec, fields1=(a, cp), fields2=(b, ap), fields3=(c, bp), edges=edges, edgesp=edgesp) * _symP(P_acp, k1vec, k1pvec) * _symP(P_bap, k2vec, k2pvec) * _symP(P_cbp, k3vec, k3pvec)
                    term += W3(k1vec, k3pvec, k2vec, k2pvec, fields1=(a, cp), fields2=(b, bp), fields3=(c, ap), edges=edges, edgesp=edgesp) * _symP(P_acp, k1vec, k1pvec) * _symP(P_bbp, k2vec, k2pvec) * _symP(P_cap, k3vec, k3pvec)

                    return S(k1hat, k2hat) * Sp(k1phat, k2phat) * term * w

                integ_ppp = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi, mu1p=integ_mu, mu2p=integ_mu, phi2p=integ_phi)
                mu1, mu2, phi2, mu1p, mu2p, phi2p = integ_ppp.x(['mu1', 'mu2', 'phi2', 'mu1p', 'mu2p', 'phi2p'], sparse=False)
                mu1, mu2, phi2, mu1p, mu2p, phi2p, w = (np.ravel(arr) for arr in (mu1, mu2, phi2, mu1p, mu2p, phi2p, integ_ppp.w))
                # 1/(8 pi)^2: each triangle's own (dcos theta_1 / 2)(dOmega_2 / 4 pi) normalization.
                block_PPP = M / (8. * np.pi)**2 * jax.vmap(_ppp_block_fn)(mu1, mu2, phi2, mu1p, mu2p, phi2p, w).sum(axis=0)

                # (2) Connected BB term: each bispectrum lives purely on its
                # own triangle (no mixed unprimed/primed legs); the only
                # primed/unprimed coupling is the window Q_W(k_i, k'_j) tying
                # leg i of the unprimed triangle (group1 = the full unprimed
                # field triple) to leg j of the primed one (group2 = the full
                # primed field triple).
                f, fp = (a, b, c), (ap, bp, cp)
                # r1(i)/r2(i): the other two legs of triangle i, cyclic --
                # same definition for unprimed (indexed by i) and primed
                # (indexed by j).
                r1, r2 = (1, 2, 0), (2, 0, 1)

                B_unprimed = [get_theory((f[i], f[r1[i]], f[r2[i]])) for i in range(3)]
                B_primed = [get_theory((fp[j], fp[r1[j]], fp[r2[j]])) for j in range(3)]

                def _bb_block_fn(mu1, mu2, phi2, mu1p, mu2p, phi2p, w):
                    (k1n, k2n, k3n), (k1h, k2h, k3h), (k1v, k2v, k3v) = get_kvec3(coords[0], coords[1], mu1, mu2, phi2)
                    (k1pn, k2pn, k3pn), (k1ph, k2ph, k3ph), (k1pv, k2pv, k3pv) = get_kvec3(coordsp[0], coordsp[1], mu1p, mu2p, phi2p)
                    knorm, kvec = (k1n, k2n, k3n), (k1v, k2v, k3v)
                    kpnorm, kpvec = (k1pn, k2pn, k3pn), (k1pv, k2pv, k3pv)

                    B_i = [B_unprimed[i](kvec[i], kvec[r1[i]], kvec[r2[i]]) for i in range(3)]
                    B_j = [B_primed[j](kpvec[j], kpvec[r1[j]], kpvec[r2[j]]) for j in range(3)]

                    term = 0.
                    for i in range(3):
                        qedges = edges[:, i, :] if i < 2 else knorm[i]
                        for j in range(3):
                            qpedges = edgesp[:, j, :] if j < 2 else kpnorm[j]

                            Qij = W2(kvec[i], kpvec[j],
                                     fields1=(f[i], f[r1[i]], f[r2[i]]), fields2=(fp[j], fp[r1[j]], fp[r2[j]]),
                                     edges=qedges, edgesp=qpedges,
                                     edges_is_points=(i == 2), edgesp_is_points=(j == 2))

                            term = term + Qij * B_i[i][:, None] * B_j[j][None, :]

                    return S(k1h, k2h) * Sp(k1ph, k2ph) * term * w

                block_BB = M / (8. * np.pi)**2 * jax.vmap(_bb_block_fn)(mu1, mu2, phi2, mu1p, mu2p, phi2p, w).sum(axis=0)

                # (3) P x T term: same two-triangle structure, but the
                # contracted pair becomes a simple power spectrum
                # P_{f_i f'_j}(k_i) (no reappearing field, since a 2-point
                # function only needs one momentum), and the remaining four
                # legs form a trispectrum.
                def _pt_block_fn(mu1, mu2, phi2, mu1p, mu2p, phi2p, w):
                    (k1n, k2n, k3n), (k1h, k2h, k3h), (k1v, k2v, k3v) = get_kvec3(coords[0], coords[1], mu1, mu2, phi2)
                    (k1pn, k2pn, k3pn), (k1ph, k2ph, k3ph), (k1pv, k2pv, k3pv) = get_kvec3(coordsp[0], coordsp[1], mu1p, mu2p, phi2p)
                    knorm, kvec = (k1n, k2n, k3n), (k1v, k2v, k3v)
                    kpnorm, kpvec = (k1pn, k2pn, k3pn), (k1pv, k2pv, k3pv)
                    n_b, n_bp = k1n.shape[0], k1pn.shape[0]

                    term = 0.
                    for i in range(3):
                        qedges = edges[:, i, :] if i < 2 else knorm[i]
                        for j in range(3):
                            qpedges = edgesp[:, j, :] if j < 2 else kpnorm[j]

                            fieldsP = (f[i], fp[j])
                            fieldsT = (f[r1[i]], f[r2[i]], fp[r1[j]], fp[r2[j]])
                            P_ij, T_term = get_theory(fieldsP), get_theory(fieldsT)

                            k_r1 = _bc0(kvec[r1[i]], n_b, n_bp)
                            k_r2 = _bc0(kvec[r2[i]], n_b, n_bp)
                            kp_s1 = _bc1(kpvec[r1[j]], n_b, n_bp)
                            kp_s2 = _bc1(kpvec[r2[j]], n_b, n_bp)

                            term = term + (
                                W2(kvec[i], kpvec[j], fields1=fieldsP, fields2=fieldsT,
                                   edges=qedges, edgesp=qpedges,
                                   edges_is_points=(i == 2), edgesp_is_points=(j == 2))
                                * P_ij(kvec[i])[:, None]
                                * T_term(k_r1, k_r2, kp_s1, kp_s2)
                            )

                    return S(k1h, k2h) * Sp(k1ph, k2ph) * term * w

                block_PT = M / (8. * np.pi)**2 * jax.vmap(_pt_block_fn)(mu1, mu2, phi2, mu1p, mu2p, phi2p, w).sum(axis=0)
                block = block_PPP + block_BB + block_PT

            else:
                continue

            cov[i][ip] = block
            cov[ip][i] = block.T

    return CovarianceMatrix(observable=_observable, value=cov)