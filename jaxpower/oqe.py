r"""
Optimal quadratic estimator weightings for the FKP-style power spectrum estimator.

The FKP estimator weights each object by a function of :math:`\bar{n}(z)` alone. The optimal
weighting is :math:`W C^{-1}`, with :math:`C = W \Xi W + N` the data covariance, :math:`W` the
selection function and :math:`N` the noise. It is :math:`k` dependent, so no per-particle weight
can carry it; it is applied to the painted field instead.

:func:`separable_inverse_weight` approximates :math:`C^{-1}` by a sum of separable terms
:math:`\sum_\alpha f_\alpha(k) D_\alpha(\mathbf{x})`, and :class:`SeparableWeighting` applies
:math:`W \sum_\alpha f_\alpha(k) D_\alpha(\mathbf{x})` to a field. Its normalization and shot
noise stay contractions of :math:`M \times M` matrices of FKP-like integrals
(:func:`compute_oqe2_normalization`, :func:`compute_oqe2_shotnoise`); the residual mode coupling
is carried by the window matrix. :func:`covariance_weight` applies the exact :math:`C^{-1}` by
conjugate gradients, as a reference to validate the separable approximation against.

A weighting is passed around as a factory ``MeshAttrs -> SeparableWeighting``, so that its
configuration-space factors, which are smooth, can be rebuilt on each mesh that needs them
(the working mesh, the normalization mesh) rather than resampled::

    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh = sinv(mesh.attrs)(mesh)
    spectrum = compute_mesh2_spectrum(mesh, bin=bin, los=los)
    spectrum = spectrum.clone(norm=compute_oqe2_normalization(fkp, sinv=sinv, bin=bin),
                              num_shotnoise=compute_oqe2_shotnoise(fkp, sinv=sinv, bin=bin))

References
----------
https://arxiv.org/abs/astro-ph/9304022
https://arxiv.org/abs/2012.09389
"""

from __future__ import annotations

from typing import Callable

import jax
import numpy as np
from jax import numpy as jnp

from .mesh import ComplexMeshField, MeshAttrs, RealMeshField


def to_real(mesh):
    """Return ``mesh`` represented in configuration space."""
    if isinstance(mesh, ComplexMeshField):
        return mesh.c2r()
    return mesh


def to_complex(mesh):
    """Return ``mesh`` represented in Fourier space."""
    if isinstance(mesh, RealMeshField):
        return mesh.r2c()
    return mesh


class SeparableWeighting(object):
    r"""
    The weighting :math:`u \rightarrow W(\mathbf{x})\, \mathcal{F}^{-1}\left[\sum_\alpha f_\alpha(k)
    \mathcal{F}[D_\alpha u](k)\right]`.

    The factors are given as functions of the term index and evaluated when used, one at a time,
    so that a rank-:math:`M` weighting does not hold :math:`2M` meshes. Its normalization and shot
    noise use the separable proxy :math:`\sum_\alpha f_\alpha(k) D_\alpha(\mathbf{x}) W(\mathbf{x})`
    (:meth:`norm_real`), which folds the output-side selection into the configuration-space
    factors: exact for :math:`W = 1`, and otherwise the same slowly-varying approximation the
    closed-form normalization makes anyway.

    Parameters
    ----------
    fourier : callable
        ``ia -> array``: :math:`f_\alpha(k)` on the Fourier grid, or a scalar.
    real : callable
        ``ia -> array``: :math:`D_\alpha(\mathbf{x})` on the configuration grid, or a scalar.
    nterms : int
        Number of terms :math:`M`.
    selection : array or float, default=1.
        Output-side selection :math:`W(\mathbf{x})`.
    """
    def __init__(self, fourier: Callable, real: Callable, nterms: int, selection=1.):
        self.fourier = fourier
        self.real = real
        self.nterms = int(nterms)
        self.selection = selection

    @classmethod
    def from_terms(cls, terms, selection=1.):
        """From explicit ``(fourier, real)`` pairs of arrays or scalars."""
        terms = [(jnp.asarray(fourier), jnp.asarray(real)) for fourier, real in terms]
        return cls(lambda ia: terms[ia][0], lambda ia: terms[ia][1], len(terms), selection=selection)

    def clone(self, **kwargs):
        """A copy with some of ``fourier``, ``real``, ``nterms``, ``selection`` replaced."""
        state = dict(fourier=self.fourier, real=self.real, nterms=self.nterms, selection=self.selection) | kwargs
        return self.__class__(**state)

    def norm_real(self, ia):
        """Configuration-space factor of the normalization proxy, :math:`D_\alpha W`."""
        return self.real(ia) * self.selection

    def __call__(self, mesh):
        """Apply the weighting to ``mesh``; return a :class:`~jaxpower.mesh.RealMeshField`."""
        rmesh = to_real(mesh)
        value = 0.
        for ia in range(self.nterms):
            value = value + to_complex(rmesh.clone(value=rmesh.value * self.real(ia))).value * self.fourier(ia)
        rmesh = to_complex(rmesh).clone(value=value).c2r()
        return rmesh.clone(value=rmesh.value * self.selection)


def separable_inverse_weight(mattrs: MeshAttrs, power: Callable | tuple, selection, noise, nterms: int=4,
                             nodes: np.ndarray=None, nk: int=96, nbins: int=64) -> SeparableWeighting:
    r"""
    A rank-``nterms`` separable approximation to the local inverse covariance,

    .. math:: S^{-1} \approx \sum_\alpha A_\alpha(k) \, g_\alpha(\xv),
              \qquad g_\alpha(\xv) = \frac{1}{W(\xv)^2 B_\alpha + N(\xv)},

    returned as a :class:`SeparableWeighting` without output-side selection, whose factors are
    evaluated on demand from :math:`W`, :math:`N`, the poles and the fitted coefficients.

    Writing the target as
    :math:`1/(W^2 P + N) = W^{-2}\,/\,(P(k) + s(\xv))` with :math:`s = N/W^2` shows it to be a
    Cauchy kernel, whose :math:`\epsilon`-rank grows only like
    :math:`\log(1/\epsilon)\log\kappa` in the dynamic range :math:`\kappa`. Measured on a
    masked box, the singular values fall by roughly a factor eight per term, so three terms
    reach 1% and four reach 0.1% *even with* :math:`s` spanning :math:`5\times10^{12}`. The
    fixed-pole restriction here costs about one extra term against the unconstrained optimum,
    which is a good trade: unlike an exponential-sum factorization, every :math:`g_\alpha` is
    bounded and positive, and tends smoothly to :math:`1/N` where the mask vanishes -- exactly
    as the true operator does.

    Keep ``nterms`` small: the window costs :math:`M^2` ordinary windows, and the error enters
    only the variance, never the bias, so a few per cent is plenty.

    The coefficients :math:`A_\alpha(k)` are fitted per :math:`k` by weighted least squares in
    :math:`s`, in *relative* error -- the operator spans decades, and an absolute fit would
    only capture the signal-dominated regime -- and weighted by the survey's own histogram of
    :math:`s` over cells, so that no terms are spent on values no cell realizes.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes.
    power : callable, tuple
        Fiducial power spectrum :math:`P(k)`, as a callable or a ``(k, P)`` pair.
    selection : array or float
        Selection function :math:`W(\mathbf{x})` of the covariance.
    noise : array or float
        Noise :math:`N(\mathbf{x})`.
    nterms : int, default=4
        Number of terms :math:`M`.
    nodes : array, default=None
        The poles :math:`B_\alpha`. Defaults to ``nterms`` values log-spaced across the range
        of :math:`s = N / W^2` the survey realizes -- geometric, not arithmetic: the Cauchy
        kernel's low-rank structure is geometric and uniform nodes waste terms badly. The
        poles belong in the space of :math:`s`, since the denominator is
        :math:`W^2 B_\alpha + N = W^2 (B_\alpha + s)`.
    nk : int, default=96
        Number of :math:`k` nodes at which the coefficients are fitted, then interpolated.
    nbins : int, default=64
        Number of bins in the histogram of :math:`s` used to weight the fit.

    Returns
    -------
    weighting : SeparableWeighting
    """
    shape = tuple(mattrs.meshsize)
    W = jnp.broadcast_to(jnp.asarray(selection), shape)
    N = jnp.broadcast_to(jnp.asarray(noise), shape)

    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))

    # Everything below stays on device. Nothing is fetched to the host, so nothing has to be
    # gathered off the shards and the build works unchanged across processes. That rules out
    # boolean indexing (a data-dependent shape is invalid under jit), Python branches on
    # device values, and numpy on anything derived from the mesh.
    kmin = jnp.min(jnp.where(knorm > 0., knorm, jnp.inf))
    kfit = jnp.geomspace(kmin, jnp.max(knorm), nk)
    pfit = _evaluate_power(power, kfit)

    # the survey's own distribution of s = N / W^2, as the weight of the fit
    svalue = N / jnp.where(W > 0., W, 1.)**2
    svalue = jnp.where(jnp.isfinite(svalue) & (svalue > 0.), svalue, jnp.nan)
    logs = jnp.log10(svalue)
    lo, hi = jnp.nanmin(logs), jnp.nanmax(logs)
    # a uniform geometry puts every cell at one value of s; give the fit a range to sit in
    hi = jnp.where(hi > lo, hi, lo + 1.)

    # trim the tails from the cumulative histogram rather than by sorting the mesh:
    # a quantile over this many cells is a sort, this is a single pass
    fine = jnp.histogram(logs, bins=4 * nbins, range=(lo, hi))[0]
    cdf = jnp.cumsum(fine) / jnp.maximum(jnp.sum(fine), 1)
    fine_edges = jnp.linspace(lo, hi, 4 * nbins + 1)[1:]
    lo = fine_edges[jnp.searchsorted(cdf, 1e-3)]
    hi = fine_edges[jnp.minimum(jnp.searchsorted(cdf, 1. - 1e-3), 4 * nbins - 1)]
    hi = jnp.where(hi > lo, hi, lo + 1.)

    edges = jnp.logspace(lo, hi, nbins)
    count = jnp.histogram(svalue, bins=edges)[0]
    sj = jnp.sqrt(edges[:-1] * edges[1:])
    # No mask on count > 0: an empty bin gets rho = 0 and drops out of the weighted fit of its
    # own accord, which keeps every shape static and the whole build on device.
    rho = jnp.sqrt(count / jnp.maximum(jnp.sum(count), 1))

    if nodes is None:
        # The poles live in the same space as s, not as P: the denominator is
        # W^2 B + N = W^2 (B + s). They must bracket the range of s the survey actually
        # realizes, or the Cauchy fit extrapolates and the coefficients blow up -- on a real
        # footprint, where W falls to zero at the edges, s spans decades that P never reaches.
        nodes = jnp.geomspace(sj[0], sj[-1], nterms)
    nodes = jnp.asarray(nodes)

    basis = 1. / (nodes[None, :] + sj[:, None])                 # (ns, M)

    def fit(p):
        target = 1. / (p + sj)                                   # the field to represent
        # relative-error, survey-weighted least squares
        return jnp.linalg.lstsq(basis / target[:, None] * rho[:, None], rho, rcond=None)[0]

    coeffs = jax.vmap(fit)(pfit)                                 # (nk, M)

    def fourier(ia):
        knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))
        return jnp.interp(knorm, kfit, coeffs[:, ia])

    def real(ia):
        # Outside the footprint both W and N vanish, and the reciprocal would be infinite. The
        # field is identically zero there, so any finite weight is equivalent; zero keeps the
        # operator and its normalization finite.
        return _invert(W**2 * nodes[ia] + N)

    return SeparableWeighting(fourier, real, nodes.shape[0])


def _as_pair(sinv):
    """One weighting per leg: ``sinv`` itself twice, or the pair given."""
    if isinstance(sinv, (tuple, list)):
        if len(sinv) != 2:
            raise ValueError('pass one weighting, or one per leg')
        return tuple(sinv)
    return (sinv, sinv)


def _contract(matrix, weightings, mattrs: MeshAttrs, bin=None):
    r"""
    Contract an :math:`M_0 \times M_1` matrix of configuration-space integrals with the Fourier
    factors of the two legs, :math:`\mathrm{Re} \sum_{\alpha\beta} f^{(0)}_\alpha(k) f^{(1)*}_\beta(k)
    X_{\alpha\beta}`, and average it over each :math:`k` shell.

    Parameters
    ----------
    matrix : list of list
        :math:`X_{\alpha\beta}`, the configuration-space integrals, as scalars.
    weightings : tuple of SeparableWeighting
        The weightings of the two legs on ``mattrs``; only their Fourier factors are used.
    mattrs : MeshAttrs
        Mesh on which the Fourier factors live.
    bin : BinMesh2SpectrumPoles, default=None
        Binning operator, to average over each :math:`k` shell.

    Returns
    -------
    value : array
        The contraction, shell-averaged if ``bin`` is given, else on the Fourier grid.
    """
    w0, w1 = weightings
    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))
    value = 0.
    for ia in range(w0.nterms):
        fa = w0.fourier(ia)
        for ib in range(w1.nterms):
            value = value + matrix[ia][ib] * fa * jnp.conj(w1.fourier(ib))
    value = jnp.real(jnp.broadcast_to(value, knorm.shape))
    if bin is None:
        return value
    # bin() divides by the mode count, so this is the shell average of A(k)
    return bin(value)


def _norm_matrix(weightings, density):
    """The matrix :math:`\\sum_\\mathbf{x} \\rho\\, D^{(0)}_\\alpha W^{(0)} D^{(1)}_\\beta W^{(1)}`, one term at a time."""
    w0, w1 = weightings
    symmetric = w0 is w1
    matrix = [[None] * w1.nterms for _ in range(w0.nterms)]
    for ia in range(w0.nterms):
        ra = w0.norm_real(ia)
        for ib in range(ia if symmetric else 0, w1.nterms):
            matrix[ia][ib] = jnp.sum(density * ra * w1.norm_real(ib))
            if symmetric: matrix[ib][ia] = matrix[ia][ib]
    return matrix


def compute_oqe2_normalization(*fkps, sinv=None, bin=None, cellsize: float=10., **kwargs):
    r"""
    The power spectrum normalization under a separable OQE weighting.

    The FKP normalization generalizes with one substitution. In the same slowly-varying-window
    approximation the FKP norm already makes, the weighted field is
    :math:`y(k) = \sum_\alpha f_\alpha(k) \, \mathrm{FFT}[D_\alpha \bar{n} w \delta]`, so

    .. math:: \langle |y(k)|^2 \rangle \simeq P(k) A(k),
              \qquad A(k) = \sum_{\alpha\beta} f_\alpha(k) f_\beta^*(k) A_{\alpha\beta},

    .. math:: A_{\alpha\beta} = \alpha_r \int \mathrm{d}\xv \, D_\alpha(\xv) D_\beta(\xv)
              \bar{n}_d(\xv) \bar{n}_r(\xv),

    which is the FKP normalization integral with :math:`D_\alpha D_\beta` inserted. Three
    properties make this the natural convention rather than one among several:

    - it reduces exactly to :func:`~jaxpower.mesh2.compute_fkp2_normalization` at
      :math:`M = 1`, :math:`f = 1`, :math:`D = 1`;
    - it preserves the invariance under :math:`S^{-1} \rightarrow c S^{-1}`, since both the
      numerator and :math:`A` scale as :math:`c^2`. A normalization lacking this would make
      the estimate depend on an arbitrary constant;
    - it makes a single separable term a no-op: under
      :math:`f_\alpha \rightarrow \lambda(k) f_\alpha` both sides pick up :math:`\lambda^2`,
      so at :math:`M = 1` any :math:`f(k)` returns exactly the FKP answer with weight
      :math:`D`. The gain therefore comes only from the :math:`\alpha \neq \beta` cross terms.

    Parameters
    ----------
    fkps : FKPField
        FKP fields.
    sinv : callable, tuple
        ``MeshAttrs -> SeparableWeighting``, rebuilt on the normalization mesh; its proxy
        (:meth:`~SeparableWeighting.norm_real`) is contracted. A pair gives one weighting per leg,
        e.g. for two tracers each weighted by its own. ``None`` for the FKP normalization.
    bin : BinMesh2SpectrumPoles, default=None
        Binning operator. Used to shell-average :math:`A(k)` and to return one entry per
        multipole.
    cellsize : float, default=10.
        Cell size of the normalization mesh.
    kwargs : dict
        Optional arguments for :func:`~jaxpower.mesh.compute_normalization`.

    Returns
    -------
    norm : array, list
    """
    from .mesh import FKPField, _iter_meshes
    from .mesh2 import compute_fkp2_normalization

    if sinv is None:
        return compute_fkp2_normalization(*fkps, bin=bin, cellsize=cellsize, **kwargs)

    for fkp in fkps:
        if not isinstance(fkp, FKPField):
            raise ValueError('an FKPField is required to normalize')
    if len(fkps) == 1:
        # cross data against randoms, as the FKP normalization does, to remove the common noise
        terms = [(fkps[0].data, fkps[0].randoms, fkps[0].data.sum() / fkps[0].randoms.sum(), 1.)]
    elif len(fkps) == 2:
        # Two legs: two tracers, or two legs of one tracer weighted differently, as `optimal_weights`
        # produces for local PNG. As `compute_fkp2_normalization` does for two fields, cross each
        # leg's data against the *other* leg's randoms, so the common noise drops, and symmetrize:
        # both terms pair a leg-0 density with a leg-1 density.
        terms = [(fkps[0].data, fkps[1].randoms, fkps[1].data.sum() / fkps[1].randoms.sum(), 0.5),
                 (fkps[1].data, fkps[0].randoms, fkps[0].data.sum() / fkps[0].randoms.sum(), 0.5)]
    else:
        raise NotImplementedError('OQE normalization takes one or two FKP fields')

    product, mattrs = 0., None
    for data, randoms, alpha, weight in terms:
        meshes = list(_iter_meshes(data, randoms, cellsize=cellsize, **kwargs))
        mattrs = meshes[0].attrs
        # alpha differs between the two terms, so it is folded in here rather than factored out
        product = product + weight * alpha * meshes[0].value * meshes[1].value
        del meshes
    alpha = 1.

    # the configuration-space factors, rebuilt on the normalization mesh
    sinvs = _as_pair(sinv)
    matrix = _norm_matrix([sinv(mattrs) for sinv in sinvs], alpha * product / mattrs.cellsize.prod())

    # the Fourier factors live on the working mesh, not the normalization mesh
    wattrs = bin.mattrs
    norm = _contract(matrix, [sinv(wattrs) for sinv in sinvs], wattrs, bin=bin)
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_oqe2_shotnoise(*fkps, sinv=None, bin=None, fields: tuple=None, cellsize: float=10., **kwargs):
    r"""
    The shot noise under a separable OQE weighting.

    It takes the same form as the normalization, being the same contraction of an
    :math:`M \times M` matrix,

    .. math:: S(k) = \sum_{\alpha\beta} f_\alpha(k) f_\beta^*(k) S_{\alpha\beta},
              \qquad S_{\alpha\beta} = \int \mathrm{d}\xv \, D_\alpha D_\beta
              \left( \bar{n}_{d,w^2} + \alpha_r^2 \bar{n}_{r,w^2} \right),

    with :math:`\bar{n}_{w^2}` the density painted with the squared weights (the product of the
    two legs' weights, and one weighting per leg, for two legs sharing positions; zero for two
    fields at different positions, as :func:`~jaxpower.mesh2.compute_fkp2_shotnoise`). Evaluating it as
    a mesh integral rather than a particle sum is exact up to the smoothness of
    :math:`D_\alpha D_\beta`, which is the same approximation the normalization makes, and it
    keeps the weighting entirely on the mesh.

    Parameters
    ----------
    fkps : FKPField
        FKP fields.
    sinv : callable, tuple
        ``MeshAttrs -> SeparableWeighting``, or one per leg; see :func:`compute_oqe2_normalization`.
    bin : BinMesh2SpectrumPoles, default=None
        Binning operator.
    fields : tuple, default=None
        Field identifiers, as in :func:`~jaxpower.mesh2.compute_fkp2_shotnoise`: pass e.g.
        ``(0, 0)`` for two legs sharing the same positions. Two different fields have no shot noise.
    cellsize : float, default=10.
        Cell size of the mesh used for the integral.
    kwargs : dict
        Optional arguments for painting.

    Returns
    -------
    shotnoise : array, list
    """
    from .mesh import FKPField, _iter_meshes
    from .mesh2 import compute_fkp2_shotnoise, _format_meshes

    if sinv is None:
        return compute_fkp2_shotnoise(*fkps, bin=bin, fields=fields)

    for fkp in fkps:
        if not isinstance(fkp, FKPField):
            raise ValueError('an FKPField is required to estimate the shot noise')
    if len(fkps) > 2:
        raise NotImplementedError('OQE shot noise takes one or two FKP fields')
    fkps, fields = _format_meshes(*fkps, fields=fields)
    if fields[1] != fields[0]:
        # different positions: no shot noise
        return [jnp.zeros_like(bin.xavg) for ell in bin.ells] if bin is not None else 0.

    # fkp.particles is data - alpha * randoms, so squaring its weights gives
    # sum_d w^2 + alpha^2 sum_r w^2 in one pass, exactly as compute_fkp2_shotnoise sums it.
    # Two legs of one tracer share their positions and differ only in weight, so the product
    # w1 w2 replaces the square -- which is what `compute_fkp2_shotnoise` sums when it is told
    # the two fields are the same (fields[1] == fields[0]).
    particles = fkps[0].particles
    particles = particles.clone(weights=particles.weights * fkps[1].particles.weights)
    mesh = next(iter(_iter_meshes(particles, cellsize=cellsize, **kwargs)))
    mattrs = mesh.attrs
    # painting conserves the summed weights, so the cell sum is already sum_i w_i^2
    density = mesh.value
    del mesh

    sinvs = _as_pair(sinv)
    matrix = _norm_matrix([sinv(mattrs) for sinv in sinvs], density)

    wattrs = bin.mattrs
    shotnoise = _contract(matrix, [sinv(wattrs) for sinv in sinvs], wattrs, bin=bin)
    if bin is not None:
        # shot noise enters the monopole only, as in compute_fkp2_shotnoise
        return [shotnoise * (ell == 0) for ell in bin.ells]
    return shotnoise


def _evaluate_power(power, k):
    """Evaluate a power spectrum, given as a callable or a (k, P) pair, on a 1-D array."""
    if power is None:
        return jnp.ones_like(k)
    if callable(power):
        return jnp.broadcast_to(power(k), k.shape)
    kk, pk = power
    return jnp.interp(k, jnp.asarray(kk), jnp.asarray(pk), left=0., right=0.)


def _invert(value):
    """Reciprocal, leaving zeros at zero."""
    good = value > 0.
    return jnp.where(good, 1. / jnp.where(good, value, 1.), 0.)


def _cg(matvec, b, x0=None, *, tol=1e-8, atol=0., maxiter=None, M=None):
    r"""
    Preconditioned conjugate gradients that survives a sharded right-hand side.

    :func:`jax.scipy.sparse.linalg.cg` cannot be used here. Its ``_isolve`` calls
    ``jax.device_put((b, x0))`` with no device argument, which resolves to a single default
    device and refuses a globally sharded, non-fully-addressable array -- so it works on one
    process, where the mesh is a full replica, and fails the moment the field is genuinely
    distributed. Everything below is ordinary jax acting on whatever sharding ``b`` carries:
    the reductions become collectives and no array is ever moved.

    Stops on the *true* residual, :math:`\|Ax - b\| \le \max(\mathrm{tol}\|b\|,
    \mathrm{atol})`, which is the quantity the estimator's diagnostics measure; a
    preconditioned residual would read differently for the same solve.

    Parameters
    ----------
    matvec : callable
        ``x -> A x``, with ``A`` symmetric positive definite.
    b : array
        Right-hand side, of any sharding.
    x0 : array, default=None
        Initial guess. Defaults to zero.
    tol, atol : float
        Relative and absolute tolerances on the residual.
    maxiter : int, default=None
        Iteration cap. Defaults to ``10 * b.size``, as scipy does.
    M : callable, default=None
        Preconditioner ``r -> M r``. Defaults to the identity.

    Returns
    -------
    x : array
    """
    if M is None:
        M = lambda value: value
    if maxiter is None:
        maxiter = 10 * b.size

    def dot(u, v):
        return jnp.sum(u * v)

    threshold = jnp.maximum(tol * jnp.sqrt(dot(b, b)), atol)
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - matvec(x)
    z = M(r)
    rz = dot(r, z)

    def cond(carry):
        x, r, z, p, rz, iteration = carry
        return (jnp.sqrt(dot(r, r)) > threshold) & (iteration < maxiter)

    def body(carry):
        x, r, z, p, rz, iteration = carry
        ap = matvec(p)
        pap = dot(p, ap)
        # guard the degenerate steps rather than branching: a Python branch here would be a
        # host fetch, and on a singular operator pap or rz can reach zero
        alpha = jnp.where(pap != 0., rz / jnp.where(pap != 0., pap, 1.), 0.)
        x = x + alpha * p
        r = r - alpha * ap
        z = M(r)
        rz_new = dot(r, z)
        beta = jnp.where(rz != 0., rz_new / jnp.where(rz != 0., rz, 1.), 0.)
        return (x, r, z, z + beta * p, rz_new, iteration + 1)

    carry = (x, r, z, z, rz, jnp.array(0))
    return jax.lax.while_loop(cond, body, carry)[0]


def covariance_weight(mattrs: MeshAttrs, power: Callable | tuple, selection, noise, tol: float=1e-8,
                      maxiter: int=None, preconditioner: bool=True) -> Callable:
    r"""
    The exact :math:`C^{-1}`, :math:`C = W \Xi W + N`, applied by preconditioned conjugate
    gradients: a reference to validate :func:`separable_inverse_weight` against at the field
    level. It has no closed-form normalization and no window.

    :math:`C` is symmetric positive definite, the domain of conjugate gradients. A solve stopped
    early is not a linear operator, so keep ``tol`` tight. Avoid mesh sizes carrying an FFT radix
    of 3 or 7, where the solve stalls near 1e-3; prefer products of 2 and 5.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes.
    power : callable, tuple
        Fiducial power spectrum :math:`P(k)`, the (isotropic) signal covariance :math:`\Xi`.
    selection : array or float
        Selection function :math:`W(\mathbf{x})`.
    noise : array or float
        Noise :math:`N(\mathbf{x})`, finite everywhere so that :math:`C` is invertible.
    tol : float, default=1e-8
        Relative tolerance on the residual.
    maxiter : int, default=None
        Iteration cap; see :func:`_cg`.
    preconditioner : bool, default=True
        Precondition with the exact configuration-space diagonal of :math:`C`,
        :math:`1 / (W^2 \xi(0) + N)`, which costs no transforms and handles the footprint
        contrast that makes :math:`C` ill-conditioned.

    Returns
    -------
    apply : callable
        ``mesh -> mesh``, applying :math:`C^{-1}`; returns a :class:`~jaxpower.mesh.RealMeshField`.
    """
    from .mesh import _get_hermitian_weights

    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))
    pk = jnp.broadcast_to(_evaluate_power(power, knorm), knorm.shape)
    # xi(0) = sum_k P(k) / N, the configuration-space diagonal of the signal, summed over the full
    # Fourier grid: the Hermitian layout stores half of it, so the conjugate modes are weighted back in
    weights = 1.
    if mattrs.is_hermitian:
        weights = _get_hermitian_weights(mattrs.kcoords(kind='separation', sparse=True), mattrs.meshsize, sharding_mesh=None).reshape(pk.shape)
    xi0 = jnp.sum(pk * weights) / mattrs.meshsize.prod(dtype=mattrs.rdtype)
    diagonal = _invert(jnp.asarray(selection)**2 * xi0 + jnp.asarray(noise))

    def apply(mesh):
        mesh = to_real(mesh)

        def matvec(value):
            signal = mesh.clone(value=selection * value).r2c()
            return selection * signal.clone(value=signal.value * pk).c2r().value + noise * value

        M = (lambda value: diagonal * value) if preconditioner else None
        return mesh.clone(value=_cg(matvec, mesh.value, tol=tol, atol=0., maxiter=maxiter, M=M))

    return apply
