r"""
Optimal quadratic estimator weightings, as linear operators on mesh fields.

The FKP estimator weights each object by a function of :math:`\bar{n}(z)` alone. The optimal
weighting is the inverse data covariance :math:`S^{-1}`, which is not a configuration-space
diagonal at all: it is :math:`k` dependent, and no per-particle weight can carry that. This
module provides the operator algebra needed to apply such a weighting to a painted field,
together with the two weightings that are cheap enough to use in production and the
normalization that goes with them.

The useful class is the **sum of separable terms**,

.. math:: S^{-1} = \sum_\alpha f_\alpha(k) D_\alpha(\xv),

(:class:`SeparableOperator`). It is wide enough to contain a rank-:math:`M` approximation to
the local inverse covariance (:func:`separable_inverse_weight`) and a weighting whose
anisotropy follows the local line of sight (:func:`local_multipole_weight`), and narrow
enough that its normalization stays a contraction of an :math:`M \times M` matrix of
ordinary FKP-like integrals (:func:`compute_oqe2_normalization`).

Usage is a single insertion at the paint step::

    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh = sinv(mesh)                                   # ComplexMeshField
    spectrum = compute_mesh2_spectrum(mesh, bin=bin, los=los)
    spectrum = spectrum.clone(norm=compute_oqe2_normalization(fkp, sinv=sinv, bin=bin),
                              num_shotnoise=compute_oqe2_shotnoise(fkp, sinv=sinv, bin=bin))

Conventions
-----------
A "field" is a real-valued mesh; it may be *represented* either in configuration space
(:class:`~jaxpower.mesh.RealMeshField`) or in Fourier space
(:class:`~jaxpower.mesh.ComplexMeshField`), and operators convert between the two as
needed. With the unnormalized FFT convention of this package,
:math:`\hat{u}(k) = \sum_x u(x) e^{-i k x}`, the two natural inner products are related by
Parseval,

.. math:: \sum_k \hat{u}^*(k) \hat{v}(k) = N \sum_x u(x) v(x),

with :math:`N` the number of mesh cells. The overall factor :math:`N` is representation
*independent*, so transposition is unambiguous: the transpose of a chain is the chain of
transposes in reverse order, with :meth:`r2c` / :meth:`c2r` inserted freely.

Notes
-----
The estimator proper -- the Fisher matrix, the noise bias and the unwindowed spectrum -- lives
in ``jax-oqe``, which builds on this module. Here the weighting is used inside the ordinary
FKP-style estimator, so the mode coupling is carried by the window matrix rather than by a
Fisher normalization.

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

from .mesh import BaseMeshField, ComplexMeshField, MeshAttrs, RealMeshField


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


class MeshOperator(object):
    """
    Base class for linear operators on mesh fields.

    Subclasses implement :meth:`__call__`, and override :attr:`T` if they are not
    self-adjoint.
    """
    def __call__(self, mesh: BaseMeshField) -> BaseMeshField:
        raise NotImplementedError

    @property
    def T(self) -> 'MeshOperator':
        """Transpose. Self-adjoint by default."""
        return self

    def __matmul__(self, other: 'MeshOperator') -> 'MeshOperator':
        """``(self @ other)(mesh) == self(other(mesh))``."""
        return Chain(other, self)

    def __add__(self, other: 'MeshOperator') -> 'MeshOperator':
        """``(self + other)(mesh) == self(mesh) + other(mesh)``."""
        return Sum(self, other)


@jax.tree_util.register_pytree_node_class
class Identity(MeshOperator):
    """Identity operator."""

    def __call__(self, mesh):
        return mesh

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


@jax.tree_util.register_pytree_node_class
class RealOperator(MeshOperator):
    r"""
    Multiplication by a real-space field, :math:`u(\xv) \rightarrow w(\xv) u(\xv)`.

    This is the pointing matrix :math:`P` (with ``value`` the selection function
    :math:`\bar{n}(\xv) w_\mathrm{FKP}(\xv)`), and also the shot-noise operator
    :math:`N` for a Poisson field. It is diagonal in configuration space, hence
    self-adjoint for a real ``value``.

    Parameters
    ----------
    value : RealMeshField, array-like or float
        The configuration-space weight :math:`w(\xv)`.
    """
    def __init__(self, value):
        if isinstance(value, BaseMeshField):
            value = to_real(value).value
        self.value = value

    def __call__(self, mesh):
        mesh = to_real(mesh)
        return mesh.clone(value=mesh.value * self.value)

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.value = children[0]
        return new


@jax.tree_util.register_pytree_node_class
class FourierOperator(MeshOperator):
    r"""
    Multiplication by a Fourier-space filter, :math:`\hat{u}(k) \rightarrow f(k) \hat{u}(k)`.

    This is the usual weighting :math:`S^{-1}`, with :math:`f(k) = 1 / P_\mathrm{fid}(k)`;
    see :func:`ideal_weight`. Self-adjoint provided :math:`f` is real and even,
    :math:`f(-k) = f(k)`, which holds for any function of :math:`|k|` and for
    even Legendre multipoles of :math:`\hat{k} \cdot \hat{n}`.

    Parameters
    ----------
    value : ComplexMeshField, array-like or float
        The Fourier-space filter :math:`f(k)`, on the mesh Fourier grid.
    """
    def __init__(self, value):
        if isinstance(value, BaseMeshField):
            value = to_complex(value).value
        self.value = value

    def __call__(self, mesh):
        mesh = to_complex(mesh)
        return mesh.clone(value=mesh.value * self.value)

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.value = children[0]
        return new


@jax.tree_util.register_pytree_node_class
class Chain(MeshOperator):
    """
    Composition of operators, applied in the order given:
    ``Chain(a, b)(mesh) == b(a(mesh))``.

    Parameters
    ----------
    operators : MeshOperator
        Operators, applied left to right.
    """
    def __init__(self, *operators):
        self.operators = list(operators)

    def __call__(self, mesh):
        for operator in self.operators:
            mesh = operator(mesh)
        return mesh

    @property
    def T(self):
        return Chain(*[operator.T for operator in self.operators[::-1]])

    def tree_flatten(self):
        return (self.operators,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.operators = children[0]
        return new


@jax.tree_util.register_pytree_node_class
class Sum(MeshOperator):
    """
    Sum of operators: ``Sum(a, b)(mesh) == a(mesh) + b(mesh)``.

    The terms may return different representations (one Fourier-diagonal, one
    configuration-diagonal), so the result is assembled in configuration space.

    Parameters
    ----------
    operators : MeshOperator
        Operators to sum.
    """
    def __init__(self, *operators):
        self.operators = list(operators)

    def __call__(self, mesh):
        out = None
        for operator in self.operators:
            value = to_real(operator(mesh))
            out = value if out is None else out.clone(value=out.value + value.value)
        return out

    @property
    def T(self):
        return Sum(*[operator.T for operator in self.operators])

    def tree_flatten(self):
        return (self.operators,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.operators = children[0]
        return new


@jax.tree_util.register_pytree_node_class
class FunctionOperator(MeshOperator):
    """
    Escape hatch: wrap arbitrary callables.

    Use this for a weighting that is not a simple diagonal, e.g. a conjugate-gradient
    solve against the full covariance. It is the caller's responsibility for
    ``transpose`` to be the actual transpose of ``function``;
    :func:`check_transpose` will tell you whether it is.

    Parameters
    ----------
    function : callable
        ``mesh -> mesh``.
    transpose : callable, optional
        Transpose of ``function``. Defaults to ``function`` (self-adjoint).
    """
    def __init__(self, function: Callable, transpose: Callable=None):
        self.function = function
        self.transpose = transpose if transpose is not None else function

    def __call__(self, mesh):
        return self.function(mesh)

    @property
    def T(self):
        return FunctionOperator(self.transpose, self.function)

    def tree_flatten(self):
        return (), (self.function, self.transpose)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@jax.tree_util.register_pytree_node_class
class SeparableOperator(MeshOperator):
    r"""
    A sum of separable terms, :math:`M = \sum_\alpha f_\alpha(k) D_\alpha(\xv)`:

    .. math:: (M u)(k) = \sum_\alpha f_\alpha(k) \int \mathrm{d}\xv\, e^{-ik\xv} D_\alpha(\xv) u(\xv).

    This is the widest class whose normalization stays an ordinary FKP-like integral, see
    :func:`compute_oqe2_normalization`. It is also what a varying-line-of-sight operator
    *is*: taking :math:`\alpha = (\ell, m)`,
    :math:`f_{\ell m}(k) = w_\ell(k) y_{\ell m}(\hat{k})` and
    :math:`D_{\ell m}(\xv) = y_{\ell m}(\hat{\xv}) d(\xv)` gives a weighting whose anisotropy
    is oriented along the local line of sight -- something no single :math:`f(k) D(\xv)` can
    represent.

    Composing on the right with a configuration-space diagonal stays in the class, since
    :math:`D_\alpha \rightarrow D_\alpha W`, which is how the pointing matrix is absorbed;
    see :meth:`with_pointing`. Composing on the right with a Fourier diagonal does not, and
    is not supported.

    The transpose is :math:`M^{T} v = \sum_\alpha \int e^{-ik\xv} D_\alpha(\xv)
    \int_{k'} e^{ik'\xv} f_\alpha^*(k') v(k')`, which is outside the class; it is returned as
    a plain :class:`FunctionOperator`. :math:`M` need not be self-adjoint.

    Parameters
    ----------
    terms : list
        ``(fourier, real)`` pairs, either of which may be ``None`` for one.
    """
    def __init__(self, terms):
        self.terms = [(None if f is None else jnp.asarray(f), None if d is None else jnp.asarray(d)) for f, d in terms]

    def __call__(self, mesh):
        rmesh = to_real(mesh)
        out = None
        for fourier, real in self.terms:
            value = to_complex(rmesh if real is None else rmesh.clone(value=rmesh.value * real)).value
            if fourier is not None: value = value * fourier
            out = value if out is None else out + value
        return to_complex(rmesh).clone(value=out)

    @property
    def T(self):
        terms = self.terms

        def transpose(mesh):
            cmesh = to_complex(mesh)
            out = None
            for fourier, real in terms:
                value = to_real(cmesh if fourier is None else cmesh.clone(value=cmesh.value * fourier.conj()))
                if real is not None: value = value.clone(value=value.value * real)
                out = value.value if out is None else out + value.value
            return to_real(cmesh).clone(value=out)

        return FunctionOperator(transpose, self.__call__)

    def with_pointing(self, pointing: MeshOperator) -> 'SeparableOperator':
        """``self @ pointing``: absorb a configuration-space diagonal applied *before* this."""
        value = _real_diagonal(pointing)
        if value is None:
            raise ValueError('only a configuration-space diagonal can be absorbed into a SeparableOperator')
        return SeparableOperator([(f, value if d is None else d * value) for f, d in self.terms])

    def tree_flatten(self):
        return (self.terms,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.terms = children[0]
        return new


def _power_on_mesh(mattrs: MeshAttrs, power: Callable | tuple=None, inverse: bool=False, fill: float=0.):
    """Evaluate ``power`` on the Fourier grid of ``mattrs``; optionally invert it."""
    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))
    if power is None:
        value = jnp.ones_like(knorm)
    elif callable(power):
        value = power(knorm)
    else:
        k, pk = power
        value = jnp.interp(knorm, jnp.asarray(k), jnp.asarray(pk), left=0., right=0.)
    value = jnp.broadcast_to(value, knorm.shape)
    if inverse:
        value = jnp.where((value == 0.) | ~jnp.isfinite(value), fill, 1. / jnp.where(value == 0., 1., value))
    return value


def _real_diagonal(operator: MeshOperator):
    """The configuration-space diagonal of a real-diagonal operator, or ``None``."""
    if operator is None or isinstance(operator, Identity):
        return 1.
    if isinstance(operator, RealOperator):
        return operator.value
    return None


def _evaluate_power(power, k):
    """Evaluate a power spectrum, given as a callable or a (k, P) pair, on a 1-D array."""
    if power is None:
        return jnp.ones_like(k)
    if callable(power):
        return jnp.broadcast_to(power(k), k.shape)
    kk, pk = power
    return jnp.interp(k, jnp.asarray(kk), jnp.asarray(pk), left=0., right=0.)


def _poles_on_mesh(mattrs: MeshAttrs, poles: dict=None, power: Callable | tuple=None):
    """Evaluate a set of power spectrum multipoles on the Fourier grid. Returns (ells, values)."""
    if poles is None:
        return (0,), [_power_on_mesh(mattrs, power)]
    ells = tuple(sorted(poles))
    if any(ell % 2 for ell in ells):
        raise NotImplementedError('only even multipoles are supported in the signal covariance')
    return ells, [_power_on_mesh(mattrs, poles[ell]) for ell in ells]


def ideal_weight(mattrs: MeshAttrs, power: Callable | tuple=None, fill: float=0.) -> FourierOperator:
    r"""
    The standard FKP-like weighting :math:`S^{-1} = 1 / P_\mathrm{fid}(k)`.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes.
    power : callable, tuple, default=None
        Fiducial power spectrum, as a callable of :math:`k` (the wavenumber norm),
        or a ``(k, P(k))`` tuple of arrays, which is then linearly interpolated.
        If ``None``, unit power (i.e. :math:`S^{-1} = 1`).
    fill : float, default=0.
        Value of :math:`1 / P_\mathrm{fid}` where :math:`P_\mathrm{fid}` vanishes or is
        not defined (including :math:`k = 0`). Zero, i.e. downweight to nothing, is the
        safe default.

    Returns
    -------
    operator : FourierOperator
    """
    return FourierOperator(_power_on_mesh(mattrs, power, inverse=True, fill=fill))


def separable_inverse_weight(mattrs: MeshAttrs, power: Callable | tuple=None, pointing: MeshOperator=None,
                             noise: MeshOperator=None, nterms: int=4, nodes: np.ndarray=None,
                             nk: int=96, nbins: int=64) -> SeparableOperator:
    r"""
    A rank-``nterms`` separable approximation to the local inverse covariance,

    .. math:: S^{-1} \approx \sum_\alpha A_\alpha(k) \, g_\alpha(\xv),
              \qquad g_\alpha(\xv) = \frac{1}{W(\xv)^2 B_\alpha + N(\xv)},

    which is a :class:`SeparableOperator`, so its normalization is exactly computable by
    :func:`compute_oqe2_normalization`.

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

    Keep ``nterms`` small. The normalization costs :math:`M(M+1)/2` pairs, quadratically, and
    the error enters only the variance, never the bias, so a few per cent is plenty.

    The coefficients :math:`A_\alpha(k)` are fitted per :math:`k` by weighted least squares in
    :math:`s`, in *relative* error -- the operator spans decades, and an absolute fit would
    only capture the signal-dominated regime -- and weighted by the survey's own histogram of
    :math:`s` over cells, so that no terms are spent on values no cell realizes.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes.
    power : callable, tuple, default=None
        Fiducial power spectrum :math:`P(k)`.
    pointing : MeshOperator, default=None
        Pointing matrix, a configuration-space diagonal :math:`W(\xv)`.
    noise : MeshOperator, default=None
        Noise covariance, a configuration-space diagonal :math:`N(\xv)`.
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
    operator : SeparableOperator
    """
    wdiag, ndiag = _real_diagonal(pointing), _real_diagonal(noise)
    if wdiag is None or ndiag is None:
        raise ValueError('pointing and noise must be configuration-space diagonals')
    shape = tuple(mattrs.meshsize)
    W = jnp.broadcast_to(jnp.asarray(wdiag), shape)
    N = jnp.broadcast_to(jnp.asarray(ndiag), shape)

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

    terms = []
    for alpha in range(nodes.shape[0]):
        amp = jnp.interp(knorm, kfit, coeffs[:, alpha])
        # Outside the footprint both W and N vanish, and the reciprocal would be infinite.
        # The field is identically zero there, so any finite weight is equivalent; zero is the
        # one that keeps the operator and its normalization finite. A survey fills a small
        # fraction of its cube, so this is the common case, not an edge case.
        terms.append((amp, _invert(W**2 * nodes[alpha] + N)))
    return SeparableOperator(terms)


def local_multipole_weight(mattrs: MeshAttrs, poles: dict=None, shotnoise: float=0.,
                           diagonal: MeshOperator=None, ells: tuple=(0, 2), nmu: int=65) -> SeparableOperator:
    r"""
    A weighting whose anisotropy follows the **local** line of sight, as a
    :class:`SeparableOperator`.

    It is the local, WKB-style inverse of the data covariance: at each point the weighting
    behaves as :math:`1 / (P(k, \hat{k}\cdot\hat{\xv}) + \bar{n}^{-1})`, expanded in Legendre
    multipoles of the *local* :math:`\hat{k} \cdot \hat{\xv}`,

    .. math:: S^{-1} = \sum_{\ell m} w_\ell(k) y_{\ell m}(\hat{k}) \cdot y_{\ell m}(\hat{\xv}) d(\xv),
              \qquad w_\ell(k) = \frac{2\ell + 1}{2} \int \mathrm{d}\mu \frac{L_\ell(\mu)}{P(k, \mu) + \bar{n}^{-1}}.

    Stop at :math:`\ell = 2`: going to :math:`\ell = 4` adds almost nothing while taking
    five times as long, since the normalization costs
    :math:`N_\alpha(N_\alpha + 1)/2` pairs and :math:`N_\alpha` goes from 6 to 15.

    Warning
    -------
    As built this is a control variate, not an estimator weighting. Measured on a
    local-line-of-sight Kaiser field behind a mask filling 7% of the box, it gives 2.3 times
    the scatter of a plain :func:`ideal_weight`. The reason is a shape mismatch, not a scale
    one: :math:`w_\ell(k)` is already an inverse power, and multiplying it by a ``diagonal``
    that is itself :math:`1/(W^2\xi(0) + N)` downweights twice over. Rescaling ``diagonal`` by
    a constant cannot fix that -- the estimator is exactly invariant under
    :math:`S^{-1} \rightarrow c S^{-1}` -- and indeed doing so reproduces the same numbers to
    every digit. What would be needed is for the two factors to *jointly* approximate
    :math:`1/(W(\xv)^2 P(k, \mu) + N(\xv))`, rather than each contributing an inverse of its
    own. Prefer :func:`separable_inverse_weight`, which does exactly that.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes.
    poles : dict, default=None
        Power spectrum multipoles ``{ell: callable or (k, P_ell(k))}`` defining
        :math:`P(k, \mu)`.
    shotnoise : float, default=0.
        Representative noise power inside the footprint, added to :math:`P(k, \mu)`.
    diagonal : MeshOperator, default=None
        The configuration-space factor :math:`d(\xv)`, typically the inverse selection
        function. Defaults to one.
    ells : tuple, default=(0, 2)
        Multipole orders to carry. Even only.
    nmu : int, default=65
        Points in the :math:`\mu` quadrature for :math:`w_\ell`.

    Returns
    -------
    operator : SeparableOperator
    """
    from .utils import get_Ylm, get_legendre

    ells = tuple(ells)
    if any(ell % 2 for ell in ells):
        raise NotImplementedError('only even multipoles are supported')
    xvec, kvec = mattrs.rcoords(sparse=True), mattrs.kcoords(sparse=True)
    knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
    _, values = _poles_on_mesh(mattrs, poles=poles)
    pells = tuple(sorted(poles)) if poles is not None else (0,)

    mus = np.linspace(-1., 1., nmu)
    weights = np.trapezoid(np.eye(nmu), mus, axis=0)  # quadrature weights of the mu integral

    diag = _real_diagonal(diagonal)
    if diag is None:
        raise ValueError('diagonal must be a configuration-space diagonal')

    terms = []
    for ell in ells:
        wl = jnp.zeros_like(knorm)
        for mu, dw in zip(mus, weights):  # accumulate rather than stack, to hold one mesh at a time
            power = sum(value * get_legendre(pell)(mu) for pell, value in zip(pells, values))
            wl = wl + dw * get_legendre(ell)(mu) / (power + shotnoise)
        wl = (2 * ell + 1) / 2. * wl
        for m in range(-ell, ell + 1):
            ylm = get_Ylm(ell, m, reduced=True, real=True)
            terms.append((wl * ylm(*kvec), ylm(*xvec) * diag))
    return SeparableOperator(terms)


def check_transpose(operator: MeshOperator, mattrs: MeshAttrs, seed: int=42) -> float:
    r"""
    Numerically check that ``operator.T`` really is the transpose of ``operator``.

    Draws two random real fields :math:`u, v` and returns the relative mismatch of
    :math:`\langle v, A u \rangle` and :math:`\langle A^{T} v, u \rangle`, where the
    inner product is :math:`\sum_\xv u(\xv) v(\xv)` in configuration space. A correct
    transpose returns ~0 (up to floating-point round-off).

    Parameters
    ----------
    operator : MeshOperator
        Operator to test.
    mattrs : MeshAttrs
        Mesh attributes.
    seed : int, default=42
        Random seed.

    Returns
    -------
    mismatch : float
    """
    from jax import random

    keys = random.split(random.key(seed), 2)
    u, v = (mattrs.create(kind='real', fill=random.normal(key, tuple(mattrs.meshsize), dtype=mattrs.rdtype)) for key in keys)
    lhs = jnp.sum(to_real(v).value * to_real(operator(u)).value)
    rhs = jnp.sum(to_real(operator.T(v)).value * to_real(u).value)
    return float(abs(lhs - rhs) / (abs(lhs) + abs(rhs)) * 2.)


def _as_separable(sinv, mattrs: MeshAttrs) -> SeparableOperator:
    """
    Resolve ``sinv`` to a :class:`SeparableOperator` on ``mattrs``.

    ``sinv`` may be a callable of :class:`MeshAttrs`, which is what allows the configuration
    space factors to be rebuilt on the coarse normalization mesh rather than resampled from
    the working one. A plain operator is accepted only if it already lives on ``mattrs``.
    """
    # either nesting is accepted: a Weighting of two factories, or a factory returning a
    # Weighting. The second is what a builder that memoizes by mesh naturally produces, since
    # one build yields both the field operator and its proxy.
    if callable(sinv) and not isinstance(sinv, (MeshOperator, Weighting)):
        sinv = sinv(mattrs)
    if isinstance(sinv, Weighting):  # normalize with the separable proxy, not the field operator
        sinv = sinv.norm
    if callable(sinv) and not isinstance(sinv, MeshOperator):
        sinv = sinv(mattrs)
    if isinstance(sinv, (Identity, type(None))):
        return SeparableOperator([(None, None)])
    if isinstance(sinv, FourierOperator):
        return SeparableOperator([(sinv.value, None)])
    if isinstance(sinv, RealOperator):
        return SeparableOperator([(None, sinv.value)])
    if not isinstance(sinv, SeparableOperator):
        raise ValueError(f'{sinv} is not a separable weighting, so it has no closed-form normalization; '
                         'wrap it in a Weighting(field=..., norm=<separable proxy>), or pass a '
                         'SeparableOperator, a diagonal, or a callable of MeshAttrs returning one')
    for _, real in sinv.terms:
        if real is not None and jnp.ndim(real) and tuple(jnp.shape(real)) != tuple(mattrs.meshsize):
            raise ValueError('the configuration-space factors of sinv do not live on the requested mesh; '
                             'pass sinv as a callable of MeshAttrs so it can be rebuilt there')
    return sinv


def resolve_field(sinv, mattrs: MeshAttrs) -> MeshOperator:
    """
    Resolve a weighting specification to the operator that acts on the painted field.

    Accepts an operator, a :class:`Weighting` (whose ``field`` is taken, not its
    normalization proxy), or a callable of :class:`MeshAttrs` returning either.
    """
    if callable(sinv) and not isinstance(sinv, (MeshOperator, Weighting)):
        sinv = sinv(mattrs)
    if isinstance(sinv, Weighting):
        sinv = sinv.field
    if callable(sinv) and not isinstance(sinv, MeshOperator):
        sinv = sinv(mattrs)
    if not isinstance(sinv, MeshOperator):
        raise ValueError(f'{sinv} is not a mesh operator, nor a callable of MeshAttrs returning one')
    return sinv


def as_weight_pair(sinv):
    r"""
    Normalize a weighting specification to the two legs of the quadratic.

    The estimator is a quadratic form, so it has two legs, and they need not carry the same
    weighting. Passing a pair ``(sinv1, sinv2)`` of weightings built from **independent**
    subsets of the randoms is what removes the noise bias of a weighting estimated from the
    data itself: writing :math:`D = \bar{D} + \delta`, the cross expectation

    .. math:: \langle D^{(1)}_\alpha(\xv) D^{(2)}_\beta(\xv') \rangle
              = \bar{D}_\alpha(\xv) \bar{D}_\beta(\xv')

    has no :math:`\langle \delta \delta \rangle` term at all, because the two noises are
    independent. The same cancellation is what makes
    :func:`~jaxpower.mesh2.compute_fkp2_normalization` cross the data against the randoms.

    Returns
    -------
    sinv1, sinv2 : the two legs, the same object twice if only one was given.
    """
    if isinstance(sinv, (tuple, list)):
        if len(sinv) != 2:
            raise ValueError('pass one weighting, or a pair for the two legs of the quadratic')
        return tuple(sinv)
    return (sinv, sinv)


def _real_factors(sinv: SeparableOperator, shape):
    """The configuration-space factors :math:`D_\\alpha`, broadcast to ``shape``."""
    return [jnp.broadcast_to(jnp.asarray(1. if real is None else real), shape) for _, real in sinv.terms]


def _contract(matrix, sinv1: SeparableOperator, sinv2: SeparableOperator, mattrs: MeshAttrs, bin=None):
    r"""
    Contract an :math:`M \times M` matrix of configuration-space integrals with the Fourier
    factors of the two legs,
    :math:`\sum_{\alpha\beta} f^{(1)}_\alpha(k) f^{(2)*}_\beta(k) X_{\alpha\beta}`, and average
    it over each :math:`k` shell.

    The real part is taken: the estimator contracts the quadratic into a real scalar, and for
    a single leg the sum already pairs :math:`(\alpha, \beta)` with :math:`(\beta, \alpha)`.
    """
    f1 = [None if f is None else jnp.asarray(f) for f, _ in sinv1.terms]
    f2 = [None if f is None else jnp.asarray(f) for f, _ in sinv2.terms]
    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))
    value = None
    for alpha, fa in enumerate(f1):
        for beta, fb in enumerate(f2):
            term = jnp.broadcast_to(jnp.asarray(matrix[alpha][beta]), knorm.shape)
            if fa is not None: term = term * fa
            if fb is not None: term = term * fb.conj()
            value = term if value is None else value + term
    value = jnp.real(value)
    if bin is None:
        return value
    # bin() divides by the mode count, so this is the shell average of A(k)
    return bin(value)


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

    For an anisotropic weighting such as :func:`local_multipole_weight`, where
    :math:`f_{\ell m}` carries :math:`y_{\ell m}(\hat{k})`, :math:`A` mixes multipoles and is
    no longer a scalar per :math:`k` per :math:`\ell`. What is returned is then the
    shell-averaged contraction, and the residual mixing belongs in the window matrix.

    Parameters
    ----------
    fkps : FKPField
        FKP fields.
    sinv : SeparableOperator, MeshOperator or callable
        The weighting. Pass a callable of :class:`MeshAttrs` so that the configuration-space
        factors can be rebuilt on the (coarser) normalization mesh; they are smooth by
        construction, so this costs nothing in accuracy and avoids resampling.
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
        # Two legs of one tracer, weighted differently -- what `optimal_weights` produces for
        # local PNG. This mirrors `compute_fkp2_normalization` term for term for two fields:
        # cross each leg's data against the *other* leg's randoms, so the common noise drops,
        # and symmetrize. It reduces to the single-leg expression when the legs are equal.
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

    # the configuration-space factors of each leg, rebuilt on the normalization mesh
    sinv1, sinv2 = as_weight_pair(sinv)
    reals1 = _real_factors(_as_separable(sinv1, mattrs), tuple(mattrs.meshsize))
    reals2 = _real_factors(_as_separable(sinv2, mattrs), tuple(mattrs.meshsize))
    cellvolume = mattrs.cellsize.prod()
    matrix = [[alpha * jnp.sum(product * ra * rb) / cellvolume for rb in reals2] for ra in reals1]

    # the Fourier factors live on the working mesh, not the normalization mesh
    wattrs = bin.mattrs
    norm = _contract(matrix, _as_separable(sinv1, wattrs), _as_separable(sinv2, wattrs), wattrs, bin=bin)
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_oqe2_shotnoise(*fkps, sinv=None, bin=None, cellsize: float=10., **kwargs):
    r"""
    The shot noise under a separable OQE weighting.

    It takes the same form as the normalization, being the same contraction of an
    :math:`M \times M` matrix,

    .. math:: S(k) = \sum_{\alpha\beta} f_\alpha(k) f_\beta^*(k) S_{\alpha\beta},
              \qquad S_{\alpha\beta} = \int \mathrm{d}\xv \, D_\alpha D_\beta
              \left( \bar{n}_{d,w^2} + \alpha_r^2 \bar{n}_{r,w^2} \right),

    with :math:`\bar{n}_{w^2}` the density painted with the squared weights. Evaluating it as
    a mesh integral rather than a particle sum is exact up to the smoothness of
    :math:`D_\alpha D_\beta`, which is the same approximation the normalization makes, and it
    keeps the weighting entirely on the mesh.

    Parameters
    ----------
    fkps : FKPField
        FKP fields.
    sinv : SeparableOperator, MeshOperator or callable
        The weighting, see :func:`compute_oqe2_normalization`.
    bin : BinMesh2SpectrumPoles, default=None
        Binning operator.
    cellsize : float, default=10.
        Cell size of the mesh used for the integral.
    kwargs : dict
        Optional arguments for painting.

    Returns
    -------
    shotnoise : array, list
    """
    from .mesh import FKPField, _iter_meshes
    from .mesh2 import compute_fkp2_shotnoise

    if sinv is None:
        return compute_fkp2_shotnoise(*fkps, bin=bin)

    for fkp in fkps:
        if not isinstance(fkp, FKPField):
            raise ValueError('an FKPField is required to estimate the shot noise')
    if len(fkps) > 2:
        raise NotImplementedError('OQE shot noise takes one or two FKP fields')

    # fkp.particles is data - alpha * randoms, so squaring its weights gives
    # sum_d w^2 + alpha^2 sum_r w^2 in one pass, exactly as compute_fkp2_shotnoise sums it.
    # Two legs of one tracer share their positions and differ only in weight, so the product
    # w1 w2 replaces the square -- which is what `compute_fkp2_shotnoise` sums when it is told
    # the two fields are the same (fields[1] == fields[0]).
    particles = fkps[0].particles
    weights = particles.weights * (fkps[-1].particles.weights if len(fkps) == 2 else particles.weights)
    particles = particles.clone(weights=weights)
    mesh = next(iter(_iter_meshes(particles, cellsize=cellsize, **kwargs)))
    mattrs = mesh.attrs
    # painting conserves the summed weights, so the cell sum is already sum_i w_i^2
    density = mesh.value
    del mesh

    sinv1, sinv2 = as_weight_pair(sinv)
    reals1 = _real_factors(_as_separable(sinv1, mattrs), tuple(mattrs.meshsize))
    reals2 = _real_factors(_as_separable(sinv2, mattrs), tuple(mattrs.meshsize))
    matrix = [[jnp.sum(density * ra * rb) for rb in reals2] for ra in reals1]

    wattrs = bin.mattrs
    shotnoise = _contract(matrix, _as_separable(sinv1, wattrs), _as_separable(sinv2, wattrs), wattrs, bin=bin)
    if bin is not None:
        # shot noise enters the monopole only, as in compute_fkp2_shotnoise
        return [shotnoise * (ell == 0) for ell in bin.ells]
    return shotnoise


def _invert(value):
    """Reciprocal, leaving zeros at zero."""
    good = value > 0.
    return jnp.where(good, 1. / jnp.where(good, value, 1.), 0.)


def _signal_variance(mattrs: MeshAttrs, power: Callable | tuple=None) -> jax.Array:
    r"""
    :math:`\xi(0) = N^{-1} \sum_k P_\mathrm{fid}(k)`, the configuration-space diagonal of the
    signal operator, summed over the *full* Fourier grid (the Hermitian layout stores only
    half of it, so the conjugate modes are weighted back in).
    """
    from .mesh import _get_hermitian_weights

    pk = _power_on_mesh(mattrs, power)
    weights = 1.
    if mattrs.is_hermitian:
        weights = _get_hermitian_weights(mattrs.kcoords(kind='separation', sparse=True), sharding_mesh=None).reshape(pk.shape)
    return jnp.sum(pk * weights) / mattrs.meshsize.prod(dtype=mattrs.rdtype)


def covariance_operator(mattrs: MeshAttrs, power: Callable | tuple=None, pointing: MeshOperator=None,
                        noise: MeshOperator=None) -> MeshOperator:
    r"""
    The data covariance :math:`C = P \Xi P^{T} + N`, as an operator.

    :math:`\Xi` is the signal covariance, diagonal in Fourier space with
    :math:`P_\mathrm{fid}(k)` on its diagonal; :math:`P` is the pointing matrix and
    :math:`N` the noise, both diagonal in configuration space. No basis diagonalizes both,
    which is the whole reason the inverse has to be applied iteratively.

    The signal is taken isotropic here. That is a weighting, not a model: it need only be
    approximately right, and an anisotropic signal with a varying line of sight costs some
    27 times as many transforms per application.

    Returns
    -------
    operator : MeshOperator
        Self-adjoint.
    """
    signal = FourierOperator(_power_on_mesh(mattrs, power))
    term = signal if pointing is None else Chain(pointing.T, signal, pointing)
    if noise is None:
        return term
    return Sum(term, noise)


def real_jacobi(mattrs: MeshAttrs, power: Callable | tuple=None, pointing: MeshOperator=None,
                noise: MeshOperator=None) -> RealOperator:
    r"""
    Configuration-space Jacobi preconditioner,
    :math:`1 / (W(\xv)^2 \xi(0) + N(\xv))` -- the exact diagonal of :math:`C`.

    It resolves the mask and the noise cell by cell, and sees the signal only through its
    variance :math:`\xi(0)`. That is the one that matters for a survey geometry, where the
    configuration-space contrast between the footprint and the outside is what makes
    :math:`C` ill-conditioned. Being a configuration-space diagonal, it costs no transforms.

    This is the default, and it is not close: a Fourier-space Jacobi preconditioner is worse
    than no preconditioner at all on total transforms, since it doubles the per-iteration
    cost without buying back enough iterations.
    """
    wdiag, ndiag = _real_diagonal(pointing), _real_diagonal(noise)
    if wdiag is None or (noise is not None and ndiag is None):
        return None
    diag = jnp.asarray(wdiag)**2 * _signal_variance(mattrs, power)
    if noise is not None:
        diag = diag + jnp.asarray(ndiag)
    return RealOperator(_invert(diag))


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


def covariance_weight(mattrs: MeshAttrs, power: Callable | tuple=None, pointing: MeshOperator=None,
                      noise: MeshOperator=None, preconditioner: MeshOperator | str='real',
                      tol: float=1e-8, maxiter: int=None, solver: str='cg') -> FunctionOperator:
    r"""
    The minimum-variance weighting :math:`S^{-1} = C^{-1}`, applied by preconditioned
    conjugate gradients.

    This only buys variance, never correctness: the estimator is unbiased for *any*
    weighting, provided the normalization is computed from the same one. It is worth its cost
    where the geometry is severe.

    Being iterative, it is not a :class:`SeparableOperator`, so
    :func:`compute_oqe2_normalization` cannot normalize it in closed form. Normalize it with
    :func:`compute_oqe2_mc_normalization`, which probes the response directly and holds for
    any operator.

    Warning
    -------
    A conjugate gradient solve stopped early is not a linear operator: :math:`S^{-1}` then
    depends on the data and no longer matches the one the normalization was built from.
    :func:`check_transpose` catches it -- its mismatch tracks ``tol``.

    With a varying line of sight, avoid mesh sizes carrying an FFT radix of 3 or 7: the solve
    stalls near 1e-3 at 12, 14 and 24, and converges normally at 16, 20, 32 and 64. The
    operator is fine there; it is the compiled solver loop. Prefer products of 2 and 5.

    Returns
    -------
    operator : FunctionOperator
        Self-adjoint, since :math:`C^{-1}` is.
    """
    covariance = covariance_operator(mattrs, power=power, pointing=pointing, noise=noise)

    if isinstance(preconditioner, str):
        if preconditioner == 'real':
            preconditioner = real_jacobi(mattrs, power=power, pointing=pointing, noise=noise)
        elif preconditioner == 'none':
            preconditioner = None
        else:
            raise ValueError(f'unknown preconditioner {preconditioner}')

    if solver != 'cg':
        # Not an arbitrary restriction. C = W Xi W^T + N is symmetric positive definite by
        # construction -- Xi is a Fourier diagonal with P(k) >= 0, so W^T Xi W is positive
        # semi-definite, and the noise floor keeps N > 0 everywhere -- which is precisely the
        # domain of conjugate gradients. bicgstab and gmres exist for non-symmetric operators;
        # if C were non-symmetric then cg would be wrong here and check_transpose would say so.
        # They would also drag in jax.scipy.sparse.linalg's device_put, so offering them would
        # mean offering something both unnecessary and single-process only.
        raise ValueError(f'{solver!r} is not supported: C is symmetric positive definite, so '
                         'conjugate gradients is the appropriate solver')
    solve = _cg

    def apply(mesh):
        mesh = to_real(mesh)

        def matvec(value):
            return to_real(covariance(mesh.clone(value=value))).value

        M = None
        if preconditioner is not None:
            def M(value):
                return to_real(preconditioner(mesh.clone(value=value))).value

        return mesh.clone(value=solve(matvec, mesh.value, tol=tol, atol=0., maxiter=maxiter, M=M))

    return FunctionOperator(apply)


class Weighting(object):
    r"""
    A weighting to apply to the field, together with the separable proxy that normalizes it.

    For a :class:`SeparableOperator` the two are the same object and the normalization is
    exact. For a weighting that is not separable -- a conjugate gradient solve against the
    full covariance, say -- there is no closed-form :math:`A(k)`, and the proxy supplies one.

    An approximate normalization is legitimate here, which is worth being explicit about.
    This estimator keeps the FKP-style convention and lets the **window matrix** carry the
    residual mode coupling. :math:`A(k)` therefore does not have to be right in any absolute
    sense: it has to be *defined and reproducible*, because the window maps theory to
    measurement through whatever convention was used. A poorly chosen :math:`A(k)` leaves a
    smooth multiplicative factor, which the window reproduces exactly and which the fit then
    sees as part of the geometry. It costs a less diagonal window, not a bias.

    The one hard requirement is that the window be computed with the same ``field`` operator
    and the same ``norm`` proxy as the measurement. Mismatch them and the multiplicative
    factor no longer cancels.

    Parameters
    ----------
    field : callable
        ``MeshAttrs -> MeshOperator``, applied to the painted field.
    norm : callable, default=None
        ``MeshAttrs -> SeparableOperator``, used for the normalization and the shot noise.
        Defaults to ``field``, which then has to be separable.
    """
    def __init__(self, field, norm=None):
        self.field = field
        self.norm = norm if norm is not None else field

    def __call__(self, mattrs):
        return resolve_field(self.field, mattrs)


def compute_oqe2_mc_normalization(*fkps, sinv=None, bin=None, nmocks: int=4, seed: int=42,
                                  resampler='cic', interlacing=0, compensate=False):
    r"""
    The normalization of an arbitrary weighting, by probing it with a white field.

    :math:`A(k)` is the response of the estimator to unit power,
    :math:`A(k) = \partial \langle |y(k)|^2 \rangle / \partial P(k)`. Probe it directly:

    .. math:: A(k) = \left\langle [M(W^{(1)} u)](k) \,
              [M(W^{(2)} u)]^{*}(k) \right\rangle_k

    with :math:`u` a unit-power white field and :math:`M` whatever weighting is applied. The
    field is white, not fiducial, because the response wanted is the one to unit power; and
    it is the selection function :math:`W` that carries the geometry, exactly as in the
    slowly-varying picture where the weighted field is :math:`W \delta`.

    This holds for **any** operator, so it is what normalizes a conjugate gradient weighting,
    which has no closed form. Where the weighting is separable, prefer
    :func:`compute_oqe2_normalization`: it is exact and costs no realizations.

    The two :math:`W` come from **disjoint halves of the randoms**. Squaring a single
    :math:`W` would square its Poisson noise along with it, and that noise does not average
    away -- it is the same bias that
    :func:`~jaxpower.mesh2.compute_fkp2_normalization` avoids by crossing the data against
    the randoms rather than squaring either. Crossing two disjoint halves removes it, because
    their noises are independent and the cross term has zero mean.

    At :math:`M = 1` this reduces to :math:`\alpha^2 \int n_r^{(1)} n_r^{(2)}`, k-independent
    and equal to the FKP normalization -- which is the null test that pins the convention.

    Parameters
    ----------
    fkps : FKPField
        FKP field, supplying the randoms and :math:`\alpha`.
    sinv : MeshOperator, Weighting or callable
        The weighting, resolved by :func:`resolve_field`. A pair is accepted for the two legs.
    bin : BinMesh2SpectrumPoles
        Binning operator.
    nmocks : int, default=4
        White realizations to average. The estimate is unbiased at any ``nmocks``; more only
        reduces its scatter, which enters the measurement as a calibration and so does not
        average down over data realizations.

    Returns
    -------
    norm : list
    """
    from .mesh import FKPField
    from jax import random

    for fkp in fkps:
        if not isinstance(fkp, FKPField):
            raise ValueError('an FKPField is required to normalize')
    if len(fkps) > 2:
        raise NotImplementedError('OQE normalization takes one or two FKP fields')

    mattrs = bin.mattrs
    kw = dict(resampler=resampler, interlacing=interlacing, compensate=compensate, out='real')
    # One FKP field probes both legs with the same randoms; two legs of one tracer -- what
    # `optimal_weights` produces -- probe each leg with its own weights. The positions are
    # shared either way, so the same disjoint split serves both: leg one takes one half and
    # leg two the other, and no object contributes to both legs.
    each = list(fkps) * 2 if len(fkps) == 1 else list(fkps)
    split = random.uniform(random.key(seed), (each[0].randoms.size,)) < 0.5
    pointings = []
    for fkp, mask in zip(each, (split, ~split)):
        alpha = fkp.data.sum() / fkp.randoms.sum()
        randoms = fkp.randoms.clone(attrs=mattrs)
        half = randoms[mask]
        half = half.clone(weights=half.weights * (randoms.sum() / half.sum()))
        pointings.append(alpha * half.paint(**kw).value)

    sinv1, sinv2 = as_weight_pair(sinv)
    operators = [resolve_field(s, mattrs) for s in (sinv1, sinv2)]

    value = 0.
    for imock in range(nmocks):
        # unit POWER, not unit variance per cell: the response wanted is to P(k) = 1, and
        # painting conventions put a cell volume between the two
        u = random.normal(random.key(seed + 1 + imock), tuple(mattrs.meshsize), dtype=mattrs.rdtype)
        u = u / jnp.sqrt(mattrs.cellsize.prod())
        legs = [to_complex(operator(mattrs.create(kind='real', fill=u * pointing))).value
                for operator, pointing in zip(operators, pointings)]
        value = value + jnp.real(legs[0] * legs[1].conj())
    norm = bin(value / nmocks)
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_oqe2_mc_shotnoise(*fkps, sinv=None, bin=None, nmocks: int=4, seed: int=42,
                              resampler='cic', interlacing=0, compensate=False):
    r"""
    The shot noise of an arbitrary weighting, by probing it with the survey's own noise.

    The companion of :func:`compute_oqe2_mc_normalization`: where that probes the response to
    unit *power*, this probes the response to the *noise*,

    .. math:: S(k) = \left\langle |M \nu|^2(k) \right\rangle_k,
              \qquad \langle \nu^2 \rangle(\xv) = N(\xv),

    with :math:`N` the Poisson variance, painted as the summed squared weights of
    ``fkp.particles``. Unlike the normalization, the two legs are not crossed here: the noise
    is the field's own, not an estimate of it, so squaring it is what is wanted.

    Returns
    -------
    shotnoise : list
    """
    from .mesh import FKPField
    from jax import random

    for fkp in fkps:
        if not isinstance(fkp, FKPField):
            raise ValueError('an FKPField is required to estimate the shot noise')
    if len(fkps) > 2:
        raise NotImplementedError('OQE shot noise takes one or two FKP fields')

    mattrs = bin.mattrs
    cellvolume = mattrs.cellsize.prod()
    particles = fkps[0].particles
    kw = dict(resampler=resampler, interlacing=interlacing, compensate=compensate, out='real')
    # Summed squared weights per cell, the Poisson variance of the painted field. Two legs of
    # one tracer share their positions, so the product w1 w2 replaces the square -- the same
    # substitution `compute_fkp2_shotnoise` makes when told the two fields are the same.
    weights = particles.weights * (fkps[-1].particles.weights if len(fkps) == 2 else particles.weights)
    variance = jnp.clip(particles.clone(weights=weights, attrs=mattrs).paint(**kw).value, 0., None)

    sinv1, sinv2 = as_weight_pair(sinv)
    operators = [resolve_field(s, mattrs) for s in (sinv1, sinv2)]

    value = 0.
    for imock in range(nmocks):
        u = random.normal(random.key(seed + 1 + imock), tuple(mattrs.meshsize), dtype=mattrs.rdtype)
        nu = mattrs.create(kind='real', fill=u * jnp.sqrt(variance))
        legs = [to_complex(operator(nu)).value for operator in operators]
        value = value + jnp.real(legs[0] * legs[1].conj())
    shotnoise = bin(value / nmocks)
    if bin is not None:
        # as in compute_fkp2_shotnoise, the Poisson term enters the monopole only
        return [shotnoise * (ell == 0) for ell in bin.ells]
    return shotnoise
