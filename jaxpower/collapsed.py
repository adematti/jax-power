r"""
Collapsed covariance spectra: the (P, B) covariance from the soft modes of the estimator integrands.

Every mesh estimator in this package is a sum over the box of a product of shell-filtered fields,

.. math::
    \hat S_a = \frac{1}{N_a} \sum_x F_a(x), \qquad N_a = \sum_x n_a(x),

where :math:`F_a(x)` is the estimator integrand -- the product that the bin operator's reduce step
sums -- and :math:`n_a(x)` the same product built from the unit field, so that :math:`N_a` is the
number of closed polygons. Following arXiv:2604.18581, transform the integrand instead of summing
it,

.. math::
    F_a(q) = \sum_x e^{-i q x} F_a(x), \qquad n_a(q) = \sum_x e^{-i q x} n_a(x),

and define, per soft shell :math:`|q|`, the collapsed covariance spectrum of two configurations

.. math::
    C_{ab}(q) = \frac{\langle F_a(q) F_b(q)^* \rangle_{|q| \in \mathrm{shell}}}
                     {\langle n_a(q) n_b(q)^* \rangle_{|q| \in \mathrm{shell}}}.

At :math:`q = 0` this is exactly :math:`\hat S_a \hat S_b`, the sample-covariance summand. At
:math:`q \neq 0` the :math:`\langle \hat S_a \rangle \langle \hat S_b \rangle` pairing needs
:math:`q = 0` and vanishes in a periodic box, so no mean is subtracted; the expectation of
:math:`C_{ab}(q)` is the bin-averaged collapsed :math:`2n`-point function at soft momentum
:math:`q`, whose :math:`q \to 0` limit is the covariance block: :math:`T(k, -k, k', -k') / V` for
:math:`\mathrm{Cov}[P, P]`, :math:`P_5 / V` for :math:`\mathrm{Cov}[P, B]`, :math:`P_6 / V` for
:math:`\mathrm{Cov}[B, B]`, multipole weights included. Every soft mode is an independent draw of
that object, where the sample covariance uses the single :math:`q = 0` mode per realization.

**Which families are flat in q.** A term's count of contributing configurations depends on the
soft mismatch only through legs that are both binned and tied to each other across the two
estimators. With the normalisation above, every family is flat in :math:`q` (up to the physical
:math:`P_{\rm lin}(q)` response term of the consistency relation) except two:

* the Gaussian term of :math:`\mathrm{Cov}[P, P]` on a shared shell, whose pairing count is one
  power of the pair count :math:`N_{\rm pair}(q)` against the two powers of the normalisation. It
  is removed mode by mode: exactly from the realization's own :math:`|\delta|^2` for a global
  line of sight (:func:`compute_mesh2_collapsed_disconnected`, the data-only estimator of
  arXiv:2604.18581, App. A), and with Gaussian mocks for any line of sight or geometry
  (:func:`compute_mesh2_collapsed_rdn0`, the realization-dependent term of CMB lensing);
* the :math:`BB` family of :math:`\mathrm{Cov}[B, B]` through a tie between two binned legs
  (:math:`p_i + p'_j = q`, both in binned shells), which is present only on elements sharing a
  binned shell. Its geometric factor is the pair count of the two shells at :math:`q`, the same
  object as the P x P count, and is left to the extrapolation step.

The :math:`PPP`, :math:`PT`, :math:`PB` families and every tie through a free (closure) leg are
flat, because either the tie forces equal momenta or the closure leg absorbs the mismatch.

Conventions: the fields are built on the full complex grid (one c2c transform per configuration
and multipole); ``value`` carries the :math:`(2 \ell + 1)` and tripolar coefficients that the
spectrum estimators carry, ``count`` is the unit-field product without any angular weight, and
``norm`` the estimator normalisation, so that ``value / count / norm`` at the zero mode is the raw
(shot noise not subtracted) spectrum value.

The soft modes are obtained by a separable partial Fourier sum of the real-space integrand onto
the small cube :math:`|q_i| \leq q_{\rm max}`, three contractions of one axis each, rather than by a
transform of the whole grid followed by a gather. The result does not depend on the Fourier
layout of the backend, and the contractions over sharded axes are reduced by XLA, so the module
runs with ``fft_backend='jax'`` or ``'jaxdecomp'``, on one or several devices.
"""

import operator
import functools
from functools import partial
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import jax
from jax import numpy as jnp

from .mesh import BaseMeshField, MeshAttrs, RealMeshField, ComplexMeshField, staticarray, __format_meshes
from .mesh2 import BinMesh2SpectrumPoles, _format_los as _format_los2
from .mesh3 import BinMesh3SpectrumPoles, _format_los as _format_los3, _iter_triposh
from .types import (Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles,
                    CovarianceMatrix, ObservableTree)
from .utils import get_legendre, get_Ylm, register_pytree_dataclass


prod = partial(functools.reduce, operator.mul)


def _complex_attrs(mattrs):
    """The full complex grid the collapsed fields live on."""
    return mattrs.clone(dtype=mattrs.cdtype)


# ======================================================================================
# soft modes
# ======================================================================================

@partial(register_pytree_dataclass, meta_fields=['nmax', 'cube_index', 'mirror', 'ibin', 'offsets', 'nshells'])
@dataclass(init=False, frozen=True)
class BinSoftModes(object):
    r"""
    Selection of the soft modes :math:`q` of the Fourier grid, grouped in shells of :math:`|q|`.

    Shell 0 is the single zero mode, kept on purpose: there the collapsed covariance spectrum is
    the product of the two estimators. Modes are stored sorted by shell, so that a shell is a
    contiguous slice ``offsets[i]:offsets[i + 1]`` of the transformed vectors.

    Parameters
    ----------
    mattrs : MeshAttrs or BaseMeshField
        Mesh attributes.
    edges : array-like, dict, or None, optional
        Shell edges in :math:`|q|`, above the zero mode. A dict takes 'max' (default 0.03) and
        'step' (default: the smallest fundamental frequency), and puts the shell centers at the
        multiples of 'step', i.e. edges at half-integer multiples, starting at ``step / 2``.
    """
    edges: jax.Array = None
    qavg: jax.Array = None
    nmodes: jax.Array = None
    mattrs: MeshAttrs = None
    nmax: tuple = None
    cube_index: staticarray = None
    mirror: staticarray = None
    ibin: staticarray = None
    offsets: tuple = None
    nshells: int = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges=None):
        if not isinstance(mattrs, MeshAttrs):
            mattrs = mattrs.attrs
        mattrs = _complex_attrs(mattrs)
        meshsize = np.asarray(mattrs.meshsize)
        kfun = np.asarray(mattrs.kfun)
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            step = edges.get('step', kfun.min())
            qmax = edges.get('max', 0.03)
            nq = int(np.floor(qmax / step + 0.5))
            edges = step * (np.arange(nq + 1) + 0.5)
        edges = np.asarray(edges, dtype='f8')
        if edges.ndim == 2:
            assert np.allclose(edges[1:, 0], edges[:-1, 1])
            edges = np.append(edges[:, 0], edges[-1, 1])
        assert edges.ndim == 1 and len(edges) >= 2 and edges[0] > 0., 'edges must be increasing and start above the zero mode'
        qmax = edges[-1]
        # The soft cube, in units of the fundamental frequencies; small, so numpy throughout
        nmax = np.floor(qmax / kfun).astype(int)
        assert np.all(2 * nmax < meshsize), 'the soft cube must fit in the grid'
        grids = np.meshgrid(*[np.arange(-n, n + 1) for n in nmax], indexing='ij')
        ivec = np.stack([g.ravel() for g in grids], axis=-1)  # (nmodes, ndim)
        qvec = ivec * kfun
        qnorm = np.sqrt(np.sum(qvec**2, axis=-1))
        ibin = np.digitize(qnorm, edges)  # 0 below edges[0], len(edges) above
        zero = np.all(ivec == 0, axis=-1)
        ibin = np.where(zero, 0, ibin)  # zero mode is shell 0; other modes below edges[0] are dropped
        keep = zero | ((ibin >= 1) & (ibin < len(edges)))
        ivec, qnorm, ibin = ivec[keep], qnorm[keep], ibin[keep]
        order = np.lexsort((qnorm, ibin))
        ivec, qnorm, ibin = ivec[order], qnorm[order], ibin[order]
        nshells = len(edges)
        cube_shape = tuple(2 * nmax + 1)
        cube_index = np.ravel_multi_index(tuple(ivec[:, axis] + nmax[axis] for axis in range(len(nmax))), cube_shape)
        mcube_index = np.ravel_multi_index(tuple(-ivec[:, axis] + nmax[axis] for axis in range(len(nmax))), cube_shape)
        # Position of -q in the sorted list
        lookup = {int(i): pos for pos, i in enumerate(cube_index)}
        mirror = np.array([lookup[int(i)] for i in mcube_index], dtype='i8')
        nmodes = np.bincount(ibin, minlength=nshells)
        qavg = np.bincount(ibin, weights=qnorm, minlength=nshells) / np.maximum(nmodes, 1)
        offsets = tuple(int(x) for x in np.concatenate([[0], np.cumsum(nmodes)]))
        full_edges = np.concatenate([[[0., 0.]], np.column_stack([edges[:-1], edges[1:]])], axis=0)
        self.__dict__.update(edges=jnp.asarray(full_edges), qavg=jnp.asarray(qavg), nmodes=jnp.asarray(nmodes), mattrs=mattrs,
                             nmax=tuple(int(n) for n in nmax), cube_index=staticarray(cube_index), mirror=staticarray(mirror),
                             ibin=staticarray(ibin), offsets=offsets, nshells=nshells)

    @property
    def size(self):
        """Number of soft modes, all shells included."""
        return len(self.cube_index)

    def _dft_matrix(self, axis, dtype):
        r""":math:`e^{-2 i \pi m n / N}` for the soft frequencies :math:`m` and the grid indices :math:`n` of one axis."""
        size = int(self.mattrs.meshsize[axis])
        m = np.arange(-self.nmax[axis], self.nmax[axis] + 1)
        # Integer phases, reduced modulo N before the division, so that large grids lose no precision
        phase = (m[:, None] * np.arange(size)[None, :]) % size
        return jnp.asarray(np.exp(-2j * np.pi * phase / size), dtype=dtype)

    def transform(self, value, method='auto'):
        r"""
        The soft modes :math:`\sum_x e^{-i q x} f(x)` of a real-space array (natural layout), in shell order.

        ``method='dft'`` contracts one axis at a time onto the soft cube, :math:`N^3 (2 n_{\rm max} + 1)`
        operations; ``'fft'`` transforms the whole grid with the backend's FFT (jaxdecomp when
        sharded) and extracts the cube's corner, whatever the backend's Fourier layout. ``'auto'``
        takes the partial sum while the cube is narrower than about twice the FFT's :math:`\log_2 N`,
        and the FFT beyond: at a survey's fundamental frequency, a soft cube reaching
        :math:`q = 0.03` is 55 modes wide and the FFT is ~15 times cheaper.
        """
        value = value.value if isinstance(value, BaseMeshField) else value
        if method == 'auto':
            method = 'dft' if max(2 * n + 1 for n in self.nmax) <= 2. * np.log2(max(int(m) for m in self.mattrs.meshsize)) else 'pruned'
        if method == 'fft':
            return self._fft_transform(value)
        if method == 'pruned':
            return self._pruned_transform(value)
        dtype = (1j * jnp.zeros((), dtype=value.dtype)).dtype
        out = value.astype(dtype)
        ndim = out.ndim
        # Contract the last axis first: after each step the contracted axis is replaced by a soft axis
        # appended at the end, and the axes still to contract keep their positions.
        for axis in reversed(range(ndim)):
            out = jnp.tensordot(out, self._dft_matrix(axis, dtype), axes=([axis], [1]))
        out = jnp.transpose(out, tuple(reversed(range(ndim))))
        return jnp.take(out.reshape(-1), jnp.asarray(self.cube_index), axis=0)

    def _pruned_transform(self, value):
        """1D FFT along the last (local) axis, its soft frequencies kept, then the partial sum over the two others."""
        dtype = (1j * jnp.zeros((), dtype=value.dtype)).dtype
        size, n = int(self.mattrs.meshsize[-1]), self.nmax[-1]
        out = jnp.fft.fft(value.astype(dtype), axis=-1)
        out = jnp.take(out, jnp.asarray(np.arange(-n, n + 1) % size), axis=-1)
        # Contract the remaining axes, last first, as the partial sum does; the soft axes pile up at the end
        ndim = out.ndim
        for axis in reversed(range(ndim - 1)):
            out = jnp.tensordot(out, self._dft_matrix(axis, dtype), axes=([axis], [1]))
        # Axes are now (z, y, x) soft
        out = jnp.transpose(out, tuple(reversed(range(ndim))))
        return jnp.take(out.reshape(-1), jnp.asarray(self.cube_index), axis=0)

    def _fft_transform(self, value):
        """Full backend FFT, then the soft cube's corner along each array dimension of the backend's layout."""
        cattrs = self.mattrs
        out = cattrs.r2c(value.astype(cattrs.dtype))
        # Which array dimension holds which Cartesian axis: the one along which its frequencies vary
        dims = [int(np.argmax(np.shape(k))) for k in cattrs.kcoords(kind='index', sparse=True)]
        for axis, dim in enumerate(dims):
            size, n = int(self.mattrs.meshsize[axis]), self.nmax[axis]
            out = jnp.take(out, jnp.asarray(np.arange(-n, n + 1) % size), axis=dim)
        # Back to (x, y, z) order, the cube's own
        out = jnp.transpose(out, dims)
        return jnp.take(out.reshape(-1), jnp.asarray(self.cube_index), axis=0)

    def conj_mirror(self, soft):
        """``conj(x(-q))`` for a transformed vector ``x(q)``: the transform of the conjugate field."""
        return jnp.conj(jnp.take(soft, jnp.asarray(self.mirror), axis=-1))

    def hermitian_part(self, soft, imag=False):
        """Transform of the real (or imaginary) part of the field whose transform is ``soft``."""
        mirrored = self.conj_mirror(soft)
        if imag:
            return -0.5j * (soft - mirrored)
        return 0.5 * (soft + mirrored)

    def shell_sum(self, value):
        """Sum over the modes of each shell; ``value`` has the modes on its last axis. Returns (..., nshells)."""
        return jnp.stack([value[..., lo:hi].sum(axis=-1) for lo, hi in zip(self.offsets[:-1], self.offsets[1:])], axis=-1)

    def shell_matmul(self, a, b):
        r"""Per shell, :math:`\sum_{q \in {\rm shell}} a(q) b(q)^*` for ``a`` of shape (..., na, nmodes) and ``b`` of shape (..., nb, nmodes). Returns (nshells, ..., na, nb)."""
        return jnp.stack([jnp.matmul(a[..., lo:hi], jnp.conj(jnp.swapaxes(b[..., lo:hi], -1, -2)))
                          for lo, hi in zip(self.offsets[:-1], self.offsets[1:])], axis=0)


# ======================================================================================
# collapsed fields
# ======================================================================================

@partial(register_pytree_dataclass, meta_fields=['ells', 'basis', 'kind', 'los'])
@dataclass(init=False, frozen=True)
class CollapsedField(object):
    r"""
    Soft modes of the integrand of a spectrum estimator, one vector per multipole and configuration.

    ``value`` has shape ``(nells, nbins, nmodes)`` and carries the estimator's angular
    coefficients; ``count`` has shape ``(nbins, nmodes)`` and is the unit-field product, without
    angular weight; ``norm`` is the estimator normalisation. ``value / count / norm`` at the zero
    mode is the raw spectrum, which :meth:`spectrum` returns as the matching ``lsstypes`` object.
    """
    value: jax.Array = None
    count: jax.Array = None
    norm: jax.Array = None
    edges: jax.Array = None
    xavg: jax.Array = None
    nmodes: jax.Array = None
    soft: BinSoftModes = None
    ells: tuple = None
    basis: str = None
    kind: str = None
    los: tuple = None

    def __init__(self, value, count, norm, edges, xavg, nmodes, soft, ells, basis, kind, los):
        self.__dict__.update(value=value, count=count, norm=norm, edges=edges, xavg=xavg, nmodes=nmodes, soft=soft,
                             ells=tuple(ells), basis=basis, kind=kind, los=los)

    @property
    def nbins(self):
        return self.value.shape[1]

    def clone(self, **kwargs):
        """Copy, with some attributes replaced; e.g. ``norm`` to use the survey or box normalisation of the pipeline."""
        state = {name: getattr(self, name) for name in ['value', 'count', 'norm', 'edges', 'xavg', 'nmodes', 'soft', 'ells', 'basis', 'kind', 'los']}
        state.update(kwargs)
        return self.__class__(**state)

    @property
    def size(self):
        """Number of data vector entries, multipoles times configurations."""
        return self.value.shape[0] * self.value.shape[1]

    def flat(self):
        """``value`` as a (size, nmodes) array, multipole-major then configuration, the data vector order."""
        return self.value.reshape(self.size, -1)

    def raw(self):
        """The estimator-normalised soft modes, ``value / count / norm``, of shape (nells, nbins, nmodes)."""
        return self.value / self.count[None, ...] / self.norm

    def spectrum(self):
        """The raw spectrum, read off the zero mode."""
        raw = np.asarray(self.raw()[..., 0].real)
        attrs = dict(los=self.los if self.los == 'local' else np.asarray(self.los))
        poles = []
        for ill, ell in enumerate(self.ells):
            num_raw = raw[ill] * np.asarray(self.norm)
            if self.kind == 'mesh2':
                poles.append(Mesh2SpectrumPole(k=np.asarray(self.xavg), k_edges=np.asarray(self.edges), nmodes=np.asarray(self.nmodes),
                                               num_raw=num_raw, norm=np.asarray(self.norm) * np.ones_like(num_raw), ell=ell, attrs=attrs))
            else:
                poles.append(Mesh3SpectrumPole(k=np.asarray(self.xavg), k_edges=np.asarray(self.edges), nmodes=np.asarray(self.nmodes),
                                               num_raw=num_raw, norm=np.asarray(self.norm), ell=ell, basis=self.basis, attrs=attrs))
        if self.kind == 'mesh2':
            return Mesh2SpectrumPoles(poles)
        return Mesh3SpectrumPoles(poles)


def _to_full_fourier(mesh, cattrs):
    """Complex Fourier transform of a mesh on the full grid, as a plain array."""
    if isinstance(mesh, ComplexMeshField):
        mesh = mesh.c2r()
    return cattrs.r2c(mesh.value.astype(cattrs.dtype))


def _to_real(mesh, cattrs):
    """Real-space values of a mesh, cast to the complex grid dtype."""
    if isinstance(mesh, ComplexMeshField):
        mesh = mesh.c2r()
    return mesh.value.astype(cattrs.dtype)


def _shard(value, cattrs):
    """Pin a full-grid array to the mesh's sharding: a broadcast of the sparse wavenumbers is otherwise free to come out replicated."""
    sharding_mesh = cattrs.sharding_mesh
    if not sharding_mesh.axis_names:
        return value
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.lax.with_sharding_constraint(value, NamedSharding(sharding_mesh, P(*sharding_mesh.axis_names)))


def _shell_mask(knorm, edge):
    return ((knorm >= edge[0]) & (knorm < edge[1])).astype(knorm.dtype)


def _mesh2_legs(meshes, fields, ells, los, cattrs):
    r"""
    The two Fourier-space legs of the power spectrum integrand, per multipole.

    The integrand of shell :math:`k` and multipole :math:`\ell` is
    ``factor * conj(c2r[U mask]) * c2r[V mask]``, with ``U`` built from the first mesh and ``V``
    from the second: for a global line of sight ``U = delta_0``, ``V = L_ell(mu) delta_1``,
    ``factor = 2 ell + 1``; for a local one ``U`` is the Yamamoto sum
    :math:`\sum_m y_{\ell m}(\hat k) \mathrm{FFT}[\delta_0 y_{\ell m}(\hat x)]` and
    ``V = delta_1``, ``factor = 4 pi`` (1 for the monopole), as in :func:`compute_mesh2_spectrum`.

    Returns the list of ``(U, V, factor, terms)``, whether the two meshes were swapped
    ('endpoint'), and the line-of-sight vector. ``terms`` lists ``(xfactor, kU, kV)`` with
    ``U = sum kU(k) FFT[delta_0 xfactor]`` and ``V = kV(k) delta_1``, which is what the self-pair
    line needs (``xfactor`` ``None`` for 1).
    """
    los, vlos, swap = _format_los2(los, ndim=cattrs.ndim)
    kvec = cattrs.kcoords(sparse=True)
    knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
    if vlos is None:
        if swap: meshes = meshes[::-1]
        rmesh0 = _to_real(meshes[0], cattrs)
        V = _to_full_fourier(meshes[1], cattrs)
        xvec = cattrs.rcoords(sparse=True)
        legs = []
        for ell in ells:
            if ell == 0:
                legs.append((cattrs.r2c(rmesh0), V, 1., [(None, 1., 1.)]))
                continue
            Ylms = [get_Ylm(ell, m, reduced=False, real=True) for m in range(-ell, ell + 1)]

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += cattrs.r2c(rmesh0 * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            U = jax.lax.scan(partial(f, Ylms), init=jnp.zeros_like(V), xs=np.arange(len(Ylms)))[0]
            legs.append((U, V, 4. * np.pi, [(Ylm(*xvec), Ylm(*kvec), 1.) for Ylm in Ylms]))
        return legs, swap, vlos
    U = _to_full_fourier(meshes[0], cattrs)
    V = U if fields[1] == fields[0] else _to_full_fourier(meshes[1], cattrs)
    mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / knorm.at[(0,) * cattrs.ndim].set(1.)
    legs = []
    for ell in ells:
        legendre = get_legendre(ell)(mu)
        legs.append((U, V * legendre, 2. * ell + 1., [(None, 1., legendre)]))
    return legs, swap, vlos


def paint_contacts(particles, orders=(2, 3), **kwargs):
    r"""
    The fields of the self-pair (and self-triplet) lines: the particles painted with their weights to the power ``order``.

    For an :class:`FKPField` the combined particles ``data - alpha randoms`` are used, so that the
    randoms enter with :math:`(-\alpha)^n`. Paint with the same ``resampler``, ``interlacing`` and
    ``compensate`` as the data meshes. The self lines are exact for an interlaced, compensated
    painting, the convention of :func:`compute_fkp3_shotnoise`.

    Returns
    -------
    contacts : dict
        ``{order: RealMeshField}``.
    """
    from .mesh import FKPField
    if isinstance(particles, FKPField):
        particles = particles.particles
    return {order: particles.clone(weights=particles.weights**order).paint(**kwargs, out='real') for order in orders}


def compute_mesh2_collapsed(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh2SpectrumPoles=None, soft: BinSoftModes=None,
                            los: str | np.ndarray='z', contacts: dict=None, count: jax.Array=None) -> CollapsedField:
    r"""
    Soft modes of the power spectrum estimator's integrand, the local :math:`P_\ell(k)` field.

    For a global line of sight the integrand of shell :math:`k` is
    :math:`\overline{d_k(x)} \, d_{k, \ell}(x)` with :math:`d_k = \mathrm{c2r}[\Theta_k \delta]`
    and :math:`d_{k, \ell} = \mathrm{c2r}[\Theta_k \mathcal{L}_\ell(\mu) \delta]`; for a local
    line of sight the Legendre factor is the Yamamoto sum of :func:`compute_mesh2_spectrum`. Same
    conventions as that function, so that the zero mode reproduces it exactly. The integrand is
    bilinear in the two meshes, which :func:`compute_mesh2_collapsed_rdn0` relies on.

    Parameters
    ----------
    meshes : RealMeshField or ComplexMeshField
        One mesh, or two for a cross spectrum.
    bin : BinMesh2SpectrumPoles
        Binning operator; only its edges and multipoles are used.
    soft : BinSoftModes
        Soft modes to compute.
    los : str or array-like, optional
        Line of sight, as in :func:`compute_mesh2_spectrum`.
    contacts : dict, optional
        The fields of :func:`paint_contacts` (order 2 is used), to remove the self pairs: at
        :math:`q \neq 0` their line is not a constant but the soft modes of the painted
        :math:`w^2` field times a geometric factor, and removing it makes the zero mode the
        shot-noise subtracted spectrum. Only for an auto spectrum (same field on both legs).
    count : array, optional
        The unit-field counts of a previous call on the same binning and soft modes: they do not depend on the
        meshes, and :func:`compute_mesh2_collapsed_rdn0` computes them once for all its mocks.

    Returns
    -------
    CollapsedField
    """
    meshes, fields = __format_meshes(*meshes, nmeshes=2)
    mattrs = meshes[0].attrs
    cattrs = _complex_attrs(mattrs)
    rdtype = mattrs.rdtype
    norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod()
    ells = tuple(bin.ells)
    knorm = _shard(jnp.sqrt(sum(kk**2 for kk in cattrs.kcoords(sparse=True))), cattrs)
    ones = _shard(jnp.ones_like(knorm, dtype=cattrs.dtype), cattrs)
    edges = jnp.asarray(bin.edges)
    legs, swap, vlos = _mesh2_legs(meshes, fields, ells, los, cattrs)
    selfs = None
    if contacts is not None and fields[0] == fields[1]:
        # Soft modes of the painted w^2 field, with each term's position factor; independent of the bin
        rho2 = _to_real(contacts[2], cattrs)
        selfs = [[soft.transform(rho2 if xfactor is None else rho2 * xfactor) for xfactor, kU, kV in terms] for U, V, factor, terms in legs]

    def per_bin(edge):
        mask = _shell_mask(knorm, edge)
        out = []
        # A leg shared by the multipoles (V for a local line of sight, U for a global one) is transformed once
        filtered = {}

        def c2r_masked(X):
            if id(X) not in filtered:
                filtered[id(X)] = cattrs.c2r(X * mask)
            return filtered[id(X)]

        for ill, (U, V, factor, terms) in enumerate(legs):
            value = soft.transform(jnp.conj(c2r_masked(U)) * c2r_masked(V))
            if selfs is not None:
                # Self pairs: sum_j w_j^2 x(x_j) e^{-i q x_j} times (1/N) sum_p kU^*(p) mask(p) kV(p + q) mask(p + q)
                for (xfactor, kU, kV), S in zip(terms, selfs[ill]):
                    value = value - S * soft.transform(jnp.conj(cattrs.c2r(kU * ones * mask)) * cattrs.c2r(kV * ones * mask))
            out.append(factor * value)
        if count is not None:
            return jnp.stack(out), jnp.zeros(soft.size)
        n = cattrs.c2r(ones * mask)
        return jnp.stack(out), soft.transform(jnp.conj(n) * n).real

    value, count_ = jax.lax.map(per_bin, edges)
    if count is None: count = count_
    value = jnp.moveaxis(value, 0, 1)  # (nells, nbins, nmodes)
    if swap: value = soft.conj_mirror(value)
    # The integrand of an even multipole is a real field, of an odd one an imaginary field:
    # keep the transform of that part, as the spectrum keeps the real or imaginary sum.
    value = jnp.stack([soft.hermitian_part(value[ill], imag=(ell % 2 == 1)) for ill, ell in enumerate(ells)])
    return CollapsedField(value=value, count=count, norm=norm, edges=edges, xavg=bin.xavg, nmodes=bin.nmodes, soft=soft,
                          ells=ells, basis=None, kind='mesh2', los='local' if vlos is None else tuple(float(v) for v in vlos))


def compute_mesh2_collapsed_disconnected(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh2SpectrumPoles=None, soft: BinSoftModes=None,
                                         los: str | np.ndarray='z'):
    r"""
    Data-only estimate of the Gaussian part of :math:`\sum_q F_a(q) F_b(q)^*` for a global line of sight.

    With the integrand written as ``factor * conj(c2r[U]) * c2r[V]`` (see :func:`_mesh2_legs`), the
    two Wick pairings of the Gaussian term are, per soft mode,

    .. math::
        \frac{f_a f_b}{N^2} \sum_p \Theta(p) \Theta(p + q) \left\{ [U_a^* U_b](p) [V_a V_b^*](p + q) + [U_a^* V_b](p) [V_a U_b^*](p + q) \right\},

    where each pair product stands for its expectation. Replacing it by the realization's own
    product at the same mode (arXiv:2604.18581, Eq. 41) gives an estimate that carries the exact
    mode geometry of the shell at every :math:`q`, and each pairing is the soft transform of a
    product of two :math:`\mathrm{c2r}` fields. The replacement is exact when the pair covariance
    is diagonal in the modes, which holds for a periodic box with a global line of sight; for a
    local line of sight or a survey window use :func:`compute_mesh2_collapsed_rdn0`.

    Returns
    -------
    disconnected : jax.Array
        Shape ``(nshells, n, n)``, with ``n = nells * nbins`` in the data vector order: the shell
        sums of the Gaussian part of ``value_a value_b^*``, in the units of
        :attr:`CollapsedField.value`. Only same-shell entries are nonzero, as for the Gaussian term.
    """
    meshes, fields = __format_meshes(*meshes, nmeshes=2)
    mattrs = meshes[0].attrs
    cattrs = _complex_attrs(mattrs)
    if _format_los2(los, ndim=mattrs.ndim)[1] is None:
        raise NotImplementedError('the data-only estimate is exact for a global line of sight only; use compute_mesh2_collapsed_rdn0')
    ells = tuple(bin.ells)
    if any(ell % 2 for ell in ells):
        raise NotImplementedError('odd multipoles have no Gaussian term on a shared shell to remove here')
    knorm = jnp.sqrt(sum(kk**2 for kk in cattrs.kcoords(sparse=True)))
    legs, swap, vlos = _mesh2_legs(meshes, fields, ells, los, cattrs)
    nmesh = mattrs.meshsize.prod(dtype=mattrs.rdtype)
    nells = len(ells)

    def per_bin(edge):
        mask = _shell_mask(knorm, edge)
        rows = []
        for Ua, Va, fa, _ in legs:
            row = []
            for Ub, Vb, fb, _ in legs:
                # (1/N) sum_p A^*(p) B(p + q) is the soft transform of conj(c2r[A]) c2r[B]
                first = soft.transform(jnp.conj(cattrs.c2r(Ua * jnp.conj(Ub) * mask)) * cattrs.c2r(Va * jnp.conj(Vb) * mask))
                second = soft.transform(jnp.conj(cattrs.c2r(Ua * jnp.conj(Vb) * mask)) * cattrs.c2r(Va * jnp.conj(Ub) * mask))
                row.append(soft.shell_sum(fa * fb * (first + second) / nmesh))
            rows.append(jnp.stack(row))
        return jnp.stack(rows)  # (nells, nells, nshells)

    disc = jax.lax.map(per_bin, jnp.asarray(bin.edges))  # (nbins, nells, nells, nshells)
    nbins = disc.shape[0]
    out = jnp.zeros((soft.nshells, nells * nbins, nells * nbins), dtype=disc.dtype)
    ibins = np.arange(nbins)
    for ia in range(nells):
        for ib in range(nells):
            out = out.at[:, ia * nbins + ibins, ib * nbins + ibins].set(jnp.moveaxis(disc[:, ia, ib], -1, 0))
    return out


def compute_mesh2_collapsed_rdn0(mesh: RealMeshField | ComplexMeshField, gaussian: Callable, nmocks: int, bin: BinMesh2SpectrumPoles=None,
                                 soft: BinSoftModes=None, los: str | np.ndarray='z', mean_field: bool=True, mocks: dict=None, return_parts: bool=False):
    r"""
    Realization-dependent estimate of the disconnected part of :math:`\sum_q F_a(q) F_b(q)^*`, for any line of sight or geometry.

    The integrand is a bilinear form :math:`F(x, y)` of the two meshes, so the three Gaussian
    pairings of :math:`F(\delta, \delta) F(\delta, \delta)^*` can be estimated with Gaussian mocks
    :math:`g` of a fiducial power spectrum, drawn with the same geometry (as the realization-
    dependent :math:`N^0` of CMB lensing):

    .. math::
        \hat D = \langle F(\delta, g) F(\delta, g)^* + F(g, \delta) F(g, \delta)^* + F(\delta, g) F(g, \delta)^* + F(g, \delta) F(\delta, g)^* \rangle_g
               - \langle F(g, g') F(g, g')^* + F(g, g') F(g', g)^* \rangle_{g \neq g'}
               + F(\delta, \delta) \bar F^* + \bar F F(\delta, \delta)^* - \bar F_1 \bar F_2^*,

    with :math:`\bar F = \langle F(g, g) \rangle_g` the mean field and :math:`\bar F_{1, 2}` its
    estimates on the two halves of the mocks, so that their product carries no mock noise bias.
    Its expectation is the Gaussian part of the data's :math:`F F^*`, exactly, and its error is
    second order in the mismatch between the fiducial and the true power spectrum. The mean field
    vanishes at :math:`q \neq 0` in a periodic box with a global line of sight, where
    ``mean_field=False`` saves the noise it adds.

    Parameters
    ----------
    mesh : RealMeshField or ComplexMeshField
        The data.
    gaussian : callable
        ``gaussian(i)`` returns Gaussian mock ``i``, with the geometry of the data; seed it apart
        from the data.
    nmocks : int
        Number of mocks, at least 2 (even, for the mean-field halves).
    bin, soft, los
        As in :func:`compute_mesh2_collapsed`.
    mean_field : bool, optional
        Whether to include the mean-field pairing.
    mocks : dict, optional
        The data-independent part, from a previous call's ``mocks`` output, to be reused across
        data realizations with the same mocks.
    return_parts : bool, optional
        Also return ``{'gaussian': ..., 'mean_field': ...}``, the two pieces of the estimate: the
        Gaussian pairings alone are the Gaussian covariance at the zero mode, where the mean field
        is :math:`\langle \hat P_a \rangle \langle \hat P_b \rangle`.

    Returns
    -------
    disconnected : jax.Array
        Shape ``(nshells, n, n)``, in the units and order of :func:`compute_mesh2_collapsed_disconnected`.
    mocks : dict
        The data-independent part: the mock-mock pairings and the mean-field halves.
    """
    assert nmocks >= 2, 'the realization-dependent estimate needs at least 2 mocks'
    # The counts depend on the binning only: computed once, with the first data field
    count = compute_mesh2_collapsed(mesh, bin=bin, soft=soft, los=los).count
    compute = partial(compute_mesh2_collapsed, bin=bin, soft=soft, los=los, count=count)
    if mocks is None:
        mock_mock, mfs = 0., [0., 0.]
        half = nmocks // 2
        for i in range(nmocks):
            gi, gj = gaussian(i), gaussian((i + 1) % nmocks)
            fij, fji = compute(gi, gj).flat(), compute(gj, gi).flat()
            mock_mock = mock_mock + soft.shell_matmul(fij, fij) + soft.shell_matmul(fij, fji)
            if mean_field:
                mfs[i >= half] = mfs[i >= half] + compute(gi).flat() / (half if i < half else nmocks - half)
        mocks = dict(mock_mock=mock_mock / nmocks, mean_field=mfs if mean_field else None)
    disc = 0.
    for i in range(nmocks):
        gi = gaussian(i)
        fdg, fgd = compute(mesh, gi).flat(), compute(gi, mesh).flat()
        disc = disc + soft.shell_matmul(fdg, fdg) + soft.shell_matmul(fgd, fgd) + soft.shell_matmul(fdg, fgd) + soft.shell_matmul(fgd, fdg)
    gaussian_part = disc / nmocks - mocks['mock_mock']
    mean_field_part = 0.
    if mean_field and mocks['mean_field'] is not None:
        mf1, mf2 = mocks['mean_field']
        mf = (mf1 + mf2) / 2.
        fdd = compute(mesh).flat()
        mean_field_part = soft.shell_matmul(fdd, mf) + soft.shell_matmul(mf, fdd) - (soft.shell_matmul(mf1, mf2) + soft.shell_matmul(mf2, mf1)) / 2.
    disc = gaussian_part + mean_field_part
    if return_parts:
        return disc, mocks, {'gaussian': gaussian_part, 'mean_field': mean_field_part}
    return disc, mocks


def _distinct_correction(legs, kernels, xfactors, contacts, fields, cattrs):
    r"""
    Coincident-particle part of the product of three legs, to subtract for distinct triplets.

    Leg :math:`i` is :math:`x_i(x) \, d_i(x)` with :math:`d_i(x) = \sum_j w_j K_i(x - x_j)` and a
    position factor :math:`x_i`; ``legs`` holds the full legs, ``kernels`` the :math:`K_i` on the
    grid (c2r of the leg's Fourier filter applied to the unit field), ``contacts`` the Fourier
    transforms of the painted :math:`w^2` and :math:`w^3` fields. With inclusion-exclusion, the
    distinct part is the product minus the three pair lines plus twice the triple line.
    """
    def pair(i, j):
        return cattrs.c2r(contacts[2] * cattrs.r2c(kernels[i] * kernels[j])) * xfactors[i] * xfactors[j]

    correction = 0.
    for (i, j), k in [((0, 1), 2), ((0, 2), 1), ((1, 2), 0)]:
        if fields[i] == fields[j]:
            correction = correction + pair(i, j) * legs[k]
    if fields[0] == fields[1] == fields[2]:
        triple = cattrs.c2r(contacts[3] * cattrs.r2c(kernels[0] * kernels[1] * kernels[2])) * xfactors[0] * xfactors[1] * xfactors[2]
        correction = correction - 2. * triple
    return correction


def compute_mesh3_collapsed(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh3SpectrumPoles=None, soft: BinSoftModes=None,
                            los: str | np.ndarray='z', contacts: dict=None) -> CollapsedField:
    r"""
    Soft modes of the bispectrum estimator's integrand, the local :math:`B(k_1, k_2)` field.

    Same construction and angular coefficients as :func:`compute_mesh3_spectrum`: in the Sugiyama
    basis the integrand is the tripolar sum over :math:`m` of
    :math:`\mathrm{c2r}[\Theta_{k_1} y_{\ell_1 m_1} \delta] \, \mathrm{c2r}[\Theta_{k_2} y_{\ell_2 m_2} \delta] \, y_{L M}(\hat n) \delta(x)`
    with the third leg free; in the Scoccimarro basis, the product of the three shell-filtered
    fields with a Legendre weight on the third, global line of sight only. The zero mode
    reproduces :func:`compute_mesh3_spectrum` exactly.

    Parameters
    ----------
    meshes : RealMeshField or ComplexMeshField
        One mesh, or up to three.
    bin : BinMesh3SpectrumPoles
        Binning operator; its edges, multipoles and basis are used.
    soft : BinSoftModes
        Soft modes to compute.
    los : str or array-like, optional
        Line of sight, as in :func:`compute_mesh3_spectrum`.
    contacts : dict, optional
        The fields of :func:`paint_contacts` (orders 2 and 3), to keep only distinct particle
        triplets. With the legs written as :math:`d_i(x) = \sum_j w_j K_i(x - x_j)`, the
        coincidence of legs :math:`i, j` is :math:`\mathrm{c2r}[\mathrm{FFT}(\rho_2) \, \mathrm{FFT}(K_i K_j)]`
        times the third leg, and of all three :math:`\mathrm{c2r}[\mathrm{FFT}(\rho_3) \, \mathrm{FFT}(K_1 K_2 K_3)]`;
        inclusion-exclusion gives the distinct part. A pair is removed only between legs carrying
        the same field. The zero mode is then the shot-noise subtracted spectrum of
        :func:`compute_fkp3_shotnoise`.

    Returns
    -------
    CollapsedField
    """
    meshes, fields = __format_meshes(*meshes, nmeshes=3)
    mattrs = meshes[0].attrs
    cattrs = _complex_attrs(mattrs)
    rdtype = mattrs.rdtype
    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)**2
    los, vlos = _format_los3(los, ndim=mattrs.ndim)
    ells = [tuple(ell) if np.ndim(ell) else int(ell) for ell in bin.ells]
    kvec = cattrs.kcoords(sparse=True)
    knorm = _shard(jnp.sqrt(sum(kk**2 for kk in kvec)), cattrs)
    ones = _shard(jnp.ones_like(knorm, dtype=cattrs.dtype), cattrs)
    # c2r of the unit field: the count's free third leg
    delta0 = cattrs.c2r(ones)
    edges = jnp.asarray(bin.edges)
    cc = None
    if contacts is not None:
        cc = {order: cattrs.r2c(_to_real(contacts[order], cattrs)) for order in (2, 3)}

    if 'scoccimarro' in bin.basis:
        if vlos is None:
            raise NotImplementedError('the collapsed Scoccimarro estimator is written for a global line of sight')
        cmeshes = [_to_full_fourier(mesh, cattrs) for mesh in meshes]
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / knorm.at[(0,) * mattrs.ndim].set(1.)
        legendres = [get_legendre(ell)(mu) for ell in ells]

        def per_conf(edge):
            masks = [_shell_mask(knorm, edge[i]) for i in range(3)]
            d1 = cattrs.c2r(cmeshes[0] * masks[0])
            d2 = cattrs.c2r(cmeshes[1] * masks[1])
            out = []
            for ell, legendre in zip(ells, legendres):
                d3 = cattrs.c2r(cmeshes[2] * legendre * masks[2])
                integrand = d1 * d2 * d3
                if cc is not None:
                    kernels = [cattrs.c2r(ones * masks[0]), cattrs.c2r(ones * masks[1]), cattrs.c2r(legendre * masks[2])]
                    integrand = integrand - _distinct_correction([d1, d2, d3], kernels, [1., 1., 1.], cc, fields, cattrs)
                out.append((2 * ell + 1) * soft.transform(integrand))
            n = prod(cattrs.c2r(ones * _shell_mask(knorm, edge[i])) for i in range(3))
            return jnp.stack(out), soft.transform(n).real

        value, count = jax.lax.map(per_conf, edges)
        value = jnp.moveaxis(value, 0, 1)
        value = jnp.stack([soft.hermitian_part(value[ill], imag=(ell % 2 == 1)) for ill, ell in enumerate(ells)])
    else:
        cmeshes = [_to_full_fourier(mesh, cattrs) for mesh in meshes[:2]]
        rmesh3 = _to_real(meshes[2], cattrs)
        xvec = cattrs.rcoords(sparse=True)
        losvec = xvec if vlos is None else vlos
        values = []
        for ell1, ell2, ell3 in ells:
            Ylms = [[get_Ylm(ell, m, reduced=True, real=False, conj=True) for m in range(-ell, ell + 1)] for ell in (ell1, ell2, ell3)]
            xs = _iter_triposh(ell1, ell2, ell3, los=los)
            rows = [tuple(np.asarray(x)[i] for x in xs) for i in range(len(xs[0]))]

            # One branch per tripolar row, scanned with lax.switch as compute_mesh3_spectrum does: only one row's
            # filtered fields and self-pair kernels are alive at a time, where an unrolled loop keeps them all
            def make_branch(im1, im2, im3, Ylms=Ylms):
                def branch(masks):
                    mask1, mask2 = masks
                    Y1, Y2 = Ylms[0][im1](*kvec), Ylms[1][im2](*kvec)
                    # The third leg's line-of-sight harmonic, a position factor, built here rather than kept for every row
                    losfactor = Ylms[2][im3](*losvec)
                    third = losfactor * rmesh3
                    d1 = cattrs.c2r(cmeshes[0] * Y1 * mask1)
                    d2 = cattrs.c2r(cmeshes[1] * Y2 * mask2)
                    integrand = d1 * d2 * third
                    if cc is not None:
                        # The third leg is the raw field: its kernel is the unit impulse
                        kernels = [cattrs.c2r(Y1 * ones * mask1), cattrs.c2r(Y2 * ones * mask2), delta0]
                        integrand = integrand - _distinct_correction([d1, d2, third], kernels, [1., 1., losfactor], cc, fields, cattrs)
                    return soft.transform(integrand)
                return branch

            branches = [make_branch(int(row[0]), int(row[1]), int(row[2])) for row in rows]
            coeffs = jnp.asarray([row[3] for row in rows], dtype=cattrs.dtype)
            syms = jnp.asarray([row[4] for row in rows], dtype=cattrs.dtype)

            def per_conf(edge, branches=branches, coeffs=coeffs, syms=syms):
                masks = (_shell_mask(knorm, edge[0]), _shell_mask(knorm, edge[1]))

                def step(acc, x):
                    irow, coeff, sym = x
                    Fq = jax.lax.switch(irow, branches, masks)
                    return acc + coeff * (Fq + sym * soft.conj_mirror(Fq)), None

                return jax.lax.scan(step, jnp.zeros(soft.size, dtype=cattrs.dtype), (jnp.arange(len(branches)), coeffs, syms))[0]

            if rows:
                value = jax.lax.map(per_conf, edges)
            else:
                value = jnp.zeros((len(edges), soft.size), dtype=cattrs.dtype)
            values.append(soft.hermitian_part(value, imag=((ell1 + ell2) % 2 == 1)))
        value = jnp.stack(values)

        def per_count(edge):
            n = cattrs.c2r(ones * _shell_mask(knorm, edge[0])) * cattrs.c2r(ones * _shell_mask(knorm, edge[1])) * delta0
            return soft.transform(n).real

        count = jax.lax.map(per_count, edges)
    return CollapsedField(value=value, count=count, norm=norm, edges=edges, xavg=bin.xavg, nmodes=bin.nmodes[0], soft=soft,
                          ells=ells, basis=bin.basis, kind='mesh3', los='local' if vlos is None else tuple(float(v) for v in vlos))


# ======================================================================================
# collapsed covariance
# ======================================================================================

class CollapsedCovariance(object):
    r"""
    The collapsed covariance spectrum :math:`C_{ab}(q)` of a set of observables, one matrix per soft shell.

    ``value`` has shape ``(nshells, n, n)`` in the flattened order of ``observable`` (an
    :class:`ObservableTree` of the raw spectra); shell 0 is the outer product of the
    estimators. ``disconnected`` is the estimate of the disconnected part on the same shells,
    where one was supplied (zero elsewhere), so that ``value - disconnected`` is the connected
    spectrum.
    """
    _name = 'collapsed_covariance'

    def __init__(self, value, observable, q, q_edges, nmodes, disconnected=None, attrs=None):
        self._value = np.asarray(value)
        self._observable = observable
        self._q, self._q_edges, self._nmodes = np.asarray(q), np.asarray(q_edges), np.asarray(nmodes)
        self._disconnected = np.zeros_like(self._value) if disconnected is None else np.asarray(disconnected)
        self._attrs = dict(attrs or {})

    @property
    def attrs(self):
        return self._attrs

    @property
    def observable(self):
        return self._observable

    @property
    def q(self):
        return self._q

    @property
    def q_edges(self):
        return self._q_edges

    @property
    def nmodes(self):
        return self._nmodes

    @property
    def nshells(self):
        return len(self._q)

    @property
    def shape(self):
        return self._value.shape

    def value(self, connected=False):
        """The spectra, of shape (nshells, n, n); with ``connected``, the disconnected part is subtracted."""
        if connected:
            return self._value - self._disconnected
        return self._value

    def disconnected(self):
        return self._disconnected

    def get(self, ishell, connected=False):
        """The matrix of one shell as a :class:`CovarianceMatrix`."""
        attrs = dict(self._attrs, q=float(self._q[ishell]), q_edges=tuple(float(x) for x in self._q_edges[ishell]), nmodes=int(self._nmodes[ishell]))
        return CovarianceMatrix(value=self.value(connected=connected)[ishell], observable=self._observable, attrs=attrs)

    def clone(self, **kwargs):
        state = dict(value=self._value, observable=self._observable, q=self._q, q_edges=self._q_edges, nmodes=self._nmodes,
                     disconnected=self._disconnected, attrs=self._attrs)
        state.update(kwargs)
        return self.__class__(**state)

    def __getstate__(self, to_file=False):
        state = {'name': self._name, 'value': self._value, 'disconnected': self._disconnected, 'q': self._q, 'q_edges': self._q_edges,
                 'nmodes': self._nmodes, 'attrs': dict(self._attrs), 'observable': self._observable.__getstate__(to_file=to_file)}
        return state

    def __setstate__(self, state):
        from lsstypes.base import from_state
        self._value = np.asarray(state['value'])
        self._disconnected = np.asarray(state['disconnected'])
        self._q, self._q_edges, self._nmodes = np.asarray(state['q']), np.asarray(state['q_edges']), np.asarray(state['nmodes'])
        # The HDF5 writer drops an empty dict, as for lsstypes' CovarianceMatrix
        self._attrs = dict(state.get('attrs', {}))
        self._observable = from_state(state['observable'])

    def write(self, filename, **kwargs):
        """Write to disk; the observable is written with its own lsstypes state."""
        from lsstypes.base import _write
        _write(filename, self.__getstate__(to_file=True), **kwargs)

    @classmethod
    def read(cls, filename):
        from lsstypes.base import _read
        state = _read(filename)
        assert str(state.pop('name')) == cls._name, 'not a collapsed covariance file'
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new


def compute_collapsed_covariance(fields: list, disconnected: dict=None, observable_fields: list=None) -> CollapsedCovariance:
    r"""
    Assemble the collapsed covariance spectrum of a list of collapsed fields.

    Parameters
    ----------
    fields : list of CollapsedField
        The collapsed fields, in the order the observable is stacked; all on the same
        :class:`BinSoftModes`.
    disconnected : dict, optional
        ``{(ifield, jfield): array}``, the shell sums of the disconnected part of
        ``value_i value_j^*``, of shape ``(nshells, n_i, n_j)``, as returned by
        :func:`compute_mesh2_collapsed_disconnected` or :func:`compute_mesh2_collapsed_rdn0`.
    observable_fields : list, optional
        Field labels for the observable tree, one tuple per collapsed field (e.g. ``(0, 0)`` for
        a power spectrum); default ``(0,) * n``.

    Returns
    -------
    CollapsedCovariance
    """
    soft = fields[0].soft
    disconnected = dict(disconnected or {})
    sizes = [field.size for field in fields]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    ntot = int(offsets[-1])
    nshells = soft.nshells
    value = np.zeros((nshells, ntot, ntot), dtype='f8')
    disc = np.zeros_like(value)
    for ia, fa in enumerate(fields):
        for ib, fb in enumerate(fields):
            if ib < ia:
                continue
            # Ratio of shell sums: sum_q F_a F_b^* / sum_q n_a n_b^*, per pair of entries
            num = soft.shell_matmul(fa.flat(), fb.flat())  # (nshells, na, nb)
            den = soft.shell_matmul(fa.count.astype(num.dtype), fb.count.astype(num.dtype)).real  # (nshells, nbins_a, nbins_b)
            # Entries are ordered multipole-major, then configuration, so the count tiles along both
            den = jnp.tile(den, (1, len(fa.ells), len(fb.ells)))
            scale = float(fa.norm * fb.norm)
            sl_a, sl_b = slice(offsets[ia], offsets[ia + 1]), slice(offsets[ib], offsets[ib + 1])
            block = np.asarray((num / den).real) / scale
            value[:, sl_a, sl_b] = block
            value[:, sl_b, sl_a] = np.swapaxes(block, 1, 2)
            for key, transpose in [((ia, ib), False), ((ib, ia), True)]:
                if key in disconnected and (not transpose or ia != ib):
                    d = jnp.asarray(disconnected[key])
                    if transpose: d = jnp.conj(jnp.swapaxes(d, 1, 2))
                    dblock = np.asarray((d / den).real) / scale
                    disc[:, sl_a, sl_b] = dblock
                    disc[:, sl_b, sl_a] = np.swapaxes(dblock, 1, 2)
                    break
    if observable_fields is None:
        observable_fields = [(0,) * (2 if field.kind == 'mesh2' else 3) for field in fields]
    observable = ObservableTree([field.spectrum() for field in fields], fields=list(observable_fields))
    return CollapsedCovariance(value=value, observable=observable, q=soft.qavg, q_edges=soft.edges, nmodes=soft.nmodes, disconnected=disc)


# ======================================================================================
# per-mode storage
# ======================================================================================

class CollapsedModes(object):
    r"""
    The soft modes of every entry's estimator integrand and of the density field, for one realization.

    Storing the modes rather than their shell sums keeps everything the extrapolation to
    :math:`q \to 0` needs, at the size of the data vector times the number of soft modes:

    * the collapsed covariance spectrum :math:`C_{ab}(q)` (:meth:`covariance`);
    * the squeezed cross spectrum :math:`X_a(q) = \sum_q F_a(q) \delta^*(q) / \sum_q n_a(q)`
      (:meth:`squeezed`), the analog of the squeezed bispectrum of arXiv:2604.18581, whose
      :math:`q`-dependence is the response of entry :math:`a` to a long mode;
    * the power spectrum :math:`P(q)` of the soft modes themselves (:meth:`power`);
    * per-mode versions of the three (:meth:`mode_quantities`), whose scatter over the modes of a
      shell estimates the covariance of the shell averages within a single realization.

    With the normalisations used, a response :math:`F_a \propto r_a n_a \delta` gives
    :math:`X_a = r_a P` and :math:`C_{ab} \supset r_a r_b P = X_a X_b / P`, whatever the
    normalisation of :math:`F_a`, which is what makes the joint model of the extrapolation consistent.

    Parameters
    ----------
    value : array
        ``(n, nmodes)`` complex: the soft modes of each entry's integrand, data vector order.
    count : array
        ``(n, nmodes)``: the unit-field counts of each entry.
    norm : array
        ``(n,)``: the estimator normalisation of each entry.
    delta : array
        ``(nmodes,)`` complex: the soft modes of the (mean-subtracted, painted) density mesh.
    delta_norm : float
        Normalisation of the power spectrum, :math:`\hat P = |\delta|^2 / \mathrm{delta\_norm}` per mode.
    delta_shotnoise : float
        Shot noise of that power spectrum, in the same units, subtracted by :meth:`power`.
    ibin, mirror : array
        Shell index of each mode, and position of :math:`-q`.
    q, q_edges, nmodes : array
        Per shell: mean :math:`|q|`, edges, number of modes. Shell 0 is the zero mode.
    observable : ObservableTree
        The measured spectra (the zero mode), labelling the data vector.
    disconnected : array, optional
        ``(nshells, n, n)``: shell sums of the disconnected part of ``value_a value_b^*``.
    gaussian_zero : array, optional
        ``(n, n)``: the Gaussian part alone at the zero mode, in the units of ``disconnected``, where the latter
        also holds the mean field :math:`\\langle \\hat P_a \\rangle \\langle \\hat P_b \\rangle` (survey geometry).
        An array, not an attribute: HDF5 attributes are limited to 64 kB.
    """
    _name = 'collapsed_modes'
    _array_names = ['value', 'count', 'norm', 'delta', 'ibin', 'mirror', 'q', 'q_edges', 'nmodes', 'disconnected', 'gaussian_zero']

    def __init__(self, value, count, norm, delta, delta_norm, delta_shotnoise, ibin, mirror, q, q_edges, nmodes, observable,
                 disconnected=None, gaussian_zero=None, attrs=None):
        self.value = np.asarray(value)
        self.count = np.asarray(count, dtype='f8')
        self.norm = np.asarray(norm, dtype='f8')
        self.delta = np.asarray(delta)
        self.delta_norm = float(delta_norm)
        self.delta_shotnoise = float(delta_shotnoise)
        self.ibin = np.asarray(ibin, dtype='i8')
        self.mirror = np.asarray(mirror, dtype='i8')
        self.q, self.q_edges, self.nmodes = np.asarray(q), np.asarray(q_edges), np.asarray(nmodes)
        self.observable = observable
        n = self.value.shape[0]
        self.disconnected = np.zeros((len(self.q), n, n)) if disconnected is None else np.asarray(disconnected)
        self.gaussian_zero = np.zeros((n, n)) if gaussian_zero is None else np.asarray(gaussian_zero)
        self.attrs = dict(attrs or {})

    @property
    def nshells(self):
        return len(self.q)

    @property
    def size(self):
        return self.value.shape[0]

    def _shell_sum(self, value):
        return np.stack([value[..., self.ibin == ishell].sum(axis=-1) for ishell in range(self.nshells)], axis=0)

    def covariance(self) -> CollapsedCovariance:
        """The collapsed covariance spectra, with their disconnected part, as :func:`compute_collapsed_covariance` returns them."""
        num = np.stack([self.value[:, self.ibin == i] @ self.value[:, self.ibin == i].conj().T for i in range(self.nshells)])
        den = np.stack([self.count[:, self.ibin == i] @ self.count[:, self.ibin == i].T for i in range(self.nshells)])
        scale = self.norm[:, None] * self.norm[None, :]
        value = (num / den).real / scale
        disc = (self.disconnected / den).real / scale
        return CollapsedCovariance(value=value, observable=self.observable, q=self.q, q_edges=self.q_edges, nmodes=self.nmodes, disconnected=disc, attrs=self.attrs)

    def power(self):
        """``(nshells,)``: the shot-noise subtracted power spectrum of the soft modes, per shell."""
        p = np.abs(self.delta)**2 / self.delta_norm - self.delta_shotnoise
        return self._shell_sum(p) / np.maximum(self.nmodes, 1)

    def squeezed(self):
        r"""``(nshells, n)``: :math:`X_a(q) = \sum_q F_a(q) \delta^*(q) / \sum_q n_a(q)`, normalised like the covariance spectra."""
        dn = self.delta / np.sqrt(self.delta_norm)
        num = self._shell_sum(self.value * dn.conj()[None, :]).real
        den = self._shell_sum(self.count)
        return num / den / self.norm[None, :]

    def response_ratio(self, numerator, denominator, k, qmin=0., kratio=5.):
        r"""
        The stacked response ratio :math:`\sum_a \sum_q X_{n(a)}(q) / \sum_a \sum_q X_{d(a)}(q)` of the rows ``numerator``
        to the rows ``denominator`` (paired element by element), each pair summed over its shells :math:`q_{\rm min} < q \leq
        k_a / k_{\rm ratio}`, ``k`` the rows' wavenumbers.

        On one realisation a per-row fit of :math:`X_a = r_a P + c_a` and a ratio of the fitted :math:`r` is noise-dominated
        (on GLAM cutsky mocks, post / pre-reconstruction :math:`r` ratios spread over 0.25-1.2 from one mock); summing the
        responses first, over every row and usable shell, is not (0.34-0.51 per mock, 0.46 for their ensemble). No mean
        field is subtracted: with one realisation there is none to subtract; cut the lowest shells with ``qmin`` instead,
        where the window's mean field dominates.
        """
        numerator, denominator, k = (np.atleast_1d(np.asarray(x)) for x in (numerator, denominator, k))
        if not (len(numerator) == len(denominator) == len(k)):
            raise ValueError('numerator, denominator and k must pair up row by row')
        squeezed, q = self.squeezed(), np.asarray(self.q)  # X_a(q), (nshells, nrows)
        total_numerator = total_denominator = 0.
        for row_numerator, row_denominator, k_row in zip(numerator, denominator, k):
            # the shells soft enough for this row (q << k), above the mean-field-dominated ones
            shells = (q > max(qmin, 0.)) & (q <= k_row / kratio)
            total_numerator += squeezed[shells, row_numerator].sum()
            total_denominator += squeezed[shells, row_denominator].sum()
        return float(total_numerator / total_denominator)

    def mode_quantities(self, ia, ib, shell):
        r"""
        Per-mode versions of :math:`\hat P`, :math:`X_a`, :math:`X_b`, :math:`C_{ab}` on one shell, over independent modes only.

        Each is normalised by the shell's mean count, so that the mean over the shell's modes is the
        shell statistic; the modes :math:`q` and :math:`-q` carry conjugate values and only one of
        each pair is kept. Returns ``(nindependent, 4)``; the disconnected part is not subtracted.
        """
        idx = np.flatnonzero((self.ibin == shell) & (self.mirror >= np.arange(len(self.ibin))))
        full = np.flatnonzero(self.ibin == shell)
        dn = self.delta[idx] / np.sqrt(self.delta_norm)
        p = np.abs(dn)**2 - self.delta_shotnoise
        xa = (self.value[ia, idx] * dn.conj()).real / self.count[ia, full].mean() / self.norm[ia]
        xb = (self.value[ib, idx] * dn.conj()).real / self.count[ib, full].mean() / self.norm[ib]
        c = (self.value[ia, idx] * self.value[ib, idx].conj()).real / (self.count[ia, full] * self.count[ib, full]).mean() / (self.norm[ia] * self.norm[ib])
        return np.column_stack([p, xa, xb, c])

    def __getstate__(self, to_file=False):
        state = {name: getattr(self, name) for name in self._array_names}
        state.update(name=self._name, delta_norm=self.delta_norm, delta_shotnoise=self.delta_shotnoise, attrs=dict(self.attrs),
                     observable=self.observable.__getstate__(to_file=to_file))
        return state

    def __setstate__(self, state):
        from lsstypes.base import from_state
        for name in self._array_names:
            # Files written before gaussian_zero existed do not carry it
            if name == 'gaussian_zero' and name not in state:
                self.gaussian_zero = np.zeros((len(state['norm']),) * 2)
                continue
            setattr(self, name, np.asarray(state[name]))
        self.delta_norm, self.delta_shotnoise = float(state['delta_norm']), float(state['delta_shotnoise'])
        # The HDF5 writer drops an empty dict, as for lsstypes' CovarianceMatrix
        self.attrs = dict(state.get('attrs', {}))
        self.observable = from_state(state['observable'])

    def write(self, filename, **kwargs):
        from lsstypes.base import _write
        _write(filename, self.__getstate__(to_file=True), **kwargs)

    @classmethod
    def read(cls, filename):
        from lsstypes.base import _read
        state = _read(filename)
        assert str(state.pop('name')) == cls._name, 'not a collapsed modes file'
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new


def compute_collapsed_modes(fields: list, mesh, delta_norm: float, delta_shotnoise: float=0., disconnected: dict=None,
                            observable_fields: list=None) -> CollapsedModes:
    r"""
    Gather the collapsed fields of one realization, and the soft modes of its density mesh, into a :class:`CollapsedModes`.

    Parameters
    ----------
    fields : list of CollapsedField
        In the order the data vector is stacked; all on the same :class:`BinSoftModes`.
    mesh : RealMeshField
        The (mean-subtracted) density mesh the fields were built from.
    delta_norm : float
        Power spectrum normalisation, e.g. :func:`compute_box2_normalization`.
    delta_shotnoise : float, optional
        Power spectrum shot noise, in the units of ``|delta|^2 / delta_norm``.
    disconnected : dict, optional
        ``{(ifield, jfield): (nshells, n_i, n_j)}``, as for :func:`compute_collapsed_covariance`.
    observable_fields : list, optional
        As for :func:`compute_collapsed_covariance`.
    """
    soft = fields[0].soft
    value = np.concatenate([np.asarray(field.flat()) for field in fields], axis=0)
    count = np.concatenate([np.tile(np.asarray(field.count), (len(field.ells), 1)) for field in fields], axis=0)
    norm = np.concatenate([np.full(field.size, float(field.norm)) for field in fields])
    sizes = [field.size for field in fields]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    disc = np.zeros((soft.nshells, offsets[-1], offsets[-1]), dtype=value.dtype)
    for (ia, ib), d in dict(disconnected or {}).items():
        disc[:, offsets[ia]:offsets[ia + 1], offsets[ib]:offsets[ib + 1]] = np.asarray(d)
        if ia != ib:
            disc[:, offsets[ib]:offsets[ib + 1], offsets[ia]:offsets[ia + 1]] = np.conj(np.swapaxes(np.asarray(d), 1, 2))
    delta = np.asarray(soft.transform(mesh))
    if observable_fields is None:
        observable_fields = [(0,) * (2 if field.kind == 'mesh2' else 3) for field in fields]
    observable = ObservableTree([field.spectrum() for field in fields], fields=list(observable_fields))
    return CollapsedModes(value=value, count=count, norm=norm, delta=delta, delta_norm=delta_norm, delta_shotnoise=delta_shotnoise,
                          ibin=np.asarray(soft.ibin), mirror=np.asarray(soft.mirror), q=np.asarray(soft.qavg), q_edges=np.asarray(soft.edges),
                          nmodes=np.asarray(soft.nmodes), observable=observable, disconnected=disc)
