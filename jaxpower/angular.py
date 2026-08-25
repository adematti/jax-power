r"""
Angular power spectrum estimation, from catalogs (direct summation) or pixelated (healpix) maps.

The direct summation path follows https://arxiv.org/abs/2312.12285 (Baleato Lizancos & White 2024):
harmonic coefficients are computed as :math:`a_{\ell m} = \sum_i w_i Y_{\ell m}^*(\hat{n}_i)`,
i.e. a direct sum over particles, with no pixelization involved.
The FKP-style field is formed by linear subtraction, data :math:`-\ \alpha` randoms.

Conventions
-----------
- :class:`AlmField` stores :math:`a_{\ell m}` as a dense complex array of shape ``(ellmax + 1, ellmax + 1)``,
  indexed by ``[ell, m]`` with :math:`m \geq 0` (zeros for :math:`m > \ell`); fields are assumed real,
  such that :math:`a_{\ell, -m} = (-1)^m a_{\ell m}^*`.
- :class:`PixelField` stores a field value per healpix (ring-ordered) pixel, in units of "per steradian"
  when painted from particles, such that its quadrature :math:`a_{\ell m} = \sum_p f_p \Omega_p Y_{\ell m}^*(\hat{n}_p)`
  matches the direct summation up to pixelization effects. This equals ``healpy.map2alm(map, iter=0)``.
- The measured (raw) binned power is :math:`\mathrm{num}_b = \sum_{\ell \in b} \sum_m a^{(1)}_{\ell m} a^{(2)*}_{\ell m} / N_b`
  with :math:`N_b = \sum_{\ell \in b} (2\ell + 1)`. The normalized spectrum is
  :math:`C_b = (\mathrm{num}_b - \mathrm{num}_b^{\mathrm{shot}}) / \mathrm{norm}` with
  :math:`\mathrm{norm} = \int d\Omega\, \bar{n}_1 \bar{n}_2 / (4\pi)` (such that, for a full-sky uniform sample,
  the shot noise is :math:`1/\bar{n}` per steradian).

Note
----
Use ``jax.config.update('jax_enable_x64', True)`` for accurate recurrences at :math:`\ell \gtrsim 100`.
"""

import math
import operator
import functools
import itertools
from functools import partial, lru_cache
from dataclasses import dataclass

import numpy as np
import jax
from jax import numpy as jnp

from .mesh import staticarray, ParticleField, FKPField, _make_input_tuple, split_particles, __format_meshes
from .mesh2 import _format_meshes
from .utils import register_pytree_dataclass, _legendre_recurrence_coeffs
from .types import Angular2Spectrum, Angular3Spectrum, WindowMatrix


_format_meshes3 = partial(__format_meshes, nmeshes=3)


prod = functools.partial(functools.reduce, operator.mul)


@dataclass(frozen=True)
class AngularAttrs(object):
    """
    Attributes for angular (sphere) computation; analog of :class:`MeshAttrs`.

    Parameters
    ----------
    ellmax : int
        Maximum multipole order :math:`\\ell_\\mathrm{max}`.
    nside : int, optional
        Healpix resolution, for the pixelated ('healpix') backend only.
    """
    ellmax: int = None
    nside: int = None

    def __post_init__(self):
        assert self.ellmax is not None and self.ellmax >= 0

    @property
    def nell(self):
        return self.ellmax + 1

    @property
    def ells(self):
        return np.arange(self.ellmax + 1)

    @property
    def npix(self):
        assert self.nside is not None, 'provide nside for pixelated computation'
        return 12 * self.nside**2

    @property
    def pixarea(self):
        return 4. * np.pi / self.npix

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = dict(self) | kwargs
        return self.__class__(**state)

    # For mapping, as :class:`MeshAttrs`
    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self.__annotations__.keys()


@partial(register_pytree_dataclass, meta_fields=['attrs'])
@dataclass(frozen=True)
class AlmField(object):
    """
    Harmonic coefficients :math:`a_{\\ell m}` of a real field on the sphere.

    ``value`` is a dense complex array of shape ``(ellmax + 1, ellmax + 1)``, indexed by ``[ell, m]``
    with :math:`m \\geq 0` (zeros for :math:`m > \\ell`).
    """
    value: jax.Array = None
    attrs: AngularAttrs = None

    @property
    def ells(self):
        return np.arange(self.attrs.ellmax + 1)

    def conj(self):
        return self.clone(value=self.value.conj())

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = dict(value=self.value, attrs=self.attrs) | kwargs
        return self.__class__(**state)

    def __add__(self, other):
        assert isinstance(other, AlmField) and other.attrs.ellmax == self.attrs.ellmax
        return self.clone(value=self.value + other.value)

    def __sub__(self, other):
        assert isinstance(other, AlmField) and other.attrs.ellmax == self.attrs.ellmax
        return self.clone(value=self.value - other.value)

    def __mul__(self, other):
        assert not isinstance(other, AlmField)
        return self.clone(value=self.value * other)

    __rmul__ = __mul__

    def to_pixel(self, nside: int=None, backend: str=None) -> 'PixelField':
        r"""
        Inverse spherical harmonic transform, :math:`f(\hat{n}) = \sum_{\ell m} a_{\ell m} Y_{\ell m}(\hat{n})`,
        assuming a real field (:math:`a_{\ell,-m} = (-1)^m a_{\ell m}^*`). See :func:`alm2map`.
        """
        return alm2map(self, nside=nside, backend=backend)

    def to_healpy(self):
        """Return alm packed in healpy order (m-major)."""
        import healpy as hp
        ellmax = self.attrs.ellmax
        value = np.asarray(self.value)
        ls, ms = np.tril_indices(ellmax + 1)
        alm = np.zeros(hp.Alm.getsize(ellmax), dtype=value.dtype)
        alm[hp.Alm.getidx(ellmax, ls, ms)] = value[ls, ms]
        return alm

    @classmethod
    def from_healpy(cls, alm, ellmax=None):
        """Build :class:`AlmField` from healpy-ordered alm."""
        import healpy as hp
        alm = np.asarray(alm)
        if ellmax is None:
            ellmax = hp.Alm.getlmax(alm.size)
        ls, ms = np.tril_indices(ellmax + 1)
        value = np.zeros((ellmax + 1,) * 2, dtype=alm.dtype)
        value[ls, ms] = alm[hp.Alm.getidx(ellmax, ls, ms)]
        return cls(value=jnp.asarray(value), attrs=AngularAttrs(ellmax=ellmax))


@partial(register_pytree_dataclass, meta_fields=['attrs'])
@dataclass(frozen=True)
class PixelField(object):
    """
    Field on the sphere sampled on healpix (ring-ordered) pixels; analog of :class:`RealMeshField`.

    ``value`` has shape ``(12 * nside**2,)``. When painted from particles, ``value`` is a surface
    density (summed weights per steradian), see :func:`to_pixel`.
    """
    value: jax.Array = None
    attrs: AngularAttrs = None

    @property
    def nside(self):
        return self.attrs.nside

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = dict(value=self.value, attrs=self.attrs) | kwargs
        return self.__class__(**state)

    def to_alm(self, ellmax=None, backend: str=None, batch_size=None) -> AlmField:
        r"""
        Spherical harmonic transform, with uniform pixel weights
        (same as ``healpy.map2alm(..., iter=0, use_weights=False)``).

        Parameters
        ----------
        ellmax : int, optional
            Maximum multipole order; defaults to ``attrs.ellmax``.
        backend : str, optional
            'jax_healpy' (default when importable): FFT-over-rings transform, jit- and grad-compatible,
            and the only backend that parallelizes over shards.
            'healpy': ``healpy.map2alm``, wrapped in :func:`jax.pure_callback` so it can still be traced;
            the transform itself runs on host, so a sharded map is gathered first.
            'quadrature': direct summation :math:`a_{\ell m} = \Omega_\mathrm{pix} \sum_p f_p Y_{\ell m}^*(\hat{n}_p)`
            over pixel centers. Same result, but :math:`\mathcal{O}(N_\mathrm{pix} \ell_\mathrm{max}^2)`,
            hence only practical at low resolution; kept as an independent reference.
        batch_size : int, optional
            Number of pixels processed at once ('quadrature' backend only).

        Returns
        -------
        alm : AlmField

        Note
        ----
        All backends support reverse-mode differentiation (:func:`jax.grad`): 'healpy' through the
        analytic adjoint of :func:`_map2alm_healpy`, 'jax_healpy' through s2fft. Only 'quadrature'
        supports forward mode (:func:`jax.jacfwd`) as well, the other two both relying on ``custom_vjp``.
        """
        attrs = self.attrs if ellmax is None else self.attrs.clone(ellmax=ellmax)
        backend = _get_pixel_backend(backend)
        if backend == 'jax_healpy':
            import jax_healpy as jhp
            # s2fft's healpix transform requires L = lmax + 1 >= 2 * nside, which excludes the usual
            # well-sampled regime (ellmax <~ 2 nside). Since the iter=0 quadrature gives each a_lm
            # independently of lmax, transform at the smallest allowed lmax and truncate.
            ellmax = max(attrs.ellmax, 2 * attrs.nside - 1)
            value = jhp.map2alm(self.value, lmax=ellmax, iter=0, pol=False, healpy_ordering=False)
            # s2fft layout is (ellmax + 1, 2 * ellmax + 1), indexed [ell, ellmax + m]; keep m >= 0
            value = value[:attrs.ellmax + 1, ellmax:ellmax + attrs.ellmax + 1]
        elif backend == 'healpy':
            value = _map2alm_healpy(self.value, attrs.ellmax, attrs.nside)
        else:
            import healpy as hp
            positions = np.column_stack(hp.pix2vec(attrs.nside, np.arange(attrs.npix)))
            weights = self.value * attrs.pixarea
            value = _compute_alm_direct(jnp.asarray(positions), weights, ellmax=attrs.ellmax, batch_size=batch_size)
        return AlmField(value=value, attrs=attrs)


def _map2alm_healpy_impl(value: jax.Array, ellmax: int) -> jax.Array:
    """``healpy.map2alm`` (uniform pixel weights, ``iter=0``) wrapped in :func:`jax.pure_callback`."""
    import healpy as hp
    cdtype = jnp.zeros(0, dtype=value.dtype).astype(complex).dtype
    size = hp.Alm.getsize(ellmax)

    def host(value):
        alm = hp.map2alm(np.asarray(value, dtype='f8'), lmax=ellmax, iter=0, use_pixel_weights=False)
        return alm.astype(cdtype)

    # 'sequential': unlike the element-wise vec2pix, the transform acts on a whole map, so a batch
    # of maps must be transformed one at a time
    alm = jax.pure_callback(host, jax.ShapeDtypeStruct((size,), cdtype), value, vmap_method='sequential')
    ls, ms = np.tril_indices(ellmax + 1)
    return jnp.zeros((ellmax + 1,) * 2, dtype=cdtype).at[ls, ms].set(alm[hp.Alm.getidx(ellmax, ls, ms)])


def _alm2map_healpy(alm: jax.Array, nside: int, dtype=None) -> jax.Array:
    r"""
    ``healpy.alm2map`` wrapped in :func:`jax.pure_callback`, taking the dense ``[ell, m]`` layout:
    :math:`f_p = \sum_\ell [a_{\ell 0} Y_{\ell 0}(\hat{n}_p) + 2 \mathrm{Re} \sum_{m > 0} a_{\ell m} Y_{\ell m}(\hat{n}_p)]`,
    i.e. assuming a real field.
    """
    import healpy as hp
    ellmax = alm.shape[0] - 1
    npix = 12 * nside**2
    if dtype is None: dtype = alm.real.dtype
    ls, ms = np.tril_indices(ellmax + 1)
    index = hp.Alm.getidx(ellmax, ls, ms)
    size = hp.Alm.getsize(ellmax)

    def host(alm):
        packed = np.zeros(size, dtype='c16')
        packed[index] = np.asarray(alm)[ls, ms]
        return hp.alm2map(packed, nside, lmax=ellmax, pol=False).astype(dtype)

    return jax.pure_callback(host, jax.ShapeDtypeStruct((npix,), dtype), alm, vmap_method='sequential')


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _map2alm_healpy(value: jax.Array, ellmax: int, nside: int) -> jax.Array:
    """
    ``healpy.map2alm`` (uniform pixel weights, ``iter=0``) wrapped in :func:`jax.pure_callback`,
    so it can be traced, returning the dense ``[ell, m]`` layout of :class:`AlmField`.

    Callbacks are opaque to autodiff, so the gradient is supplied analytically: the transform is
    *linear* in the map, and the adjoint of :math:`a_{\\ell m} = \\Omega_\\mathrm{pix} \\sum_p f_p Y_{\\ell m}^*(\\hat{n}_p)`
    is ``healpy.alm2map``, see :func:`_map2alm_healpy_bwd`.

    Note
    ----
    Unlike the painting, a harmonic transform is not element-wise: a sharded map is gathered on host.
    """
    return _map2alm_healpy_impl(value, ellmax)


def _map2alm_healpy_fwd(value, ellmax, nside):
    return _map2alm_healpy_impl(value, ellmax), None


def _map2alm_healpy_bwd(ellmax, nside, res, cotangent):
    r"""
    Adjoint of the (linear) quadrature transform. JAX pairs a complex cotangent with the conjugate,
    so over the stored :math:`m \geq 0` coefficients
    :math:`\bar{f}_p = \Omega_\mathrm{pix} \mathrm{Re} \sum_{\ell, m \geq 0} \bar{a}_{\ell m}^* Y_{\ell m}(\hat{n}_p)`,
    whereas ``healpy.alm2map`` weights :math:`m > 0` by 2 (the implicit :math:`m < 0` partners of a real
    field); hence the conjugation and halving below.
    """
    pixarea = 4. * np.pi / (12 * nside**2)
    cotangent = jnp.conj(cotangent).at[:, 1:].multiply(0.5)
    return (pixarea * _alm2map_healpy(cotangent, nside=nside, dtype=cotangent.real.dtype),)


_map2alm_healpy.defvjp(_map2alm_healpy_fwd, _map2alm_healpy_bwd)


def _dense_to_s2fft(value: jax.Array, ellmax: int=None) -> jax.Array:
    r"""
    Convert the dense ``[ell, m >= 0]`` layout to the s2fft layout ``(L, 2L - 1)`` indexed
    ``[ell, ellmax + m]``, filling :math:`m < 0` with :math:`a_{\ell,-m} = (-1)^m a_{\ell m}^*`
    (real field). Zero-pads up to ``ellmax`` if provided.
    """
    ellmax_in = value.shape[0] - 1
    if ellmax is None: ellmax = ellmax_in
    signs = jnp.asarray((-1.)**np.arange(ellmax_in + 1), dtype=value.real.dtype)
    flm = jnp.zeros((ellmax + 1, 2 * ellmax + 1), dtype=value.dtype)
    flm = flm.at[:ellmax_in + 1, ellmax:ellmax + ellmax_in + 1].set(value)
    # m < 0, reversed in m; drop the m = 0 column, already set above
    flm = flm.at[:ellmax_in + 1, ellmax - ellmax_in:ellmax].set((signs * jnp.conj(value))[:, :0:-1])
    return flm


def alm2map(alm: AlmField, nside: int=None, backend: str=None) -> 'PixelField':
    r"""
    Inverse spherical harmonic transform, :math:`f(\hat{n}) = \sum_{\ell m} a_{\ell m} Y_{\ell m}(\hat{n})`,
    assuming a real field, so that only :math:`m \geq 0` need be stored.

    Parameters
    ----------
    alm : AlmField
        Input harmonic coefficients.
    nside : int, optional
        Healpix resolution of the output map; defaults to ``alm.attrs.nside``.
    backend : str, optional
        'jax_healpy' (default when importable), jit- and grad-compatible, or 'healpy',
        wrapped in :func:`jax.pure_callback`.

    Returns
    -------
    pixel : PixelField
    """
    attrs = alm.attrs
    if nside is not None: attrs = attrs.clone(nside=nside)
    assert attrs.nside is not None, 'provide nside'
    backend = _get_pixel_backend(backend)
    if backend == 'jax_healpy':
        import jax_healpy as jhp
        # same s2fft constraint as in PixelField.to_alm; zero-padding the alm leaves the map unchanged
        ellmax = max(attrs.ellmax, 2 * attrs.nside - 1)
        value = jhp.alm2map(_dense_to_s2fft(alm.value, ellmax=ellmax), attrs.nside, lmax=ellmax,
                            pol=False, healpy_ordering=False)
        value = value.real  # the field is real by construction, the reality condition being imposed above
    else:
        value = _alm2map_healpy(alm.value, nside=attrs.nside)
    return PixelField(value=value, attrs=attrs)


def _get_method(method: str=None, attrs: AngularAttrs=None):
    """
    Resolve the estimation method: if not given, use 'healpix' when a resolution ``nside`` is
    available, and the pixel-free 'direct' summation otherwise.
    """
    if method is None:
        return 'direct' if getattr(attrs, 'nside', None) is None else 'healpix'
    assert method in ('direct', 'healpix'), f'unknown method {method}'
    if method == 'healpix' and getattr(attrs, 'nside', None) is None:
        raise ValueError("method='healpix' requires attrs.nside; provide it, or use method='direct'")
    return method


def _get_pixel_backend(backend: str=None):
    """Return the backend to use for pixelated operations, defaulting to 'jax_healpy' if importable."""
    if backend is None:
        try:
            import jax_healpy  # noqa: F401
        except ImportError:
            return 'healpy'
        return 'jax_healpy'
    assert backend in ('jax_healpy', 'healpy', 'quadrature'), f'unknown backend {backend}'
    return backend


def to_pixel(particles: ParticleField | FKPField, attrs: AngularAttrs, backend: str=None) -> PixelField:
    """
    Paint particles onto a healpix map (nearest pixel); positions are interpreted as directions
    (observer at origin). The returned :attr:`PixelField.value` is summed weights per steradian.

    Parameters
    ----------
    particles : ParticleField or FKPField
        Input particles; for :class:`FKPField`, the data :math:`-\\ \\alpha` randoms field is painted.
    attrs : AngularAttrs
        Angular attributes, providing ``nside``.
    backend : str, optional
        'jax_healpy' (default when importable), which is jit- and grad-compatible,
        or 'healpy', which runs on host and breaks the trace.

    Returns
    -------
    pixel : PixelField
    """
    if isinstance(particles, FKPField):
        particles = particles.particles
    if isinstance(particles, ParticleField):
        positions, weights = particles.positions, particles.weights
    else:
        positions, weights = particles
    backend = _get_pixel_backend(backend)
    if backend == 'quadrature':  # 'quadrature' only qualifies the transform, not the painting
        backend = _get_pixel_backend(None)
    positions, weights = jnp.asarray(positions), jnp.asarray(weights)
    if backend == 'healpy':
        ipix = _vec2pix_healpy(positions, nside=attrs.nside)
    else:
        import jax_healpy as jhp
        ipix = jhp.vec2pix(attrs.nside, positions[..., 0], positions[..., 1], positions[..., 2])
    # scatter-add in JAX in both cases, so sharding semantics do not depend on the backend
    value = jnp.zeros(attrs.npix, dtype=weights.dtype).at[ipix].add(weights)
    return PixelField(value=value / attrs.pixarea, attrs=attrs)


def _vec2pix_healpy(positions: jax.Array, nside: int) -> jax.Array:
    """
    ``healpy.vec2pix`` wrapped in :func:`jax.pure_callback`, so it can be traced (jit, vmap)
    and applied to sharded positions. The operation is element-wise in the particles.
    """
    import healpy as hp

    def host(positions):
        positions = np.asarray(positions)
        return hp.vec2pix(nside, positions[..., 0], positions[..., 1], positions[..., 2]).astype(np.int64)

    out_type = jax.ShapeDtypeStruct(jnp.shape(positions)[:-1], jnp.zeros(0, dtype=np.int64).dtype)
    return jax.pure_callback(host, out_type, positions, vmap_method='broadcast_all')


def _compute_alm_direct(positions: jax.Array, weights: jax.Array, ellmax: int, batch_size: int=None) -> jax.Array:
    r"""
    Direct summation :math:`a_{\ell m} = \sum_i w_i Y_{\ell m}^*(\hat{n}_i)` (https://arxiv.org/abs/2312.12285).

    Parameters
    ----------
    positions : jax.Array
        Particle positions, shape (N, 3); interpreted as directions (normalized internally).
    weights : jax.Array
        Particle weights, shape (N,).
    ellmax : int
        Maximum multipole order.
    batch_size : int, optional
        Number of particles processed at once; memory scales as ``batch_size * (ellmax + 1)``.

    Returns
    -------
    alm : jax.Array
        Dense complex array of shape ``(ellmax + 1, ellmax + 1)``, indexed by ``[ell, m]`` (zeros for m > ell).
    """
    ellmax = int(ellmax)
    positions = jnp.asarray(positions)
    rdtype = positions.dtype
    weights = jnp.asarray(weights, dtype=rdtype)
    cdtype = jnp.zeros(0, dtype=rdtype).astype(complex).dtype
    # guard degenerate positions: particle exchange pads shards with zero-weight particles at the
    # mean position, which sits at the origin for a box centered there. 0 / 0 would give NaN, and
    # NaN * 0 stays NaN, poisoning the whole sum
    norm = jnp.linalg.norm(positions, axis=-1, keepdims=True)
    unit = positions / jnp.where(norm == 0., 1., norm)
    size = unit.shape[0]
    if batch_size is None:
        batch_size = max(1, 2**22 // (ellmax + 1))
    batch_size = min(batch_size, size)
    nbatch = (size + batch_size - 1) // batch_size
    pad = nbatch * batch_size - size
    if pad:
        unit = jnp.concatenate([unit, jnp.broadcast_to(jnp.array([0., 0., 1.], dtype=rdtype), (pad, 3))], axis=0)
        weights = jnp.concatenate([weights, jnp.zeros(pad, dtype=rdtype)], axis=0)
    unit = unit.reshape(nbatch, batch_size, 3)
    weights = weights.reshape(nbatch, batch_size)

    a, ab, d, onehot = _legendre_recurrence_coeffs(ellmax)
    a, ab, d, onehot = (jnp.asarray(tmp, dtype=rdtype) for tmp in (a, ab, d, onehot))
    c00 = 1. / math.sqrt(4. * math.pi)
    ms = jnp.arange(ellmax + 1, dtype=rdtype)

    def compute_chunk(unit, weights):
        x = unit[..., 2]
        sin = jnp.hypot(unit[..., 0], unit[..., 1])
        phi = jnp.arctan2(unit[..., 1], unit[..., 0])
        wphase = weights[:, None] * jnp.exp(-1j * phi[:, None] * ms)  # (batch, ellmax + 1)
        lam0 = jnp.zeros((batch_size, ellmax + 1), dtype=rdtype).at[:, 0].set(c00)
        alm0 = jnp.sum(wphase * lam0, axis=0)

        def body(carry, xs):
            lam1, lam2, diag = carry
            a_row, ab_row, d_ell, oh = xs
            diag = d_ell * sin * diag
            lam = a_row * (x[:, None] * lam1) - ab_row * lam2
            lam = lam + diag[:, None] * oh
            alm_ell = jnp.sum(wphase * lam, axis=0)
            return (lam, lam1, diag), alm_ell

        if ellmax == 0:
            return alm0[None, :]
        init = (lam0, jnp.zeros_like(lam0), jnp.full((batch_size,), c00, dtype=rdtype))
        _, alms = jax.lax.scan(body, init, (a[1:], ab[1:], d[1:], onehot[1:]))
        return jnp.concatenate([alm0[None, :], alms], axis=0)

    def body(carry, xs):
        return carry + compute_chunk(*xs), None

    alm, _ = jax.lax.scan(body, jnp.zeros((ellmax + 1,) * 2, dtype=cdtype), (unit, weights))
    return alm


def to_alm(field, weights=None, attrs: AngularAttrs=None, method: str=None, backend: str=None, batch_size: int=None) -> AlmField:
    """
    Compute harmonic coefficients of the input field.

    Parameters
    ----------
    field : ParticleField, FKPField, PixelField, AlmField, or jax.Array
        Input field. Particle positions are interpreted as directions (observer at origin).
        For :class:`FKPField`, the field is data :math:`-\\ \\alpha` randoms.
        A plain array of positions (N, 3) can be given, together with ``weights``.
    weights : jax.Array, optional
        Weights, if ``field`` is a plain array of positions.
    attrs : AngularAttrs, optional
        Angular attributes (``ellmax``, and ``nside`` for ``method='healpix'``).
    method : str, optional
        'direct': pixel-free direct summation over particles (https://arxiv.org/abs/2312.12285).
        'healpix': paint particles onto a healpix map of resolution ``attrs.nside``, then transform.
        If ``None`` (default), 'healpix' is used when ``attrs.nside`` is set, else 'direct'.
    backend : str, optional
        Backend for ``method='healpix'``: 'jax_healpy' (default when importable), which keeps the whole
        path jit- and grad-compatible, or 'healpy' / 'quadrature', see :meth:`PixelField.to_alm`.
    batch_size : int, optional
        Number of particles processed at once.

    Returns
    -------
    alm : AlmField
    """
    if isinstance(field, AlmField):
        if attrs is not None and attrs.ellmax != field.attrs.ellmax:
            assert attrs.ellmax <= field.attrs.ellmax, f'input AlmField has ellmax = {field.attrs.ellmax} < {attrs.ellmax}'
            return AlmField(value=field.value[:attrs.ellmax + 1, :attrs.ellmax + 1], attrs=field.attrs.clone(ellmax=attrs.ellmax))
        return field
    assert attrs is not None, 'provide attrs (ellmax)'
    method = _get_method(method, attrs)
    if isinstance(field, PixelField):
        return field.to_alm(ellmax=attrs.ellmax, backend=backend, batch_size=batch_size)
    if method == 'healpix':
        return to_pixel(field, attrs=attrs, backend=backend).to_alm(backend=backend, batch_size=batch_size)
    if isinstance(field, FKPField):
        field = field.particles
    if isinstance(field, ParticleField):
        positions, weights = field.positions, field.weights
    else:
        positions = field
    if weights is None:
        weights = jnp.ones_like(positions[..., 0])
    value = _compute_alm_direct(positions, weights, ellmax=attrs.ellmax, batch_size=batch_size)
    return AlmField(value=value, attrs=attrs)


def _make_edges_angular(attrs, edges):
    ellmax = attrs.ellmax
    if edges is None:
        edges = {}
    if isinstance(edges, dict):
        step = edges.get('step', 1)
        edges = np.arange(edges.get('min', 0), edges.get('max', ellmax + 1) + 1e-5, step)
    edges = np.asarray(edges)
    if edges.ndim == 2:  # coming from ObservableTree
        assert np.allclose(edges[1:, 0], edges[:-1, 1])
        edges = np.append(edges[:, 0], edges[-1, 1])
    edges = np.column_stack([edges[:-1], edges[1:]])
    ells = np.arange(ellmax + 1)
    # wbin[b, ell] = 1 if ell in bin b (and ell <= ellmax)
    wbin = (ells >= edges[:, [0]]) & (ells < edges[:, [1]])
    nmodes = np.sum(wbin * (2 * ells + 1), axis=-1)
    assert np.all(nmodes > 0), 'empty ell-bins (of edges <= ellmax)'
    xavg = np.sum(wbin * (2 * ells + 1) * ells, axis=-1) / nmodes
    return dict(edges=staticarray(edges), xavg=staticarray(xavg), nmodes=staticarray(nmodes), wbin=staticarray(wbin), attrs=attrs)


@partial(register_pytree_dataclass, meta_fields=['edges', 'xavg', 'nmodes', 'wbin', 'attrs'])
@dataclass(init=False, frozen=True)
class BinAngular2Spectrum(object):
    r"""
    Binning operator for the angular power spectrum: bandpowers in :math:`\ell`, with
    :math:`(2\ell + 1)` weights.

    Parameters
    ----------
    attrs : AngularAttrs
        Angular attributes (``ellmax``).
    edges : array-like or dict, optional
        ``edges`` may be:
        - a numpy array containing the :math:`\ell`-bin edges; bin :math:`b` collects
          :math:`\mathrm{edges}[b] \leq \ell < \mathrm{edges}[b + 1]`.
        - a dictionary, with keys 'min' (minimum :math:`\ell`, defaults to 0),
          'max' (defaults to ``ellmax + 1``), 'step' (defaults to 1).
        - ``None``, defaults to unit bins covering :math:`0 \leq \ell \leq \ell_\mathrm{max}`.
    """
    edges: staticarray = None
    xavg: staticarray = None
    nmodes: staticarray = None
    wbin: staticarray = None
    attrs: AngularAttrs = None

    def __init__(self, attrs: AngularAttrs, edges: staticarray | dict | None=None):
        if not isinstance(attrs, AngularAttrs):
            attrs = attrs.attrs
        kw = _make_edges_angular(attrs, edges)
        self.__dict__.update(kw)

    def __call__(self, num):
        r"""
        Bin per-:math:`\ell` raw power :math:`\mathrm{num}_\ell = \sum_m a^{(1)}_{\ell m} a^{(2)*}_{\ell m}`
        (array of shape ``(ellmax + 1,)``): return :math:`\sum_{\ell \in b} \mathrm{num}_\ell / N_b`
        with :math:`N_b = \sum_{\ell \in b} (2\ell + 1)`.
        """
        return jnp.asarray(self.wbin, dtype=num.dtype) @ num / jnp.asarray(self.nmodes, dtype=num.dtype)


def _compute_cross_power(alm1: AlmField, alm2: AlmField):
    r"""Per-:math:`\ell` raw power :math:`\sum_{m=-\ell}^{\ell} a^{(1)}_{\ell m} a^{(2)*}_{\ell m}` for real fields."""
    assert alm1.attrs.ellmax == alm2.attrs.ellmax
    prod = alm1.value * jnp.conj(alm2.value)
    return jnp.real(prod[:, 0]) + 2. * jnp.sum(jnp.real(prod[:, 1:]), axis=-1)


def compute_angular2(*fields, bin: BinAngular2Spectrum=None, method: str=None, backend: str=None, batch_size: int=None):
    """Dispatch to :func:`compute_angular2_spectrum` (single ``bin`` type for now)."""
    if isinstance(bin, BinAngular2Spectrum):
        return compute_angular2_spectrum(*fields, bin=bin, method=method, backend=backend, batch_size=batch_size)
    raise ValueError(f'bin must be BinAngular2Spectrum, not {type(bin)}')


def compute_angular2_spectrum(*fields, bin: BinAngular2Spectrum=None, method: str=None, backend: str=None, batch_size: int=None) -> Angular2Spectrum:
    r"""
    Compute the angular power spectrum :math:`C_\ell` (auto or cross).

    Parameters
    ----------
    fields : ParticleField, FKPField, PixelField or AlmField
        Input field(s) (autocorrelation if one provided).
        Particle positions are interpreted as directions (observer at origin).
    bin : BinAngular2Spectrum
        Binning operator.
    method : str, optional
        'direct', 'healpix', or ``None`` (default) to pick from ``attrs.nside``, see :func:`to_alm`.
    batch_size : int, optional
        Number of particles processed at once.

    Returns
    -------
    spectrum : Angular2Spectrum
        Raw (unnormalized) spectrum; apply normalization and shot noise with
        ``spectrum.clone(norm=..., num_shotnoise=...)``, see
        :func:`compute_fkp_angular2_normalization` and :func:`compute_fkp_angular2_shotnoise`.
    """
    fields = _make_input_tuple(*fields)
    assert 1 <= len(fields) <= 2
    alms = [to_alm(field, attrs=bin.attrs, method=method, backend=backend, batch_size=batch_size) for field in fields]
    if len(alms) == 1: alms = alms * 2
    num = bin(_compute_cross_power(*alms))
    return Angular2Spectrum(ell=bin.xavg, ell_edges=bin.edges, num_raw=num, nmodes=np.asarray(bin.nmodes),
                            norm=jnp.ones_like(num), num_shotnoise=jnp.zeros_like(num))


def compute_angular_normalization(*particles: ParticleField | PixelField, attrs: AngularAttrs=None, nside: int=None) -> jax.Array:
    r"""
    Compute normalization :math:`\int d\Omega \prod_i n_i(\Omega) / (4\pi)` for input fields,
    with each :math:`n_i` painted on a healpix map of resolution ``nside``.

    Parameters
    ----------
    particles : ParticleField or PixelField
        Input fields, assumed uncorrelated (else the Poisson self-pair bias is not removed).
    attrs : AngularAttrs, optional
        Angular attributes, providing ``nside``.
    nside : int, optional
        Healpix resolution, if ``attrs`` is not provided.

    Returns
    -------
    normalization : jax.Array

    Warning
    -------
    Input particles are considered uncorrelated.
    """
    if attrs is None:
        attrs = AngularAttrs(ellmax=0, nside=nside)
    elif nside is not None:
        attrs = attrs.clone(nside=nside)
    normalization = 1.
    for particle in _make_input_tuple(*particles):
        value = particle.value if isinstance(particle, PixelField) else to_pixel(particle, attrs=attrs).value
        normalization = normalization * value
    return jnp.sum(normalization) * attrs.pixarea / (4. * np.pi)


def compute_fkp_angular2_normalization(*fkps: FKPField, bin: BinAngular2Spectrum=None, nside: int=None, split=None, fields: tuple=None):
    r"""
    Compute the FKP normalization :math:`\int d\Omega\, \bar{n}_1 \bar{n}_2 / (4\pi)` for the angular power spectrum,
    with :math:`\bar{n}` estimated from the (:math:`\alpha`-scaled) randoms painted on a healpix map of resolution ``nside``.

    Painting the same catalog twice would bias :math:`\int \bar{n}^2 d\Omega` by its Poisson self-pairs.
    Instead, two *uncorrelated* catalogs are crossed: data :math:`\times` randoms (default, this is the pypower
    normalization), or disjoint random subsamples if ``split`` is provided.

    Parameters
    ----------
    fkps : FKPField or ParticleField
        FKP fields (:math:`\bar{n} = \alpha \times` randoms) or particles (:math:`\bar{n} =` particles).
    bin : BinAngular2Spectrum, optional
        Binning operator. Only used to provide the default ``nside``.
    nside : int, optional
        Healpix resolution for the :math:`\bar{n}` estimate; defaults to ``bin.attrs.nside`` or 64.
    split : int, list, optional
        If provided, seed(s) used to select disjoint random subsamples, one per field,
        instead of crossing data and randoms.
    fields : tuple, optional
        Field identifiers; pass e.g. [0, 0] if two fields sharing the same positions are given as input.
        Disjoint random subsamples will be selected, if ``split`` is provided.

    Returns
    -------
    norm : float
    """
    if nside is None:
        nside = getattr(bin.attrs, 'nside', None) if bin is not None else None
    if nside is None:
        nside = 64
    attrs = AngularAttrs(ellmax=0, nside=nside)
    normalization = partial(compute_angular_normalization, attrs=attrs)

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    def get_alpha(fkp):
        return fkp.data.sum() / fkp.randoms.sum() if isinstance(fkp, FKPField) else 1.

    if split is not None:
        fkps, fields = _format_meshes(*fkps, fields=fields)
        alpha = prod(map(get_alpha, fkps))
        randoms = list(split_particles(*[get_randoms(fkp) for fkp in fkps], seed=split, fields=fields))
        alpha *= prod(get_randoms(fkp).sum() / randoms.sum() for fkp, randoms in zip(fkps, randoms, strict=True))
        norm = alpha * normalization(*randoms)
    else:
        # This is the pypower normalization
        fkps, fields = _format_meshes(*fkps)  # fields not provided on purpose
        if fields[1] == fields[0]:
            fkp = fkps[0]
            if isinstance(fkp, FKPField):
                norm = get_alpha(fkp) * normalization(fkp.data, fkp.randoms)  # cross to remove common noise
            else:
                # single catalog: nothing uncorrelated to cross it with, so subtract the Poisson self-pairs
                # (exact, as painting is nearest-neighbor: self-pairs all land in the same pixel)
                norm = normalization(fkp, fkp) - jnp.sum(fkp.weights**2) / (attrs.pixarea * 4. * np.pi)
        else:
            norm = get_alpha(fkps[1]) * normalization(fkps[0].data, fkps[1].randoms)  # cross to remove common noise
            norm += get_alpha(fkps[0]) * normalization(fkps[1].data, fkps[0].randoms)
            norm = norm / 2.
    return norm


def compute_fkp_angular2_shotnoise(*fkps: FKPField, bin: BinAngular2Spectrum=None, fields: tuple=None):
    r"""
    Shot noise :math:`(\sum_D w^2 + \alpha^2 \sum_R w^2) / (4\pi)` (raw, same units as ``num_raw``);
    zero for cross-correlations of distinct fields.

    Parameters
    ----------
    fkps : FKPField or ParticleField
        FKP fields or particles.
    bin : BinAngular2Spectrum
        Binning operator.
    fields : tuple, optional
        Field identifiers; pass e.g. [0, 0] if two fields sharing the same positions are given.

    Returns
    -------
    num_shotnoise : jax.Array
        Array of shape ``(nbins,)`` (flat in :math:`\ell`).
    """
    fkps = _make_input_tuple(*fkps)
    fkps, fields = _format_meshes(*fkps, fields=fields)
    if fields[1] == fields[0]:
        fkp = fkps[0]
        if isinstance(fkp, FKPField):
            alpha = fkp.data.sum() / fkp.randoms.sum()
            num = jnp.sum(fkp.data.weights**2) + alpha**2 * jnp.sum(fkp.randoms.weights**2)
        else:
            num = jnp.sum(fkp.weights**2)
        num = num / (4. * np.pi)
    else:
        num = 0.
    return num * jnp.ones(len(bin.xavg))


@partial(register_pytree_dataclass, meta_fields=['edges', 'xavg', 'nmodes', 'ibands', 'band_edges', 'wbin', 'attrs'])
@dataclass(init=False, frozen=True)
class BinAngular3Spectrum(object):
    r"""
    Binning operator for the angular bispectrum: triplets of :math:`\ell`-bands.

    The estimator is the binned (filtered-map) bispectrum: with the band-filtered maps
    :math:`M_i(\hat{n}) = \sum_{\ell \in i, m} a_{\ell m} Y_{\ell m}(\hat{n})`,

    .. math:: \int d\Omega\, M_{i} M_{j} M_{k} = \sum_{\ell_1 \in i, \ell_2 \in j, \ell_3 \in k} h_{\ell_1 \ell_2 \ell_3}^2 b_{\ell_1 \ell_2 \ell_3},

    with :math:`h^2_{\ell_1 \ell_2 \ell_3} = \frac{(2\ell_1 + 1)(2\ell_2 + 1)(2\ell_3 + 1)}{4\pi}
    \begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ 0 & 0 & 0 \end{pmatrix}^2`
    and :math:`b` the reduced bispectrum. Dividing by :math:`N_{ijk} = \sum h^2` (the number of triangles,
    stored as :attr:`nmodes`) hence estimates an :math:`h^2`-weighted band average of :math:`b`.

    Only band triplets :math:`i \leq j \leq k` with :math:`N_{ijk} > 0` are kept, which enforces the
    triangle and parity (:math:`\ell_1 + \ell_2 + \ell_3` even) conditions.

    Parameters
    ----------
    attrs : AngularAttrs
        Angular attributes (``ellmax``, and ``nside`` for the filtered maps).
    edges : array-like or dict, optional
        :math:`\ell`-band edges, see :class:`BinAngular2Spectrum`.

    References
    ----------
    https://arxiv.org/abs/0912.5516 (binned bispectrum estimator)
    """
    edges: staticarray = None
    xavg: staticarray = None
    nmodes: staticarray = None
    ibands: staticarray = None
    band_edges: staticarray = None
    wbin: staticarray = None
    attrs: AngularAttrs = None

    def __init__(self, attrs: AngularAttrs, edges: staticarray | dict | None=None):
        if not isinstance(attrs, AngularAttrs):
            attrs = attrs.attrs
        # the estimator is intrinsically pixel-based (it integrates a product of band-filtered maps),
        # so fail here rather than at the first transform
        assert attrs.nside is not None, 'provide attrs.nside: the bispectrum estimator needs band-filtered maps'
        bin2 = _make_edges_angular(attrs, edges)
        band_edges, wbin = np.asarray(bin2['edges']), np.asarray(bin2['wbin'])
        nbands = len(band_edges)
        ells = np.arange(attrs.ellmax + 1)
        # h^2, computed per band block to avoid materializing the full (ellmax + 1)^3 array
        ibands, nmodes, xavg = [], [], []
        for i in range(nbands):
            for j in range(i, nbands):
                for k in range(j, nbands):
                    l1, l2, l3 = (ells[wbin[idx]].reshape((1,) * axis + (-1,) + (1,) * (2 - axis))
                                  for axis, idx in enumerate((i, j, k)))
                    h2 = ((2. * l1 + 1.) * (2. * l2 + 1.) * (2. * l3 + 1.) / (4. * np.pi)
                          * _compute_wigner3j000_sq(l1, l2, l3))
                    total = h2.sum()
                    if total <= 0.: continue
                    ibands.append((i, j, k))
                    nmodes.append(total)
                    xavg.append([np.sum(h2 * l) / total for l in (l1, l2, l3)])
        assert ibands, 'no valid band triplet'
        ibands = np.array(ibands)
        self.__dict__.update(edges=staticarray(band_edges[ibands]), xavg=staticarray(np.array(xavg)),
                             nmodes=staticarray(np.array(nmodes)), ibands=staticarray(ibands),
                             band_edges=staticarray(band_edges), wbin=staticarray(wbin), attrs=attrs)

    @property
    def nbands(self):
        """Number of :math:`\\ell`-bands."""
        return len(self.band_edges)

    def __call__(self, maps: jax.Array):
        r"""
        Given the band-filtered maps, of shape ``(nbands, npix)``, return
        :math:`\int d\Omega\, M_i M_j M_k / N_{ijk}` for each band triplet.
        """
        pixarea = 4. * np.pi / maps.shape[-1]
        i, j, k = (np.asarray(self.ibands)[:, idim] for idim in range(3))
        num = jnp.sum(maps[i] * maps[j] * maps[k], axis=-1) * pixarea
        return num / jnp.asarray(self.nmodes, dtype=num.dtype)

    def filter(self, alm: AlmField, nside: int=None, backend: str=None) -> jax.Array:
        """Return the band-filtered maps, of shape ``(nbands, npix)``."""
        wbin = jnp.asarray(self.wbin, dtype=alm.value.real.dtype)

        def one(w):
            return alm2map(alm.clone(value=alm.value * w[:, None]), nside=nside, backend=backend).value

        return jnp.stack([one(w) for w in wbin])


def compute_angular3_spectrum(*fields, bin: BinAngular3Spectrum=None, method: str=None,
                              backend: str=None, batch_size: int=None) -> Angular3Spectrum:
    r"""
    Compute the angular bispectrum :math:`b_{\ell_1 \ell_2 \ell_3}`, binned in :math:`\ell`-bands,
    with the binned (filtered-map) estimator, see :class:`BinAngular3Spectrum`.

    Parameters
    ----------
    fields : ParticleField, FKPField, PixelField or AlmField
        Input field(s): one (auto-bispectrum) or three.
        Particle positions are interpreted as directions (observer at origin).
    bin : BinAngular3Spectrum
        Binning operator.
    method : str, optional
        'direct', 'healpix', or ``None`` (default) to pick from ``attrs.nside``, see :func:`to_alm`.
    backend : str, optional
        Backend for the harmonic transforms, see :meth:`PixelField.to_alm`.
    batch_size : int, optional
        Number of particles processed at once.

    Returns
    -------
    spectrum : Angular3Spectrum
        Raw (unnormalized) bispectrum; apply normalization and shot noise with
        ``spectrum.clone(norm=..., num_shotnoise=...)``, see
        :func:`compute_fkp_angular3_normalization` and :func:`compute_fkp_angular3_shotnoise`.
    """
    fields = _make_input_tuple(*fields)
    assert len(fields) in (1, 3), 'provide one (auto) or three fields'
    alms = [to_alm(field, attrs=bin.attrs, method=method, backend=backend, batch_size=batch_size) for field in fields]
    if len(alms) == 1: alms = alms * 3
    maps = [bin.filter(alm, nside=bin.attrs.nside, backend=backend) for alm in alms]
    if len(fields) == 1:
        num = bin(maps[0])
    else:
        # symmetrize over the 3! assignments of the fields to the (sorted) band triplet
        pixarea = 4. * np.pi / maps[0].shape[-1]
        i, j, k = (np.asarray(bin.ibands)[:, idim] for idim in range(3))
        num = 0.
        for perm in itertools.permutations(range(3)):
            num = num + jnp.sum(maps[perm[0]][i] * maps[perm[1]][j] * maps[perm[2]][k], axis=-1)
        num = num * pixarea / 6. / jnp.asarray(bin.nmodes)
    return Angular3Spectrum(ell=np.asarray(bin.xavg), ell_edges=np.asarray(bin.edges), num_raw=num,
                            nmodes=np.asarray(bin.nmodes), norm=jnp.ones_like(num), num_shotnoise=jnp.zeros_like(num))


def compute_fkp_angular3_normalization(*fkps: FKPField, bin: BinAngular3Spectrum=None, nside: int=None,
                                       split=None, fields: tuple=None):
    r"""
    Compute the FKP normalization :math:`\int d\Omega\, \bar{n}_1 \bar{n}_2 \bar{n}_3 / (4\pi)`
    for the angular bispectrum, see :func:`compute_fkp_angular2_normalization` for the conventions.

    With this normalization, a uniform full-sky Poisson sample of mean density :math:`\bar{n}`
    has shot noise :math:`1/\bar{n}^2` (plus the :math:`C_\ell` terms), as in 3D.

    Parameters
    ----------
    fkps : FKPField or ParticleField
        FKP fields or particles (one, or three).
    bin : BinAngular3Spectrum, optional
        Binning operator, providing the default ``nside``.
    nside : int, optional
        Healpix resolution for the :math:`\bar{n}` estimate.
    split : int, list, optional
        Random seed for splitting the randoms into 3 disjoint samples, whose product is then free of
        the Poisson self-terms. If ``None``, the same randoms are used for the three legs.
    fields : tuple, optional
        Field identifiers; pass e.g. [0, 0, 1] if the first two fields share the same positions.

    Returns
    -------
    norm : float
    """
    if nside is None:
        nside = getattr(bin.attrs, 'nside', None) if bin is not None else None
    if nside is None:
        nside = 64
    attrs = AngularAttrs(ellmax=0, nside=nside)

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    fkps, fields = _format_meshes3(*fkps, fields=fields)
    alpha = prod(map(lambda fkp: fkp.data.sum() / fkp.randoms.sum() if isinstance(fkp, FKPField) else 1., fkps))
    if split is not None:
        randoms = list(split_particles(*[get_randoms(fkp) for fkp in fkps], seed=split, fields=fields))
        alpha *= prod(get_randoms(fkp).sum() / randoms.sum() for fkp, randoms in zip(fkps, randoms, strict=True))
    else:
        randoms = [get_randoms(fkp) for fkp in fkps]
    return alpha * compute_angular_normalization(*randoms, attrs=attrs)


def compute_fkp_angular3_shotnoise(*fkps: FKPField, bin: BinAngular3Spectrum=None, method: str=None,
                                   backend: str=None, batch_size: int=None, fields: tuple=None):
    r"""
    Compute the Poisson shot noise of the angular bispectrum, in the same (raw) units as
    :attr:`Angular3Spectrum.num_raw`, entirely from the catalogs.

    Decomposing the triple sum over particles by coincidences, the two-point term is
    :math:`\int d\Omega\, A_{ij} M_k` with :math:`A_{ij}(\hat{n}) = \sum_p w_p^2 K_i K_j`. Expanding
    the kernel product in Legendre polynomials, :math:`K_i K_j = \sum_L c^{ij}_L \frac{2L + 1}{4\pi} P_L`
    with :math:`c^{ij}_L = \sum_{\ell_1 \in i, \ell_2 \in j} h^2_{\ell_1 \ell_2 L} / (2L + 1)`, so it
    collapses onto the cross pseudo-spectrum :math:`X_L = \sum_m X_{\ell m} a_{\ell m}^*` between the
    :math:`w^2`-weighted field and the field itself. Removing the self-pairs it shares with the
    three-point term,

    .. math::

        \mathrm{num}^\mathrm{shot}_{ijk} = \frac{1}{N_{ijk}} \sum_{3\ \mathrm{pairings}}
        \sum_{\ell_1 \ell_2 L} h^2_{\ell_1 \ell_2 L} \frac{X_L}{2L + 1} - \frac{2 S_3}{4\pi},

    with :math:`S_3 = \sum_D w^3 - \alpha^3 \sum_R w^3`.

    Nothing needs to be supplied: the :math:`C_\ell`-like term is measured, not modelled. For a uniform
    Poisson sample :math:`X_L / (2L + 1) \to S_3 / 4\pi` and this collapses to the flat
    :math:`S_3 / 4\pi`; normalized by :func:`compute_fkp_angular3_normalization`, it then reduces to
    the familiar :math:`[C_{\ell_1} + C_{\ell_2} + C_{\ell_3}] / \bar{n} + 1 / \bar{n}^2`.

    Parameters
    ----------
    fkps : FKPField or ParticleField
        FKP fields or particles.
    bin : BinAngular3Spectrum
        Binning operator.
    method : str, optional
        'direct', 'healpix', or ``None`` to pick from ``attrs.nside``, see :func:`to_alm`.
    backend : str, optional
        Backend for the harmonic transforms, see :meth:`PixelField.to_alm`.
    batch_size : int, optional
        Number of particles processed at once.
    fields : tuple, optional
        Field identifiers; the shot noise vanishes unless the three fields share the same points.

    Returns
    -------
    num_shotnoise : jax.Array
        Array of shape ``(nbins,)``.
    """
    fkps, fields = _format_meshes3(*fkps, fields=fields)
    num = jnp.zeros(len(bin.nmodes))
    if not all(field == fields[0] for field in fields):  # distinct fields never share points
        return num
    fkp = fkps[0]

    if isinstance(fkp, FKPField):
        alpha = fkp.data.sum() / fkp.randoms.sum()
        sum3 = jnp.sum(fkp.data.weights**3) - alpha**3 * jnp.sum(fkp.randoms.weights**3)
        # the w^2-weighted field is a *sum*: both (+1)^2 and (-alpha)^2 are positive
        field_w2 = (fkp.data.clone(weights=fkp.data.weights**2)
                    + fkp.randoms.clone(weights=alpha**2 * fkp.randoms.weights**2))
    else:
        sum3 = jnp.sum(fkp.weights**3)
        field_w2 = fkp.clone(weights=fkp.weights**2)

    kw = dict(attrs=bin.attrs, method=method, backend=backend, batch_size=batch_size)
    xl = _compute_cross_power(to_alm(field_w2, **kw), to_alm(fkp, **kw))
    xl = xl / (2. * np.arange(bin.attrs.ellmax + 1) + 1.)
    return _average_cl_over_bands(bin, xl) - 2. * sum3 / (4. * np.pi)


def _get_cl_array(spectrum, ellmax: int):
    """Return :math:`C_\\ell` for ``ell = 0, ..., ellmax``, from an observable, a callable or an array."""
    ells = np.arange(ellmax + 1)
    if callable(spectrum):
        return jnp.asarray(spectrum(ells))
    if hasattr(spectrum, 'coords'):  # Angular2Spectrum
        x, y = np.asarray(spectrum.coords('ell')), jnp.asarray(spectrum.value())
        return jnp.interp(jnp.asarray(ells, dtype=y.dtype), jnp.asarray(x, dtype=y.dtype), y)
    spectrum = jnp.asarray(spectrum)
    assert spectrum.shape[0] == ellmax + 1, 'provide C_ell for ell = 0 to ellmax'
    return spectrum


def _average_cl_over_bands(bin: BinAngular3Spectrum, cl: jax.Array):
    r"""
    Return :math:`\sum_\mathrm{3\ pairings} \bar{C}`, with :math:`\bar{C}` the :math:`h^2`-weighted
    average of :math:`C_{\ell}` over each leg of the triplet, of shape ``(nbins,)``.
    """
    ells = np.arange(bin.attrs.ellmax + 1)
    wbin = np.asarray(bin.wbin)
    out = []
    for i, j, k in np.asarray(bin.ibands):
        l1, l2, l3 = (ells[wbin[idx]].reshape((1,) * axis + (-1,) + (1,) * (2 - axis))
                      for axis, idx in enumerate((i, j, k)))
        h2 = ((2. * l1 + 1.) * (2. * l2 + 1.) * (2. * l3 + 1.) / (4. * np.pi)
              * _compute_wigner3j000_sq(l1, l2, l3))
        total = h2.sum()
        out.append(sum(jnp.sum(jnp.asarray(h2) * cl[np.broadcast_to(l, h2.shape)]) for l in (l1, l2, l3)) / total)
    return jnp.array(out)


def _compute_wigner3j(ell1, ell2, ell3, m1, m2, m3):
    r"""
    Wigner 3j symbol :math:`\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ m_1 & m_2 & m_3 \end{pmatrix}`
    from the Racah formula, vectorized (numpy).

    Note
    ----
    The sum over :math:`k` alternates in sign, so this loses precision for large :math:`\ell`
    (fine in double precision up to :math:`\ell \sim 30`, which is the regime where the bispectrum
    window matrix below is affordable anyway).
    """
    from scipy.special import gammaln
    ell1, ell2, ell3, m1, m2, m3 = np.broadcast_arrays(*(np.asarray(x) for x in (ell1, ell2, ell3, m1, m2, m3)))
    valid = ((m1 + m2 + m3 == 0) & (np.abs(m1) <= ell1) & (np.abs(m2) <= ell2) & (np.abs(m3) <= ell3)
             & (ell3 >= np.abs(ell1 - ell2)) & (ell3 <= ell1 + ell2))
    lgf = lambda n: gammaln(np.where(n >= 0, n, 0) + 1)
    logpref = 0.5 * (lgf(ell1 + ell2 - ell3) + lgf(ell1 - ell2 + ell3) + lgf(-ell1 + ell2 + ell3) - lgf(ell1 + ell2 + ell3 + 1)
                     + lgf(ell1 + m1) + lgf(ell1 - m1) + lgf(ell2 + m2) + lgf(ell2 - m2) + lgf(ell3 + m3) + lgf(ell3 - m3))
    kmax = int(max(1, np.max(np.where(valid, ell1 + ell2 - ell3, 0))))
    ks = np.arange(kmax + 1).reshape((-1,) + (1,) * ell1.ndim)
    args = [ks, ell1 + ell2 - ell3 - ks, ell1 - m1 - ks, ell2 + m2 - ks, ell3 - ell2 + m1 + ks, ell3 - ell1 - m2 + ks]
    kvalid = functools.reduce(operator.and_, (arg >= 0 for arg in args))
    terms = np.where(kvalid, (-1.)**ks * np.exp(logpref - sum(lgf(arg) for arg in args)), 0.)
    return np.where(valid, (-1.)**(ell1 - ell2 - m3) * terms.sum(axis=0), 0.)


def _compute_wigner3j000_sq(ell1, ell2, ell3):
    r"""Squared Wigner 3j symbol :math:`\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ 0 & 0 & 0 \end{pmatrix}^2` (vectorized, numpy)."""
    from scipy.special import gammaln
    ell1, ell2, ell3 = np.broadcast_arrays(ell1, ell2, ell3)
    J = ell1 + ell2 + ell3
    g = J // 2
    valid = (J % 2 == 0) & (ell3 >= np.abs(ell1 - ell2)) & (ell3 <= ell1 + ell2)
    lgf = lambda n: gammaln(np.where(n >= 0, n, 0) + 1)
    logres = (lgf(J - 2 * ell1) + lgf(J - 2 * ell2) + lgf(J - 2 * ell3) - lgf(J + 1)
              + 2 * (lgf(g) - lgf(g - ell1) - lgf(g - ell2) - lgf(g - ell3)))
    return np.where(valid, np.exp(logres), 0.)


def compute_angular2_spectrum_window(*fields, edgesin: staticarray | dict=None, bin: BinAngular2Spectrum=None,
                                     norm=None, method: str=None, backend: str=None, batch_size: int=None) -> WindowMatrix:
    r"""
    Compute the mode-coupling (window) matrix of the pseudo-:math:`C_\ell` estimator:
    :math:`\langle \tilde{C}_\ell \rangle = \sum_{\ell'} M_{\ell \ell'} C_{\ell'}` with
    :math:`M_{\ell \ell'} = \frac{2\ell' + 1}{4\pi} \sum_L (2L + 1) W_L
    \begin{pmatrix} \ell & \ell' & L \\ 0 & 0 & 0 \end{pmatrix}^2`,
    where :math:`W_L` is the pseudo-spectrum of the mask :math:`\bar{n}(\Omega)`.
    The returned matrix maps the theory :math:`C_{\ell'}` (piecewise constant on ``edgesin``)
    to the normalized observable ``spectrum.value``.

    Parameters
    ----------
    fields : ParticleField, FKPField, PixelField or AlmField
        Mask field(s) :math:`\bar{n}(\Omega)` (one for auto, two for cross).
        For :class:`FKPField`, :math:`\alpha \times` randoms is used.
        If not an :class:`AlmField`, the mask harmonic coefficients are computed up to
        ``bin.attrs.ellmax + ellmax(edgesin)``.
    edgesin : array-like or dict
        Theory :math:`\ell`-bin edges (see :class:`BinAngular2Spectrum`).
    bin : BinAngular2Spectrum
        Binning operator for the observable side.
    norm : float, optional
        Normalization of the observable; defaults to the FKP normalization computed
        from the same mask harmonic coefficients, :math:`\sum_L (2L + 1) W_L / (4\pi)`
        (consistent with :func:`compute_fkp_angular2_normalization`).
    method : str, optional
        'direct', 'healpix', or ``None`` (default) to pick from ``attrs.nside``, see :func:`to_alm`.
    batch_size : int, optional
        Number of particles processed at once.

    Returns
    -------
    wmat : WindowMatrix
    """
    fields = _make_input_tuple(*fields)
    assert 1 <= len(fields) <= 2
    if isinstance(edgesin, dict):
        ellmax_in = int(edgesin.get('max', bin.attrs.ellmax + 1)) - 1
    else:
        ellmax_in = int(np.max(np.asarray(edgesin))) - 1
    binin = BinAngular2Spectrum(bin.attrs.clone(ellmax=ellmax_in), edges=edgesin)
    ellmax_mask = bin.attrs.ellmax + binin.attrs.ellmax
    mask_attrs = bin.attrs.clone(ellmax=ellmax_mask)

    def get_alm(field):
        if isinstance(field, FKPField):
            alpha = field.data.sum() / field.randoms.sum()
            return alpha * to_alm(field.randoms, attrs=mask_attrs, method=method, backend=backend, batch_size=batch_size)
        if isinstance(field, AlmField):
            return field
        return to_alm(field, attrs=mask_attrs, method=method, backend=backend, batch_size=batch_size)

    alms = [get_alm(field) for field in fields]
    if len(alms) == 1: alms = alms * 2
    ellmax_mask = min(alm.attrs.ellmax for alm in alms)
    alms = [to_alm(alm, attrs=AngularAttrs(ellmax=ellmax_mask)) for alm in alms]
    ellsw = np.arange(ellmax_mask + 1)
    W = np.asarray(_compute_cross_power(*alms)) / (2. * ellsw + 1.)  # mask pseudo-spectrum W_L

    if norm is None:
        norm = np.sum((2. * ellsw + 1.) * W) / (4. * np.pi)

    ells = np.arange(bin.attrs.ellmax + 1)[:, None]
    ellsin = np.arange(binin.attrs.ellmax + 1)[None, :]
    M = np.zeros((ells.size, ellsin.size))
    for L in range(ellmax_mask + 1):
        M += (2. * L + 1.) * W[L] * _compute_wigner3j000_sq(ells, ellsin, L)
    M *= (2. * ellsin + 1.) / (4. * np.pi)
    # bin: observable rows with (2l+1)/nmodes weights; theory columns summed (C piecewise constant)
    wout = np.asarray(bin.wbin) * (2. * ells.T + 1.) / np.asarray(bin.nmodes)[:, None]
    wmat = wout @ M @ np.asarray(binin.wbin).T / norm

    zeros = np.zeros_like(bin.xavg)
    observable = Angular2Spectrum(ell=bin.xavg, ell_edges=bin.edges, num_raw=zeros, nmodes=np.asarray(bin.nmodes))
    theory = Angular2Spectrum(ell=binin.xavg, ell_edges=binin.edges, num_raw=np.zeros_like(binin.xavg), nmodes=np.asarray(binin.nmodes))
    return WindowMatrix(observable=observable, theory=theory, value=wmat)


def _gather_phi(phi: np.ndarray, m):
    r"""
    Index :math:`\Phi_{\ell m}`, stored for :math:`m \geq 0` only, at (possibly negative) ``m``,
    using :math:`\Phi_{\ell,-m} = (-1)^m \Phi_{\ell m}^*` (the mask and band kernel being real).
    """
    m = np.asarray(m)
    out = phi[np.abs(m)]
    if not np.any(m < 0):
        return out
    negative = (m < 0).reshape(m.shape + (1,) * (out.ndim - m.ndim))
    return np.where(negative, (-1.)**np.abs(m).reshape(negative.shape) * np.conj(out), out)


def _band_triplets(wbin: np.ndarray, ellmax: int, group: tuple):
    r"""
    Enumerate the canonical *ordered* band triplets under the permutation ``group`` (the stabilizer of
    the field triple), returning ``(ibands, orbits, nmodes, xavg)``.

    ``B(x_0, x_1, x_2)`` is invariant under permuting its arguments by :math:`\sigma \in` ``group``,
    so the bins :math:`(b_0, b_1, b_2)` and :math:`(b_{\sigma_0}, b_{\sigma_1}, b_{\sigma_2})` carry the
    same value and are merged; ``orbits`` lists the distinct ordered members of each merged bin.
    """
    ells = np.arange(ellmax + 1)
    nbands = len(wbin)
    ibands, orbits, nmodes, xavg = [], [], [], []
    for bands in itertools.product(range(nbands), repeat=3):
        orbit = sorted({tuple(bands[sigma[axis]] for axis in range(3)) for sigma in group})
        if orbit[0] != bands: continue  # keep one representative per orbit
        ls = [ells[wbin[idx]].reshape((1,) * axis + (-1,) + (1,) * (2 - axis)) for axis, idx in enumerate(bands)]
        h2 = ((2. * ls[0] + 1.) * (2. * ls[1] + 1.) * (2. * ls[2] + 1.) / (4. * np.pi)
              * _compute_wigner3j000_sq(*ls))
        total = h2.sum()
        if total <= 0.: continue
        ibands.append(bands)
        orbits.append(orbit)
        nmodes.append(total)
        xavg.append([np.sum(h2 * l) / total for l in ls])
    return np.array(ibands), orbits, np.array(nmodes), np.array(xavg)


def _get_mask_map(field, attrs: AngularAttrs, backend: str=None) -> np.ndarray:
    """Return the mask (mean density) :math:`\\bar{n}` as a numpy map of resolution ``attrs.nside``."""
    if isinstance(field, PixelField):
        return np.asarray(field.value)
    if isinstance(field, FKPField):
        alpha = field.data.sum() / field.randoms.sum()
        return alpha * np.asarray(to_pixel(field.randoms, attrs=attrs, backend=backend).value)
    return np.asarray(to_pixel(field, attrs=attrs, backend=backend).value)


def compute_angular3_spectrum_window(*masks, edgesin: staticarray | dict=None, bin: BinAngular3Spectrum=None,
                                     ellmaxin: int=None, norm=None, backend: str=None, fields: tuple=None,
                                     dtype=np.complex128) -> WindowMatrix:
    r"""
    Compute the window (mode-coupling) matrix of the binned angular bispectrum, mapping the theory
    reduced bispectrum :math:`b_{\ell_1 \ell_2 \ell_3}` (piecewise constant on band triplets) to the
    normalized observable :attr:`Angular3Spectrum.value`.

    The masked estimator is linear in :math:`b`:

    .. math::

        \langle \mathrm{num}_{ijk} \rangle = \sum_{\ell_1 \ell_2 \ell_3} b_{\ell_1 \ell_2 \ell_3}
        \int d\Omega \sum_{m_1 m_2 m_3} G^{m_1 m_2 m_3}_{\ell_1 \ell_2 \ell_3}
        \Phi^i_{\ell_1 m_1} \Phi^j_{\ell_2 m_2} \Phi^k_{\ell_3 m_3},

    with the mask-weighted, band-filtered harmonics
    :math:`\Phi^i_{\ell m}(\hat{n}) = \int d\Omega' K_i(\hat{n} \cdot \hat{n}') \bar{n}(\hat{n}') Y_{\ell m}(\hat{n}')`
    and :math:`G` the Gaunt coefficient. For a full-sky uniform mask this reduces to
    :math:`\delta_{ijk, i'j'k'} N_{ijk}`, i.e. the identity once normalized.

    With three (possibly different) masks, the estimator symmetrizes over the :math:`3!` assignments
    of the fields to the legs, so

    .. math::

        W_{ijk, (b_0 b_1 b_2)} = \frac{1}{6} \sum_\pi \sum_{\ell_1 \in b_{\pi_0}, \ell_2 \in b_{\pi_1}, \ell_3 \in b_{\pi_2}}
        \int d\Omega \sum_{m} G\, \Phi^{\pi_0, i}_{\ell_1 m_1} \Phi^{\pi_1, j}_{\ell_2 m_2} \Phi^{\pi_2, k}_{\ell_3 m_3}.

    The cross-bispectrum :math:`B^{ABC}(\ell_A, \ell_B, \ell_C)` is invariant only under permuting fields
    *and* multipoles together, so the theory side is indexed by **ordered** band triplets, merged under
    the stabilizer of the field triple: all :math:`3!` orderings for an auto-bispectrum (giving the sorted
    triplets), none for three distinct fields.

    Warning
    -------
    This is exact, but costs :math:`\mathcal{O}(\ell_\mathrm{max}^2)` spherical harmonic transforms
    per distinct mask, plus :math:`\mathcal{O}(N_\mathrm{obs} N_\mathrm{theory} \ell^2 N_\mathrm{pix})`
    for the Gaunt contraction, so it is only affordable at modest :math:`\ell_\mathrm{max}`.
    The peak memory is that of :math:`\Phi`, ``itemsize * nbands * npix * (ellmaxin + 1)(ellmaxin + 2) / 2``
    per distinct mask (only :math:`m \geq 0` is stored, and only the :math:`\ell` spanned by a theory
    band) -- quadratic in :math:`\ell_\mathrm{max}`, and quartic once ``nside`` is made to track
    :math:`\ell_\mathrm{max}` as the quadrature requires. Pass ``dtype='complex64'`` to halve it.

    Parameters
    ----------
    masks : ParticleField, FKPField or PixelField
        Mask(s) :math:`\bar{n}(\Omega)`, one (auto) or three; for :class:`FKPField`,
        :math:`\alpha \times` randoms is used. Pass ``None`` to repeat the previous mask.
    edgesin : array-like or dict, optional
        Theory :math:`\ell`-band edges; defaults to the observable's bands.
    bin : BinAngular3Spectrum
        Binning operator for the observable side.
    ellmaxin : int, optional
        Maximum theory multipole; defaults to ``bin.attrs.ellmax``.
    norm : float, optional
        Normalization of the observable; defaults to :math:`\int \bar{n}_1 \bar{n}_2 \bar{n}_3 d\Omega / (4\pi)`,
        consistent with :func:`compute_fkp_angular3_normalization`.
    backend : str, optional
        Backend used to paint the mask, see :func:`to_pixel`.
    fields : tuple, optional
        Field identifiers; pass e.g. [0, 0, 1] if the first two legs share the same mask, which merges
        the theory bins accordingly.
    dtype : optional
        Precision of the stored :math:`\Phi`; 'complex64' halves the peak memory, at a cost well below
        the healpix quadrature error of the estimator itself. The contraction always accumulates in
        double precision.

    Returns
    -------
    wmat : WindowMatrix
    """
    import healpy as hp
    try:
        from scipy.special import sph_harm_y
        Ylm = lambda ell, m, theta, phi: sph_harm_y(ell, m, theta, phi)
    except ImportError:
        from scipy.special import sph_harm
        Ylm = lambda ell, m, theta, phi: sph_harm(m, ell, phi, theta)

    masks, ids = _format_meshes3(*masks, fields=fields)
    attrs = bin.attrs
    npix, pixarea = attrs.npix, attrs.pixarea
    cdtype = np.dtype(dtype)
    uids = list(dict.fromkeys(ids))
    nbars = {uid: _get_mask_map(masks[ids.index(uid)], attrs=attrs, backend=backend) for uid in uids}
    if norm is None:
        norm = np.sum(prod(nbars[uid] for uid in ids)) * pixarea / (4. * np.pi)

    if ellmaxin is None: ellmaxin = attrs.ellmax
    if edgesin is None: edgesin = np.append(np.asarray(bin.band_edges)[:, 0], np.asarray(bin.band_edges)[-1, 1])
    wbinin = np.asarray(_make_edges_angular(attrs.clone(ellmax=ellmaxin), edgesin)['wbin'])
    # stabilizer of the field triple: the permutations of the legs leaving the masks unchanged
    group = tuple(sigma for sigma in itertools.permutations(range(3))
                  if tuple(ids[sigma[axis]] for axis in range(3)) == tuple(ids))
    ibandsin, orbitsin, nmodesin, xavgin = _band_triplets(wbinin, ellmaxin, group)

    ellmax, wbin = attrs.ellmax, np.asarray(bin.wbin)
    nbands = len(wbin)
    theta, phi = hp.pix2ang(attrs.nside, np.arange(npix))

    def band_filter(map):
        """Band-filter a real map: map -> alm -> keep each band -> map, for all bands."""
        alm = hp.map2alm(map, lmax=ellmax, iter=0, use_pixel_weights=False)
        ls = hp.Alm.getlm(ellmax)[0]
        return np.stack([hp.alm2map(np.where(w[ls], alm, 0.), attrs.nside, lmax=ellmax, pol=False) for w in wbin])

    # Phi[uid][ell][m, iband, ipix], one set of transforms per distinct mask.
    # Only m >= 0 is stored: the mask and the band kernel are real, so
    # Phi_{ell,-m} = (-1)^m conj(Phi_{ell,m}), applied on the fly by _gather_phi.
    # Only the ell actually spanned by a theory band are built.
    ellsin_needed = np.flatnonzero(np.any(wbinin, axis=0))
    Phi = {}
    for uid in uids:
        nbar, phi_uid = nbars[uid], {}
        for ell in ellsin_needed:
            ell = int(ell)
            phi_ell = np.zeros((ell + 1, nbands, npix), dtype=cdtype)
            for m in range(ell + 1):
                ylm = nbar * Ylm(ell, m, theta, phi)
                phi_ell[m] = band_filter(ylm.real) + 1j * band_filter(ylm.imag)
            phi_uid[ell] = phi_ell
        Phi[uid] = phi_uid

    obs = np.asarray(bin.ibands)
    ellsin = np.arange(ellmaxin + 1)

    @lru_cache(maxsize=None)
    def contract(iobs, legs, ell1, ell2, ell3):
        """Gaunt contraction for one observed triplet, one theory ell-triplet and one mask assignment."""
        i, j, k = obs[iobs]
        h = np.sqrt((2. * ell1 + 1.) * (2. * ell2 + 1.) * (2. * ell3 + 1.) / (4. * np.pi)) \
            * _compute_wigner3j(ell1, ell2, ell3, 0, 0, 0)
        if abs(h) < 1e-14: return 0.
        A = Phi[legs[0]][ell1][:, i]
        B = Phi[legs[1]][ell2][:, j]
        C = Phi[legs[2]][ell3][:, k]
        ms2 = np.arange(-ell2, ell2 + 1)
        total = 0.
        for m1 in range(-ell1, ell1 + 1):
            m3 = -(m1 + ms2)
            keep = np.abs(m3) <= ell3
            if not keep.any(): continue
            g = _compute_wigner3j(ell1, ell2, ell3, m1, ms2[keep], m3[keep])
            # accumulate in double precision even if Phi is stored single
            term = _gather_phi(A, m1) * _gather_phi(B, ms2[keep]) * _gather_phi(C, m3[keep])
            total = total + np.sum(g[:, None] * term.astype(np.complex128))
        return h * total * pixarea

    perms = list(itertools.permutations(range(3)))
    wmat = np.zeros((len(obs), len(nmodesin)))
    for iobs in range(len(obs)):
        for itheory, orbit in enumerate(orbitsin):
            value = 0.
            for bands in orbit:  # theory bins merged by the stabilizer carry the same value
                for perm in perms:
                    legs = tuple(ids[axis] for axis in perm)
                    for ell1 in ellsin[wbinin[bands[perm[0]]]]:
                        for ell2 in ellsin[wbinin[bands[perm[1]]]]:
                            for ell3 in ellsin[wbinin[bands[perm[2]]]]:
                                value += contract(iobs, legs, int(ell1), int(ell2), int(ell3))
            wmat[iobs, itheory] = np.real(value) / 6.
    wmat = wmat / np.asarray(bin.nmodes)[:, None] / norm

    zeros = np.zeros(len(obs))
    observable = Angular3Spectrum(ell=np.asarray(bin.xavg), ell_edges=np.asarray(bin.edges),
                                  num_raw=zeros, nmodes=np.asarray(bin.nmodes))
    band_edges = np.asarray(_make_edges_angular(attrs.clone(ellmax=ellmaxin), edgesin)['edges'])
    theory = Angular3Spectrum(ell=xavgin, ell_edges=band_edges[ibandsin],
                              num_raw=np.zeros(len(nmodesin)), nmodes=nmodesin)
    return WindowMatrix(observable=observable, theory=theory, value=wmat)


def compute_angular2_spectrum_mean(window: WindowMatrix, theory) -> Angular2Spectrum:
    """
    Mean (window-convolved) angular power spectrum given ``theory``,
    a callable :math:`C(\\ell)` or an array over ``window.theory`` :math:`\\ell`.
    """
    ellin = np.asarray(window.theory.ell)
    theory = theory(ellin) if callable(theory) else jnp.asarray(theory)
    value = jnp.asarray(window.value()) @ theory
    return window.observable.clone(value=value)
