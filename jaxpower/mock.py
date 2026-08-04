import itertools
from collections.abc import Callable

import numpy as np
import jax
from jax import random
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from .mesh import RealMeshField, ParticleField, MeshAttrs, exchange_particles, create_sharded_random, _process_seed
from .mesh2 import _get_los_vector
from .angular import AngularAttrs, AlmField, PixelField, alm2map, _get_cl_array, _make_edges_angular
from .utils import get_legendre, get_Ylm
from .types import ObservableTree


def generate_gaussian_mesh(mattrs: MeshAttrs, power: Callable=lambda kvec: 1., seed: int=42,
                           unitary_amplitude: bool=False):

    r"""
    Generate a Gaussian random field mesh with a given power spectrum.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes (box size, mesh size, etc.).
    power : Callable
        Function returning the power spectrum as a function of :math:`k`-vector.
    seed : int, tuple, optional
        Random seed for mesh generation.
        Provide (seed, ids) to ensure reproducibility when changing the number of devices; see :func:`create_sharded_random` for details.
    unitary_amplitude : bool, optional
        If ``True``, normalize the amplitude to be unitary.

    Returns
    -------
    mesh : RealMeshField
        Generated mesh field with the specified power spectrum.
    """
    mesh = mattrs.create(kind='real', fill=create_sharded_random(random.normal, seed, shape=mattrs.meshsize, out_specs=P(*mattrs.sharding_mesh.axis_names))).r2c()

    def kernel(value, kvec):
        ker = jnp.sqrt(power(kvec) / mesh.cellsize.prod())
        if unitary_amplitude:
            ker *= jnp.sqrt(mesh.meshsize.prod(dtype=float)) / jnp.abs(value)
        return value * ker

    mesh = mesh.apply(kernel, kind='wavenumber')
    return mesh.c2r()


def generate_spectrum2_mesh(mattrs: MeshAttrs, poles: ObservableTree | dict[Callable], seed: int=42, los: str='x', unitary_amplitude: bool=False, **kwargs):
    """
    Generate a Gaussian random field mesh with input power spectrum multipoles.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes (box size, mesh size, etc.).
    poles : dict or Mesh2SpectrumPoles or list
        Dictionary of multipole order to power spectrum function, or :class:`Mesh2SpectrumPoles`, or list of power spectra.
    seed : int, tuple, optional
        Random seed for mesh generation.
        Provide (seed, ids) to ensure reproducibility when changing the number of devices; see :func:`create_sharded_random` for details.
    los : str, optional
        Line-of-sight specification ('x', 'y', 'z', or 'local').
    unitary_amplitude : bool, optional
        If True, normalize the amplitude to be unitary.
    kwargs : dict
        Additional arguments for interpolation.

    Returns
    -------
    mesh : RealMeshField
        Generated mesh field with the specified multipole power spectra.
    """
    ells = (0, 2, 4)
    kin = None
    if isinstance(poles, ObservableTree):
        edges = next(iter(poles)).edges('k')
        kin = jnp.append(edges[..., 0], edges[-1, 1])
        poles = {ell: poles.get(ell).value() for ell in poles.ells}
    if isinstance(poles, list):
        poles = {ell: pole for ell, pole in zip(ells, poles)}
    ells = list(poles)

    key, ids = _process_seed(seed)
    assert key is not None, 'provide random key for mesh generation'

    def generate_normal(key):
        mesh = mattrs.create(kind='real', fill=create_sharded_random(random.normal, (key, ids), shape=mattrs.meshsize, out_specs=P(*mattrs.sharding_mesh.axis_names))).r2c()
        if unitary_amplitude:
            mesh *= jnp.sqrt(mattrs.meshsize.prod(dtype=float)) / jnp.abs(mesh.value)
        return mesh

    kvec = mattrs.kcoords(sparse=True)
    knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
    kshape = np.broadcast_shapes(*(kk.shape for kk in kvec))

    is_callable = all(callable(pole) for pole in poles.values())
    if not is_callable:
        from .utils import Interpolator1D
        interp = Interpolator1D(kin, knorm, edges=len(kin) == len(poles[0]) + 1, **kwargs)

    def get_theory(ell=None, pole=None):
        if pole is None:
            pole = poles[ell]
        if is_callable:
            return pole(knorm)
        else:
            return interp(pole)

    if los == 'local':

        @jax.checkpoint
        def get_meshs(keys):

            if is_callable:
                p0, p2, p4 = (get_theory(ell) / mattrs.cellsize.prod() for ell in ells)
            else:
                p0, p2, p4 = (poles[ell] / mattrs.cellsize.prod() for ell in ells)

            a11 = 35. / 18. * p4
            a00 = p0 - 1. / 5. * a11
            # Cholesky decomposition; clip to guard against round-off for rank-deficient
            # (e.g. exact Kaiser) inputs, which sit on the PSD boundary
            l00 = jnp.sqrt(jnp.maximum(a00, 0.))
            del a00

            a10 = 1. / 2. * p2 - 1. / 7. * a11
            l10 = jnp.where(l00 == 0., 0., a10 / l00)
            del a10

            def _interp(pole):
                if is_callable:
                    return pole.reshape(kshape)
                else:
                    return get_theory(pole=pole).reshape(kshape)

            # The mesh for ell = 0
            normal = generate_normal(keys[0])
            mesh = (normal * _interp(l00)).c2r()
            del l00
            mesh2 = normal * _interp(l10)
            del normal
            # The mesh for ell = 2
            mesh2 += generate_normal(keys[1]) * _interp(jnp.sqrt(jnp.maximum(a11 - l10**2, 0.)))
            del a11, l10
            return mesh, mesh2

        mesh, mesh2 = get_meshs(random.split(key))
        xvec = mesh.coords(sparse=True)
        ell = 2
        Ylms = [get_Ylm(ell, m, real=True) for m in range(-ell, ell + 1)]

        @jax.checkpoint
        def f(carry, im):
            carry += 4. * jnp.pi / (2 * ell + 1) * (mesh2 * jax.lax.switch(im, Ylms, *kvec)).c2r() * jax.lax.switch(im, Ylms, *xvec)
            return carry, im

        mesh = jax.lax.scan(f, init=mesh, xs=np.arange(len(Ylms)))[0]
        #mesh += 4. * jnp.pi / (2 * ell + 1) * sum((mesh2 * Ylm(*kvec)).c2r() * Ylm(*xvec) for Ylm in Ylms)  # total mesh, mesh0 + mesh2 * L2(mu)

        del mesh2
        return mesh

    else:
        vlos = _get_los_vector(los, ndim=mattrs.ndim)
        mesh = generate_normal(key)

        def kernel(value, kvec):
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)
            ker = sum(get_theory(ell) / mattrs.cellsize.prod() * get_legendre(ell)(mu) for ell in ells)
            ker = jnp.sqrt(ker)
            if unitary_amplitude:
                ker *= jnp.sqrt(mattrs.meshsize.prod(dtype=value.real.dtype)) / jnp.abs(value)
            return value * ker

        mesh = mesh.apply(kernel, kind='wavenumber').c2r()
        return mesh


#: Backward-compatible alias of :func:`generate_spectrum2_mesh`.
generate_anisotropic_gaussian_mesh = generate_spectrum2_mesh


def generate_spectrum2_alm(attrs: AngularAttrs, cl: Callable | np.ndarray=lambda ell: 1., seed: int=42,
                           unitary_amplitude: bool=False) -> AlmField:
    r"""
    Generate the harmonic coefficients of a Gaussian random field on the sphere with a given
    angular power spectrum, the analog of :func:`generate_gaussian_mesh` for :mod:`.angular`.

    The field is real, so only :math:`m \geq 0` is stored (:math:`a_{\ell,-m} = (-1)^m a_{\ell m}^*`),
    with :math:`a_{\ell 0}` real of variance :math:`C_\ell` and, for :math:`m > 0`, real and imaginary
    parts of variance :math:`C_\ell / 2` each, so that
    :math:`\langle a_{\ell m} a_{\ell' m'}^* \rangle = C_\ell \delta_{\ell \ell'} \delta_{m m'}`.
    This is the convention of :func:`compute_angular2_spectrum`, whose estimate
    :math:`\sum_m |a_{\ell m}|^2 / (2\ell + 1)` is then unbiased for :math:`C_\ell`.

    Parameters
    ----------
    attrs : AngularAttrs
        Angular attributes, providing ``ellmax`` (and ``nside``, carried over to the output for a
        subsequent :meth:`AlmField.to_pixel`).
    cl : Callable, array-like or Angular2Spectrum, optional
        Angular power spectrum: a function of :math:`\ell`, an array over ``ell = 0, ..., ellmax``,
        or a measured :class:`Angular2Spectrum` (interpolated).
    seed : int, tuple, optional
        Random seed.
    unitary_amplitude : bool, optional
        If ``True``, fix :math:`|a_{\ell m}|` to :math:`\sqrt{C_\ell}` and randomize only the phase,
        so that the measured :math:`C_\ell` has no scatter.

    Returns
    -------
    alm : AlmField

    Examples
    --------
    >>> attrs = AngularAttrs(ellmax=64, nside=64)
    >>> alm = generate_spectrum2_alm(attrs, cl=lambda ell: 1e-3 / (1. + (ell / 10.)**2), seed=42)
    >>> pixel = alm.to_pixel()  # healpix map, if nside is set
    """
    ellmax = attrs.ellmax
    cl = jnp.asarray(_get_cl_array(cl, ellmax=ellmax))
    key = _process_seed(seed)[0]
    assert key is not None, 'provide random key for mock generation'

    ells = np.arange(ellmax + 1)
    lower = ells[:, None] >= ells[None, :]  # m <= ell
    keys = random.split(key, 2)
    real = random.normal(keys[0], (ellmax + 1,) * 2, dtype=cl.dtype)
    imag = random.normal(keys[1], (ellmax + 1,) * 2, dtype=cl.dtype)

    if unitary_amplitude:
        # unit modulus, random phase; m = 0 is real, so only its sign is random
        norm = jnp.sqrt(real**2 + imag**2)
        real, imag = real / norm, imag / norm
        value = jnp.sqrt(cl)[:, None] * (real + 1j * imag)
        value = value.at[:, 0].set(jnp.sqrt(cl) * jnp.sign(real[:, 0]))
    else:
        value = jnp.sqrt(cl / 2.)[:, None] * (real + 1j * imag)
        value = value.at[:, 0].set(jnp.sqrt(cl) * real[:, 0])
    return AlmField(value=jnp.where(lower, value, 0.), attrs=attrs)


def _format_spectrum3_bins(spectrum3, nbands):
    """Return the target as a list of (sorted band triplet, amplitude), checking the band indices."""
    if not isinstance(spectrum3, dict):
        raise ValueError('spectrum3 must be a dict {(i, j, k): amplitude} of sorted band indices')
    toret = []
    for bands, amplitude in spectrum3.items():
        bands = tuple(int(band) for band in bands)
        assert len(bands) == 3, f'{bands} is not a band triplet'
        assert all(0 <= band < nbands for band in bands), f'{bands} out of range for {nbands:d} bands'
        toret.append((tuple(sorted(bands)), amplitude))
    return toret


def _inject_spectrum3_alm(alm: AlmField, cl, edges, spectrum3, nside=None, backend=None):
    r"""
    Quadratic term injecting a bispectrum in the given :math:`\ell`-band triplets, see
    :func:`generate_spectrum3_alm`. Returns its harmonic coefficients.

    With :math:`u_a = \sum_{\ell \in a, m} (a^G_{\ell m} / C_\ell) Y_{\ell m}` the band-filtered,
    inverse-:math:`C_\ell`-weighted map, each ordering of a band triplet contributes
    :math:`(A / 6) W_c(\ell) [u_a u_b]_{\ell m}`.
    """
    attrs = alm.attrs
    wbin = np.asarray(_make_edges_angular(attrs, edges)['wbin'])
    rdtype = alm.value.real.dtype
    cl = jnp.asarray(cl, dtype=rdtype)
    icl = jnp.where(cl > 0., 1. / jnp.where(cl > 0., cl, 1.), 0.)

    # u_a, one inverse transform per band
    us = [alm2map(alm.clone(value=alm.value * (jnp.asarray(w, dtype=rdtype) * icl)[:, None]),
                  nside=nside, backend=backend) for w in wbin]

    pairs = {}

    def get_pair(a, b):
        """alm of u_a u_b, one forward transform per unordered pair."""
        key = (min(a, b), max(a, b))
        if key not in pairs:
            product = us[key[0]].clone(value=us[key[0]].value * us[key[1]].value)
            pairs[key] = product.to_alm(ellmax=attrs.ellmax, backend=backend).value
        return pairs[key]

    toret = jnp.zeros_like(alm.value)
    for bands, amplitude in _format_spectrum3_bins(spectrum3, len(wbin)):
        for perm in set(itertools.permutations(bands)):  # the target is symmetric
            toret = toret + (amplitude / 6.) * jnp.asarray(wbin[perm[2]], dtype=rdtype)[:, None] * get_pair(perm[0], perm[1])
    return toret


def _inject_spectrum3_mesh(mesh: RealMeshField, power: Callable, edges, spectrum3):
    r"""
    Quadratic term injecting a bispectrum in the given :math:`k`-band triplets, see
    :func:`generate_spectrum3_mesh`. Returns it as a :class:`RealMeshField`.

    With :math:`u_a = \mathrm{c2r}[W_a(k) \delta(k) / P(k)]`, each ordering of a band triplet
    contributes :math:`(A / 6) W_c(k) \mathrm{r2c}[u_a u_b](k)`. Since ``r2c`` is the unnormalized
    transform and ``c2r`` carries the :math:`1/N`, no extra volume factor appears.

    Contributions are grouped by unordered pair :math:`(a, b)`, so each product needs a single
    forward transform, and the :math:`W_c` are collapsed into one weight per pair. The band
    membership is held as a single index array rather than one boolean mesh per band, and the
    accumulation runs as a :func:`jax.lax.scan`, keeping the footprint independent of the number
    of bands and triplets.
    """
    mattrs = mesh.attrs
    cmesh = mesh.r2c()
    kvec = mattrs.kcoords(sparse=True)
    edges = np.asarray(edges)
    nbands = len(edges) - 1

    # band index per mode, `nbands` meaning "outside any band": one integer mesh, instead of
    # `nbands` boolean meshes
    knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
    iband = jnp.digitize(knorm, edges) - 1
    iband = jnp.where((iband < 0) | (iband >= nbands), nbands, iband)
    del knorm

    pk = power(kvec)
    ipk = jnp.where(pk > 0., 1. / jnp.where(pk > 0., pk, 1.), 0.)
    del pk

    # u_a, one inverse transform per band
    def get_u(band):
        return cmesh.clone(value=cmesh.value * (iband == band) * ipk).c2r().value

    us = jax.lax.map(get_u, jnp.arange(nbands))

    # collapse the orderings into one weight vector per unordered pair, indexed by the third band
    # (the trailing entry, for modes outside any band, stays zero)
    weights = {}
    for bands, amplitude in _format_spectrum3_bins(spectrum3, nbands):
        for perm in set(itertools.permutations(bands)):  # the target is symmetric
            pair = (min(perm[0], perm[1]), max(perm[0], perm[1]))
            weights.setdefault(pair, np.zeros(nbands + 1, dtype=float))[perm[2]] += amplitude / 6.

    def body(carry, xs):
        pair, weight = xs
        product = mesh.clone(value=us[pair[0]] * us[pair[1]]).r2c().value
        return carry + weight[iband] * product, None

    xs = (jnp.asarray(list(weights), dtype=int), jnp.asarray(np.array(list(weights.values())), dtype=cmesh.value.real.dtype))
    toret = jax.lax.scan(body, jnp.zeros_like(cmesh.value), xs)[0]
    return cmesh.clone(value=toret).c2r()


def _local_transform(value, alpha2_local):
    r"""Local (quadratic) transform :math:`\delta = g + \alpha_2^\mathrm{local} (g^2 - \langle g^2 \rangle)`."""
    return value + alpha2_local * (value**2 - jnp.mean(value**2))


def generate_spectrum3_mesh(mattrs: MeshAttrs, power: Callable=lambda kvec: 1., alpha2_local: float=0.,
                            edges: np.ndarray=None, spectrum3: dict=None, seed: int=42,
                            unitary_amplitude: bool=False) -> RealMeshField:
    r"""
    Generate a weakly non-Gaussian mesh with a known bispectrum, of the local ("squeezed") shape.

    The field is :math:`\delta = g + \alpha_2^\mathrm{local} (g^2 - \langle g^2 \rangle)` with :math:`g` Gaussian
    of power spectrum :math:`P`, whose bispectrum follows from Wick's theorem,

    .. math:: B(k_1, k_2, k_3) = 2 \alpha_2^\mathrm{local} [P(k_1) P(k_2) + P(k_2) P(k_3) + P(k_3) P(k_1)] + \mathcal{O}((\alpha_2^\mathrm{local})^3),

    while the power spectrum is :math:`P + \mathcal{O}((\alpha_2^\mathrm{local})^2)`.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes.
    power : Callable
        Power spectrum of the Gaussian field, as a function of the :math:`k`-vector.
    alpha2_local : float
        Amplitude of the quadratic term. Keep :math:`\alpha_2^\mathrm{local} \sigma_g \ll 1` for the
        :math:`\mathcal{O}((\alpha_2^\mathrm{local})^3)` terms to be negligible.
    edges : array-like, optional
        1D :math:`k`-bin edges defining the bands used by ``spectrum3``.
    spectrum3 : dict, optional
        Bispectrum to inject, as ``{(i, j, k): amplitude}`` over sorted band indices of ``edges``,
        e.g. ``{(0, 1, 1): 1e8}``. A bin-restricted target is separable, so this costs one inverse
        transform per band plus one forward transform per band pair. Keep the amplitude well below
        :math:`P^2`, the construction being perturbative.
    seed : int, tuple, optional
        Random seed.
    unitary_amplitude : bool, optional
        If ``True``, the Gaussian field is generated with unitary amplitude.

    Returns
    -------
    mesh : RealMeshField

    Note
    ----
    Only the local shape is generated. An arbitrary target bispectrum would require the quadratic
    kernel :math:`K(k_1, k_2) = B(k_1, k_2, k_3) / (3 P(k_1) P(k_2))`, i.e. a non-separable convolution.
    """
    mesh = generate_gaussian_mesh(mattrs, power=power, seed=seed, unitary_amplitude=unitary_amplitude)
    toret = mesh.clone(value=_local_transform(mesh.value, alpha2_local))
    if spectrum3:
        toret = toret.clone(value=toret.value + _inject_spectrum3_mesh(mesh, power, edges, spectrum3).value)
    return toret


def generate_spectrum3_alm(attrs: AngularAttrs, cl: Callable | np.ndarray=lambda ell: 1., alpha2_local: float=0.,
                           edges: np.ndarray | dict=None, spectrum3: dict=None,
                           seed: int=42, nside: int=None, unitary_amplitude: bool=False,
                           backend: str=None) -> AlmField:
    r"""
    Generate the harmonic coefficients of a weakly non-Gaussian field on the sphere with a known
    bispectrum, of the local shape; the analog of :func:`generate_spectrum3_mesh` for :mod:`.angular`.

    The field is :math:`\delta = g + \alpha_2^\mathrm{local} (g^2 - \langle g^2 \rangle)` with :math:`g` Gaussian
    of angular power spectrum :math:`C_\ell`, whose reduced bispectrum is

    .. math:: b_{\ell_1 \ell_2 \ell_3} = 2 \alpha_2^\mathrm{local} [C_{\ell_1} C_{\ell_2} + C_{\ell_2} C_{\ell_3} + C_{\ell_3} C_{\ell_1}] + \mathcal{O}((\alpha_2^\mathrm{local})^3),

    in the normalization of :func:`compute_angular3_spectrum`. Note that for a white :math:`C_\ell`
    this is exactly constant, which makes it a convenient input for window-matrix tests.

    Parameters
    ----------
    attrs : AngularAttrs
        Angular attributes, providing ``ellmax`` and ``nside``.
    cl : Callable, array-like or Angular2Spectrum, optional
        Angular power spectrum of the Gaussian field, see :func:`generate_spectrum2_alm`.
    alpha2_local : float
        Amplitude of the quadratic term. Keep :math:`\alpha_2^\mathrm{local} \sigma_g \ll 1`.
    edges : array-like or dict, optional
        1D :math:`\ell`-band edges defining the bands used by ``spectrum3``,
        see :class:`BinAngular2Spectrum`.
    spectrum3 : dict, optional
        Reduced bispectrum to inject, as ``{(i, j, k): amplitude}`` over sorted band indices of
        ``edges``, e.g. ``{(0, 1, 1): 0.4}``. A bin-restricted target is separable, so this costs one
        inverse transform per band plus one forward transform per band pair. Keep the amplitude well
        below :math:`C_\ell^2`, the construction being perturbative.
    seed : int, tuple, optional
        Random seed.
    nside : int, optional
        Healpix resolution at which the quadratic term is formed; defaults to ``attrs.nside``.
        Take it comfortably above ``ellmax``: squaring the field doubles its harmonic content,
        which the grid must support to avoid aliasing back into :math:`\ell \leq \ell_\mathrm{max}`.
    unitary_amplitude : bool, optional
        If ``True``, the Gaussian field is generated with unitary amplitude.
    backend : str, optional
        Backend for the harmonic transforms, see :meth:`PixelField.to_alm`.

    Returns
    -------
    alm : AlmField

    Note
    ----
    Only the local shape is generated; an arbitrary target :math:`b_{\ell_1 \ell_2 \ell_3}` would
    require the pair-product sum with kernel :math:`b_{\ell_1 \ell_2 \ell_3} / (3 C_{\ell_1} C_{\ell_2})`,
    i.e. :math:`\mathcal{O}(\ell_\mathrm{max}^2)` transforms per realization.

    Warning
    -------
    Bands containing :math:`\ell = 0` fall short of the formula above. Subtracting
    :math:`\langle g^2 \rangle` zeroes the monopole of the quadratic term -- as it must, the mean
    overdensity being zero -- so the permutation in which the quadratic leg carries :math:`\ell = 0`
    is absent. Bin from :math:`\ell \geq 1` (or compare against a target built the same way) if you
    rely on the analytic form.
    """
    cl = _get_cl_array(cl, ellmax=attrs.ellmax)
    alm = generate_spectrum2_alm(attrs, cl=cl, seed=seed, unitary_amplitude=unitary_amplitude)
    pixel = alm.to_pixel(nside=nside, backend=backend)
    pixel = pixel.clone(value=_local_transform(pixel.value, alpha2_local))
    toret = pixel.to_alm(ellmax=attrs.ellmax, backend=backend)
    if spectrum3:
        toret = toret.clone(value=toret.value + _inject_spectrum3_alm(alm, cl, edges, spectrum3, nside=nside, backend=backend))
    return toret


def generate_uniform_particles(mattrs: MeshAttrs, size: int=None, seed: int=42, exchange=False, **kwargs):
    """
    Generate uniformly distributed particles in the input box.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh attributes (box size, mesh size, etc.).
    size : int
        Number of particles to generate.
    seed : int, tuple, optional
        Random seed for mesh generation.
        Provide (seed, ids) to ensure reproducibility when changing the number of devices; see :func:`create_sharded_random` for details.
    exchange : bool, default=False
        If ``True``, perform particle exchange for distributed computation.
    kwargs : dict
        Other arguments for :class:`ParticleField`.

    Returns
    -------
    particles : ParticleField
        Generated uniformly distributed particles.
    """
    def sample(key, shape):
        return mattrs.boxsize * random.uniform(key, shape + (len(mattrs.boxsize),), dtype=mattrs.rdtype) - mattrs.boxsize / 2. + mattrs.boxcenter
    positions = create_sharded_random(sample, seed, shape=size, out_specs=P(mattrs.sharding_mesh.axis_names,))
    #positions = exchange_particles(mattrs, positions=positions, return_inverse=False)[0]
    return ParticleField(positions, attrs=mattrs, exchange=exchange, **kwargs)
