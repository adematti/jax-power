from collections.abc import Callable

import numpy as np
from jax import random
from jax import numpy as jnp

from .mesh import RealMeshField, ParticleField, MeshAttrs
from .power import _get_los_vector, legendre, get_real_Ylm


def generate_gaussian_mesh(power: Callable, seed: int=42,
                           boxsize: float | np.ndarray=1000., meshsize: int | np.ndarray=128,
                           unitary_amplitude: bool=False, boxcenter=0.):

    """Generate :class:`RealMeshField` with input power."""

    if isinstance(seed, int):
        seed = random.key(seed)

    attrs = MeshAttrs(boxsize=boxsize, meshsize=meshsize, boxcenter=boxcenter)
    mesh = RealMeshField(random.normal(seed, attrs.meshsize), boxsize=boxsize, boxcenter=boxcenter).r2c()

    def kernel(value, kvec):
        ker = jnp.sqrt(power(kvec) / mesh.cellsize.prod())
        if unitary_amplitude:
            ker *= jnp.sqrt(mesh.meshsize.prod(dtype=float)) / jnp.abs(value)
        return value * ker

    mesh = mesh.apply(kernel, kind='wavenumber')
    return mesh.c2r()


def generate_anisotropic_gaussian_mesh(poles: dict[Callable], seed: int=42, los: str='x',
                                       boxsize: float | np.ndarray=1000., meshsize: int | np.ndarray=128,
                                       unitary_amplitude: bool=False, boxcenter=0.):

    """Generate :class:`RealMeshField` with input power spectrum multipoles."""

    ells = (0, 2, 4)
    if isinstance(poles, list):
        poles = {ell: p for ell, p in zip(ells, poles)}
    ells = list(poles)

    if isinstance(seed, int):
        seed = random.key(seed)

    attrs = MeshAttrs(meshsize=meshsize, boxsize=boxsize, boxcenter=boxcenter)

    def _safe_divide(num, denom):
        with np.errstate(divide='ignore', invalid='ignore'):
            return jnp.where(denom == 0., 0., num / denom)

    def generate_normal(seed):
        mesh = RealMeshField(random.normal(seed, attrs.meshsize), attrs=attrs).r2c()
        if unitary_amplitude:
            mesh *= jnp.sqrt(attrs.meshsize.prod(dtype=float)) / jnp.abs(mesh.value)
        return mesh

    if los == 'local':
        seeds = random.split(seed)

        kvec = attrs.kcoords(sparse=True, hermitian=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        shape = knorm.shape
        knorm = knorm.ravel()

        a11 = 35. / 18. * poles[4](knorm).reshape(shape) / attrs.cellsize.prod()
        a00 = poles[0](knorm).reshape(shape) / attrs.cellsize.prod() - 1. / 5. * a11
        # Cholesky decomposition
        l00 = jnp.sqrt(a00)
        del a00

        a10 = 1. / 2. * poles[2](knorm).reshape(shape) / attrs.cellsize.prod() - 1. / 7. * a11
        l10 = a10 / jnp.where(knorm.reshape(shape) == 0., 1., l00)
        del a10

        # The mesh for ell = 0
        normal = generate_normal(seeds[0])
        mesh = (normal * l00).c2r()
        del l00
        mesh2 = normal * l10
        del normal
        # The mesh for ell = 2
        mesh2 += generate_normal(seeds[1]) * jnp.sqrt(a11 - l10**2)
        del a11, l10

        xvec = mesh.coords()
        ell = 2
        Ylms = [get_real_Ylm(ell, m, batch=True) for m in range(-ell, ell + 1)]
        mesh += 4. * jnp.pi / (2 * ell + 1) * sum((mesh2 * Ylm(*kvec)).c2r() * Ylm(*xvec) for Ylm in Ylms)  # total mesh, mesh0 + mesh2 * L2(mu)

        return mesh

    else:
        vlos = _get_los_vector(los, ndim=len(attrs.meshsize))
        mesh = RealMeshField(random.normal(seed, attrs.meshsize), attrs=attrs).r2c()
        def kernel(value, kvec):
            knorm = jnp.sqrt(sum(kk**2 for kk in kvec)).ravel()
            mu = _safe_divide(sum(kk * ll for kk, ll in zip(kvec, vlos)).ravel(), knorm)
            ker = 0.
            for ell in ells:
                ker += poles[ell](knorm) / attrs.cellsize.prod() * legendre(ell)(mu)
            ker = jnp.sqrt(ker).reshape(value.shape)
            if unitary_amplitude:
                ker *= jnp.sqrt(attrs.meshsize.prod(dtype=float)) / jnp.abs(value)
            return value * ker

        mesh = mesh.apply(kernel, kind='wavenumber')
        return mesh.c2r()



def generate_uniform_particles(size, seed: int=42, boxsize: float | np.ndarray=1000., boxcenter: float | np.ndarray=0., meshsize=None):

    """Generate :class:`ParticleField` in input box."""

    if isinstance(seed, int):
        seed = random.key(seed)

    attrs = MeshAttrs(boxsize=boxsize, meshsize=meshsize, boxcenter=boxcenter)
    positions = attrs.boxsize * random.uniform(seed, (size, len(attrs.boxsize))) - attrs.boxsize / 2. + attrs.boxcenter
    return ParticleField(positions, **attrs)
