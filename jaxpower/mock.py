from collections.abc import Callable

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from .mesh import RealMeshField, ParticleField, MeshAttrs
from .power import _get_los_vector, legendre, get_real_Ylm
from .utils import BinnedStatistic


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
    kin = None
    if isinstance(poles, tuple):
        kin, poles = poles
    if isinstance(poles, BinnedStatistic):
        kin, poles = poles._edges[0], {proj: poles.view(projs=proj) for proj in poles.projs}
    if isinstance(poles, list):
        poles = {ell: pole for ell, pole in zip(ells, poles)}
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

    kvec = attrs.kcoords(sparse=True, hermitian=True)
    knorm = jnp.sqrt(sum(kk**2 for kk in kvec)).ravel()
    kshape = np.broadcast_shapes(*(kk.shape for kk in kvec))

    is_callable = all(callable(pole) for pole in poles.values())
    if not is_callable:
        from .utils import Interpolator1D
        interp = Interpolator1D(kin, knorm, edges=len(kin) == len(poles[0]) + 1)

    def get_theory(ell=None, pole=None):
        if pole is None:
            pole = poles[ell]
        if is_callable:
            return pole(knorm)
        else:
            return interp(pole)


    if los == 'local':

        @jax.checkpoint
        def get_meshs(seeds):

            if is_callable:
                p0, p2, p4 = (get_theory(ell) / attrs.cellsize.prod() for ell in ells)
            else:
                p0, p2, p4 = (poles[ell] / attrs.cellsize.prod() for ell in ells)

            a11 = 35. / 18. * p4
            a00 = p0 - 1. / 5. * a11
            # Cholesky decomposition
            l00 = jnp.sqrt(a00)
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
            normal = generate_normal(seeds[0])
            mesh = (normal * _interp(l00)).c2r()
            del l00
            mesh2 = normal * _interp(l10)
            del normal
            # The mesh for ell = 2
            mesh2 += generate_normal(seeds[1]) * _interp(jnp.sqrt(a11 - l10**2))
            del a11, l10
            return mesh, mesh2

        mesh, mesh2 = get_meshs(random.split(seed))
        xvec = mesh.coords()
        ell = 2
        Ylms = [get_real_Ylm(ell, m, batch=True) for m in range(-ell, ell + 1)]

        @jax.checkpoint
        def f(carry, im):
            carry += 4. * jnp.pi / (2 * ell + 1) * (mesh2 * jax.lax.switch(im, Ylms, *kvec)).c2r() * jax.lax.switch(im, Ylms, *xvec)
            return carry, im

        mesh = jax.lax.scan(f, init=mesh, xs=np.arange(len(Ylms)))[0]
        #mesh += 4. * jnp.pi / (2 * ell + 1) * sum((mesh2 * Ylm(*kvec)).c2r() * Ylm(*xvec) for Ylm in Ylms)  # total mesh, mesh0 + mesh2 * L2(mu)

        del mesh2
        return mesh

    else:
        vlos = _get_los_vector(los, ndim=len(attrs.meshsize))
        mesh = generate_normal(seed)

        def kernel(value, kvec):
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)).ravel() / jnp.where(knorm == 0., 1., knorm)
            ker = sum(get_theory(ell) / attrs.cellsize.prod() * legendre(ell)(mu) for ell in ells)
            ker = jnp.sqrt(ker).reshape(value.shape)
            if unitary_amplitude:
                ker *= jnp.sqrt(attrs.meshsize.prod(dtype=float)) / jnp.abs(value)
            return value * ker

        mesh = mesh.apply(kernel, kind='wavenumber').c2r()
        return mesh



def generate_uniform_particles(size, seed: int=42, boxsize: float | np.ndarray=1000., boxcenter: float | np.ndarray=0., meshsize=None):

    """Generate :class:`ParticleField` in input box."""

    if isinstance(seed, int):
        seed = random.key(seed)

    attrs = MeshAttrs(boxsize=boxsize, meshsize=meshsize, boxcenter=boxcenter)
    positions = attrs.boxsize * random.uniform(seed, (size, len(attrs.boxsize))) - attrs.boxsize / 2. + attrs.boxcenter
    return ParticleField(positions, **attrs)
