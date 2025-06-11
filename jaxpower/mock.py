from collections.abc import Callable
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from .mesh import RealMeshField, ParticleField, MeshAttrs, exchange_particles, create_sharded_random
from .mesh2 import _get_los_vector
from .utils import BinnedStatistic, get_legendre, get_real_Ylm


def generate_gaussian_mesh(attrs, power: Callable, seed: int=42,
                           unitary_amplitude: bool=False):

    """Generate :class:`RealMeshField` with input power."""

    if isinstance(seed, int):
        seed = random.key(seed)

    mesh = attrs.create(kind='real', fill=create_sharded_random(random.normal, seed, shape=attrs.meshsize)).r2c()

    def kernel(value, kvec):
        ker = jnp.sqrt(power(kvec) / mesh.cellsize.prod())
        if unitary_amplitude:
            ker *= jnp.sqrt(mesh.meshsize.prod(dtype=float)) / jnp.abs(value)
        return value * ker

    mesh = mesh.apply(kernel, kind='wavenumber')
    return mesh.c2r()


def generate_anisotropic_gaussian_mesh(attrs, poles: dict[Callable], seed: int=42, los: str='x', unitary_amplitude: bool=False, **kwargs):

    """Generate :class:`RealMeshField` with input power spectrum multipoles."""

    ells = (0, 2, 4)
    kin = None
    if isinstance(poles, BinnedStatistic):
        kin, poles = jnp.append(poles._edges[0][..., 0], poles._edges[0][-1, 1]), {proj: poles.view(projs=proj) for proj in poles.projs}
    if isinstance(poles, list):
        poles = {ell: pole for ell, pole in zip(ells, poles)}
    ells = list(poles)

    if isinstance(seed, int):
        seed = random.key(seed)

    def generate_normal(seed):
        mesh = attrs.create(kind='real', fill=create_sharded_random(random.normal, seed, shape=attrs.meshsize)).r2c()
        if unitary_amplitude:
            mesh *= jnp.sqrt(attrs.meshsize.prod(dtype=float)) / jnp.abs(mesh.value)
        return mesh

    kvec = attrs.kcoords(sparse=True)
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
        xvec = mesh.coords(sparse=True)
        ell = 2
        Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]

        @jax.checkpoint
        def f(carry, im):
            carry += 4. * jnp.pi / (2 * ell + 1) * (mesh2 * jax.lax.switch(im, Ylms, *kvec)).c2r() * jax.lax.switch(im, Ylms, *xvec)
            return carry, im

        mesh = jax.lax.scan(f, init=mesh, xs=np.arange(len(Ylms)))[0]
        #mesh += 4. * jnp.pi / (2 * ell + 1) * sum((mesh2 * Ylm(*kvec)).c2r() * Ylm(*xvec) for Ylm in Ylms)  # total mesh, mesh0 + mesh2 * L2(mu)

        del mesh2
        return mesh

    else:
        vlos = _get_los_vector(los, ndim=attrs.ndim)
        mesh = generate_normal(seed)

        def kernel(value, kvec):
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)
            ker = sum(get_theory(ell) / attrs.cellsize.prod() * get_legendre(ell)(mu) for ell in ells)
            ker = jnp.sqrt(ker)
            if unitary_amplitude:
                ker *= jnp.sqrt(attrs.meshsize.prod(dtype=value.real.dtype)) / jnp.abs(value)
            return value * ker

        mesh = mesh.apply(kernel, kind='wavenumber').c2r()
        return mesh


def generate_uniform_particles(attrs, size, seed: int=42):

    """Generate :class:`ParticleField` in input box."""

    if isinstance(seed, int):
        seed = random.key(seed)

    def sample(key, shape):
        return attrs.boxsize * random.uniform(seed, shape + (len(attrs.boxsize),), dtype=attrs.dtype) - attrs.boxsize / 2. + attrs.boxcenter

    positions = create_sharded_random(sample, seed, shape=size, out_specs=0)
    positions = exchange_particles(attrs, positions=positions, return_inverse=False)[0]
    return ParticleField(positions, attrs=attrs)
