from jax import random
from jax import numpy as jnp

from .mesh import RealMeshField, staticarray


def generate_gaussian_random_field(power, seed=42, boxsize=1000., meshsize=128, unitary_amplitude=False):

    if isinstance(seed, int):
        seed = random.key(seed)

    ndim = 3
    for value in [boxsize, meshsize]:
        try: ndim = len(value)
        except: pass
    shape = staticarray.fill(meshsize, ndim)
    mesh = RealMeshField(random.normal(seed, shape), boxsize=boxsize)
    mesh = mesh.r2c()

    def kernel(value, kvec):
        k = sum(kk**2 for kk in kvec)**0.5
        ker = jnp.sqrt(power(k.ravel()).reshape(k.shape) / mesh.cellsize.prod())
        if unitary_amplitude:
            ker *= mesh.meshsize.prod()**0.5 / jnp.abs(value)
        return value * ker

    mesh = mesh.apply(kernel, kind='wavenumber')
    return mesh.c2r()
