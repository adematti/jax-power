import numpy as np
from jax import random
from jax import numpy as jnp

from .mesh import RealMeshField, ParticleField, staticarray
from collections.abc import Callable
from typing import Union


def _get_ndim(*args):
    ndim = 3
    for value in args:
        try: ndim = len(value)
        except: pass
    return ndim


def generate_gaussian_mesh(power: Callable, seed: int=42, boxsize: Union[float, np.ndarray]=1000., meshsize: Union[int, np.ndarray]=128, unitary_amplitude: bool=False, boxcenter=0.):

    """Generate :class:`RealMeshField` with input power."""

    if isinstance(seed, int):
        seed = random.key(seed)

    ndim = _get_ndim(boxsize, meshsize, boxcenter)
    shape = staticarray.fill(meshsize, ndim)
    mesh = RealMeshField(random.normal(seed, shape), boxsize=boxsize, boxcenter=boxcenter)
    mesh = mesh.r2c()

    def kernel(value, kvec):
        ker = jnp.sqrt(power(kvec) / mesh.cellsize.prod())
        if unitary_amplitude:
            ker *= mesh.meshsize.prod()**0.5 / jnp.abs(value)
        return value * ker

    mesh = mesh.apply(kernel, kind='wavenumber')
    return mesh.c2r()


def generate_uniform_particles(size, seed: int=42, boxsize: Union[float, np.ndarray]=1000., boxcenter: Union[float, np.ndarray]=0., meshsize=None):

    """Generate :class:`ParticleField` in input box."""

    if isinstance(seed, int):
        seed = random.key(seed)

    ndim = _get_ndim(boxsize, boxcenter, meshsize)
    boxsize = staticarray.fill(boxsize, ndim)
    boxcenter = staticarray.fill(boxcenter, ndim)
    positions = boxsize * random.uniform(seed, (size, ndim)) - boxsize / 2. + boxcenter
    return ParticleField(positions, boxsize=boxsize, boxcenter=boxcenter, meshsize=meshsize)