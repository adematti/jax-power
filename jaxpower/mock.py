from collections.abc import Callable

import numpy as np
from jax import random
from jax import numpy as jnp

from .mesh import RealMeshField, ParticleField, MeshAttrs


def generate_gaussian_mesh(power: Callable, seed: int=42, boxsize: float | np.ndarray=1000., meshsize: int | np.ndarray=128, unitary_amplitude: bool=False, boxcenter=0.):

    """Generate :class:`RealMeshField` with input power."""

    if isinstance(seed, int):
        seed = random.key(seed)

    attrs = MeshAttrs(boxsize=boxsize, meshsize=meshsize, boxcenter=boxcenter)
    mesh = RealMeshField(random.normal(seed, attrs.meshsize), boxsize=boxsize, boxcenter=boxcenter)
    mesh = mesh.r2c()

    def kernel(value, kvec):
        ker = jnp.sqrt(power(kvec) / mesh.cellsize.prod())
        if unitary_amplitude:
            ker *= mesh.meshsize.prod()**0.5 / jnp.abs(value)
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
