from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp
from jax import random

from jaxpower import resamplers
from jaxpower.mesh import staticarray, BaseMeshField, RealMeshField, ParticleField


dirname = Path('_tests')


def test_static_array():
    array = staticarray.fill(3., 4)
    assert np.allclose(array, 3.) and array.shape == (4,)
    assert jnp.zeros(shape=staticarray.fill(3, 4)).shape == (3, 3, 3, 3)


def test_base_mesh():
    mesh = BaseMeshField(jnp.zeros((4, 3)))
    mesh2 = BaseMeshField(jnp.zeros((4, 3)), boxsize=(4., 3.))

    @jax.jit
    def test(mesh, mesh2):
        return jax.tree.map(jnp.power, mesh, mesh2)

    mesh3 = test(mesh, mesh2)
    assert np.allclose(mesh3.value, mesh.value**mesh2.value)
    assert np.allclose(mesh3.boxsize, mesh.boxsize)

    bak = mesh
    mesh += 2
    assert np.allclose(mesh, bak.value + 2)
    bak = mesh
    mesh /= 3
    assert np.allclose(mesh, bak.value / 3.)
    bak = mesh
    mesh = mesh.conj()
    assert np.allclose(mesh, bak.value.conj())

    mesh2 = mesh.at[1, 2].set(100.)
    assert np.allclose(mesh2[1, 2], 100)


def test_real_mesh():
    mesh = RealMeshField(random.uniform(random.key(42), shape=(10, 21, 13)), boxsize=(1000., 102., 2320.))
    mesh2 = mesh.r2c()
    assert mesh2.shape != mesh.shape
    assert np.allclose(mesh2.c2r(), mesh)
    mesh = RealMeshField(random.uniform(random.key(42), shape=(10, 21, 13)) + 0. * 1j, boxsize=(1000., 102., 2320.))
    mesh2 = mesh.r2c()
    assert mesh2.shape == mesh.shape
    assert np.allclose(mesh2.c2r(), mesh)
    positions = random.uniform(random.key(42), shape=(10, 3))
    for compensate in [True, False]:
        for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
            assert mesh.read(positions, resampler=resampler, compensate=compensate).shape == positions.shape[:1]



def test_resamplers():
    for shape in [(3,), (4, 3, 5)]:
        meshshape = np.array(shape)
        size = 100
        positions = meshshape * random.uniform(random.key(42), shape=(size, len(meshshape)))
        weights = 1. + random.uniform(random.key(42), shape=(size,))
        for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
            mesh = jnp.zeros(meshshape)
            mesh = getattr(resamplers, resampler).paint(mesh, positions, weights)
            assert np.allclose(mesh.sum(), weights.sum())



def test_particle_field():
    boxsize = 100.
    positions = boxsize * random.uniform(random.key(42), shape=(10, 3))
    weights = 1. + random.uniform(random.key(42), shape=(10,))
    particle = ParticleField(positions, weights=weights, cellsize=10)
    particle = particle + particle
    assert particle.positions.shape[0] == positions.shape[0] * 2
    assert np.all(particle.boxsize > 40)
    assert particle.meshsize.shape == positions.shape[1:]
    for interlacing in [1, 2, 3]:
        for compensate in [False, True]:
            for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
                value = particle.paint(resampler=resampler, compensate=compensate, interlacing=interlacing)
                assert np.allclose(value.sum(), particle.weights.sum())
                assert value.shape == tuple(particle.meshsize)


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)

    test_resamplers()
    test_static_array()
    test_base_mesh()
    test_real_mesh()
    test_particle_field()