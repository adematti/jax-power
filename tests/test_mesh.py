import time
from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp
from jax import random

from jaxpower import resamplers
from jaxpower.mesh import staticarray, MeshAttrs, BaseMeshField, RealMeshField, ParticleField


dirname = Path('_tests')


def test_static_array():
    array = staticarray.fill(3., 4)
    assert np.allclose(array, 3.) and array.shape == (4,)
    assert jnp.zeros(shape=staticarray.fill(3, 4)).shape == (3, 3, 3, 3)


def test_mesh_attrs():
    attrs = MeshAttrs(meshsize=100, boxsize=100.)
    dict(**attrs)
    attrs.create()


def test_base_mesh():
    mesh = BaseMeshField(jnp.ones((4, 3)))
    mesh2 = BaseMeshField(jnp.full((4, 3), 3.), boxsize=(4., 3.))

    if False:
        @jax.jit
        def test(mesh, mesh2):
            return jax.tree.map(jnp.power, mesh, mesh2)

        mesh3 = test(mesh, mesh2)
        assert np.allclose(mesh3.value, mesh.value**mesh2.value)
        assert np.allclose(mesh3.boxsize, mesh.boxsize)

    bak = mesh
    mesh += 2
    assert isinstance(mesh, BaseMeshField) and np.allclose(mesh, bak.value + 2)
    bak = mesh
    mesh /= 3
    assert isinstance(mesh, BaseMeshField) and np.allclose(mesh, bak.value / 3.)
    bak = mesh
    mesh = mesh.conj()
    assert isinstance(mesh, BaseMeshField) and np.allclose(mesh, bak.value.conj())
    bak = mesh
    mesh = mesh + mesh
    assert np.allclose(mesh, bak.value * 2)
    bak = mesh
    mesh = mesh.at[1, 2].set(100.)
    assert isinstance(mesh, BaseMeshField) and np.allclose(mesh[1, 2], 100)

    fn = dirname / 'mesh.npz'
    mesh.save(fn)
    mesh2 = RealMeshField.load(fn)
    mesh2.attrs.boxsize
    assert np.allclose(mesh2.value, mesh.value)


def test_real_mesh():
    for engine in ['jax', 'jaxdecomp']:
        mesh = RealMeshField(random.uniform(random.key(42), shape=(10, 21, 13)), boxsize=(1000., 102., 2320.), fft_engine=engine)
        mesh2 = mesh.rebin(factor=(2, 1, 1))
        assert mesh2.shape == (5, 21, 13)
        mesh2 = mesh.r2c()
        assert mesh2.shape != mesh.shape
        assert np.allclose(mesh2.c2r(), mesh)
        positions = random.uniform(random.key(42), shape=(10, 3))
        for compensate in [True, False]:
            for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
                assert mesh.read(positions, resampler=resampler, compensate=compensate).shape == positions.shape[:1]

        mesh = RealMeshField(random.uniform(random.key(42), shape=(10, 21, 13)) + 0. * 1j, boxsize=(1000., 102., 2320.), fft_engine=engine)
        mesh2 = mesh.r2c()
        if engine == 'jax': assert mesh2.shape == mesh.shape, (mesh2.shape, mesh.shape)
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


def test_jit():

    from jaxpower.resamplers import tsc
    mesh = jnp.zeros((32,) * 3)
    positions = random.uniform(random.key(64), shape=(100000, 3))
    t0 = time.time()
    paint = jax.jit(tsc.paint)
    mesh = paint(mesh, positions)
    #values = tsc.read(mesh, positions)
    print(f'time for jit {time.time() - t0:.2f}')
    t0 = time.time()
    nmock = 10
    for i in range(nmock):
        mesh = paint(mesh, positions)
        jax.block_until_ready(mesh)
    print(f'time per iteration {(time.time() - t0) / nmock:.2f}')

    positions = ParticleField(positions, meshsize=mesh.shape, boxsize=1., boxcenter=0.5)
    t0 = time.time()
    positions.paint(resampler='tsc')
    print(f'time for jit {time.time() - t0:.2f}')
    t0 = time.time()
    nmock = 10
    for i in range(nmock):
        mesh = positions.paint(resampler='tsc')
        jax.block_until_ready(mesh)
    print(f'time per iteration {(time.time() - t0) / nmock:.2f}')


def test_particle_field():
    boxsize = 100.
    positions = boxsize * random.uniform(random.key(42), shape=(10, 3))
    weights = 1. + random.uniform(random.key(42), shape=(10,))
    particle = ParticleField(positions, weights=weights, cellsize=10)
    particle = particle + particle
    assert np.allclose(particle.cellsize, 10.)
    assert particle.positions.shape[0] == positions.shape[0] * 2
    assert np.all(particle.boxsize > 40)
    assert particle.meshsize.shape == positions.shape[1:]
    for interlacing in [1, 2, 3]:
        for compensate in [False, True]:
            for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
                value = particle.paint(resampler=resampler, compensate=compensate, interlacing=interlacing)
                assert np.allclose(value.sum(), particle.weights.sum())
                assert value.shape == tuple(particle.meshsize)


def test_timing():
    from jaxpower import generate_uniform_particles
    size = int(1e8)
    attrs = MeshAttrs(boxsize=1000., meshsize=512)
    particles = generate_uniform_particles(attrs, size, seed=42)

    kw = dict(resampler='tsc', interlacing=0, compensate=False, out='real')
    value = particles.paint(**kw)
    jax.block_until_ready(value)
    t0 = time.time()
    for i in range(10):
        value = particles.clone(weights=particles.weights * 1.1).paint(**kw)
        jax.block_until_ready(value)
    print(time.time() - t0)


def test_dtype():
    for dtype in ['f4', 'f8']:
        attrs = MeshAttrs(boxsize=1000., meshsize=64, dtype=dtype, fft_engine='jax')
        mesh = attrs.create(kind='real', fill=0.)
        assert mesh.value.dtype == mesh.dtype == attrs.dtype, (mesh.value.dtype, attrs.dtype)
        mesh = mesh.r2c()
        assert mesh.value.dtype.itemsize == 2 * attrs.dtype.itemsize, (mesh.value.dtype, attrs.dtype)
        mesh = mesh.c2r()
        assert mesh.value.dtype == mesh.dtype == attrs.dtype, (mesh.value.dtype, attrs.dtype)



if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)
    test_real_mesh()
    test_base_mesh()
    test_mesh_attrs()
    test_resamplers()
    test_static_array()
    test_particle_field()
    test_dtype()