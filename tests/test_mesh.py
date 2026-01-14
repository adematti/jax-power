import os
import time
from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P

from jaxpower import resamplers, kernels
from jaxpower.mesh import staticarray, MeshAttrs, BaseMeshField, RealMeshField, ParticleField, get_sharding_mesh, create_sharding_mesh, get_mesh_attrs


dirname = Path('_tests')


def test_static_array():
    array = staticarray.fill(3., 4)
    assert np.allclose(array, 3.) and array.shape == (4,)
    assert jnp.zeros(shape=staticarray.fill(3, 4)).shape == (3, 3, 3, 3)


def test_mesh_attrs():
    from jaxpower import generate_uniform_particles
    from jaxpower.mesh import next_fft_size

    attrs = get_mesh_attrs(boxsize=1000., boxcenter=0., cellsize=10.)
    assert np.allclose(attrs.meshsize, 100)
    assert np.allclose(attrs.cellsize, 10.)
    attrs = get_mesh_attrs(boxsize=1000., boxcenter=0., meshsize=128)
    assert np.allclose(attrs.meshsize, 128)
    assert np.allclose(attrs.cellsize, 1000. / 128)

    attrs = MeshAttrs(meshsize=100, boxsize=100.)
    randoms = generate_uniform_particles(attrs, size=100, seed=84)
    attrs = get_mesh_attrs(randoms.positions, meshsize=128)
    assert np.allclose(attrs.meshsize, 128)
    attrs = get_mesh_attrs(randoms.positions, cellsize=12.)
    assert np.allclose(attrs.cellsize, 12.)
    primes = (2, 3, 5)
    attrs = get_mesh_attrs(randoms.positions, cellsize=12., primes=primes)
    assert np.allclose(attrs.meshsize, 18)
    assert np.allclose(attrs.cellsize, 12.)
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

    mesh2.r2c().apply(kernels.gaussian(radius=10.))


def test_real_mesh():
    for backend in ['jax', 'jaxdecomp']:
        mesh = RealMeshField(random.uniform(random.key(42), shape=(10, 21, 13)), boxsize=(1000., 102., 2320.), fft_backend=backend)
        mesh2 = mesh.r2c()
        assert mesh2.shape != mesh.shape
        assert np.allclose(mesh2.c2r(), mesh)
        positions = random.uniform(random.key(42), shape=(10, 3))
        for compensate in [True, False]:
            for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
                assert mesh.read(positions, resampler=resampler, compensate=compensate).shape == positions.shape[:1]

        mesh = RealMeshField(random.uniform(random.key(42), shape=(10, 21, 13)) + 0. * 1j, boxsize=(1000., 102., 2320.), fft_backend=backend)
        mesh2 = mesh.r2c()
        if backend == 'jax': assert mesh2.shape == mesh.shape, (mesh2.shape, mesh.shape)
        assert np.allclose(mesh2.c2r(), mesh)
        positions = random.uniform(random.key(42), shape=(10, 3))
        for compensate in [True, False]:
            for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
                assert mesh.read(positions, resampler=resampler, compensate=compensate).shape == positions.shape[:1]


def test_resamplers():
    for shape in [(3,), (4, 3, 5)]:
        meshshape = np.array(shape)
        size = 10000
        positions = meshshape * random.uniform(random.key(42), shape=(size, len(meshshape)))
        weights = 1. + random.uniform(random.key(42), shape=(size,))
        for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
            mesh = jnp.zeros(meshshape)
            mesh = getattr(resamplers, resampler).paint(mesh, positions, weights)
            assert np.allclose(mesh.sum(), weights.sum())
            mesh = random.uniform(random.key(64), shape=meshshape)
            weights = getattr(resamplers, resampler).read(mesh, positions)


def get_random_catalog(mattrs, size=10000, seed=42):
    from jaxpower import create_sharded_random, generate_uniform_particles
    seed = jax.random.key(seed)
    particles = generate_uniform_particles(mattrs, size, seed=seed, exchange=False)
    def sample(key, shape):
        return jax.random.uniform(key, shape, dtype=mattrs.dtype)
    weights = create_sharded_random(sample, seed, shape=particles.size, out_specs=P(mattrs.sharding_mesh.axis_names,))
    return particles.clone(weights=weights)


def _identity_fn(x):
    return x


def allgather(array):
    sharding_mesh = get_sharding_mesh()
    if sharding_mesh.axis_names:
        array = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, P()))(array)
        return array.addressable_data(0)
    return array


def allgather_particles(particles):
    from typing import NamedTuple

    class Particles(NamedTuple):
        positions: jax.Array
        weights: jax.Array

    return Particles(allgather(particles.positions), allgather(particles.weights))


def test_sharded_paint_read():

    mattrs = MeshAttrs(boxsize=1000., boxcenter=0., meshsize=64)
    pattrs = mattrs.clone(boxsize=800.)

    def f(data, positions, exchange=False):
        mesh = data.paint(resampler=resampler, compensate=False, interlacing=0)
        #print(mesh.std())
        return mesh.read(positions, resampler=resampler, compensate=False, exchange=exchange)

    for resampler in ['ngp', 'cic', 'tsc', 'pcs']:
        with create_sharding_mesh():
            data = get_random_catalog(pattrs, seed=42).clone(attrs=mattrs).exchange(backend='jax')
            randoms = get_random_catalog(pattrs, size=12000, seed=43).clone(attrs=mattrs)
            ref_data = allgather_particles(data)
            ref_randoms = allgather_particles(randoms)
            rweights = randoms.weights
            randoms = randoms.exchange(backend='jax', return_inverse=True)
            #print(jnp.argmin(diff), diff.min())
            weights = allgather(randoms.exchange_inverse(f(data, randoms.positions)))
            #diff = jnp.abs(randoms.exchange_inverse(randoms.weights) - rweights)
            rweights = allgather(randoms.exchange_inverse(randoms.weights))

        ref_data = ParticleField(ref_data.positions, ref_data.weights, attrs=mattrs, exchange=False)
        ref_weights = f(ref_data, ref_randoms.positions)
        assert np.allclose(ref_randoms.weights, rweights)
        assert np.allclose(weights, ref_weights)


def test_paint_jit():

    from jaxpower.resamplers import tsc
    mesh = jnp.zeros((256,) * 3)
    positions = random.uniform(random.key(64), shape=(1000000, 3))
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

    positions = ParticleField(positions, attrs=dict(meshsize=mesh.shape, boxsize=1., boxcenter=0.5))
    t0 = time.time()
    positions.paint(resampler='tsc', interlacing=3, compensate=True)
    print(f'time for jit {time.time() - t0:.2f}')
    t0 = time.time()
    nmock = 10
    for i in range(nmock):
        mesh = positions.paint(resampler='tsc', interlacing=3, compensate=True)
        jax.block_until_ready(mesh)
    print(f'time per iteration {(time.time() - t0) / nmock:.2f}')


def test_particle_field():
    boxsize = 100.
    positions = boxsize * random.uniform(random.key(42), shape=(10, 3))
    weights = 1. + random.uniform(random.key(42), shape=(10,))
    attrs = get_mesh_attrs(positions, cellsize=10)
    particle = ParticleField(positions, weights=weights, attrs=attrs)
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
        attrs = MeshAttrs(boxsize=1000., meshsize=64, dtype=dtype, fft_backend='jax')
        mesh = attrs.create(kind='real', fill=0.)
        assert mesh.value.dtype == mesh.dtype == attrs.dtype, (mesh.value.dtype, attrs.dtype)
        mesh = mesh.r2c()
        assert mesh.value.dtype.itemsize == 2 * attrs.dtype.itemsize, (mesh.value.dtype, attrs.dtype)
        mesh = mesh.c2r()
        assert mesh.value.dtype == mesh.dtype == attrs.dtype, (mesh.value.dtype, attrs.dtype)


def test_sharded_normalization():
    from jaxpower import compute_fkp2_normalization, compute_fkp3_normalization, generate_uniform_particles, FKPField
    from jaxpower.mesh import create_sharded_array

    size = int(1e5)

    with create_sharding_mesh() as sharding_mesh:
        pattrs = MeshAttrs(meshsize=(128,) * 3, boxsize=1121., boxcenter=1500.)
        data = generate_uniform_particles(pattrs, size, seed=(42, 'index'))
        randoms = generate_uniform_particles(pattrs, size, seed=(84, 'index'))
        index = create_sharded_array(lambda start, shape: start * np.prod(shape) + jnp.arange(np.prod(shape)).reshape(shape), jnp.arange(sharding_mesh.devices.size),
                                    shape=(randoms.size,), in_specs=P(sharding_mesh.axis_names), out_specs=P(sharding_mesh.axis_names))
        #assert np.allclose(data.positions.std(), 288.2335461652553)
        fkp = FKPField(data.exchange(backend='jax'), randoms.exchange(backend='jax'))
        index = fkp.randoms.exchange_direct(index)
        norm = compute_fkp2_normalization(fkp, cellsize=None, split=None)
        assert np.allclose(norm, 7.087479585302545)
        norm = compute_fkp2_normalization(fkp, cellsize=10., split=None)
        assert np.allclose(norm, 6.996265569638566)
        norm = compute_fkp2_normalization(fkp, cellsize=10., split=(42, index))
        assert np.allclose(norm, 6.932929039885748)


def test_sharded_random():
    from jaxpower.mesh import create_sharded_array, create_sharded_random

    size = int(100)
    with create_sharding_mesh() as sharding_mesh:
        result = create_sharded_random(jax.random.normal, (jax.random.key(42), 'index'), shape=(size,), out_specs=P(sharding_mesh.axis_names))
        assert np.allclose(result.sum(), -6.613044420294692)
        assert np.allclose(result.std(), 0.9735605177416136)
        shape = (size, 3, 4)
        result = create_sharded_random(jax.random.normal, (jax.random.key(42), 'index'), shape=shape, out_specs=P(sharding_mesh.axis_names))
        assert result.shape == shape
        result = create_sharded_random(lambda key, shape: jax.random.normal(key, shape=shape + (3,)), (jax.random.key(42), 'index'), shape=shape, out_specs=P(sharding_mesh.axis_names))
        assert result.shape == shape + (3,)


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)

    test_real_mesh()
    test_base_mesh()
    test_mesh_attrs()
    test_resamplers()
    test_paint_jit()
    test_static_array()
    test_particle_field()
    test_dtype()
    os.environ["XLA_FLAGS"] = " --xla_force_host_platform_device_count=4"
    test_sharded_paint_read()
    test_sharded_random()
    test_sharded_normalization()