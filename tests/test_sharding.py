import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from jaxpower import (compute_mesh2_spectrum, generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, RealMeshField, ParticleField, FKPField, Mesh2SpectrumPole, Mesh2SpectrumPoles, read, WindowMatrix, MeshAttrs, BinMesh2SpectrumPoles, compute_mesh2_spectrum_mean, compute_mesh2_spectrum_window, compute_normalization, utils, create_sharding_mesh, create_sharded_array, create_sharded_random, exchange_particles)


dirname = Path('_tests')


def test_jaxdecomp():
    import jaxdecomp
    mattrs = MeshAttrs(meshsize=[8, 16, 32], boxsize=1000.)
    mesh = mattrs.create(kind='real', fill=partial(random.normal, key=random.key(43)))
    delta = mesh.value
    delta_k = jaxdecomp.fft.pfft3d(delta) #.astype(jnp.complex64))
    jax.block_until_ready(delta_k)
    delta_ref = jnp.fft.fftn(delta)
    delta_ref = jnp.transpose(delta_ref, axes=(1, 2, 0))
    diff = jnp.abs(delta_ref - delta_k)
    print(diff.max())


def test_sharding():
    from jax.experimental import mesh_utils, multihost_utils
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map
    from scipy import special

    def spherical_jn(ell):
        return lambda x: jax.pure_callback(partial(special.spherical_jn, ell), x, x)

    meshsize = (64,) * 3
    device_mesh_shape = (4,)
    devices = mesh_utils.create_device_mesh(device_mesh_shape)

    def _identity_fn(x):
        return x

    with jax.sharding.Mesh(devices, axis_names=('x',)) as sharding_mesh:
        mattrs = MeshAttrs(meshsize=meshsize, boxsize=1000.)
        if False:
            t0 = time.time()
            n = 5
            for i in range(n):
                mattrs.kcoords(sparse=True)
            print((time.time() - t0) / n)

        if False:
            knorm = sum(xx**2 for xx in mattrs.kcoords(sparse=True))
            jax.debug.inspect_array_sharding(knorm, callback=print)
            tmp = spherical_jn(2)(knorm)
            jax.debug.inspect_array_sharding(tmp, callback=print)

        if True:

            @partial(shard_map, mesh=sharding_mesh, in_specs=P(*sharding_mesh.axis_names), out_specs=P(*sharding_mesh.axis_names))
            def concatenate(array):
                return jnp.concatenate([array] * 2, axis=0)

            array = create_sharded_array(lambda shape: 2 + jnp.arange(shape[0]) + 21 * jax.process_index(), shape=(12,))
            #jax.debug.inspect_array_sharding(array, callback=print)
            #print(jax.process_index(), [_.data for _ in array.addressable_shards])
            array = jnp.concatenate([array] * 2, axis=0)
            #array = concatenate(array)
            #jax.debug.inspect_array_sharding(array, callback=print)
            print(jax.process_index(), jax.jit(_identity_fn, out_shardings=array.sharding.with_spec(P()))(array).addressable_data(0))


def test_exchange_array():
    from jax.experimental import mesh_utils
    from jax.sharding import PartitionSpec as P
    from jaxpower.mesh import _exchange_array_jax, _exchange_inverse_jax, _exchange_array_mpi, _exchange_inverse_mpi


    def _identity_fn(x):
        return x

    def allgather(array):
        return jax.jit(_identity_fn, out_shardings=array.sharding.with_spec(P(None)))(array).addressable_data(0)

    test = 0
    if test == 0:
        device_mesh_shape = (4,)
        devices = mesh_utils.create_device_mesh(device_mesh_shape)
        with jax.sharding.Mesh(devices, axis_names=('x',)) as sharding_mesh:

            nprocs = sharding_mesh.devices.size

            if False:
                array = create_sharded_random(jax.random.uniform, jax.random.key(42), shape=(1024,))
                device = jnp.floor(array * nprocs).astype('i4')
                exchanged, *indices = _exchange_array_jax(array, device, pad=jnp.nan, return_indices=True)
                #tmp = [_.data[~np.isnan(_.data)] for _ in exchanged.addressable_shards]
                #print(jax.process_index(), [(tt.min(), tt.max()) for tt in tmp])
                array2 = _exchange_inverse_jax(exchanged, *indices)
                assert np.allclose(allgather(array2), allgather(array))

            if False:
                from mpi4py import MPI
                mpicomm = MPI.COMM_WORLD
                array = np.random.uniform(size=int(1e5))
                device = np.clip(np.floor(array * nprocs).astype('i4'), 0, mpicomm.size - 1)
                print(device.min(), device.max())
                exchanged, *indices = _exchange_array_mpi(array, device, return_indices=True, mpicomm=mpicomm)
                array2 = _exchange_inverse_mpi(exchanged, *indices, mpicomm=mpicomm)
                assert np.allclose(array2, array)

            if True:
                positions = np.random.uniform(size=(int(1e4) + jax.process_index(), 3))
                weights = np.random.uniform(size=positions.shape[0])
                mattrs = MeshAttrs(boxsize=1., boxcenter=0.5, meshsize=4)
                for backend in ['mpi', 'jax']:
                    positions, exchange, inverse = exchange_particles(mattrs, positions, backend=backend, return_type='jax', return_inverse=True)
                    weights2 = inverse(exchange(weights))
                    assert np.allclose(weights2, weights)

    if test == 1:
        from mpi4py import MPI
        mpicomm = MPI.COMM_WORLD
        rng = np.random.RandomState(seed=42 + mpicomm.rank)
        array = rng.uniform(size=int(1e5))
        device = np.clip(np.floor(array * mpicomm.size).astype('i4'), 0, mpicomm.size - 1)
        device = np.clip(device, 0, mpicomm.size - 2)
        exchanged, indices = _exchange_array_mpi(array, device, return_indices=True, mpicomm=mpicomm)
        array_gathered = np.concatenate(mpicomm.allgather(array))
        device_gathered = np.concatenate(mpicomm.allgather(device))
        assert all(mpicomm.allgather(np.allclose(exchanged, array_gathered[device_gathered == mpicomm.rank])))
        array2 = _exchange_inverse_mpi(exchanged, indices, mpicomm=mpicomm)
        assert np.allclose(array2, array)


def test_halo():
    import jaxdecomp
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map
    from jaxpower.mesh import pad_halo, exchange_halo, unpad_halo

    test = 0
    if test == 0:
        print(jax.process_count())
        device_mesh_shape = (2, 2)
        halo_size = 2
        meshsize = (8,) * 3
        with create_sharding_mesh(device_mesh_shape=device_mesh_shape) as sharding_mesh:
            @partial(shard_map, mesh=sharding_mesh, in_specs=P(*sharding_mesh.axis_names), out_specs=P(*sharding_mesh.axis_names))
            def paint(value):
                value = value.at[2 * halo_size:3 * halo_size].set(-100.)
                return value

            value = create_sharded_random(jax.random.uniform, jax.random.key(42), meshsize)
            value, offset = pad_halo(value, halo_size=halo_size)
            print(value.shape, halo_size, meshsize)
            #jaxdecomp.fft.pfft3d(value.astype(jnp.complex64))
            value = paint(value)
            #jax.debug.inspect_array_sharding(value, callback=print))
            #print(jax.process_index(), value.addressable_data(1)[halo_size:2 * halo_size])
            value = exchange_halo(value, halo_size=halo_size)
            #print(jax.process_index(), value.addressable_data(1)[halo_size:2 * halo_size])
            value = unpad_halo(value, halo_size=halo_size)
            print(jax.process_index(), value.addressable_data(0)[:halo_size])


def test_mesh_power():

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    f, b = 0.8, 1.5
    beta = f / b
    edgesin = np.linspace(0.001, 0.7, 100)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    kin = np.mean(edgesin, axis=-1)
    ellsin = (0, 2, 4)
    theory = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])

    poles = []
    for ell, value in zip(ellsin, theory):
        poles.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=value, ell=ell))
    theory = Mesh2SpectrumPoles(poles)

    meshsize = (32,) * 3
    with create_sharding_mesh(meshsize=meshsize) as mesh:

        mattrs = MeshAttrs(meshsize=meshsize, boxsize=1000.)
        bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))

        @partial(jax.jit, static_argnames=['los', 'ells'])
        def mock(seed, los='x'):
            mesh = generate_anisotropic_gaussian_mesh(mattrs, theory, seed=seed, los=los, unitary_amplitude=True)
            return compute_mesh2_spectrum(mesh, los={'local': 'firstpoint'}.get(los, los), bin=bin)

        #mock(random.key(43), los='x')
        #mock(random.key(43), los='local')

        for flag in ['smooth', 'infinite'][1:]:
            for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')]:
                print(flag, los, thlos, flush=True)
                mean = compute_mesh2_spectrum_mean(mattrs, theory=(theory, thlos) if thlos is not None else theory, bin=bin, los=los)
                wmatrix = compute_mesh2_spectrum_window(mattrs, edgesin=edgesin, ellsin=(ellsin, thlos) if thlos is not None else ellsin, bin=bin, los=los, pbar=True, flags=(flag,))


def test_mem():
    import jaxdecomp

    meshsize = (10,) * 3
    test = 1
    if test == 0:
        array = create_sharded_random(jax.random.normal, jax.random.key(42), meshsize)
        print(jaxdecomp.fft.pfft3d(array.astype(jnp.complex64)).std())
    elif test == 1:
        print(jax.process_count())
        device_mesh_shape = (2, 2)
        with create_sharding_mesh(device_mesh_shape=device_mesh_shape) as sharding_mesh:
            array = create_sharded_random(jax.random.normal, jax.random.key(42), meshsize)
            jax.debug.inspect_array_sharding(array, callback=print)
            print(array.addressable_shards)
            #print(jaxdecomp.fft.pfft3d(array.astype(jnp.complex64)).std())


def test_particles():
    rank = jax.process_index()
    size = 1000
    mattrs = MeshAttrs(meshsize=[8, 16, 32], boxsize=1000.)
    per_host_positions = mattrs.boxsize * random.uniform(random.key(rank), (size, len(mattrs.boxsize))) - mattrs.boxsize / 2. + mattrs.boxcenter

    mesh = ParticleField(positions=per_host_positions, attrs=mattrs, exchange=True).paint(resmpler='tsc', interlacing=3, compensate=True)
    ells = (0, 2, 4)
    bin = BinMesh2SpectrumPoles(mattrs, ells=ells, edges={'step': 0.01})
    compute_mesh2_spectrum(mesh, los='firstpoint', bin=bin)


def get_mock_fn(kind='data'):
    base_dir = Path('_tests')
    if kind == 'mesh':
        return base_dir / '{}.npz'.format(kind)
    return base_dir / '{}.npy'.format(kind)


def save_reference_mock():
    mattrs = MeshAttrs(meshsize=(32, 64, 100), boxsize=1000., boxcenter=1200.)
    size = 128**2
    data_positions = generate_uniform_particles(mattrs, size + 1, seed=42).positions
    randoms_positions = generate_uniform_particles(mattrs, 4 * size + 1, seed=43).positions

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    f, b = 0.8, 1.5
    beta = f / b
    edgesin = np.linspace(0.001, 0.7, 100)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    kin = np.mean(edgesin, axis=-1)
    ellsin = (0, 2, 4)
    theory = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])

    poles = []
    for ell, value in zip(ellsin, theory):
        poles.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=value, ell=ell))
    theory = Mesh2SpectrumPoles(poles)

    mesh = generate_anisotropic_gaussian_mesh(mattrs, theory, seed=random.key(42), los='x', unitary_amplitude=True)
    data_weights = mesh.read(data_positions).real

    utils.mkdir(get_mock_fn().parent)
    data = {'positions': data_positions, 'weights': data_weights}
    randoms = {'positions': randoms_positions, 'weights': jnp.ones_like(randoms_positions[..., 0])}
    np.save(get_mock_fn('data'), data, allow_pickle=True)
    np.save(get_mock_fn('randoms'), randoms, allow_pickle=True)

    data = ParticleField(**data, attrs=mattrs)
    randoms = ParticleField(**randoms, attrs=mattrs)
    fkp = FKPField(data, randoms)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh.write(get_mock_fn('mesh'))
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))
    power = compute_mesh2_spectrum(mesh, bin=bin, los='firstpoint')
    power.write(get_mock_fn('mesh_power'))


def compute_power_spectrum():

    size = jax.process_count()
    rank = jax.process_index()

    def load(kind='data'):
        catalog = np.load(get_mock_fn(kind=kind), allow_pickle=True)[()]
        #return {name: catalog[name] for name in ['positions', 'weights']}
        total_size = catalog['positions'].shape[0]
        print('total_size', total_size)
        sl = slice(rank * total_size // size, (rank + 1) * total_size // size)
        return {name: catalog[name][sl] for name in ['positions', 'weights']}

    ref_mesh_power = read(get_mock_fn('mesh_power'))
    los = ref_mesh_power.attrs.pop('los')
    ref_mesh_power.attrs.pop('dtype')
    mattrs = MeshAttrs(**ref_mesh_power.attrs)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))
    ref_fkp_power = read(get_mock_fn('fkp_power'))

    with create_sharding_mesh(meshsize=mattrs.meshsize) as sharding_mesh:
        #positions, weights = make_particles_from_local(**load(kind='data'))
        #jax.debug.inspect_array_sharding(positions, callback=print)
        #print(positions.shape)
        data = ParticleField(**load(kind='data'), attrs=mattrs, exchange=True)
        randoms = ParticleField(**load(kind='randoms'), attrs=mattrs, exchange=True)
        #data = ParticleField(**load(kind='data'), attrs=mattrs)
        #randoms = ParticleField(**load(kind='randoms'), attrs=mattrs)
        #print(data.weights.min(), data.weights.max())
        #jax.debug.inspect_array_sharding(data.positions, callback=print)
        fkp = FKPField(data, randoms)
        #fkp = FKPField.same_mesh(fkp)[0]
        mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        #t0 = time.time()
        #for i in range(3): jax.block_until_ready(fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real'))
        #print(time.time() - t0)
        #jax.block_until_ready(mesh.value)
        jax.debug.inspect_array_sharding(mesh.value, callback=print)
        #mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        #mesh = RealMeshField.load(get_mock_fn(kind='mesh'))
        #mesh = mesh.clone(value=jax.device_put(mesh.value, jax.sharding.NamedSharding(sharding_mesh, spec=P(*sharding_mesh.axis_names))))
        mesh_power = compute_mesh2_spectrum(mesh, bin=bin, los='firstpoint')
    diff = jnp.abs(mesh_power.value() - ref_mesh_power.value())
    print(diff)
    assert np.allclose(mesh_power.value(), ref_mesh_power.value(), rtol=1e-6, atol=1e-2)
    #print(power.view().sum(), ref.view().sum(), diff, diff[~jnp.isnan(diff)].max())


def test_scaling_raw():
    import jax
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    import jaxdecomp

    device_mesh_shape = (2, 2)
    sharding_mesh = jax.make_mesh(device_mesh_shape, axis_names=('x', 'y'), axis_types=(jax.sharding.AxisType.Auto,) * len(device_mesh_shape))

    with jax.set_mesh(sharding_mesh):

        def mesh_shard_shape(shape: tuple):
            return tuple(s // pdim for s, pdim in zip(shape, sharding_mesh.devices.shape)) + shape[sharding_mesh.devices.ndim:]

        shape = (1024,) * 3
        key = jax.random.key(43)
        value = shard_map(partial(jax.random.normal, shape=mesh_shard_shape(shape), dtype='float32'), mesh=sharding_mesh, in_specs=P(), out_specs=P('x', 'y'))(key)  # yapf: disable

        @jax.jit
        #@partial(shard_map, mesh=sharding_mesh, in_specs=P('x', 'y'), out_specs=P('y', 'x'))
        def fft(value):
            #value = jax.lax.with_sharding_constraint(value, jax.sharding.NamedSharding(sharding_mesh, spec=P('x', 'y')))
            return jaxdecomp.fft.pfft3d(value)

        cvalue = fft(value)
        jax.debug.inspect_array_sharding(cvalue, callback=print)
        n = 15
        t0 = time.time()
        for i in range(n): jax.block_until_ready(fft(value + 0.001))
        print(time.time() - t0)


def test_scaling():

    with create_sharding_mesh() as sharding_mesh:
        from jaxpower import MeshAttrs, create_sharded_random
        mattrs = MeshAttrs(meshsize=1024, boxsize=1000.)
        mesh = mattrs.create(kind='real', fill=create_sharded_random(jax.random.normal, jax.random.key(42), shape=mattrs.meshsize))

        @jax.jit
        def f(mesh):
            return mesh.r2c()

        jax.block_until_ready(f(mesh))
        n = 15
        t0 = time.time()
        for i in range(n): jax.block_until_ready(f(mesh + 0.001))
        print(time.time() - t0)


def test_sharding_routines():
    from jaxpower.mesh import make_array_from_process_local_data

    with create_sharding_mesh() as sharding_mesh:
        mattrs = MeshAttrs(boxsize=1000., meshsize=64)
        particles = generate_uniform_particles(mattrs, size=1000, seed=42, exchange=False)
        exchanged = particles.exchange(backend='jax')

        local_size = particles.weights.addressable_shards[0].data.shape[0]
        rng = np.random.RandomState(seed=42)
        values = rng.uniform(0., 1., size=(local_size, 10))
        values = make_array_from_process_local_data(values, pad='mean')
        values = exchanged.exchange_direct(values, pad='mean')
        print(values.shape)


if __name__ == '__main__':
    from jax import config
    config.update('jax_enable_x64', True)
    import warnings
    warnings.simplefilter("error")
    test_exchange_array()
    #save_reference_mock()
    # Setting up distributed jax
    #jax.distributed.initialize()
    #test_sharding_routines()
    #test_halo()
    #test_exchange_array()
    #test_jaxdecomp()
    #test_mesh_power()
    #test_sharding()
    #test_mem()
    #compute_power_spectrum()
    #test_scaling_raw()
    #test_scaling()
    # Closing distributed jax
    #jax.distributed.shutdown()