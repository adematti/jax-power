import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from jaxpower import (compute_mesh_pspec, PowerSpectrumMultipoles, generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, RealMeshField, ParticleField, FKPField,
                      compute_fkp_pspec, BinnedStatistic, WindowMatrix, MeshAttrs, bin_pspec, compute_mesh_pspec_mean, compute_mesh_pspec_window, compute_normalization, utils, create_sharding_mesh, make_particles_from_local, create_sharded_array, create_sharded_random)


dirname = Path('_tests')


def test_jaxdecomp():
    import jaxdecomp
    attrs = MeshAttrs(meshsize=[8, 16, 32], boxsize=1000.)
    mesh = attrs.create(kind='real', fill=partial(random.normal, key=random.key(43)))
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
    with jax.sharding.Mesh(devices, axis_names=('x',)) as sharding_mesh:
        attrs = MeshAttrs(meshsize=meshsize, boxsize=1000.)
        if False:
            t0 = time.time()
            n = 5
            for i in range(n):
                attrs.kcoords(sparse=True)
            print((time.time() - t0) / n)

        if False:
            knorm = sum(xx**2 for xx in attrs.kcoords(sparse=True))
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


def _identity_fn(x):
  return x


def global_array_from_single_device_arrays(sharding, arrays, return_slices=False, pad=None):
    if pad is None:
        pad = np.nan
    if not callable(pad):
        constant_values = pad
        pad = lambda array, pad_width: jnp.pad(array, pad_width, mode='constant', constant_values=constant_values)
    ndevices = sharding.num_devices
    per_host_chunks = arrays
    ndim = per_host_chunks[0].ndim
    per_host_size = jnp.array([per_host_chunk.shape[0] for per_host_chunk in per_host_chunks])
    all_size = jax.make_array_from_process_local_data(sharding, per_host_size)
    all_size = jax.jit(_identity_fn, out_shardings=sharding.with_spec(P()))(all_size).addressable_data(0)
    max_size = all_size.max().item()
    if not np.all(all_size == max_size):
        per_host_chunks = [pad(per_host_chunk, [(0, max_size - per_host_chunk.shape[0])] + [(0, 0)] * (ndim - 1)) for per_host_chunk in per_host_chunks]
    global_shape = (max_size * ndevices,) + per_host_chunks[0].shape[1:]
    tmp = jax.make_array_from_single_device_arrays(global_shape, sharding, per_host_chunks)
    del per_host_chunks
    slices = [slice(j * max_size, j * max_size + all_size[j].item()) for j in range(ndevices)]
    if return_slices:
        return tmp, slices
    return tmp


def allgather_single_device_arrays(sharding, arrays, return_slices=False, **kwargs):
    tmp, slices = global_array_from_single_device_arrays(sharding, arrays, return_slices=True, **kwargs)
    tmp = jax.jit(_identity_fn, out_shardings=sharding.with_spec(P()))(tmp).addressable_data(0)  # replicated accross all devices
    tmp = jnp.concatenate([tmp[sl] for sl in slices], axis=0)
    if return_slices:
        sizes = np.cumsum([0] + [sl.stop - sl.start for sl in slices])
        slices = [slice(start, stop) for start, stop in zip(sizes[:-1], sizes[1:])]
        return tmp, slices
    return tmp


def exchange_array(array, device, return_indices=False, pad=jnp.nan):
    # Exchange array along the first (0) axis
    # TODO: generalize to any axis
    sharding = array.sharding
    ndevices = sharding.num_devices
    per_host_arrays = [_.data for _ in array.addressable_shards]
    per_host_devices = [_.data for _ in device.addressable_shards]
    devices = sharding.mesh.devices.ravel().tolist()
    local_devices = [_.device for _ in per_host_arrays]
    per_host_final_arrays = [None] * len(local_devices)
    ndim = array.ndim
    per_host_indices = [[] for i in range(len(local_devices))]
    slices = [None] * ndevices

    for idevice in range(ndevices):
        # single-device arrays
        per_host_chunks = []
        for ilocal_device, (per_host_array, per_host_device, local_device) in enumerate(zip(per_host_arrays, per_host_devices, local_devices)):
            mask_idevice = per_host_device == idevice
            per_host_chunks.append(jax.device_put(per_host_array[mask_idevice], local_device, donate=True))
            if return_indices: per_host_indices[ilocal_device].append(jax.device_put(np.flatnonzero(mask_idevice), local_device, donate=True))
        tmp, slices[idevice] = allgather_single_device_arrays(sharding, per_host_chunks, return_slices=True)
        del per_host_chunks
        if devices[idevice] in local_devices:
            per_host_final_arrays[local_devices.index(devices[idevice])] = jax.device_put(tmp, devices[idevice], donate=True)
        del tmp

    final = global_array_from_single_device_arrays(sharding, per_host_final_arrays, pad=pad)
    if return_indices:
        for ilocal_device, local_device in enumerate(local_devices):
            per_host_indices[ilocal_device] = jax.device_put(jnp.concatenate(per_host_indices[ilocal_device]), local_device, donate=True)
        return final, per_host_indices, slices
    return final


def exchange_inverse(array, per_host_indices, slices):
    sharding = array.sharding
    ndevices = sharding.num_devices
    per_host_arrays = [_.data for _ in array.addressable_shards]
    devices = sharding.mesh.devices.ravel().tolist()
    local_devices = [_.device for _ in per_host_arrays]
    per_host_final_arrays = [None] * len(local_devices)

    for idevice in range(ndevices):
        per_host_chunks = []
        for ilocal_device, (per_host_array, local_device) in enumerate(zip(per_host_arrays, local_devices)):
            sl = slices[devices.index(local_device)][idevice]
            per_host_chunks.append(jax.device_put(per_host_array[sl], local_device, donate=True))
        tmp = allgather_single_device_arrays(sharding, per_host_chunks, return_slices=False)
        del per_host_chunks
        if devices[idevice] in local_devices:
            ilocal_device = local_devices.index(devices[idevice])
            indices = per_host_indices[ilocal_device]
            tmp = jax.device_put(tmp, devices[idevice], donate=True)
            tmp = jnp.empty_like(tmp).at[indices].set(tmp)
            per_host_final_arrays[local_devices.index(devices[idevice])] = jax.device_put(tmp, devices[idevice], donate=True)
        del tmp

    return global_array_from_single_device_arrays(sharding, per_host_final_arrays)


def test_exchange_array():
    from jax.experimental import mesh_utils, multihost_utils
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    def _create_device_index(shape, sharding_mesh=None):
        shard_shape = jax.sharding.NamedSharding(sharding_mesh, spec=P(sharding_mesh.axis_names,)).shard_shape(shape)
        def f(idevice):
            return jnp.full(shard_shape, idevice, dtype='i4')
        f = shard_map(f, mesh=sharding_mesh, in_specs=P(sharding_mesh.axis_names,), out_specs=P(sharding_mesh.axis_names,))
        return f(jnp.arange(sharding_mesh.devices.size))

    def _create_index(shape, sharding_mesh=None):
        sharding = jax.sharding.NamedSharding(sharding_mesh, spec=P(sharding_mesh.axis_names,))
        index = jnp.arange(shape[0])
        return jax.device_put(index, sharding)

    def allgather(array):
        return jax.jit(_identity_fn, out_shardings=array.sharding.with_spec(P(None)))(array).addressable_data(0)

    test = 0
    if test == 0:
        device_mesh_shape = (4,)
        devices = mesh_utils.create_device_mesh(device_mesh_shape)
        with jax.sharding.Mesh(devices, axis_names=('x',)) as sharding_mesh:
            nprocs = sharding_mesh.devices.size
            array = create_sharded_random(jax.random.uniform, jax.random.key(42), shape=(int(1024),))
            device = jnp.floor(array * nprocs).astype('i4')
            exchanged, *indices = exchange_array(array, device, pad=jnp.nan, return_indices=True)
            tmp = [_.data[~np.isnan(_.data)] for _ in exchanged.addressable_shards]
            print(jax.process_index(), [(tt.min(), tt.max()) for tt in tmp])
            array2 = exchange_inverse(exchanged, *indices)
            assert np.allclose(allgather(array2), allgather(array))

    if test == 2:
        device_mesh_shape = (4,)
        devices = mesh_utils.create_device_mesh(device_mesh_shape)
        with jax.sharding.Mesh(devices, axis_names=('x',)) as sharding_mesh:
            positions = jax.random.uniform(random.key(42), shape=(18, 3))
            positions = make_particles_from_local(positions)
            print(positions.shape)


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
    kinedges = np.linspace(0.001, 0.7, 30)
    kin = (kinedges[:-1] + kinedges[1:]) / 2.
    ells = (0, 2, 4)
    poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])
    theory = BinnedStatistic(x=[kin] * len(ells), edges=[np.array(list(zip(kinedges[:-1], kinedges[1:])))] * len(ells), value=poles, projs=ells)

    list_los = ['x', 'endpoint']

    meshsize = (32,) * 3
    with create_sharding_mesh(meshsize=meshsize) as mesh:

        attrs = MeshAttrs(meshsize=meshsize, boxsize=1000.)
        bin = bin_pspec(attrs, edges={'step': 0.01})
        ellsin = ells

        @partial(jax.jit, static_argnames=['los', 'ells'])
        def mock(seed, los='x', ells=(0, 2, 4)):
            mesh = generate_anisotropic_gaussian_mesh(attrs, theory, seed=seed, los=los, unitary_amplitude=True)
            return compute_mesh_pspec(mesh, ells=ells, los={'local': 'firstpoint'}.get(los, los), bin=bin)

        #mock(random.key(43), los='x')
        #mock(random.key(43), los='local')

        for flag in ['smooth', 'infinite'][1:]:
            for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')]:
                print(flag, los, thlos, flush=True)
                mean = compute_mesh_pspec_mean(attrs, theory=(theory, thlos) if thlos is not None else theory, ells=ells, bin=bin, los=los)
                wmatrix = compute_mesh_pspec_window(attrs, edgesin=kinedges, ellsin=(ellsin, thlos) if thlos is not None else ellsin, ells=ells, bin=bin, los=los, pbar=True, flags=(flag,))


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
    attrs = MeshAttrs(meshsize=[8, 16, 32], boxsize=1000.)
    per_host_positions = attrs.boxsize * random.uniform(random.key(rank), (size, len(attrs.boxsize))) - attrs.boxsize / 2. + attrs.boxcenter

    mesh = ParticleField(positions=per_host_positions, attrs=attrs).paint(resmpler='tsc', interlacing=3, compensate=True, pexchange=True)
    ells = (0, 2, 4)
    bin = bin_pspec(attrs, edges={'step': 0.01})
    compute_mesh_pspec(mesh, ells=ells, los='firstpoint', bin=bin)


def get_mock_fn(kind='data'):
    base_dir = Path('_tests')
    if kind == 'mesh':
        return base_dir / '{}.npz'.format(kind)
    return base_dir / '{}.npy'.format(kind)


def save_reference_mock():
    attrs = MeshAttrs(meshsize=(32, 64, 100), boxsize=1000., boxcenter=1200.)
    size = 128**2
    data_positions = generate_uniform_particles(attrs, size + 1, seed=42).positions
    randoms_positions = generate_uniform_particles(attrs, 4 * size + 1, seed=43).positions

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    f, b = 0.8, 1.5
    beta = f / b
    kinedges = np.linspace(0.001, 0.7, 30)
    kin = (kinedges[:-1] + kinedges[1:]) / 2.
    ells = (0, 2, 4)
    poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])
    theory = BinnedStatistic(x=[kin] * len(ells), edges=[kinedges] * len(ells), value=poles, projs=ells)
    mesh = generate_anisotropic_gaussian_mesh(attrs, theory, seed=random.key(42), los='x', unitary_amplitude=True)
    data_weights = mesh.read(data_positions).real

    utils.mkdir(get_mock_fn().parent)
    data = {'positions': data_positions, 'weights': data_weights}
    randoms = {'positions': randoms_positions, 'weights': jnp.ones_like(randoms_positions[..., 0])}
    np.save(get_mock_fn('data'), data, allow_pickle=True)
    np.save(get_mock_fn('randoms'), randoms, allow_pickle=True)

    data = ParticleField(**data, attrs=attrs)
    randoms = ParticleField(**randoms, attrs=attrs)
    fkp = FKPField(data, randoms)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mesh.save(get_mock_fn('mesh'))
    bin = bin_pspec(attrs, edges={'step': 0.01})
    power = compute_mesh_pspec(mesh, bin=bin, ells=(0, 2, 4), los='firstpoint')
    power.save(get_mock_fn('mesh_power'))
    power = compute_fkp_pspec(fkp, resampler='tsc', interlacing=3, bin=bin, ells=(0, 2, 4), los='firstpoint')
    power.save(get_mock_fn('fkp_power'))


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

    ref_mesh_power = PowerSpectrumMultipoles.load(get_mock_fn('mesh_power'))
    los = ref_mesh_power.attrs.pop('los')
    ref_mesh_power.attrs.pop('dtype')
    attrs = MeshAttrs(**ref_mesh_power.attrs)
    bin = bin_pspec(attrs, edges={'step': 0.01})
    ref_fkp_power = PowerSpectrumMultipoles.load(get_mock_fn('fkp_power'))

    with create_sharding_mesh(meshsize=attrs.meshsize) as sharding_mesh:
        #positions, weights = make_particles_from_local(**load(kind='data'))
        #jax.debug.inspect_array_sharding(positions, callback=print)
        #print(positions.shape)
        data = ParticleField(*make_particles_from_local(**load(kind='data')), attrs=attrs)
        randoms = ParticleField(*make_particles_from_local(**load(kind='randoms')), attrs=attrs)
        #data = ParticleField(**load(kind='data'), attrs=attrs)
        #randoms = ParticleField(**load(kind='randoms'), attrs=attrs)
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
        mesh_power = compute_mesh_pspec(mesh, bin=bin, ells=(0, 2, 4), los='firstpoint')
        fkp_power = compute_fkp_pspec(fkp, bin=bin, resampler='tsc', interlacing=3, ells=(0, 2, 4), los='firstpoint')
    diff = jnp.abs(mesh_power.view() - ref_mesh_power.view())
    print(diff)
    diff = jnp.abs(fkp_power.view() - ref_fkp_power.view())
    print(diff)
    assert np.allclose(mesh_power.view(), ref_mesh_power.view(), rtol=1e-6, atol=1e-2)
    assert np.allclose(fkp_power.view(), ref_fkp_power.view(), rtol=1e-6, atol=1e-2)
    #print(power.view().sum(), ref.view().sum(), diff, diff[~jnp.isnan(diff)].max())


def test_scaling():

    with create_sharding_mesh() as sharding_mesh:
        from jaxpower import MeshAttrs, create_sharded_random
        attrs = MeshAttrs(meshsize=600, boxsize=1000.)
        mesh = attrs.create(kind='real', fill=create_sharded_random(jax.random.normal, jax.random.key(42), shape=attrs.meshsize))

        @jax.jit
        def f(mesh):
            return mesh.r2c()

        jax.block_until_ready(f(mesh))
        n = 15
        t0 = time.time()
        for i in range(n): jax.block_until_ready(f(mesh + 0.001))
        print(time.time() - t0)


def test_scaling2():
    import jax
    from jax.sharding import PartitionSpec as P
    from jax.experimental import mesh_utils
    from jax.experimental.shard_map import shard_map
    import jaxdecomp

    device_mesh_shape = (4, 4)
    devices = mesh_utils.create_device_mesh(device_mesh_shape)

    with jax.sharding.Mesh(devices, axis_names=('x', 'y')) as sharding_mesh:

        def mesh_shard_shape(shape: tuple):
            return tuple(s // pdim for s, pdim in zip(shape, sharding_mesh.devices.shape)) + shape[sharding_mesh.devices.ndim:]

        shape = (512,) * 3
        key = jax.random.key(43)
        value = shard_map(partial(jax.random.normal, shape=mesh_shard_shape(shape), dtype='float32'), sharding_mesh, in_specs=P(), out_specs=P('x', 'y'))(key)  # yapf: disable

        @jax.jit
        def f(value):
            value = jax.lax.with_sharding_constraint(value, jax.sharding.NamedSharding(sharding_mesh, spec=P('x', 'y')))
            return jaxdecomp.fft.pfft3d(value)

        jax.block_until_ready(f(value))
        n = 15
        t0 = time.time()
        for i in range(n): jax.block_until_ready(f(value + 0.001))
        print(time.time() - t0)



if __name__ == '__main__':
    from jax import config
    config.update('jax_enable_x64', True)
    import warnings
    warnings.simplefilter("error")
    #print(jax.devices())
    #save_reference_mock()
    # Setting up distributed jax
    #jax.distributed.initialize()
    #test_halo()
    #test_exchange_array()
    #test_jaxdecomp()
    #test_mesh_power()
    #test_sharding()
    #test_mem()
    #compute_power_spectrum()
    #test_scaling2()
    # Closing distributed jax
    #jax.distributed.shutdown()