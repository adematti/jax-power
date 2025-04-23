import os
import time
from pathlib import Path
from functools import partial

import numpy as np

from mockfactory import Catalog, setup_logging


def get_clustering_positions_weights(catalog, zrange=None):
    from cosmoprimo.fiducial import DESI
    fiducial = DESI()
    catalog.get(catalog.columns())
    z = catalog['Z']
    weight = catalog['WEIGHT'] * catalog['WEIGHT_FKP']
    mask = Ellipsis
    if zrange:
        mask = (z >= zrange[0]) & (z <= zrange[1])
    dist = fiducial.comoving_radial_distance(z[mask])
    return np.column_stack([catalog['RA'][mask], catalog['DEC'][mask], dist]).astype(dtype='f8'), np.asarray(weight[mask]).astype(dtype='f8')


def get_mock_fn(kind='power'):
    base_dir = Path('_tests')
    return base_dir / '{}.npy'.format(kind)


def compute_jaxpower(fn, data_fn, all_randoms_fn, zrange=(0.4, 1.1), **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp_power, compute_fkp_normalization_and_shotnoise, create_sharding_mesh, make_particles_from_local, BinAttrs)

    data = Catalog.read(data_fn)
    randoms = Catalog.read(all_randoms_fn)
    data = get_clustering_positions_weights(data, zrange=zrange)
    randoms = get_clustering_positions_weights(randoms, zrange=zrange)
    data = ParticleField(*make_particles_from_local(*data), **attrs)
    randoms = ParticleField(*make_particles_from_local(*randoms), **attrs)
    fkp = FKPField(data, randoms)
    t0 = time.time()
    #norm, shotnoise_nonorm = compute_fkp_normalization_and_shotnoise(fkp)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    jax.block_until_ready(mesh)
    if jax.process_index() == 0: print(f'painting {time.time() - t0:.2f}')
    del fkp
    bin = BinAttrs(mesh.attrs, edges={'step': 0.001})
    t0 = time.time()
    power = jitted_compute_mesh_power(mesh, bin=bin)#.clone(norm=norm, shotnoise_nonorm=shotnoise_nonorm)
    jax.block_until_ready(power)
    if jax.process_index() == 0: print(f'power {time.time() - t0:.2f}')
    power.save(fn)


def compute_pypower(fn, data_fn, all_randoms_fn, zrange=(0.4, 1.1), **attrs):
    from pypower import CatalogFFTPower
    data = Catalog.read(data_fn)
    randoms = Catalog.read(all_randoms_fn)
    data_positions, data_weights = get_clustering_positions_weights(data, zrange=zrange)
    randoms_positions, randoms_weights = get_clustering_positions_weights(randoms, zrange=zrange)
    power = CatalogFFTPower(data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', resampler='tsc', interlacing=3, edges={'step': 0.001}, **poles_args, **attrs)
    power.save(fn)


if __name__ == '__main__':
    tracer = 'QSO'
    region = 'NGC'
    #cellsize, meshsize = 8., 1500 #1620
    #cellsize, meshsize = 20., 952
    cellsize, meshsize = 20., 750
    cutsky_args = dict(zrange=(0.8, 2.1), cellsize=cellsize, boxsize=cellsize * meshsize, boxcenter=(193.,  34.,  2806.))
    setup_logging()
    t0 = time.time()
    todo = 'jaxpower'
    #todo = 'pypower'

    poles_args = dict(ells=(0, 2, 4), los='firstpoint')

    if todo == 'jaxpower':
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        from jax import config
        config.update('jax_enable_x64', True)
        from jax import numpy as jnp
        #jax.distributed.initialize()
        from jaxpower import compute_mesh_power, create_sharding_mesh
        jitted_compute_mesh_power = jax.jit(partial(compute_mesh_power, **poles_args))
        jitted_compute_mesh_power = partial(compute_mesh_power, **poles_args)

    for imock in range(2):
        catalog_dir = Path(f'/global/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{imock:d}/mock{imock:d}/LSScats/')
        data_fn = catalog_dir / f'{tracer}_{region}_clustering.dat.fits'
        all_randoms_fn = [catalog_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(2)]

        if todo == 'jaxpower':
            fn = get_mock_fn(f'jaxpower_{imock:d}')
            if jax.process_count() > 1: 
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower(fn, data_fn, all_randoms_fn, **cutsky_args)
            else:
                compute_jaxpower(fn, data_fn, all_randoms_fn, **cutsky_args)

        if todo == 'pypower':
            compute_pypower(get_mock_fn(f'pypower_{imock:d}'), data_fn, all_randoms_fn, **cutsky_args)

    print('Elapsed time: {:.2f}'.format(time.time() - t0))

    if todo == 'jaxpower': jax.distributed.shutdown()