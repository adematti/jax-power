import os
import time
from pathlib import Path
from functools import partial

import numpy as np

from mockfactory import Catalog, setup_logging


def get_clustering_positions_weights(*fns, zrange=None):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)

    def get(fn):
        catalog = None
        if mpicomm.rank == 0:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            catalog = catalog[['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']]
        catalog = Catalog.scatter(catalog, mpicomm=mpicomm)
        z = catalog['Z']
        weight = catalog['WEIGHT'] * catalog['WEIGHT_FKP']
        mask = Ellipsis
        if zrange:
            mask = (z >= zrange[0]) & (z <= zrange[1])
        dist = fiducial.comoving_radial_distance(z[mask])
        return np.column_stack([catalog['RA'][mask], catalog['DEC'][mask], dist]).astype(dtype='f8'), np.asarray(weight[mask], dtype='f8')

    pw = [get(fn) for fn in fns]
    return np.concatenate([_[0] for _ in pw], axis=0), np.concatenate([_[1] for _ in pw], axis=0)


def get_clustering_positions_weights2(*fns, zrange=None):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)

    def get(fns):
        catalog = Catalog.read(fns, mpicomm=mpicomm)
        catalog.get(catalog.columns())  # Faster to read all columns at once
        z = catalog['Z']
        weight = catalog['WEIGHT'] * catalog['WEIGHT_FKP']
        mask = Ellipsis
        if zrange:
            mask = (z >= zrange[0]) & (z <= zrange[1])
        dist = fiducial.comoving_radial_distance(z[mask])
        return np.column_stack([catalog['RA'][mask], catalog['DEC'][mask], dist]).astype(dtype='f8'), np.asarray(weight[mask], dtype='f8')

    return get(fns)


def get_mock_fn(kind='power'):
    base_dir = Path('_tests')
    return base_dir / '{}.npy'.format(kind)


def compute_jaxpower(fn, data_fn, all_randoms_fn, zrange=(0.4, 1.1), ells=(0, 2, 4), **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp2_spectrum, compute_fkp2_spectrum_normalization, compute_fkp2_spectrum_shotnoise, create_sharding_mesh, make_particles_from_local, BinMesh2Spectrum)
    t0 = time.time()
    data = get_clustering_positions_weights(data_fn, zrange=zrange)
    randoms = get_clustering_positions_weights(*all_randoms_fn, zrange=zrange)
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    mpicomm.barrier()
    t1 = time.time()
    data = ParticleField(*make_particles_from_local(*data), **attrs)
    randoms = ParticleField(*make_particles_from_local(*randoms), **attrs)
    fkp = FKPField(data, randoms)
    t2 = time.time()
    norm, num_shotnoise = compute_fkp2_spectrum_normalization(fkp), compute_fkp2_spectrum_shotnoise(fkp)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    jax.block_until_ready(mesh)
    t3 = time.time()
    del fkp
    bin = BinMesh2Spectrum(mesh.attrs, edges={'step': 0.001}, ells=ells)
    power = jitted_compute_mesh2_spectrum(mesh, bin=bin)#.clone(norm=norm, num_shotnoise=num_shotnoise)
    power.attrs.udpate(mesh=dict(mesh.attrs), zrange=zrange)
    jax.block_until_ready(power)
    t4 = time.time()
    if jax.process_index() == 0:
        print(f'reading {t1 - t0:.2f} fkp {t2 - t1:.2f} painting {t3 - t2:.2f} power {t4 - t3:.2f}')
    power.save(fn)


def compute_jaxpower_window(fn, power_fn, data_fn, all_randoms_fn):
    from jaxpower import (ParticleField, FKPField, compute_mesh2_spectrum_window, create_sharding_mesh, make_particles_from_local, BinMesh2Spectrum, BinnedStatistic, MeshAttrs)
    power = BinnedStatistic.load(power_fn)
    zrange = power.attrs['zrange']
    attrs = MeshAttrs(**power.attrs['mesh'])
    data = get_clustering_positions_weights(data_fn, zrange=zrange)
    randoms = get_clustering_positions_weights(*all_randoms_fn, zrange=zrange)
    data = ParticleField(*make_particles_from_local(*data), **attrs)
    randoms = ParticleField(*make_particles_from_local(*randoms), **attrs)
    mesh = randoms.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    bin = BinMesh2Spectrum(mesh.attrs, edges=power.edges(projs=0))
    ells = power.projs()
    edgesin = np.linspace(bin.edges.min(), bin.edges.max(), 2 * (len(bin.edges) - 1))
    wmatrix = compute_mesh2_spectrum_window(mesh, edgesin=edgesin, ellsin=(ells, 'local'), bin=bin, ells=ells, pbar=True)
    wmatrix.save(fn)


def compute_pypower(fn, data_fn, all_randoms_fn, zrange=(0.4, 1.1), **attrs):
    from pypower import CatalogFFTPower
    t0 = time.time()
    data_positions, data_weights = get_clustering_positions_weights2(data_fn, zrange=zrange)
    randoms_positions, randoms_weights = get_clustering_positions_weights2(*all_randoms_fn, zrange=zrange)
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    mpicomm.barrier()
    t1 = time.time()
    power = CatalogFFTPower(data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', resampler='tsc', interlacing=3, edges={'step': 0.001}, **poles_args, **attrs, wnorm=1., num_shotnoise=0.)
    t2 = time.time()
    if mpicomm.rank == 0:
        print(f'reading {t1 - t0:.2f} power {t2 - t1:.2f}')
    power.save(fn)


if __name__ == '__main__':

    tracer = 'QSO'
    region = 'NGC'
    #cellsize, meshsize = 8., 1500 #1620
    cellsize, meshsize = 20., 1024  # 40GB nodes
    #cellsize, meshsize = 20., 950
    #cellsize, meshsize = 20., 750
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
        jax.distributed.initialize()
        from jaxpower import compute_mesh2_spectrum, create_sharding_mesh
        jitted_compute_mesh2_spectrum = jax.jit(partial(compute_mesh2_spectrum, **poles_args), donate_argnums=[0])
        #jitted_compute_mesh2_spectrum = partial(compute_mesh2_spectrum, **poles_args)

    for imock in range(4):
        catalog_dir = Path(f'/dvs_ro/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{imock:d}/mock{imock:d}/LSScats/')
        data_fn = catalog_dir / f'{tracer}_{region}_clustering.dat.fits'
        all_randoms_fn = [catalog_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(10)]

        if todo == 'jaxpower':
            fn = get_mock_fn(f'jaxpower_{imock:d}')
            if jax.process_count() > 1:
                with create_sharding_mesh() as sharding_mesh:
                    power = compute_jaxpower(fn, data_fn, all_randoms_fn, **cutsky_args)
                    jax.block_until_ready(power)
            else:
                power = compute_jaxpower(fn, data_fn, all_randoms_fn, **cutsky_args)
                jax.block_until_ready(power)

        if todo == 'pypower':
            compute_pypower(get_mock_fn(f'pypower_{imock:d}'), data_fn, all_randoms_fn, **cutsky_args)

    print('Elapsed time: {:.2f}'.format(time.time() - t0))

    if todo == 'jaxpower': jax.distributed.shutdown()