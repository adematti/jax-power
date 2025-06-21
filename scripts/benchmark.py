import os
import time
from pathlib import Path
from functools import partial

import numpy as np

from mockfactory import Catalog, sky_to_cartesian, setup_logging


def get_clustering_positions_weights(*fns, zrange=None):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            catalog = catalog[['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']]
            if zrange:
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    positions, weights = [], []
    for irank, catalog in catalogs:
        catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        weight = catalog['WEIGHT'] * catalog['WEIGHT_FKP']
        dist = fiducial.comoving_radial_distance(catalog['Z'])
        positions.append(sky_to_cartesian(dist, catalog['RA'], catalog['DEC'], dtype='f8'))
        weights.append(np.asarray(weight, dtype='f8'))
    return np.concatenate(positions, axis=0), np.concatenate(weights, axis=0)



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
        return sky_to_cartesian(dist, catalog['RA'], catalog['DEC'], dtype='f8'), np.asarray(weight[mask], dtype='f8')

    return get(fns)


def get_mock_fn(kind='power'):
    base_dir = Path('_tests')
    return base_dir / '{}.npy'.format(kind)


def compute_jaxpower(fn, data_fn, all_randoms_fn, zrange=(0.4, 1.1), ells=(0, 2, 4), los='firstpoint', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp2_spectrum, compute_fkp2_spectrum_normalization, compute_fkp2_spectrum_shotnoise, create_sharding_mesh, make_particles_from_local, exchange_particles, BinMesh2Spectrum, get_mesh_attrs)
    t0 = time.time()
    data = get_clustering_positions_weights(data_fn, zrange=zrange)
    randoms = get_clustering_positions_weights(*all_randoms_fn, zrange=zrange)
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    mpicomm.barrier()
    t1 = time.time()

    attrs = get_mesh_attrs(data[0], randoms[0], **attrs)
    data = ParticleField(*data, attrs=attrs, exchange=True, backend='mpi')
    randoms = ParticleField(*randoms, attrs=attrs,  exchange=True, backend='mpi')
    fkp = FKPField(data, randoms)
    t2 = time.time()
    #norm, num_shotnoise = compute_fkp2_spectrum_normalization(fkp), compute_fkp2_spectrum_shotnoise(fkp)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=False, out='real', pexchange=False)
    jax.block_until_ready(mesh)
    t3 = time.time()
    del fkp
    bin = BinMesh2Spectrum(mesh.attrs, edges={'step': 0.001}, ells=ells)
    power = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los)#.clone(norm=norm, num_shotnoise=num_shotnoise)
    power.attrs.update(mesh=dict(mesh.attrs), zrange=zrange)
    jax.block_until_ready(power)
    t4 = time.time()
    if False:
        from jaxpower import BinParticle2Correlation, compute_particle2
        bin = BinParticle2Correlation(mesh.attrs, edges={'step': 0.1}, selection={'theta': (0., 0.05)}, ells=ells)
        corr = compute_particle2(data, bin=bin, los=los)
        power = power.clone(num=corr.to_power(power).num)
    t5 = time.time()

    if jax.process_index() == 0:
        print(f'reading {t1 - t0:.2f} fkp {t2 - t1:.2f} painting {t3 - t2:.2f} power {t4 - t3:.2f} theta-cut {t5 - t4:.2f}')
    power.save(fn)


def compute_thetacut(fn, data_fn, all_randoms_fn, zrange=(0.4, 1.1), ells=(0, 2, 4), los='firstpoint', **attrs):
    t0 = time.time()
    data_positions, data_weights = get_clustering_positions_weights2(data_fn, zrange=zrange)
    randoms_positions, randoms_weights = get_clustering_positions_weights2(*all_randoms_fn, zrange=zrange)
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    mpicomm.barrier()
    t1 = time.time()

    positions = np.concatenate([data_positions, randoms_positions], axis=0)
    weights = np.concatenate([data_weights, - data_weights.sum() / randoms_weights.sum() * randoms_weights], axis=0)

    rnorm = np.sqrt(np.sum(positions**2, axis=-1))
    ctheta = np.arccos(positions[..., 2] / rnorm) * 180. / np.pi
    #smax = np.sqrt(3) * 2e4
    smax = 35472.400001

    import sys
    sys.path.insert(0, '/global/u2/a/adematti/cosmodesi/cucount/build')
    try:
        from cucount import count2, Particles, BinAttrs, SelectionAttrs
    except ImportError:
        raise
    particles1 = Particles(positions, weights)
    particles2 = Particles(positions, weights)
    print(particles1.size)
    battrs = BinAttrs(s=(0., smax, 0.1), pole=(0, 5, 2, los))
    sattrs = SelectionAttrs(theta=(0., 0.05))
    counts = count2(particles1, particles2, battrs=battrs, sattrs=sattrs)
    np.save(fn, counts)


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
    bin = BinMesh2Spectrum(mesh.attrs, edges=power.edges(projs=0), ells=power.projs)
    ells = power.projs()
    edgesin = np.linspace(bin.edges.min(), bin.edges.max(), 2 * (len(bin.edges) - 1))
    wmatrix = compute_mesh2_spectrum_window(mesh, edgesin=edgesin, ellsin=(ells, 'local'), bin=bin, pbar=True)
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
    direct_selection_attrs = {'theta': (0., 0.05)}
    direct_edges = {'min': 0., 'step': 0.1}
    direct_attrs = {'nthreads': mpicomm.size}
    #direct_selection_attrs = direct_edges = direct_attrs = None
    power = CatalogFFTPower(data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', resampler='tsc', interlacing=3, edges={'step': 0.001}, **attrs, wnorm=1., shotnoise_nonorm=0., direct_selection_attrs=direct_selection_attrs, direct_edges=direct_edges, direct_attrs=direct_attrs)
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
    #cellsize, meshsize = 80., 256
    cutsky_args = dict(zrange=(0.8, 2.1), cellsize=cellsize, boxsize=cellsize * meshsize, boxcenter=(193.,  34.,  2806.), ells=(0, 2, 4), los='firstpoint')
    setup_logging()
    t0 = time.time()
    #todo = 'jaxpower'
    #todo = 'pypower'
    todo = 'thetacut'

    ells = (0, 2, 4)
    los = 'firstpoint'

    if todo == 'jaxpower':
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        from jax import config
        config.update('jax_enable_x64', True)
        from jax import numpy as jnp
        jax.distributed.initialize()
        from jaxpower import compute_mesh2_spectrum, create_sharding_mesh
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])

    for imock in range(1):
        catalog_dir = Path(f'/dvs_ro/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{imock:d}/mock{imock:d}/LSScats/')
        data_fn = catalog_dir / f'{tracer}_{region}_clustering.dat.fits'
        all_randoms_fn = [catalog_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(10)][:5]

        if todo == 'jaxpower':
            fn = get_mock_fn(f'jaxpower_{imock:d}')
            if jax.process_count() > 1:
                with create_sharding_mesh() as sharding_mesh:
                    power = compute_jaxpower(fn, data_fn, all_randoms_fn, **cutsky_args)
                    jax.block_until_ready(power)
            else:
                power = compute_jaxpower(fn, data_fn, all_randoms_fn, **cutsky_args)
                jax.block_until_ready(power)

        if todo == 'thetacut':
            fn = get_mock_fn(f'thetacut_{imock:d}')
            power = compute_thetacut(fn, data_fn, all_randoms_fn, **cutsky_args)
            print(power)

        if todo == 'pypower':
            compute_pypower(get_mock_fn(f'pypower_{imock:d}'), data_fn, all_randoms_fn, **cutsky_args)

    print('Elapsed time: {:.2f}'.format(time.time() - t0))

    if todo == 'jaxpower': jax.distributed.shutdown()