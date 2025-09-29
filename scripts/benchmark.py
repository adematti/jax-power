import os
import time
import logging
from pathlib import Path
from functools import partial

import numpy as np

from mockfactory import Catalog, sky_to_cartesian, setup_logging


logger = logging.getLogger('benchmark')


def select_region(ra, dec, region=None):
    # print('select', region)
    if region in [None, 'ALL', 'GCcomb']:
        return np.ones_like(ra, dtype='?')
    mask_ngc = (ra > 100 - dec)
    mask_ngc &= (ra < 280 + dec)
    mask_n = mask_ngc & (dec > 32.375)
    mask_s = (~mask_n) & (dec > -25.)
    if region == 'NGC':
        return mask_ngc
    if region == 'SGC':
        return ~mask_ngc
    if region == 'N':
        return mask_n
    if region == 'S':
        return mask_s
    if region == 'SNGC':
        return mask_ngc & mask_s
    if region == 'SSGC':
        return (~mask_ngc) & mask_s
    if footprint is None: load_footprint()
    north, south, des = footprint.get_imaging_surveys()
    mask_des = des[hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)]
    if region == 'DES':
        return mask_des
    if region == 'SnoDES':
        return mask_s & (~mask_des)
    if region == 'SSGCnoDES':
        return (~mask_ngc) & mask_s & (~mask_des)
    raise ValueError('unknown region {}'.format(region))


def get_proposal_mattrs(tracer):
    if 'BGS' in tracer:
        mattrs = dict(boxsize=4000., cellsize=7)
    elif 'LRG' in tracer:
        mattrs = dict(boxsize=7000., cellsize=7)
    elif 'LRG+ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'ELG' in tracer:
        mattrs = dict(boxsize=9000., cellsize=10)
    elif 'QSO' in tracer:
        mattrs = dict(boxsize=10000., cellsize=10)
    else:
        raise NotImplementedError(f'tracer {tracer} is unknown')
    mattrs.update(cellsize=40)
    return mattrs


def get_clustering_rdzw(*fns, zrange=None, region=None, tracer=None, **kwargs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            for name in ['WEIGHT', 'WEIGHT_FKP']:
                if name not in catalog: catalog[name] = catalog.ones()
            if tracer is not None and 'Z' not in catalog:
                catalog['Z'] = catalog[f'Z_{tracer}']
            catalog = catalog[['RA', 'DEC', 'Z', 'WEIGHT', 'WEIGHT_FKP']]
            if zrange is not None:
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] <= zrange[1])
                catalog = catalog[mask]
            if region is not None:
                mask = select_region(catalog['RA'], catalog['DEC'], region)
                catalog = catalog[mask]
            catalogs[ifn] = (irank, catalog)

    rdzw = []
    for irank, catalog in catalogs:
        if mpicomm.size > 1:
            catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=irank)
        weight = catalog['WEIGHT'] #* catalog['WEIGHT_FKP']
        rdzw.append([catalog['RA'], catalog['DEC'], catalog['Z'], weight])
    return [np.concatenate([arrays[i] for arrays in rdzw], axis=0) for i in range(4)]


def get_clustering_positions_weights(*fns, **kwargs):
    from cosmoprimo.fiducial import TabulatedDESI, DESI
    fiducial = TabulatedDESI()  # faster than DESI/class (which takes ~30 s for 10 random catalogs)
    ra, dec, z, weights = get_clustering_rdzw(*fns, **kwargs)
    weights = np.asarray(weights, dtype='f8')
    dist = fiducial.comoving_radial_distance(z)
    positions = sky_to_cartesian(dist, ra, dec, dtype='f8')
    return positions, weights


def get_measurement_fn(kind='mesh2_spectrum_poles', tracer='LRG', region='NGC', zrange=(0.8, 1.1), cut=False, imock=0, **kwargs):
    if imock is None:
        import glob
        return sorted(glob.glob(get_measurement_fn(kind=kind, tracer=tracer, region=region, zrange=zrange, imock='*')))
    base_dir = Path('_tests')
    return base_dir / f'{kind}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_{imock}{"_thetacut" if cut else ""}.h5'


def compute_jaxpower_mesh2_spectrum(fn, get_data, get_randoms, cut=None, ells=(0, 2, 4), los='firstpoint', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, create_sharding_mesh, make_particles_from_local, exchange_particles, BinMesh2SpectrumPoles, BinParticle2CorrelationPoles, BinParticle2SpectrumPoles, compute_particle2, get_mesh_attrs)
    t0 = time.time()
    data = get_data()
    randoms = get_randoms()
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    mpicomm.barrier()
    t1 = time.time()
    mattrs = get_mesh_attrs(data[0], randoms[0], **attrs)
    #attrs = get_mesh_attrs(data[0], **attrs)
    kw = dict(exchange=True, backend='mpi')
    data = ParticleField(*data, attrs=mattrs, **kw)
    randoms = ParticleField(*randoms, attrs=mattrs, **kw)
    fkp = FKPField(data, randoms)
    #del data, randoms
    t2 = time.time()
    if cut is not None:
        bin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, selection={'theta': (0., 0.05)}, ells=ells)
        #bin = BinParticle2SpectrumPoles(attrs, edges=np.linspace(0.01, 0.1, 10), selection={'theta': (0., 0.05)}, ells=ells)
        cut = compute_particle2(fkp.particles, bin=bin, los=los)
        #print(cut.clone(num_shotnoise=None).view().sum(), cut.view().sum(), attrs.meshsize)
    t3 = time.time()
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    norm, num_shotnoise = compute_fkp2_normalization(fkp, bin=bin), compute_fkp2_shotnoise(fkp, bin=bin)
    #f = (data - data.sum() / randoms.sum() * randoms).clone(attrs=data.attrs)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    jax.block_until_ready(mesh)
    t4 = time.time()
    del fkp
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    jax.block_until_ready(spectrum)
    t5 = time.time()
    if cut is not None:
        cut = cut.to_spectrum(spectrum)
        spectrum = spectrum.clone(value=spectrum.value() - cut.value())
    if jax.process_index() == 0:
        logger.info(f'reading {t1 - t0:.2f} exchange {t2 - t1:.2f} theta-cut {t3 - t2:.2f} painting {t4 - t3:.2f} spectrum {t5 - t4:.2f}')
    spectrum.write(fn)


def compute_pypower_mesh2_spectrum(output_fn, get_data, get_randoms, cut=None, ells=(0, 2, 4), los='firstpoint', **attrs):
    from pypower import CatalogFFTPower
    from lsstypes.external import from_pypower
    t0 = time.time()
    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    mpicomm.barrier()
    t1 = time.time()
    direct_selection_attrs = {'theta': (0., 0.05)}
    direct_edges = {'min': 0., 'step': 0.1}
    direct_attrs = {'nthreads': mpicomm.size}
    if cut is None:
        direct_selection_attrs = direct_edges = direct_attrs = None
    spectrum = CatalogFFTPower(data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', resampler='tsc', interlacing=3, edges={'step': 0.001}, ells=ells, los=los, **attrs, direct_selection_attrs=direct_selection_attrs, direct_edges=direct_edges, direct_attrs=direct_attrs).poles
    t2 = time.time()
    if mpicomm.rank == 0:
        logger.info(f'reading {t1 - t0:.2f} spectrum {t2 - t1:.2f}')
    from_pypower(spectrum).write(output_fn)


def compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_randoms, basis='scoccimarro', ells=[0, 2], los='local', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum)
    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01}, basis=basis, ells=ells) #, buffer_size=4)
    #norm = compute_fkp3_normalization(fkp, bin=bin, cellsize=None)
    norm = compute_fkp3_normalization(fkp, split=42, bin=bin, cellsize=None) #10)
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    kw = dict(resampler='cic', interlacing=False, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(fkp, los=los, bin=bin, **kw)
    mesh = fkp.paint(**kw, out='real')
    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
    if jax.process_index() == 0:
        logger.info(f'Writing to {output_fn}')
        spectrum.write(output_fn)


def compute_triumvirate_mesh3_spectrum(output_fn, get_data, get_randoms, ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama', los='local', boxsize=10000., cellsize=10.):
    from lsstypes.external import from_triumvirate
    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_bispec
    from triumvirate.parameters import ParameterSet
    from triumvirate.logger import setup_logger

    logger = setup_logger(20)
    data_positions, data_weights = get_data()
    data = ParticleCatalogue(*np.array(data_positions.T), ws=data_weights, nz=np.ones_like(data_weights))
    randoms_positions, randoms_weights = get_randoms()
    randoms = ParticleCatalogue(*np.array(randoms_positions.T), ws=randoms_weights, nz=np.ones_like(randoms_weights))

    boxsize = boxsize * np.ones(3, dtype=float)
    cellsize = cellsize * np.ones(3, dtype=float)
    meshsize = np.ceil(boxsize / cellsize).astype(int)
    boxsize = meshsize * cellsize
    edges = np.arange(0., np.pi / cellsize.max(), 0.01)

    results = []
    for ell in ells:
        paramset = dict(norm_convention='mesh', form='diag' if 'diagonal' in basis else 'full', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None), range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='cic', interlace='off', alignment='centre', padfactor=0., boxsize=dict(zip('xyz', boxsize)), ngrid=dict(zip('xyz', meshsize)), verbose=20)

        paramset = ParameterSet(param_dict=paramset)
        results.append(compute_bispec(data, randoms, paramset=paramset, logger=logger))

    spectrum = from_triumvirate(results, ells=ells)
    logger.info(f'Writing to {output_fn}')
    spectrum.write(output_fn)


def get_catalog_fn(catalog='data', tracer='LRG', region='NGC', zrange=(0.8, 1.1), nran=1, imock=0, **kwargs):
    catalog_dir = Path(f'/dvs_ro/cfs/cdirs/desi//survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit_v4_2/altmtl{imock:d}/mock{imock:d}/LSScats/')
    if catalog == 'data':
        return catalog_dir / f'{tracer}_{region}_clustering.dat.fits'
    if catalog == 'randoms':
        return [catalog_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(10)][:1]
    raise ValueError('issue with input args')


if __name__ == '__main__':

    #cellsize, meshsize = 10., 1024  # 40GB nodes
    #catalog_args = dict(tracer='QSO', zrange=(0.8, 2.1), region='NGC')
    catalog_args = dict(tracer='LRG', zrange=(0.8, 1.1), region='NGC')
    setup_logging()
    t0 = time.time()
    todo = 'mesh2_spectrum'
    todo = 'mesh2_spectrum_pypower'
    todo = 'mesh3_spectrum_sugiyama'
    #todo = 'mesh3_spectrum_sugiyama_triumvirate'

    ells = (0, 2, 4)
    los = 'firstpoint'
    with_jax = todo in ['mesh2_spectrum', 'mesh3_spectrum_sugiyama']

    if with_jax:
        #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        from jax import config
        config.update('jax_enable_x64', True)
        from jax import numpy as jnp
        jax.distributed.initialize()
        from jaxpower import compute_mesh2_spectrum, create_sharding_mesh
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])

    for imock in range(1):
        data_fn = get_catalog_fn(imock=imock, catalog='data', **catalog_args)
        all_randoms_fn = get_catalog_fn(imock=imock, catalog='randoms', **catalog_args)
        get_data = lambda: get_clustering_positions_weights(data_fn, **catalog_args)
        get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, **catalog_args)

        spectrum_args = dict(**get_proposal_mattrs(catalog_args['tracer']), ells=(0, 2, 4), cut=None)

        if todo == 'mesh2_spectrum':
            output_fn = get_measurement_fn(imock=imock, **catalog_args, **spectrum_args, kind='mesh2_spectrum_poles')
            with create_sharding_mesh() as sharding_mesh:
                spectrum = compute_jaxpower_mesh2_spectrum(output_fn, get_data, get_randoms, **spectrum_args)

        if todo == 'mesh2_spectrum_pypower':
            output_fn = get_measurement_fn(imock=imock, **catalog_args, **spectrum_args, kind='mesh2_spectrum_poles_pypower')
            compute_pypower_mesh2_spectrum(output_fn, get_data, get_randoms, **spectrum_args)

        if todo == 'mesh3_spectrum_sugiyama':
            bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)][:1], cellsize=30)
            bispectrum_args.pop('cut')
            output_fn = get_measurement_fn(imock=imock, **catalog_args, kind=f'mesh3_spectrum_poles_{bispectrum_args["basis"]}')
            with create_sharding_mesh() as sharding_mesh:
                compute_jaxpower_mesh3_spectrum(output_fn, get_data, get_randoms, **bispectrum_args)
                jax.clear_caches()

        if todo == 'mesh3_spectrum_sugiyama_triumvirate':
            bispectrum_args = spectrum_args | dict(basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], cellsize=30)
            bispectrum_args.pop('cut')
            output_fn = get_measurement_fn(imock=imock, **catalog_args, kind=f'mesh3_spectrum_poles_{bispectrum_args["basis"]}_triumvirate')
            compute_triumvirate_mesh3_spectrum(output_fn, get_data, get_randoms, **bispectrum_args)

    
    print('Elapsed time: {:.2f}'.format(time.time() - t0))

    if with_jax: jax.distributed.shutdown()
