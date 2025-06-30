import os
import time
from pathlib import Path

import numpy as np

from mockfactory import Catalog, sky_to_cartesian, setup_logging

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


def get_clustering_rdzw(*fns, zrange=None, region=None, tracer=None):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalogs = [None] * len(fns)
    for ifn, fn in enumerate(fns):
        irank = ifn % mpicomm.size
        catalogs[ifn] = (irank, None)
        if mpicomm.rank == irank:  # Faster to read catalogs from one rank
            catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
            catalog.get(catalog.columns())  # Faster to read all columns at once
            if 'WEIGHT_FKP' not in catalog: catalog['WEIGHT_FKP'] = catalog.ones()
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
        weight = catalog['WEIGHT'] * catalog['WEIGHT_FKP']
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


def get_data_fn(zsnap=0.950, imock=0):
    sznap = f'{zsnap:.3f}'.replace('.', 'p')
    szrange = '0p8to1p6'
    return f'/dvs_ro/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/AbacusSummit_base_c000_ph{imock:03d}/CutSky/ELG_v5/z{zsnap:.3f}/forclustering/cutsky_abacusHF_DR2_ELG_LOP_z{sznap}_zcut_{szrange}_clustering.dat.fits'


def get_randoms_fn(iran=0):
    return f'/dvs_ro/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/randoms/rands_intiles_DARK_nomask_{iran:d}.fits'


def get_measurement_fn(zsnap=0.950, imock=0, kind='mesh2spectrum'):
    sznap = f'{zsnap:.3f}'.replace('.', 'p')
    szrange = '0p8to1p6'
    return f'/global/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/desipipe_test/AbacusSummit_base_c000_ph{imock:03d}/CutSky/ELG_v5/{kind}_abacusHF_DR2_ELG_LOP_z{sznap}_zcut_{szrange}.npy'


def compute_jaxpower(output_fn, data_fn, all_randoms_fn, tracer='ELG_LOP', region='NGC', zrange=(0.8, 1.1), ells=(0, 2, 4), los='firstpoint', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp2_spectrum_normalization, compute_fkp2_spectrum_shotnoise, BinMesh2Spectrum, get_mesh_attrs)
    t0 = time.time()
    data = get_clustering_positions_weights(data_fn, zrange=zrange, tracer=tracer, region=region)
    randoms = get_clustering_positions_weights(*all_randoms_fn, zrange=zrange, tracer=tracer, region=region)
    attrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=attrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=attrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    norm, num_shotnoise = compute_fkp2_spectrum_normalization(fkp), compute_fkp2_spectrum_shotnoise(fkp)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    del fkp
    bin = BinMesh2Spectrum(mesh.attrs, edges={'step': 0.001}, ells=ells)
    power = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    power.attrs.update(mesh=dict(mesh.attrs), zrange=zrange)
    jax.block_until_ready(power)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Done in {t1 - t0:.2f}')
    power.save(output_fn)


def compute_thetacut(output_fn, data_fn, all_randoms_fn, tracer='ELG_LOP', region='NGC', zrange=(0.4, 1.1), ells=(0, 2, 4), los='firstpoint', power_fn=None, output_power_fn=None, **attrs):
    from jaxpower import get_mesh_attrs, ParticleField, FKPField, BinParticle2Correlation, compute_particle2, Spectrum2Poles
    data = get_clustering_positions_weights(data_fn, zrange=zrange, region=region)
    randoms = get_clustering_positions_weights(*all_randoms_fn, zrange=zrange, tracer=tracer, region=region)

    attrs = get_mesh_attrs(data[0], randoms[0], **attrs)
    data = ParticleField(*data, attrs=attrs)
    randoms = ParticleField(*randoms, attrs=attrs)
    fkp = FKPField(data, randoms)
    bin = BinParticle2Correlation(attrs, edges={'step': 0.1}, selection={'theta': (0., 0.05)}, ells=ells)
    cut = compute_particle2(fkp.particles, bin=bin, los=los)
    cut.save(output_fn)
    if power_fn is not None and output_power_fn is not None:
        power = Spectrum2Poles.load(power_fn)
        cut = cut.to_spectrum(power)
        power = power.clone(num=[power.num[iproj] - cut.num[iproj] for iproj in range(len(power.projs))])
        power.save(output_power_fn)


def compute_pypower(fn, data_fn, all_randoms_fn, tracer='ELG_LOP', region='NGC', zrange=(0.4, 1.1), **attrs):
    from pypower import CatalogFFTPower
    data_positions, data_weights = get_clustering_positions_weights(data_fn, zrange=zrange, tracer=tracer, region=region)
    randoms_positions, randoms_weights = get_clustering_positions_weights(*all_randoms_fn, zrange=zrange, tracer=tracer, region=region)
    direct_selection_attrs = {'theta': (0., 0.05)}
    direct_edges = {'min': 0., 'step': 0.1}
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    direct_attrs = {'nthreads': mpicomm.size}
    direct_selection_attrs = direct_edges = direct_attrs = None
    power = CatalogFFTPower(data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', resampler='tsc', interlacing=3, edges={'step': 0.001}, **attrs, direct_selection_attrs=direct_selection_attrs, direct_edges=direct_edges, direct_attrs=direct_attrs)
    power.save(fn)
        

if __name__ == '__main__':

    tracer = 'ELG_LOP'
    region = 'NGC'
    cellsize, meshsize = 10., 1024  # 40GB nodes
    #cellsize, meshsize = 40., 256
    cutsky_args = dict(zrange=(0.8, 2.1), cellsize=cellsize, boxsize=cellsize * meshsize, ells=(0, 2, 4), tracer=tracer, region=region, los='firstpoint')
    setup_logging()
    t0 = time.time()
    #todo = 'jaxpower'
    #todo = 'thetacut'
    todo = 'pypower'

    ells = (0, 2, 4)
    los = 'firstpoint'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    if todo == 'jaxpower':
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        from jax import config
        config.update('jax_enable_x64', True)
        from jax import numpy as jnp
        jax.distributed.initialize()
        from jaxpower import compute_mesh2_spectrum, create_sharding_mesh
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])

    for imock in range(1):
        data_fn = get_data_fn(imock=imock)
        all_randoms_fn = [get_randoms_fn(iran=iran) for iran in range(4)]

        if todo == 'jaxpower':
            output_fn = get_measurement_fn(imock=imock, kind='mesh2spectrum')
            if jax.process_count() > 1:
                with create_sharding_mesh() as sharding_mesh:
                    compute_jaxpower(output_fn, data_fn, all_randoms_fn, **cutsky_args)
            else:
                compute_jaxpower(fn, data_fn, all_randoms_fn, **cutsky_args)

        if todo == 'thetacut':
            power_fn = get_measurement_fn(imock=imock, kind='mesh2spectrum')
            output_power_fn = get_measurement_fn(imock=imock, kind='mesh2spectrum_thetacut')
            output_fn = get_measurement_fn(imock=imock, kind='thetacut')
            compute_thetacut(output_fn, data_fn, all_randoms_fn, power_fn=power_fn, output_power_fn=output_power_fn, **cutsky_args)

        if todo == 'pypower':
            output_fn = get_measurement_fn(imock=imock, kind='pypower')
            compute_pypower(output_fn, data_fn, all_randoms_fn, **cutsky_args)

    if todo in ['jaxpower']: jax.distributed.shutdown()