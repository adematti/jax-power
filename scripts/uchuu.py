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
            for name in catalog.columns():
                array = catalog[name]
                del catalog[name]
                catalog[name.upper()] = array
            for name in ['WEIGHT', 'WEIGHT_FKP']:
                if name not in catalog: catalog[name] = catalog.ones()
            if tracer is not None and 'Z' not in catalog:
                for name in ['ZOBS', f'Z_{tracer}']:
                    if name in catalog:
                        catalog['Z'] = catalog[name]
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


def test_radec(*fns):
    import fitsio
    ra, dec = [], []
    for fn in fns:
        cat = fitsio.read(fn)
        ra.append(cat['RA'])
        dec.append(cat['DEC'])
    ra, dec = np.concatenate(ra), np.concatenate(dec)
    # Define structured dtype
    dtype = [('RA', 'f8'), ('DEC', 'f8')]
    # Create structured array
    structured = np.zeros(ra.shape[0], dtype=dtype)
    structured['RA'] = ra
    structured['DEC'] = dec
    unique = np.unique(structured).size
    print('Fraction of duplicated RA/DEC {:.3f}'.format(1 - unique / structured.size))


def get_box_clustering_positions(fn, los='x', **kwargs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    catalog = None
    mpiroot = 0
    boxsize, scalev = None, None

    if mpicomm.rank == mpiroot:  # Faster to read catalogs from one rank
        catalog = Catalog.read(fn, mpicomm=MPI.COMM_SELF)
        catalog.get(catalog.columns())  # Faster to read all columns at once
        boxsize = catalog.header['BOXSIZE']
        scalev = catalog.header['VELZ2KMS']
    boxsize, scalev = mpicomm.bcast((boxsize, scalev), root=mpiroot)

    if mpicomm.size > 1:
        catalog = Catalog.scatter(catalog, mpicomm=mpicomm, mpiroot=mpiroot)

    positions = np.column_stack([catalog['X'], catalog['Y'], catalog['Z']])
    velocities = np.column_stack([catalog['VX'], catalog['VY'], catalog['VZ']]) / scalev
    vlos = los
    if isinstance(los, str):
        vlos = [0.] * 3
        vlos['xyz'.index(los)] = 1.
    vlos = np.array(vlos)
    positions = positions + np.sum(velocities * vlos, axis=-1)[..., None] * vlos[None, :]
    return (positions + boxsize / 2.) % boxsize - boxsize / 2.


def get_data_fn(tracer='LRG1', imock=0, version='final', **kwargs):
    if version == 'final':
        return f'/pscratch/sd/e/efdez/Uchuu-GLAM/GLAM/final_mocks/GLAM-Uchuu_{tracer[:3]}_{imock:02d}_Y3_cut_sky_clustering.dat.fits'
    return f'/dvs_ro/cfs/cdirs/desi/mocks/cai/GLAM-Uchuu/cut_skies/{tracer}/GLAM-Uchuu_{tracer[:3]}_{imock:02d}_Y3_cut_sky_clustering.dat.fits'


def get_randoms_fn(tracer='LRG1', imock=0, iran=0, version='final', **kwargs):
    if version == 'final':
        return f'/pscratch/sd/e/efdez/Uchuu-GLAM/GLAM/final_mocks/randoms/GLAM-Uchuu_{tracer[:3]}_{imock:02d}_Y3_cut_sky_clustering.ran.fits'
    return f'/dvs_ro/cfs/cdirs/desi/mocks/cai/GLAM-Uchuu/cut_skies/{tracer}/GLAM-Uchuu_{tracer[:3]}_{iran:d}_Y3_cut_sky_clustering.ran.fits'


def get_measurement_fn(tracer='LRG1', imock=0, region='NGC', kind='mesh2spectrum', zrange=(0.8, 1.1), version='final', **kwargs):
    if version == 'final':
        return f'/global/cfs/projectdirs/desi/mocks/cai/GLAM-Uchuu/desipipe_test/final_mocks/{kind}_GLAM-Uchuu_{tracer[:3]}_{imock:03d}_Y3_cut_sky_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}.npy'
    return f'/global/cfs/projectdirs/desi/mocks/cai/GLAM-Uchuu/desipipe_test/cut_skies/{tracer}/{kind}_GLAM-Uchuu_{tracer[:3]}_{imock:03d}_Y3_{region}.npy'


def compute_spectrum(output_fn, get_data, get_randoms, ells=(0, 2, 4), los='firstpoint', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp2_spectrum_normalization, compute_fkp2_spectrum_shotnoise, BinMesh2Spectrum, get_mesh_attrs, compute_mesh2_spectrum)
    t0 = time.time()
    data, randoms = get_data(), get_randoms()
    attrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=attrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=attrs, exchange=True, backend='jax')
    print(data.size, randoms.size, data.positions.sum(axis=0), randoms.positions.sum(axis=0))
    fkp = FKPField(data, randoms)
    norm, num_shotnoise = compute_fkp2_spectrum_normalization(fkp), compute_fkp2_spectrum_shotnoise(fkp)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    wsum_data1 = data.sum()
    del fkp, data, randoms
    bin = BinMesh2Spectrum(mesh.attrs, edges={'step': 0.001}, ells=ells)
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    spectrum.attrs.update(mesh=dict(mesh.attrs), los=los, wsum_data1=wsum_data1)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Done in {t1 - t0:.2f}')
    spectrum.save(output_fn)


def compute_spectrum_window(output_fn, get_randoms, spectrum_fn=None, kind='smooth'):
    from jax import numpy as jnp
    from jaxpower import (ParticleField, compute_mesh2_spectrum_window, BinMesh2Spectrum, BinMesh2Correlation, compute_mesh2_correlation, compute_smooth2_spectrum_window, BinnedStatistic, MeshAttrs)
    spectrum = BinnedStatistic.load(spectrum_fn)
    attrs = MeshAttrs(**spectrum.attrs['mesh'])
    randoms = ParticleField(*get_randoms(), attrs=attrs, exchange=True, backend='jax')
    randoms = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms
    los = spectrum.attrs['los']
    num_shotnoise = jnp.sum(randoms.weights**2)
    mesh = randoms.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    del randoms
    ells, norm = spectrum.projs, spectrum.norm
    edges = spectrum.edges(projs=0)
    bin = BinMesh2Spectrum(attrs, edges=edges, ells=ells)
    edgesin = bin.edges
    output_fn = str(output_fn)

    if kind == 'smooth':
        sbin = BinMesh2Correlation(attrs, edges={}, ells=list(range(0, 9, 2)))
        xi = compute_mesh2_correlation(mesh, bin=sbin, los=los).clone(norm=norm, num_zero=None, num_shotnoise=num_shotnoise / attrs.cellsize.prod())
        xi.save(output_fn.replace('window_mesh2spectrum', 'window_xi_mesh2spectrum'))
        del mesh
        wmatrix = compute_smooth2_spectrum_window(xi, edgesin=edgesin, ellsin=ells, bin=bin)
    else:
        wmatrix = compute_mesh2_spectrum_window(mesh, edgesin=edgesin, ellsin=ells, los=spectrum.attrs['los'], bin=bin, pbar=True, flags=('infinite',), norm=norm)
    wmatrix.attrs['norm'] = norm
    wmatrix.save(output_fn)


def compute_thetacut(output_fn, get_data, get_randoms, spectrum_fn=None, output_spectrum_fn=None, **attrs):
    from jaxpower import MeshAttrs, ParticleField, FKPField, BinParticle2Correlation, compute_particle2, Spectrum2Poles
    spectrum = Spectrum2Poles.load(spectrum_fn)
    ells, los = spectrum.projs, spectrum.attrs['los']
    attrs = MeshAttrs(**spectrum.attrs['mesh'])

    data, randoms = get_data(), get_randoms()
    data = ParticleField(*data, attrs=attrs)
    randoms = ParticleField(*randoms, attrs=attrs)
    fkp = FKPField(data, randoms)

    bin = BinParticle2Correlation(attrs, edges={'step': 0.1}, selection={'theta': (0., 0.05)}, ells=ells)
    cut = compute_particle2(fkp.particles, bin=bin, los=los)
    cut.save(output_fn)
    if spectrum_fn is not None and output_spectrum_fn is not None:
        cut = cut.to_spectrum(spectrum)
        spectrum = spectrum.clone(num=[spectrum.num[iproj] - cut.num[iproj] for iproj in range(len(spectrum.projs))])
        spectrum.save(output_spectrum_fn)


def compute_thetacut_window(output_fn, get_randoms, spectrum_fn=None, window_fn=None, **attrs):
    from jax import numpy as jnp
    from jaxpower import MeshAttrs, ParticleField, FKPField, BinParticle2Correlation, BinMesh2Spectrum, compute_particle2, Spectrum2Poles, WindowMatrix, compute_smooth2_spectrum_window, BinnedStatistic
    spectrum = Spectrum2Poles.load(spectrum_fn)
    ells, norm = spectrum.projs, spectrum.norm
    attrs = MeshAttrs(**spectrum.attrs['mesh'])
    output_fn = str(output_fn)

    wmatrix = WindowMatrix.load(window_fn)
    edgesin = wmatrix.theory.edges(projs=0)
    edgesin = jnp.arange(edgesin.min(), 10. * edgesin.max(), edgesin[0, 1] - edgesin[0, 0])
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    theory = BinnedStatistic(value=[jnp.zeros_like(edgesin[:, 0])] * len(wmatrix.theory.projs),
                             edges=[edgesin] * len(wmatrix.theory.projs), projs=wmatrix.theory.projs)
    wmatrix = wmatrix.interp(theory, axis='t', extrap=True)
    edgesin = wmatrix.theory.edges(projs=0)
    ellsin = wmatrix.theory.projs

    randoms = get_randoms()
    randoms = ParticleField(*randoms, attrs=attrs)
    particles = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms
    del randoms

    bin = BinParticle2Correlation(attrs, edges={'step': 0.1}, selection={'theta': (0., 0.05)}, ells=list(range(0, 9, 2)))
    cut = compute_particle2(particles, bin=bin, los=los).clone(norm=norm, num_zero=None)
    #cut.save(output_fn.replace('window_mesh2spectrum_thetacut', 'window_xi_thetacut'))

    bin = BinMesh2Spectrum(attrs, edges=spectrum.edges(projs=0), ells=ells)
    cut = compute_smooth2_spectrum_window(cut, edgesin=edgesin, ellsin=ellsin, bin=bin)
    #cut.save(output_fn.replace('window_mesh2spectrum_thetacut', 'window_thetacut'))

    wmatrix = wmatrix.clone(value=wmatrix.view() - cut.view())
    wmatrix.save(output_fn)


def compute_bispectrum(output_fn, get_data, get_randoms, basis='scoccimarro', los='local', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp3_spectrum_normalization, compute_fkp3_spectrum_shotnoise, BinMesh3Spectrum, get_mesh_attrs, compute_mesh3_spectrum)
    t0 = time.time()
    data, randoms = get_data(), get_randoms()
    attrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=attrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=attrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    ells = [(0, 0, 0), (0, 0, 2)] if 'sugiyama' in basis else [0, 2]
    bin = BinMesh3Spectrum(attrs, edges={'step': 0.01}, basis=basis, ells=ells, buffer_size=4)
    #norm = compute_fkp3_spectrum_normalization(fkp, cellsize=None)
    norm = compute_fkp3_spectrum_normalization(fkp, split=42, cellsize=None)
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_spectrum_shotnoise(fkp, los=los, bin=bin, **kw)
    mesh = fkp.paint(**kw, out='real')
    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
    spectrum.save(output_fn)


def get_proposal_boxsize(tracer):
    if 'BGS' in tracer:
        return 4000.
    if 'LRG' in tracer:
        return 7000.
    if 'LRG+ELG' in tracer:
        return 9000.
    if 'ELG' in tracer:
        return 9000.
    if 'QSO' in tracer:
        return 10000.
    raise NotImplementedError(f'tracer {tracer} is unknown')


if __name__ == '__main__':

    #catalog_args = dict(tracer='LRG1', region='NGC')
    catalog_args = dict(version='final', tracer='LRG', region='NGC', zrange=(0.4, 0.6))
    #catalog_args = dict(version='final', tracer='BGS', region='NGC', zrange=(0.1, 0.4))
    cutsky_args = dict(cellsize=10., boxsize=get_proposal_boxsize(catalog_args['tracer']), ells=(0, 2, 4))
    box_args = dict(boxsize=2000., boxcenter=0., meshsize=512, los='x')
    setup_logging()

    todo = []
    #todo = ['spectrum-box']
    #todo = ['window-spectrum-box']
    #todo = ['spectrum', 'window-spectrum'][:1]
    todo = ['bispectrum']
    #todo = ['thetacut', 'window-thetacut'][:1]
    #todo = ['pypower', 'window-pypower'][:1]
    #todo = ['covariance-spectrum']

    nmocks = 20
    t0 = time.time()

    is_distributed = any(td in ['spectrum', 'bispectrum', 'window-spectrum', 'spectrum-box', 'window-spectrum-box', 'covariance-spectrum'] for td in todo)
    if is_distributed:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        jax.distributed.initialize()
    from jax import config
    config.update('jax_enable_x64', True)

    from jaxpower.mesh import create_sharding_mesh

    for imock in range(nmocks):
        data_fn = get_data_fn(imock=imock, **catalog_args)
        if not Path(data_fn).exists():
            print(data_fn)
            continue
        #all_randoms_fn = [get_randoms_fn(iran=iran + 1, **catalog_args) for iran in range(4)]
        all_randoms_fn = [get_randoms_fn(imock=imock, **catalog_args)]
        get_data = lambda: get_clustering_positions_weights(data_fn, **catalog_args)
        get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, **catalog_args)

        if 'spectrum' in todo:
            output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum')
            with create_sharding_mesh() as sharding_mesh:
                compute_spectrum(output_fn, get_data, get_randoms, **cutsky_args)

        if 'bispectrum' in todo:
            output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh3spectrum_scoccimarro')
            with create_sharding_mesh() as sharding_mesh:
                args = cutsky_args | dict(basis='scoccimarro', cellsize=15.)
                args.pop('ells')
                compute_bispectrum(output_fn, get_data, get_randoms, **args)

        if 'thetacut' in todo:
            spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum')
            output_spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum_thetacut')
            output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='thetacut')
            with create_sharding_mesh() as sharding_mesh:
                compute_thetacut(output_fn, get_data, get_randoms, spectrum_fn=spectrum_fn, output_spectrum_fn=output_spectrum_fn)

        if imock == 0:  # any mock as input
            if 'window-spectrum' in todo:
                spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum')
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_mesh2spectrum')
                with create_sharding_mesh() as sharding_mesh:
                    compute_spectrum_window(output_fn, get_randoms, spectrum_fn=spectrum_fn)

            if 'window-thetacut' in todo:
                spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum')
                window_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_mesh2spectrum')
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_mesh2spectrum_thetacut')
                with create_sharding_mesh() as sharding_mesh:
                    compute_thetacut_window(output_fn, get_randoms, spectrum_fn=spectrum_fn, window_fn=window_fn)

            if 'covariance-spectrum' in todo:
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='theory_mesh2spectrum')
                spectrum_fns = [get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='mesh2spectrum') for imock in range(nmocks)]
                window_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='window_mesh2spectrum')
                target_spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum')
                compute_theory(output_fn, spectrum_fns, window_fn, target_spectrum_fn=target_spectrum_fn)
                theory_fn = output_fn

                spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum')
                window_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_mesh2spectrum')
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='covariance_mesh2spectrum')
                with create_sharding_mesh() as sharding_mesh:
                    compute_spectrum_covariance(output_fn, get_randoms, spectrum_fn=spectrum_fn, theory_fn=theory_fn)

    if is_distributed:
        jax.distributed.shutdown()
    print('Elapsed time: {:.2f} s'.format(time.time() - t0))