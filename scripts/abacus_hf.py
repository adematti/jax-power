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


def get_data_fn(tracer='ELG_LOP', zsnap=0.950, imock=0, **kwargs):
    sznap = f'{zsnap:.3f}'.replace('.', 'p')
    dirname = tracer
    if tracer == 'QSO':
        szrange = '0p8to3p5'
    if tracer == 'ELG_LOP':
        szrange = '0p8to1p6'
        dirname = 'ELG_v5'
    if tracer == 'LRG':
        szrange = '0p4to1p1'
    return f'/dvs_ro/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/AbacusSummit_base_c000_ph{imock:03d}/CutSky/{dirname}/z{zsnap:.3f}/forclustering/cutsky_abacusHF_DR2_{tracer}_z{sznap}_zcut_{szrange}_clustering.dat.fits'


def get_randoms_fn(iran=0, **kwargs):
    return f'/dvs_ro/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/randoms/rands_intiles_DARK_nomask_{iran:d}_v2.fits'


def get_measurement_fn(tracer='ELG_LOP', zsnap=0.950, imock=0, region='NGC', kind='mesh2spectrum', zrange=(0.8, 1.1), **kwargs):
    sznap = f'{zsnap:.3f}'.replace('.', 'p')
    dirname = tracer
    if tracer == 'ELG_LOP':
        dirname = 'ELG_v5'
    return f'/global/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/desipipe_test/AbacusSummit_base_c000_ph{imock:03d}/CutSky/{dirname}/{kind}_abacusHF_DR2_{tracer}_z{sznap}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}.npy'


def compute_spectrum(output_fn, get_data, get_randoms, ells=(0, 2, 4), los='firstpoint', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp2_spectrum_normalization, compute_fkp2_spectrum_shotnoise, BinMesh2Spectrum, get_mesh_attrs, compute_mesh2_spectrum)
    t0 = time.time()
    data, randoms = get_data(), get_randoms()
    attrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=attrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=attrs, exchange=True, backend='jax')
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


def compute_pypower(output_fn, get_data, get_randoms, **attrs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    from pypower import CatalogFFTPower
    data_positions, data_weights = get_data()
    randoms_positions, randoms_weights = get_randoms()
    direct_selection_attrs = {'theta': (0., 0.05)}
    direct_edges = {'min': 0., 'step': 0.1}
    direct_attrs = {'nthreads': mpicomm.size}
    #direct_selection_attrs = direct_edges = direct_attrs = None
    power = CatalogFFTPower(data_positions1=data_positions, data_weights1=data_weights, randoms_positions1=randoms_positions, randoms_weights1=randoms_weights, position_type='pos', resampler='tsc', interlacing=3, edges={'step': 0.001}, **attrs, direct_selection_attrs=direct_selection_attrs, direct_edges=direct_edges, direct_attrs=direct_attrs, mpicomm=mpicomm, mpiroot=None)
    power.save(output_fn)


def compute_pypower_window(output_fn, get_randoms, spectrum_fn=None, output_window_cut_fn=None, **attrs):
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    from pypower import CatalogFFTPower, CatalogSmoothWindow, PowerSpectrumMultipoles, PowerSpectrumOddWideAngleMatrix, PowerSpectrumSmoothWindowMatrix
    output_window_fn = output_fn
    randoms_positions, randoms_weights = get_randoms()
    direct_selection_attrs = {'theta': (0., 0.05)}
    direct_edges = {'min': 0., 'step': 0.1}
    direct_attrs = {'nthreads': mpicomm.size}
    power = PowerSpectrumMultipoles.load(spectrum_fn)
    boxsize = power.attrs['boxsize']
    edges = power.edges[0]
    boxscales = [1., 5., 20.][:1]
    windows = []
    boxsizes = boxsize * np.array(boxscales)
    edges = {'step': 2. * np.pi / np.max(boxsizes)}
    for iboxsize, boxsize in enumerate(boxsizes):
        windows.append(CatalogSmoothWindow(randoms_positions1=randoms_positions, randoms_weights1=randoms_weights,
                                           power_ref=power, edges=edges, boxsize=boxsize, position_type='pos',
                                           direct_attrs=direct_attrs,
                                           direct_selection_attrs=direct_selection_attrs if iboxsize == 0 else None,
                                           direct_edges=direct_edges if iboxsize == 0 else None,
                                           mpicomm=mpicomm, mpiroot=None).poles)

    if windows[0].mpicomm.rank == 0:
        windows[0].log_info('Concatenating windows')
        windows = windows[0].concatenate_x(*windows[::-1], frac_nyq=0.9)

        for output_fn, nodirect in zip([output_window_cut_fn, output_window_fn], [False, True]):
            window = windows
            if nodirect:
                window = windows.deepcopy()
                window.power_direct_nonorm[...] = 0.
                for name in ['corr_direct_nonorm', 'sep_direct']: setattr(window, name, None)
    
            # Let us compute the wide-angle and window function matrix
            ellsin = (0, 2, 4)  # input (theory) multipoles
            wa_orders = 1 # wide-angle order
            sep = np.geomspace(1e-4, 1e5, 1024 * 16) # configuration space separation for FFTlog
            kin_rebin = 2 # rebin input theory to save memory
            #sep = np.geomspace(1e-4, 2e4, 1024 * 16) # configuration space separation for FFTlog, 2e4 > sqrt(3) * 8000
            #kin_rebin = 4 # rebin input theory to save memory
            kin_lim = (0, 2e1) # pre-cut input (theory) ks to save some memory
            # Input projections for window function matrix:
            # theory multipoles at wa_order = 0, and wide-angle terms at wa_order = 1
            projsin = tuple(ellsin) + tuple(PowerSpectrumOddWideAngleMatrix.propose_out(ellsin, wa_orders=wa_orders))
            # Window matrix
            wmatrix = PowerSpectrumSmoothWindowMatrix(power, projsin=projsin, window=window, sep=sep, kin_rebin=kin_rebin, kin_lim=kin_lim)
            # We resum over theory odd-wide angle
            wmatrix.resum_input_odd_wide_angle()
            wmatrix.attrs.update(power.attrs)
            wmatrix.save(output_fn)


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


def get_box_data_fn(tracer='ELG', zsnap=0.950, imock=0, **kwargs):
    tracer = tracer[:3]
    sznap = f'{zsnap:.3f}'.replace('.', 'p')
    return f'/global/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/AbacusSummit_base_c000_ph{imock:03d}/Boxes/{tracer}/abacus_HF_{tracer}_{sznap}_DR2_v1.0_AbacusSummit_base_c000_ph{imock:03d}_clustering.dat.fits'


def get_box_measurement_fn(tracer='ELG', zsnap=0.950, imock=0, kind='mesh2spectrum', los='x', **kwargs):
    tracer = tracer[:3]
    sznap = f'{zsnap:.3f}'.replace('.', 'p')
    return f'/global/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/desipipe_test/AbacusSummit_base_c000_ph{imock:03d}/Boxes/{tracer}/{kind}_abacusHF_DR2_v1.0_{tracer}_z{sznap}_los-{los}.npy'


def compute_box_spectrum(output_fn, get_data, ells=(0, 2, 4), los='x', **attrs):
    from jaxpower import (MeshAttrs, ParticleField, FKPField, BinMesh2Spectrum, get_mesh_attrs, compute_mesh2_spectrum)
    t0 = time.time()
    data = get_data()
    attrs = MeshAttrs(**attrs)
    data = ParticleField(data, attrs=attrs, exchange=True, backend='jax')
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    shotnoise = attrs.boxsize.prod() / data.sum()
    mesh = mesh / mesh.mean()
    bin = BinMesh2Spectrum(mesh.attrs, edges={'step': 0.001}, ells=ells)
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los)
    spectrum = spectrum.clone(num_shotnoise=shotnoise * spectrum.norm)
    spectrum.attrs.update(mesh=dict(mesh.attrs), los=los)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Done in {t1 - t0:.2f}')
    spectrum.save(output_fn)


def compute_box_spectrum_window(output_fn, spectrum_fn=None):
    from jaxpower import (ParticleField, compute_mesh2_spectrum_window, BinMesh2Spectrum, BinnedStatistic, MeshAttrs)
    spectrum = BinnedStatistic.load(spectrum_fn)
    attrs = MeshAttrs(**spectrum.attrs['mesh'])
    ells = spectrum.projs
    bin = BinMesh2Spectrum(attrs, edges=spectrum.edges(projs=0), ells=ells)
    #edgesin = np.linspace(bin.edges.min(), bin.edges.max(), 2 * (len(bin.edges) - 1))
    edgesin = bin.edges
    wmatrix = compute_mesh2_spectrum_window(attrs, edgesin=edgesin, ellsin=ells, los=spectrum.attrs['los'], bin=bin)
    wmatrix.save(output_fn)


if __name__ == '__main__':

    catalog_args = dict(tracer='LRG', region='NGC', zsnap=0.950, zrange=(0.8, 1.1))
    #catalog_args = dict(tracer='QSO', region='NGC', zsnap=1.400, zrange=(0.8, 2.1))
    cutsky_args = dict(cellsize=10., boxsize=get_proposal_boxsize(catalog_args['tracer']), ells=(0, 2, 4), los='firstpoint')
    box_args = dict(boxsize=2000., boxcenter=0., meshsize=512, los='x')
    setup_logging()

    todo = []
    #todo = ['spectrum-box']
    #todo = ['window-spectrum-box']
    #todo = ['spectrum', 'window-spectrum'][1:]
    todo = ['thetacut', 'window-thetacut'][1:]
    #todo = ['pypower', 'window-pypower'][1:]

    nmocks = 25
    ells = (0, 2, 4)
    los = 'firstpoint'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    t0 = time.time()

    is_distributed = any(td in ['spectrum', 'window-spectrum', 'spectrum-box', 'window-spectrum-box'] for td in todo)
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
        all_randoms_fn = [get_randoms_fn(iran=iran, **catalog_args) for iran in range(4)]
        get_data = lambda: get_clustering_positions_weights(data_fn, **catalog_args)
        get_randoms = lambda: get_clustering_positions_weights(*all_randoms_fn, **catalog_args)

        if 'spectrum' in todo:
            output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum')
            with create_sharding_mesh() as sharding_mesh:
                compute_spectrum(output_fn, get_data, get_randoms, **cutsky_args)
    
        if 'thetacut' in todo:
            spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum')
            output_spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum_thetacut')
            output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='thetacut')
            with create_sharding_mesh() as sharding_mesh:
                compute_thetacut(output_fn, get_data, get_randoms, spectrum_fn=spectrum_fn, output_spectrum_fn=output_spectrum_fn)

        if 'pypower' in todo:
            output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='pypower')
            compute_pypower(output_fn, get_data, get_randoms, **cutsky_args)

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

            if 'window-pypower' in todo:
                spectrum_fn = get_measurement_fn(imock=imock, **catalog_args, kind='pypower')
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_pypower')
                output_cut_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_pypower_thetacut')
                compute_pypower_window(output_fn, get_randoms, spectrum_fn=spectrum_fn, output_window_cut_fn=output_cut_fn)

    for imock in range(nmocks):
        data_fn = get_box_data_fn(imock=imock, **catalog_args)
        get_data = lambda: get_box_clustering_positions(data_fn, **catalog_args, **box_args)
        output_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='mesh2spectrum')
        if 'spectrum-box' in todo:
            with create_sharding_mesh() as sharding_mesh:
                compute_box_spectrum(output_fn, get_data, **box_args)

        if imock == 0:
            if 'window-spectrum-box' in todo:
                spectrum_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='mesh2spectrum')
                output_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='window_mesh2spectrum')
                with create_sharding_mesh() as sharding_mesh:
                    compute_box_spectrum_window(output_fn, spectrum_fn=spectrum_fn)
    
    if is_distributed:
        jax.distributed.shutdown()
    print('Elapsed time: {:.2f} s'.format(time.time() - t0))