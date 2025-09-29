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
    #return f'/dvs_ro/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/AbacusSummit_base_c000_ph{imock:03d}/CutSky/{dirname}/z{zsnap:.3f}/forclustering/masked_cutsky_abacusHF_DR2_{tracer}_z{sznap}_zcut_{szrange}_clustering.dat.fits'
    return f'/dvs_ro/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/AbacusSummit_base_c000_ph{imock:03d}/CutSky/{dirname}/z{zsnap:.3f}/forclustering/cutsky_abacusHF_DR2_{tracer}_z{sznap}_zcut_{szrange}_clustering.dat.fits'


def get_randoms_fn(iran=0, **kwargs):
    #return f'/global/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/randoms/rands_intiles_DARK_0_v2.fits'
    return f'/dvs_ro/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/randoms/rands_intiles_DARK_nomask_{iran:d}_v2.fits'


def get_measurement_fn(tracer='ELG_LOP', zsnap=0.950, imock=0, region='NGC', kind='mesh2spectrum', zrange=(0.8, 1.1), **kwargs):
    sznap = f'{zsnap:.3f}'.replace('.', 'p')
    dirname = tracer
    if tracer == 'ELG_LOP':
        dirname = 'ELG_v5'
    ext = 'npy' if 'pypower' in kind else 'h5'
    return f'/global/cfs/projectdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/desipipe_test/AbacusSummit_base_c000_ph{imock:03d}/CutSky/{dirname}/{kind}_abacusHF_DR2_{tracer}_z{sznap}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}.{ext}'


def compute_spectrum(output_fn, get_data, get_randoms, ells=(0, 2, 4), los='firstpoint', **attrs):
    import jax
    from jaxpower import (ParticleField, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum)
    t0 = time.time()
    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=ells)
    norm, num_shotnoise = compute_fkp2_normalization(fkp, bin=bin), compute_fkp2_shotnoise(fkp, bin=bin)
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    wsum_data1 = data.sum()
    del fkp, data, randoms
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=num_shotnoise)
    mattrs = {name: mattrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    spectrum = spectrum.clone(attrs=dict(los=los, wsum_data1=wsum_data1, **mattrs))
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Done in {t1 - t0:.2f}')
    spectrum.write(output_fn)


def compute_spectrum_window(output_fn, get_randoms, spectrum_fn=None, kind='smooth'):
    from jax import numpy as jnp
    from jaxpower import (ParticleField, compute_mesh2_spectrum_window, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, compute_fkp2_shotnoise, compute_smooth2_spectrum_window, MeshAttrs, read)
    spectrum = read(spectrum_fn)
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    randoms = ParticleField(*get_randoms(), attrs=mattrs, exchange=True, backend='jax')
    randoms = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms
    los = spectrum.attrs['los']
    mesh = randoms.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    ells, norm = spectrum.ells, spectrum.get(0).values('norm')[0]
    edges = spectrum.get(0).edges('k')
    bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
    edgesin = bin.edges
    output_fn = str(output_fn)

    if kind == 'smooth':
        sbin = BinMesh2CorrelationPoles(mattrs, edges={}, ells=list(range(0, 9, 2)))
        num_shotnoise = compute_fkp2_shotnoise(randoms, bin=sbin)
        xi = compute_mesh2_correlation(mesh, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells), num_shotnoise=num_shotnoise)
        xi.write(output_fn.replace('window_mesh2spectrum', 'window_xi_mesh2spectrum'))
        del mesh
        wmatrix = compute_smooth2_spectrum_window(xi, edgesin=edgesin, ellsin=ells, bin=bin)
    else:
        wmatrix = compute_mesh2_spectrum_window(mesh, edgesin=edgesin, ellsin=ells, los=spectrum.attrs['los'], bin=bin, pbar=True, flags=('infinite',), norm=norm)
    wmatrix.attrs['norm'] = norm
    wmatrix.write(output_fn)


def compute_thetacut(output_fn, get_data, get_randoms, spectrum_fn=None, output_spectrum_fn=None, **attrs):
    from jaxpower import MeshAttrs, ParticleField, FKPField, BinParticle2CorrelationPoles, compute_particle2, read
    spectrum = read(spectrum_fn)
    ells, los = spectrum.ells, spectrum.attrs['los']
    mattrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})

    data, randoms = get_data(), get_randoms()
    data = ParticleField(*data, attrs=mattrs)
    randoms = ParticleField(*randoms, attrs=mattrs)
    fkp = FKPField(data, randoms)

    bin = BinParticle2CorrelationPoles(attrs, edges={'step': 0.1}, selection={'theta': (0., 0.05)}, ells=ells)
    cut = compute_particle2(fkp.particles, bin=bin, los=los)
    cut.write(output_fn)
    if spectrum_fn is not None and output_spectrum_fn is not None:
        cut = cut.to_spectrum(spectrum)
        spectrum = spectrum.clone(value=spectrum.value() - cut.value())
        spectrum.write(output_spectrum_fn)


def compute_thetacut_window(output_fn, get_randoms, spectrum_fn=None, window_fn=None, **attrs):
    from jax import numpy as jnp
    from jaxpower import MeshAttrs, ParticleField, FKPField, BinParticle2CorrelationPoles, BinMesh2SpectrumPoles, Mesh2SpectrumPole, Mesh2SpectrumPoles, compute_particle2, WindowMatrix, compute_smooth2_spectrum_window, read
    spectrum = read(spectrum_fn)
    ells, norm = spectrum.ells, spectrum.get(0).values('norm')
    attrs = MeshAttrs(**{name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    output_fn = str(output_fn)

    wmatrix = read(window_fn)
    edgesin = wmatrix.theory.get(0).edges()
    edgesin = jnp.arange(edgesin.min(), 10. * edgesin.max(), edgesin[0, 1] - edgesin[0, 0])
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    kin = jnp.mean(edgesin, axis=-1)
    theory = []
    for ell in wmatrix.theory.ells:
        theory.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=jnp.zeros_like(kin), ell=ell))
    theory = Mesh2SpectrumPoles(theory)

    wmatrix = wmatrix.at.theory.interp(theory, extrap=True)
    edgesin = wmatrix.theory.get(0).edges('k')
    ellsin = wmatrix.theory.ells

    randoms = get_randoms()
    randoms = ParticleField(*randoms, attrs=attrs)
    particles = spectrum.attrs['wsum_data1'] / randoms.sum() * randoms
    del randoms

    bin = BinParticle2CorrelationPoles(attrs, edges={'step': 0.1}, selection={'theta': (0., 0.05)}, ells=list(range(0, 9, 2)))
    cut = compute_particle2(particles, bin=bin, los='local').clone(norm=[norm] * len(bin.ells))
    #cut.write(output_fn.replace('window_mesh2spectrum_thetacut', 'window_xi_thetacut'))

    bin = BinMesh2SpectrumPoles(attrs, edges=spectrum.get(0).edges('k'), ells=ells)
    cut = compute_smooth2_spectrum_window(cut, edgesin=edgesin, ellsin=ellsin, bin=bin)
    #cut.write(output_fn.replace('window_mesh2spectrum_thetacut', 'window_thetacut'))

    wmatrix = wmatrix.clone(value=wmatrix.view() - cut.view())
    wmatrix.write(output_fn)


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
        windows[-1].save(str(output_window_fn).replace('window_pypower', 'window_pypower_xi{:d}'.format(iboxsize)))

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
            kin_rebin = 2 # rebin input theory to write memory
            #sep = np.geomspace(1e-4, 2e4, 1024 * 16) # configuration space separation for FFTlog, 2e4 > sqrt(3) * 8000
            #kin_rebin = 4 # rebin input theory to write memory
            kin_lim = (0, 2e1) # pre-cut input (theory) ks to write some memory
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
    from jaxpower import (MeshAttrs, ParticleField, FKPField, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum, compute_fkp2_shotnoise)
    t0 = time.time()
    data = get_data()
    mattrs = MeshAttrs(**attrs)
    data = ParticleField(data, attrs=mattrs, exchange=True, backend='jax')
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mean = mesh.mean()
    mesh = mesh - mean
    bin = BinMesh2SpectrumPoles(mesh.attrs, edges={'step': 0.001}, ells=ells)
    jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'], donate_argnums=[0])
    spectrum = jitted_compute_mesh2_spectrum(mesh, bin=bin, los=los)
    num_shotnoise = compute_fkp2_shotnoise()
    spectrum = spectrum.clone(norm=[pole.values('norm') * mean**2 for pole in spectrum], num_shotnoise=num_shotnoise)
    spectrum.attrs.update(mesh=dict(mesh.attrs), los=los)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Done in {t1 - t0:.2f}')
    spectrum.write(output_fn)


def compute_box_spectrum_window(output_fn, spectrum_fn=None):
    from jaxpower import (ParticleField, compute_mesh2_spectrum_window, BinMesh2SpectrumPoles, MeshAttrs, read)
    spectrum = read(spectrum_fn)
    mattrs = MeshAttrs(**spectrum.attrs['mesh'])
    ells = spectrum.ells
    bin = BinMesh2SpectrumPoles(mattrs, edges=spectrum.get(0).edges('k'), ells=ells)
    #edgesin = np.linspace(bin.edges.min(), bin.edges.max(), 2 * (len(bin.edges) - 1))
    edgesin = bin.edges
    wmatrix = compute_mesh2_spectrum_window(mattrs, edgesin=edgesin, ellsin=ells, los=spectrum.attrs['los'], bin=bin)
    wmatrix.write(output_fn)


def compute_bispectrum(output_fn, get_data, get_randoms, basis='scoccimarro', los='local', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum)
    t0 = time.time()
    data, randoms = get_data(), get_randoms()
    mattrs = get_mesh_attrs(data[0], randoms[0], check=True, **attrs)
    data = ParticleField(*data, attrs=mattrs, exchange=True, backend='jax')
    randoms = ParticleField(*randoms, attrs=mattrs, exchange=True, backend='jax')
    fkp = FKPField(data, randoms)
    ells = [(0, 0, 0), (2, 0, 2)] if 'sugiyama' in basis else [0, 2]
    bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01}, basis=basis, ells=ells, buffer_size=4)
    #norm = compute_fkp3_normalization(fkp, bin=bin, cellsize=None)
    norm = compute_fkp3_normalization(fkp, split=42, bin=bin, cellsize=None)
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(fkp, los=los, bin=bin, **kw)
    mesh = fkp.paint(**kw, out='real')
    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
    spectrum.write(output_fn)


def compute_box_bispectrum(output_fn, get_data, basis='scoccimarro', los='z', **attrs):
    from jaxpower import (ParticleField, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, get_mesh_attrs, compute_mesh3_spectrum, MeshAttrs)
    t0 = time.time()
    data = get_data()
    mattrs = MeshAttrs(**attrs)
    data = ParticleField(data, attrs=mattrs, exchange=True, backend='jax')
    mesh = data.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    mean = mesh.mean()
    mesh = mesh - mean
    ells = [(0, 0, 0), (0, 0, 2)] if 'sugiyama' in basis else [0, 2]
    bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01, 'max': 0.201}, basis=basis, ells=ells, buffer_size=2)
    #jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'], donate_argnums=[0])
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    num_shotnoise = compute_fkp3_shotnoise(data, los=los, bin=bin, **kw)
    mesh = data.paint(**kw, out='real')
    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    spectrum = spectrum.clone(norm=[pole.values('norm') * mean**3 for pole in spectrum], num_shotnoise=num_shotnoise)
    spectrum.attrs.update(mesh=dict(mesh.attrs), los=los)
    jax.block_until_ready(spectrum)
    t1 = time.time()
    if jax.process_index() == 0:
        print(f'Done in {t1 - t0:.2f}')
    spectrum.write(output_fn)


def convert_old_jaxpower(fn):
    # Convert old files
    import numpy as np
    from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles
    spectrum = np.load(fn, allow_pickle=True)[()]
    name = spectrum['__class__name__']
    if name == 'Spectrum2Poles':
        poles = []
        for ill, ell in enumerate(spectrum['_projs']):
            poles.append(Mesh2SpectrumPole(k=spectrum['_x'][ill], k_edges=spectrum['_edges'][ill],
                                           nmodes=spectrum['_weights'][ill], num_raw=spectrum['_value'][ill],
                                           num_shotnoise=spectrum['_num_shotnoise'][ill], norm=spectrum['_norm'], ell=ell))
        return Mesh2SpectrumPoles(poles)
    if name == 'Spectrum3Poles':
        poles = []
        for ill, ell in enumerate(spectrum['_projs']):
            poles.append(Mesh3SpectrumPole(k=spectrum['_x'][ill], k_edges=spectrum['_edges'][ill],
                                           nmodes=spectrum['_weights'][ill], num_raw=spectrum['_value'][ill],
                                           num_shotnoise=spectrum['_num_shotnoise'][ill], norm=spectrum['_norm'], basis=spectrum['basis'], ell=ell))
        return Mesh3SpectrumPoles(poles)
    raise NotImplementedError(f'Cannot convert {name}.')


if __name__ == '__main__':

    #fn = '/dvs_ro/cfs/cdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/desipipe_test/AbacusSummit_base_c000_ph000/CutSky/LRG/mesh2spectrum_abacusHF_DR2_LRG_z0p500_z0.4-0.6_NGC.npy'
    #fn = '/dvs_ro/cfs/cdirs/desi/mocks/cai/abacus_HF/DR2_v1.0/desipipe_test/AbacusSummit_base_c000_ph000/CutSky/LRG/mesh3spectrum_scoccimarro_abacusHF_DR2_LRG_z0p500_z0.4-0.6_NGC.npy'
    #convert_old_jaxpower(fn).select(k=slice(0, None, 2))

    #catalog_args = dict(tracer='ELG_LOP', region='SGC', zsnap=0.950, zrange=(0.8, 1.1))
    catalog_args = dict(tracer='LRG', region='NGC', zsnap=0.950, zrange=(0.8, 1.1))
    #catalog_args = dict(tracer='LRG', region='NGC', zsnap=0.725, zrange=(0.6, 0.8))
    #catalog_args = dict(tracer='LRG', region='NGC', zsnap=0.5, zrange=(0.4, 0.6))
    #catalog_args = dict(tracer='QSO', region='NGC', zsnap=1.400, zrange=(0.8, 2.1))
    cutsky_args = dict(cellsize=10., boxsize=get_proposal_boxsize(catalog_args['tracer']), ells=(0, 2, 4))
    box_args = dict(boxsize=2000., boxcenter=0., meshsize=512, los='z', ells=(0, 2, 4))
    setup_logging()

    todo = []
    #todo = ['spectrum-box']
    #todo = ['window-spectrum-box']
    #todo = ['spectrum', 'window-spectrum'][:1]
    #todo = ['rotate']
    todo = ['bispectrum']
    #todo = ['bispectrum-box']
    #todo = ['thetacut', 'window-thetacut'][:1]
    #todo = ['pypower', 'window-pypower'][:1]
    #todo = ['covariance-spectrum']

    nmocks = 25
    t0 = time.time()

    is_distributed = any(td in ['spectrum', 'bispectrum', 'thetacut', 'window-spectrum', 'window-thetacut', 'spectrum-box', 'window-spectrum-box', 'covariance-spectrum', 'bispectrum-box'] for td in todo)
    if is_distributed:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
        import jax
        jax.distributed.initialize()
    from jax import config
    config.update('jax_enable_x64', True)

    from jaxpower.mesh import create_sharding_mesh

    for imock in range(nmocks):
        data_fn = get_data_fn(imock=imock, **catalog_args)
        all_randoms_fn = [get_randoms_fn(iran=iran, **catalog_args) for iran in range(4)][:1]
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

            if 'rotate' in todo:
                window_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_mesh2spectrum')
                covariance_fn = get_measurement_fn(imock=imock, **catalog_args, kind='covariance_mesh2spectrum')
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='rotation_mesh2spectrum')
                output_window_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_mesh2spectrum_rotated')
                rotate(output_fn, window_fn, covariance_fn, Minit=None, output_window_fn=output_window_fn)
                #output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='rotation_old_mesh2spectrum')
                #rotate_old(output_fn, window_fn, covariance_fn, Minit=None)
                rotation_fn = output_fn
                data_fns = [get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum') for imock in range(nmocks)]
                output_fns = [get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum_rotated') for imock in range(nmocks)]
                postprocess_rotation(output_fns, rotation_fn, data_fns)

                target_spectrum_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='mesh2spectrum')
                spectrum_fns = [get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='mesh2spectrum') for imock in range(nmocks)]
                window_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='window_mesh2spectrum')
                output_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='theory_mesh2spectrum')
                compute_theory(output_fn, spectrum_fns, window_fn, target_spectrum_fn=target_spectrum_fn)
                theory_fn = output_fn

                window_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_mesh2spectrum_thetacut')
                covariance_fn = get_measurement_fn(imock=imock, **catalog_args, kind='covariance_mesh2spectrum')
                output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='rotation_mesh2spectrum_thetacut')
                mock_fns = [get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum_thetacut') for imock in range(nmocks)]
                output_window_fn = get_measurement_fn(imock=imock, **catalog_args, kind='window_mesh2spectrum_thetacut_rotated')
                output_covariance_fn = get_measurement_fn(imock=imock, **catalog_args, kind='covariance_mesh2spectrum_rotated')
                rotate(output_fn, window_fn, covariance_fn, mock_fns=mock_fns, theory_fn=theory_fn,
                      output_window_fn=output_window_fn, output_covariance_fn=output_covariance_fn)

                #output_fn = get_measurement_fn(imock=imock, **catalog_args, kind='rotation_old_mesh2spectrum_thetacut')
                #rotate_old(output_fn, window_fn, covariance_fn)
                rotation_fn = output_fn
                data_fns = [get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum_thetacut') for imock in range(nmocks)]
                output_fns = [get_measurement_fn(imock=imock, **catalog_args, kind='mesh2spectrum_thetacut_rotated') for imock in range(nmocks)]
                postprocess_rotation(output_fns, rotation_fn, data_fns)


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

        if 'bispectrum-box' in todo:
            output_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='mesh3spectrum_scoccimarro')
            with create_sharding_mesh() as sharding_mesh:
                args = box_args | dict(basis='scoccimarro', meshsize=256)
                args.pop('ells')
                compute_box_bispectrum(output_fn, get_data, **args)

        if imock == 0:
            if 'window-spectrum-box' in todo:
                spectrum_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='mesh2spectrum')
                output_fn = get_box_measurement_fn(imock=imock, **catalog_args, **box_args, kind='window_mesh2spectrum')
                with create_sharding_mesh() as sharding_mesh:
                    compute_box_spectrum_window(output_fn, spectrum_fn=spectrum_fn)

    if is_distributed:
        jax.distributed.shutdown()
    print('Elapsed time: {:.2f} s'.format(time.time() - t0))
