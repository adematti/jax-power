import time
from pathlib import Path
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (compute_mesh3_spectrum, BinMesh3SpectrumPoles, Mesh3SpectrumPoles, compute_mesh3_correlation, BinMesh3CorrelationPoles, Mesh3CorrelationPoles, MeshAttrs, generate_gaussian_mesh, generate_uniform_particles,
                      FKPField, compute_normalization, compute_fkp3_normalization, compute_fkp3_shotnoise, read, utils, ParticleField, compute_smooth3_spectrum_window)


dirname = Path('_tests')


def test_mesh3_spectrum(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.h5'.format(los)

    list_los = ['x', 'local']
    mattrs = MeshAttrs(meshsize=128, boxsize=1000., fft_engine='jax')

    for basis in ['scoccimarro', 'sugiyama-diagonal']:

        ells = [0] if 'scoccimarro' in basis else [(0, 0, 0)]
        bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.1}, basis=basis, ells=ells)

        @partial(jax.jit, static_argnames=['los'])
        def mock(mattrs, bin, seed, los='x'):
            mesh = generate_gaussian_mesh(mattrs, pkvec, seed=seed, unitary_amplitude=True)
            return compute_mesh3_spectrum(mesh, los=los, bin=bin)

        for los in list_los:

            nmock = 2
            t0 = time.time()
            spectrum = mock(mattrs, bin, random.key(43), los=los)
            jax.block_until_ready(spectrum)
            print(f'time for jit {time.time() - t0:.2f}')
            t0 = time.time()
            for i in range(nmock):
                spectrum = mock(mattrs, bin, random.key(i + 42), los=los)
                jax.block_until_ready(spectrum)
            print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
            #for pole in spectrum: pole.basis
            spectrum.write(get_fn(los))

        if plot:
            from matplotlib import pyplot as plt

            for los in list_los:
                spectrum = read(get_fn(los=los))
                ax = spectrum.plot().axes[0]
                ax.set_title(los)
                plt.show()


def test_timing():

    import time
    from jax import jit, random
    from jaxpower.mesh import create_sharded_random
    from jaxpower.mesh3 import get_real_Ylm, get_spherical_jn

    @jit
    def f1(mesh):
        xvec = mesh.attrs.rcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
        return jn(xnorm) * Ylm(*xvec) * mesh

    @jit
    def f2(mesh):
        kvec = mesh.attrs.rcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in kvec))
        mask = (xnorm >= 0.01) & (xnorm <= 0.03)
        return (Ylm(*kvec) * mesh * mask).c2r()

    attrs = MeshAttrs(boxsize=1000., meshsize=256)
    rmesh = attrs.create(kind='real', fill=create_sharded_random(random.normal, random.key(42), shape=attrs.meshsize))
    cmesh = rmesh.r2c()

    Ylm = get_real_Ylm(2, 0)
    jn = get_spherical_jn(2)

    f1(rmesh)
    f2(cmesh)
    jax.block_until_ready(f1(rmesh))
    jax.block_until_ready(f2(cmesh))

    def timing(f, mesh):
        t0 = time.time()
        for i in range(10):
            jax.block_until_ready(f((0.1 + i) * mesh))
        print(time.time() - t0)

    timing(f1, rmesh)
    timing(f2, cmesh)


def test_timing():

    from jaxpower.mesh import create_sharded_random

    mattrs = MeshAttrs(meshsize=256, boxsize=1000.)
    edges = [np.arange(0.01, 0.2, 0.01) for kmax in mattrs.knyq]
    bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis='scoccimarro', ells=[0], buffer_size=4)
    # batch_size is mostly useless
    #bin = BinMesh3Spectrum(mattrs, edges=edges, basis='scoccimarro', ells=[0], batch_size=12)
    cmeshs = [mattrs.create(kind='real', fill=create_sharded_random(random.normal, random.key(42), shape=mattrs.meshsize)).r2c() for axis in range(3)]


    def test(ff, jit=True):
        if jit: f = jax.jit(ff)
        else: f = ff
        bk = f(*cmeshs)
        bk = jax.block_until_ready(bk)
        t0 = time.time()
        bk = f(*cmeshs)
        bk = jax.block_until_ready(bk)
        print(time.time() - t0)
        return bk

    def f(*values):
        meshs = []

        def f2(axis, ibin):
            return bin.mattrs.c2r(values[axis].value * (bin.ibin[axis] == ibin))

        for axis in range(3):
            meshs.append(jax.lax.map(partial(f2, axis), xs=jnp.arange(len(bin._nmodes1d[axis]))))

        def f2(ibin):
            return jnp.sum(meshs[0][ibin[0]] * meshs[1][ibin[1]] * meshs[2][ibin[2]])

        return jax.lax.map(f2, bin._iedges)

    bk1 = test(f)
    bk2 = test(bin.__call__, jit=True)
    assert np.allclose(bk1, bk2)


def test_buffer():

    from jaxpower.mesh import create_sharded_random

    mattrs = MeshAttrs(meshsize=256, boxsize=1000.)
    edges = [np.arange(0.01, 0.2, 0.05) for kmax in mattrs.knyq]
    for basis in ['scoccimarro', 'sugiyama-diagonal']:
        kw = dict(edges=edges, basis=basis, ells=[0] if 'scoccimarro' in basis else [(0, 0, 0)])
        bin = BinMesh3SpectrumPoles(mattrs, **kw)
        bin_buffer = BinMesh3SpectrumPoles(mattrs, **kw, buffer_size=2)
        cmeshs = [mattrs.create(kind='real', fill=create_sharded_random(random.normal, random.key(42), shape=mattrs.meshsize)).r2c() for axis in range(3)]
        if 'sugiyama' in bin.basis:
            cmeshs[-1] = cmeshs[-1].c2r()

        bk1 = bin(*cmeshs)
        bk2 = bin_buffer(*cmeshs)
        assert np.allclose(bk2, bk1)


def test_polybin3d(plot=False):

    from PolyBin3D import PolyBin3D, BSpec

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    mattrs = MeshAttrs(meshsize=100, boxsize=1000., boxcenter=1000.)
    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e4)
    data = generate_uniform_particles(mattrs, size, seed=32)
    #data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    kw = dict(resampler='tsc', interlacing=2, compensate=True)
    mesh = data.paint(**kw)
    mean = mesh.mean()
    mesh = mesh - mean

    edges = [np.arange(0.01, 0.10, 0.02), np.arange(0.01, 0.15, 0.02), np.arange(0.01, 0.15, 0.02)]
    #edges = [np.arange(0.01, 0.08, 0.02), np.arange(0.01, 0.08, 0.02), np.arange(0.01, 0.08, 0.02)]

    bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis='scoccimarro', ells=[0, 2], buffer_size=4)

    for los in ['z', 'local'][:1]:

        kw_pb = dict(gridsize=mattrs.meshsize, boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter * (0. if los == 'z' else 1.))
        base = PolyBin3D(sightline='global' if los == 'z' else los, **kw_pb, backend='jax', real_fft=False)
        bspec = BSpec(base, k_bins=edges[0],
                      lmax=2, k_bins_squeeze=edges[1],
                      include_partial_triangles=False)

        for imock in [0]:
            t0 = time.time()
            spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
            spectrum = spectrum.clone(norm=[mean**3 * pole.norm for pole in spectrum])
            num_shotnoise = compute_fkp3_shotnoise(data, los=los, bin=bin, **kw)
            spectrum = spectrum.clone(num_shotnoise=num_shotnoise)

            jax.block_until_ready(spectrum)
            t1 = time.time()
            bk = bspec.Bk_ideal(data=mesh.value / mean, discreteness_correction=False)
            k123 = bspec.get_ks()
            t2 = time.time()
            print(f'jax-power took {t1 - t0:.2f} s')
            print(f'polybin took {t2 - t1:.2f} s')

            if plot and imock == 0:
                k = spectrum.get(0).coords('k', center='mid')
                weight = k.prod(axis=-1)
                spectrum_raw = spectrum.clone(num_shotnoise=[0. * pole.num_shotnoise for pole in spectrum])

                ax = plt.gca()
                weight = k123.prod(axis=0)
                for name in ['b0', 'b2']:
                    ax.plot(weight * bk[name], color='C0')

                for ell in [0, 2]:
                    ax.plot(weight * spectrum_raw.get(ell).value(), color='C1')
                    ax.plot(weight * spectrum.get(ell).value(), color='C1', linestyle='--')
                plt.show()


def test_triumvirate_box(plot=False):
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    mattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)

    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e4)
    data = generate_uniform_particles(mattrs, size, seed=32)
    # Triumvirate doesn't take weights for box statistics...
    #data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))

    kw = dict(resampler='cic', interlacing=False, compensate=True)
    mesh = data.paint(**kw)
    mean = mesh.mean()
    #mean = 1.
    mesh = mesh - mean
    edges = np.arange(0.01, mattrs.knyq[0], 0.01)
    ell = (0, 0, 0)
    ell = (1, 0, 1)
    #ell = (2, 0, 2)
    #ell = (3, 1, 2)
    #ell = (1, 3, 2)
    los = 'z'
    bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis='sugiyama-diagonal', ells=[ell])

    def take(array):
        if sum(ell[:2]) % 2 == 0:
            return array.real
        return array.imag

    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    num_shotnoise = compute_fkp3_shotnoise(data, bin=bin, los=los, **kw)
    nz = size / mattrs.boxsize.prod()
    norm = jnp.sum(data.weights * nz**2)
    spectrum = spectrum.map(lambda pole: pole.clone(norm=norm))
    #print(spectrum.value())
    spectrum_raw = spectrum
    spectrum = spectrum.clone(num_shotnoise=num_shotnoise)

    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_bispec_in_gpp_box
    from triumvirate.parameters import ParameterSet

    catalogue = ParticleCatalogue(*np.array(data.positions.T), nz=np.ones_like(data.weights) * nz)

    #trv_logger = setup_logger(log_level=20)
    #binning = Binning(space='fourier', scheme='lin', bin_min=edges[0], bin_max=edges[-1], num_bins=len(edges) - 1)
    paramset = dict(norm_convention='particle', form='diag', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None),
                    range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='cic', interlace=False,
                    boxsize=dict(zip('xyz', np.array(mattrs.boxsize))), ngrid=dict(zip('xyz', np.array(mattrs.meshsize))), verbose=20)
    print(paramset)
    paramset = ParameterSet(param_dict=paramset)
    results = compute_bispec_in_gpp_box(catalogue, paramset=paramset)
    #print(results['bk_raw'])
    print(spectrum.get(ell).values('num_shotnoise'))

    if plot:
        ax = plt.gca()
        ax.plot(take(results['bk_shot']), label='triumvirate shotnoise')
        ax.plot(spectrum.get(ell).values('shotnoise'), label='jaxpower shotnoise')
        ax.legend()
        plt.show()

        ax = plt.gca()
        ax.plot(take(results['bk_raw']), label='triumvirate')
        ax.plot(spectrum.get(ell).values('value') + spectrum.get(ell).values('shotnoise'), label='jaxpower')
        ax.legend()
        plt.show()


def test_triumvirate_box_correlation(plot=False):
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    mattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)

    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e4)
    data = generate_uniform_particles(mattrs, size, seed=32)
    # Triumvirate doesn't take weights for box statistics...
    #data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))

    kw = dict(resampler='cic', interlacing=False, compensate=True)
    mesh = data.paint(**kw)
    mean = mesh.mean()
    #mean = 1.
    mesh = mesh - mean
    edges = np.arange(0.1, mattrs.boxsize.min() / 3., 40.01)
    #ell = (0, 0, 0)
    ell = (1, 0, 1)
    #ell = (2, 0, 2)
    #ell = (3, 1, 2)
    #ell = (1, 3, 2)
    los = 'z'
    bin = BinMesh3CorrelationPoles(mattrs, edges=edges, basis='sugiyama-diagonal', ells=[ell])

    def take(array):
        return array.real

    correlation = compute_mesh3_correlation(mesh, los=los, bin=bin)
    num_shotnoise = compute_fkp3_shotnoise(data, bin=bin, los=los, **kw)
    nz = size / mattrs.boxsize.prod()
    norm = jnp.sum(data.weights * nz**2)
    correlation = correlation.map(lambda pole: pole.clone(norm=norm))
    #print(correlation.value())
    correlation_raw = correlation
    correlation = correlation.clone(num_shotnoise=num_shotnoise)

    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_3pcf_in_gpp_box
    from triumvirate.parameters import ParameterSet

    catalogue = ParticleCatalogue(*np.array(data.positions.T), nz=np.ones_like(data.weights) * nz)

    #trv_logger = setup_logger(log_level=20)
    #binning = Binning(space='fourier', scheme='lin', bin_min=edges[0], bin_max=edges[-1], num_bins=len(edges) - 1)
    paramset = dict(norm_convention='particle', form='diag', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None),
                    range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='cic', interlace=False,
                    boxsize=dict(zip('xyz', np.array(mattrs.boxsize))), ngrid=dict(zip('xyz', np.array(mattrs.meshsize))), verbose=20)

    paramset = ParameterSet(param_dict=paramset)
    results = compute_3pcf_in_gpp_box(catalogue, paramset=paramset)
    print(results['zeta_raw'])
    print(correlation.get(ell).values('value'))

    if plot:
        ax = plt.gca()
        ax.plot(take(results['zeta_shot']), label='triumvirate shotnoise')
        ax.plot(take(correlation.get(ell).values('shotnoise')), label='jaxpower shotnoise')
        ax.legend()
        plt.show()

        ax = plt.gca()
        ax.plot(take(results['zeta_raw']), label='triumvirate')
        ax.plot(take(correlation.get(ell).values('value') + correlation.get(ell).values('shotnoise')), label='jaxpower')
        ax.legend()
        plt.show()


def test_triumvirate_survey(plot=False):
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    mattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)
    pattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)

    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e4)
    data = generate_uniform_particles(pattrs, size, seed=32)
    # Triumvirate doesn't take weights for box statistics...
    #data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    randoms = generate_uniform_particles(pattrs, 2 * size, seed=42)
    fkp = FKPField(data, randoms)

    kw = dict(resampler='cic', interlacing=False, compensate=True)
    mesh = fkp.paint(**kw)
    edges = np.arange(0.01, mattrs.knyq[0], 0.01)
    ell = (0, 0, 0)
    #ell = (1, 0, 1)
    #ell = (2, 0, 2)
    #ell = (3, 1, 2)
    #ell = (1, 3, 2)
    los = 'local'
    bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis='sugiyama-diagonal', ells=[ell])

    def take(array):
        if sum(ell[:2]) % 2 == 0:
            return array.real
        return array.imag

    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)

    num_shotnoise = compute_fkp3_shotnoise(fkp, bin=bin, los=los, **kw)
    nz = size / mattrs.boxsize.prod()
    norm = jnp.sum(data.weights * nz**2)
    spectrum = spectrum.map(lambda pole: pole.clone(norm=norm))
    spectrum = spectrum.clone(num_shotnoise=num_shotnoise)

    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_bispec
    from triumvirate.parameters import ParameterSet
    from triumvirate.logger import setup_logger

    logger = setup_logger(20)
    data = ParticleCatalogue(*np.array(data.positions.T), ws=data.weights, nz=nz)
    randoms = ParticleCatalogue(*np.array(randoms.positions.T), ws=randoms.weights, nz=nz)

    #trv_logger = setup_logger(log_level=20)
    #binning = Binning(space='fourier', scheme='lin', bin_min=edges[0], bin_max=edges[-1], num_bins=len(edges) - 1)
    paramset = dict(norm_convention='particle', form='diag', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None),
                    range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='cic', interlace='off', alignment='centre', padfactor=0.,
                    boxsize=dict(zip('xyz', np.array(mattrs.boxsize))), ngrid=dict(zip('xyz', np.array(mattrs.meshsize))), verbose=20)

    paramset = ParameterSet(param_dict=paramset)
    results = compute_bispec(data, randoms, paramset=paramset, logger=logger)

    #print(spectrum.get(ell).values('num_shotnoise'))

    if plot:
        ax = plt.gca()
        ax.plot(take(results['bk_shot']), label='triumvirate shotnoise')
        ax.plot(spectrum.get(ell).values('shotnoise'), label='jaxpower shotnoise')
        ax.legend()
        plt.show()

        ax = plt.gca()
        k = results['k1_eff']
        ax.plot(k, k**2 * take(results['bk_raw']), label='triumvirate')
        k = spectrum.get(ell).coords('k')[..., 0]
        ax.plot(k, k**2 * (spectrum.get(ell).values('value') + spectrum.get(ell).values('shotnoise')), label='jaxpower')
        ax.legend()
        plt.show()


def test_triumvirate_survey_correlation(plot=False):
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    mattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)
    pattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)

    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e4)
    data = generate_uniform_particles(pattrs, size, seed=32)
    # Triumvirate doesn't take weights for box statistics...
    #data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    randoms = generate_uniform_particles(pattrs, 2 * size, seed=42)
    fkp = FKPField(data, randoms)

    kw = dict(resampler='cic', interlacing=False, compensate=True)
    mesh = fkp.paint(**kw)
    edges = np.arange(0.1, mattrs.boxsize.min() / 3., 40.01)
    ell = (0, 0, 0)
    ell = (1, 0, 1)
    #ell = (2, 0, 2)
    #ell = (3, 1, 2)
    #ell = (1, 3, 2)
    los = 'local'
    bin = BinMesh3CorrelationPoles(mattrs, edges=edges, basis='sugiyama-diagonal', ells=[ell])

    def take(array):
        return array.real

    correlation = compute_mesh3_correlation(mesh, los=los, bin=bin)
    num_shotnoise = compute_fkp3_shotnoise(fkp, bin=bin, los=los, **kw)
    nz = size / mattrs.boxsize.prod()
    norm = jnp.sum(data.weights * nz**2)
    correlation = correlation.map(lambda pole: pole.clone(norm=norm))
    correlation_raw = correlation
    correlation = correlation.clone(num_shotnoise=num_shotnoise)

    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_3pcf
    from triumvirate.parameters import ParameterSet
    from triumvirate.logger import setup_logger

    logger = setup_logger(20)
    data = ParticleCatalogue(*np.array(data.positions.T), ws=data.weights, nz=nz)
    randoms = ParticleCatalogue(*np.array(randoms.positions.T), ws=randoms.weights, nz=nz)

    #trv_logger = setup_logger(log_level=20)
    #binning = Binning(space='fourier', scheme='lin', bin_min=edges[0], bin_max=edges[-1], num_bins=len(edges) - 1)
    paramset = dict(norm_convention='particle', form='diag', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None),
                    range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='cic', interlace='off', alignment='centre', padfactor=0.,
                    boxsize=dict(zip('xyz', np.array(mattrs.boxsize))), ngrid=dict(zip('xyz', np.array(mattrs.meshsize))), verbose=20)

    paramset = ParameterSet(param_dict=paramset)
    results = compute_3pcf(data, randoms, paramset=paramset, logger=logger)

    print(correlation.get(ell).values('num_shotnoise'))

    if plot:
        ax = plt.gca()
        ax.plot(take(results['zeta_shot']), label='triumvirate shotnoise')
        ax.plot(take(correlation.get(ell).values('shotnoise')), label='jaxpower shotnoise')
        ax.legend()
        plt.show()

        ax = plt.gca()
        ax.plot(take(results['zeta_raw']), label='triumvirate')
        ax.plot(take(correlation.get(ell).values('value') + correlation.get(ell).values('shotnoise')), label='jaxpower')
        ax.legend()
        plt.show()


def test_normalization():
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    boxcenter = [1300., 0., 0.]
    mattrs = MeshAttrs(boxsize=1000., boxcenter=boxcenter, meshsize=128)

    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e5)
    data = generate_uniform_particles(mattrs, size, seed=32)
    data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    randoms = generate_uniform_particles(mattrs, size * 10, seed=42)
    fkp = FKPField(data, randoms)
    norm = compute_fkp3_normalization(fkp, cellsize=20., split=None)
    print('norm no split', norm)
    norm_split = compute_fkp3_normalization(fkp, cellsize=20., split=42)
    print('norm split', norm_split)


def survey_selection(size=int(1e7), seed=random.key(42), scale=0.25, paint=True):
    from jaxpower import ParticleField
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[0., 0., 1000], meshsize=64)
    xvec = mattrs.xcoords(kind='position', sparse=False)
    limits = mattrs.boxcenter - mattrs.boxsize / 4., mattrs.boxcenter + mattrs.boxsize / 4.
    # Generate Gaussian-distributed positions
    positions = scale * random.normal(seed, shape=(size, mattrs.ndim))
    #positions = scale * (2 * random.uniform(seed, shape=(size, mattrs.ndim)) - 1.)
    bscale = scale  # cut at 1 sigmas
    mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
    positions = positions * mattrs.boxsize + mattrs.boxcenter
    toret = ParticleField(positions, weights=1. * mask, attrs=mattrs)
    if paint: toret = toret.paint(resampler='ngp', interlacing=1, compensate=False)
    return toret


def test_fisher():
    from jaxpower.mesh3 import compute_fisher_scoccimarro

    selection = survey_selection()
    mattrs = selection.attrs
    edges = np.arange(0.01, mattrs.knyq[0], 0.01)
    bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis='scoccimarro', ells=[0, 2])
    fisher = compute_fisher_scoccimarro(selection, bin=bin)


def test_bincount():

    def digitize_overlapping_matrix(x, edges, weights=None):
        edges_1d = np.sort(np.unique(edges.ravel()))
        n_fine = len(edges_1d) - 1
        fine_idx = np.digitize(x, edges_1d) - 1
        if weights is None:
            counts_fine = np.bincount(fine_idx, minlength=n_fine)
        else:
            counts_fine = np.bincount(fine_idx, weights=weights, minlength=n_fine)
        M = ((edges_1d[:-1] >= edges[:, [0]]) &
            (edges_1d[1:]  <= edges[:, [1]])).astype(int)
        return M @ counts_fine

    edges = np.array([
        [0.0, 1.0],
        [0.5, 2.0],
        [2.0, 3.0],
    ])

    # Some data points and optional weights
    x = np.array([0.2, 0.6, 1.5, 2.5, 2.9])
    w = np.array([1.0, 2.0, 1.5, 0.5, 1.0])
    print(digitize_overlapping_matrix(x, edges))


def test_fftlog2(plot=False):
    from jaxpower.fftlog import SpectrumToCorrelation, CorrelationToSpectrum

    from cosmoprimo.fiducial import DESI

    k = (np.logspace(-6, 2, 1024, endpoint=False), np.logspace(-6, 2, 2048, endpoint=False))
    #k = (np.logspace(-6, 2, 1024, endpoint=False), np.logspace(-6, 2, 1024, endpoint=False))
    #k = (np.logspace(-6, 2, 128, endpoint=False), np.logspace(-6, 2, 256, endpoint=False))

    cosmo = DESI(engine='eisenstein_hu')
    pk_callable = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    if True:
        ells = [0]
        pk = pkt = pk_callable(k[0])[None, :]
        s, xi = SpectrumToCorrelation(k[0], ell=ells)(pk)
        k2, pk2 = CorrelationToSpectrum(s, ell=ells)(xi)

        if plot:
            ax = plt.gca()
            idx = 10
            for ill, ell in enumerate(ells):
                color = f'C{ill:d}'
                ax.plot(s[ill], xi[ill], linestyle='-', color=color)
            ax.set_xscale('log')
            plt.show()

            ax = plt.gca()
            for ill, ell in enumerate(ells):
                color = f'C{ill:d}'
                ax.loglog(k[0], pk[ill], linestyle='--', color=color)
                ax.loglog(k2[ill], pk2[ill], linestyle='-', color=color)
            plt.show()

    if True:
        bkt = pk_callable(k[0])[:, None] * pk_callable(k[1])
        ells = [(0, 0)]
        bk = bkt[None, :]
        s, zeta = SpectrumToCorrelation(k, ell=ells)(bk)
        k2, bk2 = CorrelationToSpectrum(s, ell=ells)(zeta)

        if plot:
            ax = plt.gca()
            idx = 10
            for ill, ell in enumerate(ells):
                color = f'C{ill:d}'
                ax.plot(s[0][ill], zeta[ill, :, idx], linestyle='-', color=color)
            ax.set_xscale('log')
            plt.show()

            ax = plt.gca()
            idx = 100
            for ill, ell in enumerate(ells):
                color = f'C{ill:d}'
                ax.loglog(k[0], bk[ill, :, idx], linestyle='--', color=color)
                ax.loglog(k2[0][ill], bk2[ill, :, idx], linestyle='-', color=color)
            plt.show()


def test_interp2d():

    def interp2d(x, y, xp, yp, fp):
        # fp shape: (len(xp), len(yp))
        interp_x = jax.vmap(lambda f: jnp.interp(x, xp, f), in_axes=1, out_axes=1)(fp)
        return jax.vmap(lambda f: jnp.interp(y, yp, f), in_axes=0, out_axes=0)(interp_x)

    x = jnp.logspace(-3, 3, 100)
    y = jnp.logspace(-2, 2, 80)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    zz = jnp.sin(xx) * jnp.cos(yy) / (1. + 0.1 * xx * yy)

    xnew = jnp.logspace(-3, 3, 150)
    ynew = jnp.logspace(-2, 2, 120)
    zznew = interp2d(xnew, ynew, x, y, zz)

    from matplotlib import pyplot as plt

    fig, lax = plt.subplots(2, 1, figsize=(6, 10))
    ax = lax[0]
    im = ax.pcolormesh(x, y, zz.T, shading='auto')
    ax.set_title('Original')
    plt.colorbar(im, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax = lax[1]
    im = ax.pcolormesh(xnew, ynew, zznew.T, shading='auto')
    ax.set_title('Interpolated')
    plt.colorbar(im, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()


def test_smooth_window(plot=False):
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    mattrs = MeshAttrs(boxsize=2000., meshsize=64, boxcenter=800.)
    ells = [(0, 0, 0), (2, 0, 2)]
    bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.03}, basis='sugiyama-diagonal', ells=ells, mask_edges='')
    #edgesin = np.arange(0., 1.1 * mattrs.knyq.max(), 0.002)
    edgesin = np.arange(0., 1.2 * mattrs.knyq.max(), 0.02)
    kin = (edgesin[:-1] + edgesin[1:]) / 2.
    bk = pk(kin)[:, None] * pk(kin)[None, :]
    edgesin = (edgesin, edgesin)
    kin = (kin, kin)

    def gaussian_survey(pattrs, mattrs=None, size=int(1e6), seed=random.key(42), scale=0.3, paint=False):
        # Generate Gaussian-distributed positions
        positions = scale * random.normal(seed, shape=(size, 3))
        bscale = scale  # cut at 1 sigma
        mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
        if mattrs is None: mattrs = pattrs
        positions = positions * pattrs.boxsize + pattrs.boxcenter
        toret = ParticleField(positions, weights=1. * mask, attrs=mattrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(mattrs, paint=True)
    norm = compute_normalization(selection, selection, selection)

    from jaxpower import get_smooth3_window_bin_attrs, interpolate_window_function
    kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, return_ellsin=True)
    poles = [1. / (1 + sum(ell)) * bk for ell in ellsin]

    from jaxpower.utils import plotter

    @plotter
    def plot_zeta(self, s=30., fig=None):
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            pole = self.get(ells=ell)
            idx = np.abs(pole.coords('s2') - s).argmin()
            ax.plot(pole.coords('s1'), pole.value().real[:, idx], label=f'$\ell={ell}$')
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log')
        return fig

    los = 'local'
    if False:
        sbin = BinMesh3CorrelationPoles(selection, edges={'step': 2 * selection.attrs.cellsize[0]}, **kw, batch_size=12)
        zeta = compute_mesh3_correlation(selection, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
        zeta = zeta.unravel()
        if plot:
            plot_zeta(zeta, show=True)

    fn = '_tests/zeta.h5'
    if False:
        zetas = []
        for scale in [1, 4]:
            selection = gaussian_survey(mattrs, mattrs=mattrs.clone(boxsize=scale * mattrs.boxsize), paint=True)
            sbin = BinMesh3CorrelationPoles(selection, edges={'step': 2 * selection.attrs.cellsize[0]}, **kw, batch_size=12)
            zeta = compute_mesh3_correlation(selection, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
            zetas.append(zeta.unravel())

        coords = jnp.logspace(-3, 4, 1024)
        xis = [interpolate_window_function(zeta, coords=coords, order=3) for zeta in zetas]
        limit = 0.4 * mattrs.boxsize[0]
        coords = list(next(iter(xis[0])).coords().values())
        mask = (coords[0] < limit)[:, None] * (coords[1] < limit)[None, :]
        weights = [jnp.maximum(mask, 1e-6), jnp.maximum(~mask, 1e-6)]
        zeta2 = zetas[0].sum(xis, weights=weights)
        if plot:
            plot_zeta(zeta2, show=True)
        zeta2.write(fn)

    zeta2 = read(fn)
    wmatrix = compute_smooth3_spectrum_window(zeta2, edgesin=edgesin, ellsin=ellsin, bin=bin)
    wpoles = wmatrix.dot(np.concatenate([p.ravel() for p in poles]), return_type=None)
    if plot:
        ax = plt.gca()
        for ill, ell in enumerate(wpoles.ells):
            color = f'C{ill:d}'
            pole = wpoles.get(ells=ell)
            factor = pole.coords('k').prod(axis=-1)
            ax.plot(factor * pole.value(), color=color, linestyle='-')
            print(ell, ellsin)
            ell = tuple(sorted(ell[:2])) + ell[2:]
            if ell in ellsin:
                i = ellsin.index(ell)
                from scipy import interpolate
                spline = interpolate.RectBivariateSpline(*kin, poles[i], kx=1, ky=1, s=0)
                pole = spline(*pole.coords('k').T, grid=False)
                ax.plot(factor * pole, color=color, linestyle=':')
        plt.show()


def test_smooth_window_synthetic(plot=False):
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    mattrs = MeshAttrs(boxsize=2000., meshsize=64, boxcenter=800.)
    ells = [(0, 0, 0), (2, 0, 2)]
    bin = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.03}, basis='sugiyama-diagonal', ells=ells, mask_edges='')
    #edgesin = np.arange(0., 1.1 * mattrs.knyq.max(), 0.002)
    edgesin = np.arange(0., 1.2 * mattrs.knyq.max(), 0.02)
    kin = (edgesin[:-1] + edgesin[1:]) / 2.
    bk = pk(kin)[:, None] * pk(kin)[None, :]
    edgesin = (edgesin, edgesin)
    kin = (kin, kin)

    from jaxpower import get_smooth3_window_bin_attrs
    kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=0, return_ellsin=True)
    poles = [1. / (1 + sum(ell)) * bk for ell in ellsin]

    from jaxpower.utils import plotter

    @plotter
    def plot_zeta(self, s=30., fig=None):
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            pole = self.get(ells=ell)
            idx = np.abs(pole.coords('s2') - s).argmin()
            ax.plot(pole.coords('s1'), pole.value().real[:, idx], label=f'$\ell={ell}$')
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log')
        return fig

    coords = [jnp.logspace(-3, 4, 1024)] * 2 #, jnp.logspace(-3, 4, 512)]
    def get_gaussian_damping(*s):
        s = np.meshgrid(*s, indexing='ij')
        return np.exp(-sum(ss**2 for ss in s) / 100.**2)
        #return np.exp(-sum(ss**2 for ss in s) / 40.**2)
        #return np.exp(-sum(ss**2 for ss in s) / 10.**2)

    from lsstypes import ObservableLeaf, ObservableTree

    zpoles = []
    for ell in kw['ells']:
        value = get_gaussian_damping(*coords)
        if ell != (0, 0, 0): value[...] = 0.
        #else: value[...] = 1.
        pole = ObservableLeaf(s1=coords[0], s2=coords[1], value=value, coords=['s1', 's2'], meta={'ell': ell})
        zpoles.append(pole)
    zeta = ObservableTree(zpoles, ells=kw['ells'])
    if plot:
        plot_zeta(zeta, s=1., show=True)

    wmatrix = compute_smooth3_spectrum_window(zeta, edgesin=edgesin, ellsin=ellsin, bin=bin)
    wpoles = wmatrix.dot(np.concatenate([p.ravel() for p in poles]), return_type=None)
    if plot:
        ax = plt.gca()
        for ill, ell in enumerate(wpoles.ells):
            color = f'C{ill:d}'
            pole = wpoles.get(ells=ell)
            factor = pole.coords('k').prod(axis=-1)
            k = pole.coords('k')[..., 0]
            ax.plot(k, factor * pole.value(), color=color, linestyle='-')
            print(ell, ellsin)
            ell = tuple(sorted(ell[:2])) + ell[2:]
            if ell in ellsin:
                i = ellsin.index(ell)
                from scipy import interpolate
                spline = interpolate.RectBivariateSpline(*kin, poles[i], kx=1, ky=1, s=0)
                pole = spline(*pole.coords('k').T, grid=False)
                ax.plot(k, factor * pole, color=color, linestyle=':')
        plt.show()


def test_basis():
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    mattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)

    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e4)
    data = generate_uniform_particles(mattrs, size, seed=32)

    kw = dict(resampler='cic', interlacing=False, compensate=True)
    mesh = data.paint(**kw)
    mean = mesh.mean()
    #mean = 1.
    mesh = mesh - mean

    if True:
        edges = np.arange(0.01, mattrs.knyq[0], 0.04)
        bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis='sugiyama', ells=[(0, 0, 0)], mask_edges='')
        spectrum = compute_mesh3_spectrum(mesh, los='local', bin=bin)
        for pole in spectrum:
            value = pole.value().reshape((len(edges) - 1,) * 2)
            print(value, np.unique(value).size, value.size)

    if True:
        edges = np.arange(0.0, 200, 4 * mattrs.cellsize[0])
        bin = BinMesh3CorrelationPoles(mattrs, edges=edges, basis='sugiyama', ells=[(0, 0, 0)], mask_edges='')
        correlation = compute_mesh3_correlation(mesh, los='local', bin=bin)
        for pole in correlation:
            value = pole.value().reshape((len(edges) - 1,) * 2)
            print(value, np.unique(value).size, value.size)


if __name__ == '__main__':

    #import os
    #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
    #os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'

    from jax import config
    config.update('jax_enable_x64', True)

    test_buffer()
    test_mesh3_spectrum()
    test_polybin3d()
    test_triumvirate_box()
    test_triumvirate_survey()
    test_normalization()
    test_fftlog2(plot=True)
    test_smooth_window_synthetic(plot=True)
    test_smooth_window(plot=True)