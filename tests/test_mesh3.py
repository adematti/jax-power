import time
from pathlib import Path
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (compute_mesh3_spectrum, BinMesh3Spectrum, MeshAttrs, Spectrum3Poles, generate_gaussian_mesh, generate_uniform_particles, FKPField, compute_normalization, compute_fkp3_spectrum_normalization, compute_fkp3_spectrum_shotnoise, utils)


dirname = Path('_tests')


def test_mesh3_spectrum(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'local']
    attrs = MeshAttrs(meshsize=128, boxsize=1000.)

    for basis in ['sugiyama-diagonal', 'scoccimarro-equilateral']:

        ells = [0] if 'scoccimarro' in basis else [(0, 0, 0)]

        bin = BinMesh3Spectrum(attrs, edges={'step': 0.01}, basis=basis, ells=ells)

        @partial(jax.jit, static_argnames=['los'])
        def mock(attrs, bin, seed, los='x'):
            mesh = generate_gaussian_mesh(attrs, pkvec, seed=seed, unitary_amplitude=True)
            return compute_mesh3_spectrum(mesh, los=los, bin=bin)

        for los in list_los:

            nmock = 2
            t0 = time.time()
            power = mock(attrs, bin, random.key(43), los=los)
            jax.block_until_ready(power)
            print(f'time for jit {time.time() - t0:.2f}')
            t0 = time.time()
            for i in range(nmock):
                power = mock(attrs, bin, random.key(i + 42), los=los)
                jax.block_until_ready(power)
            print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
            power.save(get_fn(los))

        if plot:
            from matplotlib import pyplot as plt

            for los in list_los:
                power = Spectrum3Poles.load(get_fn(los=los))
                ax = power.plot().axes[0]
                ax.set_title(los)
                plt.show()


def test_timing():

    import time
    from jax import jit, random
    from jaxpower.mesh import create_sharded_random
    from jaxpower.mesh3 import get_real_Ylm, spherical_jn

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
    jn = spherical_jn(2)

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

    attrs = MeshAttrs(meshsize=256, boxsize=1000.)
    edges = [np.arange(0.01, 0.2, 0.01) for kmax in attrs.knyq]
    bin = BinMesh3Spectrum(attrs, edges=edges, basis='scoccimarro', ells=[0], buffer_size=4)
    # batch_size is mostly useless
    #bin = BinMesh3Spectrum(attrs, edges=edges, basis='scoccimarro', ells=[0], batch_size=12)
    cmeshs = [attrs.create(kind='real', fill=create_sharded_random(random.normal, random.key(42), shape=attrs.meshsize)).r2c() for axis in range(3)]


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


def test_polybin3d():

    from PolyBin3D import PolyBin3D, BSpec

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    attrs = MeshAttrs(meshsize=100, boxsize=1000., boxcenter=1000.)
    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    edges = [np.arange(0.01, 0.10, 0.02), np.arange(0.01, 0.15, 0.02), np.arange(0.01, 0.15, 0.02)]
    #edges = [np.arange(0.01, 0.08, 0.02), np.arange(0.01, 0.08, 0.02), np.arange(0.01, 0.08, 0.02)]

    bin = BinMesh3Spectrum(attrs, edges=edges, basis='scoccimarro', ells=[0, 2])

    for los in ['z', 'local'][1:]:

        kw = dict(gridsize=attrs.meshsize, boxsize=attrs.boxsize, boxcenter=attrs.boxcenter * (0. if los == 'z' else 1.))
        base = PolyBin3D(sightline='global' if los == 'z' else los, **kw, backend='jax', real_fft=False)
        bspec = BSpec(base, k_bins=edges[0],
                    lmax=2, k_bins_squeeze=edges[1],
                    include_partial_triangles=False)

        for imock in [0]:
            mesh = generate_gaussian_mesh(attrs, pkvec, seed=imock)
            t0 = time.time()
            spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
            #print('norm', spectrum.norm, compute_normalization(*([mesh.clone(value=jnp.ones_like(mesh.value))] * 3)))
            jax.block_until_ready(spectrum)
            t1 = time.time()
            bk = bspec.Bk_ideal(data=np.array(mesh.value), discreteness_correction=False)
            k123 = bspec.get_ks()
            t2 = time.time()
            print(f'jax-power took {t1 - t0:.2f} s')
            print(f'polybin took {t2 - t1:.2f} s')

            if imock == 0:
                ax = plt.gca()
                weight = k123.prod(axis=0)
                for name in ['b0', 'b2']:
                    ax.plot(weight * bk[name], color='C0')

                k = spectrum.xavg(projs=0, method='mid')
                weight = k.prod(axis=-1)
                for projs in [0, 2]:
                    ax.plot(weight * spectrum.view(projs=projs), color='C1')
                plt.show()


def test_triumvirate_box():
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    attrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)

    mesh = generate_gaussian_mesh(attrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e4)
    data = generate_uniform_particles(attrs, size, seed=32)
    # Triumvirate doesn't take weights for box statistics...
    #data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))

    kw = dict(resampler='cic', interlacing=False, compensate=True)
    mesh = data.paint(**kw)
    mean = mesh.mean()
    #mean = 1.
    #mesh = mesh - mean
    edges = np.arange(0.01, attrs.knyq[0], 0.01)
    ell = (0, 0, 0)
    #ell = (2, 0, 2)
    los = 'z'
    bin = BinMesh3Spectrum(attrs, edges=edges, basis='sugiyama-diagonal', ells=[ell])

    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    num_shotnoise = compute_fkp3_spectrum_shotnoise(data, bin=bin, convention=None, los=los, **kw)
    spectrum = spectrum.clone(norm=spectrum.norm * mean**3)
    spectrum = spectrum.clone(num_shotnoise=num_shotnoise)

    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_bispec_in_gpp_box
    from triumvirate.parameters import ParameterSet

    catalogue = ParticleCatalogue(*np.array(data.positions.T), nz=np.ones_like(data.weights) * jnp.sum(data.weights) / attrs.boxsize.prod())

    #trv_logger = setup_logger(log_level=20)
    #binning = Binning(space='fourier', scheme='lin', bin_min=edges[0], bin_max=edges[-1], num_bins=len(edges) - 1)
    paramset = dict(norm_convention='particle', form='diag', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None),
                    range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='cic', interlace='off',
                    boxsize=dict(zip('xyz', np.array(attrs.boxsize))), ngrid=dict(zip('xyz', np.array(attrs.meshsize))), verbose=20)
    print(paramset)
    paramset = ParameterSet(param_dict=paramset)
    results = compute_bispec_in_gpp_box(catalogue, paramset=paramset)
    print(np.array(num_shotnoise).ravel() / results['bk_shot'])

    ax = plt.gca()
    #ax.plot(results['bk_raw'] - results['bk_shot'], label='triumvirate')
    ax.plot(results['bk_raw'] - results['bk_shot'], label='triumvirate')
    ax.plot(spectrum.view(), label='jaxpower')
    ax.legend()
    plt.show()


def test_triumvirate_survey():
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    attrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=64)

    mesh = generate_gaussian_mesh(attrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e4)
    data = generate_uniform_particles(attrs, size, seed=32)
    # Triumvirate doesn't take weights for box statistics...
    #data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    randoms = generate_uniform_particles(attrs, 2 * size, seed=42)
    fkp = FKPField(data, randoms)

    kw = dict(resampler='cic', interlacing=False, compensate=True)
    mesh = fkp.paint(**kw)
    #mean = 1.
    #mesh = mesh - mean
    edges = np.arange(0.01, attrs.knyq[0], 0.01)
    #ell = (0, 0, 0)
    ell = (2, 0, 2)
    los = 'local'
    bin = BinMesh3Spectrum(attrs, edges=edges, basis='sugiyama-diagonal', ells=[ell])

    spectrum = compute_mesh3_spectrum(mesh, los=los, bin=bin)
    num_shotnoise = compute_fkp3_spectrum_shotnoise(data, bin=bin, los=los, **kw)
    mean = size / attrs.meshsize.prod(dtype=float)
    spectrum = spectrum.clone(norm=spectrum.norm * mean**3)
    #spectrum = spectrum.clone(num_shotnoise=jnp.array(num_shotnoise))

    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_bispec
    from triumvirate.parameters import ParameterSet
    from triumvirate.logger import setup_logger

    logger = setup_logger(20)
    data = ParticleCatalogue(*np.array(data.positions.T), ws=data.weights, nz=size / attrs.boxsize.prod())
    randoms = ParticleCatalogue(*np.array(randoms.positions.T), ws=randoms.weights, nz=size / attrs.boxsize.prod())

    #trv_logger = setup_logger(log_level=20)
    #binning = Binning(space='fourier', scheme='lin', bin_min=edges[0], bin_max=edges[-1], num_bins=len(edges) - 1)
    paramset = dict(norm_convention='particle', form='diag', degrees=dict(zip(['ell1', 'ell2', 'ELL'], ell)), wa_orders=dict(i=None, j=None),
                    range=[edges[0], edges[-1]], num_bins=len(edges) - 1, binning='lin', assignment='cic', interlace='off', alignment='centre', padfactor=0.,
                    boxsize=dict(zip('xyz', np.array(attrs.boxsize))),
                    ngrid=dict(zip('xyz', np.array(attrs.meshsize))), verbose=20)

    paramset = ParameterSet(param_dict=paramset)
    results = compute_bispec(data, randoms, paramset=paramset, logger=logger)

    ax = plt.gca()
    #ax.plot(results['bk_raw'] - results['bk_shot'], label='triumvirate')
    ax.plot(results['bk_raw'], label='triumvirate')
    ax.plot(spectrum.view(), label='jaxpower')
    ax.legend()
    plt.show()


def test_normalization():
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    attrs = MeshAttrs(meshsize=100, boxsize=1000., boxcenter=1000.)
    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    boxcenter = [1300., 0., 0.]
    attrs = MeshAttrs(boxsize=1000., boxcenter=boxcenter, meshsize=128)

    mesh = generate_gaussian_mesh(attrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e5)
    data = generate_uniform_particles(attrs, size, seed=32)
    data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    randoms = generate_uniform_particles(attrs, size, seed=42)
    fkp = FKPField(data, randoms)
    norm = compute_fkp3_spectrum_normalization(fkp, split=42)



if __name__ == '__main__':

    #import os
    #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
    #os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'

    from jax import config
    config.update('jax_enable_x64', True)

    #test_timing()
    #test_mesh3_spectrum(plot=False)
    #test_timing()
    #test_polybin3d()
    #test_normalization()
    test_triumvirate_box()
    #test_triumvirate_survey()