import time
from pathlib import Path
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (compute_mesh3_spectrum, BinMesh3Spectrum, MeshAttrs, Spectrum3Poles, generate_gaussian_mesh, generate_uniform_particles, FKPField, compute_normalization, compute_fkp3_spectrum_normalization, utils)


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


def test_triumvirate():
    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    attrs = MeshAttrs(meshsize=100, boxsize=1000., boxcenter=1000.)
    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    boxcenter = [1300., 0., 0.]
    attrs = MeshAttrs(boxsize=1000., boxcenter=boxcenter, meshsize=64)

    mesh = generate_gaussian_mesh(attrs, pkvec, seed=42, unitary_amplitude=True)
    size = int(1e5)
    data = generate_uniform_particles(attrs, size, seed=32)
    data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))

    mesh = data.paint(resampler='cic', interlacing=False, compensate=True)
    edges = np.linspace(0.01, 0.3, 30)
    bin = BinMesh3Spectrum(attrs, edges=edges, basis='sugiyama-diagonal', ells=[(0, 0, 0)])
    spectrum = compute_mesh3_spectrum(mesh, los='z', bin=bin)

    from triumvirate.logger import setup_logger
    from triumvirate.catalogue import ParticleCatalogue
    from triumvirate.threept import compute_bispec_in_gpp_box
    from triumvirate.dataobjs import Binning

    trv_logger = setup_logger(log_level=20)
    catalogue = ParticleCatalogue(*np.array(data.positions.T), ws=np.array(data.weights), nz=np.ones_like(data.weights))

    binning = Binning(space='fourier', scheme='lin', bin_min=edges[0], bin_max=edges[-1], num_bins=len(edges) - 1)
    results = compute_bispec_in_gpp_box(
                    catalogue,
                    degrees=(0, 0, 0),
                    binning=binning,
                    form='diag',
                    sampling_params={
                        'assignment': 'cic',
                        'boxsize': list(np.array(attrs.boxsize)),
                        'ngrid': list(np.array(attrs.meshsize))},
                    logger=trv_logger)

    ax = plt.gca()
    raw = spectrum.view()
    #print(spectrum.nmodes()[0] / (results['nmodes_1'] * results['nmodes_2']))
    print(raw / results['bk_raw'] * attrs.cellsize.prod())
    exit()
    ax.plot(results['bk_raw'], label='triumvirate')
    ax.plot(jnp.concatenate(spectrum.num), label='jaxpower')
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

    from jax import config
    config.update('jax_enable_x64', True)

    #test_mesh3_spectrum(plot=False)
    #test_timing()
    #test_polybin3d()
    #test_normalization()
    test_triumvirate()