import os
import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from jaxpower import (BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                      BinMesh2CorrelationPoles, compute_mesh2_correlation,
                      generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, ParticleField, FKPField,
                      MeshAttrs, compute_mesh2, compute_mesh2_spectrum_mean, compute_mesh2_spectrum_window, compute_smooth2_spectrum_window,
                      compute_fkp2_normalization, compute_fkp2_shotnoise, compute_normalization, create_sharding_mesh, get_mesh_attrs,
                      Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles, WindowMatrix, read, utils, create_sharded_random)


dirname = Path('_tests')


def test_mesh2_spectrum(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.h5'.format(los)

    list_los = ['x', 'endpoint']

    for meshsize in [128, (128, 124, 122)]:
        mattrs = MeshAttrs(meshsize=meshsize, boxsize=1000.)
        bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))

        @partial(jax.jit, static_argnames=['los'])
        def mock(mattrs, bin, seed, los='x'):
            mesh = generate_gaussian_mesh(mattrs, pkvec, seed=seed, unitary_amplitude=True)
            return compute_mesh2_spectrum(mesh, los=los, bin=bin)

        for los in list_los:

            nmock = 5
            t0 = time.time()
            spectrum = mock(mattrs, bin, random.key(43), los=los)
            #print(spectrum)
            jax.block_until_ready(spectrum)
            print(f'time for jit {time.time() - t0:.2f}')
            t0 = time.time()
            for i in range(nmock):
                spectrum = mock(mattrs, bin, random.key(i + 42), los=los)
                jax.block_until_ready(spectrum)
            print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
            spectrum.write(get_fn(los))
            # remove first few bins because of binning effects
            assert np.allclose(spectrum.get(0).value()[2:], pk(spectrum.get(0).coords('k'))[2:], rtol=1e-2)
            assert tuple(spectrum.ells) == (0, 2, 4)

        if plot:
            from matplotlib import pyplot as plt

            for los in list_los:
                power = read(get_fn(los=los))
                ax = power.plot().axes[0]
                k = power.get(0).coords('k')
                ax.plot(k, k * pk(k))
                ax.set_title(los)
                plt.show()


def test_fkp2_shotnoise():

    mattrs = MeshAttrs(boxsize=1000., meshsize=128)
    size = int(1e5)
    data = generate_uniform_particles(mattrs, size, seed=32)
    data1 = data.clone(weights=create_sharded_random(jax.random.uniform, shape=(size,), seed=42))
    data2 = data.clone(weights=create_sharded_random(jax.random.uniform, shape=(size,), seed=84))
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.005}, ells=(0, 2))
    #num_shotnoise = compute_fkp2_shotnoise(data1, bin=bin)
    #assert np.allclose(num_shotnoise[0], jnp.sum(data1.weights**2))
    num_shotnoise = compute_fkp2_shotnoise([data1, data2], bin=bin)
    assert np.allclose(num_shotnoise[0], 0.)
    num_shotnoise = compute_fkp2_shotnoise([data1, data2], bin=bin, fields=[0, 0])
    assert np.allclose(num_shotnoise[0], jnp.sum(data1.weights * data2.weights))
    fkp = FKPField(data1, data2)
    num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
    assert np.allclose(num_shotnoise[0], jnp.sum(fkp.particles.weights**2))


def test_mesh2_correlation(plot=False):

    from cosmoprimo.fiducial import DESI

    mattrs = MeshAttrs(meshsize=128, boxsize=1000.)

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    ellsin = (0, 2, 4)
    edgesin = np.arange(0., jnp.sqrt(3.) * mattrs.knyq.max() + 0.001, 0.001)
    edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])
    kin = np.mean(edgesin, axis=-1)
    f, b = 0.8, 1.5
    beta = f / b

    def get_pk(k, pk):
        pk = pk(k)
        return np.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk,
                          0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk,
                          8. / 35 * beta ** 2 * pk])

    def get_xi(s, pk):
        return np.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk.to_xi(fftlog_kwargs={'ell': 0})(s),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk.to_xi(fftlog_kwargs={'ell': 2})(s),
                        8. / 35 * beta ** 2 * pk.to_xi(fftlog_kwargs={'ell': 4})(s)])

    poles = []
    for ell, value in zip(ellsin, get_pk(kin, pk)):
        poles.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=value, ell=ell))
    poles = Mesh2SpectrumPoles(poles)

    def get_fn(los='x'):
        return dirname / 'tmp_{}.h5'.format(los)

    list_los = ['x', 'local'][:1]
    edges = {'step': 4, 'max': 200}
    ells = (0, 2, 4)
    bin = BinMesh2CorrelationPoles(mattrs, edges=edges, ells=ells)
    bin2 = BinMesh2CorrelationPoles(mattrs, edges=edges, ells=ells, basis='bessel', batch_size=4)

    #@partial(jax.jit, static_argnames=['los'])
    def mock(mattrs, bin, seed, los='x'):
        mesh = generate_anisotropic_gaussian_mesh(mattrs, poles, los=los, seed=seed, unitary_amplitude=True, order=1)
        #mesh = generate_gaussian_mesh(mattrs, pkvec, seed=seed, unitary_amplitude=True)
        return compute_mesh2_correlation(mesh, los=los, bin=bin)

    for los in list_los:
        corr = mock(mattrs, bin, random.key(43), los=los)
        corr = corr.select(s=(30., 140.))
        corr.write(get_fn(los))

    if plot:
        from matplotlib import pyplot as plt

        for los in list_los:
            corr = read(get_fn(los=los))
            corr = corr.select(s=(30., 140.))
            ax = corr.plot().axes[0]
            s = corr.get(ells=0).s
            xi = get_xi(s, pk)
            for ill, ell in enumerate(corr.ells):
                ax.plot(s, s**2 * xi[ill], color='C{:d}'.format(ill), linestyle='--')
            ax.set_title(los)
            plt.show()

    for los in list_los:
        seed = random.key(43)
        corr = mock(mattrs, bin, seed, los=los)
        corr2 = mock(mattrs, bin2, seed, los=los)
        if plot:
            from matplotlib import pyplot as plt
            ax = plt.gca()
            for ill, ell in enumerate(corr.ells):
                color = 'C{:d}'.format(ill)
                pole = corr.get(ell)
                ax.plot(pole.coords('s'), pole.coords('s')**2 * pole.value(), color=color, linestyle='-')
                pole = corr2.get(ell)
                ax.plot(pole.coords('s'), pole.coords('s')**2 * pole.value(), color=color, linestyle='--')
            ax.set_title(los)
            plt.show()


def test_fkp2_spectrum(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.h5'.format(los)

    list_los = ['x', 'endpoint'][:1]
    boxcenter = [1300., 0., 0.]
    mattrs = MeshAttrs(boxsize=1000., boxcenter=boxcenter, meshsize=128)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))

    for los in list_los:
        @partial(jax.jit, static_argnames=['los'])
        def mock(seed, los='x'):
            mesh = generate_gaussian_mesh(mattrs, pkvec, seed=seed, unitary_amplitude=True)
            size = int(1e5)
            data = generate_uniform_particles(mattrs, size, seed=32)
            data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
            randoms = generate_uniform_particles(mattrs, size, seed=42)
            fkp = FKPField(data, randoms)
            mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='complex')
            spectrum = compute_mesh2_spectrum(mesh, bin=bin, los=los)
            norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=None)
            num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
            return spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

        nmock = 5
        t0 = time.time()
        spectrum = mock(random.key(43), los=los)
        jax.block_until_ready(spectrum)
        print(f'time for jit {time.time() - t0:.2f}')
        t0 = time.time()
        for i in range(nmock):
            jax.block_until_ready(mock(random.key(i + 42), los=los))
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
        spectrum.write(get_fn(los))

    if plot:
        from matplotlib import pyplot as plt

        for los in list_los:
            spectrum = read(get_fn(los=los))
            spectrum.select(k=(0., 0.2))
            ax = spectrum.plot().axes[0]
            k = spectrum.get(ells=0).coords('k')
            ax.plot(k, k * pk(k), color='k')
            ax.set_title(los)
            plt.show()


def test_mesh2_spectrum_mean(plot=False):

    def pk(k):
        kp = 0.01
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    f, b = 0.8, 1.5
    beta = f / b
    poles = {0: lambda k: (1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(k),
             2: lambda k: 0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(k),
             4: lambda k: 8. / 35 * beta ** 2 * pk(k)}

    list_los = [('x', None), ('endpoint', None), ('endpoint', 'local')][1:2]
    mattrs = MeshAttrs(boxsize=1000., meshsize=64, boxcenter=1000.)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))

    @partial(jax.jit, static_argnames=['los', 'thlos'])
    def mean(los='x', thlos=None):
        theory = poles
        if thlos is not None:
            theory = (poles, thlos)
        return compute_mesh2_spectrum_mean(mattrs, theory=theory, los=los, bin=bin)

    for los, thlos in list_los:

        #power_mock = mock(random.key(43), los=los)
        t0 = time.time()
        spectrum_mean = mean(los=los, thlos=thlos)
        print(f'time for jit {time.time() - t0:.2f}')
        nmock = 2
        t0 = time.time()
        for i in range(nmock):
            jax.block_until_ready(mean(los=los, thlos=thlos))
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')

        if plot:
            from matplotlib import pyplot as plt
            ax = spectrum_mean.plot().axes[0]
            for ell in spectrum_mean.ells:
                k = spectrum_mean.get(ells=ell).coords('k')
                ax.plot(k, k * poles[ell](k), color='k', linestyle='--')
            ax.set_title(los)
            plt.show()


def test_checkpoint():
    from jaxpower.utils import Interpolator1D

    def pk(k):
        kp = 0.01
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    f, b = 0.8, 1.5
    beta = f / b
    kinedges = np.linspace(0.001, 0.7, 100)
    kin = (kinedges[:-1] + kinedges[1:]) / 2.
    ells = (0, 2, 4)
    poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])

    mattrs = MeshAttrs(boxsize=2000., meshsize=350, boxcenter=1000.)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=ells)

    def make_callable(poles):
        def get_fun(ill):
            return lambda k: 1. * k * np.mean(poles[ill]) #jnp.interp(k, kin, poles[ill], left=0., right=0.)
        return {ell: get_fun(ill) for ill, ell in enumerate(ells)}

    def make_callable(poles):
        knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True, hermitian=True))).ravel()
        interp = Interpolator1D(kin, knorm)
        del knorm
        def get_fun(ill):
            return lambda k: interp(poles[ill])
        return {ell: get_fun(ill) for ill, ell in enumerate(ells)}

    def mean(poles, los='local'):
        theory = make_callable(poles)
        if los == 'local':
            theory = (theory, los)
            los = 'firstpoint'
        return compute_mesh2_spectrum_mean(mattrs, theory=theory, bin=bin, los=los)

    #print(jax.grad(lambda poles: mean(poles).view()[2])(poles))
    from jax.ad_checkpoint import checkpoint_name

    def mock(poles, los='local', seed=42):
        mesh = generate_anisotropic_gaussian_mesh(mattrs, make_callable(poles), los=los, seed=seed)
        #mesh = jax.checkpoint(lambda poles: generate_anisotropic_gaussian_mesh(mattrs, make_callable(poles), los='x', seed=seed))(poles)
        return compute_mesh2_spectrum(mesh, bin=bin, los={'local': 'firstpoint'}.get(los, los)).view()[-3]

    def power(mesh, los='local'):
        return compute_mesh2_spectrum(mesh, bin=bin, los={'local': 'firstpoint'}.get(los, los)).view()[-3]

    def gaussian_survey(mattrs, size=int(1e6), seed=random.key(42), scale=0.03, paint=False):
        # Generate Gaussian-distributed positions
        positions = scale * mattrs.boxsize * random.normal(seed, shape=(size, 3))
        toret = ParticleField(positions + mattrs.boxcenter, attrs=mattrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(mattrs, paint=True)

    #@partial(jax.checkpoint, static_argnums=(2,))
    policy = jax.checkpoint_policies.write_only_these_names('save')

    #@partial(jax.checkpoint, static_argnums=2)
    def apply_selection(mesh, selection, cv=False):
        mesh = mesh * selection
        if not cv:  # with radial integral constraint
            dmin = np.min(selection.boxcenter - selection.boxsize / 2.)
            dmax = (1. + 1e-9) * np.sqrt(np.sum((selection.boxcenter + selection.boxsize / 2.)**2))
            edges = jnp.linspace(dmin, dmax, 1000)
            rnorm = jnp.sqrt(sum(xx**2 for xx in selection.coords(sparse=True))).ravel()
            ibin = jnp.digitize(rnorm, edges, right=False)
            bw = jnp.bincount(ibin, weights=mesh.ravel(), length=len(edges) + 1)
            b = jnp.bincount(ibin, weights=selection.ravel(), length=len(edges) + 1)
            # Integral constraint
            bw = checkpoint_name(bw / jnp.where(b == 0., 1., b), 'save')  # (integral of W * delta) / (integral of W)
            mesh -= bw[ibin].reshape(selection.shape) * selection
        return mesh

    def mock_diff(theory, selection, los='local', seed=42, unitary_amplitude=True):
        mesh = generate_anisotropic_gaussian_mesh(selection.mattrs, (kinedges, list(theory)), los=los, seed=seed,
                                                  unitary_amplitude=unitary_amplitude)
        los = {'local': 'firstpoint'}.get(los, los)
        #return compute_mesh2_spectrum(apply_selection(mesh, selection, False), bin=bin, los=los).view()[-3]
        toret = [compute_mesh2_spectrum(apply_selection(mesh, selection, cv=cv), bin=bin, los=los) for cv in [False, True]]
        return toret[0].clone(value=toret[0].value() - toret[1].value()).value()[-3]

    def mock_vmap(poles, los='local'):
        seeds = random.split(random.key(42), 4)
        def func(seed):
            mesh = generate_anisotropic_gaussian_mesh(mattrs, make_callable(poles), los=los, seed=seed)
            #mesh = jax.checkpoint(lambda poles: generate_anisotropic_gaussian_mesh(attrs, make_callable(poles), los='x', seed=seed))(poles)
            return compute_mesh2_spectrum(mesh, bin=bin, los={'local': 'firstpoint'}.get(los, los)).view()
        return jnp.mean(jax.vmap(func)(seeds), axis=0)

    option = 1
    if option == 1:
        func = lambda poles, **kwargs: mock_diff(poles, **kwargs)
        arg = poles
        kw = dict(selection=selection)
    elif option == 2:
        func = lambda poles, **kwargs: mock(poles, **kwargs)
        arg = poles
        kw = dict()
    elif option == 3:
        func = lambda mesh, **kwargs: power(mesh, **kwargs)
        arg = selection
        kw = dict()
    #func = jax.checkpoint(func, policy=jax.checkpoint_policies.write_only_these_names('mock'))
    #func = jax.checkpoint(func, policy=jax.checkpoint_policies.write_anything_except_these_names('tmp1', 'tmp2'))
    from jax.ad_checkpoint import print_saved_residuals
    print_saved_residuals(func, arg, **kw)
    #exit()
    func = jax.grad(func)
    #func = jax.jacrev(func)
    func = jax.jit(func)
    for i in range(1):
        t0 = time.time()
        tmp = func(arg + i * 1e-6, **kw)
        jax.block_until_ready(tmp)
        #print(tmp)
        print(time.time() - t0)


def test_gaunt():
    import itertools
    import sympy as sp
    from sympy.physics.wigner import real_gaunt
    from jaxpower.utils import compute_sympy_real_gaunt, get_Ylm

    if False:
        for ell1, ell2, ell3 in itertools.product((0, 2, 4), (0, 2), (0, 2)):
            ms = list(itertools.product(list(range(-ell1, ell1 + 1)),
                                        list(range(-ell2, ell2 + 1)),
                                        list(range(-ell3, ell3 + 1))))
            for m1, m2, m3 in ms:
                g = compute_sympy_real_gaunt((ell1, m1), (ell2, m2), (ell3, m3))
                print(g)

    if False:
        for ell in (0, 2, 4):
            for m in range(-ell, ell + 1):
                theta = sp.Symbol("theta")
                phi = sp.Symbol("phi")
                expr = sp.Znm(ell, m, theta, phi)
                print(ell, m, expr)
                #expr = sp.simplify(sp.Znm(ell, m, theta, phi).expand(func=True))

    if False:
        rng = np.random.RandomState(seed=42)
        xyz = rng.uniform(-1., 1., (100, 3))
        xyz /= np.sqrt(np.sum(xyz**2, axis=-1))[..., None]
        for ell in (0, 2, 4):
            for m in range(-ell, ell + 1):
                tmp = get_Ylm(ell, m, modules=('scipy',), real=True)(*xyz.T)
                tmp2 = get_Ylm(ell, m, real=True)(*xyz.T)
                assert np.allclose(tmp2, tmp, rtol=1e-6, atol=1e-6)

    if False:
        for ell1, ell2, ell3 in itertools.product((0, 2, 4), (0, 2), (0, 2)):
            ms = list(itertools.product(list(range(-ell1, ell1 + 1)),
                                        list(range(-ell2, ell2 + 1)),
                                        list(range(-ell3, ell3 + 1))))
            if any(compute_sympy_real_gaunt((ell1, m1), (ell2, m2), (ell3, m3)) for m1, m2, m3 in ms):

                for m1, m2, m3 in ms:
                    mattrs = MeshAttrs(boxsize=500, meshsize=64)
                    mesh = mattrs.create(kind='hermitian_complex')
                    kvec = mesh.coords(sparse=True)
                    knorm = sum(kk**2 for kk in kvec)**0.5
                    kedges = jnp.array([0.8 * mesh.knyq.min(), 0.9 * mesh.knyq.min()])
                    kmask = (knorm >= kedges[0]) & (knorm <= kedges[-1])
                    bin = BinMesh2SpectrumPoles(mesh, edges=kedges)
                    mesh = mesh.clone(value=kmask * get_Ylm(ell1, m1, real=True)(*kvec) * get_Ylm(ell2, m2, real=True)(*kvec) * get_Ylm(ell3, m3, real=True)(*kvec))
                    value = bin(mesh)[0]
                    g = float(compute_sympy_real_gaunt((ell1, m1), (ell2, m2), (ell3, m3))) / (4 * np.pi)
                    if not np.allclose(value, g, rtol=1e-2, atol=1e-3):
                        print(ell1, ell2, ell3, m1, m2, m3, value, g)

    if False:
        ell = 2
        for m in range(-ell, ell + 1):
            print(ell, m, get_Ylm(ell, m, real=True)(0, 0, 1))

    if True:
        rng = np.random.RandomState(seed=42)
        xyz = rng.uniform(-1., 1., (100, 3))
        xyz /= np.sqrt(np.sum(xyz**2, axis=-1))[..., None]
        for ell in (0, 2, 4):
            for m in range(-ell, ell + 1):
                tmp = get_Ylm(ell, m, modules=('scipy',), real=False)(*xyz.T)
                tmp2 = get_Ylm(ell, m, real=False)(*xyz.T)
                assert np.allclose(tmp2, tmp, rtol=1e-6, atol=1e-6)


def test_window_box(plot=False):

    def get_theory(kmax=0.3, dk=0.005):
        # Return theory power spectrum
        from cosmoprimo.fiducial import DESI
        cosmo = DESI(engine='eisenstein_hu')
        z = 1.
        pk1d = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
        ellsin = (0, 2, 4)
        edgesin = jnp.arange(0., kmax, dk)
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
        kin = np.mean(edgesin, axis=-1)
        f, b = cosmo.growth_rate(z), 1.5
        beta = f / b
        shotnoise = (1e-3)**(-1)
        pk = pk1d(kin)
        theory = [(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk + shotnoise,
                    (4. / 3. * beta + 4. / 7. * beta ** 2) * pk,
                    8. / 35 * beta ** 2 * pk]

        poles = []
        for ell, value in zip(ellsin, theory):
            poles.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=value, num_shotnoise=shotnoise * np.ones_like(kin) * (ell == 0), ell=ell))
        return Mesh2SpectrumPoles(poles)

    mattrs = MeshAttrs(boxsize=1000., meshsize=64)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'min': 0, 'step': 0.01}, ells=(0, 2, 4))
    seed = random.key(42)
    los = 'z'
    theory = get_theory(kmax=mattrs.knyq.max())
    # Setting unitary_amplitude = True to reduce noise (actually, there is no noise at all, so no need for multiple realizations)
    mesh = generate_anisotropic_gaussian_mesh(mattrs, poles=theory, los=los, seed=seed, unitary_amplitude=True)
    mean = compute_mesh2_spectrum(mesh, bin=bin, los=los)

    # edges and ells for input theory
    edgesin = theory.get(ells=0).edges('k')
    ellsin = (0, 2, 4)
    # bin is still the binning operator
    wmat = compute_mesh2_spectrum_window(mattrs, edgesin=edgesin, ellsin=ellsin, bin=bin, los=los, pbar=True)
    test = wmat.dot(theory, return_type=None)
    wmat_rebin = wmat.at.theory.select(k=slice(0, None, 2)).at.theory.select(k=slice(0, -1))
    test_rebin = wmat_rebin.dot(theory.select(k=slice(0, None, 2)).select(k=slice(0, -1)), return_type=None)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for ill, ell in enumerate(test.ells):
            color = 'C{:d}'.format(ill)
            pole = mean.get(ells=ell)
            ax.plot(pole.coords('k'), pole.coords('k') * pole.value(), color=color, linestyle='-')
            pole = test.get(ells=ell)
            ax.plot(pole.coords('k'), pole.coords('k') * pole.value(), color=color, linestyle='--')
            pole = test_rebin.get(ells=ell)
            ax.plot(pole.coords('k'), pole.coords('k') * pole.value(), color=color, linestyle=':')
        plt.show()


def test_window(plot=False):

    from jaxpower.mesh2 import compute_normalization

    ellsin = (0, 2, 4)
    edgesin = np.array([0.1, 0.11])
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    kin = np.mean(edgesin, axis=-1)

    poles = []
    for ell in ellsin:
        poles.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=np.ones_like(kin), ell=ell))
    theory = Mesh2SpectrumPoles(poles)

    mattrs = MeshAttrs(boxsize=500., meshsize=64, boxcenter=1000.)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))

    def gaussian_survey(mattrs, size=int(1e6), seed=random.key(42), scale=0.2, paint=False):
        # Generate Gaussian-distributed positions
        positions = jnp.array([1., 0.2, 0.2]) * scale * random.normal(seed, shape=(size, 3))
        bscale = scale
        mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
        positions = positions * mattrs.boxsize + mattrs.boxcenter
        toret = ParticleField(positions, weights=1. * mask, attrs=mattrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(mattrs, paint=True)
    norm = [compute_normalization(selection, selection)] * len(bin.ells)

    for flag in ['smooth', 'infinite']:
        for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')]:
            print(flag, los, thlos)
            mean = compute_mesh2_spectrum_mean(selection, theory=(theory, thlos) if thlos is not None else theory, bin=bin, los=los).clone(norm=norm)
            wmatrix = compute_mesh2_spectrum_window(selection, edgesin=edgesin, ellsin=(ellsin, thlos) if thlos is not None else ellsin, bin=bin, los=los, pbar=True, norm=norm, flags=(flag,))
            if plot:
                from matplotlib import pyplot as plt
                ax = plt.gca()
                for ill, ell in enumerate(mean.ells):
                    color = 'C{:d}'.format(ill)
                    pole = mean.get(ells=ell)
                    ax.plot(pole.coords('k'), pole.value().real, color=color, linestyle='-')
                    kw = dict(ells=0)
                    if thlos == 'firstpoint': kw.update(wa_orders=0)
                    tmp = wmatrix.at.observable.get(ells=ell).at.theory.get(**kw).value().real
                    ax.plot(pole.coords('k'), tmp, color=color, linestyle='--')
                plt.show()


def test_window_timing():

    from jaxpower import compute_mesh2_spectrum_window
    from jaxpower.utils import Interpolator1D

    ellsin = (0, 2, 4)
    edgesin = np.linspace(0.01, 0.1, 20)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    kin = np.mean(edgesin, axis=-1)

    mattrs = MeshAttrs(boxsize=500., meshsize=64, boxcenter=1000.)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))

    def gaussian_survey(mattrs, size=int(1e6), seed=random.key(42), scale=0.2, paint=False):
        # Generate Gaussian-distributed positions
        positions = jnp.array([1., 0.2, 0.2]) * scale * random.normal(seed, shape=(size, 3))
        bscale = scale
        mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
        positions = positions * mattrs.boxsize + mattrs.boxcenter
        toret = ParticleField(positions, weights=1. * mask, attrs=mattrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(mattrs, paint=True)
    norm = compute_fkp2_normalization(selection, selection)

    for flag in ['smooth', 'infinite'][-1:]:
        for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')][-1:]:
            print(flag, los, thlos)
            t0 = time.time()
            wmatrix = compute_mesh2_spectrum_window(selection, edgesin=edgesin, ellsin=(ellsin, thlos) if thlos is not None else ellsin, bin=bin, los=los, pbar=True, norm=norm, flags=(flag,))
            print(f'{time.time() - t0:.2f}')


def test_smooth_window(plot=False):
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    mattrs = MeshAttrs(boxsize=2000., meshsize=100, boxcenter=800.)
    ells = (0, 2, 4)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=ells)
    #edgesin = np.arange(0., 1.1 * mattrs.knyq.max(), 0.002)
    edgesin = np.arange(0., mattrs.knyq.max(), 0.005)
    kin = (edgesin[:-1] + edgesin[1:]) / 2.
    ellsin = (0, 2, 4)
    f, b = 0.8, 1.5
    beta = f / b
    poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])[:len(ellsin)]
    #poles = poles.at[2:].set(0.)
    #klim = (0.1, 0.106)
    #poles = poles.at[..., (kin < klim[0]) | (kin > klim[1])].set(0.)

    def gaussian_survey(mattrs, size=int(1e6), seed=random.key(42), scale=0.1, paint=False):
        # Generate Gaussian-distributed positions
        positions = scale * random.normal(seed, shape=(size, 3))
        bscale = 2. * scale  # cut at 2 sigmas
        mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
        positions = positions * mattrs.boxsize + mattrs.boxcenter
        toret = ParticleField(positions, weights=1. * mask, attrs=mattrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(mattrs, paint=True)
    norm = compute_normalization(selection, selection)
    #selection = selection.clone(value=selection.value.at[...].set(1.))
    #sbin = BinMesh2CorrelationPoles(selection, edges={'step': selection.attrs.cellsize.min()}, ells=(0,))
    sbin = BinMesh2CorrelationPoles(selection, edges={}, ells=(0, 2, 4))

    from jaxpower.utils import plotter

    @plotter
    def plot_xi(self, fig=None):
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            pole = self.get(ells=ell)
            ax.plot(pole.coords('s'), pole.value().real)
        ax.legend()
        ax.grid(True)
        #ax.set_xscale('log')
        return fig

    for (los, thlos) in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')][1:2]:
        mean = compute_mesh2_spectrum_mean(selection, theory=(edgesin, list(poles)) if thlos is None else (edgesin, list(poles), thlos),
                                           bin=bin, los=los).clone(norm=[norm] * len(bin.ells))
        wmatrix = compute_mesh2_spectrum_window(selection, edgesin=edgesin, ellsin=ellsin if thlos is None else (ellsin, thlos),
                                                bin=bin, los=los, norm=norm, flags=['smooth'])
        xi = compute_mesh2_correlation(selection, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
        #plot_xi(xi, show=True)

        wmatrix2 = compute_smooth2_spectrum_window(xi, edgesin=edgesin, ellsin=ellsin, bin=bin)
        #wmatrix2.plot(show=True)
        wpoles = wmatrix.dot(poles.ravel(), return_type=None)
        wpoles2 = wmatrix2.dot(poles.ravel(), return_type=None)
        if plot:
            ax = plt.gca()
            for ill, ell in enumerate(wpoles2.ells):
                ax.plot(kin, kin * poles[ill], color='k')
                k = wpoles2.get(ells=ell).coords('k')
                ax.plot(k, k * mean.get(ells=ell).value(), color='C0')
                ax.plot(k, k * wpoles.get(ells=ell).value(), color='C1')
                ax.plot(k, k * wpoles2.get(ells=ell).value(), color='C2')
            plt.show()


def test_smooth_window(plot=False):
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    mattrs = MeshAttrs(boxsize=2000., meshsize=100, boxcenter=800.)
    ells = (0, 2, 4)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=ells)
    #edgesin = np.arange(0., 1.1 * mattrs.knyq.max(), 0.002)
    edgesin = np.arange(0., 1.2 * mattrs.knyq.max(), 0.005)
    kin = (edgesin[:-1] + edgesin[1:]) / 2.
    ellsin = (0, 2, 4)
    f, b = 0.8, 1.5
    beta = f / b
    poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])[:len(ellsin)]
    #poles = poles.at[2:].set(0.)
    #klim = (0.1, 0.106)
    #poles = poles.at[..., (kin < klim[0]) | (kin > klim[1])].set(0.)

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
    norm = compute_normalization(selection, selection)
    #selection = selection.clone(value=selection.value.at[...].set(1.))
    #sbin = BinMesh2CorrelationPoles(selection, edges={'step': selection.attrs.cellsize.min()}, ells=(0,))
    from jaxpower import get_smooth2_window_bin_attrs, interpolate_window_function
    kw = get_smooth2_window_bin_attrs(ells, ellsin)
    los = 'local'

    from jaxpower.utils import plotter

    @plotter
    def plot_xi(self, fig=None):
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            pole = self.get(ells=ell)
            color = f'C{ill:d}'
            ax.plot(pole.coords('s'), pole.value().real, color=color, label=rf'$\ell={ell}$')
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log')
        return fig

    #kw['ells'] = [0, 2, 4]
    sbin = BinMesh2CorrelationPoles(selection, edges=None, **kw)
    xi = compute_mesh2_correlation(selection, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))

    xis = []
    for scale in [1, 4]:
        selection = gaussian_survey(mattrs, mattrs=mattrs.clone(boxsize=scale * mattrs.boxsize), paint=True)
        sbin = BinMesh2CorrelationPoles(selection, edges={'step': 2 * selection.attrs.cellsize[0]}, **kw, basis='bessel', batch_size=12)
        xi = compute_mesh2_correlation(selection, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
        xis.append(xi)
    large = xis[1]
    coords = jnp.logspace(-3, 4, 4 * 1024)
    xis = [interpolate_window_function(xi, coords=coords, order=3) for xi in xis]
    limits = [0, 0.4 * mattrs.boxsize[0], 2. * mattrs.boxsize[0]]
    weights = [jnp.maximum((coords >= limits[i]) & (coords < limits[i + 1]), 1e-10) for i in range(len(limits) - 1)]
    xi2 = xis[0].sum(xis, weights=weights)
    if plot:
        from matplotlib import pyplot as plt
        ax = plot_xi(xi2).axes[0]
        for ill, ell in enumerate(large.ells):
            pole = large.get(ells=ell)
            color = f'C{ill:d}'
            ax.plot(pole.coords('s'), pole.value().real, color=color, linestyle='--')
        plt.show()

    if False:
        kbin = BinMesh2SpectrumPoles(selection, edges=None, **kw)
        pk = compute_mesh2_spectrum(selection, bin=kbin, los=los).clone(norm=[norm] * len(kbin.ells))
        xi = pk.to_correlation(s=jnp.arange(0.001, mattrs.boxsize[0], mattrs.cellsize[0]))
        if plot:
            plot_xi(xi, show=True)

    wmatrix = compute_smooth2_spectrum_window(xi, edgesin=edgesin, ellsin=ellsin, bin=bin)
    wmatrix2 = compute_smooth2_spectrum_window(xi2, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
    #wmatrix2.plot(show=True)
    wpoles = wmatrix.dot(poles.ravel(), return_type=None)
    wpoles2 = wmatrix2.dot(poles.ravel(), return_type=None)
    if plot:
        ax = plt.gca()
        for ill, ell in enumerate(wpoles.ells):
            ax.plot(kin, kin * poles[ill], color='k')
            pole = wpoles.get(ells=ell)
            ax.plot(pole.coords('k'), pole.coords('k') * pole.value(), color='C1')
            pole = wpoles2.get(ells=ell)
            ax.plot(pole.coords('k'), pole.coords('k') * pole.value(), color='C2')
        plt.show()


def test_sympy():
    import sympy as sp

    def tophat(ell):
        k, s, a = sp.symbols('k s a', real=True, positive=True)
        integrand = sp.simplify(k**2 * sp.expand_func(sp.jn(ell, k * s)))
        # i^ell; we take in the imaginary part of the odd power spectrum multipoles
        expr = (-1)**(ell // 2) / (2 * sp.pi**2) * sp.integrate(integrand, (k, 0, a))
        expr_lows = sp.series(expr, x=s, x0=0, n=8).removeO()
        print(expr)
        print(expr_lows)

    def point(ell):
        x = sp.symbols('x', real=True, positive=True)
        integrand = sp.simplify(sp.expand_func(sp.jn(ell, x)))
        # i^ell; we take in the imaginary part of the odd power spectrum multipoles
        expr = integrandells = (0, 2, 4)
        expr_lows = sp.series(expr, x=x, x0=0, n=8).removeO()
        print(expr)
        print(expr_lows)

    for ell in [0, 2, 4]:
        #tophat(ell)
        point(ell)



def test_sharded_spectrum():

    def get_theory(kmax=0.3, dk=0.005):
        # Return theory power spectrum
        from cosmoprimo.fiducial import DESI
        cosmo = DESI(engine='eisenstein_hu')
        z = 1.
        pk1d = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
        ellsin = (0, 2, 4)
        edgesin = jnp.arange(0., kmax, dk)
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
        kin = np.mean(edgesin, axis=-1)
        f, b = cosmo.growth_rate(z), 1.5
        beta = f / b
        shotnoise = (1e-3)**(-1)
        pk = pk1d(kin)
        theory = [(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk + shotnoise,
                    (4. / 3. * beta + 4. / 7. * beta ** 2) * pk,
                    8. / 35 * beta ** 2 * pk]

        poles = []
        for ell, value in zip(ellsin, theory):
            poles.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=value, num_shotnoise=shotnoise * np.ones_like(kin) * (ell == 0), ell=ell))
        return Mesh2SpectrumPoles(poles)

    def compute_spectrum():
        theory = get_theory()
        mesh = generate_anisotropic_gaussian_mesh(mattrs, poles=theory, los='z', seed=(43, 'index'), unitary_amplitude=True)
        #fill = jnp.sin(sum(xx**2 for xx in mattrs.xcoords(kind='position')))
        #mesh = mattrs.create(kind='real', fill=fill)
        print(mesh.std())
        data = generate_uniform_particles(pattrs, size, seed=(42, 'index'))
        randoms = generate_uniform_particles(pattrs, size, seed=(84, 'index'))
        data = ParticleField(data.positions, attrs=mattrs, exchange=True)
        data = data.clone(weights=(1. + mesh.read(data.positions, resampler='cic', compensate=False, exchange=False)) * data.weights)
        randoms = ParticleField(randoms.positions, attrs=mattrs, exchange=True)
        fkp = FKPField(data, randoms)
        bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))
        norm, num_shotnoise = compute_fkp2_normalization(fkp, bin=bin), compute_fkp2_shotnoise(fkp, bin=bin)
        mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        del fkp
        spectrum = compute_mesh2_spectrum(mesh, bin=bin, los='firstpoint')
        return spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

    #os.environ["XLA_FLAGS"] = " --xla_force_host_platform_device_count=4"
    size = int(1e5)
    pattrs = MeshAttrs(meshsize=(128,) * 3, boxsize=1121., boxcenter=1500.)
    mattrs = MeshAttrs(meshsize=(64,) * 3, boxsize=2000., boxcenter=1500.)
    ref_fn = dirname / 'ref_spectrum2.h5'

    with create_sharding_mesh() as sharding_mesh:
        print(sharding_mesh)
        test = compute_spectrum()
        if False and jax.process_index() == 0:
            test.write(ref_fn)
        print(test.value())
        assert np.allclose(test.value(), read(ref_fn).value())


def test_pypower():

    def get_random_catalog(mattrs, size=int(1e6), seed=42):
        seed = jax.random.key(seed)
        particles = generate_uniform_particles(mattrs, size, seed=seed)
        def sample(key, shape):
            return jax.random.uniform(key, shape, dtype=mattrs.dtype)
        weights = create_sharded_random(sample, seed, shape=particles.size, out_specs=P(*mattrs.sharding_mesh.axis_names))
        return particles.clone(weights=weights)

    def _identity_fn(x):
        return x

    def allgather(array):
        from jaxpower.mesh import get_sharding_mesh
        from jax.sharding import PartitionSpec as P
        sharding_mesh = get_sharding_mesh()
        if sharding_mesh.axis_names:
            array = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, P()))(array)
            return array.addressable_data(0)
        return array


    def allgather_particles(particles):
        from typing import NamedTuple

        class Particles(NamedTuple):
            positions: jax.Array
            weights: jax.Array

        return Particles(allgather(particles.positions), allgather(particles.weights))

    mattrs = MeshAttrs(boxsize=1000., boxcenter=600., meshsize=64)
    pattrs = mattrs.clone(boxsize=800.)
    data = get_random_catalog(pattrs, seed=42)
    randoms = get_random_catalog(pattrs, seed=84)

    edges = {'step': 0.01}
    ells = (0, 2, 4)
    kw_paint = dict(resampler='tsc', interlacing=3)
    compensate = True
    from pypower import CatalogFFTPower, CatalogMesh, MeshFFTPower, normalization, unnormalized_shotnoise

    for los in ['firstpoint', 'x', 'y', 'z'][:1]:
        print(los)
        ref_data, ref_randoms = allgather_particles(data), allgather_particles(randoms)
        # Option #1: double compensation applied to A0 only
        #pk_py = CatalogFFTPower(data_positions1=ref_data.positions, data_weights1=ref_data.weights,
        #                        randoms_positions1=ref_randoms.positions, randoms_weights1=ref_randoms.weights,
        #                        boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter, nmesh=mattrs.meshsize,
        #                        los=los, edges=edges, ells=ells, **kw_paint, position_type='pos', dtype='c16', mpiroot=0).poles
        # Option #2: compensation applied to each mesh (what is done in jaxpower)
        cmesh = CatalogMesh(data_positions=ref_data.positions, data_weights=ref_data.weights,
                           randoms_positions=ref_randoms.positions, randoms_weights=ref_randoms.weights,
                           boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter, nmesh=mattrs.meshsize, **kw_paint,
                           position_type='pos', dtype='c16', mpiroot=0)
        wnorm = normalization(cmesh)
        shotnoise_nonorm = unnormalized_shotnoise(cmesh, cmesh)
        pk_py = MeshFFTPower(cmesh.to_mesh(compensate=compensate), ells=ells, los=los, edges=edges, boxcenter=mattrs.boxcenter, wnorm=wnorm, shotnoise_nonorm=shotnoise_nonorm).poles

        data = ParticleField(data.positions, data.weights, attrs=mattrs, exchange=True)  # or data.clone(exchange=True)
        randoms = ParticleField(randoms.positions, randoms.weights, attrs=mattrs, exchange=True)
        # Now data and randoms are exchanged given MeshAttrs attrs, we can proceed as normal
        fkp = FKPField(data, randoms, attrs=mattrs)
        bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        norm, num_shotnoise = compute_fkp2_normalization(fkp, bin=bin), compute_fkp2_shotnoise(fkp, bin=bin)
        mesh = fkp.paint(**kw_paint, compensate=compensate, out='real')
        del fkp
        pk_jax = compute_mesh2_spectrum(mesh, bin=bin, los=los)
        pk_jax = pk_jax.clone(norm=norm, num_shotnoise=num_shotnoise)
        assert np.allclose(pk_jax.get(ells=0).values('norm'), pk_py.wnorm)
        assert np.allclose(pk_jax.get(ells=0).values('shotnoise'), pk_py.shotnoise)
        for ell in ells:
            #diff = np.abs(pk_jax.get(ells=ell).value() - pk_py(ell=ell, complex=False))
            assert np.allclose(pk_jax.get(ells=ell).value(), pk_py(ell=ell, complex=False), equal_nan=True, atol=1e-5, rtol=1e-5)


def test_memory():
    from jaxpower.utils import estimate_memory

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    mattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=128)

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

    los = 'firstpoint'

    ells = [0, 2, 4, 6, 8]
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 4 * mattrs.kfun.min()}, ells=ells)
    compute = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
    estimate_memory(compute, mesh, los=los, bin=bin)

    bin = BinMesh2CorrelationPoles(mattrs, edges={'step': 4 * mattrs.cellsize.min()}, ells=ells)
    compute = jax.jit(compute_mesh2_correlation, static_argnames=['los'])
    estimate_memory(compute, mesh, los=los, bin=bin)


def test_ref():
    from jaxpower.utils import estimate_memory

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    with create_sharding_mesh() as sharding_mesh:
        mattrs = MeshAttrs(boxsize=1000., boxcenter=500., meshsize=128)
        mesh = generate_gaussian_mesh(mattrs, pkvec, seed=(42, 'index'), unitary_amplitude=True)
        size = 128 * 1024
        backend = 'jax'
        data = generate_uniform_particles(mattrs, size, seed=(32, 'index')).exchange(backend=backend)
        randoms = generate_uniform_particles(mattrs, 2 * size, seed=(64, 'index')).exchange(backend=backend)

        fkp = FKPField(data, randoms)
        kw = dict(resampler='cic', interlacing=3, compensate=True)
        mesh = fkp.paint(**kw)
    
        def run(bin, los):
            compute = jax.jit(compute_mesh2, static_argnames=['los'])
            estimate_memory(compute, mesh, los=los, bin=bin)
            return compute(mesh, los=los, bin=bin)
    
        ells = (0, 2, 4)
        ref = {'x': 21.734526818678155, 'firstpoint': 21.38052302043614, 'endpoint': 21.38052302043614}
        bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 4 * mattrs.kfun.min()}, ells=ells)
        result = {}
        for los in ['x', 'firstpoint', 'endpoint']:
            result[los] = run(bin, los).value().std()
            assert np.allclose(result[los], ref[los])

        ref = {('x', None): 5.7245424391892524e-05, ('firstpoint', None): 5.5868671570865e-05, ('endpoint', None): 5.5868671570865e-05,
        ('x', 'bessel'): 5.7245424391892524e-05, ('firstpoint', 'bessel'): 5.5868671570865e-05, ('endpoint', 'bessel'): 5.5868671570865e-05}
        result = {}
        for basis in [None, 'bessel']:
            bin = BinMesh2CorrelationPoles(mattrs, edges={'step': 4 * mattrs.cellsize.min()}, ells=ells)
            for los in ['x', 'firstpoint', 'endpoint']:
                result[los, basis] = run(bin, los).value().std()
                assert np.allclose(result[los, basis], ref[los, basis])


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)
    config.update('jax_num_cpu_devices', 16)
    config.update('jax_platform_name', 'cpu')

    jax.distributed.initialize()
    test_ref()
    exit()

    #os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
    #jax.distributed.initialize()
    #test_sharded_spectrum()
    #jax.distributed.shutdown()

    test_mesh2_spectrum(plot=False)
    test_fkp2_shotnoise()
    test_mesh2_correlation(plot=False)
    test_fkp2_spectrum(plot=False)
    test_mesh2_spectrum_mean(plot=False)
    test_window_box(plot=False)
    test_window(plot=False)
    test_smooth_window(plot=False)
    #test_sharded_spectrum()
    test_pypower()
