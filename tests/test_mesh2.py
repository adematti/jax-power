import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (BinMesh2Spectrum, compute_mesh2_spectrum, Spectrum2Poles,
                      BinMesh2Correlation, compute_mesh2_correlation, Correlation2Poles,
                      generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, ParticleField, FKPField,
                      BinnedStatistic, WindowMatrix, MeshAttrs, compute_mesh2_spectrum_mean, compute_mesh2_spectrum_window, compute_fkp2_spectrum_normalization, utils)


dirname = Path('_tests')


def test_binned_statistic():

    x = [np.linspace(0.01, 0.2, 10), np.linspace(0.01, 0.2, 10)]
    v = x
    observable = BinnedStatistic(x=x, value=v, projs=[0, 2])
    assert np.allclose(observable.view(projs=2), v[0])
    assert np.allclose(observable.view(xlim=(0., 0.1), projs=0), v[0][v[0] <= 0.1])
    assert np.allclose(observable.slice(slice(0, None, 2)).view(projs=0), (v[0][::2] + v[0][1::2]) / 2.)
    fn = dirname / 'tmp.npy'
    observable.save(fn)
    observable2 = BinnedStatistic.load(fn)
    assert np.allclose(observable2.view(), observable.view())

    xt = [np.linspace(0.01, 0.2, 20), np.linspace(0.01, 0.2, 20)]
    vt = xt
    theory = BinnedStatistic(x=xt, value=vt, projs=[0, 2])
    wmat = WindowMatrix(observable=observable, theory=theory, value=np.ones((observable.size, theory.size)))
    assert np.allclose(wmat.select(xlim=(0., 0.15), axis='o').observable.x(projs=0), v[0][v[0] <= 0.15])
    tmp = wmat.dot(theory.view(), zpt=False)
    tmp2 = wmat.slice(slice(0, None, 2), axis='t').dot(theory.slice(slice(0, None, 2)).view(), zpt=False)
    assert np.allclose(tmp.sum(), tmp2.sum())
    wmat.slice(slice(0, None, 2), axis='o')
    wmat.slice(slice(0, None, 2), axis='o')
    assert np.allclose(wmat.select(axis='t', projs=0, select_projs=True).theory.view(), vt[0])
    wmat.plot()

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def mock(seed, los='x'):
        mesh = generate_gaussian_mesh(MeshAttrs(boxsize=500., meshsize=64), pkvec, seed=seed, unitary_amplitude=True)
        bin = BinMesh2Spectrum(mesh.attrs, edges={'step': 0.01}, ells=(0, 2, 4))
        return compute_mesh2_spectrum(mesh, los=los, bin=bin)

    power = mock(random.key(43), los='x')
    power.save(fn)
    power2 = BinnedStatistic.load(fn)
    assert np.allclose(power2.view(), power.view())
    assert type(power2) == Spectrum2Poles
    power2.plot(show=False)
    assert power.clone(norm=0.1).norm == 0.1
    power3 = power.clone(value=power.view())
    assert np.allclose(power3.view(), power.view())
    assert type(power3) == BinnedStatistic
    power1 = power.select(xlim=(0., 0.05))
    power2 = power.select(xlim=(0.05, 0.5))
    powerc = power.concatenate((power1, power2))
    assert np.allclose(powerc.view(), power.view())
    powers = power.sum([power] * 5)
    assert np.allclose(powers.view(), power.view())


def test_mesh2_spectrum(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'endpoint']
    attrs = MeshAttrs(meshsize=128, boxsize=1000.)
    bin = BinMesh2Spectrum(attrs, edges={'step': 0.01}, ells=(0, 2, 4))

    @partial(jax.jit, static_argnames=['los'])
    def mock(attrs, bin, seed, los='x'):
        mesh = generate_gaussian_mesh(attrs, pkvec, seed=seed, unitary_amplitude=True)
        return compute_mesh2_spectrum(mesh, los=los, bin=bin)

    for los in list_los:

        nmock = 5
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
        # remove first few bins because of binning effects
        assert np.allclose(power.view(projs=0)[2:], pk(power.x(projs=0))[2:], rtol=1e-2)
        assert tuple(power.projs) == (0, 2, 4)

    if plot:
        from matplotlib import pyplot as plt

        for los in list_los:
            power = Spectrum2Poles.load(get_fn(los=los))
            ax = power.plot().axes[0]
            k = power.x(projs=0)
            ax.plot(k, k * pk(k))
            ax.set_title(los)
            plt.show()


def test_mesh2_correlation(plot=False):

    from cosmoprimo.fiducial import DESI

    attrs = MeshAttrs(boxsize=500., boxcenter=[1300., 0., 0.], meshsize=32)

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    ellsin = (0, 2, 4)
    edgesin = np.arange(0., jnp.sqrt(3.) * attrs.knyq.max() + 0.2, 0.001)
    edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])
    kin = (edgesin[..., 0] + edgesin[..., 1]) / 2.
    f, b = 0.8, 1.5
    beta = f / b

    def get_pk(k, pk):
        pk = pk(k)
        return np.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk,
                          0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk,
                          8. / 35 * beta ** 2 * pk])

    poles = BinnedStatistic(x=[kin] * len(ellsin), edges=[edgesin] * len(ellsin), value=list(get_pk(kin, pk)), projs=ellsin)

    def pkvec(kvec):
        from jaxpower.utils import get_legendre
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        vlos = (1, 0, 0)
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / knorm
        p = jnp.sum(get_pk(knorm, pk) * jnp.stack([get_legendre(ell)(mu) for ell in ellsin]), axis=0)
        return jnp.where(knorm == 0., 0., p)

    def get_xi(s, pk):
        return np.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk.to_xi(fftlog_kwargs={'ell': 0})(s),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk.to_xi(fftlog_kwargs={'ell': 2})(s),
                        8. / 35 * beta ** 2 * pk.to_xi(fftlog_kwargs={'ell': 4})(s)])

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'local'][:1]
    attrs = MeshAttrs(meshsize=128, boxsize=1000.)
    bin = BinMesh2Correlation(attrs, edges={'step': 4, 'max': 200}, ells=(0, 2, 4))

    #@partial(jax.jit, static_argnames=['los'])
    def mock(attrs, bin, seed, los='x'):
        mesh = generate_anisotropic_gaussian_mesh(attrs, poles, los=los, seed=seed, unitary_amplitude=True, order=1)
        #mesh = generate_gaussian_mesh(attrs, pkvec, seed=seed, unitary_amplitude=True)
        return compute_mesh2_correlation(mesh, los=los, bin=bin)

    for los in list_los:
        corr = mock(attrs, bin, random.key(43), los=los)
        corr.save(get_fn(los))

    if plot:
        from matplotlib import pyplot as plt

        for los in list_los:
            corr = Correlation2Poles.load(get_fn(los=los))
            corr = corr.select(xlim=(30., 140.))
            ax = corr.plot().axes[0]
            s = corr.x(projs=0)
            xi = get_xi(s, pk)
            for ill, ell in enumerate(corr.projs):
                ax.plot(s, s**2 * xi[ill], color='C{:d}'.format(ill), linestyle='--')
            ax.set_title(los)
            plt.show()


def test_fkp2_spectrum(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'endpoint']
    boxcenter = [1300., 0., 0.]
    attrs = MeshAttrs(boxsize=1000., boxcenter=boxcenter, meshsize=128)
    bin = BinMesh2Spectrum(attrs, edges={'step': 0.01}, ells=(0, 2, 4))

    for los in list_los:
        @partial(jax.jit, static_argnames=['los'])
        def mock(seed, los='x'):
            mesh = generate_gaussian_mesh(attrs, pkvec, seed=seed, unitary_amplitude=True)
            size = int(1e5)
            data = generate_uniform_particles(attrs, size, seed=32)
            data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
            randoms = generate_uniform_particles(attrs, size, seed=42)
            fkp = FKPField(data, randoms)
            mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='complex')
            return compute_mesh2_spectrum(mesh, bin=bin, los=los)

        nmock = 5
        t0 = time.time()
        power = mock(random.key(43), los=los)
        jax.block_until_ready(power)
        print(f'time for jit {time.time() - t0:.2f}')
        t0 = time.time()
        for i in range(nmock):
            jax.block_until_ready(mock(random.key(i + 42), los=los))
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
        power.save(get_fn(los))

    if plot:
        from matplotlib import pyplot as plt

        for los in list_los:
            power = Spectrum2Poles.load(get_fn(los=los))
            ax = power.plot().axes[0]
            k = power.x(projs=0)
            ax.plot(k, k * pk(k))
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
    attrs = MeshAttrs(boxsize=1000., meshsize=64, boxcenter=1000.)
    bin = BinMesh2Spectrum(attrs, edges={'step': 0.01}, ells=(0, 2, 4))

    @partial(jax.jit, static_argnames=['los', 'thlos'])
    def mean(los='x', thlos=None):
        theory = poles
        if thlos is not None:
            theory = (poles, thlos)
        return compute_mesh2_spectrum_mean(attrs, theory=theory, los=los, bin=bin)

    for los, thlos in list_los:

        #power_mock = mock(random.key(43), los=los)
        t0 = time.time()
        power_mean = mean(los=los, thlos=thlos)
        print(f'time for jit {time.time() - t0:.2f}')
        nmock = 2
        t0 = time.time()
        for i in range(nmock):
            jax.block_until_ready(mean(los=los, thlos=thlos))
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')

        if plot:
            from matplotlib import pyplot as plt
            ax = power_mean.plot().axes[0]
            for ell in power_mean.projs:
                k = power_mean.x(projs=ell)
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

    attrs = MeshAttrs(boxsize=2000., meshsize=350, boxcenter=1000.)
    bin = BinMesh2Spectrum(attrs, edges={'step': 0.01}, ells=ells)

    def make_callable(poles):
        def get_fun(ill):
            return lambda k: 1. * k * np.mean(poles[ill]) #jnp.interp(k, kin, poles[ill], left=0., right=0.)
        return {ell: get_fun(ill) for ill, ell in enumerate(ells)}

    def make_callable(poles):
        knorm = jnp.sqrt(sum(kk**2 for kk in attrs.kcoords(sparse=True, hermitian=True))).ravel()
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
        return compute_mesh2_spectrum_mean(attrs, theory=theory, bin=bin, los=los)

    #print(jax.grad(lambda poles: mean(poles).view()[2])(poles))
    from jax.ad_checkpoint import checkpoint_name

    def mock(poles, los='local', seed=42):
        mesh = generate_anisotropic_gaussian_mesh(attrs, make_callable(poles), los=los, seed=seed)
        #mesh = jax.checkpoint(lambda poles: generate_anisotropic_gaussian_mesh(attrs, make_callable(poles), los='x', seed=seed))(poles)
        return compute_mesh2_spectrum(mesh, bin=bin, los={'local': 'firstpoint'}.get(los, los)).view()[-3]

    def power(mesh, los='local'):
        return compute_mesh2_spectrum(mesh, bin=bin, los={'local': 'firstpoint'}.get(los, los)).view()[-3]

    def gaussian_survey(attrs, size=int(1e6), seed=random.key(42), scale=0.03, paint=False):
        # Generate Gaussian-distributed positions
        positions = scale * attrs.boxsize * random.normal(seed, shape=(size, 3))
        toret = ParticleField(positions + attrs.boxcenter, attrs=attrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(attrs, paint=True)

    #@partial(jax.checkpoint, static_argnums=(2,))
    policy = jax.checkpoint_policies.save_only_these_names('save')

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
        mesh = generate_anisotropic_gaussian_mesh(selection.attrs, (kinedges, list(theory)), los=los, seed=seed,
                                                  unitary_amplitude=unitary_amplitude)
        los = {'local': 'firstpoint'}.get(los, los)
        #return compute_mesh2_spectrum(apply_selection(mesh, selection, False), bin=bin, los=los).view()[-3]
        toret = [compute_mesh2_spectrum(apply_selection(mesh, selection, cv=cv), bin=bin, los=los) for cv in [False, True]]
        return toret[0].clone(value=toret[0].view() - toret[1].view()).view()[-3]

    def mock_vmap(poles, los='local'):
        seeds = random.split(random.key(42), 4)
        def func(seed):
            mesh = generate_anisotropic_gaussian_mesh(attrs, make_callable(poles), los=los, seed=seed)
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
    #func = jax.checkpoint(func, policy=jax.checkpoint_policies.save_only_these_names('mock'))
    #func = jax.checkpoint(func, policy=jax.checkpoint_policies.save_anything_except_these_names('tmp1', 'tmp2'))
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
    from jaxpower.utils import compute_real_gaunt, get_real_Ylm

    if False:
        for ell1, ell2, ell3 in itertools.product((0, 2, 4), (0, 2), (0, 2)):
            ms = list(itertools.product(list(range(-ell1, ell1 + 1)),
                                        list(range(-ell2, ell2 + 1)),
                                        list(range(-ell3, ell3 + 1))))
            for m1, m2, m3 in ms:
                g = compute_real_gaunt((ell1, m1), (ell2, m2), (ell3, m3))
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
                tmp = get_real_Ylm(ell, m, modules=('scipy',), batch=False)(*xyz.T)
                tmp2 = get_real_Ylm(ell, m, batch=False)(*xyz.T)
                assert np.allclose(tmp2, tmp, rtol=1e-6, atol=1e-6)

    if False:
        for ell1, ell2, ell3 in itertools.product((0, 2, 4), (0, 2), (0, 2)):
            ms = list(itertools.product(list(range(-ell1, ell1 + 1)),
                                        list(range(-ell2, ell2 + 1)),
                                        list(range(-ell3, ell3 + 1))))
            if any(compute_real_gaunt((ell1, m1), (ell2, m2), (ell3, m3)) for m1, m2, m3 in ms):

                for m1, m2, m3 in ms:
                    mattrs = MeshAttrs(boxsize=500, meshsize=64)
                    mesh = mattrs.create(kind='hermitian_complex')
                    kvec = mesh.coords(sparse=True)
                    knorm = sum(kk**2 for kk in kvec)**0.5
                    kedges = jnp.array([0.8 * mesh.knyq.min(), 0.9 * mesh.knyq.min()])
                    kmask = (knorm >= kedges[0]) & (knorm <= kedges[-1])
                    bin = BinMesh2Spectrum(mesh, edges=kedges)
                    mesh = mesh.clone(value=kmask * get_real_Ylm(ell1, m1)(*kvec) * get_real_Ylm(ell2, m2)(*kvec) * get_real_Ylm(ell3, m3)(*kvec))
                    value = bin(mesh)[0]
                    g = float(compute_real_gaunt((ell1, m1), (ell2, m2), (ell3, m3))) / (4 * np.pi)
                    if not np.allclose(value, g, rtol=1e-2, atol=1e-3):
                        print(ell1, ell2, ell3, m1, m2, m3, value, g)

    if True:
        ell = 2
        for m in range(-ell, ell + 1):
            print(ell, m, get_real_Ylm(ell, m)(0, 0, 1))


def test_window_box(plot=False):

    from jaxpower.mesh2 import compute_normalization

    def get_theory(kmax=0.3, dk=0.005):
        # Return theory power spectrum
        from cosmoprimo.fiducial import DESI
        cosmo = DESI(engine='eisenstein_hu')
        z = 1.
        pk1d = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
        ellsin = (0, 2, 4)
        edgesin = jnp.arange(0., kmax, dk)
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
        kin = (edgesin[..., 0] + edgesin[..., 1]) / 2.
        f, b = cosmo.growth_rate(z), 1.5
        beta = f / b
        shotnoise = (1e-3)**(-1)
        pk = pk1d(kin)
        poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk + shotnoise,
                            (4. / 3. * beta + 4. / 7. * beta ** 2) * pk,
                            8. / 35 * beta ** 2 * pk])
        return Spectrum2Poles(k=[kin] * len(ellsin), edges=[edgesin] * len(ellsin), num=list(poles), num_shotnoise=shotnoise, ells=ellsin)

    mattrs = MeshAttrs(boxsize=1000., meshsize=64)
    bin = BinMesh2Spectrum(mattrs, edges={'min': 0, 'step': 0.01}, ells=(0, 2, 4))
    seed = random.key(42)
    los = 'z'
    theory = get_theory(kmax=mattrs.knyq.max())
    # Setting unitary_amplitude = True to reduce noise (actually, there is no noise at all, so no need for multiple realizations)
    mesh = generate_anisotropic_gaussian_mesh(mattrs, poles=theory, los=los, seed=seed, unitary_amplitude=True)
    mean = compute_mesh2_spectrum(mesh, bin=bin, los=los)

    # edges and ells for input theory
    edgesin = theory.edges(projs=0)
    ellsin = (0, 2, 4)
    # bin is still the binning operator
    wmat = compute_mesh2_spectrum_window(mattrs, edgesin=edgesin, ellsin=ellsin, bin=bin, los=los, pbar=True)
    test = wmat.dot(theory, return_type=None)
    wmat_rebin = wmat.slice(slice(0, None, 2), axis='t').slice(slice(0, -1), axis='t')
    test_rebin = wmat_rebin.dot(theory.slice(slice(0, None, 2)).slice(slice(0, -1)), return_type=None)
    #test_rebin = wmat_rebin.dot(get_theory(kmax=wmat_rebin.theory.edges()[0].max() + 1e-4, dk=0.01), return_type=None)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for ill, ell in enumerate(test.projs):
            color = 'C{:d}'.format(ill)
            ax.plot(mean.x(ell), mean.x(ell) * mean.view(projs=ell), color=color, linestyle='-')
            ax.plot(test.x(ell), test.x(ell) * test.view(projs=ell), color=color, linestyle='--')
            ax.plot(test_rebin.x(ell), test_rebin.x(ell) * test_rebin.view(projs=ell), color=color, linestyle=':')
        plt.show()


def test_window(plot=False):

    from jaxpower.mesh2 import compute_normalization

    ellsin = (0, 2, 4)
    edgesin = np.array([0.1, 0.11])
    xin = (edgesin[:-1] + edgesin[1:]) / 2.
    theory = BinnedStatistic(x=[xin] * len(ellsin), edges=[np.array(list(zip(edgesin[:-1], edgesin[1:])))] * len(ellsin), value=[np.ones_like(xin)] + [np.ones_like(xin)] * (len(ellsin) - 1), projs=ellsin)

    attrs = MeshAttrs(boxsize=500., meshsize=64, boxcenter=1000.)
    bin = BinMesh2Spectrum(attrs, edges={'step': 0.01}, ells=(0, 2, 4))

    def gaussian_survey(attrs, size=int(1e6), seed=random.key(42), scale=0.2, paint=False):
        # Generate Gaussian-distributed positions
        positions = jnp.array([1., 0.2, 0.2]) * scale * random.normal(seed, shape=(size, 3))
        bscale = scale
        mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
        positions = positions * attrs.boxsize + attrs.boxcenter
        toret = ParticleField(positions, weights=1. * mask, attrs=attrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(attrs, paint=True)
    norm = compute_normalization(selection, selection)

    for flag in ['smooth', 'infinite']:
        for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')]:
            print(flag, los, thlos)
            mean = compute_mesh2_spectrum_mean(selection, theory=(theory, thlos) if thlos is not None else theory, bin=bin, los=los).clone(norm=norm)
            wmatrix = compute_mesh2_spectrum_window(selection, edgesin=edgesin, ellsin=(ellsin, thlos) if thlos is not None else ellsin, bin=bin, los=los, pbar=True, norm=norm, flags=(flag,))
            if plot:
                from matplotlib import pyplot as plt
                ax = plt.gca()
                for iproj, proj in enumerate(mean.projs):
                    color = 'C{:d}'.format(iproj)
                    ax.plot(mean.x(projs=proj), mean.view(projs=proj).real, color=color, linestyle='-')
                    projin = (0, 0) if thlos == 'firstpoint' else 0
                    ax.plot(mean.x(projs=proj), wmatrix.select(projs=proj, select_projs=True, axis='o').select(projs=projin, select_projs=True, axis='t').view().real, color=color, linestyle='--')
                plt.show()


def test_window_timing():

    from jaxpower import compute_mesh2_spectrum_window
    from jaxpower.utils import Interpolator1D

    ellsin = (0, 2, 4)
    edgesin = np.linspace(0.01, 0.1, 20)
    xin = (edgesin[:-1] + edgesin[1:]) / 2.
    theory = BinnedStatistic(x=[xin] * len(ellsin), edges=[np.array(list(zip(edgesin[:-1], edgesin[1:])))] * len(ellsin), value=[np.ones_like(xin)] + [np.ones_like(xin)] * (len(ellsin) - 1), projs=ellsin)

    attrs = MeshAttrs(boxsize=500., meshsize=64, boxcenter=1000.)
    bin = BinMesh2Spectrum(attrs, edges={'step': 0.01}, ells=(0, 2, 4))

    def gaussian_survey(attrs, size=int(1e6), seed=random.key(42), scale=0.2, paint=False):
        # Generate Gaussian-distributed positions
        positions = jnp.array([1., 0.2, 0.2]) * scale * random.normal(seed, shape=(size, 3))
        bscale = scale
        mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
        positions = positions * attrs.boxsize + attrs.boxcenter
        toret = ParticleField(positions, weights=1. * mask, attrs=attrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(attrs, paint=True)
    norm = compute_fkp2_spectrum_normalization(selection, selection)

    for flag in ['smooth', 'infinite'][-1:]:
        for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')][-1:]:
            print(flag, los, thlos)
            t0 = time.time()
            wmatrix = compute_mesh2_spectrum_window(selection, edgesin=edgesin, ellsin=(ellsin, thlos) if thlos is not None else ellsin, bin=bin, los=los, pbar=True, norm=norm, flags=(flag,))
            print(f'{time.time() - t0:.2f}')


def test_smooth_window():
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    attrs = MeshAttrs(boxsize=2000., meshsize=100, boxcenter=800.)
    ells = (0, 2, 4)
    bin = BinMesh2Spectrum(attrs, edges={'step': 0.005}, ells=ells)
    #edgesin = np.arange(0., 1.1 * attrs.knyq.max(), 0.002)
    edgesin = np.arange(0., attrs.knyq.max(), 0.002)
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

    def gaussian_survey(attrs, size=int(1e6), seed=random.key(42), scale=0.1, paint=False):
        # Generate Gaussian-distributed positions
        positions = scale * random.normal(seed, shape=(size, 3))
        bscale = 2. * scale  # cut at 2 sigmas
        mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
        positions = positions * attrs.boxsize + attrs.boxcenter
        toret = ParticleField(positions, weights=1. * mask, attrs=attrs)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    if False:
        selection = gaussian_survey(attrs, size=int(1e7), paint=True)
        norm = compute_fkp2_spectrum_normalization(selection, selection)
        selection = selection.r2c()
        selection = (selection * selection.conj()).c2r()
        edges = np.arange(0., selection.boxsize.max(), selection.cellsize.min())
        bin = BinMesh2Spectrum(selection, edges=edges)
        pole = bin(selection) / norm / selection.cellsize.prod()
        print(pole)

        from matplotlib import pyplot as plt
        ax = plt.gca()
        ax.plot(bin.xavg, pole)
        ax.set_xscale('log')
        plt.show()

    selection = gaussian_survey(attrs, paint=True)
    norm = compute_fkp2_spectrum_normalization(selection, selection)

    for thlos in ['firstpoint', 'local'][:1]:
        mean = compute_mesh2_spectrum_mean(selection, theory=(edgesin, list(poles), thlos),
                                           bin=bin, los='firstpoint').clone(norm=norm)
        wmatrix = compute_mesh2_spectrum_window(selection, edgesin=edgesin, ellsin=(ellsin, thlos),
                                            bin=bin, los='firstpoint', norm=norm, flags=('smooth',), pbar=True)
        wpoles = wmatrix.dot(poles, return_type=None)
        ax = plt.gca()
        for iproj, proj in enumerate(mean.projs):
            ax.plot(kin, kin * poles[iproj], color='k')
            k = mean.x(projs=proj)
            ax.plot(k, k * mean.view(projs=proj), color='C0')
            ax.plot(k, k * wpoles.view(projs=proj), color='C1')
        plt.show()



def test_wmatrix(plot=False):
    xo = np.linspace(0., 0.2, 21)
    xt = np.linspace(0., 0.2, 41)
    ellsin = (0, 2)
    ells = (0, 2)
    theory = BinnedStatistic(x=[xt] * len(ellsin), projs=ellsin)
    observable = BinnedStatistic(x=[xo] * len(ells), projs=ells)
    theory2 = BinnedStatistic(x=[np.linspace(0., 0.3, 61)] * len(ellsin), projs=ellsin)
    observable2 = BinnedStatistic(x=[np.linspace(0., 0.3, 61)] * len(ells), projs=ells)

    def f(xo, xt):
        sigma = 0.02
        delta = (xo - xt) / sigma
        return np.exp(-delta**2)

    value = np.block([[f(*np.meshgrid(xo, xt, indexing='ij')) for xt in theory.x()] for xo in observable.x()])
    wmatrix = WindowMatrix(observable=observable, theory=theory, value=value)
    if plot: wmatrix.plot(show=True)

    wmatrix1 = wmatrix.slice(slice(0, -1), axis='o')
    assert wmatrix1.shape[0] == wmatrix.shape[0] - 1 * len(ellsin)
    wmatrix1 = wmatrix1.select(axis='o', xlim=(0., 0.081))
    wmatrix2 = wmatrix.select(axis='o', xlim=(0.081, 0.5))
    wmatrixc = wmatrix.concatenate([wmatrix1, wmatrix2], axis='o')

    assert np.allclose(wmatrixc._value, wmatrix._value)
    wmatrixs = wmatrix.sum([wmatrix] * 3)
    assert np.allclose(wmatrixs._value, wmatrix._value * 3)

    wmatrix2 = wmatrix.interp(theory2, axis='t', extrap=True)
    wmatrix3 = wmatrix.interp(observable2, axis='o', extrap=True)

    fn = dirname / 'tmp.npy'
    wmatrix3.save(fn)
    wmatrix3 = WindowMatrix.load(fn)

    if plot:
        wmatrix2.plot(show=True)
        wmatrix3.plot(show=True)
        wmatrix3.plot_slice(indices=5, show=True)


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


def test_mem():

    from jaxpower import MeshAttrs, create_sharded_random, create_sharding_mesh
    with create_sharding_mesh() as sharding_mesh:
        attrs = MeshAttrs(meshsize=1600, boxsize=1000.)
        #attrs = MeshAttrs(meshsize=850, boxsize=1000.)
        bin = BinMesh2Spectrum(attrs, edges={'step': 0.001}, ells=(0, 2, 4))
        mesh = attrs.create(kind='real', fill=create_sharded_random(jax.random.normal, jax.random.key(42), shape=attrs.meshsize))
        compute = partial(compute_mesh2_spectrum, bin=bin, los='firstpoint')
        compute = jax.jit(compute)
        t0 = time.time()
        mesh_power = compute(mesh)
        jax.block_until_ready(mesh_power)
        print(time.time() - t0)


def test_split():
    from pathlib import Path

    import numpy as np
    from jax import numpy as jnp
    from jax import random
    from matplotlib import pyplot as plt

    from cosmoprimo.fiducial import DESI
    from jaxpower import (compute_mesh2_spectrum, Spectrum2Poles, generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, RealMeshField, ParticleField, FKPField,
                          BinnedStatistic, WindowMatrix, MeshAttrs, BinMesh2Spectrum, compute_mesh2_spectrum_mean, compute_mesh2_spectrum_window, compute_fkp2_spectrum_normalization, utils, create_sharding_mesh, make_particles_from_local, create_sharded_array, create_sharded_random)

    attrs = MeshAttrs(meshsize=(128,) * 3, boxsize=1000., boxcenter=1200.)
    size = int(1e-4 * attrs.boxsize.prod())
    data = generate_uniform_particles(attrs, size + 1, seed=42)
    randoms = generate_uniform_particles(attrs, 4 * size + 1, seed=43)

    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)

    f, b = 0.8, 1.5
    beta = f / b
    kinedges = np.linspace(0.001, 0.7, 100)
    kin = (kinedges[:-1] + kinedges[1:]) / 2.
    ells = (0, 2, 4)
    poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])
    theory = BinnedStatistic(x=[kin] * len(ells), edges=[np.array(list(zip(kinedges[:-1], kinedges[1:])))] * len(ells), value=poles, projs=ells)
    mesh = generate_anisotropic_gaussian_mesh(attrs, theory, seed=random.key(42), los='local', unitary_amplitude=True)
    data = data.clone(weights=1. + mesh.read(data.positions))

    fkp = FKPField(data, randoms, attrs=attrs.clone(boxsize=2. * attrs.boxsize))  # x2 padding
    from jaxpower.mesh import _paint
    for split_fkp in fkp.split(nsplits=2):
        t0 = time.time()
        mesh = split_fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        jax.block_until_ready(mesh)
        print(f'cache {_paint._cache_size()} {time.time() - t0:.2f}')


if __name__ == '__main__':

    #import warnings
    #warnings.simplefilter("error")
    #test_window_timing()
    #test_sympy()
    #test_window()
    #test_wmatrix()
    #test_gaunt()
    #test_smooth_window()
    #test_checkpoint()
    #test_gaunt()
    test_window_box(plot=False)
    test_binned_statistic()
    test_wmatrix()
    test_mesh2_spectrum(plot=False)
    #test_fkp2_spectrum(plot=False)
    test_mesh2_spectrum_mean(plot=False)
    test_mesh2_correlation(plot=False)
    test_window()
    #test_split()