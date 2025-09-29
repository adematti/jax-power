import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                      BinMesh2CorrelationPoles, compute_mesh2_correlation,
                      generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, ParticleField, FKPField,
                      MeshAttrs, compute_mesh2_spectrum_mean, compute_mesh2_spectrum_window, compute_smooth2_spectrum_window,
                      compute_fkp2_normalization, compute_fkp2_shotnoise, compute_normalization,
                      Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles, WindowMatrix, read, utils)


dirname = Path('_tests')


def test_mesh2_spectrum(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.h5'.format(los)

    list_los = ['x', 'endpoint']
    mattrs = MeshAttrs(meshsize=128, boxsize=1000.)
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
    bin = BinMesh2CorrelationPoles(mattrs, edges={'step': 4, 'max': 200}, ells=(0, 2, 4))

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
                    bin = BinMesh2SpectrumPoles(mesh, edges=kedges)
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
    norm = compute_normalization(selection, selection, bin=bin)

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
        mattrs = MeshAttrs(meshsize=1600, boxsize=1000.)
        #mattrs = MeshAttrs(meshsize=850, boxsize=1000.)
        bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=(0, 2, 4))
        mesh = mattrs.create(kind='real', fill=create_sharded_random(jax.random.normal, jax.random.key(42), shape=mattrs.meshsize))
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
    from jaxpower import (compute_mesh2_spectrum, Mesh2SpectrumPoles, generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, RealMeshField, ParticleField, FKPField,
                          WindowMatrix, MeshAttrs, BinMesh2SpectrumPoles, compute_mesh2_spectrum_mean, compute_mesh2_spectrum_window, compute_fkp2_normalization, utils, create_sharding_mesh, make_particles_from_local, create_sharded_array, create_sharded_random)

    mattrs = MeshAttrs(meshsize=(128,) * 3, boxsize=1000., boxcenter=1200.)
    size = int(1e-4 * mattrs.boxsize.prod())
    data = generate_uniform_particles(mattrs, size + 1, seed=42)
    randoms = generate_uniform_particles(mattrs, 4 * size + 1, seed=43)

    cosmo = DESI()
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)

    f, b = 0.8, 1.5
    beta = f / b
    edgesin = np.linspace(0.001, 0.7, 100)
    edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])
    kin = np.mean(edgesin, axis=-1)
    ellsin = (0, 2, 4)
    theory = [(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk,
                        (4. / 3. * beta + 4. / 7. * beta ** 2) * pk,
                        8. / 35 * beta ** 2 * pk]

    poles = []
    for ell, value in zip(ellsin, theory):
        poles.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=value, ell=ell))
    theory = Mesh2SpectrumPoles(poles)

    mesh = generate_anisotropic_gaussian_mesh(mattrs, theory, seed=random.key(42), los='local', unitary_amplitude=True)
    data = data.clone(weights=1. + mesh.read(data.positions))

    fkp = FKPField(data, randoms, attrs=mattrs.clone(boxsize=2. * mattrs.boxsize))  # x2 padding
    from jaxpower.mesh import _paint
    for split_fkp in fkp.split(nsplits=2):
        t0 = time.time()
        mesh = split_fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        jax.block_until_ready(mesh)
        print(f'cache {_paint._cache_size()} {time.time() - t0:.2f}')


def test_sharding():

    # Initialize JAX distributed environment
    #jax.distributed.initialize()

    # Let's simulate distributed calculation
        # Let's create some mesh mock!
    import jax
    from jax import numpy as jnp
    from jaxpower import MeshAttrs, Mesh2SpectrumPoles, generate_anisotropic_gaussian_mesh

    meshsize = 64


    def get_theory(kmax=0.3, dk=0.001):
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

    attrs = MeshAttrs(boxsize=1000., meshsize=meshsize)
    poles = get_theory(4 * attrs.knyq.max())

    from jaxpower import create_sharding_mesh, exchange_particles, ParticleField, compute_fkp2_shotnoise, compute_mesh2, get_mesh_attrs

    with create_sharding_mesh() as sharding_mesh:  # specify how to spatially distribute particles / mesh
        print('Sharding mesh {}.'.format(sharding_mesh))
        # Generate a Gaussian mock --- everything is already distributed
        attrs = MeshAttrs(boxsize=1000., meshsize=meshsize, boxcenter=700.)
        pattrs = attrs.clone(boxsize=800.)
        size = int(1e6)
        data = generate_uniform_particles(pattrs, size, seed=42)
        mmesh = generate_anisotropic_gaussian_mesh(attrs, poles, los='local', seed=68, order=1, unitary_amplitude=True)
        pattrs = attrs.clone(boxsize=800.)
        size = int(1e6)
        data = generate_uniform_particles(pattrs, size, seed=42)
        data = data.clone(weights=1. + mmesh.read(data.positions, resampler='cic', compensate=True))
        randoms = generate_uniform_particles(pattrs, 2 * size, seed=43)
        # Now, pick MeshAttrs
        attrs = get_mesh_attrs(data.positions, randoms.positions, boxpad=2., meshsize=128)
        # Create ParticleField, exchanging particles
        data = ParticleField(data.positions, attrs=attrs, exchange=True)  # or data.clone(exchange=True)
        randoms = ParticleField(randoms.positions, attrs=attrs, exchange=True)
        # Now data and randoms are exchanged given MeshAttrs attrs, we can proceed as normal
        fkp = FKPField(data, randoms, attrs=attrs)
        norm, num_shotnoise = compute_fkp2_normalization(fkp), compute_fkp2_shotnoise(fkp)
        mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        del fkp
        bin = BinMesh2SpectrumPoles(attrs, edges={'step': 0.01}, ells=(0, 2, 4))
        pk = compute_mesh2(mesh, bin=bin, los='firstpoint')
        pk = pk.clone(norm=norm, num_shotnoise=num_shotnoise)

    # Close JAX distributed environment
    #jax.distributed.shutdown()


def test_pypower():

    def get_random_catalog(mattrs, size=int(1e6), seed=42):
        from jaxpower import create_sharded_random
        seed = jax.random.key(seed)
        particles = generate_uniform_particles(mattrs, size, seed=seed)
        def sample(key, shape):
            return jax.random.uniform(key, shape, dtype=mattrs.dtype)
        weights = create_sharded_random(sample, seed, shape=particles.size, out_specs=0)
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


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)

    test_mesh2_spectrum(plot=False)
    test_mesh2_correlation(plot=False)
    test_fkp2_spectrum(plot=False)
    test_mesh2_spectrum_mean(plot=False)
    test_window_box(plot=False)
    test_window(plot=False)
    test_smooth_window(plot=False)
    test_pypower()
