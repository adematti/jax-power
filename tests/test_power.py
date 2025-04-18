import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (compute_mesh_power, PowerSpectrumMultipoles, generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, ParticleField, FKPField,
                      compute_fkp_power, BinnedStatistic, WindowMatrix, MeshAttrs, BinAttrs, compute_mean_mesh_power, compute_mesh_window, compute_normalization, utils)


dirname = Path('_tests')


def test_binned_statistic():

    x = [np.linspace(0.01, 0.2, 10), np.linspace(0.01, 0.2, 10)]
    v = x
    observable = BinnedStatistic(x=x, value=v, projs=[0, 2])
    assert np.allclose(observable.view(projs=2), v[0])
    assert np.allclose(observable.view(xlim=(0., 0.1), projs=0), v[0][v[0] <= 0.1])
    assert np.allclose(observable.select(rebin=2).view(projs=0), (v[0][::2] + v[0][1::2]) / 2.)
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
        mesh = generate_gaussian_mesh(MeshAttrs(boxsize=2000., meshsize=64), pkvec, seed=seed, unitary_amplitude=True)
        return compute_mesh_power(mesh, ells=(0, 2, 4), los=los, edges={'step': 0.01})

    power = mock(random.key(43), los='x')
    power.save(fn)
    power2 = BinnedStatistic.load(fn)
    assert np.allclose(power2.view(), power.view())
    assert type(power2) == PowerSpectrumMultipoles
    power2.plot(show=True)
    assert power.clone(norm=0.1).norm == 0.1
    power3 = power.clone(value=power.view())
    assert np.allclose(power3.view(), power.view())
    assert type(power3) == BinnedStatistic


def test_mesh_power(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'endpoint']

    @partial(jax.jit, static_argnames=['los', 'ells'])
    def mock(seed, los='x', ells=(0, 2, 4)):
        attrs = MeshAttrs(meshsize=128, boxsize=1000.)
        mesh = generate_gaussian_mesh(attrs, pkvec, seed=seed, unitary_amplitude=True)
        return compute_mesh_power(mesh, ells=ells, los=los, edges={'step': 0.01})

    for los in list_los:

        nmock = 5
        t0 = time.time()
        power = mock(random.key(43), los=los)
        print(f'time for jit {time.time() - t0:.2f}')
        t0 = time.time()
        for i in range(nmock):
            jax.block_until_ready(mock(random.key(i + 42), los=los))
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
        power.save(get_fn(los))
        # remove first few bins because of binning effects
        assert np.allclose(power.view(projs=0)[2:], pk(power.x(projs=0))[2:], rtol=1e-2)
        power = mock(random.key(i + 42), los=los, ells=(4,))
        assert tuple(power.projs) == (4,)

    if plot:
        from matplotlib import pyplot as plt

        for los in list_los:
            power = PowerSpectrumMultipoles.load(get_fn(los=los))
            ax = power.plot().axes[0]
            k = power.x(projs=0)
            ax.plot(k, k * pk(k))
            ax.set_title(los)
            plt.show()


def test_fkp_power(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'endpoint']

    for los in list_los:
        @partial(jax.jit, static_argnames=['los'])
        def mock(seed, los='x'):
            boxcenter = [1300., 0., 0.]
            attrs = MeshAttrs(boxsize=1000., boxcenter=boxcenter, meshsize=128)
            mesh = generate_gaussian_mesh(attrs, pkvec, seed=seed, unitary_amplitude=True)
            size = int(1e6)
            data = generate_uniform_particles(attrs, size, seed=32)
            data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
            randoms = generate_uniform_particles(attrs, size, seed=42)
            fkp = FKPField(data, randoms)
            return compute_fkp_power(fkp, ells=(0, 2, 4), los=los, edges={'step': 0.01})

        nmock = 5
        t0 = time.time()
        power = mock(random.key(43), los=los)
        print(f'time for jit {time.time() - t0:.2f}')
        t0 = time.time()
        for i in range(nmock):
            jax.block_until_ready(mock(random.key(i + 42), los=los))
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
        power.save(get_fn(los))

    if plot:
        from matplotlib import pyplot as plt

        for los in list_los:
            power = PowerSpectrumMultipoles.load(get_fn(los=los))
            ax = power.plot().axes[0]
            k = power.x(projs=0)
            ax.plot(k, k * pk(k))
            ax.set_title(los)
            plt.show()


def test_mean_power(plot=False):

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

    @partial(jax.jit, static_argnames=['los', 'thlos'])
    def mean(los='x', thlos=None):
        theory = poles
        if thlos is not None:
            theory = (poles, thlos)
        return compute_mean_mesh_power(attrs, theory=theory, ells=(0, 2, 4), los=los, edges={'step': 0.01})

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
    edges = {'step': 0.01}

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
        return compute_mean_mesh_power(attrs, theory=theory, ells=(0, 2, 4), edges=edges, los=los)

    #print(jax.grad(lambda poles: mean(poles).view()[2])(poles))
    from jax.ad_checkpoint import checkpoint_name

    def mock(poles, los='local', seed=42):
        mesh = generate_anisotropic_gaussian_mesh(attrs, make_callable(poles), los=los, seed=seed)
        #mesh = jax.checkpoint(lambda poles: generate_anisotropic_gaussian_mesh(attrs, make_callable(poles), los='x', seed=seed))(poles)
        return compute_mesh_power(mesh, ells=ells, edges=edges, los={'local': 'firstpoint'}.get(los, los)).view()[-3]

    def power(mesh, los='local'):
        return compute_mesh_power(mesh, ells=ells, edges=edges, los={'local': 'firstpoint'}.get(los, los)).view()[-3]

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
        #return compute_mesh_power(apply_selection(mesh, selection, False), edges=edges, los=los, ells=ells).view()[-3]
        toret = [compute_mesh_power(apply_selection(mesh, selection, cv=cv), edges=edges, los=los, ells=ells) for cv in [False, True]]
        return toret[0].clone(value=toret[0].view() - toret[1].view()).view()[-3]

    def mock_vmap(poles, los='local'):
        seeds = random.split(random.key(42), 4)
        def func(seed):
            mesh = generate_anisotropic_gaussian_mesh(attrs, make_callable(poles), los=los, seed=seed)
            #mesh = jax.checkpoint(lambda poles: generate_anisotropic_gaussian_mesh(attrs, make_callable(poles), los='x', seed=seed))(poles)
            return compute_mesh_power(mesh, ells=(0, 2, 4), edges=edges, los={'local': 'firstpoint'}.get(los, los)).view()
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
    from jaxpower.utils import compute_real_gaunt
    from jaxpower.power import get_real_Ylm

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
                bin = BinAttrs(mesh, edges=kedges)
                mesh = mesh.clone(value=kmask * get_real_Ylm(ell1, m1)(*kvec) * get_real_Ylm(ell2, m2)(*kvec) * get_real_Ylm(ell3, m3)(*kvec))
                value = bin(mesh)[0]
                g = float(compute_real_gaunt((ell1, m1), (ell2, m2), (ell3, m3))) / (4 * np.pi)
                if not np.allclose(value, g, rtol=1e-2, atol=1e-3):
                     print(ell1, ell2, ell3, m1, m2, m3, value, g)


def test_power_to_correlation():

    from matplotlib import pyplot as plt
    from jaxpower.utils import TophatPowerToCorrelation, BesselPowerToCorrelation, Interpolator1D
    from jaxpower import BinAttrs

    def pk(k):
        kp = 0.01
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    mattrs = MeshAttrs(boxsize=500, meshsize=256)
    kfun = mattrs.kfun[0]
    knyq = mattrs.knyq[0]

    kinedges = np.linspace(0., np.sqrt(3.) * 1.1 * knyq, 500)
    kin = (kinedges[:-1] + kinedges[1:]) / 2.
    pkin = jnp.array(pk(kin))#.at[kin < mattrs.knyq[0] / 2.].set(0.)

    edges = np.arange(0., np.sqrt(3.) * 1.1 * knyq, 0.01)
    bin = BinAttrs(mattrs, edges=edges)
    volume = mattrs.kfun.prod() * bin.nmodes
    volume2 = 4. / 3. * np.pi * (edges[1:]**3 - edges[:-1]**3)
    _k = (edges[:-1] + edges[1:]) / 2.
    _damping = (volume / volume2).at[_k < knyq].set(1.)

    def damping(k):
        toret = np.interp(k, _k, _damping, left=1., right=0.)
        return toret
        #kmax = np.sqrt(3.) * mattrs.knyq[0]
        #dk = kmax - k
        #k0 = mattrs.knyq[0]
        #toret = np.where(k > k0, (dk / (kmax - k0))**2, 1.)
        #toret = np.where(dk < 0, 0., toret)

    def find_unique(xvec, x0, xmin=0., xmax=np.inf):
        x2 = sum(xx**2 for xx in xvec).ravel()
        x2 = x2[(x2 >= xmin**2) & (x2 <= xmax**2)]
        _, index, counts = np.unique(np.int64(x2 / (0.5 * x0)**2 + 0.5), return_index=True, return_counts=True)
        return np.sqrt(x2[index]), counts

    if False:
        mk, nk = find_unique(mattrs.kcoords(sparse=True), mattrs.kfun.min())
        volume = mattrs.kfun.prod() * nk
        volume2 = 4. / 3. * np.pi * utils.weights_trapz(mk**3)
        ax = plt.gca()
        mask = Ellipsis #mk < mattrs.knyq[0] / 2.
        ax.plot(mk[mask], volume[mask] / volume2[mask])
        plt.show()
        exit()

    if False:
        edges = np.arange(0., np.sqrt(3.) * mattrs.knyq[0], 0.01)
        bin = BinAttrs(mattrs, edges=edges)
        volume = mattrs.kfun.prod() * bin.nmodes
        volume2 = 4. / 3. * np.pi * utils.weights_trapz(bin.xavg**3)
        ax = plt.gca()
        ax.plot(bin.xavg, volume / volume2)
        ax.plot(bin.xavg, damping(bin.xavg))
        plt.show()
        exit()

    def kernel(value, kvec):
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        interp = Interpolator1D(kinedges, knorm, order=0, edges=True)
        toret = interp(pkin) / mattrs.cellsize.prod()
        #return toret
        return toret.at[0, 0, 0].set(0.)

    if False:
        ax = plt.gca()
        s = np.linspace(0.1, mattrs.boxsize[0] / 2, 1000)
        xi = pk.to_xi()(s)
        ax.plot(s, s**2 * xi, color='k')

        xi = mattrs.create(kind='complex').apply(kernel, kind='wavenumber').c2r()
        sedges = np.linspace(0.1, mattrs.boxsize[0] / 2, 1000)
        bin = BinAttrs(xi, edges=sedges)
        xi = bin(xi).real
        ax.plot(bin.xavg, bin.xavg**2 * xi)

    ell = 2

    ax = plt.gca()

    if True:
        s = bin.xavg
        tophat = TophatPowerToCorrelation(kinedges, s, edges=True, ell=ell)
        #tophat = BesselPowerToCorrelation(kinedges, s, edges=True)
        xi = tophat(pkin) # * damping(kin))
        ax.plot(s, s**2 * xi)

    if False:
        s = bin.xavg
        mk, nk = find_unique(mattrs.kcoords(sparse=True), mattrs.kfun.min())
        interp = Interpolator1D(kinedges, mk, order=0, edges=True)
        tophat = BesselPowerToCorrelation(mk, s, ell=ell) #, volume=mattrs.kfun.prod() * nk)
        xi = tophat(interp(pkin).at[0].set(0.) * damping(mk))
        ax.plot(s, s**2 * xi)

    if True:
        mk, nk = find_unique(mattrs.kcoords(sparse=True), mattrs.kfun.min())
        interp = Interpolator1D(kinedges, mk, order=0, edges=True)
        volume = mattrs.kfun.prod() * nk
        #s = seval = np.linspace(0.1, mattrs.boxsize[0], 1000)
        s = seval = bin.xavg
        from scipy import special
        w = (-1)**(ell // 2) / (2. * np.pi)**3 * volume * special.spherical_jn(ell, seval[..., None] * mk)
        tmp = interp(pkin).at[0].set(0.)
        xi = w.dot(tmp)
        ax.plot(bin.xavg, bin.xavg**2 * xi)

    plt.show()


def test_power_to_correlation2():

    from matplotlib import pyplot as plt
    from jaxpower.utils import TophatPowerToCorrelation, BesselPowerToCorrelation, Interpolator1D
    from jaxpower.power import BinAttrs

    def pk(k):
        kp = 0.01
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    mattrs = MeshAttrs(boxsize=500, meshsize=128)
    kfun = mattrs.kfun[0]
    knyq = mattrs.knyq[0]

    kinedges = np.linspace(0., 0.9 * knyq, 200)
    kin = (kinedges[:-1] + kinedges[1:]) / 2.
    pkin = jnp.array(pk(kin))
    pkin = jnp.where(np.isin(np.arange(len(kinedges) - 1), [50, 51]), pkin, 0.)

    def find_unique(xvec, x0, xmin=0., xmax=np.inf):
        x2 = sum(xx**2 for xx in xvec).ravel()
        x2 = x2[(x2 >= xmin**2) & (x2 <= xmax**2)]
        _, index, counts = np.unique(np.int64(x2 / (0.5 * x0)**2 + 0.5), return_index=True, return_counts=True)
        return np.sqrt(x2[index]), counts

    sunique = find_unique(mattrs.xcoords(kind='separation', sparse=True), mattrs.cellsize.min())[0]
    snorm = jnp.sqrt(sum(xx**2 for xx in mattrs.xcoords(kind='separation', sparse=True)))

    if False:
        tophat = TophatPowerToCorrelation(kinedges, sunique, edges=True)
        #tophat = BesselPowerToCorrelation(kinedges, s, edges=True)
        xi = tophat(pkin)
        interp = Interpolator1D(sunique, snorm, order=0)
        value = interp(xi)#.astype(complex)
        mesh = mattrs.create(kind='real')
        mesh = mesh.clone(value=value)
        mesh = mesh.r2c() * mattrs.cellsize.prod()
        bin = BinAttrs(mesh, edges=kinedges)
        pk = bin(mesh).real

    if False:
        xi = pk.to_xi()
        value = xi(snorm)
        value[0, 0, 0] = 0. #1 / (2. * np.pi)**3 * np.trapz(4. * np.pi * pk.k**2 * pk.pk, x=pk.k)
        value = interp(xi)#.astype(complex)
        mesh = mattrs.create(kind='real')
        mesh = mesh.clone(value=value)
        mesh = mesh.r2c() * mattrs.cellsize.prod()
        bin = BinAttrs(mesh, edges=kinedges)
        pk = bin(mesh).real

    if True:
        mk, nk = find_unique(mattrs.kcoords(hermitian=False, sparse=True), mattrs.kfun.min())
        interp = Interpolator1D(kinedges, mk, order=0, edges=True)
        volume = mattrs.kfun.prod() * nk
        ell = 0
        #s = seval = np.linspace(0.1, mattrs.boxsize[0], 1000)
        #sedges = np.linspace(0.1, mattrs.boxsize[0] / 2, 1000)
        #seval = (sedges[1:] + sedges[:-1]) / 2.
        seval = sunique
        from scipy import special
        w = (-1)**(ell // 2) / (2. * np.pi)**3 * volume * special.spherical_jn(ell, seval[..., None] * mk)
        tmp = interp(pkin)#.at[0].set(0.)
        xi = w.dot(tmp)
        interp = Interpolator1D(sunique, snorm, order=0)
        value = interp(xi)#.astype(complex)
        mesh = mattrs.create(kind='real')
        mesh = mesh.clone(value=value)
        mesh = mesh.r2c() * mattrs.cellsize.prod()
        bin = BinAttrs(mesh, edges=kinedges)
        pk = bin(mesh).real

        if False:
            def kernel(value, kvec):
                knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
                interp = Interpolator1D(kinedges, knorm, order=0, edges=True)
                toret = interp(pkin) / mattrs.cellsize.prod()
                #return toret
                return toret.at[0, 0, 0].set(0.)

            xi2 = mattrs.create(kind='complex').apply(kernel, kind='wavenumber').c2r()
            bin = BinAttrs(xi2, edges=sedges)
            xi2 = bin(xi2, bin).real

            ax = plt.gca()
            ax.plot(seval, seval**2 * xi, color='k')
            ax.plot(bin.xavg, bin.xavg**2 * xi2, color='k')
            plt.show()
            exit()

    mesh2 = mattrs.create(kind='hermitian_complex')
    knorm = jnp.sqrt(sum(xx**2 for xx in mattrs.kcoords(kind='separation', hermitian=True, sparse=True)))
    interp = Interpolator1D(kinedges, knorm, order=0, edges=True)
    mesh2 = mesh2.clone(value=interp(pkin))
    bin = BinAttrs(mesh2, edges=kinedges)
    pk2 = bin(mesh2).real

    #mesh += mesh2.mean() - mesh.mean()
    #pk = bin(mesh).real

    ax = plt.gca()
    mask = kin < knyq
    ax.plot(kin[mask], kin[mask] * pkin[mask], color='k')
    ax.plot(kin[mask], kin[mask] * pk[mask])
    #ax.plot(kin, kin * pk2)
    plt.show()


def test_power_to_correlation3():

    from matplotlib import pyplot as plt
    from jaxpower.utils import TophatPowerToCorrelation, BesselPowerToCorrelation, Interpolator1D
    from jaxpower import BinAttrs

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    mattrs = MeshAttrs(boxsize=1500, meshsize=64)
    kinedges = np.linspace(0., 1.01 * np.sqrt(3) * mattrs.knyq[0], 200)
    #kinedges = np.linspace(0., mattrs.knyq[0], 200)
    kin = (kinedges[:-1] + kinedges[1:]) / 2.
    klim = (0.1, 0.102)
    mask = (kin > klim[0]) & (kin < klim[1])
    pkin = jnp.array(pk(kin)) * mask

    def gaussian_survey(attrs, size=int(1e7), seed=random.key(42), scale=0.24, paint=False):
        # Generate Gaussian-distributed positions
        positions = scale * random.normal(seed, shape=(size, 3))
        bscale = scale  # cut at 1 sigmas
        mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
        positions = positions * attrs.boxsize + attrs.boxcenter
        toret = ParticleField(positions, weights=1. * mask, boxcenter=attrs.boxcenter, boxsize=attrs.boxsize, meshsize=attrs.meshsize)
        if paint: toret = toret.paint(resampler='cic', interlacing=1, compensate=False)
        return toret

    selection = gaussian_survey(mattrs, paint=True)
    norm = compute_normalization(selection, selection)
    selection = selection.r2c()
    selection = (selection * selection.conj()).c2r()
    selection /= norm

    def kernel(value, kvec):
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        interp = Interpolator1D(kinedges, knorm, order=0, edges=True)
        toret = interp(pkin) / mattrs.cellsize.prod()
        #return toret
        return toret.at[0, 0, 0].set(0.)

    pk = mattrs.create(kind='complex').apply(kernel, kind='wavenumber')
    kbin = BinAttrs(pk, edges=kinedges)
    xi = pk.c2r() * selection
    bin = BinAttrs(xi)
    bxi = bin(xi)
    bpk = kbin(xi.r2c())
    snorm = jnp.sqrt(sum(xx**2 for xx in xi.coords(kind='separation', sparse=True)))
    interp = Interpolator1D(bin.edges, snorm, order=0, edges=True)
    xi = xi.clone(value=interp(bxi))
    bxi2 = bin(xi)
    bpk2 = kbin(xi.r2c())

    if True:
        kbin2 = BinAttrs(pk) #, edges=dict(max=mattrs.knyq[0]))
        kinterp = Interpolator1D(kinedges, kbin2.xavg, order=0, edges=True)
        bessel = BesselPowerToCorrelation(kbin2.xavg, bin.xavg, ell=0, volume=mattrs.kfun.prod() * kbin2.nmodes)
        bxi3 = bessel(kinterp(pkin))
    else:
        tophat = TophatPowerToCorrelation(kinedges, bin.xavg, edges=True)
        bxi3 = tophat(pkin)
    xi = xi.clone(value=interp(bxi3).astype(xi.dtype) * selection.value)
    bxi3 = bin(xi)
    bpk3 = kbin(xi.r2c())

    if True:
        ax = plt.gca()
        ax.plot(bin.xavg, bin.xavg**2 * bxi, color='C0')
        ax.plot(bin.xavg, bin.xavg**2 * bxi2, color='C1')
        ax.plot(bin.xavg, bin.xavg**2 * bxi3, color='C2')
        plt.show()

    if True:
        ax = plt.gca()
        mask = kin < mattrs.knyq[0]
        ax.plot(kin[mask], kin[mask] * pkin[mask], color='k')
        mask = kbin.xavg < mattrs.knyq[0]
        ax.plot(kbin.xavg[mask], kbin.xavg[mask] * bpk[mask], color='C0')
        ax.plot(kbin.xavg[mask], kbin.xavg[mask] * bpk2[mask], color='C1')
        ax.plot(kbin.xavg[mask], kbin.xavg[mask] * bpk3[mask], color='C2')
        plt.show()


def test_window():

    from jaxpower import compute_mesh_window
    from jaxpower.utils import Interpolator1D

    ellsin = (0, 2, 4)
    edgesin = np.array([0.1, 0.11])
    xin = (edgesin[:-1] + edgesin[1:]) / 2.
    theory = BinnedStatistic(x=[xin] * len(ellsin), edges=[edgesin] * len(ellsin), value=[np.ones_like(xin)] + [np.ones_like(xin)] * (len(ellsin) - 1), projs=ellsin)

    attrs = MeshAttrs(boxsize=500., meshsize=64, boxcenter=1000.)
    edges = {'step': 0.01}

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
    ells = (0, 2, 4)

    for flag in ['smooth', 'infinite']:
        for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')]:
            print(flag, los, thlos)
            mean = compute_mean_mesh_power(selection, theory=(theory, thlos) if thlos is not None else theory, ells=ells, edges=edges, los=los).clone(norm=norm)
            wmatrix = compute_mesh_window(selection, edgesin=edgesin, ellsin=(ellsin, thlos) if thlos is not None else ellsin, ells=ells, edges=edges, los=los, pbar=True, norm=norm, flags=(flag,))
            from matplotlib import pyplot as plt
            ax = plt.gca()
            for iproj, proj in enumerate(mean.projs):
                color = 'C{:d}'.format(iproj)
                ax.plot(mean.x(projs=proj), mean.view(projs=proj), color=color, linestyle='-')
                projin = (0, 0) if thlos == 'firstpoint' else 0
                ax.plot(mean.x(projs=proj), wmatrix.select(projs=proj, select_projs=True, axis='o').select(projs=projin, select_projs=True, axis='t').view(), color=color, linestyle='--')
            plt.show()


def test_window_timing():

    from jaxpower import compute_mesh_window
    from jaxpower.utils import Interpolator1D

    ellsin = (0, 2, 4)
    edgesin = np.linspace(0.01, 0.1, 20)
    xin = (edgesin[:-1] + edgesin[1:]) / 2.
    theory = BinnedStatistic(x=[xin] * len(ellsin), edges=[edgesin] * len(ellsin), value=[np.ones_like(xin)] + [np.ones_like(xin)] * (len(ellsin) - 1), projs=ellsin)

    attrs = MeshAttrs(boxsize=500., meshsize=64, boxcenter=1000.)
    edges = {'step': 0.01}

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
    ells = (0, 2, 4)

    for flag in ['smooth', 'infinite'][-1:]:
        for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')][-1:]:
            print(flag, los, thlos)
            t0 = time.time()
            wmatrix = compute_mesh_window(selection, edgesin=edgesin, ellsin=(ellsin, thlos) if thlos is not None else ellsin, ells=ells, edges=edges, los=los, pbar=True, norm=norm, flags=(flag,))
            print(f'{time.time() - t0:.2f}')


def test_smooth_window():
    from matplotlib import pyplot as plt
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)

    attrs = MeshAttrs(boxsize=2000., meshsize=100, boxcenter=800.)
    edges = {'step': 0.005}
    #edgesin = np.arange(0., 1.1 * attrs.knyq.max(), 0.002)
    edgesin = np.arange(0., attrs.knyq.max(), 0.002)
    kin = (edgesin[:-1] + edgesin[1:]) / 2.
    ellsin = (0, 2, 4)
    f, b = 0.8, 1.5
    beta = f / b
    ells = (0, 2, 4)
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
        norm = compute_normalization(selection, selection)
        selection = selection.r2c()
        selection = (selection * selection.conj()).c2r()
        edges = np.arange(0., selection.boxsize.max(), selection.cellsize.min())
        bin = BinAttrs(selection, edges=edges)
        pole = bin(selection) / norm / selection.cellsize.prod()
        print(pole)

        from matplotlib import pyplot as plt
        ax = plt.gca()
        ax.plot(bin.xavg, pole)
        ax.set_xscale('log')
        plt.show()

    selection = gaussian_survey(attrs, paint=True)
    norm = compute_normalization(selection, selection)

    for thlos in ['firstpoint', 'local'][:1]:
        mean = compute_mean_mesh_power(selection, theory=(edgesin, list(poles), thlos),
                                       ells=ells, edges=edges, los='firstpoint').clone(norm=norm)
        wmatrix = compute_mesh_window(selection, edgesin=edgesin, ellsin=(ellsin, thlos),
                                      edges=edges, ells=ells, los='firstpoint', norm=norm, flags=('smooth',), pbar=True)
        wpoles = wmatrix.dot(poles, return_type=None)
        ax = plt.gca()
        for iproj, proj in enumerate(mean.projs):
            ax.plot(kin, kin * poles[iproj], color='k')
            k = mean.x(projs=proj)
            ax.plot(k, k * mean.view(projs=proj), color='C0')
            ax.plot(k, k * wpoles.view(projs=proj), color='C1')
        plt.show()


def tophat_bessel():

    from matplotlib import pyplot as plt
    from jaxpower.utils import TophatPowerToCorrelation, TophatCorrelationToPower, BesselPowerToCorrelation

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    xi = pk.to_xi()

    #attrs = MeshAttrs(boxsize=500., meshsize=64, boxcenter=800.)
    kedgesin = np.arange(0., 1., 0.01)
    kin = (kedgesin[:-1] + kedgesin[1:]) / 2.
    f, b = 0.8, 1.5
    beta = f / b
    ells = (0, 2, 4)[:1]
    poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])
    sedges = jnp.linspace(0., 300, 1000)
    s = (sedges[:-1] + sedges[1:]) / 2.
    s = 3. / 4. * (sedges[1:]**4 - sedges[:-1]**4) / (sedges[1:]**3 - sedges[:-1]**3)
    ax = plt.gca()
    for ill, ell in enumerate(ells):
        tophat1 = TophatPowerToCorrelation(kedgesin, s, ell=ell, edges=True)
        tmp = tophat1(poles[ill])
        tophat2 = TophatCorrelationToPower(sedges, kin, ell=ell, edges=True)
        #poles2 = tophat2(xi(s))
        poles2 = tophat2(tmp)
        if False:
            smask = s < 200
            ax.plot(s[smask], s[smask]**2 * xi(s[smask]), color='k')
            ax.plot(s[smask], s[smask]**2 * tmp[smask], color='C0')
        else:
            ax.plot(kin, kin * poles[ill], color='k')
            ax.plot(kin, kin * poles2, color='C0')
            #ax.plot(kin, poles2 / poles[ill], color='C0')
    plt.show()


def test_wmatrix():
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

    value = np.bmat([[f(*np.meshgrid(xo, xt, indexing='ij')) for xt in theory.x()] for xo in observable.x()])
    wmatrix = WindowMatrix(observable=observable, theory=theory, value=value)
    wmatrix.plot(show=True)
    wmatrix2 = wmatrix.interp(theory2, axis='t', extrap=True)
    wmatrix2.plot(show=True)
    wmatrix3 = wmatrix.interp(observable2, axis='o', extrap=True)
    wmatrix3.plot(show=True)


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
        expr = integrand
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
        mesh = attrs.create(kind='real', fill=create_sharded_random(jax.random.normal, jax.random.key(42), shape=attrs.meshsize))
        compute = partial(compute_mesh_power, edges={'step': 0.001}, ells=(0, 2, 4), los='firstpoint')
        compute = jax.jit(compute)
        t0 = time.time()
        mesh_power = compute(mesh)
        jax.block_until_ready(mesh_power)
        print(time.time() - t0)


if __name__ == '__main__':

    #test_window_timing()
    #test_sympy()
    #test_window()
    #test_wmatrix()
    #test_gaunt()
    #test_smooth_window()
    #tophat_bessel()
    #test_power_to_correlation3()
    #test_checkpoint()
    #test_power_to_correlation()
    #test_power_to_correlation2()
    test_mesh_power(plot=False)
    exit()
    test_binned_statistic()
    test_mesh_power(plot=False)
    test_fkp_power(plot=False)
    test_mean_power(plot=True)
    test_window()
    test_wmatrix()