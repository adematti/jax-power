from pathlib import Path

import numpy as np
import jax
from jax import jit, random
from jax import numpy as jnp
from matplotlib import pyplot as plt

from jaxpower import (BinnedStatistic, CovarianceMatrix, MeshAttrs, compute_mesh2_covariance_window, compute_fkp2_covariance_window, compute_spectrum2_covariance,
                      generate_anisotropic_gaussian_mesh, generate_uniform_particles, BinMesh2Spectrum, Spectrum2Poles, compute_mesh2_spectrum)


dirname = Path('_tests')


def test_covmatrix(plot=False):
    xo = np.linspace(0., 0.2, 21)
    ells = (0, 2)
    observables = [BinnedStatistic(x=[xo] * len(ells), projs=ells, name='power'), BinnedStatistic(x=[xo] * len(ells[:1]), projs=ells[:1], name='monopole')]
    value = np.eye(3 * xo.size)
    cov = CovarianceMatrix(value=value, observables=observables)
    assert cov.shape == value.shape
    cov2 = cov.select(observables='power', select_observables=True)
    assert cov2.shape == (42,) * 2
    cov2 = cov2.select(xlim=(0., 0.1))
    assert cov2.shape == (22,) * 2

    nmocks = 100
    rng = np.random.RandomState(seed=42)
    observables = [BinnedStatistic(x=[xo] * len(ells), value=rng.uniform(0., 1., size=(len(ells), xo.size)), projs=ells, name='power') for i in range(nmocks)]
    cov2 = BinnedStatistic.cov(observables)

    fn = dirname / 'tmp.npy'
    cov2.save(fn)
    cov2 = CovarianceMatrix.load(fn)

    if plot:
        cov.plot(show=True)
        cov2.plot(corrcoef=True, show=True)


def export_sympy():
    import itertools
    from jaxpower.utils import export_legendre_product, compute_sympy_correlation_function_derivative, compute_sympy_bessel, compute_sympy_legendre

    #print(export_sympy_legendre_product(ellmax=8, n=3))
    #for ell in range(6): print(ell, compute_sympy_correlation_function_derivative(ell))
    for ell in range(11): print('_registered_bessel[{:d}] = (lambda x: {},\nlambda x: {})'.format(ell, *compute_sympy_bessel(ell)))
    for ell in range(11): print('_registered_legendre[{:d}] = lambda x: {}'.format(ell, compute_sympy_legendre(ell)))


def get_theory(kmax=0.3, dk=0.005):
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    ellsin = (0, 2, 4)
    edgesin = np.arange(0., kmax, dk)
    edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])
    kin = (edgesin[..., 0] + edgesin[..., 1]) / 2.
    f, b = 0.8, 2.
    #f = 0.
    beta = f / b
    pk = 5 * pk(kin)
    #pk = jnp.full(kin.shape, pk(0.1))
    shotnoise = (1e-4)**(-1)
    poles = np.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk + shotnoise,
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk,
                        8. / 35 * beta ** 2 * pk])
    return Spectrum2Poles(k=[kin] * len(ellsin), edges=[edgesin] * len(ellsin), num=list(poles), num_shotnoise=shotnoise, ells=ellsin)


def mock_fn(basename='box', imock=None):
    fn = dirname / basename / 'poles_{:d}.npy'
    fn = str(fn)
    if imock is None:
        imock = list(range(100))
    if isinstance(imock, list):
        return [fn.format(i) for i in imock]
    return fn.format(imock)


def save_box_mocks():
    attrs = MeshAttrs(boxsize=500., boxcenter=[0., 0., 1200], meshsize=128)
    theory = get_theory(kmax=attrs.knyq.max())
    bin = BinMesh2Spectrum(attrs, edges=theory.edges(projs=0), ells=(0, 2, 4))
    los = 'z'

    @jit
    def mock(seed):
        mesh = generate_anisotropic_gaussian_mesh(attrs, poles=theory, los=los, seed=seed)
        return compute_mesh2_spectrum(mesh, bin=bin, los=los)

    for i, fn in enumerate(mock_fn(basename='box')):
        print(i, end=" ", flush=True)
        poles = mock(random.key(i))
        poles.save(fn)
    print()


def test_box2_covariance(plot=False):

    attrs = MeshAttrs(boxsize=500., boxcenter=[0., 0., 1200], meshsize=128)
    theory = get_theory(kmax=attrs.knyq.max())
    bin = BinMesh2Spectrum(attrs, edges=theory.edges(projs=0))
    theory = theory.clone(nmodes=bin.nmodes)

    cov_mocks = Spectrum2Poles.cov(list(map(Spectrum2Poles.load, mock_fn(basename='box'))))
    if False: #plot:
        observable = cov_mocks.observables()[0]
        fig = observable.plot()
        theory.plot(fig=fig, show=True)

    observable = cov_mocks.observables()[0]
    cov_mocks.slice(slice(0, None, 2))
    cov_smooth = compute_spectrum2_covariance(attrs, theory, flags=('smooth',))
    cov_mesh = compute_spectrum2_covariance(attrs, theory, flags=tuple())
    cov_mocks, cov_smooth, cov_mesh = [cov.slice(slice(0, None, 2)) for cov in (cov_mocks, cov_smooth, cov_mesh)]

    if plot:
        #cov_mesh.plot(corrcoef=True, show=True)
        ytransform = lambda x, y: x**4 * y
        kw = dict(ytransform=ytransform)
        fig = cov_smooth.plot_diag(**kw, color='C0')
        cov_mesh.plot_diag(**kw, color='C1', fig=fig)
        cov_mocks.plot_diag(**kw, color='C2', fig=fig, show=True)


def survey_selection(size=int(1e7), seed=random.key(42), scale=0.25, paint=True):
    from jaxpower import ParticleField
    attrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 2200], meshsize=128)
    xvec = attrs.xcoords(kind='position', sparse=False)
    limits = attrs.boxcenter - attrs.boxsize / 4., attrs.boxcenter + attrs.boxsize / 4.
    if False:
        mask = True
        for i, xx in enumerate(xvec): mask &= (xx >= limits[0][i]) & (xx < limits[1][i])
        return attrs.create(kind='real', fill=0.).clone(value=1. * mask)
    # Generate Gaussian-distributed positions
    positions = scale * random.normal(seed, shape=(size, attrs.ndim))
    #positions = scale * (2 * random.uniform(seed, shape=(size, attrs.ndim)) - 1.)
    bscale = scale  # cut at 1 sigmas
    mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
    positions = positions * attrs.boxsize + attrs.boxcenter
    toret = ParticleField(positions, weights=1. * mask, boxcenter=attrs.boxcenter, boxsize=attrs.boxsize, meshsize=attrs.meshsize)
    if paint: toret = toret.paint(resampler='ngp', interlacing=1, compensate=False)
    return toret


def save_cutsky_mocks():
    from jaxpower import compute_normalization

    selection = survey_selection()
    attrs = selection.attrs
    theory = get_theory(kmax=np.sqrt(3) * attrs.knyq.max(), dk=0.001)
    bin = BinMesh2Spectrum(attrs, edges=theory.slice(slice(0, None, 5)).select(xlim=(0., attrs.knyq.max())).edges(projs=0), ells=(0, 2, 4))
    los = 'local'

    norm = compute_normalization(selection, selection)

    @jit
    def mock(seed):
        mesh = generate_anisotropic_gaussian_mesh(attrs, poles=theory, los=los, seed=seed)
        return compute_mesh2_spectrum(mesh * selection, bin=bin, los=los).clone(norm=norm)

    for i, fn in enumerate(mock_fn(basename='cutsky')):
        print(i, end=" ", flush=True)
        poles = mock(random.key(i))
        poles.save(fn)
    print()


def test_cutsky2_covariance(plot=False):

    selection = survey_selection()
    attrs = selection.attrs
    theory = get_theory(kmax=attrs.knyq.max(), dk=0.001)

    if False:
        cov_mocks = Spectrum2Poles.cov(list(map(Spectrum2Poles.load, mock_fn(basename='cutsky'))))
        #cov_mocks = cov_mocks.select(xlim=klim)
        fig = cov_mocks.observables()[0].plot()
        theory.plot(fig=fig, show=True)
        cov_mocks.plot(corrcoef=True, show=True)
        exit()

    delta = 0.08
    #edges = {'step': attrs.cellsize.min()}
    edges = None
    los = 'local'
    windows = compute_mesh2_covariance_window(selection, los=los, edges=edges)

    def smooth_window(window, xmin):
        num = []
        for iproj, proj in enumerate(window.projs):
            _num = window.num[iproj]
            if proj != 0:
                _num = jnp.where(window._x[iproj] >= xmin, _num, 0.)
            num.append(_num)
        return window.clone(num=num)

    windows = {k: smooth_window(w, 4 * attrs.cellsize.min()) for k, w in windows.items()}

    from jaxpower.utils import plotter

    @plotter
    def plot(self, fig=None):
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self._x[ill], self.value[ill].real, label=self._get_label_proj(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(self._label_x)
        #ax.set_xscale('log')
        return fig

    if plot:
        plot(windows[0, 0, 0, 0].clone(num_zero=None), show=True)

    klim = (0., attrs.knyq.max())
    cov = compute_spectrum2_covariance(windows, theory, delta=delta).slice(slice(0, None, 5)).select(xlim=klim)
    cov_fftlog = compute_spectrum2_covariance(windows, theory, delta=delta, flags=['smooth', 'fftlog']).slice(slice(0, None, 5)).select(xlim=klim)
    cov_mocks = Spectrum2Poles.cov(list(map(Spectrum2Poles.load, mock_fn(basename='cutsky')))).select(xlim=klim)
    pattrs = attrs.clone(boxsize=2. * 0.25 * attrs.boxsize)
    #cov_smooth = compute_spectrum2_covariance(pattrs, theory, delta=delta).slice(slice(0, None, 5)).select(xlim=klim)

    #cov, cov_smooth, cov_mocks = (cov.select(projs=0, select_projs=True) for cov in [cov, cov_smooth, cov_mocks])

    if plot:
        ytransform = lambda x, y: x**4 * y
        #ytransform = lambda x, y: x**2 * y
        kw = dict(ytransform=ytransform, offset=np.arange(3))
        cov.plot(corrcoef=True, show=True)
        #cov_mocks.plot(corrcoef=True, show=True)
        #fig = None
        fig = cov.plot_diag(**kw, color='C0')
        fig = cov_fftlog.plot_diag(**kw, color='C1', fig=fig)
        #fig = cov_smooth.plot_diag(**kw, color='C2', fig=fig)
        fig = cov_mocks.plot_diag(**kw, color='C3', fig=fig)
        fig.axes[0].get_legend().remove()
        plt.show()


def save_fkp_mocks():
    from jaxpower import FKPField, compute_fkp2_spectrum_normalization, compute_fkp2_spectrum_shotnoise
    attrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200], meshsize=128)
    pattrs = attrs.clone(boxsize=1000., meshsize=64)
    theory = get_theory(kmax=np.sqrt(3) * attrs.knyq.max()).clone(num_shotnoise=0.)
    bin = BinMesh2Spectrum(attrs, edges=theory.select(xlim=(0., attrs.knyq.max())).edges(projs=0), ells=(0, 2, 4))
    los = 'local'

    size = int(1e-4 * pattrs.boxsize.prod())

    @jit
    def mock(seed):
        mesh = generate_anisotropic_gaussian_mesh(attrs, poles=theory, los=los, seed=seed)
        data = generate_uniform_particles(pattrs, size, seed=seed).clone(attrs=attrs)
        data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
        randoms = generate_uniform_particles(pattrs, 5 * size, seed=42).clone(attrs=attrs)
        fkp = FKPField(data, randoms)
        mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='complex')
        norm = compute_fkp2_spectrum_normalization(fkp)
        shotnoise = compute_fkp2_spectrum_shotnoise(fkp)
        return compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=norm, num_shotnoise=shotnoise)

    for i, fn in enumerate(mock_fn(basename='fkp')):
        print(i, end=" ", flush=True)
        poles = mock(random.key(i))
        poles.save(fn)


def test_fkp2_window(plot=False):

    from cosmoprimo.fiducial import DESI
    attrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200], meshsize=128)
    pattrs = attrs.clone(boxsize=1000., meshsize=64)

    size = int(1e5)
    randoms = generate_uniform_particles(pattrs, size, seed=32).clone(attrs=attrs)
    windows = compute_fkp2_covariance_window(randoms, edges={'step': attrs.cellsize.min()}, interlacing=2, resampler='tsc', los='z')

    from jaxpower.utils import plotter

    @plotter
    def plot(self, fig=None):
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self._x[ill], self.value[ill].real, label=self._get_label_proj(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(self._label_x)
        #ax.set_xscale('log')
        return fig

    if plot:
        plot(windows[0][0, 0, 0, 0].clone(num_zero=None), show=True)


def test_fkp2_covariance(plot=False):

    if False:
        cov_mocks = Spectrum2Poles.cov(list(map(Spectrum2Poles.load, mock_fn(basename='fkp'))))
        #cov_mocks = cov_mocks.select(xlim=klim)
        cov_mocks.observables()[0].plot(show=True)
        cov_mocks.plot(corrcoef=True, show=True)
        exit()

    attrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200], meshsize=128)
    pattrs = attrs.clone(boxsize=1000., meshsize=64)
    theory = get_theory(kmax=attrs.knyq.max()).clone(num_shotnoise=0.)

    size = int(1e6)
    randoms = generate_uniform_particles(pattrs, size, seed=32).clone(attrs=attrs)
    #edges = {'step': attrs.cellsize.min()}
    edges = None
    windows = compute_fkp2_covariance_window(randoms, edges=edges, interlacing=2, resampler='tsc', los='local')
    covs = compute_spectrum2_covariance(windows, theory, delta=0.2)
    cov = sum(covs)
    klim = (0., attrs.knyq.max())
    cov = cov.select(xlim=klim)

    cov_mocks = Spectrum2Poles.cov(list(map(Spectrum2Poles.load, mock_fn(basename='fkp'))))
    cov_mocks = cov_mocks.select(xlim=klim)
    #cov_smooth = compute_spectrum2_covariance(pattrs, theory, flags=('smooth',)).select(xlim=klim)

    if plot:
        ytransform = lambda x, y: x**4 * y
        kw = dict(ytransform=ytransform)
        #cov.plot(corrcoef=True, show=True)
        #fig = None
        fig = cov.plot_diag(**kw, color='C0')
        cov_mocks.plot_diag(**kw, color='C2', fig=fig, show=True)


def test_fftlog2():
    # Taken from https://github.com/eelregit/covdisc
    from mcfit import P2xi, xi2P
    # We want matching s for different l, but lowring=False with an even N is
    # problematic, because the forward and backward transformation matrix are no
    # longer inverse of each other -- their product is the identity matrix with
    # some huge corner oscillations
    # Therefore, I use Odd N (powers of 3) here
    lowring = False
    lgkmin, lgkmax = -6, 2
    Nk = 3645
    k = np.logspace(lgkmin, lgkmax, num=Nk, endpoint=False)

    def get_ell(l_max):
        return np.arange(0, l_max+1, 2)

    def get_H(k, l_max=8, lowring=False):
        """Return Hankel circulant matrix for all (l, k, k')"""
        ell = get_ell(l_max)
        return np.stack([P2xi(k, l=l, N=len(k), lowring=lowring).matrix(full=False)[2] for l in ell], axis=0)

    H = get_H(k)

    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)(k)

    from jaxpower.fftlog import PowerToCorrelation, CorrelationToPower

    fftlog = PowerToCorrelation(k, ell=0, lowring=lowring, minfolds=False)
    tmp2 = fftlog(pk)[1]
    tmp3 = P2xi(k, l=0, N=len(k), lowring=lowring)(pk)[1]

    H0 = jax.jacfwd(lambda fun: fftlog(fun, ignore_prepostfactor=True)[1])(jnp.zeros_like(k))


    class Correlation2Power(object):

        def __init__(self, k, ells):
            from jaxpower.fftlog import PowerToCorrelation
            fftlog = PowerToCorrelation(k, ell=ells[0], lowring=False, minfolds=False)
            self._H = jax.jacfwd(lambda fun: fftlog(fun, extrap=False, ignore_prepostfactor=True)[1])(jnp.zeros_like(k))
            self._fftlog = PowerToCorrelation(k, ell=ells[1], lowring=False, minfolds=False)
            dlnk = jnp.diff(jnp.log(k)).mean()
            self._postfactor = 2 * np.pi**2 / dlnk / (k[..., None] * k)**1.5
            self.k = k
            self.s = fftlog.y

        def __call__(self, fun):
            fun = self._H * fun
            _, fun = self._fftlog(fun, extrap=False, ignore_prepostfactor=True)
            return self.k, self._postfactor * fun


    def rescale_kk(k, power):
        return k[:, None]**power * k**power

    def get_s(k, lowring=False):
        return P2xi(k, N=len(k), lowring=lowring).y


    def get_Qlls(s, l_max, lp_max):
        ell = get_ell(l_max)
        ell_prime = get_ell(lp_max)
        Q = np.zeros((len(ell), len(ell_prime), len(s)))
        for j, lp in enumerate(ell_prime):
            for i, l in enumerate(ell):
                Q[i, j] = (2 * lp + 1) * (2 * l + 1) * np.exp(-(s / 100)**2)
        return Q


    def get_Qllkk(k, l_max=8, lp_max=8):
        """Return Q_ll(k, k) for all (l, l', k, k')"""
        s = get_s(k)

        ell = get_ell(l_max)
        ell_prime = get_ell(lp_max)

        Qlls = get_Qlls(s, l_max, lp_max)

        Q = H[:len(ell), None, ...] * Qlls[..., None, :]

        for j, lp in enumerate(ell_prime):
            for i, l in enumerate(ell):
                #Q[i, j] = Q[i, j] @ H[j]
                _, Q[i, j] = P2xi(k, l=lp, N=len(k),
                                lowring=lowring)(Q[i, j], axis=1,
                                                extrap=False, convonly=True)

        dlnk = np.diff(np.log(k)).mean()
        Q *= 2 * np.pi**2 / dlnk / rescale_kk(k, 1.5)
        return Q

    def get_Qllkk2(k, l_max=8, lp_max=8):
        """Return Q_ll(k, k) for all (l, l', k, k')"""
        s = get_s(k)

        ell = get_ell(l_max)
        ell_prime = get_ell(lp_max)

        Qll = get_Qlls(s, l_max, lp_max)
        Q = np.zeros((len(ell), len(ell_prime), len(k), len(k)))

        for j, lp in enumerate(ell_prime):
            for i, l in enumerate(ell):
                #Q[i, j] = Q[i, j] @ H[j]
                fftlog = Correlation2Power(k, (l, lp))
                assert np.allclose(fftlog.s, s)
                _, Q[i, j] = fftlog(Qll[i, j])

        return Q

    Qref = get_Qllkk(k, l_max=2, lp_max=2)
    Q = get_Qllkk2(k, l_max=2, lp_max=2)
    assert np.allclose(Q, Qref)
    #ax = plt.gca()
    #ax.pcolormesh(d[0])
    #plt.show()


def test_from_pypower():
    attrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200], meshsize=128)
    theory = get_theory(kmax=np.sqrt(3) * attrs.knyq.max()).clone(num_shotnoise=0.)
    bin = BinMesh2Spectrum(attrs, edges=theory.select(xlim=(0., attrs.knyq.max())).edges(projs=0), ells=(0, 2, 4))
    los = 'local'

    size = int(1e-4 * attrs.boxsize.prod())
    seed = 42
    mesh = generate_anisotropic_gaussian_mesh(attrs, poles=theory, los=los, seed=seed)
    data = generate_uniform_particles(attrs, size, seed=seed).clone(attrs=attrs)
    data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    randoms = generate_uniform_particles(attrs, 5 * size, seed=42).clone(attrs=attrs)


    from pypower import CatalogFFTPower
    power = CatalogFFTPower(data_positions1=data.positions, data_weights1=data.weights,
                            randoms_positions1=randoms.positions, randoms_weights1=randoms.weights,
                            position_type='pos', edges=np.linspace(0., 0.2, 21), ells=(0, 2, 4), los='firstpoint',
                            nmesh=64, interlacing=2, resampler='tsc')

    def from_pypower(poles):
        return Spectrum2Poles(k=poles.k, num=poles.power_nonorm, edges=np.array(list(zip(poles.edges[0][:-1], poles.edges[0][1:]))),
                              ells=poles.ells, nmodes=poles.nmodes, num_shotnoise=poles.shotnoise_nonorm, num_zero=poles.power_zero_nonorm, norm=poles.wnorm)

    poles = from_pypower(power.poles)
    poles.slice(slice(0, None, 2))
    assert np.allclose(poles.view(), np.concatenate(power.poles()), equal_nan=True)


if __name__ == '__main__':

    #from jax import config
    #config.update('jax_enable_x64', True)
    #test_fftlog2()

    #export_sympy()
    #from jax import config
    #config.update('jax_enable_x64', True)

    #test_covmatrix(plot=True)
    #save_box_mocks()
    #test_box2_covariance(plot=True)
    #save_cutsky_mocks()
    #test_cutsky2_covariance(plot=True)
    #test_cutsky2_covariance_fftlog(plot=True)
    #test_fkp2_window(plot=True)
    #test_fkp2_covariance(plot=True)
    test_from_pypower()