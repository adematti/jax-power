from pathlib import Path

import numpy as np
import jax
from jax import jit, random
from jax import numpy as jnp
from matplotlib import pyplot as plt

from jaxpower import (MeshAttrs, compute_mesh2_covariance_window, compute_fkp2_covariance_window, compute_spectrum2_covariance,
                      generate_anisotropic_gaussian_mesh, generate_uniform_particles, BinMesh2SpectrumPoles, Mesh2SpectrumPole, Mesh2SpectrumPoles,
                      read, compute_mesh2_spectrum, interpolate_window_function)
from jaxpower.types import ObservableTree


dirname = Path('_tests')


def export_sympy():
    import itertools
    from jaxpower.utils import export_legendre_product, compute_sympy_bessel_tophat_integral, compute_sympy_bessel, compute_sympy_legendre

    #print(export_sympy_legendre_product(ellmax=8, n=3))
    for ell in range(6): print('_registered_bessel_tophat_integral[{:d}] = (lambda x: {},\nlambda x: {})'.format(ell, *compute_sympy_bessel_tophat_integral(ell)))
    #for ell in range(11): print('_registered_bessel[{:d}] = (lambda x: {},\nlambda x: {})'.format(ell, *compute_sympy_bessel(ell)))
    #for ell in range(11): print('_registered_legendre[{:d}] = lambda x: {}'.format(ell, compute_sympy_legendre(ell)))


def get_theory(kmax=0.3, dk=0.005):
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    ellsin = (0, 2, 4)
    edgesin = np.arange(0., kmax, dk)
    edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])
    kin = np.mean(edgesin, axis=-1)
    f, b = 0.8, 2.
    #f = 0.
    beta = f / b
    pk = 5 * pk(kin)
    #pk = jnp.full(kin.shape, pk(0.1))
    shotnoise = (1e-4)**(-1)
    theory = [(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk + shotnoise,
                0.99 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk,
                8. / 35 * beta ** 2 * pk]

    poles = []
    for ell, value in zip(ellsin, theory):
        poles.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=value, num_shotnoise=shotnoise * np.ones_like(kin) * (ell == 0), ell=ell))
    return Mesh2SpectrumPoles(poles)


def mock_fn(basename='box', imock=None):
    fn = dirname / basename / 'poles_{:d}.h5'
    fn = str(fn)
    if imock is None:
        imock = list(range(100))
    if isinstance(imock, list):
        return [fn.format(i) for i in imock]
    return fn.format(imock)


def save_box_mocks():
    mattrs = MeshAttrs(boxsize=500., boxcenter=[0., 0., 1200], meshsize=128)
    theory = get_theory(kmax=mattrs.knyq.max())
    bin = BinMesh2SpectrumPoles(mattrs, edges=theory.get(ells=0).edges('k'), ells=(0, 2, 4))
    los = 'z'

    @jit
    def mock(seed):
        mesh = generate_anisotropic_gaussian_mesh(mattrs, poles=theory, los=los, seed=seed)
        return compute_mesh2_spectrum(mesh, bin=bin, los=los)

    for i, fn in enumerate(mock_fn(basename='box')):
        print(i, end=" ", flush=True)
        poles = mock(random.key(i))
        poles.write(fn)
    print()


def test_box2_covariance(plot=False):

    mattrs = MeshAttrs(boxsize=500., boxcenter=[0., 0., 1200], meshsize=128)
    theory = get_theory(kmax=mattrs.knyq.max())
    bin = BinMesh2SpectrumPoles(mattrs, edges=theory.get(0).edges('k'))
    theory = theory.clone(nmodes=[bin.nmodes] * len(theory.ells))

    cov_mocks = Mesh2SpectrumPoles.cov(list(map(read, mock_fn(basename='box'))))

    if plot:
        observable = cov_mocks.observable
        fig = observable.plot()
        theory.plot(fig=fig, show=True)

    observable = cov_mocks.observable
    cov_smooth = compute_spectrum2_covariance(mattrs, theory, flags=('smooth',))
    cov_mesh = compute_spectrum2_covariance(mattrs, theory, flags=tuple())

    cov_mocks, cov_smooth, cov_mesh = [cov.at.observable.select(k=slice(0, None, 2)) for cov in (cov_mocks, cov_smooth, cov_mesh)]

    if plot:
        #cov_mesh.plot(corrcoef=True, show=True)
        ytransform = lambda x, y: x**4 * y
        kw = dict(ytransform=ytransform)
        fig = cov_smooth.plot_diag(**kw, color='C0')
        cov_mesh.plot_diag(**kw, color='C1', fig=fig)
        cov_mocks.plot_diag(**kw, color='C2', fig=fig, show=True)


def survey_selection(size=int(1e7), seed=random.key(42), scale=0.25, paint=True):
    from jaxpower import ParticleField
    mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 2200], meshsize=128)
    if False:
        xvec = mattrs.xcoords(kind='position', sparse=False)
        limits = mattrs.boxcenter - mattrs.boxsize / 4., mattrs.boxcenter + mattrs.boxsize / 4.
        mask = True
        for i, xx in enumerate(xvec): mask &= (xx >= limits[0][i]) & (xx < limits[1][i])
        return mattrs.create(kind='real', fill=0.).clone(value=1. * mask)
    # Generate Gaussian-distributed positions
    positions = scale * random.normal(seed, shape=(size, mattrs.ndim))
    #positions = scale * (2 * random.uniform(seed, shape=(size, mattrs.ndim)) - 1.)
    bscale = scale  # cut at 1 sigmas
    mask = jnp.all((positions > -bscale) & (positions < bscale), axis=-1)
    positions = positions * mattrs.boxsize + mattrs.boxcenter
    toret = ParticleField(positions, weights=1. * mask, attrs=mattrs)
    if paint: toret = toret.paint(resampler='cic', interlacing=0, compensate=False)
    return toret


def save_cutsky_mocks():
    from jaxpower import compute_normalization

    selection = survey_selection()
    mattrs = selection.attrs
    theory = get_theory(kmax=np.sqrt(3) * mattrs.knyq.max(), dk=0.001)
    bin = BinMesh2SpectrumPoles(mattrs, edges=theory.select(k=slice(0, None, 5)).select(k=(0., mattrs.knyq.max())).get(ells=0).edges('k'), ells=(0, 2, 4))
    los = 'local'

    norm = compute_normalization(selection, selection, bin=bin)

    @jit
    def mock(seed):
        mesh = generate_anisotropic_gaussian_mesh(mattrs, poles=theory, los=los, seed=seed)
        return compute_mesh2_spectrum(mesh * selection, bin=bin, los=los).clone(norm=norm)

    for i, fn in enumerate(mock_fn(basename='cutsky')):
        print(i, end=" ", flush=True)
        poles = mock(random.key(i))
        poles.write(fn)
    print()


def test_cutsky2_covariance(plot=False):

    selection = survey_selection(paint=False)
    selection = selection.clone(attrs=selection.attrs.clone(boxsize=3000.)).paint(resampler='cic', interlacing=0, compensate=False)
    mattrs = selection.attrs
    theory = get_theory(kmax=mattrs.knyq.max(), dk=0.001)

    if False:
        cov_mocks = Mesh2SpectrumPoles.cov(list(map(read, mock_fn(basename='cutsky'))))
        #cov_mocks = cov_mocks.select(xlim=klim)
        fig = cov_mocks.observable.plot()
        theory.plot(fig=fig, show=True)
        cov_mocks.plot(corrcoef=True, show=True)
        exit()

    delta = 0.08

    from jaxpower.utils import plotter

    @plotter
    def plot(self, fig=None):
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            pole = self.get(ell)
            ax.plot(pole.coords('s'), pole.value())
        ax.legend()
        ax.grid(True)
        #ax.set_xscale('log')
        return fig

    kw_window = dict(edges=None, basis=None, los='local')
    windows = compute_mesh2_covariance_window(selection, **kw_window)

    def smooth_window(windows, xmin):
        num = []
        for window in windows:
            for ell in window.ells:
                _num = window.get(ells=ell).value()
                if ell != 0:
                    _num = jnp.where(window.get(ells=ell).coords('s') >= xmin, _num, 0.)
                num.append(_num)
        return windows.clone(value=np.concatenate(num))

    windows = smooth_window(windows, 4 * mattrs.cellsize.min())
    if False: #plot:
        plot(windows.get(fields=(0, 0, 0, 0)), show=True)

    klim = (0., mattrs.knyq.max())
    cov = compute_spectrum2_covariance(windows, theory, delta=delta).at.observable.select(k=slice(0, None, 5)).at.observable.select(k=klim)

    kw_window = dict(edges={'step': mattrs.cellsize.min()}, basis='bessel', los='local')
    #windows = compute_mesh2_covariance_window(selection, **kw_window)
    coords = jnp.logspace(-2., 4., 1024)
    windows = interpolate_window_function(windows, coords)
    cov_fftlog = compute_spectrum2_covariance(windows, theory, delta=delta, flags=['smooth', 'fftlog']).at.observable.select(k=slice(0, None, 5)).at.observable.select(k=klim)
    cov_mocks = Mesh2SpectrumPoles.cov(list(map(read, mock_fn(basename='cutsky')))).at.observable.select(k=klim)
    #pattrs = mattrs.clone(boxsize=2. * 0.25 * mattrs.boxsize)
    #cov_smooth = compute_spectrum2_covariance(pattrs, theory, delta=delta).at.observable.select(k=slice(0, None, 5))at.observable.select(k=klim)

    #cov, cov_smooth, cov_mocks = (cov.at.observable.get(ells=0) for cov in [cov, cov_smooth, cov_mocks])

    if plot:
        ytransform = lambda x, y: x**4 * y
        #ytransform = lambda x, y: x**2 * y
        kw = dict(ytransform=ytransform, offset=np.arange(3))
        #cov.plot(corrcoef=True, show=True)
        cov_fftlog.plot(corrcoef=True, show=True)
        #cov_mocks.plot(corrcoef=True, show=True)
        #fig = None
        fig = cov.plot_diag(**kw, color='C0')
        fig = cov_fftlog.plot_diag(**kw, color='C1', fig=fig)
        #fig = cov_smooth.plot_diag(**kw, color='C2', fig=fig)
        fig = cov_mocks.plot_diag(**kw, color='C3', fig=fig)
        fig.axes[0].get_legend().remove()
        plt.show()


def save_fkp_mocks():
    from jaxpower import FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise
    mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200], meshsize=128)
    pattrs = mattrs.clone(boxsize=1000., meshsize=64)
    theory = get_theory(kmax=np.sqrt(3) * mattrs.knyq.max())
    bin = BinMesh2SpectrumPoles(mattrs, edges=theory.select(k=(0., mattrs.knyq.max())).get(0).edges('k'), ells=(0, 2, 4))
    los = 'local'

    size = int(1e-4 * pattrs.boxsize.prod())

    @jit
    def mock_shotnoise(seed):
        seeds = random.split(seed)
        #mesh = generate_anisotropic_gaussian_mesh(attrs, poles=theory, los=los, seed=seed)
        data = generate_uniform_particles(pattrs, size, seed=seeds[0]).clone(attrs=mattrs)
        #data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
        randoms = generate_uniform_particles(pattrs, 5 * size, seed=seeds[1]).clone(attrs=mattrs)
        fkp = FKPField(data, randoms)
        mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='complex')
        norm = (size / pattrs.boxsize.prod())**2 * pattrs.boxsize.prod()
        #norm = compute_fkp2_normalization(fkp, cellsize=None)
        num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
        return compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=[norm] * len(bin.ells), num_shotnoise=num_shotnoise)

    @jit
    def mock(seed):
        seeds = random.split(seed)
        mesh = generate_anisotropic_gaussian_mesh(mattrs, poles=theory, los=los, seed=seed)
        data = generate_uniform_particles(pattrs, size, seed=seeds[0]).clone(attrs=mattrs)
        data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
        randoms = generate_uniform_particles(pattrs, 5 * size, seed=seeds[1]).clone(attrs=mattrs)
        fkp = FKPField(data, randoms)
        mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='complex')
        norm = (size / pattrs.boxsize.prod())**2 * pattrs.boxsize.prod()
        #norm = compute_fkp2_normalization(fkp, cellsize=None)
        num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
        return compute_mesh2_spectrum(mesh, bin=bin, los=los).clone(norm=[norm] * len(bin.ells), num_shotnoise=num_shotnoise)

    for i, fn in enumerate(mock_fn(basename='fkp_shotnoise')):
        print(i, end=" ", flush=True)
        poles = mock_shotnoise(random.key(i))
        poles.write(fn)

    for i, fn in enumerate(mock_fn(basename='fkp')):
        print(i, end=" ", flush=True)
        poles = mock(random.key(i))
        poles.write(fn)
    print()


def test_fkp2_window(plot=False):

    from cosmoprimo.fiducial import DESI
    mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200], meshsize=128)
    pattrs = mattrs.clone(boxsize=1000., meshsize=64)

    size = int(1e5)
    randoms = generate_uniform_particles(pattrs, size, seed=32).clone(attrs=mattrs)
    windows = compute_fkp2_covariance_window(randoms, edges={'step': mattrs.cellsize.min()}, interlacing=2, resampler='tsc', los='z')

    from jaxpower.utils import plotter

    @plotter
    def plot(self, fig=None):
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            pole = self.get(ell)
            ax.plot(pole.coords('s'), pole.value())
        ax.legend()
        ax.grid(True)
        #ax.set_xscale('log')
        return fig

    if plot:
        plot(windows[0].get((0, 0, 0, 0)), show=True)


def test_fkp2_covariance(plot=False):
    mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200], meshsize=128)
    pattrs = mattrs.clone(boxsize=1000., meshsize=64)

    theory = get_theory(kmax=mattrs.knyq.max())
    size = int(1e-4 * pattrs.boxsize.prod())

    ialpha = 5
    randoms = generate_uniform_particles(pattrs, size * ialpha, seed=32).clone(attrs=mattrs)
    #edges = {'step': attrs.cellsize.min()}
    edges = None
    windows = compute_fkp2_covariance_window(randoms, edges=edges, interlacing=2, resampler='tsc', los='local', alpha=1. / ialpha)

    covs = compute_spectrum2_covariance(windows, theory, delta=0.2)
    klim = (0., mattrs.knyq.max())
    covs = [cov.at.observable.select(k=klim) for cov in covs]

    cov_mocks = Mesh2SpectrumPoles.cov(list(map(read, mock_fn(basename='fkp')))).at.observable.select(k=klim)
    cov_mocks_shotnoise = Mesh2SpectrumPoles.cov(list(map(read, mock_fn(basename='fkp_shotnoise')))).at.observable.select(k=klim)

    ratio_shotnoise = cov_mocks.observable.get(ells=0).values('shotnoise')[0] / cov_mocks_shotnoise.observable.get(ells=0).values('shotnoise')[0]
    print(ratio_shotnoise)
    print([np.isnan(cov.value()).any() for cov in covs])

    if plot:
        ytransform = lambda x, y: x**2 * y
        kw = dict(ytransform=ytransform)
        fig = covs[2].plot_diag(**kw, color='C0')
        cov_mocks_shotnoise.plot_diag(**kw, color='C1', fig=fig, show=True)
    if plot:
        ytransform = lambda x, y: x**2 * y
        kw = dict(ytransform=ytransform)
        cov_ws = covs[1].clone(value=ratio_shotnoise * covs[1].value())
        cov_ss = covs[2].clone(value=ratio_shotnoise**2 * covs[2].value())
        fig = covs[0].clone(value=covs[0].value() + cov_ws.value() + cov_ss.value()).plot_diag(**kw, color='C0')
        cov_mocks.plot_diag(**kw, color='C1', fig=fig, show=True)


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

    from jaxpower.fftlog import SpectrumToCorrelation, CorrelationToSpectrum

    fftlog = SpectrumToCorrelation(k, ell=0, lowring=lowring, minfolds=False).fftlog
    tmp2 = fftlog(pk)[1]
    tmp3 = P2xi(k, l=0, N=len(k), lowring=lowring)(pk)[1]

    H0 = jax.jacfwd(lambda fun: fftlog(fun, ignore_prepostfactor=True)[1])(jnp.zeros_like(k))


    class Correlation2SpectrumK(object):

        def __init__(self, k, ells):
            from jaxpower.fftlog import SpectrumToCorrelation
            fftlog = SpectrumToCorrelation(k, ell=ells[0], lowring=False, minfolds=False).fftlog
            self._H = jax.jacfwd(lambda fun: fftlog(fun, extrap=False, ignore_prepostfactor=True)[1])(jnp.zeros_like(k))
            self._fftlog = SpectrumToCorrelation(k, ell=ells[1], lowring=False, minfolds=False).fftlog
            dlnk = jnp.diff(jnp.log(k)).mean()
            self._postfactor = 2 * np.pi**2 / dlnk / (k[..., None] * k)**1.5
            self.k = k
            self.s = fftlog.y

        def __call__(self, fun):
            fun = self._H * fun
            _, fun = self._fftlog(fun, extrap=False, ignore_prepostfactor=True)
            return self.k, self._postfactor * fun


    class Correlation2Spectrum(object):

        def __init__(self, s, ells):
            from jaxpower.fftlog import CorrelationToSpectrum
            fftlog = CorrelationToSpectrum(s, ell=ells[0], lowring=False, minfolds=False).fftlog
            self._H = jax.jacfwd(lambda fun: fftlog(fun, extrap=False, ignore_prepostfactor=True)[1])(jnp.zeros_like(s))
            self._fftlog = CorrelationToSpectrum(s, ell=ells[1], lowring=False, minfolds=False).fftlog
            k = self._fftlog.y
            dlnk = jnp.diff(jnp.log(k)).mean()
            self._postfactor = 2 * np.pi**2 / dlnk / (k[..., None] * k)**1.5
            self.k = k
            self.s = s

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
                #fftlog = Correlation2SpectrumK(k, (l, lp))
                #assert np.allclose(fftlog.s, s)
                fftlog = Correlation2Spectrum(s, (l, lp))
                assert np.allclose(fftlog.k, k)
                _, Q[i, j] = fftlog(Qll[i, j])

        return Q

    Qref = get_Qllkk(k, l_max=2, lp_max=2)
    Q = get_Qllkk2(k, l_max=2, lp_max=2)
    assert np.allclose(Q, Qref)
    #ax = plt.gca()
    #ax.pcolormesh(d[0])
    #plt.show()



if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)
    #export_sympy()
    #save_box_mocks()
    #test_box2_covariance(plot=True)
    #save_cutsky_mocks()
    test_cutsky2_covariance(plot=True)
    #save_fkp_mocks()
    #test_fkp2_window(plot=True)
    #test_fkp2_covariance(plot=True)