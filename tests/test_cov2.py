from pathlib import Path

import numpy as np

from jaxpower import BinnedStatistic, CovarianceMatrix, MeshAttrs, compute_fkp2_window, compute_fkp2_spectrum_covariance, generate_uniform_particles


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


def test_fkp2_covariance(plot=False):

    from jaxpower.utils import export_legendre_product

    #print(export_legendre_product(ellmax=8, n=3))

    from cosmoprimo.fiducial import DESI

    attrs = MeshAttrs(boxsize=500., boxcenter=[1300., 0., 0.], meshsize=64)

    cosmo = DESI(engine='eisenstein_hu')
    pk = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)
    ellsin = (0, 2, 4)
    edgesin = np.arange(0., attrs.knyq.max(), 0.005)
    edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])
    kin = (edgesin[..., 0] + edgesin[..., 1]) / 2.
    f, b = 0.8, 1.5
    beta = f / b
    poles = np.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                        0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                        8. / 35 * beta ** 2 * pk(kin)])
    poles = BinnedStatistic(x=[kin] * len(ellsin), edges=[edgesin] * len(ellsin), value=list(poles), projs=ellsin)

    size = int(1e5)
    fkp = generate_uniform_particles(attrs, size, seed=32)
    windows = compute_fkp2_window(fkp, edges={'step': attrs.cellsize.min()}, interlacing=2, resampler='tsc')
    covs = compute_fkp2_spectrum_covariance(windows, poles, delta=0.03)
    cov = sum(covs)

    if plot:
        cov.plot(corrcoef=True, show=True)


if __name__ == '__main__':

    test_covmatrix(plot=True)
    #test_fkp2_covariance(plot=True)