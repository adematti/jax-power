from pathlib import Path

import numpy as np

from jaxpower import BinnedStatistic, CovarianceMatrix, WindowMatrix, WindowRotationSpectrum2, setup_logging


dirname = Path('_tests')


def test_rotation_spectrum2(plot=False):
    xo = np.linspace(0., 0.2, 21)
    ells = (0, 2, 4)
    observable = BinnedStatistic(x=[xo] * len(ells), projs=ells, name='observable')
    value = np.eye(3 * xo.size)
    covmatrix = CovarianceMatrix(value=value, observables=observable)

    xt = np.linspace(0., 0.2, 41)
    theory = BinnedStatistic(x=[xo] * len(ells), projs=ells, name='theory')

    def f(xo, xt):
        sigma = 0.02
        delta = (xo - xt) / sigma
        return np.exp(-delta**2)

    value = np.block([[(1. if ello == ellt else 1. / 10) * f(*np.meshgrid(observable.x(projs=ello), theory.x(projs=ellt), indexing='ij')) for ellt in ells] for ello in ells])
    value = value / np.sum(value, axis=-1)[..., None]
    wmatrix = WindowMatrix(observable=observable, theory=theory, value=value)
    if plot:
        wmatrix.plot(show=True)
        #fig = wmatrix.plot_slice(indices=10)
        #wmatrix.plot_slice(indices=10, fig=fig, show=True)

    rotation = WindowRotationSpectrum2(wmatrix=wmatrix, covmatrix=covmatrix)
    rotation.setup()
    print(rotation.loss(rotation.init))
    rotation.fit()

    fn = dirname / 'tmp.npy'
    rotation.save(fn)

    rotation = WindowRotationSpectrum2.load(fn)
    if plot:
        rotation.plot_compactness(show=True)
        rotation.plot_wmatrix_slice(indices=10, show=True)


if __name__ == '__main__':

    setup_logging()
    test_rotation_spectrum2(plot=True)