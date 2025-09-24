from pathlib import Path

import numpy as np

from jaxpower import Mesh2SpectrumPole, Mesh2SpectrumPoles, CovarianceMatrix, WindowMatrix, WindowRotationSpectrum2, setup_logging


dirname = Path('_tests')


def test_rotation_spectrum2(plot=False):
    ko_edges = np.linspace(0., 0.2, 21)
    ko_edges = np.column_stack([ko_edges[:-1], ko_edges[1:]])
    ko = np.mean(ko_edges, axis=-1)
    ells = (0, 2, 4)

    observable = Mesh2SpectrumPoles([Mesh2SpectrumPole(k=ko, k_edges=ko_edges, num_raw=np.zeros_like(ko), ell=ell) for ell in ells])
    value = np.eye(3 * ko.size)
    covmatrix = CovarianceMatrix(value=value, observable=observable)

    kt_edges = np.linspace(0., 0.2, 41)
    kt_edges = np.column_stack([kt_edges[:-1], kt_edges[1:]])
    kt = np.mean(kt_edges, axis=-1)
    theory = Mesh2SpectrumPoles([Mesh2SpectrumPole(k=kt, k_edges=kt_edges, num_raw=np.zeros_like(kt), ell=ell) for ell in ells])

    def f(xo, xt):
        sigma = 0.02
        delta = (xo - xt) / sigma
        return np.exp(-delta**2)

    value = np.block([[(1. if ello == ellt else 1. / 10) * f(*np.meshgrid(observable.get(ello).coords('k'), theory.get(ellt).coords('k'), indexing='ij')) for ellt in ells] for ello in ells])
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