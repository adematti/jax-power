import numpy as np
from jaxpower.types import Particle2CorrelationPole, Particle2CorrelationPoles, Particle3CorrelationPole, Particle3CorrelationPoles


def test_types():

    def get_correlation2(seed=42):
        ells = [0, 2, 4]
        rng = np.random.RandomState(seed=seed)
        correlation = []
        for ell in ells:
            s_edges = np.linspace(0., 200, 41)
            s_edges = np.column_stack([s_edges[:-1], s_edges[1:]])
            s = np.mean(s_edges, axis=-1)
            correlation.append(Particle2CorrelationPole(s=s, s_edges=s_edges, num_raw=rng.uniform(size=s.size)))
        return Particle2CorrelationPoles(correlation, ells=ells)

    def get_correlation3(seed=42, basis='sugiyama', full=False):
        rng = np.random.RandomState(seed=seed)

        assert basis in ['sugiyama', 'sugiyama-diagonal']
        if 'scoccimarro' in basis:
            ndim = 3
            ells = [0, 2]
        else:
            ndim = 2
            ells = [(0, 0, 0), (2, 0, 2)]

        correlation = []
        for ell in ells:
            uedges = np.linspace(0., 100, 41)
            uedges = [np.column_stack([uedges[:-1], uedges[1:]])] * ndim
            s = [np.mean(uedge, axis=-1) for uedge in uedges]
            nmodes1d = [np.ones(uedge.shape[0], dtype='i') for uedge in uedges]

            def _product(array):
                if not isinstance(array, (tuple, list)):
                    array = [array] * ndim
                if 'diagonal' in basis or 'equilateral' in basis:
                    grid = [np.array(array[0])] * ndim
                else:
                    grid = np.meshgrid(*array, sparse=False, indexing='ij')
                return np.column_stack([tmp.ravel() for tmp in grid])

            def get_order_mask(edges):
                xmid = _product([np.mean(edge, axis=-1) for edge in edges])
                mask = True
                for i in range(xmid.shape[1] - 1): mask &= xmid[:, i] <= xmid[:, i + 1]  # select k1 <= k2 <= k3...
                return mask

            mask = get_order_mask(uedges)
            if full: mask = Ellipsis
            # of shape (nbins, ndim, 2)
            s_edges = np.concatenate([_product([edge[..., 0] for edge in uedges])[..., None], _product([edge[..., 1] for edge in uedges])[..., None]], axis=-1)[mask]
            s = _product(s)[mask]
            nmodes = np.prod(_product(nmodes1d)[mask], axis=-1)
            k = np.mean(s_edges, axis=-1)
            correlation.append(Particle3CorrelationPole(s=s, s_edges=s_edges, nmodes=nmodes, num_raw=rng.uniform(size=k.shape[0])))
        return Particle3CorrelationPoles(correlation, ells=ells)

    correlation = get_correlation2()
    spectrum = correlation.to_spectrum(k=np.linspace(0.001, 0.2, 20))
    assert spectrum.size == 3 * 20

    correlation = get_correlation3()
    spectrum = correlation.to_spectrum(k=[np.linspace(0.001, 0.2, 20)] * 2)
    assert spectrum.get(ells=(0, 0, 0)).shape == (20, 20)
    spectrum = correlation.to_spectrum(k=np.column_stack([np.linspace(0.001, 0.2, 20)] * 2))
    assert spectrum.get(ells=(0, 0, 0)).shape == (20,)


if __name__ == '__main__':

    test_types()