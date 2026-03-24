import dataclass
import itertools
import operator
import functools
from functools import partial

import numpy as np
from scipy import special
from jax import numpy as jnp

from .mesh import MeshAttrs
from . import utils
from .utils import wigner_3j


prod = partial(functools.reduce, operator.mul)


@dataclass.dataclass
class Integral1D(object):

    x: np.ndarray
    w: np.ndarray


def integration(a=0., b=1., size=5, method='gauleg'):
    """Return weights for Gauss-Legendre integration."""
    if method == 'leggauss':
        nodes, weights = np.polynomial.legendre.leggauss(size)
        nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        weights = 0.5 * (b - a) * weights
    return Integral1D(x=nodes, w=weights)


def get_Sell(ells):
    ell1, ell2, ell3 = ells
    H = wigner_3j(ell1, ell2, ell3, 0, 0, 0)

    if abs(H) < 1e-7:

        def Sell(mu1, mu2, phi2, grid=False):
            return 0.

    else:

        ms = [np.arange(-ell, ell + 1) for ell in ells]
        coeffs = []
        for m1, m2, m3 in itertools.product(*ms):
            gaunt = 1 / H * wigner_3j(ell1, ell2, ell3, m1, m2, m3)
            if abs(gaunt) < 1e-7:
                continue
            coeffs.append((m1, m2, m3, gaunt))

        def get_Ylm(ell, m, mu):
            amp = 1.
            fac = 1
            for n in range(ell - abs(m) + 1, ell + abs(m) + 1): fac *= n  # (ell + |m|)!/(ell - |m|)!
            amp *= np.sqrt(1. / fac)
            toret = amp * special.lpmv(abs(m), ell, mu)
            return toret

        def Sell(mu1, mu2, phi2, grid=False):
            toret = 1.
            for m1, m2, m3, gaunt in coeffs:
                toret *= gaunt
                Ylms = [get_Ylm(ells[0], m1, mu1),
                        get_Ylm(ells[1], m2, mu2),
                        get_Ylm(ells[2], m3, 1.) * (np.cos if (ell1 + ell2) % 2 == 0 else np.sin)(m2 * phi2)]
                if grid:
                    Ylms = np.meshgrid(*Ylms, indexing='ij', sparse=True)
                toret *= prod(Ylms)
            return toret

    return Sell



def compute_spectrum3_covariance(window2, observable, theory=None):

    if isinstance(window2, MeshAttrs):
        mattrs = window2
        volume = mattrs.boxsize.prod()

    cov = [[None for _ in observable.items(level=None)] for _ in observable.items(level=None)]
    integ_mu = integration(-1., 1., size=6)
    integ_phi = integration(0., np.pi, size=4)

    def inverse_nmodes(edges1, edges2):
        edges = [np.maximum(edges1[0], edges2[0]), np.minimum(edges1[1], edges2[1])]
        mask = edges[1] > edges[0]
        invnmodes = 1. / (4. / 3. * np.pi) * mask * (edges[1]**3 - edges[0]**3) / (edges1[1]**3 - edges1[0]**3) / (edges2[1]**3 - edges2[0]**3)
        invnmodes *= (2. * np.pi)**3 / volume
        return invnmodes

    def d_inverse_nmodes(edges, k):
        mask = (k >= edges[0]) & (k <= edges[1])
        invnmodes = 1. / (4. / 3. * np.pi) * mask / (edges[1]**3 - edges[0]**3)
        invnmodes *= (2. * np.pi)**3 / volume
        return invnmodes

    for i1, (label1, observable1) in enumerate(observable.items(level=None)):
        for i2, (label2, observable2) in enumerate(observable.items(level=None)):
            if i2 < i1: continue
            nfields1, nfields2 = len(label1['fields']), len(label2['fields'])
            ell1, ell2 = label1['ells'], label2['ells']
            edges1, edges2 = [obs.edges('k').T for obs in [observable1, observable2]]
            center = 'mid_if_edges_and_nan'
            k1, k2 = [obs.coords('k', center=center).T for obs in [observable1, observable2]]
            th = theory(label1=label1, label2=label2)
            if nfields1 == nfields2 == 2:
                leg1, leg2 = utils.get_legendre(ell1), utils.get_legendre(ell2)
                invnmodes = inverse_nmodes(edges1, edges2)
                block = 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * invnmodes
                block *= jnp.sum(th(k1, integ_mu.x) * th(k2, integ_mu.x) * leg1(integ_mu.x) * leg2(integ_mu.x) * integ_mu.w, axis=-1) / 2.
            elif nfields1 == 1 and nfields2 == 2:
                N = (2 * ell2[0] + 1) * (2 * ell2[1] + 1) * (2 * ell2[2] + 1)
                H = wigner_3j(*ell2, 0, 0, 0)
                Sell = get_Sell(ell2)
                block = 2 * (2 * ell1 + 1) * N * H**2 * Sell(integ_mu.x, integ_mu.x, integ_phi.x, grid=True)
                leg = utils.get_legendre(ell1)
                block *= (d_inverse_nmodes(edges1, k1[0]) * leg(integ_mu.x))
