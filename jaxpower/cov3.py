import dataclasses
import itertools
import operator
import functools
from functools import partial

import numpy as np
from scipy import special
import jax
from jax import numpy as jnp

from .mesh import MeshAttrs
from . import utils
from .utils import wigner_3j, get_legendre


prod = partial(functools.reduce, operator.mul)


@dataclasses.dataclass
class Integral1D:
    x: np.ndarray
    w: np.ndarray


class IntegralND:

    def __init__(self, **kwargs):
        self.integ = dict(kwargs)

    def x(self, names=None, sparse=True):
        """Points at which to evaluate the integrand. If `names` is None, return all points as a list. Otherwise, return the points corresponding to the given names."""
        all_names = list(self.integ.keys())
        if names is None:
            names = all_names
        grids = np.meshgrid(*[self.integ[name].x for name in all_names], sparse=sparse, indexing='ij')
        if np.ndim(names) == 0:
            return grids[all_names.index(names)]
        return [grids[all_names.index(name)] for name in names]

    @property
    def w(self):
        """Weights for the integral, i.e. product of the weights of each dimension."""
        w = [integ.w for integ in self.integ.values()]
        return prod(np.meshgrid(*w, sparse=True, indexing='ij'))

    @property
    def ndim(self):
        """Dimension of the integral, i.e. number of integration variables."""
        return len(self.integ)

    def __call__(self, integrand):
        """Integrate the given integrand over the dimensions of this :class:`IntegralND` instance."""
        axes = tuple(range(integrand.ndim - self.ndim, integrand.ndim))
        return np.sum(integrand * self.w, axis=axes)


def integration(a=0., b=1., size=5, method='leggauss'):
    nodes, weights = np.polynomial.legendre.leggauss(size)
    nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights
    return Integral1D(x=nodes, w=weights)


def unitvec(mu, phi):
    s = np.sqrt(np.clip(1. - mu**2, 0., None))
    return np.stack([s * np.cos(phi), s * np.sin(phi), mu], axis=-1)


def get_kvec1(knorm, mu):
    khat = unitvec(mu, np.zeros_like(mu))
    kvec = knorm[..., None] * khat
    return knorm, khat, kvec


def get_kvec3(k1norm, k2norm, mu1, mu2, phi2):
    k1hat = unitvec(mu1, np.zeros_like(mu1))
    k2hat = unitvec(mu2, phi2)
    k1vec = k1norm[..., None] * k1hat
    k2vec = k2norm[..., None] * k2hat
    k3vec = -k1vec - k2vec
    k3norm = np.sqrt(np.sum(k3vec**2, axis=-1))
    k3hat = k3vec / k3norm[..., None]
    return (k1norm, k2norm, k3norm), (k1hat, k2hat, k3hat), (k1vec, k2vec, k3vec)


def get_kvec5(k1norm, k2norm, k2pnorm, mu1, mu2, phi2, mu2p, phi2p):
    k1hat = unitvec(mu1, np.zeros_like(mu1))
    k2hat = unitvec(mu2, phi2)
    k2phat = unitvec(mu2p, phi2p)

    k1vec = k1norm[..., None] * k1hat
    k2vec = k2norm[..., None] * k2hat
    k2pvec = k2pnorm[..., None] * k2phat

    k3vec = -k1vec - k2vec
    k3pvec = -k1vec - k2pvec

    k3norm = np.sqrt(np.sum(k3vec**2, axis=-1))
    k3pnorm = np.sqrt(np.sum(k3pvec**2, axis=-1))

    k3hat = k3vec / k3norm[..., None]
    k3phat = k3pvec / k3pnorm[..., None]

    return (k1norm, k2norm, k3norm, k2pnorm, k3pnorm), \
           (k1hat, k2hat, k3hat, k2phat, k3phat), \
           (k1vec, k2vec, k3vec, k2pvec, k3pvec)


def get_S(ells):
    ell1, ell2, ell3 = ells
    H = wigner_3j(ell1, ell2, ell3, 0, 0, 0)

    if abs(H) < 1e-12:
        return lambda *args: 0.

    ms = [np.arange(-ell, ell + 1) for ell in ells]
    coeffs = []
    for m1, m2, m3 in itertools.product(*ms):
        gaunt = wigner_3j(ell1, ell2, ell3, m1, m2, m3) / H
        if abs(gaunt) > 1e-12:
            coeffs.append((m1, m2, m3, gaunt))

    def get_Ylm(ell, m, xhat):
        mu = xhat[..., 2]
        phi = np.arctan2(xhat[..., 1], xhat[..., 0])
        fac = special.factorial(ell - abs(m), exact=False) / special.factorial(ell + abs(m), exact=False)
        amp = np.sqrt(fac)
        return amp * special.lpmv(abs(m), ell, mu) * np.exp(1j * m * phi)

    def Sell(*xhats):
        out = 0.
        for m1, m2, m3, gaunt in coeffs:
            out = out + gaunt * prod(
                get_Ylm(ell, m, xhat)
                for ell, m, xhat in zip(ells, (m1, m2, m3), xhats)
            )
        return out.real if ((ell1 + ell2 + ell3) % 2 == 0) else out.imag

    return Sell


def compute_spectrum3_covariance(window2, observable, theory=None):

    if isinstance(window2, MeshAttrs):
        mattrs = window2
        volume = mattrs.boxsize.prod()
        window2_callable = None
    else:
        volume = None
        window2_callable = window2

    cov = [[None for _ in observable.items(level=None)] for _ in observable.items(level=None)]
    integ_mu = integration(-1., 1., size=6)
    integ_phi = integration(0., 2. * np.pi, size=6)

    def inverse_nmodes(edges1, edges2):
        edges = [np.maximum(edges1[0], edges2[0]), np.minimum(edges1[1], edges2[1])]
        mask = edges[1] > edges[0]
        invnmodes = 1. / (4. / 3. * np.pi) * mask * (edges[1]**3 - edges[0]**3)
        invnmodes /= (edges1[1]**3 - edges1[0]**3) * (edges2[1]**3 - edges2[0]**3)
        invnmodes *= (2. * np.pi)**3 / volume
        return invnmodes

    def d_inverse_nmodes(edges, k):
        mask = (k >= edges[0]) & (k <= edges[1])
        invnmodes = 1. / (4. / 3. * np.pi) * mask / (edges[1]**3 - edges[0]**3)
        invnmodes *= (2. * np.pi)**3 / volume
        return invnmodes

    def W2(edgesp, k, kvec=None, kpvec=None):
        if window2_callable is None:
            return d_inverse_nmodes(edgesp, k)
        return window2_callable(kvec, kpvec)

    def W3norm(kvec, kpvec):
        if window2_callable is None:
            return volume
        return window2_callable(kvec, kpvec)

    def get_N(ell1, ell2, ell3):
        return (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1)

    def get_H(ell1, ell2, ell3):
        return wigner_3j(ell1, ell2, ell3, 0, 0, 0)

    def get_theory(fields):
        return None if theory is None else theory(tuple(fields))

    for i1, (label1, observable1) in enumerate(observable.items(level=None)):
        for i2, (label2, observable2) in enumerate(observable.items(level=None)):
            if i2 < i1:
                continue

            fields1, fields2 = tuple(label1['fields']), tuple(label2['fields'])
            nfields1, nfields2 = len(fields1), len(fields2)
            ell, ellp = label1['ells'], label2['ells']
            edges, edgesp = [obs.edges('k').T for obs in [observable1, observable2]]
            center = 'mid_if_edges_and_nan'
            coords, coordsp = [obs.coords('k', center=center).T for obs in [observable1, observable2]]

            # ------------------------------------------------------------------
            # PP block
            # ------------------------------------------------------------------
            if nfields1 == 2 and nfields2 == 2:
                a, b = fields1
                ap, bp = fields2
                P_ad, P_bc, P_ac, P_bd = get_theory((a, bp)), get_theory((b, ap)), get_theory((a, ap)), get_theory((b, bp))
                T_abapbp = get_theory((a, b, ap, bp))

                integ = IntegralND(mu=integ_mu)
                mu = integ.x('mu')
                leg, legp = get_legendre(ell), get_legendre(ellp)

                (knorm, khat, kvec) = get_kvec1(coords, mu)
                (kpnorm, kphat, kpvec) = get_kvec1(coordsp, mu)

                if window2_callable is None:
                    invnmodes = inverse_nmodes(edges, edgesp)
                    block = (2 * ell + 1) * (2 * ellp + 1) * invnmodes * integ((P_ad(kvec) * P_bc(kpvec) + P_ac(kvec) * P_bd(kpvec)) * leg(mu) * legp(mu)) / 2.
                else:
                    term1 = W2(edgesp, knorm, kvec, kpvec) * P_ad(kvec) * P_bc(kpvec)
                    term2 = W2(edgesp, knorm, kvec, -kpvec) * P_ac(kvec) * P_bd(kpvec)
                    block = (2 * ell + 1) * (2 * ellp + 1) * integ((term1 + term2) * leg(mu) * legp(mu)) / 2.

                if T_abapbp is not None:
                    block = block + (2 * ell + 1) * (2 * ellp + 1) / (volume if volume is not None else 1.) * integ(T_abapbp(kvec, -kvec, kpvec, -kpvec) * leg(mu) * legp(mu)) / 2.

            # ------------------------------------------------------------------
            # PB block
            # ------------------------------------------------------------------
            elif nfields1 == 2 and nfields2 == 3:
                a, b = fields1
                c, d, e = fields2
                P_ac, P_ad, P_ae, P_bc, P_bd, P_be = get_theory((a, c)), get_theory((a, d)), get_theory((a, e)), get_theory((b, c)), get_theory((b, d)), get_theory((b, e))
                B_bde, B_bce, B_bcd, B_ade, B_ace, B_acd = get_theory((b, d, e)), get_theory((b, c, e)), get_theory((b, c, d)), get_theory((a, d, e)), get_theory((a, c, e)), get_theory((a, c, d))

                integ = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                mu1, mu2, phi2 = integ.x(['mu1', 'mu2', 'phi2'])

                leg = get_legendre(ell)
                Sp = get_S(ellp)

                (k1norm, k2norm, k3norm), (k1hat, k2hat, k3hat), (k1vec, k2vec, k3vec) = get_kvec3(coordsp[0], coordsp[1], mu1, mu2, phi2)

                pref = (2 * ell + 1) * get_N(*ellp) * get_H(*ellp)**2

                term1 = W2(edges, k1norm, k1vec, k1vec) * leg(mu1) * Sp(k1hat, k2hat, k3hat) * P_ac(k1vec) * B_bde(k1vec, k2vec, k3vec)
                term2 = W2(edges, k2norm, k2vec, k2vec) * leg(mu1) * Sp(k2hat, k1hat, k3hat) * P_ad(k2vec) * B_bce(k2vec, k1vec, k3vec)
                term3 = W2(edges, k3norm, k3vec, k3vec) * leg(mu1) * Sp(k3hat, k1hat, k2hat) * P_ae(k3vec) * B_bcd(k3vec, k1vec, k2vec)
                term4 = W2(edges, k1norm, k1vec, k1vec) * leg(mu1) * Sp(k1hat, k2hat, k3hat) * P_bc(k1vec) * B_ade(k1vec, k2vec, k3vec)
                term5 = W2(edges, k2norm, k2vec, k2vec) * leg(mu1) * Sp(k2hat, k1hat, k3hat) * P_bd(k2vec) * B_ace(k2vec, k1vec, k3vec)
                term6 = W2(edges, k3norm, k3vec, k3vec) * leg(mu1) * Sp(k3hat, k1hat, k2hat) * P_be(k3vec) * B_acd(k3vec, k1vec, k2vec)

                block = pref * integ(term1 + term2 + term3 + term4 + term5 + term6)

            # ------------------------------------------------------------------
            # BP block
            # ------------------------------------------------------------------
            elif nfields1 == 3 and nfields2 == 2:
                a, b, c = fields1
                dp, ep = fields2
                P_ad, P_bd, P_cd, P_ae, P_be, P_ce = get_theory((a, dp)), get_theory((b, dp)), get_theory((c, dp)), get_theory((a, ep)), get_theory((b, ep)), get_theory((c, ep))
                B_bce, B_ace, B_abe, B_bcd, B_acd, B_abd = get_theory((b, c, ep)), get_theory((a, c, ep)), get_theory((a, b, ep)), get_theory((b, c, dp)), get_theory((a, c, dp)), get_theory((a, b, dp))

                integ = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                mu1, mu2, phi2 = integ.x(['mu1', 'mu2', 'phi2'])

                S = get_S(ell)
                legp = get_legendre(ellp)

                (k1norm, k2norm, k3norm), (k1hat, k2hat, k3hat), (k1vec, k2vec, k3vec) = get_kvec3(coords[0], coords[1], mu1, mu2, phi2)

                pref = (2 * ellp + 1) * get_N(*ell) * get_H(*ell)**2

                term1 = W2(edgesp, k1norm, k1vec, k1vec) * legp(mu1) * S(k1hat, k2hat, k3hat) * P_ad(k1vec) * B_bce(k1vec, k2vec, k3vec)
                term2 = W2(edgesp, k2norm, k2vec, k2vec) * legp(mu1) * S(k2hat, k1hat, k3hat) * P_bd(k2vec) * B_ace(k2vec, k1vec, k3vec)
                term3 = W2(edgesp, k3norm, k3vec, k3vec) * legp(mu1) * S(k3hat, k1hat, k2hat) * P_cd(k3vec) * B_abe(k3vec, k1vec, k2vec)
                term4 = W2(edgesp, k1norm, k1vec, k1vec) * legp(mu1) * S(k1hat, k2hat, k3hat) * P_ae(k1vec) * B_bcd(k1vec, k2vec, k3vec)
                term5 = W2(edgesp, k2norm, k2vec, k2vec) * legp(mu1) * S(k2hat, k1hat, k3hat) * P_be(k2vec) * B_acd(k2vec, k1vec, k3vec)
                term6 = W2(edgesp, k3norm, k3vec, k3vec) * legp(mu1) * S(k3hat, k1hat, k2hat) * P_ce(k3vec) * B_abd(k3vec, k1vec, k2vec)

                block = pref * integ(term1 + term2 + term3 + term4 + term5 + term6)

            # ------------------------------------------------------------------
            # BB block
            # ------------------------------------------------------------------
            elif nfields1 == 3 and nfields2 == 3:

                a, b, c = fields1
                ap, bp, cp = fields2

                S, Sp = get_S(ell), get_S(ellp)
                M = get_N(*ell) * get_N(*ellp) * get_H(*ell)**2 * get_H(*ellp)**2

                # (1) Gaussian PPP term
                integ = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)
                mu1, mu2, phi2 = integ.x(['mu1', 'mu2', 'phi2'])

                (k1norm, k2norm, k3norm), (k1hat, k2hat, k3hat), (k1vec, k2vec, k3vec) = get_kvec3(coords[0], coords[1], mu1, mu2, phi2)

                P_aap, P_abp, P_acp = get_theory((a, ap)), get_theory((a, bp)), get_theory((a, cp))
                P_bap, P_bbp, P_bcp = get_theory((b, ap)), get_theory((b, bp)), get_theory((b, cp))
                P_cap, P_cbp, P_ccp = get_theory((c, ap)), get_theory((c, bp)), get_theory((c, cp))

                term1 = W3norm(k3vec, k3vec) * S(k1hat, k2hat, k3hat) * Sp(k1hat, k2hat, k3hat) * W2(edgesp[0], k1norm, k1vec, k1vec) * W2(edgesp[1], k2norm, k2vec, k2vec) * P_aap(k1vec) * P_bbp(k2vec) * P_ccp(k3vec)
                term2 = W3norm(k3vec, k3vec) * S(k1hat, k2hat, k3hat) * Sp(k2hat, k1hat, k3hat) * W2(edgesp[0], k2norm, k1vec, k2vec) * W2(edgesp[1], k1norm, k2vec, k1vec) * P_abp(k1vec) * P_bap(k2vec) * P_ccp(k3vec)
                term3 = W3norm(k3vec, k2vec) * S(k1hat, k2hat, k3hat) * Sp(k1hat, k3hat, k2hat) * W2(edgesp[0], k1norm, k1vec, k1vec) * W2(edgesp[1], k3norm, k2vec, k3vec) * P_aap(k1vec) * P_bcp(k2vec) * P_cbp(k3vec)
                term4 = W3norm(k3vec, k1vec) * S(k1hat, k2hat, k3hat) * Sp(k3hat, k1hat, k2hat) * W2(edgesp[0], k3norm, k1vec, k3vec) * W2(edgesp[1], k1norm, k2vec, k1vec) * P_abp(k1vec) * P_bcp(k2vec) * P_cap(k3vec)
                term5 = W3norm(k3vec, k2vec) * S(k1hat, k2hat, k3hat) * Sp(k2hat, k3hat, k1hat) * W2(edgesp[0], k2norm, k1vec, k2vec) * W2(edgesp[1], k3norm, k2vec, k3vec) * P_acp(k1vec) * P_bap(k2vec) * P_cbp(k3vec)
                term6 = W3norm(k3vec, k1vec) * S(k1hat, k2hat, k3hat) * Sp(k3hat, k2hat, k1hat) * W2(edgesp[0], k3norm, k1vec, k3vec) * W2(edgesp[1], k2norm, k2vec, k2vec) * P_acp(k1vec) * P_bbp(k2vec) * P_cap(k3vec)

                block_PPP = M / (8. * np.pi) * integ(term1 + term2 + term3 + term4 + term5 + term6)

                # (2) Connected BB term
                integ = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi, mu2p=integ_mu, phi2p=integ_phi)
                mu1, mu2, phi2, mu2p, phi2p = integ.x(['mu1', 'mu2', 'phi2', 'mu2p', 'phi2p'])

                (k1norm, k2norm, k3norm, k2pnorm, k3pnorm), (k1hat, k2hat, k3hat, k2phat, k3phat), (k1vec, k2vec, k3vec, k2pvec, k3pvec) = get_kvec5(coords[0], coords[1], coordsp[1], mu1, mu2, phi2, mu2p, phi2p)

                k1pnorm, k1pvec, k1phat = coordsp[0], -k1vec, -k1hat
                kalphavec = k1vec + k2vec - k2pvec
                kalphanorm = np.sqrt(np.sum(kalphavec**2, axis=-1))
                kalphahat = kalphavec / kalphanorm[..., None]

                B_bcap, B_acap, B_abap = get_theory((b, c, ap)), get_theory((a, c, ap)), get_theory((a, b, ap))
                B_bcbp, B_acbp, B_abbp = get_theory((b, c, bp)), get_theory((a, c, bp)), get_theory((a, b, bp))
                B_bccp, B_accp, B_abcp = get_theory((b, c, cp)), get_theory((a, c, cp)), get_theory((a, b, cp))
                B_bp_cpa, B_bp_cpb, B_bp_cpc = get_theory((bp, cp, a)), get_theory((bp, cp, b)), get_theory((bp, cp, c))
                B_ap_cpa, B_ap_cpb, B_ap_cpc = get_theory((ap, cp, a)), get_theory((ap, cp, b)), get_theory((ap, cp, c))
                B_ap_bpa, B_ap_bpb, B_ap_bpc = get_theory((ap, bp, a)), get_theory((ap, bp, b)), get_theory((ap, bp, c))
                B_c_bpcp = get_theory((c, bp, cp))

                pref = M

                term1 = S(k1hat, k2hat, k3hat) * Sp(-k1hat, k2phat, k3phat) * W2(edgesp[0], k1norm, k1vec, -k1vec) * B_bcap(k2vec, k3vec, -k1vec) * B_bp_cpa(k2pvec, k3pvec, k1vec)
                term2 = S(k1hat, k2hat, k3hat) * Sp(-k2hat, k2phat, k3phat) * W2(edgesp[0], k2norm, k2vec, -k2vec) * B_acap(k1vec, k3vec, -k2vec) * B_bp_cpb(k2pvec, k3pvec, k2vec)
                term3 = S(k1hat, k2hat, k3hat) * Sp(-k3hat, k2phat, k3phat) * W2(edgesp[0], k3norm, k3vec, -k3vec) * B_abap(k1vec, k2vec, -k3vec) * B_bp_cpc(k2pvec, k3pvec, k3vec)
                term4 = S(k1hat, k2hat, k3hat) * Sp(k1phat, -k1hat, k3phat) * W2(edgesp[1], k1norm, k1vec, -k1vec) * B_bcbp(k2vec, k3vec, -k1vec) * B_ap_cpa(k1pvec, k3pvec, k1vec)
                term5 = S(k1hat, k2hat, k3hat) * Sp(k1phat, -k2hat, k3phat) * W2(edgesp[1], k2norm, k2vec, -k2vec) * B_acbp(k1vec, k3vec, -k2vec) * B_ap_cpb(k1pvec, k3pvec, k2vec)
                term6 = S(k1hat, k2hat, k3hat) * Sp(k1phat, -k3hat, k3phat) * W2(edgesp[1], k3norm, k3vec, -k3vec) * B_abbp(k1vec, k2vec, -k3vec) * B_ap_cpc(k1pvec, k3pvec, k3vec)
                term7 = S(-k3phat, k2hat, k3hat) * Sp(k1phat, k2phat, k3phat) * W2(edgesp[0], k3pnorm, k3pvec, k1pvec) * B_bccp(k2vec, k3vec, -k3pvec) * B_ap_bpa(k1pvec, k2pvec, k1vec)
                term8 = S(k1hat, -k3phat, k3hat) * Sp(k1phat, k2phat, k3phat) * W2(edgesp[1], k3pnorm, k3pvec, k2pvec) * B_accp(k1vec, k3vec, -k3pvec) * B_ap_bpb(k1pvec, k2pvec, k2vec)
                term9 = S(k1hat, k2hat, -kalphahat) * Sp(-kalphahat, k2phat, k3phat) * W2(edgesp[0], kalphanorm, kalphavec, -kalphavec) * B_abap(k1vec, k2vec, -kalphavec) * B_c_bpcp(-kalphavec, k2pvec, k3pvec)

                block_BB = pref * integ(term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)

                # (3) P × T term
                T_bc_bpcp, T_ac_bpcp, T_ab_bpcp = get_theory((b, c, bp, cp)), get_theory((a, c, bp, cp)), get_theory((a, b, bp, cp))
                T_bc_apcp, T_ac_apcp, T_ab_apcp = get_theory((b, c, ap, cp)), get_theory((a, c, ap, cp)), get_theory((a, b, ap, cp))
                T_bc_apbp, T_ac_apbp, T_ab_apbp = get_theory((b, c, ap, bp)), get_theory((a, c, ap, bp)), get_theory((a, b, ap, bp))

                integ = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi, mu2p=integ_mu, phi2p=integ_phi)
                mu1, mu2, phi2, mu2p, phi2p = integ.x(['mu1', 'mu2', 'phi2', 'mu2p', 'phi2p'])

                (k1norm, k2norm, k3norm, k2pnorm, k3pnorm), (k1hat, k2hat, k3hat, k2phat, k3phat), (k1vec, k2vec, k3vec, k2pvec, k3pvec) = get_kvec5(coords[0], coords[1], coordsp[1], mu1, mu2, phi2, mu2p, phi2p)

                k1pnorm, k1pvec, k1phat = coordsp[0], -k1vec, -k1hat
                kbetavec = k1vec + k2vec + k2pvec
                kbetanorm = np.sqrt(np.sum(kbetavec**2, axis=-1))
                kbetahat = kbetavec / kbetanorm[..., None]

                term1 = S(k1hat, k2hat, k3hat) * Sp(-k1hat, k2phat, k3phat) * W2(edgesp[0], k1norm, k1vec, -k1vec) * get_theory((a, ap))(k1vec) * T_bc_bpcp(k2vec, k3vec, k2pvec, k3pvec)
                term2 = S(k1hat, k2hat, k3hat) * Sp(-k2hat, k2phat, k3phat) * W2(edgesp[0], k2norm, k2vec, -k2vec) * get_theory((b, ap))(k2vec) * T_ac_bpcp(k1vec, k3vec, k2pvec, k3pvec)
                term3 = S(k1hat, k2hat, k3hat) * Sp(-k3hat, k2phat, k3phat) * W2(edgesp[0], k3norm, k3vec, -k3vec) * get_theory((c, ap))(k3vec) * T_ab_bpcp(k1vec, k2vec, k2pvec, k3pvec)
                term4 = S(k1hat, k2hat, k3hat) * Sp(k1phat, -k1hat, k3phat) * W2(edgesp[1], k1norm, k1vec, -k1vec) * get_theory((a, bp))(k1vec) * T_bc_apcp(k2vec, k3vec, k1pvec, k3pvec)
                term5 = S(k1hat, k2hat, k3hat) * Sp(k1phat, -k2hat, k3phat) * W2(edgesp[1], k2norm, k2vec, -k2vec) * get_theory((b, bp))(k2vec) * T_ac_apcp(k1vec, k3vec, k1pvec, k3pvec)
                term6 = S(k1hat, k2hat, k3hat) * Sp(k1phat, -k3hat, k3phat) * W2(edgesp[1], k3norm, k3vec, -k3vec) * get_theory((c, bp))(k3vec) * T_ab_apcp(k1vec, k2vec, k1pvec, k3pvec)
                term7 = S(-k3phat, k2hat, k3hat) * Sp(k1phat, k2phat, k3phat) * W2(edgesp[0], k3pnorm, k3pvec, k1pvec) * get_theory((c, cp))(k3pvec) * T_bc_apbp(k2vec, k3vec, k1pvec, k2pvec)
                term8 = S(k1hat, -k3phat, k3hat) * Sp(k1phat, k2phat, k3phat) * W2(edgesp[1], k3pnorm, k3pvec, k2pvec) * get_theory((c, cp))(k3pvec) * T_ac_apbp(k1vec, k3vec, k1pvec, k2pvec)
                term9 = S(k1hat, k2hat, k3hat) * Sp(-kbetahat, k2phat, k3phat) * W2(edgesp[0], kbetanorm, kbetavec, -kbetavec) * get_theory((c, ap))(k3vec) * T_ab_apbp(k1vec, k2vec, -kbetavec, k2pvec)

                block_PT = M * integ(term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)
                block = block_PPP + block_BB + block_PT

            else:
                continue

            cov[i1][i2] = block
            cov[i2][i1] = block.T

    return cov
