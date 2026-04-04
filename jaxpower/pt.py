from jax import numpy as jnp


def spectrum3(k1vec, k2vec, fields, f, sigma2v, Sigma2, deltaSigma2, bias_params, pk_callable, pknw_callable,
               damping='lor', Pshot_const=0.):
    """
    JAX-friendly cross-bispectrum model B^{abc}(k1, k2, k3), with k3 = -k1 - k2.

    Parameters
    ----------
    k1vec, k2vec : array_like[..., 3]
        Wavevectors of the first two bispectrum legs.
    fields : tuple
        Tuple (a, b, c) of field identifiers.
    f, sigma2v, Sigma2, deltaSigma2 : float
        RSD / IR / FoG parameters.
    bias_params : dict
        Mapping field -> parameters.
        Each entry can be either a dict with keys
        ('b1', 'b2', 'bs', 'c1', 'c2', 'Bshot', 'Pshot', 'X_FoG_b')
        or a tuple/list in that order.
    pk_callable : callable
        Callable returning the linear power spectrum P(k) for a given k.
    pknw_callable : callable
        Callable returning the no-wiggle linear power spectrum P_nw(k) for a given k.
    damping : str, default='lor'
        One of ('lor', 'exp', 'vdg').
    Pshot_const : float, default=0.
        Constant shot-noise contribution.

    Returns
    -------
    bispectrum : array_like
        Modeled bispectrum B^{abc}(k1, k2, k3).
    """
    a, b, c = fields

    def _get_bias_params(field):
        pars = bias_params[field]
        if isinstance(pars, dict):
            return (
                pars['b1'], pars['b2'], pars['bs'], pars['c1'], pars['c2'],
                pars.get('Bshot', 0.), pars.get('Pshot', 0.), pars.get('X_FoG_b', 1.)
            )
        return tuple(pars)

    def _norm(kvec):
        return jnp.sqrt(jnp.sum(kvec**2, axis=-1))

    def _mu(kvec, knorm):
        return jnp.where(knorm > 0., kvec[..., 2] / knorm, 0.)

    def _xcos(kivec, kjvec, ki, kj):
        denom = ki * kj
        return jnp.where(denom > 0., jnp.sum(kivec * kjvec, axis=-1) / denom, 0.)

    def _Z1(field, mu):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)
        return b1 + f * mu**2

    def _Z1eft(field, k, mu):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)
        return _Z1(field, mu) - (c1 * mu**2 + c2 * mu**4) * k**2

    def _Z2(field, ki, kj, xij, mui, muj):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)
        km = ki * mui + kj * muj
        term1 = b2 / 2. + bs / 2. * (xij**2 - 1. / 3.)
        term2 = km / 2. * (mui / ki * f * (b1 + f * muj**2) + muj / kj * f * (b1 + f * mui**2))
        F2 = 5. / 7. + xij / 2. * (ki / kj + kj / ki) + 2. / 7. * xij**2
        G2 = 3. / 7. + xij / 2. * (ki / kj + kj / ki) + 4. / 7. * xij**2
        term3 = b1 * F2
        mu2 = km**2 / (ki**2 + kj**2 + 2. * ki * kj * xij)
        term4 = f * mu2 * G2
        return term1 + term2 + term3 + term4

    def _IR_pk(k, mu):
        pk = pk_callable(k)
        pknw = pknw_callable(k)
        eIR = (1. + f * mu**2 * (2. + f)) * Sigma2 + (f * mu)**2 * (mu**2 - 1.) * deltaSigma2
        return pknw + (pk - pknw) * jnp.exp(-eIR * k**2)

    def _shot_leg(field, k, mu, Z1eft, pkIR):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)
        return (b1 * Bshot + 2. * Pshot * f * mu**2) * Z1eft * pkIR

    def _fog_weight(k1, mu1, k2, mu2, k3, mu3):
        Xa = _get_bias_params(a)[-1]
        Xb = _get_bias_params(b)[-1]
        Xc = _get_bias_params(c)[-1]
        l2_raw = (k1 * mu1)**2 + (k2 * mu2)**2 + (k3 * mu3)**2
        if damping == 'lor':
            l2 = 0.5 * ((f * Xa * k1 * mu1)**2 + (f * Xb * k2 * mu2)**2 + (f * Xc * k3 * mu3)**2)
            return 1. / (1. + l2 * sigma2v)
        if damping == 'exp':
            l2 = 0.5 * ((f * Xa * k1 * mu1)**2 + (f * Xb * k2 * mu2)**2 + (f * Xc * k3 * mu3)**2)
            return jnp.exp(-l2 * sigma2v)
        if damping == 'vdg':
            X2 = (Xa**2 + Xb**2 + Xc**2) / 3.
            l123_2 = -0.5 * l2_raw * f**2
            expterm = l123_2 * sigma2v / (1. - l123_2 * X2)
            return jnp.exp(expterm) / (1. - l123_2 * X2)**1.5
        return jnp.ones_like(k1)

    k1vec = jnp.asarray(k1vec)
    k2vec = jnp.asarray(k2vec)
    k3vec = -k1vec - k2vec

    k1 = _norm(k1vec)
    k2 = _norm(k2vec)
    k3 = _norm(k3vec)

    mu1 = _mu(k1vec, k1)
    mu2 = _mu(k2vec, k2)
    mu3 = _mu(k3vec, k3)

    x12 = _xcos(k1vec, k2vec, k1, k2)
    x23 = _xcos(k2vec, k3vec, k2, k3)
    x31 = _xcos(k3vec, k1vec, k3, k1)

    pkIR1 = _IR_pk(k1, mu1)
    pkIR2 = _IR_pk(k2, mu2)
    pkIR3 = _IR_pk(k3, mu3)

    Z1eft1 = _Z1eft(a, k1, mu1)
    Z1eft2 = _Z1eft(b, k2, mu2)
    Z1eft3 = _Z1eft(c, k3, mu3)

    B12 = 2. * _Z2(c, k1, k2, x12, mu1, mu2) * Z1eft1 * pkIR1 * Z1eft2 * pkIR2
    B23 = 2. * _Z2(a, k2, k3, x23, mu2, mu3) * Z1eft2 * pkIR2 * Z1eft3 * pkIR3
    B31 = 2. * _Z2(b, k3, k1, x31, mu3, mu1) * Z1eft3 * pkIR3 * Z1eft1 * pkIR1

    W = _fog_weight(k1, mu1, k2, mu2, k3, mu3)

    shot = _shot_leg(a, k1, mu1, Z1eft1, pkIR1) + _shot_leg(b, k2, mu2, Z1eft2, pkIR2) + _shot_leg(c, k3, mu3, Z1eft3, pkIR3) + Pshot_const

    return W * (B12 + B23 + B31) + shot


def spectrum4(k1vec, k2vec, k3vec, fields, f, sigma2v, Sigma2, deltaSigma2, bias_params, pk_callable, pknw_callable,
              damping=None, Pshot_const=0.):
    """
    JAX-friendly tree-level cross-trispectrum T^{abcd}(k1, k2, k3, k4),
    with k4 = -k1 - k2 - k3, including the same wiggle damping as in spectrum3.

    Parameters
    ----------
    k1vec, k2vec, k3vec : array_like[..., 3]
        Wavevectors of the first three legs.
    fields : tuple
        Tuple (a, b, c, d) of field identifiers.
    f, sigma2v, Sigma2, deltaSigma2 : float
        RSD / IR / FoG parameters.
    bias_params : dict
        Mapping field -> parameters.
    pk_callable : callable
        Callable returning the linear power spectrum P(k).
    pknw_callable : callable
        Callable returning the no-wiggle linear power spectrum P_nw(k).
    damping : {None, 'lor', 'exp', 'vdg'}, default=None
        Optional phenomenological FoG damping on the full trispectrum.
    Pshot_const : float, default=0.
        Constant shot-noise contribution, kept for interface symmetry.

    Returns
    -------
    trispectrum : array_like
        Tree-level trispectrum model.
    """
    a, b, c, d = fields

    def _get_bias_params(field):
        pars = bias_params[field]
        if isinstance(pars, dict):
            return (
                pars['b1'], pars['b2'], pars['bs'], pars.get('c1', 0.), pars.get('c2', 0.),
                pars.get('Bshot', 0.), pars.get('Pshot', 0.), pars.get('X_FoG_b', 1.)
            )
        return tuple(pars)

    def _norm(kvec):
        return jnp.sqrt(jnp.sum(kvec**2, axis=-1))

    def _mu(kvec, knorm):
        return jnp.where(knorm > 0., kvec[..., 2] / knorm, 0.)

    def _xcos(kivec, kjvec, ki, kj):
        denom = ki * kj
        return jnp.where(denom > 0., jnp.sum(kivec * kjvec, axis=-1) / denom, 0.)

    def _Z1(field, mu):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)
        return b1 + f * mu**2

    def _Z1eff(field, k, mu):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)
        return _Z1(field, mu) - (c1 * mu**2 + c2 * mu**4) * k**2

    def _F2(ki, kj, xij):
        return 5. / 7. + xij / 2. * (ki / kj + kj / ki) + 2. / 7. * xij**2

    def _G2(ki, kj, xij):
        return 3. / 7. + xij / 2. * (ki / kj + kj / ki) + 4. / 7. * xij**2

    def _alpha(ki, kj, xij):
        return 1. + (kj / ki) * xij

    def _beta(ki, kj, xij):
        return xij * (ki**2 + kj**2 + 2. * ki * kj * xij) / (2. * ki * kj)

    def _F3_unsym(ki, kj, kk, xij, xik, xjk):
        q12sq = ki**2 + kj**2 + 2. * ki * kj * xij
        q12 = jnp.sqrt(q12sq)
        x_12_3 = jnp.where(q12 > 0., (ki * xik + kj * xjk) / q12, 0.)
        F2_12 = _F2(ki, kj, xij)
        G2_12 = _G2(ki, kj, xij)
        return (7. * _alpha(q12, kk, x_12_3) * F2_12 + 2. * _beta(q12, kk, x_12_3) * G2_12) / 18.

    def _G3_unsym(ki, kj, kk, xij, xik, xjk):
        q12sq = ki**2 + kj**2 + 2. * ki * kj * xij
        q12 = jnp.sqrt(q12sq)
        x_12_3 = jnp.where(q12 > 0., (ki * xik + kj * xjk) / q12, 0.)
        F2_12 = _F2(ki, kj, xij)
        G2_12 = _G2(ki, kj, xij)
        return (3. * _alpha(q12, kk, x_12_3) * F2_12 + 6. * _beta(q12, kk, x_12_3) * G2_12) / 18.

    def _F3(ki, kj, kk, xij, xik, xjk):
        t1 = _F3_unsym(ki, kj, kk, xij, xik, xjk)
        t2 = _F3_unsym(ki, kk, kj, xik, xij, xjk)
        t3 = _F3_unsym(kj, ki, kk, xij, xjk, xik)
        t4 = _F3_unsym(kj, kk, ki, xjk, xij, xik)
        t5 = _F3_unsym(kk, ki, kj, xik, xjk, xij)
        t6 = _F3_unsym(kk, kj, ki, xjk, xik, xij)
        return t1 + t2 + t3 + t4 + t5 + t6

    def _G3(ki, kj, kk, xij, xik, xjk):
        t1 = _G3_unsym(ki, kj, kk, xij, xik, xjk)
        t2 = _G3_unsym(ki, kk, kj, xik, xij, xjk)
        t3 = _G3_unsym(kj, ki, kk, xij, xjk, xik)
        t4 = _G3_unsym(kj, kk, ki, xjk, xij, xik)
        t5 = _G3_unsym(kk, ki, kj, xik, xjk, xij)
        t6 = _G3_unsym(kk, kj, ki, xjk, xik, xij)
        return t1 + t2 + t3 + t4 + t5 + t6

    def _Z2(field, ki, kj, xij, mui, muj):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)
        km = ki * mui + kj * muj
        term1 = b2 / 2. + bs / 2. * (xij**2 - 1. / 3.)
        term2 = km / 2. * (mui / ki * f * (b1 + f * muj**2) + muj / kj * f * (b1 + f * mui**2))
        term3 = b1 * _F2(ki, kj, xij)
        mu2 = km**2 / (ki**2 + kj**2 + 2. * ki * kj * xij)
        term4 = f * mu2 * _G2(ki, kj, xij)
        return term1 + term2 + term3 + term4

    def _A2(field, ki, kj, xij, mui, muj):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)
        q12vec_z = ki * mui + kj * muj
        q12 = jnp.sqrt(ki**2 + kj**2 + 2. * ki * kj * xij)
        mu12 = jnp.where(q12 > 0., q12vec_z / q12, 0.)
        return b1 * _F2(ki, kj, xij) + b2 / 2. + bs / 2. * (xij**2 - 1. / 3.) + f * mu12**2 * _G2(ki, kj, xij)

    def _Z3(field, k1vec, k2vec, k3vec):
        b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = _get_bias_params(field)

        k1 = _norm(k1vec)
        k2 = _norm(k2vec)
        k3 = _norm(k3vec)
        k123vec = k1vec + k2vec + k3vec
        k = _norm(k123vec)

        mu = _mu(k123vec, k)
        mu1 = _mu(k1vec, k1)
        mu2 = _mu(k2vec, k2)
        mu3 = _mu(k3vec, k3)

        x12 = _xcos(k1vec, k2vec, k1, k2)
        x13 = _xcos(k1vec, k3vec, k1, k3)
        x23 = _xcos(k2vec, k3vec, k2, k3)

        F3 = _F3(k1, k2, k3, x12, x13, x23)
        G3 = _G3(k1, k2, k3, x12, x13, x23)

        A1_1 = _Z1(field, mu1)
        A1_2 = _Z1(field, mu2)
        A1_3 = _Z1(field, mu3)

        A2_12 = _A2(field, k1, k2, x12, mu1, mu2)
        A2_23 = _A2(field, k2, k3, x23, mu2, mu3)
        A2_31 = _A2(field, k3, k1, x13, mu3, mu1)

        k12vec = k1vec + k2vec
        k23vec = k2vec + k3vec
        k31vec = k3vec + k1vec

        k12 = _norm(k12vec)
        k23 = _norm(k23vec)
        k31 = _norm(k31vec)

        mu12 = _mu(k12vec, k12)
        mu23 = _mu(k23vec, k23)
        mu31 = _mu(k31vec, k31)

        G2_12 = _G2(k1, k2, x12)
        G2_23 = _G2(k2, k3, x23)
        G2_31 = _G2(k3, k1, x13)

        pref = f * mu * k

        t12 = pref * A2_12 * (mu3 / k3)
        t23 = pref * A2_23 * (mu1 / k1)
        t31 = pref * A2_31 * (mu2 / k2)

        u12 = pref * A1_1 * (mu23 / k23) * G2_23
        u23 = pref * A1_2 * (mu31 / k31) * G2_31
        u31 = pref * A1_3 * (mu12 / k12) * G2_12

        v12 = 0.5 * pref**2 * A1_1 * (mu2 / k2) * (mu3 / k3)
        v23 = 0.5 * pref**2 * A1_2 * (mu3 / k3) * (mu1 / k1)
        v31 = 0.5 * pref**2 * A1_3 * (mu1 / k1) * (mu2 / k2)

        return b1 * F3 + f * mu**2 * G3 + t12 + t23 + t31 + u12 + u23 + u31 + v12 + v23 + v31

    def _IR_pk(k, mu):
        pk = pk_callable(k)
        pknw = pknw_callable(k)
        eIR = (1. + f * mu**2 * (2. + f)) * Sigma2 + (f * mu)**2 * (mu**2 - 1.) * deltaSigma2
        return pknw + (pk - pknw) * jnp.exp(-eIR * k**2)

    def _fog_weight(k1, mu1, k2, mu2, k3, mu3, k4, mu4):
        if damping is None:
            return jnp.ones_like(k1)
        Xa = _get_bias_params(a)[-1]
        Xb = _get_bias_params(b)[-1]
        Xc = _get_bias_params(c)[-1]
        Xd = _get_bias_params(d)[-1]
        l2_raw = (k1 * mu1)**2 + (k2 * mu2)**2 + (k3 * mu3)**2 + (k4 * mu4)**2
        if damping == 'lor':
            l2 = 0.5 * ((f * Xa * k1 * mu1)**2 + (f * Xb * k2 * mu2)**2 + (f * Xc * k3 * mu3)**2 + (f * Xd * k4 * mu4)**2)
            return 1. / (1. + l2 * sigma2v)
        if damping == 'exp':
            l2 = 0.5 * ((f * Xa * k1 * mu1)**2 + (f * Xb * k2 * mu2)**2 + (f * Xc * k3 * mu3)**2 + (f * Xd * k4 * mu4)**2)
            return jnp.exp(-l2 * sigma2v)
        if damping == 'vdg':
            X2 = (_get_bias_params(a)[-1]**2 + _get_bias_params(b)[-1]**2 + _get_bias_params(c)[-1]**2 + _get_bias_params(d)[-1]**2) / 4.
            l1234_2 = -0.5 * l2_raw * f**2
            expterm = l1234_2 * sigma2v / (1. - l1234_2 * X2)
            return jnp.exp(expterm) / (1. - l1234_2 * X2)**2
        return jnp.ones_like(k1)

    def _t2211_term(fi, fj, fk, fl, kivec, kjvec, kkvec, klvec):
        ki, kj = _norm(kivec), _norm(kjvec)
        mui, muj = _mu(kivec, ki), _mu(kjvec, kj)
        qvec = kivec + klvec
        q = _norm(qvec)
        muq = _mu(qvec, q)
        x_iq = _xcos(-kivec, qvec, ki, q)
        x_jmq = _xcos(-kjvec, -qvec, kj, q)
        Zi = _Z1eff(fi, ki, mui)
        Zj = _Z1eff(fj, kj, muj)
        Z2k = _Z2(fk, ki, q, x_iq, -mui, muq)
        Z2l = _Z2(fl, kj, q, x_jmq, -muj, -muq)
        Pki = _IR_pk(ki, mui)
        Pkj = _IR_pk(kj, muj)
        Pq = _IR_pk(q, muq)
        return Zi * Zj * Z2k * Z2l * Pki * Pkj * Pq

    def _t3111_term(fi, fj, fk, fl, kivec, kjvec, kkvec):
        ki, kj, kk = _norm(kivec), _norm(kjvec), _norm(kkvec)
        mui, muj, muk = _mu(kivec, ki), _mu(kjvec, kj), _mu(kkvec, kk)
        Zi, Zj, Zk = _Z1eff(fi, ki, mui), _Z1eff(fj, kj, muj), _Z1eff(fk, kk, muk)
        Z3l = _Z3(fl, kivec, kjvec, kkvec)
        Pki = _IR_pk(ki, mui)
        Pkj = _IR_pk(kj, muj)
        Pkk = _IR_pk(kk, muk)
        return Zi * Zj * Zk * Z3l * Pki * Pkj * Pkk

    k1vec = jnp.asarray(k1vec)
    k2vec = jnp.asarray(k2vec)
    k3vec = jnp.asarray(k3vec)
    k4vec = -k1vec - k2vec - k3vec

    k1, k2, k3, k4 = _norm(k1vec), _norm(k2vec), _norm(k3vec), _norm(k4vec)
    mu1, mu2, mu3, mu4 = _mu(k1vec, k1), _mu(k2vec, k2), _mu(k3vec, k3), _mu(k4vec, k4)

    t2211 = (
        _t2211_term(a, b, c, d, k1vec, k2vec, k3vec, k4vec) +
        _t2211_term(a, b, d, c, k1vec, k2vec, k4vec, k3vec) +
        _t2211_term(a, c, b, d, k1vec, k3vec, k2vec, k4vec) +
        _t2211_term(a, c, d, b, k1vec, k3vec, k4vec, k2vec) +
        _t2211_term(a, d, b, c, k1vec, k4vec, k2vec, k3vec) +
        _t2211_term(a, d, c, b, k1vec, k4vec, k3vec, k2vec) +
        _t2211_term(b, c, a, d, k2vec, k3vec, k1vec, k4vec) +
        _t2211_term(b, c, d, a, k2vec, k3vec, k4vec, k1vec) +
        _t2211_term(b, d, a, c, k2vec, k4vec, k1vec, k3vec) +
        _t2211_term(b, d, c, a, k2vec, k4vec, k3vec, k1vec) +
        _t2211_term(c, d, a, b, k3vec, k4vec, k1vec, k2vec) +
        _t2211_term(c, d, b, a, k3vec, k4vec, k2vec, k1vec)
    )

    t3111 = (
        _t3111_term(a, b, c, d, k1vec, k2vec, k3vec) +
        _t3111_term(a, b, d, c, k1vec, k2vec, k4vec) +
        _t3111_term(a, c, d, b, k1vec, k3vec, k4vec) +
        _t3111_term(b, c, d, a, k2vec, k3vec, k4vec)
    )

    W = _fog_weight(k1, mu1, k2, mu2, k3, mu3, k4, mu4)
    return W * (4. * t2211 + 6. * t3111) + Pshot_const



# 1-loop power spectrum

import dataclasses
import functools
import operator
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


# ======================================================================================
# Gauss-Legendre integration helpers
# ======================================================================================

prod = partial(functools.reduce, operator.mul)


@dataclasses.dataclass
class Integral1D:
    x: np.ndarray
    w: np.ndarray


class IntegralND:

    def __init__(self, **kwargs):
        self.integ = dict(kwargs)

    def x(self, names=None, sparse=True):
        all_names = list(self.integ.keys())
        if names is None:
            names = all_names
        grids = np.meshgrid(*[self.integ[name].x for name in all_names], sparse=sparse, indexing='ij')
        if np.ndim(names) == 0:
            return grids[all_names.index(names)]
        return [grids[all_names.index(name)] for name in names]

    @property
    def w(self):
        w = [integ.w for integ in self.integ.values()]
        return prod(np.meshgrid(*w, sparse=True, indexing='ij'))

    @property
    def ndim(self):
        return len(self.integ)

    def __call__(self, integrand):
        axes = tuple(range(integrand.ndim - self.ndim, integrand.ndim))
        return np.sum(integrand * self.w, axis=axes)


def integration(a=0., b=1., size=5, method='leggauss'):
    nodes, weights = np.polynomial.legendre.leggauss(size)
    nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights
    return Integral1D(x=nodes, w=weights)


def make_integral_nd(**kwargs):
    integ = IntegralND(**kwargs)
    xs = {name: jnp.asarray(val) for name, val in zip(integ.integ.keys(), integ.x())}
    w = jnp.asarray(integ.w)
    return xs, w


# ======================================================================================
# Helpers
# ======================================================================================

TWOPI2 = 2.0 * jnp.pi**2
TWOPI3 = (2.0 * jnp.pi) ** 3


def S2(mu):
    return mu**2 - 1.0 / 3.0


def F2(k1, k2, mu12):
    return 5.0 / 7.0 + 0.5 * mu12 * (k1 / k2 + k2 / k1) + 2.0 / 7.0 * mu12**2


def G2(k1, k2, mu12):
    return 3.0 / 7.0 + 0.5 * mu12 * (k1 / k2 + k2 / k1) + 4.0 / 7.0 * mu12**2


def Lfun(r, eps=1e-8):
    return jnp.log((1.0 + r) / jnp.maximum(jnp.abs(1.0 - r), eps))


def K13_dd(r):
    r2 = r * r
    return (
        12.0 / r2 - 158.0 + 100.0 * r2 - 42.0 * r2**2
        + 3.0 / r**3 * (r2 - 1.0) ** 3 * (7.0 * r2 + 2.0) * Lfun(r)
    )


def K13_dt(r):
    r2 = r * r
    return (
        24.0 / r2 - 202.0 + 56.0 * r2 - 30.0 * r2**2
        + 3.0 / r**3 * (r2 - 1.0) ** 3 * (5.0 * r2 + 4.0) * Lfun(r)
    )


def K13_tt(r):
    r2 = r * r
    return (
        12.0 / r2 - 82.0 + 4.0 * r2 - 6.0 * r2**2
        + 3.0 / r**3 * (r2 - 1.0) ** 3 * (r2 + 2.0) * Lfun(r)
    )


def j0(x):
    x2 = x * x
    return jnp.where(x2 < 1e-6, 1.0 - x2 / 6.0 + x2 * x2 / 120.0, jnp.sin(x) / x)


def j2(x):
    x2 = x * x
    series = x2 / 15.0 - x2 * x2 / 210.0
    exact = ((3.0 / x**3 - 1.0 / x) * jnp.sin(x) - 3.0 * jnp.cos(x) / x**2)
    return jnp.where(x2 < 1e-6, series, exact)


def K2_delta(k1, k2, mu12, b1, b2, bs2):
    return b1 * F2(k1, k2, mu12) + 0.5 * b2 + bs2 * S2(mu12)


def glsum1d(f, w):
    return jnp.sum(f * w, axis=-1)


def glsum2d(f, w):
    return jnp.sum(f * w, axis=(-2, -1))


def glsum3d(f, w):
    return jnp.sum(f * w, axis=(-3, -2, -1))


def sigma_v2_linear(pk_callable, pmin=5e-4, pmax=10.0, npk=400):
    p, w = make_integral_nd(p=integration(a=pmin, b=pmax, size=npk))
    return (1.0 / (6.0 * jnp.pi**2)) * glsum1d(pk_callable(p["p"]), w)


def sigma_ir2(pknw_callable, ks=0.2, kbao=1.0 / 105.0, npk=400):
    p, w = make_integral_nd(p=integration(a=5e-4, b=ks, size=npk))
    pp = p["p"]
    x = pp / kbao
    pnw = pknw_callable(pp)
    Sigma2 = (1.0 / (6.0 * jnp.pi**2)) * glsum1d(pnw * (1.0 - j0(x) + 2.0 * j2(x)), w)
    dSigma2 = (1.0 / (2.0 * jnp.pi**2)) * glsum1d(pnw * j2(x), w)
    return Sigma2, dSigma2


# ======================================================================================
# 1-loop SPT matter power spectra: Pdd, Pdt, Ptt
# ======================================================================================

def spt_1loop_matter(k, pk_callable, nr=400, nmu=10, rmin=5e-4, rmax=10.0):
    k = jnp.atleast_1d(k)

    xs2d, w2d = make_integral_nd(
        r=integration(a=rmin, b=rmax, size=nr),
        mu=integration(a=-1.0, b=1.0, size=nmu),
    )
    r, mu = xs2d["r"], xs2d["mu"]

    xs1d, w1d = make_integral_nd(r=integration(a=rmin, b=rmax, size=nr))
    r13 = xs1d["r"]

    def I22(kind, kk):
        y2 = 1.0 + r * r - 2.0 * r * mu
        y = jnp.sqrt(jnp.maximum(y2, 1e-30))
        P1 = pk_callable(kk * r)
        P2 = pk_callable(kk * y)
        A = 3.0 * r + 7.0 * mu - 10.0 * r * mu * mu
        B = -r + 7.0 * mu - 6.0 * r * mu * mu
        num = {"dd": A * A, "dt": A * B, "tt": B * B}[kind]
        pref = {"dd": 1.0 / 98.0, "dt": 1.0 / 196.0, "tt": 1.0 / 98.0}[kind]
        return kk**3 / TWOPI2 * pref * glsum2d(P1 * P2 * num / y2**2, w2d)

    def I13(kind, kk):
        Pk = pk_callable(kk)
        Pkr = pk_callable(kk * r13)
        ker = {"dd": K13_dd(r13), "dt": K13_dt(r13), "tt": K13_tt(r13)}[kind]
        pref = {"dd": 1.0 / 252.0, "dt": 1.0 / 252.0, "tt": 1.0 / 84.0}[kind]
        return kk**3 / TWOPI2 * pref * Pk * glsum1d(Pkr * ker, w1d)

    def one_k(kk):
        P11 = pk_callable(kk)
        P22_dd, P22_dt, P22_tt = I22("dd", kk), I22("dt", kk), I22("tt", kk)
        P13_dd, P13_dt, P13_tt = I13("dd", kk), I13("dt", kk), I13("tt", kk)
        return {
            "P11": P11,
            "P22_dd": P22_dd, "P22_dt": P22_dt, "P22_tt": P22_tt,
            "P13_dd": P13_dd, "P13_dt": P13_dt, "P13_tt": P13_tt,
            "Pdd": P11 + P22_dd + 2.0 * P13_dd,
            "Pdt": P11 + P22_dt + 2.0 * P13_dt,
            "Ptt": P11 + P22_tt + 2.0 * P13_tt,
        }

    out = jax.vmap(one_k)(k)
    return {name: out[name] for name in out}


# ======================================================================================
# 1-loop bias basis
# ======================================================================================

def bias_terms_1loop(k, pk_callable, nr=400, nmu=10, rmin=5e-4, rmax=10.0):
    k = jnp.atleast_1d(k)

    xs2d, w2d = make_integral_nd(
        r=integration(a=rmin, b=rmax, size=nr),
        mu=integration(a=-1.0, b=1.0, size=nmu),
    )
    r, mu = xs2d["r"], xs2d["mu"]

    y = jnp.sqrt(jnp.maximum(1.0 + r * r - 2.0 * r * mu, 1e-30))
    mu_qkmq = (mu - r) / y
    mu_mqk = mu

    F2_q = F2(r, y, mu_qkmq)
    G2_q = G2(r, y, mu_qkmq)
    S2_q = S2(mu_qkmq)
    S2_mqk = S2(mu_mqk)

    def one_k(kk):
        Pq = pk_callable(kk * r)
        Pp = pk_callable(kk * y)

        def integ(expr):
            return kk**3 / (4.0 * jnp.pi**2) * glsum2d(r**2 * expr, w2d)

        Pb1b2   = integ(F2_q * Pq * Pp)
        Pb1bs2  = integ(F2_q * S2_q * Pq * Pp)
        Pb2th   = integ(G2_q * Pq * Pp)
        Pbs2th  = integ(G2_q * S2_q * Pq * Pp)

        Pb2b2   = 0.5 * integ(Pq * (Pp - Pq))
        Pb2bs2  = 0.5 * integ(Pq * (Pp * S2_q - (2.0 / 3.0) * Pq))
        Pbs2bs2 = 0.5 * integ(Pq * (Pp * S2_q**2 - (4.0 / 9.0) * Pq))

        sigma3sq = (105.0 / 16.0) * integ(
            Pq * (S2_q * ((2.0 / 7.0) * S2_mqk - 4.0 / 21.0) + 8.0 / 63.0)
        )

        return {
            "Pb1b2": Pb1b2,
            "Pb1bs2": Pb1bs2,
            "Pb2b2": Pb2b2,
            "Pb2bs2": Pb2bs2,
            "Pbs2bs2": Pbs2bs2,
            "Pb2th": Pb2th,
            "Pbs2th": Pbs2th,
            "sigma3sq": sigma3sq,
        }

    out = jax.vmap(one_k)(k)
    return {name: out[name] for name in out}


# ======================================================================================
# Biased real-space spectra for two tracers a, b
# ======================================================================================

def biased_cross_realspace_spectra(k, pk_callable, tracer_a, tracer_b, matter=None, bias_terms=None):
    if matter is None:
        matter = spt_1loop_matter(k, pk_callable)
    if bias_terms is None:
        bias_terms = bias_terms_1loop(k, pk_callable)

    PL = matter["P11"]
    Pdd_m, Pdt_m, Ptt_m = matter["Pdd"], matter["Pdt"], matter["Ptt"]

    ba1, ba2, bas2, ba3 = tracer_a["b1"], tracer_a["b2"], tracer_a["bs2"], tracer_a["b3nl"]
    bb1, bb2, bbs2, bb3 = tracer_b["b1"], tracer_b["b2"], tracer_b["bs2"], tracer_b["b3nl"]

    bt = bias_terms

    Pdd_ab = (
        ba1 * bb1 * Pdd_m
        + (ba1 * bb2 + bb1 * ba2) * bt["Pb1b2"]
        + (ba1 * bbs2 + bb1 * bas2) * bt["Pb1bs2"]
        + ba2 * bb2 * bt["Pb2b2"]
        + (ba2 * bbs2 + bb2 * bas2) * bt["Pb2bs2"]
        + bas2 * bbs2 * bt["Pbs2bs2"]
        + (ba1 * bb3 + bb1 * ba3) * bt["sigma3sq"] * PL
    )

    Pa_th = ba1 * Pdt_m + ba2 * bt["Pb2th"] + bas2 * bt["Pbs2th"] + ba3 * bt["sigma3sq"] * PL
    Pb_th = bb1 * Pdt_m + bb2 * bt["Pb2th"] + bbs2 * bt["Pbs2th"] + bb3 * bt["sigma3sq"] * PL

    return {
        "Pdd_ab": Pdd_ab,
        "Pa_th": Pa_th,
        "Pb_th": Pb_th,
        "Ptt": Ptt_m,
        **matter,
        **bt,
    }


# ======================================================================================
# Tree-level mixed bispectrum for A term
# ======================================================================================

def Btree_mixed(k1, k2, k3, mu12, mu23, mu31, P1, P2, P3, X, Y, Z, tracer_a, tracer_b):
    pars = {"a": tracer_a, "b": tracer_b}

    def K1(lbl):
        return 1.0 if lbl == "th" else pars[lbl]["b1"]

    def K2(lbl, ka, kb, muab):
        if lbl == "th":
            return G2(ka, kb, muab)
        p = pars[lbl]
        return K2_delta(ka, kb, muab, p["b1"], p["b2"], p["bs2"])

    return 2.0 * (
        K2(X, k2, k3, mu23) * K1(Y) * K1(Z) * P2 * P3
        + K2(Y, k3, k1, mu31) * K1(Z) * K1(X) * P3 * P1
        + K2(Z, k1, k2, mu12) * K1(X) * K1(Y) * P1 * P2
    )


# ======================================================================================
# A and D terms for cross spectrum
# ======================================================================================

def A_D_cross(
    k, mu, pk_callable, f, tracer_a, tracer_b,
    nr=400, nmu=10, nphi=16, rmin=5e-4, rmax=10.0, sigma_v2=None
):
    k = jnp.atleast_1d(k)
    mu = jnp.atleast_1d(mu)
    if sigma_v2 is None:
        sigma_v2 = sigma_v2_linear(pk_callable, pmin=rmin, pmax=rmax, npk=nr)

    xs3d, w3d = make_integral_nd(
        r=integration(a=rmin, b=rmax, size=nr),
        x=integration(a=-1.0, b=1.0, size=nmu),
        phi=integration(a=0.0, b=2.0 * np.pi, size=nphi),
    )
    r, x, phi = xs3d["r"], xs3d["x"], xs3d["phi"]

    def one_kmu(kk, muu):
        s = jnp.sqrt(jnp.maximum(1.0 - muu**2, 0.0))

        y = jnp.sqrt(jnp.maximum(1.0 + r**2 - 2.0 * r * x, 1e-30))
        p = kk * r
        q = kk * y

        mu_p = x * muu + jnp.sqrt(jnp.maximum(1.0 - x**2, 0.0)) * s * jnp.cos(phi)
        mu_q = (muu - r * mu_p) / y

        mu_pq = (x - r) / y
        mu_kp = x
        mu_qk = (1.0 - r * x) / y

        Pp, Pq, Pk = pk_callable(p), pk_callable(q), pk_callable(kk)

        PkK_ab_k = (
            tracer_a["b1"] * tracer_b["b1"] * Pk
            + f * muu**2 * (tracer_a["b1"] + tracer_b["b1"]) * Pk
            + f**2 * muu**4 * Pk
        )

        PkK_ab_q = (
            tracer_a["b1"] * tracer_b["b1"] * Pq
            + f * mu_q**2 * (tracer_a["b1"] + tracer_b["b1"]) * Pq
            + f**2 * mu_q**4 * Pq
        )

        def Bsigma(left, right):
            B_th_lr = Btree_mixed(
                p, kk, q, mu_kp, mu_qk, mu_pq, Pp, Pk, Pq,
                "th", left, right, tracer_a, tracer_b
            )
            B_th_lth = Btree_mixed(
                p, kk, q, mu_kp, mu_qk, mu_pq, Pp, Pk, Pq,
                "th", left, "th", tracer_a, tracer_b
            )
            B_th_thr = Btree_mixed(
                p, kk, q, mu_kp, mu_qk, mu_pq, Pp, Pk, Pq,
                "th", "th", right, tracer_a, tracer_b
            )
            B_th_thth = Btree_mixed(
                p, kk, q, mu_kp, mu_qk, mu_pq, Pp, Pk, Pq,
                "th", "th", "th", tracer_a, tracer_b
            )
            return (
                B_th_lr
                + f * mu_q**2 * B_th_lth
                + f * muu**2 * B_th_thr
                + f**2 * muu**2 * mu_q**2 * B_th_thth
            )

        Bsigma_ab = 0.5 * (Bsigma("a", "b") + Bsigma("b", "a"))

        Aab = 2.0 * kk * muu * f / TWOPI3 * glsum3d(r**2 * (mu_p / p) * Bsigma_ab, w3d)

        Fa_p = (mu_p / p) * (tracer_a["b1"] * Pp + f * mu_p**2 * Pp)
        Fb_p = (mu_p / p) * (tracer_b["b1"] * Pp + f * mu_p**2 * Pp)
        Fa_q = (mu_q / q) * (tracer_a["b1"] * Pq + f * mu_q**2 * Pq)
        Fb_q = (mu_q / q) * (tracer_b["b1"] * Pq + f * mu_q**2 * Pq)

        Bterm = (kk * muu * f) ** 2 / TWOPI3 * glsum3d(
            r**2 * 0.5 * (Fa_p * Fb_q + Fb_p * Fa_q), w3d
        )

        Cterm = (kk * muu * f) ** 2 / TWOPI3 * glsum3d(
            r**2 * (mu_p**2 / p**2) * Pp * PkK_ab_q, w3d
        )

        Dab = Bterm + Cterm - (kk * muu * f) ** 2 * sigma_v2 * PkK_ab_k
        return Aab, Dab

    A, D = jax.vmap(lambda kk: jax.vmap(lambda mm: one_kmu(kk, mm))(mu))(k)
    return {"A": A, "D": D}


# ======================================================================================
# Kaiser + A + D + EFT
# ======================================================================================

def Ps_cross_me(k, mu, Pdd_ab, Pa_th, Pb_th, Ptt, A, D, f):
    mu2 = mu[None, :]**2
    return (
        Pdd_ab[:, None]
        + f * mu2 * (Pa_th[:, None] + Pb_th[:, None])
        + f**2 * mu2**2 * Ptt[:, None]
        + A + D
    )


def Ps_cross_eft(
    k, mu, pk_callable, f, tracer_a, tracer_b,
    Pdd_ab, Pa_th, Pb_th, Ptt, A, D,
    alpha0=0.0, alpha2=0.0, alpha4=0.0,
    ctilde=0.0, Pshot=0.0
):
    k = jnp.atleast_1d(k)
    mu = jnp.atleast_1d(mu)
    mu2 = mu[None, :]**2
    PL = pk_callable(k)

    Ps_me = Ps_cross_me(k, mu, Pdd_ab, Pa_th, Pb_th, Ptt, A, D, f)

    PkK_lin_ab = (
        tracer_a["b1"] * tracer_b["b1"] * PL[:, None]
        + f * mu2 * (tracer_a["b1"] + tracer_b["b1"]) * PL[:, None]
        + f**2 * mu2**2 * PL[:, None]
    )

    Pct = (alpha0 + alpha2 * mu2 + alpha4 * mu2**2) * (k[:, None]**2) * PL[:, None]
    Pnlo = ctilde * (f * k[:, None] * mu[None, :])**4 * PkK_lin_ab

    return Ps_me + Pct + Pnlo + Pshot


# ======================================================================================
# IR resummation
# ======================================================================================

def Ps_cross_ir_resummed(
    k, mu, pk_callable, pknw_callable, f, tracer_a, tracer_b, Ps_eft, Ps_eft_nw,
    ks=0.2, kbao=1.0 / 105.0
):
    k = jnp.atleast_1d(k)
    mu = jnp.atleast_1d(mu)
    mu2 = mu[None, :]**2

    Sigma2, dSigma2 = sigma_ir2(pknw_callable, ks=ks, kbao=kbao, npk=400)
    Sigma_tot2 = (
        (1.0 + f * mu2 * (2.0 + f)) * Sigma2
        + f**2 * mu2 * (mu2 - 1.0) * dSigma2
    )
    damp = jnp.exp(-(k[:, None]**2) * Sigma_tot2)

    Pw = pk_callable(k) - pknw_callable(k)
    Kcross = (tracer_a["b1"] + f * mu2) * (tracer_b["b1"] + f * mu2)

    return (
        damp * Ps_eft
        + (1.0 - damp) * Ps_eft_nw
        + damp * Kcross * Pw[:, None] * (k[:, None]**2) * Sigma_tot2
    )


# ======================================================================================
# High-level wrapper
# ======================================================================================

def full_cross_ps(
    k, mu, pk_callable, pknw_callable, f, tracer_a, tracer_b,
    alpha0=0.0, alpha2=0.0, alpha4=0.0, ctilde=0.0, Pshot=0.0,
    matter=None, bias_terms=None,
    ad_kwargs=None
):
    if ad_kwargs is None:
        ad_kwargs = {}

    real = biased_cross_realspace_spectra(
        k, pk_callable, tracer_a, tracer_b, matter=matter, bias_terms=bias_terms
    )
    AD = A_D_cross(k, mu, pk_callable, f, tracer_a, tracer_b, **ad_kwargs)
    Ps_eft = Ps_cross_eft(
        k, mu, pk_callable, f, tracer_a, tracer_b,
        real["Pdd_ab"], real["Pa_th"], real["Pb_th"], real["Ptt"],
        AD["A"], AD["D"],
        alpha0=alpha0, alpha2=alpha2, alpha4=alpha4, ctilde=ctilde, Pshot=Pshot
    )

    real_nw = biased_cross_realspace_spectra(k, pknw_callable, tracer_a, tracer_b)
    AD_nw = A_D_cross(k, mu, pknw_callable, f, tracer_a, tracer_b, **ad_kwargs)
    Ps_eft_nw = Ps_cross_eft(
        k, mu, pknw_callable, f, tracer_a, tracer_b,
        real_nw["Pdd_ab"], real_nw["Pa_th"], real_nw["Pb_th"], real_nw["Ptt"],
        AD_nw["A"], AD_nw["D"],
        alpha0=alpha0, alpha2=alpha2, alpha4=alpha4, ctilde=ctilde, Pshot=Pshot
    )

    Ps_ir = Ps_cross_ir_resummed(
        k, mu, pk_callable, pknw_callable, f, tracer_a, tracer_b, Ps_eft, Ps_eft_nw
    )

    return {
        "real": real,
        "AD": AD,
        "Ps_eft": Ps_eft,
        "real_nw": real_nw,
        "AD_nw": AD_nw,
        "Ps_eft_nw": Ps_eft_nw,
        "Ps_ir": Ps_ir,
    }


# ======================================================================================
# Optional multipoles
# ======================================================================================

def legendre(ell, mu):
    if ell == 0:
        return jnp.ones_like(mu)
    if ell == 2:
        return 0.5 * (3.0 * mu**2 - 1.0)
    if ell == 4:
        return (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
    raise ValueError("Only ell = 0, 2, 4 implemented")


def multipole_from_Pkmu(mu, Pkmu, ell):
    L = legendre(ell, mu)[None, :]
    return (2 * ell + 1) * jnp.trapezoid(Pkmu * L, mu, axis=1)


# ======================================================================================
# Example
# ======================================================================================

if __name__ == "__main__":
    k = jnp.logspace(-2.5, -0.3, 80)
    mu = jnp.linspace(0.0, 1.0, 81)

    def pk_callable(q):
        return jnp.where(q > 0.0, q * jnp.exp(-q / 0.25), 0.0)

    def pknw_callable(q):
        return jnp.where(q > 0.0, q * jnp.exp(-q / 0.28), 0.0)

    tracer_a = {"b1": 2.0, "b2": 0.5, "bs2": -0.3, "b3nl": 0.1}
    tracer_b = {"b1": 1.7, "b2": 0.2, "bs2": -0.2, "b3nl": 0.05}

    out = full_cross_ps(
        k, mu,
        pk_callable=pk_callable,
        pknw_callable=pknw_callable,
        f=0.8,
        tracer_a=tracer_a,
        tracer_b=tracer_b,
        alpha0=10.0,
        alpha2=-15.0,
        alpha4=5.0,
        ctilde=2.0,
        Pshot=0.0,
        ad_kwargs=dict(nr=400, nmu=10, nphi=16, rmin=5e-4, rmax=10.0),
    )

    Ps_ir = out["Ps_ir"]
    P0 = multipole_from_Pkmu(mu, Ps_ir, 0)
    P2 = multipole_from_Pkmu(mu, Ps_ir, 2)
    P4 = multipole_from_Pkmu(mu, Ps_ir, 4)
