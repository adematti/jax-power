import heapq
import itertools
import math
import os
import functools
import operator
from functools import lru_cache, partial, reduce

import numpy as np
import jax
import jax.numpy as jnp

from .utils import get_legendre, get_spherical_jn, wigner_3j, get_S


prod = partial(functools.reduce, operator.mul)


class Integral1D:

    def __init__(self, x: np.ndarray=None, w: np.ndarray=None):
        self._x = x
        self._w = w

    def x(self):
        return self._x

    @property
    def w(self):
        return self._w

    @property
    def ndim(self):
        return 1

    def __call__(self, integrand):
        axes = tuple(range(integrand.ndim - self.ndim, integrand.ndim))
        return jnp.sum(integrand * self.w, axis=axes)


class IntegralND:

    def __init__(self, **kwargs):
        self.integ = dict(kwargs)

    def __getitem__(self, name: str):
        return self.integ[name]

    def x(self, names: list | str=None, sparse: bool=True):
        all_names = list(self.integ.keys())
        if names is None:
            names = all_names
        grids = np.meshgrid(*[self.integ[name].x() for name in all_names], sparse=sparse, indexing='ij')
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
        return jnp.sum(integrand * self.w, axis=axes)


def integration(a=0., b=1., size=5, method='leggauss'):
    if method == 'midpoint':
        # Uniform midpoint rule: for smooth periodic integrands over a full
        # period this is spectrally accurate (trapezoid-equivalent), better
        # per node than Gauss-Legendre; the weights equal the cell widths,
        # so each node owns the cell [x_i - h/2, x_i + h/2] exactly.
        h = (b - a) / size
        nodes = a + (np.arange(size) + 0.5) * h
        weights = np.full(size, h)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(size)
        nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        weights = 0.5 * (b - a) * weights
    return Integral1D(x=nodes, w=weights)


def _format_bias_params(bias_params, nfields=2):
    for field, params in bias_params.items():
        if not isinstance(params, dict):
            bias_params = {'a': bias_params}  # single tracer
            break
    fields = list(bias_params)
    fields = fields + [fields[-1]] * (nfields - len(fields))
    return fields, bias_params


# 1-loop SPT matter power spectra: P_dd, P_dθ, P_θθ
#
# Conventions
# -----------
# r  = q / k
# μ  = k·q / (k q)
# y² = 1 + r² - 2 r μ = |k - q|² / k²
#
# Spectra:
#   P_ab(k) = P_11(k) + P_22_ab(k) + P_13_ab(k)
#
# with
#
#   P_22_ab(k)
#   = (k³ / 2π²) ∫ dr ∫ dμ P_L(kr) P_L(ky) K_22_ab(r, μ)
#
#   P_13_ab(k)
#   = (k³ / 2π²) P_L(k) ∫ dr P_L(kr) K_13_ab(r)
#
# where
#
#   K_22_dd = (1/98) A² / y⁴
#   K_22_dθ = (1/98) A B / y⁴
#   K_22_θθ = (1/98) B² / y⁴
#
#   A = 3r + 7μ - 10rμ²
#   B = -r + 7μ - 6rμ²
#
# and
#
#   K_13_ab(r) = K13_ab(r) / 504
#
# The K13_* functions below implement the standard 1-loop SPT kernels
# with a Taylor patch around r = 1 to avoid catastrophic cancellation.


def S2(mu):
    return mu**2 - 1.0 / 3.0


def F2(k1, k2, mu12):
    return 5.0 / 7.0 + 0.5 * mu12 * (k1 / k2 + k2 / k1) + 2.0 / 7.0 * mu12**2


def G2(k1, k2, mu12):
    return 3.0 / 7.0 + 0.5 * mu12 * (k1 / k2 + k2 / k1) + 4.0 / 7.0 * mu12**2


def Lfun(r, eps=1e-8):
    r"""
    Logarithmic piece appearing in the P13 kernels:

        L(r) = log[(1 + r) / |1 - r|]

    The apparent singularity at r = 1 is integrable.
    """
    return jnp.log((1.0 + r) / jnp.maximum(jnp.abs(1.0 - r), eps))


def K13_dd(r, thresh=1e-2):
    r"""
    Density-density 13 kernel:

        K13_dd(r)
        = 12/r² - 158 + 100 r² - 42 r⁴
          + 3/r³ (r² - 1)³ (7r² + 2) L(r)

    Near r = 1, use the Taylor expansion

        K13_dd(r) ≈ -88 + 8Δ - 116Δ²
        Δ = r - 1

    to stabilize the cancellation between polynomial and logarithmic pieces.
    """
    r2 = r * r
    exact = (
        12.0 / r2 - 158.0 + 100.0 * r2 - 42.0 * r2**2
        + 3.0 / r**3 * (r2 - 1.0) ** 3 * (7.0 * r2 + 2.0) * Lfun(r)
    )
    dr = r - 1.0
    series = -88.0 + 8.0 * dr - 116.0 * dr**2
    return jnp.where(jnp.abs(dr) < thresh, series, exact)


def K13_dt(r, thresh=1e-2):
    r"""
    Density-velocity-divergence 13 kernel:

        K13_dθ(r)
        = 24/r² - 202 + 56 r² - 30 r⁴
          + 3/r³ (r² - 1)³ (5r² + 4) L(r)

    Taylor patch near r = 1:

        K13_dθ(r) ≈ -152 - 56Δ - 52Δ²
    """
    r2 = r * r
    exact = (
        24.0 / r2 - 202.0 + 56.0 * r2 - 30.0 * r2**2
        + 3.0 / r**3 * (r2 - 1.0) ** 3 * (5.0 * r2 + 4.0) * Lfun(r)
    )
    dr = r - 1.0
    series = -152.0 - 56.0 * dr - 52.0 * dr**2
    return jnp.where(jnp.abs(dr) < thresh, series, exact)


def K13_tt(r, thresh=1e-2):
    r"""
    Velocity-divergence auto 13 kernel:

        K13_θθ(r)
        = 12/r² - 82 + 4 r² - 6 r⁴
          + 3/r³ (r² - 1)³ (r² + 2) L(r)

    Taylor patch near r = 1:

        K13_θθ(r) ≈ -72 - 40Δ + 4Δ²
    """
    r2 = r * r
    exact = (
        12.0 / r2 - 82.0 + 4.0 * r2 - 6.0 * r2**2
        + 3.0 / r**3 * (r2 - 1.0) ** 3 * (r2 + 2.0) * Lfun(r)
    )
    dr = r - 1.0
    series = -72.0 - 40.0 * dr + 4.0 * dr**2
    return jnp.where(jnp.abs(dr) < thresh, series, exact)


def compute_spt_matter_1loop(
    k: jnp.ndarray,
    pk_callable: callable,
    integ_mu: Integral1D=integration(a=-1.0, b=1.0, size=10),
    integ_r: Integral1D=integration(a=5e-4, b=10.0, size=100),
):
    r"""
    Compute 1-loop SPT matter spectra:
        P_dd(k), P_dθ(k), P_θθ(k)

    Formula:
        P_ab = P_11 + P_22_ab + P_13_ab
    """
    k = jnp.atleast_1d(k)
    integ_rmu = IntegralND(r=integ_r, mu=integ_mu)

    def I22(kind, kk):
        # P22:
        # ∫ dr dμ P_L(kr) P_L(ky) K22_ab(r,μ)
        r, mu = integ_rmu.x(['r', 'mu'])
        y2 = 1.0 + r * r - 2.0 * r * mu
        y = jnp.sqrt(jnp.maximum(y2, 1e-30))

        pk1 = pk_callable(kk * r)
        pk2 = pk_callable(kk * y)

        A = 3.0 * r + 7.0 * mu - 10.0 * r * mu * mu
        B = -r + 7.0 * mu - 6.0 * r * mu * mu

        ker = 1.0 / 98.0 * {"dd": A * A, "dt": A * B, "tt": B * B}[kind]

        return kk**3 / (2.0 * jnp.pi**2) * integ_rmu(pk1 * pk2 * ker / y2**2)

    r = integ_r.x()

    def I13(kind, kk):
        # P13:
        # P_L(k) ∫ dr P_L(kr) K13_ab(r)
        pk = pk_callable(kk)
        pkr = pk_callable(kk * r)

        ker = 1.0 / 504.0 * {"dd": K13_dd, "dt": K13_dt, "tt": K13_tt}[kind](r)

        return kk**3 / (2.0 * jnp.pi**2) * pk * integ_r(pkr * ker)

    def one_k(kk):
        P11 = pk_callable(kk)

        P22_dd = I22("dd", kk)
        P22_dt = I22("dt", kk)
        P22_tt = I22("tt", kk)

        P13_dd = I13("dd", kk)
        P13_dt = I13("dt", kk)
        P13_tt = I13("tt", kk)

        return {"k": kk, "P11": P11, "P22_dd": P22_dd, "P22_dt": P22_dt, "P22_tt": P22_tt, "P13_dd": P13_dd, "P13_dt": P13_dt, "P13_tt": P13_tt,
                "Pdd": P11 + P22_dd + P13_dd, "Pdt": P11 + P22_dt + P13_dt, "Ptt": P11 + P22_tt + P13_tt}

    out = jax.vmap(one_k)(k)
    return {name: out[name] for name in out}


# 1-loop bias basis

def compute_bias_terms_1loop(k: jnp.ndarray,
                     pk_callable: callable,
                     integ_mu: Integral1D=integration(a=-1.0, b=1.0, size=10),
                     integ_r: Integral1D=integration(a=5e-4, b=10.0, size=100)):

    k = jnp.atleast_1d(k)
    integ_rmu = IntegralND(r=integ_r, mu=integ_mu)
    r, mu = integ_rmu.x(['r', 'mu'])

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
            return kk**3 / (4.0 * jnp.pi**2) * integ_rmu(r**2 * expr)

        Pb1b2 = integ(F2_q * Pq * Pp)
        Pb1bs2 = integ(F2_q * S2_q * Pq * Pp)
        Pb2t = integ(G2_q * Pq * Pp)
        Pbs2t = integ(G2_q * S2_q * Pq * Pp)

        Pb2b2 = 0.5 * integ(Pq * (Pp - Pq))
        Pb2bs2 = 0.5 * integ(Pq * (Pp * S2_q - (2.0 / 3.0) * Pq))
        Pbs2bs2 = 0.5 * integ(Pq * (Pp * S2_q**2 - (4.0 / 9.0) * Pq))

        sigma3sq = (105.0 / 16.0) * integ(Pq * (S2_q * ((2.0 / 7.0) * S2_mqk - 4.0 / 21.0) + 8.0 / 63.0))

        return {"k": kk, "Pb1b2": Pb1b2, "Pb1bs2": Pb1bs2, "Pb2b2": Pb2b2, "Pb2bs2": Pb2bs2, "Pbs2bs2": Pbs2bs2, "Pb2t": Pb2t, "Pbs2t": Pbs2t, "sigma3sq": sigma3sq}

    out = jax.vmap(one_k)(k)
    return {name: out[name] for name in out}



# Biased real-space spectra for two tracers a, b

def spectrum2_real_tracer(matter, bias, bias_params):
    PL = matter["P11"]
    Pdd_m, Pdt_m, Ptt_m = matter["Pdd"], matter["Pdt"], matter["Ptt"]
    fields, bias_params = _format_bias_params(bias_params, nfields=2)
    a, b = fields
    ba1, ba2, bas2, ba3 = [bias_params[a][name] for name in ['b1', 'b2', 'bs', 'b3nl']]
    bb1, bb2, bbs2, bb3 = [bias_params[b][name] for name in ['b1', 'b2', 'bs', 'b3nl']]
    Pdd_ab = (
        ba1 * bb1 * Pdd_m
        + (ba1 * bb2 + bb1 * ba2) * bias["Pb1b2"]
        + (ba1 * bbs2 + bb1 * bas2) * bias["Pb1bs2"]
        + ba2 * bb2 * bias["Pb2b2"]
        + (ba2 * bbs2 + bb2 * bas2) * bias["Pb2bs2"]
        + bas2 * bbs2 * bias["Pbs2bs2"]
        + (ba1 * bb3 + bb1 * ba3) * bias["sigma3sq"] * PL
    )
    Pdt_a = ba1 * Pdt_m + ba2 * bias["Pb2t"] + bas2 * bias["Pbs2t"] + ba3 * bias["sigma3sq"] * PL
    Ptd_b = bb1 * Pdt_m + bb2 * bias["Pb2t"] + bbs2 * bias["Pbs2t"] + bb3 * bias["sigma3sq"] * PL
    return dict(Pdd_ab=Pdd_ab, Pdt_a=Pdt_a, Ptd_b=Ptd_b, Ptt=Ptt_m)


def compute_sigma2v(pk_callable, integ_k=integration(a=5e-4, b=10., size=100)):
    return (1.0 / (6.0 * jnp.pi**2)) * integ_k(pk_callable(integ_k.x()))


# Main multitracer TNS A and B terms


def compute_tns_A_B_terms(k, Pdd, Pdt=None, Ptt=None,
    integ_r=integration(a=5e-4, b=10.0, size=100),
    integ_x=integration(a=-1.0, b=1.0, size=20),
):
    r"""
    Multitracer TNS A and B terms following Appendix A of arXiv:2007.09011,
    assuming c_A = c_B = 1.

    Output is organized in powers of f, mu^2 and bias terms.
    """
    if Pdt is None:
        Pdt = Pdd
    if Ptt is None:
        Ptt = Pdd

    k = jnp.atleast_1d(k)

    integ_rx = IntegralND(r=integ_r, x=integ_x)
    r, x = integ_rx.x(["r", "x"])

    def _safe_sqrt(x, eps=1e-30):
        return jnp.sqrt(jnp.maximum(x, eps))

    def _field_kernel2(field):
        if field == 1:
            return F2
        return G2

    def tree_bispectrum_abc(k1, k2, k3, mu12, mu23, mu31, P1, P2, P3, a, b, c):
        Ka = _field_kernel2(a)(k2, k3, mu23)
        Kb = _field_kernel2(b)(k3, k1, mu31)
        Kc = _field_kernel2(c)(k1, k2, mu12)
        return 2.0 * (Ka * P2 * P3 + Kb * P3 * P1 + Kc * P1 * P2)

    def tree_bispectrum_2ab(k1, k2, k3, mu12, mu23, mu31, P1, P2, P3, a, b):
        return tree_bispectrum_abc(k1, k2, k3, mu12, mu23, mu31, P1, P2, P3, 2, a, b)

    def A_basis_coeffs(r, x):
        # https://arxiv.org/pdf/2007.09011
        # Appendix A2.2 coefficients: B-term, Eq. (A23)
        D = 1.0 + r**2 - 2.0 * r * x
        xm1 = x**2 - 1.0

        bb, bA_only, bB_only, const = {}, {}, {}, {}

        bb[("A", 1, 1, 1)] = r * x
        bB_only[("A", 1, 2, 1)] = -r**2 * (-2.0 + 3.0 * r * x) * xm1 / (2.0 * D)
        bA_only[("A", 1, 1, 2)] = r * x

        bB_only[("A", 2, 2, 1)] = (
            r * (2.0 * x + r * (2.0 - 6.0 * x**2) + r**2 * x * (-3.0 + 5.0 * x**2))
            / (2.0 * D)
        )
        const[("A", 2, 2, 2)] = -r**2 * (-2.0 + 3.0 * r * x) * xm1 / (2.0 * D)
        const[("A", 3, 2, 2)] = (
            r * (2.0 * x + r * (2.0 - 6.0 * x**2) + r**2 * x * (-3.0 + 5.0 * x**2))
            / (2.0 * D)
        )

        bb[("At", 1, 1, 1)] = -r**2 * (-1.0 + r * x) / D
        const[("At", 1, 2, 2)] = -r**2 * (-1.0 + 3.0 * r * x) * xm1 / (2.0 * D)
        const[("At", 2, 2, 2)] = (
            r**2 * (-1.0 + 3.0 * r * x + 3.0 * x**2 - 5.0 * r * x**3)
            / (2.0 * D)
        )

        bA_only[("Ah", 1, 1, 2)] = r**2 * (-1.0 + 3.0 * r * x) * xm1 / (2.0 * D)
        bA_only[("Ah", 2, 1, 2)] = (
            -r**2 * (1.0 - 3.0 * x**2 + r * x * (-3.0 + 5.0 * x**2))
            / (2.0 * D)
        )
        bB_only[("Ah", 2, 2, 1)] = -r**2 * (-1.0 + r * x) / D

        return bb, bA_only, bB_only, const

    def B_basis_coeffs(r, x):
        # Appendix A2.2 coefficients: B-term, Eq. (A23)
        xm1 = x**2 - 1.0

        bb, b, const = {}, {}, {}

        bb[(1, 1, 1)] = (r**2 / 2.0) * xm1
        bb[(2, 1, 1)] = (r / 2.0) * (r + 2.0 * x - 3.0 * r * x**2)

        b[(1, 1, 2)] = (3.0 * r**2 / 16.0) * xm1**2
        b[(1, 2, 1)] = (3.0 * r**4 / 16.0) * xm1**2
        b[(2, 1, 2)] = (3.0 * r / 8.0) * xm1 * (r + 2.0 * x - 5.0 * r * x**2)
        b[(2, 2, 1)] = (3.0 * r**2 / 8.0) * xm1 * (-2.0 + r**2 + 6.0 * r * x - 5.0 * r**2 * x**2)
        b[(3, 1, 2)] = (r / 16.0) * (
            4.0 * x * (3.0 - 5.0 * x**2) + r * (3.0 - 30.0 * x**2 + 35.0 * x**4)
        )
        b[(3, 2, 1)] = (r / 16.0) * (
            -8.0 * x
            + r * (-12.0 + 36.0 * x**2 + 12.0 * r * x * (3.0 - 5.0 * x**2)
                   + r**2 * (3.0 - 30.0 * x**2 + 35.0 * x**4))
        )

        const[(1, 2, 2)] = (5.0 * r**4 / 16.0) * xm1**3
        const[(2, 2, 2)] = (3.0 * r**2 / 16.0) * xm1**2 * (
            -6.0 + 5.0 * r**2 + 30.0 * r * x - 35.0 * r**2 * x**2
        )
        const[(3, 2, 2)] = (3.0 * r / 16.0) * xm1 * (
            -8.0 * x
            + r * (-12.0 + 60.0 * x**2 + 20.0 * r * x * (3.0 - 7.0 * x**2)
                   + 5.0 * r**2 * (1.0 - 14.0 * x**2 + 21.0 * x**4))
        )
        const[(4, 2, 2)] = (r / 16.0) * (
            8.0 * x * (-3.0 + 5.0 * x**2)
            - 6.0 * r * (3.0 - 30.0 * x**2 + 35.0 * x**4)
            + 6.0 * r**2 * x * (15.0 - 70.0 * x**2 + 63.0 * x**4)
            + r**3 * (5.0 - 21.0 * x**2 * (5.0 - 15.0 * x**2 + 11.0 * x**4))
        )

        return bb, b, const

    def one_k(kk):
        Dgeom = 1.0 + r**2 - 2.0 * r * x
        y = _safe_sqrt(Dgeom)

        p = kk * r
        q = kk * y

        Pk, Pp, Pq = Pdd(kk), Pdd(p), Pdd(q)

        mu_pq = (x - r) / y
        mu_qmk = (r * x - 1.0) / y
        mu_mkp = -x

        mu_qp = mu_pq
        mu_pmk = -x
        mu_mkq = mu_qmk

        mu_qmk2 = mu_qmk
        mu_mkp2 = -x
        mu_pq2 = mu_pq

        prefac = kk**3 / (4.0 * jnp.pi**2)

        # A: organize by f^1, f^2, f^3
        A_coeffs = dict(zip(['bb', 'bA', 'bB', '0'], A_basis_coeffs(r, x)))
        A_int = {}

        for a in (1, 2):
            for b in (1, 2):
                fpow = a + b - 1
                Bs = {'A': tree_bispectrum_2ab(p, q, kk, mu_pq, mu_qmk, mu_mkp, Pp, Pq, Pk, a, b),
                      'At': tree_bispectrum_2ab(q, p, kk, mu_qp, mu_pmk, mu_mkq, Pq, Pp, Pk, a, b),
                      'Ah': tree_bispectrum_2ab(q, kk, p, mu_qmk2, mu_mkp2, mu_pq2, Pq, Pk, Pp, a, b)}
                for Aname, B in Bs.items():
                    for n in (1, 2, 3):
                        key = (Aname, n, a, b)
                        for bterm in A_coeffs:
                            if key in A_coeffs[bterm]:
                                A_int[fpow, n, bterm] = A_int.get((fpow, n, bterm), 0.) + A_coeffs[bterm][key] * B

        for key in A_int:
            A_int[key] = prefac * integ_rx(A_int[key])

        # B: organize by f^2, f^3, f^4
        P12_p, P12_q = Pdt(p), Pdt(q)
        P22_p, P22_q = Ptt(p), Ptt(q)

        B_coeffs = dict(zip(['bb', 'b', '0'], B_basis_coeffs(r, x)))
        B_int = {}

        for a in (1, 2):
            for b in (1, 2):
                fpow = a + b
                P_a2_q = P12_q if a == 1 else P22_q
                P_b2_p = P12_p if b == 1 else P22_p
                common = (-1)**fpow * P_a2_q * P_b2_p / Dgeom**a
                for bterm in B_coeffs:
                    for n in (1, 2, 3, 4):
                        key = (n, a, b)
                        if key in B_coeffs[bterm]:
                            B_int[fpow, n, bterm] = B_int.get((fpow, n, bterm), 0.) + common * B_coeffs[bterm][key]

        for key in B_int:
            B_int[key] = prefac * integ_rx(B_int[key])

        return {'A': A_int, 'B': B_int}

    return jax.vmap(lambda kk: one_k(kk))(k)

# Kaiser + A + D + EFT

def fog_damping(*kmu_X, f=1., sigma2v=1., damping='lor'):
    r"""
    Finger-of-God damping kernel W.

    Parameters
    ----------
    kmu_X : tuples
        One ``(k * mu, X_FoG)`` pair per power spectrum leg: two (identical) pairs
        for the auto power spectrum, three for the bispectrum.
    f : float
        Growth rate :math:`f_0` (each ``k * mu`` is multiplied by ``f``).
    sigma2v : float
        Velocity dispersion :math:`\sigma_v^2`.
    damping : {None, 'exp', 'lor', 'vdg'}
        ``None`` returns 1 (no damping).

    Notes
    -----
    With :math:`\lambda_X^2 = \frac{f^2}{2} \sum_i (k_i \mu_i X_i)^2` and
    :math:`\lambda^2 = \frac{f^2}{2} \sum_i (k_i \mu_i)^2`:
    'exp' returns :math:`e^{-\lambda_X^2 \sigma_v^2}`, 'lor' returns
    :math:`1 / (1 + \lambda_X^2 \sigma_v^2)`, and 'vdg' returns
    :math:`e^{-\lambda^2 \sigma_v^2 / (1 + \lambda_X^2)} / (1 + \lambda_X^2)^{n - 3/2}`
    with :math:`n` the number of legs (2 for the power spectrum, 3 for the bispectrum).

    Matches FOLPS's ``fog_damping`` (folps.py).
    """
    if damping is None:
        return 1.
    lX2 = 0.5 * f**2 * sum((kmu * X)**2 for kmu, X in kmu_X)
    if damping == 'lor':
        return 1. / (1. + lX2 * sigma2v)
    if damping == 'exp':
        return jnp.exp(-lX2 * sigma2v)
    if damping == 'vdg':
        l2 = 0.5 * f**2 * sum(kmu**2 for kmu, _ in kmu_X)
        denom = 1. + lX2
        return jnp.exp(-l2 * sigma2v / denom) / denom**(len(kmu_X) - 1.5)
    raise ValueError(f"damping must be None, 'exp', 'lor' or 'vdg', got {damping!r}")


def spectrum2_redshift_tracer_eft(matter, bias, A_B, sigma2v, mu, f,
                                  bias_params, alpha0=0.0, alpha2=0.0, alpha4=0.0, ctilde=0.0, shot=0.0, damping='lor'):
    k = matter['k']
    mu = jnp.atleast_1d(mu)
    mu2 = mu[None, :]**2
    PL = matter['P11']

    fields, bias_params = _format_bias_params(bias_params, nfields=2)
    a, b = fields
    real = spectrum2_real_tracer(matter, bias, bias_params=bias_params)
    bA, bB = bias_params[a]["b1"], bias_params[b]["b1"]
    bb = {'bb': bA * bB, 'bA': bA, 'bB': bB, '0': 1., 'b': bA + bB}

    A = sum(f**fpow * mu2**mu2pow * bb[bterm] * value[:, None] for (fpow, mu2pow, bterm), value in A_B['A'].items())
    B = sum(f**fpow * mu2**mu2pow * bb[bterm] * value[:, None] for (fpow, mu2pow, bterm), value in A_B['B'].items())

    Ps = real['Pdd_ab'][:, None] + f * mu2 * (real['Pdt_a'][:, None] + real['Ptd_b'][:, None]) + f**2 * mu2**2 * real['Ptt'][:, None] + A + B

    PkK_lin_ab = bA * bB * PL[:, None] + f * mu2 * (bA + bB) * PL[:, None] + f**2 * mu2**2 * PL[:, None]
    #alpha0 = bA * bias_params[b]['alpha0'] + bB * bias_params[a]['alpha0']
    #alpha2 = bias_params[a]['alpha2'] + bias_params[b]['alpha2']
    Pct = (alpha0 + alpha2 * mu2 + alpha4 * mu2**2) * (k[:, None]**2) * PL[:, None]
    kmu = k[:, None] * mu[None, :]
    Pnlo = ctilde * (f * kmu)**4 * PkK_lin_ab
    W = fog_damping((kmu, bias_params[a]['X_FoG']), (kmu, bias_params[b]['X_FoG']), f=f, sigma2v=sigma2v, damping=damping)
    # The damping multiplies the whole spectrum, counterterms and stochastic sector included.
    # It is a property of the line-of-sight velocity field, which every term is observed
    # through, not of the perturbative bracket alone.
    return W * (Ps + Pct + Pnlo + shot)


# IR resummation

def compute_sigma2ir(pk_callable, kbao=1.0 / 105.0, integ_k=integration(a=5e-4, b=0.2, size=100)):
    k = integ_k.x()
    x = k / kbao
    pk_now = pk_callable(k)
    j0, j2 = get_spherical_jn(0), get_spherical_jn(2)
    sigma2 = (1.0 / (6.0 * jnp.pi**2)) * integ_k(pk_now * (1.0 - j0(x) + 2.0 * j2(x)))
    sigma2_delta = (1.0 / (2.0 * jnp.pi**2)) * integ_k(pk_now * j2(x))
    return sigma2, sigma2_delta


def _spectrum2_ir_resum(k, mu, pk, pknow, pk_eft, pknow_eft, f, sigma2, sigma2_delta, b1a, b1b=None):
    k = jnp.atleast_1d(k)
    mu = jnp.atleast_1d(mu)
    mu2 = mu[None, :]**2

    sigma2_tot = (1.0 + f * mu2 * (2.0 + f)) * sigma2 + f**2 * mu2 * (mu2 - 1.0) * sigma2_delta
    damp = jnp.exp(-(k[:, None]**2) * sigma2_tot)
    wiggles = pk - pknow
    Kcross = (b1a + f * mu2) * (b1b + f * mu2)
    return damp * pk_eft + (1.0 - damp) * pknow_eft + damp * Kcross * wiggles[:, None] * (k[:, None]**2) * sigma2_tot


# High-level wrapper

def prepare_spectrum2_redshift_tracer(k, pk_callable, pknow_callable, kbao=1.0 / 105.):
    sigma2v = compute_sigma2v(pk_callable)
    sigma2, sigma2_delta = compute_sigma2ir(pk_callable, kbao=kbao)
    matter = compute_spt_matter_1loop(k, pk_callable)
    bias = compute_bias_terms_1loop(k, pk_callable)
    A_B = compute_tns_A_B_terms(k, pk_callable)
    table = dict(matter=matter, bias=bias, A_B=A_B, sigma2=sigma2, sigma2_delta=sigma2_delta, sigma2v=sigma2v)
    matter = compute_spt_matter_1loop(k, pknow_callable)
    bias = compute_bias_terms_1loop(k, pknow_callable)
    A_B = compute_tns_A_B_terms(k, pknow_callable)
    table_now = dict(matter=matter, bias=bias, A_B=A_B)
    return table, table_now


def spectrum2_redshift_tracer(mu_or_kvec, table, table_now, f, bias_params, **ct_params):
    """
    Evaluate the redshift-space power spectrum.

    Parameters
    ----------
    mu_or_kvec : array
        1-D array of cos(theta) values *or* an array of shape ``(..., 3)``
        containing 3-D wavevectors.  In the kvec case the function projects
        P(k, mu) onto multipoles and reconstructs P(kvec) via Legendre
        expansion, supporting arbitrary leading dimensions.
    """
    mu_or_kvec = jnp.asarray(mu_or_kvec)
    is_kvec = mu_or_kvec.ndim >= 2 and mu_or_kvec.shape[-1] == 3

    if is_kvec:
        kvec = mu_or_kvec
        orig_shape = kvec.shape[:-1]
        knorm = jnp.sqrt(jnp.sum(kvec**2, axis=-1))
        mu_kvec = jnp.where(knorm > 0., kvec[..., 2] / knorm, 0.)
        ells = list(range(0, 8, 2))
        to_poles = ProjectToPoles(mu=10, ells=ells)
        mu = to_poles.mu  # evaluate EFT on integration nodes, not on kvec mu values
    else:
        mu = mu_or_kvec

    fields, bias_params = _format_bias_params(bias_params, nfields=2)
    a, b = fields
    pk_eft = spectrum2_redshift_tracer_eft(table['matter'], table['bias'], table['A_B'], table['sigma2v'], mu, f, bias_params, **ct_params)
    pknow_eft = spectrum2_redshift_tracer_eft(table_now['matter'], table_now['bias'], table_now['A_B'], table['sigma2v'], mu, f, bias_params, **ct_params)
    pk_ir = _spectrum2_ir_resum(table['matter']['k'], mu, table['matter']['P11'], table_now['matter']['P11'], pk_eft, pknow_eft, f, table['sigma2'], table['sigma2_delta'], bias_params[a]['b1'], bias_params[b]['b1'])

    if is_kvec:
        k_table = table['matter']['k']
        poles = to_poles(pk_ir)  # (n_ells, nk_table)
        # vmap over ells; each pole has shape (nk_table,) -> interp to (N_flat,)
        pole_at_k = jax.vmap(lambda pole: jnp.interp(knorm.ravel(), k_table, pole))(poles)  # (n_ells, N_flat)
        pk_ir = sum(pole_at_k[i].reshape(orig_shape) * get_legendre(ell)(mu_kvec)
                    for i, ell in enumerate(ells))

    return pk_ir


class ProjectToPoles:

    """Helper class to compute multipoles using Legendre polynomials."""

    def __init__(self, ells=(0, 2, 4), mu=8):
        self.ells = list(ells)
        integ = integration(a=0, b=1, size=mu)
        self.mu, w = integ.x(), integ.w
        self.w = jnp.array([w * (2 * ell + 1) * get_legendre(ell)(self.mu) for ell in self.ells])

    def __call__(self, f):
        return jnp.sum(f * self.w[(slice(None),) + (None,) * (f.ndim - 1) + (slice(None),)], axis=-1)


class ProjectToSell:

    """Helper class to compute Sugiyama multipoles using Legendre polynomials."""

    def __init__(self, ells=((0, 0, 0), (2, 0, 2)), size=6):
        self.ells = [tuple(ell) for ell in ells]
        # Integrate over (mu, x, phi): mu the line-of-sight cosine of k1hat, x = k1hat . k2hat the
        # triangle shape, and phi the azimuth of k2hat about k1hat. Since the solid angle element
        # is unchanged, dOmega2 = dmu2 dphi2 = dx dphi, this is the same measure (and the same
        # normalization) as parameterizing k2hat by its own line-of-sight angles -- but carrying x
        # as a *direct* integration variable resolves the 1 / k3 structure at the folded
        # configuration k1 ~ -k2 far better, and that is what limits convergence. Deriving x from
        # (mu1, mu2, phi2) instead smears it over all three variables: at size=6 that costs 10%
        # on B000 and 11% on B202 by k = 0.2. This is FOLPS's parameterization
        # (folps.py, Sugiyama_Bell, precision=[Nphi, Nx, Nmu]).
        integ = IntegralND(mu=integration(-1., 1., size=size),
                           x=integration(-1., 1., size=size),
                           phi=integration(0., 2. * np.pi, size=size))

        def get_N(ell1, ell2, ell3):
            return (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1)

        def get_H(ell1, ell2, ell3):
            return wigner_3j(ell1, ell2, ell3, 0, 0, 0)

        mu, x, phi = (a.ravel() for a in integ.x(['mu', 'x', 'phi'], sparse=False))
        zero, one = np.zeros_like(mu), np.ones_like(mu)
        smu = np.sqrt(np.clip(1. - mu**2, 0., None))
        k1hat = np.stack([smu, zero, mu], axis=-1)
        # Orthonormal frame about k1hat, with e3 perpendicular to the line of sight
        e2 = np.stack([-mu, zero, smu], axis=-1)
        e3 = np.stack([zero, -one, zero], axis=-1)
        sx = np.sqrt(np.clip(1. - x**2, 0., None))
        k2hat = x[:, None] * k1hat + sx[:, None] * (np.cos(phi)[:, None] * e2 + np.sin(phi)[:, None] * e3)

        # Normalized angular measure dmu / 2 * dx / 2 * dphi / (2 pi), so that a constant
        # bispectrum projects to itself in the (0, 0, 0) multipole (raw integ.w sums to 8 pi)
        w = integ.w.ravel() / (8. * np.pi)
        self.k1hat, self.k2hat = k1hat, k2hat
        self.w = np.array([w * get_N(*ell) * get_H(*ell)**2 * get_S(ell, z3=True)(k1hat, k2hat) for ell in self.ells])

    def __call__(self, f):
        return jnp.sum(f * self.w[(slice(None),) + (None,) * (f.ndim - 1) + (slice(None),)], axis=-1)



# Note: shot noise formula isn't correct for multitracer, but let's marginalize over it anyway...

def spectrum3_redshift_tracer(k1vec, k2vec, pk_callable, pknow_callable, f, bias_params, shot=0., sigma2v=None, sigma2=None, sigma2_delta=None, damping='lor', srnl=None):
    """
    JAX-friendly cross-bispectrum model B^{abc}(k1, k2, k3), with k3 = -k1 - k2.

    Parameters
    ----------
    k1vec, k2vec : array_like[..., 3]
        Wavevectors of the first two bispectrum legs.
    fields : tuple
        Tuple (a, b, c) of field identifiers.
    f, sigma2v, sigma2, sigma2_delta : float
        RSD / IR / FoG parameters.
    bias_params : dict
        Mapping field -> parameters.
        Each entry can be either a dict with keys
        ('b1', 'b2', 'bs', 'c1', 'c2', 'snb0', 'sn0', 'X_FoG')
        or a tuple/list in that order.
    pk_callable : callable
        Callable returning the linear power spectrum P(k) for a given k.
    pknow_callable : callable
        Callable returning the no-wiggle linear power spectrum P_now(k) for a given k.
    damping : str, default='lor'
        One of ('lor', 'exp', 'vdg').
    shot : float, default=0.
        Constant shot-noise contribution.

    Returns
    -------
    bispectrum : array_like
        Modeled bispectrum B^{abc}(k1, k2, k3).
    """
    fields, bias_params = _format_bias_params(bias_params, nfields=3)
    a, b, c = fields

    if sigma2v is None:
        sigma2v = compute_sigma2v(pk_callable)
    if sigma2 is None or sigma2_delta is None:
        sigma2, sigma2_delta = compute_sigma2ir(pknow_callable)

    def _get_bias_params(field, names=None):
        pars = bias_params[field]
        if names is None:
            names = ['b1', 'b2', 'bs', 'c1', 'c2', 'snb0', 'sn0', 'X_FoG']
        if isinstance(names, str):
            return pars[names]
        return [pars[name] for name in names]

    def _norm(kvec):
        return jnp.sqrt(jnp.sum(kvec**2, axis=-1))

    def _mu(kvec, knorm):
        return jnp.where(knorm > 0., kvec[..., 2] / knorm, 0.)

    def _xcos(kivec, kjvec, ki, kj):
        denom = ki * kj
        return jnp.where(denom > 0., jnp.sum(kivec * kjvec, axis=-1) / denom, 0.)

    def _safe_div(num, denom):
        # Drops the contribution at denom == 0 -- e.g. ki == kj and xij ==
        # -1 (anti-parallel, equal-magnitude legs) is a genuine, valid
        # squeezed/folded configuration (ki+kj -> 0), not an error; avoids
        # 0/0 -> NaN there, matching spectrum4_redshift_tracer's guard.
        return jnp.where(denom > 0., num / jnp.where(denom > 0., denom, 1.), 0.)

    def _Z1(field, mu):
        b1 = _get_bias_params(field, 'b1')
        return b1 + f * mu**2

    def _srnl_Z1(field, k, mu):
        """The linear kernel with short-range non-locality: b_d(k, mu) + f mu^2 b_e(k, mu).

        Same series, and the same coefficients, as the power spectrum uses -- it is the same
        bias field -- so this adds no parameters of its own.
        """
        from math import factorial
        b1 = _get_bias_params(field, 'b1')
        R, Rpar, order = srnl['R'], srnl['Rpar'], srnl['order']
        x = (k * R)**2 / 2.
        y = (k * mu * Rpar)**2 / 2.
        ij = [(i, j) for n in range(1, order + 1) for i in range(n, -1, -1)
              for j in [n - i]]
        sd = se = 0.
        for m, (i, j) in enumerate(ij):
            w = x**i * y**j / (factorial(i) * factorial(j))
            sd = sd + srnl['bd'][m] * w
            se = se + srnl['be'][m] * w
        return b1 * (1. + sd) + f * mu**2 * (1. + se)

    def _Z1eft(field, k, mu):
        # Bispectrum EFT counterterms, on the linear kernel: matches FOLPS
        # (folps.py, Z1eft1/2/3), which uses this same (c1 mu^2 + c2 mu^4) k^2 basis and sign.
        # These are *independent* of the power spectrum counterterms alpha0 / alpha2 / alpha4
        # (see spectrum2_redshift_tracer_eft), again as in FOLPS: P and B carry their own sets.
        c1, c2 = _get_bias_params(field, ['c1', 'c2'])
        base = _Z1(field, mu) if srnl is None else _srnl_Z1(field, k, mu)
        return base - (c1 * mu**2 + c2 * mu**4) * k**2

    def _Z2(field, ki, kj, xij, mui, muj):
        b1, b2, bs = _get_bias_params(field, ['b1', 'b2', 'bs'])
        km = ki * mui + kj * muj
        term1 = b2 / 2. + bs / 2. * (xij**2 - 1. / 3.)
        term2 = km / 2. * (_safe_div(mui, ki) * f * (b1 + f * muj**2) + _safe_div(muj, kj) * f * (b1 + f * mui**2))
        F2 = 5. / 7. + xij / 2. * (_safe_div(ki, kj) + _safe_div(kj, ki)) + 2. / 7. * xij**2
        G2 = 3. / 7. + xij / 2. * (_safe_div(ki, kj) + _safe_div(kj, ki)) + 4. / 7. * xij**2
        term3 = b1 * F2
        mu2 = _safe_div(km**2, ki**2 + kj**2 + 2. * ki * kj * xij)
        term4 = f * mu2 * G2
        return term1 + term2 + term3 + term4

    def _IR_pk(k, mu):
        pk = pk_callable(k)
        pknw = pknow_callable(k)
        eIR = (1. + f * mu**2 * (2. + f)) * sigma2 + (f * mu)**2 * (mu**2 - 1.) * sigma2_delta
        return pknw + (pk - pknw) * jnp.exp(-eIR * k**2)

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

    X1, X2, X3 = [_get_bias_params(field, 'X_FoG') for field in fields]
    W = fog_damping((k1 * mu1, X1), (k2 * mu2, X2), (k3 * mu3, X3), f=f, sigma2v=sigma2v, damping=damping)

    def _shot_leg(field, k, mu, Z1eft, pkIR):
        b1, snb0, sn0 = _get_bias_params(field, ['b1', 'snb0', 'sn0'])
        return (b1 * snb0 + 2. * sn0 * f * mu**2) * Z1eft * pkIR

    legs = (_shot_leg(a, k1, mu1, Z1eft1, pkIR1) + _shot_leg(b, k2, mu2, Z1eft2, pkIR2)
            + _shot_leg(c, k3, mu3, Z1eft3, pkIR3))

    # The damping multiplies the whole spectrum, the stochastic legs and the fully coincident
    # partition included. `shot**2` used to be held out on the grounds that two points at the
    # same place have no relative displacement to be smeared -- but they are still observed
    # through the same line-of-sight velocity field as everything else.
    return W * (B12 + B23 + B31 + legs + shot**2)




# ======================================================================================
# Tree-level n-point spectra, n = 2 .. 6, from one labelled-tree construction
#
# This is what Sugiyama, Saito, Beutler & Seo (arXiv:1908.06234) Appendix A writes out term by
# term -- P (A3), B (A5), T (A6-A7), P5 (A8-A9), P6 (A11-A12). Rather than transcribe five lists
# of permutations, all five come out of one statement about labelled trees: at tree level the
# connected n-point spectrum is the sum over all labelled trees on n nodes, one node per external
# leg, each edge a linear propagator,
#
#     P_n(k_1 .. k_n) = sum_trees [prod_i (deg i)!] prod_i Z_{deg i}(..) prod_e P_L(|Q_e|)
#
# with, at node i and incident edge e, the argument -sum_{j in C} k_j where C is the component
# that edge detaches from i. The prod (deg i)! is the leg-pairing multiplicity.
#
# Two things this construction settles that the paper gets wrong, both checked rather than
# assumed: Eq. (A11) is missing a topology at n = 6 (the six labelled stars K_{1,5}, prefactor
# 5!, giving 6^4 = 1296 trees rather than 1290), and Eq. (A9)'s multiplicities at n = 5 are 40
# and 36 against Cayley's 60 and 60. ``paper=True`` reproduces the paper rather than the right
# answer.
#
# F_n and G_n come from the standard recursion, evaluated by dynamic programming over the 2^n
# subsets of the leg set rather than by summing n! orderings. Z_n comes from expanding
# (1 + delta) e^{i f k mu u} and reading off which legs go to the density factor and which to the
# velocity factors. These are the same kernels `spectrum3_redshift_tracer` and
# `spectrum4_redshift_tracer` above write out by hand at n = 3 and 4, and they agree with them --
# which is what `validate_partition_bias` and `tests/test_cov3.py` pin.
# ======================================================================================

#: An edge momentum below this fraction of the largest external leg is treated as identically
#: zero. See the note in :func:`make_spectrum_redshift_tracer`; override for a convergence check.
_ZERO_EDGE = 1e-9


# --------------------------------------------------------------------------------------
# vector helpers. The guards return 0 where a magnitude vanishes; this is the convention
# `jaxpower.pt` uses and the limit the kernels take once the propagators are included.
# --------------------------------------------------------------------------------------

def _norm(v):
    return jnp.sqrt(jnp.sum(v**2, axis=-1))


def _safe_div(num, den):
    return jnp.where(den > 0., num / jnp.where(den > 0., den, 1.), 0.)


def _mu(v, k=None):
    if k is None:
        k = _norm(v)
    return _safe_div(v[..., 2], k)


def _alpha(v1, v2):
    r""":math:`\alpha(k_1, k_2) = 1 + k_1 \cdot k_2 / k_1^2`."""
    return 1. + _safe_div(jnp.sum(v1 * v2, axis=-1), jnp.sum(v1**2, axis=-1))


def _beta(v1, v2):
    r""":math:`\beta(k_1, k_2) = (k_1 \cdot k_2) |k_1 + k_2|^2 / (2 k_1^2 k_2^2)`."""
    den = 2. * jnp.sum(v1**2, axis=-1) * jnp.sum(v2**2, axis=-1)
    return _safe_div(jnp.sum(v1 * v2, axis=-1) * jnp.sum((v1 + v2)**2, axis=-1), den)


def _sigma2(v1, v2):
    r"""The second Galileon's angular kernel, :math:`(\hat k_1 \cdot \hat k_2)^2 - 1`.

    This is ``jaxpower.pt``'s ``angK``, with its guard convention: a vanishing magnitude gives
    :math:`-1`, not a nan.
    """
    return _safe_div(jnp.sum(v1 * v2, axis=-1), _norm(v1) * _norm(v2))**2 - 1.


# --------------------------------------------------------------------------------------
# F_n, G_n by dynamic programming over subsets
# --------------------------------------------------------------------------------------

def _subsets(mask):
    """Proper non-empty submasks of ``mask``."""
    sub = (mask - 1) & mask
    while sub:
        yield sub
        sub = (sub - 1) & mask


def fg_tables(vs):
    r"""Symmetrised :math:`F_n` and :math:`G_n` for every subset of ``vs``.

    Returns ``(F, G, ksum)``, dicts keyed by bitmask over ``range(len(vs))``. ``F[mask]`` is
    :math:`F^{(s)}_{|mask|}` evaluated on the legs in ``mask``, ``ksum[mask]`` their vector sum.

    The recursion is

    .. math::
        F^{(s)}_n(S) = \frac{1}{(2n + 3)(n - 1)} \sum_{\emptyset \neq A \subsetneq S}
            \frac{G^{(s)}_{|A|}(A)}{\binom{n}{|A|}}
            \Big[(2n + 1)\, \alpha(k_A, k_{S \setminus A}) F^{(s)}(S \setminus A)
                 + 2\, \beta(k_A, k_{S \setminus A}) G^{(s)}(S \setminus A)\Big]

    and the same with :math:`(3, 2n)` in place of :math:`(2n + 1, 2)` for :math:`G`. The
    :math:`1/\binom{n}{|A|}` is what turns the unsymmetrised recursion into the symmetric kernel:
    of the :math:`n!` orderings, :math:`|A|! (n - |A|)!` give the same split.
    """
    m = len(vs)
    ksum = {}
    for mask in range(1, 1 << m):
        low = mask & -mask
        rest = mask ^ low
        ksum[mask] = vs[low.bit_length() - 1] if not rest else ksum[rest] + vs[low.bit_length() - 1]
    F, G = {}, {}
    for i in range(m):
        F[1 << i] = G[1 << i] = 1.
    for mask in range(1, 1 << m):
        n = bin(mask).count('1')
        if n < 2:
            continue
        accF = accG = 0.
        for sub in _subsets(mask):
            comp = mask ^ sub
            a, b = _alpha(ksum[sub], ksum[comp]), _beta(ksum[sub], ksum[comp])
            w = G[sub] / math.comb(n, bin(sub).count('1'))
            accF = accF + w * ((2 * n + 1) * a * F[comp] + 2. * b * G[comp])
            accG = accG + w * (3. * a * F[comp] + 2 * n * b * G[comp])
        norm = float((2 * n + 3) * (n - 1))
        F[mask], G[mask] = accF / norm, accG / norm
    return F, G, ksum


# --------------------------------------------------------------------------------------
# Z_n
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _partitions(m):
    """Every ``(density_mask, (velocity_mask, ...))`` split of ``range(m)``.

    The velocity blocks are returned sorted, so each unordered collection appears once.
    """
    out = []
    full = (1 << m) - 1
    for dens in range(1 << m):
        rest = full ^ dens
        out += [(dens, tuple(sorted(blocks))) for blocks in _set_partitions(rest)]
    return out


def _set_partitions(mask):
    """Every partition of the bits of ``mask`` into non-empty blocks (as tuples of masks)."""
    if mask == 0:
        return [()]
    low = mask & -mask
    rest = mask ^ low
    out = []
    # `low` either starts its own block or joins one of the blocks of a partition of `rest`.
    for part in _set_partitions(rest):
        out.append(part + (low,))
        for i in range(len(part)):
            out.append(part[:i] + (part[i] | low,) + part[i + 1:])
    return out


class BiasModel(object):
    r"""The real-space density kernels :math:`K_a` entering :math:`Z_n`.

    ``K_1 = b1``; ``K_2 = b1 F_2 + b2 / 2 + bs / 2 (x^2 - 1/3)``; ``K_3`` adds the second-order
    bias operators in the form ``jaxpower.pt`` carries them (Philcox's ``b2 F_2 + 2 g2 angK G_2``
    averaged over orderings, with ``g2 = bs / 2`` and ``b2`` shifted by ``2 bs / 3``).

    Above third order the same second-order sector is carried in full -- both ``delta^2`` and the
    tidal ``G_2``, so ``b2`` and ``bs`` reach ``K_4`` and ``K_5`` -- at no new parameter, see
    :meth:`__call__`. There is no knob for this. Truncating to ``K_a = b1 F_a`` above third order,
    which is what the paper does, is not a smaller model but an inconsistent one: it keeps the
    fitted ``b2`` and ``bs`` where the expansion happens to name them and throws them away where
    it does not. The two truncations also do not differ by a normalisation -- they pull generic
    and degenerate configurations in opposite directions, which is exactly what makes a
    configuration-dependent excess impossible to attribute when the truncation is a free choice.

    The paper's own model is linear bias, ``b2 = bs = 0``, and then ``K_a = b1 F_a`` at every
    order regardless; :attr:`linear` short-circuits to that. The quadratic terms are here so the
    same construction can be validated against ``jaxpower.pt``'s tree bispectrum and trispectrum,
    which carry them.
    """

    def __init__(self, b1=1., b2=0., bs=0.):
        self.b1, self.b2, self.bs = b1, b2, bs
        # The linear short cut is a python branch, so it has to be decided once, here. A jax
        # tracer cannot be compared with a float at all -- which is what a fit differentiating
        # through these kernels supplies -- so a traced bias takes the general branch instead.
        # That reduces to the same expression when b2 = bs = 0, at the cost of evaluating it.
        try:
            self._linear = bool(float(b2) == 0.) and bool(float(bs) == 0.)
        except Exception:
            self._linear = False

    @property
    def linear(self):
        return self._linear

    def __call__(self, mask, F, G, ksum, vs):
        a = bin(mask).count('1')
        if a == 0:
            return 1.
        out = self.b1 * F[mask]
        if a == 1 or self.linear:
            return out
        idx = [i for i in range(len(vs)) if mask >> i & 1]
        b2p, g2 = self.b2 + 2. / 3. * self.bs, self.bs / 2.
        if a == 2:
            i, j = idx
            x = _safe_div(jnp.sum(vs[i] * vs[j], axis=-1), _norm(vs[i]) * _norm(vs[j]))
            return out + b2p / 2. + g2 * (x**2 - 1.)
        if a == 3:
            # Philcox's K3 for one ordering, averaged over the 3! orderings. The genuinely
            # third-order bias operators (b3, g3, g21) are absent from this basis and are zero.
            acc = 0.
            for p in itertools.permutations(idx):
                m12 = (1 << p[0]) | (1 << p[1])
                m23 = (1 << p[1]) | (1 << p[2])
                acc = acc + b2p * F[m12] + 2. * g2 * _sigma2(vs[p[0]], ksum[m23]) * G[m23]
            return out + acc / 6.
        # A second-order operator is a product of exactly two fields, so at every order it is
        # one sum over the two-block partitions of the legs -- the b2 and b_K^2 already in K_2
        # and K_3 reappear in K_4 and K_5 at no new parameter. Symmetrising over the a! orderings,
        #
        #     K_a |_2nd = (1 / a!) sum_{A|B} |A|! |B|!
        #                 [ b2' F_{|A|}(A) F_{|B|}(B)
        #                   + 2 g2 sigma^2(k_A, k_B) G_{|A|}(A) G_{|B|}(B) ]
        #
        # The tidal term takes G, not F: g2 here is the coefficient of G_2[Phi_v], as Philcox's
        # `2 g2 angK Gn2` in K_3 fixes it, with Gamma_3 in the third-order block that is zero.
        # Three blocks would need genuinely third-order operators this model does not have.
        # `validate_partition_bias` checks all of it against the explicit K_2 and K_3 above.
        loc = tid = 0.
        for part in _set_partitions(mask):
            if len(part) != 2:
                continue
            ba, bb = part
            w = float(math.factorial(bin(ba).count('1')) * math.factorial(bin(bb).count('1')))
            loc = loc + w * F[ba] * F[bb]
            tid = tid + w * _sigma2(ksum[ba], ksum[bb]) * G[ba] * G[bb]
        return out + (b2p * loc + 2. * g2 * tid) / math.factorial(a)


def make_Zn(n, f=0., bias=None):
    r"""Return ``Z(vs)``, the symmetric tree-level redshift-space kernel of order ``n``.

    ``vs`` is a sequence of ``n`` arrays of shape ``(..., 3)``; the line of sight is ``z``.
    """
    bias = bias if isinstance(bias, BiasModel) else BiasModel(**(bias or {}))
    parts = _partitions(n)
    nfact = float(math.factorial(n))

    def Z(vs):
        vs = list(vs)
        assert len(vs) == n
        F, G, ksum = fg_tables(vs)
        full = (1 << n) - 1
        ktot = ksum[full]
        knorm = _norm(ktot)
        fkmu = f * ktot[..., 2]        # f k mu, with mu = khat . zhat
        out = 0.
        for dens, vels in parts:
            w = math.factorial(bin(dens).count('1'))
            for v in vels:
                w *= math.factorial(bin(v).count('1'))
            term = (w / nfact) * bias(dens, F, G, ksum, vs)
            for v in vels:
                term = term * _safe_div(ksum[v][..., 2], _norm(ksum[v])**2) * G[v]
                term = term * fkmu
            out = out + term
        del knorm
        return out

    return Z


# --------------------------------------------------------------------------------------
# labelled trees
# --------------------------------------------------------------------------------------

def labelled_trees(n):
    """All ``n ** (n - 2)`` labelled trees on ``range(n)``, as edge lists (Prufer decoding)."""
    if n == 1:
        return [[]]
    if n == 2:
        return [[(0, 1)]]
    out = []
    for seq in itertools.product(range(n), repeat=n - 2):
        deg = [1] * n
        for x in seq:
            deg[x] += 1
        leaves = [i for i in range(n) if deg[i] == 1]
        heapq.heapify(leaves)
        edges, dd = [], list(deg)
        for x in seq:
            leaf = heapq.heappop(leaves)
            edges.append((leaf, x))
            dd[x] -= 1
            if dd[x] == 1:
                heapq.heappush(leaves, x)
        u, v = heapq.heappop(leaves), heapq.heappop(leaves)
        edges.append((u, v))
        out.append(edges)
    return out


def _degrees(edges, n):
    d = [0] * n
    for (u, v) in edges:
        d[u] += 1
        d[v] += 1
    return d


def _component(edges, drop, start, n):
    """Nodes reachable from ``start`` once edge index ``drop`` is removed."""
    adj = {i: [] for i in range(n)}
    for e, (u, v) in enumerate(edges):
        if e == drop:
            continue
        adj[u].append(v)
        adj[v].append(u)
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return sorted(seen)


def tree_classes(n):
    """Group the labelled trees by shape.

    Returns ``[(representative_edges, [sigma, ...]), ...]`` where ``sigma[i]`` is the external
    leg sitting at node ``i`` of the representative. One traced body per class then covers every
    labelling of that class, which is what keeps the :math:`n = 6` graph (1296 trees, 6 shapes)
    compilable.
    """
    reps, groups = [], []
    for edges in labelled_trees(n):
        eset = {frozenset(e) for e in edges}
        placed = False
        for idx, rep in enumerate(reps):
            rset = {frozenset(e) for e in rep}
            for perm in itertools.permutations(range(n)):
                if {frozenset((perm[u], perm[v])) for (u, v) in edges} == rset:
                    sigma = [0] * n
                    for u in range(n):
                        sigma[perm[u]] = u
                    groups[idx].append(sigma)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            reps.append(edges)
            groups.append([list(range(n))])
        del eset
    return list(zip(reps, groups))


# --------------------------------------------------------------------------------------
# the n-point spectra
# --------------------------------------------------------------------------------------

def make_spectrum_redshift_tracer(n, pk_callable, f=0., bias=None, paper=False, jit=True):
    r"""Return ``P_n(*legs)``, the connected tree-level redshift-space :math:`n`-point spectrum.

    ``legs`` are ``n`` arrays of shape ``(..., 3)`` summing to zero.

    ``paper=True`` reproduces Sugiyama's own term list rather than the complete one: at
    :math:`n = 6` it drops the six labelled stars :math:`K_{1,5}` that Eq. (A11) omits. It
    changes nothing at :math:`n \leq 5`, where every topology is listed (Eq. A9's *multiplicities*
    are wrong -- 40 and 36 against 60 and 60 -- but Cayley's counts are used either way).
    """
    bias = bias if isinstance(bias, BiasModel) else BiasModel(**(bias or {}))
    Z = {m: make_Zn(m, f=f, bias=bias) for m in range(1, n)}

    def make_body(edges):
        degs = _degrees(edges, n)
        comps = [_component(edges, e, v, n) for e, (u, v) in enumerate(edges)]
        incid = [[(e, -1. if i == u else 1.) for e, (u, v) in enumerate(edges) if i in (u, v)]
                 for i in range(n)]
        mult = float(reduce(lambda x, y: x * y, [math.factorial(d) for d in degs], 1))

        def body(kc):
            Q = [reduce(lambda x, y: x + y, [kc[m] for m in comp]) for comp in comps]
            # An internal line whose momentum vanishes IDENTICALLY -- the edge separating the two
            # closed triangles of a Cov[B, B] configuration, or the (k, -k) pair of a Cov[P, B]
            # one -- must be snapped to zero. Summing the legs in a different order leaves a
            # round-off residual of order 1e-16 k, and the vertex kernels at each end diverge as
            # 1/q, so that residual is amplified by (k/q)^2 ~ 1e32: measured, the tree P6 on two
            # closed triangles came out 1e38 instead of 1e22, with the three shape classes that
            # can carry such an edge cancelling against each other at that absurd scale. The
            # physical value is zero -- a periodic box has no k = 0 mode, so P_lin(0) = 0 -- and
            # the limit is clean, because a subtree whose external momenta sum to zero has soft
            # coefficient (k_A . q)/q^2 = 0. The cut is far below any configuration with weight:
            # the angular measure near a genuine degeneracy goes as q^2.
            scale = reduce(jnp.maximum, [_norm(k) for k in kc])
            Q = [jnp.where((_norm(q) > _ZERO_EDGE * scale)[..., None], q, 0.) for q in Q]
            term = mult
            for i in range(n):
                term = term * Z[degs[i]]([s * Q[e] for (e, s) in incid[i]])
            for q in Q:
                term = term * pk_callable(_norm(q))
            return term

        return body

    classes = tree_classes(n)
    if paper and n >= 6:
        classes = [(rep, sig) for rep, sig in classes if max(_degrees(rep, n)) < n - 1]
    tables = [(make_body(rep), np.asarray(sig, dtype=np.int32)) for rep, sig in classes]
    ntree = sum(len(sig) for _, sig in classes)

    def _P(q):
        out = 0.
        for body, sig in tables:
            if len(sig) == 1:
                out = out + body([q[i] for i in sig[0]])
                continue

            def step(carry, row, _b=body):
                return carry + _b([jnp.take(q, row[i], axis=0) for i in range(n)]), None

            acc, _ = jax.lax.scan(step, jnp.zeros(q.shape[1:-1]), jnp.asarray(sig))
            out = out + acc
        return out

    _P = jax.jit(_P) if jit else _P

    def P(*legs):
        """The ``n - 1`` independent legs; the last closes the sum.

        This is the convention of :func:`spectrum3_redshift_tracer` and
        :func:`spectrum4_redshift_tracer` above, which take two and three wavevectors for the
        bispectrum and trispectrum: a connected n-point lives on ``sum k_i = 0``, so the closing
        leg is not an independent argument and passing it would invite the two to disagree.
        """
        assert len(legs) == n - 1, f'the connected {n}-point takes its {n - 1} independent legs'
        legs = [jnp.asarray(v) for v in legs]
        legs = legs + [-reduce(lambda a, b: a + b, legs)]
        return _P(jnp.stack(jnp.broadcast_arrays(*legs)))

    P.ntree = ntree
    P.nclass = len(classes)
    # A list, not a dict: two distinct shapes share the degree sequence (3, 2, 2, 1, 1, 1) at
    # n = 6 -- the paper's T322111a and T322111b -- and a dict would silently merge them.
    P.shapes = [(tuple(sorted(_degrees(rep, n), reverse=True)),
                 int(reduce(lambda x, y: x * y, [math.factorial(d) for d in _degrees(rep, n)], 1)),
                 len(sig)) for rep, sig in classes]
    return P


def with_fog_damping(fun, n, f, X_FoG, sigma2v, damping='lor'):
    r"""Wrap a connected ``n``-point in the Fingers-of-God factor ``jaxpower.pt`` gives ``B``
    and ``T``.

    ``spectrum3_redshift_tracer`` multiplies the whole tree bispectrum by
    ``fog_damping((k_i mu_i, X_i) for each of its three legs)`` and
    ``spectrum4_redshift_tracer`` does the same over four; the tree spectra built here carry
    none. Mixing the two in one covariance is not a small inconsistency at :math:`k \sim 0.3`:
    the 2-, 3- and 4-point are suppressed while the 5- and 6-point are not, so every correlation
    the latter carry is inflated relative to the variances that normalise it.

    Measured on the 500 Abacus LRG boxes, supplying it takes the full :math:`k < 0.3` covariance
    from a smallest correlation eigenvalue of -0.275 to **+0.045** -- positive definite over the
    whole range -- and brings the P-vs-B canonical correlation onto the mocks' at every cut. It
    introduces no parameter: ``X_FoG`` and ``sigma2v`` are the ones the bispectrum fit already
    used. The cost is that :math:`\sigma(B_{000})` falls from 0.955 to 0.874 of the mocks'.

    ``k mu`` is the leg's line-of-sight component, so no extra geometry is needed.
    """
    def wrapped(*legs):
        # The damping kernel needs every leg's line-of-sight component, the closing one included;
        # the spectrum itself takes only the independent ones. Handing `fun` the closed list is
        # the mistake to avoid here.
        legs = [jnp.asarray(v) for v in legs]
        closed = legs + [-reduce(lambda a, b: a + b, legs)]
        W = fog_damping(*[(leg[..., 2], X_FoG) for leg in closed], f=f, sigma2v=sigma2v,
                        damping=damping)
        return W * fun(*legs)

    return wrapped


def validate_partition_bias(atol=1e-12, size=200, seed=0):
    """The two-block partition sum must reproduce this module's own K_2 and K_3.

    Both were validated against ``jaxpower.pt``'s Z_2 and Z_3, so they are the reference, and the
    sum is what continues them into K_4 and K_5. It is checked here term by term -- the local
    part alone with ``bs = 0``, the tidal part alone with ``b2' = 0``, and the two together --
    because the two failures the continuation invites cancel in neither: dropping the block
    factorials leaves the local part short by 2/3 at n = 3, and putting F where G belongs leaves
    the tidal part wrong at n = 3 while both stay exact at n = 2.
    """
    rng = np.random.default_rng(seed)
    b1 = 2.0
    out = {}
    # (b2, bs) with, in turn, no tidal operator, no local one (b2' = b2 + 2 bs / 3 = 0), both.
    cases = {'local': (1.0, 0.), 'tidal': (0.8, -1.2), 'both': (1.0, -0.7)}
    for n in (2, 3):
        vs = [jnp.asarray(rng.normal(scale=0.1, size=(size, 3))) for _ in range(n)]
        F, G, ksum = fg_tables(vs)
        mask = (1 << n) - 1
        loc = tid = 0.
        for part in _set_partitions(mask):
            if len(part) != 2:
                continue
            ba, bb = part
            w = float(math.factorial(bin(ba).count('1')) * math.factorial(bin(bb).count('1')))
            loc = loc + w * np.asarray(F[ba]) * np.asarray(F[bb])
            tid = tid + w * np.asarray(_sigma2(ksum[ba], ksum[bb])) \
                * np.asarray(G[ba]) * np.asarray(G[bb])
        lin = np.asarray(BiasModel(b1=b1)(mask, F, G, ksum, vs))
        for name, (b2, bs) in cases.items():
            ref = np.asarray(BiasModel(b1=b1, b2=b2, bs=bs)(mask, F, G, ksum, vs)) - lin
            b2p, g2 = b2 + 2. / 3. * bs, bs / 2.
            pred = (b2p * loc + 2. * g2 * tid) / math.factorial(n)
            # Against the sample's typical magnitude, not element by element: the two sides group
            # the same terms differently, so where one configuration happens to cancel, a
            # per-element ratio reports round-off on a vanishing denominator rather than an error.
            err = float(np.max(np.abs(ref - pred)) / np.median(np.abs(ref)))
            out[n, name] = err
            assert err < atol, \
                f'K_{n} ({name}) disagrees with the partition sum by {err:.3e}'
    return out


def make_spectra_redshift_tracer(pk_callable, f=0., bias=None, nmax=6, paper=False, jit=True):
    """``{2: P, 3: B, 4: T, 5: P5, 6: P6}`` up to ``nmax``, sharing one bias model."""
    return {n: make_spectrum_redshift_tracer(n, pk_callable, f=f, bias=bias, paper=paper, jit=jit)
            for n in range(2, nmax + 1)}
