import functools
import operator
import itertools
from functools import partial

import numpy as np
from scipy import special
import jax
import jax.numpy as jnp

from .utils import get_legendre, get_spherical_jn, wigner_3j


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
    lX2 = 0.5 * f**2 * sum((kmu * X)**2 for kmu, X in kmu_X)
    if damping == 'lor':
        return 1. / (1. + lX2 * sigma2v)
    elif damping == 'exp':
        return jnp.exp(-lX2 * sigma2v)
    raise NotImplementedError(f'damping {damping} is not implemented')


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
    return W * Ps + Pct + Pnlo + shot


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


def spectrum2_redshift_tracer(mu, table, table_now, f, bias_params, **ct_params):

    interp_kmu = None
    if mu.ndim == 2:  # kvec

        def _norm(kvec):
            return jnp.sqrt(jnp.sum(kvec**2, axis=-1))

        def _mu(kvec, knorm):
            return jnp.where(knorm > 0., kvec[..., 2] / knorm, 0.)

        kvec = mu
        knorm = _norm(kvec)
        mu = _mu(kvec, knorm)
        interp_kmu = knorm, mu
        ells = list(range(0, 8, 2))
        to_poles = ProjectToMultipoles(mu=10, ells=ells)

    fields, bias_params = _format_bias_params(bias_params, nfields=2)
    a, b = fields
    pk_eft = spectrum2_redshift_tracer_eft(table['matter'], table['bias'], table['A_B'], table['sigma2v'], mu, f, bias_params, **ct_params)
    pknow_eft = spectrum2_redshift_tracer_eft(table_now['matter'], table_now['bias'], table_now['A_B'], table['sigma2v'], mu, f, bias_params, **ct_params)
    pk_ir = _spectrum2_ir_resum(table['matter']['k'], mu, table['matter']['P11'], table_now['matter']['P11'], pk_eft, pknow_eft, f, table['sigma2'], table['sigma2_delta'], bias_params[a]['b1'], bias_params[b]['b1'])
    if interp_kmu is not None:
        poles = to_poles(pk_ir)
        poles = jax.vmap(lambda pole: jnp.interp(knorm, table['matter']['k'], pole))(poles.T)
        pk_ir = sum(pole * get_legendre(ell)(mu) for ell, pole in zip(ells, poles))
    return pk_ir


class ProjectToMultipoles:

    """Helper class to compute multipoles using Legendre polynomials."""

    def __init__(self, ells=(0, 2, 4), mu=8):
        self.ells = list(ells)
        integ = integration(a=0, b=1, size=mu)
        self.mu, w = integ.x(), integ.w
        self.w = np.array([w * (2 * ell + 1) * get_legendre(ell)(self.mu) for ell in self.ells])

    def __call__(self, f):
        return jnp.sum(f * self.w[(slice(None),) + (None,) * (f.ndim - 1) + (slice(None),)], axis=-1)


def get_S(ells, z3=False):

    ell1, ell2, ell3 = ells
    H = wigner_3j(ell1, ell2, ell3, 0, 0, 0)

    if abs(H) < 1e-12:
        return lambda *args: 0.

    def get_Ylm(ell, m, xhat):
        mu = xhat[..., 2]
        phi = np.arctan2(xhat[..., 1], xhat[..., 0])
        fac = special.factorial(ell - abs(m), exact=False) / special.factorial(ell + abs(m), exact=False)
        amp = np.sqrt(fac)
        return amp * special.lpmv(abs(m), ell, mu) * np.exp(1j * m * phi)

    if z3:  # last vector is z, so m3 = 0
        coeffs = []
        mmax = min(ell1, ell2)
        for m in range(-mmax, mmax + 1):
            gaunt = wigner_3j(ell1, ell2, ell3, m, -m, 0) / H
            if abs(gaunt) > 1e-12:
                coeffs.append((m, gaunt))

        def Sell(xhat1, xhat2):
            out = 0.0
            for m, gaunt in coeffs:
                out += gaunt * get_Ylm(ell1, m, xhat1) * get_Ylm(ell2, -m, xhat2)
            return out.real if ((ell1 + ell2 + ell3) % 2 == 0) else out.imag
    else:
        ms = [np.arange(-ell, ell + 1) for ell in ells]
        coeffs = []
        for m1, m2, m3 in itertools.product(*ms):
            gaunt = wigner_3j(ell1, ell2, ell3, m1, m2, m3) / H
            if abs(gaunt) > 1e-12:
                coeffs.append((m1, m2, m3, gaunt))

        def Sell(*xhats):
            out = 0.
            for m1, m2, m3, gaunt in coeffs:
                out = out + gaunt * prod(
                    get_Ylm(ell, m, xhat)
                    for ell, m, xhat in zip(ells, (m1, m2, m3), xhats, strict=True)
                )
            return out.real if ((ell1 + ell2 + ell3) % 2 == 0) else out.imag

    return Sell


class ProjectToSell:

    """Helper class to compute multipoles using Legendre polynomials."""

    def __init__(self, ells=((0, 0, 0), (2, 0, 2)), size=6):
        self.ells = [tuple(ell) for ell in ells]
        integ_mu = integration(-1., 1., size=size)
        integ_phi = integration(0., 2. * np.pi, size=size)
        integ = IntegralND(mu1=integ_mu, mu2=integ_mu, phi2=integ_phi)

        def get_N(ell1, ell2, ell3):
            return (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1)

        def get_H(ell1, ell2, ell3):
            return wigner_3j(ell1, ell2, ell3, 0, 0, 0)

        def unitvec(mu, phi):
            s = np.sqrt(np.clip(1. - mu**2, 0., None))
            return np.stack([s * np.cos(phi), s * np.sin(phi), mu], axis=-1)

        mu1, mu2, phi2 = integ.x(['mu1', 'mu2', 'phi2'], sparse=False)
        k1hat = unitvec(mu1, np.zeros_like(mu1)).reshape(-1, 3)
        k2hat = unitvec(mu2, phi2).reshape(-1, 3)

        w = integ.w.ravel()
        self.k1hat, self.k2hat = k1hat, k2hat
        self.w = np.array([w * get_N(*ell) * get_H(*ell)**2 * get_S(ell, z3=True)(k1hat, k2hat) for ell in self.ells])

    def __call__(self, f):
        return jnp.sum(f * self.w[(slice(None),) + (None,) * (f.ndim - 1) + (slice(None),)], axis=-1)



# Note: shot noise formula isn't correct for multitracer, but let's marginalize over it anyway...

def spectrum3_redshift_tracer(k1vec, k2vec, pk_callable, pknow_callable, f, bias_params, shot=0., sigma2v=None, sigma2=None, sigma2_delta=None, damping='lor'):
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
        ('b1', 'b2', 'bs', 'c1', 'c2', 'Bshot', 'Pshot', 'X_FoG')
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
            names = ['b1', 'b2', 'bs', 'c1', 'c2', 'Bshot', 'Pshot', 'X_FoG']
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

    def _Z1(field, mu):
        b1 = _get_bias_params(field, 'b1')
        return b1 + f * mu**2

    def _Z1eft(field, k, mu):
        c1, c2 = _get_bias_params(field, ['c1', 'c2'])
        return _Z1(field, mu) - (c1 * mu**2 + c2 * mu**4) * k**2

    def _Z2(field, ki, kj, xij, mui, muj):
        b1, b2, bs = _get_bias_params(field, ['b1', 'b2', 'bs'])
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
        b1, Bshot, Pshot = _get_bias_params(field, ['b1', 'Bshot', 'Pshot'])
        return (b1 * Bshot + 2. * Pshot * f * mu**2) * Z1eft * pkIR

    shot = _shot_leg(a, k1, mu1, Z1eft1, pkIR1) + _shot_leg(b, k2, mu2, Z1eft2, pkIR2) + _shot_leg(c, k3, mu3, Z1eft3, pkIR3) + shot**2

    return W * (B12 + B23 + B31) + shot


def spectrum4_redshift_tracer(k1vec, k2vec, k3vec, pk_callable, pknow_callable, f, bias_params, shot=0., sigma2v=None, sigma2=None, sigma2_delta=None, damping='lor'):
    """
    JAX-friendly tree-level cross-trispectrum T^{abcd}(k1, k2, k3, k4),
    with k4 = -k1 - k2 - k3, including the same wiggle damping as in spectrum3.

    Parameters
    ----------
    k1vec, k2vec, k3vec : array_like[..., 3]
        Wavevectors of the first three legs.
    f, sigma2v, sigma2, sigma2_delta : float
        RSD / IR / FoG parameters.
    bias_params : dict
        Mapping field -> parameters.
    pk_callable : callable
        Callable returning the linear power spectrum P(k).
    pknow_callable : callable
        Callable returning the no-wiggle linear power spectrum P_now(k).
    damping : {None, 'lor', 'exp', 'vdg'}, default=None
        Optional phenomenological FoG damping on the full trispectrum.
    shot : float, default=0.
        Constant shot-noise contribution.

    Returns
    -------
    trispectrum : array_like
        Tree-level trispectrum model.
    """
    fields, bias_params = _format_bias_params(bias_params, nfields=4)
    a, b, c, d = fields

    if sigma2v is None:
        sigma2v = compute_sigma2v(pk_callable)
    if sigma2 is None or sigma2_delta is None:
        sigma2, sigma2_delta = compute_sigma2ir(pknow_callable)

    def _get_bias_params(field, names=None):
        pars = bias_params[field]
        if names is None:
            names = ['b1', 'b2', 'bs', 'c1', 'c2', 'Bshot', 'Pshot', 'X_FoG']
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

    def _Z1(field, mu):
        b1 = _get_bias_params(field, 'b1')
        return b1 + f * mu**2

    def _Z1eff(field, k, mu):
        c1, c2 = _get_bias_params(field, ['c1', 'c2'])
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
        b1, b2, bs = _get_bias_params(field, ['b1', 'b2', 'bs'])
        km = ki * mui + kj * muj
        term1 = b2 / 2. + bs / 2. * (xij**2 - 1. / 3.)
        term2 = km / 2. * (mui / ki * f * (b1 + f * muj**2) + muj / kj * f * (b1 + f * mui**2))
        term3 = b1 * _F2(ki, kj, xij)
        mu2 = km**2 / (ki**2 + kj**2 + 2. * ki * kj * xij)
        term4 = f * mu2 * _G2(ki, kj, xij)
        return term1 + term2 + term3 + term4

    def _A2(field, ki, kj, xij, mui, muj):
        b1, b2, bs = _get_bias_params(field, ['b1', 'b2', 'bs'])
        q12vec_z = ki * mui + kj * muj
        q12 = jnp.sqrt(ki**2 + kj**2 + 2. * ki * kj * xij)
        mu12 = jnp.where(q12 > 0., q12vec_z / q12, 0.)
        return b1 * _F2(ki, kj, xij) + b2 / 2. + bs / 2. * (xij**2 - 1. / 3.) + f * mu12**2 * _G2(ki, kj, xij)

    def _Z3(field, k1vec, k2vec, k3vec):
        b1 = _get_bias_params(field, 'b1')

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
        pknw = pknow_callable(k)
        eIR = (1. + f * mu**2 * (2. + f)) * sigma2 + (f * mu)**2 * (mu**2 - 1.) * sigma2_delta
        return pknw + (pk - pknw) * jnp.exp(-eIR * k**2)

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

    X1, X2, X3, X4 = [_get_bias_params(field, 'X_FoG') for field in fields]
    W = fog_damping((k1 * mu1, X1), (k2 * mu2, X2), (k3 * mu3, X3), (k4 * mu4, X4), f=f, sigma2v=sigma2v, damping=damping)

    def _shot_leg(field, k, mu, Z1eft, pkIR):
        b1, Bshot, Pshot = _get_bias_params(field, ['b1', 'Bshot', 'Pshot'])
        return (b1 * Bshot + 2. * Pshot * f * mu**2) * Z1eft * pkIR

    leg1 = _shot_leg(a, k1, mu1, _Z1eff(a, k1, mu1), _IR_pk(k1, mu1))
    leg2 = _shot_leg(b, k2, mu2, _Z1eff(b, k2, mu2), _IR_pk(k2, mu2))
    leg3 = _shot_leg(c, k3, mu3, _Z1eff(c, k3, mu3), _IR_pk(k3, mu3))
    leg4 = _shot_leg(d, k4, mu4, _Z1eff(d, k4, mu4), _IR_pk(k4, mu4))

    Pshot1, Pshot2, Pshot3, Pshot4 = [_get_bias_params(field, 'Pshot') for field in fields]
    shot = 0.25 * (leg1 * leg2 + leg1 * leg3 + leg1 * leg4 + leg2 * leg3 + leg2 * leg4 + leg3 * leg4)\
    + 0.5 * (leg1 * Pshot1 + leg2 * Pshot2 + leg3 * Pshot3 + leg4 * Pshot4) + shot

    return W * (4. * t2211 + 6. * t3111) + shot