r"""Purely ISOTROPIC (spherically symmetric Gaussian) selection function, at the survey's typical width.

Why this is the clean ellmax test: for a Gaussian selection W = exp(-d^2/2sigma^2), completing the
square gives Q(s1,s2) = exp(-[s1^2+s2^2-s1.s2]/3sigma^2) EXACTLY, whose isotropic part has the exact
separable expansion (ms.tex eq:gaussian_q000_exact)

    Q000(s1,s2) = exp(-(s1^2+s2^2)/3sigma^2) sinh(z)/z = sum_m h_m(s1) h_m(s2),  z = s1 s2/3sigma^2

reaching machine precision with ~12 terms. So the RESUMMED analytic window matrix is EXACT here --
no Laguerre fit, none of the ~20% offset that stopped the analytic arm adjudicating on the octant
window. ellmax convergence can then be measured against ground truth instead of another approximation.

Everything else (box, mesh, painting, 3 independent random realizations, cross-leg measurement) is
kept identical to survey_window_geom.py so the two are directly comparable.
"""
import numpy as np, jax, jax.numpy as jnp
from jax import random
from jaxpower import MeshAttrs, ParticleField, compute_normalization
import survey_window_geom as SG

MESHSIZE = SG.MESHSIZE
BOXSIZE = float(SG.get_mattrs().boxsize[0])          # 2302, same as the survey
NRAND = SG.NRAND
KW_PAINT = SG.KW_PAINT


def get_mattrs(boxsize=BOXSIZE, meshsize=MESHSIZE):
    return MeshAttrs(boxsize=boxsize, meshsize=meshsize, boxcenter=[0., 0., 0.])


def survey_width(frac=(0.5, 1. / np.e)):
    """Effective width of the MEASURED survey Q000, so sigma can be matched to it.

    Reports the s at which the s2->0 profile falls to each fraction, plus the measure-weighted
    second moment sqrt(<s^2>) with weight s^2 Q ds -- the scale that sets the required ellmax,
    since the ell-sum term peaks near s ~ ell/k.
    """
    import os
    from lsstypes import read
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', '_tests')
    pole = read(os.path.join(OUT, 'p3_scocc_Q_ellwmax0.h5')).get(ells=(0, 0, 0))
    S = np.asarray(list(pole.coords().values())[0], dtype='f8')
    w = np.asarray(pole.value(), dtype='f8').real[:, 0]
    out = {}
    for f in frac:
        i = np.argmax(w < f * w[0])
        out[f] = float(np.interp(f * w[0], [w[i], w[i - 1]], [S[i], S[i - 1]])) if i else np.nan
    d = np.gradient(S)
    m2 = float(np.sqrt((d * S**2 * w * S**2).sum() / max((d * S**2 * w).sum(), 1e-300)))
    return out, m2


def make_particles(seed=0, sigma=None, nrand=NRAND, mattrs=None):
    """Randoms drawn FROM the isotropic Gaussian selection (positions ~ N(centre, sigma^2))."""
    if mattrs is None: mattrs = get_mattrs()
    pos = sigma * random.normal(random.key(seed), (nrand, 3)) + jnp.asarray(mattrs.boxcenter)
    return ParticleField(pos, weights=jnp.ones(nrand), attrs=mattrs)


def selection(sigma, nreal=3, nrand=NRAND, mattrs=None):
    if mattrs is None: mattrs = get_mattrs()
    parts = [make_particles(seed=s, sigma=sigma, nrand=nrand, mattrs=mattrs) for s in range(nreal)]
    Ws = [p.paint(**KW_PAINT) for p in parts]
    return parts, Ws, compute_normalization(*Ws)


def analytic_q000(sigma, s1, s2):
    """Q000(s1,s2) for a Gaussian selection -- EXACT (eq:gaussian_q000_exact).

    Evaluated as exp(-(s1^2+s2^2)/3sigma^2) sinh(z)/z with z = s1 s2 / 3 sigma^2, using sinh(z)/z ->
    1 + z^2/6 + ... at small z to avoid 0/0.
    """
    a = 1. / (3. * sigma**2)
    S1, S2 = np.meshgrid(np.asarray(s1), np.asarray(s2), indexing='ij')
    z = S1 * S2 * a
    sh = np.where(z < 1e-8, 1. + z**2 / 6., np.sinh(np.minimum(z, 700.)) / np.where(z == 0., 1., z))
    return np.exp(-a * (S1**2 + S2**2)) * sh


def hm(m, r, sigma):
    """h_m(r) = a^m r^(2m) e^(-a r^2)/sqrt((2m+1)!), a = 1/3sigma^2, so sum_m h_m(s1) h_m(s2) = Q000."""
    from scipy.special import gammaln
    a = 1. / (3. * sigma**2)
    r = np.maximum(np.asarray(r, dtype='f8'), 1e-30)
    return np.exp(m * np.log(a) - 0.5 * gammaln(2 * m + 2) + 2 * m * np.log(r) - a * r**2)


if __name__ == '__main__':
    frac, m2 = survey_width()
    print('MEASURED survey window Q000 (s2 -> 0 profile):')
    for f, v in frac.items():
        print(f'  falls to {f:.3f} of its peak at s = {v:.0f}')
    print(f'  measure-weighted sqrt(<s^2>) = {m2:.0f}   <- the scale that sets required ellmax')
    print('\nMatching an isotropic Gaussian (Q000 = exp(-(s1^2+s2^2)/3sigma^2) sinh(z)/z):')
    for sg in (200., 400., 600., 800.):
        s = np.geomspace(1., 8000., 2000); d = np.gradient(s)
        q = np.diag(analytic_q000(sg, s, s))
        w = analytic_q000(sg, s, np.array([s[0]]))[:, 0]
        m2g = float(np.sqrt((d * s**2 * w * s**2).sum() / (d * s**2 * w).sum()))
        i = np.argmax(w < 0.5 * w[0])
        print(f'  sigma={sg:5.0f}: half-max at s={s[i]:6.0f}, measure-weighted sqrt(<s^2>)={m2g:6.0f}')
