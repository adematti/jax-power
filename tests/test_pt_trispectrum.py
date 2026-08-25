r"""``spectrum4_redshift_tracer`` at the parallelogram T(k, -k, k', -k').

That configuration is what every element of Cov[P, P] is built from, and it is a degenerate
point of the generic (2,2,1,1) + (3,1,1,1) permutation sum: four of the twelve (2,2,1,1) terms
carry the internal momentum ``q = k1 + k2``, which vanishes there, and their individual
``1/q^2`` poles cancel only in the sum. Evaluating at ``q = 0`` exactly, the ``_safe_div``
guards return zero for those four terms rather than their finite limit.

The function therefore evaluates just off the degenerate point, shifting ``k2 -> k2 - eps k1``
(momentum conservation then puts the same shift into ``k3 + k4``). These tests check that the
shift is (a) harmless -- the answer is the limit of the generic sum, flat over decades of eps
-- and (b) necessary, by exhibiting what the previous closed-form branch returned instead.

Everything here is run in the pure matter limit (``b1 = 1``, ``f = 0``, no counterterms, no
FoG) so that nothing bias- or RSD-related can be confused with the kinematics.
"""
import os

import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jaxpower.pt import spectrum4_redshift_tracer

F = 0.
BIAS = dict(b1=1.0, b2=0., bs=0., c1=0., c2=0., X_FoG=0., snb0=0., sn0=0.)
CASES = [(np.array([0., 0., 0.08]), np.array([0.05, 0., 0.03])),
         (np.array([0., 0., 0.15]), np.array([0., 0.12, 0.06])),
         (np.array([0.02, 0., 0.03]), np.array([0., 0., 0.25]))]


def _cosmo():
    k = np.logspace(-4., 1., 512)
    pk = 2.0e4 * (k / 0.05)**0.96 / (1. + (k / 0.05)**2.4)
    kj, pkj = jnp.asarray(k), jnp.asarray(pk)
    # left=0., not the jnp.interp default of clamping to P(k_min). The tree trispectrum's
    # collapsed channels carry P(q) with q -> 0 and are finite only because P(q) -> 0 kills
    # their 1/q^2 kernels; a clamped constant there is a spurious contribution (44% on T in
    # one matter configuration) and makes the parallelogram limit depend on the direction of
    # approach by up to 28%. With P(q -> 0) = 0 that direction dependence is 1e-5.
    f = lambda q: jnp.interp(q, kj, pkj, left=0., right=0.)
    return f, f


def _T(k, kp, eps=0.):
    """T(k1, k2, k3, k4) with k2 = -k (1 - eps); eps = 0 is the parallelogram itself."""
    pk, pknow = _cosmo()
    k1v = jnp.asarray(k)[None]
    k2v = jnp.asarray(-k * (1. - eps))[None]
    k3v = jnp.asarray(kp)[None]
    return float(spectrum4_redshift_tracer(k1v, k2v, k3v, pk, pknow, f=F,
                                           bias_params={0: dict(BIAS)})[0])


@pytest.mark.parametrize('icase', range(len(CASES)))
def test_generic_sum_has_a_limit(icase):
    """The generic permutation sum converges as the degeneracy is approached.

    If the 1/q^2 poles of the four vanishing-q terms did not cancel, this would move by ~10^8
    across the scan instead of by <0.1%.
    """
    k, kp = CASES[icase]
    v = np.array([_T(k, kp, eps) for eps in (1e-4, 1e-5, 1e-6, 1e-7)])
    assert np.all(np.abs(v / v[0] - 1.) < 1e-3), v / v[0]


@pytest.mark.parametrize('icase', range(len(CASES)))
def test_parallelogram_equals_the_limit(icase):
    """T at the parallelogram is that limit -- which is the point of the regularization."""
    k, kp = CASES[icase]
    lim = _T(k, kp, 1e-5)
    got = _T(k, kp, 0.)
    assert np.isclose(got, lim, rtol=1e-3), (got, lim, got / lim)


@pytest.mark.parametrize('icase', range(len(CASES)))
def test_limit_does_not_depend_on_the_direction_of_approach(icase):
    """Shrinking k2 along k1, or moving it transversally, must give the same limit.

    It does not if P(k) is clamped below the theory grid instead of going to zero: the
    collapsed channels then contribute a constant whose angular structure survives the limit,
    and the two paths differ by up to 28%. This is the test that says the parallelogram value
    is well defined after all.
    """
    k, kp = CASES[icase]
    pk, pknow = _cosmo()
    t = np.array([-k[1], k[0], 0.])
    n = np.sqrt((t**2).sum())
    t = t / n if n > 0 else np.array([1., 0., 0.])
    eps = 1e-5
    k2_long = -k * (1. - eps)
    k2_tran = -k + eps * np.sqrt((k**2).sum()) * t
    out = [float(spectrum4_redshift_tracer(jnp.asarray(k)[None], jnp.asarray(k2)[None],
                                           jnp.asarray(kp)[None], pk, pknow, f=F,
                                           bias_params={0: dict(BIAS)})[0])
           for k2 in (k2_long, k2_tran)]
    assert np.isclose(out[0], out[1], rtol=1e-3), out


@pytest.mark.parametrize('icase', range(len(CASES)))
def test_legacy_reduced_branch_is_exactly_double(icase):
    """The closed-form branch it replaced returns exactly 2x the limit.

    Not 1.4x-2.0x "depending on configuration" -- that spread was an artifact of a clamped
    P(k -> 0) contaminating the limit. With a physical P it is 2.000 in every configuration,
    which is the double count the coefficient counting predicts: `_para_channel`'s (3,1,1,1)
    and (2,2,1,1) pieces are each added in BOTH of its two channel calls.
    """
    k, kp = CASES[icase]
    lim = _T(k, kp, 1e-5)
    os.environ['JAXPOWER_PT_PARA_REDUCED'] = '1'
    try:
        legacy = _T(k, kp, 0.)
    finally:
        os.environ.pop('JAXPOWER_PT_PARA_REDUCED', None)
    assert np.isclose(legacy / lim, 2., rtol=1e-3), (legacy, lim, legacy / lim)


def test_regularization_is_inert_away_from_the_degeneracy():
    """A generic configuration must not be touched by the shift."""
    pk, pknow = _cosmo()
    k1v = jnp.array([[0.031, 0.017, 0.052]])
    k2v = jnp.array([[-0.045, 0.062, 0.011]])
    k3v = jnp.array([[0.070, -0.024, -0.038]])
    ref = float(spectrum4_redshift_tracer(k1v, k2v, k3v, pk, pknow, f=F,
                                          bias_params={0: dict(BIAS)})[0])
    os.environ['JAXPOWER_PT_PARA_EPS'] = '1e-2'      # 1000x larger: must change nothing
    try:
        import importlib
        import jaxpower.pt
        got = float(jaxpower.pt.spectrum4_redshift_tracer(k1v, k2v, k3v, pk, pknow, f=F,
                                                          bias_params={0: dict(BIAS)})[0])
    finally:
        os.environ.pop('JAXPOWER_PT_PARA_EPS', None)
    assert np.isclose(got, ref, rtol=1e-12), (got, ref)
