r"""The stochastic sectors of ``spectrum3_redshift_tracer`` and ``spectrum4_redshift_tracer``.

Two independent checks, neither of which trusts the algebra in ``pt.py``.

**Dimensions.** With ``[P] = L^3``, ``[B] = L^6``, ``[T] = L^9`` and ``[snb0] = [sn0] = L^3``,
every stochastic term of the n-point spectrum must have powers ``(p, q)`` of ``(P, sn)``
satisfying ``p + q = n - 1``. Measured by rescaling ``P -> lambda P`` and ``sn -> mu sn``
independently and reading the two exponents. This is what catches the term that used to sit in
``spectrum4_redshift_tracer``: ``0.25 leg_i leg_j`` has ``(p, q) = (2, 2)``, i.e. ``p + q = 4``.

**The Poisson limit.** For a Poisson sampling of a continuous field the discrete n-point
spectrum is a sum over set partitions of the legs: a block of size ``m`` contributes
``sn^(m-1)`` times the continuous spectrum of the blocks at their summed momenta. Written out,

    P^(N) = P + sn
    B^(N) = B + sn [P(k1) + P(k2) + P(k3)] + sn^2
    T^(N) = T + sn sum_{6 pairs} B(k_i+k_j, k_k, k_l)
              + sn^2 sum_{3 pairings} P(k_i+k_j) + sn^2 sum_{4 legs} P(k_l) + sn^3

The reference side is built here from the model's *own* tree ``B`` and ``P`` (obtained by
calling ``spectrum3_redshift_tracer`` / ``spectrum2`` with the stochastic amplitudes switched
off), so the test compares the assembly, not the kernels. Counterterms are set to zero because
the model's shot leg contracts ``Z1`` with ``Z1eft`` while the partition expansion contracts
``Z1eft`` with itself; the two agree identically only at ``c1 = c2 = 0``.
"""
import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jaxpower.pt import spectrum3_redshift_tracer, spectrum4_redshift_tracer


ZEFF, F = 0.5, 0.76
SN = 2000.


def _cosmo():
    k = np.logspace(-4., 1., 512)
    # A smooth, wiggle-free stand-in: the test is about the stochastic assembly, not about
    # transfer functions, and a closed form keeps the test independent of any Boltzmann code.
    pk = 2.0e4 * (k / 0.05)**0.96 / (1. + (k / 0.05)**2.4)
    kj, pkj = jnp.asarray(k), jnp.asarray(pk)
    # left=0., not the jnp.interp default of clamping to P(k_min). The tree trispectrum's
    # collapsed channels carry P(q) with q -> 0 and are finite only because P(q) -> 0 kills
    # their 1/q^2 kernels; a clamped constant there is a spurious contribution (44% on T in
    # one matter configuration) and makes the parallelogram limit depend on the direction of
    # approach by up to 28%. With P(q -> 0) = 0 that direction dependence is 1e-5.
    f = lambda q: jnp.interp(q, kj, pkj, left=0., right=0.)
    return f, f


BIAS = dict(b1=2.0, b2=0.3, bs=-0.4, c1=0., c2=0., X_FoG=0., snb0=0., sn0=0.)


def _tetrad(scale=1.):
    """A generic (non-degenerate, non-parallelogram) set of three independent legs."""
    return (jnp.array([[0.031, 0.017, 0.052]]) * scale,
            jnp.array([[-0.045, 0.062, 0.011]]) * scale,
            jnp.array([[0.070, -0.024, -0.038]]) * scale)


def _T(bias, shot=0., scale=1., pkscale=1.):
    pk, pknow = _cosmo()
    return float(spectrum4_redshift_tracer(*_tetrad(scale),
                                           lambda q: pkscale * pk(q),
                                           lambda q: pkscale * pknow(q),
                                           f=F, bias_params={0: dict(bias)}, shot=shot)[0])


def _B(bias, k1vec, k2vec, shot=0., pkscale=1.):
    pk, pknow = _cosmo()
    return float(spectrum3_redshift_tracer(k1vec, k2vec,
                                           lambda q: pkscale * pk(q),
                                           lambda q: pkscale * pknow(q),
                                           f=F, bias_params={0: dict(bias)}, shot=shot)[0])


def _P(bias, kvec, pkscale=1.):
    """Z1eft^2 P_IR at one momentum -- the model's own galaxy power, no stochastic part."""
    pk, pknow = _cosmo()
    from jaxpower.pt import compute_sigma2ir
    sigma2, sigma2_delta = compute_sigma2ir(lambda q: pkscale * pknow(q))
    kvec = np.asarray(kvec).reshape(3)
    k = float(np.sqrt(np.sum(kvec**2)))
    mu = float(kvec[2] / k) if k > 0 else 0.
    Z1 = bias['b1'] + F * mu**2 - (bias['c1'] * mu**2 + bias['c2'] * mu**4) * k**2
    eIR = (1. + F * mu**2 * (2. + F)) * sigma2 + (F * mu)**2 * (mu**2 - 1.) * sigma2_delta
    pkl, pknw = float(pkscale * pk(k)), float(pkscale * pknow(k))
    return Z1**2 * (pknw + (pkl - pknw) * np.exp(-eIR * k**2))


@pytest.mark.parametrize('order', [3, 4])
def test_stochastic_dimensions(order):
    """The stochastic sector must be homogeneous of degree ``order - 1`` in (P, sn) jointly.

    Each individual term has its own ``(p, q)`` -- the trispectrum's are (2, 1), (1, 2) and
    (0, 3) -- so the two exponents measured separately are weighted averages and need not sum
    to anything in particular. What every admissible term does share is ``p + q = order - 1``,
    and that makes the *joint* rescaling ``P -> s P``, ``sn -> s sn`` exactly homogeneous:
    ``f(s, s) = s^(order-1) f(1, 1)`` for any mixture. So the joint exponent is asserted
    tightly, and the separate ones are only reported.

    The term this catches, ``0.25 leg_i leg_j``, is ``(p, q) = (2, 2)``: a joint exponent of 4.
    """
    lam, mu = 2., 3.
    bias0 = dict(BIAS)

    def bias_sn(scale):
        return dict(BIAS, snb0=scale * SN, sn0=scale * SN / 2.)

    if order == 3:
        k1v, k2v = _tetrad()[0], _tetrad()[1]
        f = lambda b, s, sh: _B(b, k1v, k2v, shot=sh, pkscale=s)
    else:
        f = lambda b, s, sh: _T(b, shot=sh, pkscale=s)

    ref = f(bias_sn(1.), 1., SN) - f(bias0, 1., 0.)
    joint = f(bias_sn(lam), lam, lam * SN) - f(bias0, lam, 0.)
    e = np.log(joint / ref) / np.log(lam)
    p = np.log((f(bias_sn(1.), lam, SN) - f(bias0, lam, 0.)) / ref) / np.log(lam)
    q = np.log((f(bias_sn(mu), 1., mu * SN) - f(bias0, 1., 0.)) / ref) / np.log(mu)
    print(f'order {order}: joint exponent {e:.6f} (must be {order - 1}); '
          f'separate (p, q) = ({p:.3f}, {q:.3f})')
    assert abs(e - (order - 1)) < 1e-8, f'joint exponent {e:.6f} != {order - 1}'


def test_bispectrum_poisson_limit():
    """B^(N) - B == sn [P(k1) + P(k2) + P(k3)] + sn^2."""
    k1v, k2v = _tetrad()[0], _tetrad()[1]
    k3v = -k1v - k2v
    bias0 = dict(BIAS)
    bias1 = dict(BIAS, snb0=SN, sn0=SN / 2.)
    got = _B(bias1, k1v, k2v, shot=SN) - _B(bias0, k1v, k2v)
    ref = SN * sum(_P(bias0, kv) for kv in (k1v, k2v, k3v)) + SN**2
    assert np.isclose(got, ref, rtol=1e-10), (got, ref)


def test_trispectrum_poisson_limit():
    """T^(N) - T == the full set-partition expansion, built from the model's own P and B."""
    k1v, k2v, k3v = _tetrad()
    k4v = -k1v - k2v - k3v
    kv = (k1v, k2v, k3v, k4v)
    bias0 = dict(BIAS)
    bias1 = dict(BIAS, snb0=SN, sn0=SN / 2.)

    got = _T(bias1, shot=SN) - _T(bias0)

    # one pair coincides -> sn B(k_i + k_j, k_k, k_l)
    ref = 0.
    for i, j, k_, l_ in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
                         (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]:
        ref += SN * _B(bias0, kv[i] + kv[j], kv[k_])
    # two disjoint pairs -> sn^2 P(k_i + k_j)
    for i, j in [(0, 1), (0, 2), (0, 3)]:
        ref += SN**2 * _P(bias0, kv[i] + kv[j])
    # a triple -> sn^2 P(k_l)
    for l_ in range(4):
        ref += SN**2 * _P(bias0, kv[l_])
    # all four
    ref += SN**3

    assert np.isclose(got, ref, rtol=1e-8), (got, ref, got / ref)


def test_no_stochastic_terms_is_unchanged():
    """With snb0 = sn0 = shot = 0 the trispectrum is the tree one -- no regression."""
    assert _T(dict(BIAS), shot=0.) == _T(dict(BIAS, snb0=0., sn0=0.), shot=0.)


def test_parallelogram_pair_guard():
    """At T(k, -k, k', -k') the vanishing pair momenta must not be evaluated.

    Two of the six pairs have k_i + k_j == 0 exactly there; P and B are undefined at zero
    momentum and the covariance calls this configuration for every Cov[P, P] element.
    """
    kv = jnp.array([[0., 0., 0.08]])
    kpv = jnp.array([[0.05, 0., 0.03]])
    pk, pknow = _cosmo()
    bias1 = dict(BIAS, snb0=SN, sn0=SN / 2.)
    out = spectrum4_redshift_tracer(kv, -kv, kpv, pk, pknow, f=F,
                                    bias_params={0: dict(bias1)}, shot=SN)
    assert np.isfinite(np.asarray(out)).all()
    # and the guard must actually remove something: the same legs with a tiny offset that
    # makes both pair momenta nonzero give a different, larger answer
    out_off = spectrum4_redshift_tracer(kv, -kv * 0.98, kpv, pk, pknow, f=F,
                                        bias_params={0: dict(bias1)}, shot=SN)
    assert not np.isclose(float(out[0]), float(out_off[0]), rtol=1e-3)
