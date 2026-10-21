r"""The vertex model of ``make_spectrum_redshift_tracer``: every nonlinear vertex ``Z_m`` (m >= 2) times
``exp(-(m - 1) sigma^2 lambda^2 / 2) (1 + a K^2)^(m - 1)``.

Two checks: ``vertex_damping = vertex_k2 = 0`` is the tree, bit for bit; and the arguments reproduce the construction the
covariance was validated with, which swapped ``pt.make_Zn`` for a dressed kernel at run time.
"""
import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jaxpower import pt


def _pk():
    k = np.logspace(-4., 1., 512)
    pk = 2.0e4 * (k / 0.05)**0.96 / (1. + (k / 0.05)**2.4)
    kj, pkj = jnp.asarray(k), jnp.asarray(pk)
    return lambda q: jnp.interp(q, kj, pkj, left=0., right=0.)


BIAS = dict(b1=1.9, b2=0.5, bs=-1.8, X_FoG=1.4)
F = 0.76


def _legs(n, seed=3):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.normal(scale=0.06, size=(4, 3))) for _ in range(n - 1)]


@pytest.mark.parametrize('n', [4, 5, 6])
def test_vertex_zero_is_tree(n):
    pk = _pk()
    legs = _legs(n)
    tree = pt.make_spectrum_redshift_tracer(n, pk, f=F, bias=BIAS)(*legs)
    zero = pt.make_spectrum_redshift_tracer(n, pk, f=F, bias=BIAS, vertex_damping=0., vertex_k2=0.)(*legs)
    assert np.array_equal(np.asarray(tree), np.asarray(zero))


@pytest.mark.parametrize('n', [4, 5])
def test_vertex_matches_runtime_dressing(n):
    """The arguments against the run-time swap of ``pt.make_Zn`` used by the validation scripts."""
    pk = _pk()
    sigma, a = 3., 9.
    legs = _legs(n)
    new = pt.make_spectrum_redshift_tracer(n, pk, f=F, bias=BIAS, vertex_damping=sigma, vertex_k2=a)(*legs)
    make_Zn0 = pt.make_Zn

    def make_Zn(m, f=0., bias=None):
        Z0 = make_Zn0(m, f=f, bias=bias)
        if m == 1:
            return Z0

        def Z(vs):
            kz = sum(v[..., 2] for v in vs)
            k2 = jnp.sum(sum(vs)**2, axis=-1)
            return Z0(vs) * jnp.exp(-0.5 * (m - 1) * sigma**2 * kz**2) * (1. + a * k2)**(m - 1)
        return Z

    pt.make_Zn = make_Zn
    try:
        old = pt.make_spectrum_redshift_tracer(n, pk, f=F, bias=BIAS)(*legs)
    finally:
        pt.make_Zn = make_Zn0
    tree = pt.make_spectrum_redshift_tracer(n, pk, f=F, bias=BIAS)(*legs)
    print(f'n = {n}: dressed / tree {np.asarray(new / tree).round(3)}')
    assert np.allclose(np.asarray(new), np.asarray(old), rtol=1e-12, atol=0.)
    assert not np.allclose(np.asarray(new), np.asarray(tree))
