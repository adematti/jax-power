r"""The P5 term in the box PB branch of ``compute_spectrum3_covariance``.

`Cov[P, B]` is zero for a Gaussian field, so the whole block is non-Gaussian and the five-point
term is a *leading* contribution rather than a correction. cov3 used to build only the six
`P x B` tie terms of arXiv:1908.06234 Eq. (23); those carry a radial delta tying a bispectrum
leg to the power-spectrum wavenumber, so they live on the diagonal and vanish identically once
the bins are far enough apart. The P5 term of its Eq. (26)-(27) has no tie and is what populates
the off-diagonal -- as that paper puts it, "the P5 term dominates the off-diagonal elements".

These tests pin the properties that make the implementation trustworthy rather than its values:
it must be switchable off to exactly the old answer, it must touch only the PB block, and it
must vanish when the amplitudes that multiply it do.
"""
import os

import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jaxpower import MeshAttrs, BinMesh2SpectrumPoles, BinMesh3SpectrumPoles, types
from jaxpower.cov3 import compute_spectrum3_covariance
from jaxpower.pt import (prepare_spectrum2_redshift_tracer, spectrum2_redshift_tracer,
                         spectrum3_redshift_tracer, spectrum4_redshift_tracer, ProjectToPoles)
from jaxpower.utils import get_legendre

# Small on purpose: these tests pin PROPERTIES (switchability, locality,
# vanishing without shot noise), not values, so the cheapest grid that
# exercises every code path is the right one.
# The sugiyama mask requires k1_hi + k2_hi <= k_Nyquist, so the grid has to be
# fine enough for the edges below or the binning comes out EMPTY (which shows up
# as a zero-size reduction, not as an obvious message).
BOXSIZE, MESHSIZE, SHOTNOISE = 500., 64, 3000.
BIAS = dict(b1=2.0, b2=0.4, bs=-0.3, b3nl=0.1, c1=0.1, c2=0.2, X_FoG=1.,
            snb0=0., sn0=0.)


def _theory():
    k = np.logspace(-4., 1., 400)
    pk = 2.0e4 * (k / 0.05)**0.96 / (1. + (k / 0.05)**2.4)
    kj, pkj = jnp.asarray(k), jnp.asarray(pk)
    f = lambda q: jnp.interp(q, kj, pkj, left=0., right=0.)
    table, table_now = prepare_spectrum2_redshift_tracer(
        jnp.asarray(np.logspace(-3., np.log10(0.6), 60)), f, f)
    ells = [0, 2, 4]
    to_poles = ProjectToPoles(ells=ells, mu=6)
    poles = to_poles(spectrum2_redshift_tracer(to_poles.mu, table, table_now, 0.76,
                                               {0: dict(BIAS)}))
    kt = table['matter']['k']

    def P(kvec):
        kvec = jnp.asarray(kvec)
        kn = jnp.sqrt(jnp.sum(kvec**2, axis=-1))
        mu = jnp.where(kn > 0., kvec[..., 2] / jnp.where(kn > 0., kn, 1.), 0.)
        at = jax.vmap(lambda pl: jnp.interp(kn.ravel(), kt, pl))(poles)
        return sum(at[i].reshape(kn.shape) * get_legendre(e)(mu) for i, e in enumerate(ells))

    def B(k1, k2, k3):
        return spectrum3_redshift_tracer(k1, k2, f, f, f=0.76, bias_params={0: dict(BIAS)})

    def T(k1, k2, k3, k4):
        return spectrum4_redshift_tracer(k1, k2, k3, f, f, f=0.76, bias_params={0: dict(BIAS)})

    return lambda fields: {2: P, 3: B, 4: T}.get(len(fields), None)


def _observable(mattrs, edges):
    bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=(0,))
    bin3 = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=[(0, 0, 0)],
                                 basis='sugiyama-diagonal')
    s2 = types.Mesh2SpectrumPoles([types.Mesh2SpectrumPole(
        k=np.asarray(bin2.xavg), k_edges=np.asarray(bin2.edges),
        num_raw=np.zeros(len(np.asarray(bin2.xavg))), nmodes=np.asarray(bin2.nmodes),
        norm=1., ell=0)])
    s3 = types.Mesh3SpectrumPoles([types.Mesh3SpectrumPole(
        k=np.asarray(bin3.xavg), k_edges=np.asarray(bin3.edges),
        num_raw=np.zeros(len(np.asarray(bin3.xavg))), nmodes=np.asarray(bin3.nmodes[0]),
        norm=1., ell=(0, 0, 0), basis='sugiyama-diagonal')])
    return types.ObservableTree([s2, s3], fields=[(0, 0), (0, 0, 0)]), len(np.asarray(bin2.xavg))


def _run(mattrs, obs, theory, shotnoise, **env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        return np.asarray(compute_spectrum3_covariance(
            mattrs, mattrs, obs, theory=theory, shotnoise=shotnoise, cache={}).value())
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_p5_switch_and_locality():
    """Off -> exactly the tie-only answer; on -> only the PB block moves; stays symmetric."""
    mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=MESHSIZE, boxcenter=0.)
    obs, n2 = _observable(mattrs, np.arange(0.05, 0.151, 0.05))
    th = _theory()
    off = _run(mattrs, obs, th, SHOTNOISE, COV3_NO_P5=1, COV3_QUAD_SIZE=4)
    zero = _run(mattrs, obs, th, SHOTNOISE, COV3_P5_QK=0, COV3_QUAD_SIZE=4)
    assert np.allclose(off, zero, rtol=1e-12), np.max(np.abs(off / zero - 1.))

    on = _run(mattrs, obs, th, SHOTNOISE, COV3_P5_QK=4, COV3_QUAD_SIZE=4)
    # the P and B diagonal blocks are untouched: P5 lives only in the PB branch
    assert np.allclose(on[:n2, :n2], off[:n2, :n2], rtol=1e-12)
    assert np.allclose(on[n2:, n2:], off[n2:, n2:], rtol=1e-12)
    # the cross block does change
    assert not np.allclose(on[:n2, n2:], off[:n2, n2:], rtol=1e-6)

    # The P5 CONTRIBUTION must be symmetric. Asserting that the total is would test cov3
    # rather than this term: its PP block carries a quadrature asymmetry of its own -- ~5e-6
    # on this deliberately tiny configuration (64^3, COV3_QUAD_SIZE = 4, two bins) -- which is
    # present with the term switched off, unchanged by it, and is a separate question.
    dif = on - off
    assert np.allclose(dif, dif.T, atol=1e-10 * np.abs(dif).max()), \
        np.abs(dif - dif.T).max() / np.abs(dif).max()

    def _asym(m):
        return np.abs(m - m.T).max() / np.abs(m).max()

    assert _asym(on) <= 1.5 * _asym(off) + 1e-12, (_asym(on), _asym(off))


def test_p5_vanishes_without_shotnoise():
    """Its two implemented lines carry 1/nbar and 1/nbar^2, so shotnoise = 0 removes them."""
    mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=MESHSIZE, boxcenter=0.)
    obs, _ = _observable(mattrs, np.arange(0.05, 0.151, 0.05))
    th = _theory()
    on = _run(mattrs, obs, th, 0., COV3_P5_QK=4, COV3_QUAD_SIZE=4)
    off = _run(mattrs, obs, th, 0., COV3_NO_P5=1, COV3_QUAD_SIZE=4)
    assert np.allclose(on, off, rtol=1e-12), np.max(np.abs(on - off))
