from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (BinMesh2SpectrumPoles, compute_mesh2_spectrum, generate_gaussian_mesh,
                      generate_uniform_particles, FKPField, MeshAttrs,
                      compute_fkp2_normalization, compute_fkp2_shotnoise)
from jaxpower.oqe import (SeparableWeighting, separable_inverse_weight, compute_oqe2_normalization,
                          compute_oqe2_shotnoise, covariance_weight, to_real)


def pk(k):
    kp = 0.03
    return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)


def get_fkp(mattrs, size=int(1e5), seed=42):
    """A cutsky-like FKP field: clustered data against uniform randoms."""
    mesh = generate_gaussian_mesh(mattrs, lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec))),
                                  seed=seed, unitary_amplitude=True)
    data = generate_uniform_particles(mattrs, size, seed=seed + 1)
    data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    randoms = generate_uniform_particles(mattrs, 4 * size, seed=seed + 2)
    return FKPField(data, randoms)


def _spectrum(fkp, bin, sinv=None, los='x'):
    """The FKP estimator, optionally with an OQE weighting (a factory of MeshAttrs) applied to the painted field."""
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    if sinv is None:
        norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=None)
        num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
    else:
        mesh = sinv(mesh.attrs)(mesh)
        norm = compute_oqe2_normalization(fkp, sinv=sinv, bin=bin, cellsize=None)
        num_shotnoise = compute_oqe2_shotnoise(fkp, sinv=sinv, bin=bin, cellsize=None)
    spectrum = compute_mesh2_spectrum(mesh, bin=bin, los=los)
    return spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)


def test_null(meshsize=64):
    """The whole OQE path with a unit weighting must reproduce the FKP estimator."""
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))
    fkp = get_fkp(mattrs)
    unit = lambda mattrs: SeparableWeighting.from_terms([(1., 1.)])

    ref = _spectrum(fkp, bin)
    test = _spectrum(fkp, bin, sinv=unit)
    assert np.allclose(compute_oqe2_normalization(fkp, sinv=unit, bin=bin, cellsize=None)[0],
                       compute_fkp2_normalization(fkp, bin=bin, cellsize=None)[0], rtol=1e-10)
    assert np.allclose(compute_oqe2_shotnoise(fkp, sinv=unit, bin=bin, cellsize=None)[0],
                       compute_fkp2_shotnoise(fkp, bin=bin)[0], rtol=1e-10)
    for ell in bin.ells:
        r, t = ref.get(ells=ell).value(), test.get(ells=ell).value()
        assert np.allclose(t, r, rtol=1e-8, atol=1e-8 * np.abs(r).max()), f'ell = {ell}'


def test_null_cross(meshsize=64):
    """
    Two tracers, each with its own unit weighting (one per leg), must reproduce the FKP cross
    spectrum: same normalization, and no shot noise between fields at different positions.
    """
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2))
    fkps = [get_fkp(mattrs, seed=42), get_fkp(mattrs, size=int(8e4), seed=84)]
    unit = lambda mattrs: SeparableWeighting.from_terms([(1., 1.)])
    kw = dict(resampler='tsc', interlacing=3, compensate=True, out='real')
    meshes = [fkp.paint(**kw) for fkp in fkps]

    ref = compute_mesh2_spectrum(*meshes, bin=bin, los='x').clone(norm=compute_fkp2_normalization(*fkps, bin=bin, cellsize=None),
                                                                 num_shotnoise=compute_fkp2_shotnoise(*fkps, bin=bin))
    norm = compute_oqe2_normalization(*fkps, sinv=(unit, unit), bin=bin, cellsize=None)
    shotnoise = compute_oqe2_shotnoise(*fkps, sinv=(unit, unit), bin=bin, cellsize=None)
    test = compute_mesh2_spectrum(*[unit(mesh.attrs)(mesh) for mesh in meshes], bin=bin, los='x').clone(norm=norm, num_shotnoise=shotnoise)
    assert np.allclose(norm[0], compute_fkp2_normalization(*fkps, bin=bin, cellsize=None)[0], rtol=1e-10)
    assert np.allclose(shotnoise[0], 0.)
    for ell in bin.ells:
        r, t = ref.get(ells=ell).value(), test.get(ells=ell).value()
        assert np.allclose(t, r, rtol=1e-8, atol=1e-8 * np.abs(r).max()), f'ell = {ell}'


def test_rescaling_invariance(meshsize=64):
    """The estimator is exactly invariant under ``sinv -> c sinv``: numerator and normalization both scale as ``c^2``."""
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2))
    fkp = get_fkp(mattrs)
    xnorm = jnp.sqrt(sum(xx**2 for xx in mattrs.rcoords(sparse=True)))
    diag = jnp.exp(-xnorm / 2e3)

    ref = _spectrum(fkp, bin, sinv=lambda mattrs: SeparableWeighting.from_terms([(1., diag)]))
    for c in [1e-3, 7., 1e4]:
        test = _spectrum(fkp, bin, sinv=lambda mattrs: SeparableWeighting.from_terms([(1., c * diag)]))
        for ell in bin.ells:
            r, t = ref.get(ells=ell).value(), test.get(ells=ell).value()
            assert np.allclose(t, r, rtol=1e-8, atol=1e-8 * np.abs(r).max()), f'c = {c}, ell = {ell}'


def test_single_term_shape(meshsize=64):
    """
    A single separable term returns the FKP answer with weight ``D``, whatever ``f(k)``: exact mode by
    mode, approximate within a bin, where ``f^2`` reweights the modes of the shell.
    """
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0,))
    fkp = get_fkp(mattrs)
    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))

    ref = _spectrum(fkp, bin, sinv=lambda mattrs: SeparableWeighting.from_terms([(1., 1.)]))
    test = _spectrum(fkp, bin, sinv=lambda mattrs: SeparableWeighting.from_terms([(1. / (1. + (knorm / 0.1)**2), 1.)]))
    r, t = ref.get(ells=0).value(), test.get(ells=0).value()
    dev = np.abs(t - r) / np.abs(r).max()
    print(f'single-term deviation: max {dev.max():.2e}, median {np.median(dev):.2e}')
    assert dev.max() < 5e-2


def test_separable_inverse_weight(meshsize=64, nterms=4):
    """The rank-M fit must reproduce 1 / (W^2 P + N) over the values the mesh actually holds."""
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    xnorm = jnp.sqrt(sum(xx**2 for xx in mattrs.rcoords(sparse=True)))
    W = jnp.exp(-(xnorm / 2e3)**2)
    N = 1e-4 * jnp.ones_like(W)
    weighting = separable_inverse_weight(mattrs, power=pk, selection=W, noise=N, nterms=nterms)
    assert weighting.nterms == nterms

    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))
    pkmesh = jnp.where(knorm > 0., pk(jnp.where(knorm > 0., knorm, 1.)), 0.)
    approx = sum(weighting.fourier(ia) * weighting.real(ia) for ia in range(nterms))
    exact = 1. / (W**2 * pkmesh + N)
    mask = knorm > 0.
    rel = np.abs(np.asarray((approx - exact)[mask] / exact[mask]))
    print(f'rank-{nterms} relative error: median {np.median(rel):.2e}, 99th {np.percentile(rel, 99):.2e}')
    assert np.median(rel) < 1e-2


def test_covariance_weight(meshsize=64):
    """
    The conjugate gradient C^-1, kept as a field-level reference for the separable weighting: it must
    be symmetric -- a solve stopped early is not even linear -- and give a finite field.
    W and N are built from the field itself, so the covariance describes the data it weights.
    """
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1500., 0., 0.], meshsize=meshsize)
    fkp = get_fkp(mattrs, size=int(1e6))
    cellvolume = mattrs.cellsize.prod()
    alpha = fkp.data.sum() / fkp.randoms.sum()
    kw = dict(resampler='cic', interlacing=0, compensate=False, out='real')
    W = alpha * fkp.randoms.clone(attrs=mattrs).paint(**kw).value
    particles = fkp.particles
    N = jnp.clip(particles.clone(weights=particles.weights**2, attrs=mattrs).paint(**kw).value, 0., None)
    W = W * jnp.sqrt(float(compute_fkp2_normalization(fkp, cellsize=None)) / (jnp.sum(W**2) * cellvolume))
    N = N * float(compute_fkp2_shotnoise(fkp)) / (jnp.sum(N) * cellvolume)

    cinv = covariance_weight(mattrs, power=pk, selection=W, noise=N, tol=1e-10)
    keys = random.split(random.key(42), 2)
    u, v = (mattrs.create(kind='real', fill=random.normal(key, tuple(mattrs.meshsize), dtype=mattrs.rdtype)) for key in keys)
    lhs, rhs = jnp.sum(v.value * cinv(u).value), jnp.sum(cinv(v).value * u.value)
    mismatch = float(abs(lhs - rhs) / (abs(lhs) + abs(rhs)) * 2.)
    print(f'symmetry mismatch of the CG weighting: {mismatch:.2e}')
    assert mismatch < 1e-6
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    assert jnp.all(jnp.isfinite(cinv(mesh).value))


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    test_null()
    test_null_cross()
    test_rescaling_invariance()
    test_single_term_shape()
    test_separable_inverse_weight()
    test_covariance_weight()
