from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (BinMesh2SpectrumPoles, compute_mesh2_spectrum, generate_gaussian_mesh,
                      generate_uniform_particles, FKPField, MeshAttrs,
                      compute_fkp2_normalization, compute_fkp2_shotnoise)
from jaxpower.oqe import (Identity, RealOperator, FourierOperator, SeparableOperator,
                          ideal_weight, separable_inverse_weight, local_multipole_weight,
                          check_transpose, compute_oqe2_normalization, compute_oqe2_shotnoise,
                          Weighting, covariance_weight, real_jacobi, resolve_field,
                          compute_oqe2_mc_normalization, compute_oqe2_mc_shotnoise)


dirname = Path('_tests')


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


def _spectrum(fkp, bin, sinv=None, los='x', mc=False, nmocks=8):
    """The FKP estimator, optionally with an OQE weighting applied to the painted field."""
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    if sinv is None:
        norm = compute_fkp2_normalization(fkp, bin=bin, cellsize=None)
        num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)
    elif mc:
        mesh = resolve_field(sinv, mesh.attrs)(mesh)
        norm = compute_oqe2_mc_normalization(fkp, sinv=sinv, bin=bin, nmocks=nmocks)
        num_shotnoise = compute_oqe2_mc_shotnoise(fkp, sinv=sinv, bin=bin, nmocks=nmocks)
    else:
        mesh = resolve_field(sinv, mesh.attrs)(mesh)
        norm = compute_oqe2_normalization(fkp, sinv=sinv, bin=bin, cellsize=None)
        num_shotnoise = compute_oqe2_shotnoise(fkp, sinv=sinv, bin=bin, cellsize=None)
    spectrum = compute_mesh2_spectrum(mesh, bin=bin, los=los)
    return spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)


def test_transpose(meshsize=32):
    """Every operator must report the transpose it actually has."""
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))
    xnorm = jnp.sqrt(sum(xx**2 for xx in mattrs.rcoords(sparse=True)))

    operators = {'identity': Identity(),
                 'real': RealOperator(jnp.exp(-xnorm / 1e3)),
                 'fourier': ideal_weight(mattrs, pk),
                 'separable': SeparableOperator([(1. / (1. + knorm / 0.1), jnp.exp(-xnorm / 1e3)),
                                                 (jnp.exp(-knorm / 0.2), 1. + 0. * xnorm)])}
    for name, operator in operators.items():
        mismatch = check_transpose(operator, mattrs)
        print(f'transpose mismatch of {name}: {mismatch:.2e}')
        assert mismatch < 1e-9, name


def test_null(meshsize=64):
    """
    The whole OQE path with a unit weighting must reproduce the FKP estimator.

    This isolates the plumbing from the physics: it exercises the operator application, the
    new normalization and the new shot noise, and every one of them has to be the identity.
    """
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2, 4))
    fkp = get_fkp(mattrs)

    ref = _spectrum(fkp, bin)
    test = _spectrum(fkp, bin, sinv=Identity())

    assert np.allclose(compute_oqe2_normalization(fkp, sinv=Identity(), bin=bin, cellsize=None)[0],
                       compute_fkp2_normalization(fkp, bin=bin, cellsize=None)[0], rtol=1e-10)
    assert np.allclose(compute_oqe2_shotnoise(fkp, sinv=Identity(), bin=bin, cellsize=None)[0],
                       compute_fkp2_shotnoise(fkp, bin=bin)[0], rtol=1e-10)
    for ell in bin.ells:
        r, t = ref.get(ells=ell).value(), test.get(ells=ell).value()
        assert np.allclose(t, r, rtol=1e-8, atol=1e-8 * np.abs(r).max()), f'ell = {ell}'


def test_rescaling_invariance(meshsize=64):
    """
    The estimator is exactly invariant under ``sinv -> c sinv``.

    Both the numerator and the normalization scale as ``c^2``, so a mis-scaled weighting is
    never the problem -- only a mis-shaped one. A normalization lacking this property would
    make the estimate depend on an arbitrary constant.
    """
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0, 2))
    fkp = get_fkp(mattrs)
    xnorm = jnp.sqrt(sum(xx**2 for xx in mattrs.rcoords(sparse=True)))
    diag = jnp.exp(-xnorm / 2e3)

    ref = _spectrum(fkp, bin, sinv=SeparableOperator([(None, diag)]))
    for c in [1e-3, 7., 1e4]:
        test = _spectrum(fkp, bin, sinv=SeparableOperator([(None, c * diag)]))
        for ell in bin.ells:
            r, t = ref.get(ells=ell).value(), test.get(ells=ell).value()
            assert np.allclose(t, r, rtol=1e-8, atol=1e-8 * np.abs(r).max()), f'c = {c}, ell = {ell}'


def test_single_term_shape(meshsize=64):
    """
    A single separable term returns the FKP answer with weight ``D``, whatever ``f(k)``.

    The equality is exact mode by mode -- numerator and normalization both carry ``f^2``.
    Within a bin it is only approximate, because ``f^2`` reweights the modes of the shell, so
    the estimate becomes an ``f^2``-weighted shell average rather than a flat one. Hence the
    loose tolerance here and the sharp one in :func:`test_rescaling_invariance`: the residual
    measures the variation of ``f`` across a bin, not an error.
    """
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0,))
    fkp = get_fkp(mattrs)
    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))

    ref = _spectrum(fkp, bin, sinv=SeparableOperator([(None, 1.)]))
    test = _spectrum(fkp, bin, sinv=SeparableOperator([(1. / (1. + (knorm / 0.1)**2), 1.)]))
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
    sinv = separable_inverse_weight(mattrs, power=pk, pointing=RealOperator(W),
                                    noise=RealOperator(N), nterms=nterms)
    assert len(sinv.terms) == nterms

    knorm = jnp.sqrt(sum(kk**2 for kk in mattrs.kcoords(sparse=True)))
    pkmesh = jnp.where(knorm > 0., pk(jnp.where(knorm > 0., knorm, 1.)), 0.)
    approx = sum(f * d for f, d in sinv.terms)
    exact = 1. / (W**2 * pkmesh + N)
    mask = knorm > 0.
    rel = np.abs(np.asarray((approx - exact)[mask] / exact[mask]))
    print(f'rank-{nterms} relative error: median {np.median(rel):.2e}, 99th {np.percentile(rel, 99):.2e}')
    assert np.median(rel) < 1e-2


def test_local_multipole_weight(meshsize=32):
    """It builds, it is separable, and it carries 1 + 3 + ... terms for ells = (0, 2, ...)."""
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    poles = {0: pk, 2: lambda k: 0.5 * pk(k)}
    sinv = local_multipole_weight(mattrs, poles=poles, shotnoise=1e4, ells=(0, 2))
    assert len(sinv.terms) == 1 + 5
    mesh = mattrs.create(kind='real', fill=random.normal(random.key(42), tuple(mattrs.meshsize), dtype=mattrs.rdtype))
    assert jnp.all(jnp.isfinite(sinv(mesh).value))


def test_covariance_weight(meshsize=64):
    """
    The conjugate gradient weighting, end to end, against plain FKP.

    Two things make this a valid test where a comparison against the separable weighting is
    not. The geometry is flat -- uniform randoms -- so both estimators carry a near-trivial
    window and must agree; with a varying selection function they carry *different* windows
    and are expected to differ until each is convolved with its own. And W and N are built
    from the field itself rather than set by hand, so the weighting describes the data it
    weights: a covariance assuming far less noise than the field has will up-weight that
    noise enormously, and the shot-noise term then swamps the numerator.

    C^-1 has no closed-form A(k), so both the normalization and the shot noise are probed.

    The sample is dense on purpose. Where the shot noise dominates, P_hat is a difference of
    two nearly cancelling terms and any error in S(k) is amplified into it -- at n_bar P ~ 0.4
    this comparison reads 1.14 while nothing is wrong with the numerator. Bound the sum, not
    the terms.
    """
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1500., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01}, ells=(0,))
    fkp = get_fkp(mattrs, size=int(1e6))  # signal-dominated: see the note above

    # W and N from the field, normalized so the local model integrates to the global one
    cellvolume = mattrs.cellsize.prod()
    alpha = fkp.data.sum() / fkp.randoms.sum()
    kw = dict(resampler='cic', interlacing=0, compensate=False, out='real')
    W = alpha * fkp.randoms.clone(attrs=mattrs).paint(**kw).value
    particles = fkp.particles
    N = jnp.clip(particles.clone(weights=particles.weights**2, attrs=mattrs).paint(**kw).value, 0., None)
    W = W * jnp.sqrt(float(compute_fkp2_normalization(fkp, cellsize=None)) / (jnp.sum(W**2) * cellvolume))
    N = N * float(compute_fkp2_shotnoise(fkp)) / (jnp.sum(N) * cellvolume)
    pointing, noise = RealOperator(W), RealOperator(N)

    cg = covariance_weight(mattrs, power=pk, pointing=pointing, noise=noise, tol=1e-10)
    # a solve stopped early is not a linear operator, and this is what catches it
    mismatch = check_transpose(cg, mattrs)
    print(f'transpose mismatch of the CG weighting: {mismatch:.2e}')
    assert mismatch < 1e-6

    ref = _spectrum(fkp, bin).get(ells=0)
    test = _spectrum(fkp, bin, sinv=lambda ma: cg, mc=True, nmocks=8).get(ells=0)
    k, r, t = ref.coords('k'), ref.value(), test.value()
    # stay inside the grid: k_Nyq is pi / cellsize, and an empty selection silently gives nan
    sel = (k > 0.02) & (k < 0.8 * np.pi / mattrs.cellsize.max())
    assert sel.sum() > 3, 'empty k selection'
    dev = np.median(np.abs(t[sel] / r[sel] - 1.))
    print(f'CG against FKP on a flat geometry: median deviation {dev:.3f}')
    assert dev < 0.3


def test_weighting_requires_proxy(meshsize=16):
    """A non-separable weighting without a proxy must say so, not fail obscurely."""
    mattrs = MeshAttrs(boxsize=1000., boxcenter=[1300., 0., 0.], meshsize=meshsize)
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.05}, ells=(0,))
    fkp = get_fkp(mattrs, size=int(1e4))
    cg = covariance_weight(mattrs, power=pk, pointing=RealOperator(1.), noise=RealOperator(1e-2))
    try:
        compute_oqe2_normalization(fkp, sinv=lambda ma: cg, bin=bin, cellsize=None)
    except ValueError as exc:
        assert 'Weighting' in str(exc)
    else:
        raise AssertionError('a non-separable weighting must be refused')


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    test_transpose()
    test_null()
    test_rescaling_invariance()
    test_single_term_shape()
    test_separable_inverse_weight()
    test_local_multipole_weight()
    test_covariance_weight()
    test_weighting_requires_proxy()
