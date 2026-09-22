r"""Tests of the n-point mesh estimator (``jaxpower.meshn``).

* ``test_meshn_matches_mesh2_mesh3``: with a mask allowing equal shells, the n-generic code at
  n = 2 and n = 3 reproduces :func:`compute_mesh2_spectrum` and the Scoccimarro
  :func:`compute_mesh3_spectrum` (mode counts and values) to machine precision.
* ``test_meshn_local_transform``: the field ``g + alpha (g^2 - <g^2>)`` has closed-form tree-level
  n-point spectra, because a purely quadratic vertex makes every labelled tree a path:
  ``P_n = 2^(n-2) alpha^(n-2) sum_{paths} P(k_a) P(|k_a + k_b|) ... P(k_z)``. On a tiny mesh every
  closed n-gon of the grid is enumerated, so the estimator's expectation (the uniform average of
  ``P_n`` over them) is exact, and the mock mean must match it: this pins the mode count and the
  normalization at n = 4 and n = 5, with no dependence on any perturbation-theory code.
* ``test_meshn_gaussian_null``: on disjoint shells a Gaussian field gives zero at n = 4, 5 and 6
  (at n = 6 the surviving B B pairing vanishes for a Gaussian field too).
* ``test_meshn_io``: the output type round-trips through write and read.
"""
import itertools
import time
from functools import partial

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from jax import random
from jax import numpy as jnp

from jaxpower import (MeshAttrs, generate_gaussian_mesh, generate_spectrum3_mesh, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                      BinMesh3SpectrumPoles, compute_mesh3_spectrum, BinMeshNSpectrumPoles, compute_meshn_spectrum, compute_mesh4_spectrum,
                      compute_mesh5_spectrum, compute_mesh6_spectrum, MeshNSpectrumPoles)


def pk(k):
    kp = 0.03
    return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)


def pkvec(kvec):
    return pk(jnp.sqrt(sum(kk**2 for kk in kvec)))


def test_meshn_matches_mesh2_mesh3():
    mattrs = MeshAttrs(meshsize=64, boxsize=1000., fft_backend='jax')
    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=False)
    edges = np.arange(0.02, 0.15, 0.02)

    # n = 2 vs the power spectrum: only the diagonal (k, k) survives, giving the standard estimator
    bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=(0, 2))
    ref = compute_mesh2_spectrum(mesh, bin=bin2, los='z')
    binn = BinMeshNSpectrumPoles(mattrs, edges=edges, ells=(0, 2), ndim=2, basis='scoccimarro-diagonal')
    test = compute_meshn_spectrum(mesh, bin=binn, los='z')
    for ell in (0, 2):
        assert np.allclose(test.get(ell).nmodes, ref.get(ell).nmodes), (ell, test.get(ell).nmodes, ref.get(ell).nmodes)
        assert np.allclose(test.get(ell).value(), ref.get(ell).value(), rtol=1e-6, atol=0.), ell

    # n = 3 vs the Scoccimarro bispectrum, with its default mask (equal shells allowed)
    mask3 = ['mid1 <= mid2', 'mid2 <= mid3', '(mid3 >= jnp.abs(mid1 - mid2)) & (mid3 <= jnp.abs(mid1 + mid2))', '(edge1 + edge2 + edge3)[:, 1] <= 2 * nyq']
    bin3 = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=(0, 2), basis='scoccimarro')
    ref = compute_mesh3_spectrum(mesh, bin=bin3, los='z')
    binn = BinMeshNSpectrumPoles(mattrs, edges=edges, ells=(0, 2), ndim=3, mask_edges=mask3)
    test = compute_meshn_spectrum(mesh, bin=binn, los='z')
    assert np.allclose(binn.edges, bin3.edges)
    for ell in (0, 2):
        assert np.allclose(test.get(ell).nmodes, ref.get(ell).nmodes, rtol=1e-6), ell  # mesh3 counts in float32
        assert np.allclose(test.get(ell).value(), ref.get(ell).value(), rtol=1e-6, atol=0.), ell
    # Same with the per-configuration path, and under jit
    binn0 = BinMeshNSpectrumPoles(mattrs, edges=edges, ells=(0, 2), ndim=3, mask_edges=mask3, buffer_size=0)
    test0 = jax.jit(compute_meshn_spectrum, static_argnames=['los'])(mesh, bin=binn0, los='z')
    for ell in (0, 2):
        assert np.allclose(test0.get(ell).value(), ref.get(ell).value(), rtol=1e-6, atol=0.), ell
    # Cross-spectrum path: three different mesh objects, and a different edge array on one leg
    test = compute_meshn_spectrum(mesh, mesh.r2c(), mesh, bin=binn, los='z')
    assert np.allclose(test.get(0).value(), ref.get(0).value(), rtol=1e-6, atol=0.)
    binn = BinMeshNSpectrumPoles(mattrs, edges=[edges, edges, np.arange(0.02, 0.15, 0.04)], ells=0, ndim=3, mask_edges=mask3)
    test = compute_meshn_spectrum(mesh, bin=binn, los='z')
    assert np.all(np.isfinite(test.get(0).value()))
    print('meshn reproduces mesh2 and mesh3')


def _closed_ngons(mattrs, edges, iedges):
    """All closed n-gons of the (full, non-hermitian) grid with legs in the given 1d bins; brute force."""
    kf = 2. * np.pi / np.asarray(mattrs.boxsize)
    kvec = [np.fft.fftfreq(n, d=1. / n) * kf_ for n, kf_ in zip(mattrs.meshsize, kf)]
    kvec = np.stack(np.meshgrid(*kvec, indexing='ij'), axis=-1).reshape(-1, 3)
    knorm = np.sqrt(np.sum(kvec**2, axis=-1))
    shells = [kvec[(knorm >= edges[i]) & (knorm < edges[i + 1])] for i in iedges]
    # Extend partial sums leg by leg
    partial_sum = shells[0]
    legs = [shells[0]]
    for shell in shells[1:-1]:
        new = partial_sum[:, None, :] + shell[None, :, :]
        legs = [leg[:, None, :].repeat(len(shell), axis=1).reshape(-1, 3) for leg in legs] + [np.broadcast_to(shell[None], new.shape).reshape(-1, 3)]
        partial_sum = new.reshape(-1, 3)
    last = -partial_sum
    lnorm = np.sqrt(np.sum(last**2, axis=-1))
    mask = (lnorm >= edges[iedges[-1]]) & (lnorm < edges[iedges[-1] + 1])
    return [leg[mask] for leg in legs] + [last[mask]]


def _path_spectrum(alpha, pkfun, *legs):
    r"""Tree-level connected n-point of ``g + alpha (g^2 - <g^2>)``: sum over Hamiltonian paths.

    Path ``a - b - ... - z``: the ends are linear legs, the interior legs quadratic; the edges carry
    ``P(k_a)``, ``P(|k_a + k_b|)``, ..., ``P(k_z)``. Each labelled path comes with the leg-pairing
    multiplicity ``2^(n-2)`` (two ways to attach the two lines of each quadratic vertex), and there
    are ``n! / 2`` labelled paths.
    """
    n = len(legs)
    norm = lambda v: np.sqrt(np.sum(v**2, axis=-1))
    total = 0.
    for perm in itertools.permutations(range(n)):
        if perm[0] > perm[-1]:  # each undirected path once
            continue
        term = 1.
        partial_sum = 0.
        for i in range(n - 1):
            partial_sum = partial_sum + legs[perm[i]]
            term = term * pkfun(norm(partial_sum))
        total = total + term
    return 2.**(n - 2) * alpha**(n - 2) * total


def test_meshn_local_transform(nmocks=20000, plot=False):
    # Nyquist at 10 kf: the five smallest shells below sum to 16.25 kf on their upper edges,
    # inside the 2 nyq anti-aliasing guard
    mattrs = MeshAttrs(meshsize=20, boxsize=1000., fft_backend='jax')
    kf = 2. * np.pi / 1000.

    def pk(k):
        return 2e5 * jnp.exp(-(k / (4. * kf))**2)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))
    alpha = 0.5  # alpha sigma_g ~ 0.12: the O(alpha^4) loop is ~1% of the tree term
    # Half-fundamental-wide shells so that the brute-force enumeration below stays small
    edges = kf * np.array([1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25])

    for ndim, compute in [(4, compute_mesh4_spectrum), (5, compute_mesh5_spectrum)]:
        bin = BinMeshNSpectrumPoles(mattrs, edges=edges, ells=0, ndim=ndim)
        # Keep the enumeration affordable: the smallest shells
        iedges = np.asarray(bin._iedges)
        keep = np.flatnonzero(np.prod(np.asarray(bin.nmodes1d[0])[iedges[:, :-1]], axis=-1) <= 6e6)
        keep = keep[:6 if ndim == 4 else 3]
        assert len(keep) > 0

        @jax.jit
        def mock(seed):
            mesh = generate_spectrum3_mesh(mattrs, pkvec, alpha2_local=alpha, seed=seed, unitary_amplitude=False)
            mesh = mesh - mesh.mean()
            return compute(mesh, bin=bin, los='z').get(0).value()

        t0 = time.time()
        values = np.array([mock(random.key(i)) for i in range(nmocks)])
        print(f'n = {ndim:d}: {nmocks:d} mocks in {time.time() - t0:.1f} s, {len(bin.edges):d} configurations')
        mean, err = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(nmocks)

        pknp = lambda k: np.asarray(pk(jnp.asarray(k)))
        ratios, weights = [], []
        for iconf in keep:
            legs = _closed_ngons(mattrs, edges, iedges[iconf])
            nclosed = len(legs[0])
            # The mode count is the number of closed n-gons of the grid, exactly
            assert np.isclose(nclosed, bin.nmodes[0][iconf], rtol=1e-6), (nclosed, bin.nmodes[0][iconf])
            theory = np.mean(_path_spectrum(alpha, pknp, *legs))
            pull = (mean[iconf] - theory) / err[iconf]
            print(f'  bins {tuple(int(i) for i in iedges[iconf])}: {nclosed:d} closed {ndim:d}-gons, mock {mean[iconf]:.4e} +- {err[iconf]:.1e}, theory {theory:.4e}, ratio {mean[iconf] / theory:.4f}, pull {pull:+.2f}')
            assert abs(pull) < 4., pull
            ratios.append(mean[iconf] / theory)
            weights.append((theory / err[iconf])**2)
        ratios, weights = np.array(ratios), np.array(weights)
        ratio = np.sum(weights * ratios) / np.sum(weights)
        ratio_err = 1. / np.sqrt(np.sum(weights))
        print(f'  n = {ndim:d}: mock / theory = {ratio:.4f} +- {ratio_err:.4f} over {len(keep):d} configurations')
        assert abs(ratio - 1.) < 0.03 + 3. * ratio_err, ratio


def test_meshn_gaussian_null(nmocks=50):
    mattrs = MeshAttrs(meshsize=64, boxsize=1000., fft_backend='jax')
    edges = np.arange(0.02, 0.09, 0.01)
    for ndim, compute in [(4, compute_mesh4_spectrum), (5, compute_mesh5_spectrum), (6, compute_mesh6_spectrum)]:
        bin = BinMeshNSpectrumPoles(mattrs, edges=edges, ells=(0, 2), ndim=ndim)
        assert np.all(np.diff(np.asarray(bin._iedges), axis=1) > 0)

        @jax.jit
        def mock(seed):
            mesh = generate_gaussian_mesh(mattrs, pkvec, seed=seed, unitary_amplitude=False)
            spectrum = compute(mesh, bin=bin, los='z')
            return jnp.stack([spectrum.get(ell).value() for ell in (0, 2)])

        values = np.array([mock(random.key(i)) for i in range(nmocks)])
        mean, err = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(nmocks)
        pull = mean / err
        print(f'n = {ndim:d}: {len(bin.edges):d} configurations, max |pull| {np.abs(pull).max():.2f}')
        if ndim < 6:  # at n = 6 the B B pairing survives on disjoint shells; it vanishes for a Gaussian field too
            pass
        assert np.abs(pull).max() < 4.5


def test_meshn_io():
    mattrs = MeshAttrs(meshsize=32, boxsize=1000., fft_backend='jax')
    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42)
    bin = BinMeshNSpectrumPoles(mattrs, edges=np.arange(0.02, 0.09, 0.01), ells=(0, 2), ndim=4)
    spectrum = compute_mesh4_spectrum(mesh, bin=bin, los='z')
    assert spectrum.ndim == 4
    from pathlib import Path
    from jaxpower import read
    fn = Path('_tests') / 'tmp_meshn.h5'
    fn.parent.mkdir(exist_ok=True)
    spectrum.write(fn)
    spectrum2 = read(fn)
    assert isinstance(spectrum2, MeshNSpectrumPoles)
    assert np.allclose(spectrum2.get(0).value(), spectrum.get(0).value())
    assert spectrum.get(0).unravel().value().ndim == 4


if __name__ == '__main__':
    test_meshn_matches_mesh2_mesh3()
    test_meshn_io()
    test_meshn_gaussian_null()
    test_meshn_local_transform()
