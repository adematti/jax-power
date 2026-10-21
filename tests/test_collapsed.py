r"""Tests of the collapsed covariance estimator (``jaxpower.collapsed``).

* ``test_zero_mode_matches_spectra``: at the zero mode the collapsed field, divided by its count
  and normalisation, is the spectrum estimator itself -- :func:`compute_mesh2_spectrum` with a
  global and a local line of sight, :func:`compute_mesh3_spectrum` in the Sugiyama (both lines of
  sight) and Scoccimarro bases -- to machine precision, and shell 0 of the collapsed covariance is
  the outer product of the estimators.
* ``test_gaussian_expectation``: on a Gaussian field the expectation of the collapsed covariance
  spectrum has a closed form on the grid, shell by shell, including the mode geometry of the
  shell at every :math:`q`; the mock mean must match it, and the data-only disconnected estimate
  must remove it, leaving a connected part consistent with zero.
* ``test_io``: the container round-trips through write and read.
"""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from jax import random
from jax import numpy as jnp

from jaxpower import (MeshAttrs, generate_gaussian_mesh, BinMesh2SpectrumPoles, compute_mesh2_spectrum, BinMesh3SpectrumPoles, compute_mesh3_spectrum,
                      BinSoftModes, compute_mesh2_collapsed, compute_mesh2_collapsed_disconnected, compute_mesh2_collapsed_rdn0, compute_mesh3_collapsed,
                      compute_collapsed_covariance, CollapsedCovariance, CollapsedModes)


def pk(k):
    kp = 0.03
    return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)


def pkvec(kvec):
    return pk(jnp.sqrt(sum(kk**2 for kk in kvec)))


def test_zero_mode_matches_spectra():
    mattrs = MeshAttrs(meshsize=64, boxsize=1000., fft_backend='jax')
    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42, unitary_amplitude=False)
    soft = BinSoftModes(mattrs, edges={'max': 0.02})
    assert soft.nmodes[0] == 1 and soft.nshells == 4, (soft.nmodes, soft.nshells)
    # The first shell holds the 6 axis modes and the 12 face-diagonal ones
    kf = float(mattrs.kfun.min())
    assert soft.nmodes[1] == 18 and kf < soft.qavg[1] < 1.5 * kf, (soft.nmodes, soft.qavg / kf)
    edges = np.arange(0.02, 0.15, 0.03)

    for los in ['z', 'local']:
        bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=(0, 2, 4))
        ref = compute_mesh2_spectrum(mesh, bin=bin2, los=los)
        field = jax.jit(compute_mesh2_collapsed, static_argnames=['los'])(mesh, bin=bin2, soft=soft, los=los)
        test = field.spectrum()
        for ell in (0, 2, 4):
            assert np.allclose(test.get(ell).value(), ref.get(ell).value(), rtol=1e-8, atol=0.), (los, ell, test.get(ell).value(), ref.get(ell).value())
        # A pure imaginary transform of the real integrand: the zero mode is real
        assert np.allclose(field.value[..., 0].imag, 0., atol=1e-10 * np.abs(field.value[..., 0].real).max())

    ells3 = [(0, 0, 0), (2, 0, 2), (1, 1, 0), (1, 1, 2)]
    for los in ['z', 'local']:
        bin3 = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=ells3, basis='sugiyama')
        ref = compute_mesh3_spectrum(mesh, bin=bin3, los=los)
        field = jax.jit(compute_mesh3_collapsed, static_argnames=['los'])(mesh, bin=bin3, soft=soft, los=los)
        test = field.spectrum()
        for ell in ells3:
            assert np.allclose(test.get(ell).value(), ref.get(ell).value(), rtol=1e-8, atol=0.), (los, ell, test.get(ell).value(), ref.get(ell).value())

    bin3 = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=(0, 2), basis='scoccimarro')
    ref = compute_mesh3_spectrum(mesh, bin=bin3, los='z')
    field3 = jax.jit(compute_mesh3_collapsed, static_argnames=['los'])(mesh, bin=bin3, soft=soft, los='z')
    test = field3.spectrum()
    # The reference counts its closed triangles by transforming a boolean indicator, which jax
    # promotes to single precision: its mode count, hence its value, is good to ~1e-7 only.
    print('scoccimarro count, collapsed / mesh3 - 1:', np.asarray(test.get(0).nmodes) / np.asarray(ref.get(0).nmodes) - 1.)
    for ell in (0, 2):
        assert np.allclose(test.get(ell).value(), ref.get(ell).value(), rtol=1e-6, atol=0.), (ell, test.get(ell).value(), ref.get(ell).value())
    assert np.allclose(test.get(0).nmodes, ref.get(0).nmodes, rtol=1e-6)

    # Shell 0 of the collapsed covariance is the outer product of the raw spectra
    bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=(0, 2))
    field2 = compute_mesh2_collapsed(mesh, bin=bin2, soft=soft, los='z')
    cov = compute_collapsed_covariance([field2, field3])
    raw = np.concatenate([np.concatenate([field.spectrum().get(ell).value() for ell in field.ells]) for field in [field2, field3]])
    assert cov.shape == (soft.nshells, raw.size, raw.size)
    assert np.allclose(cov.value()[0], raw[:, None] * raw[None, :], rtol=1e-8)
    # The other shells are symmetric and finite
    for ishell in range(1, soft.nshells):
        v = cov.value()[ishell]
        assert np.all(np.isfinite(v)) and np.allclose(v, v.T)
    print('zero mode: ok')


def _gaussian_expectation(mattrs, soft, edges, ells, power):
    """Exact grid expectation of the collapsed covariance spectrum of a Gaussian field, ``(nshells, nells, nells, nbins)``."""
    cattrs = mattrs.clone(dtype=mattrs.cdtype)
    kvec = cattrs.kcoords(sparse=True)
    knorm = np.asarray(jnp.sqrt(sum(kk**2 for kk in kvec)))
    mu = np.asarray(sum(kk * ll for kk, ll in zip(kvec, [0., 0., 1.])) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0, 0, 0)].set(1.))
    P = np.array(power(jnp.asarray(knorm)))  # a copy: np.asarray of a jax array is read-only
    P[0, 0, 0] = 0.
    from jaxpower.utils import get_legendre
    L = [np.asarray(get_legendre(ell)(jnp.asarray(mu))) for ell in ells]
    nmax = np.asarray(soft.nmax)
    ivec = np.stack(np.unravel_index(np.asarray(soft.cube_index), tuple(2 * nmax + 1)), axis=-1) - nmax
    nbins = len(edges)
    num = np.zeros((soft.nshells, len(ells), len(ells), nbins))
    den = np.zeros((soft.nshells, nbins))
    for imode, (shift, ishell) in enumerate(zip(ivec, np.asarray(soft.ibin))):
        roll = tuple(-int(s) for s in shift)  # X(p + q) = roll(X, -q)
        Pq = np.roll(P, roll, axis=(0, 1, 2))
        Lq = [np.roll(l, roll, axis=(0, 1, 2)) for l in L]
        kq = np.roll(knorm, roll, axis=(0, 1, 2))
        for ibin, edge in enumerate(edges):
            mask = ((knorm >= edge[0]) & (knorm < edge[1])) & ((kq >= edge[0]) & (kq < edge[1]))
            npair = mask.sum()
            den[ishell, ibin] += npair**2
            for ia in range(len(ells)):
                for ib in range(len(ells)):
                    num[ishell, ia, ib, ibin] += (2 * ells[ia] + 1) * (2 * ells[ib] + 1) * np.sum(mask * P * Pq * (Lq[ia] * Lq[ib] + Lq[ia] * L[ib]))
    return num / den[:, None, None, :]


def test_gaussian_expectation(nmocks=200):
    mattrs = MeshAttrs(meshsize=32, boxsize=1000., fft_backend='jax')
    kf = float(mattrs.kfun.min())
    soft = BinSoftModes(mattrs, edges=kf * (np.arange(4) + 0.5))
    edges = np.array([[0.03, 0.05], [0.05, 0.07], [0.07, 0.09]])
    ells = (0, 2)
    bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)

    @jax.jit
    def mock(seed):
        mesh = generate_gaussian_mesh(mattrs, pkvec, seed=seed, unitary_amplitude=False)
        field = compute_mesh2_collapsed(mesh, bin=bin2, soft=soft, los='z')
        disc = compute_mesh2_collapsed_disconnected(mesh, bin=bin2, soft=soft, los='z')
        return field, disc

    values, connected = [], []
    for i in range(nmocks):
        field, disc = mock(random.key(i))
        cov = compute_collapsed_covariance([field], disconnected={(0, 0): disc})
        values.append(cov.value())
        connected.append(cov.value(connected=True))
    values, connected = np.array(values), np.array(connected)
    mean, err = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(nmocks)
    cmean, cerr = connected.mean(axis=0), connected.std(axis=0, ddof=1) / np.sqrt(nmocks)

    exact = _gaussian_expectation(mattrs, soft, edges, ells, pk)
    nbins = len(edges)
    # Shell 0 is the outer product of two estimators: its mean is Cov + <P_a><P_b>, so only shells >= 1 are tested here
    for ishell in range(1, soft.nshells):
        for ia, ella in enumerate(ells):
            for ib, ellb in enumerate(ells):
                for ibin in range(nbins):
                    i, j = ia * nbins + ibin, ib * nbins + ibin
                    pull = (mean[ishell, i, j] - exact[ishell, ia, ib, ibin]) / err[ishell, i, j]
                    ratio = mean[ishell, i, j] / exact[ishell, ia, ib, ibin] if exact[ishell, ia, ib, ibin] != 0. else np.nan
                    print(f'shell {ishell} ell ({ella}, {ellb}) bin {ibin}: mean / exact = {ratio:.4f}, pull {pull:.2f}; connected residual / exact = {cmean[ishell, i, j] / exact[ishell, ia, ia, ibin]:.4f}, pull {cmean[ishell, i, j] / cerr[ishell, i, j]:.2f}')
                    assert abs(pull) < 4.5, (ishell, ella, ellb, ibin, pull)
                    if ia == ib:
                        # The data-only subtraction removes the Gaussian part; the residual bias is O(1 / N_pair)
                        assert abs(cmean[ishell, i, j]) < 0.05 * abs(exact[ishell, ia, ib, ibin]) + 4.5 * cerr[ishell, i, j], (ishell, ella, ibin)
    # The geometric factor: the same-shell Gaussian term grows with q, as 1 / N_pair(q)
    assert np.all(exact[2, 0, 0] > exact[1, 0, 0]) and np.all(exact[3, 0, 0] > exact[2, 0, 0])
    print('gaussian expectation: ok')


def _anisotropic(amplitude=1., beta=0.5):
    """Kaiser-like power along z, which a local line of sight sees as position dependent."""
    def power(kvec):
        k2 = sum(kk**2 for kk in kvec)
        mu2 = jnp.where(k2 > 0., kvec[2]**2 / jnp.where(k2 > 0., k2, 1.), 0.)
        return amplitude * pk(jnp.sqrt(k2)) * (1. + beta * mu2)**2
    return power


def test_rdn0(ndata=100, ngaussian=10):
    """
    The Gaussian-mock (realization-dependent) disconnected estimate.

    With a global line of sight it must agree with the data-only estimate on average; with a local
    line of sight -- the Yamamoto integrand, box displaced from the observer so that angles are
    wide, and an anisotropic field so that the mean field is nonzero -- it must leave a connected
    residual consistent with zero on Gaussian data. The mocks use a fiducial spectrum 5% off in
    amplitude, which the estimate absorbs at second order.
    """
    mattrs = MeshAttrs(meshsize=32, boxsize=1000., boxcenter=(0., 0., 800.), fft_backend='jax')
    kf = float(mattrs.kfun.min())
    soft = BinSoftModes(mattrs, edges=kf * (np.arange(3) + 0.5))
    edges = np.array([[0.03, 0.05], [0.05, 0.07], [0.07, 0.09]])
    ells = (0, 2)
    bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
    truth, fiducial = _anisotropic(1.), _anisotropic(1.05)

    def gaussian_for(idata):
        # Independent mocks for each data realization, seeded apart from the data: with shared mocks
        # their noise would be a common offset that no number of data realizations averages down
        def gaussian(i):
            return generate_gaussian_mesh(mattrs, fiducial, seed=random.fold_in(random.key(100000), 1000 * idata + i), unitary_amplitude=False)
        return gaussian

    for los, mean_field in [('z', False), ('local', True)]:

        @jax.jit
        def one(idata):
            mesh = generate_gaussian_mesh(mattrs, truth, seed=random.fold_in(random.key(0), idata), unitary_amplitude=False)
            field = compute_mesh2_collapsed(mesh, bin=bin2, soft=soft, los=los)
            rdn0, mocks = compute_mesh2_collapsed_rdn0(mesh, gaussian_for(idata), ngaussian, bin=bin2, soft=soft, los=los, mean_field=mean_field)
            disc = compute_mesh2_collapsed_disconnected(mesh, bin=bin2, soft=soft, los=los) if los == 'z' else jnp.zeros_like(rdn0)
            return field, rdn0, disc, mocks['mean_field'][0] if mean_field else jnp.zeros(())

        values, residuals, others = [], [], []
        for i in range(ndata):
            field, rdn0, disc, mf = one(i)
            cov = compute_collapsed_covariance([field], disconnected={(0, 0): rdn0})
            values.append(cov.value())
            residuals.append(cov.value(connected=True))
            if los == 'z':
                others.append(compute_collapsed_covariance([field], disconnected={(0, 0): disc}).disconnected())
        values, residuals = np.array(values), np.array(residuals)
        diag = np.array([np.diagonal(v, axis1=1, axis2=2) for v in values])  # (ndata, nshells, n)
        rdiag = np.array([np.diagonal(v, axis1=1, axis2=2) for v in residuals])
        mean, rmean, rerr = diag.mean(axis=0), rdiag.mean(axis=0), rdiag.std(axis=0, ddof=1) / np.sqrt(ndata)
        for ishell in range(1, soft.nshells):
            print(f'los {los}, shell {ishell}: connected residual / total = {np.round(rmean[ishell] / mean[ishell], 4)}, pulls {np.round(rmean[ishell] / rerr[ishell], 2)}')
            assert np.all(np.abs(rmean[ishell]) < 0.05 * np.abs(mean[ishell]) + 4.5 * rerr[ishell]), (los, ishell)
        if mean_field:
            mf = np.asarray(mf)
            print(f'los {los}: mean field |F(q)| / |F(0)| on shell 1 = {np.abs(mf[:, soft.offsets[1]:soft.offsets[2]]).max(axis=1) / np.abs(mf[:, 0])}')
        if los == 'z':
            others = np.array([np.diagonal(v, axis1=1, axis2=2) for v in others])
            rdn0 = np.array([np.diagonal(v, axis1=1, axis2=2) for v in values - residuals])
            ratio = rdn0.mean(axis=0)[1:] / others.mean(axis=0)[1:]
            print(f'los z: Gaussian-mock / data-only estimate, shells >= 1: {np.round(ratio, 4)}')
            assert np.allclose(ratio, 1., atol=0.05), ratio
    print('rdn0: ok')


def test_sharded_matches_single(ndevices=4):
    """
    The soft-mode transform of a sharded real array equals that of the same array on one device.

    This is the one sharding-sensitive step that is new here. The full pipeline on several devices
    is :func:`test_jaxdecomp_matches_jax`: XLA's CPU FFT refuses sharded operands, so on CPU the
    collapsed fields themselves can only run on one device.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jaxpower import create_sharding_mesh
    if len(jax.devices()) < ndevices:
        print(f'sharded: skipped, {len(jax.devices())} devices')
        return
    mattrs = MeshAttrs(meshsize=(32, 24, 16), boxsize=1000., fft_backend='jax')
    kf = float(mattrs.kfun.min())
    soft = BinSoftModes(mattrs, edges=kf * (np.arange(4) + 0.5))
    rng = np.random.default_rng(42)
    value = rng.normal(size=tuple(mattrs.meshsize)) + 1j * rng.normal(size=tuple(mattrs.meshsize))
    ref = np.asarray(jax.jit(soft.transform)(jnp.asarray(value)))
    # Against the full transform read at the soft modes
    full = np.fft.fftn(value)
    nmax = np.asarray(soft.nmax)
    ivec = np.stack(np.unravel_index(np.asarray(soft.cube_index), tuple(2 * nmax + 1)), axis=-1) - nmax
    direct = full[tuple((ivec % np.asarray(mattrs.meshsize)).T)]
    assert np.allclose(ref, direct, rtol=1e-10, atol=1e-10 * np.abs(direct).max())
    # The FFT route gives the same modes
    assert np.allclose(np.asarray(soft.transform(jnp.asarray(value), method='fft')), direct, rtol=1e-10, atol=1e-10 * np.abs(direct).max())
    assert np.allclose(np.asarray(soft.transform(jnp.asarray(value), method='pruned')), direct, rtol=1e-10, atol=1e-10 * np.abs(direct).max())
    with create_sharding_mesh(device_mesh_shape=(2, ndevices // 2)) as sharding_mesh:
        sharded = jax.device_put(jnp.asarray(value), NamedSharding(sharding_mesh, P('x', 'y')))
        test = np.asarray(jax.jit(soft.transform)(sharded))
    assert np.allclose(test, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max()), np.abs(test - ref).max()
    print(f'sharded transform on {ndevices} devices: ok')


def test_jaxdecomp_matches_jax():
    """With the jaxdecomp backend, on however many GPUs are visible, the collapsed fields equal those of the jax backend."""
    try:
        import jaxdecomp
    except ImportError:
        print('jaxdecomp: skipped, not importable')
        return
    if jax.devices()[0].platform != 'gpu':
        print('jaxdecomp: skipped, no GPU')
        return
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jaxpower import create_sharding_mesh, RealMeshField
    ref_attrs = MeshAttrs(meshsize=32, boxsize=1000., boxcenter=(0., 0., 800.), fft_backend='jax')
    value = np.asarray(generate_gaussian_mesh(ref_attrs, pkvec, seed=42, unitary_amplitude=False).value)
    edges = np.array([[0.03, 0.05], [0.05, 0.07], [0.07, 0.09]])
    kf = float(ref_attrs.kfun.min())

    soft = BinSoftModes(ref_attrs, edges=kf * (np.arange(3) + 0.5))
    bin2 = BinMesh2SpectrumPoles(ref_attrs, edges=edges, ells=(0, 2))
    bin3 = BinMesh3SpectrumPoles(ref_attrs, edges=edges, ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama')

    def run(mesh, mattrs):
        out = []
        for los in ['z', 'local']:
            out.append(jax.jit(compute_mesh2_collapsed, static_argnames=['los'])(mesh, bin=bin2, soft=soft, los=los).value)
            out.append(jax.jit(compute_mesh3_collapsed, static_argnames=['los'])(mesh, bin=bin3, soft=soft, los=los).value)
        return [np.asarray(v) for v in out]

    ref = run(RealMeshField(jnp.asarray(value), attrs=ref_attrs), ref_attrs)
    with create_sharding_mesh() as sharding_mesh:
        mattrs = ref_attrs.clone(fft_backend='jaxdecomp')
        sharded = jax.device_put(jnp.asarray(value), NamedSharding(sharding_mesh, P(*sharding_mesh.axis_names)))
        test = run(RealMeshField(sharded, attrs=mattrs), mattrs)
    for r, t in zip(ref, test):
        assert np.allclose(t, r, rtol=1e-8, atol=1e-10 * np.abs(r).max()), np.abs(t - r).max() / np.abs(r).max()
    # And the FFT route of the transform in jaxdecomp's own layout, against the partial sum
    rng = np.random.default_rng(1)
    field = rng.normal(size=(32, 32, 32)) + 1j * rng.normal(size=(32, 32, 32))
    with create_sharding_mesh() as sharding_mesh:
        mattrs = ref_attrs.clone(fft_backend='jaxdecomp')
        soft_jd = BinSoftModes(mattrs, edges=kf * (np.arange(3) + 0.5))
        sharded = jax.device_put(jnp.asarray(field), NamedSharding(sharding_mesh, P(*sharding_mesh.axis_names)))
        fft = np.asarray(jax.jit(partial(soft_jd.transform, method='fft'))(sharded))
        pruned = np.asarray(jax.jit(partial(soft_jd.transform, method='pruned'))(sharded))
        dft = np.asarray(jax.jit(partial(soft_jd.transform, method='dft'))(sharded))
    assert np.allclose(fft, dft, rtol=1e-10, atol=1e-10 * np.abs(dft).max()), np.abs(fft - dft).max() / np.abs(dft).max()
    assert np.allclose(pruned, dft, rtol=1e-10, atol=1e-10 * np.abs(dft).max()), np.abs(pruned - dft).max() / np.abs(dft).max()
    print(f'jaxdecomp on {len(jax.devices())} devices: ok')


def _poisson_catalog(mattrs, size, seed):
    from jaxpower import generate_uniform_particles
    return generate_uniform_particles(mattrs, size=size, seed=seed)


def test_contacts_zero_mode():
    r"""
    With the self lines removed, the zero mode is the shot-noise subtracted spectrum of jaxpower's own estimators.

    Exactly for the power spectrum, which is asserted. For the bispectrum jaxpower's shot noise is
    itself an approximation, so the difference is only reported, and the exact check is the Poisson
    zero mean of :func:`test_poisson_null`. In the Sugiyama basis the two differ by 1-4% of the shot
    line, independent of the grid resolution: :func:`compute_fkp3_shotnoise` evaluates the pair line
    of legs 1 and 2 with spherical Bessel functions at the bin-mean :math:`k` (``S113``), where the
    collapsed line is the exact shell sum. Over realizations that difference is noise, with zero mean
    and 1-2% of the estimator's own scatter (``claude_collapsed/diag_contacts3_seeds.py``). The
    comparison is therefore made on the scale of the raw spectrum, not of the shot line: for
    :math:`\ell_1 = \ell_2 = 1` the shot line has zero mean and is itself noise, and both
    subtractions leave a Poisson mean consistent with zero (``claude_collapsed/diag_odd_ells.py``).
    In the Scoccimarro basis :func:`compute_fkp3_shotnoise` gives the pair lines a uniform weight
    over the third shell (arXiv:1506.02729, Eq. 58), where the collapsed line carries the exact
    triangle counts.
    """
    from jaxpower import compute_fkp2_shotnoise, compute_fkp3_shotnoise
    from jaxpower.collapsed import paint_contacts
    mattrs = MeshAttrs(meshsize=32, boxsize=1000., boxcenter=(0., 0., 800.), fft_backend='jax')
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    particles = _poisson_catalog(mattrs, size=40000, seed=random.key(7))
    mesh = particles.paint(**kw, out='real')
    mesh = mesh - mesh.mean()
    contacts = paint_contacts(particles, **kw)
    soft = BinSoftModes(mattrs, edges={'max': 0.02})
    # Below the Nyquist frequency (0.1005 here): above it the two disagree before any self pair is
    # removed, because jaxpower's Hermitian weights count the self-conjugate kz = Nyquist plane twice
    edges = np.arange(0.02, 0.101, 0.02)
    for los in ['z', 'local']:
        bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=(0, 2))
        ref = compute_mesh2_spectrum(mesh, bin=bin2, los=los)
        ref = ref.clone(num_shotnoise=compute_fkp2_shotnoise(particles, bin=bin2))
        test = compute_mesh2_collapsed(mesh, bin=bin2, soft=soft, los=los, contacts=contacts).spectrum()
        # The scale is the shot noise itself: the subtracted spectrum of a Poisson sample is noise about zero
        scale = float(np.asarray(compute_fkp2_shotnoise(particles, bin=bin2)[0])[0] / np.asarray(ref.get(0).values('norm'))[0])
        for ell in (0, 2):
            diff = np.abs(test.get(ell).value() - ref.get(ell).value()).max() / scale
            print(f'mesh2 los {los} ell {ell}: max |collapsed - jaxpower| / shot noise = {diff:.2e}')
            assert diff < 1e-6, (los, ell, diff)
        ells3 = [(0, 0, 0), (2, 0, 2), (1, 1, 0)]
        bin3 = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=ells3, basis='sugiyama')
        raw = compute_mesh3_spectrum(mesh, bin=bin3, los=los)
        ref = raw.clone(num_shotnoise=compute_fkp3_shotnoise(particles, bin=bin3, los=los, **kw))
        test = compute_mesh3_collapsed(mesh, bin=bin3, soft=soft, los=los, contacts=contacts).spectrum()
        for ell in ells3:
            diff = np.abs(test.get(ell).value() - ref.get(ell).value()).max() / np.abs(raw.get(ell).value()).max()
            print(f'mesh3 sugiyama los {los} ell {ell}: max |collapsed - jaxpower| / max |raw| = {diff:.2e} (reported only)')
    bin3 = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=(0, 2), basis='scoccimarro')
    raw = compute_mesh3_spectrum(mesh, bin=bin3, los='z')
    ref = raw.clone(num_shotnoise=compute_fkp3_shotnoise(particles, bin=bin3, los='z', **kw))
    test = compute_mesh3_collapsed(mesh, bin=bin3, soft=soft, los='z', contacts=contacts).spectrum()
    for ell in (0, 2):
        diff = np.abs(test.get(ell).value() - ref.get(ell).value()).max() / np.abs(raw.get(ell).value()).max()
        print(f'mesh3 scoccimarro ell {ell}: max |collapsed - jaxpower| / max |raw| = {diff:.2e} (reported only)')
    print('contacts zero mode: ok')


def test_poisson_null(nmocks=100, size=40000):
    r"""
    Unclustered points: once the self lines are removed, the collapsed P x B vanishes and so does the connected P x P.

    For a Poisson sample every connected spectrum is zero and the distinct-particle estimators
    leave no coincidence line to carry one, so the only surviving term is the Gaussian P x P, which
    the data-only estimate removes. Without the self lines P x B does not vanish, which is what
    gives the test its power.
    """
    from jaxpower.collapsed import paint_contacts
    mattrs = MeshAttrs(meshsize=32, boxsize=1000., fft_backend='jax')
    kw = dict(resampler='tsc', interlacing=3, compensate=True)
    kf = float(mattrs.kfun.min())
    soft = BinSoftModes(mattrs, edges=kf * (np.arange(3) + 0.5))
    edges = np.array([[0.03, 0.05], [0.05, 0.07], [0.07, 0.09]])
    bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=(0, 2))
    bin3 = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama')
    bin3s = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=[(0, 0, 0), (1, 1, 0), (1, 1, 2), (2, 0, 2)], basis='sugiyama')
    bin3c = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=(0, 2), basis='scoccimarro')

    @partial(jax.jit, static_argnames=['with_contacts'])
    def one(idata, with_contacts=True):
        particles = _poisson_catalog(mattrs, size=size, seed=random.fold_in(random.key(3), idata))
        mesh = particles.paint(**kw, out='real')
        mesh = mesh - mesh.mean()
        contacts = paint_contacts(particles, **kw) if with_contacts else None
        f2 = compute_mesh2_collapsed(mesh, bin=bin2, soft=soft, los='z', contacts=contacts)
        f3 = compute_mesh3_collapsed(mesh, bin=bin3, soft=soft, los='z', contacts=contacts)
        disc = compute_mesh2_collapsed_disconnected(mesh, bin=bin2, soft=soft, los='z')
        means = [soft.shell_sum(compute_mesh3_collapsed(mesh, bin=b, soft=soft, los='z', contacts=contacts).raw().real) for b in (bin3s, bin3c)] if with_contacts else []
        return f2, f3, disc, means

    n2 = 2 * len(edges)
    for with_contacts in [True, False]:
        pp, pb, ppdiag, bmeans = [], [], [], []
        for i in range(nmocks):
            f2, f3, disc, means = one(i, with_contacts=with_contacts)
            if with_contacts:
                bmeans.append([np.asarray(m) for m in means])
            cov = compute_collapsed_covariance([f2, f3], disconnected={(0, 0): disc})
            v, c = cov.value(), cov.value(connected=True)
            pp.append(np.diagonal(c[:, :n2, :n2], axis1=1, axis2=2))
            ppdiag.append(np.diagonal(v[:, :n2, :n2], axis1=1, axis2=2))
            # P x B in units of sqrt(diag P diag B)
            dP, dB = np.diagonal(v[:, :n2, :n2], axis1=1, axis2=2), np.diagonal(v[:, n2:, n2:], axis1=1, axis2=2)
            pb.append(v[:, :n2, n2:] / np.sqrt(np.abs(dP[:, :, None] * dB[:, None, :])))
        pp, pb, ppdiag = np.array(pp), np.array(pb), np.array(ppdiag)
        ppm, ppe = pp.mean(axis=0), pp.std(axis=0, ddof=1) / np.sqrt(nmocks)
        pbm, pbe = pb.mean(axis=0), pb.std(axis=0, ddof=1) / np.sqrt(nmocks)
        for ishell in range(1, soft.nshells):
            pull_pp = ppm[ishell] / ppe[ishell]
            pull_pb = pbm[ishell] / pbe[ishell]
            print(f'contacts {with_contacts}, shell {ishell}: P x P connected / total {np.round(ppm[ishell] / ppdiag.mean(axis=0)[ishell], 4)}, max |pull| {np.abs(pull_pp).max():.2f}; '
                  f'P x B correlation mean {np.abs(pbm[ishell]).mean():.4f}, max |pull| {np.abs(pull_pb).max():.2f}')
            if with_contacts:
                assert np.abs(pull_pp).max() < 4.5 and np.abs(pull_pb).max() < 4.5, (ishell, pull_pp, pull_pb)
        if with_contacts:
            # Zero mean of the distinct-triplet bispectrum field, per multipole, configuration and shell
            for ib, basis in enumerate(['sugiyama', 'scoccimarro']):
                b = np.array([m[ib] for m in bmeans])  # (nmocks, nells, nconf, nshells)
                pull = b.mean(axis=0) / (b.std(axis=0, ddof=1) / np.sqrt(nmocks))
                print(f'bispectrum {basis}: zero-mean pulls over multipoles, configurations and shells, max |pull| {np.abs(pull).max():.2f}, rms {np.sqrt(np.mean(pull**2)):.2f} over {pull.size} entries')
                assert np.sqrt(np.mean(pull**2)) < 1.5 and np.abs(pull).max() < 5., basis
        if not with_contacts:
            assert np.abs(pbm[1:] / pbe[1:]).max() > 10., 'without the self lines, P x B must be visibly nonzero'
    print('poisson null: ok')


def test_closed_form_trispectrum(nmocks=20000, batch=200):
    r"""
    A connected term against its closed form: the collapsed P x P of ``g + alpha (g^2 - <g^2>)`` on disjoint shells.

    On disjoint shells no Gaussian pairing survives at :math:`q \neq 0`, and the expectation of the
    collapsed covariance spectrum is, exactly on the grid, :math:`(1/V)` times the average of the
    tree-level trispectrum :math:`T(-p, p + q, p', -p' - q)` over the pairs with :math:`p, p + q`
    in one shell and :math:`p', p' + q` in the other, the shell's soft modes pooled as the estimator
    pools them. The trispectrum is the path sum of ``test_meshn._path_spectrum``, which needs no
    perturbation-theory code; the one-loop :math:`\alpha^4` term is about 1% at this amplitude.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_meshn import _path_spectrum
    from jaxpower import generate_spectrum3_mesh
    mattrs = MeshAttrs(meshsize=20, boxsize=1000., fft_backend='jax')
    kf = 2. * np.pi / 1000.

    def pk(k):
        return 2e5 * jnp.exp(-(k / (4. * kf))**2)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))
    alpha = 0.5
    edges = kf * np.array([1.5, 2.5, 3.5])
    bin2 = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=(0,))
    soft = BinSoftModes(mattrs, edges=kf * np.array([0.5, 1.5, 2.1]))

    @jax.jit
    def batch_moments(seeds):
        def one(seed):
            mesh = generate_spectrum3_mesh(mattrs, pkvec, alpha2_local=alpha, seed=seed, unitary_amplitude=False)
            mesh = mesh - mesh.mean()
            field = compute_mesh2_collapsed(mesh, bin=bin2, soft=soft, los='z')
            num = soft.shell_matmul(field.flat(), field.flat()).real  # (nshells, 2, 2)
            return num[:, 0, 1]
        return jax.vmap(one)(seeds)

    values = np.concatenate([np.asarray(batch_moments(jnp.stack([random.fold_in(random.key(11), i) for i in range(start, start + batch)])))
                             for start in range(0, nmocks, batch)])
    # The estimator's normalisation, from one realization
    mesh = generate_spectrum3_mesh(mattrs, pkvec, alpha2_local=alpha, seed=random.key(0), unitary_amplitude=False)
    field = compute_mesh2_collapsed(mesh - mesh.mean(), bin=bin2, soft=soft, los='z')
    den = np.asarray(soft.shell_matmul(field.count.astype(complex), field.count.astype(complex)).real)[:, 0, 1]
    norm = float(field.norm)
    mean = values.mean(axis=0) / den / norm**2
    err = values.std(axis=0, ddof=1) / np.sqrt(nmocks) / den / norm**2

    # Exact grid expectation
    kvec = [np.fft.fftfreq(n, d=1. / n) * kf for n in mattrs.meshsize]
    kvec = np.stack(np.meshgrid(*kvec, indexing='ij'), axis=-1).reshape(-1, 3)
    knorm = np.sqrt(np.sum(kvec**2, axis=-1))
    shells = [kvec[(knorm >= edges[i]) & (knorm < edges[i + 1])] for i in range(2)]
    nmax = np.asarray(soft.nmax)
    ivec = np.stack(np.unravel_index(np.asarray(soft.cube_index), tuple(2 * nmax + 1)), axis=-1) - nmax
    qvecs = ivec * kf
    pknp = lambda k: np.asarray(pk(jnp.asarray(k)))
    num = np.zeros(soft.nshells)
    npairs = np.zeros(soft.nshells)
    for q, ishell in zip(qvecs, np.asarray(soft.ibin)):
        pairs = []
        for shell, (lo, hi) in zip(shells, [edges[:2], edges[1:]]):
            pq = np.sqrt(np.sum((shell + q)**2, axis=-1))
            pairs.append(shell[(pq >= lo) & (pq < hi)])
        pa, pb = pairs
        npairs[ishell] += len(pa) * len(pb)
        if not len(pa) or not len(pb):
            continue
        k1 = np.repeat(-pa, len(pb), axis=0)
        k2 = np.repeat(pa + q, len(pb), axis=0)
        k3 = np.tile(pb, (len(pa), 1))
        k4 = np.tile(-pb - q, (len(pa), 1))
        num[ishell] += np.sum(_path_spectrum(alpha, pknp, k1, k2, k3, k4))
    volume = float(np.prod(np.asarray(mattrs.boxsize)))
    theory = num / npairs / volume
    for ishell in range(1, soft.nshells):
        ratio, pull = mean[ishell] / theory[ishell], (mean[ishell] - theory[ishell]) / err[ishell]
        print(f'closed form, soft shell {ishell}: collapsed {mean[ishell]:.4e} +- {err[ishell]:.1e}, exact {theory[ishell]:.4e}, ratio {ratio:.4f}, pull {pull:+.2f}')
        assert abs(pull) < 4. and abs(ratio - 1.) < 0.03 + 3. * err[ishell] / abs(theory[ishell]), (ishell, ratio, pull)
    print('closed-form trispectrum: ok')


def test_io():
    mattrs = MeshAttrs(meshsize=32, boxsize=1000., fft_backend='jax')
    mesh = generate_gaussian_mesh(mattrs, pkvec, seed=42)
    soft = BinSoftModes(mattrs, edges={'max': 0.02})
    edges = np.arange(0.02, 0.15, 0.03)
    field2 = compute_mesh2_collapsed(mesh, bin=BinMesh2SpectrumPoles(mattrs, edges=edges, ells=(0, 2)), soft=soft, los='z')
    field3 = compute_mesh3_collapsed(mesh, bin=BinMesh3SpectrumPoles(mattrs, edges=edges, ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama'), soft=soft, los='z')
    cov = compute_collapsed_covariance([field2, field3])
    fn = Path('_tests') / 'tmp_collapsed.h5'
    fn.parent.mkdir(exist_ok=True)
    cov.write(fn)
    cov2 = CollapsedCovariance.read(fn)
    assert np.allclose(cov2.value(), cov.value()) and np.allclose(cov2.q, cov.q) and np.allclose(cov2.nmodes, cov.nmodes)
    assert cov2.get(1).observable.value().shape == cov.observable.value().shape
    assert np.allclose(cov2.observable.value(), cov.observable.value())
    print('io: ok')


def test_response_ratio():
    """The stacked response ratio of rows that are an exact multiple c of other rows is c, whatever the noise in the soft
    modes, and only the shells qmin < q <= k / kratio enter."""
    rng = np.random.default_rng(1)
    q = np.array([0., 0.005, 0.01, 0.015, 0.02])
    ibin = np.repeat(np.arange(len(q)), [1, 6, 18, 26, 42])
    nm = len(ibin)
    pre = rng.normal(size=(3, nm)) + 1j * rng.normal(size=(3, nm))
    c = 0.46
    value = np.concatenate([pre, c * pre])
    modes = CollapsedModes(value=value, count=np.ones_like(value.real), norm=np.ones(6), delta=rng.normal(size=nm) + 1j * rng.normal(size=nm),
                           delta_norm=1., delta_shotnoise=0., ibin=ibin, mirror=np.arange(nm), q=q, q_edges=np.column_stack([q, q]),
                           nmodes=np.bincount(ibin), observable=None)
    k = np.array([0.08, 0.1, 0.12])
    assert np.isclose(modes.response_ratio([3, 4, 5], [0, 1, 2], k), c, rtol=1e-12)
    assert np.isclose(modes.response_ratio([3, 4, 5], [0, 1, 2], k, qmin=0.011), c, rtol=1e-12)
    # a mismatched row in the numerator moves it; excluded by qmin when it only differs on the cut shells
    X = modes.squeezed()
    sel = (q > 0.) & (q <= 0.1 / 5.)
    manual = (X[sel, 4].sum() + X[(q > 0.) & (q <= 0.08 / 5.), 3].sum() + X[(q > 0.) & (q <= 0.12 / 5.), 5].sum()) / \
             (X[sel, 1].sum() + X[(q > 0.) & (q <= 0.08 / 5.), 0].sum() + X[(q > 0.) & (q <= 0.12 / 5.), 2].sum())
    assert np.isclose(modes.response_ratio([3, 4, 5], [0, 1, 2], k), manual, rtol=1e-12)
    print('response ratio: ok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmocks', type=int, default=200)
    parser.add_argument('--only', type=str, nargs='*', default=None, help='names of the tests to run')
    args = parser.parse_args()
    tests = dict(zero_mode=test_zero_mode_matches_spectra, io=test_io, response_ratio=test_response_ratio, gaussian=lambda: test_gaussian_expectation(nmocks=args.nmocks),
                 rdn0=test_rdn0, contacts=test_contacts_zero_mode, poisson=test_poisson_null, trispectrum=test_closed_form_trispectrum, sharded=test_sharded_matches_single, jaxdecomp=test_jaxdecomp_matches_jax)
    for name, test in tests.items():
        if args.only is None or name in args.only:
            test()
