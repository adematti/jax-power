from pathlib import Path

import numpy as np
from jax import numpy as jnp

from jaxpower import (MeshAttrs, generate_uniform_particles,
                      BinMesh2SpectrumPoles, BinMesh3SpectrumPoles,
                      Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh3SpectrumPoles,
                      interpolate_window_function)
from jaxpower import types
from jaxpower.types import Mesh3SpectrumPole
from jaxpower.cov3 import compute_fkp2_covariance_window, compute_fkp3_covariance_window, compute_spectrum3_covariance
from jaxpower.pt import (prepare_spectrum2_redshift_tracer, spectrum2_redshift_tracer,
                          spectrum3_redshift_tracer, spectrum4_redshift_tracer)


dirname = Path('_tests')


def get_theory(kmax=0.5):
    from cosmoprimo.fiducial import DESI

    cosmo = DESI(engine='eisenstein_hu')
    kt = np.linspace(0.001, 0.3, 201)
    pkt = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)(kt)

    f = 0.8
    bias_params = {0: {"b1": 2.0, "b2": 0.5, "bs": -0.3, "b3nl": 0.1,
                       "c1": 0.1, "c2": 0.2, "X_FoG": 2., "Bshot": 0.1, "Pshot": 0.1}}

    def pk_callable(q):
        return jnp.interp(q, kt, pkt)

    def pknow_callable(q):
        return jnp.interp(q, kt, pkt)

    k = jnp.logspace(-3, jnp.log10(kmax), 80)
    table, table_now = prepare_spectrum2_redshift_tracer(k, pk_callable, pknow_callable)

    def P(kvec):
        return spectrum2_redshift_tracer(kvec, table, table_now, f, bias_params)

    def B(k1vec, k2vec, k3vec):
        return spectrum3_redshift_tracer(k1vec, k2vec, pk_callable, pknow_callable,
                                         f=f, bias_params=bias_params)

    def T(k1vec, k2vec, k3vec, k4vec):
        return spectrum4_redshift_tracer(k1vec, k2vec, k3vec, pk_callable, pknow_callable,
                                          f=f, bias_params=bias_params)

    def theory(fields):
        n = len(fields)
        if n == 2:
            return P
        if n == 3:
            return B
        if n == 4:
            return T
        return None

    return theory


def test_fkp3_covariance(plot=False):
    mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200.], meshsize=64)
    pattrs = mattrs.clone(boxsize=1000., meshsize=64)

    theory = get_theory(kmax=mattrs.knyq.max())
    size = int(1e-4 * pattrs.boxsize.prod())

    randoms = generate_uniform_particles(pattrs, size, seed=32).clone(attrs=mattrs)
    edges = {'step': 40.}

    window2_fn = dirname / "window_fkp2_cov.h5"
    window3_fn = dirname / "window_fkp3_cov.h5"

    if window2_fn.exists():
        window2 = types.read(window2_fn)
    else:
        window2 = compute_fkp2_covariance_window(
            randoms,
            edges=edges,
            interlacing=2,
            resampler="tsc",
            los="local",
            group_sizes=(2, 3, 4),
            max_total_size=6,
            ells=[0, 2, 4]
        )
        window2.write(window2_fn)
        print('2-point window computed')

    if window3_fn.exists():
        window3 = types.read(window3_fn)
    else:
        window3 = compute_fkp3_covariance_window(
            randoms,
            edges=edges,
            interlacing=2,
            resampler="tsc",
            los="local",
            buffer_size=50,
            ells=[(0, 0, 0)]
        )
        window3.write(window3_fn)
        print('3-point window computed')

    # Create observables
    bin2 = BinMesh2SpectrumPoles(
        mattrs,
        edges={"step": 0.01, "min": 0.01},
        ells=[0, 2, 4],
    )
    bin3 = BinMesh3SpectrumPoles(
        mattrs,
        edges={"step": 0.01, "min": 0.01},
        ells=[(0, 0, 0), (2, 0, 2)],
        basis="sugiyama-diagonal",
    )

    observable2 = Mesh2SpectrumPoles([
        Mesh2SpectrumPole(
            k=bin2.xavg,
            k_edges=bin2.edges,
            nmodes=bin2.nmodes,
            num_raw=jnp.zeros_like(bin2.xavg),
            ell=ell,
        )
        for ell in bin2.ells
    ])
    observable3 = Mesh3SpectrumPoles([
        Mesh3SpectrumPole(
            k=bin3.xavg,
            k_edges=bin3.edges,
            nmodes=bin3.nmodes[ill],
            num_raw=jnp.zeros_like(bin3.xavg[..., 0]),
            basis=bin3.basis,
            ell=ell,
        )
        for ill, ell in enumerate(bin3.ells)
    ])
    observable = types.ObservableTree([observable2, observable3], fields=[(0, 0), (0, 0, 0)])
    coords = jnp.logspace(-3, 4, 1024)
    window2 = interpolate_window_function(window2, coords=coords, order=3)
    window3 = window3.map(lambda pole: pole.unravel())
    window3 = interpolate_window_function(window3, coords=coords, order=3)

    cov = compute_spectrum3_covariance(
        window2,
        window3,
        observable,
        theory=theory,
        shotnoise=1. / 1e-4,
        cache={},
        batch_size=16,
    )

    for block in cov.value():
        if block is not None:
            assert np.all(np.isfinite(block))

    value = cov.value()
    assert np.allclose(value, value.T, rtol=1e-5, atol=1e-12)

    if plot:
        fig = cov.plot_diag(ytransform=lambda x, y: x**2 * y, color="C0", show=True)

    return cov


def test_fkp2_covariance_pp_vs_ww(plot=False):
    """
    Cross-check compute_spectrum3_covariance's PP (spectrum-spectrum) block
    against the independent compute_spectrum2_covariance ("WW" Gaussian
    term) reference, reusing the same FKP window2 as test_fkp3_covariance.

    Resolved pitfall: compute_spectrum2_covariance must be called with
    flags=['smooth', 'fftlog'], not just ['smooth'] -- the latter silently
    uses method='bessel' (direct spherical-Bessel summation) for the window
    block instead of method='fftlog' (Correlation2Spectrum-based), which is
    the *only* method compute_QW_AB / compute_spectrum3_covariance's PP
    block use. Comparing 'bessel' against 'fftlog' gave a spurious
    order-of-magnitude (and ell-dependent) mismatch that looked like a
    window-normalization bug but was just a method mismatch; with
    'fftlog' on both sides the two agree closely.
    """
    from jaxpower.cov2 import compute_spectrum2_covariance

    mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200.], meshsize=64)
    pattrs = mattrs.clone(boxsize=1000., meshsize=64)
    theory = get_theory(kmax=mattrs.knyq.max())
    size = int(1e-4 * pattrs.boxsize.prod())

    randoms = generate_uniform_particles(pattrs, size, seed=32).clone(attrs=mattrs)
    edges = {'step': 40.}

    window2_fn = dirname / "window_fkp2_cov.h5"
    if window2_fn.exists():
        window2 = types.read(window2_fn)
    else:
        window2 = compute_fkp2_covariance_window(
            randoms,
            edges=edges,
            interlacing=2,
            resampler="tsc",
            los="local",
            group_sizes=(2, 3, 4),
            max_total_size=6,
            ells=[0, 2, 4],
        )
        window2.write(window2_fn)

    coords = jnp.logspace(-3, 4, 1024)
    window2 = interpolate_window_function(window2, coords=coords, order=3)

    bin2 = BinMesh2SpectrumPoles(mattrs, edges={"step": 0.01, "min": 0.01}, ells=(0, 2, 4))
    observable2 = Mesh2SpectrumPoles([
        Mesh2SpectrumPole(
            k=bin2.xavg,
            k_edges=bin2.edges,
            nmodes=bin2.nmodes,
            num_raw=jnp.zeros_like(bin2.xavg),
            ell=ell,
        )
        for ell in bin2.ells
    ])
    observable = types.ObservableTree([observable2], fields=[(0, 0)])

    # window3=None: this observable has no bispectrum entries, so the PB/BP/
    # PPP/BB/PT branches (the only ones needing window3) are never reached.
    shotnoise = 1. / 1e-4
    cov3 = compute_spectrum3_covariance(window2, None, observable, theory=theory, shotnoise=shotnoise, cache={})
    val3 = np.block(cov3.value())
    assert np.all(np.isfinite(val3))
    assert np.allclose(val3, val3.T, rtol=1e-5, atol=1e-6)

    # Build a matching P_ell(k) theory table via Legendre projection of the
    # same P(k, mu) theory model, for the compute_spectrum2_covariance
    # reference (which expects precomputed multipole tables, not a callable).
    # theory((0,0)) is the *bare* P(k,mu) (test_cov3.get_theory's own
    # theory(fields) adds no shot noise); compute_spectrum3_covariance's PP
    # block instead calls its own internal get_theory((a,ap)), which adds
    # +shotnoise on top since shotnoise != 0 here -- add it to the ell=0
    # multipole only (shot noise is isotropic) for a fair comparison.
    P_aa = theory((0, 0))
    nodes, weights = np.polynomial.legendre.leggauss(20)
    k = np.asarray(bin2.xavg)

    def P_ell(k, ell):
        kk = jnp.asarray(k)[:, None] * jnp.ones_like(jnp.asarray(nodes))[None, :]
        mu = jnp.ones_like(jnp.asarray(k))[:, None] * jnp.asarray(nodes)[None, :]
        kvec = jnp.stack([kk * jnp.sqrt(1 - mu**2), jnp.zeros_like(kk), kk * mu], axis=-1)
        pkmu = np.asarray(P_aa(kvec))
        legendre_ell = np.polynomial.legendre.Legendre.basis(ell)(nodes)
        out = (2 * ell + 1) / 2. * np.sum(pkmu * legendre_ell[None, :] * weights[None, :], axis=-1)
        if ell == 0:
            out = out + shotnoise
        return out

    poles_table = Mesh2SpectrumPoles([
        Mesh2SpectrumPole(k=k, k_edges=bin2.edges, nmodes=bin2.nmodes, num_raw=jnp.asarray(P_ell(k, ell)), ell=ell)
        for ell in bin2.ells
    ])

    # compute_spectrum2_covariance expects window2.fields as flat (a,b,c,d)
    # tuples (compute_mesh2_covariance_window's convention), while
    # compute_fkp2_covariance_window stores grouped fields1/fields2 labels.
    # Adapt the size-2-group subset (auto/cross power spectra) to the flat
    # convention so it can be looked up the same way.
    items, flat_fields = [], []
    for label, item in window2.items():
        f1, f2 = label['fields1'], label['fields2']
        if len(f1) == 2 and len(f2) == 2:
            items.append(item)
            flat_fields.append(f1 + f2)
    window2_flat = types.ObservableTree(items, fields=flat_fields)

    cov2 = compute_spectrum2_covariance(window2_flat, poles_table, flags=['smooth', 'fftlog'])
    val2 = np.block(cov2.value())
    assert np.all(np.isfinite(val2))
    assert np.allclose(val2, val2.T, rtol=1e-5, atol=1e-6)

    n = len(k)
    ratio = val3 / np.where(np.abs(val2) > 1e-300, val2, np.nan)
    print("val3 / val2 ratio (diagonal), median per ell:")
    for ib, ell in enumerate(bin2.ells):
        r = np.diag(ratio)[ib * n:(ib + 1) * n]
        print(f"  ell={ell}: median={np.nanmedian(r):.4g} min={np.nanmin(r):.4g} max={np.nanmax(r):.4g}")

    assert np.allclose(val3, val2, rtol=2e-2, atol=1e-6 * np.max(np.abs(val2)))

    if plot:
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for ax, val, title in zip(axs, [val3, val2], ['compute_spectrum3_covariance (PP)', 'compute_spectrum2_covariance (WW)']):
            ax.imshow(np.log10(np.abs(val) + 1e-300))
            ax.set_title(title)
        plt.show()

    return val3, val2


if __name__ == '__main__':

    test_fkp3_covariance(plot=True)
    test_fkp2_covariance_pp_vs_ww(plot=True)
