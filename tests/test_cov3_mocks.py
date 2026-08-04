"""End-to-end P + B covariance workflow and mock validation, mirroring the
desi-clustering pipeline (clustering_statistics.spectrum3_tools), regrouping
the former scripts/example_fit_bias_covariance.py and
tests/test_cov3_zeldovich.py behind two flags:

--mock {gaussian, zeldovich}
    gaussian : anisotropic Gaussian mesh with Kaiser multipoles; tracer =
        uniform particles weighted by 1 + delta. The (shot-noise subtracted)
        bispectrum then has essentially no physical signal: B and the PB
        cross-covariance are exercised through their shot-noise pieces only.
    zeldovich : linear field -> Zel'dovich displacement psi = ik/k^2 delta_L;
        tracer = uniform Lagrangian particles with bias weights
        w = 1 + (b1 - 1) delta_L(q), displaced (+ RSD along z). Genuine
        gravitational bispectrum: the PB and BB non-Gaussian covariance
        terms carry physical signal.

--geometry {box, cutsky}
    box : tracer fills the full periodic box; global-z line of sight;
        analytic covariance in the periodic limit,
        compute_spectrum3_covariance(mattrs, mattrs, ...).
    cutsky : sharp sub-box selection (a stand-in for a survey mask), FKP
        estimator with fixed uniform randoms, firstpoint (P) / local (B)
        line of sight; analytic covariance from the FKP covariance windows,
        compute_spectrum3_covariance(window2, window3, ...). NOTE the
        periodic-box covariance is only a ~25%-level approximation to this
        geometry (sharp-volume idealization of the painted selection).

Pipeline (any combination):
1. nmocks realizations, FKP measurement of P_ell(k) (ells 0, 2, 4) and the
   Sugiyama-diagonal bispectrum B_{l1 l2 L}(k, k) (ells (0,0,0), (2,0,2)),
   cached one file per mock under _tests/.
2. Preliminary fit of the jaxpower.pt tracer model to the mock mean
   (production fits the single data vector), chi2 weighted by the periodic
   P-only Gaussian covariance, L-BFGS-B on jax gradients -- the no-window
   branch of run_preliminary_fit_mesh3_spectrum. By default all bias
   parameters are free for 'zeldovich'; only (b1, b2) for 'gaussian' (the
   noise-only bispectrum data cannot constrain the rest).
3. Analytic P + B covariance at the fitted theory (P / B / T callables),
   compared block by block against the mock sample covariance.

Run with the cosmodesi environment, e.g.:
    python tests/test_cov3_mocks.py --mock zeldovich --geometry cutsky --nmocks 300
"""

import argparse
from pathlib import Path

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (MeshAttrs, BinMesh2SpectrumPoles, BinMesh3SpectrumPoles,
                      generate_gaussian_mesh, generate_anisotropic_gaussian_mesh,
                      generate_uniform_particles, FKPField,
                      compute_mesh2_spectrum, compute_mesh3_spectrum,
                      compute_fkp2_normalization, compute_fkp2_shotnoise,
                      interpolate_window_function)
from jaxpower.mesh3 import compute_fkp3_shotnoise
from jaxpower import types
import lsstypes
from jaxpower.cov3 import (compute_fkp2_covariance_window, compute_fkp3_covariance_window,
                           compute_spectrum3_covariance)
from jaxpower.pt import (prepare_spectrum2_redshift_tracer, spectrum2_redshift_tracer,
                         spectrum3_redshift_tracer, spectrum4_redshift_tracer,
                         ProjectToPoles, ProjectToSell)
from jaxpower.utils import get_legendre

dirname = Path(__file__).parent / '_tests'

# ---- shared geometry / tracer parameters ----
mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200.], meshsize=64)
pattrs = mattrs.clone(boxsize=1000., meshsize=64)   # cutsky selection volume
nbar = 1e-4          # randoms (and Lagrangian) density
b1, f = 2.0, 0.8     # Eulerian linear bias (b_L = b1 - 1), growth rate

# jit the estimators once: without this, per-realization retracing (see the
# compute_mesh3_spectrum docstring) accumulates compiler memory over
# hundreds of mocks and eventually aborts XLA.
_c2_jit = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
_c3_jit = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])


def get_pk_callable():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='eisenstein_hu')
    kt = np.linspace(0.001, 0.5, 400)
    pkt = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)(kt)
    return lambda q: jnp.interp(q, jnp.asarray(kt), jnp.asarray(pkt))


def get_selection_attrs(geometry):
    """The tracer selection volume: full periodic box or the sub-box."""
    return mattrs if geometry == 'box' else pattrs


def get_randoms(geometry):
    """Fixed randoms defining the selection (and, for cutsky, the covariance
    windows -- same seed-32 realization as test_fkp3_covariance_periodic_approx)."""
    sattrs = get_selection_attrs(geometry)
    return generate_uniform_particles(sattrs, int(nbar * sattrs.boxsize.prod()), seed=32).clone(attrs=mattrs)


def generate_gaussian_fkp(pk_callable, randoms, geometry, seed=42):
    """Kaiser anisotropic Gaussian mesh; tracer = uniform particles in the
    selection volume weighted by 1 + delta (weights may go negative --
    a linear, unbiased density tracer with exactly Gaussian statistics)."""
    coefs = {0: b1**2 + 2. / 3. * b1 * f + f**2 / 5.,
             2: 4. / 3. * b1 * f + 4. / 7. * f**2,
             4: 8. / 35. * f**2}
    poles_in = {ell: (lambda k, c=c: (c * pk_callable(k)).astype(mattrs.rdtype)) for ell, c in coefs.items()}
    seeds = random.split(random.key(seed), 2)
    mesh = generate_anisotropic_gaussian_mesh(mattrs, poles_in, seed=seeds[0], los='z', unitary_amplitude=False)
    sattrs = get_selection_attrs(geometry)
    data = generate_uniform_particles(sattrs, int(nbar * sattrs.boxsize.prod()), seed=seeds[1]).clone(attrs=mattrs)
    data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
    return FKPField(data, randoms)


def generate_zeldovich_fkp(pk_callable, randoms, geometry, seed=42):
    """One Zel'dovich realization: linear Gaussian field delta_L on the full
    periodic box; ZA displacement psi = ik/k^2 delta_L; uniform Lagrangian
    particles over the full box with Lagrangian bias weights
    w = 1 + (b1 - 1) delta_L(q); Eulerian (+ RSD along z, s = x + f psi_z)
    positions periodically wrapped, then restricted to the selection.

    The selection is applied by ZEROING the weights, not boolean-masking:
    fixed-size arrays keep every jitted kernel's shapes constant across
    mocks (a variable particle count would retrigger XLA compilation each
    realization and exhaust compiler memory over hundreds of mocks), and
    zero-weight particles are exactly equivalent to removed ones for the
    FKP estimator (paint sums, sum(w), sum(w^2) shot noise)."""
    seeds = random.split(random.key(seed), 2)
    dmesh = generate_gaussian_mesh(mattrs, power=lambda kvec: pk_callable(jnp.sqrt(sum(kk**2 for kk in kvec))), seed=seeds[0])
    dk = dmesh.r2c()

    def psi_kernel(axis):
        def kernel(value, kvec):
            k2 = sum(kk**2 for kk in kvec)
            k2 = jnp.where(k2 == 0., 1., k2)
            return value * 1j * kvec[axis] / k2
        return kernel

    psis = [dk.apply(psi_kernel(axis), kind='wavenumber').c2r() for axis in range(3)]

    size = int(nbar * mattrs.boxsize.prod())
    particles = generate_uniform_particles(mattrs, size, seed=seeds[1])
    q = particles.positions
    weights = 1. + (b1 - 1.) * dmesh.read(q, resampler='cic', compensate=True)
    disp = jnp.stack([psi.read(q, resampler='cic', compensate=True) for psi in psis], axis=-1)
    # RSD along z (global LOS; the box center sits at z = 1200 so the
    # firstpoint LOS of the cutsky measurement is close to z).
    disp = disp.at[..., 2].multiply(1. + f)
    x = q + disp
    lo = jnp.asarray(mattrs.boxcenter) - jnp.asarray(mattrs.boxsize) / 2.
    x = lo + (x - lo) % jnp.asarray(mattrs.boxsize)
    if geometry == 'cutsky':
        mask = jnp.all(jnp.abs(x - jnp.asarray(pattrs.boxcenter)) <= jnp.asarray(pattrs.boxsize) / 2., axis=-1)
        weights = weights * mask
    data = particles.clone(positions=x, weights=weights, attrs=mattrs)
    return FKPField(data, randoms)


def generate_mock(mock, geometry, pk_callable, randoms, seed=42):
    gen = {'gaussian': generate_gaussian_fkp, 'zeldovich': generate_zeldovich_fkp}[mock]
    return gen(pk_callable, randoms, geometry, seed=seed)


def measure(fkp, bin2, bin3, geometry, norm2=None):
    """FKP measurement of P_ell and the Sugiyama-diagonal bispectrum. The
    bispectrum normalization uses the analytic alpha^3 nbar^3 V form (the
    single-catalog int nbar^3 moment estimator is shot-noise-biased at these
    densities; production uses split= disjoint random subsamples instead)."""
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
    los2, los3 = ('z', 'z') if geometry == 'box' else ('firstpoint', 'local')
    sattrs = get_selection_attrs(geometry)
    if norm2 is None:
        norm2 = compute_fkp2_normalization(fkp, bin=bin2, cellsize=10.)
    fmesh = fkp.paint(**kw_paint, out='complex')
    s2 = _c2_jit(fmesh, bin=bin2, los=los2).clone(
        norm=norm2, num_shotnoise=compute_fkp2_shotnoise(fkp, bin=bin2))
    alpha = float(fkp.data.weights.sum() / fkp.randoms.weights.sum())
    norm3 = alpha**3 * nbar**3 * sattrs.boxsize.prod()
    s3 = _c3_jit(fmesh, bin=bin3, los=los3)
    s3 = s3.map(lambda pole: pole.clone(norm=norm3)).clone(
        num_shotnoise=compute_fkp3_shotnoise(fkp, bin=bin3, los=los3, **kw_paint))
    return types.ObservableTree([s2, s3], fields=[(0, 0), (0, 0, 0)]), norm2


def run_preliminary_fit(spectrum2, spectrum3, mattrs_cov, pk_callable, shotnoise, free=None):
    """Mirror of run_preliminary_fit_mesh3_spectrum's no-window branch
    (clustering_statistics.spectrum3_tools): binned tracer-model multipoles
    at the measured coordinates, chi2 weighted by the periodic P-only
    Gaussian covariance, L-BFGS-B on jax gradients. Returns the fitted
    theory (P / B / T callables for compute_spectrum3_covariance), with the
    best-fit binned model attached as .spectrum2 / .spectrum3."""
    pknow_callable = pk_callable
    k_table = jnp.logspace(-3, np.log10(float(mattrs.knyq.max())), 80)
    table, table_now = prepare_spectrum2_redshift_tracer(k_table, pk_callable, pknow_callable)

    fid_bias = {'b1': 2.0, 'b2': 0.5, 'bs': -0.3, 'b3nl': 0.1,
                'c1': 0.1, 'c2': 0.2, 'X_FoG': 2., 'snb0': 0.1, 'sn0': 0.1}
    names = list(free) if free is not None else list(fid_bias)

    def make_theory(bias):
        bias_params = {0: dict(bias)}
        ells_P = list(range(0, 8, 2))
        to_poles_P = ProjectToPoles(mu=10, ells=ells_P)
        poles_P = to_poles_P(spectrum2_redshift_tracer(to_poles_P.mu, table, table_now, f, bias_params))
        k_table_P = table['matter']['k']

        def P(kvec):
            kvec = jnp.asarray(kvec)
            knorm = jnp.sqrt(jnp.sum(kvec**2, axis=-1))
            mu = jnp.where(knorm > 0., kvec[..., 2] / jnp.where(knorm > 0., knorm, 1.), 0.)
            pole_at_k = jax.vmap(lambda pole: jnp.interp(knorm.ravel(), k_table_P, pole))(poles_P)
            return sum(pole_at_k[ill].reshape(knorm.shape) * get_legendre(ell)(mu)
                       for ill, ell in enumerate(ells_P))

        def B(k1vec, k2vec, k3vec):
            return spectrum3_redshift_tracer(k1vec, k2vec, pk_callable, pknow_callable, f=f, bias_params=bias_params)

        def T(k1vec, k2vec, k3vec, k4vec):
            return spectrum4_redshift_tracer(k1vec, k2vec, k3vec, pk_callable, pknow_callable, f=f, bias_params=bias_params)

        def theory(fields):
            return {2: P, 3: B, 4: T}.get(len(fields), None)

        theory.bias = dict(bias)
        return theory

    observable = types.ObservableTree([spectrum2, spectrum3], fields=[(0, 0), (0, 0, 0)])
    data = np.concatenate([np.asarray(obs.value()).ravel() for _, obs in observable.items(level=None)])

    def theory_cov(fields):
        return make_theory(fid_bias)(fields) if len(fields) == 2 else None

    covariance = compute_spectrum3_covariance(mattrs_cov, mattrs_cov, observable, theory=theory_cov,
                                              shotnoise=shotnoise, cache={})
    Cinv = jnp.asarray(np.linalg.inv(np.asarray(covariance.value())))

    to_poles = ProjectToPoles(ells=spectrum2.ells, mu=10)
    k2d = np.asarray(next(iter(spectrum2)).coords('k'))
    mu = np.asarray(to_poles.mu)
    kvec_P = k2d[:, None, None] * np.stack([np.sqrt(1. - mu**2), np.zeros_like(mu), mu], axis=-1)
    to_Sell = ProjectToSell(ells=spectrum3.ells, size=6)
    k3d = np.asarray(next(iter(spectrum3)).coords('k'))
    k1vec_B = k3d[:, 0, None, None] * np.asarray(to_Sell.k1hat)[None, ...]
    k2vec_B = k3d[:, 1, None, None] * np.asarray(to_Sell.k2hat)[None, ...]

    def model_vector(x):
        bias_params = {0: fid_bias | dict(zip(names, x))}
        P3d = spectrum2_redshift_tracer(jnp.asarray(kvec_P), table, table_now, f, bias_params)
        p_poles = to_poles(P3d)
        B3d = spectrum3_redshift_tracer(jnp.asarray(k1vec_B), jnp.asarray(k2vec_B),
                                        pk_callable, pknow_callable, f=f, bias_params=bias_params)
        b_poles = to_Sell(B3d)
        return jnp.concatenate([p_poles.ravel(), b_poles.ravel()])

    def chi2(x):
        r = jnp.asarray(data) - model_vector(x)
        return r @ Cinv @ r

    value_and_grad = jax.jit(jax.value_and_grad(chi2))
    from scipy import optimize
    res = optimize.minimize(value_and_grad, x0=np.array([fid_bias[name] for name in names]),
                            jac=True, method='L-BFGS-B')
    best = dict(zip(names, np.asarray(res.x)))
    print(f'Preliminary fit: {best}, chi2 = {res.fun:.1f} ({data.size} data points)')
    theory = make_theory(fid_bias | best)
    theory.chi2 = float(res.fun)
    model = np.asarray(model_vector(jnp.asarray(res.x)))
    size2 = np.asarray(spectrum2.value()).size
    theory.spectrum2 = spectrum2.clone(value=model[:size2])
    theory.spectrum3 = spectrum3.clone(value=model[size2:])
    return theory


def get_covariance_windows(randoms):
    """FKP covariance windows from the (cutsky) randoms, cached on disk
    (same files as test_fkp3_covariance_periodic_approx), interpolated for
    the Hankel transforms of the covariance assembly."""
    window2_fn = dirname / 'window_fkp2_cov.h5'
    window3_fn = dirname / 'window_fkp3_cov.h5'
    if window2_fn.exists():
        window2 = types.read(window2_fn)
    else:
        window2 = compute_fkp2_covariance_window(
            randoms, edges={'step': 40.}, interlacing=2, resampler='tsc', los='local',
            group_sizes=(2, 3, 4), max_total_size=6, ells=[0, 2, 4])
        window2.write(window2_fn)
    if window3_fn.exists():
        window3 = types.read(window3_fn)
    else:
        window3 = compute_fkp3_covariance_window(
            randoms, edges={'step': 40.}, interlacing=2, resampler='tsc', los='local',
            buffer_size=50, ells=[(0, 0, 0)])
        window3.write(window3_fn)
    coords = jnp.logspace(-3, 4, 1024)
    window2 = interpolate_window_function(window2, coords=coords, order=3)
    window3 = window3.map(lambda pole: pole.unravel())
    window3 = interpolate_window_function(window3, coords=coords, order=3)
    return window2, window3


def test_cov3_mocks(mock='zeldovich', geometry='cutsky', nmocks=300, free=None, plot=False):
    import time
    assert mock in ('gaussian', 'zeldovich') and geometry in ('box', 'cutsky')
    if free is None and mock == 'gaussian':
        # noise-only bispectrum data: only (b1, b2), as in the former
        # example_fit_bias_covariance.py
        free = ('b1', 'b2')
    pk_callable = get_pk_callable()
    randoms = get_randoms(geometry)

    bin2 = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01, 'min': 0.01}, ells=(0, 2, 4))
    bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01, 'min': 0.01},
                                 ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama-diagonal')

    # ---- 1) mocks ----
    mockdir = dirname / f'cov3_mocks_{mock}_{geometry}'
    mockdir.mkdir(parents=True, exist_ok=True)
    mocks, norm2 = [], None
    t0 = time.time()
    for i in range(nmocks):
        fn = mockdir / f'mock_{i:03d}.h5'
        if fn.exists():
            obs = types.read(fn)
        else:
            fkp = generate_mock(mock, geometry, pk_callable, randoms, seed=i + 1)
            obs, norm2 = measure(fkp, bin2, bin3, geometry, norm2=norm2)
            obs.write(fn)
        mocks.append(obs)
        if (i + 1) % 25 == 0:
            print(f'{i + 1}/{nmocks} mocks ({time.time() - t0:.0f}s)', flush=True)

    cov_mock = lsstypes.cov(mocks) if nmocks > 1 else None
    mean = lsstypes.mean(mocks)
    sn = float(np.mean(mean.get(fields=(0, 0), ells=0).values('shotnoise')))
    print(f'measured shotnoise = {sn:.4g}')

    # ---- 2) preliminary fit (to the mock mean; production fits the data itself) ----
    theory = run_preliminary_fit(mean.get(fields=(0, 0)), mean.get(fields=(0, 0, 0)),
                                 get_selection_attrs(geometry), pk_callable, sn, free=free)

    # ---- 3) analytic covariance at the fitted theory ----
    observable = mocks[0]
    t0 = time.time()
    if geometry == 'box':
        cov_ana = compute_spectrum3_covariance(mattrs, mattrs, observable, theory=theory,
                                               shotnoise=sn, cache={})
    else:
        window2, window3 = get_covariance_windows(randoms)
        cov_ana = compute_spectrum3_covariance(window2, window3, observable, theory=theory,
                                               shotnoise=sn, cache={}, batch_size=16)
    print(f'analytic covariance done in {time.time() - t0:.0f}s', flush=True)
    cov_ana.write(dirname / f'cov3_{mock}_{geometry}_analytic.h5')

    # ---- 4) comparison ----
    k2 = np.asarray(bin2.xavg)
    kb = np.asarray(bin3.xavg).mean(axis=-1)
    nk, nb = len(k2), len(kb)
    names = ['P0', 'P2', 'P4', 'B000', 'B202']
    edges_flat = np.concatenate([[0], np.cumsum([nk, nk, nk, nb, nb])])

    for ill, ell in enumerate((0, 2, 4)):
        d = np.asarray(mean.get(fields=(0, 0), ells=ell).value())
        m = np.asarray(theory.spectrum2.get(ells=ell).value())
        print(f'P{ell} model/mock-mean: {np.array2string(m / d, precision=2)}')
    for ell in [(0, 0, 0), (2, 0, 2)]:
        d = np.asarray(mean.get(fields=(0, 0, 0), ells=ell).value())
        m = np.asarray(theory.spectrum3.get(ells=ell).value())
        print(f'B{ell} model/mock-mean: {np.array2string(m / d, precision=2)}')

    if cov_mock is None:
        return cov_ana, None

    v_mock = np.asarray(cov_mock.value())
    v_ana = np.asarray(cov_ana.value())
    np.save(dirname / f'cov3_{mock}_{geometry}_mock.npy', v_mock)

    print(f'\nnmocks = {nmocks}: per-bin diag scatter ~ {np.sqrt(2. / (nmocks - 1)):.3f}')
    print('=== diag ratio analytic/mock per block ===')
    for i, name in enumerate(names):
        sl = slice(edges_flat[i], edges_flat[i + 1])
        r = np.diag(v_ana)[sl] / np.diag(v_mock)[sl]
        print(f'{name:5s}: {np.array2string(r, precision=2)}')

    def corrmat(v):
        d = np.sqrt(np.diag(v))
        with np.errstate(invalid='ignore'):
            return v / np.outer(d, d)
    cm, ca = corrmat(v_mock), corrmat(v_ana)
    print('\n=== PB blocks, matched-k ===')
    for iP, iB, nameP, nameB in [(0, 3, 'P0', 'B000'), (1, 4, 'P2', 'B202')]:
        print(f'-- {nameP} x {nameB} --')
        for j in range(nb):
            i_ = np.argmin(np.abs(k2 - kb[j]))
            r, c = edges_flat[iP] + i_, edges_flat[iB] + j
            print(f'  k={kb[j]:.3f}: ana/mock={v_ana[r, c] / v_mock[r, c]:6.2f} | '
                  f'corr mock={cm[r, c]:+.3f} ana={ca[r, c]:+.3f}')

    if plot:
        from matplotlib import pyplot as plt
        fig, lax = plt.subplots(1, len(names), figsize=(16, 3.2))
        for i, (name, ax) in enumerate(zip(names, lax)):
            sl = slice(edges_flat[i], edges_flat[i + 1])
            kk = k2 if i < 3 else kb
            ax.plot(kk, np.diag(v_mock)[sl], color='C1', label='mocks')
            ax.plot(kk, np.diag(v_ana)[sl], color='C0', label='analytic')
            ax.set_yscale('log')
            ax.set_title(f'{name} ({mock}, {geometry})')
        lax[0].legend()
        plt.tight_layout()
        plt.show()

    return cov_ana, cov_mock


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mock', choices=['gaussian', 'zeldovich'], default='zeldovich')
    parser.add_argument('--geometry', choices=['box', 'cutsky'], default='cutsky')
    parser.add_argument('--nmocks', type=int, default=300)
    parser.add_argument('--free', nargs='+', default=None,
                        help='free bias parameters for the preliminary fit '
                             '(default: all for zeldovich, b1 b2 for gaussian)')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    test_cov3_mocks(mock=args.mock, geometry=args.geometry, nmocks=args.nmocks,
                    free=args.free, plot=args.plot)
