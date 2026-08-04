"""End-to-end mock validation of the scoccimarro-basis bispectrum window matrix.

No PT model is needed: we convolve the MEASURED unwindowed bispectrum multipoles
with the MEASURED window and compare against the MEASURED windowed bispectrum,
using matched Zel'dovich realizations. Both the convolution input and its target
come from the same mock ensemble.

  python validate_scocc_window_mocks.py --nmocks 16 --ellmax 2 --ninsub 16 --exact-box-limit

Geometries
----------
cutsky   : Gaussian selection (sigma = boxsize/6), local LOS. The physics test.
periodic : uniform selection, GLOBAL los='z'. Here the window is exactly the
           identity, so the matrix must reproduce its own input -- a clean
           end-to-end box-limit test with the window's own measurement and
           normalization chain in the loop. Use THIS for the identity test, not
           `box`.
box      : uniform selection but local LOS. NOT an identity test: the box center
           sits at z=1500 with size 1000, so the LOS swings by ~+-18 deg and the
           uniform selection carries genuine wide-angle structure (its L>0
           window multipoles do not vanish). Kept only for contrast.

Window-grid pitfall (cost several wrong results before it was found)
-------------------------------------------------------------------
The window's separation binning must be built with `mask_edges=''`. The default
sugiyama mask imposes (s1 + s2) <= vecmax, which truncated the window at s ~ 232
out of an available 500 (vecmax = cellsize * meshsize / 2) and left a 4x7 grid.
`interpolate_window_function(coords=256, order=3)` then resampled that onto
s in [1, 1e4] -- almost all extrapolation, since it extends a full decade past
the data and pads with only one anchor point per end. The cubic spline rang and
the interpolated Q_000 ran from -0.346 to 1.439 (negative!) where the exact
answer is 1.0 everywhere. Every mock number computed that way was wrong.

Sanity check before trusting any window: for a uniform selection the RAW
measured Q_000 must be exactly 1, and it must survive interpolation. Run
`--geometry periodic`; it should give median |nsigma| well below 1.
"""
import argparse
from pathlib import Path

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (MeshAttrs, BinMesh3SpectrumPoles, BinMesh3CorrelationPoles,
                      generate_gaussian_mesh, generate_uniform_particles, FKPField,
                      compute_mesh3_spectrum, compute_mesh3_correlation, compute_normalization,
                      compute_fkp3_normalization, interpolate_window_function,
                      get_smooth3_window_bin_attrs, compute_smooth3_spectrum_window)
from jaxpower.mesh3 import compute_fkp3_shotnoise
from jaxpower.types import Mesh3SpectrumPole
from lsstypes import ObservableTree

dirname = Path(__file__).parent
# Durable home for the mock measurements: the session scratchpad is wiped
# between sessions (this dataset was lost twice that way), and regenerating 16
# Zel'dovich realizations costs ~30 min. Keep it inside the repo's test-output
# directory instead.
CACHE_DIR = Path('/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/jax-power/tests/_tests')
CACHE_FN = CACHE_DIR / 'scocc_window_mock_bk.npz'


def _box_suffix():
    """Cache-name token for the box size. Deliberately EMPTY at the historical
    BOXSIZE=1000 so the existing cache set keeps its names; any other box gets an
    explicit tag. Without this the filenames omit boxsize entirely and a run at a
    different box silently loads box-1000 measurements."""
    return '' if BOXSIZE == 1000. else f'_box{BOXSIZE:g}'


def window_cache_fn(geometry, ellwmax, wcoords, worder, swstep, los='local'):
    """Cache for the measured+interpolated window Q. The multipole set required
    depends on (ells, ellsin, ellwmax) but NOT on ellmax -- the Gaunt J set is
    fixed by (L, L') alone -- so one cached Q serves every ellmax. The loader
    asserts the multipole set matches rather than assuming it."""
    ms = int(np.min(mattrs.meshsize))
    sw = swstep if swstep else BOXSIZE / 32.
    return CACHE_DIR / f'scocc_window_Q_{geometry}{_box_suffix()}_mesh{ms}_ellwmax{ellwmax}_w{wcoords}_o{worder}_sw{sw:g}_sig{mask_sigma:g}_dk{DK:g}_kmax{KMAX:g}{'' if KMIN is None else f'_kmin{KMIN:g}'}_los{los}.h5'


def cache_fn_for(geometry='cutsky', los='local'):
    ms = int(np.min(mattrs.meshsize))
    suff = '_periodic' if geometry == 'periodic' else ''
    return CACHE_DIR / f'scocc_window_mock_bk{suff}{_box_suffix()}_mesh{ms}_nbar{nbar:g}_sig{mask_sigma:g}_dk{DK:g}_kmax{KMAX:g}{'' if KMIN is None else f'_kmin{KMIN:g}'}_rsd{"z" if geometry == "periodic" else los}.npz'

BOXSIZE = 1000.
mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=64, boxcenter=[0., 0., 1500.])
b1, f = 2.0, 0.8
nbar = 1e-3
RFAC = 10          # randoms oversampling; the binding cost at high nbar
mask_sigma = mattrs.boxsize[0] / 6.
DK, KMAX = 0.02, 0.1      # k bin width and kmax; kNyq = pi*meshsize/boxsize must exceed KMAX
KMIN = None               # first k edge; None -> DK (theory range == output range, which
                          # biases the lowest bin -- set 0. to extend the grid below it)


def set_geometry(meshsize=None, sigma=None, boxsize=None):
    """Set meshsize and/or the selection width independently. set_meshsize resets
    sigma to boxsize/6, so call this instead when sigma is chosen explicitly."""
    global mattrs, mask_sigma, BOXSIZE
    if boxsize is not None:
        BOXSIZE = float(boxsize)
    if boxsize is not None or meshsize is not None:
        ms = int(meshsize) if meshsize is not None else int(np.min(mattrs.meshsize))
        mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=ms, boxcenter=[0., 0., 1500.])
    if sigma is not None:
        mask_sigma = float(sigma)
    return mattrs


def set_meshsize(meshsize):
    """Rebuild the geometry. meshsize matters: kNyq = pi * meshsize / boxsize, so
    meshsize=32 gives kNyq = 0.1005 while the k bins run to 0.18 -- half the bins
    are then above Nyquist and aliased (bin3 is built with mask_edges='', which
    disables the guard that would drop them). meshsize=64 gives kNyq = 0.201,
    covering the whole range."""
    global mattrs, mask_sigma
    mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=int(meshsize), boxcenter=[0., 0., 1500.])
    mask_sigma = mattrs.boxsize[0] / 6.
    return mattrs

_c3_jit = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])


def get_pk_callable():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='eisenstein_hu')
    kt = np.linspace(0.001, 0.5, 400)
    pkt = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)(kt)
    return lambda q: jnp.interp(q, jnp.asarray(kt), jnp.asarray(pkt))


def mask_weight(positions):
    """Smooth Gaussian survey selection, centered on the box."""
    d2 = sum((positions[..., i] - mattrs.boxcenter[i])**2 for i in range(3))
    return jnp.exp(-0.5 * d2 / mask_sigma**2)


def get_window_mesh(geometry='cutsky'):
    """The selection function on the mesh (analytic, noise-free).

    geometry='box' gives the uniform (periodic) selection, W == 1. The
    corresponding window matrix should then be the identity, so convolving the
    measured box bispectrum with it must return that same measurement -- an
    end-to-end box-limit test with real mock data and real error bars, with the
    window's own measurement/normalization in the loop (unlike the synthetic
    Q_000 = const unit tests, which bypass it).
    """
    if geometry == 'box':
        return mattrs.create(kind='real', fill=1.)
    # xcoords returns a tuple of per-axis coordinate arrays, not a stacked array
    xvec = mattrs.xcoords(kind='position', sparse=False)
    d2 = sum((xx - cc)**2 for xx, cc in zip(xvec, mattrs.boxcenter))
    return mattrs.create(kind='real', fill=jnp.exp(-0.5 * d2 / mask_sigma**2))


def generate_zeldovich(pk_callable, seed=42, rsd_los='z'):
    """One Zel'dovich realization on the full periodic box.

    rsd_los='z'     : plane-parallel RSD, psi_z -> (1+f) psi_z. Self-consistent
                      ONLY with an estimator using los='z' (the periodic case).
    rsd_los='local' : RSD along the LOCAL line of sight, psi -> psi + f (psi.xhat) xhat
                      with xhat the radial direction from the observer at the origin.
                      This is what the cutsky case requires: the estimator there uses
                      los='local', so imprinting the RSD about a fixed zhat instead
                      leaves the quadrupole defined about one axis but MEASURED about
                      another -- an inconsistency that shows up in ell=2 and not in
                      ell=0, since the monopole is insensitive to the LOS convention.
    """
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
    if rsd_los == 'local':
        # radial unit vector at the particle's Lagrangian position (observer at origin)
        xhat = q / jnp.sqrt(jnp.sum(q**2, axis=-1))[..., None]
        disp = disp + f * jnp.sum(disp * xhat, axis=-1)[..., None] * xhat
    else:
        disp = disp.at[..., 2].multiply(1. + f)
    x = q + disp
    lo = jnp.asarray(mattrs.boxcenter) - jnp.asarray(mattrs.boxsize) / 2.
    x = lo + (x - lo) % jnp.asarray(mattrs.boxsize)
    return particles.clone(positions=x, weights=weights, attrs=mattrs)


# The randoms are IDENTICAL for every realization (fixed seed) and so is the split
# seed, so compute_fkp3_normalization's randoms integral -- its entire cost -- is the
# same every time. Only alpha = sum(data)/sum(randoms) varies, hence
#     norm_i = norm_0 * (Sd_i / Sd_0)^3
# for the 3-point normalization. Measured against the real function: agrees to
# float32 roundoff (worst ratio 1.8e-7) at 2-34 ms instead of ~2 s, i.e. ~4 s saved
# per mock (two measurements). Keyed on id(randoms) so the windowed and unwindowed
# random sets get separate references.
# The randoms are identical for every realization, and painting is LINEAR in the
# particle weights (deposit, interlacing and compensation all are), so
#     paint(data - alpha*randoms) == paint(data) - alpha*paint(randoms)
# and the randoms mesh can be painted ONCE. That removes 5e6 of the 7.5e6 particles
# from every call. Verified against FKPField.paint: agrees to 1.1e-6 (float32
# roundoff) with a x2.4-2.8 speedup on the paint, which is ~75% of the per-mock cost.
_RMESH_CACHE = {}


def _paint_fkp_cached(data, randoms, **kw_paint):
    key = (id(randoms), tuple(sorted((k, str(v)) for k, v in kw_paint.items())))
    if key not in _RMESH_CACHE:
        _RMESH_CACHE[key] = (randoms.paint(**kw_paint), randoms.sum())
    rmesh, sr = _RMESH_CACHE[key]
    return data.paint(**kw_paint) - (data.sum() / sr) * rmesh


def clear_paint_cache():
    _RMESH_CACHE.clear()


_NORM_CACHE = {}


def _fkp3_normalization_cached(fkp, randoms, cellsize=None, split=42):
    key = (id(randoms), float(cellsize), split)
    Sd = float(fkp.data.sum())
    if key not in _NORM_CACHE:
        n0 = compute_fkp3_normalization(fkp, cellsize=cellsize, split=split)
        _NORM_CACHE[key] = (np.asarray(n0), Sd)
        return n0
    n0, S0 = _NORM_CACHE[key]
    return type(n0)(n0 * (Sd / S0)**3) if np.ndim(n0) else n0 * (Sd / S0)**3


def clear_norm_cache():
    _NORM_CACHE.clear()


def measure(data, randoms, bin3, los='local'):
    """FKP bispectrum measurement. The normalization MUST use split= (disjoint
    random subsamples): a single catalog used for all three legs self-correlates
    at zero separation and biases I3.

    los='z' (global) with uniform randoms is the genuine periodic-box setting:
    the LOS does not vary, so a uniform selection's L>0 window multipoles vanish
    and the window matrix is the exact identity. Reusing the FKP path here (dense
    uniform randoms) rather than a delta field + compute_box3_normalization is
    deliberate -- the latter derives its density factors from mesh.sum(), which
    is ~0 for a mean-subtracted delta, so the normalization collapses."""
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
    fkp = FKPField(data, randoms)
    fmesh = _paint_fkp_cached(data, randoms, **kw_paint)
    norm3 = _fkp3_normalization_cached(fkp, randoms, cellsize=2 * mattrs.cellsize[0], split=42)
    s3 = _c3_jit(fmesh, bin=bin3, los=los)
    # Shot noise at interlacing=1, NOT **kw_paint's interlacing=3. Passing kw_paint
    # makes compute_fkp3_shotnoise repaint the whole field, and that single call
    # dominated the runtime: 21.95 s against 10.44 s for the paint itself and 1.46 s
    # for the normalization (~60% of the total). Interlacing suppresses aliasing in
    # the shot-noise term to no useful purpose -- measured, the shot-noise values move
    # by 5.3e-5 between interlacing 1 and 3, against window effects of 40-126% and
    # statistical errors of 38-74%. Dropping to interlacing=1 costs 4.49 s instead of
    # 21.95 s, i.e. ~35 s per mock, ~90 min over a 150-mock run.
    kw_shotnoise = dict(kw_paint, interlacing=1)
    s3 = s3.map(lambda pole: pole.clone(norm=norm3)).clone(
        num_shotnoise=compute_fkp3_shotnoise(fkp, bin=bin3, los=los, **kw_shotnoise))
    return s3


def _window_bin_attrs(ells, ellsin, ellmax, ellwmax=None):
    """Window multipoles Q_{l1'' l2'' L''} to measure and hand to the window matrix.

    The set is whatever `get_smooth3_window_bin_attrs` requires for the requested
    (ells, ellsin, ellmax), optionally capped so ALL THREE indices satisfy
    max(l1'', l2'', L'') <= ellwmax.

    Capping every index (rather than only l2'', with l1'' left to run up to
    l2'' + L'' through the 3j band) keeps the set inside the range where the
    measured window is trustworthy. The measured multipoles decay smoothly through
    l1'' <= 4 and then become numerically meaningless -- at mesh 64 they reach
    max|Q| = 4.6e8 at (9,5,4) against Q_000 ~= 1, already within the measured
    separation range, so it is estimator noise rather than an interpolation
    artifact. Feeding those poisons the ell=2 output specifically (its Gaunt set
    needs J = 4, so l1'' reaches l2'' + 4) while leaving ell=0 clean.

    Absent multipoles are treated as ZERO downstream, so simply not providing them
    is the supported way to truncate. Measured: ellwmax=0 (Q000 only, the isotropic
    limit) reproduces the full ellwmax=2 result to 3 decimals, so the L''>0
    multipoles can be dropped outright at sigma = boxsize/10.
    """
    kw, _ = get_smooth3_window_bin_attrs(ells, ellsin=ellsin, return_ellsin=True,
                                         basis='scoccimarro', ellmax=ellmax)
    kw.setdefault('mask_edges', '')
    if ellwmax is not None:
        kw['ells'] = [q for q in kw['ells'] if max(np.atleast_1d(q).ravel()) <= ellwmax]
    return kw


def run(nmocks=16, seed0=1000, ellmax=2, ellwmax=5, ninsub=1, noutsub=1, interp='tophat', exact_box_limit=False,
        permute_in=True, permute_lgt0='full',
        theory_patch=None,
        ellsin_theory=None, from_cache=False, wcoords=256, worder=3, swstep=None, batch_size=None, buffer_size=0, geometry='cutsky',
        theory_los='local', los='local', wpole_ellcut=None, kmin_theory=None):
    pk_callable = get_pk_callable()
    ells = [0, 2]
    ellsin_theory = ellsin_theory if ellsin_theory is not None else ells
    edges3 = np.arange(DK if KMIN is None else KMIN, KMAX + 1e-9, DK)
    knyq = float(np.pi * np.min(mattrs.meshsize) / BOXSIZE)
    assert edges3[-1] <= knyq + 1e-9, f'kmax={edges3[-1]:g} exceeds kNyq={knyq:g}: raise meshsize'
    # buffer_size (not batch_size) is the estimator knob that computes several
    # band-powers at once in BinMesh3SpectrumPoles
    # Probe only ORDERED triangles k1 <= k2 <= k3 (plus the triangle inequality),
    # which is the estimator's own default convention. Passing mask_edges='' -- as
    # this harness previously did -- disables it and measures every permutation of
    # each triangle: 64 bins / 52 "valid" instead of 17 ordered ones, i.e. ~3x
    # redundant copies that are perfectly correlated, so any "N comparisons"
    # statistic quoted over them badly overstates independence. It also drops the
    # triangle-invalid bins that made compute_I(0, kout) vanish and left NaNs.
    bin3 = BinMesh3SpectrumPoles(mattrs, edges=edges3, basis='scoccimarro', ells=ells, buffer_size=buffer_size,
                                 mask_edges='mid1 <= mid2; mid2 <= mid3; '
                                            '(mid3 >= jnp.abs(mid1 - mid2)) & (mid3 <= jnp.abs(mid1 + mid2))')
    print(f'binning: dk={DK:g} kmax={edges3[-1]:g} ({len(edges3)-1} k bins, {len(bin3.xavg)} triangles), '
          f'kNyq={knyq:.4f}; mask_sigma={mask_sigma:g}; kernel width ~1/sigma={1./mask_sigma:.4f} vs dk={DK:g}; '
          f'buffer_size={buffer_size}', flush=True)
    _Veff = (2. * np.pi)**1.5 * mask_sigma**3
    print(f'selection: V_eff={_Veff:.3e} ({100*_Veff/BOXSIZE**3:.2f}% of box), N_eff=nbar*V_eff={nbar*_Veff:.0f} '
          f'galaxies; data={int(nbar*BOXSIZE**3):.3g} randoms={int(nbar*BOXSIZE**3*RFAC):.3g}', flush=True)

    cache_fn = cache_fn_for(geometry, los)
    _cached = None
    if from_cache and cache_fn.exists():
        cache = np.load(cache_fn)
        if cache['truth_vals'].shape[0] < nmocks:
            # Do NOT silently report statistics from fewer realizations than asked
            # for: a short timing run writes this same file, and reusing it would
            # quietly change the error bars underneath the comparison.
            print(f'cache has only {cache["truth_vals"].shape[0]} mocks < requested {nmocks}; regenerating',
                  flush=True)
        else:
            _cached = (cache['truth_vals'][:nmocks], cache['wind_vals'][:nmocks])
            print(f'loaded {_cached[0].shape[0]} mocks from {cache_fn}', flush=True)
    if _cached is not None:
        truth_vals, wind_vals = _cached
    else:
        randoms_box = generate_uniform_particles(mattrs, int(nbar * mattrs.boxsize.prod() * RFAC), seed=999).clone(attrs=mattrs)
        randoms_win = randoms_box.clone(weights=mask_weight(randoms_box.positions))
        truth_vals, wind_vals = [], []
        from time import time as _tm
        for i in range(nmocks):
            _t0m = _tm()
            _los = 'z' if geometry == 'periodic' else los
            data = generate_zeldovich(pk_callable, seed=seed0 + i, rsd_los=_los)
            if geometry == 'periodic':
                # identical measurement on both sides: the uniform-selection
                # window (global LOS) must reproduce its own input
                s3_truth = s3_wind = measure(data, randoms_box, bin3, los='z')
            else:
                s3_truth = measure(data, randoms_box, bin3, los=_los)
                data_win = data.clone(weights=data.weights * mask_weight(data.positions))
                s3_wind = measure(data_win, randoms_win, bin3, los=_los)
            truth_vals.append(np.stack([np.asarray(p.value()) for p in s3_truth]))
            wind_vals.append(np.stack([np.asarray(p.value()) for p in s3_wind]))
            print(f'mock {i + 1}/{nmocks} done in {_tm() - _t0m:.1f}s', flush=True)
        truth_vals, wind_vals = np.array(truth_vals), np.array(wind_vals)
        assert truth_vals.shape[0] == nmocks and nmocks > 0, \
            f'refusing to cache {truth_vals.shape[0]} mocks (nmocks={nmocks}): an empty or short ' \
            'cache silently poisons later runs -- a --nmocks 0 probe did exactly that'
        # self-describing: per-mock values plus the geometry/tracer/binning that
        # produced them, so the dataset can be reused without this script
        cache_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_fn, truth_vals=truth_vals, wind_vals=wind_vals,
                 ells=np.asarray(ells), edges3=edges3, xavg=np.asarray(bin3.xavg),
                 kedges=np.asarray(bin3.edges), basis='scoccimarro', los='local',
                 boxsize=np.asarray(mattrs.boxsize), meshsize=np.asarray(mattrs.meshsize),
                 boxcenter=np.asarray(mattrs.boxcenter), b1=b1, f=f, nbar=nbar,
                 mask_sigma=mask_sigma, seed0=seed0, nmocks=truth_vals.shape[0],
                 note=('truth_vals/wind_vals: (nmocks, nells, ntriangles) FKP bispectrum '
                       'multipoles, unwindowed (uniform randoms) and windowed (Gaussian '
                       'selection, sigma=mask_sigma). Zeldovich mocks, RSD along z, '
                       'normalization via compute_fkp3_normalization(split=42).'))
        print(f'saved {truth_vals.shape[0]} mocks to {cache_fn}', flush=True)

    if theory_los == 'z':
        # The window matrix's B_{L'} is defined for a GLOBAL line of sight, so the
        # theory vector must be the los='z' measurement. The default local-LOS
        # 'truth' is measured over the FULL box, which the observer at the origin
        # sees across +-18 deg, and it therefore carries a wide-angle quadrupole
        # deficit (12% measured, ell=0 unaffected). The windowed sample, by
        # contrast, is a sigma=50 mask at distance 1500 subtending only ~2 deg --
        # essentially plane-parallel. Feeding the local-LOS theory to predict it
        # mismatches the convention and biases ell=2 low.
        pfn = cache_fn_for('periodic', 'z')
        assert pfn.exists(), f'need the global-LOS (periodic) cache at this binning: {pfn}'
        pcache = np.load(pfn)
        assert pcache['truth_vals'].shape[0] >= nmocks, 'periodic cache too small'
        truth_vals = pcache['truth_vals'][:nmocks]
        print(f'theory vector: GLOBAL los=z, from {pfn.name} (same seeds => matched realizations)', flush=True)

    if theory_patch is not None:
        # Replace the theory values on selected triangles by a deterministic model, for bins
        # whose MEASUREMENT is unusable. Only the fully squeezed (0.015,0.015,0.015) triangle
        # qualifies at this binning: its relative scatter is 16 against <=0.1 for every other
        # leg-below-0.02 bin, and feeding it raw injects a +-7e10 per-mock fluctuation into
        # every low-k output row. Setting all realizations to the same value removes that
        # noise from the paired residual as well as from the mean.
        assert geometry not in ('box', 'periodic'), 'theory_patch would also alter wind_vals here'
        _pmask, _pvals = theory_patch
        truth_vals = np.array(truth_vals)
        for _a in range(truth_vals.shape[1]):
            truth_vals[:, _a, _pmask] = _pvals[_a][_pmask]
        print(f'theory_patch: replaced {int(np.sum(_pmask))} triangles with the model', flush=True)

    nm = truth_vals.shape[0]
    truth_mean = truth_vals.mean(axis=0)
    if geometry in ('box', 'periodic'):
        # uniform selection: the "windowed" measurement IS the box measurement,
        # so the window matrix must reproduce its own input
        wind_vals = truth_vals
    wind_mean, wind_err = wind_vals.mean(axis=0), wind_vals.std(axis=0) / np.sqrt(nm)

    # ---- measured window multipoles Q_{l1 l2 L}(s1, s2), local LOS ----
    import time as _t; _tw = _t.time()
    qfn = window_cache_fn(geometry, ellwmax, wcoords, worder, swstep, 'z' if geometry == 'periodic' else los)
    kw_expect = _window_bin_attrs(ells, ellsin_theory, ellmax, ellwmax)
    if qfn.exists():
        from jaxpower import read as _read
        Q = _read(qfn)
        if kw_expect is not None:
            got, want = sorted(tuple(q) for q in Q.ells), sorted(tuple(q) for q in kw_expect['ells'])
            assert got == want, f'cached window multipole set {got} != required {want}'
        print(f'loaded window Q ({len(Q.ells)} multipoles) from {qfn.name} in {_t.time() - _tw:.1f}s', flush=True)
        _need_build = False
    else:
        _need_build = True

    if _need_build:
        W = get_window_mesh('box' if geometry == 'periodic' else geometry)
        normW = compute_normalization(W, W, W)
        kw = _window_bin_attrs(ells, ellsin_theory, ellmax, ellwmax)
        # Window separation grid. Two things matter here:
        #  - mask_edges='': the default sugiyama mask imposes s1 + s2 <= vecmax,
        #    which truncated the window at s ~ 232 out of an available 500
        #    (vecmax = cellsize * meshsize / 2). The window is not a measurement
        #    subject to that triangle/Nyquist argument -- we want it over the full
        #    separation range the Hankel transforms will ask for.
        #  - an explicit fine step out to vecmax, so the interpolation below is
        #    mostly interpolation rather than extrapolation.
        #  - the separation STEP is a physical scale, deliberately NOT tied to
        #    cellsize: the window is smooth, so its sampling need not follow the mesh,
        #    and tying it to cellsize made the mesh-64 window measurement ~4x more
        #    (s1,s2) pairs on top of 8x the mesh cost -- which ran for 11 h without
        #    finishing. boxsize/32 reproduces the (validated) mesh-32 binning at any
        #    meshsize.
        svecmax = float(mattrs.cellsize.min()) * float(np.min(mattrs.meshsize)) / 2.
        sedges = np.arange(0., svecmax, swstep if swstep else BOXSIZE / 32.)
        if wpole_ellcut is not None:
            # Drop the window's own high-ell multipoles BEFORE measuring them: the
            # discrete 9j coupling asks for l'' up to ~2*ellmax, where the measured
            # window is noise rather than signal, and measuring them is also the
            # expensive part. Absent multipoles are treated as zero downstream.
            nall = len(kw['ells'])
            kw['ells'] = [q for q in kw['ells'] if max(np.atleast_1d(q).ravel()) <= wpole_ellcut]
            print(f'window multipoles: kept {len(kw["ells"])}/{nall} with max(l1,l2,L) <= {wpole_ellcut}', flush=True)
        sbin = BinMesh3CorrelationPoles(mattrs, edges=sedges, **kw)
        print(f'window separation grid: {len(sedges) - 1} bins, step={sedges[1] - sedges[0]:g}, '
              f'max={sedges[-1]:g}; {len(sbin.ells)} multipoles', flush=True)
        Q = compute_mesh3_correlation(W, bin=sbin, los='z' if geometry == 'periodic' else los).clone(norm=[normW] * len(sbin.ells)).unravel()
        Q = interpolate_window_function(Q, coords=wcoords, order=worder)
        print(f'window Q built in {_t.time() - _tw:.1f}s', flush=True)
        try:
            Q.write(qfn); print(f'saved window Q to {qfn.name}', flush=True)
        except Exception as exc:
            print(f'(could not cache window Q: {exc})', flush=True)

    # ---- optionally silence the window's own high-ell multipoles ----
    # The discrete method's 9j coupling drags the window up to l'' ~ 2*ellmax,
    # where the FFT-measured multipoles are noise, not signal (smooth ~1e-4 decay
    # at l=5 jumps to ~1e3 at l=7, ~2e5 at l=8) -- that is what produced the
    # median |nsigma| ~ 5.7e5 blow-up at ellmax=4. ZERO those poles rather than
    # dropping them, so the multipole set (and hence the code path) is unchanged
    # and no lookup can go missing.
    if wpole_ellcut is not None:
        nz = 0
        poles = []
        for q in Q.ells:
            pole = Q.get(ells=q)
            if max(np.atleast_1d(q).ravel()) > wpole_ellcut:
                pole = pole.clone(value=jnp.zeros_like(pole.value())); nz += 1
            poles.append(pole)
        Q = ObservableTree(poles, ells=list(Q.ells))
        print(f'window poles: zeroed {nz}/{len(Q.ells)} with max(l1,l2,L) > {wpole_ellcut}', flush=True)

    # ---- theory input: share bin3's OWN grid, so no interpolation/matching ----
    xavg = np.asarray(bin3.xavg)
    valid_in = (xavg[:, 2] >= np.abs(xavg[:, 0] - xavg[:, 1])) & (xavg[:, 2] <= xavg[:, 0] + xavg[:, 1])
    if kmin_theory is not None:
        # Drop theory triangles with any leg below kmin_theory WITHOUT re-measuring: lets a
        # grid measured from k=0 be reused as a k>=kmin_theory theory vector. Needed because
        # the [0, 0.02] bin is not usable -- its fully-squeezed triangle has a relative
        # scatter of 201 and a negative ell=2 mean (pure noise), and all such triangles feed
        # the convolution of every output bin.
        ndrop = int((valid_in & (xavg.min(axis=1) < kmin_theory - 1e-9)).sum())
        valid_in = valid_in & (xavg.min(axis=1) >= kmin_theory - 1e-9)
        print(f'theory vector: dropped {ndrop} triangles with a leg below kmin_theory={kmin_theory:g}', flush=True)
    k_in, kedges_in = jnp.asarray(xavg[valid_in]), bin3.edges[valid_in]
    edgesin = ObservableTree([Mesh3SpectrumPole(k=k_in, k_edges=kedges_in, num_raw=jnp.zeros_like(k_in[..., 0]), basis='sugiyama')
                              for _ in ellsin_theory],
                             ells=list(ellsin_theory), wa_orders=[(0, 0)] * len(ellsin_theory))

    import time
    t0 = time.time()
    # noutsub: output-side bin averaging of L_{ell2}(cos theta12), ms.tex caveat (ii). Removed
    # here when the grid path went (the original did not accept it); re-added now that
    # compute_smooth3_spectrum_window implements it.
    # permute_in: sum each theory bin over all distinct permutations of its (k1',k2',k3')
    # box. edgesin here is the ORDERED-triangle set, so without it the convolution covers
    # only the ordered octant and the ell=0 row sums come out 0.41-0.83 instead of ~Q_inf.
    wmatrix = compute_smooth3_spectrum_window(Q, edgesin=edgesin, ellsin=ellsin_theory, bin=bin3,
                                              ellmax=ellmax, ninsub=ninsub, noutsub=noutsub,
                                              interp=interp, exact_box_limit=exact_box_limit,
                                              permute_in=permute_in, permute_lgt0=permute_lgt0)
    print(f'window matrix built in {time.time() - t0:.1f}s (ninsub={ninsub}, '
          f'exact_box_limit={exact_box_limit})', flush=True)

    theory = [truth_mean[ells.index(ell)][valid_in] for ell in ellsin_theory]
    theory_vec = np.concatenate([t.ravel() for t in theory])
    pred = wmatrix.dot(theory_vec, return_type=None)

    # ---- PAIRED errors ----
    # theory and target are measured on the SAME realizations, so cosmic variance
    # is common to both and cancels in the difference. Quoting std(wind_i)/sqrt(N)
    # therefore leaves the (large) common mode in the error bar and badly
    # understates the discrepancy: at sigma=50 that unpaired error is 75% (ell=0)
    # and 189% (ell=2) of the signal, which is what made every model -- full
    # matrix, identity, even no window at all -- sit indistinguishably near 1 sigma.
    # The window matrix is linear, so pred_i = W . truth_i costs one dot per mock.
    pred_vals = []
    for i in range(nm):
        tv = np.concatenate([np.asarray(truth_vals[i][ells.index(ell)][valid_in]).ravel() for ell in ellsin_theory])
        pi = wmatrix.dot(tv, return_type=None)
        pred_vals.append(np.stack([np.asarray(pi.get(ells=ell).value()) for ell in ells]))
    pred_vals = np.asarray(pred_vals)                      # (nmocks, nells, nout)
    dif = pred_vals - np.asarray(wind_vals)                # per-realization residual
    paired_err = dif.std(axis=0) / np.sqrt(nm)

    # IDENTITY-window reference: B~ = Qinf * B, with Qinf = Q_000(s -> 0), i.e. the
    # window's own normalization. This is the window matrix reduced to a single
    # scalar with no k dependence at all, so the gap between it and the measured
    # points is the TOTAL effect of the window; the gap between it and the full
    # prediction is what the convolution actually supplies.
    wval = np.asarray(wmatrix.value())
    nout, nin = len(xavg), int(valid_in.sum())
    jof = -np.ones(nout, dtype=int)
    jof[np.where(valid_in)[0]] = np.arange(nin)
    Qinf = float(np.real(np.asarray(Q.get(ells=(0, 0, 0)).value()).ravel()[0]))
    print(f'identity-window reference: Qinf = Q_000(s->0) = {Qinf:.6f}', flush=True)
    pred_ident = np.zeros((len(ells), nout))
    for a, ell in enumerate(ells):
        if ell not in list(ellsin_theory): continue
        b = list(ellsin_theory).index(ell)
        for i in range(nout):
            j = jof[i]
            if j >= 0:
                pred_ident[a, i] = Qinf * theory[b][j]

    valid = valid_in
    nsig_all = []
    print()
    print('=== predicted (theory x window) vs measured windowed mean ===')
    for ill, ell in enumerate(ells):
        pred_vals = np.asarray(pred.get(ells=ell).value())
        print(f'--- ell={ell} ---')
        for ik in np.where(valid)[0]:
            err = wind_err[ill, ik]
            nsig = (pred_vals[ik] - wind_mean[ill, ik]) / err if err > 0 else np.nan
            if np.isfinite(nsig): nsig_all.append(abs(nsig))
            print(f'{tuple(np.round(xavg[ik], 3))}   {pred_vals[ik]: .4e}   {wind_mean[ill, ik]: .4e}   {err:.2e}   {nsig: .2f}')

    nsig_all = np.array(nsig_all)
    print()
    print(f'SUMMARY geometry={geometry} nmocks={nm} ellmax={ellmax} ellwmax={ellwmax} '
          f'ninsub={ninsub} noutsub={noutsub} interp={interp} exact_box_limit={exact_box_limit}')
    print(f'  |nsigma|: n={len(nsig_all)} median={np.median(nsig_all):.3f} mean={nsig_all.mean():.3f} '
          f'max={nsig_all.max():.3f}  frac<=1: {(nsig_all <= 1).mean():.3f}  frac<=3: {(nsig_all <= 3).mean():.3f}')

    return dict(ells=ells, xavg=xavg, valid=valid, nmocks=nm,
                pred=np.stack([np.asarray(pred.get(ells=ell).value()) for ell in ells]),
                pred_ident=pred_ident, Qinf=Qinf,
                wind_mean=wind_mean, wind_err=wind_err, truth_mean=truth_mean,
                pred_vals=pred_vals, paired_err=paired_err, truth_vals=truth_vals, wind_vals=np.asarray(wind_vals),
                valid_in=valid_in,
                nsig=nsig_all, geometry=geometry, ellmax=ellmax, ellwmax=ellwmax,
                ninsub=ninsub, noutsub=noutsub, interp=interp, exact_box_limit=exact_box_limit,
                meshsize=int(np.min(mattrs.meshsize)), nbar=nbar)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nmocks', type=int, default=16)
    ap.add_argument('--nbar', type=float, default=None, help='tracer density; N_eff = nbar * (2pi)^1.5 sigma^3 sets the error bars')
    ap.add_argument('--rfac', type=float, default=None, help='randoms oversampling factor (default 10)')
    ap.add_argument('--mask-sigma', type=float, default=None, help='selection width [Mpc/h]; default boxsize/6')
    ap.add_argument('--dk', type=float, default=None)
    ap.add_argument('--kmax', type=float, default=None)
    ap.add_argument('--buffer-size', type=int, default=0, help='band-powers computed at once in the estimator (BinMesh3SpectrumPoles)')
    ap.add_argument('--batch-size', type=int, default=None, help='batch for the window-matrix theory-bin map')
    ap.add_argument('--meshsize', type=int, default=64, help='kNyq = pi*meshsize/boxsize; 64 keeps all k bins below Nyquist')
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--ellmax', type=int, default=2)
    ap.add_argument('--wpole-ellcut', type=int, default=None, help="zero window multipoles with max(l1,l2,L) above this; the discrete 9j coupling reaches l''~2*ellmax where the measured window is pure noise")
    ap.add_argument('--ellwmax', type=int, default=2, help='cap on ALL window multipole indices max(l1,l2,L); absent multipoles are treated as zero')
    ap.add_argument('--ninsub', type=int, default=1)
    ap.add_argument('--interp', default='tophat', choices=['tophat', 'spline', 'tophat-rebin', 'spline-read'], help='theory-side/output-side binning primitives')
    ap.add_argument('--noutsub', type=int, default=1, help='output-side bin averaging of L_ell2(cos theta12); ms.tex caveat (ii)')
    ap.add_argument('--exact-box-limit', action='store_true')
    ap.add_argument('--no-permute-in', action='store_true', help='pre-fix behaviour: sum theory k\' over the ordered octant only')
    ap.add_argument('--ellsin-theory', type=int, nargs='+', default=None)
    ap.add_argument('--from-cache', action='store_true')
    ap.add_argument('--wcoords', type=int, default=256)
    ap.add_argument('--swstep', type=float, default=None, help='window separation step [Mpc/h]; default boxsize/32, NOT tied to cellsize')
    ap.add_argument('--worder', type=int, default=3, help='window interpolation spline order; 3 (cubic) RINGS on coarse window grids -> use 1')
    ap.add_argument('--geometry', choices=['cutsky', 'box', 'periodic'], default='cutsky')
    ap.add_argument('--los', choices=['local', 'z'], default='local', help='LOS used CONSISTENTLY for the RSD imprint, the estimator and the window measurement')
    ap.add_argument('--theory-los', choices=['local', 'z'], default='local', help="'z': take the theory vector from the GLOBAL-LOS (periodic) cache, the convention the window matrix assumes")
    args = ap.parse_args()
    # Assign into THIS module's globals. `import validate_scocc_window_mocks as
    # _self` does not work when the file is run as a script: the running module is
    # __main__ and that import binds a SECOND copy, so run() -- which reads
    # __main__'s globals -- never saw the override. --nbar/--rfac were silently
    # ignored that way (and --dk/--kmax only appeared to work because the values
    # passed happened to equal the defaults).
    _g = globals()
    for _name, _val in [('nbar', args.nbar), ('RFAC', args.rfac), ('DK', args.dk), ('KMAX', args.kmax)]:
        if _val is not None: _g[_name] = _val
    set_geometry(args.meshsize, args.mask_sigma)
    run(nmocks=args.nmocks, seed0=args.seed0, ellmax=args.ellmax, ellwmax=args.ellwmax, ninsub=args.ninsub, noutsub=args.noutsub, interp=args.interp, exact_box_limit=args.exact_box_limit,
        permute_in=not args.no_permute_in,
        batch_size=args.batch_size, buffer_size=args.buffer_size, theory_los=args.theory_los, los=args.los, ellsin_theory=args.ellsin_theory, from_cache=args.from_cache, wcoords=args.wcoords, worder=args.worder, swstep=args.swstep,
        geometry=args.geometry)
