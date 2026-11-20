# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package does

`jax-power` estimates auto and cross power spectra (and bispectra) for cosmological surveys. It supports periodic boxes and survey geometries, global and local line-of-sight, and distributed multi-GPU computation via JAX and `jaxdecomp`.

## Installation

```bash
pip install -e .
```

Key dependencies: `jax`, `jaxdecomp` (distributed FFT), `lsstypes` (observable data structures, installed from GitHub).

## Running tests

Tests use plain Python (no pytest runner is required — each file has an `if __name__ == '__main__':` block), but pytest works too:

```bash
# Run a full test file
python tests/test_mesh2.py

# Run a single test function
python -c "from tests.test_mesh2 import test_mesh2_spectrum; test_mesh2_spectrum(plot=True)"

# With pytest
pytest tests/test_mesh2.py::test_mesh2_spectrum -s
```

Test output files are written to `tests/_tests/`. Multi-GPU tests require `jax.distributed.initialize()` and must be launched with `mpirun` or equivalent.

## Architecture

### Core data model

**`MeshAttrs`** (`mesh.py`) is the central configuration object: it holds `meshsize`, `boxsize`, `boxcenter`, and derived quantities (`cellsize`, `kfun`, etc.). Almost every function accepts a `MeshAttrs` or a mesh field (which carries its own `MeshAttrs` as `.attrs`).

**Mesh fields** (`mesh.py`):
- `RealMeshField` — real-space field on a 3-D grid (wraps a JAX array + `MeshAttrs`)
- `ComplexMeshField` — Fourier-space field; supports `r2c`/`c2r` round-trips
- `ParticleField` — particle positions + weights; `.paint()` produces a mesh field
- `FKPField` — composite `data - randoms` field for survey geometry

Fields are frozen `@dataclass`s registered as JAX pytrees so they can pass through `jax.jit` boundaries.

### JAX pytree registration pattern

All dataclasses that must survive `jax.jit` / `jax.vmap` / `jax.grad` are registered as pytrees. Two mechanisms are used:

- **`@register_pytree_dataclass(meta_fields=[...])`** (`utils.py:653`) — decorator for `Bin*` dataclasses. Fields listed in `meta_fields` become static aux data (must be hashable); all other annotated fields are dynamic (JAX arrays).
- **`jax.tree_util.register_dataclass(data_fields=..., meta_fields=...)`** — used directly for `FKPField`.
- **`staticarray`** (`mesh.py:46`) — immutable numpy array subclass used for numpy arrays that live in `meta_fields`; it is hashable and equality-comparable so JAX can cache JIT traces.
- Observable types (`Spectrum`, `Correlation`, `WindowMatrix`, `CovarianceMatrix`) from `lsstypes` are registered via `make_leaf_pytree` / `make_tree_pytree` / `make_window_pytree` in `types.py`.

### Computation pipeline

1. **Define geometry**: `MeshAttrs(meshsize=..., boxsize=...)` or `get_mesh_attrs(data_positions, randoms_positions, ...)` for survey data.
2. **Define binning**: `BinMesh2SpectrumPoles(mattrs, edges=..., ells=(0,2,4))` — a pytree-compatible static config.
3. **Paint particles**: `ParticleField(...).paint(resampler='tsc', interlacing=3, compensate=True)` → `RealMeshField`.
4. **Compute spectrum**: `compute_mesh2_spectrum(mesh, bin=bin, los='firstpoint')` → `Mesh2SpectrumPoles`.
5. **Apply normalization**: `spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)`.

For bispectra, replace `mesh2` → `mesh3` throughout.

### Module map

| Module | Role |
|---|---|
| `mesh.py` | `MeshAttrs`, all field classes, sharding, particle exchange |
| `mesh2.py` | 2-point power spectrum: `BinMesh2SpectrumPoles`, `compute_mesh2_spectrum`, window matrices |
| `mesh3.py` | 3-point bispectrum: `BinMesh3SpectrumPoles`, `compute_mesh3_spectrum` |
| `meshn.py` | n-point spectrum with all legs binned in `|k|` (trispectrum and higher, for covariance validation): `BinMeshNSpectrumPoles`, `compute_meshn_spectrum`, `compute_mesh{4,5,6}_spectrum` |
| `particle2/3.py` | Direct particle-pair / triplet statistics (uses `cucount.jax`) |
| `cov2.py` | 2-point covariance and window matrices; `compute_fkp2_covariance_window` |
| `cov3.py` | Power spectrum x bispectrum covariance in a periodic box, arXiv:1908.06234 term by term: `compute_spectrum3_covariance`, plus the survey-window builders `compute_fkp2/3_covariance_window`. Multitracer throughout; every term has a survey window. Build those windows with `split=<seed>`, or the randoms' self-pairs inflate `1/Q_W(s->0)` by up to 33% |
| `oqe.py` | Optimal quadratic estimator weightings for the FKP-style estimator. A weighting is a factory `MeshAttrs -> SeparableWeighting`, rebuilt on each mesh that needs it; `SeparableWeighting(fourier, real, nterms, selection)` applies `W IFFT[sum_a f_a FFT[D_a u]]`, with the factors `fourier(a)`, `real(a)` evaluated on demand (a rank-M weighting stores W, N and the fit, not 2M meshes); `norm_real(a)` = `D_a W` is the normalization proxy. `separable_inverse_weight` fits the rank-M terms to `C^-1`; `compute_oqe2_normalization` / `compute_oqe2_shotnoise` contract the proxy in closed form (one or two FKP fields sharing the weighting); `covariance_weight` applies the exact `C^-1` by preconditioned conjugate gradients (sharding-safe `_cg`), a field-level reference with no normalization or window. The selection sits on the output side too -- dropping it stays unbiased and is ~70x noisier. Only what `desi-clustering` needs lives here |
| `ic.py` | Exact window matrices of template deprojection (integral constraints) for the power spectrum, on the sphere with the local line of sight: the RIC of shuffled randoms, imaging weights fitted on the catalog (linear regression, or a calibrated harmonic response such as SYSNet), cap means. One input container, `Template(angular, radial, redges, mean=None, response=None, group=None)`: HEALPix map(s) of a sky pattern (`(npix,)` or `(nsplit, npix)` independent random subsets) times a radial profile on the shells `redges` it carries (shapes checked; all templates of a model must share them); the survey regions (`geometry`) have no mean/response; the templates of what the pipeline removes (`ic`: `ric_templates`, `amr_templates`, `harmonic_templates` (with the normalization of the weights per region), or hand-built) carry the mean window over which their amplitude is measured and its response `'diag'` / `'dense'` / rank-one `'outer'`, or `'harmonic'` (calibrated weights such as SYSNet: mode-coupling matrices, built with `harmonic_templates`, which adds the normalization of the weights to mean 1 per region as in the pipeline (required by `IC2Model`): their survey mean is then zero, and `gic_templates` skips them; mixable with the others, e.g. SYSNet + RIC, the harmonic x regular cross terms through `N^{Ra}_{L'L} = C_L'[Theta_R sum_M U_{a,LM} Y_LM, Theta_a]`, one SHT per mean window, degree and harmonic region; `removed['h']`, pairs `('f', 'h')`, `('h', 'h')`, `('h', 't')`). Removed fields `'t'` (regular templates, plus the global integral constraint of alpha as the last template, `W (mean_W delta - mean of the templates)`: `gic_templates(geometry, ic)`, added by `gic=True`) and `'h'`; the field is `W delta`, the plain window convolution, so Delta P includes the global IC (relative to the merged arm of the mocks, which has alpha: subtract the model with `ic=[]`). `IC2Model(geometry, ic, norm, lmax, wsum=None)` (shells from the templates; `norm` the measured spectrum's normalization, divided by alpha^2 = (wsum / int W)^2 with `wsum` the sum of the data weights, so that Delta P is normalized as the measurement) normalizes non-negative maps, removes the pixel window, adds the alpha template and computes the geometric factors (healpy pair-count functions, one SHT per template pattern and degree, the dominant cost). JAX: `compute_smooth2_ic_window` (one jitted `lax.map` over theory columns; any theory multipoles `ellsin`; Delta P = -P^ft - P^tf + P^tt and each term, theory bands integrated exactly with `BesselIntegral`, l <= 8 for the bispectrum removal kernels; harmonic templates too), `compute_smooth3_ic_window` (bispectrum: sparse `D^BB` from the theory Sugiyama multipoles on the (k1', k2') grid -- each leg replaced by the removed modes q, the third leg's triangles read as the pairs (q, k2), (k1, q) -- with q a grid magnitude the window's six multipoles suffice (as (-(q + k2), k2) the squeezed triangles sit in the opening angle: 39% low at k = 0.12 even 10% with l <= 8) -- (l2, l1, L) from (l1, l2, L) by symmetry -- and dense `D^BP` (modulation x theory P at nodes); composed with the geometry window if given; ONE closure-leg rule for every term (removal, modulation, shot noise of the removal), any bins (diagonal or not): the closure leg composed for k3 >= `qsplit` (0.02), below replaced by its exact squeezed-limit term when given -- `closure_kernel` (C^BB, `'BB'` = W^geo D^BB + C^BB, `'CBB'` alone), `closure_modulation` (C^BP from P^eta_W2, added to `'BP'`, `'CBP'` alone), `closure_shotnoise` (C^BN from -N_W2, added to `'BN'`, `'CBN'` alone); qsplit=0: plain composition, qsplit=inf: exact closure at all q; P^eta, N and their W^2 versions given as lsstypes multipoles (`modulation`, `shotnoise`, `closure_*`: e.g. Mesh2SpectrumPoles on the observable of the window they come from), the theory P grid as `pin` (lsstypes; `out['BP'].dot(pin)`); ONE interpolation rule for every tabulated function (`_interp_matrix`: linear, first value below, zero above); with `shotnoise`, N the `('ft', 'shotnoise')` window relative to alpha, the shot noise of the removal, which the estimator's own shot-noise terms miss (a leg carries the removal of another leg's particle, which clusters with the third): `'BN'`, Delta B^N = sn BN P, the D^BP form with P^eta = -N; with `closure_shotnoise` (N_W2), the shot-noise column of the closure kernel, independent of `closure_kernel`), `compute_smooth3_closure_window(window, kernel=, modulation=P^eta_W2, shotnoise=N_W2, pin=, qmax=0.02)` (lsstypes multipoles at points q) -> dict `CBB`/`CBP`/`CBN`, ONE function for the three exact closure terms (pair integral over the two observed shells, one binned leg at its bin centre, the other -(k + q) through the two-shell overlap Lambda_{k1k2}, symmetrized over the reference leg; the q points get bands half-way to their neighbours, q < qmax kept; C^BP / C^BN, notes Eq. 153: binned legs carrying P, closure leg carrying P^eta_W2 or -N_W2, observed-level, not composed with the geometry window; a data / randoms mismatch pattern would enter the same way and is NOT modelled -- a defect of the catalogs, removed by the ell-cut / k-cut; works without `window` from `k`, `kedges` (not C^BB); C^BB: the exact removal on the closure leg below qmax with the window on the binned legs, any bins, linear in the theory multipoles; kernel = 'ft' window of an IC2Model whose field leg is W^2, i.e. -tau_W2 (tau with field leg W^2, norm int W^3; the W^2 subscript also for tau^eta_W2 -> P^eta_W2 and N_W2) (Theta^2 from random subsets disjoint from the templates', radial g^2 int Theta^2, norm int W^3), relative to alpha: templates + gic_templates([W geometry], templates) minus gic_templates alone, gic=False: theory independent, B read at the true long mode; called by `compute_smooth3_ic_window(closure_kernel=K)`; W^geo D^BB undercounts the closure removal of angular templates), `sketch_templates(templates, nprobe, blocks)` (random-sign probes of many rank-one templates sharing radial profile and response, e.g. the l_cut^2 of `ellcut_templates`, one block per degree L: Hutchinson, unbiased for everything linear in sum_a Theta_a x U_a -- P^ft, P^tf, their shot noise, the closure kernels tau_W2, N_W2, hence D^BB, C^BB, BN -- not for P^tt; IC2Model's cost and memory scale with the number of patterns), `fkp_modulation_templates` + `compute_smooth2_modulation_window` (weights following the data's own n(z), eta = -beta deltabar_cap: P^eta, the `modulation` input of `compute_smooth3_ic_window` for D^BP; for imaging weights P^eta = -P^ft + P^tt). Memory of `compute_smooth2_ic_window`: the regular templates go over blocks of 8 shells r (`_correlation_rows`), S^ft and S^tt projected a block of rows at a time (`project(..., rows=)`), never the full kernels K(r, s, column) (290 GB per rank for the DR2 QSO, 870 shells x lmax 755); only the template x template kernels are kept for all shells. Peak memory per rank in the 'theory columns' log lines, and the memory XLA plans before they run. Harmonic (SYSNet) stage: only two-operand einsums (a 4-operand one planned ~200 GB per rank for the DR2 ELG at z 1.1-1.6). `IC2Model(iter=0)` by default: the analysis of the field x template products is the pixel sum (iter=3: windows change ~2e-4, model 2x slower); a matrix product of degree maps gives the same and was slower (768 syntheses per mean window) |
| `types.py` | JAX pytree registration for `lsstypes` observable classes |
| `resamplers.py` | NGP/CIC/TSC/PCS painting kernels |
| `kernels.py` | Spectral kernels (gradient, Gaussian smoothing) |
| `fftlog.py` | FFTlog Bessel/correlation transforms |
| `pt.py` | Perturbation theory: the one-loop tracer spectra, and the tree-level n-point for n = 2..6 (`make_spectra_redshift_tracer`) that `cov3.py` needs for P5 and P6. All n-point callables take the `n - 1` independent legs |
| `mock.py` | `generate_gaussian_mesh`, `generate_uniform_particles` |
| `rotation.py` | Window matrix rotation (`WindowRotationSpectrum2`) |
| `utils.py` | Legendre/spherical harmonics, logging, `register_pytree_dataclass` |

### Distributed / multi-GPU

Wrap all computation in `with create_sharding_mesh() as sharding_mesh:`. The context manager sets up `jaxdecomp` for distributed 3-D FFTs. `create_sharded_array` and `create_sharded_random` create globally-shaped arrays sharded across devices using `shard_map`. `ParticleField(..., exchange=True)` redistributes particles so each device owns the particles that fall in its spatial shard.

### JIT usage

`compute_mesh2_spectrum` and similar top-level functions are JIT-friendly. The line-of-sight `los` argument must be declared static:

```python
compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
```

`Bin*` objects are pytrees with static meta fields, so they safely pass through `jax.jit` without retracing.

### Observable I/O

Results are `lsstypes` observable objects. Use `.write('output.h5')` and `read('output.h5')` (exported from `jaxpower` directly).
