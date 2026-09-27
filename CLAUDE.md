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
