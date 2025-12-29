# 🚀 jax-power: JAX-Powered Power Spectrum Estimation

**`jax-power`** is a package for estimating **auto** and **cross power spectra**.
It supports periodic boxes and survey geometries, global and local line-of-sight.

Distributed, multi-GPU computation with JAX.

---

## 📦 Installation

You can install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/adematti/jax-power.git
```

Alternatively, if you plan to contribute or modify the code, install in editable (development) mode:

```bash
git clone https://github.com/adematti/jax-power.git
cd jax-power
pip install -e .
```

### Requirements

- Python ≥ 3.9
- `jax`, `jaxlib` (with GPU or TPU support, if applicable)
- `numpy`
- [`jaxdecomp`](https://github.com/DifferentiableUniverseInitiative/jaxDecomp) — for distributed FFT and halo exchange

We recommend following the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) to ensure correct setup for your hardware (CPU/GPU/TPU).

---

## 🧪 Quick Example: Auto Power Spectrum with Multi-GPU

```python
import jax
# Initialize JAX distributed environment
jax.distributed.initialize()

from jax import numpy as jnp
from jaxpower import (
    get_mesh_attrs,
    compute_mesh2_spectrum,
    ParticleField,
    FKPField,
    create_sharding_mesh,
    BinMesh2SpectrumPoles,
    compute_fkp2_normalization,
    compute_fkp2_shotnoise
)

with create_sharding_mesh() as sharding_mesh:  # distribute mesh and particles

    # Create MeshAttrs from positions (assumed already sharded across processes)
    mattrs = get_mesh_attrs(data_positions, randoms_positions, boxpad=2., meshsize=128)

    # Define FKP field = data - randoms
    data = ParticleField(data_positions, data_weights, attrs=mattrs, exchange=True)
    randoms = ParticleField(randoms_positions, randoms_weights, attrs=mattrs, exchange=True)
    fkp = FKPField(data, randoms)
    # Warning!
    # If you access data.positions, data.weights (same for randoms) in a distributed context,
    # You'll see they are reordered w.r.t. data_positions, data_weights
    # This is such that the local portion of the 3D mesh receives the relevant particles

    # Define k-bin edges and multipoles
    bin = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.001}, ells=(0, 2, 4))

    # Compute normalization and shot noise terms
    norm = compute_fkp2_normalization(fkp, bin=bin)
    num_shotnoise = compute_fkp2_shotnoise(fkp, bin=bin)

    # Paint FKP field to mesh
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
    del fkp  # cleanup

    # JIT the power spectrum function
    compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])

    # Compute P(k)
    spectrum = compute_mesh2_spectrum(mesh, bin=bin, los='firstpoint')
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

    # Save result
    spectrum.write('spectrum.h5')

# Shut down distributed environment
jax.distributed.shutdown()
```

📝 Example notebooks are available in the `nb/` directory.

---

## 📚 Citation

Multi-GPU 3D FFT and halo exchange support is provided by:

> **`jaxdecomp`** — [https://github.com/DifferentiableUniverseInitiative/jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp)
> Developed by *Wassim Kabalan* and *François Lanusse*.
> 📄 *Publication incoming!*

---

## 🙏 Acknowledgments

Thanks to **Hugo Simon-Onfroy** for providing JIT-friendly resamplers via [`montecosmo`](https://github.com/hsimonfroy).

---