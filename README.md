# JAX-powered power spectrum estimator

**jax-power** is a package for auto and cross power spectrum and associated window function estimation,
for periodic boxes, survey geometry, in the flat-sky or plane-parallel configurations.

A typical auto power spectrum estimation is as simple as (for multi-GPU):
```
import jax
# Initialize JAX distributed environment
jax.distributed.initialize()

from jax import numpy as jnp
from jaxpower import get_mesh_attrs, compute_mesh2_spectrum, ParticleField, FKPField, create_sharding_mesh

with create_sharding_mesh() as sharding_mesh:  # specify how to spatially distribute particles / mesh

    # Create MeshAttrs
    attrs = get_mesh_attrs(data_positions, randoms_positions, boxpad=2., meshsize=128)
    # Input ``data_positions``, ``data_weights``, ``randoms_positions``, ``randoms_weights`` are assumed scattered over the different processes.
    data = ParticleField(data_positions, data_weights, attrs=attrs, exchange=True)
    randoms = ParticleField(randoms_positions, randoms_weights, attrs=attrs, exchange=True)
    fkp = FKPField(data, randoms)
    norm, num_shotnoise = compute_fkp2_spectrum_normalization(fkp), compute_fkp2_spectrum_shotnoise(fkp)
    # particles are already exchanged in ``get_particle_field``
    mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real', pexchange=False)
    del fkp
    # Tip: can be done once for many P(k) evaluation
    bin = BinMesh2Spectrum(mesh.attrs, edges={'step': 0.001}, ells=(0, 2, 4))
    # Then compute power spectrum
    # One can jit the function
    compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
    pk = compute_mesh2_spectrum(mesh, bin=bin, los='firstpoint')
    # Add the normalization and shot noise information
    pk = power.clone(norm=norm, num_shotnoise=num_shotnoise)
    pk.save('power.npy')

# Close JAX distributed environment
jax.distributed.shutdown()
```

Example notebooks presenting most use cases are provided in directory nb/.


## Citations

Multi-GPU 3D FFT and halo exhcange is handled with (publication incoming!):
[jaxdecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp)
by Wassim Kabalan and François Lanusse.