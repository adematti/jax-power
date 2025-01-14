from pathlib import Path

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import compute_mesh_power, PowerSpectrumMultipoles, generate_gaussian_mesh, generate_uniform_particles, FKPField, compute_fkp_power


dirname = Path('_tests')


def test_mesh_power(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    @jax.jit
    def mock(seed):
        mesh = generate_gaussian_mesh(pkvec, seed=seed, meshsize=256, unitary_amplitude=True)
        return compute_mesh_power(mesh, ells=(0, 2, 4), edges={'step': 0.01})

    power = mock(random.key(43))
    fn = dirname / 'tmp.npy'
    power.save(fn)
    power = PowerSpectrumMultipoles.load(fn)
    if plot:
        from matplotlib import pyplot as plt
        ax = power.plot()
        ax.plot(power.k, power.k * pk(power.k))
        plt.show()
    # remove first few bins because of binning effects
    assert np.allclose(power.power[0, 2:], pk(power.k)[2:], rtol=1e-2)


def test_fkp_power(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))
    @jax.jit
    def mock(seed):
        attrs = dict(boxsize=1000., boxcenter=0., meshsize=128)
        mesh = generate_gaussian_mesh(pkvec, seed=seed, unitary_amplitude=True, **attrs)
        size = int(1e6)
        data = generate_uniform_particles(size, seed=32, **attrs)
        data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
        randoms = generate_uniform_particles(size, seed=42, **attrs)
        fkp = FKPField(data, randoms)
        return compute_fkp_power(fkp, ells=(0, 2, 4), edges={'step': 0.01})

    power = mock(random.key(43))
    fn = dirname / 'tmp.npy'
    power.save(fn)
    power = PowerSpectrumMultipoles.load(fn)
    if plot:
        from matplotlib import pyplot as plt
        ax = power.plot()
        ax.plot(power.k, power.k * pk(power.k))
        plt.show()


if __name__ == '__main__':

    test_mesh_power(plot=True)
    test_fkp_power(plot=True)