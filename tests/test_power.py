from pathlib import Path

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import compute_mesh_power, PowerSpectrumMultipoles, generate_gaussian_random_field


dirname = Path('_tests')


def test_power(plot=False):

    def pk(k):
        kp = 0.15
        return k**3 * jnp.exp(-k / kp**2)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    @jax.jit
    def mock(seed):
        mesh = generate_gaussian_random_field(pkvec, seed=seed, meshsize=256, unitary_amplitude=True)
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


if __name__ == '__main__':

    test_power(plot=True)