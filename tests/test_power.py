import time
from pathlib import Path
from functools import partial

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

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'endpoint']

    for los in list_los:
        @partial(jax.jit, static_argnames=['los'])
        def mock(seed, los='x'):
            mesh = generate_gaussian_mesh(pkvec, seed=seed, meshsize=128, unitary_amplitude=True)
            return compute_mesh_power(mesh, ells=(0, 2, 4), los=los, edges={'step': 0.01})

        power = mock(random.key(43))

        nmock = 5
        t0 = time.time()
        power = mock(random.key(43), los=los)
        print(f'time for jit {time.time() - t0:.2f}')
        t0 = time.time()
        for i in range(nmock):
            mock(random.key(i + 42), los=los)
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
        power.save(get_fn(los))

    power = PowerSpectrumMultipoles.load(get_fn(los='x'))
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

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'endpoint']

    for los in list_los:
        @partial(jax.jit, static_argnames=['los'])
        def mock(seed, los='x'):
            boxcenter = [1300., 0., 0.]
            attrs = dict(boxsize=1000., boxcenter=boxcenter, meshsize=128)
            mesh = generate_gaussian_mesh(pkvec, seed=seed, unitary_amplitude=True, **attrs)
            size = int(1e6)
            data = generate_uniform_particles(size, seed=32, **attrs)
            data = data.clone(weights=1. + mesh.read(data.positions, resampler='cic', compensate=True))
            randoms = generate_uniform_particles(size, seed=42, **attrs)
            fkp = FKPField(data, randoms)
            return compute_fkp_power(fkp, ells=(0, 2, 4), los=los, edges={'step': 0.01})

        nmock = 5
        t0 = time.time()
        power = mock(random.key(43), los=los)
        print(f'time for jit {time.time() - t0:.2f}')
        t0 = time.time()
        for i in range(nmock):
            mock(random.key(i + 42), los=los)
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
        power.save(get_fn(los))

    if plot:
        from matplotlib import pyplot as plt

        for los in list_los:
            ax = power.plot()
            ax.plot(power.k, power.k * pk(power.k))
            ax.set_title(los)
            plt.show()


if __name__ == '__main__':

    test_mesh_power(plot=False)
    test_fkp_power(plot=True)