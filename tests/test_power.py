import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (compute_mesh_power, PowerSpectrumMultipoles, generate_gaussian_mesh, generate_uniform_particles, FKPField,
                      compute_fkp_power, BinnedStatistic, WindowMatrix, MeshAttrs, compute_mean_mesh_power)


dirname = Path('_tests')


def test_binned_statistic():

    x = [np.linspace(0.01, 0.2, 10), np.linspace(0.01, 0.2, 10)]
    v = x
    observable = BinnedStatistic(x=x, value=v, projs=[0, 2])
    assert np.allclose(observable.view(projs=2), v[0])
    assert np.allclose(observable.view(xlim=(0., 0.1), projs=0), v[0][v[0] <= 0.1])
    assert np.allclose(observable.select(rebin=2).view(projs=0), (v[0][::2] + v[0][1::2]) / 2.)
    fn = dirname / 'tmp.npy'
    observable.save(fn)
    observable2 = BinnedStatistic.load(fn)
    assert np.allclose(observable2.view(), observable.view())

    xt = [np.linspace(0.01, 0.2, 20), np.linspace(0.01, 0.2, 20)]
    vt = xt
    theory = BinnedStatistic(x=xt, value=vt, projs=[0, 2])
    wmat = WindowMatrix(observable=observable, theory=theory, value=np.ones((observable.size, theory.size)))
    assert np.allclose(wmat.select(xlim=(0., 0.15), axis='o').observable.x(projs=0), v[0][v[0] <= 0.15])
    tmp = wmat.dot(theory.view(), zpt=False)
    tmp2 = wmat.slice(slice(0, None, 2), axis='t').dot(theory.slice(slice(0, None, 2)).view(), zpt=False)
    assert np.allclose(tmp.sum(), tmp2.sum())
    wmat.slice(slice(0, None, 2), axis='o')
    wmat.slice(slice(0, None, 2), axis='o')
    assert np.allclose(wmat.select(axis='t', projs=0, select_projs=True).theory.view(), vt[0])
    wmat.plot()

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def mock(seed, los='x'):
        mesh = generate_gaussian_mesh(pkvec, seed=seed, meshsize=64, unitary_amplitude=True)
        return compute_mesh_power(mesh, ells=(0, 2, 4), los=los, edges={'step': 0.01})

    power = mock(random.key(43), los='x')
    power.save(fn)
    power2 = BinnedStatistic.load(fn)
    assert np.allclose(power2.view(), power.view())
    assert type(power2) == PowerSpectrumMultipoles
    power2.plot(show=True)
    assert power.clone(norm=0.1).norm == 0.1
    power3 = power.clone(value=power.view())
    assert np.allclose(power3.view(), power.view())
    assert type(power3) == BinnedStatistic


def test_mesh_power(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'endpoint']

    @partial(jax.jit, static_argnames=['los', 'ells'])
    def mock(seed, los='x', ells=(0, 2, 4)):
        mesh = generate_gaussian_mesh(pkvec, seed=seed, meshsize=128, unitary_amplitude=True)
        return compute_mesh_power(mesh, ells=ells, los=los, edges={'step': 0.01})

    for los in list_los[1:]:

        nmock = 5
        t0 = time.time()
        power = mock(random.key(43), los=los)
        print(f'time for jit {time.time() - t0:.2f}')
        t0 = time.time()
        for i in range(nmock):
            mock(random.key(i + 42), los=los)
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
        power.save(get_fn(los))
        power = mock(random.key(i + 42), los=los, ells=(4,))
        assert tuple(power.projs) == (4,)

    power = PowerSpectrumMultipoles.load(get_fn(los='x'))
    k = power.x(projs=0)
    if plot:
        from matplotlib import pyplot as plt
        ax = power.plot().axes[0]
        ax.plot(k, k * pk(k))
        plt.show()
    # remove first few bins because of binning effects
    assert np.allclose(power.view(projs=0)[2:], pk(k)[2:], rtol=1e-2)


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
            ax = power.plot().axes[0]
            k = power.x(projs=0)
            ax.plot(k, k * pk(k))
            ax.set_title(los)
            plt.show()


def test_mean_power(plot=False):

    def pk(k):
        kp = 0.01
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    f, b = 0.8, 1.5
    beta = f / b
    poles = {0: lambda k: (1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(k),
             2: lambda k: 0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(k),
             4: lambda k: 8. / 35 * beta ** 2 * pk(k)}

    list_los = [('x', None), ('endpoint', None), ('endpoint', 'local')]
    attrs = MeshAttrs(boxsize=1000., meshsize=64, boxcenter=1000.)

    #@partial(jax.jit, static_argnames=['los', 'thlos'])
    def mean(los='x', thlos=None):
        theory = poles
        if thlos is not None:
            theory = (poles, thlos)
        return compute_mean_mesh_power(attrs, theory=theory, ells=(0, 2, 4), los=los, edges={'step': 0.01}, pbar=True)

    for los, thlos in list_los:

        #power_mock = mock(random.key(43), los=los)
        t0 = time.time()
        power_mean = mean(los=los, thlos=thlos)
        print(f'time for jit {time.time() - t0:.2f}')
        nmock = 2
        t0 = time.time()
        for i in range(nmock):
            jax.block_until_ready(mean(los=los, thlos=thlos))
        print(f'time per iteration {(time.time() - t0) / nmock:.2f}')

        if plot:
            from matplotlib import pyplot as plt
            ax = power_mean.plot().axes[0]
            for ell in power_mean.projs:
                k = power_mean.x(projs=ell)
                ax.plot(k, k * poles[ell](k), color='k', linestyle='--')
            ax.set_title(los)
            plt.show()


if __name__ == '__main__':

    test_binned_statistic()
    test_mesh_power(plot=False)
    test_fkp_power(plot=False)
    test_mean_power(plot=True)