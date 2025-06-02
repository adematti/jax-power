import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (compute_mesh3_spectrum, BinMesh3Spectrum, MeshAttrs, Spectrum3Poles, generate_gaussian_mesh, utils)


dirname = Path('_tests')


def test_mesh3_spectrum(plot=False):

    def pk(k):
        kp = 0.03
        return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)

    pkvec = lambda kvec: pk(jnp.sqrt(sum(kk**2 for kk in kvec)))

    def get_fn(los='x'):
        return dirname / 'tmp_{}.npy'.format(los)

    list_los = ['x', 'local']
    attrs = MeshAttrs(meshsize=128, boxsize=1000.)

    for basis in ['sugiyama-diagonal', 'scoccimarro-equilateral']:

        ells = [0] if 'scoccimarro' in basis else [(0, 0, 0)]

        bin = BinMesh3Spectrum(attrs, edges={'step': 0.01}, basis=basis, ells=ells)

        @partial(jax.jit, static_argnames=['los'])
        def mock(attrs, bin, seed, los='x'):
            mesh = generate_gaussian_mesh(attrs, pkvec, seed=seed, unitary_amplitude=True)
            return compute_mesh3_spectrum(mesh, los=los, bin=bin)

        for los in list_los:

            nmock = 2
            t0 = time.time()
            power = mock(attrs, bin, random.key(43), los=los)
            jax.block_until_ready(power)
            print(f'time for jit {time.time() - t0:.2f}')
            t0 = time.time()
            for i in range(nmock):
                power = mock(attrs, bin, random.key(i + 42), los=los)
                jax.block_until_ready(power)
            print(f'time per iteration {(time.time() - t0) / nmock:.2f}')
            power.save(get_fn(los))

        if plot:
            from matplotlib import pyplot as plt

            for los in list_los:
                power = Spectrum3Poles.load(get_fn(los=los))
                ax = power.plot().axes[0]
                ax.set_title(los)
                plt.show()


def test_timing():

    import time
    from jax import jit, random
    from jaxpower.mesh import create_sharded_random
    from jaxpower.bspec import get_real_Ylm, spherical_jn

    @jit
    def f1(mesh):
        xvec = mesh.attrs.rcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
        return jn(xnorm) * Ylm(*xvec) * mesh

    @jit
    def f2(mesh):
        kvec = mesh.attrs.rcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in kvec))
        mask = (xnorm >= 0.01) & (xnorm <= 0.03)
        return (Ylm(*kvec) * mesh * mask).c2r()

    attrs = MeshAttrs(boxsize=1000., meshsize=256)
    rmesh = attrs.create(kind='real', fill=create_sharded_random(random.normal, random.key(42), shape=attrs.meshsize))
    cmesh = rmesh.r2c()

    Ylm = get_real_Ylm(2, 0)
    jn = spherical_jn(2)

    f1(rmesh)
    f2(cmesh)
    jax.block_until_ready(f1(rmesh))
    jax.block_until_ready(f2(cmesh))

    def timing(f, mesh):
        t0 = time.time()
        for i in range(10):
            jax.block_until_ready(f((0.1 + i) * mesh))
        print(time.time() - t0)

    timing(f1, rmesh)
    timing(f2, cmesh)


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('error')
    test_mesh3_spectrum(plot=False)
    #test_timing()