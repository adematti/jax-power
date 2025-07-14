"""
Implementation of various resamplers, with attributes ``read``, ``paint`` and ``compensate``.

Reference
---------
https://arxiv.org/pdf/1512.07295
"""

import itertools
from collections.abc import Callable

import numpy as np
import jax
from jax import numpy as jnp


def _compensate_tophat_convolution_kernel(order: int):

    def fn(value, kvec):
        kernel = 1.
        for kk in kvec:
            kernel *= jnp.sinc(kk / (2 * jnp.pi))**(-order)
        return value * kernel
    fn.kind = 'circular'

    return fn


# Stolen from Hugo's montecosmo: https://github.com/hsimonfroy/montecosmo/blob/f4d318329a332d1a984e6a1fde6f5d59c4dd4336/montecosmo/nbody.py#L175


from itertools import product


paint_kernels = [
    lambda s: jnp.full(jnp.shape(s)[-1:], jnp.inf), # Dirac
    lambda s: jnp.full(jnp.shape(s)[-1:], 1.), # NGP
    lambda s: 1 - s, # CIC
    lambda s: (s <= 1/2) * (3/4 - s**2) + (1/2 < s) / 2 * (3/2 - s)**2, # TSC
    lambda s: (s <= 1) / 6 * (4 - 6 * s**2 + 3 * s**3) + (1 < s) / 6 * (2 - s)**3, # PCS
]


def paint(mesh: tuple | jnp.ndarray, positions, weights=1., order: int=2):
    """
    Paint the positions onto the mesh.
    If mesh is a tuple, paint on a zero mesh with such shape.
    """
    if isinstance(mesh, tuple):
        mesh = jnp.zeros(mesh)
    else:
        mesh = jnp.asarray(mesh)

    dtype = 'int16' # int16 -> +/- 32_767, trkl
    shape = np.asarray(mesh.shape, dtype=dtype)
    def wrap(idx):
        return idx % shape

    id0 = (jnp.round if order % 2 else jnp.floor)(positions).astype(dtype)
    ishifts = np.arange(order) - (order - 1) // 2
    ishifts = np.array(list(product(* len(shape) * (ishifts,))), dtype=dtype)

    def step(carry, ishift):
        idx = id0 + ishift
        s = jnp.abs(idx - positions)
        idx, ker = wrap(idx), paint_kernels[order](s).prod(-1)

        idx = jnp.unstack(idx, axis=-1)
        # idx = tuple(jnp.moveaxis(idx, -1, 0)) # TODO: JAX >= 0.4.28 for unstack
        carry = carry.at[idx].add(weights * ker)
        return carry, None

    mesh = jax.lax.scan(step, mesh, ishifts)[0]
    return mesh


def read(mesh: jnp.ndarray, positions, order: int=2):
    """Read the value at the positions from the mesh."""
    dtype = 'int16' # int16 -> +/- 32_767, trkl
    shape = np.asarray(mesh.shape, dtype=dtype)

    def wrap(idx):
        return idx % shape

    id0 = (jnp.round if order % 2 else jnp.floor)(positions).astype(dtype)
    ishifts = np.arange(order) - (order - 1) // 2
    ishifts = np.array(list(product(* len(shape) * (ishifts,))), dtype=dtype)

    def step(carry, ishift):
        idx = id0 + ishift
        s = jnp.abs(idx - positions)
        idx, ker = wrap(idx), paint_kernels[order](s).prod(-1)

        idx = jnp.unstack(idx, axis=-1)
        # idx = tuple(jnp.moveaxis(idx, -1, 0)) # TODO: JAX >= 0.4.28 for unstack
        carry += mesh[idx] * ker
        return carry, None

    out = jnp.zeros(id0.shape[:-1])
    out = jax.lax.scan(step, out, ishifts)[0]
    return out


from functools import partial

for order, name in enumerate(['dirac', 'ngp', 'cic', 'tsc', 'pcs']):
    globals()[name] = type(name, (), dict(paint=partial(paint, order=order), read=partial(read, order=order), compensate=_compensate_tophat_convolution_kernel(order), order=order))
