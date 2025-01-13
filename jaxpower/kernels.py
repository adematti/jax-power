from typing import Any, Union

import numpy as np
from jax import numpy as jnp



def gradient(axis: Union[int, tuple, list] = 0, order: Union[str, None] = None):
    """
    Return the gradient kernel in the requested direction.

    Parameters
    ----------
    axis : int, array
        Axis where to take the gradient.
        Take care not to repeat indices (except if desired).
    order : str
        'finite_difference' for finite difference kernel.

    Returns
    -------
    fn : callable
        Kernel callable to be applied to mesh: ``mesh.apply(fn)``.
    """
    axis = np.atleast_1d(axis).flat

    def fn(value, kvec):
        kernel = 1
        for ax in axis:
            if order == 'finite_difference':
                w = kvec[ax]
                a = 1 / 6.0 * (8 * jnp.sin(w) - jnp.sin(2 * w))
                kernel *= a * 1j
            else:
                kernel *= 1j * kvec[ax]
                # FIXME: why this in jaxpm?
                #wts = jnp.squeeze(wts)
                #wts[len(wts) // 2] = 0
                #wts = wts.reshape(kvec[direction].shape)
        return value * kernel

    fn.kind = 'circular'
    return fn


def invlaplace(order: Union[str, None] = None):
    """
    Return the inverse Laplace kernel.

    cf. [Feng+2016](https://arxiv.org/pdf/1603.00476)

    Parameters
    ----------
    order : str
        'finite_difference' for finite difference kernel.

    Returns
    -------
    fn : callable
        Kernel callable to be applied to mesh: ``mesh.apply(fn)``.
    """
    def fn(value, kvec):
        if order == 'finite_difference':
            kk = sum((ki * jnp.sinc(ki / (2 * jnp.pi)))**2 for ki in kvec)
        else:
            kk = sum(ki**2 for ki in kvec)
        return jnp.where(kk == 0, 0, - value / kk)
    fn.kind = 'circular'
    return fn


def longrange(r_split: float = 0.):
    """
    Return long range kernel.

    Parameters
    ----------
    r_split : float
        Splitting radius.

    Returns
    --------
    Returns
    -------
    fn : callable
        Kernel callable to be applied to mesh: ``mesh.apply(fn)``.

    TODO: @modichirag add documentation
    """
    def fn(value, kvec):
        if r_split != 0:
            kk = sum(ki**2 for ki in kvec)
            kernel = jnp.exp(-kk * r_split**2)
        else:
            kernel = 1.
        return value * kernel
    fn.kind = 'circular'
    return fn