import os
from functools import partial
from dataclasses import dataclass, asdict
from typing import Any, Union

import numpy as np
import jax
from jax import numpy as jnp
from scipy import special

from . import utils
from .mesh import RealMeshField, ComplexMeshField, HermitianComplexMeshField
from .utils import mkdir


@partial(jax.tree_util.register_dataclass, data_fields=['k', 'power_nonorm', 'nmodes', 'norm', 'shotnoise_nonorm'], meta_fields=['edges', 'ells'])
@dataclass(init=False)
class MeshFFTPower(object):

    """TODO: implement Legendre polynomials in JAX, mu-wedges and varying line-of-sight."""
    k : np.ndarray
    power_nonorm: jax.Array
    nmodes: np.ndarray
    norm: jax.Array
    shotnoise_nonorm: jax.Array
    edges: np.ndarray
    ells: tuple

    def __init__(self, *meshs: Union[RealMeshField, ComplexMeshField, HermitianComplexMeshField], edges: Union[np.ndarray, dict, None]=None,
                 ells: Union[int, tuple]=0, los: Union[str, np.ndarray]='x', norm: Union[jax.Array, float]=1., shotnoise_nonorm: Union[jax.Array, float]=0.):
        meshs = list(meshs)
        assert 1 <= len(meshs) <= 2
        for imesh, mesh in enumerate(meshs):
            if not isinstance(mesh, (ComplexMeshField, HermitianComplexMeshField)):
                meshs[imesh] = mesh.r2c()
        kfun, knyq = np.min(meshs[0].kfun), np.min(meshs[0].knyq)
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            kmin = edges.get('min', 0.)
            kmax = edges.get('max', knyq)
            kstep = edges.get('step', kfun)
            edges = np.arange(kmin, kmax, kstep)
        else:
            edges = np.asarray(edges)

        if len(meshs) == 1:
            power = meshs[0] * meshs[0].conj()
        else:
            power = meshs[0] * meshs[1].conj()

        boxsize, meshsize = power.boxsize, power.meshsize
        hermitian = isinstance(power, HermitianComplexMeshField)
        nmodes = jnp.full_like(power.value, 1 + hermitian, dtype='i4')
        if hermitian:
            nmodes = nmodes.at[..., 0].set(1)
            if power.shape[-1] % 2 == 0:
                nmodes = nmodes.at[..., -1].set(1)

        kvec = power.coords(kind='wavenumber', sparse=True)
        k = sum(kk**2 for kk in kvec)**0.5
        k = k.ravel()
        power = power.ravel()
        nmodes = nmodes.ravel()
        ibin = jnp.digitize(k, edges, right=False)
        power = (power * nmodes).astype(power.dtype)

        vlos = los
        ndim = len(boxsize)
        if isinstance(los, str):
            vlos = [0.] * ndim
            vlos['xyz'.index(los)] = 1.
        vlos = np.array(vlos)
        isscalar = np.ndim(ells) == 0
        if isscalar: ells = (ells,)
        mu = None
        poles = []
        for ell in ells:
            if ell == 0:
                leg = 1.
            else:
                if mu is None:
                    mu = np.where(k == 0., 0., sum(kk * ll for kk, ll in zip(kvec, vlos)) / k)
                leg = special.legendre(ell)(mu)
            poles.append(jnp.bincount(ibin, weights=power * leg, length=edges.size + 1)[1:-1])
        ells = tuple(ells)
        power = jnp.array(poles)
        dtype = power.real.dtype
        k = (k * nmodes).astype(dtype)
        k = jnp.bincount(ibin, weights=k, length=edges.size + 1)[1:-1]
        nmodes = jnp.bincount(ibin, weights=nmodes, length=edges.size + 1)[1:-1]
        k /= nmodes
        power *= jnp.prod(boxsize / meshsize**2) / nmodes
        self.k, self.power_nonorm, self.nmodes, self.edges = k, power, nmodes, edges
        self.ells = ells
        self.norm = jnp.asarray(norm, dtype=dtype)
        self.shotnoise_nonorm = jnp.asarray(shotnoise_nonorm, dtype=dtype)

    @property
    def shotnoise(self):
        return self.shotnoise_nonorm / self.norm

    @property
    def power(self):
        return (self.power_nonorm - self.shotnoise_nonorm) / self.norm

    def save(self, fn):
        fn = str(fn)
        mkdir(os.path.dirname(fn))
        np.save(fn, asdict(self), allow_pickle=True)

    @classmethod
    def load(cls, fn):
        fn = str(fn)
        state = np.load(fn, allow_pickle=True)[()]
        new = cls.__new__(cls)
        new.__dict__.update(**state)
        return new

    def plot(self, ax=None, fn=None, kw_save=None, show=False):
        r"""
        Plot power spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        fn : string, default=None
            If not ``None``, file name where to save figure.

        kw_save : dict, default=None
            Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            Whether to show figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from matplotlib import pyplot as plt
        fig = None
        if ax is None: fig, ax = plt.subplots()
        for ill, ell in enumerate(self.ells):
            ax.plot(self.k, self.k * self.power[ill].real, label=r'$\ell = {:d}$'.format(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        if fn is not None:
            utils.savefig(fn, fig=fig, **(kw_save or {}))
        if show:
            plt.show()
        return ax


class ParticleFFTPower(MeshFFTPower):

    """TODO."""