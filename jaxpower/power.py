import os
from functools import partial
from dataclasses import dataclass, asdict, field, fields
from typing import Any, Union

import numpy as np
import jax
from jax import numpy as jnp
from scipy import special

from . import utils
from .mesh import RealMeshField, ComplexMeshField, HermitianComplexMeshField, ParticleField, get_common_mesh_attrs
from .utils import legendre, mkdir


@partial(jax.tree_util.register_dataclass, data_fields=['k', 'power_nonorm', 'nmodes', 'norm', 'shotnoise_nonorm'], meta_fields=['edges', 'ells'])
@dataclass(frozen=True)
class PowerSpectrumMultipoles(object):

    """Class to store power spectrum multipoles."""

    k : np.ndarray
    power_nonorm: jax.Array
    nmodes: np.ndarray
    edges: np.ndarray
    ells: tuple
    norm: jax.Array = 1.
    shotnoise_nonorm: jax.Array = 0.

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = asdict(self)
        state.update(kwargs)
        return self.__class__(**state)

    @property
    def shotnoise(self):
        """Shot noise."""
        return self.shotnoise_nonorm / self.norm

    @property
    def power(self) -> jax.Array:
        """Power spectrum estimate."""
        power_nonorm = self.power_nonorm
        if 0 in self.ells:
            power_nonorm = power_nonorm.at[self.ells.index(0)].add(- self.shotnoise_nonorm)
        return power_nonorm / self.norm

    def save(self, fn):
        """Save power spectrum multipoles to file."""
        fn = str(fn)
        mkdir(os.path.dirname(fn))
        np.save(fn, asdict(self), allow_pickle=True)

    @classmethod
    def load(cls, fn):
        """Load power spectrum from file."""
        fn = str(fn)
        state = np.load(fn, allow_pickle=True)[()]
        new = cls.__new__(cls)
        new.__dict__.update(**state)
        return new

    def plot(self, ax=None, fn: str=None, kw_save: dict=None, show: bool=False):
        r"""
        Plot power spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        fn : str, default=None
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


def compute_mesh_power(*meshs: Union[RealMeshField, ComplexMeshField, HermitianComplexMeshField], edges: Union[np.ndarray, dict, None]=None,
                       ells: Union[int, tuple]=0, los: Union[str, np.ndarray]='x') -> PowerSpectrumMultipoles:
    r"""
    Compute power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, HermitianComplexMeshField
        Input meshs.

    edges : np.ndarray, dict, default=None
        ``kedges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, default='x'
        Line-of-sight direction.

    Returns
    -------
    power : PowerSpectrumMultipoles
    """
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
                knonzero = jnp.where(k == 0., 1., k)
                mu = sum(kk * ll for kk, ll in zip(kvec, vlos)).ravel() / knonzero
            leg = (2 * ell + 1) * legendre(ell)(mu)
        poles.append(jnp.bincount(ibin, weights=power * leg, length=edges.size + 1)[1:-1])
    ells = tuple(ells)
    power = jnp.array(poles)
    dtype = power.real.dtype
    k = (k * nmodes).astype(dtype)
    k = jnp.bincount(ibin, weights=k, length=edges.size + 1)[1:-1]
    nmodes = jnp.bincount(ibin, weights=nmodes, length=edges.size + 1)[1:-1]
    k /= nmodes
    power /= nmodes
    norm = meshsize.prod() / jnp.prod(boxsize / meshsize)
    return PowerSpectrumMultipoles(k, power_nonorm=power, nmodes=nmodes, edges=edges, ells=ells, norm=norm)


@partial(jax.tree_util.register_dataclass, data_fields=['data', 'randoms'], meta_fields=[])
@dataclass(frozen=True, init=False)
class FKPField(object):
    """
    Class defining the FKP field, data - randoms.

    References
    ----------
    https://arxiv.org/abs/astro-ph/9304022
    """
    data: ParticleField
    randoms: ParticleField

    def __init__(self, data, randoms, **kwargs):
        data, randoms = ParticleField.same_mesh(data, randoms, **kwargs)
        self.__dict__.update(data=data, randoms=randoms)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = {name: getattr(self, name) for name in ['data', 'randoms']} | kwargs
        return self.__class__(**state)

    def paint(self, resampler: Union[str, callable]='cic', interlacing: int=1,
              compensate: bool=False, dtype=None, out: str='real'):
        fkp = self.data - self.data.sum() / self.randoms.sum() * self.randoms
        return fkp.paint(resampler=resampler, interlacing=interlacing, compensate=compensate, dtype=dtype, out=out)

    @staticmethod
    def same_mesh(*others, **kwargs):
        attrs = get_common_mesh_attrs(*([other.data for other in others] + [other.randoms for other in others]), **kwargs)
        return tuple(other.clone(**attrs) for other in others)


def compute_normalization(*inputs: Union[RealMeshField, ParticleField], resampler='cic', **kwargs) -> jax.Array:
    """
    Return normalization, in 1 / volume unit.

    Warning
    -------
    Input particles are considered uncorrelated.
    """
    meshs, particles = [], []
    attrs = {}
    for inp in inputs:
        if isinstance(inp, RealMeshField):
            meshs.append(inp)
            attrs = {name: getattr(inp, name) for name in ['boxsize', 'boxcenter', 'meshsize']}
        else:
            particles.append(inp)
    if particles: particles = ParticleField.same_mesh(*particles, **attrs)
    normalization = 1
    for mesh in meshs:
        normalization *= mesh
    for particle in particles:
        normalization *= particle.paint(resampler=resampler, interlacing=1, compensate=False)
    return normalization.sum() / normalization.cellsize.prod()


def compute_fkp_power(*fkps: FKPField, edges: Union[np.ndarray, dict, None]=None,
                      resampler='tsc', interlacing=3, ells: Union[int, tuple]=0, los: Union[str, np.ndarray]='x') -> PowerSpectrumMultipoles:
    r"""
    Compute power spectrum from FKP field.

    Parameters
    ----------
    meshs : FKPField
        Input FKP fields.

    edges : np.ndarray, dict, default=None
        ``kedges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    resampler : str, callable
        Resampler to read particule weights from mesh.
        One of ['ngp', 'cic', 'tsc', 'pcs'].

    interlacing : int, default=1
        If 1, no interlacing correction.
        If > 1, order of interlacing correction.
        Typically, 3 gives reliable power spectrum estimation up to :math:`k \sim k_\mathrm{nyq}`.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, default='x'
        Line-of-sight direction.

    Returns
    -------
    power : PowerSpectrumMultipoles
    """
    fkps = FKPField.same_mesh(*fkps)
    meshs = [fkp.paint(resampler=resampler, interlacing=interlacing, compensate=True, out='complex') for fkp in fkps]
    # TODO: generalize to N fkp fields
    if len(fkps) > 1:
        shotnoise = 0.
        randoms = [fkp.randoms for fkp in fkps]
        alpha2 = jnp.array([fkp.data.sum() / fkp.randoms.sum() for fkp in fkps]).prod()
    else:
        fkp = fkps[0]
        alpha = fkp.data.sum() / fkp.randoms.sum()
        shotnoise = jnp.sum(fkp.data.weights**2) + alpha**2 * jnp.sum(fkp.randoms.weights**2)
        randoms = [fkp.randoms[:fkp.randoms.size // 2], fkp.randoms[fkp.randoms.size // 2:]]
        alpha2 = jnp.array([fkp.data.sum() / randoms.sum() for randoms in randoms]).prod()
    norm = alpha2 * compute_normalization(*randoms, cellsize=10.)
    return compute_mesh_power(*meshs, edges=edges, ells=ells, los=los).clone(norm=norm, shotnoise_nonorm=shotnoise)