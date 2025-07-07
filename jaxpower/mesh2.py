import os
from functools import partial
from dataclasses import dataclass, field, asdict
from collections.abc import Callable
import itertools
from pathlib import Path

import numpy as np
from scipy import special
import jax
from jax import numpy as jnp

from .utils import get_legendre, get_spherical_jn, get_real_Ylm, real_gaunt, plotter, BinnedStatistic, WindowMatrix, register_pytree_dataclass
from .mesh import BaseMeshField, RealMeshField, ComplexMeshField, ParticleField, staticarray, MeshAttrs, _get_hermitian_weights, _find_unique_edges, _get_bin_attrs, _bincount, get_mesh_attrs


@jax.tree_util.register_pytree_node_class
class Spectrum2Poles(BinnedStatistic):

    _label_x = r'$k$ [$h/\mathrm{Mpc}$]'
    _label_proj = r'$\ell$'
    _label_value = r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]'
    _data_fields = BinnedStatistic._data_fields + ['_num_shotnoise', '_num_zero', '_volume']
    _select_x_fields = BinnedStatistic._select_x_fields + ['_num_shotnoise', '_num_zero', '_volume']
    _select_proj_fields = BinnedStatistic._select_proj_fields + ['_num_shotnoise', '_num_zero', '_volume']
    _sum_fields = BinnedStatistic._sum_fields + ['_num_shotnoise', '_num_zero']
    _init_fields = {'k': '_x', 'num': '_value', 'nmodes': '_weights', 'edges': '_edges', 'ells': '_projs', 'norm': '_norm',
                    'num_shotnoise': '_num_shotnoise', 'num_zero': '_num_zero', 'name': 'name', 'attrs': 'attrs'}

    def __init__(self, k: np.ndarray, num: jax.Array, ells: tuple, nmodes: np.ndarray=None, edges: np.ndarray=None, volume: np.ndarray=None, norm: jax.Array=1.,
                 num_shotnoise: jax.Array=0., num_zero: jax.Array=None, name: str=None, attrs: dict=None):

        def _tuple(item):
            if item is None:
                return None
            if not isinstance(item, (tuple, list)):
                item = (item,) * len(ells)
            return tuple(item)
        super().__init__(x=_tuple(k), edges=_tuple(edges), projs=ells, value=num,
                         weights=_tuple(nmodes), norm=norm, name=name, attrs=attrs)
        if volume is None: volume = tuple(nmodes.copy() for nmodes in self.nmodes())
        else: volume = _tuple(volume)
        if num_zero is None: num_zero = _tuple(0.)
        num_zero = list(num_zero)
        for ill, value in enumerate(num_zero):
            if jnp.size(value) <= 1: num_zero[ill] = jnp.where((self._edges[ill][..., 0] <= 0.) & (self._edges[ill][..., 1] >= 0.), value, 0.)
        num_zero = tuple(num_zero)
        if not isinstance(num_shotnoise, (tuple, list)):
             num_shotnoise = (num_shotnoise,) + (0,) * (len(ells) - 1)
        num_shotnoise = list(num_shotnoise)
        for ill, value in enumerate(num_shotnoise):
            if jnp.size(value) <= 1: num_shotnoise[ill] = jnp.zeros_like(self._value[ill]).at[...].set(value)
        num_shotnoise = tuple(num_shotnoise)
        self.__dict__.update(_volume=volume, _num_shotnoise=num_shotnoise, _num_zero=num_zero)

    @property
    def num(self):
        """Power spectrum with shot noise *not* subtracted."""
        return self._value

    k = BinnedStatistic.x
    kavg = BinnedStatistic.xavg
    nmodes = BinnedStatistic.weights

    def shotnoise(self, projs=Ellipsis):
        """Shot noise."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        num_shotnoise = [s / self._norm for s in self._num_shotnoise]
        if isscalar: return num_shotnoise[iprojs]
        return [num_shotnoise[iproj] for iproj in iprojs]

    def volume(self, projs=Ellipsis):
        """Volume (optionally restricted to input projs)."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return self._volume[iprojs]
        return [self._volume[iproj] for iproj in iprojs]

    @property
    def ells(self):
        return self._projs

    @property
    def value(self):
        """Power spectrum estimate."""
        toret = list(self.num)
        for ill, ell in enumerate(self.ells):
            toret[ill] = (toret[ill] - self._num_zero[ill] - self._num_shotnoise[ill]).real / self._norm
        return tuple(toret)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot power spectrum.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self._x[ill], self._x[ill] * self.value[ill].real, label=self._get_label_proj(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(self._label_x)
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig


@jax.tree_util.register_pytree_node_class
class Correlation2Poles(BinnedStatistic):

    _label_x = r'$s$ [$\mathrm{Mpc}/h$]'
    _label_proj = r'$\ell$'
    _label_value = r'$\xi_{\ell}(s)$'
    _data_fields = BinnedStatistic._data_fields + ['_num_shotnoise', '_num_zero', '_volume']
    _select_x_fields = BinnedStatistic._select_x_fields + ['_num_shotnoise', '_num_zero', '_volume']
    _select_proj_fields = BinnedStatistic._select_proj_fields + ['_num_shotnoise', '_num_zero', '_volume']
    _sum_fields = BinnedStatistic._sum_fields + ['_num_shotnoise', '_num_zero']
    _init_fields = {'s': '_x', 'num': '_value', 'nmodes': '_weights', 'edges': '_edges', 'volume': '_volume', 'ells': '_projs', 'norm': '_norm',
                    'num_shotnoise': '_num_shotnoise', 'num_zero': '_num_zero', 'name': 'name', 'attrs': 'attrs'}

    def __init__(self, s: np.ndarray, num: jax.Array, ells: tuple, nmodes: np.ndarray=None, edges: np.ndarray=None, volume: np.ndarray=None, norm: jax.Array=1.,
                 num_shotnoise: jax.Array=0., num_zero: jax.Array=None, name: str=None, attrs: dict=None):

        def _tuple(item):
            if item is None:
                return None
            if not isinstance(item, (tuple, list)):
                item = (item,) * len(ells)
            return tuple(item)

        super().__init__(x=_tuple(s), edges=_tuple(edges), projs=ells, value=num,
                         weights=_tuple(nmodes), norm=norm, name=name, attrs=attrs)
        if volume is None: volume = tuple(nmodes.copy() for nmodes in self.nmodes())
        else: volume = _tuple(volume)
        if num_zero is None: num_zero = _tuple(0.)
        num_zero = list(num_zero)
        for ill, value in enumerate(num_zero):
            if jnp.size(value) <= 1: num_zero[ill] = jnp.zeros_like(self._value[ill]).at[...].set(value)
        num_zero = tuple(num_zero)
        if not isinstance(num_shotnoise, (tuple, list)):
             num_shotnoise = (num_shotnoise,) + (0,) * (len(ells) - 1)
        num_shotnoise = list(num_shotnoise)
        for ill, value in enumerate(num_shotnoise):
            if jnp.size(value) <= 1: num_shotnoise[ill] = jnp.where((self._edges[ill][..., 0] <= 0.) & (self._edges[ill][..., 1] >= 0.), value, 0.)
        num_shotnoise = tuple(num_shotnoise)
        self.__dict__.update(_volume=volume, _num_shotnoise=num_shotnoise, _num_zero=num_zero)
    @property
    def num(self):
        """Correlation function with shot noise *not* subtracted."""
        return self._value

    s = BinnedStatistic.x
    savg = BinnedStatistic.xavg
    nmodes = BinnedStatistic.weights

    def shotnoise(self, projs=Ellipsis):
        """Shot noise."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        num_shotnoise = [s / self._norm for s in self._num_shotnoise]
        if isscalar: return num_shotnoise[iprojs]
        return [num_shotnoise[iproj] for iproj in iprojs]
    
    def volume(self, projs=Ellipsis):
        """Volume (optionally restricted to input projs)."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return self._volume[iprojs]
        return [self._volume[iproj] for iproj in iprojs]

    @property
    def ells(self):
        return self._projs

    @property
    def value(self):
        """Correlation function estimate."""
        toret = list(self.num)
        for ill, ell in enumerate(self.ells):
            toret[ill] = (toret[ill] - self._num_zero[ill] - self._num_shotnoise[ill]).real / self._norm
        return tuple(toret)

    def to_spectrum(self, k):
        from .utils import BesselIntegral
        num = []
        for ill, ell in enumerate(self.ells):
            if isinstance(k, BinnedStatistic):
                kk = k._x[ill]
            elif isinstance(k, (tuple, list)):
                kk = k[ill]
            else:
                kk = k
            #integ = BesselIntegral(self.edges(projs=ell), kk, ell=ell, method='trapz', mode='forward', edges=True, volume=False)
            integ = BesselIntegral(self.x(projs=ell), kk, ell=ell, method='rect', mode='forward', edges=False, volume=False)
            #num.append(integ(self._value[ill]))
            # self.weights = volume factor
            volume = self.volume(projs=ell)
            xi = jnp.where(volume == 0., 0., self.view(projs=ell))
            num.append(integ(volume * xi))
        num = tuple(num)
        if isinstance(k, BinnedStatistic):
            return k.clone(num=num)
        else:
            return Spectrum2Poles(k=k, num=num, ells=self.ells, norm=self._norm, num_shotnoise=self._num_shotnoise, num_zero=self._num_zero, attrs=self.attrs)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot correlation function.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self._x[ill], self._x[ill]**2 * self.value[ill].real, label=self._get_label_proj(ell))
            #ax.plot(self._x[ill], self.value[ill].real, label=self._get_label_proj(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(self._label_x)
        ax.set_ylabel(r'$s^2 \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig


@partial(register_pytree_dataclass, meta_fields=['ells'])
@dataclass(init=False, frozen=True)
class BinMesh2Spectrum(object):

    edges: jax.Array = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    ibin: jax.Array = None
    wmodes: jax.Array = None
    ells: tuple = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, ells: int | tuple=0, mode_oversampling: int=0):
        if not isinstance(mattrs, MeshAttrs):
            mattrs = mattrs.attrs
        hermitian = mattrs.hermitian
        vec = mattrs.kcoords(kind='separation', sparse=True)
        vec0 = mattrs.kfun.min()

        wmodes = None
        if hermitian:
            wmodes = _get_hermitian_weights(vec, sharding_mesh=None)
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            step = edges.get('step', None)
            if step is None:
                edges = _find_unique_edges(vec, vec0, xmin=edges.get('min', 0.), xmax=edges.get('max', np.inf))
            else:
                edges = np.arange(edges.get('min', 0.), edges.get('max', vec0 * np.min(mattrs.meshsize) / 2.), step)
        if edges.ndim == 2:  # coming from BinnedStatistic
            assert np.allclose(edges[1:, 0], edges[:-1, 1])
            edges = np.append(edges[:, 0], edges[-1, 1])
        shifts = [jnp.arange(-mode_oversampling, mode_oversampling + 1)] * len(mattrs.meshsize)
        shifts = list(itertools.product(*shifts))
        ibin, nmodes, xsum = [], 0, 0
        for shift in shifts:
            coords = jnp.sqrt(sum((xx + ss)**2 for (xx, ss) in zip(vec, shift)))
            bin = _get_bin_attrs(coords, edges, weights=wmodes)
            del coords
            ibin.append(bin[0])
            nmodes += bin[1]
            xsum += bin[2]
        edges = np.column_stack([edges[:-1], edges[1:]])
        ells = _format_ells(ells)
        self.__dict__.update(edges=edges, nmodes=nmodes / len(shifts), xavg=xsum / nmodes, ibin=ibin, wmodes=wmodes, ells=ells)

    def __call__(self, mesh, antisymmetric=False, remove_zero=False):
        weights = self.wmodes
        value = mesh.value if isinstance(mesh, BaseMeshField) else mesh
        if remove_zero:
            value = value.at[(0,) * value.ndim].set(0.)
        return _bincount(self.ibin, value, weights=weights, length=len(self.xavg), antisymmetric=antisymmetric) / self.nmodes


@partial(register_pytree_dataclass, meta_fields=['ells'])
@dataclass(init=False, frozen=True)
class BinMesh2Correlation(object):

    edges: jax.Array = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    ibin: jax.Array = None
    ells: tuple = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, ells: int | tuple=0, mode_oversampling: int=0):
        if not isinstance(mattrs, MeshAttrs):
            mattrs = mattrs.attrs
        vec = mattrs.xcoords(kind='separation', sparse=True)
        vec0 = mattrs.cellsize.min()

        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            step = edges.get('step', None)
            if step is None:
                edges = _find_unique_edges(vec, vec0, xmin=edges.get('min', 0.), xmax=edges.get('max', np.inf))
            else:
                edges = np.arange(edges.get('min', 0.), edges.get('max', vec0 * np.min(mattrs.meshsize) / 2.), step)
        if edges.ndim == 2:  # coming from BinnedStatistic
            assert np.allclose(edges[1:, 0], edges[:-1, 1])
            edges = np.append(edges[:-1, 0], edges[-1, 1])
        shifts = [jnp.arange(-mode_oversampling, mode_oversampling + 1)] * len(mattrs.meshsize)
        shifts = list(itertools.product(*shifts))
        ibin, nmodes, xsum = [], 0, 0
        for shift in shifts:
            coords = jnp.sqrt(sum((xx + ss)**2 for (xx, ss) in zip(vec, shift)))
            bin = _get_bin_attrs(coords, edges, weights=None)
            del coords
            ibin.append(bin[0])
            nmodes += bin[1]
            xsum += bin[2]
        edges = np.column_stack([edges[:-1], edges[1:]])
        ells = _format_ells(ells)
        self.__dict__.update(edges=edges, nmodes=nmodes / len(shifts), xavg=xsum / nmodes, ibin=ibin, ells=ells)

    def __call__(self, mesh, remove_zero=False):
        value = mesh.value if isinstance(mesh, BaseMeshField) else mesh
        if remove_zero:
            value = value.at[(0,) * value.ndim].set(0.)
        return _bincount(self.ibin, value, weights=None, length=len(self.xavg)) / self.nmodes


def _get_los_vector(los: str | np.ndarray, ndim=3):
    vlos = None
    if isinstance(los, str):
        vlos = [0.] * ndim
        vlos['xyz'.index(los)] = 1.
    else:
        vlos = los
    return staticarray(vlos)


def _get_zero(mesh):
    return mesh[(0,) * mesh.ndim]


def _format_meshs(*meshs):
    meshs = list(meshs)
    assert 1 <= len(meshs) <= 2
    meshs = meshs + [None] * (2 - len(meshs))
    same = [0]
    for mesh in meshs[1:]: same.append(same[-1] if mesh is None else same[-1] + 1)
    for imesh, mesh in enumerate(meshs):
        if mesh is None:
            meshs[imesh] = meshs[imesh - 1]
    return meshs, same[1] == same[0]


def _format_ells(ells):
    if np.ndim(ells) == 0: ells = (ells,)
    ells = tuple(sorted(ells))
    return ells


def _format_los(los, ndim=3):
    vlos, swap = None, False
    if isinstance(los, str) and los in ['firstpoint', 'local', 'endpoint']:
        if los == 'local': los = 'firstpoint'
        swap = los == 'endpoint'
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    return los, vlos, swap


def compute_mesh2(*meshs: RealMeshField | ComplexMeshField, bin: BinMesh2Spectrum | BinMesh2Correlation=None, los: str | np.ndarray='x'):
    if isinstance(bin, BinMesh2Spectrum):
        return compute_mesh2_spectrum(*meshs, bin=bin, los=los)
    elif isinstance(bin, BinMesh2Correlation):
        return compute_mesh2_correlation(*meshs, bin=bin, los=los)
    raise ValueError(f'bin must be either BinMesh2Spectrum or BinMesh2Correlation, not {type(bin)}')


def compute_mesh2_spectrum(*meshs: RealMeshField | ComplexMeshField, bin: BinMesh2Spectrum=None, los: str | np.ndarray='x') -> Spectrum2Poles:
    r"""
    Compute power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField
        Input mesh(s).

    edges : np.ndarray, dict, default=None
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, array, default=None
        If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    power : Spectrum2Poles
    """
    meshs, autocorr = _format_meshs(*meshs)
    rdtype = meshs[0].real.dtype
    mattrs = meshs[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod()

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
    attrs = dict(los=vlos if vlos is not None else los)
    ells = bin.ells

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    if vlos is None:  # local, varying line-of-sight
        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]
        if swap: meshs = meshs[::-1]

        rmesh1 = meshs[0]
        A0 = _2c(rmesh1 if autocorr else meshs[1])
        del meshs

        num, num_zero = [], []
        if 0 in ells:
            if autocorr:
                Aell = A0.clone(value=A0.real**2 + A0.imag**2)  # saves a bit of memory
            else:
                Aell = _2c(rmesh1) * A0.conj()
            num.append(bin(Aell, antisymmetric=False))
            num_zero.append(_get_zero(Aell))
            del Aell

        if nonzeroells:
            rmesh1 = _2r(rmesh1)
            # The real-space grid
            xvec = rmesh1.coords(sparse=True)
            # The Fourier-space grid
            kvec = A0.coords(sparse=True)

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += _2c(rmesh1 * jax.lax.switch(im, Ylm, *xvec)).conj() * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            for ell in nonzeroells:
                Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
                #jax.debug.inspect_array_sharding(jnp.zeros_like(A0.value), callback=print)
                xs = np.arange(len(Ylms))
                Aell = jax.lax.scan(partial(f, Ylms), init=A0.clone(value=jnp.zeros_like(A0.value)), xs=xs)[0] * A0
                #Aell = sum(_2c(rmesh1 * Ylm(*xvec)) * Ylm(*kvec) for Ylm in Ylms).conj() * A0
                # Project on to 1d k-basis (averaging over mu=[-1, 1])
                num.append(4. * jnp.pi * bin(Aell, antisymmetric=bool(ell % 2), remove_zero=True))
                num_zero.append(4. * jnp.pi * 0.)
                del Aell

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        num, num_zero = map(jnp.array, (num, num_zero))
        if swap: num, num_zero = map(jnp.conj, (num, num_zero))
        # Format the num results into :class:`Spectrum2Poles` instance
        num_zero /= bin.nmodes[0]
        return Spectrum2Poles(bin.xavg, num=num, nmodes=bin.nmodes, volume=mattrs.kfun.prod() * bin.nmodes, edges=bin.edges, ells=ells, norm=norm, num_zero=num_zero, attrs=attrs)

    else:  # fixed line-of-sight

        for imesh, mesh in enumerate(meshs[:1 if autocorr else 2]):
            meshs[imesh] = _2c(mesh)
        if autocorr:
            Aell = meshs[0].clone(value=meshs[0].real**2 + meshs[0].imag**2)  # saves a bit of memory
        else:
            Aell = meshs[0] * meshs[1].conj()
        del meshs

        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        num, num_zero = [], []
        if 0 in ells:
            num.append(bin(Aell, antisymmetric=False))
            num_zero.append(_get_zero(Aell))

        if nonzeroells:
            kvec = Aell.coords(sparse=True)
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)
            for ell in nonzeroells:  # TODO: jax.lax.scan
                num.append((2 * ell + 1) * bin(Aell * get_legendre(ell)(mu), antisymmetric=bool(ell % 2), remove_zero=True))
                num_zero.append(0.)

        num, num_zero = map(jnp.array, (num, num_zero))
        num_zero /= bin.nmodes[0]
        return Spectrum2Poles(bin.xavg, num=num, nmodes=bin.nmodes, volume=mattrs.kfun.prod() * bin.nmodes, edges=bin.edges, ells=ells, norm=norm, num_zero=num_zero, attrs=attrs)


def compute_mesh2_correlation(*meshs: RealMeshField | ComplexMeshField, bin: BinMesh2Correlation=None, los: str | np.ndarray='x') -> Correlation2Poles:
    r"""
    Compute 2-pt correlation function from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField
        Input mesh(s).

    edges : np.ndarray, dict, default=None
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, array, default=None
        If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    corr : Correlation2Poles
    """
    meshs, autocorr = _format_meshs(*meshs)
    rdtype = meshs[0].real.dtype
    mattrs = meshs[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod()

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
    attrs = dict(los=vlos if vlos is not None else los)
    ells = bin.ells

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    if vlos is None:  # local, varying line-of-sight
        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]
        if swap: meshs = meshs[::-1]

        rmesh1 = meshs[0]
        A0 = _2c(rmesh1 if autocorr else meshs[1])
        del meshs

        num, num_zero = [], []
        if 0 in ells:
            if autocorr:
                Aell = A0.clone(value=A0.real**2 + A0.imag**2)  # saves a bit of memory
            else:
                Aell = _2c(rmesh1) * A0.conj()
            Aell = Aell.c2r()  # convert to real space
            num.append(bin(Aell))
            num_zero.append(Aell.mean())
            del Aell

        if nonzeroells:
            rmesh1 = _2r(rmesh1)
            # The real-space grid
            xvec = rmesh1.coords(sparse=True)
            # The separation grid
            svec = rmesh1.coords(kind='separation', sparse=True)

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += _2r(_2c(rmesh1 * jax.lax.switch(im, Ylm, *xvec)).conj() * A0) * jax.lax.switch(im, Ylm, *svec)
                return carry, im

            for ell in nonzeroells:
                Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
                xs = np.arange(len(Ylms))
                Aell = jax.lax.scan(partial(f, Ylms), init=rmesh1.clone(value=jnp.zeros_like(rmesh1.value)), xs=xs)[0]
                num.append(4. * jnp.pi * bin(Aell, remove_zero=True))
                num_zero.append(4. * jnp.pi * 0.)
                del Aell

        num, num_zero = map(lambda array: jnp.array(array) / mattrs.cellsize.prod(), (num, num_zero))
        if swap: num, num_zero = map(jnp.conj, (num, num_zero))
        # Format the num results into :class:`Correlation2Poles` instance
        num_zero /= bin.nmodes[0]
        return Correlation2Poles(bin.xavg, num=num, nmodes=bin.nmodes, volume=mattrs.cellsize.prod() * bin.nmodes, edges=bin.edges, ells=ells, norm=norm, num_zero=num_zero, attrs=attrs)

    else:  # fixed line-of-sight

        for imesh, mesh in enumerate(meshs[:1 if autocorr else 2]):
            meshs[imesh] = _2c(mesh)
        if autocorr:
            Aell = meshs[0].clone(value=meshs[0].real**2 + meshs[0].imag**2)  # saves a bit of memory
        else:
            Aell = meshs[0] * meshs[1].conj()
        Aell = Aell.c2r()  # convert to real space
        del meshs

        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        num, num_zero = [], []
        if 0 in ells:
            num.append(bin(Aell))
            num_zero.append(Aell.mean())

        if nonzeroells:
            svec = Aell.coords(kind='separation', sparse=True)
            mu = sum(ss * ll for ss, ll in zip(svec, vlos)) / jnp.sqrt(sum(ss**2 for ss in svec)).at[(0,) * mattrs.ndim].set(1.)
            for ell in nonzeroells:  # TODO: jax.lax.scan
                num.append((2 * ell + 1) * bin(Aell * get_legendre(ell)(mu), remove_zero=True))
                num_zero.append(0.)

        num, num_zero = map(lambda array: jnp.array(array) / mattrs.cellsize.prod(), (num, num_zero))
        num_zero /= bin.nmodes[0]
        return Correlation2Poles(bin.xavg, num=num, nmodes=bin.nmodes, volume=mattrs.cellsize.prod() * bin.nmodes, edges=bin.edges, ells=ells, norm=norm, num_zero=num_zero, attrs=attrs)


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

    def __init__(self, data, randoms, attrs=None):
        if attrs is not None:
            data = data.clone(attrs=attrs)
            randoms = randoms.clone(attrs=attrs)
        self.__dict__.update(data=data, randoms=randoms)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = asdict(self) | kwargs
        return self.__class__(**state)

    def exchange(self, **kwargs):
        return self.clone(data=self.data.clone(exchange=True, **kwargs), randoms=self.randoms.clone(exchange=True, **kwargs), attrs=self.attrs)

    @property
    def attrs(self):
        return self.data.attrs

    def split(self, nsplits=1, extent=None):
        from .mesh import _get_extent
        if extent is None:
            extent = _get_extent(self.data.positions, self.randoms.positions)
        for data, randoms in zip(self.data.split(nsplits=nsplits, extent=extent), self.randoms.split(nsplits=nsplits, extent=extent)):
            new = self.clone(data=data, randoms=randoms)
            yield new

    @property
    def particles(self):
        particles = getattr(self, '_particles', None)
        if particles is None:
            self.__dict__['_particles'] = particles = (self.data - self.data.sum() / self.randoms.sum() * self.randoms).clone(attrs=self.data.attrs)
        return particles

    def paint(self, resampler: str | Callable='cic', interlacing: int=1,
              compensate: bool=False, dtype=None, out: str='real', **kwargs):
        return self.particles.paint(resampler=resampler, interlacing=interlacing, compensate=compensate, dtype=dtype, out=out, **kwargs)


def compute_normalization(*inputs: RealMeshField | ParticleField, resampler='cic', **kwargs) -> jax.Array:
    """
    Return normalization, in volume**(1 - len(inputs)) unit.

    Warning
    -------
    Input particles are considered uncorrelated.
    """
    meshs, particles = [], []
    attrs = dict(kwargs)
    for inp in inputs:
        if isinstance(inp, RealMeshField):
            meshs.append(inp)
            attrs = {name: getattr(inp, name) for name in ['boxsize', 'boxcenter', 'meshsize']}
        else:
            particles.append(inp)
    if particles:
        cattrs = dict(particles[0].attrs)
        if 'cellsize' in attrs: cattrs.pop('meshsize')
        attrs = particles[0].attrs.clone(**get_mesh_attrs(**(cattrs | attrs)))
        particles = [particle.clone(attrs=attrs) for particle in particles]
    normalization = 1
    for mesh in meshs:
        normalization *= mesh
    for particle in particles:
        normalization *= particle.paint(resampler=resampler, interlacing=0, compensate=False)
    return normalization.sum() * normalization.cellsize.prod()**(1 - len(inputs))


def compute_fkp2_spectrum_normalization(*fkps, cellsize=10.):
    # This is the pypower normalization - move to new one?
    fkps, autocorr = _format_meshs(*fkps)
    if autocorr:
        fkp = fkps[0]
        randoms = [fkp.data, fkp.randoms]  # cross to remove common noise
        #mask = random.uniform(random.key(42), shape=fkp.randoms.size) < 0.5
        #randoms = [fkp.randoms[mask], fkp.randoms[~mask]]
        #randoms = [fkp.randoms[:fkp.randoms.size // 2], fkp.randoms[fkp.randoms.size // 2:]]
        alpha = fkp.data.sum() / fkp.randoms.sum()
        norm = alpha * compute_normalization(*randoms, cellsize=cellsize)
    else:
        randoms = [fkps[0].data, fkps[1].randoms]  # cross to remove common noise
        alpha2 = fkps[1].data.sum() / fkps[1].randoms.sum()
        norm = alpha2 * compute_normalization(*randoms, cellsize=cellsize)
        randoms = [fkps[1].data, fkps[0].randoms]
        alpha2 = fkps[0].data.sum() / fkps[0].randoms.sum()
        norm += alpha2 * compute_normalization(*randoms, cellsize=cellsize)
        norm = norm / 2
    return norm


def compute_fkp2_spectrum_shotnoise(*fkps):
    # This is the pypower normalization - move to new one?
    fkps, autocorr = _format_meshs(*fkps)
    if autocorr:
        fkp = fkps[0]
        alpha = fkp.data.sum() / fkp.randoms.sum()
        shotnoise = jnp.sum(fkp.data.weights**2) + alpha**2 * jnp.sum(fkp.randoms.weights**2)
    else:
        shotnoise = 0.
    return shotnoise


def compute_wide_angle_spectrum2_poles(poles: dict[Callable]):
    r"""
    Add (first) wide-angle order power spectrum multipoles to input dictionary of poles.

    Parameters
    ----------
    poles : dict[Callable]
        A dictionary of callables, with keys the multipole orders :math:`\ell`.
        Non-provided poles are assumed zero.

    Returns
    -------
    poles : Dictionary of callables with keys (multipole order, wide-angle order) :math:`(\ell, n)`.
    """
    toret = {(ell, 0): pole for ell, pole in poles.items()}
    for ell in range(max(list(toret) + [0]) + 1):
        tmp = []
        if ell - 1 in poles:
            p = poles[ell - 1]
            coeff = - ell * (ell - 1) / (2. * (2. * ell - 1))
            tmp.append(coeff * (ell - 1), p)
            tmp.append(- coeff, lambda k: k * jax.jacfwd(p)(k))
        if ell + 1 in poles:
            p = poles[ell + 1]
            coeff = - (ell + 1) * (ell + 2) / (2. * (2. * ell + 3))
            tmp.append(coeff * (ell + 1), p)
            tmp.append(coeff, lambda k: k * jax.jacfwd(p)(k))

        def func(k):
            return sum(coeff * p(k) for coeff, p in tmp)

        toret[(ell, 1)] = func
    return toret


def compute_smooth2_spectrum_window(window, edgesin: np.ndarray, ellsin: tuple=None, bin: BinMesh2Spectrum=None) -> WindowMatrix:

    r"""Compute "smooth" power spectrum window matrix given input configuration-space window function."""

    from .utils import legendre_product, BesselIntegral
    tophat_method = 'rect'
    ells = bin.ells

    if isinstance(edgesin, BinnedStatistic):
        kin = edgesin._edges[0]
        ellsin = edgesin.projs

    if edgesin.ndim == 2:
        kin = edgesin
        edgesin = None
    else:
        kin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

    kout = jnp.where(bin.nmodes == 0, 0., bin.xavg)
    window = window.clone(num_zero=None)

    wmat_tmp = {}
    spherical_jn = {ell: get_spherical_jn(ell) for ell in set(ells)}

    for ell1 in ellsin:
        wmat_tmp[ell1] = 0
        for ill, ell in enumerate(ells):
            Qs = sum(legendre_product(ell, ell1, q) * window.view(projs=q).real if q in window.projs else jnp.zeros(()) for q in list(range(abs(ell - ell1), ell + ell1 + 1)))
            snmodes = window.volume()[0]
            savg = jnp.where(snmodes == 0, 0., window.x()[0])
            Qs = jnp.where(snmodes == 0, 0., Qs)
            #integ = BesselIntegral(window.edges(projs=0), kout, ell=ell, method='rect', mode='forward', edges=True, volume=False)

            def f(kin):
                tophat_Qs = BesselIntegral(kin, savg, ell=ell1, edges=True, method=tophat_method, mode='backward').w[..., 0] * Qs
                def f2(kout):
                    integ = BesselIntegral(savg, kout, ell=ell, method='rect', mode='forward', edges=False, volume=False)
                    return integ(snmodes * tophat_Qs)
                #    return (-1)**(ell // 2) * jnp.sum(snmodes * spherical_jn[ell](kout * savg) * tophat_Qs)
                batch_size = int(min(max(1e7 / savg.size, 1), kout.size))
                power = (2 * ell + 1) * jax.lax.map(f2, kout, batch_size=batch_size)
                #power = (2 * ell + 1) * integ(snmodes * tophat_Qs)
                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                return power.ravel()

            batch_size = int(min(max(1e7 / (kout.size * savg.size), 1), kin.shape[0]))
            wmat_tmp[ell1] += jax.lax.map(f, xs=kin, batch_size=batch_size)

    wmat = jnp.concatenate(list(wmat_tmp.values()), axis=0).T

    observable = BinnedStatistic(x=[bin.xavg] * len(ells), value=[jnp.zeros_like(bin.xavg)] * len(ells), edges=[bin.edges] * len(ells), projs=ells)
    xin = np.mean(kin, axis=-1)
    theory = BinnedStatistic(x=[xin] * len(ellsin), value=[jnp.zeros_like(xin)] * len(ellsin), edges=[kin] * len(ellsin), projs=ellsin)
    wmat = WindowMatrix(observable, theory, wmat)
    return wmat


def compute_mesh2_spectrum_window(*meshs: RealMeshField | ComplexMeshField | MeshAttrs, edgesin: np.ndarray, ellsin: tuple=None,
                                  bin: BinMesh2Spectrum=None, los: str | np.ndarray='x',
                                  buffer=None, batch_size=None, pbar=False, norm=None, flags=tuple()) -> WindowMatrix:
    r"""
    Compute mean power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, MeshAttrs
        Input mesh(s).
        A :class:`MeshAttrs` instance can be directly provided, in case the selection function is trivial (constant).

    edges : np.ndarray, dict, default=None
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, array, default=None
        If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    wmat : WindowMatrix
    """
    tophat_method = 'exact'

    from .utils import BesselIntegral

    meshs, autocorr = _format_meshs(*meshs)
    periodic = isinstance(meshs[0], MeshAttrs)
    if periodic:
        assert autocorr
        rdtype = float
        mattrs = meshs[0]
    else:
        rdtype = meshs[0].real.dtype
        mattrs = meshs[0].attrs

    _norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod()
    if norm is None: norm = _norm
    rnorm = _norm / norm / mattrs.meshsize.prod(dtype=rdtype)

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
    ells = bin.ells

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    if isinstance(edgesin, BinnedStatistic):
        kin = edgesin._edges[0]
        ellsin = edgesin.projs

    if edgesin.ndim == 2:
        kin = edgesin
        edgesin = None
    else:
        kin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

    def np_map(f, xs):
        return jnp.array(list(map(f, xs)))

    svec = mattrs.xcoords(kind='separation', sparse=True)

    if pbar:
        from tqdm import tqdm
        t = tqdm(total=len(kin), bar_format='{l_bar}{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        def round(n):
            return int(n * 1e6) / 1e6

    if vlos is not None:

        if np.ndim(ellsin) == 0: ellsin = (ellsin,)
        ellsin = list(ellsin)

        if not periodic:

            for imesh, mesh in enumerate(meshs[:1 if autocorr else 2]):
                meshs[imesh] = _2c(mesh)
            if autocorr:
                meshs[1] = meshs[0]

            Q = _2r(meshs[0] * meshs[1].conj()) / mattrs.meshsize.prod(dtype=rdtype)
        else:
            Q = None

        kvec = mattrs.kcoords(sparse=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)

        def _bin(Aell):
            power = []
            for ell in ells:
                leg = get_legendre(ell)(mu)
                odd = ell % 2
                if odd: leg += get_legendre(ell)(-mu)
                power.append((2 * ell + 1) / (1 + odd) * bin(Aell * leg, remove_zero=True))
            return jnp.concatenate(power)

        def my_map(f, xs):
            if pbar:
                return np_map(f, xs)
            return jax.lax.map(f, xs=xs, batch_size=batch_size)

        wmat = []

        if 'smooth' in flags:

            spherical_jn = {ell: get_spherical_jn(ell) for ell in set(ells)}
            sedges = None
            #sedges = np.arange(0., rmesh1.boxsize.max() / 2., rmesh1.cellsize.min() / 4.)
            sbin = BinMesh2Correlation(mattrs, edges=sedges)
            kout = jnp.where(bin.nmodes == 0, 0., bin.xavg)

            for ellin in ellsin:

                wmat_tmp = []
                for ill, ell in enumerate(ells):
                    snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                    smu = sum(xx * ll for xx, ll in zip(svec, vlos)) / jnp.where(snorm == 0., 1., snorm)
                    Qs = sbin((Q if Q is not None else 1.) * get_legendre(ell)(smu) * get_legendre(ellin)(smu))
                    if ell != 0: Qs = Qs.at[0].set(0.)
                    Qs = jnp.where(sbin.nmodes == 0, 0., Qs)
                    savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                    snmodes = sbin.nmodes

                    del snorm

                    def f(kin):
                        tophat_Qs = BesselIntegral(kin, savg, ell=ellin, edges=True, method=tophat_method, mode='backward').w[..., 0] * rnorm * mattrs.boxsize.prod() * Qs

                        def f2(kout):
                            return (-1)**(ell // 2) * jnp.sum(snmodes * spherical_jn[ell](kout * savg) * tophat_Qs)

                        batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / savg.size, 1), kout.size))
                        power = jax.lax.map(f2, kout, batch_size=batch_size)

                        if pbar:
                            t.update(n=round(1 / len(ells) / len(ellsin)))
                        return (2 * ell + 1) * power

                    wmat_tmp.append(my_map(f, kin))
                wmat.append(jnp.concatenate(wmat_tmp, axis=-1))

        elif 'infinite' in flags:
            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
            smu = sum(xx * ll for xx, ll in zip(svec, vlos)) / jnp.where(snorm == 0., 1., snorm)

            for ellin in ellsin:
                legin = get_legendre(ellin)(smu)

                def f(kin):
                    Aell = BesselIntegral(kin, snorm, ell=ellin, edges=True, method=tophat_method, mode='backward').w[..., 0] * legin * rnorm * mattrs.boxsize.prod(dtype=rdtype)
                    Aell = mattrs.create(kind='real', fill=Aell)
                    if Q is not None: Aell *= Q
                    power = _bin(_2c(Aell))
                    if pbar:
                        t.update(n=round(1 / len(ellsin)))
                    return power

                wmat.append(my_map(f, kin))

        else:   

            for ellin in ellsin:
                legin = get_legendre(ellin)(mu)

                def f(kin):
                    Aell = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm < kin[-1])) * legin * rnorm * mattrs.meshsize.prod(dtype=rdtype))
                    if Q is not None: Aell = _2c(Q * _2r(Aell))
                    power = _bin(Aell)
                    if pbar:
                        t.update(n=round(1 / len(ellsin)))
                    return power

                wmat.append(my_map(f, kin))

        wmat = jnp.concatenate(wmat, axis=0).T

    else:
        theory_los = 'firstpoint'
        if len(ellsin) == 2 and isinstance(ellsin[1], str):
            ellsin, theory_los = ellsin
        if np.ndim(ellsin) == 0: ellsin = (ellsin,)
        ellsin = list(ellsin)

        # In this case, theory must be a dictionary of (multipole, wide_angle_order)
        if swap: meshs = meshs[::-1]

        if periodic:
            meshs = [mattrs.create(kind='real', fill=1.)]

        rmesh1 = _2r(meshs[0])
        rmesh2 = rmesh1 if autocorr else _2r(meshs[1])
        A0 = _2c(meshs[0] if autocorr else meshs[1])
        del meshs

        # The real-space grid
        xvec = mattrs.xcoords(sparse=True)

        # The Fourier-space grid
        kvec = A0.coords(sparse=True)

        Ylms = {ell: [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}
        spherical_jn = {ell: get_spherical_jn(ell) for ell in set(ells) | set(ellsin)}
        has_buffer = False

        if isinstance(buffer, str) and buffer in ['gpu', 'cpu']:
            buffer = jax.devices(buffer)[0]
            has_buffer = True
        elif isinstance(buffer, (str, Path)):
            buffer = str(buffer)
            has_buffer = True

        def my_map(f, xs):
            if has_buffer or pbar:
                return np_map(f, xs)
            return jax.lax.map(f, xs=xs, batch_size=batch_size)

        def dump_to_buffer(mesh, key):
            toret = None
            if buffer is None:
                toret = mesh
            elif isinstance(buffer, str):
                key = '_'.join(list(map(str, key)))
                toret = os.path.join(buffer, f'mesh_{key}.npz')
                mesh.save(toret)
            else:
                toret = jax.device_put(mesh, device=buffer, donate=True)
            return toret

        def load_from_buffer(obj):
            if buffer is None:
                toret = obj
            elif isinstance(buffer, str):
                toret = RealMeshField.load(obj)
            else:
                toret = jax.device_put(obj)
            return toret

        if theory_los == 'firstpoint':

            ellsin = [ellin if isinstance(ellin, tuple) else (ellin, 0) for ellin in ellsin]
            wmat_tmp = {}
            if 'smooth' in flags:
                sedges = None
                #sedges = np.arange(0., rmesh1.boxsize.max() / 2., rmesh1.cellsize.min() / 4.)
                sbin = BinMesh2Correlation(rmesh1, edges=sedges)
                kout = jnp.where(bin.nmodes == 0, 0., bin.xavg)

                for ell1, wa1 in ellsin:
                    wmat_tmp[ell1, wa1] = 0
                    for ill, ell in enumerate(ells):
                        Qs = 0.
                        xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                        snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                        for im1, Yl1m1 in enumerate(Ylms[ell1]):
                            for im, Ylm in enumerate(Ylms[ell]):
                                Q = (4. * np.pi) / (2 * ell1 + 1) * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xvec) * Yl1m1(*xvec)).conj() * A0) * snorm**wa1
                                Qs += 4. * np.pi * sbin(Q * Ylm(*svec) * Yl1m1(*svec)) * rnorm * mattrs.cellsize.prod()

                        if ell != 0: Qs = Qs.at[0].set(0.)
                        Qs = jnp.where(sbin.nmodes == 0, 0., Qs)
                        savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                        snmodes = sbin.nmodes

                        del xnorm, snorm

                        def f(kin):
                            tophat_Qs = BesselIntegral(kin, savg, ell=ell1, edges=True, method=tophat_method, mode='backward').w[..., 0] * Qs

                            def f2(kout):
                                return (-1)**(ell // 2) * jnp.sum(snmodes * spherical_jn[ell](kout * savg) * tophat_Qs)

                            batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / savg.size, 1), kout.size))
                            power = jax.lax.map(f2, kout, batch_size=batch_size)

                            if pbar:
                                t.update(n=round(1 / len(ells) / len(ellsin)))
                            power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                            return power.ravel()

                        wmat_tmp[ell1, wa1] += my_map(f, kin)

            elif 'infinite' in flags:
                for ell1, wa1 in ellsin:
                    wmat_tmp[ell1, wa1] = 0
                    for ill, ell in enumerate(ells):
                        for im, Ylm in enumerate(Ylms[ell]):
                            xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                            Qs = 0.
                            for im1, Yl1m1 in enumerate(Ylms[ell1]):
                                Q = (4. * np.pi) / (2 * ell1 + 1) * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xvec) * Yl1m1(*xvec)).conj() * A0) * snorm**wa1
                                Qs += Yl1m1(*svec) * Q * rnorm * mattrs.cellsize.prod()
                            del xnorm

                            def f(kin):
                                tophat_Qs = BesselIntegral(kin, snorm, ell=ell1, edges=True, method=tophat_method, mode='backward').w[..., 0] * Qs
                                power = 4 * jnp.pi * bin(Ylm(*kvec) * _2c(tophat_Qs), antisymmetric=bool(ell % 2), remove_zero=ell == 0)
                                if pbar:
                                    t.update(n=round(1. / sum(len(Ylms[ell]) for ell in ells) / len(ellsin)))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()
                            wmat_tmp[ell1, wa1] += my_map(f, kin)
            else:

                for ell1, wa1 in ellsin:
                    wmat_tmp[ell1, wa1] = 0
                    Qs = {}
                    for im1, Yl1m1 in enumerate(Ylms[ell1]):
                        for ill, ell in enumerate(ells):
                            if 'recompute' in flags: Qs = {}
                            xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                            for im, Ylm in enumerate(Ylms[ell]):
                                key = ell, im, im1
                                tmp = (4. * np.pi) / (2 * ell1 + 1) * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xvec) * Yl1m1(*xvec)).conj() * A0) * snorm**wa1
                                Qs[key] = dump_to_buffer(tmp, key)
                            del xnorm, snorm
                            if 'recompute' in flags:
                                def f(kin):
                                    Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                                    knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                                    xi = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * rnorm * Yl1m1(*kvec)).c2r()
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im1])

                                    Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                    power = 4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0)
                                    del Aell[ell]
                                    if pbar:
                                        t.update(n=round((im1 + 1) / sum(len(Ylms[ell]) for ell in ells)) / sum(len(Ylms[ell]) for ell, _ in ellsin))
                                    power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                    return power.ravel()

                                wmat_tmp[ell1, wa1] += my_map(f, kin)

                    if 'recompute' not in flags:
                        def f(kin):
                            Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                            knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                            for im1, Yl1m1 in enumerate(Ylms[ell1]):
                                def kernel(*args):
                                    kmask = (knorm >= kin[0]) & (knorm <= kin[-1])
                                    return kmask * rnorm * Yl1m1(*kvec)

                                xi = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * rnorm * Yl1m1(*kvec)).c2r()
                                for ell in Aell:
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im1])

                            power = []
                            for ill, ell in enumerate(ells):
                                Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                power.append(4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0))
                                del Aell[ell]
                            if pbar:
                                t.update(n=round((im1 + 1) / sum(len(Ylms[ell]) for ell, _ in ellsin)))
                            return jnp.concatenate(power)

                        wmat_tmp[ell1, wa1] = my_map(f, kin)

            wmat = jnp.concatenate(list(wmat_tmp.values()), axis=0).T

        elif theory_los == 'local':

            wmat_tmp = {}
            if 'smooth' in flags:
                sedges = None
                #sedges = np.arange(0., rmesh1.boxsize.max() / 2., rmesh1.cellsize.min() / 4.)
                sbin = BinMesh2Correlation(rmesh1, edges=sedges)
                kout = jnp.where(bin.nmodes == 0, 0., bin.xavg)

                for ell1, ell2 in itertools.product((2, 0), (2, 0)):
                    wmat_tmp[ell1, ell2] = 0
                    for ill, ell in enumerate(ells):
                        ps = [p for p in (0, 2, 4) if real_gaunt((p, 0), (ell1, 0), (ell2, 0))]
                        Qs = {p: 0. for p in ps}
                        for (im1, Yl1m1), (im2, Yl2m2) in itertools.product(enumerate(Ylms[ell1]), enumerate(Ylms[ell2])):
                            for im, Ylm in enumerate(Ylms[ell]):
                                Q = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                                for p in ps:
                                    key = p
                                    tmp = 0.
                                    for imp, Ypmp in enumerate(Ylms[p]):
                                        rg = real_gaunt((p, imp - p), (ell1, im1 - ell1), (ell2, im2 - ell2))
                                        if rg:  # rg != 0
                                            tmp += rg * Ylm(*svec) * Ypmp(*svec) * Q
                                    if hasattr(tmp, 'shape'):
                                        tmp = 4. * np.pi * sbin(tmp) * rnorm * mattrs.cellsize.prod()
                                        Qs[key] += dump_to_buffer(tmp, key)

                        for p in ps:
                            Q = load_from_buffer(Qs[p])
                            if ell != 0: Q = Q.at[0].set(0.)
                            Q = jnp.where(sbin.nmodes == 0, 0., Q)
                            savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                            snmodes = sbin.nmodes

                            def f(kin):
                                tophat_Q = BesselIntegral(kin, savg, ell=p, edges=True, method=tophat_method, mode='backward').w[..., 0] * Q

                                def f2(kout):
                                    return (-1)**(ell // 2) * jnp.sum(snmodes * spherical_jn[ell](kout * savg) * tophat_Q)

                                batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / savg.size, 1), kout.size))
                                power = jax.lax.map(f2, kout, batch_size=batch_size)
                                if pbar:
                                    t.update(n=round(1 / len(ells) / 6))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()

                            wmat_tmp[ell1, ell2] += my_map(f, kin)

            elif 'infinite' in flags:
                for ell1, ell2 in itertools.product((2, 0), (2, 0)):
                    wmat_tmp[ell1, ell2] = 0
                    for ill, ell in enumerate(ells):
                        ps = [p for p in (0, 2, 4) if real_gaunt((p, 0), (ell1, 0), (ell2, 0))]
                        for im, Ylm in enumerate(Ylms[ell]):
                            Qs = {p: 0. for p in ps}
                            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                            for (im1, Yl1m1), (im2, Yl2m2) in itertools.product(enumerate(Ylms[ell1]), enumerate(Ylms[ell2])):
                                Q = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                                for p in ps:
                                    for imp, Ypmp in enumerate(Ylms[p]):
                                        rg = real_gaunt((p, imp - p), (ell1, im1 - ell1), (ell2, im2 - ell2))
                                        if rg:  # rg != 0
                                            Qs[p] += rg * Ypmp(*svec) * Q
                            for p in Qs:
                                Qs[p] = dump_to_buffer(Qs[p] * rnorm * mattrs.cellsize.prod(), p)

                            def f(kin):
                                xi = 0.
                                for p in ps:
                                    tophat = BesselIntegral(kin, snorm, ell=p, edges=True, method=tophat_method, mode='backward').w[..., 0]
                                    Q = load_from_buffer(Qs[p])
                                    xi += tophat * Q
                                power = 4 * jnp.pi * bin(Ylm(*kvec) * _2c(xi), antisymmetric=bool(ell % 2), remove_zero=ell == 0)
                                if pbar:
                                    t.update(n=round(1 / sum(len(Ylms[ell]) for ell in ells) / 4))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()
                            wmat_tmp[ell1, ell2] += my_map(f, kin)

            else:

                for ell1, ell2 in itertools.product((2, 0), (2, 0)):
                    wmat_tmp[ell1, ell2] = 0
                    Qs = {}
                    for im12, (Yl1m1, Yl2m2) in enumerate(itertools.product(Ylms[ell1], Ylms[ell2])):
                        for ill, ell in enumerate(ells):
                            if 'recompute' in flags: Qs = {}
                            for im, Ylm in enumerate(Ylms[ell]):
                                key = ell, im, im12
                                tmp = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                                Qs[key] = dump_to_buffer(tmp, key)
                            if 'recompute' in flags:
                                def f(kin):
                                    knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                                    Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                                    xi = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * rnorm * Yl1m1(*kvec) * Yl2m2(*[-kk for kk in kvec])).c2r()
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im12])
                                    Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                    power = 4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0)
                                    if pbar:
                                        t.update(n=round((im + 1) / sum(len(Ylms[ell]) for ell in ells) / 36))
                                    power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                    return power.ravel()
                                wmat_tmp[ell1, ell2] += my_map(f, kin)

                    if 'recompute' not in flags:
                        knorm = jnp.sqrt(sum(xx**2 for xx in kvec))

                        def f(kin):
                            Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                            for im12, (Yl1m1, Yl2m2) in enumerate(itertools.product(Ylms[ell1], Ylms[ell2])):
                                xi = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * rnorm * Yl1m1(*kvec) * Yl2m2(*[-kk for kk in kvec])).c2r()
                                # Typically takes ~ 2x the time to load all Qs than the above FFT
                                # Not great, but... recomputing 15 FFTs would have taken more time
                                for ell in Aell:
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im12])
                            power = []
                            for ill, ell in enumerate(ells):
                                Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                power.append(4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0))
                                del Aell[ell]
                            if pbar:
                                t.update(n=round((im12 + 1) / 36))
                            return jnp.concatenate(power)

                        wmat_tmp[ell1, ell2] = my_map(f, kin)

            wmat = jnp.zeros((len(ellsin),) + wmat_tmp[0, 0].shape, dtype=wmat_tmp[0, 0].dtype)
            coeff2 = {(0, 0): [(0, 1), (4, -7. / 18.)],
                      (0, 2): [(2, 1. / 2.), (4, -5. / 18.)],
                      (2, 2): [(4, 35. / 18.)]}
            coeff2[2, 0] = coeff2[0, 2]
            for illin, ellin in enumerate(ellsin):
                for ell1, ell2 in coeff2:
                    coeff = sum(coeff * (ell == ellin) for ell, coeff in coeff2[ell1, ell2])
                    wmat = wmat.at[illin].add(coeff * wmat_tmp[ell1, ell2])
            wmat = wmat.reshape(-1, wmat.shape[-1]).T

        else:
            raise NotImplementedError(f'theory los {theory_los} not implemented')

    observable = BinnedStatistic(x=[bin.xavg] * len(ells), value=[jnp.zeros_like(bin.xavg)] * len(ells), edges=[bin.edges] * len(ells),
                                 weights=[bin.nmodes] * len(ells), projs=ells)
    xin = np.mean(kin, axis=-1)
    theory = BinnedStatistic(x=[xin] * len(ellsin), value=[jnp.zeros_like(xin)] * len(ellsin), edges=[kin] * len(ellsin), projs=ellsin)
    wmat = WindowMatrix(observable, theory, wmat, attrs={'norm': norm})
    return wmat



def compute_mesh2_spectrum_mean(*meshs: RealMeshField | ComplexMeshField | MeshAttrs, theory: Callable | dict[Callable],
                                bin: BinMesh2Spectrum=None, los: str | np.ndarray='x') -> Spectrum2Poles:
    r"""
    Compute mean power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, ComplexMeshField, MeshAttrs
        Input mesh(s).
        A :class:`MeshAttrs` instance can be directly provided, in case the selection function is trivial (constant).

    theory : Callable, dict[Callable]
        Mean theory power spectrum. Either a callable (if ``los`` is an axis),
        or a dictionary of callables, with keys the multipole orders :math:`\ell`.
        Also possible to add wide-angle order :math:`n`, such that the key is the tuple :math:`(\ell, n)`.

    edges : np.ndarray, dict, default=None
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, array, default=None
        If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    power : Spectrum2Poles
    """
    meshs, autocorr = _format_meshs(*meshs)
    periodic = isinstance(meshs[0], MeshAttrs)
    if periodic:
        assert autocorr
        rdtype = float
        mattrs = meshs[0]
    else:
        rdtype = meshs[0].real.dtype
        mattrs = meshs[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod()
    rnorm = norm / mattrs.meshsize.prod(dtype=rdtype)

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
    attrs = dict(los=vlos if vlos is not None else los)
    ells = bin.ells

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    poles = theory
    kin = None
    theory_los = 'firstpoint'
    if isinstance(poles, tuple) and isinstance(poles[-1], str):
        theory_los = poles[-1]
        if len(poles) == 2: poles = poles[0]
        else: poles = poles[:-1]
    if isinstance(poles, tuple):
        kin, poles = poles
    if isinstance(poles, BinnedStatistic):
        kin, poles = jnp.append(poles._edges[0][..., 0], poles._edges[0][-1, 1]) if poles._edges[0] is not None else poles._x[0], {proj: poles.view(projs=proj) for proj in poles.projs}
    if isinstance(poles, list):
        poles = {ell: pole for ell, pole in zip((0, 2, 4), poles)}
    kvec = mattrs.kcoords(sparse=True)
    knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
    is_poles = not callable(poles)
    if is_poles:
        is_callable = all(callable(pole) for pole in poles.values())
        if not is_callable:
            from .utils import Interpolator1D
            interp = Interpolator1D(kin, knorm, edges=len(kin) == len(poles[0]) + 1)

    def get_theory(ell=None, pole=None):
        if pole is None:
            pole = poles[ell]
        if is_callable:
            return pole(knorm)
        else:
            return interp(pole)

    if vlos is not None:

        if not periodic:

            for imesh, mesh in enumerate(meshs[:1 if autocorr else 2]):
                meshs[imesh] = _2c(mesh)
            if autocorr:
                meshs[1] = meshs[0]

            Q = _2r(meshs[0] * meshs[1].conj())
        else:
            Q = None

        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)

        pkvec = theory

        if is_poles:
            def pkvec(*args):
                return sum(get_theory(ell) * get_legendre(ell)(mu) for ell in poles)

        Aell = mattrs.create(kind='complex').apply(lambda value, kvec: pkvec(kvec) * rnorm, kind='wavenumber')
        if Q is not None: Aell = _2c(Q * _2r(Aell))
        else: Aell *= mattrs.meshsize.prod(dtype=rdtype)

        power, power_zero = [], []
        for ell in ells:
            leg = get_legendre(ell)(mu)
            odd = ell % 2
            if odd: leg += get_legendre(ell)(-mu)
            power.append((2 * ell + 1) / (1 + odd) * bin(Aell * leg))
            power_zero.append(0.)
            if ell == 0:
                power_zero[-1] += _get_zero(Aell)

        return Spectrum2Poles(bin.xavg, num=power, nmodes=bin.nmodes, edges=bin.edges, ells=ells, norm=norm, num_zero=power_zero, attrs=mattrs)

    else:
        poles = {ell if isinstance(ell, tuple) else (ell, 0): pole for ell, pole in poles.items()} # wide-angle = 0 as a default

        ellsin = [mode[0] for mode in poles]

        if swap: meshs = meshs[::-1]

        if periodic:
            meshs = [mattrs.create(kind='real', fill=1.)]

        rmesh1 = _2r(meshs[0])
        rmesh2 = rmesh1 if autocorr else _2r(meshs[1])
        A0 = _2c(meshs[0] if autocorr else meshs[1])
        del meshs

        # The real-space grid
        xhat = mattrs.xcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in xhat))

        shat = mattrs.xcoords(kind='separation', sparse=True)
        snorm = jnp.sqrt(sum(xx**2 for xx in shat))

        # The Fourier-space grid
        khat = A0.coords(sparse=True)

        Ylms = {ell: [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}

        power, power_zero = [], []
        for ell in ells:
            Aell = 0.
            for Ylm in Ylms[ell]:
                Q = 0.
                if theory_los == 'firstpoint':
                    for ell1, wa1 in poles:
                        kernel_ell1 = get_theory((ell1, wa1)) * rnorm
                        for Yl1m1 in Ylms[ell1]:
                            xi = mattrs.create(kind='complex', fill=kernel_ell1 * Yl1m1(*khat)).c2r() * snorm**wa1
                            Q += (4. * np.pi) / (2 * ell1 + 1) * xi * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xhat) * Yl1m1(*xhat)).conj() * A0)

                elif theory_los == 'local':
                    coeff2 = {(0, 0): [(0, 1), (4, -7. / 18.)],
                              (0, 2): [(2, 1. / 2.), (4, -5. / 18.)],
                              (2, 2): [(4, 35. / 18.)]}
                    coeff2[2, 0] = coeff2[0, 2]
                    for ell1, ell2 in itertools.product((0, 2), (0, 2)):
                        if is_callable:
                            kernel_ell2 = sum(coeff * get_theory((ell, 0)) for ell, coeff in coeff2[ell1, ell2])
                        else:
                            pole = sum(coeff * poles[ell, 0] for ell, coeff in coeff2[ell1, ell2])
                            kernel_ell2 = get_theory(pole=pole)
                        kernel_ell2 = kernel_ell2 * rnorm
                        for Yl1m1, Yl2m2 in itertools.product(Ylms[ell1], Ylms[ell2]):
                            xi = mattrs.create(kind='complex', fill=kernel_ell2 * Yl1m1(*khat) * Yl2m2(*[-kk for kk in khat])).c2r()
                            Q += (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * xi * _2r(_2c(rmesh1 * Ylm(*xhat) * Yl1m1(*xhat)).conj() * _2c(rmesh2 * Yl2m2(*xhat)))

                else:
                    raise NotImplementedError(f'theory los {theory_los} not implemented')

                Aell += _2c(Q) * Ylm(*khat)
            # Project on to 1d k-basis (averaging over mu=[-1, 1])
            power.append(4. * jnp.pi * bin(Aell, antisymmetric=bool(ell % 2)))
            power_zero.append(0.)
            if ell == 0:
                power_zero[-1] += 4. * jnp.pi * _get_zero(Aell)

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        power, power_zero = jnp.array(power), jnp.array(power_zero)
        if swap: power, power_zero = power.conj(), power_zero.conj()
        # Format the power results into :class:`Spectrum2Poles` instance
        return Spectrum2Poles(bin.xavg, num=power, nmodes=bin.nmodes, edges=bin.edges, ells=ells, norm=norm,
                              num_zero=power_zero, attrs=attrs)
