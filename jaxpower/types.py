import jax
import numpy as np

from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles, ObservableLeaf, ObservableTree, WindowMatrix, CovarianceMatrix
#from lsstypes import Mesh2SpectrumPole as _Mesh2SpectrumPole, Mesh2SpectrumPoles as _Mesh2SpectrumPoles
from lsstypes.base import register_type
from lsstypes.utils import plotter




@register_type
class Mesh2CorrelationPole(ObservableLeaf):
    r"""
    Container for a correlation function multipole :math:`\xi_\ell(s)`.

    Stores the binned correlation function for a given multipole order :math:`\ell`, including normalization and number of modes.

    Parameters
    ----------
    s : array-like
        Bin centers for separation :math:`s`.
    s_edges : array-like
        Bin edges for separation :math:`s`.
    value : array-like
        Correlation function multipole values for each bin.
    nmodes : array-like, optional
        (Isotropic-average of) RR (random-random) pair counts for each bin (default: ones).
    norm : array-like, optional
        Normalization factor (default: ones).
    ell : int, optional
        Multipole order :math:`\ell`.
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh2correlationpole'

    def __init__(self, s=None, s_edges=None, num_raw=None, norm=None, nmodes=None, ell=None, attrs=None, **kwargs):
        kw = dict(s=s, s_edges=s_edges)
        if s_edges is None: kw.pop('s_edges')
        self.__pre_init__(**kw, coords=['s'], attrs=attrs)
        if norm is None: norm = np.ones_like(num_raw)
        if nmodes is None: nmodes = np.ones_like(num_raw, dtype='i4')
        self._values_names = ['value', 'num_shotnoise', 'norm', 'nmodes']
        for name in list(kwargs):
            if name in self._values_names: pass
            elif name in ['volume']: self._values_names.append(name)
            else: raise ValueError('{name} not unknown')
        self._update(num_raw=num_raw, norm=norm, nmodes=nmodes, **kwargs)
        if ell is not None:
            self._meta['ell'] = ell

    def _update(self, **kwargs):
        for name in list(kwargs):
            if name in ['s', 's_edges'] + self._values_names:
                self._data[name] = kwargs.pop(name)
        for name in list(kwargs):
            if name in ['num_raw']:
                self._data['value'] = kwargs.pop(name) / self.norm
        if kwargs:
            raise ValueError(f'{kwargs} unknown')

    def _binweight(self, name=None):
        # weight, normalized
        if name == 'nmodes':
            return False, False
        return self.nmodes, True

    @classmethod
    def _sumweight(cls, observables, name=None):
        if name is None or name in ['value']:
            s = sum(observable.norm for observable in observables)
            return [observable.norm / s for observable in observables]
        if name in ['nmodes']:
            return [1. / len(observables)] * len(observables)
        return [1] * len(observables)  # just sum

    def _plabel(self, name):
        if name == 's':
            return r'$s$ [$\mathrm{Mpc}/h$]'
        if name == 'value':
            return r'$\xi_\ell(s)$'
        return None

    @plotter
    def plot(self, fig=None, **kwargs):
        r"""
        Plot a correlation function multipole.

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
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ax.plot(self.s, self.s**2 * self.value(), **kwargs)
        ax.set_xlabel(self._plabel('s'))
        ax.set_ylabel(r'$s^2 \xi_\ell(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig


@register_type
class Mesh2CorrelationPoles(ObservableTree):
    r"""
    Container for multiple correlation function multipoles :math:`\xi_\ell(s)`.

    Stores a collection of `Mesh2CorrelationPole` objects for different multipole orders :math:`\ell`.

    Parameters
    ----------
    poles : list of Mesh2CorrelationPole
        List of correlation function multipole objects.
    ells : list of int, optional
        Multipole orders :math:`\ell` for each pole (default: inferred from `poles`).
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh2correlationpoles'

    def __init__(self, poles, ells=None, attrs=None):
        """Initialize correlattion function multipoles."""
        if ells is None: ells = [pole.ell for pole in poles]
        super().__init__(poles, ells=ells, attrs=attrs)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot the correlattion function multipoles.

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
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ell in self.ells:
            pole = self.get(ell)
            pole.plot(fig=ax, label=rf'$\ell = {ell:d}$')
        ax.legend(frameon=False)
        return fig


@register_type
class Mesh3SpectrumPole(ObservableLeaf):
    """
    Container for a bispectrum multipole :math:`B_\ell(k)`.

    Stores the binned bispectrum for a given multipole order :math:`\ell`, including shot noise, normalization, and mode counts.

    Parameters
    ----------
    k : array-like
        Bin centers for wavenumber :math:`k`.
    k_edges : array-like
        Bin edges for wavenumber :math:`k`.
    num_raw : array-like
        Raw power spectrum measurements.
    num_shotnoise : array-like, optional
        Shot noise contribution (default: zeros).
    norm : array-like, optional
        Normalization factor (default: ones).
    nmodes : array-like, optional
        Number of modes per bin (default: ones).
    ell : tuple, int, optional
        Multipole order :math:`\ell`.
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh3spectrumpole'

    def __init__(self, k=None, k_edges=None, num_raw=None, num_shotnoise=None, norm=None, nmodes=None, ell=None, basis='', attrs=None):
        kw = dict(k=k, k_edges=k_edges)
        if k_edges is None: kw.pop('k_edges')
        self.__pre_init__(**kw, coords=['k'], attrs=attrs)
        if num_shotnoise is None: num_shotnoise = np.zeros_like(num_raw)
        if norm is None: norm = np.ones_like(num_raw)
        if nmodes is None: nmodes = np.ones_like(num_raw, dtype='i4')
        self._update(**kw, num_raw=num_raw, num_shotnoise=num_shotnoise, norm=norm, nmodes=nmodes)
        if ell is not None:
            self._meta['ell'] = ell
        self._meta['basis'] = basis

    def _update(self, **kwargs):
        self._values_names = ['value', 'num_shotnoise', 'norm', 'nmodes']
        for name in list(kwargs):
            if name in ['k', 'k_edges'] + self._values_names:
                self._data[name] = kwargs.pop(name)
        for name in list(kwargs):
            if name in ['num_raw']:
                self._data['value'] = (kwargs.pop(name) - self.num_shotnoise) / self.norm
        if kwargs:
            raise ValueError(f'Could not interpret arguments {kwargs}')

    def _plabel(self, name):
        if name == 'k':
            if 'scoccimarro' in self.basis:
                return r'$k_1, k_2, k_3$ [$h/\mathrm{Mpc}$]'
            return r'$k_1, k_2$ [$h/\mathrm{Mpc}$]'
        if name == 'value':
            if 'scoccimarro' in self.basis:
                return r'$B_{\ell_3}(k_3)$ [$(\mathrm{Mpc}/h)^{6}$]'
            return r'$B_{\ell_1, \ell_2, \ell_3}(k_1, k_2)$ [$(\mathrm{Mpc}/h)^{6}$]'
        return None

    def _binweight(self, name=None):
        # weight, normalized
        if name == 'nmodes':
            return False, False
        return self.nmodes, True

    @classmethod
    def _sumweight(cls, observables, name=None):
        if name is None or name in ['value']:
            s = sum(observable.norm for observable in observables)
            return [observable.norm / s for observable in observables]
        if name in ['nmodes']:
            return None  # keep the first nmodes
        return [1] * len(observables)  # just sum

    @plotter
    def plot(self, fig=None, **kwargs):
        r"""
        Plot bispectrum multipole.

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
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ax.plot(np.arange(len(self.k)), self.k.prod(axis=-1) * self.value(), **kwargs)
        ax.set_xlabel('bin index')
        if 'scoccimarro' in self.basis:
            ax.set_ylabel(r'$k_1 k_2 k_3 B_{\ell_1 \ell_2 \ell_3}(k_1, k_2, k_3)$ [$(\mathrm{Mpc}/h)^{6}$]')
        else:
            ax.set_ylabel(r'$k_1 k_2 B_{\ell_1 \ell_2 \ell_3}(k_1, k_2)$ [$(\mathrm{Mpc}/h)^{4}$]')
        return fig


@register_type
class Mesh3SpectrumPoles(ObservableTree):
    """
    Container for multiple bispectrum multipoles :math:`B_\ell(k)`.

    Stores a collection of `Mesh3SpectrumPole` objects for different multipole orders :math:`\ell`, allowing joint analysis and plotting.

    Parameters
    ----------
    poles : list of Mesh3SpectrumPole
        List of bispectrum multipole objects.

    ells : list of int or tuples, optional
        Multipole orders :math:`\ell` for each pole (default: inferred from `poles`).

    attrs : dict, optional
        Additional attributes.
    """
    _name = 'mesh3spectrumpoles'

    def __init__(self, poles, ells=None, attrs=None):
        """Initialize bispectrum multipoles."""
        if ells is None: ells = [pole.ell for pole in poles]
        super().__init__(poles, ells=ells, attrs=attrs)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot the bispectrum multipoles.

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
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ell in self.ells:
            pole = self.get(ell)
            pole.plot(fig=ax, label=rf'$\ell = {ell}$')
        ax.legend(frameon=False)
        return fig



def make_leaf_pytree(cls):

    def tree_flatten(self):
        children = tuple(self._data[name] for name in self._coords_names + self._values_names)
        aux_data = {name: getattr(self, name) for name in ['_attrs', '_meta', '_coords_names', '_values_names']}
        return children, aux_data

    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        new._data = {name: child for name, child in zip(new._coords_names + new._values_names, children)}
        return new

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = classmethod(tree_unflatten)
    return jax.tree_util.register_pytree_node_class(cls)


def make_tree_pytree(cls):

    def tree_flatten(self):
        children = tuple(self._branches)
        aux_data = {name: getattr(self, name) for name in ['_attrs', '_meta', '_labels', '_strlabels']}
        return children, aux_data

    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        new._branches = list(children)
        return new

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = classmethod(tree_unflatten)
    return jax.tree_util.register_pytree_node_class(cls)


def make_window_pytree(cls):

    def tree_flatten(self):
        children = (self._value, self._observable, self._theory)
        aux_data = {name: getattr(self, name) for name in ['_attrs']}
        return children, aux_data

    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        new._value, new._observable, new._theory = tuple(children)
        return new

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = classmethod(tree_unflatten)
    return jax.tree_util.register_pytree_node_class(cls)


def make_covariance_pytree(cls):

    def tree_flatten(self):
        children = (self._value, self._observable)
        aux_data = {name: getattr(self, name) for name in ['_attrs']}
        return children, aux_data

    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        new._value, new._observable = tuple(children)
        return new

    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = classmethod(tree_unflatten)
    return jax.tree_util.register_pytree_node_class(cls)


ObservableLeaf = make_leaf_pytree(ObservableLeaf)
ObservableTree = make_tree_pytree(ObservableTree)

Mesh2SpectrumPole = make_leaf_pytree(Mesh2SpectrumPole)
Mesh2SpectrumPoles = make_tree_pytree(Mesh2SpectrumPoles)
Mesh2CorrelationPole = make_leaf_pytree(Mesh2CorrelationPole)
Mesh2CorrelationPoles = make_tree_pytree(Mesh2CorrelationPoles)
Mesh3SpectrumPole = make_leaf_pytree(Mesh3SpectrumPole)
Mesh3SpectrumPoles = make_tree_pytree(Mesh3SpectrumPoles)

WindowMatrix = make_window_pytree(WindowMatrix)
CovarianceMatrix = make_covariance_pytree(CovarianceMatrix)