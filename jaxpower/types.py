import numpy as np
import jax
from jax import numpy as jnp

from lsstypes import (Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles, Mesh3CorrelationPole, Mesh3CorrelationPoles,
                      ObservableLeaf, ObservableTree, WindowMatrix, CovarianceMatrix, read, write)
from lsstypes.base import from_state, _edges_names, register_type, _check_data_names, _check_data_shapes
from lsstypes.utils import plotter, my_ones_like, my_zeros_like


def make_leaf_pytree(cls):

    def tree_flatten(self):
        children = list(self._data[name] for name in self._coords_names + self._values_names)
        edges_names = []
        for name in _edges_names(self._coords_names):
            if name in self._data:
                children.append(self._data[name])
                edges_names.append(name)
        aux_data = {name: getattr(self, name) for name in ['_attrs', '_meta', '_coords_names', '_values_names']}
        aux_data['_edges_names'] = edges_names
        return tuple(children), aux_data

    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        aux_data = dict(aux_data)
        edges_names = aux_data.pop('_edges_names')
        new.__dict__.update(aux_data)
        new._data = {name: child for name, child in zip(new._coords_names + new._values_names + edges_names, children)}
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
Mesh3CorrelationPole = make_leaf_pytree(Mesh3CorrelationPole)
Mesh3CorrelationPoles = make_tree_pytree(Mesh3CorrelationPoles)


WindowMatrix = make_window_pytree(WindowMatrix)
CovarianceMatrix = make_covariance_pytree(CovarianceMatrix)


import operator
import functools

prod = functools.partial(functools.reduce, operator.mul)


def _pole_transform(xout, pole, label, mode='forward'):
    # for mode = 'forward': xin is s, xout is k
    # for mode = 'backward': xin is k, xout is s
    from .utils import BesselIntegral

    ell = label['ells']
    multidim = isinstance(ell, (tuple, list))  # 3-point, sugiyama basis
    if not multidim: ell = [ell]

    def unravel(leaf):
        if hasattr(leaf, 'unravel') and leaf.is_raveled:
            return leaf.unravel()
        return leaf

    if multidim:
        pole = unravel(pole)

    leaf = None
    if isinstance(xout, ObservableLeaf):
        leaf = xout
    elif isinstance(xout, ObservableTree):
        leaf = xout.get(**label)
    if leaf is not None:
        xout = list(leaf.coords().values())
        if multidim and len(xout) == 1:
            xout = xout[0]
    index_out = None
    if not isinstance(xout, (tuple, list)):
        if multidim:
            assert xout.ndim == 2
            xout_unraveled, index_out = [], []
            for i in range(xout.shape[1]):
                unique, index = np.unique(xout[:, i], return_inverse=True)
                xout_unraveled.append(unique)
                index_out.append(index)
        else:
            assert xout.ndim == 1
            xout_unraveled = [xout]
        xout = [xout]
    else:
        xout_unraveled = xout
    xin_unraveled = list(pole.edges(default=None).values())
    is_edges = all(xx is not None for xx in xin_unraveled)
    if not is_edges:
        xin_unraveled = list(pole.coords().values())

    # sugiyama basis: bessel transform on the first 2 ells
    weights = [BesselIntegral(xxin, xxout, ell=ell, method='exact' if is_edges else 'rect', mode=mode, edges=is_edges, volume=False).w for xxin, xxout, ell in zip(xin_unraveled, xout_unraveled, ell[:len(xout_unraveled)], strict=True)]
    transformed = pole.value()
    if 'volume' in pole.values():
        volume = pole.values('volume')
        transformed = jnp.where(volume == 0., 0., transformed)
    else:
        volume = 1.
    value = volume * transformed
    for idim, ww in enumerate(weights):
        # tensordot would move idim to first axis, so let's simplify
        value = np.moveaxis(value, idim, -1)
        value = np.tensordot(value, ww.T, axes=(-1, 0))
        value = np.moveaxis(value, -1, idim)
    if index_out is not None:
        value = value[tuple(index_out)]
    if leaf is not None:
        return leaf.clone(value=value)
    base_coord = 'k' if mode == 'forward' else 's'
    if multidim and index_out is None:
        coords = [f'{base_coord}{idim:d}' for idim in range(len(xout_unraveled))]
    else:
        coords = [base_coord]
    return ObservableLeaf(**dict(zip(coords, xout)), value=value, coords=coords)


def observable_correlation_to_spectrum(correlation, k):
    """
    Convert the correlation function multipoles to power spectrum multipoles.

    Parameters
    ----------
    k : array-like or Particle2SpectrumPoles
        Wavenumber bin centers or :class:`Particle2SpectrumPoles` instance.

    Returns
    -------
    Particle2SpectrumPoles
    """
    pole_to_spectrum = functools.partial(_pole_transform, k, mode='forward')
    if isinstance(correlation, ObservableTree):  # multipoles
        branches = []
        for label in correlation.labels():
            branches.append(pole_to_spectrum(correlation.get(**label), label))
        return ObservableTree(branches, **correlation.labels(return_type='unflatten'))
    return pole_to_spectrum(correlation)


def observable_spectrum_to_correlation(spectrum, s):
    """
    Convert the power spectrum multipoles to correlation function multipoles.

    Parameters
    ----------
    s : array-like or Particle2CorrelationPoles
        Separation bin centers or :class:`Particle2CorrelationPoles` instance.

    Returns
    -------
    Particle2CorrelationPoles
    """
    pole_to_correlation = functools.partial(_pole_transform, s, mode='backward')
    if isinstance(spectrum, ObservableTree):  # multipoles
        branches = []
        for label in spectrum.labels():
            branches.append(pole_to_correlation(spectrum.get(**label), label))
        return ObservableTree(branches, **spectrum.labels(return_type='unflatten'))
    return pole_to_correlation(spectrum)


Mesh2SpectrumPole.to_correlation = observable_spectrum_to_correlation
Mesh2SpectrumPoles.to_correlation = observable_spectrum_to_correlation


class Particle2SpectrumPole(Mesh2SpectrumPole): pass


class Particle2SpectrumPoles(Mesh2SpectrumPoles): pass


class Particle2CorrelationPole(Mesh2CorrelationPole):

    def to_spectrum(self, k):
        """
        Convert the correlation function multipole to power spectrum multipole.

        Parameters
        ----------
        k : array-like or Particle2SpectrumPoles
            Wavenumber bin centers or :class:`Particle2SpectrumPoles` instance.

        Returns
        -------
        Particle2SpectrumPole
        """
        return observable_correlation_to_spectrum(self, k)


class Particle2CorrelationPoles(Mesh2CorrelationPoles):

    def to_spectrum(self, k):
        """
        Convert the correlation function multipoles to power spectrum multipoles.

        Parameters
        ----------
        k : array-like or Particle2SpectrumPoles
            Wavenumber bin centers or :class:`Particle2SpectrumPoles` instance.

        Returns
        -------
        Particle2SpectrumPoles
        """
        return observable_correlation_to_spectrum(self, k)



@register_type
class Angular2Spectrum(Mesh2SpectrumPole):
    r"""
    Container for an angular power spectrum :math:`C_\ell`.

    Stores the binned (bandpower) angular power spectrum, including shot noise, normalization, and mode counts.

    Parameters
    ----------
    ell : array-like
        Bin centers for multipole :math:`\ell`.
    ell_edges : array-like
        Bin edges for multipole :math:`\ell`.
    num_raw : array-like
        Raw angular power spectrum measurements.
    num_shotnoise : array-like, optional
        Shot noise contribution (default: zeros).
    norm : array-like, optional
        Normalization factor (default: ones).
    nmodes : array-like, optional
        Number of modes :math:`\sum_{\ell \in b} (2\ell + 1)` per bin (default: ones).
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'angular2_spectrum'

    def __init__(self, ell=None, ell_edges=None, num_raw=None, num_shotnoise=None, norm=None, nmodes=None, attrs=None, **kwargs):
        kw = dict(ell=ell, ell_edges=ell_edges)
        if ell_edges is None: kw.pop('ell_edges')
        self.__pre_init__(**kw, coords=['ell'], attrs=attrs)
        if num_shotnoise is None: num_shotnoise = my_zeros_like(num_raw)
        if norm is None: norm = my_ones_like(num_raw)
        if nmodes is None: nmodes = my_ones_like(num_raw, dtype='i4')
        self._values_names = ['value', 'num_shotnoise', 'norm', 'nmodes']
        for name in list(kwargs):
            if name not in self._values_names:
                raise ValueError(f'{name} not unknown')
        self._update(num_raw=num_raw, num_shotnoise=num_shotnoise, norm=norm, nmodes=nmodes, **kwargs)
        _check_data_names(self)
        _check_data_shapes(self)

    def _plabel(self, name):
        if name == 'ell':
            return r'$\ell$'
        if name == 'value':
            return r'$C_\ell$'
        return None

    @plotter
    def plot(self, fig=None, **kwargs):
        r"""
        Plot the angular power spectrum.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.
        fn : str, Path, default=None
            Optionally, path where to save figure.
        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.
        show : bool, default=False
            If ``True``, show figure.
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        ax.plot(self.ell, self.value, **kwargs)
        ax.set_xlabel(self._plabel('ell'))
        ax.set_ylabel(self._plabel('value'))
        return fig


Angular2Spectrum = make_leaf_pytree(Angular2Spectrum)


@register_type
class Angular3Spectrum(Mesh3SpectrumPole):
    r"""
    Container for an angular bispectrum :math:`b_{\ell_1 \ell_2 \ell_3}`, binned in :math:`\ell`-bands.

    Triplets of bands are raveled: ``ell`` has shape ``(nbins, 3)`` (and ``ell_edges`` shape ``(nbins, 3, 2)``),
    since the triangle and parity conditions leave a subset of the band triplets.

    Parameters
    ----------
    ell : array-like
        Band centers, shape ``(nbins, 3)``.
    ell_edges : array-like
        Band edges, shape ``(nbins, 3, 2)``.
    num_raw : array-like
        Raw bispectrum measurements.
    num_shotnoise : array-like, optional
        Shot noise contribution (default: zeros).
    norm : array-like, optional
        Normalization factor (default: ones).
    nmodes : array-like, optional
        Number of triangles :math:`N_{i_1 i_2 i_3} = \sum_{\ell_1 \ell_2 \ell_3} h^2_{\ell_1 \ell_2 \ell_3}` per bin.
    attrs : dict, optional
        Additional attributes.
    """
    _name = 'angular3_spectrum'

    def __init__(self, ell=None, ell_edges=None, num_raw=None, num_shotnoise=None, norm=None, nmodes=None, attrs=None):
        kw = dict(ell=ell, ell_edges=ell_edges)
        if ell_edges is None: kw.pop('ell_edges')
        coords = ['ell']
        if isinstance(ell, tuple):
            kw = {f'ell{idim + 1:d}': coord for idim, coord in enumerate(ell)}
            coords = list(kw)
            if ell_edges is not None:
                kw.update({f'ell{idim + 1:d}_edges': edge for idim, edge in enumerate(ell_edges)})
        self.__pre_init__(**kw, coords=coords, attrs=attrs)
        if num_shotnoise is None: num_shotnoise = my_zeros_like(num_raw)
        if norm is None: norm = my_ones_like(num_raw)
        if nmodes is None: nmodes = my_ones_like(num_raw, dtype='f8')
        self._values_names = ['value', 'num_shotnoise', 'norm', 'nmodes']
        self._update(**kw, num_raw=num_raw, num_shotnoise=num_shotnoise, norm=norm, nmodes=nmodes)
        _check_data_names(self)
        _check_data_shapes(self)
        self._meta['basis'] = ''

    def _plabel(self, name):
        if name.startswith('ell'):
            return r'$\ell_{}$'.format(name[3:]) if name[3:] else r'$\ell$'
        if name == 'value':
            return r'$b_{\ell_1 \ell_2 \ell_3}$'
        return None


Angular3Spectrum = make_leaf_pytree(Angular3Spectrum)


class Particle3SpectrumPole(Mesh3SpectrumPole): pass


class Particle3SpectrumPoles(Mesh3SpectrumPoles): pass


class Particle3CorrelationPole(Mesh3CorrelationPole):

    def to_spectrum(self, k):
        """
        Convert the correlation function multipole to power spectrum multipole.

        Parameters
        ----------
        k : array-like or Particle3SpectrumPoles
            Wavenumber bin centers or :class:`Particle3SpectrumPoles` instance.

        Returns
        -------
        Particle3SpectrumPole
        """
        return observable_correlation_to_spectrum(self, k)


class Particle3CorrelationPoles(Mesh3CorrelationPoles):

    def to_spectrum(self, k):
        """
        Convert the correlation function multipoles to power spectrum multipoles.

        Parameters
        ----------
        k : array-like or Particle3SpectrumPoles
            Wavenumber bin centers or :class:`Particle3SpectrumPoles` instance.

        Returns
        -------
        Particle3SpectrumPoles
        """
        return observable_correlation_to_spectrum(self, k)