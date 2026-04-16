import jax
from jax import numpy as jnp

from lsstypes import (Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles, Mesh3CorrelationPole, Mesh3CorrelationPoles,
                      ObservableLeaf, ObservableTree, WindowMatrix, CovarianceMatrix, read, write)
from lsstypes.base import from_state, _edges_names, register_type


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
    ndim = isinstance(ell, (tuple, list))  # 3-point

    def unravel(leaf):
        if hasattr(leaf, 'unravel'):
            return leaf.unravel()
        return unravel

    if ndim:
        pole = unravel(pole)

    leaf = None
    if isinstance(xout, ObservableLeaf):
        leaf = xout
    elif isinstance(xout, ObservableTree):
        leaf = xout.get(**label)
    if leaf is not None:
        if ndim:
            leaf = unravel(leaf)
        xout = list(leaf.coords().values())

    xout = xout if isinstance(xout, (tuple, list)) else [xout]
    xin = list(pole.coords().values())
    w = [BesselIntegral(xxin, xxout, ell=ell, method='rect', mode=mode, edges=False, volume=False).w for xxin, xxout in zip(xin, xout, strict=True)]
    w = prod(jnp.meshgrid(w, indexing='ij', sparse=False))
    # self.weights = volume factor
    transformed = pole.value()
    if 'volume' in pole.values():
        volume = pole.values('volume')
        transformed = jnp.where(volume == 0., 0., transformed)
    else:
        volume = 1.
    value = jnp.sum(w * volume * transformed)
    if leaf is not None:
        return leaf.clone(value=value)
    coords = list(pole.coords())
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



class Particle3SpectrumPole(Mesh2SpectrumPole): pass


class Particle3SpectrumPoles(Mesh2SpectrumPoles): pass


class Particle3CorrelationPole(Mesh2CorrelationPole):

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


class Particle3CorrelationPoles(Mesh2CorrelationPoles):

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