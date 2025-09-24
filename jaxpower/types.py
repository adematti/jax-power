import jax
import numpy as np

from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles, Mesh3SpectrumPole, Mesh3SpectrumPoles, ObservableLeaf, ObservableTree, WindowMatrix, CovarianceMatrix, read, write


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