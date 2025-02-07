import os
import operator
from functools import partial
from collections.abc import Callable
from typing import Any

import numpy as np
import jax
from jax import numpy as jnp
from dataclasses import dataclass, field, fields, asdict
from . import resamplers, utils


class staticarray(np.ndarray):
    """
    Class overriding a numpy array, so that it can be passed in ``meta_fields``.
    Otherwise an error is raised whenever a ``jax.tree.map`` is fed with two different dataclass instances.
    """

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        # Finally, we must return the newly created object:
        return obj

    def __setitem__(self, key, value):
        raise ValueError('staticarray is static')

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        if not isinstance(other, np.ndarray):
            return False
        return other.shape == self.shape and np.all(self.view(np.ndarray) == np.asarray(other))

    @classmethod
    def fill(cls, fill: float | np.ndarray, shape: int | tuple, **kwargs):
        """
        Create a :class:`staticarray` of shape ``shape``, filled with ``fill``,
        which can be a numpy array, as long as its shape is brodcastable with ``shape``.
        """
        fill = np.array(fill)
        toret = np.empty_like(fill, shape=shape, **kwargs)
        toret[...] = fill
        return cls(toret)


def _get_ndim(*args):
    ndim = 3
    for value in args:
        try: ndim = len(value)
        except: pass
    return ndim



def fftfreq(shape: tuple, kind: str='wavenumber', sparse: bool | None=None,
            hermitian: bool=False, spacing: np.ndarray | float=1.):
    r"""
    Return mesh frequencies.

    Parameters
    ----------
    kind : str, default='wavenumber'
        Either 'circular' (in :math:`(-\pi, \pi)`) or 'wavenumber' (in ``spacing`` units).

    sparse : bool, default=None
        If ``None``, return a tuple of 1D-arrays.
        If ``False``, return a tuple of broadcastable arrays.
        If ``True``, return a tuple of broadcastable arrays of same shape.

    hermitian : bool, default=False
        If ``True``, last axis is of size ``shape[-1] // 2 + 1``.

    spacing : float, default=1.
        Sampling spacing, typically ``cellsize``.

    Returns
    -------
    freqs : tuple
    """
    ndim = len(shape)
    spacing = staticarray.fill(spacing, ndim)
    if kind is None:
        toret = [jnp.arange(s) for s in shape]
    else:
        if kind == 'circular':
            period = (2 * jnp.pi,) * ndim
        else:  # wavenumber
            period = 2 * jnp.pi / spacing
        toret = []
        for axis, s in enumerate(shape):
            k = (jnp.fft.rfftfreq if axis == ndim - 1 and hermitian else jnp.fft.fftfreq)(s) * period[axis]
            toret.append(k)
    if sparse is None:
        return tuple(toret)
    return jnp.meshgrid(*toret, sparse=sparse, indexing='ij')


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class MeshAttrs(object):

    meshsize: staticarray = None
    boxsize: staticarray = None
    boxcenter: staticarray = 0.

    def __init__(self, meshsize=None, boxsize=None, boxcenter=0.):
        state = dict(meshsize=meshsize, boxsize=boxsize, boxcenter=boxcenter)
        ndim = _get_ndim(*[meshsize, boxsize, boxcenter])
        if meshsize is not None:  # meshsize may not be provided (e.g. for particles) and it is fine
            meshsize = staticarray.fill(meshsize, ndim, dtype='i4')
            if boxsize is None: boxsize = 1. * meshsize
        if boxsize is not None: boxsize = staticarray.fill(boxsize, ndim, dtype=float)
        if boxcenter is not None: boxcenter = staticarray.fill(boxcenter, ndim, dtype=float)
        self.__dict__.update(meshsize=meshsize, boxsize=boxsize, boxcenter=boxcenter)

    def tree_flatten(self):
        return tuple(), asdict(self)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        return new

    # For mapping
    def __getitem__(self, key):
            return getattr(self, key)

    def keys(self):
        return self.__annotations__.keys()

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = asdict(self)
        state.update(kwargs)
        return self.__class__(**state)

    @property
    def cellsize(self):
        return self.boxsize / self.meshsize

    @property
    def kfun(self):
        """Fundamental wavenumber."""
        return 2. * np.pi / self.boxsize

    @property
    def knyq(self):
        """Nyquist wavenumber."""
        return np.pi * self.meshsize / self.boxsize

    def xcoords(self, kind: str='position', sparse: bool | None=None):
        """
        Return mesh spatial coordinates.

        Parameters
        ----------
        kind : str, default='position'.
            Either 'index' (index on mesh) or 'position' (spatial coordinates).

        sparse : bool, default=None
            If ``None``, return a tuple of 1D-arrays.
            If ``False``, return a tuple of broadcastable arrays.
            If ``True``, return a tuple of broadcastable arrays of same shape, :attr:`shape`.

        Returns
        -------
        coords : tuple
        """
        toret = [jnp.arange(s) for s in self.meshsize]
        if kind != 'index':
            toret = [idx * cell + offset for idx, cell, offset in zip(toret, self.cellsize, self.boxcenter - self.boxsize / 2.)]
        if sparse is None:
            return tuple(toret)
        return jnp.meshgrid(*toret, sparse=sparse, indexing='ij')

    def kcoords(self, kind: str='wavenumber', sparse: bool | None=None, hermitian: bool=False):
        r"""
        Return mesh Fourier coordinates.

        Parameters
        ----------
        kind : str, default='wavenumber'
            Either 'circular' (in :math:`(-\pi, \pi)`) or 'wavenumber' (in ``spacing`` units).

        sparse : bool, default=None
            If ``None``, return a tuple of 1D-arrays.
            If ``False``, return a tuple of broadcastable arrays.
            If ``True``, return a tuple of broadcastable arrays of same shape.

        hermitian : bool, default=False
        If ``True``, last axis is of size ``shape[-1] // 2 + 1``.

        Returns
        -------
        coords : tuple
        """
        return fftfreq(self.meshsize, kind=kind, sparse=sparse, hermitian=hermitian, spacing=self.boxsize / self.meshsize)

    def create(self, kind: str='real', fill: float=None, dtype=None):
        """
        Create mesh with these given attributes.

        Parameters
        ----------
        kind : str, default='real'
            Type of mesh to return, 'real', 'complex', or 'hermitian_complex'.

        fill : float, complex, default=None
            Optionally, value to fill mesh with.
            Empty as a default.

        dtype : DTypeLike, default=None
            Data type, float or complex.

        Returns
        -------
        mesh : New mesh.
        """
        kind = {'real': RealMeshField, 'complex': ComplexMeshField, 'hermitian_complex': HermitianComplexMeshField}.get(kind, kind)
        name = kind.__name__.lower()
        shape = tuple(self.meshsize)
        if 'hermitian' in name:
            shape = shape[:-1] + (shape[-1] // 2 + 1,)
        if dtype is None:
            dtype = complex if 'complex' in name else float
        if fill is None:
            value = jnp.empty(shape, dtype=dtype)
        else:
            value = jnp.full(shape, fill, dtype=dtype)
        return kind(value, **self)


# Note: I couldn't make it work properly with jax dataclass registration as tree_unflatten would pass all fields to __init__, which I don't want
@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class BaseMeshField(object):

    """Representation of a N-dim field as a N-dim array."""

    value: jax.Array = field(repr=False)
    attrs: MeshAttrs = None

    def __init__(self, value, *args, **kwargs):
        state = dict(value=jnp.asarray(value))
        shape = state['value'].shape
        if 'attrs' in kwargs:
            args = args + (kwargs.pop('attrs'),)
        if len(args):  # attrs is provided directly
            assert len(args) == 1, f"Do not understand args {args}"
            state['attrs'] = args[0].clone(**kwargs)
        else:
            meshsize = kwargs.pop('meshsize', None)
            if meshsize is None: meshsize = shape
            meshsize = staticarray.fill(meshsize, len(shape), dtype='i4')
            state['attrs'] = MeshAttrs(meshsize=meshsize, **kwargs)
        self.__dict__.update(state)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in ['value']), {name: getattr(self, name) for name in ['attrs']}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(value=children[0], attrs=aux_data['attrs'])
        return new

    def __getitem__(self, item):
        return self.value.__getitem__(item)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = {name: getattr(self, name) for name in ['value', 'attrs']}
        name = 'value'
        # value is provided, set meshsize to value.shape as a default
        if kwargs.get(name, state[name]).shape != state[name].shape and 'attrs' not in kwargs:
            kwargs.setdefault('meshsize', kwargs[name].shape)
        for name in list(kwargs):
            if name in state:
                state[name] = kwargs.pop(name)
        state['attrs'] = state['attrs'].clone(**kwargs)
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(state)
        return new

    @property
    def at(self):
        return _UpdateHelper(self)

    def __array__(self):
        # for numpy
        return np.array(self.value)

    def save(self, fn):
        """Save mesh to file."""
        fn = str(fn)
        utils.mkdir(os.path.dirname(fn))
        np.savez(fn, **{name: getattr(self, name) for name in ['value', 'attrs']})

    @classmethod
    def load(cls, fn):
        """Load mesh from file."""
        fn = str(fn)
        state = dict(np.load(fn, allow_pickle=True))
        new = cls.__new__(cls)
        state['attrs'] = state['attrs'][()]
        new.__dict__.update(**state)
        return new


def _set_property(base, name: str):
    setattr(base, name, property(lambda self: getattr(self.value, name)))


for name in ['ndim',
             'shape',
             'size',
             'dtype',
             'real',
             'imag']:
    _set_property(BaseMeshField, name)


def _set_property(base, name: str):
    setattr(base, name, property(lambda self: getattr(self.attrs, name)))


for name in [field.name for field in fields(MeshAttrs)] + ['cellsize']:
    _set_property(BaseMeshField, name)


def _set_binary(base, name: str, op: Callable[[Any, Any], Any]) -> None:
    def fn(self, other):
        if type(other) == type(self):
            return jax.tree.map(op, self, other)
        return jax.tree.map(lambda x: op(x, other), self)

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


def _set_unary(base, name: str, op: Callable[[Any], Any]) -> None:
    def fn(self):
        return jax.tree.map(op, self)

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


def _rev(op):
    def __rev(x, y):
        return op(y, x)

    return __rev


for name, op in [
    ("__add__", operator.add),
    ("__sub__", operator.sub),
    ("__mul__", operator.mul),
    ("__matmul__", operator.matmul),
    ("__truediv__", operator.truediv),
    ("__floordiv__", operator.floordiv),
    ("__mod__", operator.mod),
    ("__pow__", operator.pow),
    ("__lshift__", operator.lshift),
    ("__rshift__", operator.rshift),
    ("__and__", operator.and_),
    ("__xor__", operator.xor),
    ("__or__", operator.or_),
    ("__radd__", _rev(operator.add)),
    ("__rsub__", _rev(operator.sub)),
    ("__rmul__", _rev(operator.mul)),
    ("__rmatmul__", _rev(operator.matmul)),
    ("__rtruediv__", _rev(operator.truediv)),
    ("__rfloordiv__", _rev(operator.floordiv)),
    ("__rmod__", _rev(operator.mod)),
    ("__rpow__", _rev(operator.pow)),
    ("__rlshift__", _rev(operator.lshift)),
    ("__rrshift__", _rev(operator.rshift)),
    ("__rand__", _rev(operator.and_)),
    ("__rxor__", _rev(operator.xor)),
    ("__ror__", _rev(operator.or_)),
    ("__lt__", operator.lt),
    ("__le__", operator.le),
    ("__eq__", operator.eq),
    ("__ne__", operator.ne),
    ("__gt__", operator.gt),
    ("__ge__", operator.ge),
]:
    _set_binary(BaseMeshField, name, op)


for name, op in [
    ("__neg__", operator.neg),
    ("__pos__", operator.pos),
    ("__abs__", operator.abs),
    ("__invert__", operator.invert),
    ("conj", jnp.conj)
]:
    _set_unary(BaseMeshField, name, op)


def _set_method(base, name: str) -> None:

    def fn(self, *args, **kwargs):
        return getattr(self.value, name)(*args, **kwargs)

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


for name in [
    'sum',
    'mean',
    'var',
    'std',
    'min',
    'max',
    'argmin',
    'argmax',
    'ravel'
]:
    _set_method(BaseMeshField, name)



@dataclass
class _UpdateHelper(object):

    value: BaseMeshField

    def __getitem__(self, item):
        return _UpdateRef(self.value, item)


@dataclass
class _UpdateRef(object):

    value: BaseMeshField
    item: Any


def _set_binary_at(base, name: str) -> None:
    def fn(self, other, **kwargs):
        if type(other) == type(self.value):
            return jax.tree.map(lambda x, z: getattr(x.at[self.item], name)(z, **kwargs), self.value, other)
        return jax.tree.map(lambda x: getattr(x.at[self.item], name)(other, **kwargs), self.value)

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


for name in ['set',
             'add',
             'multiply',
             'divide',
             'power',
             'min',
             'max']:
    _set_binary_at(_UpdateRef, name)


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class RealMeshField(BaseMeshField):

    """A :class:`BaseMeshField` containing the values of a real (or complex) field."""

    def rebin(self, factor: int | tuple, axis: int | tuple=None, reduce: Callable=jnp.sum):
        """Rebin mesh by factors ``factor`` along axes ``axis``, with reduction operation ``reduce``."""
        value = utils.rebin(self.value, factor, axis=axis, reduce=reduce)
        return self.clone(value=value)

    @property
    def spacing(self):
        """Spacing between two mesh nodes."""
        return self.cellsize

    def coords(self, kind: str='position', sparse: bool | None=None):
        """
        Return mesh spatial coordinates.

        Parameters
        ----------
        kind : str, default='position'.
            Either 'index' (index on mesh) or 'position' (spatial coordinates).

        sparse : bool, default=None
            If ``None``, return a tuple of 1D-arrays.
            If ``False``, return a tuple of broadcastable arrays.
            If ``True``, return a tuple of broadcastable arrays of same shape, :attr:`shape`.

        Returns
        -------
        coords : tuple
        """
        return self.attrs.xcoords(kind=kind, sparse=sparse)

    def r2c(self):
        """
        FFT, from real to complex. If :attr:`value` is of complex type, return :class:`ComplexMeshField`.
        Else, a :class:`HermitianComplexMeshField`.
        """
        if jnp.iscomplexobj(self.value):
            fft = partial(ComplexMeshField, jnp.fft.fftn(self.value))
        else:
            fft = partial(HermitianComplexMeshField, jnp.fft.rfftn(self.value))
        return fft(self.attrs)

    def apply(self, fn: Callable, kind: str=Ellipsis, sparse: bool | None=False):
        """
        Apply input kernel ``fn`` to mesh.

        Parameters
        ----------
        fn : Callable
            Callable that takes the mesh value as input, and return a new mesh value.

        kind : str, default=Ellipsis
            To pass- mesh coordinates as an additional argument to ``fn``:
            - ``None``: no extra argument passed
            - 'index' or 'position': see :meth:`coords`
            - ``Ellipsis``: tries to find ``kind`` attribute attached ``fn``. If not present, falls back to ``None``.

        sparse : bool, default=False
            Used only if ``kind`` is not ``None``, see :meth:`coords`.

        Returns
        -------
        mesh : RealMeshField
        """
        args = [self.value]
        if kind is Ellipsis:
            kind = getattr(fn, 'kind', None)  # in case it is fn's attribute
        if kind is None:
            pass
        elif kind in ['index', 'position']:
            args.append(self.coords(kind=kind, sparse=sparse))
        else:
            raise ValueError('do not recognize kind = {}'.format(kind))
        value = fn(*args)
        assert value.shape == self.shape, f'kernel does not return correct shape: {value.shape} vs expected {self.shape}'
        return self.clone(value=value)

    @partial(jax.jit, static_argnames=['resampler', 'compensate'])
    def read(self, positions: jax.Array, resampler: str | Callable='cic', compensate: bool=False):
        """
        Read mesh, at input ``positions``.

        Parameters
        ----------
        positions : jax.Array
            Array of positions.

        resampler : str, Callable
            Resampler to read particule weights from mesh.
            One of ['ngp', 'cic', 'tsc', 'pcs'].

        compensate : bool, default=False
            If ``True``, applies compensation to the mesh before reading.

        Returns
        -------
        values : jax.Array
        """
        resampler = getattr(resamplers, resampler, resampler)
        if isinstance(compensate, bool):
            if compensate:
                kernel_compensate = resampler.compensate
            else:
                kernel_compensate = None
        else:
            kernel_compensate = compensate
        mesh = self
        if kernel_compensate is not None:
            mesh = mesh.r2c().apply(kernel_compensate).c2r()
        if isinstance(positions, ParticleField):
            positions = positions.positions
        return resampler.read(mesh.value, (positions + self.boxsize / 2. - self.boxcenter) / self.cellsize)


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class ComplexMeshField(BaseMeshField):

    """A :class:`BaseMeshField` containing the values of a field in Fourier space."""

    @property
    def spacing(self):
        """Spacing between two mesh nodes."""
        return self.kfun

    @property
    def kfun(self):
        """Fundamental wavenumber."""
        return self.attrs.kfun

    @property
    def knyq(self):
        """Nyquist wavenumber."""
        return self.attrs.knyq

    def coords(self, kind: str='wavenumber', sparse: bool | None=None):
        r"""
        Return mesh Fourier coordinates.

        Parameters
        ----------
        kind : str, default='wavenumber'
            Either 'circular' (in :math:`(-\pi, \pi)`) or 'wavenumber' (in ``spacing`` units).

        sparse : bool, default=None
            If ``None``, return a tuple of 1D-arrays.
            If ``False``, return a tuple of broadcastable arrays.
            If ``True``, return a tuple of broadcastable arrays of same shape.

        Returns
        -------
        coords : tuple
        """
        return self.attrs.kcoords(kind=kind, sparse=sparse, hermitian=False)

    def c2r(self):
        """FFT, from complex to real. Return :class:`RealMeshField`."""
        return RealMeshField(jnp.fft.ifftn(self.value), attrs=self.attrs)

    def apply(self, fn, sparse=False, kind=Ellipsis):
        """
        Apply input kernel ``fn`` to mesh.

        Parameters
        ----------
        fn : Callable
            Callable that takes the mesh value as input, and return a new mesh value.

        kind : str, default=Ellipsis
            To pass- mesh coordinates as an additional argument to ``fn``:
            - ``None``: no extra argument passed
            - 'wavenumber' or 'circular': see :meth:`coords`
            - ``Ellipsis``: tries to find ``kind`` attribute attached ``fn``. If not present, falls back to ``None``.

        sparse : bool, default=False
            Used only if ``kind`` is not ``None``, see :meth:`coords`.

        Returns
        -------
        mesh : ComplexMeshField
        """
        args = [self.value]
        if kind is Ellipsis:
            kind = getattr(fn, 'kind', None)  # in case it is fn's attribute
        if kind is None:
            pass
        elif kind in ['circular', 'wavenumber']:
            args.append(self.coords(kind=kind, sparse=sparse))
        else:
            raise ValueError('do not recognize kind = {}'.format(kind))
        value = fn(*args)
        assert value.shape == self.shape, f'kernel does not return correct shape: {value.shape} vs expected {self.shape}'
        return self.clone(value=value)


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class HermitianComplexMeshField(ComplexMeshField):

    """Same as :class:`ComplexMeshField`, but with Hermitian symmetry."""

    def coords(self, kind: str='wavenumber', sparse: bool | None=None):
        r"""
        Return mesh Fourier coordinates.

        Parameters
        ----------
        kind : str, default='wavenumber'
            Either 'circular' (in :math:`(-\pi, \pi)`) or 'wavenumber' (in ``spacing`` units).

        sparse : bool, default=None
            If ``None``, return a tuple of 1D-arrays.
            If ``False``, return a tuple of broadcastable arrays.
            If ``True``, return a tuple of broadcastable arrays of same shape.

        Returns
        -------
        coords : tuple
        """
        return self.attrs.kcoords(kind=kind, sparse=sparse, hermitian=True)

    def c2r(self):
        """FFT, from complex to real. Return :class:`RealMeshField`."""
        # shape must be provided in case meshsize is odd
        return RealMeshField(jnp.fft.irfftn(self.value, s=self.meshsize), attrs=self.attrs)


def get_mesh_attrs(*positions: np.ndarray, meshsize: np.ndarray | int=None,
                   boxsize: np.ndarray | float=None, boxcenter: np.ndarray | float=None,
                   cellsize: np.ndarray | float=None, boxpad: np.ndarray | float=2.,
                   check: bool=False, dtype=None):
    """
    Compute enclosing box.
    Differentiable if ``meshsize`` is provided.

    Parameters
    ----------
    positions : (list of) (N, 3) arrays, default=None
        If ``boxsize`` and / or ``boxcenter`` is ``None``, use this (list of) position arrays
        to determine ``boxsize`` and / or ``boxcenter``.

    meshsize : array, int, default=None
        Mesh size, i.e. number of mesh nodes along each axis.
        If not provided, see ``value``.

    boxsize : float, default=None
        Physical size of the box.
        If not provided, see ``positions``.

    boxcenter : array, float, default=None
        Box center.
        If not provided, see ``positions``.

    cellsize : array, float, default=None
        Physical size of mesh cells.
        If not ``None``, ``boxsize`` is ``None`` and mesh size ``meshsize`` is not ``None``, used to set ``boxsize`` to ``meshsize * cellsize``.
        If ``meshsize`` is ``None``, it is set to (the nearest integer(s) to) ``boxsize / cellsize`` if ``boxsize`` is provided,
        else to the nearest even integer to ``boxsize / cellsize``, and ``boxsize`` is then reset to ``meshsize * cellsize``.

    boxpad : float, default=2.
        When ``boxsize`` is determined from ``positions``, take ``boxpad`` times the smallest box enclosing ``positions`` as ``boxsize``.

    check : bool, default=False
        If ``True``, and input ``positions`` (if provided) are not contained in the box, raise a :class:`ValueError`.

    Returns
    -------
    attrs : dictionary, with:
        - boxsize : array, physical size of the box
        - boxcenter : array, box center
        - meshsize : array, shape of mesh.
    """
    if boxsize is None or boxcenter is None or check:
        if not positions:
            raise ValueError('positions must be provided if boxsize and boxcenter are not specified, or check is True')
        # Find bounding coordinates
        nonempty_positions = [pos for pos in positions if pos.size]
        if not nonempty_positions:
            raise ValueError('<= 1 particles found; cannot infer boxsize')
        pos_min = min(jax.tree.map(partial(jnp.min, axis=0), nonempty_positions))
        pos_max = max(jax.tree.map(partial(jnp.max, axis=0), nonempty_positions))
        delta = jnp.abs(pos_max - pos_min)
        if boxcenter is None: boxcenter = 0.5 * (pos_min + pos_max)
        if boxsize is None:
            if cellsize is not None and meshsize is not None:
                boxsize = meshsize * cellsize
            else:
                boxsize = delta.max() * boxpad
        if check and (boxsize < delta).any():
            raise ValueError('boxsize {} too small to contain all data (max {})'.format(boxsize, delta))

    ndim = None
    if positions is not None:
        if dtype is None:
            dtype = positions[0].dtype
        if ndim is None:
            ndim = positions[0].shape[-1]

    boxsize = staticarray.fill(boxsize, ndim, dtype=dtype)
    toret = dict()
    if meshsize is None:
        if cellsize is not None:
            cellsize = staticarray.fill(cellsize, ndim, dtype=dtype)
            toret['meshsize'] = meshsize = np.ceil(boxsize / cellsize).astype('i4')
            toret['boxsize'] = meshsize * cellsize  # enforce exact cellsize
        else:
            raise ValueError('meshsize (or cellsize) must be specified')
    else:
        meshsize = staticarray.fill(meshsize, ndim, dtype='i4')
        toret['meshsize'] = meshsize
    boxcenter = staticarray.fill(boxcenter, ndim, dtype=dtype)
    toret = dict(boxsize=boxsize, boxcenter=boxcenter) | toret
    return MeshAttrs(**{name: staticarray(value) for name, value in toret.items()})


def _get_common_mesh_cache_attrs(*attrs, **kwargs):
    gather = attrs[0]
    for a in attrs:
         if a != gather:
            raise RuntimeError("Not same attrs, {} vs {}".format(a, gather))
    return gather | kwargs


def get_common_mesh_attrs(*particles, **kwargs):
    """Get common mesh attributes for multiple :class:`ParticleField`."""
    attrs = _get_common_mesh_cache_attrs(*[p._cache_attrs for p in particles], **kwargs)
    return get_mesh_attrs(*[p.positions for p in particles], **attrs)


# Note: I couldn't make it work properly with jax dataclass registration as tree_unflatten would pass all fields to __init__, which I don't want
@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class ParticleField(object):
    """
    Representation of a N-dim field as a collection of particles / weights in N-dim space.
    We can e.g. add two :class:`ParticleField`'s like:

    >>> pf1 + pf2

    I also started an implementation to concatenate ``positions`` and ``weights`` arrays only if needed.
    Note sure it's worth the complexity.
    """

    positions: jax.Array = field(repr=False)
    weights: jax.Array = field(repr=False)
    _attrs: MeshAttrs | None = field(init=False, repr=False)
    _cache_attrs: dict = field(init=True, repr=False)

    def __init__(self, positions: jax.Array, weights: jax.Array | None=None, **kwargs):
        weights = jnp.ones_like(positions, shape=positions.shape[0]) if weights is None else weights
        assert '_cache_attrs' not in kwargs
        kwargs = {name: staticarray(value) if value is not None else value for name, value in kwargs.items()}
        self.__dict__.update(positions=positions, weights=weights, _attrs=None, _cache_attrs=kwargs)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = asdict(self)
        for name in ['positions', 'weights']:
            if name in kwargs:
                state[name] = kwargs.pop(name)
        state.pop('_attrs')
        attrs = state.pop('_cache_attrs')
        attrs.update(kwargs)
        state.update(attrs)
        return self.__class__(**state)

    @property
    def attrs(self):
        if self._attrs is None:
            self.__dict__.update(_attrs=get_mesh_attrs(self.positions, **self._cache_attrs))
        return self._attrs

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in ['positions', 'weights']), {name: getattr(self, name) for name in ['_attrs', '_cache_attrs']}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(positions=children[0], weights=children[1], **aux_data)
        return new

    def __getitem__(self, name):
        """Array-like slicing."""
        return self.clone(positions=self.positions[name], weights=self.weights[name])

    @property
    def size(self):
        return self.weights.size

    def sum(self, *args, **kwargs):
        """Sum of :attr:`weights`."""
        return self.weights.sum(*args, **kwargs)

    @staticmethod
    def same_mesh(*others, **kwargs):
        attrs = get_common_mesh_attrs(*others, **kwargs)
        return tuple(other.clone(**attrs) for other in others)

    @classmethod
    def concatenate(cls, others, weights=None, **kwargs):
        """Sum multiple :class:`ParticleField`, with input weights."""
        if weights is None:
            weights = [1] * len(weights)
        else:
            assert len(weights) == len(others)
        gather = {name: [] for name in ['positions', 'weights']}
        for other, factor in zip(others, weights):
            if not isinstance(other, ParticleField):
                raise RuntimeError("Type of `other` not understood.")
            gather['positions'].append(other.positions)
            gather['weights'].append(factor * other.weights)
        cache_attrs = _get_common_mesh_cache_attrs(*[other._cache_attrs for other in others], **kwargs)
        for name, value in gather.items():
            gather[name] = jnp.concatenate(value, axis=0)
        return cls(**gather, **cache_attrs)

    def __add__(self, other):
        if isinstance(other, ParticleField):
            return self.concatenate([self, other], [1, 1])
        else:
            raise RuntimeError("Type of `other` not understood.")

    def __sub__(self, other):
        if isinstance(other, ParticleField):
            return self.concatenate([self, other], [1, -1])
        else:
            raise RuntimeError("Type of `other` not understood.")

    def __mul__(self, other):
        return self.concatenate([self], [other])

    def __rmul__(self, other):
        return self.concatenate([self], [other])

    def __truediv__(self, other):
        return self.concatenate([self], [1. / other])

    def __rtruediv__(self, other):
        return self.concatenate([self], [1. / other])

    def paint(self, resampler: str | Callable='cic', interlacing: int=0,
              compensate: bool=False, dtype=None, out: str='real'):
        r"""
        Paint particles to mesh.

        Parameters
        ----------
        resampler : str, Callable
            Resampler to read particule weights from mesh.
            One of ['ngp', 'cic', 'tsc', 'pcs'].

        interlacing : int, default=0
            If 0 or 1, no interlacing correction.
            If > 1, order of interlacing correction.
            Typically, 3 gives reliable power spectrum estimation up to :math:`k \sim k_\mathrm{nyq}`.

        compensate : bool, default=False
            If ``True``, applies compensation to the mesh after painting.

        dtype : default=None
            Mesh array type.

        out : str, default='real'
            If 'real', return a :class:`RealMeshField`, else :class:`HermitianComplexMeshField`
            or :class:`ComplexMeshField` if ``dtype`` is complex.

        Returns
        -------
        mesh : Output mesh.
        """
        return _paint(self.attrs, self.positions, weights=self.weights, resampler=resampler, interlacing=interlacing, compensate=compensate, dtype=dtype, out=out)

@partial(jax.jit, static_argnames=['attrs', 'resampler',  'interlacing', 'compensate', 'dtype', 'out'])
def _paint(attrs, positions, weights=None, resampler: str | Callable='cic', interlacing: int=0, compensate: bool=False, dtype=None, out: str='real'):

    resampler = getattr(resamplers, resampler, resampler)
    interlacing = max(interlacing, 1)
    shifts = np.arange(interlacing) * 1. / interlacing
    positions = (positions + attrs.boxsize / 2. - attrs.boxcenter) / attrs.cellsize

    if isinstance(compensate, bool):
        if compensate:
            kernel_compensate = resampler.compensate
        else:
            kernel_compensate = None
    else:
        kernel_compensate = compensate

    def ppaint(positions):
        return RealMeshField(resampler.paint(jnp.zeros(attrs.meshsize, dtype=dtype), positions, weights.astype(dtype) if dtype is not None else weights), boxsize=attrs.boxsize, boxcenter=attrs.boxcenter)

    toret = ppaint(positions)
    if interlacing > 1:
        toret = toret.r2c()
        for shift in shifts[1:]: # remove 0 shift, already computed

            # convention is F(k) = \sum_{r} e^{-ikr} F(r)
            # shifting by "shift * cellsize" we compute F(k) = \sum_{r} e^{-ikr} F(r - shift * cellsize)
            # i.e. F(k) = e^{- i shift * kc} e^{-ikr} F(r)
            # Hence compensation below

            def kernel_shift(value, kvec):
                kvec = sum(kvec)
                return value * jnp.exp(shift * 1j * kvec)

            toret += ppaint(positions + shift).r2c().apply(kernel_shift, kind='circular')
        if kernel_compensate is not None:
            toret = toret.apply(kernel_compensate)
        toret /= interlacing
        if out == 'real': toret = toret.c2r()
    elif kernel_compensate is not None:
        toret = toret.r2c().apply(kernel_compensate)
        if out == 'real': toret = toret.c2r()
    else:
        if out != 'real': toret = toret.r2c()
    return toret


def _set_property(base, name: str):
    setattr(base, name, property(lambda self: getattr(self.attrs, name)))


for name in [field.name for field in fields(MeshAttrs)] + ['cellsize']:
    _set_property(ParticleField, name)


# For functional programming interface

def c2r(mesh: ComplexMeshField | HermitianComplexMeshField) -> RealMeshField:
    return mesh.c2r()


def r2c(mesh: RealMeshField) -> ComplexMeshField | HermitianComplexMeshField:
    return mesh.r2c()


def apply(mesh: RealMeshField | ComplexMeshField | HermitianComplexMeshField,
          fn: Callable, sparse=False, **kwargs) -> RealMeshField | ComplexMeshField | HermitianComplexMeshField:
    return mesh.apply(fn, sparse=sparse, **kwargs)


def read(mesh: RealMeshField, positions: jax.Array, *args, **kwargs) -> jax.Array:
    return mesh.read(positions, *args, **kwargs)


def paint(mesh: ParticleField, *args, **kwargs) -> RealMeshField | ComplexMeshField | HermitianComplexMeshField:
    return mesh.paint(*args, **kwargs)