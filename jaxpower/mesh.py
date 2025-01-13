import operator
from functools import partial
from collections.abc import Callable
from typing import Any, Union

import numpy as np
import jax
from jax import numpy as jnp
from dataclasses import dataclass, field, asdict
from . import resamplers


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
    def fill(cls, fill: Union[float, np.ndarray], shape: Union[int, tuple], **kwargs):
        """
        Create a :class:`staticarray` of shape ``shape``, filled with ``fill``,
        which can be a numpy array, as long as its shape is brodcastable with ``shape``.
        """
        fill = np.array(fill)
        toret = np.empty_like(fill, shape=shape, **kwargs)
        toret[...] = fill
        return cls(toret)


@partial(jax.tree_util.register_dataclass, data_fields=['value'], meta_fields=['boxsize', 'boxcenter', 'attrs'])
@dataclass(frozen=True)
class BaseMeshField(object):

    """Representation of a N-dim field as a N-dim array."""

    value: jax.Array = field(repr=False)
    boxsize: staticarray = None
    boxcenter: staticarray = 0.
    meshsize: staticarray = field(default=None, init=False, hash=None)
    attrs: dict = field(default_factory=lambda: dict())

    def __post_init__(self):
        state = asdict(self)
        state['value'] = jnp.asarray(state['value'])
        shape = state['value'].shape
        dtype = state['value'].real.dtype
        if state['meshsize'] is None: state['meshsize'] = shape
        if state['boxsize'] is None: state['boxsize'] = tuple(1. * s for s in self.shape)
        for name, dt in zip(['meshsize', 'boxsize', 'boxcenter'], ['i4', dtype, dtype]):
            state[name] = staticarray.fill(state[name], len(shape), dtype=dt)
        self.__dict__.update(state)

    def __getitem__(self, item):
        return self.value.__getitem__(item)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = asdict(self)
        state.update(kwargs)
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(state)
        return new

    @property
    def at(self):
        return _UpdateHelper(self)

    def __array__(self):
        # for numpy
        return np.array(self.value)

    @property
    def cellsize(self):
        return self.boxsize / self.meshsize


def _set_property(base, name: str):
    setattr(base, name, property(lambda self: getattr(self.value, name)))


for name in ['ndim',
             'shape',
             'size',
             'dtype',
             'real',
             'imag']:
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


@partial(jax.tree_util.register_dataclass, data_fields=['value'], meta_fields=['boxsize', 'boxcenter', 'attrs'])
class RealMeshField(BaseMeshField):

    """A :class:`BaseMeshField` containing the values of a real (or complex) field."""

    def coords(self, kind: str='position', sparse: Union[bool, None]=None):
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
        toret = [np.arange(s) for s in self.shape]
        if kind != 'index':
            toret = [idx * box + center - box / 2. for idx, box, center in zip(toret, self.boxsize, self.boxcenter)]
        if sparse is None:
            return tuple(toret)
        return jnp.meshgrid(*toret, sparse=sparse, indexing='ij')

    def r2c(self):
        """
        FFT, from real to complex. If :attr:`value` is of complex type, return :class:`ComplexMeshField`.
        Else, a :class:`HermitianComplexMeshField`.
        """
        if jnp.iscomplexobj(self.value):
            fft = partial(ComplexMeshField, jnp.fft.fftn(self.value))
        else:
            fft = partial(HermitianComplexMeshField, jnp.fft.rfftn(self.value), meshsize=self.meshsize)
        return fft(boxsize=self.boxsize, boxcenter=self.boxcenter, attrs=self.attrs)

    def apply(self, fn: callable, kind: str=Ellipsis, sparse: Union[bool, None]=False):
        """
        Apply input kernel ``fn`` to mesh.

        Parameters
        ----------
        fn : callable
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
        return self.clone(value=value)

    def read(self, positions: jax.Array, resampler: Union[str, callable]='cic', compensate: bool=False):
        """
        Read mesh, at input ``positions``.

        Parameters
        ----------
        positions : jax.Array
            Array of positions.

        resampler : str, callable
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
        return resampler.read(mesh.value, (positions - self.boxsize / 2. - self.boxcenter) / self.cellsize)


def fftfreq(shape: tuple, kind: str='wavenumber', sparse: Union[bool, None]=None,
            hermitian: bool=False, spacing: Union[np.ndarray, float]=1.):
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
        toret = [np.arange(s) for s in shape]
    else:
        if kind == 'circular':
            period = (2 * np.pi,) * ndim
        else:  # wavenumber
            period = 2 * np.pi / spacing
        toret = []
        for axis, s in enumerate(shape):
            k = (np.fft.rfftfreq if axis == ndim - 1 and hermitian else np.fft.fftfreq)(s) * period[axis]
            toret.append(k)
    if sparse is None:
        return tuple(toret)
    return np.meshgrid(*toret, sparse=sparse, indexing='ij')


@partial(jax.tree_util.register_dataclass, data_fields=['value'], meta_fields=['boxsize', 'boxcenter', 'attrs'])
class ComplexMeshField(BaseMeshField):

    """A :class:`BaseMeshField` containing the values of a field in Fourier space."""

    @property
    def kfun(self):
        """Fundamental wavenumber."""
        return 2. * np.pi / self.boxsize

    @property
    def knyq(self):
        """Nyquist wavenumber."""
        return np.pi * self.meshsize / self.boxsize

    def coords(self, kind: str='wavenumber', sparse: Union[bool, None]=None):
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
        return fftfreq(self.shape, kind=kind, sparse=sparse, hermitian=False, spacing=self.boxsize / self.meshsize)

    def c2r(self):
        """FFT, from complex to real. Return :class:`RealMeshField`."""
        return RealMeshField(jnp.fft.ifftn(self.value), boxsize=self.boxsize, boxcenter=self.boxcenter, attrs=self.attrs)

    def apply(self, fn, sparse=False, kind=Ellipsis):
        """
        Apply input kernel ``fn`` to mesh.

        Parameters
        ----------
        fn : callable
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
        return self.clone(value=value)


@partial(jax.tree_util.register_dataclass, data_fields=['value'], meta_fields=['boxsize', 'boxcenter', 'meshsize', 'attrs'])
@dataclass(frozen=True)
class HermitianComplexMeshField(ComplexMeshField):

    """Same as :class:`ComplexMeshField`, but with Hermitian symmetry."""

    meshsize: staticarray = field(default=None)

    def coords(self, kind: str='wavenumber', sparse: Union[bool, None]=None):
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
        return fftfreq(self.meshsize, kind=kind, sparse=sparse, hermitian=True, spacing=self.boxsize / self.meshsize)

    def c2r(self):
        """FFT, from complex to real. Return :class:`RealMeshField`."""
        # shape must be provided in case meshsize is odd
        return RealMeshField(jnp.fft.irfftn(self.value, s=self.meshsize), boxsize=self.boxsize, boxcenter=self.boxcenter, attrs=self.attrs)


def get_mesh_cache_attrs(*positions: np.ndarray, meshsize: Union[np.ndarray, int]=None,
                         boxsize: Union[np.ndarray, float]=None, boxcenter: Union[np.ndarray, float]=None,
                         cellsize: Union[np.ndarray, float]=None, boxpad: Union[np.ndarray, float]=2.,
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
    return {name: staticarray(value) for name, value in toret.items()}


@partial(jax.tree_util.register_dataclass, data_fields=['positions', 'weights'], meta_fields=['_cache_attrs'])
@dataclass(init=False, frozen=True)
class ParticleField(object):
    """
    Representation of a N-dim field as a collection of particles / weights in N-dim space.
    We can e.g. add two :class:`ParticleField`'s like:

    >>> pf1 + pf2

    I also started an implementation to concatenate ``positions`` and ``weights`` arrays only if needed.
    Note sure it's worth the complexity.
    """

    positions: jax.Array
    weights: jax.Array = None
    _attrs: dict = None
    _cache_attrs: dict = None

    def __init__(self, positions: jax.Array, weights: Union[jax.Array, None]=None, **kwargs):
        weights = jnp.ones_like(positions, shape=positions.shape[0]) if weights is None else weights
        self.__dict__.update(positions=positions, weights=weights, _attrs={}, _cache_attrs=kwargs)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        return self.__class__(self.positions, weights=self.weights, **(self._cache_attrs | kwargs))

    def sum(self, *args, **kwargs):
        """Sum of :attr:`weights`."""
        return self.weights.sum(*args, **kwargs)

    @classmethod
    def concatenate(cls, others, weights=None):
        """Sum multiple :class:`ParticleField`, with input weights."""
        if weights is None:
            weights = [1] * len(weights)
        else:
            assert len(weights) == len(others)
        gather = {name: [] for name in ['positions', 'weights']}
        cache_attrs = others[0]._cache_attrs
        for other, factor in zip(others, weights):
            if not isinstance(other, ParticleField):
                raise RuntimeError("Type of `other` not understood.")
            if set(other._cache_attrs) != set(cache_attrs) or not all(np.all(cache_attrs[name] == other._cache_attrs[name]) for name in cache_attrs):
                raise RuntimeError("Not same _cache_attrs, {} vs {}".format(other._cache_attrs, cache_attrs))
            gather['positions'].append(other.positions)
            gather['weights'].append(factor * other.weights)
        for name, value in gather.items():
            gather[name] = jnp.concatenate(gather[name], axis=0)
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

    def paint(self, resampler: Union[str, callable]='cic', interlacing: int=1,
              compensate: bool=False, dtype=None, out: str='real'):
        r"""
        Paint particles to mesh.

        Parameters
        ----------
        resampler : str, callable
            Resampler to read particule weights from mesh.
            One of ['ngp', 'cic', 'tsc', 'pcs'].

        interlacing : int, default=1
            If 1, no interlacing correction.
            If > 1, order of interlacing correction.
            Typically, 3 gives reliable power spectrum estimation up to :math:`k \sim k_\mathrm{nyq}`.

        compensate : bool, default=False
            If ``True``, applies compensation to the mesh after painting.

        dtype : defaulat=None
            Mesh array type.

        out : str, default='real'
            If 'real', return a :class:`RealMeshField`, else :class:`HermitianComplexMeshField`
            or :class:`ComplexMeshField` if ``dtype`` is complex.

        Returns
        -------
        mesh : Output mesh.
        """
        resampler = getattr(resamplers, resampler, resampler)
        shifts = np.arange(interlacing) * 1. / interlacing
        positions = (self.positions - self.boxsize / 2. - self.boxcenter) / self.cellsize

        def _paint(positions):
            return RealMeshField(resampler.paint(jnp.zeros(tuple(self.meshsize), dtype=dtype), positions, self.weights), boxsize=self.boxsize, boxcenter=self.boxcenter)

        if isinstance(compensate, bool):
            if compensate:
                kernel_compensate = resampler.compensate
            else:
                kernel_compensate = None
        else:
            kernel_compensate = compensate
        toret = _paint(positions)
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

                toret += _paint(positions + shift).r2c().apply(kernel_shift, kind='circular')
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


def _set_property(base, name):

    def fn(self):
        if name not in self._attrs:
            self._attrs.update(get_mesh_cache_attrs(self.positions, **self._cache_attrs))
            if 'cellsize' not in self._attrs:
                self._attrs['cellsize'] = self._attrs['boxsize'] / self._attrs['meshsize']
        return self._attrs[name]

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, property(fn))


for name in ['boxsize',
             'boxcenter',
             'meshsize',
             'cellsize']:
    _set_property(ParticleField, name)


# For functional programming interface

def c2r(mesh: Union[ComplexMeshField, HermitianComplexMeshField]) -> RealMeshField:
    return mesh.c2r()


def r2c(mesh: RealMeshField) -> Union[ComplexMeshField, HermitianComplexMeshField]:
    return mesh.r2c()


def apply(mesh: Union[RealMeshField, ComplexMeshField, HermitianComplexMeshField],
          fn: Callable, sparse=False, **kwargs) -> Union[RealMeshField, ComplexMeshField, HermitianComplexMeshField]:
    return mesh.apply(fn, sparse=sparse, **kwargs)


def read(mesh: RealMeshField, positions: jax.Array, *args, **kwargs) -> jax.Array:
    return mesh.read(positions, *args, **kwargs)


def paint(mesh: ParticleField, *args, **kwargs) -> Union[RealMeshField, ComplexMeshField, HermitianComplexMeshField]:
    return mesh.paint(*args, **kwargs)