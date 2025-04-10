import os
import operator
import functools
from functools import partial
from collections.abc import Callable
from typing import Any

import numpy as np
import jax
from jax import numpy as jnp
from jax import sharding
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from dataclasses import dataclass, field, fields, asdict
from . import resamplers, utils

try:
    import jaxdecomp
except ImportError:
    jaxdecomp = None


def get_sharding_mesh():
    from jax._src import mesh as mesh_lib
    return mesh_lib.thread_resources.env.physical_mesh


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


def _get_ndim(*args, default=3):
    ndim = default
    for value in args:
        try: ndim = len(value)
        except: pass
    return ndim


def create_sharding_mesh(meshsize=None, device_mesh_shape=None):

    if device_mesh_shape is None:
        count = jax.process_count()
        if meshsize is None:
            meshsize = 0
        ndim = _get_ndim(meshsize, default=2)
        meshsize = staticarray.fill(meshsize, ndim, dtype='i4')

        def closest_divisors_to_sqrt(n):
            sqrt_n = int(np.sqrt(n))

            for i in range(sqrt_n, 0, -1):
                if n % i == 0:
                    a = i
                    b = n // i
                    if meshsize[0] % a == meshsize[1] % b == 0:
                        return (a, b)
                    if meshsize[0] % b == meshsize[1] % a == 0:
                        return (b, a)
            # Fallback, n is prime
            if meshsize[0] % n == 0:
                return (n, 1)
            if meshsize[1] % n == 0:
                return (1, n)
            raise ValueError('did not find any 2-dim mesh shape that divides meshsize = {}'.format(meshsize))

        device_mesh_shape = closest_divisors_to_sqrt(count)
    device_mesh_shape = tuple(s for s in device_mesh_shape if s > 1)
    if len(device_mesh_shape) > 2:
        raise NotImplementedError('device mesh with ndim {:d} > 2 not implemented'.format(len(device_mesh_shape)))
    device_mesh_shape = (1,) * (2 - len(device_mesh_shape)) + device_mesh_shape
    devices = mesh_utils.create_device_mesh(device_mesh_shape)
    return sharding.Mesh(devices, axis_names=('x', 'y'))


def default_sharding_mesh(func: Callable):

    @functools.wraps(func)
    def wrapper(*args, sharding_mesh=None, **kwargs):
        if sharding_mesh is None:
            sharding_mesh = get_sharding_mesh()
        return func(*args, sharding_mesh=sharding_mesh, **kwargs)

    return wrapper


@partial(jax.jit, static_argnames=('sparse', 'sharding_mesh'))
def _get_freq_mesh(*freqs, sparse=None, sharding_mesh: jax.sharding.Mesh=None):
    def get_mesh(*freqs):
        if sparse is None:
            return tuple(freqs)
        return tuple(jnp.meshgrid(*freqs, sparse=sparse, indexing='ij'))

    if sharding_mesh is not None and sharding_mesh.axis_names:
        ndim = len(freqs)
        in_specs = tuple(P(name) for name in sharding_mesh.axis_names)
        remaining = ndim - len(sharding_mesh.axis_names)
        if remaining:
            in_specs += tuple(P(None) for i in range(remaining))
        if sparse is None:
            out_specs = in_specs
        elif sparse:
            out_specs = []
            for i in range(ndim):
                tmp = [None] * ndim
                if i < len(sharding_mesh.axis_names): tmp[i] = sharding_mesh.axis_names[i]
                out_specs.append(P(*tmp))
            out_specs = tuple(out_specs)
        else:
            out_specs = tuple(P(*sharding_mesh.axis_names) for i in range(ndim))
        get_mesh = shard_map(get_mesh, sharding_mesh, in_specs=in_specs, out_specs=out_specs)
    return get_mesh(*freqs)


@default_sharding_mesh
def fftfreq(shape: tuple, kind: str='separation', sparse: bool | None=None,
            hermitian: bool=False, spacing: np.ndarray | float=1.,
            axis_order: tuple=None, sharding_mesh: jax.sharding.Mesh=None):
    r"""
    Return mesh frequencies.

    Parameters
    ----------
    kind : str, default='separation'
        Either 'separation' (i.e. wavenumbers) or 'position'.

    sparse : bool, default=None
        If ``None``, return a tuple of 1D-arrays.
        If ``False``, return a tuple of broadcastable arrays.
        If ``True``, return a tuple of broadcastable arrays of same shape.

    hermitian : bool, default=False
        If ``True``, last axis is of size ``shape[-1] // 2 + 1``.

    spacing : float, default=1.
        Sampling spacing, typically ``cellsize`` (real space) or ``kfun`` (Fourier space).

    Returns
    -------
    freqs : tuple
    """
    ndim = len(shape)
    spacing = staticarray.fill(spacing, ndim)

    toret = []
    if kind == 'position':
        for axis, s in enumerate(shape):
            k = jnp.arange(s) * spacing[axis]
            toret.append(k)
    elif kind == 'separation':
        for axis, s in enumerate(shape):
            k = (jnp.fft.rfftfreq if axis == ndim - 1 and hermitian else jnp.fft.fftfreq)(s) * spacing[axis] * s
            toret.append(k)

    ndim = len(toret)
    if axis_order is None:
        axis_order = list(range(ndim))
    toret = [toret[axis] for axis in axis_order]
    toret = _get_freq_mesh(*toret, sparse=sparse, sharding_mesh=sharding_mesh)
    return tuple(toret[axis_order.index(axis)] for axis in range(len(toret)))


@default_sharding_mesh
def shard_shape(shape: tuple, sharding_mesh: jax.sharding.Mesh=None):
    if not len(sharding_mesh.axis_names):
        return tuple(shape)
    return tuple(s // pdim for s, pdim in zip(shape, sharding_mesh.devices.shape)) + shape[sharding_mesh.devices.ndim:]


@default_sharding_mesh
def pad_halo(value, halo_size=0, sharding_mesh: jax.sharding.Mesh=None):
    pad_width = [(halo_size,) * 2] * len(sharding_mesh.axis_names)
    pad_width += [(0,) * 2] * len(value.shape[len(sharding_mesh.axis_names):])

    def pad(value):
        return jnp.pad(value, tuple(pad_width))

    offset = jnp.array([width[0] for width in pad_width])
    return shard_map(pad, mesh=sharding_mesh, in_specs=(P(*sharding_mesh.axis_names),), out_specs=P(*sharding_mesh.axis_names))(value), offset


@default_sharding_mesh
def exchange_halo(value, halo_size=0, sharding_mesh: jax.sharding.Mesh=None):
    extents = [(halo_size // 2,) * 2] * len(sharding_mesh.axis_names)
    return jaxdecomp.halo_exchange(value, halo_extents=tuple(extents), halo_periods=(True,) * len(extents))


@default_sharding_mesh
def make_particles_from_local(per_host_positions, per_host_weights, sharding_mesh: jax.sharding.Mesh=None):

    if not len(sharding_mesh.axis_names):
        return per_host_positions, per_host_weights

    per_host_size = jnp.array(per_host_positions.size[:1])
    sizes = jax.make_array_from_process_local_data(sharding_mesh, per_host_size)
    max_size = jnp.max(sizes)
    pad_width = [(0, max_size - per_host_positions.size)] + [(0, 0)] * (per_host_positions.ndim - 1)
    per_host_positions = jnp.pad(per_host_positions, pad_width=pad_width)
    sharding = jax.sharding.NamedSharding(sharding_mesh, P(sharding_mesh.axis_names,))
    global_size = max_size * sizes.size
    positions = jax.make_array_from_process_local_data(sharding_mesh, per_host_positions, global_shape=(global_size,) + per_host_positions.shape[-1:])
    pad_width = [(0, max_size - per_host_weights.size)] + [(0, 0)] * (per_host_weights.ndim - 1)
    per_host_weights = jnp.pad(per_host_weights, pad_width=pad_width, mode='constant', constant_values=0.)
    weights = jax.make_array_from_process_local_data(sharding_mesh, per_host_weights, global_shape=(global_size,))
    return positions, weights


def exchange_particles(attrs, positions: jax.Array):
    sharding_mesh = attrs.sharding_mesh
    devices = sharding_mesh.devices

    if not len(sharding_mesh.axis_names):

        def exchange(positions=None, weights=None):
            toret = []
            if positions is not None:
                toret.append(positions)
            if weights is not None:
                toret.append(weights)
            return toret

        def inverse(values):
            return values

        exchange.inverse = inverse

        return exchange

    fidx_out_devices = ((positions + attrs.boxsize / 2. - attrs.boxcenter) % attrs.boxsize)[..., :devices.ndim] / (attrs.boxsize[:devices.ndim] / devices)
    idx_out_devices = devices[jnp.floor(fidx_out_devices).astype('i4')]
    count_out_device = jnp.bincount(idx_out_devices, length=devices.max() + 1)[1:]
    count_out_max = count_out_device.max()

    p_shape = positions.shape
    p_sharding = positions.sharding
    out_sharding = jax.sharding.NamedSharding(sharding_mesh, P(sharding_mesh.axis_names,))

    def exchange(positions=None, weights=None):

        def get_positions(idevice):
            mask = jnp.all(idx_out_devices == idevice.start, axis=-devices.ndim)
            p = positions[mask]
            pad_width = [(0, count_out_max - count_out_device[devices[idevice]])] + [(0, 0)] * (p.ndim - 1)
            return jnp.pad(p, pad_width=pad_width)

        def get_weights(idevice):
            mask = jnp.all(idx_out_devices == idevice.start, axis=-devices.ndim)
            w = weights[mask]
            pad_width = [(0, count_out_max - count_out_device[devices[idevice]])] + [(0, 0)] * (w.ndim - 1)
            w = jnp.pad(w, pad_width=pad_width, mode='constant', constant_values=0.)
            return w

        toret = []
        if positions is not None:
            toret.append(jax.make_array_from_callback((devices.size, count_out_max) + positions.shape[-1:], out_sharding, get_positions))
        if weights is not None:
            toret.append(jax.make_array_from_callback((devices.size, count_out_max), out_sharding, get_weights))
        if len(toret) == 1:
            toret = toret[0]
        return toret

    def inverse(values):

        def f(carry, idevice):
            mask = idx_out_devices == idevice
            carry = carry.at[mask].set(jnp.cumsum(mask)[mask])
            return carry, idevice

        index_in_out = idx_out_devices.copy()
        for idevice in devices.ravel():
            index_in_out, _ = f(index_in_out, idevice)

        #index_in_out = jax.lax.scan(f, index_in_out, devices.ravel())[0]  # it will never be jitted
        index_in_out = (idx_out_devices, index_in_out)

        def get_values(sl):
            return values[tuple(tmp[sl] for tmp in index_in_out)]

        return jax.make_array_from_callback(p_shape, p_sharding, get_values)

    exchange.inverse = inverse
    return exchange


@default_sharding_mesh
def unpad_halo(value, halo_size, sharding_mesh=None):

    def unpad(value):
        crop = []
        for axis in range(len(sharding_mesh.axis_names)):
            slices = [slice(None) for i in range(value.ndim)]
            slices[axis] = slice(halo_size, halo_size + halo_size // 2)
            slices_add = list(slices_add)
            slices_add[axis] = slice(0, halo_size // 2)
            value = value.at[tuple(slices)].set(value[tuple(slices_add)])
            slices[axis] = slice(-(halo_size + halo_size // 2), -halo_size)
            slices_add = list(slices_add)
            slices_add[axis] = slice(-halo_size // 2)
            value = value.at[tuple(slices)].set(value[tuple(slices_add)])
            crop.append(slice(halo_size, -halo_size))
        return value[tuple(crop)]

    return shard_map(unpad, mesh=sharding_mesh, in_specs=(P(*sharding_mesh.axis_names),), out_specs=P(*sharding_mesh.axis_names))(value)


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class MeshAttrs(object):

    meshsize: staticarray = None
    boxsize: staticarray = None
    boxcenter: staticarray = 0.
    dtype: Any = None
    fft_engine: str = None

    def __init__(self, meshsize=None, boxsize=None, boxcenter=0., dtype=float, fft_engine='auto'):
        ndim = _get_ndim(*[meshsize, boxsize, boxcenter])
        if meshsize is not None:  # meshsize may not be provided (e.g. for particles) and it is fine
            meshsize = staticarray.fill(meshsize, ndim, dtype='i4')
            if boxsize is None: boxsize = 1. * meshsize
        if boxsize is not None: boxsize = staticarray.fill(boxsize, ndim, dtype=float)
        if boxcenter is not None: boxcenter = staticarray.fill(boxcenter, ndim, dtype=float)
        dtype = jnp.zeros((), dtype=dtype).dtype  # default dtype is float32, except if config.update('jax_enable_x64', True)
        if fft_engine == 'auto':
            fft_engine = 'jaxdecomp' if ndim == 3 and jaxdecomp is not None else 'jax'
        self.__dict__.update(meshsize=meshsize, boxsize=boxsize, boxcenter=boxcenter, dtype=dtype, fft_engine=fft_engine)

    @property
    def sharding_mesh(self):
        return get_sharding_mesh()

    @property
    def hermitian(self):
        return self.fft_engine != 'jaxdecomp' and jnp.issubdtype(self.dtype, jnp.floating)

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
    def ndim(self):
        return len(self.meshsize)

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

    def rcoords(self, kind: str='position', sparse: bool | None=None):
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
        offset = None
        spacing = self.cellsize
        if kind == 'position':
            offset = self.boxcenter - self.boxsize / 2.
        if kind == 'index':
            spacing = 1
            kind = 'position'
        toret = fftfreq(self.meshsize, kind=kind, sparse=sparse, hermitian=False, spacing=spacing, sharding_mesh=self.sharding_mesh)
        if offset is not None:
            toret = tuple(tmp + off for off, tmp in zip(offset, toret))
        return toret

    def ccoords(self, kind: str='wavenumber', sparse: bool | None=None):
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
        spacing = self.kfun
        if kind == 'wavenumber':
            kind = 'separation'
        if kind == 'circular':
            spacing = 2. * np.pi / self.meshsize
            kind = 'separation'
        axis_order = None
        if self.fft_engine == 'jaxdecomp':
            axis_order = tuple(np.roll(np.arange(self.ndim), shift=len(self.sharding_mesh.axis_names)))
        return fftfreq(self.meshsize, kind=kind, sparse=sparse, hermitian=self.hermitian, spacing=spacing, axis_order=axis_order, sharding_mesh=self.sharding_mesh)

    xcoords = rcoords
    kcoords = ccoords

    def create(self, kind: str='real', fill: float | jax.Array | Callable=None):
        """
        Create mesh with these given attributes.

        Parameters
        ----------
        kind : str, default='real'
            Type of mesh to return, 'real' or 'complex'.

        fill : float, complex, default=None
            Optionally, value to fill mesh with.
            Empty as a default.

        Returns
        -------
        mesh : New mesh.
        """
        kind = {'real': RealMeshField, 'complex': ComplexMeshField}.get(kind, kind)
        name = kind.__name__.lower()
        shape = tuple(self.meshsize)
        if 'complex' in name:
            if self.hermitian:
                shape = shape[:-1] + (shape[-1] // 2 + 1,)
            if self.fft_engine == 'jaxdecomp':
                shape = tuple(np.roll(shape, shift=len(self.sharding_mesh.axis_names)))
        itemsize = jnp.zeros((), dtype=self.dtype).real.dtype.itemsize
        dtype = jnp.dtype('c{:d}'.format(2 * itemsize) if 'complex' in name else 'f{:d}'.format(itemsize))
        if callable(fill):
            fun = fill
        elif fill is None:
            fun = jnp.empty
        elif getattr(fill, 'shape', None) == shape:
            return kind(fill, attrs=self)
        else:
            fun = lambda shape, **kwargs: jnp.full(shape, fill, **kwargs)
        fun = partial(fun, shape=shard_shape(shape, sharding_mesh=self.sharding_mesh), dtype=dtype)
        if self.sharding_mesh.axis_names:
            fun = shard_map(fun, self.sharding_mesh, in_specs=tuple(), out_specs=P(*self.sharding_mesh.axis_names))
        return kind(fun(), attrs=self)

    def r2c(self, value):
        """FFT, from real to complex."""
        if self.fft_engine == 'jaxdecomp':
            value = jaxdecomp.fft.pfft3d(value)
        else:
            if self.hermitian:
                value = jnp.fft.rfftn(value)
            else:
                value = jnp.fft.fftn(value)
        return value

    def c2r(self, value):
        """FFT, from complex to real."""
        if self.fft_engine == 'jaxdecomp':
            value = jaxdecomp.fft.pifft3d(value)
            if jnp.issubdtype(self.dtype, jnp.floating):
                value = value.real.astype(self.dtype)
        else:
            if self.hermitian:
                value = jnp.fft.irfftn(value, s=tuple(self.meshsize))
            else:
                value = jnp.fft.ifftn(value)
        return value


# Note: I couldn't make it work properly with jax dataclass registration as tree_unflatten would pass all fields to __init__, which I don't want
@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class BaseMeshField(object):

    """Representation of a N-dim field as a N-dim array."""

    value: jax.Array = field(repr=False)
    attrs: MeshAttrs = None

    def __init__(self, value, *args, **kwargs):
        value = jnp.asarray(value)
        shape = value.shape
        if 'attrs' in kwargs:
            args = args + (kwargs.pop('attrs'),)
        if len(args):  # attrs is provided directly
            assert len(args) == 1, f"Do not understand args {args}"
            mattrs = args[0].clone(**kwargs)
        else:
            meshsize = kwargs.pop('meshsize', None)
            if meshsize is None: meshsize = shape
            meshsize = staticarray.fill(meshsize, len(shape), dtype='i4')
            mattrs = MeshAttrs(meshsize=meshsize, **kwargs)
        mattrs = mattrs.clone(dtype=value.dtype if mattrs.meshsize.prod(dtype='i8') == value.size else value.real.dtype)  # second option = hermitian
        self.__dict__.update(value=value, attrs=mattrs)

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
        """FFT, from real to complex."""
        return ComplexMeshField(self.attrs.r2c(self.value), self.attrs)

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

    def read(self, positions: jax.Array, resampler: str | Callable='cic', compensate: bool=False, pexchange: bool=True):
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
        sharding_mesh = self.attrs.sharding_mesh
        with_sharding = bool(sharding_mesh.axis_names)
        exchange = None
        if with_sharding and pexchange:
            exchange = exchange_particles(self.attrs, positions)
            positions = exchange(positions=positions)
        toret = _read(self, positions, resampler=resampler, compensate=compensate)
        if exchange is not None:
            toret = exchange.inverse(toret)
        return toret


@partial(jax.jit, static_argnames=['resampler', 'compensate'])
def _read(mesh, positions: jax.Array, resampler: str | Callable='cic', compensate: bool=False):
    """WARNING: in case of multiprocessing, positions and weights are assumed to be exchanged!"""

    resampler = getattr(resamplers, resampler, resampler)
    if isinstance(compensate, bool):
        if compensate:
            kernel_compensate = resampler.compensate
        else:
            kernel_compensate = None
    else:
        kernel_compensate = compensate

    if kernel_compensate is not None:
        mesh = mesh.r2c().apply(kernel_compensate).c2r()

    if isinstance(positions, ParticleField):
        positions = positions.positions

    value, attrs = mesh.value, mesh.attrs
    sharding_mesh = attrs.sharding_mesh
    with_sharding = bool(sharding_mesh.axis_names)

    positions = (positions + attrs.boxsize / 2. - attrs.boxcenter) / attrs.cellsize

    if with_sharding:
        kw_sharding = dict(halo_size=resampler.order, sharding_mesh=sharding_mesh)
        value, offset = pad_halo(value, **kw_sharding)
        value = exchange_halo(value, **kw_sharding)
        positions = positions + offset
        return shard_map(resampler.read, in_specs=(P(*sharding_mesh.axis_names),) * 2, out_specs=P(*sharding_mesh.axis_names))(value, positions)
    return resampler.read(value, positions)


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
        return self.attrs.kcoords(kind=kind, sparse=sparse)

    def c2r(self):
        """FFT, from complex to real. Return :class:`RealMeshField`."""
        return RealMeshField(self.attrs.c2r(self.value), attrs=self.attrs)

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


@default_sharding_mesh
def get_mesh_attrs(*positions: np.ndarray, meshsize: np.ndarray | int=None,
                   boxsize: np.ndarray | float=None, boxcenter: np.ndarray | float=None,
                   cellsize: np.ndarray | float=None, boxpad: np.ndarray | float=2.,
                   check: bool=False, dtype=None, sharding_mesh=None, **kwargs):
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
        axis = tuple(range(len(nonempty_positions[0].shape[:-1])))
        pos_min = min(jax.tree.map(partial(jnp.min, axis=axis), nonempty_positions))
        pos_max = max(jax.tree.map(partial(jnp.max, axis=axis), nonempty_positions))
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
            meshsize = np.ceil(boxsize / cellsize).astype('i4')
            sharding_mesh = get_sharding_mesh()
            if sharding_mesh.axis_names:
                ndevices = staticarray(sharding_mesh.devices.shape + (1,) * (meshsize.ndim - sharding_mesh.devices.ndim))
                meshsize = (meshsize + ndevices - 1) // ndevices * ndevices  # to make it a multiple of devices
            toret['meshsize'] = meshsize
            toret['boxsize'] = meshsize * cellsize  # enforce exact cellsize
        else:
            raise ValueError('meshsize (or cellsize) must be specified')
    else:
        meshsize = staticarray.fill(meshsize, ndim, dtype='i4')
        toret['meshsize'] = meshsize
    boxcenter = staticarray.fill(boxcenter, ndim, dtype=dtype)
    toret = dict(boxsize=boxsize, boxcenter=boxcenter) | toret
    return MeshAttrs(**{name: staticarray(value) for name, value in toret.items()}, dtype=dtype, **kwargs)


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

    def __init__(self, positions: jax.Array, weights: jax.Array | None=None, make_from=None, **kwargs):
        weights = jnp.ones_like(positions, shape=positions.shape[:-1]) if weights is None else weights
        if make_from == 'local':
            positions, weights = make_particles_from_local(positions, weights)
        assert '_cache_attrs' not in kwargs
        _cache_attrs = dict(kwargs.pop('attrs', {}))
        _cache_attrs.update({name: staticarray(value) if name in ['meshsize', 'boxsize', 'boxcenter'] and value is not None else value for name, value in kwargs.items()})
        self.__dict__.update(positions=positions, weights=weights, _attrs=None, _cache_attrs=_cache_attrs)

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
              compensate: bool=False, out: str='real', dtype=None, pexchange=True):
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
        attrs = self.attrs
        if dtype is not None: attrs = attrs.clone(dtype=dtype)
        sharding_mesh = attrs.sharding_mesh
        with_sharding = bool(sharding_mesh.axis_names)
        positions, weights = self.positions, self.weights
        if with_sharding and pexchange:
            exchange = exchange_particles(attrs, positions)
            positions, weights = exchange(positions=positions, weights=weights)
        return _paint(attrs, positions, weights, resampler=resampler, interlacing=interlacing, compensate=compensate, out=out)



@partial(jax.jit, static_argnames=['attrs', 'resampler',  'interlacing', 'compensate', 'out'])
def _paint(attrs, positions, weights=None, resampler: str | Callable='cic', interlacing: int=0, compensate: bool=False, out: str='real'):
    """WARNING: in case of multiprocessing, positions and weights are assumed to be exchanged!"""

    resampler = getattr(resamplers, resampler, resampler)
    interlacing = max(interlacing, 1)
    shifts = np.arange(interlacing) * 1. / interlacing

    sharding_mesh = attrs.sharding_mesh
    with_sharding = bool(sharding_mesh.axis_names)

    positions = (positions + attrs.boxsize / 2. - attrs.boxcenter) / attrs.cellsize

    if isinstance(compensate, bool):
        if compensate:
            kernel_compensate = resampler.compensate
        else:
            kernel_compensate = None
    else:
        kernel_compensate = compensate

    def ppaint(positions):
        mesh = attrs.create(kind='real', fill=0.)
        value = mesh.value
        w = None
        if weights is not None:
            w = weights.astype(value.dtype)
        if with_sharding:
            kw_sharding = dict(halo_size=resampler.order + interlacing, sharding_mesh=sharding_mesh)
            value, offset = pad_halo(value, **kw_sharding)
            positions = positions + offset
            value = shard_map(resampler.paint, in_specs=(P(*sharding_mesh.axis_names),) * 3, out_specs=P(*sharding_mesh.axis_names))(value, positions, w)
            value = exchange_halo(value, **kw_sharding)
            value = unpad_halo(value, **kw_sharding)
        else:
            value = resampler.paint(value, positions, w)
        return mesh.clone(value=value)

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


def _find_unique_edges(xvec, x0, xmin=0., xmax=np.inf):
    x2 = sum(xx**2 for xx in xvec).ravel()
    x2 = x2[(x2 >= xmin**2) & (x2 <= xmax**2)]
    _, index, counts = jnp.unique(np.int64(x2 / (0.5 * x0)**2 + 0.5), return_index=True, return_counts=True)
    x = jnp.sqrt(x2[index])
    tmp = (x[:-1] + x[1:]) / 2.
    edges = jnp.insert(tmp, jnp.array([0, len(tmp)]), jnp.array([tmp[0] - (x[1] - x[0]), tmp[-1] + (x[-1] - x[-2])]))
    return edges, x, counts


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class BinAttrs(object):

    edges: jax.Array = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    ibin: jax.Array = None
    ibin_zero: jax.Array = None
    wmodes: jax.Array = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, kind='complex', mode_oversampling: int=0, **kwargs):
        if not isinstance(mattrs, MeshAttrs):
            kind = 'complex' if 'complex' in mattrs.__class__.__name__.lower() else 'real'
            mattrs = mattrs.attrs
        hermitian = mattrs.hermitian
        if kind == 'real':
            vec = mattrs.xcoords(kind='separation', sparse=True, **kwargs)
            vec0 = mattrs.cellsize.min()
        else:
            vec = mattrs.kcoords(kind='separation', sparse=True, **kwargs)
            vec0 = mattrs.kfun.min()
        nonsingular = None
        if hermitian:
            shape = np.broadcast_shapes(*[xx.shape for xx in vec])

            def get_nonsingular(zvec):
                nonsingular = jnp.ones(shard_shape(shape, sharding_mesh=mattrs.sharding_mesh), dtype='i4')
                # Get the indices that have positive freq along symmetry axis = -1
                nonsingular += zvec > 0.
                return nonsingular.ravel()

            if self.sharding_mesh.axis_names:
                get_nonsingular = shard_map(get_nonsingular, mattrs.sharding_mesh,
                                            in_specs=P(*[axis if vec[-1].shape[iaxis] else None for iaxis, axis in enumerate(self.sharding_mesh.axis_names)]),
                                            out_specs=P(self.sharding_mesh.axis_names))
            nonsingular = get_nonsingular(vec[-1])
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            step = edges.get('step', None)
            if step is None:
                edges = _find_unique_edges(vec, vec0, xmin=edges.get('min', 0.), xmax=edges.get('max', np.inf))[0]
            else:
                edges = jnp.arange(edges.get('min', 0.), edges.get('max', vec0 * mattrs.meshsize / 2.), step)
        import itertools
        shifts = [np.arange(-mode_oversampling, mode_oversampling + 1)] * len(mattrs.meshsize)
        shifts = list(itertools.product(*shifts))
        ibin, nmodes, xsum = [], 0, 0

        def _get_bin_attrs(coords: jax.Array, edges: np.ndarray, weights: None | jax.Array=None):
            r"""Return bin index, binned number of modes and coordinates."""
            coords = coords.ravel()
            ibin = jnp.digitize(coords, edges, right=False)
            nmodes = jnp.bincount(ibin, weights=weights, length=len(edges) + 1)[1:-1]
            x = jnp.bincount(ibin, weights=coords if weights is None else coords * weights, length=len(edges) + 1)[1:-1]
            return ibin, nmodes, x
        
        get_bin_attrs = _get_bin_attrs
        if self.sharding_mesh.axis_names:

            @partial(shard_map, mesh=self.sharding_mesh,
                     in_specs=(P(*self.sharding_mesh.axis_names), P(None), P(self.sharding_mesh.axis_names)),
                     out_specs=(P(self.sharding_mesh.axis_names), P(None), P(None)))
            def get_bin_attrs(coords, edges, weights):
                r"""Return bin index, binned number of modes and coordinates."""
                ibin, nmodes, x = _get_bin_attrs(coords, edges, weights)
                nmodes = jax.lax.psum(nmodes, self.sharding_mesh.axis_names)
                x = jax.lax.psum(x, self.sharding_mesh.axis_names)
                return ibin, nmodes, x

        for shift in shifts:
            bin = get_bin_attrs(jnp.sqrt(sum((xx + ss)**2 for (xx, ss) in zip(vec, shift))), edges, nonsingular)
            ibin.append(bin[0])
            nmodes += bin[1]
            xsum += bin[2]
        self.__dict__.update(edges=edges, nmodes=nmodes / len(shifts), xavg=xsum / nmodes, ibin=ibin, wmodes=nonsingular)

    @property
    def sharding_mesh(self):
        return get_sharding_mesh()

    def __call__(self, mesh, antisymmetric=False, remove_zero=False):
        weights = self.wmodes
        value = getattr(mesh, 'value', mesh)
        if remove_zero:
            value = value.at[(0,) * value.ndim].set(0.)

        def _bincount(value, *ibin):
            value = value.ravel()
            if weights is not None:
                if antisymmetric: value = value.imag
                else: value = value.real
                value *= weights
            length = len(self.xavg) + 2
            count = lambda ib: jnp.bincount(ib, weights=value, length=length)
            if jnp.iscomplexobj(value):  # bincount much slower with complex numbers
                count = lambda ib: jnp.bincount(ib, weights=value.real, length=length) + 1j * jnp.bincount(ib, weights=value.imag, length=length)
            value = sum(count(ib) for ib in ibin)
            return value[1:-1] / len(ibin) / self.nmodes

        bincount = _bincount

        if self.sharding_mesh.axis_names:

            @partial(shard_map, mesh=self.sharding_mesh, in_specs=(P(*self.sharding_mesh.axis_names),) + (P(self.sharding_mesh.axis_names),) * len(self.ibin), out_specs=P(None))
            def bincount(value, *ibin):
                value = _bincount(value, *ibin)
                return jax.lax.psum(value, self.sharding_mesh.axis_names)

        return bincount(value, *self.ibin)

    def tree_flatten(self):
        return (asdict(self),), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        return new


# For functional programming interface

def c2r(mesh: ComplexMeshField) -> RealMeshField:
    return mesh.c2r()


def r2c(mesh: RealMeshField) -> ComplexMeshField:
    return mesh.r2c()


def apply(mesh: RealMeshField | ComplexMeshField,
          fn: Callable, sparse=False, **kwargs) -> RealMeshField | ComplexMeshField:
    return mesh.apply(fn, sparse=sparse, **kwargs)


def read(mesh: RealMeshField, positions: jax.Array, *args, **kwargs) -> jax.Array:
    return mesh.read(positions, *args, **kwargs)


def paint(mesh: ParticleField, *args, **kwargs) -> RealMeshField | ComplexMeshField:
    return mesh.paint(*args, **kwargs)


def bin(mesh, battrs=None, antisymmetric=False, remove_zero=False, *args, **kwargs):
    """Bin input mesh."""
    input_battrs = battrs is not None
    if not input_battrs:
        battrs = BinAttrs(mesh, *args, **kwargs)
    value = battrs(mesh, antisymmetric=antisymmetric, remove_zero=remove_zero)
    if not input_battrs:
        return battrs
    return value