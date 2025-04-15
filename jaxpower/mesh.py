import os
import operator
import functools
from functools import partial
from collections.abc import Callable
import numbers
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
        if isinstance(other, numbers.Number) and not self.shape:
            return self.view(np.ndarray) == other
        try:
            return np.all(self.view(np.ndarray) == np.asarray(other))
        except ValueError:
            return False

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
        count = len(jax.devices())
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


@partial(jax.jit, static_argnames=('func', 'shape', 'out_specs', 'sharding_mesh'))
@default_sharding_mesh
def create_sharded_array(func, shape, out_specs=None, sharding_mesh=None):
    if np.ndim(shape) == 0: shape = (shape,)
    shape = tuple(shape)
    if sharding_mesh.axis_names:
        if out_specs is None:
            out_specs = P(*sharding_mesh.axis_names)
        if not isinstance(out_specs, P):
            axis_names = []
            axis = out_specs
            if np.ndim(axis) == 0:
                out_specs = P((None,) * axis + (sharding_mesh.axis_names,))
            else:
                assert len(axis) <= len(sharding_mesh.axis_names), 'cannot have more array axes than device mesh'
                out_specs = [None] * np.max(axis)
                for iax, ax in enumerate(axis): out_specs[ax] = sharding_mesh.axis_names[iax]
                out_specs = P(*out_specs)
        shard_shape = jax.sharding.NamedSharding(sharding_mesh, spec=out_specs).shard_shape(shape)
        f = shard_map(partial(func, shape=shard_shape), mesh=sharding_mesh, in_specs=(), out_specs=out_specs)
    else:
        f = partial(func, shape=shape)
    return f()


@partial(jax.jit, static_argnames=('func', 'shape', 'out_specs', 'sharding_mesh'))
@default_sharding_mesh
def create_sharded_random(func, key, shape, out_specs=None, sharding_mesh=None):
    if np.ndim(shape) == 0: shape = (shape,)
    shape = tuple(shape)
    if sharding_mesh.axis_names:
        key = jnp.array(jax.random.split(key, sharding_mesh.devices.size))
        if out_specs is None:
            out_specs = P(*sharding_mesh.axis_names)
        if not isinstance(out_specs, P):
            axis_names = []
            axis = out_specs
            if np.ndim(axis) == 0:
                out_specs = P((None,) * axis + (sharding_mesh.axis_names,))
            else:
                assert len(axis) <= len(sharding_mesh.axis_names), 'cannot have more array axes than device mesh'
                out_specs = [None] * np.max(axis)
                for iax, ax in enumerate(axis): out_specs[ax] = sharding_mesh.axis_names[iax]
                out_specs = P(*out_specs)
        shard_shape = jax.sharding.NamedSharding(sharding_mesh, spec=out_specs).shard_shape(shape)
        f = shard_map(lambda key: func(key[0], shape=shard_shape), mesh=sharding_mesh, in_specs=P(sharding_mesh.axis_names,), out_specs=out_specs)
    else:
        f = partial(func, shape=shape)
    return f(key)


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
def mesh_shard_shape(shape: tuple, sharding_mesh: jax.sharding.Mesh=None):
    if not len(sharding_mesh.axis_names):
        return tuple(shape)
    return tuple(s // pdim for s, pdim in zip(shape, sharding_mesh.devices.shape)) + shape[sharding_mesh.devices.ndim:]


@default_sharding_mesh
@partial(jax.jit, static_argnames=['halo_size', 'sharding_mesh'])
def pad_halo(value, halo_size=0, sharding_mesh: jax.sharding.Mesh=None):
    pad_width = [(2 * halo_size if sharding_mesh.devices.shape[axis] > 1 else 0,) * 2 for axis in range(len(sharding_mesh.axis_names))]
    pad_width += [(0,) * 2] * len(value.shape[len(sharding_mesh.axis_names):])

    def pad(value):
        return jnp.pad(value, tuple(pad_width))

    offset = jnp.array([width[0] for width in pad_width])
    return shard_map(pad, mesh=sharding_mesh, in_specs=(P(*sharding_mesh.axis_names),), out_specs=P(*sharding_mesh.axis_names))(value), offset


@default_sharding_mesh
@partial(jax.jit, static_argnames=['halo_size', 'sharding_mesh'])
def unpad_halo(value, halo_size=0, sharding_mesh=None):

    def unpad(value):
        crop = []
        for axis in range(len(sharding_mesh.axis_names)):
            if sharding_mesh.devices.shape[axis] > 1:
                slices = [slice(None) for i in range(value.ndim)]
                slices[axis] = slice(2 * halo_size, 3 * halo_size)
                slices_add = list(slices)
                slices_add[axis] = slice(0, halo_size)
                value = value.at[tuple(slices)].add(value[tuple(slices_add)])
                slices[axis] = slice(value.shape[axis] - 3 * halo_size, value.shape[axis] - 2 * halo_size)
                slices_add = list(slices)
                slices_add[axis] = slice(value.shape[axis] - halo_size, value.shape[axis])
                value = value.at[tuple(slices)].add(value[tuple(slices_add)])
                crop.append(slice(2 * halo_size, value.shape[axis] - 2 * halo_size))
            else:
                crop.append(slice(None))
        return value[tuple(crop)]

    return shard_map(unpad, mesh=sharding_mesh, in_specs=(P(*sharding_mesh.axis_names),), out_specs=P(*sharding_mesh.axis_names))(value)


@default_sharding_mesh
@partial(jax.jit, static_argnames=['halo_size', 'sharding_mesh'])
def exchange_halo(value, halo_size=0, sharding_mesh: jax.sharding.Mesh=None):
    extents = (halo_size,) * len(sharding_mesh.axis_names)
    return jaxdecomp.halo_exchange(value, halo_extents=tuple(extents), halo_periods=(True,) * len(extents))


@default_sharding_mesh
def make_particles_from_local(positions, weights=None, sharding_mesh: jax.sharding.Mesh=None):

    per_host_positions, per_host_weights = positions, weights

    if not len(sharding_mesh.axis_names):
        if per_host_weights is None:
            return per_host_positions
        return per_host_positions, per_host_weights
    nlocal = len(jax.local_devices())
    sharding = jax.sharding.NamedSharding(sharding_mesh, P(sharding_mesh.axis_names,))
    per_host_size = np.repeat(per_host_positions.shape[0], nlocal)
    sizes = jax.make_array_from_process_local_data(sharding, per_host_size)
    per_host_sum = np.repeat(per_host_positions.sum(axis=0, keepdims=True), nlocal, axis=0)
    mean = jax.make_array_from_process_local_data(sharding, per_host_sum).sum(axis=0) / sizes.sum()
    max_size = (sizes.max().item() + nlocal - 1) // nlocal * nlocal
    pad_width = [(0, max_size - per_host_positions.shape[0])] + [(0, 0)] * (per_host_positions.ndim - 1)
    per_host_positions = np.append(per_host_positions, np.repeat(mean[None, :], pad_width[0][1], axis=0), axis=0)
    positions = jax.make_array_from_process_local_data(sharding, per_host_positions)#, global_shape=(global_size,) + per_host_positions.shape[-1:])
    if per_host_weights is None:
        return positions
    pad_width = [(0, max_size - per_host_weights.shape[0])] + [(0, 0)] * (per_host_weights.ndim - 1)
    per_host_weights = jnp.pad(per_host_weights, pad_width=pad_width, mode='constant', constant_values=0.)
    weights = jax.make_array_from_process_local_data(sharding, per_host_weights)
    return positions, weights


def _identity_fn(x):
  return x


def global_array_from_single_device_arrays(sharding, arrays, return_slices=False, pad=None):
    if pad is None:
        pad = np.nan
    if not callable(pad):
        constant_values = pad
        pad = lambda array, pad_width: jnp.pad(array, pad_width, mode='constant', constant_values=constant_values)
    ndevices = sharding.num_devices
    per_host_chunks = arrays
    ndim = per_host_chunks[0].ndim
    per_host_size = jnp.array([per_host_chunk.shape[0] for per_host_chunk in per_host_chunks])
    all_size = jax.make_array_from_process_local_data(sharding, per_host_size)
    all_size = jax.jit(_identity_fn, out_shardings=sharding.with_spec(P()))(all_size).addressable_data(0)
    max_size = all_size.max().item()
    if not np.all(all_size == max_size):
        per_host_chunks = [pad(per_host_chunk, [(0, max_size - per_host_chunk.shape[0])] + [(0, 0)] * (ndim - 1)) for per_host_chunk in per_host_chunks]
    global_shape = (max_size * ndevices,) + per_host_chunks[0].shape[1:]
    tmp = jax.make_array_from_single_device_arrays(global_shape, sharding, per_host_chunks)
    del per_host_chunks
    slices = [slice(j * max_size, j * max_size + all_size[j].item()) for j in range(ndevices)]
    if return_slices:
        return tmp, slices
    return tmp


def allgather_single_device_arrays(sharding, arrays, return_slices=False, **kwargs):
    tmp, slices = global_array_from_single_device_arrays(sharding, arrays, return_slices=True, **kwargs)
    tmp = jax.jit(_identity_fn, out_shardings=sharding.with_spec(P()))(tmp).addressable_data(0)  # replicated accross all devices
    tmp = jnp.concatenate([tmp[sl] for sl in slices], axis=0)
    if return_slices:
        sizes = np.cumsum([0] + [sl.stop - sl.start for sl in slices])
        slices = [slice(start, stop) for start, stop in zip(sizes[:-1], sizes[1:])]
        return tmp, slices
    return tmp


def exchange_array(array, device, pad=jnp.nan, return_indices=False):
    # Exchange array along the first (0) axis
    # TODO: generalize to any axis
    sharding = array.sharding
    ndevices = sharding.num_devices
    per_host_arrays = [_.data for _ in array.addressable_shards]
    per_host_devices = [_.data for _ in device.addressable_shards]
    devices = sharding.mesh.devices.ravel().tolist()
    local_devices = [_.device for _ in per_host_arrays]
    per_host_final_arrays = [None] * len(local_devices)
    ndim = array.ndim
    per_host_indices = [[] for i in range(len(local_devices))]
    slices = [None] * ndevices

    for idevice in range(ndevices):
        # single-device arrays
        per_host_chunks = []
        for ilocal_device, (per_host_array, per_host_device, local_device) in enumerate(zip(per_host_arrays, per_host_devices, local_devices)):
            mask_idevice = per_host_device == idevice
            per_host_chunks.append(jax.device_put(per_host_array[mask_idevice], local_device, donate=True))
            if return_indices: per_host_indices[ilocal_device].append(jax.device_put(np.flatnonzero(mask_idevice), local_device, donate=True))
        tmp, slices[idevice] = allgather_single_device_arrays(sharding, per_host_chunks, return_slices=True)
        del per_host_chunks
        if devices[idevice] in local_devices:
            per_host_final_arrays[local_devices.index(devices[idevice])] = jax.device_put(tmp, devices[idevice], donate=True)
        del tmp

    final = global_array_from_single_device_arrays(sharding, per_host_final_arrays, pad=pad)
    if return_indices:
        for ilocal_device, local_device in enumerate(local_devices):
            per_host_indices[ilocal_device] = jax.device_put(jnp.concatenate(per_host_indices[ilocal_device]), local_device, donate=True)
        return final, (per_host_indices, slices)
    return final


def exchange_inverse(array, indices):
    # Reciprocal exchange
    indices = per_host_indices, slices
    sharding = array.sharding
    ndevices = sharding.num_devices
    per_host_arrays = [_.data for _ in array.addressable_shards]
    devices = sharding.mesh.devices.ravel().tolist()
    local_devices = [_.device for _ in per_host_arrays]
    per_host_final_arrays = [None] * len(local_devices)

    for idevice in range(ndevices):
        per_host_chunks = []
        for ilocal_device, (per_host_array, local_device) in enumerate(zip(per_host_arrays, local_devices)):
            sl = slices[devices.index(local_device)][idevice]
            per_host_chunks.append(jax.device_put(per_host_array[sl], local_device, donate=True))
        tmp = allgather_single_device_arrays(sharding, per_host_chunks, return_slices=False)
        del per_host_chunks
        if devices[idevice] in local_devices:
            ilocal_device = local_devices.index(devices[idevice])
            indices = per_host_indices[ilocal_device]
            tmp = jax.device_put(tmp, devices[idevice], donate=True)
            tmp = jnp.empty_like(tmp).at[indices].set(tmp)
            per_host_final_arrays[local_devices.index(devices[idevice])] = jax.device_put(tmp, devices[idevice], donate=True)
        del tmp

    return global_array_from_single_device_arrays(sharding, per_host_final_arrays)


@default_sharding_mesh
def exchange_particles(attrs, positions: jax.Array=None, return_inverse=False, sharding_mesh=None):
    sharding = positions.sharding

    if not len(sharding_mesh.axis_names):

        def exchange(values, pad=0.):
            return values

        def inverse(values):
            return values

        if return_inverse:
            return positions, exchange, inverse
        return positions, exchange

    shape_devices = staticarray(sharding_mesh.devices.shape + (1,) * (attrs.ndim - sharding_mesh.devices.ndim))
    size_devices = shape_devices.prod(dtype='i4')

    idx_out_devices = ((positions + attrs.boxsize / 2. - attrs.boxcenter) % attrs.boxsize) / (attrs.boxsize / shape_devices)
    idx_out_devices = jnp.ravel_multi_index(jnp.unstack(jnp.floor(idx_out_devices).astype('i4'), axis=-1), tuple(shape_devices))
    shifts = jnp.meshgrid(*[np.arange(s) * attrs.boxsize[i] / s for i, s in enumerate(shape_devices)], indexing='ij', sparse=False)
    shifts = jnp.stack(shifts, axis=-1).reshape(-1, len(shifts))

    mean = positions.mean(axis=0)
    pad = lambda array, pad_width: jnp.append(array, jnp.repeat(jax.device_get(mean)[None, :], pad_width[0][1], axis=0), axis=0)
    positions = exchange_array(positions, idx_out_devices, pad=pad, return_indices=return_inverse)

    def f(positions, idevice):
        return positions - shifts[idevice[0]]

    positions = shard_map(f, mesh=sharding_mesh, in_specs=(sharding.spec, sharding.spec), out_specs=sharding.spec)(positions, jnp.arange(size_devices))

    def exchange(values, pad=0.):
        return exchange_array(values, idx_out_devices, pad=pad, return_indices=False)

    if return_inverse:
        positions, indices = positions

        def inverse(values):
            return exchange_inverse(values, indices)

        return positions, exchange, inverse
    return positions, exchange


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
            axis_order = tuple(np.roll(np.arange(self.ndim), shift=2))
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
            fun = partial(jnp.empty, dtype=dtype)
        elif getattr(fill, 'shape', None) == shape:
            return kind(fill, attrs=self)
        else:
            fun = lambda shape: jnp.full(shape, fill, dtype=dtype)
        value = create_sharded_array(fun, shape, out_specs=P(*self.sharding_mesh.axis_names), sharding_mesh=self.sharding_mesh)
        return kind(value, attrs=self)

    def r2c(self, value):
        """FFT, from real to complex."""
        if self.fft_engine == 'jaxdecomp':
            value = jax.lax.with_sharding_constraint(value, jax.sharding.NamedSharding(self.sharding_mesh, spec=P(*self.sharding_mesh.axis_names)))
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
            value = jax.lax.with_sharding_constraint(value, jax.sharding.NamedSharding(self.sharding_mesh, spec=P(*self.sharding_mesh.axis_names)))
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
        inverse = None
        if with_sharding and pexchange:
            positions, exchange, inverse = exchange_particles(self.attrs, positions, return_inverse=True)
        toret = _read(self, positions, resampler=resampler, compensate=compensate)
        if inverse is not None:
            toret = inverse(toret)
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
        pos_min = jnp.array([jnp.min(p, axis=axis) for p in nonempty_positions]).min(axis=0)
        pos_max = jnp.array([jnp.max(p, axis=axis) for p in nonempty_positions]).max(axis=0)
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

    rdtype = jnp.zeros((), dtype=dtype).real.dtype
    boxsize = staticarray.fill(boxsize, ndim, dtype=rdtype)
    toret = dict()
    if meshsize is None:
        if cellsize is not None:
            cellsize = staticarray.fill(cellsize, ndim, dtype=rdtype)
            meshsize = np.ceil(boxsize / cellsize).astype('i4')
            if sharding_mesh.axis_names:
                same = np.all(meshsize.view(np.ndarray) == meshsize[0])
                if same:
                    # FIXME
                    shape_devices = max(sharding_mesh.devices.shape)
                else:
                    shape_devices = staticarray(sharding_mesh.devices.shape + (1,) * (ndim - sharding_mesh.devices.ndim))
                meshsize = (meshsize + shape_devices - 1) // shape_devices * shape_devices  # to make it a multiple of devices
            toret['meshsize'] = meshsize
            toret['boxsize'] = meshsize * cellsize  # enforce exact cellsize
        else:
            raise ValueError('meshsize (or cellsize) must be specified')
    else:
        meshsize = staticarray.fill(meshsize, ndim, dtype='i4')
        toret['meshsize'] = meshsize
    boxcenter = staticarray.fill(boxcenter, ndim, dtype=rdtype)
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


@default_sharding_mesh
@partial(jax.jit, static_argnames=['axis', 'sharding_mesh'])
def _local_concatenate(arrays, axis=0, sharding_mesh: jax.sharding.Mesh=None):

    def f(arrays):
        return jnp.concatenate(arrays, axis=axis)

    if sharding_mesh.axis_names:
        f = shard_map(f, mesh=sharding_mesh, in_specs=P(sharding_mesh.axis_names), out_specs=P(sharding_mesh.axis_names))

    return f(arrays)


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
    def concatenate(cls, others, weights=None, local=False, **kwargs):
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
            if local: value = _local_concatenate(value, axis=0)
            else: value = jnp.concatenate(value, axis=0)
            gather[name] = value
        return cls(**gather, **cache_attrs)

    def __add__(self, other):
        if isinstance(other, ParticleField):
            return self.concatenate([self, other], [1, 1])
        else:
            raise RuntimeError(f"Type of `other` {type(other)} not understood.")

    def __sub__(self, other):
        if isinstance(other, ParticleField):
            return self.concatenate([self, other], [1, -1])
        else:
            raise RuntimeError(f"Type of `other` {type(other)} not understood.")

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
            positions, exchange = exchange_particles(attrs, positions)
            weights = exchange(weights)
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

    _paint = resampler.paint
    if with_sharding:
        _paint = shard_map(_paint, mesh=sharding_mesh, in_specs=(P(*sharding_mesh.axis_names), P(sharding_mesh.axis_names), P(sharding_mesh.axis_names)), out_specs=P(*sharding_mesh.axis_names))

    def ppaint(positions):
        mesh = attrs.create(kind='real', fill=0.)
        value = mesh.value
        w = None
        if weights is not None:
            w = weights.astype(value.dtype)
        positions = positions % attrs.meshsize
        if with_sharding:
            kw_sharding = dict(halo_size=resampler.order + interlacing, sharding_mesh=sharding_mesh)
            #print(value.shape)
            value, offset = pad_halo(value, **kw_sharding)
            #print('padded', value.shape, offset)
            positions = positions + offset
            #print(positions.min(axis=0), positions.max(axis=0), value.shape)
            value = _paint(value, positions, w)
            #hs = kw_sharding['halo_size']
            #print(value[:2 * hs].std(), value[-2 * hs:].std(), value[:, :2 * hs].std(), value[:, -2 * hs:].std())
            value = exchange_halo(value, **kw_sharding)
            #print('exchanged', value[:2 * hs].std(), value[-2 * hs:].std(), value[:, :2 * hs].std(), value[:, -2 * hs:].std())
            value = unpad_halo(value, **kw_sharding)
            #print('unpadded', value.shape)
        else:
            value = _paint(value, positions, w)
        return mesh.clone(value=value)

    if interlacing <= 1:
        toret = ppaint(positions)
        if kernel_compensate is not None:
            toret = toret.r2c().apply(kernel_compensate)
            if out == 'real': toret = toret.c2r()
        elif out != 'real': toret = toret.r2c()
    else:
        @jax.checkpoint
        def f(carry, shift):
            def kernel_shift(value, kvec):
                kvec = sum(kvec)
                return value * jnp.exp(shift * 1j * kvec) / interlacing

            carry += ppaint(positions + shift).r2c().apply(kernel_shift, kind='circular')
            return carry, shift
        toret = jax.lax.scan(f, init=attrs.create(kind='complex', fill=0.), xs=shifts)[0]
        if kernel_compensate is not None:
            toret = toret.apply(kernel_compensate)
        if out == 'real': toret = toret.c2r()
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


@default_sharding_mesh
@partial(jax.jit, static_argnames=('sharding_mesh',))
def _get_hermitian_weights(coords, sharding_mesh=None):
    shape = np.broadcast_shapes(*[xx.shape for xx in coords])

    def get_nonsingular(zvec):
        nonsingular = jnp.ones(shape, dtype='i4')
        # Get the indices that have positive freq along symmetry axis = -1
        nonsingular += zvec > 0.
        return nonsingular.ravel()

    if sharding_mesh.axis_names:
        shape = jax.sharding.NamedSharding(sharding_mesh, spec=P(*sharding_mesh.axis_names)).shard_shape(shape)
        get_nonsingular = shard_map(get_nonsingular, mesh=sharding_mesh,
                                    in_specs=P(*[axis if coords[-1].shape[iaxis] else None for iaxis, axis in enumerate(sharding_mesh.axis_names)]),
                                    out_specs=P(sharding_mesh.axis_names))
    return get_nonsingular(coords[-1])


@default_sharding_mesh
@partial(jax.jit, static_argnames=('edges', 'linear', 'sharding_mesh'))  # linear saves memory
def _get_bin_attrs(coords, edges: staticarray, weights: None | jax.Array=None, linear: bool=False, sharding_mesh=None):

    def _get_attrs(coords, edges, weights):
        r"""Return bin index, binned number of modes and coordinates."""
        coords = coords.ravel()
        if linear: ibin = jnp.clip(jnp.floor((coords - edges[0]) / (edges[1] - edges[0])).astype('i4') + 1, 0, len(edges))
        else: ibin = jnp.digitize(coords, edges, right=False)
        x = jnp.bincount(ibin, weights=coords if weights is None else coords * weights, length=len(edges) + 1)[1:-1]
        del coords
        nmodes = jnp.bincount(ibin, weights=weights, length=len(edges) + 1)[1:-1]
        return ibin, nmodes, x

    get_attrs = _get_attrs

    if sharding_mesh.axis_names:

        def get_attrs(coords, edges, weights):
            r"""Return bin index, binned number of modes and coordinates."""
            ibin, nmodes, x = _get_attrs(coords, edges, weights)
            nmodes = jax.lax.psum(nmodes, sharding_mesh.axis_names)
            x = jax.lax.psum(x, sharding_mesh.axis_names)
            return ibin, nmodes, x

        get_attrs = shard_map(get_attrs, mesh=sharding_mesh,
                    in_specs=(P(*sharding_mesh.axis_names), P(None), P(sharding_mesh.axis_names)),
                    out_specs=(P(sharding_mesh.axis_names), P(None), P(None)))

    return get_attrs(coords, edges, weights)


@default_sharding_mesh
@partial(jax.jit, static_argnames=('length', 'antisymmetric', 'sharding_mesh'))
def _bincount(ibin, value, weights=None, length=None, antisymmetric=False, sharding_mesh=None):

    def _count(value, *ibin):
        value = value.ravel()
        if weights is not None:
            if antisymmetric: value = value.imag
            else: value = value.real
            value *= weights
        count = lambda ib: jnp.bincount(ib, weights=value, length=length)
        if jnp.iscomplexobj(value):  # bincount much slower with complex numbers
            count = lambda ib: jnp.bincount(ib, weights=value.real, length=length) + 1j * jnp.bincount(ib, weights=value.imag, length=length)
        value = sum(count(ib) for ib in ibin)
        return value[1:-1] / len(ibin)

    count = _count

    if sharding_mesh.axis_names:

        def count(value, *ibin):
            value = _count(value, *ibin)
            return jax.lax.psum(value, sharding_mesh.axis_names)

        count = shard_map(count, mesh=sharding_mesh, in_specs=(P(*sharding_mesh.axis_names),) + (P(sharding_mesh.axis_names),) * len(ibin), out_specs=P(None))

    return count(value, *ibin)



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
        wmodes = None
        if hermitian:
            wmodes = _get_hermitian_weights(vec, sharding_mesh=None)
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            step = edges.get('step', None)
            if step is None:
                edges = _find_unique_edges(vec, vec0, xmin=edges.get('min', 0.), xmax=edges.get('max', np.inf))[0]
            else:
                edges = np.arange(edges.get('min', 0.), edges.get('max', vec0 * mattrs.meshsize / 2.), step)
        import itertools
        shifts = [jnp.arange(-mode_oversampling, mode_oversampling + 1)] * len(mattrs.meshsize)
        shifts = list(itertools.product(*shifts))
        ibin, nmodes, xsum = [], 0, 0
        edges = staticarray(edges)
        linear = np.allclose(edges, np.linspace(edges[0], edges[-1], len(edges)), atol=1e-8, rtol=1e-8)
        for shift in shifts:
            coords = jnp.sqrt(sum((xx + ss)**2 for (xx, ss) in zip(vec, shift)))
            bin = _get_bin_attrs(coords, edges, wmodes, linear=linear)
            del coords
            ibin.append(bin[0])
            nmodes += bin[1]
            xsum += bin[2]
        self.__dict__.update(edges=edges, nmodes=nmodes / len(shifts), xavg=xsum / nmodes, ibin=ibin, wmodes=wmodes)

    @property
    def sharding_mesh(self):
        return get_sharding_mesh()

    def __call__(self, mesh, antisymmetric=False, remove_zero=False):
        weights = self.wmodes
        value = getattr(mesh, 'value', mesh)
        if remove_zero:
            value = value.at[(0,) * value.ndim].set(0.)
        return _bincount(self.ibin, value, weights=weights, length=len(self.xavg) + 2, antisymmetric=antisymmetric) / self.nmodes

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