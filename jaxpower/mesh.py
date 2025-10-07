import os
import operator
import functools
import itertools
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


def _np_array_fill(fill: float | np.ndarray, shape: int | tuple, **kwargs):
    fill = np.array(fill)
    toret = np.empty_like(fill, shape=shape, **kwargs)
    toret[...] = fill
    return toret


def _jnp_array_fill(fill: float | np.ndarray, shape: int | tuple, **kwargs):
    fill = jnp.array(fill)
    toret = jnp.empty_like(fill, shape=shape, **kwargs).at[...].set(fill)
    return toret


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
        return cls(_np_array_fill(fill, shape, **kwargs))


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
        meshsize = _np_array_fill(meshsize, ndim, dtype='i4')

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
            axis = out_specs
            if np.ndim(axis) == 0:
                out_specs = P((None,) * axis + tuple(sharding_mesh.axis_names))
            else:
                assert len(axis) <= len(sharding_mesh.axis_names), 'cannot have more array axes than device mesh'
                out_specs = [None] * (1 + np.max(axis))
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
            hermitian: bool=False, spacing: np.ndarray | float=1., dtype=None,
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
    spacing = _jnp_array_fill(spacing, ndim)

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
    if dtype is not None: toret = [tt.astype(dtype) for tt in toret]
    toret = _get_freq_mesh(*toret, sparse=sparse, sharding_mesh=sharding_mesh)
    return tuple(toret[axis_order.index(axis)] for axis in range(len(toret)))


@default_sharding_mesh
def mesh_shard_shape(shape: tuple, sharding_mesh: jax.sharding.Mesh=None):
    if not len(sharding_mesh.axis_names):
        return tuple(shape)
    return tuple(s // pdim for s, pdim in zip(shape, sharding_mesh.devices.shape)) + shape[sharding_mesh.devices.ndim:]


@default_sharding_mesh
@partial(jax.jit, static_argnames=['halo_size', 'factor', 'sharding_mesh'])
def pad_halo(value, halo_size=0, factor=2, sharding_mesh: jax.sharding.Mesh=None):
    pad_width = [(factor * halo_size if sharding_mesh.devices.shape[axis] > 1 else 0,) * 2 for axis in range(len(sharding_mesh.axis_names))]
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
def make_array_from_global_data(array, sharding_mesh: jax.sharding.Mesh=None):
    shape = array.shape
    sharding = jax.sharding.NamedSharding(sharding_mesh, P(sharding_mesh.axis_names,))
    return jax.make_array_from_callback(shape, sharding, lambda index: array[index])



@default_sharding_mesh
def make_array_from_process_local_data(per_host_array, per_host_size=None, pad=0., sharding_mesh: jax.sharding.Mesh=None):

    if not len(sharding_mesh.axis_names):
        return per_host_array

    nlocal = len(jax.local_devices())
    sharding = jax.sharding.NamedSharding(sharding_mesh, P(sharding_mesh.axis_names,))

    sizes = None
    def get_sizes():
        return jax.make_array_from_process_local_data(sharding, np.repeat(per_host_array.shape[0], nlocal))

    if not callable(pad):
        if isinstance(pad, str):
            if pad == 'global_mean':
                if sizes is None:
                    sizes = get_sizes()
                per_host_sum = np.repeat(per_host_array.sum(axis=0, keepdims=True), nlocal, axis=0)
                pad = jax.make_array_from_process_local_data(sharding, per_host_sum).sum(axis=0) / sizes.sum()[None, ...]
            elif pad == 'mean':
                pad = np.mean(per_host_array, axis=0)[None, ...]
            else:
                raise ValueError('mean or global_mean supported only')
        constant_values = pad
        pad = lambda array, pad_width: np.append(array, np.repeat(constant_values, pad_width[0][1], axis=0), axis=0)

    if per_host_size is None:
        if sizes is None: sizes = get_sizes()
        per_host_size = (sizes.max().item() + nlocal - 1) // nlocal * nlocal

    pad_width = [(0, per_host_size - per_host_array.shape[0])] + [(0, 0)] * (per_host_array.ndim - 1)
    per_host_array = pad(per_host_array, pad_width=pad_width)
    return jax.make_array_from_process_local_data(sharding, per_host_array)


@default_sharding_mesh
def make_particles_from_local(positions, weights=None, sharding_mesh: jax.sharding.Mesh=None):
    positions = make_array_from_process_local_data(positions, pad='mean', sharding_mesh=sharding_mesh)
    if weights is None:
        return positions
    weights = make_array_from_process_local_data(weights, pad=0., sharding_mesh=sharding_mesh)
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
    # Line below fails with no attribute .with_spec for jax 0.6.2
    #all_size = jax.jit(_identity_fn, out_shardings=sharding.with_spec(P()))(all_size).addressable_data(0)
    all_size = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding.mesh, P()))(all_size).addressable_data(0)
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
    # Line below fails with no attribute .with_spec for jax 0.6.2
    #tmp = jax.jit(_identity_fn, out_shardings=sharding.with_spec(P()))(tmp).addressable_data(0)  # replicated accross all devices
    tmp = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding.mesh, P()))(tmp).addressable_data(0)
    tmp = jnp.concatenate([tmp[sl] for sl in slices], axis=0)
    if return_slices:
        sizes = np.cumsum([0] + [sl.stop - sl.start for sl in slices])
        slices = [slice(start, stop) for start, stop in zip(sizes[:-1], sizes[1:])]
        return tmp, slices
    return tmp


def _exchange_array_jax(array, device, pad=jnp.nan, return_indices=False):
    # Exchange array along the first (0) axis
    # TODO: generalize to any axis
    sharding = array.sharding
    ndevices = sharding.num_devices
    per_host_arrays = [_.data for _ in array.addressable_shards]
    per_host_devices = [_.data for _ in device.addressable_shards]
    devices = sharding.mesh.devices.ravel().tolist()
    local_devices = [_.device for _ in per_host_arrays]
    per_host_final_arrays = [None] * len(local_devices)
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


def _exchange_inverse_jax(array, indices):
    # Reciprocal exchange
    per_host_indices, slices = indices
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
def _exchange_particles_jax(attrs, positions: jax.Array | np.ndarray=None, return_inverse=False, sharding_mesh=None):

    if not len(sharding_mesh.axis_names):

        def exchange(values, pad=0.):
            return values

        def inverse(values):
            return values

        if return_inverse:
            return positions, exchange, inverse
        return positions, exchange

    shape_devices = np.array(sharding_mesh.devices.shape + (1,) * (attrs.ndim - sharding_mesh.devices.ndim))
    #size_devices = shape_devices.prod(dtype='i4')

    idx_out_devices = ((positions + attrs.boxsize / 2. - attrs.boxcenter) % attrs.boxsize) / (attrs.boxsize / shape_devices)
    idx_out_devices = jnp.ravel_multi_index(jnp.unstack(jnp.floor(idx_out_devices).astype('i4'), axis=-1), tuple(shape_devices))

    mean = positions.mean(axis=0)
    pad = lambda array, pad_width: jnp.append(array, jnp.repeat(jax.device_get(mean)[None, :], pad_width[0][1], axis=0), axis=0)
    positions = _exchange_array_jax(positions, idx_out_devices, pad=pad, return_indices=return_inverse)
    if return_inverse:
        positions, indices = positions

    def exchange(values, pad=0.):
        return _exchange_array_jax(values, idx_out_devices, pad=pad, return_indices=False)

    if return_inverse:

        def inverse(values):
            return _exchange_inverse_jax(values, indices)

        return positions, exchange, inverse
    return positions, exchange



def _send_mpi(data, dest, tag=0, mpicomm=None):
    """
    Send input array ``data`` to process ``dest``.

    Parameters
    ----------
    data : array
        Array to send.

    dest : int
        Rank of process to send array to.

    tag : int, default=0
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.
    """
    from mpi4py import MPI

    data = np.asarray(data)
    shape, dtype = (data.shape, data.dtype)
    data = np.ascontiguousarray(data)

    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in send; please specify specific data type')

    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    mpicomm.send((shape, dtype), dest=dest, tag=tag)
    mpicomm.Send([data, dt], dest=dest, tag=tag)
    dt.Free()


def _recv_mpi(source=None, tag=None, mpicomm=None):
    """
    Receive array from process ``source``.

    Parameters
    ----------
    source : int, default=MPI.ANY_SOURCE
        Rank of process to receive array from.

    tag : int, default=MPI.ANY_TAG
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.

    Returns
    -------
    data : array
    """
    from mpi4py import MPI
    if source is None: source = MPI.ANY_SOURCE
    if tag is None: tag = MPI.ANY_TAG

    shape, dtype = mpicomm.recv(source=source, tag=tag)
    data = np.zeros(shape, dtype=dtype)

    duplicity = np.prod(shape[1:], dtype='intp')
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    mpicomm.Recv([data, dt], source=source, tag=tag)
    dt.Free()
    return data


def _exchange_array_mpi(array, process, return_indices=False, mpicomm=None):
    # Exchange array along the first (0) axis
    # TODO: generalize to any axis
    per_host_indices, per_host_final_arrays, slices = [], [], []
    tag = 42
    start = 0
    for irank in range(mpicomm.size):
        mask_irank = process == irank
        tmp = array[mask_irank]
        if mpicomm.rank != irank:
            _send_mpi(tmp, dest=irank, tag=tag, mpicomm=mpicomm)
            tmp = _recv_mpi(source=irank, tag=tag, mpicomm=mpicomm)
        per_host_final_arrays.append(tmp)
        if return_indices:
            per_host_indices.append(np.flatnonzero(mask_irank))
            slices.append(slice(start, start + tmp.shape[0]))
            start += tmp.shape[0]
    final = np.concatenate(per_host_final_arrays, axis=0)
    del per_host_final_arrays
    if return_indices:
        per_host_indices = np.concatenate(per_host_indices, axis=0)
        return final, (per_host_indices, slices)
    return final


def _exchange_array_mpi(array, process, return_indices=False, mpicomm=None):
    # Exchange array along the first (0) axis
    # TODO: generalize to any axis
    per_host_indices, per_host_final_arrays, slices = [], [], []
    tag = 42

    for irank in range(mpicomm.size):
        mask_irank = process == irank
        if return_indices:
            per_host_indices.append(np.flatnonzero(mask_irank))
        tmp = array[mask_irank]
        if mpicomm.rank != irank:
            _send_mpi(tmp, dest=irank, tag=tag, mpicomm=mpicomm)
        if mpicomm.rank == irank:
            start = 0
            for source in range(mpicomm.size):
                if source == irank:
                    per_host_final_arrays.append(tmp)
                else:
                    per_host_final_arrays.append(_recv_mpi(source=source, tag=tag, mpicomm=mpicomm))
                if return_indices:
                    slices.append(slice(start, start + per_host_final_arrays[-1].shape[0]))
                    start += per_host_final_arrays[-1].shape[0]
    final = np.concatenate(per_host_final_arrays, axis=0)
    del per_host_final_arrays
    if return_indices:
        per_host_indices = np.concatenate(per_host_indices, axis=0)
        return final, (per_host_indices, slices)
    return final


def _exchange_inverse_mpi(array, indices, mpicomm=None):
    # Reciprocal exchange
    per_host_indices, slices = indices

    def reorder(array, indices):
        toret = np.empty_like(array)
        toret[indices] = array
        return toret

    per_host_final_arrays = []
    tag = 42
    for irank in range(mpicomm.size):
        tmp = array[slices[irank]]
        if mpicomm.rank != irank:
            _send_mpi(tmp, dest=irank, tag=tag, mpicomm=mpicomm)
        if mpicomm.rank == irank:
            for source in range(mpicomm.size):
                if source == irank:
                    per_host_final_arrays.append(tmp)
                else:
                    per_host_final_arrays.append(_recv_mpi(source=source, tag=tag, mpicomm=mpicomm))
    final = reorder(np.concatenate(per_host_final_arrays, axis=0), per_host_indices)
    return final


@default_sharding_mesh
def _exchange_particles_mpi(attrs, positions: jax.Array | np.ndarray=None, return_inverse=False, sharding_mesh=None, return_type='nparray', mpicomm=None):

    shape_devices = np.array(sharding_mesh.devices.shape + (1,) * (attrs.ndim - sharding_mesh.devices.ndim))
    idx_out_devices = ((positions + attrs.boxsize / 2. - attrs.boxcenter) % attrs.boxsize) / (attrs.boxsize / shape_devices)
    idx_out_devices = np.ravel_multi_index(tuple(np.floor(idx_out_devices).astype('i4').T), tuple(shape_devices))
    shifts = np.meshgrid(*[np.arange(s) * attrs.boxsize[i] / s for i, s in enumerate(shape_devices)], indexing='ij', sparse=False)
    shifts = np.stack(shifts, axis=-1).reshape(-1, len(shifts))

    positions = _exchange_array_mpi(positions, idx_out_devices, return_indices=return_inverse, mpicomm=mpicomm)

    if return_inverse:
        positions, indices = positions

    if return_type == 'jax':
        positions = make_array_from_process_local_data(positions, pad='mean', sharding_mesh=sharding_mesh)

    def exchange(values, pad=0.):
        per_host_array = _exchange_array_mpi(values, idx_out_devices, return_indices=False, mpicomm=mpicomm)
        if return_type == 'jax':
            return make_array_from_process_local_data(per_host_array, pad=pad, sharding_mesh=sharding_mesh)
        return per_host_array

    if return_inverse:
        positions, indices = positions

        def inverse(values):
            return _exchange_inverse_mpi(values, indices, mpicomm=mpicomm)

        return positions, exchange, inverse
    return positions, exchange

def _get_distributed_backend(array, backend='auto', **kwargs):

    def get_mpicomm():
        mpicomm = None
        try: from mpi4py import MPI
        except: MPI = None
        if MPI is not None:
            mpicomm = MPI.COMM_WORLD
        return mpicomm

    if backend == 'auto' and isinstance(array, np.ndarray):
        if jax.distributed.is_initialized():
            mpicomm = get_mpicomm()
            if mpicomm is not None and mpicomm.size == len(jax.devices()):
                backend = 'mpi'
                kwargs.setdefault('mpicomm', mpicomm)
    if backend == 'auto':
        backend = 'jax'
    if backend == 'mpi':
        kwargs.setdefault('mpicomm', get_mpicomm())
    return backend, kwargs


@default_sharding_mesh
def exchange_particles(attrs, positions: jax.Array | np.ndarray=None, return_inverse=False, sharding_mesh=None, backend='auto', **kwargs):

    if not len(sharding_mesh.axis_names):

        def exchange(values, pad=0.):
            return values

        def inverse(values):
            return values

        if return_inverse:
            return positions, exchange, inverse
        return positions, exchange

    backend, kwargs = _get_distributed_backend(positions, backend=backend, **kwargs)
    if backend == 'jax':
        kwargs.pop('return_type', None)
        return _exchange_particles_jax(attrs, positions, return_inverse=return_inverse, sharding_mesh=sharding_mesh, **kwargs)
    return _exchange_particles_mpi(attrs, positions, return_inverse=return_inverse, sharding_mesh=sharding_mesh, **kwargs)


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class MeshAttrs(object):

    meshsize: staticarray = None
    boxsize: jax.Array = None
    boxcenter: jax.Array = 0.
    dtype: Any = None
    fft_engine: str = None

    def __init__(self, meshsize=None, boxsize=None, boxcenter=0., dtype=float, fft_engine='auto'):
        ndim = _get_ndim(*[meshsize, boxsize, boxcenter])
        if meshsize is not None:  # meshsize may not be provided (e.g. for particles) and it is fine
            meshsize = staticarray.fill(meshsize, ndim, dtype='i4')
            if boxsize is None: boxsize = 1. * meshsize
        dtype = jnp.zeros((), dtype=dtype).dtype  # default dtype is float32, except if config.update('jax_enable_x64', True)
        self.__dict__.update(dtype=dtype)
        if boxsize is not None: boxsize = _jnp_array_fill(boxsize, ndim, dtype=self.rdtype)
        if boxcenter is not None: boxcenter = _jnp_array_fill(boxcenter, ndim, dtype=self.rdtype)
        if fft_engine == 'auto':
            fft_engine = 'jaxdecomp' if ndim == 3 and jaxdecomp is not None else 'jax'
        self.__dict__.update(meshsize=meshsize, boxsize=boxsize, boxcenter=boxcenter, fft_engine=fft_engine)

    @property
    def rdtype(self):
        return jnp.zeros((), dtype=self.dtype).real.dtype

    @property
    def sharding_mesh(self):
        return get_sharding_mesh()

    @property
    def hermitian(self):
        return self.fft_engine != 'jaxdecomp' and jnp.issubdtype(self.dtype, jnp.floating)

    def tree_flatten(self):
        state = asdict(self)
        return tuple(state.pop(name) for name in ['boxsize', 'boxcenter']), state

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update({name: value for name, value in zip(['boxsize', 'boxcenter'], children)})
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
        toret = fftfreq(self.meshsize, kind=kind, sparse=sparse, hermitian=False, spacing=spacing, sharding_mesh=self.sharding_mesh, dtype=self.rdtype)
        if offset is not None:
            toret = tuple(tmp + off for off, tmp in zip(offset, toret))
        return toret

    def ccoords(self, kind: str='wavenumber', sparse: bool | None=None):
        r"""
        Return mesh Fourier coordinates.

        Parameters
        ----------
        kind : str, default='wavenumber'
            Either 'circular' (in :math:`(-\pi, \pi)`) or 'wavenumber' (in ``spacing`` units),
            or 'index' (index of Fourier mode).

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
        if kind == 'index':
            spacing = 1
            kind = 'separation'
        axis_order = None
        if self.fft_engine == 'jaxdecomp':
            axis_order = tuple(np.roll(np.arange(self.ndim), shift=2))
        return fftfreq(self.meshsize, kind=kind, sparse=sparse, hermitian=self.hermitian, spacing=spacing, axis_order=axis_order, sharding_mesh=self.sharding_mesh, dtype=self.rdtype)

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
            if self.sharding_mesh.axis_names: value = jax.lax.with_sharding_constraint(value, jax.sharding.NamedSharding(self.sharding_mesh, spec=P(*self.sharding_mesh.axis_names)))
            value = jaxdecomp.fft.pfft3d(value)
        else:
            if self.hermitian:
                value = jnp.fft.rfftn(value)
            else:
                value = jnp.fft.fftn(value)
        if jnp.issubdtype(self.dtype, jnp.floating):
            if value.dtype.itemsize != 2 * self.dtype.itemsize:
                value = value.astype('c{:d}'.format(2 * self.dtype.itemsize))
        else:
            if value.dtype.itemsize != self.dtype.itemsize:
                value = value.astype('c{:d}'.format(self.dtype.itemsize))
        return value

    def c2r(self, value):
        """FFT, from complex to real."""
        if self.fft_engine == 'jaxdecomp':
            if self.sharding_mesh.axis_names: value = jax.lax.with_sharding_constraint(value, jax.sharding.NamedSharding(self.sharding_mesh, spec=P(*self.sharding_mesh.axis_names)))
            value = jaxdecomp.fft.pifft3d(value)
            if jnp.issubdtype(self.dtype, jnp.floating):
                value = value.real.astype(self.dtype)
        else:
            if self.hermitian:
                value = jnp.fft.irfftn(value, s=tuple(self.meshsize))
            else:
                value = jnp.fft.ifftn(value)
        if jnp.issubdtype(self.dtype, jnp.floating):
            if value.dtype.itemsize != self.dtype.itemsize:
                value = value.astype('f{:d}'.format(self.dtype.itemsize))
        else:
            if value.dtype.itemsize != self.dtype.itemsize:
                value = value.astype('c{:d}'.format(self.dtype.itemsize))
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
            mattrs = mattrs.clone(dtype=value.real.dtype if 'complex' in self.__class__.__name__.lower() and mattrs.hermitian else value.dtype)
        self.__dict__.update(value=value, attrs=mattrs)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in ['value', 'attrs']), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(value=children[0], attrs=children[1])
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


for name in ['meshsize', 'boxsize', 'boxcenter'] + ['cellsize']:
    _set_property(BaseMeshField, name)


def _set_binary(base, name: str, op: Callable[[Any, Any], Any]) -> None:
    def fn(self, other):
        if type(other) == type(self):
            value = op(self.value, other.value)
        else:
            value = op(self.value, other)
        return self.clone(value=value)

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


def _set_unary(base, name: str, op: Callable[[Any], Any]) -> None:
    def fn(self):
        return self.clone(value=op(self.value))

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
            other = other.value
        value = getattr(self.value.value.at[self.item], name)(other, **kwargs)
        return self.value.clone(value=value)

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

    def read(self, positions: jax.Array, resampler: str | Callable='cic', compensate: bool=False, exchange: bool=False, **kwargs):
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
        if isinstance(positions, ParticleField):
            positions = positions.positions
        if with_sharding and exchange:
            positions, exchange, inverse = exchange_particles(self.attrs, positions, return_inverse=True, return_type='jax')
        toret = _read(self, positions, resampler=resampler, compensate=compensate, **kwargs)
        #jread = jax.jit(_read, static_argnames=['resampler', 'compensate', 'halo_add'])
        #toret = jread(self, positions, resampler=resampler, compensate=compensate, **kwargs)
        if inverse is not None:
            toret = inverse(toret)
        return toret


# Avoid this as this accumulates in the GPU memory (jax.clear_caches() to clear the cache)
@partial(jax.jit, static_argnames=['resampler', 'compensate', 'halo_add'])
def _read(mesh, positions: jax.Array, resampler: str | Callable='cic', compensate: bool=False, halo_add: int=0):
    """WARNING: in case of multiprocessing, positions and weights are assumed to be exchanged!"""

    resampler = resamplers.get_resampler(resampler)
    if isinstance(compensate, bool):
        if compensate:
            kernel_compensate = resampler.compensate
        else:
            kernel_compensate = None
    else:
        kernel_compensate = compensate

    if kernel_compensate is not None:
        mesh = mesh.r2c().apply(kernel_compensate).c2r()

    value, attrs = mesh.value, mesh.attrs
    sharding_mesh = attrs.sharding_mesh
    with_sharding = bool(sharding_mesh.axis_names)

    positions = (positions + attrs.boxsize / 2. - attrs.boxcenter) / attrs.cellsize
    out = jnp.zeros_like(positions, shape=positions.shape[:1], dtype=mesh.value.dtype)
    _read = lambda mesh, positions, out: resampler.read(mesh, positions, out=out)

    #order = resampler.order
    #ishifts = np.arange(order) - (order - 1) // 2
    #from itertools import product
    #ishifts = np.array(list(product(ishifts, ishifts, ishifts)), dtype=int)

    if with_sharding:
        shape_devices = np.array(sharding_mesh.devices.shape + (1,) * (attrs.ndim - sharding_mesh.devices.ndim))
        size_devices = shape_devices.prod(dtype='i4')
        shard_shifts = jnp.meshgrid(*[np.arange(s) * attrs.meshsize[i] / s for i, s in enumerate(shape_devices)], indexing='ij', sparse=False)
        shard_shifts = jnp.stack(shard_shifts, axis=-1).reshape(-1, len(shard_shifts))

        def s(positions, idevice):
            return positions - shard_shifts[idevice[0]]

        #idx = jnp.round(positions[12]).astype(int)
        #print('1', positions[12], idx, [(tuple(ishift), float(value[tuple(idx + ishift)])) for ishift in ishifts])
        positions = shard_map(s, mesh=sharding_mesh, in_specs=(P(sharding_mesh.axis_names),) * 2, out_specs=P(sharding_mesh.axis_names))(positions, jnp.arange(size_devices))
        #print('2', positions[12])

        kw_sharding = dict(halo_size=halo_add + resampler.order, sharding_mesh=sharding_mesh)
        value, offset = pad_halo(value, **kw_sharding, factor=1)
        positions = positions + offset
        #idx = jnp.round(positions[12]).astype(int)
        #idx = idx + jnp.array([44, 0, 0], dtype=int)
        #print('3', positions[12], jnp.round(positions[12]).astype(int), idx, [(tuple(ishift), float(value[tuple(idx + ishift)])) for ishift in ishifts])
        value = exchange_halo(value, **kw_sharding)
        _read = shard_map(_read, mesh=sharding_mesh, in_specs=(P(*sharding_mesh.axis_names), P(sharding_mesh.axis_names), P(sharding_mesh.axis_names)), out_specs=P(sharding_mesh.axis_names))
    #else:
    #    idx = jnp.round(positions[8]).astype(int)
    #    print('0', positions[8], idx, [(tuple(ishift), float(value[tuple(idx + ishift)])) for ishift in ishifts])

    return _read(value, positions, out)


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


def _get_extent(*positions):
    """Return minimum physical extent (min, max) corresponding to input positions."""
    if not positions:
        raise ValueError('positions must be provided if boxsize and boxcenter are not specified, or check is True')
    # Find bounding coordinates
    nonempty_positions = [pos for pos in positions if pos.size]
    if not nonempty_positions:
        raise ValueError('<= 1 particles found; cannot infer boxsize')
    axis = tuple(range(len(nonempty_positions[0].shape[:-1])))
    backend, kw = _get_distributed_backend(nonempty_positions[0])
    if backend == 'jax':
        pos_min = jnp.array([jnp.min(p, axis=axis) for p in nonempty_positions]).min(axis=0)
        pos_max = jnp.array([jnp.max(p, axis=axis) for p in nonempty_positions]).max(axis=0)
    else:
        pos_min = np.array([np.min(p, axis=axis) for p in nonempty_positions]).min(axis=0)
        pos_max = np.array([np.max(p, axis=axis) for p in nonempty_positions]).max(axis=0)
        mpicomm = kw['mpicomm']
        pos_min, pos_max = np.min(mpicomm.allgather(pos_min), axis=0), np.max(mpicomm.allgather(pos_max), axis=0)
    return pos_min, pos_max


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
        pos_min, pos_max = _get_extent(*positions)
        delta = jnp.abs(pos_max - pos_min)
        if boxcenter is None: boxcenter = 0.5 * (pos_min + pos_max)
        if boxsize is None:
            if cellsize is not None and meshsize is not None:
                boxsize = meshsize * cellsize
            else:
                boxsize = delta.max() * boxpad
        if check and (boxsize < delta).any():
            raise ValueError('boxsize {} too small to contain all data (max {})'.format(boxsize, delta))

    ndim = 3
    if positions:
        if dtype is None:
            dtype = positions[0].dtype
        if ndim is None:
            ndim = positions[0].shape[-1]

    rdtype = jnp.zeros((), dtype=dtype).real.dtype
    boxsize = _jnp_array_fill(boxsize, ndim, dtype=rdtype)
    toret = dict()
    if meshsize is None:
        if cellsize is not None:
            cellsize = _np_array_fill(cellsize, ndim, dtype=rdtype)
            meshsize = np.ceil(boxsize / cellsize).astype('i4')
            if sharding_mesh.axis_names:
                same = np.all(meshsize == meshsize[0])
                if same:
                    # FIXME
                    shape_devices = max(sharding_mesh.devices.shape)
                else:
                    shape_devices = np.array(sharding_mesh.devices.shape + (1,) * (ndim - sharding_mesh.devices.ndim))
                meshsize = (meshsize + shape_devices - 1) // shape_devices * shape_devices  # to make it a multiple of devices
            toret['meshsize'] = meshsize
            toret['boxsize'] = meshsize * cellsize  # enforce exact cellsize
        else:
            raise ValueError('meshsize (or cellsize) must be specified')
    else:
        meshsize = _np_array_fill(meshsize, ndim, dtype='i4')
        toret['meshsize'] = meshsize
    boxcenter = _jnp_array_fill(boxcenter, ndim, dtype=rdtype)
    toret = dict(boxsize=boxsize, boxcenter=boxcenter) | toret
    return MeshAttrs(**toret, dtype=dtype, **kwargs)


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
    Representation of a field as a collection of particles and associated weights.

    This class is used to represent a set of particles (e.g., galaxies, simulation particles, or random points)
    with positions and weights, and provides utilities for distributed computation and painting to mesh grids.

    Parameters
    ----------
    positions : jax.Array
        Array of particle positions, shape (N, ndim).
        Important: in case of parellel computation, assumed scattered (sharded) over the different processes.
    weights : jax.Array, optional
        Array of particle weights, shape (N,). If None, defaults to 1 for all particles.
        Important: in case of parellel computation, assumed scattered (sharded) over the different processes.
    attrs : MeshAttrs or dict, optional
        Mesh attributes.
    exchange : bool, default=False
        If ``True``, perform particle exchange for distributed computation.
    backend : {'auto', 'jax', 'mpi'}, default='auto'
        Backend particle exchange.
    **kwargs
        Additional keyword arguments for mesh or sharding setup.

    Attributes
    ----------
    positions : jax.Array
        Particle positions.
    weights : jax.Array
        Particle weights.
    attrs : MeshAttrs
        Mesh attributes.

    Examples
    --------
    >>> p1 = ParticleField(positions, weights, attrs, exchange=True)
    >>> p2 = ParticleField(other_positions, other_weights, attrs, exchange=True)
    >>> p_sum = p1 + p2
    >>> mesh = p_sum.paint(resampler='cic')
    """
    positions: jax.Array = field(repr=False)
    weights: jax.Array = field(repr=False)
    attrs: MeshAttrs | None = field(init=False, repr=False)

    def __init__(self, positions: jax.Array, weights: jax.Array | None=None, attrs=None, exchange=False, backend='auto', **kwargs):

        if attrs is None: raise ValueError('attrs must be provided')
        if not isinstance(attrs, MeshAttrs): attrs = MeshAttrs(**attrs)
        sharding_mesh = attrs.sharding_mesh
        with_sharding = bool(sharding_mesh.axis_names)
        if with_sharding:
            backend, kwargs = _get_distributed_backend(positions, backend=backend, **kwargs)
            sharding = jax.sharding.NamedSharding(sharding_mesh, P(sharding_mesh.axis_names))
        positions = jnp.asarray(positions)
        if weights is None:
            weights = jnp.ones_like(positions, shape=positions.shape[:-1])
        else:
            weights = jnp.asarray(weights)

        input_is_not_sharded = getattr(positions, 'sharding', None) is not None and getattr(positions.sharding, 'mesh', None) is None
        input_is_sharded = getattr(positions, 'sharding', None) is not None and getattr(positions.sharding, 'mesh', None) is not None

        inverse = tuple()
        if with_sharding and exchange:
            if backend == 'mpi' and input_is_sharded:

                def get(array):
                    return np.concatenate([_.data for _ in array.addressable_shards], axis=0)

                positions = get(positions)
                weights = get(weights)

            if backend == 'jax' and input_is_not_sharded:
                positions, weights = make_particles_from_local(positions, weights)

            if backend == 'jax':
                positions, weights = jax.device_put((positions, weights), sharding)

            positions, exchange, *inverse = exchange_particles(attrs, positions, return_type='jax', backend=backend, **kwargs)
            weights = exchange(weights)

        # If sharding and not exchange, but input arrays aren't sharded, assume input arrays are local and shard them here
        if with_sharding and (not exchange):
            if input_is_not_sharded:
                positions, weights = make_particles_from_local(positions, weights)
            else:
                positions, weights = jax.device_put((positions, weights), sharding)

        self.__dict__.update(positions=positions, weights=weights, attrs=attrs)
        if inverse: self.__dict__.update(exchange_inverse=inverse[0])

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = asdict(self) | kwargs
        return self.__class__(**state)

    def exchange(self, **kwargs):
        return self.clone(exchange=True, **kwargs)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in ['positions', 'weights', 'attrs']), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update({name: value for name, value in zip(['positions', 'weights', 'attrs'], children)})
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

    @classmethod
    def concatenate(cls, others, weights=None, local=True, **kwargs):
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
        for name, value in gather.items():
            if local: value = _local_concatenate(value, axis=0)
            else: value = jnp.concatenate(value, axis=0)
            gather[name] = value
        return cls(**gather, attrs=others[0].attrs)

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

    def split(self, nsplits=1, extent=None):
        attrs = self.attrs
        nsplits = _np_array_fill(nsplits, attrs.ndim, dtype='i4')
        split_boxsize = attrs.boxsize / nsplits
        if extent is None:
            extent = _get_extent(*self.positions)
        extent_boxsize = (extent[1] - extent[0]) / nsplits
        extent_offset = extent[0]

        def split_particles(isplit):
            isplit = np.array(isplit)
            mask = jnp.all((self.positions >= isplit * extent_boxsize + extent_offset) & (self.positions <= (isplit + 1) * extent_boxsize + extent_offset), axis=-1)
            attrs = self.attrs.clone(boxsize=split_boxsize, boxcenter=(isplit + 0.5) * extent_boxsize + extent_offset, meshsize=self.attrs.meshsize)
            return self.clone(positions=self.positions[mask], weights=self.weights[mask], attrs=attrs)

        for isplit in itertools.product(*(range(nsplit) for nsplit in nsplits)):
            yield split_particles(isplit)

    def paint(self, resampler: str | Callable='cic', interlacing: int=0,
              compensate: bool=False, out: str='real', dtype=None, **kwargs):
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
            If 'real', return a :class:`RealMeshField`, else :class:`ComplexMeshField`
            or :class:`ComplexMeshField` if ``dtype`` is complex.

        Returns
        -------
        mesh : Output mesh.
        """
        attrs = self.attrs
        if dtype is not None: attrs = attrs.clone(dtype=dtype)
        positions, weights = self.positions, self.weights
        #import time
        #t0 = time.time()
        #t1 = time.time()
        # jit is fast enough that it is not worth padding to fixed size
        #size = 2**22
        #positions = jnp.pad(positions, pad_width=((0, size - positions.shape[0]), (0, 0)))
        #weights = jnp.pad(weights,  pad_width=((0, size - weights.shape[0]),))
        return _paint(attrs, positions, weights, resampler=resampler, interlacing=interlacing, compensate=compensate, out=out, **kwargs)
        #jpaint = jax.jit(_paint, static_argnames=['resampler',  'interlacing', 'compensate', 'out', 'halo_add'])
        #return jpaint(attrs, positions, weights, resampler=resampler, interlacing=interlacing, compensate=compensate, out=out, **kwargs)
        #jax.block_until_ready(toret)
        #print(f'exchange {t1 - t0:.2f} painting {time.time() - t1:.2f}', positions.shape[0] / size, _paint._cache_size())


@partial(jax.jit, static_argnames=['resampler',  'interlacing', 'compensate', 'out', 'halo_add'])
def _paint(attrs, positions, weights=None, resampler: str | Callable='cic', interlacing: int=0, compensate: bool=False, out: str='real', halo_add: int=0):
    """WARNING: in case of multiprocessing, positions and weights are assumed to be exchanged!"""

    resampler = resamplers.get_resampler(resampler)
    interlacing = max(interlacing, 1)
    interlacing_shifts = jnp.astype(jnp.arange(interlacing) * 1. / interlacing, attrs.rdtype)

    sharding_mesh = attrs.sharding_mesh
    with_sharding = bool(sharding_mesh.axis_names)

    positions = (positions + attrs.boxsize / 2. - attrs.boxcenter) / attrs.cellsize
    #mask = jnp.where(weights == 0., jnp.nan, 1.)[..., None]
    #print(jnp.nanmin(positions * mask, axis=0), jnp.nanmax(positions * mask, axis=0), attrs.boxsize, attrs.boxcenter, attrs.meshsize)

    if isinstance(compensate, bool):
        if compensate:
            kernel_compensate = resampler.compensate
        else:
            kernel_compensate = None
    else:
        kernel_compensate = compensate

    _paint = resampler.paint
    if with_sharding:
        shape_devices = np.array(sharding_mesh.devices.shape + (1,) * (attrs.ndim - sharding_mesh.devices.ndim))
        size_devices = shape_devices.prod(dtype='i4')
        shard_shifts = jnp.meshgrid(*[np.arange(s) * attrs.meshsize[i] / s for i, s in enumerate(shape_devices)], indexing='ij', sparse=False)
        shard_shifts = jnp.stack(shard_shifts, axis=-1).reshape(-1, len(shard_shifts))

        def s(positions, idevice):
            return positions - shard_shifts[idevice[0]]

        positions = shard_map(s, mesh=sharding_mesh, in_specs=(P(sharding_mesh.axis_names),) * 2, out_specs=P(sharding_mesh.axis_names))(positions, jnp.arange(size_devices))

        _paint = shard_map(_paint, mesh=sharding_mesh, in_specs=(P(*sharding_mesh.axis_names), P(sharding_mesh.axis_names), P(sharding_mesh.axis_names)), out_specs=P(*sharding_mesh.axis_names))

    def ppaint(positions, weights=None):
        mesh = attrs.create(kind='real', fill=0.)
        value = mesh.value
        w = None
        if weights is not None:
            w = weights.astype(value.dtype)
        if with_sharding:
            kw_sharding = dict(halo_size=halo_add + resampler.order + interlacing, sharding_mesh=sharding_mesh)
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
        toret = ppaint(positions, weights)
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

            tmp = ppaint(positions + shift, weights).r2c()
            carry += tmp.apply(kernel_shift, kind='circular')
            return carry, shift

        toret = jax.lax.scan(f, init=attrs.create(kind='complex', fill=0.), xs=interlacing_shifts)[0]
        #toret = attrs.create(kind='complex', fill=0.)
        #for shift in interlacing_shifts: toret, _ = f(toret, shift)
        if kernel_compensate is not None:
            toret = toret.apply(kernel_compensate)
        if out == 'real': toret = toret.c2r()
    return toret


def _set_property(base, name: str):
    setattr(base, name, property(lambda self: getattr(self.attrs, name)))


for name in [field.name for field in fields(MeshAttrs)] + ['cellsize']:
    _set_property(ParticleField, name)


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


@default_sharding_mesh
def _find_unique_edges(xvec, x0, xmin=0., xmax=np.inf, sharding_mesh=None):
    x2 = sum(xx**2 for xx in xvec).ravel()

    def get_x(x2):
        x2 = x2[(x2 >= xmin**2) & (x2 <= xmax**2)]
        _, index = jnp.unique(np.int64(x2 / (0.5 * x0)**2 + 0.5), return_index=True)
        return x2[index]

    if sharding_mesh.axis_names:
        x2 = np.concatenate([_.data for _ in x2.addressable_shards], axis=0)
        x2 = get_x(x2)
        x2 = make_array_from_process_local_data(x2, pad=-1, sharding_mesh=sharding_mesh)
        x2 = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, spec=P(None)))(x2)
        x2 = x2[x2 > -1]
        x2 = get_x(x2)
    else:
        x2 = get_x(x2)
    x = jnp.sqrt(x2)
    tmp = (x[:-1] + x[1:]) / 2.
    edges = jnp.insert(tmp, jnp.array([0, len(tmp)]), jnp.array([tmp[0] - (x[1] - x[0]), tmp[-1] + (x[-1] - x[-2])]))
    return edges


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
@partial(jax.jit, static_argnames=('sharding_mesh', 'ravel'))  # if we want to save some memory, consider linear binning
def _get_bin_attrs(coords, edges: jax.Array, weights: None | jax.Array=None, sharding_mesh=None, ravel=True):

    def _get_attrs(coords, edges, weights):
        r"""Return bin index, binned number of modes and coordinates."""
        shape = coords.shape
        coords = coords.ravel()

        ibin = jnp.digitize(coords, edges, right=False)
        x = jnp.bincount(ibin, weights=coords if weights is None else coords * weights, length=len(edges) + 1)[1:-1]
        del coords
        nmodes = jnp.bincount(ibin, weights=weights, length=len(edges) + 1)[1:-1]
        if not ravel:
            ibin = ibin.reshape(shape)
        return ibin, nmodes, x

    get_attrs = _get_attrs

    if sharding_mesh.axis_names and sharding_mesh.devices.size > 1:  # sharding_mesh.devices.size > 1, otherwise, obscure error...

        def get_attrs(coords, edges, weights):
            r"""Return bin index, binned number of modes and coordinates."""
            ibin, nmodes, x = _get_attrs(coords, edges, weights)
            nmodes = jax.lax.psum(nmodes, sharding_mesh.axis_names)
            x = jax.lax.psum(x, sharding_mesh.axis_names)
            return ibin, nmodes, x

        get_attrs = shard_map(get_attrs, mesh=sharding_mesh,
                    in_specs=(P(*sharding_mesh.axis_names), P(None), P(sharding_mesh.axis_names)),
                    out_specs=(P(sharding_mesh.axis_names) if ravel else P(*sharding_mesh.axis_names), P(None), P(None)))

    return get_attrs(coords, edges, weights)


@default_sharding_mesh
def _get_bin_attrs_edges2d(coords, edges: jax.Array, weights: None | jax.Array=None, sharding_mesh=None, ravel=True):
    edges_1d = jnp.unique(edges.ravel())
    battrs = _get_bin_attrs(coords, edges_1d, weights=weights, sharding_mesh=sharding_mesh, ravel=ravel)
    M = ((edges_1d[:-1] >= edges[:, [0]]) & (edges_1d[1:]  <= edges[:, [1]])).astype(int)  # rebinning matrix
    return battrs, edges_1d, M


@default_sharding_mesh
@partial(jax.jit, static_argnames=('length', 'antisymmetric', 'sharding_mesh'))
def _bincount(ibin, value, weights=None, length=None, antisymmetric=False, sharding_mesh=None):

    if not isinstance(ibin, (tuple, list)):
        ibin = [ibin]

    def _count(value, *ibin):
        value = value.ravel()
        if weights is not None:
            if antisymmetric: value = value.imag
            else: value = value.real
            value *= weights
        count = lambda ib: jnp.bincount(ib, weights=value, length=length + 2)
        if jnp.iscomplexobj(value):  # bincount much slower with complex numbers
            count = lambda ib: jnp.bincount(ib, weights=value.real, length=length + 2) + 1j * jnp.bincount(ib, weights=value.imag, length=length + 2)
        value = sum(count(ib.ravel() if ib.ndim > 1 else ib) for ib in ibin)
        return value[1:-1] / len(ibin)

    count = _count

    if sharding_mesh.axis_names:

        def count(value, *ibin):
            value = _count(value, *ibin)
            return jax.lax.psum(value, sharding_mesh.axis_names)

        in_specs = (P(*sharding_mesh.axis_names),)
        for ib in ibin:
            in_specs += (P(sharding_mesh.axis_names) if ib.ndim <= 1 else P(*sharding_mesh.axis_names),)
        count = shard_map(count, mesh=sharding_mesh, in_specs=in_specs, out_specs=P(None))

    return count(value, *ibin)