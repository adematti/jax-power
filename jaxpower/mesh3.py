import numbers
import itertools
import operator
import functools
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from dataclasses import dataclass

from .mesh import (BaseMeshField, MeshAttrs, RealMeshField, ComplexMeshField, ParticleField, FKPField, staticarray, get_sharding_mesh, _get_hermitian_weights, _find_unique_edges, _get_bin_attrs, _bincount, create_sharded_random,
compute_normalization, compute_box_normalization, split_particles, __format_meshes)
from .mesh2 import _get_los_vector
from .types import Mesh3SpectrumPole, Mesh3SpectrumPoles, Mesh3CorrelationPole, Mesh3CorrelationPoles, ObservableLeaf, ObservableTree, WindowMatrix
from .utils import real_gaunt, get_legendre, get_spherical_jn, get_Ylm, wigner_3j, wigner_9j, register_pytree_dataclass


prod = partial(functools.reduce, operator.mul)


def _make_edges3(kind, mattrs, edges, ells, basis='scoccimarro', batch_size=None, buffer_size=0, mask_edges=None):
    assert basis in ['sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-diagonal']
    if 'scoccimarro' in basis:
        ndim = 3
    else:
        ndim = 2
        mattrs = mattrs.clone(dtype=mattrs.cdtype)

    # Argument after e.q. 53 of https://arxiv.org/pdf/1512.07295
    if mask_edges is None:
        if 'scoccimarro' in basis:
            mask_edges = ['mid1 <= mid2', 'mid2 <= mid3']
            mask_edges += ['(mid3 >= jnp.abs(mid1 - mid2)) & (mid3 <= jnp.abs(mid1 + mid2))']
            mask_edges += ['(edge1 + edge2 + edge3)[:, 1] <= 2 * nyq']
        else:  # sugiyama
            mask_edges = ['mid1 <= mid2']
            mask_edges += ['(edge1 + edge2)[:, 1] <= nyq']
    if isinstance(mask_edges, str):
        mask_edges = mask_edges.strip()
        mask_edges = tuple(mask_edges.split(';'))

    if isinstance(mask_edges, (list, tuple)):
        mask_edges_list = mask_edges

        def mask_edges(*edges):
            mask = np.ones_like(edges[0], shape=edges[0].shape[0], dtype=bool)
            d = {'nyq': vecmax}
            for i, edge in enumerate(edges):
                d[f'mid{i + 1:d}'] = np.mean(edge, axis=-1)
                d[f'edge{i + 1:d}'] = edge
            for c in mask_edges_list:
                c = c.strip()
                if not c: continue
                mask &= eval(c, {'np': np, 'jnp': jnp}, d)
            return mask

    wmodes = None
    if kind == 'complex':
        vec = mattrs.kcoords(kind='separation', sparse=True)
        vec0 = mattrs.kfun.min()
        if mattrs.is_hermitian:
            wmodes = _get_hermitian_weights(vec, sharding_mesh=None)
    else:
        vec = mattrs.xcoords(kind='separation', sparse=True)
        vec0 = mattrs.cellsize.min()
    vecmax = vec0 * np.min(mattrs.meshsize) / 2.

    if edges is None:
        edges = {}
    if not isinstance(edges, (tuple, list)):
        edges = [edges] * ndim
    edges = list(edges)
    for iedge, edge in enumerate(edges):
        if isinstance(edge, dict):
            step = edge.get('step', None)
            if step is None:
                edge = _find_unique_edges(vec, vec0, xmin=edge.get('min', 0.), xmax=edge.get('max', np.inf))
            else:
                edge = np.arange(edge.get('min', 0.), edge.get('max', vecmax), step)
        else:
            edge = np.asarray(edge)
        if edge.ndim == 2:
            assert np.allclose(edge[1:, 0], edge[:-1, 1])
            edge = np.append(edge[:, 0], edge[-1, 1])
        edges[iedge] = edge

    ells = _format_ells(ells, basis=basis)

    coords = jnp.sqrt(sum(xx**2 for xx in vec))
    ibin1d, nmodes1d, xavg1d, edges1d = [], [], [], []
    for edge in edges[:ndim]:
        ib, nm, x = _get_bin_attrs(coords, edge, wmodes, ravel=False)
        ib = ib - 1
        x /= nm
        ibin1d.append(ib)
        nmodes1d.append(nm)
        xavg1d.append(x)
        edges1d.append(jnp.column_stack([edge[:-1], edge[1:]]))
    edges1d = edges1d + [edges1d[-1]] * (ndim - len(edges1d))
    ibin1d = ibin1d + [ibin1d[-1]] * (ndim - len(ibin1d))
    nmodes1d = nmodes1d + [nmodes1d[-1]] * (ndim - len(nmodes1d))
    iedges1d = [jnp.arange(len(xx)) for xx in nmodes1d]
    xavg1d = xavg1d + [xavg1d[-1]] * (ndim - len(xavg1d))

    def _cproduct(array):
        grid = jnp.meshgrid(*array, sparse=False, indexing='ij')
        return jnp.column_stack([tmp.ravel() for tmp in grid])

    def _product(array):
        if not isinstance(array, (tuple, list)):
            array = [array] * ndim
        if 'diagonal' in basis:
            grid = [jnp.array(array[0])] * ndim
            return jnp.column_stack([tmp.ravel() for tmp in grid])
        else:
            return _cproduct(array)

    # of shape (nbins, ndim, 2)
    edges = jnp.concatenate([_product([edge[..., 0] for edge in edges1d])[..., None], _product([edge[..., 1] for edge in edges1d])[..., None]], axis=-1)
    list_edges = [edges[:, i, :] for i in range(ndim)]
    mask = mask_edges(*list_edges)
    edges = edges[mask]
    xavg = _product(xavg1d)[mask]
    nmodes = jnp.prod(_product(nmodes1d)[mask], axis=-1)
    iedges = _product(iedges1d)[mask]
    nmodes = [nmodes] * len(ells)

    if 'diagonal' in basis:
        # No gain in buffer
        # But since we can have multiple meshes in memory:
        if batch_size is None:
            batch_size = max(buffer_size, 1)
        buffer_size = 0
    if batch_size is None:
        batch_size = 1

    if buffer_size > 1:
        split_iedges = []

        for axis in range(ndim):
            # Number of batches
            iedges1d_axis = iedges1d[axis][np.isin(iedges1d[axis], iedges[..., axis])]
            nsplits = (len(iedges1d_axis) + buffer_size - 1) // buffer_size
            split_iedges.append(jnp.array_split(iedges1d_axis, nsplits, axis=0))
        _buffer_global_iedges, _buffer_iedges, _buffer_iedges1d = [], [], []
        size_max, usize_max = 0, 0

        for biedges1d in itertools.product(*split_iedges):
            biedges = _cproduct(biedges1d)
            mask = jax.vmap(lambda biedge: jnp.all(iedges == biedge, axis=-1).any())(biedges)
            if mask.sum():
                _buffer_global_iedges.append(biedges[mask])
                _buffer_iedges.append(_cproduct([jnp.arange(len(iedge)) for iedge in biedges1d])[mask])
                _buffer_iedges1d.append(biedges1d)
                size_max = max(size_max, len(_buffer_iedges[-1]))
                usize_max = max(usize_max, *[len(b) for b in _buffer_iedges1d[-1]])

        #print('Number of FFTs', sum(sum(len(bb) for bb in b) for b in _buffer_iedges1d))
        # Pad to be able to use jax.lax.map, otherwise compilation time is prohibitive
        for i in range(len(_buffer_global_iedges)):
            _buffer_global_iedges[i] = jnp.pad(_buffer_global_iedges[i], [(0, size_max - len(_buffer_global_iedges[i])), (0, 0)], mode='edge')
            _buffer_iedges[i] = jnp.pad(_buffer_iedges[i], [(0, size_max - len(_buffer_iedges[i])), (0, 0)], mode='edge')
            _buffer_iedges1d[i] = jnp.stack([jnp.pad(b, (0, usize_max - len(b)), mode='edge') for b in _buffer_iedges1d[i]])

        #print('Number of FFTs padded', sum(sum(len(bb) for bb in b) for b in _buffer_iedges1d))
        _buffer_global_iedges = jnp.stack(_buffer_global_iedges).reshape(-1, ndim)
        _buffer_sort = jnp.array([jnp.flatnonzero(jnp.all(iedge == _buffer_global_iedges, axis=1))[0] for iedge in iedges])
        # _buffer_iedges = (N-dim bins, corresponding unique bins along each dim, how to sort the N-dim bins to obtain the requested bins)
        _buffer_iedges = (jnp.stack(_buffer_iedges), jnp.stack(_buffer_iedges1d), _buffer_sort)
    else:
        _buffer_iedges = None

    return dict(edges=edges, ibin1d=tuple(ibin1d), edges1d=tuple(edges1d), nmodes1d=tuple(nmodes1d), xavg1d=tuple(xavg1d), nmodes=nmodes, xavg=xavg, wmodes=wmodes, mattrs=mattrs, basis=basis, batch_size=batch_size, buffer_size=buffer_size, _iedges=iedges, _buffer_iedges=_buffer_iedges, ells=ells)


@partial(register_pytree_dataclass, meta_fields=['basis', 'batch_size', 'buffer_size', 'ells'])
@dataclass(init=False, frozen=True)
class BinMesh3SpectrumPoles(object):
    """
    Binning operator for 3D mesh to bispectrum.

    Parameters
    ----------
    mattrs : MeshAttrs or BaseMeshField
        Mesh attributes or mesh field.
    edges : array-like, dict, or None, optional
        Bin edges or binning configuration.
    ells : int or tuple, optional
        Multipole orders.
    basis : str, optional
        Binning basis ('sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-diagonal').
    batch_size : int, optional
        Batch size for JAX mapping.
    buffer_size : int, optional
        Buffer size for chunked binning: number of meshes that van be kept into memory.
    """

    edges: jax.Array = None
    nmodes1d: jax.Array = None
    edges1d: tuple = None
    xavg1d: tuple = None
    ibin1d: tuple = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    wmodes: jax.Array = None
    _iedges: jax.Array = None
    _buffer_iedges: tuple = None
    mattrs: MeshAttrs = None
    basis: str = 'sugiyama'
    batch_size: int = 1
    buffer_size: int = 0
    ells: tuple = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, ells=0, basis='sugiyama', batch_size=None, buffer_size=0, mask_edges=None):
        if not isinstance(mattrs, MeshAttrs):
            mattrs = mattrs.attrs
        kw = _make_edges3('complex', mattrs, edges, ells, basis=basis, batch_size=batch_size, buffer_size=buffer_size, mask_edges=mask_edges)
        self.__dict__.update(kw)
        xmid = jnp.mean(self.edges, axis=-1).T

        if 'scoccimarro' in basis:
            symfactor = jnp.ones_like(xmid[0])
            symfactor = jnp.where((xmid[1] == xmid[0]) | (xmid[2] == xmid[0]) | (xmid[2] == xmid[1]), 2, symfactor)
            symfactor = jnp.where((xmid[1] == xmid[0]) & (xmid[2] == xmid[0]), 6, symfactor)

            def bin(axis, ibin):
                return mattrs.c2r(self.ibin1d[axis] == ibin)

            def reduce(meshes):
                return prod(meshes).sum()

            nmodes = self.bin_reduce_meshs(bin, reduce) * self.mattrs.meshsize.prod(dtype=self.mattrs.rdtype)**2
            nmodes = [nmodes] * len(ells)
            self.__dict__.update(nmodes=nmodes)

    def bin_reduce_meshs(self, bin, reduce):
        """
        Reduce over mesh bins using provided binning and reduction functions.

        Parameters
        ----------
        bin : callable
            Function to bin mesh along each axis.
        reduce : callable
            Function to reduce binned meshes.

        Returns
        -------
        result : array-like
            Reduced mesh values.
        """
        if self._buffer_iedges is None:

            def f(ibin):
                meshes = (bin(axis, ibin_) for axis, ibin_ in enumerate(ibin))
                return reduce(meshes)

            return jax.lax.map(f, self._iedges, batch_size=self.batch_size)

        else:

            def f(args):
                iedges, uiedges = args

                iter_binned_meshs = [jax.lax.map(partial(bin, axis), edge, batch_size=self.batch_size) for axis, edge in enumerate(uiedges)]

                def f_reduce(index):
                    meshes = (value[index[axis]] for axis, value in enumerate(iter_binned_meshs))
                    return reduce(meshes)

                return jax.lax.map(f_reduce, iedges)

            return jax.lax.map(f, self._buffer_iedges[:2]).ravel()[self._buffer_iedges[2]]

    def __call__(self, *meshes, remove_zero=False):
        """
        Bin and reduce input meshes to compute the bispectrum.

        Parameters
        ----------
        meshes : array-like or BaseMeshField
            Input meshes.
        remove_zero : bool, optional
            Whether to remove the zero mode.

        Returns
        -------
        binned : array-like
        """
        values = []
        ndim = 3 if 'scoccimarro' in self.basis else 2
        norm = self.mattrs.meshsize.prod(dtype=self.mattrs.rdtype)**2
        for imesh, mesh in enumerate(meshes):
            value = mesh.value if isinstance(mesh, BaseMeshField) else mesh
            if remove_zero:
                if imesh < ndim:  # 0, 1 for sugiyama, 0, 1, 2 for scoccimaro
                    value = value.at[(0,) * value.ndim].set(0.)
                else:
                    value = value - jnp.mean(value)
            values.append(value)

        def bin(axis, ibin):
            return self.mattrs.c2r(values[axis] * (self.ibin1d[axis] == ibin))

        def reduce(meshes):
            return prod(meshes, 1. if 'scoccimarro' in self.basis else values[2]).sum()

        return self.bin_reduce_meshs(bin, reduce) * norm



@partial(register_pytree_dataclass, meta_fields=['basis', 'batch_size', 'buffer_size', 'ells', 'klimit'])
@dataclass(init=False, frozen=True)
class BinMesh3CorrelationPoles(object):
    """
    Binning operator for 3D mesh to 3pcf.

    Parameters
    ----------
    mattrs : MeshAttrs or BaseMeshField
        Mesh attributes or mesh field.
    edges : array-like, dict, or None, optional
        Bin edges or binning configuration.
    ells : int or tuple, optional
        Multipole orders.
    basis : str, optional
        Binning basis ('sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-equilateral').
    batch_size : int, optional
        Batch size for JAX mapping.
    buffer_size : int, optional
        Buffer size for chunked binning: number of meshes that van be kept into memory.
    """

    edges: jax.Array = None
    nmodes1d: jax.Array = None
    edges1d: tuple = None
    xavg1d: tuple = None
    ibin1d: tuple = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    wmodes: jax.Array = None
    _iedges: jax.Array = None
    _buffer_iedges: tuple = None
    nmodes1d: jax.Array = None
    mattrs: MeshAttrs = None
    basis: str = 'sugiyama'
    batch_size: int = 1
    buffer_size: int = 0
    ells: tuple = None
    klimit: tuple = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, ells=0, basis='sugiyama', batch_size=None, buffer_size=0, mask_edges=None, klimit=None):
        if not isinstance(mattrs, MeshAttrs):
            mattrs = mattrs.attrs
        kw = _make_edges3('real', mattrs, edges, ells, basis=basis, batch_size=batch_size, buffer_size=buffer_size, mask_edges=mask_edges)
        self.__dict__.update(kw)
        if isinstance(klimit, bool) and klimit: klimit = (0, mattrs.knyq.min())
        self.__dict__.update(klimit=klimit)

    bin_reduce_meshs = BinMesh3SpectrumPoles.bin_reduce_meshs

    def __call__(self, *meshes, ell=(0, 0, 0), remove_zero=False):
        """
        Bin and reduce input meshes to compute the bispectrum.

        Parameters
        ----------
        meshes : array-like or BaseMeshField
            Input meshes.
        remove_zero : bool, optional
            Whether to remove the zero mode.

        Returns
        -------
        binned : array-like
        """
        values = []
        ndim = 2
        ell = ell[:2]
        norm = (1j)**sum(ell) / self.mattrs.cellsize.prod()**2
        knorm = jnp.sqrt(sum(kk**2 for kk in self.mattrs.kcoords(sparse=True)))
        jns = [get_spherical_jn(ell) for ell in ell]

        for imesh, mesh in enumerate(meshes):
            value = mesh.value if isinstance(mesh, BaseMeshField) else mesh
            if remove_zero:
                if imesh < ndim:  # 0, 1 for sugiyama, 0, 1, 2 for scoccimaro
                    value = value.at[(0,) * value.ndim].set(0.)
                else:
                    value = value - jnp.mean(value)
            values.append(value)

        def bin(axis, ibin):
            x = knorm * self.xavg1d[axis][ibin]
            jn = jns[axis](x)
            if self.klimit is not None: jn *= (knorm >= self.klimit[0]) * (knorm < self.klimit[1])
            return self.mattrs.c2r(values[axis] * jn)

        def reduce(meshes):
            return prod(meshes, values[2]).sum()

        return self.bin_reduce_meshs(bin, reduce) * norm



_format_meshes = partial(__format_meshes, nmeshes=3)


def _format_ells(ells, basis: str='sugiyama'):
    """
    Format multipole orders for bispectrum binning,
    depending on 'basis'.

    - 'sugiyama': list of 3-tuples
    - 'scoccimaro': list of integers
    """
    if 'scoccimarro' in basis:
        if np.ndim(ells) == 0:
            ells = [ells]
        ells = list(ells)
    else:
        msg = 'ells must be (a list of) (ell1, ell2, L)'
        assert np.ndim(ells) != 0, msg
        if np.ndim(ells[0]) == 0:
            assert len(ells) == 3, msg
            ells = [ells]
        ells = list(ells)
    return ells


def _format_los(los, ndim=3):
    """Format the line-of-sight specification."""
    vlos, swap = None, False
    if isinstance(los, str) and los in ['local']:
        pass
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    return los, vlos


def _iter_triposh(*ells, los='local'):
    """Iterate over allowed m combinations for Gaunt coefficients."""
    ell1, ell2, ell3 = ells
    ms = [np.arange(-ell, ell + 1) for ell in ells]
    if los == 'z':
        ms[-1] = [0]
    toret, acc = [], []
    for m1, m2, m3 in itertools.product(*ms):
        # In https://arxiv.org/pdf/1803.02132, the total coefficient is
        # H(ell1 ell2, L) wigner_3j(ell1, ell2, L, m1, m2, M) (2ell1 + 1) (2ell2 + 1) (2L + 1)
        # i.e. (2ell1 + 1) (2ell2 + 1) (2L + 1) wigner_3j(ell1, ell2, L, 0, 0, 0) wigner_3j(ell1 ell2 L, m1, m2, M)
        # Gaunt below is:
        # sqrt((2ell1 + 1) (2ell2 + 1) (2L + 1) / 4pi) wigner_3j(ell1, ell2, L, 0, 0, 0) wigner_3j(ell1 ell2 L, m1, m2, M)
        # The ratio between the 2 is compensated by our definition of Spherical Harmonics, which includes coefficents sqrt((2ell + 1) / 4 pi)
        #gaunt = real_gaunt((ell1, im1), (ell2, im2), (ell3, im3))
        gaunt = (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) * wigner_3j(ell1, ell2, ell3, 0, 0, 0) * wigner_3j(ell1, ell2, ell3, m1, m2, m3)
        if abs(gaunt) < 1e-7:
            continue
        sym = 0.
        neg = (-m1, -m2, -m3)
        if neg in acc:
            idx = acc.index(neg)
            toret[idx][-1] = (-1)**ell3
            continue
        toret.append([m1 + ell1, m2 + ell2, m3 + ell3, gaunt, sym])  # m indexing starting from 0
        acc.append(toret[-1][:3])
    if toret:
        return [np.array(xx) for xx in zip(*toret)]
    return [np.zeros((0,), dtype=int) for _ in range(5)]



def compute_mesh3(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh3SpectrumPoles | BinMesh3CorrelationPoles=None, los: str | np.ndarray='z'):
    """
    Dispatch to :func:`compute_mesh3_spectrum` or :func:`compute_mesh3_correlation`
    depending on type of input ``bin``.

    Parameters
    ----------
    meshes : RealMeshField or ComplexMeshField
        Input meshes.
    bin : BinMesh3SpectrumPoles or BinMesh3CorrelationPoles
        Binning operator.
    los : str or array-like, optional
        Line-of-sight specification.
        If ``los`` is 'local', use local (varying) line-of-sight.
        Else, global line-of-sight: may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector. In case of the sugiyama basis, 'z' only is supported.

    Note
    ----
    jit is recommended when running for many realizations.
    Otherwise OOM may occur.

    Returns
    -------
    result : Mesh3SpectrumPoles or Mesh3CorrelationPoles
    """
    if isinstance(bin, BinMesh3SpectrumPoles):
        return compute_mesh3_spectrum(*meshes, bin=bin, los=los)
    elif isinstance(bin, BinMesh3CorrelationPoles):
        return compute_mesh3_correlation(*meshes, bin=bin, los=los)
    raise ValueError(f'bin must be either BinMesh3SpectrumPoles or BinMesh3CorrelationPoles, not {type(bin)}')



def compute_mesh3_spectrum(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh3SpectrumPoles=None, los: str | np.ndarray='x'):
    """
    Compute the bispectrum multipoles from mesh.

    Parameters
    ----------
    meshes : RealMeshField or ComplexMeshField
        Input meshes.
    bin : BinMesh3SpectrumPoles
        Binning operator.
    los : str or array-like, optional
        Line-of-sight specification.
        If ``los`` is 'local', use local (varying) line-of-sight.
        Else, global line-of-sight: may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector. In case of the sugiyama basis, 'z' only is supported.

    Note
    ----
    jit is recommended when running for many realizations.
    Otherwise OOM may occur.

    Returns
    -------
    result : Mesh3SpectrumPoles
    """
    meshes, fields = _format_meshes(*meshes)
    rdtype = meshes[0].real.dtype
    mattrs = meshes[0].attrs
    if 'sugiyama' in bin.basis:
        mattrs = mattrs.clone(dtype=mattrs.cdtype)
    meshes = [mesh.clone(attrs=mattrs) for mesh in meshes]
    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)**2

    los, vlos = _format_los(los, ndim=mattrs.ndim)
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

    # The real-space grid
    xvec = mattrs.rcoords(sparse=True)
    # The Fourier-space grid
    kvec = mattrs.kcoords(sparse=True)

    num = []
    if 'scoccimarro' in bin.basis:

        if vlos is None:
            meshes = [_2c(mesh) for mesh in meshes[:2]] + [_2r(meshes[2])]

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += _2c(meshes[2] * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            for ill3, ell3 in enumerate(ells):
                Ylms = [get_Ylm(ell3, m, reduced=False, real=True) for m in range(-ell3, ell3 + 1)]
                xs = np.arange(len(Ylms))
                tmp = tuple(meshes[i] for i in range(2)) + (jax.lax.scan(partial(f, Ylms), init=meshes[0].clone(value=jnp.zeros_like(meshes[0].value)), xs=xs)[0],)
                tmp = (4. * np.pi) * bin(*tmp) / bin.nmodes[ill3]
                num.append(tmp)

        else:

            meshes = [_2c(mesh) for mesh in meshes]
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)

            for ill3, ell3 in enumerate(ells):
                tmp = meshes[:2] + [meshes[2] * get_legendre(ell3)(mu)]
                #tmp = (2 * ell3 + 1) * bin(*tmp, remove_zero=True) / bin.nmodes[ill3]
                tmp = (2 * ell3 + 1) * bin(*tmp) / bin.nmodes[ill3]
                num.append(tmp)

            num = [num.real if ell % 2 == 0 else num.imag for ell, num in zip(ells, num)]

    else:

        meshes = [_2c(mesh) for mesh in meshes[:2]] + [_2r(meshes[2])]

        def get_f(ells):
            Ylms = [[get_Ylm(ell, m, reduced=True, real=False, conj=True) for m in range(-ell, ell + 1)] for ell in ells]
            xs = _iter_triposh(*ells, los=los)
            branches = []
            for row in zip(*xs):
                def branch(kvec, los, row=row):
                    coeff, sym, im = row[3], row[4], row[:3]
                    tmp = tuple(meshes[i] * Ylms[i][im[i]](*kvec) for i in range(2))
                    tmp += (Ylms[2][im[2]](*los) * meshes[2],)
                    tmp = coeff.astype(mattrs.rdtype) * bin(*tmp)  # remove_zero=True)
                    # Cast back to the mesh dtype: with x64 enabled (e.g.
                    # flipped globally by a cosmoprimo import), the numpy
                    # complex128 Ylm constants promote the branch output and
                    # break the complex64 scan carry.
                    return (tmp + sym.astype(mattrs.rdtype) * tmp.conj()).astype(mattrs.cdtype)
                branches.append(branch)

            def f(carry, idx):
                los = xvec if vlos is None else vlos
                carry += jax.lax.switch(idx, branches, kvec, los)
                return carry, idx

            init = jnp.zeros(len(bin.edges), dtype=mattrs.cdtype)
            return f, init, np.arange(len(branches))

        for ill, (ell1, ell2, ell3) in enumerate(ells):
            f, init, xs = get_f((ell1, ell2, ell3))
            if xs.size:
                num_ = jax.lax.scan(f, init=init, xs=xs)[0] / bin.nmodes[ill]
            else:
                num_ = init
            num.append(num_.real if (ell1 + ell2) % 2 == 0 else num_.imag)

    spectrum = []
    for ill, ell in enumerate(ells):
        spectrum.append(Mesh3SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes[ill], num_raw=num[ill], norm=norm, attrs=attrs, basis=bin.basis, ell=ell))
    return Mesh3SpectrumPoles(spectrum)



def compute_mesh3_correlation(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh3CorrelationPoles=None, los: str | np.ndarray='x'):
    """
    Compute the 3pcf multipoles from mesh.

    Parameters
    ----------
    meshes : RealMeshField or ComplexMeshField
        Input meshes.
    bin : BinMesh3CorrelationPoles
        Binning operator.
    los : str or array-like, optional
        Line-of-sight specification.
        If ``los`` is 'local', use local (varying) line-of-sight.
        Else, global line-of-sight: may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector. In case of the sugiyama basis, 'z' only is supported.

    Note
    ----
    jit is recommended when running for many realizations.
    Otherwise OOM may occur.

    Returns
    -------
    result : Mesh3CorrelationPoles
    """
    meshes, fields = _format_meshes(*meshes)
    rdtype = meshes[0].real.dtype
    mattrs = meshes[0].attrs
    if 'sugiyama' in bin.basis:
        mattrs = mattrs.clone(dtype=mattrs.cdtype)
    meshes = [mesh.clone(attrs=mattrs) for mesh in meshes]

    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)**2

    los, vlos = _format_los(los, ndim=mattrs.ndim)
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

    # The real-space grid
    xvec = mattrs.rcoords(sparse=True)
    # The Fourier-space grid
    kvec = mattrs.kcoords(sparse=True)

    num = []
    if 'scoccimarro' in bin.basis:

        raise NotImplementedError

    else:

        meshes = [_2c(mesh) for mesh in meshes[:2]] + [_2r(meshes[2])]

        def get_f(ells):
            Ylms = [[get_Ylm(ell, m, reduced=True, real=False, conj=True) for m in range(-ell, ell + 1)] for ell in ells]
            xs = _iter_triposh(*ells, los=los)
            branches = []
            for row in zip(*xs):
                def branch(kvec, los, row=row):
                    coeff, sym, im = row[3], row[4], row[:3]
                    tmp = tuple(meshes[i] * Ylms[i][im[i]](*kvec) for i in range(2))
                    tmp += (Ylms[2][im[2]](*los) * meshes[2],)
                    tmp = coeff.astype(mattrs.rdtype) * bin(*tmp, ell=ells) # remove_zero=True)
                    return tmp + sym.astype(mattrs.rdtype) * tmp.conj()
                branches.append(branch)

            def f(carry, idx):
                # Cast a global LOS to the mesh's real dtype: _get_los_vector
                # builds it from Python floats, so it is float64, and the Ylm
                # evaluated on it then promotes the branch output to complex128
                # while `carry` is mattrs.cdtype, that can be complex64.
                los = xvec if vlos is None else tuple(jnp.asarray(ll, dtype=mattrs.rdtype) for ll in vlos)
                carry += jax.lax.switch(idx, branches, kvec, los)
                return carry, idx

            init = jnp.zeros(len(bin.edges), dtype=mattrs.cdtype)
            return f, init, np.arange(len(branches))

        for (ell1, ell2, ell3) in ells:
            f, init, xs = get_f((ell1, ell2, ell3))
            if xs.size:
                num_ = jax.lax.scan(f, init=init, xs=xs)[0]
            else:
                num_ = init
            num.append(num_.real)
        #num.append(bin(*meshes, ell=ells[0]).real)

    correlation = []
    for ill, ell in enumerate(ells):
        correlation.append(Mesh3CorrelationPole(s=bin.xavg, s_edges=bin.edges, nmodes=bin.nmodes[ill], num_raw=num[ill], norm=norm, attrs=attrs, basis=bin.basis, ell=ell))
    return Mesh3CorrelationPoles(correlation)


def compute_box3_normalization(*inputs: RealMeshField | ParticleField, bin: BinMesh3SpectrumPoles=None) -> jax.Array:
    """Compute normalization, assuming constant density, for the bispectrum."""
    norm = compute_box_normalization(*_format_meshes(*inputs)[0])
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_fkp3_normalization(*fkps, bin: BinMesh3SpectrumPoles=None, cellsize=10., split=None, fields: tuple=None, **kwargs):
    """
    Compute the FKP normalization for the bispectrum.

    Parameters
    ----------
    fkps : FKPField or PaticleField
        FKP or particle fields.
    bin : BinMesh3SpectrumPoles, optional
        Binning operator. Only used to return a list of normalization factors for each multipole.
    cellsize : float, optional
        Cell size.
    split : int or None, optional
        Random seed for splitting.
        This is useful to get unbiased estimate of the normalization.
        The input particle fields are split into 3 disjoint samples.
        Each sample is painted on a mesh, and the normalization is computed from the product of the 3 meshes.
        If ``None``, no splitting is performed.
    fields : tuple, default=None
        Field identifiers; pass e.g. [0, 0, 1] if the first two fields share the same positions;
        disjoint random subsamples will be selected with ``split``.
    kwargs : dict
        Optional arguments for :func:`compute_normalization`.

    Returns
    -------
    norm : float, list
    """
    kwargs.update(cellsize=cellsize)
    fkps, fields = _format_meshes(*fkps, fields=fields)
    alpha = prod(map(lambda fkp: fkp.data.sum() / fkp.randoms.sum() if isinstance(fkp, FKPField) else 1., fkps))

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    if split is not None:
        randoms = list(split_particles(*[get_randoms(fkp) for fkp in fkps], seed=split, fields=fields))
        alpha *= prod(get_randoms(fkp).sum() / randoms.sum() for fkp, randoms in zip(fkps, randoms, strict=True))
    else:
        fkps, fields = _format_meshes(*fkps)
        randoms = [get_randoms(fkp) for fkp in fkps]
    norm = alpha * compute_normalization(*randoms, **kwargs)
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


@partial(jax.jit, donate_argnums=[0, 1], static_argnames=['los'])
def _compute_scoccimarro_S(cmeshw, cmeshw2, sumw3, bin=None, los='z'):

    mattrs = cmeshw.attrs
    ells = bin.ells
    shotnoise = [jnp.zeros_like(bin.xavg[..., 0]) for ill in range(len(ells))]

    los, vlos = _format_los(los, ndim=mattrs.ndim)
    # The real-space grid
    xvec = mattrs.rcoords(sparse=True)
    # The Fourier-space grid
    kvec = mattrs.kcoords(sparse=True)

    mattrs = cmeshw.attrs

    def bin_mesh2(mesh, axis):
        nmodes1d = bin.nmodes1d[axis]
        return _bincount(bin.ibin1d[axis] + 1, getattr(mesh, 'value', mesh), weights=bin.wmodes, length=len(nmodes1d)) / nmodes1d

    def apply_fourier_legendre(ell, cmesh):
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)
        return get_legendre(ell)(mu) * cmesh

    def apply_fourier_harmonics(ell, rmesh):
        Ylms = [get_Ylm(ell, m, reduced=False, real=True) for m in range(-ell, ell + 1)]

        @partial(jax.checkpoint, static_argnums=(0, 1))
        def f(Ylm, carry, im):
            inc = (rmesh * jax.lax.switch(im, Ylm, *xvec)).r2c() * jax.lax.switch(im, Ylm, *kvec)
            # Cast to the carry dtype (see _compute_sugiyama_spectrum_S113).
            carry += inc.clone(value=inc.value.astype(carry.value.dtype))
            return carry, im

        xs = np.arange(len(Ylms))
        return (4. * jnp.pi) / (2 * ell + 1) * jax.lax.scan(partial(f, Ylms), init=mattrs.create(fill=0., kind='complex'), xs=xs)[0]

    for ill, ell in enumerate(ells):
        if ell == 0:
            cmeshw3_ell = cmeshw * cmeshw2.conj()
            shotnoise[ill] = sum(bin_mesh2(cmeshw3_ell, axis=axis)[bin._iedges[..., axis]] for axis in range(mattrs.ndim)) - 2. * sumw3
        else:
            # First line of eq. 58, q1 => q3
            if vlos is not None:
                cmeshw_ell = apply_fourier_legendre(ell, cmeshw)
            else:
                cmeshw_ell = apply_fourier_harmonics(ell, cmeshw.c2r())
            sn_ell = bin_mesh2(cmeshw_ell * cmeshw2.conj(), axis=mattrs.ndim - 1)[bin._iedges[..., mattrs.ndim - 1]]
            del cmeshw_ell

            if vlos is not None:
                cmeshw3_ell = cmeshw * apply_fourier_legendre(ell, cmeshw2).conj()
            else:
                cmeshw3_ell = cmeshw * apply_fourier_harmonics(ell, cmeshw2.c2r()).conj()

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                Ylm = jax.lax.switch(im, Ylm, *kvec) * jnp.ones_like(cmeshw3_ell)
                # Second line of eq. 58
                tmp = cmeshw3_ell * Ylm
                tmp = [bin_mesh2(tmp, axis=axis) for axis in range(mattrs.ndim - 1)] + [bin_mesh2(Ylm, axis=mattrs.ndim - 1)]
                # Cast to the carry dtype (see _compute_sugiyama_spectrum_S113).
                carry += ((4 * jnp.pi) * sum(tmp[axis][bin._iedges[..., axis]] * tmp[mattrs.ndim - 1][bin._iedges[..., mattrs.ndim - 1]] for axis in range(mattrs.ndim - 1))).astype(carry.dtype)
                return carry, im

            Ylms = [get_Ylm(ell, m, reduced=False, real=True) for m in range(-ell, ell + 1)]
            xs = np.arange(len(Ylms))
            shotnoise[ill] = (2 * ell + 1) * jax.lax.scan(partial(f, Ylms), init=sn_ell, xs=xs)[0]

    return shotnoise


@partial(jax.jit, donate_argnums=[0, 1, 2, 3], static_argnames=['ells', 'los', 'axis'])
def _compute_sugiyama_spectrum_S122(rmesh, cmesh, s111=dict(), cmesh_s111=1., bin=None, ells=tuple(), los='z', axis=0):

    mattrs = rmesh.attrs
    los, vlos = _format_los(los, ndim=mattrs.ndim)

    def bin_mesh2(bin, mesh, axis):
        nmodes1d = bin.nmodes1d[axis]
        return _bincount(bin.ibin1d[axis] + 1, getattr(mesh, 'value', mesh), weights=bin.wmodes, length=len(nmodes1d)) / nmodes1d

    rmesh -= rmesh.mean()
    cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode
    xvec = mattrs.rcoords(sparse=True)
    kvec = mattrs.kcoords(sparse=True)

    @partial(jax.checkpoint, static_argnums=0)
    def f(Ylm, carry, im):
        im, s111 = im
        # Second and third lines
        los = xvec if vlos is None else vlos
        inc = jax.lax.switch(im, Ylm, *kvec) * (cmesh * (rmesh * jax.lax.switch(im, Ylm, *los)).r2c().conj() - s111 * cmesh_s111)
        # Cast to the carry dtype (see _compute_sugiyama_spectrum_S113).
        carry += inc.clone(value=inc.value.astype(carry.value.dtype))
        return carry, im

    s122 = []
    for ell in ells:
        Ylms = [get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)]
        xs = (jnp.arange(len(Ylms)), jnp.array([s111[ell, m] for m in range(-ell, ell + 1)]))
        s122.append((2 * ell + 1) * bin_mesh2(bin, jax.lax.scan(partial(f, Ylms), init=cmesh.clone(value=jnp.zeros_like(cmesh.value)), xs=xs)[0], axis))
    return s122


@partial(jax.jit, donate_argnums=[0, 1, 2], static_argnames=['ells', 'los'])
def _compute_sugiyama_spectrum_S113(rmesh, cmesh, s111=dict(), cmesh_s111=1., bin=None, ells=tuple(), los='z'):

    mattrs = rmesh.attrs
    los, vlos = _format_los(los, ndim=mattrs.ndim)

    rmesh -= rmesh.mean()
    cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode
    xvec = mattrs.rcoords(sparse=True)
    svec = mattrs.rcoords(kind='separation', sparse=True)

    @partial(jax.checkpoint, static_argnums=(0, 1))
    def f(Ylms, jl, carry, im):
        los = xvec if vlos is None else vlos
        s111, coeff, sym, im = im[-1], im[3], im[4], im[:3]
        # Fourth line
        tmp = coeff * ((rmesh * jax.lax.switch(im[2], Ylms[2], *los).conj()).r2c() * cmesh.conj() - s111 * cmesh_s111)
        snorm = jnp.sqrt(sum(xx**2 for xx in svec))
        tmp = tmp.c2r() * (jax.lax.switch(im[0], Ylms[0], *svec) * jax.lax.switch(im[1], Ylms[1], *svec)).conj()

        def fk(k):
            return jnp.sum(tmp.value * jl[0](snorm * k[0]) * jl[1](snorm * k[1]))

        tmp = jax.lax.map(fk, bin.xavg)
        # Cast to the carry dtype: with x64 enabled (e.g. flipped globally by
        # a cosmoprimo import) the complex Ylm / s111 constants promote the
        # increment and break the scan carry.
        carry += (tmp + sym * tmp.conj()).astype(carry.dtype)
        return carry, im

    s113 = []
    for ell1, ell2, ell3 in ells:
        Ylms = [[get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)] for ell in [ell1, ell2, ell3]]
        xs = _iter_triposh(ell1, ell2, ell3, los=los)
        if xs[0].size:
            # Add s111 for s, im in xs are offset by ell
            xs = xs + [jnp.array([s111[ell3, int(im3) - ell3] for im3 in xs[2]])]
            sign = (1j)**(ell1 + ell2)
            s113.append(sign * jax.lax.scan(partial(f, Ylms, [get_spherical_jn(ell1), get_spherical_jn(ell2)]), init=jnp.zeros_like(bin.xavg[..., 0], dtype=mattrs.cdtype), xs=xs)[0])
        else:
            s113.append(jnp.zeros_like(bin.xavg[..., 0]))
    return s113


@partial(jax.jit, donate_argnums=[0, 1, 2, 3], static_argnames=['ells', 'los', 'axis'])
def _compute_sugiyama_correlation_S122(rmesh, cmesh, s111=dict(), cmesh_s111=1., bin=None, ells=tuple(), los='z', axis=0):

    mattrs = rmesh.attrs
    los, vlos = _format_los(los, ndim=mattrs.ndim)

    def bin_mesh2(mesh, axis):
        nmodes1d = bin.nmodes1d[axis]
        return _bincount(bin.ibin1d[axis] + 1, getattr(mesh, 'value', mesh), weights=bin.wmodes, length=len(nmodes1d)) / nmodes1d

    rmesh -= rmesh.mean()
    cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode
    xvec = mattrs.rcoords(sparse=True)
    svec = mattrs.rcoords(kind='separation', sparse=True)

    @partial(jax.checkpoint, static_argnums=0)
    def f(Ylm, carry, im):
        im, s111 = im
        # Second and third lines
        los = xvec if vlos is None else vlos
        inc = jax.lax.switch(im, Ylm, *svec) * (cmesh * (rmesh * jax.lax.switch(im, Ylm, *los)).r2c().conj() - s111 * cmesh_s111).c2r()
        # Cast to the carry dtype (see _compute_sugiyama_spectrum_S113).
        carry += inc.clone(value=inc.value.astype(carry.value.dtype))
        return carry, im

    s122 = []
    for ell in ells:
        Ylms = [get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)]
        xs = (jnp.arange(len(Ylms)), jnp.array([s111[ell, m] for m in range(-ell, ell + 1)]))
        s122.append((2 * ell + 1) * bin_mesh2(jax.lax.scan(partial(f, Ylms), init=rmesh.clone(value=jnp.zeros_like(rmesh.value, dtype=mattrs.cdtype)), xs=xs)[0], axis))
    return s122


@partial(jax.jit, donate_argnums=[0, 1, 2], static_argnames=['ells', 'los'])
def _compute_sugiyama_correlation_S113(rmesh, cmesh, s111=dict(), cmesh_s111=1., bin=None, inter_bin=None, ells=tuple(), los='z'):

    mattrs = rmesh.attrs
    los, vlos = _format_los(los, ndim=mattrs.ndim)

    rmesh -= rmesh.mean()
    cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode
    xvec = mattrs.rcoords(sparse=True)
    svec = mattrs.rcoords(kind='separation', sparse=True)

    inter_ibin, uinter_edges, M = inter_bin

    def bin_mesh2_inter(mesh):
        tmp = M @ _bincount(inter_ibin[0], getattr(mesh, 'value', mesh), length=len(uinter_edges) - 1)
        return tmp / (bin.nmodes1d[0][bin._iedges[..., 0]] * bin.nmodes1d[1][bin._iedges[..., 1]])

    @partial(jax.checkpoint, static_argnums=(0, 1))
    def f(Ylms, carry, im):
        los = xvec if vlos is None else vlos
        s111, coeff, sym, im = im[-1], im[3], im[4], im[:3]
        # Fourth line
        tmp = coeff * ((rmesh * jax.lax.switch(im[2], Ylms[2], *los).conj()).r2c() * cmesh.conj() - s111 * cmesh_s111)
        tmp = tmp.c2r() * (jax.lax.switch(im[0], Ylms[0], *svec) * jax.lax.switch(im[1], Ylms[1], *svec)).conj()
        tmp = bin_mesh2_inter(tmp)
        # Cast to the carry dtype (see _compute_sugiyama_spectrum_S113).
        carry += (tmp + sym * tmp.conj()).astype(carry.dtype)
        return carry, im

    s113 = []
    for ell1, ell2, ell3 in ells:
        Ylms = [[get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)] for ell in [ell1, ell2, ell3]]
        xs = _iter_triposh(ell1, ell2, ell3, los=los)
        if xs[0].size:
            # Add s111 for s, im in xs are offset by ell
            xs = xs + [jnp.array([s111[ell3, int(im3) - ell3] for im3 in xs[2]])]
            sign = (-1)**(ell1 + ell2)
            s113.append(sign * jax.lax.scan(partial(f, Ylms), init=jnp.zeros_like(bin.xavg[..., 0], dtype=mattrs.cdtype), xs=xs)[0])
        else:
            s113.append(jnp.zeros_like(bin.xavg[..., 0]))
    return s113


def compute_fkp3_shotnoise(*fkps, bin=None, los: str | np.ndarray='z', resampler='cic', interlacing=False, compensate=True, fields: tuple=None, **kwargs):
    r"""
    Compute the FKP shot noise for the bispectrum.

    Parameters
    ----------
    fkps : FKPField
        FKP fields.
    bin : BinMesh3SpectrumPoles or BinMesh3CorrelationPoles, optional
        Binning operator.
    los : str or array-like, optional
        Line-of-sight specification.
        If ``los`` is 'local', use local (varying) line-of-sight.
        Else, global line-of-sight: may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector. In case of the sugiyama basis, 'z' only is supported.
    resampler : str, Callable
        Resampler to read particule weights from mesh.
        One of ['ngp', 'cic', 'tsc', 'pcs'].
    interlacing : int, default=0
        If 0 or 1, no interlacing correction.
        If > 1, order of interlacing correction.
        Typically, 3 gives reliable power spectrum estimation up to :math:`k \sim k_\mathrm{nyq}`.
    compensate : bool, default=True
        If ``True``, applies compensation to the mesh after painting.
    fields : tuple, list, optional
        Field identifiers; pass e.g. [0, 0, 1] if the first two fields share the same positions.
    kwargs : dict
        Additional arguments for :meth:`ParticleField.paint`.

    Returns
    -------
    shotnoise : list
        Shot noise for each multipole.
    """
    fkps, fields = _format_meshes(*fkps, fields=fields)
    mattrs = fkps[0].attrs
    if 'sugiyama' in bin.basis:
        mattrs = mattrs.clone(dtype=mattrs.cdtype)
    fkps = [fkp.clone(attrs=mattrs) for fkp in fkps]

    # ells
    ells = bin.ells
    shotnoise = [jnp.zeros_like(bin.xavg[..., 0]) for ill in range(len(ells))]
    kwargs.update(resampler=resampler, interlacing=interlacing, compensate=compensate)

    from . import resamplers
    resampler = resamplers.get_resampler(resampler)
    interlacing = max(interlacing, 1) >= 2

    if fields[2] == fields[1] + 1 == fields[0] + 2:
        return tuple(shotnoise)

    particles = []
    for fkp, s in zip(fkps, fields):
        if s < len(particles):
            particles.append(particles[s])
        else:
            if isinstance(fkp, FKPField):
                fkp = fkp.particles
            particles.append(fkp)

    mattrs = particles[0].attrs

    is_correlation = isinstance(bin, BinMesh3CorrelationPoles)

    if 'scoccimarro' in bin.basis:

        if is_correlation: raise NotImplementedError

        # Eq. 58 of https://arxiv.org/pdf/1506.02729, 1 => 3
        if not (fields[0] == fields[1] == fields[2]):
            raise NotImplementedError
        cmeshw = particles[2].paint(**kwargs, out='complex')
        cmeshw = cmeshw.clone(value=cmeshw.value.at[(0,) * cmeshw.ndim].set(0.))  # remove zero-mode
        cmeshw2 = particles[0].clone(weights=particles[0].weights * particles[1].weights).paint(**kwargs, out='complex')
        cmeshw2 = cmeshw2.clone(value=cmeshw2.value.at[(0,) * cmeshw2.ndim].set(0.))  # remove zero-mode
        sumw3 = jnp.sum(particles[0].weights * particles[1].weights * particles[2].weights)
        del particles

        shotnoise = _compute_scoccimarro_S(cmeshw, cmeshw2, sumw3, bin=bin, los=los)
        shotnoise = [sn.real if ell % 2 == 0 else sn.imag for ell, sn in zip(ells, shotnoise)]

    else:
        # Eq. 45 - 46 of https://arxiv.org/pdf/1803.02132
        def compute_S111(particles, ellms):
            ellms = list(ellms)
            if ellms == [(0, 0)]:
                s0 = jnp.sum(particles[0].weights * particles[1].weights * particles[2].weights)
                s111_ellm = {(ell, m): s0 if (ell, m) == (0, 0) else 0. for ell, m in ellms}
            else:
                rmesh = particles[0].clone(weights=particles[0].weights**3).paint(**kwargs, out='real')
                xvec = mattrs.rcoords(sparse=True)
                s111_ellm = {(ell, m): jnp.sum(rmesh.value * get_Ylm(ell, m, reduced=True, real=False, conj=True)(*xvec)) for ell, m in ellms}
            s111 = s111_ellm.get((0, 0), 0.)
            if is_correlation:
                mask = 1. / mattrs.cellsize.prod()**2 * jnp.all((bin.edges[..., 0] <= 0.) & (bin.edges[..., 1] >= 0.), axis=1)
                s111 *= mask
            return s111, s111_ellm

        def compensate_shotnoise():
            # NOTE: when there is no interlacing, triumvirate compensates pk with aliasing_shotnoise in the shotnoise estimation (but in bk estimation?)
            # In jaxpower, we always compensate by the standard compensate
            #if convention == 'triumvirate' and not interlacing:
            kcirc = mattrs.kcoords(kind='circular', sparse=True)
            if interlacing:
                return (1. if compensate else 1. / resampler.compensate(1., kcirc)**2)
            else:
                return resampler.aliasing_shotnoise(1., kcirc) * (resampler.compensate(1., kcirc)**2 if compensate else 1.)

        def compute_S122(particles, ells, axis=0):  # 1 == 2
            rmeshw2 = particles[1].clone(weights=particles[1].weights * particles[2].weights).paint(**kwargs, out='real')
            cmeshw = particles[0].paint(**kwargs, out='complex')

            if is_correlation:
                s122 = _compute_sugiyama_correlation_S122(rmeshw2, cmeshw, s111_ellm, cmesh_s111=compensate_shotnoise(), bin=bin, ells=tuple(ells), los=los, axis=axis)
                mask = 1. / mattrs.cellsize.prod()**2 * ((bin.edges[..., 1 - axis, 0] <= 0.) & (bin.edges[..., 1 - axis, 1] >= 0.))
            else:
                s122 = _compute_sugiyama_spectrum_S122(rmeshw2, cmeshw, s111_ellm, cmesh_s111=compensate_shotnoise(), bin=bin, ells=tuple(ells), los=los, axis=axis)
                mask = 1.
            return [s[bin._iedges[..., axis]] * mask for s in s122]

        def compute_S113(particles, ells):
            rmeshw = particles[2].paint(**kwargs, out='real')
            cmeshw2 = particles[0].clone(weights=particles[0].weights * particles[1].weights).paint(**kwargs, out='complex')
            if is_correlation:
                inter_edges = jnp.concatenate([jnp.max(bin.edges[..., 0], axis=1, keepdims=True), jnp.min(bin.edges[..., 1], axis=1, keepdims=True)], axis=-1)
                inter_edges = jnp.where(inter_edges[..., [1]] <= inter_edges[..., [0]], 0., inter_edges)
                from .mesh import _get_bin_attrs_edges2d
                inter_bin = _get_bin_attrs_edges2d(jnp.sqrt(sum(ss**2 for ss in mattrs.rcoords(sparse=True, kind='separation'))), inter_edges)
                s113 = _compute_sugiyama_correlation_S113(rmeshw, cmeshw2, s111_ellm, cmesh_s111=compensate_shotnoise(), bin=bin, inter_bin=inter_bin, ells=tuple(ells), los=los)
                mask = 1. / mattrs.cellsize.prod()**2
            else:
                s113 = _compute_sugiyama_spectrum_S113(rmeshw, cmeshw2, s111_ellm, cmesh_s111=compensate_shotnoise(), bin=bin, ells=tuple(ells), los=los)
                mask = 1.
            return [s * mask for s in s113]

        uells = sorted(sum(ells, start=tuple()))
        ellms = [(ell, m) for ell in uells for m in range(-ell, ell + 1)]
        s111_ellm = {(ell, m): 0 for ell, m in ellms}

        if fields[0] == fields[1] == fields[2]:
            s111, s111_ellm = compute_S111(particles, tuple(ellms))
            ell0 = (0, 0, 0)
            if ell0 in ells:
                shotnoise[ells.index(ell0)] += s111

        if fields[1] == fields[2]:
            def select(ell):
                return ell[2] == ell[0] and ell[1] == 0

            ells1 = [ell[0] for ell in ells if select(ell)]
            if ells1:
                particles01 = particles
                s122 = compute_S122(particles01, ells=ells1, axis=0)

                for ill, ell in enumerate(ells):
                    if select(ell):
                        idx = ells1.index(ell[0])
                        shotnoise[ill] += s122[idx]

        if fields[0] == fields[2]:
            def select(ell):
                return ell[2] == ell[1] and ell[0] == 0

            ells2 = [ell[1] for ell in ells if select(ell)]
            if ells2:
                particles01 = [particles[1], particles[0], particles[2]]
                s121 = compute_S122(particles01, ells=ells2, axis=1)
                for ill, ell in enumerate(ells):
                    if select(ell):
                        idx = ells2.index(ell[1])
                        shotnoise[ill] += s121[idx]

        if fields[0] == fields[1]:
            s113 = compute_S113(particles, ells=ells)
            for ill, ell in enumerate(ells):
                shotnoise[ill] += s113[ill]

        shotnoise = [sn.real if is_correlation or (sum(ell[:2]) % 2 == 0) else sn.imag for ell, sn in zip(ells, shotnoise)]

    return list(shotnoise)


def get_sugiyama_window_convolution_coeffs(ell, ellin):  # observed ell, theory ell
    # Prefactor, eq. 63 of https://arxiv.org/pdf/1803.02132
    # ell = (ell_1, ell_2, L)
    # ellin = (ell_1', ell_2', L')
    coeffs = []
    H = wigner_3j(*ell, 0, 0, 0)
    if abs(H) < 1e-7: return coeffs
    #for ellw in itertools.product(*([range(max(ell) + max(ellin) + 1)] * 3)):
    for ellw in itertools.product(*[range(ell_ + ellin_ + 1) for ell_, ellin_ in zip(ell, ellin)]):
        if sum(ellw) % 2: continue
        if ellw[2] % 2: continue
        # Cheap 3j triangle / parity pruning before the expensive 9j
        Hw = wigner_3j(*ellw, 0, 0, 0)
        if abs(Hw) < 1e-7: continue
        coeff = prod((2 * ell_ + 1) for ell_ in ell) * H
        for i in range(3): coeff *= wigner_3j(ell[i], ellin[i], ellw[i], 0, 0, 0)
        if abs(coeff) < 1e-7: continue
        coeff *= wigner_9j(*ellw, *ellin, *ell)
        if abs(coeff) < 1e-7: continue
        coeff /= wigner_3j(*ellin, 0, 0, 0) * Hw
        coeffs.append((ellw, coeff))
    return coeffs


@functools.lru_cache(maxsize=None)
def get_sugiyama_covariance_window_convolution_coeffs(ell, ellin):
    r"""Window-multipole coefficients for the *covariance* 4-point kernel.

    ``ell`` = (L1, L2, J) indexes the unprimed-side S-basis channel and
    ``ellin`` = (L1', L2', J') the primed-side one (both z3, M = 0). Returns
    the list of (q, coeff) such that the 4-point angular window kernel is

    .. math::
        Q_W(k_1, k_1', k_2, k_2')
        = \sum_{\ell,\ell'} \Big[\sum_q c_q\, \mathrm{Hankel}_{L_1 L_1' L_2 L_2'}[Q_{W,q}]\Big]
          S_\ell(\hat k_1, \hat k_2)\, S_{\ell'}(\hat k_1', \hat k_2'),

    following the :math:`\mathcal C^{\lambda_1\lambda_2\Lambda}_{L_1L_1'L_2L_2'}`
    kernel of ``_cov3_math.tex``, keeping only its N = 0 term (the N != 0
    azimuthal channels vanish identically under the estimators' independent
    per-side orientation averages). This differs from
    :func:`get_sugiyama_window_convolution_coeffs` (the *mean* bispectrum
    window convolution, eq. 63 of arXiv:1803.02132): here the monopole
    window feeds each diagonal channel with the Parseval weight
    :math:`(2L_1+1)(2L_2+1)(2J+1) H_{L_1L_2J}^2 = 1 / \| S_\ell \|^2`.

    The relative phase :math:`(-i)^{L_1+L_2} i^{L_1'+L_2'}` is real
    (:math:`\pm 1`) for every allowed q: the two triangle conditions with
    even-sum q force :math:`(L_1'-L_1)+(L_2'-L_2)` even. It is included here
    because the covariance Hankel matrices drop the transforms'
    :math:`i^\ell` prefactors. Normalization is anchored so that the
    ((0,0,0), (0,0,0)) channel has coeff((0,0,0)) = 1, matching the box
    limit.
    """
    L1, L2, J = ell
    L1p, L2p, Jp = ellin
    HJ = wigner_3j(L1, L2, J, 0, 0, 0)
    HJp = wigner_3j(L1p, L2p, Jp, 0, 0, 0)
    if abs(HJ) < 1e-12 or abs(HJp) < 1e-12:
        return []
    coeffs = []
    for q in itertools.product(range(L1 + L1p + 1), range(L2 + L2p + 1), range(abs(J - Jp), J + Jp + 1)):
        if sum(q) % 2 or q[2] % 2:
            continue
        Hq = wigner_3j(*q, 0, 0, 0)
        if abs(Hq) < 1e-12:
            continue
        coeff = (2 * L1 + 1) * (2 * L1p + 1) * (2 * L2 + 1) * (2 * L2p + 1)
        coeff *= wigner_3j(q[0], L1, L1p, 0, 0, 0) * wigner_3j(q[1], L2, L2p, 0, 0, 0) / Hq
        coeff *= (2 * J + 1) * (2 * Jp + 1) * HJ * HJp
        coeff *= wigner_9j(L1, L2, J, L1p, L2p, Jp, *q)
        coeff *= wigner_3j(J, Jp, q[2], 0, 0, 0)
        if abs(coeff) < 1e-10:
            continue
        coeff *= (-1) ** (((L1p - L1 + L2p - L2) // 2) % 2)
        coeffs.append((tuple(q), coeff))
    return coeffs


@functools.lru_cache(maxsize=None)
def get_scoccimarro_window_convolution_coeffs(ell, ellin, ellmax=4):
    r"""
    Coefficients for the smooth window convolution in the Scoccimarro basis,
    routing through the TripoSH (Sugiyama) basis where the radial transform is
    a diagonal 2D Hankel transform. Estimator side (thin :math:`k`-shells):

    .. math::

        \tilde{B}_L(k_1, k_2, k_3) = \sum_{\ell_1 \ell_2} \mathcal{L}_{\ell_2}(\cos\theta_{12})\,
        \tilde{B}_{\ell_1 \ell_2 L}(k_1, k_2),
        \quad \cos\theta_{12} = \frac{k_3^2 - k_1^2 - k_2^2}{2 k_1 k_2},

    with :math:`\tilde{B}_{\ell_1\ell_2L}` the TripoSH window convolution
    (:func:`get_sugiyama_window_convolution_coeffs`) of the theory TripoSH multipoles,
    themselves projected from the Scoccimarro-basis theory (eq. 25 of arXiv:1803.02132):

    .. math::

        B_{\ell_1'\ell_2'L'}(k_1', k_2') = \frac{N_{\ell_1'\ell_2'L'} H_{\ell_1'\ell_2'L'}}{\sqrt{4\pi(2L'+1)}}
        \begin{pmatrix} \ell_1' & \ell_2' & L' \\ 0 & -M' & M' \end{pmatrix}
        \int \frac{k_3'^2 dk_3'}{2\pi^2}\, I_{000}(k_1', k_2', k_3')\,
        y_{\ell_2'}^{-M'}(\cos\theta_{12}', 0)\, B_{L'M'}(k_1', k_2', k_3'),

    with :math:`I_{000} = \pi^2 \Theta(\cos\theta_{12}') / (k_1' k_2' k_3')` carrying the
    triangle condition. The 3j triangle conditions bound :math:`|\ell_1 - \ell_2| \leq L`
    (and enforce :math:`\ell_1 + \ell_2 + L` even) on both sides; the remaining sums resolve
    the internal (opening-angle) dependence and are truncated at ``ellmax``: convergence
    with ``ellmax`` should be checked, especially for squeezed triangles.

    Parameters
    ----------
    ell : int
        Output (estimator) multipole :math:`L`.
    ellin : int, tuple
        Theory multipole :math:`L'`, or :math:`(L', M')`. If :math:`M'` is not provided,
        the theory is assumed to be given as Legendre multipoles,
        :math:`B(k_1, k_2, k_3, \mu, \phi) = \sum_{L'} B_{L'}(k_1, k_2, k_3) \mathcal{L}_{L'}(\mu)`,
        i.e. :math:`B_{L'0} = \sqrt{4\pi/(2L'+1)}\, B_{L'}` (and :math:`M' \neq 0` components dropped).
        If :math:`M'` is provided, the theory is the coefficient of the (reduced, real)
        spherical harmonic :math:`y_{L'M'}` in the triangle frame; note the real-harmonic
        :math:`\sqrt{2}` convention for :math:`M' \neq 0` has not been validated.
    ellmax : int, default=4
        Truncation of the TripoSH multipole sums (internal-angle resolution).
        In the uniform-window (box) limit the chain is exact (checked to machine precision)
        once ``ellmax >= ell + max(L, L')``, with ``ell`` the Legendre content of the theory's
        internal-angle (:math:`\cos\theta_{12}'`) dependence; window convolution and sharp
        features (squeezed triangles) increase the required ``ellmax``.

    Returns
    -------
    coeffs : list of (sugiyama_ell, sugiyama_ellt, [(ellw, coeff), ...])
        ``sugiyama_ell = (ell_1, ell_2, L)`` is the estimator-side TripoSH multipole
        (to be resummed with weight :math:`\mathcal{L}_{\ell_2}(\cos\theta_{12})`),
        ``sugiyama_ellt = (ell_1', ell_2', L')`` the theory-side one (to be projected
        with weight :math:`I_{000}\, y_{\ell_2'}^{-M'}`); the projection prefactor
        :math:`N' H' (3j) / \sqrt{4\pi(2L'+1)}` is folded into the window coefficients ``coeff``.
    """
    coeffs = []
    min = None
    if isinstance(ellin, tuple):
        ellin, min = ellin
    for sugiyama_ell in itertools.product(range(ellmax + 1), range(ellmax + 1)):
        sugiyama_ell = sugiyama_ell + (ell,)
        # 3j triangle condition: |ell_1 - ell_2| <= L and ell_1 + ell_2 + L even
        if abs(wigner_3j(*sugiyama_ell, 0, 0, 0)) < 1e-7: continue
        for sugiyama_ellt in itertools.product(range(ellmax + 1), range(ellmax + 1)):
            sugiyama_ellt = sugiyama_ellt + (ellin,)
            H = wigner_3j(*sugiyama_ellt, 0, 0, 0)
            if abs(H) < 1e-7: continue
            sugiyama_coeffs = get_sugiyama_window_convolution_coeffs(sugiyama_ell, sugiyama_ellt)
            if not sugiyama_coeffs: continue
            # Theory-side projection (Scoccimarro to TripoSH), eq. 25 of arXiv:1803.02132
            scoccimarro_to_sugiyama = prod((2 * ell_ + 1) for ell_ in sugiyama_ellt) * H
            if min is None:  # Legendre-multipole input: B_{L'0} = sqrt(4 pi / (2 L' + 1)) B_{L'}, M' = 0
                scoccimarro_to_sugiyama *= H / (2 * ellin + 1)
            else:
                scoccimarro_to_sugiyama *= wigner_3j(*sugiyama_ellt, 0, -min, min) / np.sqrt(4. * np.pi * (2 * ellin + 1))
            if abs(scoccimarro_to_sugiyama) < 1e-10: continue
            sugiyama_coeffs = [(ellw, scoccimarro_to_sugiyama * coeff) for ellw, coeff in sugiyama_coeffs]
            coeffs += [(sugiyama_ell, sugiyama_ellt, sugiyama_coeffs)]
    return coeffs


def get_smooth3_window_bin_attrs(ells, ellsin=3, fields=None, return_ellsin: bool=False, basis: str='sugiyama', ellmax: int=4):
    """
    Get the window bin attributes for sugiyama basis.

    Parameters
    ----------
    ells : list
        Observed multipole orders.
    ellsin : tuple
        Theory multipole orders.
    fields : tuple, list, optional
        3-tuple or 3-list of field identifiers, e.g. [1, 1, 1] if all 3 fields are the fields,
        [1, 2, 3] if all different. To take advantage of symmetries.
    ellmax : int, default=4
        For the scoccimarro basis: truncation of the TripoSH multipole sums,
        see :func:`get_scoccimarro_window_convolution_coeffs`. Use the same value
        as passed to :func:`compute_smooth3_spectrum_window` (window multipoles
        missing from the measured window are silently treated as zero there).

    Returns
    -------
    dict
    """
    if fields is None:
        fields = [1, 1, 1]
    if 'sugiyama' in basis:
        if isinstance(ellsin, numbers.Number):
            nellsin = ellsin
            ellsin = []
            for ellin in itertools.product(*[range(nellsin + 1) for _ in range(3)]):
                if sum(ellin) % 2: continue
                if ellin[2] % 2: continue
                if fields[1] == fields[0]: ellin = tuple(sorted(ellin[:2])) + ellin[2:]
                if ellin not in ellsin:
                    ellsin.append(ellin)
        ellsin = list(ellsin)
        non_zero_ellsin, ellw = [], []
        for ellin in ellsin:  # ell1 3-tuple, wain wide-angle order
            for ill, ell in enumerate(ells):
                coeffs = get_sugiyama_window_convolution_coeffs(ell, ellin)
                if coeffs and ellin not in non_zero_ellsin:
                    non_zero_ellsin.append(ellin)
                for ellw_, _ in coeffs:
                    if fields[1] == fields[0]: ellw_ = tuple(sorted(ellw_[:2])) + ellw_[2:]
                    if ellw_ not in ellw: ellw.append(ellw_)
    elif 'scoccimarro' in basis:
        if isinstance(ellsin, numbers.Number):
            nellsin = ellsin
            ellsin = list(range(0, 2 * nellsin + 1))
        ellsin = list(ellsin)
        non_zero_ellsin, ellw = [], []
        for ellin in ellsin:  # ellin 1-integer
            for ill, ell in enumerate(ells):
                coeffs = get_scoccimarro_window_convolution_coeffs(ell, ellin, ellmax=ellmax)
                if coeffs and ellin not in non_zero_ellsin:
                    non_zero_ellsin.append(ellin)
                for _, _, ellsw_ in coeffs:
                    for ellw_, _ in ellsw_:
                        if fields[1] == fields[0]: ellw_ = tuple(sorted(ellw_[:2])) + ellw_[2:]
                        if ellw_ not in ellw: ellw.append(ellw_)
    else:
        raise NotImplementedError(f'unknown basis {basis}')
    ellw = sorted(set(ellw))

    ellw = dict(ells=ellw, basis='sugiyama', mask_edges='')
    if return_ellsin:
        return ellw, non_zero_ellsin
    return ellw


# Exact radial kernels and window-expansion helpers for the bispectrum window matrix.


def _gaussian_bessel_product(ell, k, kp, sigma):
    r"""
    Exact radial kernel for a Gaussian window, equal Bessel orders:

    .. math:: F^{(\sigma)}_{\ell\ell}(k, k') = \int_0^\infty r^2 dr\, e^{-r^2/2\sigma^2} j_\ell(kr) j_\ell(k'r)
              = \sqrt{\pi/2}\, \sigma^3 e^{-\sigma^2(k^2+k'^2)/2}\, i_\ell(\sigma^2 k k')

    with :math:`i_\ell` the modified spherical Bessel function. This is the
    building block of the Gaussian-expansion route (``eq:gaussian_radial_kernel``
    of ``jax_window_notes/ms.tex``): expanding the window multipoles on separable
    Gaussian products factorizes the radial double integral of
    ``eq:scoccimarro_window_matrix_explicit`` into a product of two of these, each
    a single elementary function.

    Unlike the 2D-FFTlog chain it replaces, this has NO absolute noise floor: it
    is accurate at any smallness, so the :math:`\ell`-sum can be pushed as far as
    the physics requires. The infinite sum can also be done outright,
    :math:`\sum_\ell (2\ell+1) \mathcal{L}_\ell(\mu) F_{\ell\ell} =
    \sqrt{\pi/2}\sigma^3 e^{-\sigma^2 |\mathbf{k}-\mathbf{k}'|^2/2}`
    (``eq:gaussian_resum``), and the box limit is the :math:`\sigma \to \infty`
    member of the family, tending to :math:`(\pi/2k^2)\delta^D(k-k')`
    independently of :math:`\ell` -- i.e. the identity.

    Evaluated as :math:`\frac{\pi}{2}\sigma^3 z^{-1/2} e^{-\sigma^2(k-k')^2/2}
    \tilde I_{\ell+1/2}(z)`, :math:`z=\sigma^2 k k'`, with
    :math:`\tilde I_\nu = e^{-z}I_\nu` the exponentially scaled Bessel function:
    the growth of :math:`i_\ell` cancels the decay of the prefactor exactly, so
    nothing overflows.

    Note this is a host-side (NumPy/SciPy) routine: JAX has no general
    :math:`I_\nu`, so it is not traceable. The window matrix is built eagerly, so
    that is not a limitation in practice.

    Parameters
    ----------
    ell : int
        Bessel order (same on both factors).
    k, kp : array_like
        Wavenumbers; broadcast against each other.
    sigma : float
        Gaussian width of the window, :math:`w(r) = e^{-r^2/2\sigma^2}`.

    Returns
    -------
    F : np.ndarray
    """
    from scipy import special
    k, kp = np.broadcast_arrays(np.asarray(k, dtype='f8'), np.asarray(kp, dtype='f8'))
    z = sigma**2 * k * kp
    small = z < 1e-12  # i_ell(z) ~ z^ell / (2ell+1)!! : vanishes for ell > 0
    zs = np.where(small, 1., z)
    out = 0.5 * np.pi * sigma**3 * zs**-0.5 * np.exp(-sigma**2 * (k - kp)**2 / 2.) * special.ive(ell + 0.5, zs)
    if np.any(small):
        lim = np.sqrt(np.pi / 2.) * sigma**3 * np.exp(-sigma**2 * (k**2 + kp**2) / 2.) if ell == 0 else 0. * k
        out = np.where(small, lim, out)
    return out


def _bessel_product_compact(ell, ellp, k, kp, moment, nquad=96):
    r"""
    Exact radial kernel for an ARBITRARY radial window, any Bessel orders:

    .. math:: F^{(w)}_{\ell\ell'}(k,k') = \frac{1}{4 i^{\ell+\ell'}}
              \int_{-1}^{1}\!dt \int_{-1}^{1}\!ds\, \mathcal{L}_\ell(t) \mathcal{L}_{\ell'}(s)\, R_w(kt + k's)

    where :math:`R_w(A) = \int_0^\infty r^2 w(r) e^{iAr} dr` (``eq:radial_compact``).
    Obtained by applying :math:`j_\ell(z) = (2i^\ell)^{-1}\int_{-1}^1 \mathcal{L}_\ell(t)e^{izt}dt`
    to both Bessel functions: a compact, smooth 2-D quadrature against a single
    1-D moment of the window, exact under modest Gauss-Legendre.

    This is what the unequal-order case needs (:math:`\ell \neq \ell'` arises as
    soon as the window is anisotropic, since :math:`H_{\ell\ell'\ell''}` then no
    longer forces equality), and it is not specific to Gaussians -- any window
    whose moment one is willing to tabulate works, which makes it a general
    replacement for the 2D-FFTlog chain.

    Parameters
    ----------
    ell, ellp : int
        Bessel orders of the two factors (need not be equal).
    k, kp : array_like
        Wavenumbers; broadcast against each other.
    moment : callable
        ``moment(A)`` returning :math:`R_w(A) = \int_0^\infty r^2 w(r) e^{iAr} dr`,
        broadcasting over its argument. For :math:`\ell + \ell'` even only the
        even (cosine) part contributes, so a real
        :math:`C_w(A) = \int_0^\infty r^2 w(r)\cos(Ar) dr` may be passed instead.
        Elementary for a Gaussian, tophat or exponential window; otherwise
        tabulate it once by 1-D quadrature and interpolate.
    nquad : int, default=96
        Gauss-Legendre nodes per axis of the compact :math:`(t, s)` square. The
        integrand is smooth (a polynomial times the window moment), so this
        converges exponentially; 96 is exact to ~8 digits in the cases checked.

    Returns
    -------
    F : np.ndarray
        Real array, broadcast over ``k``, ``kp``.
    """
    from scipy import special
    k, kp = np.broadcast_arrays(np.asarray(k, dtype='f8'), np.asarray(kp, dtype='f8'))
    t, wq = np.polynomial.legendre.leggauss(nquad)
    Pl = special.eval_legendre(ell, t) * wq
    Plp = special.eval_legendre(ellp, t) * wq
    # Accumulate over the t axis rather than materializing the full
    # (k.shape, nquad, nquad) array of arguments: for a k-grid of size n that
    # would be n * nquad^2 entries (~700 MB at n = 1e4, nquad = 96), whereas
    # this holds only (k.shape, nquad) at a time.
    out = np.zeros(k.shape, dtype='c16')
    for i in range(nquad):
        A = k[..., None] * t[i] + kp[..., None] * t
        out += Pl[i] * np.sum(np.asarray(moment(A)) * Plp, axis=-1)
    out = out / (4. * (1j)**(ell + ellp))
    # F is real for a real window: the phase and the (t, s) parity conspire to
    # cancel the imaginary part exactly (kept explicit rather than assumed).
    return out.real



def _project_window_gaussian(s1, s2, value, ell1=0, ell2=0, widths=None, nwidth=8, rcond=1e-10):
    r"""
    Project a window multipole :math:`Q_{\ell_1\ell_2L}(s_1, s_2)` onto SEPARABLE
    Gaussian-polynomial products,

    .. math:: Q_{\ell_1\ell_2 L}(s_1, s_2) \simeq \sum_{nm} c_{nm}\;
              s_1^{\ell_1} e^{-a_n s_1^2}\; s_2^{\ell_2} e^{-a_m s_2^2}

    (``eq:gaussian_mixture`` of ``jax_window_notes/ms.tex``). The point of this basis
    is that every radial integral of the window matrix then reduces to
    :func:`_gaussian_bessel_product` (equal orders) or :func:`_bessel_product_compact`
    (unequal), i.e. to elementary functions with no FFTlog grid, no ringing and no
    absolute noise floor -- and the box limit is the :math:`a \to 0` member of the
    same family, so it needs no separate treatment.

    The widths :math:`a_n` are FIXED on a logarithmic grid rather than fitted, which
    makes the problem a linear least squares in :math:`c_{nm}` -- well conditioned and
    solved in one shot, instead of the ill-conditioned nonlinear fit a variable-width
    (or shifted) Gaussian basis would require. The :math:`s^{\ell}` prefactors carry
    the small-separation behaviour of the multipole exactly.

    Parameters
    ----------
    s1, s2 : array_like
        Separation grids (1-D each).
    value : array_like
        ``(len(s1), len(s2))`` window multipole.
    ell1, ell2 : int
        Multipole orders on each leg; set the :math:`s^{\ell}` prefactors.
    widths : array_like, optional
        Gaussian widths :math:`\sigma_n` (so :math:`a_n = 1/2\sigma_n^2`). Default:
        ``nwidth`` values log-spaced across the span of the input grid.
    nwidth : int, default=8
        Number of widths per axis when ``widths`` is not given.
    rcond : float, default=1e-10
        Cutoff for small singular values in the least-squares solve.

    Returns
    -------
    coeffs : np.ndarray
        ``(nwidth, nwidth)`` coefficients :math:`c_{nm}`.
    widths : np.ndarray
        The widths used.
    residual : float
        Relative residual ``||fit - value|| / ||value||``; check this before trusting
        the expansion.
    """
    s1, s2, value = np.asarray(s1, dtype='f8'), np.asarray(s2, dtype='f8'), np.asarray(value, dtype='f8')
    if widths is None:
        smin = max(np.min(s1[s1 > 0]) if np.any(s1 > 0) else 1., 1e-3)
        widths = np.geomspace(smin, np.max(s1), nwidth)
    widths = np.asarray(widths, dtype='f8')
    a = 1. / (2. * widths**2)
    # separable design matrices per axis, then a Kronecker least squares
    B1 = s1[:, None]**ell1 * np.exp(-a[None, :] * s1[:, None]**2)     # (ns1, nw)
    B2 = s2[:, None]**ell2 * np.exp(-a[None, :] * s2[:, None]**2)     # (ns2, nw)
    A = np.einsum('in,jm->ijnm', B1, B2).reshape(s1.size * s2.size, widths.size**2)
    c, *_ = np.linalg.lstsq(A, value.reshape(-1), rcond=rcond)
    fit = (A @ c).reshape(value.shape)
    nrm = np.linalg.norm(value)
    return c.reshape(widths.size, widths.size), widths, float(np.linalg.norm(fit - value) / max(nrm, 1e-300))


def _project_window_laguerre(s1, s2, value, ell1=0, ell2=0, sigma=None, nmax=8, rcond=1e-12):
    r"""
    Project a window multipole onto the ORTHOGONAL Gauss--Laguerre basis

    .. math:: \phi^{(\ell)}_n(s) = s^{\ell}\, L_n(s^2/2\sigma^2)\, e^{-s^2/2\sigma^2}

    (``eq:gauss_laguerre_basis`` of ``jax_window_notes/ms.tex``), i.e.
    :math:`Q_{\ell_1\ell_2L}(s_1,s_2) \simeq \sum_{nm} c_{nm}\phi^{(\ell_1)}_n(s_1)\phi^{(\ell_2)}_m(s_2)`.

    Preferred over :func:`_project_window_gaussian`: a mixture of Gaussians of
    *different widths* is strongly non-orthogonal, so its design matrix is
    ill-conditioned and the least squares is unstable -- measured, that basis leaves
    a 3e-2 residual on a function it represents EXACTLY, with a residual that is not
    even monotonic in the number of widths. The Laguerre functions share a single
    width and are orthogonal, so the fit is stable and converges with ``nmax``.

    The closed forms survive: :math:`L_n(s^2/2\sigma^2)e^{-s^2/2\sigma^2}` is a finite
    combination of :math:`s^{2p}e^{-as^2}`, and :math:`s^{2p}e^{-as^2} =
    (-\partial/\partial a)^p e^{-as^2}`, so every radial integral is an
    :math:`a`-derivative of :func:`_gaussian_bessel_product`.

    Returns
    -------
    coeffs : np.ndarray
        ``(nmax, nmax)``.
    sigma : float
    residual : float
        Relative residual; should fall monotonically with ``nmax``.
    """
    from numpy.polynomial import laguerre
    s1, s2, value = np.asarray(s1, dtype='f8'), np.asarray(s2, dtype='f8'), np.asarray(value, dtype='f8')
    if sigma is None:
        # Width matched to the input's own SECOND MOMENT, not to the grid extent.
        # Getting this wrong is the dominant error: with sigma tied to max(s), a
        # narrow window needs many high-order terms and the fit stalls (measured
        # 2e-2 at nmax=12 for a sigma=50 input on a grid reaching 500, versus 1e-8
        # for a sigma=150 input on the same grid).
        w = np.abs(value).sum(axis=1)
        sigma = np.sqrt(max((w * s1**2).sum() / max(w.sum(), 1e-300), 1e-300)) / np.sqrt(2.)

    def design(s, ell):
        u = s**2 / (2. * sigma**2)
        cols = []
        for n in range(nmax):
            cn = np.zeros(n + 1); cn[n] = 1.
            cols.append(s**ell * laguerre.lagval(u, cn) * np.exp(-u / 2.))
        return np.column_stack(cols)

    B1, B2 = design(s1, ell1), design(s2, ell2)
    A = np.einsum('in,jm->ijnm', B1, B2).reshape(s1.size * s2.size, nmax * nmax)
    c, *_ = np.linalg.lstsq(A, value.reshape(-1), rcond=rcond)
    fit = (A @ c).reshape(value.shape)
    return c.reshape(nmax, nmax), float(sigma), float(np.linalg.norm(fit - value) / max(np.linalg.norm(value), 1e-300))


def compute_smooth3_spectrum_window(window, edgesin: np.ndarray | tuple, ellsin: tuple=None, bin: BinMesh3SpectrumPoles=None,
                                    flags: tuple=None, batch_size: int=None, ellmax: int=4,
                                    ninsub: int=1, noutsub: int=1, interp: str='tophat',
                                    exact_box_limit: bool | float=False, permute_in: bool=True,
                                    permute_lgt0: str='slot3') -> WindowMatrix:
    """
    Compute the "smooth" (no binning effect) bispectrum window matrix.

    Parameters
    ----------
    window : ObservableTree
        Configuration-space window function, in the sugiyama basis.
    edgesin : np.ndarray | tuple
        Input bin edges.
    ellsin : tuple, optional
        Input multipole orders. Optional when ``edgesin`` is provided.
        For the scoccimarro basis, integers :math:`L'` (theory given as Legendre multipoles)
        or tuples :math:`(L', M')`, see :func:`get_scoccimarro_window_convolution_coeffs`.
    bin : BinMesh2SpectrumPoles
        Output binning.
    ellmax : int, default=4
        For the scoccimarro basis: truncation of the TripoSH multipole sums resolving
        the internal (opening-angle) dependence, see
        :func:`get_scoccimarro_window_convolution_coeffs`. Use the same value as passed to
        :func:`get_smooth3_window_bin_attrs` (window multipoles missing from ``window``
        are silently treated as zero). Convergence should be checked, especially for
        squeezed configurations.
    batch_size : int, optional
        Size of the batch for each step to execute in parallel.
    permute_in : bool, default=True
        For the scoccimarro basis: sum each theory bin's contribution over all distinct
        permutations of its :math:`(k_1', k_2', k_3')` box.

        The estimator bins ORDERED triangles :math:`k_1' \\le k_2' \\le k_3'`, so ``edgesin``
        typically covers only the ordered octant -- but the convolution integral runs over
        all of :math:`k'`-space. Summing the ordered octant alone drops every configuration
        that leaves it under window smearing, which breaks the box-limit sum rule
        :math:`\\sum_j W_{ij} = Q_\\infty` by an amount set by triangle SHAPE rather than
        scale.

        This is not a multiplicity factor: the kernel is not symmetric under permuting
        :math:`k'` at fixed :math:`k`, so each assignment carries its own kernel and is
        integrated separately. It accumulates into the SORTED bin's column, which is exact
        because :math:`B` is symmetric. Permutations that reproduce a box already present in
        ``edgesin`` are skipped, so a caller passing a full (unordered) product grid is
        unaffected and nothing is double-counted.

        For :math:`L' > 0` the theory substitution is only valid for the permutations selected
        by ``permute_lgt0`` (default ``'slot3'``); see there.

        Set ``False`` to recover the pre-fix behaviour (regression comparisons only).
    permute_lgt0 : str, default='slot3'
        Which permutations to use for the :math:`L' > 0` theory blocks (the :math:`L' = 0`
        block always uses all of them, where the substitution is exact).

        The estimator applies the output Legendre to the THIRD leg (``meshes[2]``) and the
        binning mask orders the bins, so :math:`B_{L'}` at an ordered bin is the multipole
        referred to the leg in slot 3, i.e. the LONGEST leg. A permuted box puts a different
        bin in slot 3, and the kernel then refers the line-of-sight Legendre to that leg --
        correctly, cf. Philcox 2021 Eq. (55), where the Legendre sits inside the permutation
        sum -- but the value supplied is still the long-leg multipole. For :math:`L' = 0` there
        is no line-of-sight dependence and the substitution is exact; for :math:`L' > 0` it
        silently swaps one quantity for another.

        - ``'full'``: all distinct permutations (exact only for :math:`L' = 0`).
        - ``'slot3'``: only permutations that leave slot 3 holding a leg in the SAME bin as
          the ordered triple's third, so the theory's leg reference always matches the
          kernel's. Exact for every :math:`L'`, but recovers less of :math:`k'`-space
          (the 1<->2 swaps only).
        - ``'none'``: identity only for :math:`L' > 0`, i.e. pre-fix behaviour there.

        Re-referring a quadrupole to a different leg mixes :math:`L` and involves the second
        angular coordinate that this basis drops, so the ``'full'`` permuted :math:`L' > 0`
        terms cannot be reconstructed exactly from :math:`L' = 0, 2` about one leg alone.

    Returns
    -------
    wmat : WindowMatrix
    """
    ells = bin.ells
    # the constant-window pass below must receive the CALLER's arguments, not the
    # rebound/parsed ones: ellsin is turned into (ellin, wain) pairs further down
    _edgesin_arg, _ellsin_arg = edgesin, ellsin

    if isinstance(edgesin, ObservableTree):
        ellsin = edgesin.ells
        if 'wa_orders' in edgesin.labels(return_type='keys'):
            ellsin = [(ell, wa) for ell, wa in zip(edgesin.ells, edgesin.wa_orders)]
        pole = next(iter(edgesin))
        # kin is used by the scoccimarro branch below (the compute_I weight) but was
        # only ever assigned in the raw-edges branch, so passing an ObservableTree --
        # the natural way to feed a MEASURED spectrum as theory -- raised NameError.
        # Read it from the pole itself, i.e. whatever representative k the theory
        # values are tabulated at (typically the mode-weighted bin.xavg), so the
        # angle/thin-shell weight is evaluated at the same k the values belong to.
        kin = pole.coords('k')
        edgesin = pole.edges('k')

    else:
        if not isinstance(edgesin, (list, tuple)):
            edgesin = (edgesin,)
        edgesin = tuple(edgesin)
        edgesin += (edgesin[-1],) * (bin.xavg.shape[-1] - len(edgesin))
        grid_edgesin = []
        for edge in edgesin:
            if edge.ndim == 1: edge = jnp.column_stack([edge[:-1], edge[1:]])
            grid_edgesin.append(edge)
        grid_kin = tuple(jnp.mean(edge, axis=-1) for edge in grid_edgesin)

        def _cproduct(arrays, swap=False):
            if swap: arrays = arrays[::-1]
            grid = jnp.meshgrid(*arrays, sparse=False, indexing='ij')
            return jnp.column_stack([tmp.ravel() for tmp in grid])

        # of shape (nbins, ndim, 2)
        def _get_edgesin(grid_edgesin, swap=False):
            return jnp.concatenate([_cproduct([edge[..., 0] for edge in grid_edgesin], swap=swap)[..., None],
                                    _cproduct([edge[..., 1] for edge in grid_edgesin], swap=swap)[..., None]], axis=-1)
        edgesin, edgesin_swap = (_get_edgesin(grid_edgesin, swap=swap) for swap in [False, True])
        kin = _cproduct(grid_kin)
        if 'scoccimarro' in bin.basis:
            # Bin-level triangle-overlap test: keep every bin whose (k1, k2, k3)
            # theory intersects the triangle region, not only those whose single
            # representative (midpoint) triangle does. The midpoint test
            # silently drops bins that partially overlap the allowed region,
            # breaking the box-limit sum rule sum_j W_ij = 1.
            k1lo, k1hi = edgesin[:, 0, 0], edgesin[:, 0, 1]
            k2lo, k2hi = edgesin[:, 1, 0], edgesin[:, 1, 1]
            k3lo, k3hi = edgesin[:, 2, 0], edgesin[:, 2, 1]
            gap12 = jnp.maximum(jnp.maximum(k1lo - k2hi, k2lo - k1hi), 0.)
            mask = (k3hi >= gap12) & (k3lo <= k1hi + k2hi)
            edgesin, kin = edgesin[mask], kin[mask]

    if 'sugiyama' in bin.basis:
        ellsin = [(ellin[0], tuple(ellin[1])) if isinstance(ellin[0], tuple) else (ellin, (0, 0)) for ellin in ellsin]
    else:
        def _parse(ellin):  # L', (L', M'), or ((L') or (L', M'), wa_orders)
            if isinstance(ellin, tuple) and isinstance(ellin[1], (tuple, list)):
                return (ellin[0], tuple(ellin[1]))
            return (ellin, (0, 0))
        ellsin = [_parse(ellin) for ellin in ellsin]

    kout = bin.xavg

    from .fftlog import SpectrumToCorrelation, CorrelationToSpectrum
    from .cov2 import matrix_spline_interp, matrix_rebin

    def get_w_rect(q, wain):
        transpose = False
        if q not in window.ells:
            q = q[1::-1] + q[2:]
            transpose = True
        kw = dict(ells=q)
        if 'wa_orders' in window.labels(return_type='keys'):
            kw.update(wa_orders=wain)
        elif wain != (0, 0):
            raise ValueError('wa_orders must be provided in input window')
        if kw in window.labels(return_type='flatten'):
            value = window.get(**kw).value().real
            if transpose:
                value = jnp.swapaxes(value, 0, 1)
            return value
        return jnp.zeros(())

    def tophat(k, edgein, value):
        # k tuple defining the k-grid, edgein tuple of (min, max), value flattened values
        masks = []
        for kk, edge in zip(k, edgein):
            masks.append((kk >= edge[0]) & (kk < edge[1]))
        return prod(jnp.meshgrid(*masks, indexing='ij', sparse=True)) * value

    def axis_basis_matrices(edges, k_axes, kind):
        """Per-axis (separable) replacement for the sharp tophat/read: build,
        for each axis independently, a SMALL matrix of shape (n_k_axis,
        n_unique_axis) ('spline', input/theory side: a smooth spline basis
        function per distinct bin center, matrix_spline_interp) or
        (n_unique_axis, n_k_axis) ('rebin', output side: a proper k^2-weighted
        bin average, matrix_rebin), keyed by the axis's DISTINCT bin
        centers/edges -- never the full (nbins1 x nbins2, n_k1 x n_k2) dense
        tensor (edges may be a masked/paired list, e.g. sugiyama-diagonal's
        k1=k2, not a full product grid; this stays correct and small either
        way). Returns (index_per_bin (nbins, ndim), matrices list).
        """
        edges_np = np.asarray(edges)
        centers = edges_np.mean(axis=-1)
        ndim = centers.shape[-1]
        index_per_bin = np.empty(centers.shape, dtype=np.int64)
        matrices = []
        for d in range(ndim):
            u, first_idx, inv = np.unique(centers[:, d], return_index=True, return_inverse=True)
            index_per_bin[:, d] = inv
            kk = np.asarray(k_axes[d])
            unique_edges = edges_np[first_idx, d, :]
            if kind == 'spline':
                M = matrix_spline_interp(u, kk, interp_order=3)
                # matrix_spline_interp extrapolates via the spline's own boundary
                # polynomial outside [u.min(), u.max()], and kk (the fftlog k-grid) spans a
                # far wider range than the input bins, where cubic extrapolation explodes --
                # so it must be cut off. Cut at the bin EDGES, not the bin centres: the
                # centres lose half a bin at each end (11% of the edge-to-edge range for a
                # typical binning), and the box-limit sum rule needs the theory basis to tile
                # the whole range.
                lo, hi = unique_edges.min(), unique_edges.max()
                in_range = (kk >= lo) & (kk <= hi)
                M = M * in_range[:, None]
            else:
                M = matrix_rebin(unique_edges, kk, wt=kk**2, interp_order=3)
            matrices.append(M)
        return jnp.asarray(index_per_bin), matrices

    def read(kout, k, value):
        # kout tuple of flattened output k's, k tuple defining the k-grid, value is grid
        id0, s = [], []
        mask = 1.
        for kk, kkout in zip(k, kout):
            id0_ = jnp.searchsorted(kk, kkout, side='right') - 1
            mask = mask * ((id0_ >= 0) & (id0_ < len(kk) - 1.))
            id0_ = jnp.clip(id0_, 0, len(kk) - 2)
            id0.append(id0_)
            s.append((kkout - kk[id0_]) / (kk[id0_ + 1] - kk[id0_]))
        id0, s = jnp.column_stack(id0), jnp.column_stack(s)
        ishifts = np.array(list(itertools.product(* len(k) * (np.arange(2),))), dtype=('i4'))

        def step(carry, ishift):
            idx = id0 + ishift
            ker = jnp.prod((ishift == 0) * (1 - s) + (ishift == 1) * s, axis=-1)
            idx = jnp.unstack(idx, axis=-1)
            carry += value[idx] * ker * mask
            return carry, None

        toret = jnp.zeros_like(value, shape=kout[0].size)
        return jax.lax.scan(step, toret, ishifts)[0]

    wmat_tmp = {}

    if 'scoccimarro' in bin.basis:

        def compute_I(ell, qs):
            cos = (qs[2]**2 - qs[1]**2 - qs[0]**2) / (2 * qs[0] * qs[1])
            tophat = (jnp.abs(cos) < 1.) + 1. / 2. * (jnp.abs(cos) == 1.)
            # Bin centers can fall (slightly) outside the triangle inequality
            # for coarse binning even though the bin itself has partial valid
            # overlap; tophat already zeroes the contribution there, but an
            # unclipped cos blows up sqrt(1 - cos**2) into NaN, and NaN * 0 =
            # NaN propagates through the whole matrix. Clip only for the
            # Ylm/Legendre evaluation, not for the tophat mask itself.
            cos_safe = jnp.clip(cos, -1., 1.)
            m = None
            if isinstance(ell, tuple): ell, m = ell
            toret = (-1)**ell * np.pi**2 / prod(qs) * tophat
            if m is None: toret *= get_legendre(ell)(cos_safe)
            else: toret *= get_Ylm(ell, m, reduced=True, real=True)(jnp.sqrt(1. - cos_safe**2), 0., cos_safe)
            return toret

        if ninsub > 1:
            from .pt import integration
            _integ_in = integration(-1., 1., size=ninsub)
            _u_in, _wu_in = jnp.asarray(_integ_in.x()), jnp.asarray(_integ_in.w)

        # OUTPUT-side bin averaging (ms.tex caveat (ii) on eq:scoccimarro_window_matrix_explicit:
        # "average the (k1, k2, k3) dependence over the bin"). The theory side does this via
        # ninsub; the output side carries its own rapidly varying, discontinuous factor
        # L_{ell2}(cos theta12) = (-1)^ell2 I_{ell2 ell2 0} / I_000, and evaluating it at the
        # single representative triangle bin.xavg is NOT the bin average. Since
        # d cos(theta12) / d k3 = k3 / (k1 k2), a SHORT leg makes cos(theta12) sweep the bin:
        # measured, L_2 at the midpoint is +20% off its bin average for (0.033, 0.071, 0.071)
        # and +40% for (0.033, 0.033, 0.033), but only ~1% for (0.110, 0.071, 0.071). That
        # shape dependence -- large for squeezed ISOSCELES, small for squeezed scalene or for
        # equal-but-short legs -- is exactly the pattern of the box-limit residual.
        # The bin average is the ratio of bin-integrated measures, all three legs sub-binned.
        # interp='spline': the SAME primitives the sugiyama branch uses -- a smooth spline basis on
        # the theory side (matrix_spline_interp) in place of the sharp `tophat`, and a k^2-weighted
        # bin average on the output side (matrix_rebin) in place of `read`, which interpolates
        # linearly at the bin's representative (k1, k2) with no k^2 weight and no bin average --
        # a plausible source of a low-k/high-k tilt.
        # interp names the (theory-side, output-side) pair so the two can be isolated:
        #   'tophat'        = (tophat mask, linear read)      -- the historical path
        #   'spline'        = (spline basis, k^2 rebin)        -- both swapped, as the sugiyama branch
        #   'tophat-rebin'  = (tophat mask, k^2 rebin)         -- keeps the exact bin tiling on the
        #                     theory side (which the box-limit sum rule needs) while gaining a
        #                     proper k^2-weighted bin average on the output
        #   'spline-read'   = (spline basis, linear read)
        _INTERP = {'tophat': ('tophat', 'read'), 'spline': ('spline', 'rebin'),
                   'tophat-rebin': ('tophat', 'rebin'), 'spline-read': ('spline', 'read')}
        if interp not in _INTERP:
            raise ValueError(f"interp must be one of {sorted(_INTERP)}, got {interp}")
        _interp_in, _interp_out = _INTERP[interp]
        _sc_spline = None
        if interp != 'tophat':
            assert 'scoccimarro' in bin.basis, f'interp={interp} implemented for the scoccimarro branch'

        if noutsub > 1:
            from .pt import integration as _integration_out
            _integ_out = _integration_out(-1., 1., size=noutsub)
            _u_out, _wu_out = np.asarray(_integ_out.x()), np.asarray(_integ_out.w)
            _oe = np.asarray(bin.edges)                                     # (nout, 3, 2)
            _lo, _hi = _oe[..., 0], _oe[..., 1]                             # (nout, 3)
            _mid = 0.5 * (_hi - _lo)[..., None] * (_u_out[None, None] + 1.) + _lo[..., None]
            _jac = 0.5 * (_hi - _lo)[..., None] * _wu_out[None, None]
            _g = np.stack(np.meshgrid(*[np.arange(noutsub)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
            _ksub = np.stack([_mid[:, j, _g[:, j]] for j in range(3)], axis=-1)   # (nout, nsub, 3)
            _wsub = np.prod(np.stack([_jac[:, j, _g[:, j]] for j in range(3)], axis=-1), axis=-1)
            # measure weight ~ k1^2 k2^2 k3^2 (the Theta triangle factor comes from compute_I)
            _wsub = _wsub * np.prod(_ksub**2, axis=-1)                      # (nout, nsub)
            nout_sub = _g.shape[0]
            kout_sub = jnp.asarray(_ksub.reshape(-1, 3))
            kout_weight = jnp.asarray(_wsub)

        # THEORY-SIDE k'-SPACE ENUMERATION (see permute_in in the docstring).
        # W_ij = sum over every k'-space box whose SORTED bin is j, of the kernel integrated
        # over that box. edgesin covers only the ordered octant, so the rows below expand
        # each bin into its distinct permutations, all accumulating back into column j.
        _edges_np, _kin_np = np.asarray(edgesin), np.asarray(kin)
        _bkey = lambda e: tuple(np.round(np.asarray(e).ravel(), 12))
        _existing = {_bkey(_edges_np[j]): j for j in range(len(_edges_np))}
        if permute_lgt0 not in ('full', 'slot3', 'none'):
            raise ValueError(f"permute_lgt0 must be 'full', 'slot3' or 'none', got {permute_lgt0}")

        def _make_rows(mode):
            rows_e, rows_k, rows_col = [], [], []
            for j in range(len(_edges_np)):
                _seen = set()
                _perms = itertools.permutations(range(3)) if (permute_in and mode != 'none') else [(0, 1, 2)]
                for _p in _perms:
                    # 'slot3': keep the theory's line-of-sight leg reference. Compare BINS, not
                    # positions, so a degenerate third leg (b == c) still admits its swap.
                    if mode == 'slot3' and _bkey(_edges_np[j][_p[2]]) != _bkey(_edges_np[j][2]):
                        continue
                    _pl = list(_p)
                    _e = _edges_np[j][_pl]
                    _kk = _bkey(_e)
                    # degenerate legs: this permutation reproduces a box already taken for bin j
                    if _kk in _seen: continue
                    _seen.add(_kk)
                    # that box is another bin's own column (caller passed a full product grid)
                    if _existing.get(_kk, j) != j: continue
                    rows_e.append(_e); rows_k.append(_kin_np[j][_pl]); rows_col.append(j)
            return (jnp.asarray(np.array(rows_e)), jnp.asarray(np.array(rows_k)),
                    jnp.asarray(np.array(rows_col)), len(rows_col))

        _nbins = len(_edges_np)
        _ROWS = {'full': _make_rows('full')}
        if permute_lgt0 != 'full': _ROWS[permute_lgt0] = _make_rows(permute_lgt0)
        _SPL = {}
        import logging
        for _m, _r in _ROWS.items():
            if _r[3] != _nbins:
                logging.getLogger('Mesh3').info(
                    f'theory-side k\' enumeration [{_m}]: {_nbins} bins -> {_r[3]} permuted boxes '
                    f'(x{_r[3] / _nbins:.2f}); set permute_in=False for the pre-fix behaviour')

        for ellin, wain in ellsin:  # ellin = L' or (L', M'), wain wide-angle order
            wmat_tmp[ellin, wain] = []
            m_in = ellin[1] if isinstance(ellin, tuple) else 0  # M'
            # L' = 0 always takes every permutation (the theory substitution is exact there);
            # L' > 0 follows permute_lgt0, since B_{L'} is referred to the ordered triple's
            # third leg and a permuted box may put a different leg in slot 3.
            _L_in = ellin[0] if isinstance(ellin, tuple) else ellin
            _mode = 'full' if _L_in == 0 else permute_lgt0
            _perm_edges, _perm_kin, _perm_col, _nrows = _ROWS[_mode]
            for ill, ell in enumerate(ells):  # ell = L

                # Then sum over \ell_1, \ell_2, \ell_1', \ell_2', \ell_1'', \ell_2'', L''
                tmp = jnp.zeros(shape=(len(kout), len(edgesin)))

                for sugiyama_ell, sugiyama_ellt, wcoeffs in get_scoccimarro_window_convolution_coeffs(ell, ellin, ellmax=ellmax):
                    # fftlog
                    to_spectrum = CorrelationToSpectrum(s=tuple(next(iter(window)).coords().values()), ell=sugiyama_ell, check_level=1, minfolds=0)
                    to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=sugiyama_ellt, minfolds=0)
                    Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)

                    def convolve(idx):
                        # Theory side: \int_{bin} k_3'^2 dk_3' / (2 pi^2) x I_000(k') y_{ell_2'}^{-M'}(cos theta_12', 0);
                        # compute_I((ell, m)) = (-1)^ell (pi^2 / (k_1' k_2' k_3')) Theta y_ell^m, hence the (-1)^ell_2' compensation
                        ell2t = sugiyama_ellt[1]
                        if ninsub > 1:
                            # Integrate the measure factor over the bin's k3 extent,
                            # per (k1', k2') grid point, instead of evaluating it at the
                            # single representative triangle kin[idx]: I ~ (k1 k2 k3)^-1
                            # Theta y_l is rapidly varying AND discontinuous, so the
                            # point value is not the bin average (this breaks the
                            # box-limit sum rule by tens of per cent otherwise).
                            lo3, hi3 = _perm_edges[idx, 2, 0], _perm_edges[idx, 2, 1]
                            k3n = lo3 + 0.5 * (hi3 - lo3) * (_u_in + 1.)
                            w3n = 0.5 * (hi3 - lo3) * _wu_in
                            qs = (to_spectrum.k[0][:, None, None], to_spectrum.k[1][None, :, None], k3n[None, None, :])
                            volume = (-1)**ell2t * jnp.sum(w3n * qs[2]**2 / (2. * jnp.pi**2) * compute_I((ell2t, -m_in), qs), axis=-1)
                        else:
                            volume = (_perm_edges[idx, 2, 1]**3 - _perm_edges[idx, 2, 0]**3) / (6. * jnp.pi**2) * (-1)**ell2t * compute_I((ell2t, -m_in), _perm_kin[idx].T)
                        if _interp_in == 'spline':
                            _Min, _ii = _sc_spline[0], _sc_spline[1][idx]
                            spectrum = (_Min[0][:, _ii[0]][:, None] * _Min[1][:, _ii[1]][None, :]) * volume
                        else:
                            spectrum = tophat(to_spectrum.k, _perm_edges[idx, :2], volume)
                        correlation = to_correlation(spectrum)[1]
                        correlation = correlation * Qs * to_correlation.s[0][:, None]**wain[0] * to_correlation.s[1][None, :]**wain[1]
                        # Estimator side: B_L(k1, k2, k3) = sum_{ell_1 ell_2} Legendre_{ell_2}(cos theta_12) B_{ell_1 ell_2 L}(k1, k2)
                        ell2 = sugiyama_ell[1]
                        if _interp_out == 'rebin':
                            _Mo, _io = _sc_spline[2], _sc_spline[3]
                            _rb = _Mo[0] @ to_spectrum(correlation)[1] @ _Mo[1].T
                            spectrum = _rb[_io[:, 0], _io[:, 1]]
                            spectrum = spectrum * compute_I(ell2, kout.T) * (-1)**ell2 / compute_I(0, kout.T)
                            return jnp.nan_to_num(spectrum)
                        if noutsub > 1:
                            # bin average = sum_sub w k^2 I_{ell2} read  /  sum_sub w k^2 I_0
                            sub = read(kout_sub.T, to_spectrum.k, to_spectrum(correlation)[1])
                            num = jnp.sum(kout_weight * (sub * compute_I(ell2, kout_sub.T) * (-1)**ell2).reshape(-1, nout_sub), axis=-1)
                            den = jnp.sum(kout_weight * compute_I(0, kout_sub.T).reshape(-1, nout_sub), axis=-1)
                            spectrum = num / jnp.where(den == 0., 1., den)
                        else:
                            spectrum = read(kout.T, to_spectrum.k, to_spectrum(correlation)[1])
                            spectrum *= compute_I(ell2, kout.T) * (-1)**ell2 / compute_I(0, kout.T)
                        # compute_I(0, kout) = pi^2/(k1 k2 k3) * Theta vanishes on output
                        # bins whose representative triangle violates the triangle
                        # inequality, so this division is 0/0 there and leaves NaN in
                        # those rows (measured: 252 of 729 rows, ALL triangle-invalid).
                        # Harmless in itself -- those bins are unphysical -- but a NaN
                        # in the matrix poisons any downstream dot product for a caller
                        # who does not mask, so return 0 as the grid branch does.
                        return jnp.nan_to_num(spectrum)

                    if interp != 'tophat' and _mode not in _SPL:
                        # keyed on the PERMUTED boxes: their (k1', k2') centres are a superset
                        # of the ordered bins', so building this from edgesin would leave the
                        # permuted rows indexing a basis that lacks their own centres. Cached
                        # per permutation mode, since each mode has its own row list.
                        _i_in, _M_in = axis_basis_matrices(_perm_edges[:, :2], to_spectrum.k, kind='spline')
                        _i_out, _M_out = axis_basis_matrices(np.asarray(bin.edges)[:, :2], to_spectrum.k, kind='rebin')
                        _SPL[_mode] = (_M_in, _i_in, _M_out, _i_out)
                    _sc_spline = _SPL.get(_mode)
                    # (nrows, nkout) -> sum permuted boxes back into their sorted bin's column
                    _res = jax.lax.map(convolve, jnp.arange(_nrows), batch_size=batch_size)
                    tmp += jax.ops.segment_sum(_res, _perm_col, num_segments=_nbins).T

                wmat_tmp[ellin, wain].append(tmp)

            wmat_tmp[ellin, wain] = jnp.concatenate(wmat_tmp[ellin, wain], axis=0)
    else:

        for ellin, wain in ellsin:  # ellin 3-tuple, wain wide-angle order
            wmat_tmp[ellin, wain] = []
            for ill, ell in enumerate(ells):

                def convolve(idx, swap=False):
                    idx_axes = (index_in_swap if swap else index_in)[idx]
                    theory = Min_axes[0][:, idx_axes[0]][:, None] * Min_axes[1][:, idx_axes[1]][None, :]
                    correlation = to_correlation(theory)[1]
                    correlation = correlation * Qs * to_correlation.s[0][:, None]**wain[0] * to_correlation.s[1][None, :]**wain[1]
                    # Separable output rebin: Mout_axes[d] is small
                    # (n_unique_axis, n_k_axis), so this never materializes
                    # a (nbins1 x nbins2, n_k1 x n_k2) dense tensor -- only
                    # the same (n_k1, n_k2) grid tophat/read already used.
                    rebinned = Mout_axes[0] @ to_spectrum(correlation)[1] @ Mout_axes[1].T
                    return rebinned[index_out[:, 0], index_out[:, 1]]

                tmp = jnp.zeros(shape=(len(kout), len(edgesin)))
                # fftlog
                to_spectrum = CorrelationToSpectrum(s=tuple(next(iter(window)).coords().values()), ell=ell, check_level=1, minfolds=0)
                index_in, Min_axes = axis_basis_matrices(edgesin, to_spectrum.k, kind='spline')
                index_in_swap = index_in[:, ::-1]
                index_out, Mout_axes = axis_basis_matrices(bin.edges, to_spectrum.k, kind='rebin')

                wcoeffs = get_sugiyama_window_convolution_coeffs(ell, ellin)
                Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)
                if wcoeffs:
                    to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=ellin, minfolds=0)
                    tmp += jax.lax.map(convolve, jnp.arange(edgesin.shape[0]), batch_size=batch_size).T

                ellin_swap = tuple(ellin[1::-1]) + ellin[2:]
                # Takes care of symmetry
                if ellin[1] != ellin[0] and (ellin_swap, wain) not in ellsin:
                    wcoeffs = get_sugiyama_window_convolution_coeffs(ell, ellin_swap)
                    if wcoeffs:
                        Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)
                        to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=ellin_swap, minfolds=0)
                        tmp += jax.lax.map(partial(convolve, swap=True), jnp.arange(edgesin.shape[0]), batch_size=batch_size).T

                wmat_tmp[ellin, wain].append(tmp)

            wmat_tmp[ellin, wain] = jnp.concatenate(wmat_tmp[ellin, wain], axis=0)

    wmat = jnp.concatenate(list(wmat_tmp.values()), axis=1)

    if exact_box_limit is not False:
        # Identity split: W^refac = W[Q] - Qinf W[e_000] + Qinf Delta, with
        # Qinf = Q_000(s -> 0). The truncated pipeline cannot reproduce the box
        # limit, because at fixed (k1, k2) the identity in k3 is a delta in
        # cos(theta12) and its Legendre series needs l -> infinity; handling that
        # term analytically makes the box limit exact at ANY truncation, and the
        # shared truncation error largely cancels between the two pipeline passes.
        poles, Qinf = [], None
        for q in window.ells:
            wpole = window.get(ells=q)
            value = wpole.value()
            if tuple(np.atleast_1d(q).ravel()) == (0, 0, 0):
                Qinf = float(jnp.real(value.ravel()[0]))
                value = jnp.ones_like(value)
            else:
                value = jnp.zeros_like(value)
            poles.append(wpole.clone(value=value))
        assert Qinf is not None, 'exact_box_limit needs the (0, 0, 0) window multipole'
        if exact_box_limit is not True: Qinf = float(exact_box_limit)
        window_const = ObservableTree(poles, ells=list(window.ells))
        wmat_const = compute_smooth3_spectrum_window(
            window_const, _edgesin_arg, ellsin=_ellsin_arg, bin=bin, flags=flags, batch_size=batch_size,
            ellmax=ellmax, ninsub=ninsub, noutsub=noutsub, interp=interp, exact_box_limit=False,
            permute_in=permute_in, permute_lgt0=permute_lgt0)

        # Delta: the exact binned identity. Diagonal in L, and only for the pure
        # Legendre-multipole channel -- the wide-angle terms vanish in the box limit.
        kout_ = np.asarray(bin.xavg)
        edges_ = np.asarray(edgesin)
        inside = np.all((kout_[:, None, :] >= edges_[None, :, :, 0])
                        & (kout_[:, None, :] < edges_[None, :, :, 1]), axis=-1).astype(float)
        blocks = []
        for ellin, wain in ellsin:
            L_in = ellin[0] if isinstance(ellin, tuple) else ellin
            M_in = ellin[1] if isinstance(ellin, tuple) else 0
            pure = (tuple(wain) == (0, 0)) and (M_in == 0)
            blocks.append(jnp.concatenate([jnp.asarray(inside) if (pure and ell == L_in)
                                           else jnp.zeros_like(jnp.asarray(inside))
                                           for ell in ells], axis=0))
        delta = jnp.concatenate(blocks, axis=1)
        wmat = wmat - Qinf * jnp.asarray(wmat_const.value()) + Qinf * delta

    observable = []
    for ill, ell in enumerate(ells):
        observable.append(Mesh3SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes[ill], num_raw=jnp.zeros_like(bin.xavg[..., 0]), basis=bin.basis, ell=ell))
    observable = Mesh3SpectrumPoles(observable)

    theory = []
    kin, edgesin = jnp.mean(edgesin, axis=-1), edgesin
    for ill, ell in enumerate(ellsin):
        #theory.append(ObservableLeaf(k=kin, k_edges=edgesin, value=jnp.zeros_like(kin[..., 0]), coords=['k']))
        theory.append(Mesh3SpectrumPole(k=kin, k_edges=edgesin, num_raw=jnp.zeros_like(kin[..., 0]), basis='sugiyama'))
    #theory = Mesh2SpectrumPoles(theory, ells=ellsin)
    kw = dict(ells=[ell[0] for ell in ellsin], wa_orders=[ell[1] for ell in ellsin])
    theory = ObservableTree(theory, **kw)

    return WindowMatrix(observable=observable, theory=theory, value=wmat)



def compute_fisher_scoccimarro(mattrs, bin, los: str | np.ndarray='z', apply_selection=None, power=None, seed=42, norm=None):

    raise NotImplementedError('not working (yet)')
    assert 'scoccimarro' in bin.basis, 'fisher is only available for scoccimarro basis'

    if apply_selection is None and isinstance(mattrs, RealMeshField):

        selection = mattrs
        mattrs = selection.attrs

        def apply_selection(mesh):
            return mesh * selection

    periodic = apply_selection is None
    rdtype = mattrs.rdtype

    los, vlos = _format_los(los, ndim=mattrs.ndim)

    _norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)**2
    if norm is None: norm = _norm

    if periodic:
        xmid = jnp.mean(bin.edges, axis=-1).T
        kvec = mattrs.kcoords(sparse=True)

        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)
        del knorm

        fisher = [[0. for illin in range(len(bin.ells))] for ell in range(len(bin.ells))]
        for ill, ell in enumerate(bin.ells):
            for illin, ellin in enumerate(bin.ells):
                # The Fourier-space grid
                legin = get_legendre(ellin)(mu)
                legout = get_legendre(ell)(mu)

                def f(weights, ibin):
                    mesh_prod = 1.
                    for ivalue in range(3):
                        mask = (bin.ibin1d[ivalue] == ibin[ivalue]) * weights[ivalue]
                        mesh_prod = mesh_prod * mattrs.c2r(mask.astype(mattrs.dtype))
                    return jnp.sum(mesh_prod)

                def fmap(*weights):
                    return jax.lax.map(partial(f, weights), bin._iedges[mask], batch_size=bin.batch_size)

                tmp = jnp.ones_like(xmid[0])
                mask = (xmid[1] != xmid[0]) & (xmid[2] != xmid[1])
                tmp = tmp.at[mask].set(fmap(1., 1., legin * legout))
                mask = (xmid[1] == xmid[0]) & (xmid[2] != xmid[1])
                tmp = tmp.at[mask].set(2 * fmap(1., 1., legin * legout))
                mask = (xmid[1] != xmid[0]) & (xmid[2] == xmid[1])
                tmp = tmp.at[mask].set(fmap(1., legin, legout) + fmap(1., 1., legin * legout))
                mask = (xmid[1] == xmid[0]) & (xmid[2] == xmid[1])
                tmp = tmp.at[mask].set(4. * fmap(1., legin, legout) + 2. * fmap(1., 1., legin * legout))
                fisher[ill][illin] = jnp.diag(tmp)

        fisher = np.block(fisher)

    else:

        def apply_SinvW(cmap):
            return apply_selection(cmap.c2r()).r2c()

        def apply_Ainv(cmap):
            return cmap * Ainv

        # Define Q map code
        def compute_Q(weighting, cmaps, Q_Ainv=None):
            # Filter maps appropriately
            if weighting == 'Sinv':
                cwmaps = [apply_SinvW(cmap) for cmap in cmaps]
                fisher = jnp.zeros((len(bin.ells) * len(bin._iedges),) * 2)
            else:
                cwmaps = [apply_Ainv(cmap) for cmap in cmaps]
                Q_Ainv = jnp.zeros((len(bin._iedges),) + tuple(mattrs.meshsize))
            del cmaps
            rwmaps = [cwmap.c2r() for cwmap in cwmaps]

            def apply_fourier_legendre(ell, cmesh):
                kvec = mattrs.kcoords(sparse=True)
                mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)
                return get_legendre(ell)(mu) * cmesh

            def apply_fourier_harmonics(ell, rmesh):
                Ylms = [get_Ylm(ell, m, reduced=False, real=True) for m in range(-ell, ell + 1)]
                xvec = mattrs.rcoords(sparse=True)
                kvec = mattrs.kcoords(sparse=True)

                @partial(jax.checkpoint, static_argnums=0)
                def f(Ylm, carry, im):
                    carry += mattrs.r2c(rmesh * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                    return carry, im

                xs = np.arange(len(Ylms))
                return (4. * jnp.pi) * jax.lax.scan(partial(f, Ylms), init=mattrs.create(fill=0., kind='complex'), xs=xs)[0]

            def apply_real_harmonics(ell, cmesh):
                Ylms = [get_Ylm(ell, m, reduced=False, real=True) for m in range(-ell, ell + 1)]
                xvec = mattrs.rcoords(sparse=True)
                kvec = mattrs.kcoords(sparse=True)

                @partial(jax.checkpoint, static_argnums=0)
                def f(Ylm, carry, im):
                    carry += mattrs.c2r(cmesh * jax.lax.switch(im, Ylm, *kvec)) * jax.lax.switch(im, Ylm, *xvec)
                    return carry, im

                xs = np.arange(len(Ylms))
                return (4. * jnp.pi) * jax.lax.scan(partial(f, Ylms), init=mattrs.create(fill=0., kind='real'), xs=xs)[0]

            # Compute g_{b,0} maps
            g_b0_maps, leg_maps = [], []
            for axis in range(2):
                g_b0_maps.append([mattrs.c2r(cwmaps[axis] * (bin.ibin1d[axis + 1] == b)) for b in bin._uiedges[axis + 1]])
                if vlos is None:
                    leg_maps.append([apply_fourier_harmonics(ell, rwmaps[axis]) for ell in bin.ells])

            for ibin3 in bin.uiedges[-1]:
                g_bBl_maps = []
                for axis in range(2):
                    tmp = []
                    for ell in bin.ells:
                        if vlos is not None: tmp.append(mattrs.c2r(apply_fourier_legendre(ell, cwmaps[axis] * (bin.ibin1d[2] == ibin3))))
                        else: tmp.append(mattrs.c2r(leg_maps[axis] * (bin.ibin1d[2] == ibin3)))
                    g_bBl_maps.append(tmp)

                for ibin1 in bin.uiedges[0]:
                    if weighting == 'Sinv':
                        ibins2 = jnp.flatnonzero((bin._iedges[..., 0] == ibin1) & (bin._iedges[..., 2] == ibin3))
                        if not ibins2.size: continue
                        for ill, ell in enumerate(bin.ells):
                            # Compute FT[g_{0, bA}, g_{ell, bB}]
                            ft_ABl = mattrs.r2c(g_b0_maps[0][ibin1] * g_bBl_maps[0][ill][ibin3] - g_b0_maps[1][ibin1] * g_bBl_maps[1][ill][ibin3])
                            for ibin2 in ibins2:
                                fisher = fisher.at[ibin2 + ill * len(bin._iedges)].set(jnp.sum(ft_ABl * Q_Ainv.conj() * (bin.ibin1d[1] == ibin2)))

                    if weighting == 'Ainv':
                        # Find which elements of the Q3 matrix this pair is used for (with ordering)
                        ibins1 = jnp.flatnonzero((bin._iedges[..., 0] == ibin1) & (bin._iedges[..., 1] == ibin1))
                        ibins2 = jnp.flatnonzero((bin._iedges[..., 1] == ibin1) & (bin._iedges[..., 2] == ibin3))
                        ibins3 = jnp.flatnonzero((bin._iedges[..., 2] == ibin3) & (bin._iedges[..., 0] == ibin1))
                        if ibins1.size + ibins2.size + ibins2.size:
                            continue
                        ft_ABl = []
                        for ell in bin.ells:
                            ft_ABl.append(mattrs.r2c(g_bBl_maps[0][bin.ells.index(0)][ibin1] * g_bBl_maps[0][ill][ibin3] - g_bBl_maps[1][bin.ells.index(0)][ibin1] * g_bBl_maps[1][ill][ibin3]))

                        def add_Q_element(Q_Ainv, axis, ibins):
                            # Iterate over these elements and add to the output arrays
                            for ibin2 in ibins:
                                ibin = bin._iedges[ibin, axis]
                                for ill, ell in enumerate(bin.ells):
                                    if (ell == 0) or (axis == 2):
                                        tmp = ft_ABl[ill] * (bin.ibin1d[axis] == ibin)
                                    else:
                                        if vlos is not None:
                                            tmp = apply_fourier_legendre(ell, ft_ABl[bin.ells.index(0)] * (bin.ibin1d[axis] == ibin))
                                        else:
                                            tmp = apply_real_harmonics(ell, ft_ABl[bin.ells.index(0)] * (bin.ibin1d[axis] == ibin)).r2c()
                                    Q_Ainv[ibin2 + ill * len(bin._iedges)] += tmp

                        Q_Ainv = add_Q_element(Q_Ainv, 2, ibins1)
                        Q_Ainv = add_Q_element(Q_Ainv, 0, ibins2)
                        Q_Ainv = add_Q_element(Q_Ainv, 1, ibins3)

            if weighting == 'Sinv':
                return fisher
            return Q_Ainv

        A, Ainv = 1., 1.
        if power is not None:
            kvec = mattrs.kcoords(sparse=True)
            A = power(kvec)
            Ainv = jnp.where(A == 0., 1., 1 / A)

        if isinstance(seed, int):
            seed = random.key(seed)

        seeds = random.split(seed, 2)
        cmaps = [mattrs.create(kind='real', fill=create_sharded_random(random.normal, seed, shape=mattrs.meshsize)).r2c() * jnp.sqrt(A) for seed in seeds]

        Q_Ainv = compute_Q('Ainv', cmaps)
        for ibin in range(Q_Ainv.shape[0]):
            Q_Ainv = Q_Ainv.at[ibin].set(apply_SinvW(Q_Ainv))

        fisher = compute_Q('Sinv', cmaps, Q_Ainv=Q_Ainv)
        fisher = 1. / 2. * fisher / norm

    spectrum = []
    for ill, ell in enumerate(bin.ells):
        spectrum.append(Mesh3SpectrumPole(k=bin.xavg, ell=ell, num=jnp.zeros_like(bin.xavg), nmodes=bin.nmode, edges=bin.edges, norm=norm * jnp.ones_like(bin.xavg), attrs=dict(los=vlos if vlos is not None else los), basis=bin.basis))
    observable = Mesh3SpectrumPoles(spectrum)
    theory = observable.at().clone(num=[jnp.ones_like(bin.xavg)] * len(bin.ells))
    window = WindowMatrix(observable=observable, theory=theory, fisher=fisher)
    return window