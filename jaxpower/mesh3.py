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
from .utils import real_gaunt, get_legendre, get_legendre_recurrence, get_spherical_jn, get_spherical_jn_all, get_Ylm, get_Ylm_all, wigner_3j, wigner_9j, register_pytree_dataclass


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
    #print(vecmax)

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
            #symfactor = jnp.ones_like(xmid[0])
            #symfactor = jnp.where((xmid[1] == xmid[0]) | (xmid[2] == xmid[0]) | (xmid[2] == xmid[1]), 2, symfactor)
            #symfactor = jnp.where((xmid[1] == xmid[0]) & (xmid[2] == xmid[0]), 6, symfactor)

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
        Else, a 3-vector. In case of the sugiyama basis, 'local' and 'z' only are supported.

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
        Else, a 3-vector. In case of the sugiyama basis, 'local' and 'z' only are supported.

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
        Else, a 3-vector. In case of the sugiyama basis, 'local' and 'z' only are supported.

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
        Else, a 3-vector. In case of the sugiyama basis, 'local' and 'z' only are supported.
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

    # Take the meshes as _format_meshes returned them. It has ALREADY resolved every repeated
    # field (`if mesh is None: meshes[imesh] = meshes[fields.index(field)]`).
    particles = []
    for fkp in fkps:
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
                weights = particles[0].weights * particles[1].weights * particles[2].weights
                rmesh = particles[0].clone(weights=weights).paint(**kwargs, out='real')
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
def get_scoccimarro_los_coeffs(ell, m=0, normalize=True):
    r"""
    Host-side coefficients of the TripoSH *shape factor* in the :math:`\hat{k}_3` reference-leg
    convention (desi-gqc-notes, sec. "The reference leg").

    The bispectrum estimator applies :math:`\mathcal{L}_L` to the leg that CLOSES the triangle,
    :math:`\hat{k}_3` (see :func:`compute_mesh3_spectrum`: ``meshes[2] * get_legendre(ell3)(mu)``
    for a global line of sight, ``Ylms[2][...] * meshes[2]`` for a local one). The window-matrix
    derivation is naturally written in terms of the rotational scalar

    .. math::

        \mathcal{S}^{(r)}_{\ell_1\ell_2L} = \frac{1}{H_{\ell_1\ell_2L}} \sum_{m_1 m_2 M}
        \begin{pmatrix}\ell_1&\ell_2&L\\m_1&m_2&M\end{pmatrix}
        y_{\ell_1 m_1}(\hat{k}_1)\, y_{\ell_2 m_2}(\hat{k}_2)\, y_{LM}(\hat{k}_r),

    which for :math:`r = 1` collapses to the single Legendre
    :math:`\mathcal{L}_{\ell_2}(\cos\theta_{12})`, but for :math:`r = 3` does not: evaluating it in
    the frame :math:`\hat{k}_3 = \hat{z}` forces :math:`M = 0` and puts the other two legs at
    OPPOSITE azimuths (their transverse parts must cancel), leaving

    .. math::

        \mathcal{S}^{(3)}_{\ell_1\ell_2L} = \frac{1}{H_{\ell_1\ell_2L}} \sum_{\mu} (-1)^{\mu_2}
        \begin{pmatrix}\ell_1&\ell_2&L\\\mu&\mu_2&0\end{pmatrix}
        y_{\ell_1 \mu}(\theta_{31}, 0)\, y_{\ell_2 \mu_2}(\theta_{32}, 0), \quad \mu_2 = -\mu,

    a finite sum of associated Legendre values at the two interior angles measured FROM leg 3. The
    same expression with :math:`M' \neq 0` and without the :math:`1/H` (it cancels against the
    Sugiyama denominator) is the theory-side factor :math:`\Sigma^{(3)}`, which replaces
    ``3j(l1', l2', L'; 0, -M', M') y_{l2'}^{-M'}(cos theta_12', 0)``.

    The two conventions are identical at :math:`L = 0` and differ by :math:`\mathcal{O}(1)` beyond,
    so only :math:`L \geq 2` results change. :math:`\mathcal{S}^{(3)}` is symmetric under
    :math:`(\ell_1, k_1) \leftrightarrow (\ell_2, k_2)` while :math:`\mathcal{S}^{(1)}` is not --
    which is what makes the :math:`k_1' \leftrightarrow k_2'` enumeration legitimate. See
    ``tests/test_shape_factor.py``, which checks both closed forms against the definition evaluated
    in an arbitrary frame.

    Parameters
    ----------
    ell : tuple
        :math:`(\ell_1, \ell_2, L)`.
    m : int, default=0
        :math:`M`. Non-zero only for the theory side with explicit :math:`(L', M')` multipoles.
    normalize : bool, default=True
        Divide by :math:`H_{\ell_1\ell_2L}`, giving :math:`\mathcal{S}^{(3)}` (output side).
        ``False`` gives :math:`\Sigma^{(3)}` (theory side), whose :math:`H` cancels elsewhere.

    Returns
    -------
    coeffs : tuple of (mu1, mu2, coeff)
        Terms of the :math:`\mu` sum; the harmonics themselves come from
        :func:`~jaxpower.utils.get_Ylm_all`.
    """
    ell1, ell2, L = ell
    H = wigner_3j(ell1, ell2, L, 0, 0, 0)
    coeffs = []
    for mu1 in range(-ell1, ell1 + 1):
        mu2 = -mu1 - m
        if abs(mu2) > ell2: continue
        coeff = (-1)**mu2 * wigner_3j(ell1, ell2, L, mu1, mu2, m)
        if normalize: coeff = coeff / H
        if abs(coeff) < 1e-12: continue
        coeffs.append((mu1, mu2, coeff))
    return tuple(coeffs)


@functools.lru_cache(maxsize=None)
def get_scoccimarro_window_convolution_coeffs(ell, ellin, ellmax=4):
    r"""
    Coefficients for the smooth window convolution in the Scoccimarro basis,
    routing through the TripoSH (Sugiyama) basis where the radial transform is
    a diagonal 2D Hankel transform. Estimator side (thin :math:`k`-shells):

    .. math::

        \tilde{B}_L(k_1, k_2, k_3) = \sum_{\ell_1 \ell_2} \mathcal{S}^{(3)}_{\ell_1\ell_2L}(k_1, k_2, k_3)\,
        \tilde{B}_{\ell_1 \ell_2 L}(k_1, k_2),

    with :math:`\mathcal{S}^{(3)}` the reference-leg shape factor of
    :func:`get_scoccimarro_los_coeffs` -- the estimator's line-of-sight Legendre refers to the
    leg that closes the triangle, :math:`\hat{k}_3`, so this is NOT the single Legendre
    :math:`\mathcal{L}_{\ell_2}(\cos\theta_{12})` that the :math:`\hat{k}_1` convention would give
    (the two agree at :math:`L = 0` and differ by :math:`\mathcal{O}(1)` beyond) --
    and :math:`\tilde{B}_{\ell_1\ell_2L}` the TripoSH window convolution
    (:func:`get_sugiyama_window_convolution_coeffs`) of the theory TripoSH multipoles,
    themselves projected from the Scoccimarro-basis theory (eq. 25 of arXiv:1803.02132):

    .. math::

        B_{\ell_1'\ell_2'L'}(k_1', k_2') = \frac{N_{\ell_1'\ell_2'L'} H_{\ell_1'\ell_2'L'}}{\sqrt{4\pi(2L'+1)}}
        \int \frac{k_3'^2 dk_3'}{2\pi^2}\, I_{000}(k_1', k_2', k_3')\,
        \Sigma^{(3)}_{\ell_1'\ell_2'L'M'}(k_1', k_2', k_3')\, B_{L'M'}(k_1', k_2', k_3'),

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
        ``sugiyama_ell = (ell_1, ell_2, L)`` is the estimator-side TripoSH multipole (to be
        resummed with weight :math:`\mathcal{S}^{(3)}_{\ell_1\ell_2L}`),
        ``sugiyama_ellt = (ell_1', ell_2', L')`` the theory-side one (to be projected with weight
        :math:`I_{000}\, \Sigma^{(3)}_{\ell_1'\ell_2'L'M'}`); both angular factors come from
        :func:`get_scoccimarro_los_coeffs`. The projection prefactor
        :math:`N' H' / \sqrt{4\pi(2L'+1)}` is folded into the window coefficients ``coeff`` -- but
        NOT the :math:`3j`, which now lives inside :math:`\Sigma^{(3)}`.
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
            # Theory-side projection (Scoccimarro to TripoSH), eq. 25 of arXiv:1803.02132.
            # The 3j (l1', l2', L'; 0, -M', M') that used to sit here has moved INTO the theory-side
            # angular factor: in the k_3 reference-leg convention the 3j and the harmonic no longer
            # factorize, the pair being replaced by Sigma^(3) (get_scoccimarro_los_coeffs with
            # normalize=False), whose mu = 0 term is exactly this 3j times y_{l2'}^{-M'}.
            scoccimarro_to_sugiyama = prod((2 * ell_ + 1) for ell_ in sugiyama_ellt) * H
            if min is None:  # Legendre-multipole input: B_{L'0} = sqrt(4 pi / (2 L' + 1)) B_{L'}, M' = 0
                scoccimarro_to_sugiyama *= 1. / (2 * ellin + 1)
            else:
                scoccimarro_to_sugiyama *= 1. / np.sqrt(4. * np.pi * (2 * ellin + 1))
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


def get_scoccimarro_symmetrization_matrix(kin=None, kin_ordered=None, edges_ordered=None,
                                          fix_legs=None, atol=1e-9):
    r"""
    Matrix :math:`S` scattering an ORDERED theory vector onto the UNORDERED
    :math:`(k_1', k_2', k_3')` grid that :func:`compute_smooth3_spectrum_window` integrates
    over: ``theory_unordered = S @ theory_ordered``.

    The estimator bins ORDERED triangles :math:`k_1' \leq k_2' \leq k_3'`, but the window
    convolution runs over all of :math:`k'`-space, so the matrix must be given the unordered
    grid (pass ``edgesin`` as raw per-axis edges and it builds one). Filling that grid requires
    knowing :math:`B_{L'}` at each ordering, which is a THEORY-side question and is only a
    relabelling when the multipole is invariant under the permutation.

    Validity, per multipole, if the three :math:`\delta`-fields are the same:

    - ``fix_legs=(2,)``: exact for EVERY :math:`L'`.
      The scoccimarro :math:`B_{L'}` is referred to :math:`\hat{k}_3 \cdot \hat{z}` (the estimator applies the output Legendre to the
      third leg), and swapping :math:`k_1 \leftrightarrow k_2` leaves that leg alone, so
      :math:`B_{L'}(k_1,k_2,k_3) = B_{L'}(k_2,k_1,k_3)` identically.
    - ``fix_legs=None`` (all permutations): exact for :math:`L' = 0` only, since :math:`B_0` is
      an orientation average of a rigid triangle and therefore symmetric. For :math:`L' > 0`
      a reordering moves the line-of-sight reference to a different leg, mixing :math:`L` and
      involving the second angular coordinate this basis drops -- so it is NOT a relabelling,
      and the theory must supply those entries itself.

    Rows with no admissible permutation are left ZERO, i.e. that part of :math:`k'`-space
    contributes nothing; the box-limit sum rule then falls short by the corresponding weight,
    which is the honest signal that the theory is incomplete there.

    Either grid may be left out and is then inferred from the other:

    - ``kin_ordered=None``: taken as the distinct sorted triples occurring in ``kin``, in order
      of first appearance.
    - ``kin=None``: built as every distinct permutation of ``kin_ordered``. Note this ignores
      ``fix_legs`` on purpose -- the grid is a property of :math:`k'`-space, not of any one
      multipole's symmetry, so a single grid (hence a single window matrix) serves every
      :math:`L'`, each with its own ``fix_legs`` and therefore its own :math:`S`.

    Parameters
    ----------
    kin : array_like, optional
        ``(n_unordered, 3)`` representative :math:`(k_1', k_2', k_3')` of the grid the window
        matrix is built on (``wmatrix.theory.get(...).coords('k')``). Inferred from
        ``kin_ordered`` when absent.
    kin_ordered : array_like, optional
        ``(n_ordered, 3)`` representative triangles of the ordered theory vector. Inferred from
        ``kin`` when absent.
    edges_ordered : array_like, optional
        ``(n_ordered, 3, 2)`` bin edges of the ordered vector. Permuted alongside a ``kin``
        built here and returned, so the caller can construct ``edgesin`` without redoing the
        permutation enumeration.
    fix_legs : tuple, optional
        Leg positions the permutation must leave in place, e.g. ``(2,)`` to keep the third leg
        third. ``None`` allows every permutation.
    atol : float, default=1e-9
        Tolerance of the fixed-leg check. Matching a sorted triple to a row of ``kin_ordered``
        is exact: both grids are expected to come from the same edges, hence bitwise equal.

    Returns
    -------
    S : np.ndarray
        ``(n_unordered, n_ordered)``, entries 0 or 1.
    kin : np.ndarray
        The unordered grid, as passed or as built.
    kin_ordered : np.ndarray
        The ordered grid, as passed or as inferred. Returned so the column order of ``S`` is
        never in doubt.
    edges : np.ndarray or None
        ``(n_unordered, 3, 2)`` permuted edges, when ``edges_ordered`` was given and ``kin`` was
        built here; ``None`` otherwise.
    """
    if kin is None and kin_ordered is None:
        raise ValueError('provide at least one of kin, kin_ordered')
    edges = None
    if kin_ordered is None:
        kin = np.asarray(kin, dtype='f8')
        sorted_kin = np.sort(kin, axis=-1)
        # unique returns first occurrences; re-sorting the indices restores input order
        index = np.unique(sorted_kin, axis=0, return_index=True)[1]
        kin_ordered = sorted_kin[np.sort(index)]
    elif kin is None:
        kin_ordered = np.asarray(kin_ordered, dtype='f8')
        eo = None if edges_ordered is None else np.asarray(edges_ordered, dtype='f8')
        # every permutation of every ordered row, ordered-row-major as the nested loop was
        perms = np.array(list(itertools.permutations(range(3))))
        rows = kin_ordered[:, perms].reshape(-1, 3)
        erows = None if eo is None else eo[:, perms].reshape(-1, 3, eo.shape[-1])
        # dedupe on the BOX when edges are known (degenerate legs give identical boxes),
        # else on the triple
        keys = rows if erows is None else erows
        index = np.unique(keys.reshape(len(rows), -1), axis=0, return_index=True)[1]
        index = np.sort(index)
        kin = rows[index]
        if erows is not None: edges = erows[index]
    kin, kin_ordered = np.asarray(kin, dtype='f8'), np.asarray(kin_ordered, dtype='f8')
    fix = () if fix_legs is None else tuple(int(leg) for leg in np.atleast_1d(fix_legs))
    sorted_kin = np.sort(kin, axis=-1)
    # group both grids by their sorted triple: rows sharing a group are permutations of each other
    keys = np.concatenate([np.sort(kin_ordered, axis=-1), sorted_kin])
    groups, index_1d = np.unique(keys, axis=0, return_inverse=True)
    index_1d = index_1d.ravel()
    inv_ordered, inv_kin = index_1d[:len(kin_ordered)], index_1d[len(kin_ordered):]
    # column of each group, -1 where the group has none; reversed so the FIRST ordered row wins
    column = np.full(len(groups), -1)
    column[inv_ordered[::-1]] = np.arange(len(kin_ordered))[::-1]
    # admissible only if sorting leaves every fixed leg where it already is
    admissible = np.all(np.abs(kin[:, fix] - sorted_kin[:, fix]) <= atol, axis=-1)
    S = np.zeros((len(kin), len(kin_ordered)))
    rows = np.flatnonzero(admissible & (column[inv_kin] >= 0))
    S[rows, column[inv_kin][rows]] = 1.
    return S, kin, kin_ordered, edges


def compute_smooth3_spectrum_window(window, edgesin: np.ndarray | tuple, ellsin: tuple=None, bin: BinMesh3SpectrumPoles=None,
                                    flags: tuple=None, batch_size: int=None, ellmax: int=16,
                                    ninsub: int=16, noutsub: int=8, pbar: bool=False) -> WindowMatrix:
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
    ellmax : int, default=16
        For the scoccimarro basis. Truncation of the TripoSH multipole sums resolving
        the internal (opening-angle) dependence, see
        :func:`get_scoccimarro_window_convolution_coeffs`. Use the same value as passed to
        :func:`get_smooth3_window_bin_attrs` (window multipoles missing from ``window``
        are silently treated as zero).
        **``ellmax`` does not converge on its own -- it converges jointly with the radial (Bessel)
        resolution of ``window``.** The high-order Bessel content has to be resolved
        before the extra multipoles mean anything.
        Cost is linear here -- the surviving term count is exactly ``4 * ellmax - 1`` -- and
        quadratic in the window's grid size, so 16 is deliberately chosen as the point where the
        pipeline reaches the analytic rather than as margin.
    ninsub : int, default=16
        Theory-side sub-binning, so the theory basis represents the bin rather than its midpoint.
        Scoccimarro basis: sub-binning of the :math:`k_3^{\\prime}` measure integral. Not optional:
        the point value breaks the box-limit sum rule by tens of per cent (~16% at ``ninsub = 1``,
        1.9% at 16).
        Sugiyama basis: number of Gauss-Legendre sub-nodes per theory bin per axis used to build
        the theory basis, summed so the bins keep tiling. There is no
        :math:`k_3^{\\prime}` leg to integrate here, but the midpoint spline basis is truncated at
        the range ends by its own extrapolation cut-off, which loses weight there; increasing
        ``ninsub`` tends to an exactly tiling tophat instead. ``ninsub = 1`` places the single node
        at the bin centre and so reproduces the plain midpoint spline basis exactly.
    noutsub : int, default=8
        Output-side sub-binning.
        Scoccimarro basis: the full 3-D bin average of the rapidly varying angular factor
        :math:`\\mathcal{L}_{\\ell_2}(\\cos\\theta_{12})`, all three legs sub-binned and weighted
        by :math:`k_1^2k_2^2k_3^2` times the triangle measure. Must be >= 2 here: taking that
        factor at the bin's representative triangle instead is 20-40% wrong on squeezed
        configurations, so that path is gone and ``noutsub < 2`` raises.
        Cost grows as ``noutsub**3`` in the number of sub-triangles, but only ``noutsub**2`` of
        them reach the interpolation (the gather sees the :math:`k_1, k_2` legs only), so the
        practical scaling is milder than the node count suggests.
        Sugiyama basis: sub-node refinement of the :math:`k^2`-weighted output-bin average. Expect
        little effect and use it as a cross-check rather than a correction -- these multipoles bin
        only two legs and carry no angular factor, so the output measure :math:`k_1^2k_2^2`
        factorizes and the separable ``matrix_rebin`` is already the exact 2-D bin average.
    batch_size : int, optional
        Number of theory bins whose 2D FFTlog pair is evaluated in parallel (the ``batch_size`` of
        the underlying :func:`jax.lax.map`). This is the main throughput knob: the per-bin work is a
        handful of length-``n`` FFTs, far too little to fill a GPU on its own, so the run is
        launch-latency bound at small batches. Results do not depend on it.
        ``None`` (default) picks the largest batch whose dominant ``(batch_size, n1, n2)`` complex
        intermediate stays within 1 GB, capped at 64.
        Raise it if memory allows, lower it on out-of-memory.
    pbar : bool, default=False
        Whether to show a progress bar over the terms of the multipole sums. Note the granularity is
        one term (of ``4 * ellmax - 1`` for the scoccimarro basis), each a full pass over the theory
        bins, so the bar advances rarely but the ETA is meaningful. Updating it forces a device
        synchronization per term, which is otherwise unnecessary.

    Returns
    -------
    wmat : WindowMatrix
    """
    ells = bin.ells
    if isinstance(edgesin, ObservableTree):
        ellsin = edgesin.ells
        if 'wa_orders' in edgesin.labels(return_type='keys'):
            ellsin = [(ell, wa) for ell, wa in zip(edgesin.ells, edgesin.wa_orders)]
        pole = next(iter(edgesin))
        edgesin = jnp.asarray(pole.edges('k'))

    else:
        if not isinstance(edgesin, (list, tuple)):
            edgesin = (edgesin,)
        edgesin = tuple(edgesin)
        edgesin += (edgesin[-1],) * (bin.xavg.shape[-1] - len(edgesin))
        grid_edgesin = []
        for edge in edgesin:
            if edge.ndim == 1: edge = jnp.column_stack([edge[:-1], edge[1:]])
            grid_edgesin.append(edge)
        def _cproduct(arrays):
            grid = jnp.meshgrid(*arrays, sparse=False, indexing='ij')
            return jnp.column_stack([tmp.ravel() for tmp in grid])

        # of shape (nbins, ndim, 2)
        edgesin = jnp.stack([_cproduct([edge[..., iedge] for edge in grid_edgesin]) for iedge in range(2)], axis=-1)
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
            edgesin = edgesin[mask]

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

    # interp_order=1, not 3. A cubic spline basis rings. Defined here, before the nested helpers that close over it.
    interp_order = 1

    # Per-bin work is only a few length-n FFTs, so with batch_size = 1 (what lax.map does when given
    # None) the run is dominated by launch latency rather than arithmetic -- most of all on a GPU.
    # Size the batch against the dominant (batch_size, n1, n2) complex intermediate instead.
    if batch_size is None:
        memory_target = 1024**3  # bytes
        batch_size_max = 64
        _wshape = [len(c) for c in next(iter(window)).coords().values()]
        _nbytes = int(np.prod(_wshape)) * 16  # complex128
        batch_size = int(np.clip(memory_target // max(_nbytes, 1), 1, batch_size_max))
    batch_size = max(int(batch_size), 1)

    def get_w_rect_label(q, wain):
        """Label of the window multipole a term reads, and whether it comes in transposed."""
        transpose = False
        if q not in window.ells:
            q = q[1::-1] + q[2:]
            transpose = True
        kw = dict(ells=q)
        if 'wa_orders' in window.labels(return_type='keys'):
            kw.update(wa_orders=wain)
        elif wain != (0, 0):
            raise ValueError('wa_orders must be provided in input window')
        return kw, transpose

    def has_w_rect(q, wain):
        """Whether ``window`` carries this multipole at all; absent ones contribute exactly zero."""
        kw, _ = get_w_rect_label(q, wain)
        return kw in window.labels(return_type='flatten')

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

    def axis_basis_matrices(edges, k_axes, kind, nsub=1):
        """Per-axis (separable) replacement for the sharp tophat/read: build,
        for each axis independently, a SMALL matrix of shape (n_k_axis,
        n_unique_axis) (kind='spline', input/theory side: a smooth spline basis
        function per distinct bin center, matrix_spline_interp) or
        (n_unique_axis, n_k_axis) ('rebin', output side: a proper k^2-weighted
        bin average, matrix_rebin), keyed by the axis's DISTINCT bin
        centers/edges -- never the full (nbins1 x nbins2, n_k1 x n_k2) dense
        tensor (edges may be a masked/paired list, e.g. sugiyama-diagonal's
        k1=k2, not a full product grid; this stays correct and small either
        way). Returns (index_per_bin (nbins, ndim), matrices list).

        ``nsub`` > 1 replaces each bin's single representative midpoint by
        ``nsub`` Gauss-Legendre sub-nodes spanning the bin, so the basis
        represents the BIN rather than its centre (``nsub = 1`` puts the single
        node at the centre and reproduces the midpoint construction exactly,
        which is why it is the default here and leaves the scoccimarro callers
        below untouched).
        """
        edges = np.asarray(edges)
        centers = edges.mean(axis=-1)
        ndim = centers.shape[-1]
        index_per_bin = np.empty(centers.shape, dtype=np.int64)
        matrices = []
        for d in range(ndim):
            u, first_idx, inv = np.unique(centers[:, d], return_index=True, return_inverse=True)
            index_per_bin[:, d] = inv
            kk = np.asarray(k_axes[d])
            unique_edges = edges[first_idx, d, :]
            if nsub > 1:
                # Sub-node construction, shared by both kinds. 'spline' SUMS the sub-node
                # interpolation weights and never averages them: the sum over all bins of a
                # partition-of-unity interpolant must stay 1 at every k. As nsub
                # grows the summed basis tends to the sharp tophat indicator that tiles the bins
                # exactly, i.e. the scoccimarro branch's theory-side primitive, so nsub
                # interpolates monotonically between the midpoint spline (nsub = 1) and that
                # tophat. 'rebin' instead AVERAGES, weighted by the k^2 measure, matching
                # matrix_rebin's definition.
                from .pt import integration
                # bounds are per unique bin, so the nodes come out (n_unique, nsub) directly,
                # with weights already carrying the bin width
                integ_sub = integration(unique_edges[:, :1], unique_edges[:, 1:], size=nsub)
                ksub, wsub_bin = np.asarray(integ_sub.x()), np.asarray(integ_sub.w)
                if kind == 'spline':
                    # theory known AT the sub-nodes, evaluated ON the fftlog grid
                    Msub = matrix_spline_interp(jnp.asarray(ksub.ravel()), kk, interp_order=interp_order)
                    M = jnp.sum(Msub.reshape(Msub.shape[0], *ksub.shape), axis=-1)       # (n_k, n_unique)
                    _lo, _hi = unique_edges.min(), unique_edges.max()
                    M = M * ((kk >= _lo) & (kk <= _hi))[:, None]
                else:
                    # opposite direction: the transformed spectrum is known ON the fftlog grid and
                    # is READ at the sub-nodes, then averaged over the bin with the k^2 measure
                    Msub = matrix_spline_interp(kk, jnp.asarray(ksub.ravel()), interp_order=interp_order)
                    Msub = Msub.reshape(*ksub.shape, Msub.shape[-1])                     # (n_unique, nsub, n_k)
                    wsub = jnp.asarray(wsub_bin * ksub**2)
                    M = jnp.sum(wsub[..., None] * Msub, axis=1) / jnp.sum(wsub, axis=-1)[:, None]
            elif kind == 'spline':
                M = matrix_spline_interp(u, kk, interp_order=interp_order)
                # matrix_spline_interp extrapolates via the spline's own boundary
                # polynomial outside [u.min(), u.max()], and kk (the fftlog k-grid) spans a
                # far wider range than the input bins, where cubic extrapolation explodes --
                # so it must be cut off.
                lo, hi = unique_edges.min(), unique_edges.max()
                in_range = (kk >= lo) & (kk <= hi)
                M = M * in_range[:, None]
            else:
                M = matrix_rebin(unique_edges, kk, wt=kk**2, interp_order=interp_order)
            matrices.append(M)
        return jnp.asarray(index_per_bin), matrices

    def read_index(kout, k):
        """Bilinear interpolation indices and weights for reading a grid at ``kout``.

        Split out of :func:`read` because it depends only on (kout, k) -- neither of which varies
        with the theory bin -- so callers inside a per-bin loop can hoist it and pay the
        searchsorted once per term instead of once per (term, bin).
        """
        id0, s = [], []
        mask = 1.
        for kk, kkout in zip(k, kout):
            id0_ = jnp.searchsorted(kk, kkout, side='right') - 1
            mask = mask * ((id0_ >= 0) & (id0_ < len(kk) - 1.))
            id0_ = jnp.clip(id0_, 0, len(kk) - 2)
            id0.append(id0_)
            s.append((kkout - kk[id0_]) / (kk[id0_ + 1] - kk[id0_]))
        ishifts = np.array(list(itertools.product(* len(k) * (np.arange(2),))), dtype=('i4'))
        return jnp.column_stack(id0), jnp.column_stack(s), mask, ishifts

    def read_apply(index, value):
        id0, s, mask, ishifts = index

        def step(carry, ishift):
            idx = id0 + ishift
            ker = jnp.prod((ishift == 0) * (1 - s) + (ishift == 1) * s, axis=-1)
            idx = jnp.unstack(idx, axis=-1)
            carry += value[idx] * ker * mask
            return carry, None

        toret = jnp.zeros_like(value, shape=id0.shape[0])
        return jax.lax.scan(step, toret, ishifts)[0]

    wmat_tmp = {}

    def make_pbar(total):
        """tqdm over the terms of the multipole sums, or a no-op stand-in."""
        if not pbar:
            class _NoPbar(object):
                def update(self, n=1): pass
                def close(self): pass
            return _NoPbar()
        from tqdm import tqdm
        return tqdm(total=total, desc='window terms',
                    bar_format='{l_bar}{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    if 'scoccimarro' in bin.basis:

        # Upper bound on the Bessel orders reached below, so the spherical-Bessel table covers them.
        # Enumerated rather than assumed to equal ellmax: an order ABOVE the bound would be silently
        # CLAMPED by jnp.take into a lower-order Bessel -- the same failure mode as the tabulated
        # get_legendre. Taken over ALL FOUR leg orders, output and theory: the Bessel gather uses
        # the first component of each key as well as the second.
        ellmax_legendre = max([0] + [max(ell_out_key[:2] + ell_theory_key[:2])
                                     for ellin_key, wain_key in ellsin for ell_key in ells
                                     for ell_out_key, ell_theory_key, _ in get_scoccimarro_window_convolution_coeffs(ell_key, ellin_key, ellmax=ellmax)])
        spherical_jn_all = get_spherical_jn_all(ellmax_legendre)

        def cos31_cos32(triangle):
            """Interior angles measured FROM leg 3: (3, 1) opposite leg 2, and (3, 2) opposite 1."""

            def cos_clipped(numerator, denominator):
                # Bin centers can fall (slightly) outside the triangle inequality
                # for coarse binning even though the bin itself has partial valid
                # overlap; the tophat below already zeroes the contribution there,
                # but an unclipped cos blows up sqrt(1 - cos**2) into NaN, and
                # NaN * 0 = NaN propagates through the whole matrix. Clip only for
                # the harmonic evaluation, not for the tophat mask itself.
                return jnp.clip(numerator / denominator, -1., 1.)

            return (cos_clipped(triangle[1]**2 - triangle[2]**2 - triangle[0]**2, 2 * triangle[2] * triangle[0]),
                    cos_clipped(triangle[0]**2 - triangle[1]**2 - triangle[2]**2, 2 * triangle[1] * triangle[2]))

        def compute_measure(triangle):
            """The thin-shell measure I_000 = pi^2 Theta(triangle) / (k1 k2 k3)."""
            cos = (triangle[2]**2 - triangle[1]**2 - triangle[0]**2) / (2 * triangle[0] * triangle[1])
            tophat = (jnp.abs(cos) < 1.) + 1. / 2. * (jnp.abs(cos) == 1.)
            return np.pi**2 / prod(triangle) * tophat

        # ---- table-driven shape factors: ONE compilation for every term ----
        # The reference-leg shape factor sum_mu c_mu y_{l1 mu}(theta31) y_{l2 mu2}(theta32) -- both
        # angles measured FROM leg 3, the leg the estimator's Legendre refers to -- is what the two
        # angular factors below evaluate. Written directly it needs a STATIC (ell1, ell2, L), since
        # get_Ylm casts its order to a Python int, and so costs one XLA compilation per surviving
        # term -- 2 x 63 = 126 at ellmax = 16, which DOMINATES the run (the arithmetic itself is
        # ~16 min for the Q000 window, the compilations far more).
        # The fix: the shape factor is separable, and across ALL terms only (ellmax + 1)^2 distinct
        # (ell, mu) harmonics ever appear -- 289 at ellmax = 16 against 129150 requested evaluations,
        # a 447x redundancy. So tabulate the harmonics once per batch of evaluation points and turn
        # each term into a traced GATHER plus a weighted sum. Orders become data, nothing is static,
        # and one graph serves every term.
        # The table comes from the recurrence rather than get_Ylm's closed forms: same values, but
        # one scan instead of (ellmax + 1)^2 lambdified expressions unrolled into this jit (1223 vs
        # 43033 jaxpr equations at ellmax = 16), and no high-order cancellation. Flat-indexed
        # exactly as harmonic_index, at azimuth 0 where the reduced COMPLEX harmonic is real.
        harmonic_table = get_Ylm_all(ellmax_legendre, reduced=True)

        def pack_shape_keys(keys):
            r"""Pack ``[((ell1, ell2, L), [(mu1, mu2, coeff), ...]), ...]`` into rectangular arrays.

            Returns ``(index1, index2, weight)``, each ``(nkeys, nmu)``: ``index1[k, i]`` and
            ``index2[k, i]`` are the rows of the harmonic table holding :math:`y_{\ell_1 \mu_1}`
            and :math:`y_{\ell_2 \mu_2}` for the i-th term of the k-th key, and ``weight[k, i]``
            is that term's coefficient.

            The mu-sums are ragged (1 to 2 ellmax + 1 terms, median 17 at ellmax = 16), so shorter
            ones are padded with weight 0. That is what makes the contraction below a static loop
            of the same length for every key -- one traced graph for the whole matrix -- at the
            cost of gathering on padding, which is about half the rows.
            """
            def harmonic_index(ell, m):
                """Flat index of the reduced harmonic y_{ell m} in the table below."""
                return ell * ell + ell + m

            maxlen = max([1] + [len(coeffs) for _, coeffs in keys])
            index1 = np.zeros((len(keys), maxlen), dtype=np.int32)
            index2 = np.zeros_like(index1)
            weight = np.zeros((len(keys), maxlen))
            for ikey, (ell, coeffs) in enumerate(keys):
                for imu, (mu1, mu2, coeff) in enumerate(coeffs):
                    index1[ikey, imu] = harmonic_index(ell[0], mu1)
                    index2[ikey, imu] = harmonic_index(ell[1], mu2)
                    weight[ikey, imu] = coeff
            return jnp.asarray(index1), jnp.asarray(index2), jnp.asarray(weight)

        def shape_from_tables(table1, table2, index1, index2, weight):
            r"""Shape factor of every packed key at once.

            Evaluates :math:`\sum_\mu c_\mu y_{\ell_1 \mu_1}(\theta_{31}) y_{\ell_2 \mu_2}(\theta_{32})`
            for each key, by reading the two harmonic tables -- ``(nharm,) + point_shape``, from
            :func:`~jaxpower.utils.get_Ylm_all`, one evaluated at each of the two angles
            -- at the rows :func:`pack_shape_keys` recorded. Returns ``(nkeys,) + point_shape``.

            Accumulated one mu-term at a time rather than as a single ``(nkeys, nmu) + point_shape``
            tensor summed over mu: the tables are already the largest arrays here (why the
            theory-side factor is chunked over bins at all), and this way nothing bigger than one
            term's worth of them is ever live.
            """
            expand = (slice(None),) + (None,) * (table1.ndim - 1)   # weight broadcasts over the points
            toret = 0.
            for imu in range(index1.shape[1]):   # static, = longest mu-sum; each step is a gather
                toret = toret + weight[:, imu][expand] * table1[index1[:, imu]] * table2[index2[:, imu]]
            return toret

        # Theory-side quadrature nodes. Built unconditionally: the k1', k2' legs are now integrated
        # by the same rule (see the quadrature block below), so ninsub = 1 must still give a node.
        from .pt import integration
        integ_in = integration(-1., 1., size=max(int(ninsub), 1))
        nodes_in, weights_in = jnp.asarray(integ_in.x()), jnp.asarray(integ_in.w)

        # OUTPUT-side bin averaging (ms.tex caveat (ii) on eq:scoccimarro_window_matrix_explicit:
        # "average the (k1, k2, k3) dependence over the bin"). The theory side does this via
        # ninsub; the output side carries its own rapidly varying, discontinuous factor
        # L_{ell2}(cos theta12) = (-1)^ell2 I_{ell2 ell2 0} / I_000, and evaluating it at the
        # single representative triangle bin.xavg is NOT the bin average. Since
        # d cos(theta12) / d k3 = k3 / (k1 k2), a SHORT leg makes cos(theta12) sweep the bin:
        # measured, L_2 at the midpoint is +20% off its bin average for (0.033, 0.071, 0.071)
        # and +40% for (0.033, 0.033, 0.033), but only ~1% for (0.110, 0.071, 0.071).
        # Shape dependence -- large for squeezed ISOSCELES, small for squeezed scalene or for
        # equal-but-short legs.
        # Theory-side primitive is the sharp tophat: it tiles the bins exactly, which the
        # box-limit sum rule requires, and it measured BETTER than the smooth spline basis
        # The output side is the noutsub bin average below, and it is not optional here: leaving
        # the angular factor at the bin's representative triangle is what noutsub exists to fix.
        if noutsub < 2:
            raise ValueError('the scoccimarro basis requires noutsub >= 2: its output-side angular '
                             'factor varies too fast across a bin to be taken at the bin centre')
        edges_out = np.asarray(bin.edges)                                   # (nout, 3, 2)
        # per-leg bounds, so the rule comes back mapped onto each bin: (nout, 3, noutsub)
        integ_out = integration(edges_out[..., :1], edges_out[..., 1:], size=noutsub)
        nodes_out, weights_out = np.asarray(integ_out.x()), np.asarray(integ_out.w)
        # every (i1, i2, i3) combination of the three legs' sub-nodes
        subindex = np.stack(np.meshgrid(*[np.arange(noutsub)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
        ksub_out = np.stack([nodes_out[:, leg, subindex[:, leg]] for leg in range(3)], axis=-1)   # (nout, nsub, 3)
        wsub_out = np.prod(np.stack([weights_out[:, leg, subindex[:, leg]] for leg in range(3)], axis=-1), axis=-1)
        # measure weight ~ k1^2 k2^2 k3^2 (the Theta triangle factor comes from compute_measure)
        wsub_out = wsub_out * np.prod(ksub_out**2, axis=-1)                      # (nout, nsub)
        nsub_out = subindex.shape[0]
        kout_sub = jnp.asarray(ksub_out.reshape(-1, 3))
        kout_weight = jnp.asarray(wsub_out)
        # `read` below is called with `to_spectrum.k`, a 2-tuple, so its `zip(k, kout)`
        # consumes the k1 and k2 columns ONLY -- the k3 sub-index never reaches the gather.
        # Of the noutsub^3 sub-points, only noutsub^2 are therefore distinct as far as the
        # interpolation is concerned, and gathering all of them repeats identical work
        # noutsub times. `subindex` orders the (k1, k2, k3) sub-indices with the k3 one fastest, so the i3 = 0 slice
        # ::noutsub enumerates the distinct (k1, k2) pairs. Measured end-to-end (scoccimarro,
        # wcoords=256, ellmax=2): 1.26x at noutsub=16 and ~1.03x at noutsub=8, the gain
        # growing with noutsub.
        kout_sub12 = jnp.asarray(ksub_out[:, ::noutsub, :2].reshape(-1, 2))   # (nout * noutsub^2, 2)
        nsub12 = noutsub**2

        # These weights are pure setup, but they run OUTSIDE jax.lax.map, i.e. eagerly, where
        # every primitive pays its own XLA compilation. The measure and shape factor together
        # are ~25 primitives, so left bare this costs ~25 compilations per term and the term
        # count grows as 4 * ellmax - 1. Under a single jit it is one compilation each instead.
        # Table-driven and computed for EVERY term at once, so this is a single compilation
        # rather than one per term (see harmonic_table above).
        @jax.jit
        def compute_out_weights(index1, index2, weight):
            triangle = kout_sub.T
            cos31, cos32 = cos31_cos32(triangle)
            shape = shape_from_tables(harmonic_table(cos31), harmonic_table(cos32), index1, index2, weight)
            numerator = kout_weight[None] * (compute_measure(triangle)[None] * shape).reshape(len(index1), -1, nsub_out)
            return numerator.reshape(len(index1), -1, nsub12, noutsub).sum(axis=-1)

        # The denominator is independent of BOTH the theory bin and the term, so it is built once
        # here. Left eager, unlike the weights above: it is a handful of primitives, not the ~25
        # per term that make the eager compilation worth avoiding.
        out_denominator = jnp.sum(kout_weight * compute_measure(kout_sub.T).reshape(-1, nsub_out), axis=-1)
        out_denominator = jnp.where(out_denominator == 0., 1., out_denominator)

        # Theory-side quadrature nodes on the k1', k2' legs, mirroring the k3' ones. The theory
        # bin's contribution is
        #   int_bin dk1' k1'^2 j_{l1'}(k1' r1) int_bin dk2' k2'^2 j_{l2'}(k2' r2) volume(k1', k2')
        # which for ONE bin is just the (bin-integrated) Bessel functions -- there is no need to
        # build a tophat on the full FFTlog k-grid and transform it. Evaluating it directly is
        # both far cheaper (`volume` costs ninsub^3 points instead of the whole n1 x n2 x ninsub
        # grid, of which the tophat discards all but a handful) and MORE accurate: the tophat
        # snaps the bin edges onto the log k-grid, so the integral it performs is over a
        # quantized bin. Measured against Gauss-Legendre quadrature, the snapped bin agrees to
        # ~0.1% but the TRUE bin differs by 2-20%, and refining the grid does not converge it
        # smoothly -- the effective bin width jitters with where the edges land.
        s_window = tuple(next(iter(window)).coords().values())

        # Spherical Bessel table, PRECOMPUTED over the distinct 1D theory bins.
        # Evaluating get_spherical_jn_all inside the per-bin map recomputed an IDENTICAL table for each of the
        # 4 * ellmax - 1 terms (it depends on the bin's nodes and the fixed r grid, not on the
        # term), and recomputed it per theory bin although the axis-d nodes depend only on that
        # axis's 1D bin -- of which there are ~n^(1/3) as many. Both redundancies are removed by
        # tabulating the distinct 1D bins up front and gathering (order, bin) below.
        bessel_index, bessel_table = [], []
        for leg in range(2):
            edges_1d, index_1d = np.unique(np.asarray(edgesin)[:, leg, :], axis=0, return_inverse=True)
            k_lo, k_hi = jnp.asarray(edges_1d[:, 0]), jnp.asarray(edges_1d[:, 1])
            k_nodes = k_lo[:, None] + 0.5 * (k_hi - k_lo)[:, None] * (nodes_in[None, :] + 1.)   # (nbin1d, nquad)
            # (ellmax + 1, nbin1d, nquad, n_r)
            bessel_table.append(spherical_jn_all(k_nodes[..., None] * s_window[leg][None, None, :]))
            bessel_index.append(jnp.asarray(index_1d.ravel()))

        def bessel_row(ell, table, index_bin):
            """j_ell at the bin's quadrature nodes, gathered from the precomputed table."""
            return table[ell, index_bin]

        def in_nodes(index_bin, leg, nodes_unit, weights_unit):
            """Gauss-Legendre nodes and weights on ``leg`` of theory bin ``index_bin``."""
            k_lo, k_hi = edgesin[index_bin, leg, 0], edgesin[index_bin, leg, 1]
            return k_lo + 0.5 * (k_hi - k_lo) * (nodes_unit + 1.), 0.5 * (k_hi - k_lo) * weights_unit

        # Theory-side angular factor, for ALL theory bins AND all terms at once. Chunked over
        # bins by lax.map: the harmonic table is (nharm, nquad, nquad, ninsub) per bin, so
        # materializing it for every bin at once would be ~12 GB at ellmax = 16, while a batch
        # of it is tens of MB. The result is (nbin, nkeys, nquad, nquad) -- 274 MB at 8511 bins,
        # 63 keys and ninsub = 8 -- and replaces one jitted graph per term with exactly one.
        @jax.jit
        def compute_in_volume(index1, index2, weight):
            def volume_one_bin(index_bin):
                # Gauss-Legendre nodes on all three theory legs. The k3' integral is the
                # measure factor (I ~ (k1 k2 k3)^-1 Theta Sigma is rapidly varying AND
                # discontinuous, so its point value is not the bin average -- that breaks
                # the box-limit sum rule by tens of per cent); the k1', k2' ones replace
                # the tophat, which quantized the bin onto the log k-grid.
                k1_nodes, _ = in_nodes(index_bin, 0, nodes_in, weights_in)
                k2_nodes, _ = in_nodes(index_bin, 1, nodes_in, weights_in)
                k3_nodes, k3_weights = in_nodes(index_bin, 2, nodes_in, weights_in)
                triangle = (k1_nodes[:, None, None], k2_nodes[None, :, None], k3_nodes[None, None, :])
                cos31, cos32 = cos31_cos32(triangle)
                shape = shape_from_tables(harmonic_table(cos31), harmonic_table(cos32), index1, index2, weight)
                measure = k3_weights * triangle[2]**2 / (2. * jnp.pi**2) * compute_measure(triangle)
                # (nkeys, nquad, nquad): the measure integrated over k3', at the k1', k2' nodes
                return jnp.sum(measure[None] * shape, axis=-1)

            return jax.lax.map(volume_one_bin, jnp.arange(edgesin.shape[0]), batch_size=batch_size)

        # ONE compilation for all 4 * ellmax - 1 TripoSH terms. Everything that varies between
        # terms -- the two FFTlog transforms, the window combination Qs, the output weights, the
        # theory volume and the multipole orders -- is passed as a TRACED argument. Captured as
        # closure constants instead (as this used to be), each term's FFTlog kernels are baked
        # into the graph as literals, so every term lowers to a different HLO module and
        # recompiles: at ellmax = 16 that is 63 compilations of a 2D-FFTlog scan body.
        # Note this relies on FFTlog treedefs comparing EQUAL across orders, i.e. on the value
        # equality of BaseFFTEngine -- with identity comparison jit retraces regardless.
        # `wain` is the only thing that must stay static -- it is an exponent of the separations --
        # so it is jit's static argument and jit's own cache holds one compilation per wide-angle
        # order. (Both angular factors, the only users of the static-order get_Ylm, are precomputed
        # above, so m_in no longer reaches the runner and the orders are always traced.)
        @partial(jax.jit, static_argnums=(0,))
        def run_term(wain, to_spectrum, Qs, out_weight, ell1_theory, ell2_theory, bessel_table1, bessel_table2, volume_in):

            # depends on (kout_sub12, to_spectrum.k) -- constant across theory bins, so the
            # searchsorted runs once per term here instead of once per (term, bin)
            read_index_sub = read_index(kout_sub12.T, to_spectrum.k)

            def convolve(index_bin):
                k1_nodes, k1_weights = in_nodes(index_bin, 0, nodes_in, weights_in)
                k2_nodes, k2_weights = in_nodes(index_bin, 1, nodes_in, weights_in)
                # Theory side: \int_{bin} k_3'^2 dk_3' / (2 pi^2) x I_000(k') Sigma^(3)_{l1' l2' L' M'},
                # precomputed for every bin by compute_in_volume above
                theory_weight = ((k1_weights * k1_nodes**2)[:, None] * (k2_weights * k2_nodes**2)[None, :]
                                 * volume_in[index_bin])
                # Per-axis FFTlog convention: (-i)^ell / (2 pi^2), which for a real input is
                # (-1)^(ell // 2) / (2 pi^2) at both parities (odd poles carry the imaginary part).
                phase1 = (1 - 2 * ((ell1_theory // 2) % 2)) / (2. * jnp.pi**2)
                phase2 = (1 - 2 * ((ell2_theory // 2) % 2)) / (2. * jnp.pi**2)
                bessel1 = phase1 * bessel_row(ell1_theory, bessel_table1, bessel_index[0][index_bin])   # (nquad, n_r1)
                bessel2 = phase2 * bessel_row(ell2_theory, bessel_table2, bessel_index[1][index_bin])   # (nquad, n_r2)
                # sum_ab theory_weight[a, b] bessel1[a, r] bessel2[b, s], as two small GEMMs rather
                # than a 2D FFTlog -- this is what the first transform used to do
                correlation = bessel1.T @ (theory_weight @ bessel2)
                correlation = correlation * Qs * s_window[0][:, None]**wain[0] * s_window[1][None, :]**wain[1]
                # Estimator side: B_L(k1, k2, k3) = sum_{ell_1 ell_2} S^(3)_{ell_1 ell_2 L} B_{ell_1 ell_2 L}(k1, k2).
                # The full 3-D output-bin average: all three legs sub-binned, weighted by
                # k1^2 k2^2 k3^2 and the I_000 triangle measure. The (ell_1, ell_2, L)
                # dependence is already folded into out_weight, hence no shape factor here.
                # bin average = sum_sub w k^2 I_000 S^(3) read  /  sum_sub w k^2 I_000,
                # with the gather done on the distinct (k1, k2) sub-points only
                read_sub = read_apply(read_index_sub, to_spectrum(correlation)[1])
                numerator = jnp.sum(out_weight * read_sub.reshape(-1, nsub12), axis=-1)
                spectrum = numerator / out_denominator
                # compute_measure(kout) = pi^2/(k1 k2 k3) * Theta vanishes on output
                # bins whose representative triangle violates the triangle
                # inequality, so this division is 0/0 there and leaves NaN in
                # those rows.
                # Harmless in itself -- those bins are unphysical -- but a NaN
                # in the matrix poisons any downstream dot product for a caller
                # who does not mask, so return 0 as the grid branch does.
                return jnp.nan_to_num(spectrum)

            return jax.lax.map(convolve, jnp.arange(edgesin.shape[0]), batch_size=batch_size).T

        # Count the SURVIVING terms up front (a pure-Python label lookup, no device work), so the bar
        # measures what is actually run: of the (ellmax + 1)^2 terms per block only ~4 * ellmax - 1
        # read a multipole `window` carries, the rest contributing exactly zero.
        progress = make_pbar(sum(any(has_w_rect(q, wain_key) for q, _ in wcoeffs_key)
                                 for ellin_key, wain_key in ellsin for ell_key in ells
                                 for _, _, wcoeffs_key in get_scoccimarro_window_convolution_coeffs(ell_key, ellin_key, ellmax=ellmax)))

        # Enumerate the DISTINCT angular keys over every surviving term, so both tables are built
        # once for the whole matrix instead of once per term. Many terms share a key: at ellmax = 16
        # the ellwmax = 2 window has 827 surviving terms but only 63 distinct theory-side keys.
        keys_out, keys_in = {}, {}
        for ellin_key, wain_key in ellsin:
            m_in_key = ellin_key[1] if isinstance(ellin_key, tuple) else 0
            for ell_key in ells:
                coeffs_key = get_scoccimarro_window_convolution_coeffs(ell_key, ellin_key, ellmax=ellmax)
                for ell_out_key, ell_theory_key, wcoeffs_key in coeffs_key:
                    if not any(has_w_rect(q, wain_key) for q, _ in wcoeffs_key): continue
                    keys_out.setdefault(ell_out_key, get_scoccimarro_los_coeffs(ell_out_key))
                    # m = M', the third 3j slot: the mu = 0 term of Sigma^(3) is exactly the old
                    # 3j (l1', l2', L'; 0, -M', M') times y_{l2'}^{-M'}
                    keys_in.setdefault((ell_theory_key, m_in_key),
                                       get_scoccimarro_los_coeffs(ell_theory_key, m=m_in_key, normalize=False))
        index_of_key_out = {k: i for i, k in enumerate(keys_out)}
        index_of_key_in = {k: i for i, k in enumerate(keys_in)}
        out_weights_all = compute_out_weights(*pack_shape_keys(list(keys_out.items())))
        volume_in_all = compute_in_volume(*pack_shape_keys([(ell, coeffs) for (ell, m), coeffs in keys_in.items()]))

        for ellin, wain in ellsin:  # ellin = L' or (L', M'), wain wide-angle order
            wmat_tmp[ellin, wain] = []
            m_in = ellin[1] if isinstance(ellin, tuple) else 0  # M'
            for ell in ells:  # ell = L

                # Then sum over \ell_1, \ell_2, \ell_1', \ell_2', \ell_1'', \ell_2'', L''
                block = jnp.zeros(shape=(len(kout), len(edgesin)))

                coeffs = get_scoccimarro_window_convolution_coeffs(ell, ellin, ellmax=ellmax)
                for sugiyama_ell_out, sugiyama_ell_theory, wcoeffs in coeffs:
                    # Skip terms whose window multipoles are ALL absent from `window`: they
                    # contribute exactly nothing -- yet each would otherwise pay a full pass per
                    # theory bin. This is the bulk of the cost at low ellwmax: the term count grows
                    # as (ellmax + 1)^2 per (ell <- ellin) block while only ~(ellmax + 1) terms
                    # have a non-zero window, so 67% (ellmax=2) to 89% (ellmax=8) of the work is
                    # wasted. Absent multipoles are already treated as zero downstream, so this
                    # changes no result -- only the runtime, from O(ellmax^2) to O(ellmax).
                    # The SAME predicate feeds the progress bar and the key enumeration above, so
                    # the three cannot disagree about which terms exist.
                    if not any(has_w_rect(q, wain) for q, _ in wcoeffs): continue
                    Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)
                    # fftlog
                    # lowring=False, xy=1: align the two transforms' grids EXACTLY. With the
                    # default lowring=True each Bessel kernel picks its own low-ringing offset
                    # lnxy, so whenever the forward (theory, sugiyama_ell_theory) and backward
                    # (output, sugiyama_ell_out) orders differ -- most terms here -- the forward
                    # transform's output grid is shifted by up to one spacing from the grid the
                    # window is multiplied on.
                    to_spectrum = CorrelationToSpectrum(s=s_window, ell=sugiyama_ell_out, check_level=1,
                                                        minfolds=0, lowring=False, xy=1.)

                    # Both angular factors depend on the term but NOT on the FFTlog chain, so they
                    # were built for every key at once above; this term just reads its row. The
                    # output weight is summed over the k3 sub-axis there, which is what lets the
                    # gather run on the (k1, k2) sub-grid alone: sum_{i3} w I is contracted against
                    # a read that is constant along i3.
                    out_weight = out_weights_all[index_of_key_out[sugiyama_ell_out]]
                    volume_in = volume_in_all[:, index_of_key_in[sugiyama_ell_theory, m_in]]
                    # always traced now, so every term reuses the same compilation
                    ell1_theory, ell2_theory = jnp.asarray(sugiyama_ell_theory[0]), jnp.asarray(sugiyama_ell_theory[1])
                    term = run_term(wain, to_spectrum, Qs, out_weight, ell1_theory, ell2_theory,
                                    bessel_table[0], bessel_table[1], volume_in)
                    if pbar: jax.block_until_ready(term)  # otherwise the bar races ahead of the device
                    block += term
                    progress.update(1)

                wmat_tmp[ellin, wain].append(block)

            wmat_tmp[ellin, wain] = jnp.concatenate(wmat_tmp[ellin, wain], axis=0)
        progress.close()
    else:

        def get_passes(ellin, wain, ell):
            # Theory multipoles contributing to the observed `ell`: `ellin` itself, plus its
            # k1 <-> k2 swap wherever that swap is not already a requested theory multipole.
            # Returned as (theory ell, swap, coeffs), coeff-less pairs dropped -- so the progress
            # bar below counts exactly the passes the loop then runs.
            ellin_swap = tuple(ellin[1::-1]) + ellin[2:]
            toret = [(ellin, False)]
            if ellin[1] != ellin[0] and (ellin_swap, wain) not in ellsin:
                toret.append((ellin_swap, True))
            return [(ellt, swap, coeffs) for ellt, swap in toret
                    if (coeffs := get_sugiyama_window_convolution_coeffs(ell, ellt))]

        passes = {(ellin, wain, ell): get_passes(ellin, wain, ell) for ellin, wain in ellsin for ell in ells}
        progress = make_pbar(sum(map(len, passes.values())))

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
                # lowring=False, xy=1: align the transforms' grids exactly -- see the scoccimarro
                # branch above. Every ellwmax > 0 coupling here transforms forward (ellin) and
                # back (ell) at different orders, and with per-kernel low-ringing offsets the
                # window would be multiplied on a grid shifted by up to one spacing from where
                # the correlation actually lives.
                to_spectrum = CorrelationToSpectrum(s=tuple(next(iter(window)).coords().values()), ell=ell, check_level=1, minfolds=0, lowring=False, xy=1.)
                # ninsub / noutsub apply here too, via the sub-node construction in
                # axis_basis_matrices. Their scoccimarro-side machinery does NOT carry over
                # literally: the sugiyama multipoles B_{l1 l2 L}(k1, k2) bin only two legs, so
                # there is no third-leg measure for ninsub to integrate and no rapidly varying
                # L_{ell2}(cos theta12) for noutsub to bin-average -- and because the output
                # measure k1^2 k2^2 then factorizes, the separable `rebin` already IS the exact
                # 2-D output-bin average.
                index_in, Min_axes = axis_basis_matrices(edgesin, to_spectrum.k, kind='spline', nsub=ninsub)
                index_in_swap = index_in[:, ::-1]
                index_out, Mout_axes = axis_basis_matrices(bin.edges, to_spectrum.k, kind='rebin', nsub=noutsub)

                for ellt, swap, wcoeffs in passes[ellin, wain, ell]:
                    Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)
                    to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=ellt, minfolds=0, lowring=False, xy=1.)
                    term = jax.lax.map(partial(convolve, swap=swap), jnp.arange(edgesin.shape[0]), batch_size=batch_size).T
                    if pbar: jax.block_until_ready(term)
                    tmp += term
                    progress.update(1)

                wmat_tmp[ellin, wain].append(tmp)

            wmat_tmp[ellin, wain] = jnp.concatenate(wmat_tmp[ellin, wain], axis=0)
        progress.close()

    wmat = jnp.concatenate(list(wmat_tmp.values()), axis=1)

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