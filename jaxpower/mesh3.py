import numbers
import itertools
from math import factorial
import operator
import functools
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from dataclasses import dataclass

from .mesh import (BaseMeshField, MeshAttrs, RealMeshField, ComplexMeshField, ParticleField, FKPField, staticarray, get_sharding_mesh, _get_hermitian_weights, _find_unique_edges, _get_bin_attrs, _bincount,
compute_normalization, compute_box_normalization, split_particles, __format_meshes, _gather_mpi, get_mpicomm)
from .mesh2 import _get_los_vector
from .types import Mesh3SpectrumPole, Mesh3SpectrumPoles, Mesh3CorrelationPole, Mesh3CorrelationPoles, ObservableLeaf, ObservableTree, WindowMatrix
from .utils import real_gaunt, get_legendre, get_spherical_jn, get_Ylm, get_Ylm_all, wigner_3j, wigner_9j, register_pytree_dataclass


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
            wmodes = _get_hermitian_weights(vec, mattrs.meshsize, sharding_mesh=None)
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


def _scoccimarro_bin_legendre_mu(edges, ell, nsub=12):
    """< L_ell(mu_a3) > over each bin (edges: (nbin, 3, 2)), a = 1, 2, mu_a3 = k_a^ . k3^ from the side lengths; measure
    k1 k2 k3 Theta, nsub^3 Gauss-Legendre nodes. Shape (nbin, 2)."""
    x, w = np.polynomial.legendre.leggauss(nsub)
    frac, w = jnp.asarray((x + 1.) / 2.), jnp.asarray(w / 2.)
    grid = np.stack(np.meshgrid(*[np.arange(nsub)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
    k = edges[:, None, :, 0] + (edges[:, None, :, 1] - edges[:, None, :, 0]) * frac[grid][None]
    mu13 = (k[..., 1]**2 - k[..., 0]**2 - k[..., 2]**2) / (2. * k[..., 0] * k[..., 2])
    mu23 = (k[..., 0]**2 - k[..., 1]**2 - k[..., 2]**2) / (2. * k[..., 1] * k[..., 2])
    weight = jnp.prod(w[grid], axis=-1)[None] * jnp.prod(k, axis=-1) * (jnp.abs(mu13) <= 1.)
    norm = jnp.sum(weight, axis=-1)
    norm = jnp.where(norm > 0., norm, 1.)
    legendre = get_legendre(ell)
    mu13, mu23 = jnp.clip(mu13, -1., 1.), jnp.clip(mu23, -1., 1.)
    return jnp.stack([jnp.sum(weight * legendre(mu13), axis=-1), jnp.sum(weight * legendre(mu23), axis=-1)], axis=-1) / norm[:, None]


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

            # Second line of eq. 58: legs (a, 3), a = 1, 2. The pair coincidence leaves < d w2* > on the k_a shell with the
            # Legendre still on k3^; rotating the triangle about k_a at fixed side lengths, k3^ sweeps a cone, and the
            # addition theorem gives < d w2* L_ell(k^ . n) >_{k_a shell} x L_ell(mu_a3), mu_a3 = k_a^ . k3^ fixed by the
            # side lengths, averaged over the bin with the triangle measure. (The former factorization,
            # sum_m < . Y_lm >_{k_a shell} < Y_lm >_{k3 shell}, averaged k3^ over its whole shell independently of k_a^:
            # ~0 for ell > 0, so B_2 was under-subtracted by ~2/3 of its shot noise, most for folded triangles.)
            # Verified for a global line of sight (density test on galaxy sub-cubes); the local branch follows the same
            # argument with the leg's harmonics but is not separately tested.
            if vlos is not None:
                cmeshw3_ell = cmeshw * apply_fourier_legendre(ell, cmeshw2).conj()
            else:
                cmeshw3_ell = cmeshw * apply_fourier_harmonics(ell, cmeshw2.c2r()).conj()
            lmu = _scoccimarro_bin_legendre_mu(bin.edges, ell)
            sn_ell = sn_ell + sum(bin_mesh2(cmeshw3_ell, axis=axis)[bin._iedges[..., axis]] * lmu[:, axis] for axis in range(mattrs.ndim - 1))
            shotnoise[ill] = (2 * ell + 1) * sn_ell

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

    # Take the meshes as _format_meshes returned them. It has already resolved every repeated
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


def get_smooth3_window_bin_attrs(ells, ellsin=3, fields=None, return_ellsin: bool=False, basis: str='sugiyama', ellmax: int=4):
    r"""
    Window multipoles to measure for an output binning (sugiyama or scoccimarro basis).

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
        For the scoccimarro basis: the largest window multipole leg :math:`\ell_1, \ell_2`. All the window multipoles the
        quadrature window uses are returned: :math:`\ell_1 + \ell_2` even, :math:`L` even with
        :math:`|\ell_1 - \ell_2| \leq L \leq \ell_1 + \ell_2`, and :math:`L \leq \max L_{\rm out} + \max L'` (the line-of-sight
        selection rule); theory multipoles are the even :math:`L' \leq 2\,` ``ellsin`` if an integer.

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
            ellsin = list(range(0, 2 * ellsin + 1, 2))
        non_zero_ellsin = [ellin for ellin in ellsin if (ellin[0] if np.ndim(ellin) else ellin) % 2 == 0]
        Lmax = max(ells) + max(ellin[0] if np.ndim(ellin) else ellin for ellin in non_zero_ellsin)
        ellw = [(ell1, ell2, L) for ell1 in range(ellmax + 1) for ell2 in range(ell1 if fields[1] == fields[0] else 0, ellmax + 1)
                if (ell1 + ell2) % 2 == 0 for L in range(abs(ell1 - ell2), min(ell1 + ell2, Lmax) + 1, 2)]
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
    Matrix :math:`S` scattering an ordered theory vector onto the unordered
    :math:`(k_1', k_2', k_3')` grid that :func:`compute_smooth3_spectrum_window` integrates
    over: ``theory_unordered = S @ theory_ordered``.

    The estimator bins ordered triangles :math:`k_1' \leq k_2' \leq k_3'`, but the window
    convolution runs over all of :math:`k'`-space, so the matrix must be given the unordered
    grid (pass ``edgesin`` as raw per-axis edges and it builds one). Filling that grid requires
    knowing :math:`B_{L'}` at each ordering, which is a theory-side question and is only a
    relabelling when the multipole is invariant under the permutation.

    Validity, per multipole, if the three :math:`\delta`-fields are the same:

    - ``fix_legs=(2,)``: exact for every :math:`L'`.
      The scoccimarro :math:`B_{L'}` is referred to :math:`\hat{k}_3 \cdot \hat{z}` (the estimator applies the output Legendre to the
      third leg), and swapping :math:`k_1 \leftrightarrow k_2` leaves that leg alone, so
      :math:`B_{L'}(k_1,k_2,k_3) = B_{L'}(k_2,k_1,k_3)` identically.
    - ``fix_legs=None`` (all permutations): exact for :math:`L' = 0` only, since :math:`B_0` is
      an orientation average of a rigid triangle and therefore symmetric. For :math:`L' > 0`
      a reordering moves the line-of-sight reference to a different leg, mixing :math:`L` and
      involving the second angular coordinate this basis drops -- so it is something other than a relabelling,
      and the theory must supply those entries itself.

    Rows with no admissible permutation are left at zero, i.e. that part of :math:`k'`-space
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
        # dedupe on the box when edges are known (degenerate legs give identical boxes),
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
    # column of each group, -1 where the group has none; reversed so the first ordered row wins
    column = np.full(len(groups), -1)
    column[inv_ordered[::-1]] = np.arange(len(kin_ordered))[::-1]
    # admissible only if sorting leaves every fixed leg where it already is
    admissible = np.all(np.abs(kin[:, fix] - sorted_kin[:, fix]) <= atol, axis=-1)
    S = np.zeros((len(kin), len(kin_ordered)))
    rows = np.flatnonzero(admissible & (column[inv_kin] >= 0))
    S[rows, column[inv_kin][rows]] = 1.
    return S, kin, kin_ordered, edges


def compute_smooth3_spectrum_window(window, edgesin: np.ndarray | tuple, ellsin: tuple=None, bin: BinMesh3SpectrumPoles=None,
                                    batch_size: int=None, pbar: bool=False, rows=None, return_identity: bool=False) -> WindowMatrix:
    r"""
    Compute the "smooth" (no binning effect) bispectrum window matrix.

    Sugiyama basis: the harmonic (TripoSH) convolution through 2D FFTlogs. Scoccimarro basis: a deterministic quadrature
    in the kernel variables :math:`\mathbf{q}_i = \mathbf{k}_i - \mathbf{k}_i^\prime` with every measured window multipole exact
    (desi-gqc-notes, "Scoccimarro basis"); output rows are split over the jax processes (run one per GPU, e.g.
    ``srun -n 4 --gpus-per-node=4`` with ``jax.distributed.initialize()``) and gathered over MPI.

    The accuracy settings are fixed at their converged values. Sugiyama basis: the theory basis is built from 16
    Gauss-Legendre sub-nodes per theory bin per axis, and the output bins are averaged with 8. Scoccimarro basis: the window
    is integrated on tensor cubic B-splines over the ``edgesin`` cells (on 0.02 cells ~10x better than trilinear hats, at the
    same cost), the model projected onto them (:math:`L^2`, over the triangle region) and the projection folded into the
    returned window; the quadrature rules are :data:`default_scoccimarro_rules_iso` for :math:`Q_{000}` and
    :data:`default_scoccimarro_rules_aniso` for the others, and the kernel grid is set from the window size and the theory
    step (:func:`_get_scoccimarro_qgrid`; for LRG1, cells of :math:`2\times10^{-5}` to :math:`2\times10^{-3}\,h\,\mathrm{Mpc}^{-1}`,
    35 response nodes from steps of :math:`0.0025` to :math:`0.08`, to :math:`q \simeq 0.96`).

    Parameters
    ----------
    window : ObservableTree
        Configuration-space window multipoles :math:`Q_{\ell_1\ell_2L}(s_1, s_2)`, sugiyama basis. For the scoccimarro basis
        they must include :math:`Q_{000}`; a missing :math:`(\ell_2, \ell_1, L)` is the stored :math:`(\ell_1, \ell_2, L)`
        transposed; wide-angle orders are not supported.
    edgesin : np.ndarray | tuple
        Theory bin edges. Scoccimarro basis: one 1D array (or (nbin, 2)), contiguous, the same for the three legs; the
        widths may vary (e.g. coarser at high k), the cubic basis then being only continuous where they change.
    ellsin : tuple, optional
        Theory multipole orders; optional when ``edgesin`` is an ``ObservableTree``. Scoccimarro basis (default (0, 2, 4)):
        an integer :math:`L'` is the Legendre multipole; a tuple :math:`(L', M')`, :math:`M' \geq 1`, the response to
        :math:`\mathrm{Re}\, Y_{L'M'}` of the line of sight in the theory triangle's frame (polar axis :math:`\hat{k}_3'`,
        :math:`\mathbf{k}_1'` in the :math:`x'z'` half-plane), so that a model's :math:`M' \neq 0` part enters as
        :math:`2 W_{(L', M')} B_{L'M'}`.
    bin : BinMesh3SpectrumPoles
        Output binning.
    batch_size : int, optional
        Sugiyama basis: number of theory bins whose 2D FFTlog pair is evaluated in parallel (``None``: the largest batch
        whose ``(batch_size, n1, n2)`` complex intermediate stays within 1 GB, capped at 64). Scoccimarro basis: integration
        points per output-bin point per compiled call (``None``: 1.5e5; 3e4 for :math:`M' \neq 0` columns on 40 GB cards).
        Lower it on out-of-memory; results do not depend on it.
    pbar : bool, default=False
        Show progress (sugiyama: over the terms of the multipole sums; scoccimarro: over the output rows).
    rows : array, optional
        Scoccimarro basis: output rows to compute (others are zero); default all.
    return_identity : bool, default=False
        Scoccimarro basis: also return :math:`I`, the window matrix of an infinite uniform survey (:math:`Q_{000} = 1`, the
        others zero, so :math:`k' = k`): each row is the output-bin average of the theory basis functions, with the same
        output-bin rule as the window. For predictions :math:`B_{\rm box} + (W - I)\, t` and identity-level tests.

    Returns
    -------
    wmat : WindowMatrix
        Not normalized by the measured ``norm``. Scoccimarro basis: the theory of each multipole is the model's values at
        the window's theory coordinates ``(npoint, 3)``, points :math:`(k_1, k_2, k_3)` that all close a triangle (2
        Gauss-Legendre points per leg in the theory cells the triangle condition does not cut, 3 in those it cuts: ~71k for
        0.02 cells to 0.48, within 0.01 sigma of one mock of 5 everywhere on the LRG1 window): evaluate the model there and
        multiply.
    identity : WindowMatrix, if ``return_identity`` (scoccimarro basis).
    """
    if 'scoccimarro' in bin.basis:
        value, identity, kin, kin_edges, ellsin, wa_orders = _compute_smooth3_spectrum_window_scoccimarro(
            window, edgesin, ellsin=(0, 2, 4) if ellsin is None else ellsin, bin=bin,
            rows=rows, return_identity=return_identity, batch_size=1.5e5 if batch_size is None else batch_size, pbar=pbar)
    else:
        value, identity, kin, kin_edges, ellsin, wa_orders = _compute_smooth3_spectrum_window_sugiyama(
            window, edgesin, ellsin=ellsin, bin=bin, batch_size=batch_size, pbar=pbar)
    observable = Mesh3SpectrumPoles([Mesh3SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes[ill], num_raw=jnp.zeros_like(bin.xavg[..., 0]),
                                                       basis=bin.basis, ell=ell) for ill, ell in enumerate(bin.ells)])
    kin = jnp.asarray(kin)
    theory = ObservableTree([Mesh3SpectrumPole(k=kin, k_edges=kin_edges, num_raw=jnp.zeros_like(kin[..., 0]), basis='sugiyama') for _ in ellsin],
                            ells=ellsin, wa_orders=wa_orders)
    wmat = WindowMatrix(observable=observable, theory=theory, value=jnp.asarray(value))
    if identity is not None:
        return wmat, wmat.clone(value=jnp.asarray(identity))
    return wmat


def _compute_smooth3_spectrum_window_scoccimarro(window, edgesin, ellsin=(0, 2, 4), bin=None, rows=None, return_identity=False,
                                                batch_size=1.5e5, pbar=False):
    """
    Scoccimarro window by deterministic quadrature; arguments as :func:`compute_smooth3_spectrum_window`. Returns (window
    matrix (nrow, len(ellsin) x npoint), identity matrix or None, theory points (npoint, 3), None (no theory edges), theory
    multipoles, wide-angle orders).
    """
    if 'wa_orders' in window.labels(return_type='keys') and any(np.any(np.asarray(wa)) for wa in window.wa_orders):
        raise NotImplementedError('wide-angle window orders are not supported')
    ells, ellsin = list(bin.ells), [ell if np.ndim(ell) else int(ell) for ell in ellsin]
    columns = [(int(ell[0]), int(ell[1]), False) if np.ndim(ell) else (ell, 0, True) for ell in ellsin]   # theory columns (L', M', Legendre?)

    # 1. Theory basis over the theory cells (product bins of edgesin in (k1', k2', k3') that contain a closed triangle)
    basis = _get_scoccimarro_theory_basis(edgesin[0] if isinstance(edgesin, (tuple, list)) else edgesin, basis='cubic')
    nnode = len(basis['knodes'])

    # 2. Window multipoles: the measured ones, plus the transpose (l2, l1, L) of any not measured
    s = np.asarray(next(iter(window)).coords()['s1'])
    parts = []  # (l1, l2, L, Q)
    for ell in window.ells:
        Q = np.asarray(window.get(tuple(ell)).value()).real
        parts.append(tuple(ell) + (Q,))
        if ell[0] != ell[1] and (ell[1], ell[0], ell[2]) not in [tuple(e) for e in window.ells]:
            parts.append((ell[1], ell[0], ell[2], Q.T))
    if not any(part[:3] == (0, 0, 0) for part in parts): raise ValueError('window must contain Q000')
    # quadrature rules per multipole
    def get_rules(ell):
        return default_scoccimarro_rules_iso if ell == (0, 0, 0) else default_scoccimarro_rules_aniso

    passes = {}  # multipoles with the same rules are integrated in one pass (one set of compiled kernels)
    for part in parts:
        passes.setdefault(tuple(sorted(get_rules(part[:3]).items())), []).append(part)

    # 3. Kernel magnitudes, from the window extent and the theory grid
    Q000 = next(part[3] for part in parts if part[:3] == (0, 0, 0))
    qgrid = _get_scoccimarro_qgrid(s, Q000, basis, kmax=np.max(bin.edges))
    rows_fn = [_get_scoccimarro_window_rows(group, s, qgrid, ells, columns, basis, dict(key), batch_size=batch_size) for key, group in passes.items()]

    # 4. Rows, split over the jax processes. The identity row is the q = 0 limit: the basis averaged over the output bin
    #    (Legendre L' = L block only)
    edges = [(ell, edge) for ell in ells for edge in np.asarray(bin.edges)]
    selected = np.arange(len(edges)) if rows is None else np.asarray(rows)
    mine = selected[jax.process_index()::jax.process_count()]
    value = np.zeros((len(mine), len(ellsin) * nnode))   # columns: theory multipole, then node
    identity = np.zeros_like(value) if return_identity else None
    rules_identity, inodes, shape = get_rules((0, 0, 0)), basis['inodes'], basis['shape']
    if pbar and jax.process_index() == 0:
        from tqdm import tqdm
        mine_iter = tqdm(mine, desc='window rows')
    else:
        mine_iter = mine
    for irow, row in enumerate(mine_iter):
        ell, edge = edges[row]
        value[irow] = sum(fn(edge, ell) for fn in rows_fn).T.ravel()
        if return_identity and ell in ellsin:
            points, weights = _get_scoccimarro_bin_rule(edge, rules_identity['nbin'], rules_identity['ntet'])
            box = np.einsum('p,px,py,pz->xyz', weights, *[np.asarray(_get_scoccimarro_basis_weights(jnp.asarray(points[:, axis]), basis, shape[axis])) for axis in range(3)])
            identity[irow, ellsin.index(ell) * nnode:(ellsin.index(ell) + 1) * nnode] = box.ravel()[(inodes[:, 0] * shape[1] + inodes[:, 1]) * shape[2] + inodes[:, 2]]

    # 5. Every process's rows, on every process: MPI Gatherv in host memory (variable row counts); rows nobody computed are zero
    local = [value] + ([identity] if return_identity else [])
    if jax.process_count() > 1:
        mpicomm = get_mpicomm()
        if mpicomm is None or mpicomm.size != jax.process_count():
            raise RuntimeError('gathering window rows needs one MPI rank per jax process (launch with srun / mpirun)')
        mine, *local = (_gather_mpi(array, mpiroot=None, mpicomm=mpicomm) for array in [np.asarray(mine, dtype='i8')] + local)
    matrices = [np.zeros((len(edges), array.shape[1])) for array in local]
    for full, array in zip(matrices, local): full[mine] = array

    # 6. Theory in terms of (k1, k2, k3): fold the projection of the model onto the basis into the matrices (each
    #    (nrow, len(ellsin) x nnode), columns by theory multipole, then node). The projection is the least-squares fit over the
    #    theory cells with the measure k1 k2 k3 Theta: G t = Phi^T w B, Phi the basis at the evaluation points, w their
    #    quadrature weights, G = Phi^T w Phi, with 2 Gauss-Legendre points per leg in the cells the triangle condition does
    #    not cut, and 3 in those it cuts (fewer points there misses by up to 1 sigma of one mock on the LRG window).
    import scipy.linalg
    import scipy.sparse
    cells, knodes = basis['cells'], basis['knodes']
    # evaluation points and weights, cell by cell
    corners = np.stack([cells[:, axis, code] for axis, code in enumerate(np.array(list(itertools.product((0, 1), repeat=3))).T)], axis=-1)   # (ncell, 8, 3)
    cut = ~np.all(_scoccimarro_closes_triangle(*np.moveaxis(corners, -1, 0)), axis=-1)
    kpoints, weights = [], []
    for n, mask in zip((2, 3), [~cut, cut]):
        x, wx = np.polynomial.legendre.leggauss(n)
        sub = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
        lo, width = cells[mask][..., 0], cells[mask][..., 1] - cells[mask][..., 0]
        k = lo[:, None, :] + width[:, None, :] * ((x + 1.) / 2.)[sub][None]
        weight = np.prod(width[:, None, :] / 2. * wx[sub][None], axis=-1) * np.prod(k, axis=-1)
        weight *= np.abs((k[..., 2]**2 - k[..., 0]**2 - k[..., 1]**2) / (2. * k[..., 0] * k[..., 1])) < 1.
        kpoints.append(k[weight > 0.]); weights.append(weight[weight > 0.])
    kpoints, weights = np.concatenate(kpoints), np.concatenate(weights)
    # sparse basis matrix Phi (npoint, nnode): the tensor products of the per-leg functions not vanishing at each point
    index, values = _get_scoccimarro_basis_weights(kpoints, basis)              # (npoint, 3, order + 1)
    a, b, c = (i.ravel() for i in np.meshgrid(*[np.arange(index.shape[-1])] * 3, indexing='ij'))
    lookup = np.full(np.prod(shape), -1)
    lookup[(inodes[:, 0] * shape[1] + inodes[:, 1]) * shape[2] + inodes[:, 2]] = np.arange(len(knodes))
    col = lookup[(index[:, 0, a] * shape[1] + index[:, 1, b]) * shape[2] + index[:, 2, c]]
    phivalue = values[:, 0, a] * values[:, 1, b] * values[:, 2, c]
    if np.any((col < 0) & (phivalue != 0.)): raise ValueError('a projection point reads a node outside the basis')
    col = np.maximum(col, 0)   # zero entries of the node windows
    phi = scipy.sparse.csr_matrix((phivalue.ravel(), (np.repeat(np.arange(len(kpoints)), col.shape[1]), col.ravel())), shape=(len(kpoints), len(knodes)))
    wphi = (scipy.sparse.diags(weights) @ phi).tocsr()
    # Gram matrix on the nodes with support; the B-splines that only graze the region carry almost no measure: small ridge
    gram = (phi.T @ wphi).toarray()
    supported = np.diag(gram) > 1e-14 * np.diag(gram).max()
    gram = gram[np.ix_(supported, supported)]
    cho = scipy.linalg.cholesky(gram + 1e-12 * np.diag(np.diag(gram)), lower=True)
    wphi = wphi[:, np.flatnonzero(supported)].tocsr()
    # W P = (W G^-1) Phi^T w per theory multipole: one Cholesky solve and a sparse product, P never formed
    value, *identity = [np.concatenate([np.asarray(wphi @ scipy.linalg.cho_solve((cho, True), block[:, supported].T)).T
                                        for block in np.split(matrix, len(ellsin), axis=1)], axis=1) for matrix in matrices]
    return value, identity[0] if return_identity else None, kpoints, None, list(ellsin), [(0, 0)] * len(ellsin)


def _compute_smooth3_spectrum_window_sugiyama(window, edgesin, ellsin=None, bin=None, batch_size=None, pbar=False):
    """Sugiyama window through the harmonic (TripoSH) convolution: (matrix, None, theory k, theory k edges, theory ells, wide-angle orders)."""
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

    ellsin = [(ellin[0], tuple(ellin[1])) if isinstance(ellin[0], tuple) else (ellin, (0, 0)) for ellin in ellsin]

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
        for each axis independently, a small matrix of shape (n_k_axis,
        n_unique_axis) (kind='spline', input/theory side: a smooth spline basis
        function per distinct bin center, matrix_spline_interp) or
        (n_unique_axis, n_k_axis) ('rebin', output side: a proper k^2-weighted
        bin average, matrix_rebin), keyed by the axis's distinct bin
        centers/edges -- never the full (nbins1 x nbins2, n_k1 x n_k2) dense
        tensor (edges may be a masked/paired list, e.g. sugiyama-diagonal's
        k1=k2, not a full product grid; this stays correct and small either
        way). Returns (index_per_bin (nbins, ndim), matrices list).

        ``nsub`` > 1 replaces each bin's single representative midpoint by
        ``nsub`` Gauss-Legendre sub-nodes spanning the bin, so the basis
        represents the bin rather than its centre (``nsub = 1`` puts the single
        node at the centre and reproduces the midpoint construction exactly).
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
                # Sub-node construction, shared by both kinds. 'spline' sums the sub-node
                # interpolation weights and never averages them: the sum over all bins of a
                # partition-of-unity interpolant must stay 1 at every k. As nsub
                # grows the summed basis tends to the sharp tophat indicator that tiles the bins
                # exactly, so nsub interpolates monotonically between the midpoint spline (nsub = 1) and that
                # tophat. 'rebin' instead averages, weighted by the k^2 measure, matching
                # matrix_rebin's definition.
                from .pt import integration
                # bounds are per unique bin, so the nodes come out (n_unique, nsub) directly,
                # with weights already carrying the bin width
                integ_sub = integration(unique_edges[:, :1], unique_edges[:, 1:], size=nsub)
                ksub, wsub_bin = np.asarray(integ_sub.x()), np.asarray(integ_sub.w)
                if kind == 'spline':
                    # theory known at the sub-nodes, evaluated on the fftlog grid
                    Msub = matrix_spline_interp(jnp.asarray(ksub.ravel()), kk, interp_order=interp_order)
                    M = jnp.sum(Msub.reshape(Msub.shape[0], *ksub.shape), axis=-1)       # (n_k, n_unique)
                    _lo, _hi = unique_edges.min(), unique_edges.max()
                    M = M * ((kk >= _lo) & (kk <= _hi))[:, None]
                else:
                    # opposite direction: the transformed spectrum is known on the fftlog grid and
                    # is read at the sub-nodes, then averaged over the bin with the k^2 measure
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

            # Odd-leg phase. The FFTlog transforms are per axis, with the 2-point convention that
            # an odd pole carries its imaginary part: phase (-1)^(ell // 2) on each leg. A sugiyama
            # multipole with two odd legs (ell1 + ell2 is even) is real, so its phase
            # (-i)^(ell1 + ell2) = (-1)^((ell1 + ell2) / 2) is the per-leg product times -1, on the
            # way in (theory) and on the way out (observable). Without it every odd-odd theory
            # multipole, (1,1,L), (1,3,L), ..., entered with the wrong sign, and every odd-odd
            # observable came out with it: B202 was 3-7% high on a 1000 Mpc/h cube for the terms
            # carrying the free-leg power P(k3), and B110 had the wrong sign. Checked against a
            # brute-force, multipole-free convolution (claude_window_b202_ellwmax/brute_window.py).
            sign_out = -1. if ell[0] % 2 else 1.

            def convolve(idx, swap=False, sign_in=1.):
                idx_axes = (index_in_swap if swap else index_in)[idx]
                theory = Min_axes[0][:, idx_axes[0]][:, None] * Min_axes[1][:, idx_axes[1]][None, :]
                correlation = sign_in * to_correlation(theory)[1]
                correlation = correlation * Qs * to_correlation.s[0][:, None]**wain[0] * to_correlation.s[1][None, :]**wain[1]
                # Separable output rebin: Mout_axes[d] is small
                # (n_unique_axis, n_k_axis), so this never materializes
                # a (nbins1 x nbins2, n_k1 x n_k2) dense tensor -- only
                # the same (n_k1, n_k2) grid tophat/read already used.
                rebinned = Mout_axes[0] @ (sign_out * to_spectrum(correlation)[1]) @ Mout_axes[1].T
                return rebinned[index_out[:, 0], index_out[:, 1]]

            tmp = jnp.zeros(shape=(len(kout), len(edgesin)))
            # fftlog
            # lowring=False, xy=1: align the transforms' grids exactly. Every ellwmax > 0 coupling
            # here transforms forward (ellin) and back (ell) at different orders, and with per-kernel low-ringing offsets the
            # window would be multiplied on a grid shifted by up to one spacing from where
            # the correlation actually lives.
            to_spectrum = CorrelationToSpectrum(s=tuple(next(iter(window)).coords().values()), ell=ell, check_level=1, minfolds=0, lowring=False, xy=1.)
            # Sub-nodes, fixed at their converged values: 16 Gauss-Legendre per theory bin per axis build the theory basis,
            # 8 refine the output-bin average. The multipoles B_{l1 l2 L}(k1, k2) bin only two legs, and the output measure
            # k1^2 k2^2 factorizes, so the separable `rebin` already gives the exact 2-D output-bin average.
            index_in, Min_axes = axis_basis_matrices(edgesin, to_spectrum.k, kind='spline', nsub=16)
            index_in_swap = index_in[:, ::-1]
            index_out, Mout_axes = axis_basis_matrices(bin.edges, to_spectrum.k, kind='rebin', nsub=8)

            for ellt, swap, wcoeffs in passes[ellin, wain, ell]:
                Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)
                to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=ellt, minfolds=0, lowring=False, xy=1.)
                term = jax.lax.map(partial(convolve, swap=swap, sign_in=-1. if ellt[0] % 2 else 1.), jnp.arange(edgesin.shape[0]), batch_size=batch_size).T
                if pbar: jax.block_until_ready(term)
                tmp += term
                progress.update(1)

            wmat_tmp[ellin, wain].append(tmp)

        wmat_tmp[ellin, wain] = jnp.concatenate(wmat_tmp[ellin, wain], axis=0)
    progress.close()

    wmat = jnp.concatenate(list(wmat_tmp.values()), axis=1)
    return wmat, None, jnp.mean(edgesin, axis=-1), edgesin, [ell[0] for ell in ellsin], [ell[1] for ell in ellsin]


# Scoccimarro window by deterministic quadrature (desi-gqc-notes, "Scoccimarro basis").
#
# Each window row (an output bin and multipole L) is, with the output triangle fixed (k3 along z, k1 in the xz half-plane),
# an integral of the window kernel Qhat(q1, q2) times the theory at k_i' = k_i - q_i (i = 1, 2), the line of sight averaged
# in closed form. It is taken over the kernel variables: for each leg, the magnitude q_i, the theory leg k_i' = |k_i - q_i|
# and its azimuth psi_i about k_i. The theory is expanded on a basis over the theory cells, the window computed per basis
# function, and the projection of the model onto the basis folded in at the end: the returned window acts on the model's
# values at points (k1, k2, k3).

# Quadrature rules: Gauss-Legendre points per leg piece (nleg), azimuths (npsi), points over the unsplit tail leg range
# (ntail), output-bin Gauss-Legendre points per leg (nbin), collapsed Gauss-Jacobi points per tetrahedron axis (ntet),
# single-precision angular weights (float32). Q000 reaches its floor at nleg = 2; the anisotropic parts carry polynomials
# in q^.k^ (in p^2), so nleg = 3 (2 is 0.05 sigma of one mock off); they tolerate npsi = 6 and float32. The same rules
# hold for the cubic basis (one more point per leg piece moves W . t by <= 0.002 sigma of one mock). Defaults per window
# multipole: default_scoccimarro_rules_iso for (0, 0, 0), default_scoccimarro_rules_aniso otherwise.
default_scoccimarro_rules_iso = dict(nleg=2, npsi=8, ntail=8, nbin=3, ntet=2, float32=False)
default_scoccimarro_rules_aniso = dict(nleg=3, npsi=6, ntail=8, nbin=3, ntet=2, float32=True)


def _scoccimarro_closes_triangle(k1, k2, k3):
    """Whether legs ``k1, k2, k3`` (arrays, broadcast) close a triangle, boundary included."""
    return (k3 <= k1 + k2) & (k1 <= k2 + k3) & (k2 <= k1 + k3)


def _get_scoccimarro_theory_basis(edgesin, basis='cubic'):
    r"""
    Theory basis. The theory grid is the product over :math:`(k_1', k_2', k_3')` of the 1D bins of ``edgesin`` (1D or
    (nbin, 2), contiguous, any widths); a cell is one product bin, kept if it contains a closed triangle. The basis is
    trilinear hats on the cell corners (``basis='linear'``) or tensor cubic B-splines centred on the grid knots (``'cubic'``:
    one more knot beyond each end), both uniform in the grid index: with variable widths, the cubic pieces are still
    polynomial in each cell, but only continuous where the width changes. The cubic basis has the natural boundary
    condition (zero second derivative at both ends of the range): the B-splines of the two padding knots lie almost
    entirely outside the range, where the theory is not fitted -- left free, their coefficients run to ~1e5 x the model,
    cancelling at the fit points but not where the window reads them -- so they are not nodes but fold into their two
    inward neighbours, :math:`c_0 = 2 c_1 - c_2`.

    Returns
    -------
    dict: ``cells`` (ncell, 3, 2) cell edges; ``knodes`` (nnode, 3) node coordinates; ``inodes`` (nnode, 3) their grid
    indices; ``shape`` grid size per leg; ``step`` smallest knot spacing; ``order`` 1 or 3; ``fold`` (nknot, shape[0])
    per-leg values of the node functions on the knot B-splines (the identity off the padding knots); ``index(k)`` the
    fractional knot index of ``k`` and ``knot(j)`` the coordinate of knot index ``j`` (jax arrays; piecewise linear,
    extrapolated with the end widths).
    """
    edgesin = np.asarray(edgesin, dtype='f8')
    if edgesin.ndim == 1: edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])
    width = edgesin[:, 1] - edgesin[:, 0]
    if not (np.all(width > 0.) and np.allclose(edgesin[1:, 0], edgesin[:-1, 1], rtol=0., atol=1e-6 * width.min())):
        raise ValueError('the quadrature window needs increasing, contiguous theory edges')
    if basis not in ['linear', 'cubic']:
        raise ValueError(f'unknown theory basis {basis}; choose linear or cubic')
    order = 3 if basis == 'cubic' else 1
    knots = np.append(edgesin[:, 0], edgesin[-1, 1])
    if order == 3: knots = np.concatenate([[knots[0] - width[0]], knots, [knots[-1] + width[-1]]])
    steps = np.diff(knots)
    step = steps.min()

    grid = np.arange(len(knots), dtype='f8')

    def interp(x, xs, ys):  # piecewise linear, extrapolated with the end slopes
        y = jnp.interp(x, xs, ys)
        y = jnp.where(x < xs[0], ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]), y)
        return jnp.where(x > xs[-1], ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]), y)

    index = lambda k: interp(k, knots, grid)   # fractional grid index of k
    knot = lambda j: interp(j, grid, knots)    # coordinate of grid index j

    ndim = 3  # legs k1, k2, k3
    ibin = np.stack(np.meshgrid(*[np.arange(len(edgesin))] * ndim, indexing='ij'), axis=-1).reshape(-1, ndim)
    cells = np.stack([edgesin[ibin[:, axis]] for axis in range(ndim)], axis=1)
    # a cell contains a closed triangle if k3 can reach |k1 - k2| and k1 + k2; tested in units of the smallest step, rounded,
    # so that round-off in the edges cannot flip a boundary cell
    unit = np.round(cells / step, 6)
    gap12 = np.maximum(np.maximum(unit[:, 0, 0] - unit[:, 1, 1], unit[:, 1, 0] - unit[:, 0, 1]), 0.)  # min |k1 - k2| over the cell
    # keep the cell if [k3 min, k3 max] meets [min |k1 - k2|, max k1 + k2] (the range of |k1 - k2| <= k3 <= k1 + k2 over the cell)
    cells = cells[(unit[:, 2, 1] >= gap12) & (unit[:, 2, 0] <= unit[:, 0, 1] + unit[:, 1, 1])]
    # the basis functions not vanishing on a cell (lower knot j per leg) are those of knots j to j + 1 (linear), j - 1 to j + 2 (cubic)
    offsets = np.array(list(itertools.product(range(-1, 3) if order == 3 else range(2), repeat=ndim)))
    inodes = np.unique((np.rint(np.asarray(index(cells[:, :, 0]))).astype(int)[:, None, :] + offsets[None]).reshape(-1, ndim), axis=0)
    nknot = len(knots)
    fold = np.eye(nknot)
    if order == 3:  # natural boundary condition: the padding knots 0 and nknot - 1 are not nodes
        inodes = inodes[np.all((inodes >= 1) & (inodes <= nknot - 2), axis=-1)]
        fold = fold[:, 1:-1]
        fold[0, :2], fold[-1, -2:] = [2., -1.], [-1., 2.]
    knodes = np.asarray(knot(inodes.astype('f8')))
    inodes = inodes - (order == 3)   # node index = knot index - 1 for the cubic basis
    return dict(cells=cells, knodes=knodes, inodes=inodes, shape=(fold.shape[1],) * ndim, step=step, order=order, fold=fold, index=index, knot=knot)


def _get_scoccimarro_basis_weights(k, basis, n=None):
    """
    Per-leg values at ``k`` (array) of the node functions of the theory ``basis`` (:func:`_get_scoccimarro_theory_basis`).
    With ``n`` (the number of nodes per leg): dense, ``k.shape + (n,)`` (jax, for the dense contractions); else sparse:
    (node indices, values) of a window of ``order + 1`` consecutive nodes holding every function not vanishing at ``k``
    inside the range, each ``k.shape + (order + 1,)`` (numpy).
    """
    xp = np if n is None else jnp
    x = np.asarray(basis['index'](k)) if n is None else basis['index'](k)
    cell = xp.floor(x)
    u = x - cell
    if basis['order'] == 1:   # linear hats
        offsets, values = [0, 1], [1. - u, u]
    else:                     # uniform cubic B-splines (in the knot index)
        offsets, values = [-1, 0, 1, 2], [(1. - u)**3 / 6., (3. * u**3 - 6. * u**2 + 4.) / 6., (-3. * u**3 + 3. * u**2 + 3. * u + 1.) / 6., u**3 / 6.]
    # the knot B-splines not vanishing at k, onto the node functions (rows of the fold matrix; zero beyond the knots)
    fold = xp.asarray(basis['fold'])
    nknot, nnode = fold.shape
    dense = sum(value[..., None] * xp.where(((cell + offset >= 0) & (cell + offset < nknot))[..., None], fold[xp.clip(cell + offset, 0, nknot - 1).astype(int)], 0.)
                for offset, value in zip(offsets, values))
    if n is not None: return dense
    start = np.clip(cell.astype(int) + offsets[0] - (nknot - nnode) // 2, 0, nnode - len(offsets))
    index = start[..., None] + np.arange(len(offsets))
    return index, np.take_along_axis(dense, index, axis=-1)


def _get_scoccimarro_bin_rule(edges, nbin, ntet):
    r"""
    Quadrature of output bin ``edges`` (3, 2) with the measure :math:`k_1 k_2 k_3 \Theta`: (points (npoint, 3), weights (npoint,)
    summing to 1), padded with zero-weight points to a few fixed sizes (fewer recompilations). Gauss-Legendre product
    (``nbin`` per leg) if the triangle condition does not cut the bin; otherwise its polytope, split into tetrahedra with a
    collapsed (Duffy) Gauss-Jacobi rule (``ntet`` per axis).
    """
    sizes = (32, 48, 64, 96, 128, 192, 256, 384, 512)
    corners = np.array(np.meshgrid(*edges, indexing='ij')).reshape(3, -1).T
    if np.all(_scoccimarro_closes_triangle(*corners.T)):  # every corner closes a triangle: the bin is not cut
        x, w = np.polynomial.legendre.leggauss(nbin)
        k = [lo + (hi - lo) * (x + 1.) / 2. for lo, hi in edges]
        points = np.stack(np.meshgrid(*k, indexing='ij'), axis=-1).reshape(-1, 3)
        weights = np.prod(np.stack(np.meshgrid(*[w * (hi - lo) / 2. for lo, hi in edges], indexing='ij'), axis=-1), axis=-1).ravel()
    else:
        from scipy.spatial import Delaunay
        # polytope vertices: intersections of 3 of the 9 bounding planes (6 bin faces, 3 triangle inequalities) inside all
        planes = np.array([np.eye(3)[i] * s for i in range(3) for s in (-1., 1.)] + [[-1., -1., 1.], [-1., 1., -1.], [1., -1., -1.]])
        bounds = np.array([-edges[i, 0] if s < 0 else edges[i, 1] for i in range(3) for s in (-1., 1.)] + [0.] * 3)
        vertices = []
        for triplet in itertools.combinations(range(len(planes)), 3):
            if abs(np.linalg.det(planes[list(triplet)])) < 1e-12: continue
            x = np.linalg.solve(planes[list(triplet)], bounds[list(triplet)])
            if np.all(planes @ x <= bounds + 1e-12): vertices.append(x)
        vertices = np.unique(np.round(np.array(vertices), 14), axis=0) if vertices else np.zeros((0, 3))
        if len(vertices) < 4: return np.full((sizes[0], 3), 0.1), np.zeros(sizes[0])
        # collapsed Gauss-Jacobi rule on the unit tetrahedron, exact to degree 2 ntet - 1 (Jacobian in the Jacobi weights)
        from scipy.special import roots_jacobi
        (xu, wu), (xv, wv), (xw, ww) = roots_jacobi(ntet, 2., 0.), roots_jacobi(ntet, 1., 0.), np.polynomial.legendre.leggauss(ntet)
        u, v, w = np.meshgrid((xu + 1.) / 2., (xv + 1.) / 2., (xw + 1.) / 2., indexing='ij')
        reference = np.stack([u, (1. - u) * v, (1. - u) * (1. - v) * w], axis=-1).reshape(-1, 3)
        reference_weights = ((wu[:, None, None] / 8.) * (wv[None, :, None] / 4.) * (ww[None, None, :] / 2.)).ravel()
        points, weights = [], []
        for simplex in Delaunay(vertices).simplices:  # convex polytope: the Delaunay tetrahedra tile it
            v = vertices[simplex]
            volume = abs(np.linalg.det((v[1:] - v[0]).T))
            if volume < 1e-18: continue
            points.append(v[0] + reference @ (v[1:] - v[0]))
            weights.append(reference_weights * volume)
        points, weights = np.concatenate(points), np.concatenate(weights)
    weights = weights * np.prod(points, axis=-1)
    weights = weights / weights.sum() if weights.sum() > 0. else weights
    points = np.where((weights > 0.)[:, None], points, 0.1)
    npad = next(size for size in sizes if size >= len(points)) - len(points)
    return np.concatenate([points, np.full((npad, 3), 0.1)]), np.concatenate([weights, np.zeros(npad)])


def _get_scoccimarro_qgrid(s, Q000, basis, kmax):
    r"""
    Kernel grid, (qedges, qnodes), in :math:`|q|`: the fine cells over which the kernel :math:`\hat{Q}` is integrated
    exactly (its moments), and the coarse nodes at which the response of a row is computed, then interpolated.

    - Cells: the kernel oscillates with period :math:`\sim q_D = 2\pi / D`, :math:`D` the survey size (where the ``Q000``
      diagonal, on ``s``, falls to :math:`10^{-3}` of :math:`Q_{000}(0, 0)`): the cells are fractions of :math:`q_D`
      (finest near :math:`q = 0`, where the kernel is concentrated), and never coarser than for the LRG1 window.
    - Nodes: the response varies on the theory cell scale, whatever the window: nodes 1/8, 1/4, 1, 2, 4 median steps of
      the theory ``basis`` apart, finest near :math:`q = 0` (the median: cells refined at low k only do not need it).
    - The grid stops at :math:`q_\mathrm{max}` = ``kmax`` (largest output :math:`k`) + the largest :math:`k'` the basis
      reaches: a larger :math:`q` sends every leg to :math:`k' = |k - q| \geq q - k`, outside the basis (exact truncation).

    Tuned on the LRG1 window (:math:`D` = 1590 Mpc/h), theory step 0.02.
    """
    diag = np.abs(np.diag(Q000)) / np.abs(Q000[0, 0])
    qD = min(2. * np.pi / s[np.argmax(diag < 1e-3)], 0.004)   # 0.004: LRG1 (D = 1590 Mpc/h)
    step = np.median(np.diff(np.asarray(basis['knot'](np.arange(len(basis['fold']), dtype='f8')))))
    qmax = kmax + float(basis['knot'](basis['inodes'].max() + (basis['order'] + 1) / 2))   # end of the last node's support
    qedges = np.unique(np.concatenate([np.arange(0., qD, qD / 200.), np.arange(qD, 12.5 * qD, qD / 20.),
                                       np.arange(12.5 * qD, 75. * qD, qD / 4.), np.arange(75. * qD, qmax + qD / 2., qD / 2.)]))

    def spacing(q):  # node spacing, in theory steps
        return step * next(h for qstop, h in [(1.5, 1. / 8.), (3., 1. / 4.), (6., 1.), (16., 2.), (np.inf, 4.)] if q < qstop * step)

    qnodes = [0.]
    while qnodes[-1] < qmax: qnodes.append(qnodes[-1] + spacing(qnodes[-1]))
    return qedges[qedges <= qnodes[-1] + 1e-12], np.array(qnodes)


def _get_scoccimarro_window_rows(parts, s, qgrid, ells, columns, basis, rules, qsplit=None, batch_size=1.5e5, prune=1e-7, ds=2.):
    r"""
    Return ``row(edges, ell)``: the contribution of the window multipoles ``parts`` to the row of output bin ``edges``
    (3, 2), output multipole ``ell``, as (nnode, len(columns)).

    parts: window multipoles ``(ell1, ell2, L, Q)``, ``Q`` (ns, ns) on ``s``.
    s: log-spaced separations of the window multipoles.
    qgrid: (edges of the fine :math:`|q|` cells, coarse :math:`|q|` response nodes), :func:`_get_scoccimarro_qgrid`.
    ells: output multipoles.
    columns: theory columns ``(L', M', Legendre?)``.
    basis: theory basis, :func:`_get_scoccimarro_theory_basis`.
    rules: quadrature rules, as :data:`default_scoccimarro_rules_iso`.
    qsplit: beyond it, the theory leg range is not split at the cell faces; default twice the theory step.
    batch_size: number of integration points per kernel call.
    prune: kernel-magnitude pairs whose moments are below ``prune`` times the total in every multipole are skipped.
    ds: step of the linear separation grid of the Hankel transforms.
    """
    # 1. Kernel moments. Qhat rings (the survey edges) and cannot be sampled. The response A of a row to a kernel at fixed
    #    magnitudes is smooth: it is evaluated on the coarse q nodes and interpolated (A / q^l linear in q^2, as a leg of
    #    multipole l responds as q^l), and each multipole's kernel enters only through its exact moments against that
    #    interpolation over the fine q cells. Qhat = double Hankel transform on a linear s grid, Q interpolated in log s.
    from scipy.special import spherical_jn
    qedges, qnodes = qgrid
    qsplit = 2. * basis['step'] if qsplit is None else qsplit
    qmid = (qedges[1:] + qedges[:-1]) / 2.
    sgrid = np.arange(0., s.max() + ds / 2., ds)
    wgrid = np.full(len(sgrid), ds); wgrid[[0, -1]] = ds / 2.

    def interp_matrix(x, xp):  # (len(x), len(xp)) linear interpolation weights, x clipped to xp's range
        x = np.clip(x, xp[0], xp[-1])
        j = np.clip(np.searchsorted(xp, x) - 1, 0, len(xp) - 2)
        frac = (x - xp[j]) / (xp[j + 1] - xp[j])
        toret = np.zeros((len(x), len(xp)))
        toret[np.arange(len(x)), j] = 1. - frac
        toret[np.arange(len(x)), j + 1] = frac
        return toret

    # bilinear interpolation of Q as R Q R^T: a 2D interpolator on the (nsgrid, nsgrid) grid needs ~20 GB
    R = interp_matrix(np.log(np.maximum(sgrid, s[0])), np.log(s)) * (sgrid <= s[-1])[:, None]
    ellmax = max(max(part[:2]) for part in parts)
    hankel = {ell: jnp.asarray(spherical_jn(ell, qmid[:, None] * sgrid) * sgrid**2 * wgrid) @ jnp.asarray(R) for ell in range(ellmax + 1)}
    interp = {}
    for ell in range(ellmax + 1):
        P = interp_matrix(qmid**2, qnodes**2)
        if ell:
            P[:, 1] += P[:, 0]  # A / q^ell constant below the first non-zero node
            P[:, 0] = 0.
            P[:, 1:] *= (qmid[:, None] / qnodes[1:])**ell
        interp[ell] = P * (qmid <= qnodes[-1])[:, None]
    volume = 4. * np.pi * qmid**2 * np.diff(qedges)
    toret = []
    for l1, l2, L, Q in parts:
        H = np.asarray(hankel[l1] @ jnp.asarray(Q) @ hankel[l2].T)
        mass = (4. * np.pi)**2 * (-1)**((l1 + l2) // 2) * H * volume[:, None] * volume / (2. * np.pi)**6
        toret.append(interp[l1].T @ mass @ interp[l2])
    moments = np.array(toret)

    # 2. Line-of-sight average, in closed form: per point, a short sum over harmonic products y_l1^m1(q1^) y_l2^m2(q2^)
    #    times conj(y_L'^mu(k3'^)) (Legendre column) or a Wigner-D factor ((L', M') column), with 3j coefficients
    def get_angular_coefficients(ell, part, combos, columns):
        r"""
        Coefficients of the :math:`\hat{n}`-averaged angular weight of window part ``part = (l1, l2, Lt)``, output multipole
        ``ell``: shape (len(columns), len(combos)), ``combos`` the :math:`(\ell_1, m_1, \ell_2, m_2)` products of harmonics,

        .. math::

            (2L + 1) \frac{(\ell_1 \ell_2 L_t; m_1 m_2 M)}{(\ell_1 \ell_2 L_t; 0 0 0)} (L L_t L'; 0 0 0) (L L_t L'; 0 M -M),
            \quad M = -m_1 - m_2,

        the first factor from the tripolar harmonic, the last two from :math:`\mathrm{E}_{\hat{n}}[y_L^0 y_{L_t}^M y_{L'}^{-M}]`;
        times :math:`\sqrt{(2L'+1)/4\pi}` for a :math:`(L', M')` column (:math:`Y_{L'M'}` rather than :math:`\mathcal{L}_{L'}`).
        """
        l1, l2, Lt = part
        toret = np.zeros((len(columns), len(combos)))
        for icombo, (a1, m1, a2, m2) in enumerate(combos):
            M = -m1 - m2
            if (a1, a2) != (l1, l2) or abs(M) > Lt: continue
            tripolar = wigner_3j(l1, l2, Lt, m1, m2, M) / wigner_3j(l1, l2, Lt, 0, 0, 0)
            for icol, (ellin, _, legendre) in enumerate(columns):
                if tripolar and abs(M) <= ellin:
                    norm = 1. if legendre else np.sqrt((2 * ellin + 1) / (4. * np.pi))
                    toret[icol, icombo] = norm * (2 * ell + 1) * tripolar * wigner_3j(ell, Lt, ellin, 0, 0, 0) * wigner_3j(ell, Lt, ellin, 0, M, -M)
        return toret

    def get_wigner_d(ell, m1, m2, cos):
        r"""Small Wigner :math:`d^\ell_{m_1 m_2}(\beta)` at :math:`\cos\beta` = ``cos`` (closed form, static orders)."""
        c, s = jnp.sqrt(jnp.clip((1. + cos) / 2., 0., 1.)), jnp.sqrt(jnp.clip((1. - cos) / 2., 0., 1.))
        toret = 0.
        for k in range(max(0, m2 - m1), min(ell + m2, ell - m1) + 1):
            coeff = (-1)**(m1 - m2 + k) * np.sqrt(factorial(ell + m1) * factorial(ell - m1) * factorial(ell + m2) * factorial(ell - m2)) \
                / (factorial(ell + m2 - k) * factorial(k) * factorial(m1 - m2 + k) * factorial(ell - m1 - k))
            toret = toret + coeff * c**(2 * ell + m2 - m1 - 2 * k) * s**(m1 - m2 + 2 * k)
        return toret

    knodes, inodes, shape, step, index, knot = (basis[name] for name in ['knodes', 'inodes', 'shape', 'step', 'index', 'knot'])
    nleg, npsi, ntail, float32 = rules['nleg'], rules['npsi'], rules['ntail'], rules['float32']
    legpairs = sorted({part[:2] for part in parts})
    isotropic = legpairs == [(0, 0)]
    combos = [(l1, m1, l2, m2) for l1, l2 in legpairs for m1 in range(-l1, l1 + 1) for m2 in range(-l2, l2 + 1)]
    # the harmonics y_l1^m1(q1^), y_l2^m2(q2^), y_L'^(m1 + m2)(k3'^) the angular weights need, and where each combo reads them
    ellms1, ellms2 = sorted({combo[:2] for combo in combos}), sorted({combo[2:] for combo in combos})
    # Legendre column L': conj(y_L'^mu(k3'^)), mu = m1 + m2; (L', M') column, Re Y_L'M' of the line of sight in the theory
    # triangle's frame: (D^L'_{mu M'} + (-1)^M' D^L'_{mu, -M'}) / 2 of the rotation to that frame (Wigner D)
    mus = lambda ellin: [int(np.clip(m1 + m2, -ellin, ellin)) for _, m1, _, m2 in combos]
    ellms3 = sorted({(ellin, mu) for ellin, _, legendre in columns if legendre for mu in mus(ellin)}) or [(0, 0)]
    dtriples = sorted({(ellin, mu, sign * m) for ellin, m, legendre in columns if not legendre for mu in mus(ellin) for sign in (1, -1)})
    index1 = jnp.array([ellms1.index(combo[:2]) for combo in combos])
    index2 = jnp.array([ellms2.index(combo[2:]) for combo in combos])
    index3 = [jnp.array([ellms3.index((ellin, mu)) for mu in mus(ellin)]) if legendre else
              (jnp.array([dtriples.index((ellin, mu, m)) for mu in mus(ellin)]), jnp.array([dtriples.index((ellin, mu, -m)) for mu in mus(ellin)]), (-1)**m)
              for ellin, m, legendre in columns]
    node_index = jnp.asarray((inodes[:, 0] * shape[1] + inodes[:, 1]) * shape[2] + inodes[:, 2])

    # 3. Work plan: kernel-magnitude pairs (q_a, q_b) with significant moments, grouped by the number of cell-face pieces
    #    each theory leg range spans (fixed array shapes: one compilation per group), batched to ``batch_size`` points
    def get_npieces(q):
        if q > qsplit + 1e-9: return 0
        n = int(np.ceil(2. * q / step - 1e-9)) + 1   # faces in a range of width 2q: at most this many with the smallest step
        return next(size for size in (1, 2, 3, 5, 8, 13) if n <= size)

    def get_npsi(group):  # more azimuths where the k3' range spans more node cells (its hats kink there)
        return 2 * npsi if 0 in group else npsi * int(np.ceil(np.sqrt(max(group))))

    def get_npoints(group):
        nlegs = [ntail if n == 0 else n * nleg for n in group]
        return nlegs[0] * nlegs[1] * (get_npsi(group) // 2) * get_npsi(group)

    significant = np.any(np.abs(moments) > prune * np.abs(moments).sum(axis=(1, 2))[:, None, None], axis=0)
    groups = {}
    for a, b in zip(*np.nonzero(significant)):
        groups.setdefault((get_npieces(qnodes[a]), get_npieces(qnodes[b])), []).append((a, b))
    # per output multipole: the parts' moments contracted with their angular coefficients, (npair, len(ellsin), ncombo)
    coefficients = {ell: np.array([get_angular_coefficients(ell, part[:3], combos, columns) for part in parts]) for ell in ells}
    qeval = np.maximum(qnodes, 1e-6)  # q = 0 is evaluated at a tiny q: the leg rule degenerates to p = k
    plan = {}
    for group, pairs in groups.items():
        pairs, size = np.array(pairs), max(1, int(batch_size // get_npoints(group)))
        plan[group] = []
        for start_pair in range(0, len(pairs), size):
            sub = pairs[start_pair:start_pair + size]
            npad = size - len(sub) if len(pairs) > size else 0  # pad the last batch: one shape per group
            q1, q2 = (jnp.asarray(np.concatenate([qeval[sub[:, i]], np.full(npad, 1e-6)])) for i in range(2))
            mom = np.concatenate([moments[:, sub[:, 0], sub[:, 1]], np.zeros((len(parts), npad))], axis=1)
            plan[group].append((q1, q2, {ell: jnp.asarray(np.einsum('tn,tlk->nlk', mom, coefficients[ell])) for ell in ells}))

    # 4. Integrand at one output-bin point, summed over a batch of pairs: output triangle fixed; theory leg magnitudes by
    #    Gauss-Legendre between cell faces (the basis is polynomial there) with the density k' / (2 k q); azimuths by the
    #    trapezoid rule (periodic), psi1 over [0, pi] by reflection; the basis accumulated by dense separable contractions
    #    onto the node grid (an atomic scatter of all the points onto a few nodes serialises on a GPU)
    def get_ylm(x, ellms):
        r"""
        :math:`y_\ell^m = \sqrt{4\pi/(2\ell+1)} Y_{\ell m}` of vectors ``x`` (..., 3), stacked for ``ellms``: the reduced
        harmonics of :func:`get_Ylm_all` (stable recurrence, azimuth 0) times :math:`e^{im\phi}`.
        """
        r = jnp.linalg.norm(x, axis=-1)
        cos = jnp.where(r > 0., x[..., 2] / jnp.where(r > 0., r, 1.), 1.)
        phi = jnp.arctan2(x[..., 1], x[..., 0])
        ells, ms = np.array(ellms).T
        table = get_Ylm_all(int(ells.max()), reduced=True)(cos)[ells**2 + ells + ms]
        return jnp.moveaxis(table, 0, -1) * jnp.exp(1j * ms * phi[..., None])

    def leg_rule(k, q, npieces):
        # p = |k - q| has density p / (2kq) on [|k - q|, k + q], here split at the cell faces (the basis is polynomial between them)
        lo, hi = jnp.abs(k - q), k + q
        if npieces:
            x, w = np.polynomial.legendre.leggauss(nleg)
            faces = knot(jnp.floor(index(lo))[..., None] + 1. + jnp.arange(npieces - 1))
            bounds = jnp.concatenate([lo[..., None], jnp.clip(faces, lo[..., None], hi[..., None]), hi[..., None]], axis=-1)
        else:
            x, w = np.polynomial.legendre.leggauss(ntail)
            bounds = jnp.stack([lo, hi], axis=-1)
        a, b = bounds[..., :-1, None], bounds[..., 1:, None]
        p = (a + b) / 2. + (b - a) / 2. * x
        weight = (b - a) / 2. * w * p / (2. * k[..., None, None] * q[..., None, None])
        return p.reshape(p.shape[:-2] + (-1,)), weight.reshape(weight.shape[:-2] + (-1,))

    def leg_vectors(kvec, p, q, psi):
        # k' at angle beta from k (cos beta = (k^2 + p^2 - q^2) / (2kp)) and azimuth psi measured from the triangle (xz) plane
        knorm = jnp.linalg.norm(kvec)
        khat = kvec / knorm
        e1, e2 = jnp.stack([khat[2], 0. * khat[0], -khat[0]]), jnp.array([0., 1., 0.])
        cosb = jnp.clip((knorm**2 + p**2 - q[:, None]**2) / (2. * knorm * p), -1., 1.)
        sinb = jnp.sqrt(1. - cosb**2)
        return p[..., None, None] * (cosb[..., None, None] * khat + sinb[..., None, None] * (jnp.cos(psi)[:, None] * e1 + jnp.sin(psi)[:, None] * e2))

    def get_kernel(group):
        n = get_npsi(group)
        psi1 = (jnp.arange(n // 2) + 0.5) * np.pi / (n // 2)  # [0, pi]: reflection symmetry
        psi2 = (jnp.arange(n) + 0.5) * 2. * np.pi / n
        cdtype = jnp.complex64 if float32 else jnp.complex128

        def bin_point(point, q1, q2, table):
            k, wk = point
            # the output triangle: k3 along z, k1 in the xz plane
            cos13 = (k[1]**2 - k[0]**2 - k[2]**2) / (2. * k[0] * k[2])
            k1 = k[0] * jnp.stack([jnp.sqrt(jnp.clip(1. - cos13**2, 0., 1.)), 0. * cos13, cos13])
            k2 = -k1 - jnp.stack([0. * cos13, 0. * cos13, k[2]])
            p1, w1 = leg_rule(jnp.broadcast_to(k[0], q1.shape), q1, group[0])
            p2, w2 = leg_rule(jnp.broadcast_to(k[1], q2.shape), q2, group[1])
            k1p, k2p = leg_vectors(k1, p1, q1, psi1), leg_vectors(k2, p2, q2, psi2)   # (npair, n1, m1, 3), (npair, n2, m2, 3)
            k3p = -(k1p[:, :, None, :, None] + k2p[:, None, :, None, :])            # (npair, n1, n2, m1, m2, 3)
            k3pnorm = jnp.linalg.norm(k3p, axis=-1)
            y3 = jnp.conj(get_ylm(k3p, ellms3)).astype(cdtype)
            if dtriples:  # Euler angles of the theory triangle's frame (z' along k3', k1' in the x'z' half-plane)
                cosb = k3p[..., 2] / k3pnorm
                alpha = jnp.arctan2(k3p[..., 1], k3p[..., 0])
                sinb = jnp.sqrt(jnp.clip(1. - cosb**2, 0., 1.))
                etheta = jnp.stack([cosb * jnp.cos(alpha), cosb * jnp.sin(alpha), -sinb], axis=-1)
                ephi = jnp.stack([-jnp.sin(alpha), jnp.cos(alpha), 0. * alpha], axis=-1)
                k1pf = jnp.broadcast_to(k1p[:, :, None, :, None], k3p.shape)
                gamma = jnp.arctan2(jnp.sum(k1pf * ephi, axis=-1), jnp.sum(k1pf * etheta, axis=-1))
                d3 = jnp.stack([jnp.exp(-1j * (mu * alpha + m * gamma)) * get_wigner_d(ellin, mu, m, cosb) for ellin, mu, m in dtriples], axis=-1).astype(cdtype)
            if not isotropic:  # q = k - k' (direction only)
                y12 = (get_ylm(k1 - k1p, ellms1)[..., index1][:, :, None, :, None].astype(cdtype)
                       * get_ylm(k2 - k2p, ellms2)[..., index2][:, None, :, None].astype(cdtype))
            weights = []
            for icol in range(len(columns)):
                if columns[icol][2]:
                    factor = y3[..., index3[icol]]
                else:
                    plus, minus, sign = index3[icol]
                    factor = (d3[..., plus] + sign * d3[..., minus]) / 2.
                z = jnp.real(factor if isotropic else y12 * factor)
                weights.append(jnp.einsum('pabcdk,pk->pabcd', z, table[:, icol].astype(z.dtype)).astype(jnp.float64))
            weights = jnp.stack(weights, axis=-1) * (wk / (n // 2) / n * w1[:, :, None, None, None] * w2[:, None, :, None, None])[..., None]
            # dense separable contraction of the basis (no atomic scatter): k3', then p1, then p2, into the node box
            toret = jnp.einsum('pabcdg,pabcdz->pabzg', weights, _get_scoccimarro_basis_weights(k3pnorm, basis, shape[2]))
            toret = jnp.einsum('pabzg,pax->pbxzg', toret, _get_scoccimarro_basis_weights(p1, basis, shape[0]))
            toret = jnp.einsum('pbxzg,pby->xyzg', toret, _get_scoccimarro_basis_weights(p2, basis, shape[1]))
            return toret.reshape(-1, len(columns))[node_index]

        @jax.jit
        def kernel(points, weights, q1, q2, table):  # sum over the output-bin points, one at a time to bound memory
            return jnp.sum(jax.lax.map(lambda point: bin_point(point, q1, q2, table), (points, weights)), axis=0)

        return kernel

    kernels = {group: get_kernel(group) for group in groups}

    # 5. The row: sum over the groups, with a coarser output-bin rule for the tail kernels (they spread over the bin)
    def row(edges, ell):
        rule = tuple(map(jnp.asarray, _get_scoccimarro_bin_rule(edges, rules['nbin'], rules['ntet'])))
        if not rule[1].sum(): return np.zeros((len(knodes), len(columns)))
        rule_tail = tuple(map(jnp.asarray, _get_scoccimarro_bin_rule(edges, 2, max(1, rules['ntet'] - 1))))  # tail kernels spread the bin: coarser rule
        toret = 0.
        for group, chunks in plan.items():
            for q1, q2, tables in chunks:
                toret = toret + kernels[group](*(rule_tail if 0 in group else rule), q1, q2, tables[ell])
        return np.asarray(toret)

    return row
