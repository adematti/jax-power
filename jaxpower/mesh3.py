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

from .mesh import BaseMeshField, MeshAttrs, RealMeshField, ComplexMeshField, ParticleField, staticarray, get_sharding_mesh, _get_hermitian_weights, _find_unique_edges, _get_bin_attrs, _bincount, create_sharded_random
from .mesh2 import _get_los_vector, FKPField
from .types import Mesh3SpectrumPole, Mesh3SpectrumPoles, Mesh3CorrelationPole, Mesh3CorrelationPoles, ObservableLeaf, ObservableTree, WindowMatrix
from .utils import real_gaunt, get_legendre, get_spherical_jn, get_Ylm, wigner_3j, wigner_9j, register_pytree_dataclass


prod = functools.partial(functools.reduce, operator.mul)



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
    iedges = _product([jnp.arange(len(xx)) for xx in nmodes1d])[mask]
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
            axis_iedges = jnp.arange(len(edges1d[axis]))
            # Number of batches
            nsplits = (len(axis_iedges) + buffer_size - 1) // buffer_size
            split_iedges.append(jnp.array_split(axis_iedges, nsplits, axis=0))
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



def _format_meshes(*meshes):
    """Format input meshes for autocorrelation/cross-correlation: return list of 3 meshes,
    and a list of indices corresponding to the mesh they are equal to."""
    meshes = list(meshes)
    meshes = meshes + [None] * (3 - len(meshes))
    fields = [0]
    for mesh in meshes[1:]: fields.append(fields[-1] if mesh is None else fields[-1] + 1)
    for imesh, mesh in enumerate(meshes):
        if mesh is None:
            meshes[imesh] = meshes[imesh - 1]
    return meshes, tuple(fields)


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
        return [jnp.array(xx) for xx in zip(*toret)]
    return [jnp.zeros((0,), dtype=int) for _ in range(5)]



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

        @partial(jax.checkpoint, static_argnums=0)
        def f(Ylm, carry, im):
            coeff, sym, im = im[3], im[4], im[:3]
            tmp = tuple(meshes[i] * jax.lax.switch(im[i], Ylm[i], *kvec) for i in range(2))
            los = xvec if vlos is None else vlos
            tmp += (jax.lax.switch(im[2], Ylm[2], *los) * meshes[2],)
            tmp = coeff.astype(mattrs.rdtype) * bin(*tmp) #, remove_zero=True)
            carry += tmp + sym.astype(mattrs.rdtype) * tmp.conj()
            return carry, im

        for ill, (ell1, ell2, ell3) in enumerate(ells):
            Ylms = [[get_Ylm(ell, m, reduced=True, real=False, conj=True) for m in range(-ell, ell + 1)] for ell in (ell1, ell2, ell3)]
            xs = _iter_triposh(ell1, ell2, ell3, los=los)
            if xs[0].size:
                num_ = jax.lax.scan(partial(f, Ylms), init=jnp.zeros(len(bin.edges), dtype=mattrs.cdtype), xs=xs)[0] / bin.nmodes[ill]
                #num.append(bin(*meshes, remove_zero=True) / bin.nmodes[ill])
            else:
                num_ = jnp.zeros(len(bin.edges), dtype=mattrs.cdtype)
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

        @partial(jax.checkpoint, static_argnums=(0, 1))
        def f(Ylm, ell, carry, im):
            coeff, sym, im = im[3], im[4], im[:3]
            tmp = tuple(meshes[i] * jax.lax.switch(im[i], Ylm[i], *kvec) for i in range(2))
            los = xvec if vlos is None else vlos
            tmp += (jax.lax.switch(im[2], Ylm[2], *los) * meshes[2],)
            tmp = coeff.astype(mattrs.rdtype) * bin(*tmp, ell=ell) # remove_zero=True)
            carry += tmp + sym.astype(mattrs.rdtype) * tmp.conj()
            return carry, im

        for ill, (ell1, ell2, ell3) in enumerate(ells):
            Ylms = [[get_Ylm(ell, m, reduced=True, real=False, conj=True) for m in range(-ell, ell + 1)] for ell in (ell1, ell2, ell3)]
            xs = _iter_triposh(ell1, ell2, ell3, los=los)
            if xs[0].size:
                num_ = jax.lax.scan(partial(f, Ylms, (ell1, ell2, ell3)), init=jnp.zeros(len(bin.edges), dtype=mattrs.cdtype), xs=xs)[0]
                #num.append(bin(*meshes, remove_zero=True) / bin.nmodes[ill])
            else:
                num_ = jnp.zeros(len(bin.edges), dtype=mattrs.cdtype)
            num.append(num_.real)

    correlation = []
    for ill, ell in enumerate(ells):
        correlation.append(Mesh3CorrelationPole(s=bin.xavg, s_edges=bin.edges, nmodes=bin.nmodes[ill], num_raw=num[ill], norm=norm, attrs=attrs, basis=bin.basis, ell=ell))
    return Mesh3CorrelationPoles(correlation)


from .mesh2 import compute_normalization, compute_box_normalization, split_particles


def compute_box3_normalization(*inputs: RealMeshField | ParticleField, bin: BinMesh3SpectrumPoles=None) -> jax.Array:
    """Compute normalization, assuming constant density, for the bispectrum."""
    return compute_box_normalization(*_format_meshes(*inputs)[0], bin=bin)


def compute_fkp3_normalization(*fkps, bin: BinMesh3SpectrumPoles=None, cellsize=10., split=None):
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
        The input particle fields are split such that the total number of splits is 3.
        For instance, if 3 different fields are provided, no splitting is performed; if 2 fields are provided, one of them is split in 2.
        The each split is painted on a mesh, and the normalization is computed from the product of the 3 meshes.
        If ``None``, no splitting is performed.

    Returns
    -------
    norm : float, list
    """
    fkps_none =  list(fkps) + [None] * (3 - len(fkps))
    fkps, fields = _format_meshes(*fkps)
    alpha = prod(map(lambda fkp: fkp.data.sum() / fkp.randoms.sum() if isinstance(fkp, FKPField) else 1., fkps))

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    if split is not None:
        randoms = split_particles(*[get_randoms(fkp) for fkp in fkps_none], seed=split)
        alpha *= prod(get_randoms(fkp).sum() / randoms.sum() for fkp, randoms in zip(fkps, randoms, strict=True))
    else:
        fkps, fields = _format_meshes(*fkps)
        randoms = [get_randoms(fkp) for fkp in fkps]
    kw = dict(cellsize=cellsize)
    for name in list(kw):
        if kw[name] is None: kw.pop(name)
    norm = alpha * compute_normalization(*randoms, **kw)
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_fkp3_shotnoise(*fkps, bin=None, los: str | np.ndarray='z', resampler='cic', interlacing=False, **kwargs):
    """
    Compute the FKP shot noise for the bispectrum or 3pcf.

    Parameters
    ----------
    fkps : FKPField
        FKP fields.
    bin : BinMesh3SpectrumPoles, BinMesh3CorrelationPoles, optional
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
    compensate : bool, default=False
        If ``True``, applies compensation to the mesh after painting.
    kwargs : dict
        Additional arguments for :meth:`ParticleField.paint`.

    Returns
    -------
    shotnoise : list
        Shot noise for each multipole.
    """
    if isinstance(bin, BinMesh3SpectrumPoles):
        return compute_fkp3_spectrum_shotnoise(*fkps, bin=bin, los=los, resampler=resampler, interlacing=interlacing, **kwargs)
    if isinstance(bin, BinMesh3CorrelationPoles):
        return compute_fkp3_correlation_shotnoise(*fkps, bin=bin, los=los, resampler=resampler, interlacing=interlacing, **kwargs)
    raise ValueError(f'bin must be either BinMesh3SpectrumPoles or BinMesh3CorrelationPoles, not {type(bin)}')



def compute_fkp3_spectrum_shotnoise(*fkps, bin=None, los: str | np.ndarray='z', resampler='cic', interlacing=False, compensate=True, **kwargs):
    """
    Compute the FKP shot noise for the bispectrum.

    Parameters
    ----------
    fkps : FKPField
        FKP fields.
    bin : BinMesh3SpectrumPoles, optional
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
    compensate : bool, default=False
        If ``True``, applies compensation to the mesh after painting.
    kwargs : dict
        Additional arguments for :meth:`ParticleField.paint`.

    Returns
    -------
    shotnoise : list
        Shot noise for each multipole.
    """
    fkps, fields = _format_meshes(*fkps)
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

    def bin_mesh2(mesh, axis):
        nmodes1d = bin.nmodes1d[axis]
        return _bincount(bin.ibin1d[axis] + 1, getattr(mesh, 'value', mesh), weights=bin.wmodes, length=len(nmodes1d)) / nmodes1d

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

    los, vlos = _format_los(los, ndim=mattrs.ndim)
    # The real-space grid
    xvec = mattrs.rcoords(sparse=True)
    svec = mattrs.rcoords(kind='separation', sparse=True)
    # The Fourier-space grid
    kvec = mattrs.kcoords(sparse=True)
    kcirc = mattrs.kcoords(kind='circular', sparse=True)

    if 'scoccimarro' in bin.basis:

        # Eq. 58 of https://arxiv.org/pdf/1506.02729, 1 => 3
        if not (fields[0] == fields[1] == fields[2]):
            raise NotImplementedError
        cmeshw = particles[0].paint(**kwargs, out='complex')
        cmeshw = cmeshw.clone(value=cmeshw.value.at[(0,) * cmeshw.ndim].set(0.))  # remove zero-mode
        cmeshw2 = particles[0].clone(weights=particles[0].weights**2).paint(**kwargs, out='complex')
        cmeshw2 = cmeshw2.clone(value=cmeshw2.value.at[(0,) * cmeshw2.ndim].set(0.))  # remove zero-mode
        sumw3 = jnp.sum(particles[0].weights**3)
        del particles

        ndim = 3
        def apply_fourier_legendre(ell, cmesh):
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)
            return get_legendre(ell)(mu) * cmesh

        def apply_fourier_harmonics(ell, rmesh):
            rmesh = getattr(rmesh, 'value', rmesh)
            Ylms = [get_Ylm(ell, m, reduced=False, real=True) for m in range(-ell, ell + 1)]

            @partial(jax.checkpoint, static_argnums=(0, 1))
            def f(Ylm, carry, im):
                carry += mattrs.r2c(rmesh * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            xs = np.arange(len(Ylms))
            return (4. * jnp.pi) / (2 * ell + 1) * jax.lax.scan(partial(f, Ylms), init=mattrs.create(fill=0., kind='complex'), xs=xs)[0]

        for ill, ell in enumerate(ells):
            if ell == 0:
                cmeshw3_ell = cmeshw * cmeshw2.conj()
                shotnoise[ill] = sum(bin_mesh2(cmeshw3_ell, axis=axis)[bin._iedges[..., axis]] for axis in range(ndim)) - 2. * sumw3
            else:
                # First line of eq. 58, q1 => q3
                if vlos is not None:
                    cmeshw_ell = apply_fourier_legendre(ell, cmeshw)
                else:
                    cmeshw_ell = apply_fourier_harmonics(ell, cmeshw.c2r())
                sn_ell = bin_mesh2(cmeshw_ell * cmeshw2.conj(), axis=ndim - 1)[bin._iedges[..., ndim - 1]]
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
                    tmp = [bin_mesh2(tmp, axis=axis) for axis in range(ndim - 1)] + [bin_mesh2(Ylm, axis=ndim - 1)]
                    carry += (4 * jnp.pi) * sum(tmp[axis][bin._iedges[..., axis]] * tmp[ndim - 1][bin._iedges[..., ndim - 1]] for axis in range(ndim - 1))
                    return carry, im

                Ylms = [get_Ylm(ell, m, reduced=False, real=True) for m in range(-ell, ell + 1)]
                xs = np.arange(len(Ylms))
                shotnoise[ill] = (2 * ell + 1) * jax.lax.scan(partial(f, Ylms), init=sn_ell, xs=xs)[0]

        shotnoise = [sn.real if ell % 2 == 0 else sn.imag for ell, sn in zip(ells, shotnoise)]

    else:
        # Eq. 45 - 46 of https://arxiv.org/pdf/1803.02132

        def compute_S111(particles, ellms):
            ellms = list(ellms)
            if ellms == [(0, 0)]:
                s0 = jnp.sum(particles[0].weights**3)
                s111 = [s0 if (ell, m) == (0, 0) else 0. for ell, m in ellms]
            else:
                rmesh = particles[0].clone(weights=particles[0].weights**3).paint(**kwargs, out='real')
                s111 = [jnp.sum(rmesh.value * get_Ylm(ell, m, reduced=True, real=False, conj=True)(*xvec)) for ell, m in ellms]
            return s111

        def compensate_shotnoise(s111):
            # NOTE: when there is no interlacing, triumvirate compensates pk with aliasing_shotnoise in the shotnoise estimation (but in bk estimation?)
            # In jaxpower, we always compensate by the standard compensate
            #if convention == 'triumvirate' and not interlacing:
            if not interlacing:
                return s111 * resampler.aliasing_shotnoise(1., kcirc) * (resampler.compensate(1., kcirc)**2 if compensate else 1.)
            return s111

        def compute_S122(particles, ells, axis):  # 1 == 2
            rmesh = particles[1].clone(weights=particles[1].weights**2).paint(**kwargs, out='real')
            rmesh -= rmesh.mean()
            cmesh = particles[0].paint(**kwargs, out='complex')
            cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                im, s111 = im
                # Second and third lines
                s111 = compensate_shotnoise(s111)
                los = xvec if vlos is None else vlos
                carry += jax.lax.switch(im, Ylm, *kvec) * (cmesh * (rmesh * jax.lax.switch(im, Ylm, *los)).r2c().conj() - s111)
                return carry, im

            s122 = []
            for ell in ells:
                Ylms = [get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)]
                xs = (jnp.arange(len(Ylms)), jnp.array([s111[ellms.index((ell, m))] for m in range(-ell, ell + 1)]))
                s122.append((2 * ell + 1) * bin_mesh2(jax.lax.scan(partial(f, Ylms), init=cmesh.clone(value=jnp.zeros_like(cmesh.value)), xs=xs)[0], axis))
            return s122

        def compute_S113(particles, ells):

            rmesh = particles[2].paint(**kwargs, out='real')
            rmesh -= rmesh.mean()
            cmesh = particles[0].clone(weights=particles[0].weights**2).paint(**kwargs, out='complex')
            cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode

            @partial(jax.checkpoint, static_argnums=(0, 1))
            def f(Ylms, jl, carry, im):
                los = xvec if vlos is None else vlos
                s111, coeff, sym, im = im[-1], im[3], im[4], im[:3]
                s111 = compensate_shotnoise(s111)
                # Fourth line
                tmp = coeff * ((rmesh * jax.lax.switch(im[2], Ylms[2], *los).conj()).r2c() * cmesh.conj() - s111)
                snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                tmp = tmp.c2r() * (jax.lax.switch(im[0], Ylms[0], *svec) * jax.lax.switch(im[1], Ylms[1], *svec)).conj()

                def fk(k):
                    return jnp.sum(tmp.value * jl[0](snorm * k[0]) * jl[1](snorm * k[1]))

                tmp = jax.lax.map(fk, bin.xavg)
                carry += tmp + sym * tmp.conj()
                return carry, im

            s113 = []
            for ell1, ell2, ell3 in ells:
                Ylms = [[get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)] for ell in [ell1, ell2, ell3]]
                xs = _iter_triposh(ell1, ell2, ell3, los=los)
                if xs[0].size:
                    # Add s111 for s, im in xs are offset by ell
                    xs = xs + [jnp.array([s111[ellms.index((ell3, int(im3) - ell3))] for im3 in xs[2]])]
                    sign = (1j)**(ell1 + ell2)
                    s113.append(sign * jax.lax.scan(partial(f, Ylms, [get_spherical_jn(ell1), get_spherical_jn(ell2)]), init=jnp.zeros_like(bin.xavg[..., 0], dtype=mattrs.cdtype), xs=xs)[0])
                else:
                    s113.append(jnp.zeros_like(bin.xavg[..., 0]))
            return s113

        uells = sorted(sum(ells, start=tuple()))
        ellms = [(ell, m) for ell in uells for m in range(-ell, ell + 1)]
        s111 = [0.] * len(ellms)
        if fields[0] == fields[1] == fields[2]:
            s111 = compute_S111(particles, ellms)
            ell0 = (0, 0, 0)
            if ell0 in ells:
                shotnoise[ells.index(ell0)] += s111[ellms.index((0, 0))]

        if fields[1] == fields[2]:
            def select(ell):
                return ell[2] == ell[0] and ell[1] == 0

            ells1 = [ell[0] for ell in ells if select(ell)]
            if ells1:
                particles01 = particles
                s122 = compute_S122(particles01, ells1, 0)

                for ill, ell in enumerate(ells):
                    if select(ell):
                        idx = ells1.index(ell[0])
                        shotnoise[ill] += s122[idx][bin._iedges[..., 0]]

        if fields[0] == fields[2]:
            def select(ell):
                return ell[2] == ell[1] and ell[0] == 0

            ells2 = [ell[1] for ell in ells if select(ell)]
            if ells2:
                particles01 = [particles[1], particles[0]]
                s121 = compute_S122(particles01, ells2, 1)
                for ill, ell in enumerate(ells):
                    if select(ell):
                        idx = ells2.index(ell[1])
                        shotnoise[ill] += s121[idx][bin._iedges[..., 1]]

        if fields[0] == fields[1]:
            s113 = compute_S113(particles, ells)
            for ill, ell in enumerate(ells):
                shotnoise[ill] += s113[ill]

        shotnoise = [sn.real if sum(ell[:2]) % 2 == 0 else sn.imag for ell, sn in zip(ells, shotnoise)]

    return list(shotnoise)


def compute_fkp3_correlation_shotnoise(*fkps, bin=None, los: str | np.ndarray='z', resampler='cic', interlacing=False, compensate=True, **kwargs):
    """
    Compute the FKP shot noise for the 3pcf.

    Parameters
    ----------
    fkps : FKPField
        FKP fields.
    bin : BinMesh3SpectrumPoles, optional
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
    compensate : bool, default=False
        If ``True``, applies compensation to the mesh after painting.
    kwargs : dict
        Additional arguments for :meth:`ParticleField.paint`.

    Returns
    -------
    shotnoise : list
        Shot noise for each multipole.
    """
    fkps, fields = _format_meshes(*fkps)
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

    def bin_mesh2(mesh, axis):
        nmodes1d = bin.nmodes1d[axis]
        return _bincount(bin.ibin1d[axis] + 1, getattr(mesh, 'value', mesh), weights=bin.wmodes, length=len(nmodes1d)) / nmodes1d

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
    los, vlos = _format_los(los, ndim=mattrs.ndim)
    # The real-space grid
    xvec = mattrs.rcoords(sparse=True)
    svec = mattrs.rcoords(kind='separation', sparse=True)
    # The Fourier-space grid
    kvec = mattrs.kcoords(sparse=True)
    kcirc = mattrs.kcoords(kind='circular', sparse=True)

    if 'scoccimarro' in bin.basis:

        raise NotImplementedError

    else: # sugiyama
        # Eq. 45 - 46 of https://arxiv.org/pdf/1803.02132

        def compute_S111(particles, ellms):
            ellms = list(ellms)
            if ellms == [(0, 0)]:
                s0 = jnp.sum(particles[0].weights**3)
                s111 = [s0 if (ell, m) == (0, 0) else 0. for ell, m in ellms]
            else:
                rmesh = particles[0].clone(weights=particles[0].weights**3).paint(**kwargs, out='real')
                s111 = [jnp.sum(rmesh.value * get_Ylm(ell, m, reduced=True, real=False, conj=True)(*xvec)) for ell, m in ellms]
            return s111

        def compensate_shotnoise(s111):
            # NOTE: when there is no interlacing, triumvirate compensates pk with aliasing_shotnoise in the shotnoise estimation (but in bk estimation?)
            # In jaxpower, we always compensate by the standard compensate
            #if convention == 'triumvirate' and not interlacing:
            if not interlacing:
                return s111 * resampler.aliasing_shotnoise(1., kcirc) * (resampler.compensate(1., kcirc)**2 if compensate else 1.)
            return s111

        def compute_S122(particles, ells, axis):  # 1 == 2
            rmesh = particles[1].clone(weights=particles[1].weights**2).paint(**kwargs, out='real')
            rmesh -= rmesh.mean()
            cmesh = particles[0].paint(**kwargs, out='complex')
            cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                im, s111 = im
                # Second and third lines
                s111 = compensate_shotnoise(s111)
                los = xvec if vlos is None else vlos
                carry += jax.lax.switch(im, Ylm, *svec) * (cmesh * (rmesh * jax.lax.switch(im, Ylm, *los)).r2c().conj() - s111).c2r()
                return carry, im

            s122 = []
            for ell in ells:
                Ylms = [get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)]
                xs = (jnp.arange(len(Ylms)), jnp.array([s111[ellms.index((ell, m))] for m in range(-ell, ell + 1)]))
                s122.append((2 * ell + 1) * bin_mesh2(jax.lax.scan(partial(f, Ylms), init=rmesh.clone(value=jnp.zeros_like(rmesh.value, dtype=mattrs.cdtype)), xs=xs)[0], axis))
            return s122

        def compute_S113(particles, ells):

            rmesh = particles[2].paint(**kwargs, out='real')
            rmesh -= rmesh.mean()
            cmesh = particles[0].clone(weights=particles[0].weights**2).paint(**kwargs, out='complex')
            cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode

            inter_edges = jnp.concatenate([jnp.max(bin.edges[..., 0], axis=1, keepdims=True), jnp.min(bin.edges[..., 1], axis=1, keepdims=True)], axis=-1)
            inter_edges = jnp.where(inter_edges[..., [1]] <= inter_edges[..., [0]], 0., inter_edges)
            snorm = jnp.sqrt(sum(ss**2 for ss in svec))
            from .mesh import _get_bin_attrs_edges2d
            inter_ibin, uinter_edges, M = _get_bin_attrs_edges2d(snorm, inter_edges)
            inter_ibin = inter_ibin[0]
            del snorm

            def bin_mesh2_inter(mesh):
                tmp = M @ _bincount(inter_ibin, getattr(mesh, 'value', mesh), length=len(uinter_edges) - 1)
                return tmp / (bin.nmodes1d[0][bin._iedges[..., 0]] * bin.nmodes1d[1][bin._iedges[..., 1]])

            @partial(jax.checkpoint, static_argnums=(0, 1))
            def f(Ylms, carry, im):
                los = xvec if vlos is None else vlos
                s111, coeff, sym, im = im[-1], im[3], im[4], im[:3]
                s111 = compensate_shotnoise(s111)
                # Fourth line
                tmp = coeff * ((rmesh * jax.lax.switch(im[2], Ylms[2], *los).conj()).r2c() * cmesh.conj() - s111)
                tmp = tmp.c2r() * (jax.lax.switch(im[0], Ylms[0], *svec) * jax.lax.switch(im[1], Ylms[1], *svec)).conj()
                tmp = bin_mesh2_inter(tmp)
                carry += tmp + sym * tmp.conj()
                return carry, im

            s113 = []
            for ell1, ell2, ell3 in ells:
                Ylms = [[get_Ylm(ell, m, reduced=True, real=False) for m in range(-ell, ell + 1)] for ell in [ell1, ell2, ell3]]
                xs = _iter_triposh(ell1, ell2, ell3, los=los)
                if xs[0].size:
                    # Add s111 for s, im in xs are offset by ell
                    xs = xs + [jnp.array([s111[ellms.index((ell3, int(im3) - ell3))] for im3 in xs[2]])]
                    sign = (-1)**(ell1 + ell2)
                    s113.append(sign * jax.lax.scan(partial(f, Ylms), init=jnp.zeros_like(bin.xavg[..., 0], dtype=mattrs.cdtype), xs=xs)[0])
                else:
                    s113.append(jnp.zeros_like(bin.xavg[..., 0]))
            return s113

        uells = sorted(sum(ells, start=tuple()))
        ellms = [(ell, m) for ell in uells for m in range(-ell, ell + 1)]
        s111 = [0.] * len(ellms)
        if fields[0] == fields[1] == fields[2]:
            s111 = compute_S111(particles, ellms)
            ell0 = (0, 0, 0)
            mask_shotnoise = jnp.all((bin.edges[..., 0] <= 0.) & (bin.edges[..., 1] >= 0.), axis=1)
            if ell0 in ells:
                shotnoise[ells.index(ell0)] += s111[ellms.index((0, 0))] * mask_shotnoise

        if fields[1] == fields[2]:
            def select(ell):
                return ell[2] == ell[0] and ell[1] == 0

            ells1 = [ell[0] for ell in ells if select(ell)]
            # r2 == 0
            mask_shotnoise = (bin.edges[..., 1, 0] <= 0.) & (bin.edges[..., 1, 1] >= 0.)

            if ells1:
                particles01 = particles
                s122 = compute_S122(particles01, ells1, 0)

                for ill, ell in enumerate(ells):
                    if select(ell):
                        idx = ells1.index(ell[0])
                        shotnoise[ill] += s122[idx][bin._iedges[..., 0]] * mask_shotnoise

        if fields[0] == fields[2]:
            def select(ell):
                return ell[2] == ell[1] and ell[0] == 0

            ells2 = [ell[1] for ell in ells if select(ell)]
            # r1 == 0
            mask_shotnoise = (bin.edges[..., 0, 0] <= 0.) & (bin.edges[..., 0, 1] >= 0.)

            if ells2:
                particles01 = [particles[1], particles[0]]
                s121 = compute_S122(particles01, ells2, 1)
                for ill, ell in enumerate(ells):
                    if select(ell):
                        idx = ells2.index(ell[1])
                        shotnoise[ill] += s121[idx][bin._iedges[..., 1]] * mask_shotnoise

        if fields[0] == fields[1]:
            s113 = compute_S113(particles, ells)
            for ill, ell in enumerate(ells):
                shotnoise[ill] += s113[ill]

        shotnoise = [sn.real for sn in shotnoise]

    return [sn / mattrs.cellsize.prod()**2 for sn in shotnoise]


def get_sugiyama_window_convolution_coeffs(ell, ellin):  # observed ell, theory ell
    # Prefactor, eq. 63 of https://arxiv.org/pdf/1803.02132
    # ell = (ell_1, ell_2, L)
    # ellin = (ell_1', ell_2', L')
    coeffs = []
    #for ellw in itertools.product(*([range(max(ell) + max(ellin) + 1)] * 3)):
    for ellw in itertools.product(*[range(ell_ + ellt_ + 1) for ell_, ellt_ in zip(ell, ellin)]):
        if sum(ellw) % 2: continue
        if ellw[2] % 2: continue
        coeff = prod((2 * ell_ + 1) for ell_ in ell)
        coeff *= wigner_9j(*ellw, *ellin, *ell)
        coeff *= wigner_3j(*ell, 0, 0, 0)
        for i in range(3): coeff *= wigner_3j(ell[i], ellin[i], ellw[i], 0, 0, 0)
        if abs(coeff) < 1e-7: continue
        coeff /= wigner_3j(*ellin, 0, 0, 0) * wigner_3j(*ellw, 0, 0, 0)
        coeffs.append((ellw, coeff))
    return coeffs


def get_scoccimarro_window_convolution_coeffs(ell, ellin):
    coeffs = []
    sugiyama_ell_max = sugiyama_ellt_max = 4
    min = None
    if isinstance(ellin, tuple):
        ellin, min = ellin
    for sugiyama_ell in zip(range(sugiyama_ell_max + 1), range(sugiyama_ell_max + 1), [ell]):
        if abs(wigner_3j(*sugiyama_ell, 0, 0, 0)) < 1e-7: continue
        for sugiyama_ellt in zip(range(sugiyama_ellt_max + 1), range(sugiyama_ellt_max + 1), [ellin]):
            if abs(wigner_3j(*sugiyama_ellt, 0, 0, 0)) < 1e-7: continue
            sugiyama_coeffs = get_sugiyama_window_convolution_coeffs(sugiyama_ell, sugiyama_ellt)
            if not sugiyama_coeffs: continue
            if min is not None:
                scoccimarro_to_sugiyama = prod((2 * ell_ + 1) for ell_ in sugiyama_ellt) * wigner_3j(*sugiyama_ellt, 0, 0, 0) * wigner_3j(*sugiyama_ellt, 0, -min, min)
                scoccimarro_to_sugiyama /= jnp.sqrt(4. * jnp.pi * (2 * ellin + 1))
                sugiyama_coeffs = [(ellw, scoccimarro_to_sugiyama * coeff) for ellw, coeff in sugiyama_coeffs]
            coeffs += [(sugiyama_ell, sugiyama_ellt, sugiyama_coeffs)]
    return coeffs


def get_smooth3_window_bin_attrs(ells, ellsin=3, fields=None, return_ellsin: bool=False, basis: str='sugiyama'):
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
                coeffs = get_scoccimarro_window_convolution_coeffs(ell, ellin)
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


def compute_smooth3_spectrum_window(window, edgesin: np.ndarray | tuple, ellsin: tuple=None, bin: BinMesh3SpectrumPoles=None, flags: tuple=None) -> WindowMatrix:
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
    bin : BinMesh2SpectrumPoles
        Output binning.

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

        def _cproduct(array):
            grid = jnp.meshgrid(*array, sparse=False, indexing='ij')
            return jnp.column_stack([tmp.ravel() for tmp in grid])

        # of shape (nbins, ndim, 2)
        edgesin = jnp.concatenate([_cproduct([edge[..., 0] for edge in grid_edgesin])[..., None], _cproduct([edge[..., 1] for edge in grid_edgesin])[..., None]], axis=-1)
        kin = _cproduct(grid_kin)
        if 'soccimarro' in bin.basis:
            mask = (kin[:, 2] >= jnp.abs(kin[:, 0] - kin[:, 1])) & (kin[:, 2] <= jnp.abs(kin[:, 0] + kin[:, 1]))
            edgesin, kin = edgesin[mask], kin[mask]

    if 'sugiyama' in bin.basis:
        ellsin = [(ellin[0], tuple(ellin[1])) if isinstance(ellin[0], tuple) else (ellin, (0, 0)) for ellin in ellsin]
    else:
        ellsin = [(ellin[0], tuple(ellin[1])) if isinstance(ellin, tuple) else (ellin, (0, 0)) for ellin in ellsin]

    kout = bin.xavg

    from .fftlog import SpectrumToCorrelation, CorrelationToSpectrum

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
            m = None
            if isinstance(ell, tuple): ell, m = ell
            toret = (-1)**ell * np.pi**2 / prod(qs) * tophat
            if m is None: toret *= get_legendre(ell)(cos)
            else: toret *= get_Ylm(ell, m, reduced=True, real=True)(jnp.sqrt(1. - cos**2), 0., cos)
            return toret

        for ellin, wain in ellsin:  # ellin = L', M', wain wide-angle order
            wmat_tmp[ellin, wain] = []
            for ill, ell in enumerate(ells):  # ell = L

                # Then sum over \ell_1, \ell_2, \ell_1', \ell_2', L', \ell_1'', \ell_2'', L''
                tmp = jnp.zeros(shape=(len(kout), len(edgesin)))

                for sugiyama_ell, sugiyama_ellt, wcoeffs in get_scoccimarro_window_convolution_coeffs(ell, ellin):
                    # fftlog
                    to_spectrum = CorrelationToSpectrum(s=tuple(next(iter(window)).coords().values()), ell=sugiyama_ell, check_level=1, minfolds=0)
                    to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=sugiyama_ellt, minfolds=0)
                    Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)

                    def convolve(idx):
                        volume = (edgesin[idx, 2, 1]**3 - edgesin[idx, 2, 0]**3) / (6. * jnp.pi**2) * compute_I((sugiyama_ellt[0], -sugiyama_ellt[1]), kin[idx].T)
                        spectrum = tophat(to_spectrum.k, edgesin[idx, :2], volume)
                        correlation = to_correlation(spectrum)[1]
                        correlation = correlation * Qs * to_correlation.s[0][:, None]**wain[0] * to_correlation.s[1][None, :]**wain[1]
                        spectrum = read(kout.T, to_spectrum.k, to_spectrum(correlation)[1])
                        ell2 = sugiyama_ell[1]
                        spectrum *= compute_I(ell2, kout.T) * (-1)**ell2 / compute_I(0, kout.T)
                        return spectrum

                    tmp += jax.lax.map(convolve, jnp.arange(edgesin.shape[0])).T

                wmat_tmp[ellin, wain].append(tmp)

            wmat_tmp[ellin, wain] = jnp.concatenate(wmat_tmp[ellin, wain], axis=0)
    else:

        for ellin, wain in ellsin:  # ellin 3-tuple, wain wide-angle order
            wmat_tmp[ellin, wain] = []
            for ill, ell in enumerate(ells):

                def convolve(idx):
                    spectrum = tophat(to_spectrum.k, edgesin[idx], 1.)
                    correlation = to_correlation(spectrum)[1]
                    correlation = correlation * Qs * to_correlation.s[0][:, None]**wain[0] * to_correlation.s[1][None, :]**wain[1]
                    return read(kout.T, to_spectrum.k, to_spectrum(correlation)[1])

                tmp = jnp.zeros(shape=(len(kout), len(edgesin)))
                # fftlog
                to_spectrum = CorrelationToSpectrum(s=tuple(next(iter(window)).coords().values()), ell=ell, check_level=1, minfolds=0)

                wcoeffs = get_sugiyama_window_convolution_coeffs(ell, ellin)
                Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)
                if wcoeffs:
                    to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=ellin, minfolds=0)
                    tmp += jax.lax.map(convolve, jnp.arange(edgesin.shape[0])).T

                sym_ell1 = tuple(ellin[1::-1]) + ellin[2:]
                # Takes care of symmetry
                if ellin[1] != ellin[0] and (sym_ell1, wain) not in ellsin:
                    wcoeffs = get_sugiyama_window_convolution_coeffs(ell, sym_ell1)
                    if wcoeffs:
                        Qs = sum(coeff * get_w_rect(q, wain) for q, coeff in wcoeffs)
                        to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=sym_ell1, minfolds=0)
                        tmp += jax.lax.map(convolve, jnp.arange(edgesin.shape[0])).T

                wmat_tmp[ellin, wain].append(tmp)

            wmat_tmp[ellin, wain] = jnp.concatenate(wmat_tmp[ellin, wain], axis=0)

    wmat = jnp.concatenate(list(wmat_tmp.values()), axis=1)

    observable = []
    for ill, ell in enumerate(ells):
        observable.append(Mesh3SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes[ill], num_raw=jnp.zeros_like(bin.xavg[..., 0]), basis=bin.basis, ell=ell))
    observable = Mesh3SpectrumPoles(observable)

    theory = []
    kin, edgesin = jnp.mean(edgesin, axis=-1), edgesin
    for ill, ell in enumerate(ellsin):
        theory.append(ObservableLeaf(k=kin, k_edges=edgesin, value=jnp.zeros_like(kin[..., 0]), coords=['k']))
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