r"""
Mesh estimator of the :math:`n`-point spectrum, :math:`n \geq 2`, with every leg binned in :math:`|k|`.

This is the Scoccimarro bispectrum estimator with more legs. Filtering a field to a shell,
:math:`d_i(x) = \mathrm{c2r}[\delta(k) W_i(k)]`, the sum over :math:`x` of the product
:math:`d_1(x) \cdots d_n(x)` collects every closed :math:`n`-gon with one leg in each shell,

.. math::
    \hat P_n(k_1, \ldots, k_n) = \frac{1}{N_n} \sum_{k_1 + \cdots + k_n = 0} \delta(k_1) \cdots \delta(k_n),

with :math:`N_n` the number of such :math:`n`-gons. The costs are :math:`n` inverse transforms per
configuration, and the shell-filtered meshes are shared across configurations.

The intended use is the term-by-term validation of an analytic covariance: the trispectrum enters
:math:`\mathrm{Cov}[P, P]` and :math:`\mathrm{Cov}[B, B]`, the 5-point :math:`\mathrm{Cov}[P, B]`
and the 6-point :math:`\mathrm{Cov}[B, B]`. Two properties of the estimator matter for that.

* **On disjoint shells the Gaussian disconnected part vanishes identically.** Every pairing
  carries :math:`\delta_D(k_i + k_j)`, hence :math:`|k_i| = |k_j|`, which distinct shells forbid.
  So the default binning keeps strictly increasing shells only: at :math:`n = 4` and :math:`5`
  nothing has to be subtracted, and at :math:`n = 6` only the :math:`B B` pairing (two closed
  triangles) survives, to be removed at the ensemble level from the measured triple products.
  Repeated shells are allowed through ``mask_edges``; the disconnected terms are then present.

* **The collapsed configurations** :math:`T(k, -k, k', -k')` **cannot be isolated**: on a single
  realization that estimator reduces, by Parseval, to a product of binned power spectra, i.e. to
  the sample covariance itself.

Multipoles are Legendre weights on the last leg with a global line of sight, as in the
Scoccimarro basis of :mod:`mesh3`; the default is the monopole.

The estimator normalization is the box one, :math:`N_\mathrm{mesh}^n / V^{n-1}`, the direct
generalization of :func:`mesh3.compute_mesh3_spectrum` (:math:`N_\mathrm{mesh}^3 / V^2`).
"""

import itertools
import operator
import functools
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from dataclasses import dataclass

from .mesh import (BaseMeshField, MeshAttrs, RealMeshField, ComplexMeshField, ParticleField, staticarray, _get_hermitian_weights, _find_unique_edges, _get_bin_attrs,
compute_box_normalization, __format_meshes)
from .mesh3 import _format_los
from .types import MeshNSpectrumPole, MeshNSpectrumPoles
from .utils import get_legendre, register_pytree_dataclass


prod = partial(functools.reduce, operator.mul)


def _default_mask_edges(ndim, basis):
    """Default configuration mask: strictly increasing shells, polygon closure, no aliasing."""
    if 'diagonal' in basis:
        return []
    legs = [f'mid{i + 1:d}' for i in range(ndim)]
    mask_edges = [f'{legs[i]} < {legs[i + 1]}' for i in range(ndim - 1)]
    # A closed polygon: the longest leg is at most the sum of the others
    mask_edges += [f'2 * {legs[-1]} <= ' + ' + '.join(legs)]
    # No aliasing: if k1 + ... + kn = G != 0 on the periodic grid then sum |k_i| >= |G| >= 2 nyq
    mask_edges += ['(' + ' + '.join(f'edge{i + 1:d}' for i in range(ndim)) + ')[:, 1] <= 2 * nyq']
    return mask_edges


def _make_edgesn(mattrs, edges, ells, ndim, basis='scoccimarro', batch_size=None, buffer_size=None, mask_edges=None):
    assert basis in ['scoccimarro', 'scoccimarro-diagonal'], basis
    assert ndim >= 2, 'ndim must be >= 2'
    ells = _format_ells(ells)

    default_mask = mask_edges is None
    if default_mask:
        mask_edges = _default_mask_edges(ndim, basis)
    if isinstance(mask_edges, str):
        mask_edges = tuple(c for c in mask_edges.strip().split(';'))

    vec = mattrs.kcoords(kind='separation', sparse=True)
    vec0 = mattrs.kfun.min()
    wmodes = None
    if mattrs.is_hermitian:
        wmodes = _get_hermitian_weights(vec, sharding_mesh=None)
    vecmax = vec0 * np.min(mattrs.meshsize) / 2.

    if isinstance(mask_edges, (list, tuple)):
        mask_edges_list = mask_edges

        def mask_edges(*edges):
            mask = np.ones(edges[0].shape[0], dtype=bool)
            d = {'nyq': vecmax}
            for i, edge in enumerate(edges):
                d[f'mid{i + 1:d}'] = np.mean(edge, axis=-1)
                d[f'edge{i + 1:d}'] = edge
            for c in mask_edges_list:
                c = c.strip()
                if not c: continue
                mask &= np.asarray(eval(c, {'np': np, 'jnp': jnp}, d))
            return mask

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
        edges[iedge] = np.asarray(edge)
    edges = edges + [edges[-1]] * (ndim - len(edges))
    same_edges = all(len(edge) == len(edges[0]) and np.allclose(edge, edges[0]) for edge in edges[1:])

    coords = jnp.sqrt(sum(xx**2 for xx in vec))
    ibin1d, nmodes1d, xavg1d, edges1d = [], [], [], []
    for edge in edges:
        ib, nm, x = _get_bin_attrs(coords, edge, wmodes, ravel=False)
        ib = ib - 1
        x /= nm
        ibin1d.append(ib)
        nmodes1d.append(nm)
        xavg1d.append(x)
        edges1d.append(np.column_stack([edge[:-1], edge[1:]]))

    # Configurations: (nconf, ndim) array of 1d bin indices. With the default mask and equal
    # edges on every axis the strictly-increasing condition is a combination, which is far
    # smaller than the full product (15 shells at n = 6: 5005 rows against 11 million).
    nbins1d = [len(edge) - 1 for edge in edges]
    if default_mask and same_edges and 'diagonal' not in basis:
        iedges = np.array(list(itertools.combinations(range(nbins1d[0]), ndim)), dtype='i8').reshape(-1, ndim)
    elif 'diagonal' in basis:
        iedges = np.repeat(np.arange(nbins1d[0])[:, None], ndim, axis=1)
    else:
        grid = np.meshgrid(*[np.arange(nb) for nb in nbins1d], sparse=False, indexing='ij')
        iedges = np.column_stack([tmp.ravel() for tmp in grid])
    list_edges = [edges1d[axis][iedges[:, axis]] for axis in range(ndim)]
    mask = mask_edges(*list_edges)
    iedges = iedges[mask]
    assert len(iedges), 'no configuration survives mask_edges'
    edges = jnp.asarray(np.stack([edges1d[axis][iedges[:, axis]] for axis in range(ndim)], axis=1))  # (nconf, ndim, 2)
    xavg = jnp.stack([jnp.asarray(xavg1d[axis])[iedges[:, axis]] for axis in range(ndim)], axis=1)  # (nconf, ndim)
    nmodes = jnp.prod(jnp.stack([jnp.asarray(nmodes1d[axis])[iedges[:, axis]] for axis in range(ndim)], axis=1), axis=-1)
    nmodes = [nmodes] * len(ells)

    if batch_size is None:
        batch_size = 1

    # Which unique 1d bins are needed on each axis. These, and the configurations, are static
    # (numpy) so that the python-level grouping of legs in bin_reduce_meshs works under jit.
    uiedges = [np.unique(iedges[:, axis]) for axis in range(ndim)]

    return dict(edges=edges, ibin1d=tuple(ibin1d), edges1d=tuple(jnp.asarray(edge) for edge in edges1d), nmodes1d=tuple(nmodes1d), xavg1d=tuple(xavg1d),
                nmodes=nmodes, xavg=xavg, wmodes=wmodes, mattrs=mattrs, basis=basis, batch_size=batch_size, buffer_size=buffer_size,
                _iedges=staticarray(iedges), _uiedges=tuple(staticarray(u) for u in uiedges),
                _same_edges=same_edges, ells=ells, ndim=ndim)


def _format_ells(ells):
    """Format multipole orders: (a list of) integers, Legendre weights on the last leg."""
    if np.ndim(ells) == 0:
        ells = [ells]
    ells = [int(ell) for ell in ells]
    return ells


@partial(register_pytree_dataclass, meta_fields=['basis', 'batch_size', 'buffer_size', 'ells', 'ndim', '_same_edges', '_iedges', '_uiedges'])
@dataclass(init=False, frozen=True)
class BinMeshNSpectrumPoles(object):
    r"""
    Binning operator for 3D mesh to :math:`n`-point spectrum, all :math:`n` legs binned in :math:`|k|`.

    Parameters
    ----------
    mattrs : MeshAttrs or BaseMeshField
        Mesh attributes or mesh field.
    edges : array-like, dict, list, or None, optional
        Bin edges or binning configuration, shared by all legs; or a list of one per leg.
    ells : int or tuple, optional
        Multipole orders: Legendre weights on the last leg, global line of sight.
    ndim : int, optional
        Number of legs :math:`n`.
    basis : str, optional
        'scoccimarro' (all bin combinations, subject to ``mask_edges``) or 'scoccimarro-diagonal'
        (all legs in the same bin).
    batch_size : int, optional
        Batch size for JAX mapping.
    buffer_size : int, optional
        Number of shell-filtered meshes that can be kept in memory.
        ``None`` (default) keeps every shell that is needed, so each is transformed once; ``0``
        recomputes the :math:`n` transforms for every configuration.
    mask_edges : str, list of str, or callable, optional
        Selection of the bin configurations, as conditions on ``mid{i}`` (bin centers),
        ``edge{i}`` (bin edges, of shape (nbins, 2)) and ``nyq``. The default keeps strictly
        increasing shells, ``mid1 < mid2 < ... < midn``, on which the Gaussian disconnected
        part of the estimator vanishes identically, closed polygons, and configurations that
        the periodic grid does not alias (the sum of the upper edges below twice the Nyquist
        frequency). Pass e.g. ``['mid1 <= mid2', 'mid2 <= mid3', 'mid3 <= mid4']`` to include
        repeated shells; the disconnected terms are then part of the measurement.
    """

    edges: jax.Array = None
    nmodes1d: jax.Array = None
    edges1d: tuple = None
    xavg1d: tuple = None
    ibin1d: tuple = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    wmodes: jax.Array = None
    _iedges: staticarray = None
    _uiedges: tuple = None
    _same_edges: bool = True
    mattrs: MeshAttrs = None
    basis: str = 'scoccimarro'
    batch_size: int = 1
    buffer_size: int = None
    ells: tuple = None
    ndim: int = 4

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, ells=0, ndim=4, basis='scoccimarro', batch_size=None, buffer_size=None, mask_edges=None):
        if not isinstance(mattrs, MeshAttrs):
            mattrs = mattrs.attrs
        kw = _make_edgesn(mattrs, edges, ells, ndim, basis=basis, batch_size=batch_size, buffer_size=buffer_size, mask_edges=mask_edges)
        self.__dict__.update(kw)

        # Number of closed n-gons with one leg in each shell
        def bin(axis, ibin):
            return mattrs.c2r((self.ibin1d[axis] == ibin).astype(mattrs.cdtype))

        def reduce(meshes):
            return prod(meshes).sum()

        nmodes = self.bin_reduce_meshs(bin, reduce, shared=self._same_edges) * self.mattrs.meshsize.prod(dtype=self.mattrs.rdtype)**(self.ndim - 1)
        nmodes = [nmodes] * len(self.ells)
        self.__dict__.update(nmodes=nmodes)

    def bin_reduce_meshs(self, bin, reduce, shared=None):
        """
        Reduce over mesh bins using provided binning and reduction functions.

        Parameters
        ----------
        bin : callable
            Function ``bin(axis, ibin)`` returning the shell-filtered mesh of 1d bin ``ibin`` on leg ``axis``.
        reduce : callable
            Function to reduce the :math:`n` binned meshes.
        shared : bool or list, optional
            Which legs share the same shell-filtered meshes (same input mesh and same edges):
            ``True`` for all of them, or a list of group labels, one per leg. Defaults to
            ``self._same_edges``, i.e. equal edges are taken to mean equal inputs.

        Returns
        -------
        result : array-like
            Reduced mesh values, one per configuration.
        """
        ndim = self.ndim
        if shared is None:
            shared = self._same_edges
        if isinstance(shared, bool):
            shared = [0] * ndim if shared else list(range(ndim))
        shared = list(shared)

        if self.buffer_size is not None and self.buffer_size == 0:

            def f(ibin):
                meshes = (bin(axis, ibin_) for axis, ibin_ in enumerate(ibin))
                return reduce(meshes)

            return jax.lax.map(f, jnp.asarray(self._iedges), batch_size=self.batch_size)

        # Shell-filtered meshes, computed once per group of legs
        groups = {}
        for axis, label in enumerate(shared):
            groups.setdefault(label, []).append(axis)
        stacks, labels, index = {}, [None] * ndim, [None] * ndim
        for label, axes in groups.items():
            uiedges = np.unique(np.concatenate([np.asarray(self._uiedges[axis]) for axis in axes]))
            stacks[label] = jax.lax.map(partial(bin, axes[0]), jnp.asarray(uiedges), batch_size=self.batch_size)
            for axis in axes:
                labels[axis] = label
                index[axis] = np.searchsorted(uiedges, np.asarray(self._iedges)[:, axis])
        iedges = jnp.asarray(np.stack(index, axis=1))

        def f_reduce(idx):
            meshes = (stacks[labels[axis]][idx[axis]] for axis in range(ndim))
            return reduce(meshes)

        return jax.lax.map(f_reduce, iedges, batch_size=self.batch_size)

    def __call__(self, *meshes, remove_zero=False, shared=None):
        """
        Bin and reduce input meshes to compute the :math:`n`-point spectrum.

        Parameters
        ----------
        meshes : array-like or BaseMeshField
            Input (Fourier-space) meshes, one per leg.
        remove_zero : bool, optional
            Whether to remove the zero mode.
        shared : bool or list, optional
            See :meth:`bin_reduce_meshs`. Defaults to grouping legs whose input arrays are the
            same object and whose edges are the same.

        Returns
        -------
        binned : array-like
        """
        values = []
        ndim = self.ndim
        assert len(meshes) == ndim, f'expected {ndim} meshes, got {len(meshes)}'
        norm = self.mattrs.meshsize.prod(dtype=self.mattrs.rdtype)**(ndim - 1)
        for mesh in meshes:
            value = mesh.value if isinstance(mesh, BaseMeshField) else mesh
            if remove_zero:
                value = value.at[(0,) * value.ndim].set(0.)
            values.append(value)

        if shared is None:
            shared, seen = [], []
            for axis, value in enumerate(values):
                key = (id(value), axis if not self._same_edges else 0)
                if key not in seen: seen.append(key)
                shared.append(seen.index(key))

        def bin(axis, ibin):
            return self.mattrs.c2r(values[axis] * (self.ibin1d[axis] == ibin))

        def reduce(meshes):
            return prod(meshes).sum()

        return self.bin_reduce_meshs(bin, reduce, shared=shared) * norm


def compute_meshn_spectrum(*meshes: RealMeshField | ComplexMeshField, bin: BinMeshNSpectrumPoles=None, los: str | np.ndarray='z'):
    r"""
    Compute the :math:`n`-point spectrum multipoles from mesh, :math:`n` set by ``bin.ndim``.

    Parameters
    ----------
    meshes : RealMeshField or ComplexMeshField
        Input meshes: one, or one per leg.
    bin : BinMeshNSpectrumPoles
        Binning operator.
    los : str or array-like, optional
        Global line of sight: 'x', 'y' or 'z', or a 3-vector. Only enters the multipoles
        :math:`\ell > 0`, as a Legendre weight on the last leg.

    Note
    ----
    jit is recommended when running for many realizations.
    Otherwise OOM may occur.

    Returns
    -------
    result : MeshNSpectrumPoles
    """
    ndim = bin.ndim
    meshes, fields = __format_meshes(*meshes, nmeshes=ndim)
    rdtype = meshes[0].real.dtype
    mattrs = meshes[0].attrs
    meshes = [mesh.clone(attrs=mattrs) for mesh in meshes]
    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)**(ndim - 1)

    los, vlos = _format_los(los, ndim=mattrs.ndim)
    if vlos is None:
        raise NotImplementedError('local line of sight is not supported by the n-point estimator')
    attrs = dict(los=vlos)
    ells = bin.ells

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    meshes = [_2c(mesh) for mesh in meshes]
    # Legs sharing the same field and the same edges share their shell-filtered meshes
    shared = [fields[axis] if bin._same_edges else axis for axis in range(ndim)]
    shared_ell = list(shared[:-1]) + [max(shared) + 1]

    kvec = mattrs.kcoords(sparse=True)
    mu = None
    if any(ell != 0 for ell in ells):
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)

    num = []
    for ill, ell in enumerate(ells):
        if ell == 0:
            tmp = bin(*meshes, shared=shared)
        else:
            tmp = meshes[:-1] + [meshes[-1] * get_legendre(ell)(mu)]
            tmp = (2 * ell + 1) * bin(*tmp, shared=shared_ell)
        tmp = tmp / bin.nmodes[ill]
        num.append(tmp.real if ell % 2 == 0 else tmp.imag)

    spectrum = []
    for ill, ell in enumerate(ells):
        spectrum.append(MeshNSpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes[ill], num_raw=num[ill], norm=norm, attrs=attrs, basis=bin.basis, ell=ell))
    return MeshNSpectrumPoles(spectrum)


def _compute_meshn_spectrum_fixed(ndim):

    def compute(*meshes: RealMeshField | ComplexMeshField, bin: BinMeshNSpectrumPoles=None, los: str | np.ndarray='z'):
        assert bin.ndim == ndim, f'bin.ndim must be {ndim:d}, got {bin.ndim}'
        return compute_meshn_spectrum(*meshes, bin=bin, los=los)

    compute.__name__ = f'compute_mesh{ndim:d}_spectrum'
    compute.__doc__ = rf"""
    Compute the {ndim:d}-point spectrum multipoles from mesh. See :func:`compute_meshn_spectrum`.

    Parameters
    ----------
    meshes : RealMeshField or ComplexMeshField
        Input meshes: one, or one per leg.
    bin : BinMeshNSpectrumPoles
        Binning operator, with ``ndim={ndim:d}``.
    los : str or array-like, optional
        Global line of sight: 'x', 'y' or 'z', or a 3-vector.

    Returns
    -------
    result : MeshNSpectrumPoles
    """
    return compute


compute_mesh4_spectrum = _compute_meshn_spectrum_fixed(4)
compute_mesh5_spectrum = _compute_meshn_spectrum_fixed(5)
compute_mesh6_spectrum = _compute_meshn_spectrum_fixed(6)


def compute_boxn_normalization(*inputs: RealMeshField | ParticleField, bin: BinMeshNSpectrumPoles=None) -> jax.Array:
    """Compute normalization, assuming constant density, for the :math:`n`-point spectrum."""
    nmeshes = bin.ndim if bin is not None else None
    norm = compute_box_normalization(*__format_meshes(*inputs, nmeshes=nmeshes)[0])
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm
