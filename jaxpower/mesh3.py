import itertools
import operator
import functools
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from dataclasses import dataclass

from .mesh import BaseMeshField, MeshAttrs, RealMeshField, ComplexMeshField, staticarray, get_sharding_mesh, _get_hermitian_weights, _find_unique_edges, _get_bin_attrs, _bincount, create_sharded_random
from .mesh2 import _get_los_vector, _get_zero, FKPField
from .types import Mesh3SpectrumPole, Mesh3SpectrumPoles, WindowMatrix
from .utils import real_gaunt, get_legendre, get_spherical_jn, get_real_Ylm, register_pytree_dataclass


prod = functools.partial(functools.reduce, operator.mul)


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
        Binning basis ('sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-equilateral').
    batch_size : int, optional
        Batch size for JAX mapping.
    buffer_size : int, optional
        Buffer size for chunked binning: number of meshes that van be kept into memory.
    """

    edges: jax.Array = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    ibin: jax.Array = None
    wmodes: jax.Array = None
    _iedges: jax.Array = None
    _buffer_iedges: tuple = None
    _nmodes1d: jax.Array = None
    mattrs: MeshAttrs = None
    basis: str = 'sugiyama'
    batch_size: int = 1
    buffer_size: int = 0
    ells: tuple = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, ells=0, basis='sugiyama', batch_size=1, buffer_size=0):
        kind = 'complex'
        if not isinstance(mattrs, MeshAttrs):
            kind = 'complex' if 'complex' in mattrs.__class__.__name__.lower() else 'real'
            mattrs = mattrs.attrs
        hermitian = mattrs.hermitian
        if kind == 'real':
            vec = mattrs.xcoords(kind='separation', sparse=True)
            vec0 = mattrs.cellsize.min()
        else:
            vec = mattrs.kcoords(kind='separation', sparse=True)
            vec0 = mattrs.kfun.min()
        wmodes = None
        if hermitian:
            wmodes = _get_hermitian_weights(vec, sharding_mesh=None)
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            step = edges.get('step', None)
            if step is None:
                edges = _find_unique_edges(vec, vec0, xmin=edges.get('min', 0.), xmax=edges.get('max', np.inf))
            else:
                edges = np.arange(edges.get('min', 0.), edges.get('max', vec0 * np.min(mattrs.meshsize) / 2.), step)

        assert basis in ['sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-equilateral']
        if 'scoccimarro' in basis: ndim = 3
        else: ndim = 2

        if not isinstance(edges, (tuple, list)):
            edges = [edges] * ndim

        coords = jnp.sqrt(sum(xx**2 for xx in vec))
        ibin, nmodes1d, xavg, uedges = [], [], [], []
        for edge in edges:
            ib, nm, x = _get_bin_attrs(coords, edge, wmodes, ravel=False)
            ib = ib - 1
            #ib = ib.reshape(coords.shape) - 1
            #sharding_mesh = mattrs.sharding_mesh
            #if sharding_mesh.axis_names:
            #    ib = jax.lax.with_sharding_constraint(ib, jax.sharding.NamedSharding(sharding_mesh, spec=jax.sharding.PartitionSpec(*sharding_mesh.axis_names)))
            x /= nm
            ibin.append(ib)
            nmodes1d.append(nm)
            xavg.append(x)
            uedges.append(jnp.column_stack([edge[:-1], edge[1:]]))
        uedges = uedges + [uedges[-1]] * (ndim - len(uedges))
        ibin = ibin + [ibin[-1]] * (ndim - len(ibin))
        nmodes1d = nmodes1d + [nmodes1d[-1]] * (ndim - len(nmodes1d))
        xavg = xavg + [xavg[-1]] * (ndim - len(xavg))

        def _product(array):
            if not isinstance(array, (tuple, list)):
                array = [array] * ndim
            if 'diagonal' in basis or 'equilateral' in basis:
                grid = [jnp.array(array[0])] * ndim
            else:
                grid = jnp.meshgrid(*array, sparse=False, indexing='ij')
            return jnp.column_stack([tmp.ravel() for tmp in grid])

        def get_order_mask(edges):
            xmid = _product([jnp.mean(edge, axis=-1) for edge in edges])
            mask = True
            for i in range(xmid.shape[1] - 1): mask &= xmid[:, i] <= xmid[:, i + 1]  # select k1 <= k2 <= k3...
            return mask

        mask = get_order_mask(uedges)
        # of shape (nbins, ndim, 2)
        edges = jnp.concatenate([_product([edge[..., 0] for edge in uedges])[..., None], _product([edge[..., 1] for edge in uedges])[..., None]], axis=-1)[mask]
        xavg = _product(xavg)[mask]
        nmodes = jnp.prod(_product(nmodes1d)[mask], axis=-1)
        iedges = _product([jnp.arange(len(xx)) for xx in nmodes1d])[mask]

        if 'scoccimarro' in basis:
            xmid = jnp.mean(edges, axis=-1).T
            mask = (xmid[2] >= jnp.abs(xmid[0] - xmid[1])) & (xmid[2] <= jnp.abs(xmid[0] + xmid[1]))
            xmid, edges, xavg, nmodes, iedges = xmid[..., mask], edges[mask], xavg[mask], nmodes[mask], iedges[mask]

        nmodes = [nmodes] * len(ells)

        if 'diagonal' in basis: buffer_size = 0

        if buffer_size > 1:
            split_edges, split_iedges = [], []
            for axis in range(ndim):
                axis_iedges = jnp.arange(len(uedges[axis]))
                nsplits = (len(axis_iedges) + buffer_size - 1) // buffer_size
                split_edges.append(jnp.array_split(uedges[axis], nsplits, axis=0))
                split_iedges.append(jnp.array_split(axis_iedges, nsplits, axis=0))
            _buffer_global_iedges, _buffer_iedges, _buffer_iuedges = [], [], []
            size_max, usize_max = 0, 0
            for biuedges, buedges in zip(itertools.product(*split_iedges), itertools.product(*split_edges)):
                mask = get_order_mask(buedges)
                if mask.sum():
                    _buffer_global_iedges.append(_product(biuedges)[mask])
                    _buffer_iedges.append(_product([jnp.arange(len(iedge)) for iedge in biuedges])[mask])
                    _buffer_iuedges.append(biuedges)
                    size_max = max(size_max, len(_buffer_iedges[-1]))
                    usize_max = max(usize_max, *[len(b) for b in _buffer_iuedges[-1]])

            #print('Number of FFTs', sum(sum(len(bb) for bb in b) for b in _buffer_iuedges))
            # Pad to be able to use jax.lax.map, otherwise compilation time is prohibitive
            for i in range(len(_buffer_global_iedges)):
                _buffer_global_iedges[i] = jnp.pad(_buffer_global_iedges[i], [(0, size_max - len(_buffer_global_iedges[i])), (0, 0)], mode='edge')
                _buffer_iedges[i] = jnp.pad(_buffer_iedges[i], [(0, size_max - len(_buffer_iedges[i])), (0, 0)], mode='edge')
                _buffer_iuedges[i] = jnp.stack([jnp.pad(b, (0, usize_max - len(b)), mode='edge') for b in _buffer_iuedges[i]])

            #print('Number of FFTs padded', sum(sum(len(bb) for bb in b) for b in _buffer_iuedges))
            _buffer_global_iedges = jnp.concatenate(_buffer_global_iedges, axis=0)
            _buffer_sort = jnp.array([jnp.flatnonzero(jnp.all(iedge == _buffer_global_iedges, axis=1))[0] for iedge in iedges])
            # _buffer_iedges = (N-dim bins, corresponding unique bins along each dim, how to sort the N-dim bins to obtain the requested bins)
            _buffer_iedges = (jnp.stack(_buffer_iedges), jnp.stack(_buffer_iuedges), _buffer_sort)
        else:
            _buffer_iedges = None


        ells = _format_ells(ells, basis=basis)
        self.__dict__.update(edges=edges, xavg=xavg, nmodes=nmodes, ibin=ibin, wmodes=wmodes, mattrs=mattrs, basis=basis, batch_size=batch_size, buffer_size=buffer_size, _iedges=iedges, _buffer_iedges=_buffer_iedges, _nmodes1d=nmodes1d, ells=ells)

        if 'scoccimarro' in basis:
            symfactor = jnp.ones_like(xmid[0])
            symfactor = jnp.where((xmid[1] == xmid[0]) | (xmid[2] == xmid[0]) | (xmid[2] == xmid[1]), 2, symfactor)
            symfactor = jnp.where((xmid[1] == xmid[0]) & (xmid[2] == xmid[0]), 6, symfactor)

            def bin(axis, ibin):
                return mattrs.c2r(self.ibin[axis] == ibin)

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
            return self.mattrs.c2r(values[axis] * (self.ibin[axis] == ibin))

        def reduce(meshes):
            return prod(meshes, 1. if 'scoccimarro' in self.basis else values[2]).sum()

        return self.bin_reduce_meshs(bin, reduce) * norm


def _format_meshs(*meshes):
    """Format input meshes for autocorrelation/cross-correlation: return list of 3 meshes,
    and a list of indices corresponding to the mesh they are equal to."""
    meshes = list(meshes)
    meshes = meshes + [None] * (3 - len(meshes))
    same = [0]
    for mesh in meshes[1:]: same.append(same[-1] if mesh is None else same[-1] + 1)
    for imesh, mesh in enumerate(meshes):
        if mesh is None:
            meshes[imesh] = meshes[imesh - 1]
    return meshes, tuple(same)


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
    for im1, im2, im3 in itertools.product(*ms):
        gaunt = real_gaunt((ell1, im1), (ell2, im2), (ell3, im3))
        if gaunt == 0.: continue
        sym = 0.
        neg = (-im1, -im2, -im3)
        if neg in acc:
            idx = acc.index(neg)
            toret[idx][-1] = (-1)**ell3
            continue
        toret.append([im1 + ell1, im2 + ell2, im3 + ell3, gaunt, sym])  # m indexing starting from 0
        acc.append(toret[-1][:3])
    if toret:
        return [jnp.array(xx) for xx in zip(*toret)]
    return [jnp.zeros((0,), dtype=int) for _ in range(5)]


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
    meshes, same = _format_meshs(*meshes)
    rdtype = meshes[0].real.dtype
    mattrs = meshes[0].attrs
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
                Ylms = [get_real_Ylm(ell3, m) for m in range(-ell3, ell3 + 1)]
                xs = np.arange(len(Ylms))
                tmp = tuple(meshes[i] for i in range(2)) + (jax.lax.scan(partial(f, Ylms), init=meshes[0].clone(value=jnp.zeros_like(meshes[0].value)), xs=xs)[0],)
                tmp = (4. * np.pi) * bin(*tmp, remove_zero=True) / bin.nmodes[ill3]
                num.append(tmp)

        else:

            meshes = [_2c(mesh) for mesh in meshes]
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)

            for ill3, ell3 in enumerate(ells):
                tmp = meshes[:2] + [meshes[2] * get_legendre(ell3)(mu)]
                tmp = (2 * ell3 + 1) * bin(*tmp, remove_zero=True) / bin.nmodes[ill3]
                num.append(tmp)

    else:

        meshes = [_2c(mesh) for mesh in meshes[:2]] + [_2r(meshes[2])]

        @partial(jax.checkpoint, static_argnums=0)
        def f(Ylm, carry, im):
            coeff, sym, im = im[3], im[4], im[:3]
            tmp = tuple(meshes[i] * jax.lax.switch(im[i], Ylm[i], *kvec) for i in range(2))
            los = xvec if vlos is None else vlos
            tmp += (jax.lax.switch(im[2], Ylm[2], *los) * meshes[2],)
            tmp = (4. * np.pi)**2 * coeff.astype(mattrs.rdtype) * bin(*tmp, remove_zero=True)
            carry += tmp + sym.astype(mattrs.rdtype) * tmp.conj()
            return carry, im

        for ill, (ell1, ell2, ell3) in enumerate(ells):
            Ylms = [[get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in (ell1, ell2, ell3)]
            xs = _iter_triposh(ell1, ell2, ell3, los=los)
            if xs[0].size:
                num.append(jax.lax.scan(partial(f, Ylms), init=jnp.zeros(len(bin.edges), dtype=mattrs.dtype), xs=xs)[0] / bin.nmodes[ill])
                #num.append(bin(*meshes, remove_zero=True) / bin.nmodes[ill])
            else:
                num.append(jnp.zeros(len(bin.edges), dtype=mattrs.dtype))

    spectrum = []
    for ill, ell in enumerate(ells):
        spectrum.append(Mesh3SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes[ill], num_raw=num[ill], norm=norm, attrs=attrs, basis=bin.basis, ell=ell))
    return Mesh3SpectrumPoles(spectrum)


from .mesh2 import compute_normalization


def _split_particles(*particles, seed=0):
    """
    Split input particles for estimation of the bispectrum normalization.

    Parameters
    ----------
    particles : ParticleField or None
        Input particles.
    seed : int, optional
        Random seed.

    Returns
    -------
    toret : list
        List of split particles.
    """
    toret = list(particles)
    particles_to_split, nsplits = [], 0
    # Loop in reverse order
    for particle in particles[::-1]:
        if particle is None:
            nsplits += 1
        else:
            particles_to_split.append((particle, nsplits + 1))
            nsplits = 0
    # Reorder
    particles_to_split = particles_to_split[::-1]
    # remove last one
    if isinstance(seed, int):
        seed = random.key(seed)
    seeds = random.split(seed, len(particles_to_split))
    toret = []
    for i, (particle, nsplits) in enumerate(particles_to_split):
        if nsplits == 1:
            toret.append(particle)
        else:
            x = create_sharded_random(random.uniform, seeds[i], particle.size, out_specs=0)
            for isplit in range(nsplits):
                mask = (x >= isplit / nsplits) & (x < (isplit + 1) / nsplits)
                toret.append(particle.clone(weights=particle.weights * mask))
    return toret


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
    fkps =  list(fkps) + [None] * (3 - len(fkps))
    if split is not None:
        randoms = _split_particles(*[fkp.randoms if fkp is not None else fkp for fkp in fkps], seed=split)
        fkps, same = _format_meshs(*fkps)
        fkps = [fkp.clone(randoms=randoms) for fkp, randoms in zip(fkps, randoms)]
    else:
        fkps, same = _format_meshs(*fkps)
    kw = dict(cellsize=cellsize)
    for name in list(kw):
        if kw[name] is None: kw.pop(name)
    alpha = prod(map(lambda fkp: fkp.data.sum() / fkp.randoms.sum(), fkps))
    norm = alpha * compute_normalization(*[fkp.randoms for fkp in fkps], **kw)
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_fkp3_shotnoise(*fkps, bin=None, los: str | np.ndarray='z', convention=None, resampler='cic', interlacing=False, **kwargs):
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
    convention : str, optional
        Shot noise convention.
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
    fkps, same = _format_meshs(*fkps)
    # ells
    ells = bin.ells
    shotnoise = [jnp.zeros_like(bin.xavg[..., 0]) for ill in range(len(ells))]
    kwargs.update(resampler=resampler, interlacing=interlacing)

    from . import resamplers
    resampler = resamplers.get_resampler(resampler)
    interlacing = max(interlacing, 1) >= 2

    def bin_mesh2_spectrum(mesh, axis):
        nmodes1d = bin._nmodes1d[axis]
        return _bincount(bin.ibin[axis] + 1, getattr(mesh, 'value', mesh), weights=bin.wmodes, length=len(nmodes1d)) / nmodes1d

    if same[2] == same[1] + 1 == same[0] + 2:
        return tuple(shotnoise)

    particles = []
    for fkp, s in zip(fkps, same):
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

        # Eq. 58 of https://arxiv.org/pdf/1506.02729, 1 => 3
        if not (same[0] == same[1] == same[2]):
            raise NotImplementedError
        cmeshw = particles[0].paint(**kwargs, out='complex')
        cmeshw = cmeshw.clone(value=cmeshw.value.at[(0,) * cmeshw.ndim].set(0.))  # remove zero-mode
        cmeshw2 = particles[0].clone(weights=particles[0].weights**2).paint(**kwargs, out='complex')
        sumw3 = jnp.sum(particles[0].weights**3)
        del particles

        ndim = 3
        def apply_fourier_legendre(ell, cmesh):
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)
            return get_legendre(ell)(mu) * cmesh

        def apply_fourier_harmonics(ell, rmesh):
            rmesh = getattr(rmesh, 'value', rmesh)
            Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]

            @partial(jax.checkpoint, static_argnums=(0, 1))
            def f(Ylm, carry, im):
                carry += mattrs.r2c(rmesh * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            xs = np.arange(len(Ylms))
            return (4. * jnp.pi) * jax.lax.scan(partial(f, Ylms), init=mattrs.create(fill=0., kind='complex'), xs=xs)[0]

        for ill, ell in enumerate(ells):
            if ell == 0:
                cmeshw3_ell = cmeshw * cmeshw2.conj()
                shotnoise[ill] = sum(bin_mesh2_spectrum(cmeshw3_ell, axis=axis)[bin._iedges[..., axis]] for axis in range(ndim)) - 2. * sumw3
            else:
                # First line of eq. 58, q1 => q3
                if vlos is not None:
                    cmeshw_ell = apply_fourier_legendre(ell, cmeshw)
                else:
                    cmeshw_ell = apply_fourier_harmonics(ell, cmeshw.c2r())
                sn_ell = bin_mesh2_spectrum(cmeshw_ell * cmeshw2.conj(), axis=ndim - 1)[bin._iedges[..., ndim - 1]]
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
                    tmp = [bin_mesh2_spectrum(tmp, axis=axis) for axis in range(ndim - 1)] + [bin_mesh2_spectrum(Ylm, axis=ndim - 1)]
                    carry += (4 * jnp.pi) * sum(tmp[axis][bin._iedges[..., axis]] * tmp[ndim - 1][bin._iedges[..., ndim - 1]] for axis in range(ndim - 1))
                    return carry, im

                Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
                xs = np.arange(len(Ylms))
                shotnoise[ill] = (2 * ell + 1) * jax.lax.scan(partial(f, Ylms), init=sn_ell, xs=xs)[0]

    else:
        # Eq. 45 - 46 of https://arxiv.org/pdf/1803.02132

        def compute_S111(particles, ellms):
            if convention == 'triumvirate':
                s0 = jnp.sum(particles[0].weights**3) / jnp.sqrt(4. * jnp.pi)
                s111 = [s0 if (ell, m) == (0, 0) else 0. for ell, m in ellms]
            else:
                rmesh = particles[0].clone(weights=particles[0].weights**3).paint(**kwargs, out='real')
                s111 = [jnp.sum(rmesh.value * get_real_Ylm(ell, m)(*xvec)) for ell, m in ellms]
            return s111

        def compensate_shotnoise(s111):
            # NOTE: when there is no interlacing, triumvirate compensates pk with aliasing_shotnoise in the shotnoise estimation (but in bk estimation?)
            # In jaxpower, we always compensate by the standard compensate
            #if convention == 'triumvirate' and not interlacing:
            if not interlacing:
                return s111 * resampler.aliasing_shotnoise(1., kcirc) * resampler.compensate(1., kcirc)**2
            return s111

        def compute_S122(particles, ells, axis):  # 1 == 2
            rmesh = particles[1].clone(weights=particles[1].weights**2).paint(**kwargs, out='real')
            cmesh = particles[0].paint(**kwargs, out='complex')

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                im, s111 = im
                # Second and third lines
                s111 = compensate_shotnoise(s111)
                los = xvec if vlos is None else vlos
                carry += jax.lax.switch(im, Ylm, *kvec) * ((rmesh * jax.lax.switch(im, Ylm, *los)).r2c() * cmesh.conj() - s111)
                return carry, im

            s122 = []
            for ell in ells:
                Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
                xs = (jnp.arange(len(Ylms)), jnp.array([s111[ellms.index((ell, m))] for m in range(-ell, ell + 1)]))
                s122.append(4. * jnp.pi * bin_mesh2_spectrum(jax.lax.scan(partial(f, Ylms), init=cmesh.clone(value=jnp.zeros_like(cmesh.value)), xs=xs)[0], axis))
            return s122

        def compute_S113(particles, ells):

            rmesh = particles[2].paint(**kwargs, out='real')
            cmesh = particles[0].clone(weights=particles[0].weights**2).paint(**kwargs, out='complex')
            cmesh = cmesh.clone(value=cmesh.value.at[(0,) * cmesh.ndim].set(0.))  # remove zero-mode

            @partial(jax.checkpoint, static_argnums=(0, 1))
            def f(Ylms, jl, carry, im):
                los = xvec if vlos is None else vlos
                s111, coeff, sym, im = im[-1], im[3], im[4], im[:3]
                s111 = compensate_shotnoise(s111)
                # Fourth line
                tmp = coeff * ((rmesh * jax.lax.switch(im[2], Ylms[2], *los)).r2c() * cmesh.conj() - s111)
                snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                tmp = tmp.c2r() * jax.lax.switch(im[0], Ylms[0], *svec) * jax.lax.switch(im[1], Ylms[1], *svec)

                def fk(k):
                    return jnp.sum(tmp.value * jl[0](snorm * k[0]) * jl[1](snorm * k[1]))

                tmp = (4. * np.pi)**2 * jax.lax.map(fk, bin.xavg)
                carry += tmp + sym * tmp.conj()
                return carry, im

            s113 = []
            for ell1, ell2, ell3 in ells:
                Ylms = [[get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in [ell1, ell2, ell3]]
                xs = _iter_triposh(ell1, ell2, ell3, los=los)
                if xs[0].size:
                     # Add s111 for s, im in xs are offset by ell
                    xs = xs + [jnp.array([s111[ellms.index((ell3, int(im3) - ell3))] for im3 in xs[2]])]
                    sign = (-1)**((ell1 + ell2) // 2)
                    s113.append(sign * jax.lax.scan(partial(f, Ylms, [get_spherical_jn(ell1), get_spherical_jn(ell2)]), init=jnp.zeros_like(bin.xavg[..., 0]), xs=xs)[0])
                else:
                    s113.append(jnp.zeros_like(bin.xavg[..., 0]))
            return s113

        uells = sorted(sum(ells, start=tuple()))
        ellms = [(ell, m) for ell in uells for m in range(-ell, ell + 1)]
        s111 = [0.] * len(ellms)
        if same[0] == same[1] == same[2]:
            s111 = compute_S111(particles, ellms)
            ell0 = (0, 0, 0)
            if ell0 in ells:
                shotnoise[ells.index(ell0)] += jnp.sqrt(4. * jnp.pi) * s111[ellms.index((0, 0))]

        if same[1] == same[2]:
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

        if same[0] == same[2]:
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

        if same[0] == same[1]:
            s113 = compute_S113(particles, ells)
            for ill, ell in enumerate(ells):
                shotnoise[ill] += s113[ill]

    return list(shotnoise)



def compute_fisher_scoccimarro(mattrs, bin, los: str | np.ndarray='z', apply_selection=None, power=None, seed=42, norm=None):

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
                        mask = (bin.ibin[ivalue] == ibin[ivalue]) * weights[ivalue]
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
                Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
                xvec = mattrs.rcoords(sparse=True)
                kvec = mattrs.kcoords(sparse=True)

                @partial(jax.checkpoint, static_argnums=0)
                def f(Ylm, carry, im):
                    carry += mattrs.r2c(rmesh * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                    return carry, im

                xs = np.arange(len(Ylms))
                return (4. * jnp.pi) * jax.lax.scan(partial(f, Ylms), init=mattrs.create(fill=0., kind='complex'), xs=xs)[0]

            def apply_real_harmonics(ell, cmesh):
                Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
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
                g_b0_maps.append([mattrs.c2r(cwmaps[axis] * (bin.ibin[axis + 1] == b)) for b in bin._uiedges[axis + 1]])
                if vlos is None:
                    leg_maps.append([apply_fourier_harmonics(ell, rwmaps[axis]) for ell in bin.ells])

            for ibin3 in bin.uiedges[-1]:
                g_bBl_maps = []
                for axis in range(2):
                    tmp = []
                    for ell in bin.ells:
                        if vlos is not None: tmp.append(mattrs.c2r(apply_fourier_legendre(ell, cwmaps[axis] * (bin.ibin[2] == ibin3))))
                        else: tmp.append(mattrs.c2r(leg_maps[axis] * (bin.ibin[2] == ibin3)))
                    g_bBl_maps.append(tmp)

                for ibin1 in bin.uiedges[0]:
                    if weighting == 'Sinv':
                        ibins2 = jnp.flatnonzero((bin._iedges[..., 0] == ibin1) & (bin._iedges[..., 2] == ibin3))
                        if not ibins2.size: continue
                        for ill, ell in enumerate(bin.ells):
                            # Compute FT[g_{0, bA}, g_{ell, bB}]
                            ft_ABl = mattrs.r2c(g_b0_maps[0][ibin1] * g_bBl_maps[0][ill][ibin3] - g_b0_maps[1][ibin1] * g_bBl_maps[1][ill][ibin3])
                            for ibin2 in ibins2:
                                fisher = fisher.at[ibin2 + ill * len(bin._iedges)].set(jnp.sum(ft_ABl * Q_Ainv.conj() * (bin.ibin[1] == ibin2)))

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
                                        tmp = ft_ABl[ill] * (bin.ibin[axis] == ibin)
                                    else:
                                        if vlos is not None:
                                            tmp = apply_fourier_legendre(ell, ft_ABl[bin.ells.index(0)] * (bin.ibin[axis] == ibin))
                                        else:
                                            tmp = apply_real_harmonics(ell, ft_ABl[bin.ells.index(0)] * (bin.ibin[axis] == ibin)).r2c()
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