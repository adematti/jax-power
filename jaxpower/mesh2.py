import os
import numbers
from functools import partial
from dataclasses import dataclass, field, asdict
from collections.abc import Callable
import itertools
from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp

from .types import WindowMatrix, Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles, ObservableLeaf, ObservableTree
from .utils import get_legendre, get_spherical_jn, get_Ylm, real_gaunt, legendre_product, register_pytree_dataclass
from .mesh import BaseMeshField, RealMeshField, ComplexMeshField, ParticleField, staticarray, MeshAttrs, _get_hermitian_weights, _find_unique_edges, _get_bin_attrs, _bincount, get_mesh_attrs


def _make_edges2(mattrs, edges, ells, kind='complex', mode_oversampling=0):
    wmodes = None
    if kind == 'complex':
        vec = mattrs.kcoords(kind='separation', sparse=True)
        vec0 = mattrs.kfun.min()
        if mattrs.is_hermitian:
            wmodes = _get_hermitian_weights(vec, sharding_mesh=None)
    else:
        vec = mattrs.xcoords(kind='separation', sparse=True)
        vec0 = mattrs.cellsize.min()

    if edges is None:
        edges = {}
    if isinstance(edges, dict):
        step = edges.get('step', None)
        vecmax = vec0 * np.min(mattrs.meshsize) / 2.
        if step is None:
            edges = _find_unique_edges(vec, vec0, xmin=edges.get('min', 0.), xmax=edges.get('max', np.inf))
        else:
            edges = np.arange(edges.get('min', 0.), edges.get('max', vecmax), step)
    if edges.ndim == 2:  # coming from ObservableTree
        assert np.allclose(edges[1:, 0], edges[:-1, 1])
        edges = np.append(edges[:, 0], edges[-1, 1])
    shifts = [jnp.arange(-mode_oversampling, mode_oversampling + 1)] * len(mattrs.meshsize)
    shifts = list(itertools.product(*shifts))
    ibin, nmodes, xsum = [], 0, 0
    for shift in shifts:
        coords = jnp.sqrt(sum((xx + ss)**2 for (xx, ss) in zip(vec, shift)))
        bin = _get_bin_attrs(coords, edges, weights=wmodes)
        del coords
        ibin.append(bin[0])
        nmodes += bin[1]
        xsum += bin[2]
    edges = np.column_stack([edges[:-1], edges[1:]])
    ells = _format_ells(ells)
    return dict(edges=edges, nmodes=nmodes / len(shifts), xavg=xsum / nmodes, ibin=ibin, wmodes=wmodes, ells=ells, mattrs=mattrs)


@partial(register_pytree_dataclass, meta_fields=['ells'])
@dataclass(init=False, frozen=True)
class BinMesh2SpectrumPoles(object):
    """
    Binning operator for mesh to power spectrum.

    Parameters
    ----------
    mattrs : MeshAttrs or BaseMeshField
        Mesh attributes.
    edges : array-like, dict, or None, optional
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.
    ells : int or tuple, optional
        Multipole orders.
    mode_oversampling : int, optional
        Oversampling factor for binning.
    """
    edges: jax.Array = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    ibin: jax.Array = None
    wmodes: jax.Array = None
    ells: tuple = None
    mattrs: MeshAttrs = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, ells: int | tuple=0, mode_oversampling: int=0):
        if not isinstance(mattrs, MeshAttrs):
            mattrs = mattrs.attrs
        kw = _make_edges2(mattrs, edges=edges, ells=ells, kind='complex', mode_oversampling=mode_oversampling)
        self.__dict__.update(kw)

    def __call__(self, mesh, antisymmetric=False, remove_zero=False):
        """
        Bin the mesh to compute the power spectrum.

        Parameters
        ----------
        mesh : array-like or BaseMeshField
            Input mesh.
        antisymmetric : bool, optional
            Whether the mesh is hermitian antisymmetric.
        remove_zero : bool, optional
            Whether to remove the zero mode.

        Returns
        -------
        binned : array-like
        """
        weights = self.wmodes
        value = mesh.value if isinstance(mesh, BaseMeshField) else mesh
        if remove_zero:
            value = value.at[(0,) * value.ndim].set(0.)
        return _bincount(self.ibin, value, weights=weights, length=len(self.xavg), antisymmetric=antisymmetric) / self.nmodes


@partial(register_pytree_dataclass, meta_fields=['ells', 'basis', 'batch_size'])
@dataclass(init=False, frozen=True)
class BinMesh2CorrelationPoles(object):
    """
    Binning operator for mesh to correlation function.

    Parameters
    ----------
    mattrs : MeshAttrs or BaseMeshField
        Mesh attributes.
    edges : array-like, dict, or None, optional
        ``edges`` may be:
        - a numpy array containing the :math:`s`-edges.
        - a dictionary, with keys 'min' (minimum :math:`s`, defaults to 0), 'max' (maximum :math:`s`, defaults to ``(boxsize / 2)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`s` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.
    ells : int or tuple, optional
        Multipole orders.
    mode_oversampling : int, optional
        Oversampling factor for binning.
    """
    edges: jax.Array = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    ibin: jax.Array = None
    ells: tuple = None
    basis: str = None
    batch_size: int = None
    kcut: tuple = None
    mattrs: MeshAttrs = None

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, ells: int | tuple=0, mode_oversampling: int=0, basis=None, kcut=None, batch_size: int=None):
        if not isinstance(mattrs, MeshAttrs):
            mattrs = mattrs.attrs
        kw = _make_edges2(mattrs, edges=edges, ells=ells, kind='real', mode_oversampling=mode_oversampling)
        kw.pop('wmodes')
        kw.update(basis=basis, batch_size=batch_size, kcut=kcut)
        self.__dict__.update(kw)

    def __call__(self, mesh, ell=0, remove_zero=False):
        """
        Bin the mesh to compute the correlation function.

        Parameters
        ----------
        mesh : array-like or BaseMeshField
            Input mesh.
        remove_zero : bool, optional
            Whether to remove the zero mode.

        Returns
        -------
        binned : array-like
        """
        value = mesh.value if isinstance(mesh, BaseMeshField) else mesh
        if self.basis == 'bessel':
            if remove_zero:
                value = value.at[(0,) * value.ndim].set(0.)
            jn = get_spherical_jn(ell)
            knorm = jnp.sqrt(sum(kk**2 for kk in self.mattrs.kcoords(sparse=True)))

            def bin(ibin):
                j = jn(knorm * self.xavg[ibin])
                if self.kcut is not None: j *= (knorm >= self.kcut[0]) * (knorm < self.kcut[1])
                return (-1)**(ell // 2) * jnp.sum(value * j) / self.mattrs.meshsize.prod(dtype=self.mattrs.rdtype)

            return jax.lax.map(bin, jnp.arange(len(self.xavg)), batch_size=self.batch_size)
        else:
            if remove_zero:
                value = value.at[(0,) * value.ndim].set(0.)
            return _bincount(self.ibin, value, weights=None, length=len(self.xavg)) / self.nmodes


def _get_los_vector(los: str | np.ndarray, ndim=3):
    """Return the line-of-sight vector."""
    vlos = None
    if isinstance(los, str):
        vlos = [0.] * ndim
        vlos['xyz'.index(los)] = 1.
    else:
        vlos = los
    return staticarray(vlos)


def _format_meshes(*meshes):
    """Format input meshes for autocorrelation/cross-correlation: return list of two meshes, and boolean if they are equal."""
    meshes = list(meshes)
    assert 1 <= len(meshes) <= 2
    meshes = meshes + [None] * (2 - len(meshes))
    same = [0]
    for mesh in meshes[1:]: same.append(same[-1] if mesh is None else same[-1] + 1)
    for imesh, mesh in enumerate(meshes):
        if mesh is None:
            meshes[imesh] = meshes[imesh - 1]
    return meshes, same[1] == same[0]


def _format_ells(ells):
    if np.ndim(ells) == 0: ells = (ells,)
    ells = tuple(sorted(ells))
    return ells


def _format_los(los, ndim=3):
    vlos, swap = None, False
    if isinstance(los, str) and los in ['firstpoint', 'local', 'endpoint']:
        if los == 'local': los = 'firstpoint'
        swap = los == 'endpoint'
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    return los, vlos, swap


def compute_mesh2(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh2SpectrumPoles | BinMesh2CorrelationPoles=None, los: str | np.ndarray='z'):
    """
    Dispatch to :func:`compute_mesh2_spectrum` or :func:`compute_mesh2_correlation`
    depending on type of input ``bin``.

    Parameters
    ----------
    meshes : RealMeshField or ComplexMeshField
        Input meshes.
    bin : BinMesh2SpectrumPoles or BinMesh2CorrelationPoles
        Binning operator.
    los : str or array-like, optional
        Line-of-sight specification.
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    Returns
    -------
    result : Mesh2SpectrumPoles or Mesh2CorrelationPoles
    """
    if isinstance(bin, BinMesh2SpectrumPoles):
        return compute_mesh2_spectrum(*meshes, bin=bin, los=los)
    elif isinstance(bin, BinMesh2CorrelationPoles):
        return compute_mesh2_correlation(*meshes, bin=bin, los=los)
    raise ValueError(f'bin must be either BinMesh2SpectrumPoles or BinMesh2CorrelationPoles, not {type(bin)}')


def compute_mesh2_spectrum(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh2SpectrumPoles=None, los: str | np.ndarray='z') -> Mesh2SpectrumPoles:
    r"""
    Compute the power spectrum multipoles from mesh.

    Parameters
    ----------
    meshs : RealMeshField or ComplexMeshField
        Input mesh(es).
    bin : BinMesh2SpectrumPoles
        Binning operator.
    los : str or array-like, optional
        Line-of-sight specification.
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    Returns
    -------
    result : Mesh2SpectrumPoles
    """

    meshes, autocorr = _format_meshes(*meshes)
    rdtype = meshes[0].real.dtype
    mattrs = meshes[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod() * jnp.ones_like(bin.xavg)

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
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

    if vlos is None:  # local, varying line-of-sight
        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]
        if swap: meshes = meshes[::-1]

        rmesh1 = meshes[0]
        A0 = _2c(rmesh1 if autocorr else meshes[1])
        #print('jax', rmesh1.value.std(), A0.value.std())
        del meshes

        num = []
        if 0 in ells:
            if autocorr:
                Aell = A0.clone(value=A0.real**2 + A0.imag**2)  # saves a bit of memory
            else:
                Aell = _2c(rmesh1) * A0.conj()
            num.append(bin(Aell, antisymmetric=False))
            del Aell

        if nonzeroells:
            rmesh1 = _2r(rmesh1)
            # The real-space grid
            xvec = mattrs.rcoords(sparse=True)
            # The Fourier-space grid
            kvec = mattrs.kcoords(sparse=True)

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += _2c(rmesh1 * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            for ell in nonzeroells:
                Ylms = [get_Ylm(ell, m, real=True) for m in range(-ell, ell + 1)]
                #jax.debug.inspect_array_sharding(jnp.zeros_like(A0.value), callback=print)
                xs = np.arange(len(Ylms))
                #print('jax', ell, jax.lax.scan(partial(f, Ylms), init=A0.clone(value=jnp.zeros_like(A0.value)), xs=xs)[0].value.std())
                Aell = jax.lax.scan(partial(f, Ylms), init=A0.clone(value=jnp.zeros_like(A0.value)), xs=xs)[0].conj() * A0
                #Aell = sum(_2c(rmesh1 * Ylm(*xvec)) * Ylm(*kvec) for Ylm in Ylms).conj() * A0
                # Project on to 1d k-basis (averaging over mu=[-1, 1])
                #print('jax', ell, A0.value.std(), Aell.value.std())
                num.append(4. * jnp.pi * bin(Aell, antisymmetric=bool(ell % 2), remove_zero=True))
                del Aell

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        if swap: num = list(map(jnp.conj, num))
        # Format the num results into :class:`Mesh2SpectrumPoles` instance
        spectrum = []
        for ill, ell in enumerate(ells):
            spectrum.append(Mesh2SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes, num_raw=num[ill], num_shotnoise=jnp.zeros_like(num[ill]), norm=norm,
                                              volume=mattrs.kfun.prod() * bin.nmodes, ell=ell, attrs=attrs))
        return Mesh2SpectrumPoles(spectrum)

    else:  # fixed line-of-sight

        for imesh, mesh in enumerate(meshes[:1 if autocorr else 2]):
            meshes[imesh] = _2c(mesh)
        if autocorr:
            Aell = meshes[0].clone(value=meshes[0].real**2 + meshes[0].imag**2)  # saves a bit of memory
        else:
            Aell = meshes[0] * meshes[1].conj()
        del meshes

        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        num = []
        if 0 in ells:
            num.append(bin(Aell, antisymmetric=False))

        if nonzeroells:
            kvec = mattrs.kcoords(sparse=True)
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)
            for ell in nonzeroells:  # TODO: jax.lax.scan
                num.append((2 * ell + 1) * bin(Aell * get_legendre(ell)(mu), antisymmetric=bool(ell % 2), remove_zero=True))

        spectrum = []
        for ill, ell in enumerate(ells):
            spectrum.append(Mesh2SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes, num_raw=num[ill], num_shotnoise=jnp.zeros_like(num[ill]), norm=norm,
                                              volume=mattrs.kfun.prod() * bin.nmodes, ell=ell, attrs=attrs))
        return Mesh2SpectrumPoles(spectrum)


def compute_mesh2_correlation(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh2CorrelationPoles=None, los: str | np.ndarray='z') -> Mesh2CorrelationPoles:
    """
    Compute the correlation function multipoles from mesh.

    Parameters
    ----------
    meshs : RealMeshField or ComplexMeshField
        Input meshes.
    bin : BinMesh2CorrelationPoles
        Binning operator.
    los : str or array-like, optional
        Line-of-sight specification.
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    Returns
    -------
    result : Mesh2CorrelationPoles
    """
    meshes, autocorr = _format_meshes(*meshes)
    rdtype = meshes[0].real.dtype
    mattrs = meshes[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod() * jnp.ones_like(bin.xavg)

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
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

    if vlos is None:  # local, varying line-of-sight
        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]
        if swap: meshes = meshes[::-1]

        rmesh1 = meshes[0]
        A0 = _2c(rmesh1 if autocorr else meshes[1])
        del meshes

        num = []
        if 0 in ells:
            if autocorr:
                Aell = A0.clone(value=A0.real**2 + A0.imag**2)  # saves a bit of memory
            else:
                Aell = _2c(rmesh1) * A0.conj()
            Aell = Aell.c2r()  # convert to real space
            num.append(bin(Aell))
            del Aell

        if nonzeroells:
            rmesh1 = _2r(rmesh1)
            # The real-space grid
            xvec = mattrs.rcoords(sparse=True)
            # The separation grid
            svec = mattrs.rcoords(kind='separation', sparse=True)

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += _2r(_2c(rmesh1 * jax.lax.switch(im, Ylm, *xvec)).conj() * A0) * jax.lax.switch(im, Ylm, *svec)
                return carry, im

            for ell in nonzeroells:
                Ylms = [get_Ylm(ell, m, real=True) for m in range(-ell, ell + 1)]
                xs = np.arange(len(Ylms))
                Aell = jax.lax.scan(partial(f, Ylms), init=rmesh1.clone(value=jnp.zeros_like(rmesh1.value)), xs=xs)[0]
                num.append(4. * jnp.pi * bin(Aell, remove_zero=True))
                del Aell

        if swap: num = list(map(jnp.conj, num))
        # Format the num results into :class:`Mesh2CorrelationPoles` instance
        correlation = []
        for ill, ell in enumerate(ells):
            correlation.append(Mesh2CorrelationPole(s=bin.xavg, s_edges=bin.edges, nmodes=bin.nmodes, num_raw=num[ill] / mattrs.cellsize.prod(), norm=norm,
                                                    volume=mattrs.cellsize.prod() * bin.nmodes, ell=ell, attrs=attrs))
        return Mesh2CorrelationPoles(correlation)

    else:  # fixed line-of-sight

        for imesh, mesh in enumerate(meshes[:1 if autocorr else 2]):
            meshes[imesh] = _2c(mesh)
        if autocorr:
            Aell = meshes[0].clone(value=meshes[0].real**2 + meshes[0].imag**2)  # saves a bit of memory
        else:
            Aell = meshes[0] * meshes[1].conj()
        Aell = Aell.c2r()  # convert to real space
        del meshes

        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        num = []
        if 0 in ells:
            num.append(bin(Aell))

        if nonzeroells:
            svec = mattrs.rcoords(kind='separation', sparse=True)
            mu = sum(ss * ll for ss, ll in zip(svec, vlos)) / jnp.sqrt(sum(ss**2 for ss in svec)).at[(0,) * mattrs.ndim].set(1.)
            for ell in nonzeroells:  # TODO: jax.lax.scan
                num.append((2 * ell + 1) * bin(Aell * get_legendre(ell)(mu), remove_zero=True))

        correlation = []
        for ill, ell in enumerate(ells):
            correlation.append(Mesh2CorrelationPole(s=bin.xavg, s_edges=bin.edges, nmodes=bin.nmodes, num_raw=num[ill] / mattrs.cellsize.prod(), norm=norm,
                                                    volume=mattrs.cellsize.prod() * bin.nmodes, ell=ell, attrs=attrs))
        return Mesh2CorrelationPoles(correlation)



def compute_mesh2_correlation(*meshes: RealMeshField | ComplexMeshField, bin: BinMesh2CorrelationPoles=None, los: str | np.ndarray='z') -> Mesh2CorrelationPoles:
    """
    Compute the correlation function multipoles from mesh.

    Parameters
    ----------
    meshs : RealMeshField or ComplexMeshField
        Input meshes.
    bin : BinMesh2CorrelationPoles
        Binning operator.
    los : str or array-like, optional
        Line-of-sight specification.
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    Returns
    -------
    result : Mesh2CorrelationPoles
    """
    meshes, autocorr = _format_meshes(*meshes)
    rdtype = meshes[0].real.dtype
    mattrs = meshes[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod() * jnp.ones_like(bin.xavg)

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
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

    if vlos is None:  # local, varying line-of-sight
        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]
        if swap: meshes = meshes[::-1]

        rmesh1 = meshes[0]
        A0 = _2c(rmesh1 if autocorr else meshes[1])
        del meshes

        num = []
        if 0 in ells:
            if autocorr:
                Aell = A0.clone(value=A0.real**2 + A0.imag**2)  # saves a bit of memory
            else:
                Aell = _2c(rmesh1) * A0.conj()
            if bin.basis != 'bessel': Aell = Aell.c2r()  # convert to real space
            num.append(bin(Aell, ell=0))
            del Aell

        if nonzeroells:
            rmesh1 = _2r(rmesh1)
            # The real-space grid
            xvec = mattrs.rcoords(sparse=True)
            # The separation grid
            svec = mattrs.rcoords(kind='separation', sparse=True)
            kvec = mattrs.kcoords(sparse=True)

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                if bin.basis != 'bessel':
                    carry += _2r(_2c(rmesh1 * jax.lax.switch(im, Ylm, *xvec)).conj() * A0) * jax.lax.switch(im, Ylm, *svec)
                else:
                    carry += _2c(rmesh1 * jax.lax.switch(im, Ylm, *xvec)).conj() * A0 * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            for ell in nonzeroells:
                Ylms = [get_Ylm(ell, m, real=True) for m in range(-ell, ell + 1)]
                xs = np.arange(len(Ylms))
                if bin.basis != 'bessel':
                    init = rmesh1.clone(value=jnp.zeros_like(rmesh1.value))
                else:
                    init = A0.clone(value=jnp.zeros_like(A0.value))
                Aell = jax.lax.scan(partial(f, Ylms), init=init, xs=xs)[0]
                num.append(4. * jnp.pi * bin(Aell, ell=ell, remove_zero=True))
                del Aell

        if swap: num = list(map(jnp.conj, num))
        # Format the num results into :class:`Mesh2CorrelationPoles` instance
        correlation = []
        for ill, ell in enumerate(ells):
            correlation.append(Mesh2CorrelationPole(s=bin.xavg, s_edges=bin.edges, nmodes=bin.nmodes, num_raw=num[ill] / mattrs.cellsize.prod(), norm=norm,
                                                    volume=mattrs.cellsize.prod() * bin.nmodes, ell=ell, attrs=attrs))
        return Mesh2CorrelationPoles(correlation)

    else:  # fixed line-of-sight

        for imesh, mesh in enumerate(meshes[:1 if autocorr else 2]):
            meshes[imesh] = _2c(mesh)
        if autocorr:
            Aell = meshes[0].clone(value=meshes[0].real**2 + meshes[0].imag**2)  # saves a bit of memory
        else:
            Aell = meshes[0] * meshes[1].conj()
        if bin.basis != 'bessel': Aell = Aell.c2r()  # convert to real space
        del meshes

        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        num = []
        if 0 in ells:
            num.append(bin(Aell, ell=0))

        if nonzeroells:
            svec = mattrs.rcoords(kind='separation', sparse=True)
            kvec = mattrs.kcoords(sparse=True)
            if bin.basis != 'bessel':
                mu = sum(ss * ll for ss, ll in zip(svec, vlos)) / jnp.sqrt(sum(ss**2 for ss in svec)).at[(0,) * mattrs.ndim].set(1.)
            else:
                mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)
            for ell in nonzeroells:  # TODO: jax.lax.scan
                num.append((2 * ell + 1) * bin(Aell * get_legendre(ell)(mu), ell=ell, remove_zero=True))

        correlation = []
        for ill, ell in enumerate(ells):
            correlation.append(Mesh2CorrelationPole(s=bin.xavg, s_edges=bin.edges, nmodes=bin.nmodes, num_raw=num[ill] / mattrs.cellsize.prod(), norm=norm,
                                                    volume=mattrs.cellsize.prod() * bin.nmodes, ell=ell, attrs=attrs))
        return Mesh2CorrelationPoles(correlation)


@partial(jax.tree_util.register_dataclass, data_fields=['data', 'randoms'], meta_fields=[])
@dataclass(frozen=True, init=False)
class FKPField(object):
    """
    FKP field: data minus randoms.

    Parameters
    ----------
    data : ParticleField
        Data particles.
    randoms : ParticleField
        Random particles.

    References
    ----------
    https://arxiv.org/abs/astro-ph/9304022
    """
    data: ParticleField
    randoms: ParticleField

    def __init__(self, data, randoms, attrs=None):
        """
        Initialize FKPField.

        Parameters
        ----------
        data : ParticleField
            Data particles.
        randoms : ParticleField
            Random particles.
        attrs : MeshAttrs, optional
            Mesh attributes.
        """
        if attrs is not None:
            data = data.clone(attrs=attrs)
            randoms = randoms.clone(attrs=attrs)
        self.__dict__.update(data=data, randoms=randoms)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = {name: getattr(self, name) for name in ['data', 'randoms']} | kwargs
        return self.__class__(**state)

    def exchange(self, **kwargs):
        """In distributed computation, exchange particles such that their distribution matches the mesh shards."""
        return self.clone(data=self.data.clone(exchange=True, **kwargs), randoms=self.randoms.clone(exchange=True, **kwargs), attrs=self.attrs)

    @property
    def attrs(self):
        """Mesh attributes."""
        return self.data.attrs

    def split(self, nsplits=1, extent=None):
        """Split particles into subregions."""
        from .mesh import _get_extent
        if extent is None:
            extent = _get_extent(self.data.positions, self.randoms.positions)
        for data, randoms in zip(self.data.split(nsplits=nsplits, extent=extent), self.randoms.split(nsplits=nsplits, extent=extent)):
            new = self.clone(data=data, randoms=randoms)
            yield new

    @property
    def particles(self):
        """Return the FKP field as a :class:`ParticleField`."""
        particles = getattr(self, '_particles', None)
        if particles is None:
            self.__dict__['_particles'] = particles = (self.data - self.data.sum() / self.randoms.sum() * self.randoms).clone(attrs=self.data.attrs)
        return particles

    def paint(self, resampler: str | Callable='cic', interlacing: int=1,
              compensate: bool=False, dtype=None, out: str='real', **kwargs):
        """
        Paint the FKP field onto a mesh.

        Parameters
        ----------
        resampler : str, Callable, default='cic'
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
        mesh : MeshField
        """
        return self.particles.paint(resampler=resampler, interlacing=interlacing, compensate=compensate, dtype=dtype, out=out, **kwargs)


def compute_normalization(*inputs: RealMeshField | ParticleField, bin: BinMesh2SpectrumPoles | BinMesh2CorrelationPoles=None, resampler='cic', **kwargs) -> jax.Array:
    """
    Compute normalization for input fields, in volume**(1 - len(inputs)) unit.

    Parameters
    ----------
    inputs : RealMeshField or ParticleField
        Input fields.
    resampler : str, Callable, default='cic'
        Resampling method.
    kwargs : dict
        Additional arguments for :meth:`ParticleField.paint`.

    Returns
    -------
    normalization : jax.Array

    Warning
    -------
    Input particles are considered uncorrelated.
    """
    meshes, particles = [], []
    attrs = dict(kwargs)
    for inp in inputs:
        if isinstance(inp, RealMeshField):
            meshes.append(inp)
            attrs = {name: getattr(inp, name) for name in ['boxsize', 'boxcenter', 'meshsize']}
        else:
            particles.append(inp)
    halo_add = 0
    if particles:
        cattrs = dict(particles[0].attrs)
        update_cellsize = 'cellsize' in attrs
        if update_cellsize:
            cattrs.pop('meshsize')
        attrs = particles[0].attrs.clone(**get_mesh_attrs(**(cattrs | attrs)))
        if update_cellsize:
            halo_add = int(np.ceil(np.max((attrs.boxsize - cattrs['boxsize']) / 2. / attrs.cellsize)))
        particles = [particle.clone(attrs=attrs) for particle in particles]
    normalization = 1
    for mesh in meshes:
        normalization *= mesh
    for particle in particles:
        normalization *= particle.paint(resampler=resampler, interlacing=0, compensate=False, halo_add=halo_add)
    norm = normalization.sum() * normalization.cellsize.prod()**(1 - len(inputs))
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_box_normalization(*inputs: RealMeshField | ParticleField, bin: BinMesh2SpectrumPoles | BinMesh2CorrelationPoles=None) -> jax.Array:
    """Compute normalization, assuming constant density."""
    normalization = 1.
    mattrs = inputs[0].attrs
    size = mattrs.meshsize.prod(dtype=mattrs.rdtype)
    for mesh in inputs:
        normalization *= mesh.sum() / size
    norm = normalization * size * mattrs.cellsize.prod()**(1 - len(inputs))
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_box2_normalization(*inputs: RealMeshField | ParticleField, bin: BinMesh2SpectrumPoles | BinMesh2CorrelationPoles=None) -> jax.Array:
    """Compute normalization, assuming constant density, for the power spectrum."""
    return compute_box_normalization(*_format_meshes(*inputs)[0], bin=bin)


def compute_fkp2_normalization(*fkps: FKPField, bin: BinMesh2SpectrumPoles | BinMesh2CorrelationPoles=None, cellsize: float=10.):
    """
    Compute the FKP normalization for the power spectrum.

    Parameters
    ----------
    fkps : FKPField or ParticleField
        FKP fields.
    bin : BinMesh2SpectrumPoles or BinMesh2CorrelationPoles, optional
        Binning operator. Only used to return a list of normalization factors for each multipole.
    cellsize : float, optional
        Cell size.

    Returns
    -------
    norm : float, list
    """
    # This is the pypower normalization - move to new one?
    fkps, autocorr = _format_meshes(*fkps)
    kw = dict(cellsize=cellsize)
    for name in list(kw):
        if kw[name] is None: kw.pop(name)
    if autocorr:
        fkp = fkps[0]
        randoms = [fkp.data, fkp.randoms]  # cross to remove common noise
        #mask = random.uniform(random.key(42), shape=fkp.randoms.size) < 0.5
        #randoms = [fkp.randoms[mask], fkp.randoms[~mask]]
        #randoms = [fkp.randoms[:fkp.randoms.size // 2], fkp.randoms[fkp.randoms.size // 2:]]
        alpha = fkp.data.sum() / fkp.randoms.sum()
        norm = alpha * compute_normalization(*randoms, **kw)
    else:
        randoms = [fkps[0].data, fkps[1].randoms]  # cross to remove common noise
        alpha2 = fkps[1].data.sum() / fkps[1].randoms.sum()
        norm = alpha2 * compute_normalization(*randoms, **kw)
        randoms = [fkps[1].data, fkps[0].randoms]
        alpha2 = fkps[0].data.sum() / fkps[0].randoms.sum()
        norm += alpha2 * compute_normalization(*randoms, **kw)
        norm = norm / 2
    if bin is not None:
        return [norm] * len(bin.ells)
    return norm


def compute_fkp2_shotnoise(*fkps: FKPField | ParticleField, bin: BinMesh2SpectrumPoles | BinMesh2CorrelationPoles=None):
    """
    Compute the FKP shot noise for the power spectrum.

    Parameters
    ----------
    fkps : FKPField or ParticleField
        FKP or particle fields.
    bin : BinMesh2SpectrumPoles or BinMesh2CorrelationPoles, optional
        Binning operator. Only used to return a list of shotnoise estimates for each multipole.

    Returns
    -------
    shotnoise : float, list
    """
    # This is the pypower normalization - move to new one?
    fkps, autocorr = _format_meshes(*fkps)
    if autocorr:
        particles = fkp = fkps[0]
        if isinstance(fkp, FKPField):
            particles = fkp.particles
        shotnoise = jnp.sum(particles.weights**2)
        kcut = getattr(bin, 'kcut', None)
        if kcut is not None:  # count number of modes
            kvec = bin.mattrs.kcoords(sparse=True)
            knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
            shotnoise *= jnp.sum((knorm >= kcut[0]) & (knorm <= kcut[-1])) / knorm.size
            del knorm
    else:
        shotnoise = 0.
    if bin is not None:
        if isinstance(bin, BinMesh2CorrelationPoles):
            mask_shotnoise = (bin.edges[..., 0] <= 0.) & (bin.edges[..., 1] >= 0.)
            mask_shotnoise = mask_shotnoise / bin.mattrs.cellsize.prod()
        else:
            mask_shotnoise = jnp.ones_like(bin.xavg)
        return [shotnoise * (ell == 0) * mask_shotnoise for ell in bin.ells]
    return shotnoise


def compute_wide_angle_spectrum2_poles(poles: dict[Callable]):
    r"""
    Add (first) wide-angle order power spectrum multipoles to input dictionary of poles.

    Parameters
    ----------
    poles : dict[Callable]
        A dictionary of callables, with keys the multipole orders :math:`\ell`.
        Non-provided poles are assumed zero.

    Returns
    -------
    poles : Dictionary of callables with keys (multipole order, wide-angle order) :math:`(\ell, n)`.
    """
    toret = {(ell, 0): pole for ell, pole in poles.items()}
    for ell in range(max(list(toret) + [0]) + 1):
        tmp = []
        if ell - 1 in poles:
            p = poles[ell - 1]
            coeff = - ell * (ell - 1) / (2. * (2. * ell - 1))
            tmp.append(coeff * (ell - 1), p)
            tmp.append(- coeff, lambda k: k * jax.jacfwd(p)(k))
        if ell + 1 in poles:
            p = poles[ell + 1]
            coeff = - (ell + 1) * (ell + 2) / (2. * (2. * ell + 3))
            tmp.append(coeff * (ell + 1), p)
            tmp.append(coeff, lambda k: k * jax.jacfwd(p)(k))

        def func(k):
            return sum(coeff * p(k) for coeff, p in tmp)

        toret[(ell, 1)] = func
    return toret


def interpolate_window_function(window: ObservableTree, coords: tuple | np.ndarray | int=4096, order=3):
    """
    Extrapolate / resample the configuration-space window function.

    Parameters
    ----------
    window : ObservableTree
        Window function to resample.
    coords : tuple | numpy.ndarray | int, optional
        Target coordinates for interpolation.
        - If an integer N, generate `N` log-spaced points spanning an extended
          decade range derived from the original coordinates.
        - If an array-like, it is used as the new coordinate for the first axis.
        - If a tuple of arrays, each element is used for the corresponding axis.
        When a leaf has more axes than supplied coordinates, the last supplied
        coordinate is repeated for the remaining axes.
    order : int, optional
        Spline order (degree). Default is 3 (cubic). For 2D interpolation this
        is used for both dimensions (kx = ky = order).

    Returns
    -------
    window : ObservableTree
        New (extrapolated) window function.
    """
    def get_new_coords(old_coords, arg_coords):
        if isinstance(old_coords, list):
            return [get_new_coords(old_coord, arg_coord) for old_coord, arg_coord in zip(old_coords, arg_coords)]
        start, stop = old_coords.min(), old_coords.max()
        start = 0 if start <= 0 else np.floor(np.log10(start)).astype(int)
        stop = np.ceil(np.log10(stop)).astype(int)
        delta = (stop - start) / 2.
        start, stop = start - delta, stop + delta

        if isinstance(arg_coords, numbers.Number):
            num = int(arg_coords)
            return jnp.logspace(start, stop, num)
        else:
            return arg_coords

    def pad_coords(coords):
        if isinstance(coords, list):
            return [pad_coords(coord) for coord in coords]
        # Add 0 in front, last value at the end
        return jnp.pad(coords, (1, 1), mode='constant', constant_values=(0. - (coords[1] - coords[0]), coords[-1] + (coords[-1] - coords[-2])))

    def pad_value(value):
        value = jnp.pad(value, ((0, 1),) * value.ndim, mode='constant', constant_values=0.)
        value = jnp.pad(value, ((1, 0),) * value.ndim, mode='edge')
        return value

    def remove_nan(coords, value):
        masks = []
        coords = list(coords)
        for axis in range(value.ndim):
            masks.append(np.isfinite(value).all(axis=tuple(i for i in range(value.ndim) if i != axis)))
            coords[axis] = coords[axis][masks[-1]]
        value = value[np.ix_(*masks)]
        return coords, value

    def extrapolate_leaf(leaf, coords):
        from scipy import interpolate
        old_coords = leaf.coords(center='mid_if_edges_and_nan')
        if not isinstance(coords, tuple):
            coords = (coords,)
        coords += (coords[-1],) * (len(old_coords) - len(coords))
        if len(old_coords) == 1:
            old_x = list(old_coords.values())[0]
            new_x = get_new_coords(old_x, coords[0])
            old_x = pad_coords(old_x)
            old_value = pad_value(leaf.value())
            old_x, old_value = remove_nan([old_x], old_value)
            spline = interpolate.UnivariateSpline(old_x[0], old_value, k=order, s=0, ext=3, check_finite=False)
            new_value = spline(new_x)
            new_coords = [new_x]
        elif len(old_coords) == 2:
            old_x = list(old_coords.values())
            new_x = get_new_coords(old_x, coords)
            old_x = pad_coords(old_x)
            old_value = pad_value(leaf.value())
            old_x, old_value = remove_nan(old_x, old_value)
            spline = interpolate.RectBivariateSpline(*old_x, old_value, kx=order, ky=order, s=0)
            new_value = spline(*new_x, grid=True)
            new_coords = new_x
        new_coords = {name: coord for name, coord in zip(old_coords, new_coords)}
        return ObservableLeaf(value=new_value, **new_coords, coords=list(new_coords), attrs=dict(leaf.attrs), meta=dict(leaf.meta))

    if isinstance(window, ObservableLeaf):
        return extrapolate_leaf(window, coords=coords)
    return window.map(lambda leaf: extrapolate_leaf(leaf, coords=coords))


def get_window_coeffs(ell, ell1):
    coeffs = []
    for q in range(abs(ell - ell1), ell + ell1 + 1):
        coeff = (2 * ell + 1) * legendre_product(ell, ell1, q)
        if abs(coeff) < 1e-7: continue
        coeffs.append((q, coeff))
    return coeffs


def get_smooth2_window_bin_attrs(ells, ellsin=3, return_ellsin: bool=False):
    """
    Get the window bin attributes for sugiyama basis.

    Parameters
    ----------
    ells : list
        Observed multipole orders.
    ellsin : tuple, int
        Theory multipole orders, or number of even multipoles up to which to compute the window.

    Returns
    -------
    dict
    """
    if isinstance(ellsin, numbers.Number):
        ellsin = list(range(0, 2 * ellsin - 1, 2))
    with_wide_angle = any(isinstance(ellin, tuple) for ellin in ellsin)
    ellsin = [ellin if isinstance(ellin, tuple) else (ellin, 0) for ellin in ellsin]
    non_zero_ellsin = []
    ellw = {}
    for ell1, wa1 in ellsin:  # ell1 3-tuple, wa1 wide-angle order
        if wa1 not in ellw: ellw[wa1] = []
        for ill, ell in enumerate(ells):
            coeffs = get_window_coeffs(ell, ell1)
            if coeffs and (ell1, wa1) not in non_zero_ellsin:
                non_zero_ellsin.append((ell1, wa1))
            for ell, _ in coeffs:
                if ell not in ellw[wa1]: ellw[wa1].append(ell)

    for wa in ellw:
        ellw[wa] = sorted(set(ellw[wa]))

    def _make_dict(ellw):
        return dict(ells=ellw)

    if with_wide_angle:
        ellw = [(_make_dict(ellw[wa]), wa) for wa in ellw]
    else:
        ellw = _make_dict(ellw[0])
        non_zero_ellsin = [ellin[0] for ellin in non_zero_ellsin]
    if return_ellsin:
        return ellw, non_zero_ellsin
    return ellw



def compute_smooth2_spectrum_window(window, edgesin: np.ndarray, ellsin: tuple=None, bin: BinMesh2SpectrumPoles=None, flags=('rect',)) -> WindowMatrix:
    """
    Compute the "smooth" (no binning effect) power spectrum window matrix.

    Parameters
    ----------
    window : ObservableTree
        Configuration-space window function.
    edgesin : np.ndarray
        Input bin edges.
    ellsin : tuple, optional
        Input multipole orders. Optional when ``edgesin`` is provided.
    bin : BinMesh2SpectrumPoles
        Output binning.

    Returns
    -------
    wmat : WindowMatrix
    """
    from .utils import BesselIntegral
    ells = bin.ells

    if isinstance(edgesin, ObservableTree):
        ellsin = edgesin.ells
        if 'wa_orders' in edgesin.labels(return_type='keys'):
            ellsin = [(ell, wa) for ell, wa in zip(edgesin.ells, edgesin.wa_orders)]
        edgesin = next(iter(edgesin)).edges('k')

    elif edgesin.ndim != 2:
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

    ellsin = [ellin if isinstance(ellin, tuple) else (ellin, 0) for ellin in ellsin]
    #kout = jnp.where(bin.nmodes == 0, 0., bin.xavg)
    kout = bin.xavg

    wmat_tmp = {}

    for ell1, wa1 in ellsin:
        wmat_tmp[ell1, wa1] = []
        for ill, ell in enumerate(ells):

            def get_w(q):
                kw = dict(ells=q)
                if 'wa_orders' in window.labels(return_type='keys'):
                    kw.update(wa_orders=wa1)
                elif wa1 != 0:
                    raise ValueError('wa_orders must be provided in input window')
                if kw in window.labels(return_type='flatten'):
                    return window.get(**kw).value().real
                return jnp.zeros(())

            Qs = sum(coeff * get_w(q) for q, coeff in get_window_coeffs(ell, ell1))

            #integ = BesselIntegral(window.get(0).edges('s'), kout, ell=ell, method='rect', mode='forward', edges=True, volume=False)
            if 'rect' in flags:
                tmpw = next(iter(window))
                if 'volume' in tmpw.values():
                    snmodes = tmpw.values('volume')
                else:
                    snmodes = jnp.ones_like(tmpw.value())
                savg = jnp.where(snmodes == 0, 0., tmpw.coords('s'))
                Qs = jnp.where(snmodes == 0, 0., Qs)

                def f(edgein):
                    tophat_Qs = BesselIntegral(edgein, savg, ell=ell1, edges=True, method='rect', mode='backward').w[..., 0] * Qs * savg**wa1
                    def f2(kout):
                        integ = BesselIntegral(savg, kout, ell=ell, method='rect', mode='forward', edges=False, volume=False)
                        return integ(snmodes * tophat_Qs)
                    #    return (-1)**(ell // 2) * jnp.sum(snmodes * spherical_jn[ell](kout * savg) * tophat_Qs)
                    batch_size = int(min(max(1e7 / savg.size, 1), kout.size))
                    spectrum = jax.lax.map(f2, kout, batch_size=batch_size)
                    #spectrum = jnp.zeros_like(spectrum, shape=(len(ells), spectrum.size)).at[ill].set(spectrum)
                    return spectrum#.ravel()

                batch_size = int(min(max(1e7 / (kout.size * savg.size), 1), edgesin.shape[0]))
                tmp = jax.lax.map(f, xs=edgesin, batch_size=batch_size).T
            else:
                # fftlog
                from .fftlog import SpectrumToCorrelation, CorrelationToSpectrum
                to_spectrum = CorrelationToSpectrum(s=next(iter(window)).coords('s'), ell=ell, check_level=1, lowring=False)
                to_correlation = SpectrumToCorrelation(k=to_spectrum.k, ell=ell1, lowring=False)
                kin = jnp.mean(edgesin, axis=-1)

                def convolve(theory):
                    #return (ell == ell1) * jnp.interp(kout, kin, theory, left=0., right=0.)
                    theory = jnp.interp(to_spectrum.k, kin, theory, left=0., right=0.)
                    correlation = to_correlation(theory)[1]
                    #correlation = (ell == ell1) * correlation
                    correlation = correlation * Qs * to_correlation.s**wa1
                    return jnp.interp(kout, to_spectrum.k, to_spectrum(correlation)[1], left=0., right=0.)

                tmp = jax.jacfwd(convolve)(jnp.zeros_like(edgesin[..., 0]))
                #tmp = jnp.zeros_like(tmp, shape=(len(ells) * tmp.shape[0], tmp.shape[1])).at[ill * tmp.shape[0]:(ill + 1) * tmp.shape[0]].set(tmp).T

            wmat_tmp[ell1, wa1].append(tmp)
        wmat_tmp[ell1, wa1] = jnp.concatenate(wmat_tmp[ell1, wa1], axis=0)

    wmat = jnp.concatenate(list(wmat_tmp.values()), axis=1)

    observable = []
    for ill, ell in enumerate(ells):
        observable.append(Mesh2SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes, num_raw=jnp.zeros_like(bin.xavg), num_shotnoise=jnp.zeros_like(bin.xavg), norm=jnp.ones_like(bin.xavg), ell=ell))
    observable = Mesh2SpectrumPoles(observable)

    theory = []
    kin, edgesin = jnp.mean(edgesin, axis=-1), edgesin
    for ill, ell in enumerate(ellsin):
        #theory.append(ObservableLeaf(k=kin, k_edges=edgesin, value=jnp.zeros_like(kin), coords=['k']))
        theory.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=jnp.zeros_like(kin)))
    #theory = Mesh2SpectrumPoles(theory, ells=ellsin)
    kw = dict(ells=list(ellsin))
    if isinstance(ellsin[0], tuple):  # ell, wide-angle order
        kw = dict(ells=[ell[0] for ell in ellsin], wa_orders=[ell[1] for ell in ellsin])
    theory = ObservableTree(theory, **kw)

    return WindowMatrix(observable=observable, theory=theory, value=wmat)



def compute_mesh2_spectrum_window(*meshes: RealMeshField | ComplexMeshField | MeshAttrs, edgesin: np.ndarray, ellsin: tuple=None,
                                  bin: BinMesh2SpectrumPoles=None, los: str | np.ndarray='z',
                                  buffer=None, batch_size=None, pbar=False, norm=None, flags=tuple()) -> WindowMatrix:
    r"""
    Compute the power spectrum window matrix from mesh.

    Parameters
    ----------
    meshes : RealMeshField, ComplexMeshField, MeshAttrs
        Input mesh(es).
        A :class:`MeshAttrs` instance can be directly provided, in case the selection function is trivial (constant).
    edgesin : np.ndarray
        Input bin edges.
    ellsin : tuple, list
        Input multipole orders.
    bin : BinMesh2SpectrumPoles
        Output binning.
    los : str, array, default=None
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.
    buffer : str or device, optional
        Buffer for intermediate results (meshes).
        If string, save intermerdiate meshes to this path.
    batch_size : int, optional
        Batch size for computation: how many meshes hold in memory.
    pbar : bool, optional
        Whether to show a progress bar.
    norm : float, optional
        Normalization factor.
    flags : tuple, optional
        Additional computation flags.
        - "smooth": smooth approximation to the window matrix (no binning effect)
        - "infinite": input power spectrum is turned into theory correlation function with "infinite" resolution.
        - else: input power spectrum is defined on the mesh. Useful for control variate estimates.

    Returns
    -------
    wmat : WindowMatrix
    """
    tophat_method = 'exact'

    from .utils import BesselIntegral

    meshes, autocorr = _format_meshes(*meshes)
    periodic = isinstance(meshes[0], MeshAttrs)
    if periodic:
        assert autocorr
        mattrs = meshes[0]
    else:
        mattrs = meshes[0].attrs
    rdtype = mattrs.rdtype

    _norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod()
    if norm is None: norm = _norm
    norm = jnp.array(norm)
    if norm.ndim <= 1:
        norm = norm * jnp.ones(len(bin.ells), dtype=rdtype)
    rnorm = _norm / norm / mattrs.meshsize.prod(dtype=rdtype)

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
    ells = bin.ells

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    if isinstance(edgesin, ObservableTree):
        ellsin = edgesin.ells
        edgesin = next(iter(edgesin)).edges('k')
    else:
        if edgesin.ndim != 2:
            edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

    def np_map(f, xs):
        return jnp.array(list(map(f, xs)))

    svec = mattrs.xcoords(kind='separation', sparse=True)

    if pbar:
        from tqdm import tqdm
        t = tqdm(total=len(edgesin), bar_format='{l_bar}{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        def round(n):
            return int(n * 1e6) / 1e6

    if vlos is not None:

        if np.ndim(ellsin) == 0: ellsin = (ellsin,)
        ellsin = list(ellsin)
        common = False

        if not periodic:

            for imesh, mesh in enumerate(meshes[:1 if autocorr else 2]):
                meshes[imesh] = _2c(mesh)
            if autocorr:
                meshes[1] = meshes[0]

            Q = _2r(meshes[0] * meshes[1].conj()) / mattrs.meshsize.prod(dtype=rdtype)
        else:
            Q = None
            mask_in = jnp.all(edgesin >= bin.edges[0], axis=-1) & jnp.all(edgesin <= bin.edges[-1], axis=-1)
            mask_edges = jnp.all(bin.edges >= edgesin[0], axis=-1) & jnp.all(bin.edges <= edgesin[-1], axis=-1)
            common = mask_edges.sum() == mask_in.sum() and jnp.allclose(edgesin[mask_in], bin.edges[mask_edges])

        kvec = mattrs.kcoords(sparse=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)

        def _bin(Aell):
            power = []
            for ill, ell in enumerate(ells):
                leg = get_legendre(ell)(mu)
                odd = ell % 2
                if odd: leg += get_legendre(ell)(-mu)
                power.append((2 * ell + 1) / (1 + odd) * bin(Aell * leg, remove_zero=True) * rnorm[ill])
            return jnp.concatenate(power)

        def my_map(f, xs):
            if pbar:
                return np_map(f, xs)
            return jax.lax.map(f, xs=xs, batch_size=batch_size)

        wmat = []

        if 'smooth' in flags:

            spherical_jn = {ell: get_spherical_jn(ell) for ell in set(ells)}
            sedges = None
            #sedges = np.arange(0., rmesh1.boxsize.max() / 2., rmesh1.cellsize.min() / 4.)
            sbin = BinMesh2CorrelationPoles(mattrs, edges=sedges)
            kout = jnp.where(bin.nmodes == 0, 0., bin.xavg)

            for ellin in ellsin:

                wmat_tmp = []
                for ill, ell in enumerate(ells):
                    snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                    smu = sum(xx * ll for xx, ll in zip(svec, vlos)) / jnp.where(snorm == 0., 1., snorm)
                    Qs = sbin((Q if Q is not None else 1.) * get_legendre(ell)(smu) * get_legendre(ellin)(smu))
                    if ell != 0: Qs = Qs.at[0].set(0.)
                    Qs = jnp.where(sbin.nmodes == 0, 0., Qs)
                    savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                    snmodes = sbin.nmodes

                    del snorm

                    def f(edgein):
                        tophat_Qs = BesselIntegral(edgein, savg, ell=ellin, edges=True, method=tophat_method, mode='backward').w[..., 0] * mattrs.boxsize.prod() * Qs

                        def f2(kout):
                            return (-1)**(ell // 2) * jnp.sum(snmodes * spherical_jn[ell](kout * savg) * tophat_Qs)

                        batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / savg.size, 1), kout.size))
                        power = jax.lax.map(f2, kout, batch_size=batch_size)

                        if pbar:
                            t.update(n=round(1 / len(ells) / len(ellsin)))
                        return (2 * ell + 1) * power* rnorm[ill]

                    wmat_tmp.append(my_map(f, edgesin))
                wmat.append(jnp.concatenate(wmat_tmp, axis=-1))

        elif 'infinite' in flags:
            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
            smu = sum(xx * ll for xx, ll in zip(svec, vlos)) / jnp.where(snorm == 0., 1., snorm)

            for ellin in ellsin:
                legin = get_legendre(ellin)(smu)

                def f(edgein):
                    Aell = BesselIntegral(edgein, snorm, ell=ellin, edges=True, method=tophat_method, mode='backward').w[..., 0] * legin * mattrs.boxsize.prod(dtype=rdtype)
                    Aell = mattrs.create(kind='real', fill=Aell)
                    if Q is not None: Aell *= Q
                    power = _bin(_2c(Aell))
                    if pbar:
                        t.update(n=round(1 / len(ellsin)))
                    return power

                wmat.append(my_map(f, edgesin))

        else:

            for ellin in ellsin:
                legin = get_legendre(ellin)(mu)

                if common:
                    Aell = mattrs.create(kind='complex', fill=legin * mattrs.meshsize.prod(dtype=rdtype))
                    power = _bin(Aell)
                    wmat_ellin = jnp.zeros_like(power, shape=(edgesin.shape[0], power.size))
                    wmat_ellin = wmat_ellin.at[jnp.tile(jnp.flatnonzero(mask_in), len(bin.ells)), jnp.tile(mask_edges, len(bin.ells))].set(power)

                else:

                    def f(edgein):
                        Aell = mattrs.create(kind='complex', fill=((knorm >= edgein[0]) & (knorm < edgein[-1])) * legin * mattrs.meshsize.prod(dtype=rdtype))
                        if Q is not None: Aell = _2c(Q * _2r(Aell))
                        power = _bin(Aell)
                        if pbar:
                            t.update(n=round(1 / len(ellsin)))
                        return power
                    wmat_ellin = my_map(f, edgesin)

                wmat.append(wmat_ellin)

        wmat = jnp.concatenate(wmat, axis=0).T

    else:
        theory_los = 'firstpoint'
        if len(ellsin) == 2 and isinstance(ellsin[1], str):
            ellsin, theory_los = ellsin
        if np.ndim(ellsin) == 0: ellsin = (ellsin,)
        ellsin = list(ellsin)

        # In this case, theory must be a dictionary of (multipole, wide_angle_order)
        if swap: meshes = meshes[::-1]

        if periodic:
            meshes = [mattrs.create(kind='real', fill=1.)]

        rmesh1 = _2r(meshes[0])
        rmesh2 = rmesh1 if autocorr else _2r(meshes[1])
        A0 = _2c(meshes[0] if autocorr else meshes[1])
        del meshes

        # The real-space grid
        xvec = mattrs.xcoords(sparse=True)

        # The Fourier-space grid
        kvec = A0.coords(sparse=True)

        Ylms = {ell: [get_Ylm(ell, m, real=True) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}
        spherical_jn = {ell: get_spherical_jn(ell) for ell in set(ells) | set(ellsin)}
        has_buffer = False

        if isinstance(buffer, str) and buffer in ['gpu', 'cpu']:
            buffer = jax.devices(buffer)[0]
            has_buffer = True
        elif isinstance(buffer, (str, Path)):
            buffer = str(buffer)
            has_buffer = True

        def my_map(f, xs):
            if has_buffer or pbar:
                return np_map(f, xs)
            return jax.lax.map(f, xs=xs, batch_size=batch_size)

        def dump_to_buffer(mesh, key):
            toret = None
            if buffer is None:
                toret = mesh
            elif isinstance(buffer, str):
                key = '_'.join(list(map(str, key)))
                toret = os.path.join(buffer, f'mesh_{key}.npz')
                mesh.save(toret)
            else:
                toret = jax.device_put(mesh, device=buffer, donate=True)
            return toret

        def load_from_buffer(obj):
            if buffer is None:
                toret = obj
            elif isinstance(buffer, str):
                toret = RealMeshField.load(obj)
            else:
                toret = jax.device_put(obj)
            return toret

        if theory_los == 'firstpoint':

            ellsin = [ellin if isinstance(ellin, tuple) else (ellin, 0) for ellin in ellsin]
            wmat_tmp = {}
            if 'smooth' in flags:
                sedges = None
                #sedges = np.arange(0., rmesh1.boxsize.max() / 2., rmesh1.cellsize.min() / 4.)
                sbin = BinMesh2CorrelationPoles(rmesh1, edges=sedges)
                kout = jnp.where(bin.nmodes == 0, 0., bin.xavg)

                for ell1, wa1 in ellsin:
                    wmat_tmp[ell1, wa1] = 0
                    for ill, ell in enumerate(ells):
                        Qs = 0.
                        xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                        snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                        for im1, Yl1m1 in enumerate(Ylms[ell1]):
                            for im, Ylm in enumerate(Ylms[ell]):
                                Q = (4. * np.pi) / (2 * ell1 + 1) * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xvec) * Yl1m1(*xvec)).conj() * A0) * snorm**wa1
                                Qs += 4. * np.pi * sbin(Q * Ylm(*svec) * Yl1m1(*svec)) * mattrs.cellsize.prod()

                        if ell != 0: Qs = Qs.at[0].set(0.)
                        Qs = jnp.where(sbin.nmodes == 0, 0., Qs)
                        savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                        snmodes = sbin.nmodes

                        del xnorm, snorm

                        def f(edgein):
                            tophat_Qs = BesselIntegral(edgein, savg, ell=ell1, edges=True, method=tophat_method, mode='backward').w[..., 0] * Qs

                            def f2(kout):
                                return (-1)**(ell // 2) * jnp.sum(snmodes * spherical_jn[ell](kout * savg) * tophat_Qs)

                            batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / savg.size, 1), kout.size))
                            power = jax.lax.map(f2, kout, batch_size=batch_size) * rnorm[ill]

                            if pbar:
                                t.update(n=round(1 / len(ells) / len(ellsin)))
                            power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                            return power.ravel()

                        wmat_tmp[ell1, wa1] += my_map(f, edgesin)

            elif 'infinite' in flags:
                for ell1, wa1 in ellsin:
                    wmat_tmp[ell1, wa1] = 0
                    for ill, ell in enumerate(ells):
                        for im, Ylm in enumerate(Ylms[ell]):
                            xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                            Qs = 0.
                            for im1, Yl1m1 in enumerate(Ylms[ell1]):
                                Q = (4. * np.pi) / (2 * ell1 + 1) * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xvec) * Yl1m1(*xvec)).conj() * A0) * snorm**wa1
                                Qs += Yl1m1(*svec) * Q * mattrs.cellsize.prod()
                            del xnorm

                            def f(edgein):
                                tophat_Qs = BesselIntegral(edgein, snorm, ell=ell1, edges=True, method=tophat_method, mode='backward').w[..., 0] * Qs
                                power = 4 * jnp.pi * bin(Ylm(*kvec) * _2c(tophat_Qs), antisymmetric=bool(ell % 2), remove_zero=ell == 0) * rnorm[ill]
                                if pbar:
                                    t.update(n=round(1. / sum(len(Ylms[ell]) for ell in ells) / len(ellsin)))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()
                            wmat_tmp[ell1, wa1] += my_map(f, edgesin)
            else:

                for ell1, wa1 in ellsin:
                    wmat_tmp[ell1, wa1] = 0
                    Qs = {}
                    for im1, Yl1m1 in enumerate(Ylms[ell1]):
                        for ill, ell in enumerate(ells):
                            if 'recompute' in flags: Qs = {}
                            xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                            for im, Ylm in enumerate(Ylms[ell]):
                                key = ell, im, im1
                                tmp = (4. * np.pi) / (2 * ell1 + 1) * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xvec) * Yl1m1(*xvec)).conj() * A0) * snorm**wa1
                                Qs[key] = dump_to_buffer(tmp, key)
                            del xnorm, snorm
                            if 'recompute' in flags:
                                def f(edgein):
                                    Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                                    knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                                    xi = mattrs.create(kind='complex', fill=((knorm >= edgein[0]) & (knorm <= edgein[-1])) * Yl1m1(*kvec)).c2r()
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im1])

                                    Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                    power = 4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0) * rnorm[ill]
                                    del Aell[ell]
                                    if pbar:
                                        t.update(n=round((im1 + 1) / sum(len(Ylms[ell]) for ell in ells)) / sum(len(Ylms[ell]) for ell, _ in ellsin))
                                    power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                    return power.ravel()

                                wmat_tmp[ell1, wa1] += my_map(f, edgesin)

                    if 'recompute' not in flags:
                        def f(edgein):
                            Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                            knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                            for im1, Yl1m1 in enumerate(Ylms[ell1]):
                                xi = mattrs.create(kind='complex', fill=((knorm >= edgein[0]) & (knorm <= edgein[-1])) * Yl1m1(*kvec)).c2r()
                                for ell in Aell:
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im1])

                            power = []
                            for ill, ell in enumerate(ells):
                                Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                power.append(4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0) * rnorm[ill])
                                del Aell[ell]
                            if pbar:
                                t.update(n=round((im1 + 1) / sum(len(Ylms[ell]) for ell, _ in ellsin)))
                            return jnp.concatenate(power)

                        wmat_tmp[ell1, wa1] = my_map(f, edgesin)

            wmat = jnp.concatenate(list(wmat_tmp.values()), axis=0).T

        elif theory_los == 'local':

            wmat_tmp = {}
            if 'smooth' in flags:
                sedges = None
                #sedges = np.arange(0., rmesh1.boxsize.max() / 2., rmesh1.cellsize.min() / 4.)
                sbin = BinMesh2CorrelationPoles(rmesh1, edges=sedges)
                kout = jnp.where(bin.nmodes == 0, 0., bin.xavg)

                for ell1, ell2 in itertools.product((2, 0), (2, 0)):
                    wmat_tmp[ell1, ell2] = 0
                    for ill, ell in enumerate(ells):
                        ps = [p for p in (0, 2, 4) if real_gaunt((p, 0), (ell1, 0), (ell2, 0))]
                        Qs = {p: 0. for p in ps}
                        for (im1, Yl1m1), (im2, Yl2m2) in itertools.product(enumerate(Ylms[ell1]), enumerate(Ylms[ell2])):
                            for im, Ylm in enumerate(Ylms[ell]):
                                Q = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                                for p in ps:
                                    key = p
                                    tmp = 0.
                                    for imp, Ypmp in enumerate(Ylms[p]):
                                        rg = real_gaunt((p, imp - p), (ell1, im1 - ell1), (ell2, im2 - ell2))
                                        if rg:  # rg != 0
                                            tmp += rg * Ylm(*svec) * Ypmp(*svec) * Q
                                    if hasattr(tmp, 'shape'):
                                        tmp = 4. * np.pi * sbin(tmp) * mattrs.cellsize.prod()
                                        Qs[key] += dump_to_buffer(tmp, key)

                        for p in ps:
                            Q = load_from_buffer(Qs[p])
                            if ell != 0: Q = Q.at[0].set(0.)
                            Q = jnp.where(sbin.nmodes == 0, 0., Q)
                            savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                            snmodes = sbin.nmodes

                            def f(edgein):
                                tophat_Q = BesselIntegral(edgein, savg, ell=p, edges=True, method=tophat_method, mode='backward').w[..., 0] * Q

                                def f2(kout):
                                    return (-1)**(ell // 2) * jnp.sum(snmodes * spherical_jn[ell](kout * savg) * tophat_Q)

                                batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / savg.size, 1), kout.size))
                                power = jax.lax.map(f2, kout, batch_size=batch_size) * rnorm[ill]
                                if pbar:
                                    t.update(n=round(1 / len(ells) / 6))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()

                            wmat_tmp[ell1, ell2] += my_map(f, edgesin)

            elif 'infinite' in flags:
                for ell1, ell2 in itertools.product((2, 0), (2, 0)):
                    wmat_tmp[ell1, ell2] = 0
                    for ill, ell in enumerate(ells):
                        ps = [p for p in (0, 2, 4) if real_gaunt((p, 0), (ell1, 0), (ell2, 0))]
                        for im, Ylm in enumerate(Ylms[ell]):
                            Qs = {p: 0. for p in ps}
                            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                            for (im1, Yl1m1), (im2, Yl2m2) in itertools.product(enumerate(Ylms[ell1]), enumerate(Ylms[ell2])):
                                Q = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                                for p in ps:
                                    for imp, Ypmp in enumerate(Ylms[p]):
                                        rg = real_gaunt((p, imp - p), (ell1, im1 - ell1), (ell2, im2 - ell2))
                                        if rg:  # rg != 0
                                            Qs[p] += rg * Ypmp(*svec) * Q
                            for p in Qs:
                                Qs[p] = dump_to_buffer(Qs[p] * mattrs.cellsize.prod(), p)

                            def f(edgein):
                                xi = 0.
                                for p in ps:
                                    tophat = BesselIntegral(edgein, snorm, ell=p, edges=True, method=tophat_method, mode='backward').w[..., 0]
                                    Q = load_from_buffer(Qs[p])
                                    xi += tophat * Q
                                power = 4 * jnp.pi * bin(Ylm(*kvec) * _2c(xi), antisymmetric=bool(ell % 2), remove_zero=ell == 0) * rnorm[ill]
                                if pbar:
                                    t.update(n=round(1 / sum(len(Ylms[ell]) for ell in ells) / 4))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()
                            wmat_tmp[ell1, ell2] += my_map(f, edgesin)

            else:

                for ell1, ell2 in itertools.product((2, 0), (2, 0)):
                    wmat_tmp[ell1, ell2] = 0
                    Qs = {}
                    for im12, (Yl1m1, Yl2m2) in enumerate(itertools.product(Ylms[ell1], Ylms[ell2])):
                        for ill, ell in enumerate(ells):
                            if 'recompute' in flags: Qs = {}
                            for im, Ylm in enumerate(Ylms[ell]):
                                key = ell, im, im12
                                tmp = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                                Qs[key] = dump_to_buffer(tmp, key)
                            if 'recompute' in flags:
                                def f(edgein):
                                    knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                                    Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                                    xi = mattrs.create(kind='complex', fill=((knorm >= edgein[0]) & (knorm <= edgein[-1])) * Yl1m1(*kvec) * Yl2m2(*[-kk for kk in kvec])).c2r()
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im12])
                                    Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                    power = 4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0) * rnorm[ill]
                                    if pbar:
                                        t.update(n=round((im + 1) / sum(len(Ylms[ell]) for ell in ells) / 36))
                                    power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                    return power.ravel()
                                wmat_tmp[ell1, ell2] += my_map(f, edgesin)

                    if 'recompute' not in flags:
                        knorm = jnp.sqrt(sum(xx**2 for xx in kvec))

                        def f(edgein):
                            Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                            for im12, (Yl1m1, Yl2m2) in enumerate(itertools.product(Ylms[ell1], Ylms[ell2])):
                                xi = mattrs.create(kind='complex', fill=((knorm >= edgein[0]) & (knorm <= edgein[-1])) * Yl1m1(*kvec) * Yl2m2(*[-kk for kk in kvec])).c2r()
                                # Typically takes ~ 2x the time to load all Qs than the above FFT
                                # Not great, but... recomputing 15 FFTs would have taken more time
                                for ell in Aell:
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im12])
                            power = []
                            for ill, ell in enumerate(ells):
                                Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                power.append(4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0) * rnorm[ill])
                                del Aell[ell]
                            if pbar:
                                t.update(n=round((im12 + 1) / 36))
                            return jnp.concatenate(power)

                        wmat_tmp[ell1, ell2] = my_map(f, edgesin)

            wmat = jnp.zeros((len(ellsin),) + wmat_tmp[0, 0].shape, dtype=wmat_tmp[0, 0].dtype)
            coeff2 = {(0, 0): [(0, 1), (4, -7. / 18.)],
                      (0, 2): [(2, 1. / 2.), (4, -5. / 18.)],
                      (2, 2): [(4, 35. / 18.)]}
            coeff2[2, 0] = coeff2[0, 2]
            for illin, ellin in enumerate(ellsin):
                for ell1, ell2 in coeff2:
                    coeff = sum(coeff * (ell == ellin) for ell, coeff in coeff2[ell1, ell2])
                    wmat = wmat.at[illin].add(coeff * wmat_tmp[ell1, ell2])
            wmat = wmat.reshape(-1, wmat.shape[-1]).T

        else:
            raise NotImplementedError(f'theory los {theory_los} not implemented')

    observable = []
    for ill, ell in enumerate(ells):
        observable.append(Mesh2SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes, num_raw=jnp.zeros_like(bin.xavg), num_shotnoise=jnp.zeros_like(bin.xavg), norm=jnp.ones_like(bin.xavg) * norm[ill], ell=ell))
    observable = Mesh2SpectrumPoles(observable)

    theory = []
    kin, edgesin = jnp.mean(edgesin, axis=-1), edgesin
    for ill, ell in enumerate(ellsin):
        #theory.append(ObservableLeaf(k=kin, k_edges=edgesin, value=jnp.zeros_like(kin), coords=['k']))
        theory.append(Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=jnp.zeros_like(kin)))
    #theory = Mesh2SpectrumPoles(theory, ells=ellsin)
    kw = dict(ells=list(ellsin))
    if isinstance(ellsin[0], tuple):  # ell, wide-angle order
        kw = dict(ells=[ell[0] for ell in ellsin], wa_orders=[ell[1] for ell in ellsin])
    theory = ObservableTree(theory, **kw)
    return WindowMatrix(observable=observable, theory=theory, value=wmat)


def compute_mesh2_spectrum_mean(*meshes: RealMeshField | ComplexMeshField | MeshAttrs, theory: Callable | dict[Callable],
                                bin: BinMesh2SpectrumPoles=None, los: str | np.ndarray='z') -> Mesh2SpectrumPoles:
    r"""
    Compute the mean power spectrum from mesh and theory.

    Parameters
    ----------
    meshes : RealMeshField, ComplexMeshField, ComplexMeshField, MeshAttrs
        Input mesh(es).
        A :class:`MeshAttrs` instance can be directly provided, in case the selection function is trivial (constant).
    theory : Callable, dict[Callable], ObservableTree, Mesh2SpectrumPoles
        Mean theory power spectrum. Either a callable (if ``los`` is an axis),
        or a dictionary of callables, with keys the multipole orders :math:`\ell`.
        Also possible to add wide-angle order :math:`n`, such that the key is the tuple :math:`(\ell, n)`.
        One can also directly provided a :class:`ObservableTree` or :class:`Mesh2SpectrumPoles` instance.
    bin : BinMesh2SpectrumPoles
        Output binning.
    los : str, array, default=None
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    Returns
    -------
    power : Mesh2SpectrumPoles
    """
    meshes, autocorr = _format_meshes(*meshes)
    periodic = isinstance(meshes[0], MeshAttrs)
    if periodic:
        assert autocorr
        rdtype = float
        mattrs = meshes[0]
    else:
        rdtype = meshes[0].real.dtype
        mattrs = meshes[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / mattrs.cellsize.prod()
    rnorm = norm / mattrs.meshsize.prod(dtype=rdtype)

    los, vlos, swap = _format_los(los, ndim=mattrs.ndim)
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

    poles = theory
    kin = None
    theory_los = 'firstpoint'
    if isinstance(poles, tuple) and isinstance(poles[-1], str):
        theory_los = poles[-1]
        if len(poles) == 2: poles = poles[0]
        else: poles = poles[:-1]
    if isinstance(poles, tuple):
        kin, poles = poles
    if isinstance(poles, ObservableTree):
        edges = next(iter(poles)).edges('k')
        kin = jnp.append(edges[..., 0], edges[-1, 1])
        poles = {ell: poles.get(ell).value() for ell in poles.ells}
    if isinstance(poles, list):
        poles = {ell: pole for ell, pole in zip((0, 2, 4), poles)}
    kvec = mattrs.kcoords(sparse=True)
    knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
    is_poles = not callable(poles)
    if is_poles:
        is_callable = all(callable(pole) for pole in poles.values())
        if not is_callable:
            from .utils import Interpolator1D
            interp = Interpolator1D(kin, knorm, edges=len(kin) == len(poles[0]) + 1)

    def get_theory(ell=None, pole=None):
        if pole is None:
            pole = poles[ell]
        if is_callable:
            return pole(knorm)
        else:
            return interp(pole)

    if vlos is not None:

        if not periodic:

            for imesh, mesh in enumerate(meshes[:1 if autocorr else 2]):
                meshes[imesh] = _2c(mesh)
            if autocorr:
                meshes[1] = meshes[0]

            Q = _2r(meshes[0] * meshes[1].conj())
        else:
            Q = None

        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)

        pkvec = theory

        if is_poles:
            def pkvec(*args):
                return sum(get_theory(ell) * get_legendre(ell)(mu) for ell in poles)

        Aell = mattrs.create(kind='complex').apply(lambda value, kvec: pkvec(kvec) * rnorm, kind='wavenumber')
        if Q is not None: Aell = _2c(Q * _2r(Aell))
        else: Aell *= mattrs.meshsize.prod(dtype=rdtype)

        num = []
        for ell in ells:
            leg = get_legendre(ell)(mu)
            odd = ell % 2
            if odd: leg += get_legendre(ell)(-mu)
            num.append((2 * ell + 1) / (1 + odd) * bin(Aell * leg))

        spectrum = []
        for ill, ell in enumerate(ells):
            spectrum.append(Mesh2SpectrumPole(k=bin.xavg, k_edges=bin.edges, num_raw=num[ill], nmodes=bin.nmodes, norm=jnp.ones_like(bin.xavg) * norm,
                                              volume=mattrs.kfun.prod() * bin.nmodes, ell=ell, attrs=attrs))
        return Mesh2SpectrumPoles(spectrum)

    else:
        poles = {ell if isinstance(ell, tuple) else (ell, 0): pole for ell, pole in poles.items()} # wide-angle = 0 as a default

        ellsin = [mode[0] for mode in poles]

        if swap: meshes = meshes[::-1]

        if periodic:
            meshes = [mattrs.create(kind='real', fill=1.)]

        rmesh1 = _2r(meshes[0])
        rmesh2 = rmesh1 if autocorr else _2r(meshes[1])
        A0 = _2c(meshes[0] if autocorr else meshes[1])
        del meshes

        # The real-space grid
        xhat = mattrs.xcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in xhat))

        shat = mattrs.xcoords(kind='separation', sparse=True)
        snorm = jnp.sqrt(sum(xx**2 for xx in shat))

        # The Fourier-space grid
        khat = A0.coords(sparse=True)

        Ylms = {ell: [get_Ylm(ell, m, real=True) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}

        num = []
        for ell in ells:
            Aell = 0.
            for Ylm in Ylms[ell]:
                Q = 0.
                if theory_los == 'firstpoint':
                    for ell1, wa1 in poles:
                        kernel_ell1 = get_theory((ell1, wa1)) * rnorm
                        for Yl1m1 in Ylms[ell1]:
                            xi = mattrs.create(kind='complex', fill=kernel_ell1 * Yl1m1(*khat)).c2r() * snorm**wa1
                            Q += (4. * np.pi) / (2 * ell1 + 1) * xi * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xhat) * Yl1m1(*xhat)).conj() * A0)

                elif theory_los == 'local':
                    coeff2 = {(0, 0): [(0, 1), (4, -7. / 18.)],
                              (0, 2): [(2, 1. / 2.), (4, -5. / 18.)],
                              (2, 2): [(4, 35. / 18.)]}
                    coeff2[2, 0] = coeff2[0, 2]
                    for ell1, ell2 in itertools.product((0, 2), (0, 2)):
                        if is_callable:
                            kernel_ell2 = sum(coeff * get_theory((ell, 0)) for ell, coeff in coeff2[ell1, ell2])
                        else:
                            pole = sum(coeff * poles[ell, 0] for ell, coeff in coeff2[ell1, ell2])
                            kernel_ell2 = get_theory(pole=pole)
                        kernel_ell2 = kernel_ell2 * rnorm
                        for Yl1m1, Yl2m2 in itertools.product(Ylms[ell1], Ylms[ell2]):
                            xi = mattrs.create(kind='complex', fill=kernel_ell2 * Yl1m1(*khat) * Yl2m2(*[-kk for kk in khat])).c2r()
                            Q += (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * xi * _2r(_2c(rmesh1 * Ylm(*xhat) * Yl1m1(*xhat)).conj() * _2c(rmesh2 * Yl2m2(*xhat)))

                else:
                    raise NotImplementedError(f'theory los {theory_los} not implemented')

                Aell += _2c(Q) * Ylm(*khat)
            # Project on to 1d k-basis (averaging over mu=[-1, 1])
            num.append(4. * jnp.pi * bin(Aell, antisymmetric=bool(ell % 2)))

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        if swap: num = list(map(jnp.conj, num))
        # Format the power results into :class:`Mesh2SpectrumPoles` instance
        spectrum = []
        for ill, ell in enumerate(ells):
            spectrum.append(Mesh2SpectrumPole(k=bin.xavg, k_edges=bin.edges, nmodes=bin.nmodes, num_raw=num[ill], num_shotnoise=jnp.zeros_like(num[ill]), norm=jnp.ones_like(bin.xavg) * norm,
                                              volume=mattrs.kfun.prod() * bin.nmodes, ell=ell, attrs=attrs))
        return Mesh2SpectrumPoles(spectrum)
