import os
from functools import partial, lru_cache
from dataclasses import dataclass, asdict, field
from collections.abc import Callable
import itertools
from pathlib import Path

import jax.experimental
import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from scipy import special

from . import utils
from .utils import legendre, plotter, BinnedStatistic, WindowMatrix
from .mesh import RealMeshField, ComplexMeshField, HermitianComplexMeshField, ParticleField, staticarray, MeshAttrs, BinAttrs, get_common_mesh_attrs


@jax.tree_util.register_pytree_node_class
class PowerSpectrumMultipoles(BinnedStatistic):

    _rename_fields = BinnedStatistic._rename_fields | {'_x': 'k', '_projs': 'ells', '_value': 'power_nonorm', '_weights': 'nmodes', '_norm': 'norm',
                                                       '_shotnoise_nonorm': 'shotnoise_nonorm', '_power_zero_nonorm': 'power_zero_nonorm'}
    _label_x = r'$k$ [$h/\mathrm{Mpc}$]'
    _label_proj = r'$\ell$'
    _label_value = r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]'
    _data_fields = BinnedStatistic._data_fields + ['_norm', '_shotnoise_nonorm', '_power_zero_nonorm']
    _select_proj_fields = BinnedStatistic._select_proj_fields + ['_power_zero_nonorm']

    def __init__(self, k: np.ndarray, power_nonorm: jax.Array, nmodes: np.ndarray, edges: np.ndarray, ells: tuple, norm: jax.Array=1.,
                 shotnoise_nonorm: jax.Array=0., power_zero_nonorm: jax.Array=None, name: str=None, attrs: dict=None):

        def _tuple(item):
            if not isinstance(item, (tuple, list)):
                item = (item,) * len(ells)
            return tuple(item)

        self.__dict__.update(_norm=norm, _shotnoise_nonorm=shotnoise_nonorm, _power_zero_nonorm=tuple(power_zero_nonorm))
        super().__init__(x=_tuple(k), edges=_tuple(edges), projs=ells, value=power_nonorm,
                         weights=_tuple(nmodes), name=name, attrs=attrs)

    @property
    def norm(self):
        """Power spectrum normalization."""
        return self._norm

    @property
    def shotnoise(self):
        """Shot noise."""
        return self._shotnoise_nonorm / self._norm

    @property
    def power_nonorm(self):
        """Power spectrum without shot noise."""
        return self._value

    k = BinnedStatistic.x
    kavg = BinnedStatistic.xavg
    nmodes = BinnedStatistic.weights

    @property
    def ells(self):
        return self._projs

    @property
    def value(self):
        """Power spectrum estimate."""
        toret = list(self.power_nonorm)
        if 0 in self._projs:
            ill = self._projs.index(0)
            toret[ill] = toret[ill] - self._shotnoise_nonorm
        for ill, ell in enumerate(self._projs):
            dig_zero = np.digitize(0., self._edges[ill], right=False) - 1
            if 0 <= dig_zero < self.power_nonorm[ill].shape[0]:
                with np.errstate(divide='ignore', invalid='ignore'):
                    toret[ill] = toret[ill].at[..., dig_zero].add(- self._power_zero_nonorm[ill] / self._weights[ill][dig_zero])
        return tuple(tmp / self._norm for tmp in toret)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot power spectrum.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(self._x[ill], self._x[ill] * self.value[ill].real, label=self._get_label_proj(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(self._label_x)
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig


def _get_batch_Ylm(Ylm):

    if Ylm.ell == 0:
        return lambda *coords: 1. / np.sqrt(4. * np.pi)

    def _safe_divide(num, denom):
        with np.errstate(divide='ignore', invalid='ignore'):
            return jnp.where(denom == 0., 0., num / denom)

    @jax.checkpoint
    def func(*coords):
        if coords[0].ndim < len(coords):
            coords = jnp.meshgrid(*coords, indexing='ij', sparse=True)
        coords = tuple(coords)
        def f(x):
            x = (x,) + coords[1:]
            xnorm = jnp.sqrt(sum(xx**2 for xx in x))
            xhat = tuple(_safe_divide(xx, xnorm) for xx in x)
            return Ylm(*xhat)#[0]
        nbatch = 1
        if nbatch == 1:
            return f(coords[0])
        total_size = len(coords[0])
        batch_size = total_size // nbatch
        return jnp.concatenate([f(coords[0][i * total_size // nbatch:(i + 1) * total_size // nbatch]) for i in range(nbatch)])
        #return jax.lax.map(f, coords[0].ravel(), batch_size=batch_size)

    return func


@lru_cache(maxsize=32, typed=False)
def get_real_Ylm(ell, m, batch=True):
    """
    Return a function that computes the real spherical harmonic of order (ell, m).
    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py.

    Note
    ----
    Faster (and differentiable) evaluation will be achieved if sympy is available.
    Else, fallback to scipy's functions.
    I am not using :func:`jax.scipy.special.lpmn_values` as this returns all ``ell``, ``m``'s at once,
    which is not great for memory reasons.

    Parameters
    ----------
    ell : int
        The degree of the harmonic.

    m : int
        The order of the harmonic; abs(m) <= ell.

    Returns
    -------
    Ylm : callableCallable
        A function that takes 3 arguments: (xhat, yhat, zhat)
        unit-normalized Cartesian coordinates and returns the
        specified Ylm.

    References
    ----------
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form

    """
    # Make sure ell, m are integers
    ell = int(ell)
    m = int(m)

    # Normalization of Ylms
    amp = np.sqrt((2 * ell + 1) / (4 * np.pi))
    if m != 0:
        fac = 1
        for n in range(ell - abs(m) + 1, ell + abs(m) + 1): fac *= n  # (ell + |m|)!/(ell - |m|)!
        amp *= np.sqrt(2. / fac)

    try: import sympy as sp
    except ImportError: sp = None

    # sympy is not installed, fallback to scipy
    if sp is None:

        def Ylm(xhat, yhat, zhat):
            # The cos(theta) dependence encoded by the associated Legendre polynomial
            toret = amp * (-1)**m * special.lpmv(abs(m), ell, zhat)
            # The phi dependence
            phi = np.arctan2(yhat, xhat)
            if m < 0:
                toret *= np.sin(abs(m) * phi)
            else:
                toret *= np.cos(m * phi)
            return toret

        # Attach some meta-data
        Ylm.ell = ell
        Ylm.m = m

    else:
        # The relevant cartesian and spherical symbols
        # Using intermediate variable r helps sympy simplify expressions
        x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
        xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
        phi, theta = sp.symbols('phi theta')
        defs = [(sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
                (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
                (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2))]

        # The cos(theta) dependence encoded by the associated Legendre polynomial
        expr = (-1)**m * sp.assoc_legendre(ell, abs(m), sp.cos(theta))

        # The phi dependence
        if m < 0:
            expr *= sp.expand_trig(sp.sin(abs(m) * phi))
        elif m > 0:
            expr *= sp.expand_trig(sp.cos(m * phi))

        # Simplify
        expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
        expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat), (z / r, zhat)])
        Ylm = sp.lambdify((xhat, yhat, zhat), expr, modules='jax')

        # Attach some meta-data
        Ylm.expr = expr
        Ylm.ell = ell
        Ylm.m = m

    if batch:
        return _get_batch_Ylm(Ylm)
    return Ylm


[[get_real_Ylm(ell, m, batch=True) for m in range(-ell, ell + 1)] for ell in (0, 2, 4)]


def _get_los_vector(los: str | np.ndarray, ndim=3):
    vlos = None
    if isinstance(los, str):
        vlos = [0.] * ndim
        vlos['xyz'.index(los)] = 1.
    else:
        vlos = los
    return staticarray(vlos)


def _get_edges(edges, boxsize=1., meshsize=1., **kw):
    kfun, knyq = np.min(2. * np.pi / boxsize), np.min(np.pi *  meshsize / boxsize)
    if edges is None:
        edges = {}
    if isinstance(edges, dict):
        kmin = edges.get('min', 0.)
        kmax = edges.get('max', knyq)
        kstep = edges.get('step', kfun)
        edges = np.arange(kmin, kmax, kstep)
    else:
        edges = np.asarray(edges)
    return edges


def _get_power_zero(mesh):
    toret = mesh[(0,) * mesh.ndim]
    if isinstance(mesh, HermitianComplexMeshField):
        return toret.real
    return toret


def compute_mesh_power(*meshs: RealMeshField | ComplexMeshField | HermitianComplexMeshField, edges: np.ndarray | dict | None=None,
                       ells: int | tuple=0, los: str | np.ndarray='x', mode_oversampling: int=0) -> PowerSpectrumMultipoles:
    r"""
    Compute power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, HermitianComplexMeshField
        Input mesh(s).

    edges : np.ndarray, dict, default=None
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, array, default=None
        If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    power : PowerSpectrumMultipoles
    """
    meshs = list(meshs)
    assert 1 <= len(meshs) <= 2
    rdtype = meshs[0].real.dtype
    mattrs = meshs[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.boxsize / mattrs.meshsize, dtype=rdtype)
    edges = _get_edges(edges, **mattrs)

    ndim = len(mattrs.boxsize)
    vlos, swap = None, False
    if isinstance(los, str) and los in ['firstpoint', 'endpoint']:
        swap = los == 'endpoint'
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    attrs = dict(los=vlos if vlos is not None else los) | dict(mattrs)

    if np.ndim(ells) == 0:
        ells = (ells,)
    ells = tuple(sorted(ells))

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, (ComplexMeshField, HermitianComplexMeshField)):
            mesh = mesh.r2c()
        return mesh

    autocorr = len(meshs) == 1
    if vlos is None:  # local, varying line-of-sight
        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]
        if swap: meshs = meshs[::-1]

        A0 = _2c(meshs[0] if autocorr else meshs[1])
        bin = BinAttrs(A0, edges=edges, mode_oversampling=mode_oversampling)

        power, power_zero = [], []
        if 0 in ells:
            Aell = A0 if autocorr else _2c(meshs[1])
            Aell = Aell.conj() * A0
            power.append(bin(Aell, antisymmetric=False))
            power_zero.append(_get_power_zero(Aell))

        if nonzeroells:

            rmesh1 = _2r(meshs[0])
            # The real-space grid
            xvec = rmesh1.coords(sparse=True)
            # The Fourier-space grid
            kvec = A0.coords(sparse=True)
            Ylms = {ell: [get_real_Ylm(ell, m, batch=True) for m in range(-ell, ell + 1)] for ell in nonzeroells}

            @jax.checkpoint
            def f(carry, im):
                carry += _2c(rmesh1 * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            for ell in nonzeroells:
                Ylm = Ylms[ell]
                Aell = jax.lax.scan(f, init=A0.clone(value=jnp.zeros_like(A0.value)), xs=np.arange(len(Ylm)))[0].conj() * A0
                #Aell = sum(_2c(rmesh1 * Ylm(*xvec)) * Ylm(*kvec) for Ylm in Ylms[ell]).conj() * A0
                # Project on to 1d k-basis (averaging over mu=[-1, 1])
                power.append(4. * jnp.pi * bin(Aell, antisymmetric=bool(ell % 2)))
                power_zero.append(4. * jnp.pi * 0.)

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        power, power_zero = jnp.array(power), jnp.array(power_zero)
        if swap: power, power_zero = power.conj(), power_zero.conj()
        # Format the power results into :class:`PowerSpectrumMultipoles` instance
        return PowerSpectrumMultipoles(bin.xavg, power_nonorm=power, nmodes=bin.nmodes, edges=edges, ells=ells, norm=norm,
                                       power_zero_nonorm=power_zero, attrs=attrs)

    else:  # fixed line-of-sight

        for imesh, mesh in enumerate(meshs):
            meshs[imesh] = _2c(mesh)
        if autocorr:
            meshs.append(meshs[0])

        Aell = meshs[0] * meshs[1].conj()
        del meshs
        bin = BinAttrs(Aell, edges=edges, mode_oversampling=mode_oversampling)
        kvec = Aell.coords(sparse=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * kvec[0].ndim].set(1.)
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / knorm
        del knorm
        power, power_zero = [], []
        for ell in ells:
            leg = legendre(ell)(mu)
            odd = ell % 2
            if odd: leg += legendre(ell)(-mu)
            power.append((2 * ell + 1) / (1 + odd) * bin(Aell * leg))
            power_zero.append(0.)
            if ell == 0:
                power_zero[-1] += _get_power_zero(Aell)
        power, power_zero = jnp.array(power), jnp.array(power_zero)
        return PowerSpectrumMultipoles(bin.xavg, power_nonorm=power, nmodes=bin.nmodes, edges=edges, ells=ells, norm=norm,
                                       power_zero_nonorm=power_zero, attrs=attrs)


@partial(jax.tree_util.register_dataclass, data_fields=['data', 'randoms'], meta_fields=[])
@dataclass(frozen=True, init=False)
class FKPField(object):
    """
    Class defining the FKP field, data - randoms.

    References
    ----------
    https://arxiv.org/abs/astro-ph/9304022
    """
    data: ParticleField
    randoms: ParticleField

    def __init__(self, data, randoms, **kwargs):
        data, randoms = ParticleField.same_mesh(data, randoms, **kwargs)
        self.__dict__.update(data=data, randoms=randoms)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = {name: getattr(self, name) for name in ['data', 'randoms']} | kwargs
        return self.__class__(**state)

    def paint(self, resampler: str | Callable='cic', interlacing: int=1,
              compensate: bool=False, dtype=None, out: str='real'):
        fkp = self.data - self.data.sum() / self.randoms.sum() * self.randoms
        return fkp.paint(resampler=resampler, interlacing=interlacing, compensate=compensate, dtype=dtype, out=out)

    @staticmethod
    def same_mesh(*others, **kwargs):
        attrs = get_common_mesh_attrs(*([other.data for other in others] + [other.randoms for other in others]), **kwargs)
        return tuple(other.clone(**attrs) for other in others)


def compute_normalization(*inputs: RealMeshField | ParticleField, resampler='cic', **kwargs) -> jax.Array:
    """
    Return normalization, in 1 / volume unit.

    Warning
    -------
    Input particles are considered uncorrelated.
    """
    meshs, particles = [], []
    attrs = {}
    for inp in inputs:
        if isinstance(inp, RealMeshField):
            meshs.append(inp)
            attrs = {name: getattr(inp, name) for name in ['boxsize', 'boxcenter', 'meshsize']}
        else:
            particles.append(inp)
    if particles: particles = ParticleField.same_mesh(*particles, **attrs)
    normalization = 1
    for mesh in meshs:
        normalization *= mesh
    for particle in particles:
        normalization *= particle.paint(resampler=resampler, interlacing=1, compensate=False)
    return normalization.sum() / normalization.cellsize.prod()


def compute_fkp_power(*fkps: FKPField, edges: np.ndarray | dict | None=None,
                      resampler='tsc', interlacing=3, ells: int | tuple=0, los: str | np.ndarray='x', mode_oversampling: int=0) -> PowerSpectrumMultipoles:
    r"""
    Compute power spectrum from FKP field.

    Parameters
    ----------
    meshs : FKPField
        Input FKP fields.

    edges : np.ndarray, dict, default=None
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    resampler : str, callable
        Resampler to read particule weights from mesh.
        One of ['ngp', 'cic', 'tsc', 'pcs'].

    interlacing : int, default=1
        If 1, no interlacing correction.
        If > 1, order of interlacing correction.
        Typically, 3 gives reliable power spectrum estimation up to :math:`k \sim k_\mathrm{nyq}`.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, array, default=None
        If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    power : PowerSpectrumMultipoles
    """
    fkps = FKPField.same_mesh(*fkps)
    meshs = [fkp.paint(resampler=resampler, interlacing=interlacing, compensate=True, out='complex') for fkp in fkps]
    # TODO: generalize to N fkp fields
    if len(fkps) > 1:
        shotnoise = 0.
        randoms = [fkp.randoms for fkp in fkps]
        alpha2 = jnp.array([fkp.data.sum() / fkp.randoms.sum() for fkp in fkps]).prod()
    else:
        fkp = fkps[0]
        alpha = fkp.data.sum() / fkp.randoms.sum()
        shotnoise = jnp.sum(fkp.data.weights**2) + alpha**2 * jnp.sum(fkp.randoms.weights**2)
        #mask = random.uniform(random.key(42), shape=fkp.randoms.size) < 0.5
        #randoms = [fkp.randoms[mask], fkp.randoms[~mask]]
        randoms = [fkp.randoms[:fkp.randoms.size // 2], fkp.randoms[fkp.randoms.size // 2:]]
        alpha2 = jnp.array([fkp.data.sum() / randoms.sum() for randoms in randoms]).prod()
    norm = alpha2 * compute_normalization(*randoms, cellsize=10.)
    return compute_mesh_power(*meshs, edges=edges, ells=ells, los=los, mode_oversampling=mode_oversampling).clone(norm=norm, shotnoise_nonorm=shotnoise)


def compute_wide_angle_poles(poles: dict[Callable]):
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


def compute_mesh_window(*meshs: RealMeshField | ComplexMeshField | HermitianComplexMeshField | MeshAttrs, edgesin: np.ndarray, ellsin: tuple=None,
                        edges: np.ndarray | dict | None=None, ells: int | tuple=0, los: str | np.ndarray='x', mode_oversampling: int=0,
                        buffer=None, batch_size=None, pbar=False, norm=None) -> jax.Array:
    r"""
    Compute mean power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, HermitianComplexMeshField, MeshAttrs
        Input mesh(s).
        A :class:`MeshAttrs` instance can be directly provided, in case the selection function is trivial (constant).

    edges : np.ndarray, dict, default=None
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, array, default=None
        If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    wmat : WindowMatrix
    """
    meshs = list(meshs)
    assert 1 <= len(meshs) <= 2
    periodic = isinstance(meshs[0], MeshAttrs)
    if periodic:
        assert len(meshs) == 1
        rdtype = float
        mattrs = meshs[0]
    else:
        rdtype = meshs[0].real.dtype
        mattrs = meshs[0].attrs
    if norm is None:
        norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.boxsize / mattrs.meshsize, dtype=rdtype)
    edges = _get_edges(edges, **mattrs)

    ndim = len(mattrs.boxsize)
    vlos, swap = None, False
    if isinstance(los, str) and los in ['firstpoint', 'endpoint']:
        swap = los == 'endpoint'
    else:
        vlos = _get_los_vector(los, ndim=ndim)

    if np.ndim(ells) == 0: ells = (ells,)
    ells = tuple(ells)

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, (ComplexMeshField, HermitianComplexMeshField)):
            mesh = mesh.r2c()
        return mesh

    autocorr = len(meshs) == 1
    if edgesin.ndim == 2:
        kin = edgesin
        edgesin = None
    else:
        kin = jnp.array([edgesin[:-1], edgesin[1:]]).T

    if vlos is not None:

        if np.ndim(ellsin) == 0: ellsin = (ellsin,)
        ellsin = list(ellsin)

        if not periodic:

            for imesh, mesh in enumerate(meshs):
                meshs[imesh] = _2c(mesh)
            if autocorr:
                meshs.append(meshs[0])

            Q = _2r(meshs[0] * meshs[1].conj())
        else:
            Q = None

        bin = BinAttrs(mattrs, edges=edges, kind='complex_hermitian', mode_oversampling=mode_oversampling)
        kvec = mattrs.kcoords(sparse=True, hermitian=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * kvec[0].ndim].set(1.)

        wmat = []
        for ellin in ellsin:
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / knorm
            legin = legendre(ellin)(mu)

            def f(kin):

                def pkvec(*args):
                    kmask = (knorm >= kin[0]) & (knorm <= kin[-1])
                    return kmask * legin

                Aell = mattrs.create(kind='hermitian_complex').apply(lambda value, kvec: pkvec(kvec) * norm, kind='wavenumber')
                if Q is not None: Aell = _2c(Q * _2r(Aell))

                power = []
                for ell in ells:
                    leg = legendre(ell)(mu)
                    odd = ell % 2
                    if odd: leg += legendre(ell)(-mu)
                    power.append((2 * ell + 1) / (1 + odd) * bin(Aell * leg, remove_zero=ell == 0))
                return jnp.concatenate(power)

            wmat.append(jax.lax.map(f, xs=kin, batch_size=batch_size))
        wmat = jnp.concatenate(wmat, axis=0).T

    else:
        theory_los = 'firstpoint'
        if ellsin[1] == 'local':
            theory_los = 'local'
            ellsin = ellsin[0]
        if np.ndim(ellsin) == 0: ellsin = (ellsin,)
        ellsin = list(ellsin)

        # In this case, theory must be a dictionary of (multipole, wide_angle_order)
        if swap: meshs = meshs[::-1]

        if periodic:
            meshs = [mattrs.create(kind='real', fill=1.)]

        def _safe_divide(num, denom):
            with np.errstate(divide='ignore', invalid='ignore'):
                return jnp.where(denom == 0., 0., num / denom)

        rmesh1 = _2r(meshs[0])
        rmesh2 = rmesh1 if autocorr else _2r(meshs[1])
        A0 = _2c(meshs[0] if autocorr else meshs[1])
        del meshs

        bin = BinAttrs(A0, edges=edges, mode_oversampling=mode_oversampling)

        # The real-space grid
        xvec = mattrs.xcoords(sparse=True)

        def _wrap_rslab(rslab):
            return tuple(jnp.where(rr > mattrs.boxsize[ii] / 2., rr - mattrs.boxsize[ii], rr) for ii, rr in enumerate(rslab))

        svec = _wrap_rslab(mattrs.clone(boxcenter=mattrs.boxsize / 2.).xcoords(sparse=True))

        # The Fourier-space grid
        kvec = A0.coords(sparse=True)

        Ylms = {ell: [get_real_Ylm(ell, m, batch=True) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}
        has_buffer = False
        if isinstance(buffer, str) and buffer in ['gpu', 'cpu']:
            buffer = jax.devices(buffer)[0]
            has_buffer = True
        elif isinstance(buffer, (str, Path)):
            buffer = str(buffer)
            has_buffer = True

        def my_map(f, xs):
            if has_buffer:
                return jnp.array(list(map(f, xs)))
            return jax.lax.map(f, xs=xs, batch_size=batch_size)

        if not has_buffer:
            pbar = False

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

        if pbar:
            from tqdm import tqdm
            t = tqdm(total=len(kin), bar_format='{l_bar}{bar}| {n:.3f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

            def round(n):
                return int(n * 1e6) / 1e6

        if theory_los == 'firstpoint':

            ellsin = [ellin if isinstance(ellin, tuple) else (ellin, 0) for ellin in ellsin]
            nellsin = sum(len(Ylms[ell]) for ell, _ in ellsin)
            wmat = []
            for ell1, wa1 in ellsin:
                xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                Qs = {}
                for ill, ell in enumerate(ells):
                    for im, Ylm in enumerate(Ylms[ell]):
                        for im1, Yl1m1 in enumerate(Ylms[ell1]):
                            key = ell, im, im1
                            tmp = (4. * np.pi) / (2 * ell1 + 1) * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xvec) * Yl1m1(*xvec)).conj() * A0) * snorm**wa1
                            Qs[key] = dump_to_buffer(tmp, key)

                del xnorm, snorm
                knorm = jnp.sqrt(sum(xx**2 for xx in kvec))

                wmat = []

                def f(kin):
                    Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                    for im1, Yl1m1 in enumerate(Ylms[ell1]):
                        def kernel(*args):
                            kmask = (knorm >= kin[0]) & (knorm <= kin[-1])
                            return kmask * norm / mattrs.meshsize.prod(dtype=rdtype) * Yl1m1(*kvec)

                        xi = mattrs.create(kind='hermitian_complex').apply(kernel, kind='wavenumber').c2r()
                        for ell in Aell:
                            for im in Aell[ell]:
                                Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im1])

                    power = []
                    for ill, ell in enumerate(ells):
                        Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                        power.append(4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0))
                        del Aell[ell]
                    if pbar:
                        t.update(n=round((im1 + 1) / nellsin))
                    return jnp.concatenate(power)

                wmat.append(my_map(f, kin))

            wmat = jnp.concatenate(wmat, axis=0).T

        elif theory_los == 'local':

            wmat_tmp = {}
            for ell1, ell2 in list(itertools.product((0, 2), (0, 2))):
                Qs = {}
                for im12, (Yl1m1, Yl2m2) in enumerate(itertools.product(Ylms[ell1], Ylms[ell2])):
                    for ill, ell in enumerate(ells):
                        for im, Ylm in enumerate(Ylms[ell]):
                            key = ell, im, im12
                            tmp = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                            Qs[key] = dump_to_buffer(tmp, key)

                def f(kin):
                    Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                    for im12, (Yl1m1, Yl2m2) in enumerate(itertools.product(Ylms[ell1], Ylms[ell2])):
                        knorm = jnp.sqrt(sum(xx**2 for xx in kvec))

                        def kernel(*args):
                            kmask = (knorm >= kin[0]) & (knorm <= kin[-1])
                            return kmask * norm / mattrs.meshsize.prod(dtype=rdtype) * Yl1m1(*kvec) * Yl2m2(*[-kk for kk in kvec])

                        xi = mattrs.create(kind='hermitian_complex').apply(kernel, kind='wavenumber').c2r()
                        # Typically takes ~ 2x the time to load all Qs than the above FFT
                        # Not great, but... recomputing 15 FFTs would have taken more time
                        for ell in Aell:
                            for im in Aell[ell]:
                                Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im12])

                    power = []
                    for ill, ell in enumerate(ells):
                        Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                        power.append(4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0))
                        del Aell[ell]
                    if pbar:
                        t.update(n=round((im12 + 1) / 36))
                    return jnp.concatenate(power)

                wmat_tmp[ell1, ell2] = my_map(f, kin)

            wmat = jnp.zeros((len(ellsin),) + wmat_tmp[0, 0].shape)
            coeff2 = {(0, 0): [(0, 1), (- 7. / 18., 4)],
                      (0, 2): [(2, 1. / 2.), (4, - 5. / 18.)],
                      (2, 2): [(4, 35. / 18.)]}
            coeff2[2, 0] = coeff2[0, 2]
            for illin, ellin in enumerate(ellsin):
                for ell1, ell2 in coeff2:
                    coeff = sum(coeff * (ell == ellin) for ell, coeff in coeff2[ell1, ell2])
                    wmat = wmat.at[illin].add(coeff * wmat_tmp[ell1, ell2])
            wmat = wmat.reshape(-1, wmat.shape[-1]).T

        else:
            raise NotImplementedError(f'theory los {theory_los} not implemented')

    observable = BinnedStatistic(x=[bin.xavg] * len(ells), edges=[edges] * len(ells), projs=ells)
    theory = BinnedStatistic(x=[np.mean(kin, axis=-1)] * len(ellsin), edges=[edgesin] * len(ellsin) if edgesin is not None else None, projs=ellsin)
    wmat = WindowMatrix(observable, theory, wmat, attrs={'norm': norm})
    return wmat



def compute_mean_mesh_power(*meshs: RealMeshField | ComplexMeshField | HermitianComplexMeshField | MeshAttrs, theory: Callable | dict[Callable],
                            edges: np.ndarray | dict | None=None, ells: int | tuple=0, los: str | np.ndarray='x', mode_oversampling: int=0) -> PowerSpectrumMultipoles:
    r"""
    Compute mean power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, HermitianComplexMeshField, MeshAttrs
        Input mesh(s).
        A :class:`MeshAttrs` instance can be directly provided, in case the selection function is trivial (constant).

    theory : Callable, dict[Callable]
        Mean theory power spectrum. Either a callable (if ``los`` is an axis),
        or a dictionary of callables, with keys the multipole orders :math:`\ell`.
        Also possible to add wide-angle order :math:`n`, such that the key is the tuple :math:`(\ell, n)`.

    edges : np.ndarray, dict, default=None
        ``edges`` may be:
        - a numpy array containing the :math:`k`-edges.
        - a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi / (boxsize / meshsize)``),
            'step' (if not provided :func:`find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').
        - ``None``, defaults to empty dictionary.

    ells : tuple, default=0
        Multiple orders to compute.

    los : str, array, default=None
        If ``los`` is 'firstpoint' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    power : PowerSpectrumMultipoles
    """
    meshs = list(meshs)
    assert 1 <= len(meshs) <= 2
    periodic = isinstance(meshs[0], MeshAttrs)
    if periodic:
        assert len(meshs) == 1
        rdtype = float
        mattrs = meshs[0]
    else:
        rdtype = meshs[0].real.dtype
        mattrs = meshs[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.boxsize / mattrs.meshsize, dtype=rdtype)
    edges = _get_edges(edges, **mattrs)

    ndim = len(mattrs.boxsize)
    vlos, swap = None, False
    if isinstance(los, str) and los in ['firstpoint', 'endpoint']:
        swap = los == 'endpoint'
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    attrs = dict(los=vlos if vlos is not None else los) | dict(mattrs)

    if np.ndim(ells) == 0:
        ells = (ells,)
    ells = tuple(sorted(ells))

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, (ComplexMeshField, HermitianComplexMeshField)):
            mesh = mesh.r2c()
        return mesh

    autocorr = len(meshs) == 1

    poles = theory
    kin = None
    theory_los = 'firstpoint'
    if isinstance(poles, tuple) and isinstance(poles[-1], str):
        poles, theory_los = poles
    if isinstance(poles, tuple):
        kin, poles = poles
    if isinstance(poles, BinnedStatistic):
        kin, poles = poles._edges[0], {proj: poles.view(projs=proj) for proj in poles.projs}
    if isinstance(poles, list):
        poles = {ell: pole for ell, pole in zip((0, 2, 4), poles)}

    kvec = mattrs.kcoords(sparse=True, hermitian=True)
    kshape = np.broadcast_shapes(*(kk.shape for kk in kvec))
    knorm = jnp.sqrt(sum(kk**2 for kk in kvec)).ravel()
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

            for imesh, mesh in enumerate(meshs):
                meshs[imesh] = _2c(mesh)
            if autocorr:
                meshs.append(meshs[0])

            Q = _2r(meshs[0] * meshs[1].conj())
        else:
            Q = None

        bin = BinAttrs(mattrs, edges=edges, kind='complex_hermitian', mode_oversampling=mode_oversampling)
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)).ravel() / jnp.where(knorm == 0., 1., knorm)

        pkvec = theory

        if is_poles:
            def pkvec(*args):
                return sum(get_theory(ell) * legendre(ell)(mu) for ell in theory)

        Aell = mattrs.create(kind='hermitian_complex').apply(lambda value, kvec: pkvec(kvec).reshape(kshape) * norm, kind='wavenumber')
        if Q is not None: Aell = _2c(Q * _2r(Aell))

        power, power_zero = [], []
        for ell in ells:
            leg = legendre(ell)(mu)
            odd = ell % 2
            if odd: leg += legendre(ell)(-mu)
            power.append((2 * ell + 1) / (1 + odd) * bin(Aell * leg))
            power_zero.append(0.)
            if ell == 0:
                power_zero[-1] += _get_power_zero(Aell)

        return PowerSpectrumMultipoles(bin.xavg, power_nonorm=power, nmodes=bin.nmodes, edges=edges, ells=ells, norm=norm, power_zero_nonorm=power_zero, attrs=mattrs)

    else:
        poles = {ell if isinstance(ell, tuple) else (ell, 0): pole for ell, pole in poles.items()} # wide-angle = 0 as a default

        ellsin = [mode[0] for mode in poles]

        if swap: meshs = meshs[::-1]

        if periodic:
            meshs = [mattrs.create(kind='real', fill=1.)]

        rmesh1 = _2r(meshs[0])
        rmesh2 = rmesh1 if autocorr else _2r(meshs[1])
        A0 = _2c(meshs[0] if autocorr else meshs[1])
        del meshs

        bin = BinAttrs(A0, edges=edges, mode_oversampling=mode_oversampling)
        # The real-space grid
        xhat = mattrs.xcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in xhat))

        def _wrap_rslab(rslab):
            return tuple(jnp.where(rr > mattrs.boxsize[ii] / 2., rr - mattrs.boxsize[ii], rr) for ii, rr in enumerate(rslab))

        shat = _wrap_rslab(mattrs.clone(boxcenter=mattrs.boxsize / 2.).xcoords(sparse=True))
        snorm = jnp.sqrt(sum(xx**2 for xx in shat))

        # The Fourier-space grid
        khat = A0.coords(sparse=True)

        Ylms = {ell: [get_real_Ylm(ell, m, batch=True) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}

        power, power_zero = [], []
        for ell in ells:
            Aell = 0.
            for Ylm in Ylms[ell]:
                Q = 0.
                if theory_los == 'firstpoint':
                    for ell1, wa1 in poles:
                        kernel_ell1 = get_theory((ell1, wa1))
                        kernel_ell1 = kernel_ell1.reshape(kshape) * norm / mattrs.meshsize.prod(dtype=rdtype)
                        for Yl1m1 in Ylms[ell1]:

                            def kernel(*args):
                                return kernel_ell1 * Yl1m1(*khat)

                            xi = mattrs.create(kind='hermitian_complex').apply(kernel, kind='wavenumber').c2r() * snorm**wa1
                            Q += (4. * np.pi) / (2 * ell1 + 1) * xi * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xhat) * Yl1m1(*xhat)).conj() * A0)

                elif theory_los == 'local':
                    coeff2 = {(0, 0): [(0, 1), (- 7. / 18., 4)],
                              (0, 2): [(2, 1. / 2.), (4, - 5. / 18.)],
                              (2, 2): [(4, 35. / 18.)]}
                    coeff2[2, 0] = coeff2[0, 2]
                    for ell1, ell2 in itertools.product((0, 2), (0, 2)):
                        if is_callable:
                            kernel_ell2 = sum(coeff * get_theory((ell, 0)) for ell, coeff in coeff2[ell1, ell2])
                        else:
                            pole = sum(coeff * poles[ell, 0] for ell, coeff in coeff2[ell1, ell2])
                            kernel_ell2 = get_theory(pole=pole)
                        kernel_ell2 = kernel_ell2.reshape(kshape) * norm / mattrs.meshsize.prod(dtype=rdtype)
                        for Yl1m1, Yl2m2 in itertools.product(Ylms[ell1], Ylms[ell2]):

                            def kernel(*args):
                                return kernel_ell2 * Yl1m1(*khat) * Yl2m2(*[-kk for kk in khat])

                            #xi = mattrs.create(kind='complex').apply(kernel, kind='wavenumber').c2r().real
                            xi = mattrs.create(kind='hermitian_complex').apply(kernel, kind='wavenumber').c2r()
                            Q += (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * xi * _2r(_2c(rmesh1 * Ylm(*xhat) * Yl1m1(*xhat)).conj() * _2c(rmesh2 * Yl2m2(*xhat)))

                else:
                    raise NotImplementedError(f'theory los {theory_los} not implemented')

                Aell += _2c(Q) * Ylm(*khat)
            # Project on to 1d k-basis (averaging over mu=[-1, 1])
            power.append(4. * jnp.pi * bin(Aell, antisymmetric=bool(ell % 2)))
            power_zero.append(4. * jnp.pi * 0.)

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        power, power_zero = jnp.array(power), jnp.array(power_zero)
        if swap: power, power_zero = power.conj(), power_zero.conj()
        # Format the power results into :class:`PowerSpectrumMultipoles` instance
        return PowerSpectrumMultipoles(bin.xavg, power_nonorm=power, nmodes=bin.nmodes, edges=edges, ells=ells, norm=norm,
                                       power_zero_nonorm=power_zero, attrs=attrs)
