import os
from functools import partial, lru_cache
from dataclasses import dataclass, asdict, field
from collections.abc import Callable
import itertools
from pathlib import Path

import numpy as np
from scipy import special

import jax
from jax import numpy as jnp
import jax.experimental
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from . import utils
from .utils import legendre, plotter, BinnedStatistic, WindowMatrix
from .mesh import RealMeshField, ComplexMeshField, ParticleField, staticarray, MeshAttrs, BinAttrs, get_sharding_mesh, get_common_mesh_attrs


@jax.tree_util.register_pytree_node_class
class PowerSpectrumMultipoles(BinnedStatistic):

    _rename_fields = BinnedStatistic._rename_fields | {'_x': 'k', '_projs': 'ells', '_value': 'power_nonorm', '_weights': 'nmodes', '_norm': 'norm',
                                                       '_shotnoise_nonorm': 'shotnoise_nonorm', '_power_zero_nonorm': 'power_zero_nonorm'}
    _label_x = r'$k$ [$h/\mathrm{Mpc}$]'
    _label_proj = r'$\ell$'
    _label_value = r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]'
    _data_fields = BinnedStatistic._data_fields + ['_norm', '_shotnoise_nonorm', '_power_zero_nonorm']
    _select_proj_fields = BinnedStatistic._select_proj_fields + ['_power_zero_nonorm']
    _sum_fields = BinnedStatistic._sum_fields + ['_norm', '_shotnoise_nonorm', '_power_zero_nonorm']

    def __init__(self, k: np.ndarray, power_nonorm: jax.Array, nmodes: np.ndarray, edges: np.ndarray, ells: tuple, norm: jax.Array=1.,
                 shotnoise_nonorm: jax.Array=0., power_zero_nonorm: jax.Array=None, name: str=None, attrs: dict=None):

        def _tuple(item):
            if not isinstance(item, (tuple, list)):
                item = (item,) * len(ells)
            return tuple(item)

        self.__dict__.update(_norm=jnp.asarray(norm), _shotnoise_nonorm=jnp.asarray(shotnoise_nonorm), _power_zero_nonorm=tuple(jnp.asarray(p) for p in power_zero_nonorm))
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


@lru_cache(maxsize=32, typed=False)
def get_real_Ylm(ell, m, modules=None):
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
    Ylm : callable
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

    sp = None

    if modules is None:
        try: import sympy as sp
        except ImportError: pass

    elif 'sympy' in modules:
        import sympy as sp

    elif 'scipy' not in modules:
        raise ValueError('modules must be either ["sympy", "scipy", None]')

    def _safe_divide(num, denom):
        with np.errstate(divide='ignore', invalid='ignore'):
            return jnp.where(denom == 0., 0., num / denom)

    def get_Ylm_func(func, **attrs):

        def Ylm(*xvec):
            xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
            xhat = tuple(_safe_divide(xx, xnorm) for xx in xvec)
            return func(*xhat)

        for name, value in attrs.items():
            setattr(Ylm, name, value)
        return Ylm

    # sympy is not installed, fallback to scipy
    if sp is None:

        def _Ylm(xhat, yhat, zhat):
            # The cos(theta) dependence encoded by the associated Legendre polynomial
            toret = amp * (-1)**m * special.lpmv(abs(m), ell, zhat)
            # The phi dependence
            phi = np.arctan2(yhat, xhat)
            if m < 0:
                toret *= np.sin(abs(m) * phi)
            else:
                toret *= np.cos(m * phi)
            return toret

        def func(xhat, yhat, zhat):
            shape = jnp.broadcast_shapes(jnp.shape(xhat), jnp.shape(yhat), jnp.shape(zhat))
            dtype = jnp.result_type(xhat, yhat, zhat)
            out_type = jax.ShapeDtypeStruct(shape, dtype)
            return jax.pure_callback(_Ylm, out_type, xhat, yhat, zhat)

        Ylm = get_Ylm_func(func, ell=ell, m=m)

    else:
        # The relevant cartesian and spherical symbols
        # Using intermediate variable r helps sympy simplify expressions
        x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
        xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
        phi, theta = sp.symbols('phi theta', real=True)
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
        func = sp.lambdify((xhat, yhat, zhat), expr, modules='jax')

        Ylm = get_Ylm_func(func, ell=ell, m=m, expr=expr)

    return Ylm


[[get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in (0, 2, 4)]


def _get_los_vector(los: str | np.ndarray, ndim=3):
    vlos = None
    if isinstance(los, str):
        vlos = [0.] * ndim
        vlos['xyz'.index(los)] = 1.
    else:
        vlos = los
    return staticarray(vlos)


def _get_power_zero(mesh):
    toret = mesh[(0,) * mesh.ndim]
    if mesh.attrs.hermitian:
        return toret.real
    return toret


def compute_mesh_power(*meshs: RealMeshField | ComplexMeshField, bin: BinAttrs=None,
                       ells: int | tuple=0, los: str | np.ndarray='x') -> PowerSpectrumMultipoles:
    r"""
    Compute power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField
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
    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)

    ndim = len(mattrs.boxsize)
    vlos, swap = None, False
    if isinstance(los, str) and los in ['firstpoint', 'endpoint']:
        swap = los == 'endpoint'
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    attrs = dict(los=vlos if vlos is not None else los)

    if np.ndim(ells) == 0:
        ells = (ells,)
    ells = tuple(sorted(ells))

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    autocorr = len(meshs) == 1
    if vlos is None:  # local, varying line-of-sight
        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]
        if swap: meshs = meshs[::-1]

        rmesh1 = meshs[0]
        A0 = _2c(rmesh1 if autocorr else meshs[1])
        del meshs

        power, power_zero = [], []
        if 0 in ells:
            if autocorr:
                Aell = A0.clone(value=A0.real**2 + A0.imag**2)  # saves a bit of memory
            else:
                Aell = _2c(rmesh1) * A0.conj()
            power.append(bin(Aell, antisymmetric=False))
            power_zero.append(_get_power_zero(Aell))
            del Aell

        if nonzeroells:
            rmesh1 = _2r(rmesh1)
            # The real-space grid
            xvec = rmesh1.coords(sparse=True)
            # The Fourier-space grid
            kvec = A0.coords(sparse=True)
            Ylms = {ell: [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in nonzeroells}

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += _2c(rmesh1 * jax.lax.switch(im, Ylm, *xvec)).conj() * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            for ell in nonzeroells:
                Ylm = Ylms[ell]
                #jax.debug.inspect_array_sharding(jnp.zeros_like(A0.value), callback=print)
                Aell = jax.lax.scan(partial(f, Ylm), init=A0.clone(value=jnp.zeros_like(A0.value)), xs=np.arange(len(Ylm)))[0] * A0
                #Aell = sum(_2c(rmesh1 * Ylm(*xvec)) * Ylm(*kvec) for Ylm in Ylms[ell]).conj() * A0
                # Project on to 1d k-basis (averaging over mu=[-1, 1])
                power.append(4. * jnp.pi * bin(Aell, antisymmetric=bool(ell % 2)))
                power_zero.append(4. * jnp.pi * 0.)
                del Aell

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        power, power_zero = jnp.array(power), jnp.array(power_zero)
        if swap: power, power_zero = power.conj(), power_zero.conj()
        # Format the power results into :class:`PowerSpectrumMultipoles` instance
        return PowerSpectrumMultipoles(bin.xavg, power_nonorm=power, nmodes=bin.nmodes, edges=bin.edges, ells=ells, norm=norm,
                                       power_zero_nonorm=power_zero, attrs=attrs)

    else:  # fixed line-of-sight

        for imesh, mesh in enumerate(meshs):
            meshs[imesh] = _2c(mesh)
        if autocorr:
            Aell = meshs[0].clone(value=meshs[0].real**2 + meshs[0].imag**2)  # saves a bit of memory
        else:
            Aell = meshs[0] * meshs[1].conj()
        del meshs

        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]

        power, power_zero = [], []
        if 0 in ells:
            power.append(bin(Aell, antisymmetric=False))
            power_zero.append(_get_power_zero(Aell))

        if nonzeroells:
            kvec = Aell.coords(sparse=True)
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * kvec[0].ndim].set(1.)
            for ell in nonzeroells:  # TODO: jax.lax.scan
                power.append((2 * ell + 1) * bin(Aell * legendre(ell)(mu), antisymmetric=bool(ell % 2)))
                power_zero.append(0.)

        power, power_zero = jnp.array(power), jnp.array(power_zero)
        return PowerSpectrumMultipoles(bin.xavg, power_nonorm=power, nmodes=bin.nmodes, edges=bin.edges, ells=ells, norm=norm, power_zero_nonorm=power_zero, attrs=attrs)


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

    def split(self, nsplits=1, extent=None):
        from .mesh import _get_extent
        if extent is None:
            extent = _get_extent(self.data.positions, self.randoms.positions)
        for data, randoms in zip(self.data.split(nsplits=nsplits, extent=extent), self.randoms.split(nsplits=nsplits, extent=extent)):
            new = self.clone(data=data, randoms=randoms)
            yield new

    def paint(self, resampler: str | Callable='cic', interlacing: int=1,
              compensate: bool=False, dtype=None, out: str='real', **kwargs):
        fkp = self.data - self.data.sum() / self.randoms.sum() * self.randoms
        return fkp.paint(resampler=resampler, interlacing=interlacing, compensate=compensate, dtype=dtype, out=out, **kwargs)

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
    attrs = dict(kwargs)
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


def compute_fkp_normalization_and_shotnoise(*fkps, statistic='power', cellsize=10.):
    # TODO: generalize to N fkp fields
    fkps = FKPField.same_mesh(*fkps)
    if statistic in ['power', 'power_spectrum']:
        # This is the pypower normalization - move to new one?
        if len(fkps) > 1:
            shotnoise = 0.
            randoms = [fkps[0].data, fkps[1].randoms]
            alpha2 = fkps[1].data.sum() / fkps[1].randoms.sum()
            norm = alpha2 * compute_normalization(*randoms, cellsize=cellsize)
            randoms = [fkps[1].data, fkps[0].randoms]
            alpha2 = fkps[0].data.sum() / fkps[0].randoms.sum()
            norm += alpha2 * compute_normalization(*randoms, cellsize=cellsize)
            norm = norm / 2
        else:
            fkp = fkps[0]
            alpha = fkp.data.sum() / fkp.randoms.sum()
            shotnoise = jnp.sum(fkp.data.weights**2) + alpha**2 * jnp.sum(fkp.randoms.weights**2)
            randoms = [fkp.data, fkp.randoms]
            #mask = random.uniform(random.key(42), shape=fkp.randoms.size) < 0.5
            #randoms = [fkp.randoms[mask], fkp.randoms[~mask]]
            #randoms = [fkp.randoms[:fkp.randoms.size // 2], fkp.randoms[fkp.randoms.size // 2:]]
            norm = alpha * compute_normalization(*randoms, cellsize=cellsize)
    else:
        raise NotImplementedError
    return norm, shotnoise


def compute_fkp_power(*fkps: FKPField, bin: BinAttrs=None,
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
    norm, shotnoise_nonorm = compute_fkp_normalization_and_shotnoise(*fkps, statistic='power')
    return compute_mesh_power(*[fkp.paint(resampler=resampler, interlacing=interlacing, compensate=True, out='complex') for fkp in fkps], bin=bin, ells=ells, los=los, mode_oversampling=mode_oversampling).clone(norm=norm, shotnoise_nonorm=shotnoise_nonorm)


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


def compute_mesh_window(*meshs: RealMeshField | ComplexMeshField | MeshAttrs, edgesin: np.ndarray, ellsin: tuple=None,
                        bin: BinAttrs=None, ells: int | tuple=0, los: str | np.ndarray='x',
                        buffer=None, batch_size=None, pbar=False, norm=None, flags=tuple()) -> WindowMatrix:
    r"""
    Compute mean power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, MeshAttrs
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
    tophat_method = 'rectangle'

    from .utils import TophatPowerToCorrelation, TophatCorrelationToPower

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

    _norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)
    if norm is None: norm = _norm
    rnorm = _norm / norm / mattrs.meshsize.prod(dtype=rdtype)

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
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    autocorr = len(meshs) == 1
    if edgesin.ndim == 2:
        kin = edgesin
        edgesin = None
    else:
        kin = jnp.array([edgesin[:-1], edgesin[1:]]).T

    def np_map(f, xs):
        return jnp.array(list(map(f, xs)))

    svec = mattrs.xcoords(kind='separation', sparse=True)

    def oversample(edges, factor=5):
        return jnp.interp(jnp.linspace(0., 1., (edges.size - 1) * factor + 1), jnp.linspace(0., 1., edges.size), edges)

    if pbar:
        from tqdm import tqdm
        t = tqdm(total=len(kin), bar_format='{l_bar}{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        def round(n):
            return int(n * 1e6) / 1e6

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

        kvec = mattrs.kcoords(sparse=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)

        def _bin(Aell):
            power = []
            for ell in ells:
                leg = legendre(ell)(mu)
                odd = ell % 2
                if odd: leg += legendre(ell)(-mu)
                power.append((2 * ell + 1) / (1 + odd) * bin(Aell * leg, remove_zero=ell == 0))
            return jnp.concatenate(power)

        def my_map(f, xs):
            if pbar:
                return np_map(f, xs)
            return jax.lax.map(f, xs=xs, batch_size=batch_size)

        wmat = []

        if 'smooth' in flags:

            from .utils import spherical_jn
            spherical_jn = {ell: spherical_jn(ell) for ell in set(ells)}
            sedges = None
            #sedges = np.arange(0., rmesh1.boxsize.max() / 2., rmesh1.cellsize.min() / 4.)
            sbin = BinAttrs(mattrs, edges=sedges, kind='real')
            koversampling = 5
            kbin = BinAttrs(mattrs, edges=oversample(bin.edges, koversampling), kind='complex')
            kout, koutnmodes = kbin.xavg.reshape(-1, koversampling), kbin.nmodes.reshape(-1, koversampling)
            koutnmodes /= koutnmodes.sum(axis=-1)[..., None]
            kout = jnp.where(koutnmodes == 0, 0., kout)
            del kbin

            for ellin in ellsin:

                wmat_tmp = []
                for ill, ell in enumerate(ells):
                    snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                    smu = sum(xx * ll for xx, ll in zip(svec, vlos)) / jnp.where(snorm == 0., 1., snorm)

                    Qs = sbin((Q if Q is not None else 1.) * legendre(ell)(smu) * legendre(ellin)(smu))
                    if ell != 0: Qs = Qs.at[0].set(0.)
                    Qs = jnp.where(sbin.nmodes == 0, 0., Qs)
                    savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                    snmodes = sbin.nmodes

                    del snorm

                    def f(kin):
                        tophat_Qs = TophatPowerToCorrelation(kin, seval=savg, ell=ellin, edges=True, method=tophat_method).w[..., 0] * rnorm * mattrs.cellsize.prod() * Qs

                        def f2(args):
                            kout, nout = args
                            return (-1)**(ell // 2) * jnp.sum(nout[..., None] * snmodes * spherical_jn[ell](kout[..., None] * savg) * tophat_Qs)

                        batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / (koversampling * savg.size), 1), kout.shape[0]))
                        power = jax.lax.map(f2, (kout, koutnmodes), batch_size=batch_size)

                        if pbar:
                            t.update(n=round(1 / len(ells) / len(ellsin)))
                        return (2 * ell + 1) * power

                    wmat_tmp.append(my_map(f, kin))
                wmat.append(jnp.concatenate(wmat_tmp, axis=-1))

        elif 'infinite' in flags:
            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
            smu = sum(xx * ll for xx, ll in zip(svec, vlos)) / jnp.where(snorm == 0., 1., snorm)

            for ellin in ellsin:
                legin = legendre(ellin)(smu)

                def f(kin):
                    Aell = TophatPowerToCorrelation(kin, seval=snorm, ell=ellin, edges=True, method=tophat_method).w[..., 0] * legin * rnorm * mattrs.cellsize.prod()
                    Aell = mattrs.create(kind='real', fill=Aell)
                    if Q is not None: Aell *= Q
                    power = _bin(_2c(Aell))
                    if pbar:
                        t.update(n=round(1 / len(ellsin)))
                    return power

                wmat.append(my_map(f, kin))

        else:

            for ellin in ellsin:
                legin = legendre(ellin)(mu)

                def f(kin):
                    Aell = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * legin)
                    if Q is not None: Aell = _2c(Q * _2r(Aell))
                    else: Aell *= mattrs.meshsize.prod(dtype=rdtype)
                    power = _bin(Aell)
                    if pbar:
                        t.update(n=round(1 / len(ellsin)))
                    return power

                wmat.append(my_map(f, kin))

        wmat = jnp.concatenate(wmat, axis=0).T

    else:
        theory_los = 'firstpoint'
        if len(ellsin) == 2 and isinstance(ellsin[1], str):
            ellsin, theory_los = ellsin
        if np.ndim(ellsin) == 0: ellsin = (ellsin,)
        ellsin = list(ellsin)

        # In this case, theory must be a dictionary of (multipole, wide_angle_order)
        if swap: meshs = meshs[::-1]

        if periodic:
            meshs = [mattrs.create(kind='real', fill=1.)]

        rmesh1 = _2r(meshs[0])
        rmesh2 = rmesh1 if autocorr else _2r(meshs[1])
        A0 = _2c(meshs[0] if autocorr else meshs[1])
        del meshs

        # The real-space grid
        xvec = mattrs.xcoords(sparse=True)

        # The Fourier-space grid
        kvec = A0.coords(sparse=True)

        Ylms = {ell: [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}
        from .utils import spherical_jn
        spherical_jn = {ell: spherical_jn(ell) for ell in set(ells) | set(ellsin)}
        real_gaunt = {((0, 0), (0, 0), (0, 0)): 0.28209479177387814, ((0, 0), (2, -2), (2, -2)): 0.28209479177387814, ((0, 0), (2, -1), (2, -1)): 0.28209479177387814, ((0, 0), (2, 0), (2, 0)): 0.28209479177387814,
                      ((0, 0), (2, 1), (2, 1)): 0.28209479177387814, ((0, 0), (2, 2), (2, 2)): 0.28209479177387814, ((2, -2), (0, 0), (2, -2)): 0.28209479177387814, ((2, -1), (0, 0), (2, -1)): 0.28209479177387814,
                      ((2, 0), (0, 0), (2, 0)): 0.28209479177387814, ((2, 1), (0, 0), (2, 1)): 0.28209479177387814, ((2, 2), (0, 0), (2, 2)): 0.28209479177387814, ((2, -2), (2, -2), (0, 0)): 0.28209479177387814,
                      ((2, -1), (2, -1), (0, 0)): 0.28209479177387814, ((2, 0), (2, 0), (0, 0)): 0.28209479177387814, ((2, 1), (2, 1), (0, 0)): 0.28209479177387814, ((2, 2), (2, 2), (0, 0)): 0.28209479177387814,
                      ((2, -2), (2, -2), (2, 0)): -0.1802237515728686, ((2, -2), (2, -1), (2, 1)): 0.15607834722743974, ((2, -2), (2, 0), (2, -2)): -0.1802237515728686, ((2, -2), (2, 1), (2, -1)): 0.15607834722743974,
                      ((2, -1), (2, -2), (2, 1)): 0.15607834722743974, ((2, -1), (2, -1), (2, 0)): 0.0901118757864343, ((2, -1), (2, -1), (2, 2)): -0.15607834722743985, ((2, -1), (2, 0), (2, -1)): 0.0901118757864343,
                      ((2, -1), (2, 1), (2, -2)): 0.15607834722743974, ((2, -1), (2, 2), (2, -1)): -0.15607834722743985, ((2, 0), (2, -2), (2, -2)): -0.1802237515728686, ((2, 0), (2, -1), (2, -1)): 0.0901118757864343,
                      ((2, 0), (2, 0), (2, 0)): 0.18022375157286857, ((2, 0), (2, 1), (2, 1)): 0.09011187578643429, ((2, 0), (2, 2), (2, 2)): -0.18022375157286857, ((2, 1), (2, -2), (2, -1)): 0.15607834722743974,
                      ((2, 1), (2, -1), (2, -2)): 0.15607834722743974, ((2, 1), (2, 0), (2, 1)): 0.09011187578643429, ((2, 1), (2, 1), (2, 0)): 0.09011187578643429, ((2, 1), (2, 1), (2, 2)): 0.15607834722743988,
                      ((2, 1), (2, 2), (2, 1)): 0.15607834722743988, ((2, 2), (2, -1), (2, -1)): -0.15607834722743985, ((2, 2), (2, 0), (2, 2)): -0.18022375157286857, ((2, 2), (2, 1), (2, 1)): 0.15607834722743988,
                      ((2, 2), (2, 2), (2, 0)): -0.18022375157286857, ((4, -4), (2, -2), (2, 2)): 0.23841361350444812, ((4, -4), (2, 2), (2, -2)): 0.23841361350444812, ((4, -3), (2, -2), (2, 1)): 0.16858388283618375,
                      ((4, -3), (2, -1), (2, 2)): 0.1685838828361839, ((4, -3), (2, 1), (2, -2)): 0.16858388283618375, ((4, -3), (2, 2), (2, -1)): 0.1685838828361839, ((4, -2), (2, -2), (2, 0)): 0.15607834722744057,
                      ((4, -2), (2, -1), (2, 1)): 0.18022375157286857, ((4, -2), (2, 0), (2, -2)): 0.15607834722744057, ((4, -2), (2, 1), (2, -1)): 0.18022375157286857, ((4, -1), (2, -2), (2, 1)): -0.06371871843402716,
                      ((4, -1), (2, -1), (2, 0)): 0.2207281154418226, ((4, -1), (2, -1), (2, 2)): 0.06371871843402753, ((4, -1), (2, 0), (2, -1)): 0.2207281154418226, ((4, -1), (2, 1), (2, -2)): -0.06371871843402717,
                      ((4, -1), (2, 2), (2, -1)): 0.06371871843402753, ((4, 0), (2, -2), (2, -2)): 0.04029925596769687, ((4, 0), (2, -1), (2, -1)): -0.1611970238707875, ((4, 0), (2, 0), (2, 0)): 0.24179553580618127,
                      ((4, 0), (2, 1), (2, 1)): -0.16119702387078752, ((4, 0), (2, 2), (2, 2)): 0.04029925596769688, ((4, 1), (2, -2), (2, -1)): -0.06371871843402717, ((4, 1), (2, -1), (2, -2)): -0.06371871843402717,
                      ((4, 1), (2, 0), (2, 1)): 0.2207281154418226, ((4, 1), (2, 1), (2, 0)): 0.2207281154418226, ((4, 1), (2, 1), (2, 2)): -0.06371871843402754, ((4, 1), (2, 2), (2, 1)): -0.06371871843402754,
                      ((4, 2), (2, -1), (2, -1)): -0.18022375157286857, ((4, 2), (2, 0), (2, 2)): 0.15607834722743988, ((4, 2), (2, 1), (2, 1)): 0.18022375157286857, ((4, 2), (2, 2), (2, 0)): 0.15607834722743988,
                      ((4, 3), (2, -2), (2, -1)): -0.16858388283618375, ((4, 3), (2, -1), (2, -2)): -0.16858388283618375, ((4, 3), (2, 1), (2, 2)): 0.16858388283618386, ((4, 3), (2, 2), (2, 1)): 0.16858388283618386,
                      ((4, 4), (2, -2), (2, -2)): -0.23841361350444804, ((4, 4), (2, 2), (2, 2)): 0.23841361350444806}

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
                sbin = BinAttrs(rmesh1, edges=sedges)
                koversampling = 5
                kbin = BinAttrs(A0, edges=oversample(bin.edges, koversampling))
                kout, koutnmodes = kbin.xavg.reshape(-1, koversampling), kbin.nmodes.reshape(-1, koversampling)
                koutnmodes /= koutnmodes.sum(axis=-1)[..., None]
                kout = jnp.where(koutnmodes == 0, 0., kout)
                del kbin

                for ell1, wa1 in ellsin:
                    wmat_tmp[ell1, wa1] = 0
                    for ill, ell in enumerate(ells):
                        Qs = 0.
                        xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                        snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                        for im1, Yl1m1 in enumerate(Ylms[ell1]):
                            for im, Ylm in enumerate(Ylms[ell]):
                                Q = (4. * np.pi) / (2 * ell1 + 1) * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xvec) * Yl1m1(*xvec)).conj() * A0) * snorm**wa1
                                Qs += 4. * np.pi * sbin(Q * Ylm(*svec) * Yl1m1(*svec)) * rnorm * mattrs.cellsize.prod()

                        if ell != 0: Qs = Qs.at[0].set(0.)
                        Qs = jnp.where(sbin.nmodes == 0, 0., Qs)
                        savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                        snmodes = sbin.nmodes

                        del xnorm, snorm

                        def f(kin):
                            tophat_Qs = TophatPowerToCorrelation(kin, seval=savg, ell=ell1, edges=True, method=tophat_method).w[..., 0] * Qs

                            def f2(args):
                                kout, nout = args
                                return (-1)**(ell // 2) * jnp.sum(nout[..., None] * snmodes * spherical_jn[ell](kout[..., None] * savg) * tophat_Qs)

                            batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / (koversampling * savg.size), 1), kout.shape[0]))
                            power = jax.lax.map(f2, (kout, koutnmodes), batch_size=batch_size)

                            if pbar:
                                t.update(n=round(1 / len(ells) / len(ellsin)))
                            power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                            return power.ravel()

                        wmat_tmp[ell1, wa1] += my_map(f, kin)

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
                                Qs += Yl1m1(*svec) * Q * rnorm * mattrs.cellsize.prod()
                            del xnorm

                            def f(kin):
                                tophat_Qs = TophatPowerToCorrelation(kin, seval=snorm, ell=ell1, edges=True, method=tophat_method).w[..., 0] * Qs
                                power = 4 * jnp.pi * bin(Ylm(*kvec) * _2c(tophat_Qs), antisymmetric=bool(ell % 2), remove_zero=ell == 0)
                                if pbar:
                                    t.update(n=round(1. / sum(len(Ylms[ell]) for ell in ells) / len(ellsin)))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()
                            wmat_tmp[ell1, wa1] += my_map(f, kin)
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
                                def f(kin):
                                    Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                                    knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                                    xi = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * rnorm * Yl1m1(*kvec)).c2r()
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im1])

                                    Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                    power = 4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0)
                                    del Aell[ell]
                                    if pbar:
                                        t.update(n=round((im1 + 1) / sum(len(Ylms[ell]) for ell in ells)) / sum(len(Ylms[ell]) for ell, _ in ellsin))
                                    power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                    return power.ravel()

                                wmat_tmp[ell1, wa1] += my_map(f, kin)

                    if 'recompute' not in flags:
                        def f(kin):
                            Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                            knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                            for im1, Yl1m1 in enumerate(Ylms[ell1]):
                                def kernel(*args):
                                    kmask = (knorm >= kin[0]) & (knorm <= kin[-1])
                                    return kmask * rnorm * Yl1m1(*kvec)

                                xi = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * rnorm * Yl1m1(*kvec)).c2r()
                                for ell in Aell:
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im1])

                            power = []
                            for ill, ell in enumerate(ells):
                                Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                power.append(4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0))
                                del Aell[ell]
                            if pbar:
                                t.update(n=round((im1 + 1) / sum(len(Ylms[ell]) for ell, _ in ellsin)))
                            return jnp.concatenate(power)

                        wmat_tmp[ell1, wa1] = my_map(f, kin)

            wmat = jnp.concatenate(list(wmat_tmp.values()), axis=0).T

        elif theory_los == 'local':

            wmat_tmp = {}
            if 'smooth' in flags:
                sedges = None
                #sedges = np.arange(0., rmesh1.boxsize.max() / 2., rmesh1.cellsize.min() / 4.)
                sbin = BinAttrs(rmesh1, edges=sedges)
                koversampling = 5
                kbin = BinAttrs(A0, edges=oversample(bin.edges, koversampling))
                kout, koutnmodes = kbin.xavg.reshape(-1, koversampling), kbin.nmodes.reshape(-1, koversampling)
                koutnmodes /= koutnmodes.sum(axis=-1)[..., None]
                kout = jnp.where(koutnmodes == 0, 0., kout)
                del kbin

                for ell1, ell2 in itertools.product((2, 0), (2, 0)):
                    wmat_tmp[ell1, ell2] = 0
                    for ill, ell in enumerate(ells):
                        ps = [p for p in (0, 2, 4) if real_gaunt.get(((p, 0), (ell1, 0), (ell2, 0))) is not None]
                        #ps = [p for p in (0,) if real_gaunt.get(((p, 0), (ell1, 0), (ell2, 0))) is not None]
                        Qs = {p: 0. for p in ps}
                        for (im1, Yl1m1), (im2, Yl2m2) in itertools.product(enumerate(Ylms[ell1]), enumerate(Ylms[ell2])):
                            for im, Ylm in enumerate(Ylms[ell]):
                                Q = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                                for p in ps:
                                    key = p
                                    tmp = 0.
                                    for imp, Ypmp in enumerate(Ylms[p]):
                                        rg = real_gaunt.get(((p, imp - p), (ell1, im1 - ell1), (ell2, im2 - ell2)), None)
                                        if rg is not None:  # rg != 0
                                            tmp += rg * Ylm(*svec) * Ypmp(*svec) * Q
                                    if hasattr(tmp, 'shape'):
                                        tmp = 4. * np.pi * sbin(tmp) * rnorm * mattrs.cellsize.prod()
                                        Qs[key] += dump_to_buffer(tmp, key)

                        for p in ps:
                            Q = load_from_buffer(Qs[p])
                            if ell != 0: Q = Q.at[0].set(0.)
                            Q = jnp.where(sbin.nmodes == 0, 0., Q)
                            savg = jnp.where(sbin.nmodes == 0, 0., sbin.xavg)
                            snmodes = sbin.nmodes

                            def f(kin):
                                tophat_Q = TophatPowerToCorrelation(kin, seval=savg, ell=p, edges=True, method=tophat_method).w[..., 0] * Q

                                def f2(args):
                                    kout, nout = args
                                    return (-1)**(ell // 2) * jnp.sum(nout[..., None] * snmodes * spherical_jn[ell](kout[..., None] * savg) * tophat_Q)

                                batch_size = int(min(max(mattrs.meshsize.prod(dtype=float) / (koversampling * savg.size), 1), kout.shape[0]))
                                power = jax.lax.map(f2, (kout, koutnmodes), batch_size=batch_size)
                                if pbar:
                                    t.update(n=round(1 / len(ells) / 6))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()

                            wmat_tmp[ell1, ell2] += my_map(f, kin)

            elif 'infinite' in flags:
                for ell1, ell2 in itertools.product((2, 0), (2, 0)):
                    wmat_tmp[ell1, ell2] = 0
                    for ill, ell in enumerate(ells):
                        ps = [p for p in (0, 2, 4) if real_gaunt.get(((p, 0), (ell1, 0), (ell2, 0))) is not None]
                        for im, Ylm in enumerate(Ylms[ell]):
                            Qs = {p: 0. for p in ps}
                            snorm = jnp.sqrt(sum(xx**2 for xx in svec))
                            for (im1, Yl1m1), (im2, Yl2m2) in itertools.product(enumerate(Ylms[ell1]), enumerate(Ylms[ell2])):
                                Q = (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * _2r(_2c(rmesh1 * Ylm(*xvec) * Yl1m1(*xvec)).conj() * _2c(rmesh2 * Yl2m2(*xvec)))
                                for p in ps:
                                    for imp, Ypmp in enumerate(Ylms[p]):
                                        rg = real_gaunt.get(((p, imp - p), (ell1, im1 - ell1), (ell2, im2 - ell2)), None)
                                        if rg is not None:  # rg != 0
                                            Qs[p] += rg * Ypmp(*svec) * Q
                            for p in Qs:
                                Qs[p] = dump_to_buffer(Qs[p] * rnorm * mattrs.cellsize.prod(), p)

                            def f(kin):
                                xi = 0.
                                for p in ps:
                                    tophat = TophatPowerToCorrelation(kin, seval=snorm, ell=p, edges=True, method=tophat_method).w[..., 0]
                                    Q = load_from_buffer(Qs[p])
                                    xi += tophat * Q
                                power = 4 * jnp.pi * bin(Ylm(*kvec) * _2c(xi), antisymmetric=bool(ell % 2), remove_zero=ell == 0)
                                if pbar:
                                    t.update(n=round(1 / sum(len(Ylms[ell]) for ell in ells) / 4))
                                power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                return power.ravel()
                            wmat_tmp[ell1, ell2] += my_map(f, kin)

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
                                def f(kin):
                                    knorm = jnp.sqrt(sum(xx**2 for xx in kvec))
                                    Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                                    xi = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * rnorm * Yl1m1(*kvec) * Yl2m2(*[-kk for kk in kvec])).c2r()
                                    for im in Aell[ell]:
                                        Aell[ell][im] += xi * load_from_buffer(Qs[ell, im, im12])
                                    Aell[ell] = sum(Aell[ell][im].r2c() * Ylms[ell][im](*kvec) for im in Aell[ell])
                                    power = 4 * jnp.pi * bin(Aell[ell], antisymmetric=bool(ell % 2), remove_zero=ell == 0)
                                    if pbar:
                                        t.update(n=round((im + 1) / sum(len(Ylms[ell]) for ell in ells) / 36))
                                    power = jnp.zeros_like(power, shape=(len(ells), power.size)).at[ill].set(power)
                                    return power.ravel()
                                wmat_tmp[ell1, ell2] += my_map(f, kin)

                    if 'recompute' not in flags:
                        knorm = jnp.sqrt(sum(xx**2 for xx in kvec))

                        def f(kin):
                            Aell = {ell: {im: 0. for im, Ylm in enumerate(Ylms[ell])} for ell in ells}
                            for im12, (Yl1m1, Yl2m2) in enumerate(itertools.product(Ylms[ell1], Ylms[ell2])):
                                xi = mattrs.create(kind='complex', fill=((knorm >= kin[0]) & (knorm <= kin[-1])) * rnorm * Yl1m1(*kvec) * Yl2m2(*[-kk for kk in kvec])).c2r()
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

    observable = BinnedStatistic(x=[bin.xavg] * len(ells), value=[jnp.zeros_like(bin.xavg)] * len(ells), edges=[bin.edges] * len(ells), projs=ells)
    xin = np.mean(kin, axis=-1)
    theory = BinnedStatistic(x=[xin] * len(ellsin), value=[jnp.zeros_like(xin)] * len(ellsin), edges=[edgesin] * len(ellsin) if edgesin is not None else None, projs=ellsin)
    wmat = WindowMatrix(observable, theory, wmat, attrs={'norm': norm})
    return wmat



def compute_mean_mesh_power(*meshs: RealMeshField | ComplexMeshField | MeshAttrs, theory: Callable | dict[Callable],
                            bin: BinAttrs=None, ells: int | tuple=0, los: str | np.ndarray='x') -> PowerSpectrumMultipoles:
    r"""
    Compute mean power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, ComplexMeshField, MeshAttrs
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
    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)
    rnorm = norm / mattrs.meshsize.prod(dtype=rdtype)

    ndim = len(mattrs.boxsize)
    vlos, swap = None, False
    if isinstance(los, str) and los in ['firstpoint', 'endpoint']:
        swap = los == 'endpoint'
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    attrs = dict(los=vlos if vlos is not None else los)

    if np.ndim(ells) == 0:
        ells = (ells,)
    ells = tuple(sorted(ells))

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    autocorr = len(meshs) == 1

    poles = theory
    kin = None
    theory_los = 'firstpoint'
    if isinstance(poles, tuple) and isinstance(poles[-1], str):
        theory_los = poles[-1]
        if len(poles) == 2: poles = poles[0]
        else: poles = poles[:-1]
    if isinstance(poles, tuple):
        kin, poles = poles
    if isinstance(poles, BinnedStatistic):
        kin, poles = poles._edges[0] if poles._edges[0] is not None else poles._x[0], {proj: poles.view(projs=proj) for proj in poles.projs}
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

            for imesh, mesh in enumerate(meshs):
                meshs[imesh] = _2c(mesh)
            if autocorr:
                meshs.append(meshs[0])

            Q = _2r(meshs[0] * meshs[1].conj())
        else:
            Q = None

        mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.where(knorm == 0., 1., knorm)

        pkvec = theory

        if is_poles:
            def pkvec(*args):
                return sum(get_theory(ell) * legendre(ell)(mu) for ell in poles)

        Aell = mattrs.create(kind='complex').apply(lambda value, kvec: pkvec(kvec) * rnorm, kind='wavenumber')
        if Q is not None: Aell = _2c(Q * _2r(Aell))
        else: Aell *= mattrs.meshsize.prod(dtype=rdtype)

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

        # The real-space grid
        xhat = mattrs.xcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in xhat))

        shat = mattrs.xcoords(kind='separation', sparse=True)
        snorm = jnp.sqrt(sum(xx**2 for xx in shat))

        # The Fourier-space grid
        khat = A0.coords(sparse=True)

        Ylms = {ell: [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}

        power, power_zero = [], []
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
            power.append(4. * jnp.pi * bin(Aell, antisymmetric=bool(ell % 2)))
            power_zero.append(0.)
            if ell == 0:
                power_zero[-1] += 4. * jnp.pi * _get_power_zero(Aell)

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        power, power_zero = jnp.array(power), jnp.array(power_zero)
        if swap: power, power_zero = power.conj(), power_zero.conj()
        # Format the power results into :class:`PowerSpectrumMultipoles` instance
        return PowerSpectrumMultipoles(bin.xavg, power_nonorm=power, nmodes=bin.nmodes, edges=bin.edges, ells=ells, norm=norm,
                                       power_zero_nonorm=power_zero, attrs=attrs)
