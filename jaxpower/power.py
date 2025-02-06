import os
from functools import partial, lru_cache
from dataclasses import dataclass, asdict, field
from collections.abc import Callable
import itertools

import numpy as np
import jax
from jax import numpy as jnp
from scipy import special

from . import utils
from .utils import legendre, plotter, BinnedStatistic
from .mesh import RealMeshField, ComplexMeshField, HermitianComplexMeshField, ParticleField, staticarray, MeshAttrs, get_common_mesh_attrs


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
        super().__init__(x=_tuple(k), edges=_tuple(edges), projs=ells, value=tuple(power_nonorm),
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
def get_real_Ylm(ell, m):
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
        Ylm.l = ell
        Ylm.m = m
        return Ylm

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
    Ylm.l = ell
    Ylm.m = m
    return Ylm


[[get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in (0, 2, 4)]


def project_to_basis(mesh: RealMeshField | ComplexMeshField | HermitianComplexMeshField, xedges: np.ndarray,
                     los: str | np.ndarray='x', muedges: None | np.ndarray=None, ells: None | tuple | list=None,
                     antisymmetric: bool=False, exclude_zero: bool=False, mode_oversampling: int=0):
    r"""
    Project a 3D statistic on to the specified basis. The basis will be one of:

        - 2D :math:`(x, \mu)` bins: :math:`\mu` is the cosine of the angle to the line-of-sight
        - 2D :math:`(x, \ell)` bins: :math:`\ell` is the multipole number, which specifies
          the Legendre polynomial when weighting different :math:`\mu` bins.

    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/fftpower.py.

    Notes
    -----
    In single precision (float32/complex64) nbodykit's implementation is fairly imprecise
    due to incorrect binning of :math:`x` and :math:`\mu` modes.
    Here we cast mesh coordinates to the maximum precision of input ``edges``,
    which makes computation much more accurate in single precision.

    Notes
    -----
    We deliberately set to 0 the number of modes beyond Nyquist, as it is unclear whether to count Nyquist as :math:`\mu` or :math:`-\mu`
    (it should probably be half weight for both).
    Our safe choice ensures consistent results between hermitian compressed and their associated uncompressed fields.

    Notes
    -----
    The 2D :math:`(x, \ell)` bins will be computed only if ``ells`` is specified.
    See return types for further details.
    For both :math:`x` and :math:`\mu`, binning is inclusive on the low end and exclusive on the high end,
    i.e. mode `mode` falls in bin `i` if ``edges[i] <= mode < edges[i+1]``.
    However, last :math:`\mu`-bin is inclusive on both ends: ``edges[-2] <= mu <= edges[-1]``.
    Therefore, with e.g. :math:`\mu`-edges ``[0.2, 0.4, 1.0]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 1.0`.
    Similarly, with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, the last :math:`\mu`-bin includes modes at :math:`\mu = 0.8`.

    Warning
    -------
    Integration over Legendre polynomials for multipoles is performed between the first and last :math:`\mu`-edges,
    e.g. with :math:`\mu`-edges ``[0.2, 0.4, 0.8]``, integration is performed between :math:`\mu = 0.2` and :math:`\mu = 0.8`.

    Parameters
    ----------
    mesh : RealMeshField or ComplexMeshField
        The 3D array holding the statistic to be projected to the specified basis.

    xedges : array
        List of arrays specifying the edges of the desired :math:`x` bins; assumed sorted.

    muedges : array, default=None
        List of arrays specifying the edges of the desired :math:`\mu` bins; assumed sorted.

    los : array_like, default=(0, 0, 1)
        The line-of-sight direction to use, which :math:`\mu` is defined with respect to.

    ells : tuple of ints, default=None
        If provided, a list of integers specifying multipole numbers to project the 2D :math:`(x, \mu)` bins on to.

    antisymmetric : bool, default=False
        If mesh is compressed, whether mesh is hermitian-antisymmetric (in which case negative modes are added with weight -1).

    exclude_zero : bool, default=0
        Whether to exclude zero mode from the sum.

    mode_oversampling : int, default=0
        If > 0, artificially increase the resolution of the input mesh by a factor ``2 * mode_oversampling + 1``.
        In practice, shift the coordinates of the coordinates of the input grid by ``np.arange(-mode_oversampling, mode_oversampling + 1)``
        along each of x, y, z axes.
        This reduces "discrete grid binning effects".

    Returns
    -------
    - xmean2d : array_like, (nx, nmu)
        The mean :math:`x` value in each 2D bin
    - mumean2d : array_like, (nx, nmu)
        The mean :math:`\mu` value in each 2D bin
    - y2d : array_like, (nell, nx, nmu)
        The mean ``mesh`` value in each 2D bin, for each multipole
    - n2d : array_like, (nx, nmu)
        The number of values averaged in each 2D bin
    - zero2d : array_like, (nmu, )
        Value of the power spectrum at k = 0
    """
    hermitian_symmetric = isinstance(mesh, HermitianComplexMeshField)
    if antisymmetric: hermitian_symmetric *= -1

    shape = mesh.shape
    ndim = mesh.ndim
    dtype = mesh.dtype
    xedges = np.asarray(xedges)
    #xdtype = xedges.dtype
    nx = len(xedges) - 1

    # Setup the bin edges and number of bins
    nmu = 1
    if muedges is not None:
        muedges = np.asarray(muedges)
        #xdtype = max(xdtype, muedges.dtype)
        nmu = len(muedges) - 1
    xdtype = mesh.real.dtype
    los = _get_los_vector(los, ndim=ndim)

    # Always make sure first ell value is monopole, which is just (x, mu) projection since legendre of ell = 0 is 1
    unique_ells = sorted(set([0]) | set(ells or []))
    legpoly = {ell: legendre(ell) for ell in unique_ells}
    nell = len(unique_ells)

    # valid ell values
    if any(ell < 0 for ell in unique_ells):
        raise ValueError('Multipole numbers must be non-negative integers')

    # Initialize the binning arrays
    xshape = (nx + 3, nmu + 3)
    xsize = xshape[0] * xshape[1]
    musum = jnp.zeros(xsize, dtype=xdtype)
    xsum = jnp.zeros(xsize, dtype=xdtype)
    ysum = jnp.zeros((nell, xsize), dtype=dtype)  # extra dimension for multipoles
    nsum = jnp.zeros(xsize, dtype=xdtype if mode_oversampling else int)
    # If input array is Hermitian symmetric, only half of the last axis is stored in `mesh`

    spacing = mesh.spacing
    meshsize = mesh.meshsize
    coords = mesh.coords(sparse=False)
    mesh = mesh.ravel()
    mincell = np.min(spacing)

    if hermitian_symmetric:
        nonsingular = jnp.ones(shape, dtype='i4')
        # Get the indices that have positive freq along symmetry axis = -1
        nonsingular = nonsingular.at[...].set(coords[-1] > 0.)
        nonsingular = nonsingular.ravel()

    import itertools
    shifts = [np.arange(-mode_oversampling, mode_oversampling + 1)] * len(spacing)
    shifts = list(itertools.product(*shifts))

    for shift in shifts:
        shift = spacing * shift / (2 * mode_oversampling + 1)
        xvec = coords
        mask_zero_noshift = jnp.ravel(sum(xx**2 for xx in xvec) < (mincell / 2.)**2)  # mask_zero defined *before* shift
        xvec = tuple(xx + ss for xx, ss in zip(xvec, shift))
        xnorm = jnp.ravel(sum(xx**2 for xx in xvec)**0.5)

        # Get the bin indices for x on the slab
        dig_x = jnp.digitize(xnorm, xedges, right=False)
        dig_x = jnp.where(mask_zero_noshift, nx + 2, dig_x)

        # Get the bin indices for mu on the slab
        mu = None
        if muedges is not None or unique_ells:
            mu = jnp.ravel(sum(xx * ll for xx, ll in zip(xvec, los)))
            with np.errstate(divide='ignore', invalid='ignore'):
                mu = jnp.where(xnorm >= (mincell / (2 * mode_oversampling + 1) / 2.), mu / xnorm, 0.)

        if hermitian_symmetric == 0:
            mus = [mu]
        else:
            mus = [mu, -mu]

        # Accounting for negative frequencies
        for imu, mu in enumerate(mus):
            if muedges is not None:
                # Make the multi-index
                dig_mu = jnp.digitize(mu, muedges, right=False)  # this is bins[i-1] <= x < bins[i]
                dig_mu = jnp.where(mu == muedges[-1], nmu, dig_mu) # last mu inclusive
                dig_mu = jnp.where(mask_zero_noshift, nmu + 2, dig_mu)
            else:
                dig_mu = jnp.where(mask_zero_noshift, nmu + 2, nmu)

            multi_index = jnp.ravel_multi_index([dig_x, dig_mu], (nx + 3, nmu + 3), mode='clip')

            mode_weight = None
            if hermitian_symmetric and imu:
                mode_weight = nonsingular

            # Count number of modes in each bin
            nsum = nsum.at[...].add(jnp.bincount(multi_index, weights=mode_weight, length=nsum.size))
            # Sum up x in each bin
            xsum = xsum.at[...].add(jnp.bincount(multi_index, weights=xnorm * mode_weight if mode_weight is not None else xnorm, length=nsum.size))
            # Sum up mu in each bin
            if muedges is not None: musum = musum.at[...].add(jnp.bincount(multi_index, weights=mu * mode_weight if mode_weight is not None else xnorm, length=nsum.size))

            # Compute multipoles by weighting by Legendre(ell, mu)
            for ill, ell in enumerate(unique_ells):

                weightedmesh = 1. if ell == 0 else (2. * ell + 1.) * legpoly[ell](mu)

                if mode_weight is not None:
                    # Weight the input 3D array by the appropriate Legendre polynomial
                    weightedmesh = hermitian_symmetric * weightedmesh * mesh.conj() * mode_weight  # hermitian_symmetric is 1 or -1
                else:
                    weightedmesh = weightedmesh * mesh

                # Sum up the weighted y in each bin
                tmp = jnp.bincount(multi_index, weights=weightedmesh.real, length=nsum.size)
                if jnp.iscomplexobj(ysum):
                    tmp += jnp.bincount(multi_index, weights=weightedmesh.imag, length=nsum.size)
                ysum = ysum.at[ill, ...].add(tmp)

    # It is not clear how to proceed with beyond Nyquist frequencies
    # At Nyquist, kN = - pi * N / L (appears once in mesh.x) is the same as pi * N / L, so corresponds to mu and -mu
    # Our treatment of hermitian symmetric field would sum this frequency twice (mu and -mu)
    # But this would appear only once in the uncompressed field
    # As a default, set frequencies beyond Nyquist to NaN
    # Margin for oversampling factor
    xmax = (meshsize // 2 - mode_oversampling) * spacing
    mask_beyond_nyq = np.flatnonzero(xedges >= np.min(xmax))
    xsum = xsum.reshape(xshape).at[mask_beyond_nyq].set(np.nan)
    musum = musum.reshape(xshape).at[mask_beyond_nyq].set(np.nan)
    ysum = ysum.reshape((nell,) + xshape).at[:, mask_beyond_nyq].set(np.nan)
    nsum = nsum.reshape(xshape).at[mask_beyond_nyq].set(0)

    # Reshape and slice to remove out of bounds points
    slx = slmu = slice(1, -2)
    if not exclude_zero:
        dig_zero = tuple(np.digitize(0., edges, right=False) for edges in [xedges, muedges if muedges is not None else [-1., 1.]])
        xsum = xsum.at[dig_zero].add(xsum[nx + 2, nmu + 2])
        if muedges is not None: musum = musum.at[dig_zero].add(musum[nx + 2, nmu + 2])
        ysum = ysum.at[(Ellipsis,) + dig_zero].add(ysum[:, nx + 2, nmu + 2])
        nsum = nsum.at[dig_zero].add(nsum[nx + 2, nmu + 2])

    with np.errstate(invalid='ignore', divide='ignore'):

        # 2D binned results
        if ells is None:
            ills = 0
        else:
            ills = tuple(unique_ells.index(ell) for ell in ells)
        if muedges is None:
            mumean2d = None
            slmu = nmu
        else:
            mumean2d = (musum / nsum)[slx, slmu]
        y2d = (ysum / nsum)[ills, slx, slmu]  # ell=0 is first index
        xmean2d = (xsum / nsum)[slx, slmu]
        n2d = nsum[slx, slmu]
        zero2d = ysum[ills, nx + 2, nmu + 2] / nsum[nx + 2, nmu + 2]
        if mode_oversampling: n2d = n2d / len(shifts)  # int => float

    return (xmean2d, mumean2d, y2d, n2d, zero2d)


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



def compute_mesh_power(*meshs: RealMeshField | ComplexMeshField | HermitianComplexMeshField, edges: np.ndarray | dict | None=None,
                       ells: int | tuple=0, los: str | np.ndarray='x', mode_oversampling: int=0) -> PowerSpectrumMultipoles:
    r"""
    Compute power spectrum from mesh.

    Parameters
    ----------
    meshs : RealMeshField, ComplexMeshField, HermitianComplexMeshField
        Input mesh(s).

    edges : np.ndarray, dict, default=None
        ``kedges`` may be:
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
        result = []
        nonzeroells = ells
        if nonzeroells[0] == 0: nonzeroells = nonzeroells[1:]
        if swap: meshs = meshs[::-1]

        A0 = _2c(meshs[0] if autocorr else meshs[1])
        if 0 in ells:
            Aell = A0 if autocorr else _2c(meshs[1])
        Aell = Aell.conj() * A0
        k, _, power, nmodes, power_zero = project_to_basis(Aell, edges, exclude_zero=False, mode_oversampling=mode_oversampling)
        result.append((power, power_zero))

        if nonzeroells:

            def _safe_divide(num, denom):
                with np.errstate(divide='ignore', invalid='ignore'):
                    return jnp.where(denom == 0., 0., num / denom)

            rmesh1 = _2r(meshs[0])
            # The real-space grid
            xhat = rmesh1.coords(sparse=True)
            xnorm = jnp.sqrt(sum(xx**2 for xx in xhat))
            xhat = [_safe_divide(xx, xnorm) for xx in xhat]
            del xnorm

            # The Fourier-space grid
            khat = A0.coords(sparse=True)
            knorm = jnp.sqrt(sum(kk**2 for kk in khat))
            khat = [_safe_divide(kk, knorm) for kk in khat]
            del knorm

            Ylms = {ell: [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in nonzeroells}

            for ell in nonzeroells:
                Aell = sum(_2c(rmesh1 * Ylm(*xhat)) * Ylm(*khat) for Ylm in Ylms[ell])
                Aell = Aell.conj() * A0
                # Project on to 1d k-basis (averaging over mu=[-1, 1])
                k, _, power, nmodes, power_zero = project_to_basis(Aell, edges, antisymmetric=bool(ell % 2), exclude_zero=False, mode_oversampling=mode_oversampling)
                result.append((4 * jnp.pi * power, 4 * jnp.pi * power_zero))

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        power, power_zero = (jnp.array([result[ells.index(ell)][ii] for ell in ells]) for ii in range(2))
        if swap: power, power_zero = (tmp.conj() for tmp in (power, power_zero))
        # Format the power results into :class:`PowerSpectrumMultipoles` instance
        return PowerSpectrumMultipoles(k, power_nonorm=power, nmodes=nmodes, edges=edges, ells=ells, norm=norm,
                                       power_zero_nonorm=power_zero, attrs=attrs)

    else:  # fixed line-of-sight

        for imesh, mesh in enumerate(meshs):
            meshs[imesh] = _2c(mesh)
        if autocorr:
            meshs.append(meshs[0])

        power = meshs[0] * meshs[1].conj()
        k, _, power, nmodes, power_zero = project_to_basis(power, edges, ells=ells, los=vlos, exclude_zero=False, mode_oversampling=mode_oversampling)
        return PowerSpectrumMultipoles(k, power_nonorm=power, nmodes=nmodes, edges=edges, ells=ells, norm=norm, power_zero_nonorm=power_zero, attrs=attrs)


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
        ``kedges`` may be:
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


def compute_mean_mesh_power(*meshs: RealMeshField | ComplexMeshField | HermitianComplexMeshField | MeshAttrs, theory: Callable | dict[Callable], edges: np.ndarray | dict | None=None,
                            ells: int | tuple=0, los: str | np.ndarray='x', mode_oversampling: int=0) -> PowerSpectrumMultipoles:
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
        Also possible to add wide-angle order :math:`n`, such that they key is the tuple :math:`(\ell, n)`.

    edges : np.ndarray, dict, default=None
        ``kedges`` may be:
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

    if vlos is not None:

        pkvec = theory

        if isinstance(theory, dict):
            def pkvec(kvec):
                knorm = sum(kk**2 for kk in kvec)**0.5
                mu = sum(kk * ll for kk, ll in zip(kvec, vlos))
                mu = jnp.where(knorm == 0., 0., mu / knorm)
                return sum(theory[ell](knorm) * legendre(ell)(mu) for ell in theory)

        power = mattrs.create(kind='hermitian_complex').apply(lambda value, kvec: pkvec(kvec) * norm, kind='wavenumber')

        if not periodic:

            for imesh, mesh in enumerate(meshs):
                meshs[imesh] = _2c(mesh)
            if autocorr:
                meshs.append(meshs[0])

            power = _2c(_2r(meshs[0] * meshs[1].conj()) * _2r(power))

        k, _, power, nmodes, power_zero = project_to_basis(power, edges, ells=ells, los=vlos, exclude_zero=False, mode_oversampling=mode_oversampling)
        return PowerSpectrumMultipoles(k, power_nonorm=power, nmodes=nmodes, edges=edges, ells=ells, norm=norm, power_zero_nonorm=power_zero, attrs=mattrs)

    else:
        theory_los = 'firstpoint'
        if isinstance(theory, tuple):
            theory, theory_los = theory
        assert isinstance(theory, dict)
        poles = {}
        for mode, p in theory.items():
            if isinstance(mode, tuple):
                poles[mode] = p
            else:
                poles[(mode, 0)] = p  # wide-angle = 0 as a default
        ellsin = [mode[0] for mode in poles]

        # In this case, theory must be a dictionary of (multipole, wide_angle_order)
        result = []
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
        # The real-space grid
        xhat = mattrs.xcoords(sparse=True)
        xnorm = jnp.sqrt(sum(xx**2 for xx in xhat))
        xhat = [_safe_divide(xx, xnorm) for xx in xhat]

        def _wrap_rslab(rslab):
            return tuple(jnp.where(rr > mattrs.boxsize[ii] / 2., rr - mattrs.boxsize[ii], rr) for ii, rr in enumerate(rslab))

        shat = _wrap_rslab(mattrs.clone(boxcenter=mattrs.boxsize / 2.).xcoords(sparse=True))
        snorm = jnp.sqrt(sum(xx**2 for xx in shat))
        #shat = [_safe_divide(xx, snorm) for xx in shat]

        # The Fourier-space grid
        khat = A0.coords(sparse=True)
        knorm = jnp.sqrt(sum(kk**2 for kk in khat))
        khat = [_safe_divide(kk, knorm) for kk in khat]
        del knorm

        Ylms = {ell: [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in set(ells) | set(ellsin)}

        for ell in ells:
            Aell = 0.
            for Ylm in Ylms[ell]:
                Q = 0.
                if theory_los == 'firstpoint':
                    for ell1, wa1 in poles:
                        for Yl1m1 in Ylms[ell1]:

                            def kernel(value, kvec):
                                knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
                                khat = [_safe_divide(kk, knorm) for kk in kvec]
                                return poles[ell1, wa1](knorm) * norm / mattrs.meshsize.prod(dtype=rdtype) * Yl1m1(*khat)

                            xi = mattrs.create(kind='hermitian_complex').apply(kernel, kind='wavenumber').c2r() * snorm**wa1
                            Q += (4. * np.pi) / (2 * ell1 + 1) * xi * _2r(_2c(rmesh1 * xnorm**(-wa1) * Ylm(*xhat) * Yl1m1(*xhat)).conj() * A0)

                elif theory_los == 'local':
                    coeff2 = {(0, 0): lambda k: poles[0, 0](k) - 7. / 18. * poles[4, 0](k),
                              (0, 2): lambda k: 1. / 2. * poles[2, 0](k) - 5. / 18. * poles[4, 0](k),
                              (2, 2): lambda k: 35. / 18. * poles[4, 0](k)}
                    coeff2[2, 0] = coeff2[0, 2]
                    for ell1, ell2 in itertools.product((0, 2), (0, 2)):
                        for Yl1m1, Yl2m2 in itertools.product(Ylms[ell1], Ylms[ell2]):

                            def kernel(value, kvec):
                                knorm = jnp.sqrt(sum(kk**2 for kk in kvec))
                                khat = [_safe_divide(kk, knorm) for kk in kvec]
                                return coeff2[ell1, ell2](knorm) * norm / mattrs.meshsize.prod(dtype=rdtype) * Yl1m1(*khat) * Yl2m2(*[-kk for kk in khat])

                            #xi = mattrs.create(kind='complex').apply(kernel, kind='wavenumber').c2r().real
                            xi = mattrs.create(kind='hermitian_complex').apply(kernel, kind='wavenumber').c2r()
                            Q += (4. * np.pi)**2 / ((2 * ell1 + 1) * (2 * ell2 + 1)) * xi * _2r(_2c(rmesh1 * Ylm(*xhat) * Yl1m1(*xhat)).conj() * _2c(rmesh2 * Yl2m2(*xhat)))

                else:
                    raise NotImplementedError(f'theory los {theory_los} not implemented')

                Aell += _2c(Q) * Ylm(*khat)
            # Project on to 1d k-basis (averaging over mu=[-1, 1])
            k, _, power, nmodes, power_zero = project_to_basis(Aell, edges, antisymmetric=bool(ell % 2), exclude_zero=False, mode_oversampling=mode_oversampling)
            result.append((4 * jnp.pi * power, 4 * jnp.pi * power_zero))

        # jax-mesh convention is F(k) = \sum_{r} e^{-ikr} F(r); let us correct it here
        power, power_zero = (jnp.array([result[ells.index(ell)][ii] for ell in ells]) for ii in range(2))
        if swap: power, power_zero = (tmp.conj() for tmp in (power, power_zero))
        # Format the power results into :class:`PowerSpectrumMultipoles` instance
        return PowerSpectrumMultipoles(k, power_nonorm=power, nmodes=nmodes, edges=edges, ells=ells, norm=norm,
                                       power_zero_nonorm=power_zero, attrs=attrs)