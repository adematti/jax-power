"""
Implementation of the FFTlog algorithm, very much inspired by mcfit (https://github.com/eelregit/mcfit) and implementation in
https://github.com/sfschen/velocileptors/blob/master/velocileptors/Utils/spherical_bessel_transform_fftw.py
"""

import os

import numpy as np

from scipy.special import gamma as numpy_gamma
from scipy.special import loggamma as numpy_loggamma
import jax
from jax import numpy as jnp


def gamma(x):
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(numpy_gamma, result_shape, x)


def loggamma(x):
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(numpy_loggamma, result_shape, x)


@jax.tree_util.register_pytree_node_class
class FFTlog(object):
    r"""
    Implementation of the FFTlog algorithm presented in https://jila.colorado.edu/~ajsh/FFTLog/, which computes the generic integral:

    .. math::

        G(y) = \int_{0}^{\infty} x dx F(x) K(xy)

    with :math:`F(x)` input function, :math:`K(xy)` a kernel.

    This transform is (mathematically) invariant under a power law transformation:

    .. math::

        G_{q}(y) = \int_{0}^{\infty} x dx F_{q}(x) K_{q}(xy)

    where :math:`F_{q}(x) = G(x)x^{-q}`, :math:`K_{q}(t) = K(t)t^{q}` and :math:`G_{q}(y) = G(y)y^{q}`.
    """
    def __init__(self, x, kernel, q=0, minfolds=2, lowring=True, xy=1, check_level=0, engine='jax', **engine_kwargs):
        r"""
        Initialize :class:`FFTlog`, which can perform several transforms at once.

        Parameters
        ----------
        x : array_like
            Input log-spaced coordinates. Must be strictly increasing.
            If 1D, is broadcast to the number of provided kernels.

        kernel : callable, list of callables
            Mellin transform of the kernel:
            .. math:: U_{K}(z) = \int_{0}^{\infty} t^{z-1} K(t) dt
            If a list of kernels is provided, will perform all transforms at once.

        q : float, list of floats, default=0
            Power-law tilt(s) to regularise integration.

        minfolds : int, default=2
            Padded size is ``2**n``, with minimum :math:`n` statisfying ``2**n > minfolds * x.size``.

        lowring : bool, default=True
            If ``True`` set output coordinates according to the low-ringing condition, otherwise set it with ``xy``.

        xy : float, list of floats, default=1
            Enforce the reciprocal product (i.e. ``x[0] * y[-1]``) of the input ``x`` and output ``y`` coordinates.

        check_level : int, default=0
            If non-zero run sanity checks on input.

        engine : string, default='jax'
            FFT engine. See :meth:`set_fft_engine`.

        engine_kwargs : dict
            Arguments for FFT engine.

        Note
        ----
        Kernel definition is different from that of https://jila.colorado.edu/~ajsh/FFTLog/, which uses (eq. 10):

        .. math:: U_{K}(z) = \int_{0}^{\infty} t^{z} K(t) dt

        Therefore, one should use :math:`q = 1` for Bessel functions to match :math:`q = 0` in  https://jila.colorado.edu/~ajsh/FFTLog/.
        """
        self.inparallel = isinstance(kernel, (tuple, list))
        if not self.inparallel:
            kernel = [kernel]
        kernel = list(kernel)
        if np.ndim(q) == 0:
            q = [q] * len(kernel)
        q = list(q)
        self.x = jnp.asarray(x)
        if not self.inparallel:
            self.x = self.x[None, :]
        elif self.x.ndim == 1:
            self.x = jnp.tile(self.x[None, :], (len(kernel), 1))
        if np.ndim(xy) == 0:
            xy = [xy] * len(kernel)
        xy = list(xy)
        if check_level:
            if len(self.x) != len(kernel):
                raise ValueError('x and kernel must of same length')
            if len(q) != len(kernel):
                raise ValueError('q and kernel must be lists of same length')
            if len(xy) != len(kernel):
                raise ValueError('xy and kernel must be lists of same length')
        self._setup(kernel, q, minfolds=minfolds, lowring=lowring, xy=xy, check_level=check_level)
        self.set_fft_engine(engine, **engine_kwargs)

    def set_fft_engine(self, engine='jax', **engine_kwargs):
        """
        Set up FFT engine.
        See :func:`get_fft_engine`

        Parameters
        ----------
        engine : BaseEngine, string, default='jax'
            FFT engine, or one of ['jax'].

        engine_kwargs : dict
            Arguments for FFT engine.
        """
        self._engine = get_fft_engine(engine, size=self.padded_size, nparallel=self.nparallel, **engine_kwargs)

    @property
    def nparallel(self):
        """Number of transforms performed in parallel."""
        return self.x.shape[0]

    @property
    def size(self):
        """Size of x-coordinates."""
        return self.x.shape[-1]

    def _setup(self, kernels, qs, minfolds=2, lowring=True, xy=1., check_level=0):
        """Set up u funtions."""
        self.delta = jnp.log(self.x[:, -1] / self.x[:, 0]) / (self.size - 1)

        self.padded_size = self.size
        if minfolds:
            nfolds = (self.size * minfolds - 1).bit_length()
            self.padded_size = 2**nfolds

        npad = self.padded_size - self.size
        self.padded_size_in_left, self.padded_size_in_right = npad // 2, npad - npad // 2
        self.padded_size_out_left, self.padded_size_out_right = npad - npad // 2, npad // 2

        if check_level:
            if not jnp.allclose(jnp.log(self.x[:, 1:] / self.x[:, :-1]), self.delta, rtol=1e-3):
                raise ValueError('Input x must be log-spaced')
            if self.padded_size < self.size:
                raise ValueError('Convolution size must be larger than input x size')

        if lowring:
            self.lnxy = jnp.array([delta / jnp.pi * jnp.angle(kernel(q + 1j * np.pi / delta)) for kernel, delta, q in zip(kernels, self.delta, qs)], dtype=self.x.dtype)
        else:
            self.lnxy = jnp.log(jnp.array(xy))# + self.delta

        self.y = jnp.exp(self.lnxy - self.delta)[:, None] / self.x[:, ::-1]

        m = np.arange(0, self.padded_size // 2 + 1)
        self.padded_u, self.padded_prefactor, self.padded_postfactor = [], [], []
        self.padded_x = pad(self.x, (self.padded_size_in_left, self.padded_size_in_right), axis=-1, extrap='log')
        self.padded_y = pad(self.y, (self.padded_size_out_left, self.padded_size_out_right), axis=-1, extrap='log')
        prev_kernel, prev_q, prev_delta, prev_u = None, None, None, None
        for kernel, padded_x, padded_y, lnxy, delta, q in zip(kernels, self.padded_x, self.padded_y, self.lnxy, self.delta, qs):
            self.padded_prefactor.append(padded_x**(-q))
            self.padded_postfactor.append(padded_y**(-q))
            if kernel is prev_kernel and q == prev_q and delta == prev_delta:
                u = prev_u
            else:
                u = prev_u = kernel(q + 2j * np.pi / self.padded_size / delta * m)
            self.padded_u.append(u * jnp.exp(-2j * jnp.pi * lnxy / self.padded_size / delta * m))
            prev_kernel, prev_q, prev_delta = kernel, q, delta
        self.padded_u = jnp.array(self.padded_u)
        self.padded_prefactor = jnp.array(self.padded_prefactor)
        self.padded_postfactor = jnp.array(self.padded_postfactor)

    def tree_flatten(self):
        children = (self.x, self.y, self.padded_x, self.padded_y, self.padded_u, self.padded_prefactor, self.padded_postfactor)
        aux_data = {name: getattr(self, name) for name in ['inparallel', 'padded_size', 'padded_size_in_left', 'padded_size_in_right', 'padded_size_out_left', 'padded_size_out_right', '_engine']}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(aux_data)
        new.x, new.y, new.padded_x, new.padded_y, new.padded_u, new.padded_prefactor, new.padded_postfactor = children
        return new

    def __call__(self, fun, extrap=0, keep_padding=False, ignore_prepostfactor=False):
        """
        Perform the transforms.

        Parameters
        ----------
        fun : array_like
            Function to be transformed.
            Last dimensions should match (:attr:`nparallel`, len(x)) where ``len(x)`` is the size of the input x-coordinates.
            (if :attr:`nparallel` is 1, the only requirement is the last dimension to be len(x).

        extrap : float, string, default=0
            How to extrapolate function outside of  ``x`` range to fit the integration range.
            If 'log', performs a log-log extrapolation.
            If 'edge', pad ``fun`` with its edge values.
            Else, pad ``fun`` with the provided value.
            Pass a tuple to differentiate between left and right sides.

        keep_padding : bool, default=False
            Whether to return function padded to the number of points in the integral.
            By default, crop it to its original size.

        Returns
        -------
        y : array
            Array of new coordinates.

        fftloged : array
            Transformed function.
        """
        fun = jnp.asarray(fun)
        padded_fun = pad(fun, (self.padded_size_in_left, self.padded_size_in_right), axis=-1, extrap=extrap)
        padded_prefactor, padded_postfactor = self.padded_prefactor,  self.padded_postfactor
        if ignore_prepostfactor:
            padded_prefactor = padded_postfactor = 1.
        fftloged = self._engine.backward(self._engine.forward(padded_fun * padded_prefactor) * self.padded_u) * padded_postfactor

        if not keep_padding:
            y = self.y
            fftloged = fftloged[..., self.padded_size_out_left:self.padded_size_out_left + self.size]
        else:
            y = self.padded_y
        if not self.inparallel:
            y = y[0]
            fftloged = jnp.reshape(fftloged, fun.shape if not keep_padding else fun.shape[:-1] + (self.padded_size,))
        return y, fftloged

    def inv(self):
        """Inverse the transform."""
        self.x, self.y = self.y, self.x
        self.padded_x, self.padded_y = self.y, self.x
        self.padded_prefactor, self.padded_postfactor = 1 / self.padded_postfactor, 1 / self.padded_prefactor
        self.padded_u = 1 / self.padded_u.conj()


@jax.tree_util.register_pytree_node_class
class HankelTransform(FFTlog):
    """
    Hankel transform implementation using :class:`FFTlog`.

    It relies on Bessel function kernels.
    """
    def __init__(self, x, nu=0, **kwargs):
        """
        Initialize Hankel transform.

        Parameters
        ----------
        x : array_like
            Input log-spaced coordinates.
            If 1D, is broadcast to the number of provided ``nu``.

        nu : int, list of int, default=0
            Order of Bessel functions.
            If a list is provided, will perform all transforms at once.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        if np.ndim(nu) == 0:
            kernel = BesselJKernel(nu)
        else:
            kernel = [BesselJKernel(nu_) for nu_ in nu]
        FFTlog.__init__(self, x, kernel, **kwargs)
        self.padded_prefactor *= self.padded_x**2


@jax.tree_util.register_pytree_node_class
class SpectrumToCorrelation(FFTlog):
    r"""
    Power spectrum to correlation function transform, defined as:

    .. math::
        \xi_{\ell}(s) = \frac{(-i)^{\ell}}{2 \pi^{2}} \int dk k^{2} P_{\ell}(k) j_{\ell}(ks)

    """
    def __init__(self, k, ell=0, q=0, complex=False, **kwargs):
        """
        Initialize power to correlation transform.

        Parameters
        ----------
        k : array_like
            Input log-spaced wavenumbers.
            If 1D, is broadcast to the number of provided ``ell``.

        ell : int, list of int, default=0
            Poles. If a list is provided, will perform all transforms at once.

        q : float, list of floats, default=0
            Power-law tilt(s) to regularise integration.

        complex : bool, default=False
            ``False`` assumes the imaginary part of odd power spectrum poles is provided.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        if np.ndim(ell) == 0:
            kernel = SphericalBesselJKernel(ell)
        else:
            kernel = [SphericalBesselJKernel(ell_) for ell_ in ell]
        FFTlog.__init__(self, k, kernel, q=1.5 + q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 / (2 * np.pi)**1.5
        # Convention is (-i)^ell/(2 pi^2)
        ell = np.atleast_1d(ell)
        if complex:
            phase = (-1j) ** ell
        else:
            # Prefactor is (-i)^ell, but we take in the imaginary part of odd power spectra, hence:
            # (-i)^ell = (-1)^(ell/2) if ell is even
            # (-i)^ell i = (-1)^(ell//2) if ell is odd
            phase = (-1)**(ell // 2)
        # Not in-place as phase (and hence padded_postfactor) may be complex instead of float
        self.padded_postfactor = self.padded_postfactor * phase[:, None]


@jax.tree_util.register_pytree_node_class
class CorrelationToSpectrum(FFTlog):
    r"""
    Correlation function to power spectrum transform, defined as:

    .. math::
        P_{\ell}(k) = 4 \pi i^{\ell} \int ds s^{2} \xi_{\ell}(s) j_{\ell}(ks)

    """
    def __init__(self, s, ell=0, q=0, complex=False, **kwargs):
        """
        Initialize power to correlation transform.

        Parameters
        ----------
        s : array_like
            Input log-spaced separations.
            If 1D, is broadcast to the number of provided ``ell``.

        ell : int, list of int, default=0
            Poles. If a list is provided, will perform all transforms at once.

        q : float, list of floats, default=0
            Power-law tilt(s) to regularise integration.

        complex : bool, default=False
            ``False`` returns the real part of even poles, and the imaginary part of odd poles.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        if np.ndim(ell) == 0:
            kernel = SphericalBesselJKernel(ell)
        else:
            kernel = [SphericalBesselJKernel(ell_) for ell_ in ell]
        FFTlog.__init__(self, s, kernel, q=1.5 + q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 * (2 * np.pi)**1.5
        # Convention is 4 \pi i^ell, and we return imaginary part of odd poles
        ell = np.atleast_1d(ell)
        if complex:
            phase = (-1j) ** ell
        else:
            # We return imaginary part of odd poles
            phase = (-1)**(ell // 2)
        self.padded_postfactor = self.padded_postfactor * phase[:, None]


@jax.tree_util.register_pytree_node_class
class TophatVariance(FFTlog):
    """
    Variance in tophat window.

    It relies on tophat kernel.
    """
    def __init__(self, k, q=0, **kwargs):
        """
        Initialize tophat variance transform.

        Parameters
        ----------
        k : array_like
            Input log-spaced wavenumbers.
            If 1D, is broadcast to the number of provided ``ell``.

        q : float, list of floats, default=0
            Power-law tilt(s) to regularise integration.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        kernel = TophatSqKernel(ndim=3)
        FFTlog.__init__(self, k, kernel, q=1.5 + q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 / (2 * np.pi**2)


@jax.tree_util.register_pytree_node_class
class GaussianVariance(FFTlog):
    """
    Variance in Gaussian window.

    It relies on Gaussian kernel.
    """
    def __init__(self, k, q=0, **kwargs):
        """
        Initialize Gaussian variance transform.

        Parameters
        ----------
        k : array_like
            Input log-spaced wavenumbers.
            If 1D, is broadcast to the number of provided ``ell``.

        q : float, list of floats, default=0
            Power-law tilt(s) to regularise integration.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        kernel = GaussianSqKernel()
        FFTlog.__init__(self, k, kernel, q=1.5 + q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 / (2 * np.pi**2)


def pad(array, pad_width, axis=-1, extrap=0):
    """
    Pad array along ``axis``.

    Parameters
    ----------
    array : array_like
        Input array to be padded.

    pad_width : int, tuple of ints
        Number of points to be added on both sides of the array.
        Pass a tuple to differentiate between left and right sides.

    axis : int, default=-1
        Axis along which padding is to be applied.

    extrap : string, float, default=0
        If 'log', performs a log-log extrapolation.
        If 'edge', pad ``array`` with its edge values.
        Else, pad ``array`` with the provided value.
        Pass a tuple to differentiate between left and right sides.

    Returns
    -------
    array : array
        Padded array.
    """
    array = jnp.asarray(array)

    try:
        pad_width_left, pad_width_right = pad_width
    except (TypeError, ValueError):
        pad_width_left = pad_width_right = pad_width

    try:
        extrap_left, extrap_right = extrap
    except (TypeError, ValueError):
        extrap_left = extrap_right = extrap

    axis = axis % array.ndim
    to_axis = [1] * array.ndim
    to_axis[axis] = -1

    def index(i):
        return jnp.full(1, i, dtype='i4')

    if extrap_left == 'edge':
        end = jnp.take(array, index(0), axis=axis)
        pad_left = jnp.repeat(end, pad_width_left, axis=axis)
    elif extrap_left == 'log':
        end = jnp.take(array, index(0), axis=axis)
        ratio = jnp.take(array, index(1), axis=axis) / end
        exp = jnp.arange(-pad_width_left, 0).reshape(to_axis)
        pad_left = end * ratio ** exp
    else:
        pad_left = jnp.full(array.shape[:axis] + (pad_width_left,) + array.shape[axis + 1:], extrap_left)

    if extrap_right == 'edge':
        end = jnp.take(array, index(-1), axis=axis)
        pad_right = jnp.repeat(end, pad_width_right, axis=axis)
    elif extrap_right == 'log':
        end = jnp.take(array, index(-1), axis=axis)
        ratio = jnp.take(array, index(-2), axis=axis) / end
        exp = jnp.arange(1, pad_width_right + 1).reshape(to_axis)
        pad_right = end / ratio ** exp
    else:
        pad_right = jnp.full(array.shape[:axis] + (pad_width_right,) + array.shape[axis + 1:], extrap_right)

    return jnp.concatenate([pad_left, array, pad_right], axis=axis)


class BaseFFTEngine(object):

    """Base FFT engine."""

    def __init__(self, size, nparallel=1, nthreads=None):
        """
        Initialize FFT engine.

        Parameters
        ----------
        size : int
            Array size.

        nparallel : int, default=1
            Number of FFTs to be performed in parallel.

        nthreads : int, default=None
            Number of threads.
        """
        self.size = size
        self.nparallel = nparallel
        if nthreads is not None:
            os.environ['OMP_NUM_THREADS'] = str(nthreads)
        self.nthreads = int(os.environ.get('OMP_NUM_THREADS', 1))


class JAXFFTEngine(BaseFFTEngine):

    """FFT engine based on :mod:`numpy.fft`."""

    def forward(self, fun):
        """Forward transform of ``fun``."""
        return jnp.fft.rfft(fun, axis=-1)

    def backward(self, fun):
        """Backward transform of ``fun``."""
        return jnp.fft.irfft(fun.conj(), n=self.size, axis=-1)


def get_fft_engine(engine, *args, **kwargs):
    """
    Return FFT engine.

    Parameters
    ----------
    engine : BaseFFTEngine, string
        FFT engine, or one of ['jax'].

    args, kwargs : tuple, dict
        Arguments for FFT engine.

    Returns
    -------
    engine : BaseFFTEngine
    """
    if isinstance(engine, str):
        if engine.lower() == 'jax':
            return JAXFFTEngine(*args, **kwargs)
        raise ValueError('FFT engine {} is unknown'.format(engine))
    return engine


class BaseKernel(object):

    """Base kernel."""

    def __call__(self, z):
        return self.eval(z)

    def __eq__(self, other):
        return other.__class__ == self.__class__


class BaseBesselKernel(BaseKernel):

    """Base Bessel kernel."""

    def __init__(self, nu):
        self.nu = nu

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.nu == self.nu


class BesselJKernel(BaseBesselKernel):

    """(Mellin transform of) Bessel kernel."""

    def eval(self, z):
        return jnp.exp(jnp.log(2) * (z - 1) + loggamma(0.5 * (self.nu + z)) - loggamma(0.5 * (2 + self.nu - z)))


class SphericalBesselJKernel(BaseBesselKernel):

    """(Mellin transform of) spherical Bessel kernel."""

    def eval(self, z):
        return jnp.exp(jnp.log(2) * (z - 1.5) + loggamma(0.5 * (self.nu + z)) - loggamma(0.5 * (3 + self.nu - z)))


class BaseTophatKernel(BaseKernel):

    """Base tophat kernel."""

    def __init__(self, ndim=1):
        self.ndim = ndim

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.ndim == self.ndim


class TophatKernel(BaseTophatKernel):

    """(Mellin transform of) tophat kernel."""

    def eval(self, z):
        return jnp.exp(jnp.log(2) * (z - 1) + loggamma(1 + 0.5 * self.ndim) + loggamma(0.5 * z) - loggamma(0.5 * (2 + self.ndim - z)))


class TophatSqKernel(BaseTophatKernel):

    """(Mellin transform of) square of tophat kernel."""

    def __init__(self, ndim=1):
        self.ndim = ndim

    def eval(self, z):
        if self.ndim == 1:
            return -0.25 * jnp.sqrt(jnp.pi) * jnp.exp(loggamma(0.5 * (z - 2)) - loggamma(0.5 * (3 - z)))
        elif self.ndim == 3:
            return 2.25 * jnp.sqrt(jnp.pi) * (z - 2) / (z - 6) * jnp.exp(loggamma(0.5 * (z - 4)) - loggamma(0.5 * (5 - z)))
        else:
            return jnp.exp(jnp.log(2) * (self.ndim - 1) + 2 * loggamma(1 + 0.5 * self.ndim)
                              + loggamma(0.5 * (1 + self.ndim - z)) + loggamma(0.5 * z)
                              - loggamma(1 + self.ndim - 0.5 * z) - loggamma(0.5 * (2 + self.ndim - z))) / jnp.sqrt(jnp.pi)


class GaussianKernel(BaseKernel):

    """(Mellin transform of) Gaussian kernel."""

    def eval(self, z):
        return 2**(0.5 * z - 1) * gamma(0.5 * z)


class GaussianSqKernel(BaseKernel):

    """(Mellin transform of) square of Gaussian kernel."""

    def eval(self, z):
        return 0.5 * gamma(0.5 * z)
