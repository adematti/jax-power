import os
import sys
import time
import logging
import traceback
from collections.abc import Callable
import itertools
from functools import partial, lru_cache
from contextlib import contextmanager

import numpy as np
import jax
#from jax import config
#config.update('jax_enable_x64', True)
from jax import numpy as jnp


from lsstypes.utils import plotter


logger = logging.getLogger('Utils')


def exception_handler(exc_type, exc_value, exc_traceback):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '=' * 100
    # log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')


def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return


def savefig(filename: str, fig=None, bbox_inches='tight', pad_inches=0.1, dpi=200, **kwargs):
    """
    Save figure to ``filename``.

    Warning
    -------
    Take care to close figure at the end, ``plt.close(fig)``.

    Parameters
    ----------
    filename : str
        Path to save the figure.
    fig : matplotlib.figure.Figure, optional
        Figure to save. If None, uses current figure.
    bbox_inches : str, optional
        Bounding box for saving.
    pad_inches : float, optional
        Padding around the figure.
    dpi : int, optional
        Dots per inch.
    **kwargs
        Optional arguments for :meth:`matplotlib.figure.Figure.savefig`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib import pyplot as plt
    mkdir(os.path.dirname(filename))
    logger.info('Saving figure to {}.'.format(filename))
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filename, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi, **kwargs)
    return fig


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : str, int, default=logging.INFO
        Logging level.
    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.
    filename : string, default=None
        If not ``None`` stream to file name.
    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.
    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        def format(self, record):
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s'
            return super(MyFormatter, self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level, handlers=[handler], **kwargs)
    sys.excepthook = exception_handler


@contextmanager
def set_env(**environ):
    """
    Context manager to temporarily set environment variables.

    Example:
        with set_env(MY_VAR='value'):
            # MY_VAR is set to 'value' here
        # MY_VAR is restored to its original state here
    """
    original_env = os.environ.copy()
    os.environ.update({k: str(v) for k, v in environ.items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)


class MemoryMonitor(object):
    """
    Class that monitors memory usage and clock, useful to check for memory leaks.

    >>> with MemoryMonitor() as mem:
            '''do something'''
            mem()
            '''do something else'''
    """
    def __init__(self, pid=None):
        """
        Initalize :class:`MemoryMonitor` and register current memory usage.

        Parameters
        ----------
        pid : int, default=None
            Process identifier. If ``None``, use the identifier of the current process.
        """
        import psutil
        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.mem = self.proc.memory_info().rss / 1e6
        self.time = time.time()
        msg = 'using {:.3f} [Mb]'.format(self.mem)
        print(msg, flush=True)

    def __enter__(self):
        """Enter context."""
        return self

    def __call__(self, log=None):
        """Update memory usage."""
        mem = self.proc.memory_info().rss / 1e6
        t = time.time()
        msg = 'using {:.3f} [Mb] (increase of {:.3f} [Mb]) after {:.3f} [s]'.format(mem, mem - self.mem, t - self.time)
        if log:
            msg = '[{}] {}'.format(log, msg)
        print(msg, flush=True)
        self.mem = mem
        self.time = t

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()



@lru_cache(maxsize=32, typed=False)
@lru_cache(maxsize=None)
def _get_Ylm(ell, m, modules=None, reduced=False, real=False, conj=False):
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
    if reduced:
        amp = 1.
    if m != 0:
        fac = 1
        for n in range(ell - abs(m) + 1, ell + abs(m) + 1): fac *= n  # (ell + |m|)!/(ell - |m|)!
        amp *= np.sqrt(1. / fac)
    if real and m != 0:
        amp *= np.sqrt(2.) * (-1)**m
    if not real and m < 0:
        amp *= (-1)**m

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
            toret = jnp.asarray(func(*xhat))
            if not real: toret += 0 * 1j  # for JAX jit
            return toret

        for name, value in attrs.items():
            setattr(Ylm, name, value)
        return Ylm

    # sympy is not installed, fallback to scipy
    if sp is None:

        def _Ylm(xhat, yhat, zhat):
            # The cos(theta) dependence encoded by the associated Legendre polynomial
            toret = amp * special.lpmv(abs(m), ell, zhat)
            # The phi dependence
            phi = np.arctan2(yhat, xhat)
            if real:
                if m < 0:
                    eimphi = np.sin(abs(m) * phi)
                elif m > 0:
                    eimphi = np.cos(m * phi)
                else:
                    eimphi = 1
            else:
                eimphi = np.exp(1j * m * phi)
                if conj: eimphi = eimphi.conj()
            return toret * eimphi

        def func(xhat, yhat, zhat):
            shape = jnp.broadcast_shapes(jnp.shape(xhat), jnp.shape(yhat), jnp.shape(zhat))
            dtype = jnp.result_type(xhat, yhat, zhat)
            if not real:
                dtype = (1j * jnp.zeros((), dtype=dtype)).dtype
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
        expr = sp.assoc_legendre(ell, abs(m), sp.cos(theta))

        # The phi dependence
        if real:
            if m < 0:
                eimphi = sp.sin(abs(m) * phi)
            elif m > 0:
                eimphi = sp.cos(m * phi)
            else:
                eimphi = 1
        else:
            # lambdify doesn't seem to tacke sp.exp properly with JAX
            eimphi = sp.cos(m * phi) + sp.sin(m * phi) * 1j
            if conj: eimphi = sp.conjugate(eimphi)

        expr *= sp.expand_trig(eimphi)

        # Simplify
        expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
        expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat), (z / r, zhat)])
        func = sp.lambdify((xhat, yhat, zhat), expr, modules=['jax'])

        Ylm = get_Ylm_func(func, ell=ell, m=m, expr=expr)

    return Ylm


def get_Ylm(ell, m, modules=None, reduced=False, real=False, conj=False):
    # Cached: building the harmonic lambdifies a sympy expression, which costs ~0.1 s per (ell, m)
    # and grows with ell -- ~30 s for every order up to ell = 16, as the bispectrum window's
    # reference-leg shape factor needs. The returned closure is read-only, so sharing it is safe.
    if isinstance(modules, list): modules = tuple(modules)
    return _get_Ylm(int(ell), int(m), modules=modules, reduced=bool(reduced), real=bool(real), conj=bool(conj))


get_Ylm.__doc__ = _get_Ylm.__doc__

# Store in cache
[[get_Ylm(ell, m, reduced=False, real=True) for m in range(-ell, ell + 1)] for ell in (0, 2, 4)]


_registered_legendre = [None] * 11
_registered_legendre[0] = lambda x: jnp.ones_like(x)
_registered_legendre[1] = lambda x: x
_registered_legendre[2] = lambda x: 3*x**2/2 - 1/2
_registered_legendre[3] = lambda x: 5*x**3/2 - 3*x/2
_registered_legendre[4] = lambda x: 35*x**4/8 - 15*x**2/4 + 3/8
_registered_legendre[5] = lambda x: 63*x**5/8 - 35*x**3/4 + 15*x/8
_registered_legendre[6] = lambda x: 231*x**6/16 - 315*x**4/16 + 105*x**2/16 - 5/16
_registered_legendre[7] = lambda x: 429*x**7/16 - 693*x**5/16 + 315*x**3/16 - 35*x/16
_registered_legendre[8] = lambda x: 6435*x**8/128 - 3003*x**6/32 + 3465*x**4/64 - 315*x**2/32 + 35/128
_registered_legendre[9] = lambda x: 12155*x**9/128 - 6435*x**7/32 + 9009*x**5/64 - 1155*x**3/32 + 315*x/128
_registered_legendre[10] = lambda x: 46189*x**10/256 - 109395*x**8/256 + 45045*x**6/128 - 15015*x**4/128 + 3465*x**2/256 - 63/256


def get_legendre(ell):
    """
    Return a function evaluating the Legendre polynomial of order ``ell``.

    ``ell`` may be traced, but must lie within the tabulated range (0 to
    ``len(_registered_legendre) - 1``): :func:`jax.lax.switch` *clamps* an out-of-range index
    rather than raising, which would silently return the highest tabulated order instead.
    A static ``ell`` is checked here; a traced one cannot be, so use
    :func:`get_legendre_recurrence` when the order may exceed the table.
    """
    if np.ndim(ell) == 0 and not isinstance(ell, jax.core.Tracer):
        if not 0 <= int(ell) < len(_registered_legendre):
            raise ValueError(f'Legendre order ell = {ell} outside the tabulated range '
                             f'[0, {len(_registered_legendre) - 1}]; use get_legendre_recurrence instead')

    def legendre(x):
        return jax.lax.switch(ell, _registered_legendre, x)

    return legendre


def get_legendre_recurrence(ell, ellmax: int=None):
    r"""
    Return a function evaluating the Legendre polynomial of order ``ell``, using Bonnet's
    three-term recurrence :math:`(n + 1) P_{n + 1}(x) = (2n + 1) x P_n(x) - n P_{n - 1}(x)`.

    Unlike :func:`get_legendre` this has no tabulated upper bound, and accepts a **traced**
    ``ell`` without emitting one branch per order --- :func:`jax.lax.switch` would lower every
    tabulated branch into the graph. That is what lets callers that sum over many orders (e.g. the
    TripoSH sums of :func:`~jaxpower.mesh3.compute_smooth3_spectrum_window`) share one compilation
    instead of one per order.

    The recurrence is also numerically stable at high order, where evaluating the monomial form
    would suffer cancellation: :math:`P_{16}` has coefficients of order :math:`10^5` with
    alternating signs.

    The recurrence is unrolled whenever its length is known statically -- exactly when ``ell`` is a
    Python integer, and to ``ellmax`` steps followed by a select when it is traced. This matters a
    great deal: a dynamic :func:`jax.lax.fori_loop` lowers to a while loop, whose iterations XLA
    cannot fuse, so each of the ``ell`` steps becomes its own pass over the (potentially large)
    input. Measured on an A100 over a 1.7e7-element array, the dynamic form costs 10.6 ms at
    ``ell = 16`` against 0.3 ms at ``ell = 2``, while the unrolled form is a single fused pass.

    Parameters
    ----------
    ell : int, array
        Legendre order, possibly traced. Must be >= 0.
    ellmax : int, optional
        Maximum order ``ell`` can take, used to unroll the recurrence when ``ell`` is traced.
        **Must be a true upper bound**: orders above it would silently return a lower-order
        polynomial. Ignored when ``ell`` is static. If ``None`` and ``ell`` is traced, falls back to
        the dynamic (unfused, much slower) loop.

    Returns
    -------
    legendre : callable
        Function of ``x``, returning :math:`P_{\ell}(x)`.
    """
    static_ell = None
    if np.ndim(ell) == 0 and not isinstance(ell, jax.core.Tracer):
        static_ell = int(ell)
        if static_ell < 0:
            raise ValueError(f'Legendre order must be >= 0, got {static_ell}')

    def step(n, x, pm1, p):
        # (n + 1) P_{n + 1} = (2n + 1) x P_n - n P_{n - 1}
        return p, ((2 * n + 1) * x * p - n * pm1) / (n + 1)

    def legendre(x):
        x = jnp.asarray(x)
        # carry = (P_{n - 1}, P_n), starting at n = 1
        p0 = jnp.ones_like(x)
        p1 = x * jnp.ones_like(x)

        if static_ell is not None:  # fully unrolled, no select and no wasted step
            if static_ell == 0: return p0
            pm1, p = p0, p1
            for n in range(1, static_ell):
                pm1, p = step(n, x, pm1, p)
            return p

        n_ell = jnp.asarray(ell)
        if ellmax is None:  # correct, but the iterations do not fuse
            _, toret = jax.lax.fori_loop(1, n_ell, lambda n, c: step(n, x, *c), (p0, p1))
            return jnp.where(n_ell == 0, p0, toret)

        # unrolled to a static length; every step is elementwise, so the whole chain fuses
        toret = jnp.where(n_ell == 0, p0, p1)  # covers ell = 0 and ell = 1
        pm1, p = p0, p1
        for n in range(1, int(ellmax)):
            pm1, p = step(n, x, pm1, p)
            toret = jnp.where(n_ell == n + 1, p, toret)
        return toret

    return legendre


from scipy import special


def Si_scipy(x):
    return jax.pure_callback(lambda x: special.sici(x)[0], x, x)


def Si(x):
    from jax.scipy.special import sici
    return sici(x)[0]


def get_spherical_jn_scipy(ell):
    return lambda x: jax.pure_callback(partial(special.spherical_jn, ell), x, x)


def compute_sympy_bessel_tophat_integral(ell):
    import sympy as sp
    k, x = sp.symbols('k x', real=True, positive=True)
    integrand = sp.simplify(k**2 * sp.expand_func(sp.jn(ell, k * x)))
    return sp.integrate(integrand, (k, 0, 1))


def compute_sympy_legendre(ell):
    import sympy as sp
    x = sp.symbols('x', real=True)
    expr = sp.expand_func(sp.legendre(ell, x))
    return expr


def compute_sympy_bessel(ell):
    import sympy as sp
    x = sp.symbols('x', real=True)
    return sp.expand_func(sp.jn(ell, x))


_registered_bessel_tophat_integral = {}
_registered_bessel_tophat_integral[0] = lambda x: (-jnp.cos(x)/x + jnp.sin(x)/x**2)/x
_registered_bessel_tophat_integral[1] = lambda x: (-jnp.sin(x) - 2*jnp.cos(x)/x)/x**2 + 2/x**3
_registered_bessel_tophat_integral[2] = lambda x: (x*jnp.cos(x) - 4*jnp.sin(x) + 3*Si(x))/x**3
_registered_bessel_tophat_integral[3] = lambda x: 8/x**3 + (x**2*jnp.sin(x) + 7*x*jnp.cos(x) - 15*jnp.sin(x))/x**4
_registered_bessel_tophat_integral[4] = lambda x: (-x**3*jnp.cos(x) + 11*x**2*jnp.sin(x) + 15*x**2*Si(x)/2 + 105*x*jnp.cos(x)/2 - 105*jnp.sin(x)/2)/x**5
_registered_bessel_tophat_integral[5] = lambda x: 16/x**3 + (-x**4*jnp.sin(x) - 16*x**3*jnp.cos(x) + 105*x**2*jnp.sin(x) + 315*x*jnp.cos(x) - 315*jnp.sin(x))/x**6


_registered_bessel = {}
_registered_bessel[0] = lambda x: jnp.sin(x)/x
_registered_bessel[1] = lambda x: -jnp.cos(x)/x + jnp.sin(x)/x**2
_registered_bessel[2] = lambda x: (-1/x + 3/x**3)*jnp.sin(x) - 3*jnp.cos(x)/x**2
_registered_bessel[3] = lambda x: (-6/x**2 + 15/x**4)*jnp.sin(x) + (1/x - 15/x**3)*jnp.cos(x)
_registered_bessel[4] = lambda x: (10/x**2 - 105/x**4)*jnp.cos(x) + (1/x - 45/x**3 + 105/x**5)*jnp.sin(x)
_registered_bessel[5] = lambda x: (15/x**2 - 420/x**4 + 945/x**6)*jnp.sin(x) + (-1/x + 105/x**3 - 945/x**5)*jnp.cos(x)
_registered_bessel[6] = lambda x: (-21/x**2 + 1260/x**4 - 10395/x**6)*jnp.cos(x) + (-1/x + 210/x**3 - 4725/x**5 + 10395/x**7)*jnp.sin(x)
_registered_bessel[7] = lambda x: (-28/x**2 + 3150/x**4 - 62370/x**6 + 135135/x**8)*jnp.sin(x) + (1/x - 378/x**3 + 17325/x**5 - 135135/x**7)*jnp.cos(x)
_registered_bessel[8] = lambda x: (36/x**2 - 6930/x**4 + 270270/x**6 - 2027025/x**8)*jnp.cos(x) + (1/x - 630/x**3 + 51975/x**5 - 945945/x**7 + 2027025/x**9)*jnp.sin(x)
_registered_bessel[9] = lambda x: (45/x**2 - 13860/x**4 + 945945/x**6 - 16216200/x**8 + 34459425/x**10)*jnp.sin(x) + (-1/x + 990/x**3 - 135135/x**5 + 4729725/x**7 - 34459425/x**9)*jnp.cos(x)
_registered_bessel[10] = lambda x: (-55/x**2 + 25740/x**4 - 2837835/x**6 + 91891800/x**8 - 654729075/x**10)*jnp.cos(x) + (-1/x + 1485/x**3 - 315315/x**5 + 18918900/x**7 - 310134825/x**9 + 654729075/x**11)*jnp.sin(x)


def _spherical_jn_series(ell, nterms=32):
    r"""
    Power series :math:`j_\ell(x) = \frac{x^\ell}{(2\ell+1)!!} \sum_k
    \frac{(-x^2/2)^k}{k! (2\ell+3)(2\ell+5)\cdots(2\ell+2k+1)}`.

    Entire, so it converges everywhere; ``nterms`` sets how far out in :math:`x` it stays at
    machine precision. 24 terms carry it well past the crossover used by :func:`get_spherical_jn`.
    """
    dfact = 1.
    for n in range(1, int(ell) + 1): dfact *= 2 * n + 1

    def ser(x):
        corr = term = jnp.ones_like(x)
        for k in range(1, nterms + 1):
            term = term * -x**2 / (2. * k * (2 * ell + 2 * k + 1))
            corr = corr + term
        return x**ell / dfact * corr

    return ser


def _spherical_jn_tophat_integral_series(ell, nterms=32):
    r"""
    Power series for :math:`\int_0^1 u^2 j_\ell(x u) du`, obtained by integrating that of
    :func:`_spherical_jn_series` term by term:

    .. math::

        \frac{x^\ell}{(2\ell+1)!!} \sum_k \frac{(-x^2/2)^k}{k! (2\ell+3) \cdots (2\ell+2k+1)
        \, (\ell + 2k + 3)}

    i.e. the :math:`j_\ell` series with each term divided by :math:`\ell + 2k + 3`, the power of
    :math:`u` it integrates to. Entire, like the series it comes from.
    """
    dfact = 1.
    for n in range(1, int(ell) + 1): dfact *= 2 * n + 1

    def ser(x):
        term = jnp.ones_like(x)
        corr = term / (ell + 3.)
        for k in range(1, nterms + 1):
            term = term * -x**2 / (2. * k * (2 * ell + 2 * k + 1))
            corr = corr + term / (ell + 2 * k + 3.)
        return x**ell / dfact * corr

    return ser


def get_spherical_jn(ell):
    r"""
    Return a function evaluating the spherical Bessel function of order ``ell`` (a static integer).

    The tabulated closed forms carry coefficients of order :math:`(2\ell+1)!!` divided by powers of
    :math:`x` --- :math:`6.5 \times 10^8 / x^{11}` already at :math:`\ell = 10` --- which cancel to
    give an :math:`O(1)` result. Below :math:`x \sim \ell/2` that cancellation destroys the answer:
    measured against :mod:`scipy`, the previous unconditional use of the closed form above
    :math:`x = 0.1` was wrong by 1.9e-6 at :math:`\ell = 6`, 3.1e-2 at :math:`\ell = 8` and 1.0e+3
    at :math:`\ell = 10`, over a region reaching up to :math:`x \approx 0.42 \ell - 1.4`.
    The series is used there instead, and orders beyond the table fall back to the stable
    recurrence of :func:`get_spherical_jn_all`.
    """
    ell = int(ell)
    if ell not in _registered_bessel:
        jn_all = get_spherical_jn_all(ell)
        return lambda x: jn_all(x)[ell]

    closed = _registered_bessel[ell]
    # The cancellation region measured against scipy reaches ~0.42 * ell - 1.4, but the closed
    # form only reaches full accuracy somewhat beyond it (the surviving cancellation still costs
    # ~1e-13 at ell = 10 in float64, and ~1e-4 in float32, if the crossover is placed too early).
    xswitch = max(1.5, 0.7 * ell)
    ser = _spherical_jn_series(ell)

    def jn(x):
        x = jnp.asarray(x)
        # hold the closed form away from small x, where its 1/x**(ell+1) terms overflow: the value
        # is discarded there anyway, but inf - inf would poison the result with NaN
        return jnp.where(x > xswitch, closed(jnp.where(x > xswitch, x, 1.)), ser(x))

    return jn


@lru_cache(maxsize=None)
def _legendre_recurrence_coeffs(ellmax: int):
    r"""
    Coefficients for the normalized associated Legendre recurrence
    :math:`\lambda_{\ell m} = a_{\ell m} (x \lambda_{\ell-1, m} - b_{\ell m} \lambda_{\ell-2, m})`,
    with :math:`Y_{\ell m} = \lambda_{\ell m}(\cos\theta) e^{i m \phi}` (Condon-Shortley phase included).
    """
    ls = np.arange(ellmax + 1)[:, None]
    ms = np.arange(ellmax + 1)[None, :]
    mask = ms < ls
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(mask, np.sqrt((4. * ls**2 - 1.) / np.where(mask, ls**2 - ms**2, 1.)), 0.)
        b = np.where(mask, np.sqrt(np.abs(((ls - 1.)**2 - ms**2)) / (4. * (ls - 1.)**2 - 1.)), 0.)
    ab = a * b
    # diagonal: lambda_{ll} = -sqrt((2l+1)/(2l)) sin(theta) lambda_{l-1,l-1}
    d = np.zeros(ellmax + 1)
    d[1:] = -np.sqrt((2. * ls[1:, 0] + 1.) / (2. * ls[1:, 0]))
    onehot = np.eye(ellmax + 1)
    return a, ab, d, onehot


def get_Ylm_all(ellmax: int, reduced: bool=False):
    r"""
    Return a function evaluating all spherical harmonics up to ``ellmax`` at once, at azimuth 0
    where they are real, stacked on a leading axis in the flat order :math:`\ell^2 + \ell + m`
    (so ``ell**2 + ell + m`` indexes the :math:`(\ell, m)` harmonic).

    The all-orders counterpart of :func:`get_Ylm`, whose ``reduced`` convention it shares:
    ``reduced=True`` gives :math:`y_{\ell m} = \sqrt{(\ell - m)! / (\ell + m)!} P_\ell^m(\cos\theta)`,
    ``False`` the fully normalized :math:`Y_{\ell m}(\theta, 0)`, i.e. that times
    :math:`\sqrt{(2\ell + 1) / 4\pi}`. Condon-Shortley phase included in both.

    Built from the stable recurrence
    :math:`\lambda_{\ell m} = a_{\ell m}(x \lambda_{\ell - 1, m} - b_{\ell m}\lambda_{\ell - 2, m})`
    rather than from :func:`get_Ylm`'s tabulated closed forms. Two reasons, both growing with
    ``ellmax``: those closed forms carry :math:`(2\ell+1)!!`-sized coefficients that cancel (2e-11
    at :math:`\ell = 16` in float64, 7e-3 in float32, against :mod:`scipy`), and one lambdified
    expression per :math:`(\ell, m)` unrolls into the enclosing jit -- 43033 jaxpr equations at
    ``ellmax = 16`` against 1223 here, i.e. a 35x larger graph to compile.

    :func:`get_Ylm`'s other two flags have no counterpart here, because the azimuth is fixed at 0:
    the complex harmonic is real there (measured imaginary part exactly 0), so ``conj`` would be
    the identity, and the ``real`` convention degenerates -- it gives :math:`\pm\sqrt{2}` times
    the complex harmonic for :math:`m > 0` but vanishes identically for every :math:`m < 0`, half the
    table. Callers summing over signed :math:`m` (the TripoSH shape factor) need the complex one.

    Parameters
    ----------
    ellmax : int
        Largest degree returned; the stack holds ``(ellmax + 1)**2`` harmonics.
    reduced : bool, default=False
        Drop the :math:`\sqrt{(2\ell + 1) / 4\pi}` normalization, as :func:`get_Ylm`'s own
        ``reduced`` does (and with the same default).

    Returns
    -------
    Ylm_all : callable
        Function of :math:`\cos\theta`, returning an array of shape
        ``((ellmax + 1)**2,) + cos.shape``.
    """
    ellmax = int(ellmax)
    coeffs = _legendre_recurrence_coeffs(ellmax)
    ells = np.array([ell for ell in range(ellmax + 1) for m in range(-ell, ell + 1)])
    ms = np.array([m for ell in range(ellmax + 1) for m in range(-ell, ell + 1)])
    # lambda_{ell m} is exactly Y_{ell m} at azimuth 0, and is defined for m >= 0 only: mirror m < 0 with
    # y_{ell -m} = (-1)^m y_{ell m}, and strip sqrt((2 ell + 1) / 4 pi) if the reduced one is wanted
    factor = np.where(ms < 0, (-1.)**ms, 1.)
    if reduced: factor = factor * np.sqrt(4. * np.pi / (2. * ells + 1.))

    def Ylm_all(cos):
        cos = jnp.asarray(cos)
        dtype = cos.dtype
        a, ab, d, onehot = (jnp.asarray(tmp, dtype=dtype) for tmp in coeffs)
        sin = jnp.sqrt(1. - cos**2)
        c00 = 1. / np.sqrt(4. * np.pi)
        # lam holds lambda_{ell m} for all m at fixed ell, m on the last axis
        lam = jnp.zeros(cos.shape + (ellmax + 1,), dtype=dtype).at[..., 0].set(c00)
        diag = jnp.full(cos.shape, c00, dtype=dtype)

        def step(carry, xs):
            lam1, lam2, diag = carry
            a_row, ab_row, d_ell, oh = xs
            diag = d_ell * sin * diag                       # lambda_{ell ell}, the recurrence's seed
            lam = a_row * (cos[..., None] * lam1) - ab_row * lam2 + diag[..., None] * oh
            return (lam, lam1, diag), lam

        table = lam[None]
        if ellmax:
            table = jnp.concatenate([table, jax.lax.scan(step, (lam, jnp.zeros_like(lam), diag),
                                                         (a[1:], ab[1:], d[1:], onehot[1:]))[1]])
        # (ell, ..., m) -> (row, ...), one row per (ell, m)
        table = jnp.moveaxis(table, -1, 1)[ells, np.abs(ms)]
        return jnp.asarray(factor, dtype=dtype).reshape((-1,) + (1,) * cos.ndim) * table

    return Ylm_all


def get_spherical_jn_all(ellmax: int, n_iter: int=None, xmin: float=0.1):
    r"""
    Return a function evaluating :math:`j_0` to :math:`j_{\ell_{\mathrm{max}}}` at once, stacked
    along a leading axis.

    :func:`get_spherical_jn` stops at :math:`\ell = 10` and evaluates the explicit closed form,
    whose coefficients grow like :math:`(2\ell + 1)!!` --- already :math:`10^{17}` at
    :math:`\ell = 16`, where they cancel catastrophically at moderate argument. This uses the
    standard stable combination instead, and returns every order in one pass, which is what a
    traced-order gather needs (indexing a Python table requires a static order).

    Regimes, each valid where the others are not:

    - :math:`x < x_{\mathrm{min}}`: power series
      :math:`j_n(x) = \frac{x^n}{(2n+1)!!} \left(1 - \frac{x^2}{2(2n+3)} + \ldots\right)`;
    - :math:`x > \ell`: upward recurrence
      :math:`j_{n+1} = \frac{2n+1}{x} j_n - j_{n-1}`, stable only in this regime;
    - otherwise: Miller's downward recurrence from ``n_iter``, seeded arbitrarily and rescaled at
      the end onto the known :math:`j_0` (or :math:`j_1`, whichever is larger there --- their zeros
      interlace, so they never vanish together and the normalization never degenerates).

    Parameters
    ----------
    ellmax : int
        Largest order returned.
    n_iter : int, optional
        Starting order of the downward recurrence. Defaults to ``ellmax + 40``. Only the
        :math:`x \leq \ell` regime uses it, so it need not exceed the argument.
    xmin : float, default=0.1
        Below this the series is used, which also keeps the recurrences away from :math:`x = 0`.

    Returns
    -------
    jn_all : callable
        Function of ``x``, returning an array of shape ``(ellmax + 1,) + x.shape``.
    """
    ellmax = int(ellmax)
    if n_iter is None: n_iter = ellmax + 40
    if n_iter <= ellmax:
        raise ValueError(f'n_iter = {n_iter} must exceed ellmax = {ellmax}')

    def jn_all(x):
        x = jnp.asarray(x)
        # the recurrences divide by x; the series branch covers whatever is masked out here
        safe = jnp.where(x < xmin, 1., x)
        j0 = jnp.sin(safe) / safe
        j1 = jnp.sin(safe) / safe**2 - jnp.cos(safe) / safe

        # upward: stable for x > ell, garbage (but finite) below, and masked out there
        ups = [j0] + ([j1] if ellmax >= 1 else [])
        for n in range(1, ellmax):
            ups.append((2 * n + 1) / safe * ups[n] - ups[n - 1])
        up = jnp.stack(ups)

        # Miller downward: the seed is arbitrary (the normalization below fixes the scale), but the
        # recurrence grows by ~(2n+1)/x per step -- some 1e130 over the whole sweep at x = xmin --
        # so it must be renormalized as it goes. A fixed small seed instead of renormalizing works
        # in float64 but silently flushes to zero (hence 0/0 = NaN) in float32.
        cap = jnp.asarray(np.sqrt(np.finfo(jnp.result_type(safe)).max), dtype=safe.dtype)
        jp1, jcur = jnp.zeros_like(safe), jnp.ones_like(safe)
        store = [None] * (ellmax + 1)
        for n in range(n_iter, 0, -1):
            jp1, jcur = jcur, (2 * n + 1) / safe * jcur - jp1
            scale = jnp.where(jnp.abs(jcur) > cap, 1. / cap, jnp.ones_like(cap))
            jcur, jp1 = jcur * scale, jp1 * scale
            for k in range(ellmax + 1):  # orders already stored share the rescaling
                if store[k] is not None: store[k] = store[k] * scale
            if n - 1 <= ellmax: store[n - 1] = jcur
        down = jnp.stack(store)
        # normalize on whichever of j0, j1 is larger: sin(x)/x vanishes at multiples of pi
        use0 = jnp.abs(j0) >= jnp.abs(j1)
        down = down * jnp.where(use0, j0 / store[0], j1 / store[1])

        # small-x series
        xs = jnp.where(x < xmin, x, 0.)
        ser, dfact = [], 1.
        for n in range(ellmax + 1):
            if n: dfact *= 2 * n + 1
            # sum_k (-x^2 / 2)^k / (k! (2n+3)(2n+5)...(2n+2k+1)); 4 terms leave a relative
            # truncation ~ x^10 / (2^5 5! 11!!) at worst, i.e. ~1e-16 at x = xmin = 0.1
            corr = term = 1.
            for k in range(1, 5):
                term = term * -xs**2 / (2. * k * (2 * n + 2 * k + 1))
                corr = corr + term
            ser.append(xs**n / dfact * corr)
        ser = jnp.stack(ser)

        ell = jnp.arange(ellmax + 1).reshape((-1,) + (1,) * x.ndim)
        return jnp.where(x < xmin, ser, jnp.where(x > jnp.maximum(ell, 1.), up, down))

    return jn_all


def get_spherical_jn_tophat_integral(ell):
    r"""
    Return a function of ``(xeval, edges)`` giving :math:`4\pi \int_{r_-}^{r_+} r^2 j_\ell(k r) dr`,
    with :math:`k` the evaluation points and :math:`(r_-, r_+)` the last axis of ``edges``.

    As in :func:`get_spherical_jn`, the tabulated closed form loses the argument to cancellation at
    small :math:`x = k r`: against a 40-digit reference it is wrong by 5e-3 at :math:`\ell = 4` and
    by a factor 10 at :math:`\ell = 5` at :math:`x = 0.1`, which the previous cut at 0.1 handed
    straight through. :func:`_spherical_jn_tophat_integral_series` is used below the crossover.
    """
    closed = _registered_bessel_tophat_integral[ell]
    xswitch = max(1.5, 0.7 * ell)
    ser = _spherical_jn_tophat_integral_series(ell)

    def jn_tophat(xeval, edges):
        x = xeval[..., None, None] * edges
        mask = x > xswitch
        # hold the closed form away from small x, where its inverse powers of x overflow: the value
        # is discarded there anyway, but inf - inf would poison the result with NaN
        w = jnp.where(mask, closed(jnp.where(mask, x, 1.)), ser(x)) * edges**3
        return 4. * np.pi * (w[..., 1] - w[..., 0])

    return jn_tophat


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return jnp.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]]) / 2.


class BesselIntegral(object):

    def __init__(self, xp, xeval, ell=0, edges=True, method='exact', mode='forward', volume=True):
        # If mode = 'forward', xp is 's', xeval 'k'
        # If mode = 'backward', xp is 'k', xeval 's'
        # edges = True if xp is edges
        if edges:
            edges = xp
            xp = None
        else:
            edges = jnp.concatenate([xp[:1], (xp[1:] + xp[:-1]) / 2., xp[-1:]], axis=0)
        if edges.ndim == 1:
            edges = jnp.column_stack([edges[:-1], edges[1:]])
        if xp is None:
            xp = jnp.mean(edges, axis=-1)
        assert mode in ['forward', 'backward']
        if mode == 'forward':
            norm = (-1)**(ell // 2)
        else:
            norm = (-1)**(ell // 2) / (2 * np.pi)**3
        if method == 'rect':
            x = xeval[..., None] * xp
            self.w = norm * get_spherical_jn(ell)(x)
            if volume: self.w *= (4. / 3. * np.pi) * (edges[:, 1]**3 - edges[:, 0]**3)
        elif method == 'trapz':
            x = xeval[..., None, None] * edges
            self.w = norm * jnp.sum(get_spherical_jn(ell)(x), axis=-1) / 2.
            if volume: self.w *= (4. / 3. * np.pi) * (edges[:, 1]**3 - edges[:, 0]**3)
        else:  # exact
            self.w = norm * get_spherical_jn_tophat_integral(ell)(xeval, edges)
            if not volume:
                self.w /= (4. * np.pi) / 3. * (edges[:, 1]**3 - edges[:, 0]**3)

    def __call__(self, fun: jax.Array):
        return jnp.sum(self.w * fun, axis=-1)


class Interpolator1D(object):

    def __init__(self, x: jax.Array, xeval: jax.Array, order: int=0, edges=False, extrap=False):
        self.order = order
        self.mask = 1
        if edges:
            edges = x
            x = (edges[:-1] + edges[1:]) / 2.
        else:
            tmp = (x[:-1] + x[1:]) / 2.
            edges = np.concatenate([[tmp[0] - (x[1] - x[0])], tmp, [tmp[-1] + (x[-1] - x[-2])]])
        if self.order == 0:  # simple bins
            self.idx = jnp.digitize(xeval, edges, right=False) - 1
            if not extrap: self.mask = (self.idx >= 0) & (self.idx <= len(edges) - 2)
            self.idx = jnp.where(self.mask, self.idx, 0)
        elif self.order == 1:
            self.idx = jnp.digitize(xeval, x, right=False) - 1
            if not extrap: self.mask = (self.idx >= 0) & (self.idx <= len(x) - 1)
            self.idx = jnp.clip(self.idx, 0, len(x) - 2)
            self.fidx = jnp.clip(xeval - x[self.idx], 0., 1.)
        else:
            raise NotImplementedError

    def __call__(self, fun: jax.Array):
        fun = jnp.asarray(fun)
        if self.order == 0:
            toret = fun[self.idx]
        if self.order == 1:
            toret = (1. - self.fidx) * fun[self.idx] + self.fidx * fun[self.idx + 1]
        toret *= self.mask
        return toret


def compute_sympy_real_gaunt(*ellms):
    import sympy as sp
    phi, theta = sp.symbols('phi theta', real=True)

    def _Ylm(ell, m):
        # Normalization of Ylms
        amp = sp.sqrt((2 * ell + 1) / (4 * sp.pi))
        if m != 0:
            fac = 1
            for n in range(ell - abs(m) + 1, ell + abs(m) + 1): fac *= n  # (ell + |m|)!/(ell - |m|)!
            amp *= sp.sqrt(2) / sp.sqrt(fac)
        expr = (-1)**m * sp.assoc_get_legendre(ell, abs(m), sp.cos(theta))
        # The phi dependence
        if m < 0:
            expr *= sp.sin(abs(m) * phi)
        elif m > 0:
            expr *= sp.cos(m * phi)
        return amp * expr

    Ylm123 = 1
    for ell, m in ellms:
        Ylm123 *= _Ylm(ell, m)
    expr = sp.integrate(Ylm123 * sp.sin(theta), (phi, 0, 2 * sp.pi), (theta, 0, sp.pi))
    return expr


def export_real_gaunt():
    import itertools
    toret = {}
    for ells in itertools.product((0, 2, 4), (0, 2), (0, 2)):
        for ms in itertools.product(*(list(range(-ell, ell + 1)) for ell in ells)):
            ellms = tuple(zip(ells, ms))
            tmp = float(compute_sympy_real_gaunt(*ellms))
            if tmp != 0.:
                toret[ellms] = tmp
    return toret


def compute_sympy_legendre_product(*ells):
    import sympy as sp
    mu = sp.symbols('mu', real=True)

    legendre = 1
    for ell in ells:
        legendre *= sp.legendre(ell, mu)
    expr = sp.integrate(legendre, (mu, -1, 1)) / 2
    return expr


def export_legendre_product(ellmax=8, n=3):

    import itertools
    toret = {}
    for ells in itertools.combinations_with_replacement(tuple(range(ellmax + 1)), n):
        tmp = compute_sympy_legendre_product(*ells)
        if tmp != 0.:
            toret[ells] = tmp
    return toret


_real_gaunt = {((0, 0), (0, 0), (0, 0)): 0.28209479177387814, ((0, 0), (2, -2), (2, -2)): 0.28209479177387814, ((0, 0), (2, -1), (2, -1)): 0.28209479177387814, ((0, 0), (2, 0), (2, 0)): 0.28209479177387814,
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


def real_gaunt(*ellms):
    return _real_gaunt.get(ellms, 0)


_legendre_product3 = {(0, 0, 0): 1, (0, 1, 1): 1/3, (0, 2, 2): 1/5, (0, 3, 3): 1/7, (0, 4, 4): 1/9, (0, 5, 5): 1/11, (0, 6, 6): 1/13, (0, 7, 7): 1/15, (0, 8, 8): 1/17, (1, 1, 2): 2/15, (1, 2, 3): 3/35, (1, 3, 4): 4/63,
                      (1, 4, 5): 5/99, (1, 5, 6): 6/143, (1, 6, 7): 7/195, (1, 7, 8): 8/255, (2, 2, 2): 2/35, (2, 2, 4): 2/35, (2, 3, 3): 4/105, (2, 3, 5): 10/231, (2, 4, 4): 20/693, (2, 4, 6): 5/143, (2, 5, 5): 10/429,
                      (2, 5, 7): 21/715, (2, 6, 6): 14/715, (2, 6, 8): 28/1105, (2, 7, 7): 56/3315, (2, 8, 8): 24/1615, (3, 3, 4): 2/77, (3, 3, 6): 100/3003, (3, 4, 5): 20/1001, (3, 4, 7): 35/1287, (3, 5, 6): 7/429,
                      (3, 5, 8): 56/2431, (3, 6, 7): 168/12155, (3, 7, 8): 252/20995, (4, 4, 4): 18/1001, (4, 4, 6): 20/1287, (4, 4, 8): 490/21879, (4, 5, 5): 2/143, (4, 5, 7): 280/21879, (4, 6, 6): 28/2431, (4, 6, 8): 504/46189,
                      (4, 7, 7): 2268/230945, (4, 8, 8): 36/4199, (5, 5, 6): 80/7293, (5, 5, 8): 490/46189, (5, 6, 7): 420/46189, (5, 7, 8): 360/46189, (6, 6, 6): 400/46189, (6, 6, 8): 350/46189, (6, 7, 7): 1000/138567,
                      (6, 8, 8): 600/96577, (7, 7, 8): 1750/289731, (8, 8, 8): 490/96577}


@lru_cache(maxsize=None)
def _wigner_3j(*ells):
    from sympy.physics.wigner import wigner_3j
    return float(wigner_3j(*ells))


def wigner_3j(*ells):
    return _wigner_3j(*map(int, ells))


def _delta_log_6j(a, b, c):
    """log of the Racah triangle coefficient Delta(abc); -inf if the triangle fails."""
    from scipy.special import gammaln
    if a + b - c < 0 or a - b + c < 0 or -a + b + c < 0:
        return -np.inf
    lf = lambda n: gammaln(n + 1.)
    return 0.5 * (lf(a + b - c) + lf(a - b + c) + lf(-a + b + c) - lf(a + b + c + 1))


@lru_cache(maxsize=None)
def _wigner_6j(a, b, c, d, e, f):
    """Racah formula for the 6j in floating point, via log-factorials."""
    from scipy.special import gammaln
    lf = lambda n: gammaln(n + 1.)
    dl = (_delta_log_6j(a, b, c) + _delta_log_6j(a, e, f)
          + _delta_log_6j(d, b, f) + _delta_log_6j(d, e, c))
    if not np.isfinite(dl):
        return 0.
    a1, a2, a3, a4 = a + b + c, a + e + f, d + b + f, d + e + c
    b1, b2, b3 = a + b + d + e, b + c + e + f, a + c + d + f
    lo, hi = int(max(a1, a2, a3, a4)), int(min(b1, b2, b3))
    if lo > hi:
        return 0.
    tot = 0.
    for t in range(lo, hi + 1):
        tot += (-1)**t * np.exp(dl + lf(t + 1) - lf(t - a1) - lf(t - a2) - lf(t - a3) - lf(t - a4)
                                - lf(b1 - t) - lf(b2 - t) - lf(b3 - t))
    return float(tot)


@lru_cache(maxsize=None)
def _wigner_9j(*ells):
    """9j via the Racah single sum over 6j.

    sympy's exact-rational wigner_9j costs 20-200 ms for each distinct symbol at the orders the
    scoccimarro window matrix needs (and these calls are already lru_cached, so memoisation cannot
    help further) -- which is what made ellmax >= 16 impractical. The identity

        {a b c; d e f; g h i} = sum_x (-1)^(2x) (2x+1) {a b c; f i x}{d e f; b x h}{g h i; x a d}

    with the 6j evaluated in floating point gives 20-58x, growing with ell, and agrees with sympy to
    ~1e-16. The float 6j is essential here: using sympy's exact 6j inside this sum is slower than one
    exact 9j (~0.6x), since it trades one exact symbol for ~30 of them.
    """
    a, b, c, d, e, f, g, h, i = ells
    for (x, y, z) in ((a, b, c), (d, e, f), (g, h, i), (a, d, g), (b, e, h), (c, f, i)):
        if not (abs(x - y) <= z <= x + y):
            return 0.
    lo = max(abs(a - i), abs(d - h), abs(b - f))
    hi = min(a + i, d + h, b + f)
    tot = 0.
    for x in range(int(lo), int(hi) + 1):
        t = _wigner_6j(a, b, c, f, i, x)
        if t == 0.: continue
        u = _wigner_6j(d, e, f, b, x, h)
        if u == 0.: continue
        v = _wigner_6j(g, h, i, x, a, d)
        if v == 0.: continue
        tot += (-1)**(2 * x) * (2 * x + 1) * t * u * v
    return float(tot)


def wigner_9j(*ells):
    return _wigner_9j(*map(int, ells))


def legendre_product(*ells):
    ells = tuple(sorted(ells))
    if len(ells) == 3:
        return _legendre_product3.get(ells, 0.)
    raise NotImplementedError('product of 3-legendre polynomials only is implemented')


def get_S(ells, z3=False):
    r"""
    Return a function computing the Gaunt-coefficient-weighted triple
    spherical-harmonic coupling basis :math:`S_{\ell_1\ell_2\ell_3}(\hat x_1, \hat x_2, \hat x_3)`,
    built from the JAX-traceable spherical harmonics in :func:`get_Ylm`.

    Parameters
    ----------
    ells : tuple
        (ell1, ell2, ell3).
    z3 : bool, default=False
        If True, the returned function takes only two unit vectors
        (``xhat1``, ``xhat2``); the third leg is implicitly aligned with the
        line of sight (i.e. its harmonic order is fixed to ``m3 = 0``).

    Returns
    -------
    Sell : callable
    """
    ell1, ell2, ell3 = ells
    H = wigner_3j(ell1, ell2, ell3, 0, 0, 0)

    if abs(H) < 1e-12:
        return lambda *args: 0.

    def _Ylm(ell, m, xhat):
        # No sign compensation is applied. This used to multiply by (-1)**m for m < 0, on the grounds that
        # get_Ylm's non-real convention carries an extra such factor relative to the plain
        # amp * lpmv(|m|, ell, mu) * exp(i m phi) the Gaunt coefficients were derived in. Measured,
        # it does not: get_Ylm(ell, m, reduced=True) equals sqrt(4 pi / (2 ell + 1)) Y_ell^m to 1e-16
        # for every m tested. The compensation was therefore spurious and broke the addition
        # theorem -- the convention-free identity S_(l l 0)(x1, x2, x3) = L_l(x1.x2) failed by
        # O(1) with it (l = 1 gave +0.269 against the correct -0.644) and holds to 1e-15 without.
        # Guarded by test_utils.py::test_S.
        out = get_Ylm(ell, m, reduced=True)(xhat[..., 0], xhat[..., 1], xhat[..., 2])
        # For e.g. ell = m = 0, get_Ylm's (lambdified) expression does not
        # depend on xhat at all, so it returns a bare scalar that doesn't
        # carry xhat's shape; broadcast explicitly.
        return jnp.broadcast_to(out, jnp.shape(xhat)[:-1])

    if z3:  # last vector is z, so m3 = 0
        coeffs = []
        mmax = min(ell1, ell2)
        for m in range(-mmax, mmax + 1):
            gaunt = wigner_3j(ell1, ell2, ell3, m, -m, 0) / H
            if abs(gaunt) > 1e-12:
                coeffs.append((m, gaunt))

        def Sell(xhat1, xhat2):
            out = 0.0
            for m, gaunt in coeffs:
                out = out + gaunt * _Ylm(ell1, m, xhat1) * _Ylm(ell2, -m, xhat2)
            return out.real if ((ell1 + ell2 + ell3) % 2 == 0) else out.imag

    else:
        ms = [range(-ell, ell + 1) for ell in ells]
        coeffs = []
        for m1, m2, m3 in itertools.product(*ms):
            gaunt = wigner_3j(ell1, ell2, ell3, m1, m2, m3) / H
            if abs(gaunt) > 1e-12:
                coeffs.append((m1, m2, m3, gaunt))

        def Sell(*xhats):
            out = 0.
            for m1, m2, m3, gaunt in coeffs:
                term = gaunt
                for ell, m, xhat in zip(ells, (m1, m2, m3), xhats, strict=True):
                    term = term * _Ylm(ell, m, xhat)
                out = out + term
            return out.real if ((ell1 + ell2 + ell3) % 2 == 0) else out.imag

    return Sell


def register_pytree_dataclass(cls, meta_fields=None):

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in cls._data_fields), {name: getattr(self, name) for name in cls._meta_fields}

    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(zip(cls._data_fields, children))
        new.__dict__.update(aux_data)
        return new

    cls._meta_fields = tuple(meta_fields or [])
    cls._data_fields = tuple(name for name in cls.__annotations__.keys() if name not in cls._meta_fields)
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = classmethod(tree_unflatten)
    return jax.tree_util.register_pytree_node_class(cls)


def estimate_memory(fun, *args, **kwargs):
    lowered = fun.lower(*args, **kwargs)
    compiled = lowered.compile()
    memory_analysis = compiled.memory_analysis()
    print(f"{memory_analysis}")
    print(f"Estimated memory cost: {(memory_analysis.temp_size_in_bytes + memory_analysis.argument_size_in_bytes + memory_analysis.output_size_in_bytes - memory_analysis.alias_size_in_bytes) / 1024 ** 2:.2f} MB")
