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

    def legendre(x):
        return jax.lax.switch(ell, _registered_legendre, x)

    return legendre


from scipy import special


def Si_scipy(x):
    return jax.pure_callback(lambda x: special.sici(x)[0], x, x)


def get_spherical_jn_scipy(ell):
    return lambda x: jax.pure_callback(partial(special.spherical_jn, ell), x, x)


def compute_sympy_bessel_tophat_integral(ell, n=11):
    import sympy as sp
    k, x = sp.symbols('k x', real=True, positive=True)
    integrand = sp.simplify(k**2 * sp.expand_func(sp.jn(ell, k * x)))
    expr = sp.integrate(integrand, (k, 0, 1))
    expr_lowx = sp.series(expr, x=x, x0=0, n=n).removeO()
    return expr, expr_lowx


def compute_sympy_legendre(ell):
    import sympy as sp
    x = sp.symbols('x', real=True)
    expr = sp.expand_func(sp.legendre(ell, x))
    return expr


def compute_sympy_bessel(ell, n=11):
    import sympy as sp
    x = sp.symbols('x', real=True)
    expr = sp.expand_func(sp.jn(ell, x))
    expr_lowx = sp.series(expr, x=x, x0=0, n=n).removeO()
    return expr, expr_lowx


_registered_bessel_tophat_integral = {}
_registered_bessel_tophat_integral[0] = (lambda x: (-jnp.cos(x)/x + jnp.sin(x)/x**2)/x,
                                         lambda x: -x**10/518918400 + x**8/3991680 - x**6/45360 + x**4/840 - x**2/30 + 1/3)
_registered_bessel_tophat_integral[1] = (lambda x: (-jnp.sin(x) - 2*jnp.cos(x)/x)/x**2 + 2/x**3,
                                         lambda x: x**9/47900160 - x**7/453600 + x**5/6720 - x**3/180 + x/12)
_registered_bessel_tophat_integral[2] = (lambda x: (x*jnp.cos(x) - 4*jnp.sin(x) + 3*Si_scipy(x))/x**3,
                                         lambda x: x**10/674593920 - x**8/5488560 + x**6/68040 - x**4/1470 + x**2/75)
_registered_bessel_tophat_integral[3] = (lambda x: 8/x**3 + (x**2*jnp.sin(x) + 7*x*jnp.cos(x) - 15*jnp.sin(x))/x**4,
                                         lambda x: -x**9/77837760 + x**7/831600 - x**5/15120 + x**3/630)
_registered_bessel_tophat_integral[4] = (lambda x: (-x**3*jnp.cos(x) + 11*x**2*jnp.sin(x) + 15*x**2*Si_scipy(x)/2 + 105*x*jnp.cos(x)/2 - 105*jnp.sin(x)/2)/x**5,
                                         lambda x: -x**10/1264863600 + x**8/11891880 - x**6/187110 + x**4/6615)
_registered_bessel_tophat_integral[5] = (lambda x: 16/x**3 + (-x**4*jnp.sin(x) - 16*x**3*jnp.cos(x) + 105*x**2*jnp.sin(x) + 315*x*jnp.cos(x) - 315*jnp.sin(x))/x**6,
                                         lambda x: x**9/194594400 - x**7/2702700 + x**5/83160)


_registered_bessel = {}
_registered_bessel[0] = (lambda x: jnp.sin(x)/x,
                         lambda x: -x**10/39916800 + x**8/362880 - x**6/5040 + x**4/120 - x**2/6 + 1)
_registered_bessel[1] = (lambda x: -jnp.cos(x)/x + jnp.sin(x)/x**2,
                         lambda x: x**9/3991680 - x**7/45360 + x**5/840 - x**3/30 + x/3)
_registered_bessel[2] = (lambda x: (-1/x + 3/x**3)*jnp.sin(x) - 3*jnp.cos(x)/x**2,
                         lambda x: x**10/51891840 - x**8/498960 + x**6/7560 - x**4/210 + x**2/15)
_registered_bessel[3] = (lambda x: (-6/x**2 + 15/x**4)*jnp.sin(x) + (1/x - 15/x**3)*jnp.cos(x),
                         lambda x: -x**9/6486480 + x**7/83160 - x**5/1890 + x**3/105)
_registered_bessel[4] = (lambda x: (10/x**2 - 105/x**4)*jnp.cos(x) + (1/x - 45/x**3 + 105/x**5)*jnp.sin(x),
                         lambda x: -x**10/97297200 + x**8/1081080 - x**6/20790 + x**4/945)
_registered_bessel[5] = (lambda x: (15/x**2 - 420/x**4 + 945/x**6)*jnp.sin(x) + (-1/x + 105/x**3 - 945/x**5)*jnp.cos(x),
                         lambda x: x**9/16216200 - x**7/270270 + x**5/10395)
_registered_bessel[6] = (lambda x: (-21/x**2 + 1260/x**4 - 10395/x**6)*jnp.cos(x) + (-1/x + 210/x**3 - 4725/x**5 + 10395/x**7)*jnp.sin(x),
                         lambda x: x**10/275675400 - x**8/4054050 + x**6/135135)
_registered_bessel[7] = (lambda x: (-28/x**2 + 3150/x**4 - 62370/x**6 + 135135/x**8)*jnp.sin(x) + (1/x - 378/x**3 + 17325/x**5 - 135135/x**7)*jnp.cos(x),
                         lambda x: -x**9/68918850 + x**7/2027025)
_registered_bessel[8] = (lambda x: (36/x**2 - 6930/x**4 + 270270/x**6 - 2027025/x**8)*jnp.cos(x) + (1/x - 630/x**3 + 51975/x**5 - 945945/x**7 + 2027025/x**9)*jnp.sin(x),
                         lambda x: -x**10/1309458150 + x**8/34459425)
_registered_bessel[9] = (lambda x: (45/x**2 - 13860/x**4 + 945945/x**6 - 16216200/x**8 + 34459425/x**10)*jnp.sin(x) + (-1/x + 990/x**3 - 135135/x**5 + 4729725/x**7 - 34459425/x**9)*jnp.cos(x),
                         lambda x: x**9/654729075)
_registered_bessel[10] = (lambda x: (-55/x**2 + 25740/x**4 - 2837835/x**6 + 91891800/x**8 - 654729075/x**10)*jnp.cos(x) + (-1/x + 1485/x**3 - 315315/x**5 + 18918900/x**7 - 310134825/x**9 + 654729075/x**11)*jnp.sin(x),
                          lambda x: x**10/13749310575)


def get_spherical_jn(ell):

    def jn(x):
        mask = x > 0.01
        bessel = _registered_bessel[ell]
        return jnp.where(mask, bessel[0](x), bessel[1](x))

    return jn


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return jnp.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]]) / 2.


class BesselIntegral(object):

    def __init__(self, xp, xeval, ell=0, edges=True, method='exact', mode='forward', volume=True):
        if edges:
            edges = xp
            xp = None
        else:
            edges = jnp.concatenate([xp[:1], (xp[1:] + xp[:-1]) / 2., xp[-1:]], axis=0)
        if edges.ndim == 1:
            edges = jnp.column_stack([edges[:-1], edges[1:]])
        if xp is None:
            xp = jnp.mean(edges, axis=-1)
        xeval = xeval[..., None]
        assert mode in ['forward', 'backward']
        if mode == 'forward':
            norm = (-1)**(ell // 2)
        else:
            norm = (-1)**((ell + 1) // 2) / (2 * np.pi)**3
        xmin = 0.01
        if method == 'rect':
            w = norm
            x = xeval * xp
            mask = x > xmin
            bessel = _registered_bessel[ell]
            self.w = norm * jnp.where(mask, bessel[0](x), bessel[1](x))
            #self.w = norm * get_spherical_jn(ell)(x)
            if volume: self.w *= (4. / 3. * np.pi) * (edges[:, 1]**3 - edges[:, 0]**3)
        elif method == 'trapz':
            x = xeval[..., None] * edges
            mask = x > xmin
            bessel = _registered_bessel[ell]
            self.w = norm * jnp.sum(jnp.where(mask, bessel[0](x), bessel[1](x)), axis=-1) / 2.
            if volume: self.w *= (4. / 3. * np.pi) * (edges[:, 1]**3 - edges[:, 0]**3)
        else:  # exact
            x = xeval[..., None] * edges
            mask = x > xmin
            tophat = _registered_bessel_tophat_integral[ell]
            w = jnp.where(mask, tophat[0](x), tophat[1](x)) * edges**3
            self.w =  norm * (w[..., 1] - w[..., 0])
            if volume: self.w *= (4. * np.pi)
            else: self.w /= (edges[:, 1]**3 - edges[:, 0]**3) / 3.

    def __call__(self, fun: jax.Array):
        return jnp.sum(self.w * fun, axis=-1)


class Interpolator1D(object):

    def __init__(self, x: jax.Array, xeval: jax.Array, order: int=0, edges=False, extrap=False):
        self.order = order
        self.mask = 1
        if self.order == 0:  # simple bins
            if edges:
                edges = x
            else:
                tmp = (x[:-1] + x[1:]) / 2.
                edges = np.concatenate([[tmp[0] - (x[1] - x[0])], tmp, [tmp[-1] + (x[-1] - x[-2])]])
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


def legendre_product(*ells):
    ells = tuple(sorted(ells))
    if len(ells) == 3:
        return _legendre_product3.get(ells, 0.)
    raise NotImplementedError('product of 3-legendre polynomials only is implemented')


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