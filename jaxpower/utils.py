import os
import sys
import time
import logging
import traceback
from collections.abc import Callable
from functools import partial

import numpy as np
import jax
#from jax import config
#config.update('jax_enable_x64', True)
from jax import numpy as jnp


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
    filename : string
        Path where to save figure.

    fig : matplotlib.figure.Figure, default=None
        Figure to save. Defaults to current figure.

    kwargs : dict
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


def rebin(array: np.ndarray | jax.Array, factor: int | tuple, axis: int | tuple=None, reduce: Callable=jnp.sum):
    """
    Rebin input ``array`` by factors ``factor`` along axes ``axis``,
    with reduction operation ``reduce``.
    """
    if axis is None:
        axis = list(range(array.ndim))
    if np.ndim(axis) == 0:
        axis = [axis]
    if np.ndim(factor) == 0:
        factor = [factor] * len(axis)
    factors = [1] * array.ndim
    for a, f in zip(axis, factor):
        factors[a] = f

    pairs = []
    for c, f in zip(array.shape, factors):
        pairs.append((c // f, f))

    flattened = [ll for p in pairs for ll in p]
    array = array.reshape(flattened)

    for i in range(len(factors)):
        array = reduce(array, axis=-1 * (i + 1))

    return array


# Generated by ChatGPT, https://fr.wikipedia.org/wiki/Polyn%C3%B4me_de_Legendre

def _legendre_0(x):
    return jnp.ones_like(x)

def _legendre_1(x):
    return 1. * x

def _legendre_2(x):
    return (3*x**2 - 1) / 2

def _legendre_3(x):
    return (5*x**3 - 3*x) / 2

def _legendre_4(x):
    return (35*x**4 - 30*x**2 + 3) / 8

def _legendre_5(x):
    return (63*x**5 - 70*x**3 + 15*x) / 8

def _legendre_6(x):
    return (231*x**6 - 315*x**4 + 105*x**2 - 5) / 16

def _legendre_7(x):
    return (429*x**7 - 693*x**5 + 315*x**3 - 35*x) / 16

def _legendre_8(x):
    return (6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35) / 128

def _legendre_9(x):
    return (12155*x**9 - 25740*x**7 + 18018*x**5 - 4620*x**3 + 315*x) / 128

def _legendre_10(x):
    return (46189*x**10 - 109395*x**8 + 90090*x**6 - 30030*x**4 + 3465*x**2 - 63) / 256

_ells = np.arange(11)
_registered_legendres = [globals()['_legendre_{:d}'.format(ell)] for ill, ell in enumerate(_ells)]


def get_legendre(ell):

    def f(mu):
        return jax.lax.switch(ell, _registered_legendres, mu)

    return f


from scipy import special


def Si_scipy(x):
    return jax.pure_callback(lambda x: special.sici(x)[0], x, x)


def get_spherical_jn_scipy(ell):
    return lambda x: jax.pure_callback(partial(special.spherical_jn, ell), x, x)


# Derivative of correlation function w.r.t. k-bins, precomputed with sympy; full, low-s or low-a limit
_registered_correlation_function_tophat_derivatives = {}
_registered_correlation_function_tophat_derivatives[0] = (lambda s, a: (-a * jnp.cos(a * s) / s + jnp.sin(a * s) / s**2) / (2 * jnp.pi**2 * s),
                                                          lambda s, a: -a**9 * s**6 / (90720 * jnp.pi**2) + a**7 * s**4 / (1680 * jnp.pi**2) - a**5 * s**2 / (60 * jnp.pi**2) + a**3 / (6 * jnp.pi**2))
_registered_correlation_function_tophat_derivatives[1] = (lambda s, a: ((-a * jnp.sin(a * s) - 2 * jnp.cos(a * s) / s) / s**2 + 2 / s**3) / (2 * jnp.pi**2),
                                                          lambda s, a: -a**10 * s**7 / (907200 * jnp.pi**2) + a**8 * s**5 / (13440 * jnp.pi**2) - a**6 * s**3 / (360 * jnp.pi**2) + a**4 * s / (24 * jnp.pi**2))
_registered_correlation_function_tophat_derivatives[2] = (lambda s, a: -(a * s * jnp.cos(a * s) - 4 * jnp.sin(a * s) + 3 * Si_scipy(a * s)) / (2 * jnp.pi**2 * s**3),
                                                          lambda s, a: -a**9 * s**6 / (136080 * jnp.pi**2) + a**7 * s**4 / (2940 * jnp.pi**2) - a**5 * s**2 / (150 * jnp.pi**2))
_registered_correlation_function_tophat_derivatives[3] = (lambda s, a: -(8 / s**3 + (a * s**2 * jnp.sin(a * s) + 7 * s * jnp.cos(a * s) - 15 * jnp.sin(a * s) / a) / s**4) / (2 * jnp.pi**2),
                                                          lambda s, a: -a**10 * s**7 / (1663200 * jnp.pi**2) + a**8 * s**5 / (30240 * jnp.pi**2) - a**6 * s**3 / (1260 * jnp.pi**2))
_registered_correlation_function_tophat_derivatives[4] = (lambda s, a: (-a * s**3 * jnp.cos(a * s) + 11 * s**2 * jnp.sin(a * s) + 15 * s**2 * Si_scipy(a * s) / 2 + 105 * s * jnp.cos(a * s) / (2 * a) - 105 * jnp.sin(a * s) / (2 * a**2)) / (2 * jnp.pi**2 * s**5),
                                                          lambda s, a: -a**9 * s**6 / (374220 * jnp.pi**2) + a**7 * s**4 / (13230 * jnp.pi**2))
_registered_correlation_function_tophat_derivatives[5] = (lambda s, a: (16 / s**3 + (-a * s**4 * jnp.sin(a * s) - 16 * s**3 * jnp.cos(a * s) + 105 * s**2 * jnp.sin(a * s) / a + 315 * s * jnp.cos(a * s) / a**2 - 315 * jnp.sin(a * s) / a**3) / s**6) / (2 * jnp.pi**2),
                                                          lambda s, a: -a**10 * s**7 / (5405400 * jnp.pi**2) + a**8 * s**5 / (166320 * jnp.pi**2))


_registered_bessel = {}
_registered_bessel[0] = (lambda x: jnp.sin(x) / x, lambda x: -x**6/5040 + x**4/120 - x**2/6 + 1)
_registered_bessel[2] = (lambda x: (-3*x*jnp.cos(x) + (3 - x**2)*jnp.sin(x))/x**3, lambda x: x**6/7560 - x**4/210 + x**2/15)
_registered_bessel[4] = (lambda x: (5*x*(2*x**2 - 21)*jnp.cos(x) + (x**4 - 45*x**2 + 105)*jnp.sin(x))/x**5, lambda x: -x**6/20790 + x**4/945)


def get_spherical_jn(ell):

    def jn(x):
        mask = x > 0.01
        bessel = _registered_bessel[ell]
        return jnp.where(mask, bessel[0](x), bessel[1](x))

    return jn


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return jnp.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]]) / 2.


class TophatPowerToCorrelation(object):

    def __init__(self, k: np.ndarray, seval: np.ndarray, ell=0, edges=False, method='exact'):
        if edges: edges = k
        else: edges = jnp.concatenate([k[:1], (k[1:] + k[:-1]) / 2., k[-1:]], axis=0)
        seval = seval[..., None]
        if method == 'rectangle':
            w = (-1)**(ell // 2) / (2 * np.pi**2) * (edges[1:]**3 - edges[:-1]**3) / 3.
            x = seval * (edges[1:] + edges[:-1]) / 2.
            mask = seval * edges > 0.01
            bessel = _registered_bessel[ell]
            self.w = w * jnp.where(mask, bessel[0](x), bessel[1](x))
        else:
            mask = seval * edges > 0.01
            tophat = _registered_correlation_function_tophat_derivatives[ell]
            self.w = jnp.diff(jnp.where(mask, tophat[0](seval, edges), tophat[1](seval, edges)), axis=-1)

    def __call__(self, fun: jax.Array):
        return jnp.sum(self.w * fun, axis=-1)


class TophatCorrelationToPower(object):

    def __init__(self, s: np.ndarray, keval: np.ndarray, ell=0, edges=False):
        if edges: edges = s
        else: edges = jnp.concatenate([s[:1], (s[1:] + s[:-1]) / 2., s[-1:]], axis=0)
        tophat = _registered_correlation_function_tophat_derivatives[ell]
        keval = keval[..., None]
        mask = keval * edges > 0.01
        self.w = (-1)**(ell // 2) * (2. * jnp.pi)**3 * jnp.diff(jnp.where(mask, tophat[0](keval, edges), tophat[1](keval, edges)), axis=-1)

    def __call__(self, fun: jax.Array):
        return jnp.sum(self.w * fun, axis=-1)


class BesselPowerToCorrelation(object):

    def __init__(self, k: np.ndarray, seval: np.ndarray, ell=0, edges=False, volume=None):
        if edges:
            edges = k
            if volume is None: volume = 4. / 3. * np.pi * (edges[1:]**3 - edges[:-1]**3)
            k = (edges[:-1] + edges[1:]) / 2.
        else:
            if volume is None: volume = 4. / 3. * np.pi * weights_trapz(k**3)
        self.w = (-1)**(ell // 2) / (2. * np.pi)**3 * volume * get_spherical_jn_scipy(ell)(seval[..., None] * k)

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


def _format_slice(sl, size):
    if sl is None: sl = slice(None)
    start, stop, step = sl.start, sl.stop, sl.step
    # To handle slice(0, None, 1)
    if start is None: start = 0
    if step is None: step = 1
    stop = min((size - start) // step * step, stop if stop is not None else size)
    #start, stop, step = sl.indices(len(self._x[iproj]))
    if step < 0:
        raise IndexError('positive slicing step only supported')
    return slice(start, stop, step)


def compute_real_gaunt(*ellms):
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
    gaunt = {}
    for ells in itertools.product((0, 2, 4), (0, 2), (0, 2)):
        for ms in itertools.product(*(list(range(-ell, ell + 1)) for ell in ells)):
            ellms = tuple(zip(ells, ms))
            tmp = float(compute_real_gaunt(*ellms))
            if tmp != 0.:
                gaunt[ellms] = tmp
    return gaunt


class FakeFigure(object):

    def __init__(self, axes):
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        self.axes = list(axes)

def plotter(*args, **kwargs):

    from functools import wraps

    def get_wrapper(func):
        """
        Return wrapper for plotting functions, that adds the following (optional) arguments to ``func``:

        Parameters
        ----------
        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        interactive : True or dict, default=False
            If not None, use interactive interface provided by ipywidgets. Interactive can be a dictionary
            with several entries:
                * ref_param : Use to display reference theory
        """
        @wraps(func)
        def wrapper(*args, fn=None, kw_save=None, show=False, fig=None, **kwargs):

            from matplotlib import pyplot as plt

            if fig is not None:

                if not isinstance(fig, plt.Figure):  # create fake figure that has axes
                    fig = FakeFigure(fig)

                elif not fig.axes:
                    fig.add_subplot(111)

                kwargs['fig'] = fig

            fig = func(*args, **kwargs)
            if fn is not None:
                savefig(fn, **(kw_save or {}))
            if show: plt.show()
            return fig

        return wrapper

    if kwargs or not args:
        if args:
            raise ValueError('unexpected args: {}, {}'.format(args, kwargs))
        return get_wrapper

    if len(args) != 1:
        raise ValueError('unexpected args: {}'.format(args))

    return get_wrapper(args[0])


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



class RegisteredStatistic(type):

    """Metaclass registering :class:`BinnedStatistic`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.__name__] = cls
        return cls


@jax.tree_util.register_pytree_node_class
class BinnedStatistic(metaclass=RegisteredStatistic):
    """
    Class representing binned statistic.

    Example
    -------
    >>> observable = BinnedStatistic(x=[np.linspace(0.01, 0.2, 10), np.linspace(0.01, 0.2, 10)], projs=[0, 2])
    >>> observable = observable.select(projs=2, xlim=(0., 0.15))

    Attributes
    ----------
    _x : tuple
        Coordinates.

    _edges : list, array, default=None
        Edges.

    _projs : tuple, default=None
        Projections.

    _value : tuple
        Data vector value.

    _weights : tuple
        Weights for rebinning.

    name : str
        Name.

    attrs : dict
        Other attributes.
    """
    _label_x = None
    _label_proj = None
    _label_value = None
    _data_fields = ['_x', '_edges', '_value', '_weights']
    _meta_fields = ['_projs', 'name', 'attrs', '_label_x', '_label_proj', '_label_value']
    _select_x_fields = ['_x', '_value', '_weights', '_edges']
    _select_proj_fields = ['_x', '_value', '_weights', '_edges', '_projs']
    _sum_fields = ['_value']
    _init_fields = {'x': '_x', 'edges': '_edges', 'projs': '_projs', 'value': '_value', 'weights': '_weights', 'name': 'name', 'attrs': 'attrs'}

    def __init__(self, x=None, edges=None, projs=None, value=None, weights=None, name=None, attrs=None):
        """
        Initialize observable array.
        'x' is an axis along which it may makes sense to rebin.

        Example
        -------
        >>> observable = ObservableArray(x=[np.linspace(0.01, 0.2, 10), np.linspace(0.01, 0.2, 10)], projs=[0, 2])
        >>> observable = observable.select(projs=2, xlim=(0., 0.15))

        Parameters
        ----------
        x : list, array, ObservableArray
            Coordinates.

        edges : list, array, default=None
            Edges.

        projs : list, default=None
            Projections.

        value : list, array
            Data vector value.

        weights : list, array
            Weights for rebinning.

        name : str, default=None
            Optionally, name.

        attrs : dict, default=None
            Optionally, attributes.
        """
        if isinstance(x, self.__class__):
            self.__dict__.update(x.__dict__)
            return
        state = {}
        state['name'] = str(name or self.__class__.__name__)
        state['attrs'] = dict(attrs or {})
        state['_projs'] = list(projs) if projs is not None else [None]
        if projs is None:
            x = [x]
            weights = [weights]
            if value is not None: value = [value]
        nprojs = len(state['_projs'])
        if x is None:
            if edges is not None:
                edges = [jnp.atleast_1d(xx) for xx in edges]
                state['_x'] = [(edges[:-1] + edges[1:]) / 2. for edges in edges]
            else:
                state['_x'] = [jnp.full(1, np.nan) for xx in range(nprojs)]
        else:
            state['_x'] = [jnp.atleast_1d(xx) for xx in x]
        if len(state['_x']) != nprojs:
            raise ValueError('x should be of same length as the number of projs = {:d}, found {:d}'.format(nprojs, len(state['_x'])))
        state['_edges'] = []
        if edges is None:
            for xx in state['_x']:
                if len(xx) >= 2:
                    tmp = (xx[:-1] + xx[1:]) / 2.
                    tmp = jnp.concatenate([jnp.array([tmp[0] - (xx[1] - xx[0])]), tmp, jnp.array([tmp[-1] + (xx[-1] - xx[-2])])])
                    tmp = jnp.column_stack([tmp[:-1], tmp[1:]])
                else:
                    tmp = jnp.full(xx.shape + (2,), np.nan)
                state['_edges'].append(tmp)
        else:
            for ix, xx in enumerate(edges):
                tmp = jnp.atleast_2d(xx)
                eshape = state['_x'][ix].shape + (2,)
                if tmp.shape != eshape:
                    raise ValueError('edges should be of shape {} (since x shape is {}), found {}'.format(eshape, eshape[:-1], tmp.shape))
                state['_edges'].append(tmp)

        shape = tuple(len(xx) for xx in state['_x'])
        if weights is None:
            weights = [None] * len(state['_x'])
        state['_weights'] = [jnp.atleast_1d(ww) if ww is not None else jnp.ones(len(xx), dtype=float) for xx, ww in zip(state['_x'], weights)]
        wshape = tuple(len(ww) for ww in state['_weights'])
        if wshape != shape:
            raise ValueError('weights should be of same length as x = {}, found = {}'.format(shape, wshape))
        state['_value'] = value
        if value is None:
            state['_value'] = [jnp.full(len(xx), np.nan) for xx in state['_x']]
        else:
            if getattr(value, 'ndim', 0) == 1:
                value = jnp.array_split(value, np.cumsum(shape)[:-1])
            state['_value'] = [jnp.atleast_1d(vv) for vv in value]
            vshape = tuple(len(vv) for vv in state['_value'])
            if vshape != shape:
                raise ValueError('value should be of same length as x = {}, found = {}'.format(shape, vshape))
        for name, value in state.items():
            if isinstance(value, list):
                state[name] = tuple(value)
        self.__dict__.update(state)

    def _clone_as_binned_statistic(self, **kwargs):
        init_kw = {'x': self.x(), 'value': self.value, 'weights': self.weights(), 'edges': self.edges()}
        init_kw |= {name: getattr(self, name) for name in ['projs', 'name', 'attrs']}
        init_kw.update(kwargs)
        return BinnedStatistic(**init_kw)

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        init_kw = {name: getattr(self, rename) for name, rename in self._init_fields.items()}
        not_in_init = [name for name in kwargs if name not in init_kw]
        if not_in_init:
            if type(self) == BinnedStatistic:
                raise ValueError('arguments {} cannot be passed to {}'.format(not_in_init, self.__init__))
            else:
                return self._clone_as_binned_statistic(**kwargs)  # try preceding clone
        init_kw.update(kwargs)
        return self.__class__(**init_kw)

    def copy(self):
        import copy
        new = self.__class__.__new__(self.__class__)
        state = {}
        for name in self._data_fields + self._meta_fields:
            value = getattr(self, name)
            if name in self._select_proj_fields:
                value = tuple(copy.copy(xx) for xx in value)
            else:
                value = copy.copy(value)
            state[name] = value
        new.__dict__.update(state)
        return new

    def __getstate__(self):
        state = {name: getattr(self, name) for name in self._data_fields + self._meta_fields}
        state['__class__name__'] = self.__class__.__name__
        return state

    def __setstate__(self, state):
        state = dict(state)
        state.pop('__class__name__', None)
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        cls = BinnedStatistic._registry[state.get('__class__name__', cls.__name__)]
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        """Save object."""
        state = self.__getstate__()
        #self.log_info('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        np.save(filename, state, allow_pickle=True)

    @classmethod
    def load(cls, filename):
        """Load object."""
        filename = str(filename)
        state = np.load(filename, allow_pickle=True)
        #cls.log_info('Loading {}.'.format(filename))
        state = state[()]
        return cls.from_state(state)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self._data_fields), {name: getattr(self, name) for name in self._meta_fields}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__setstate__(aux_data | {name: value for name, value in zip(cls._data_fields, children)})
        return new

    @property
    def size(self):
        """Size of the data vector."""
        return sum(len(v) for v in self._value)

    @property
    def value(self):
        return self._value

    @classmethod
    def sum(cls, others, weights=None):
        if weights is None:
            weights = [1] * len(others)
        new = None
        for other, weight in zip(others, weights):
            if new is None:
                new = other.copy()
            else:
                for name in new._sum_fields:
                    value, other_value = getattr(new, name), getattr(other, name)
                    if name in new._select_proj_fields:
                        value = tuple(vv + weight * ov for vv, ov in zip(value, other_value))
                    else:
                        value = value + weight * other_value
                    setattr(new, name, value)
        return new

    @classmethod
    def concatenate(cls, others):
        # No check performed
        new = None
        for other in others:
            if new is None:
                new = other.copy()
                values = {name: [] for name in new._select_x_fields}
            for name in values:
                values[name].append(getattr(other, name))
        for name in new._select_x_fields:
            value = values[name]
            if name in new._select_proj_fields:
                indices = [slice(None)] * len(value)
                value = tuple(jnp.concatenate([vv[iv][idx] for vv, idx in zip(value, indices)], axis=0) for iv in range(len(value[0])))
            else:
                value = jnp.concatenate(value, axis=0)
            setattr(new, name, value)
        return new

    def _index(self, xlim=None, projs=Ellipsis, method='mid', concatenate=True):
        """
        Return indices for given input x-limits and projs.

        Parameters
        ----------
        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        projs : list, default=None
            List of projections to return indices for.
            Defaults to :attr:`projs`.

        concatenate : bool, default=True
            If ``False``, return list of indices, for each input projection.

        Returns
        -------
        indices : array, list
        """
        iprojs = self._index_projs(projs)
        if not isinstance(iprojs, list): iprojs = [iprojs]
        toret = []
        for iproj in iprojs:
            if method == 'mid': xx = (self._edges[iproj][..., 0] + self._edges[iproj][..., 1]) / 2.
            else: xx = self._x[iproj]
            if xlim is not None:
                tmp = (xx >= xlim[0]) & (xx <= xlim[1])
                tmp = np.all(tmp, axis=tuple(range(1, tmp.ndim)))
            else:
                tmp = np.ones(xx.shape[0], dtype='?')
                #print(xlim, self._x[iproj].shape, self._edges[iproj].shape)
            tmp = np.flatnonzero(tmp)
            if concatenate: tmp += sum(len(xx) for xx in self._x[:iproj])
            toret.append(tmp)
        if concatenate:
            return np.concatenate(toret, axis=0)
        #if isscalar:
        #    return toret[0]
        return toret

    def _index_projs(self, projs=Ellipsis):
        """Return projs indices."""
        if projs is Ellipsis:
            return list(range(len(self._projs)))
        isscalar = not isinstance(projs, list)
        if isscalar: projs = [projs]
        toret = []
        for proj in projs:
            try: iproj = self._projs.index(proj)
            except (ValueError, IndexError):
                raise ValueError('{} could not be found in {}'.format(proj, self._projs))
            toret.append(iproj)
        if isscalar:
            return toret[0]
        return toret

    def select(self, xlim=None, projs=Ellipsis, select_projs=False, method='mid'):
        """
        Apply x-cuts for given projections.

        Parameters
        ----------
        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        rebin : int, default=1
            Optionally, rebinning factor (after ``xlim`` cut).

        projs : list, default=None
            List of projections to apply ``xlim`` and ``rebin`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : ObservableArray
        """
        iprojs = self._index_projs(projs)
        if not isinstance(iprojs, list): iprojs = [iprojs]
        state = {name: list(getattr(self, name)) for name in self._select_x_fields}
        for iproj in iprojs:
            index = self._index(xlim=xlim, projs=[self._projs[iproj]], method=method, concatenate=False)[0]
            for name in state:
                tmpidx = index
                # continuous (no rebinning), should be all fine even for edges
                state[name][iproj] = getattr(self, name)[iproj][tmpidx]
        state = {name: getattr(self, name) for name in self._data_fields + self._meta_fields} | {name: tuple(value) for name, value in state.items()}
        projs = self._projs
        if select_projs:
            for name in self._select_proj_fields:
                state[name] = tuple(state[name][iproj] for iproj in iprojs)
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(state)
        return new

    def _slice_matrix(self, edges=None, projs=Ellipsis, weighted=True, normalize=True):
        # Return, for a given slice, the corresponding matrix to apply to the data arrays.
        toret = []
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: iprojs = [iprojs]
        if edges is None:
            edges = slice(None)
        if isinstance(edges, BinnedStatistic):
            edges = list(edges._edges)
        toret = []
        for iproj in iprojs:
            if isinstance(edges, list): iedges = edges[iproj]
            else: iedges = edges
            if isinstance(iedges, slice):
                sl = _format_slice(iedges, len(self._x[iproj]))
                sl1 = self._edges[iproj][sl, ..., 0]
                sl2 = self._edges[iproj][sl.start + sl.step - 1:sl.stop + sl.step - 1:sl.step, ..., 1]
                iedges = jnp.concatenate([sl1[..., None], sl2[..., None]], axis=-1)
            matrix = np.zeros((len(iedges), len(self._weights[iproj])), dtype=float)
            for iedge, edge in enumerate(iedges):  # iterate on data vector size
                mask = (self._edges[iproj][..., 0] >= edge[..., 0]) & (self._edges[iproj][..., 1] <= edge[..., 1])  # (size, ndim, 2)
                if mask.ndim > 2: mask = mask.all(axis=1)  # (N>1)-dim vector
                if weighted:
                    matrix[iedge, mask] = self._weights[iproj][mask]
                else:
                    matrix[iedge, mask] = 1.
            if normalize: matrix /= jnp.sum(matrix, axis=-1)[:, None]
            toret.append(matrix)
        if isscalar:
            return toret[0]
        return toret

    def slice(self, edges=None, projs=Ellipsis, select_projs=False):
        """
        Apply selections to the data, slicing for given projections.

        Parameters
        ----------
        slice : slice, default=None
            Slicing to apply, defaults to ``slice(None)``.

        projs : list, default=None
            List of projections to apply ``slice`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : ObservableArray
        """
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: iprojs = [iprojs]
        state = {name: list(getattr(self, name)) for name in self._select_x_fields}
        if edges is None:
            edges = slice(None)
        if isinstance(edges, BinnedStatistic):
            edges = list(edges._edges)
        for iproj in iprojs:
            proj = self._projs[iproj]
            if isinstance(edges, list): iedges = edges[iproj]
            else: iedges = edges
            matrix = self._slice_matrix(iedges, projs=proj, weighted=False, normalize=False)
            nwmatrix = self._slice_matrix(iedges, projs=proj, weighted=True, normalize=True)
            for name in state:
                if name == '_edges':
                    _edges = []
                    for row in nwmatrix:
                        edge = state[name][iproj][row != 0]
                        _edges.append(jnp.concatenate([edge[0, ..., 0][None, ..., None], edge[-1, ..., 1][None, ..., None]], axis=-1))
                    state[name][iproj] = jnp.concatenate(_edges, axis=0)
                elif name == '_weights':
                    state[name][iproj] = matrix.dot(state[name][iproj])
                else:
                    state[name][iproj] = nwmatrix.dot(state[name][iproj])
        state = {name: getattr(self, name) for name in self._data_fields + self._meta_fields} | {name: tuple(value) for name, value in state.items()}
        projs = self._projs
        if select_projs:
            for name in self._select_proj_fields:
                state[name] = tuple(state[name][iproj] for iproj in iprojs)
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(state)
        return new

    def view(self, xlim=None, projs=Ellipsis, method='mid', return_type='nparray'):
        """
        Return observable array for input x-limits and projections.

        Parameters
        ----------
        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        projs : list, default=None
            Restrict to these projections.
            Defaults to :attr:`projs`.

        return_type : str, default='nparray'
            If 'nparray', return numpy array :attr:`flatvalue`.
            Else, return a new :class:`ObservableArray`, restricting to ``xlim`` and ``projs``.

        Returns
        -------
        new : array, ObservableArray
        """
        if (xlim, return_type) == (None, 'nparray'):  # fast way
            iprojs = self._index_projs(projs=projs)
            isscalar = not isinstance(iprojs, list)
            value = self.value
            if isscalar: return value[iprojs].copy()
            return jnp.concatenate([value[iproj] for iproj in iprojs], axis=0)
        toret = self.select(xlim=xlim, projs=projs, select_projs=True, method=method)
        if return_type is None:
            return toret
        return jnp.concatenate(toret.value, axis=0)

    @property
    def projs(self):
        return self._projs

    def x(self, projs=Ellipsis):
        """x-coordinates (optionally restricted to input projs)."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return self._x[iprojs]
        return [self._x[iproj] for iproj in iprojs]

    def xavg(self, projs=Ellipsis, method='mid'):
        """x-coordinates (optionally restricted to input projs)."""
        def get_x(iproj):
            if method == 'mid':
                return (self._edges[iproj][:-1] + self._edges[iproj][1:]) / 2.
            return self._x[iproj]
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return get_x(iprojs)
        return [get_x(iproj) for iproj in iprojs]

    def edges(self, projs=Ellipsis):
        """edges (optionally restricted to input projs)."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return self._edges[iprojs]
        return [self._edges[iproj] for iproj in iprojs]

    def weights(self, projs=Ellipsis):
        """Weights (optionally restricted to input projs)."""
        iprojs = self._index_projs(projs)
        isscalar = not isinstance(iprojs, list)
        if isscalar: return self._weights[iprojs]
        return [self._weights[iproj] for iproj in iprojs]

    def __repr__(self):
        """Return string representation of observable data."""
        return '{}(projs={}, size={:d})'.format(self.name, self._projs, self.size)

    def __array__(self, *args, **kwargs):
        return np.asarray(self.view(), *args, **kwargs)

    def _get_label_proj(self, proj):
        label = str(proj)
        if self._label_proj:
            label = f'{self._label_proj} = {label}'
        return label

    @plotter
    def plot(self, fig=None, **kwargs):
        """
        Plot data.

        Parameters
        ----------
        xlabel : str, default=None
            Optionally, label for the x-axis.

        ylabel : str, default=None
            Optionally, label for the y-axis.

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
        fig : matplotlib.figure.Figure
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for iproj, proj in enumerate(self._projs):
            ax.plot(self.x(projs=proj), self.view(projs=proj), color='C{:d}'.format(iproj), linestyle='-', label=self._get_label_proj(proj))
        ax.grid(True)
        if self._projs: ax.legend()
        ax.set_ylabel(kwargs.get('xlabel', self._label_x))
        ax.set_xlabel(kwargs.get('ylabel', self._label_value))
        return fig


@jax.tree_util.register_pytree_node_class
class WindowMatrix(object):
    """
    Class representing a window matrix.

    Attributes
    ----------
    _value : array
        Window matrix 2D array.

    _theory : BinnedStatistic
        Theory corresponding to the window matrix.

    _observable : BinnedStatistic
        (Mean) observable corresponding to the window matrix.

    attrs : dict
        Other attributes.
    """
    _data_fields = ['_observable', '_theory', '_value']
    _meta_fields = ['attrs']

    def __init__(self, observable, theory, value, attrs=None):
        """
        Initialize :class:`WindowMatrix`.

        Parameters
        ----------
        observable : BinnedStatistic
            (Mean) observable corresponding to the window matrix.

        theory : BinnedStatistic
            Theory corresponding to the window matrix.

        value : array
            Window matrix 2D array.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        if isinstance(value, self.__class__):
            self.__dict__.update(value.__dict__)
            return
        state = {}
        state['_value'] = np.asarray(value)
        shape = state['_value'].shape
        for axis, (name, value) in enumerate({'observable': observable, 'theory': theory}.items()):
            _name = '_' + name
            state[_name] = value if isinstance(value, BinnedStatistic) else BinnedStatistic(**value)
            size = state[_name].size
            if size != shape[axis]:
                raise ValueError('size = {:d} of input {} must match input matrix shape = {}'.format(size, name, shape[axis]))
        state['attrs'] = dict(attrs or {})
        self.__dict__.update(state)

    @property
    def observable(self):
        """(Mean) observable."""
        return self._observable

    @property
    def theory(self):
        """Theory."""
        return self._theory

    def _axis_index(self, axis):
        observable_names = [0, 'o', 'observable']
        theory_names = [1, 't', 'theory']
        if axis in observable_names:
            return 0, 'observable', self._observable
        if axis in theory_names:
            return 1, 'theory', self._theory
        raise ValueError('axis must be in {} or {}'.format(observable_names, theory_names))

    def _slice_matrix(self, edges, axis='o', projs=Ellipsis, normalize=False):
        # Return, for a given slice, the corresponding matrix to apply to the data arrays.
        axis, _, observable = self._axis_index(axis=axis)
        if projs is not Ellipsis and not isinstance(projs, list): projs = [projs]
        proj_indices = observable._index_projs(projs)
        list_projs = list(observable._projs)
        if edges is None:
            edges = slice(None)
        if isinstance(edges, BinnedStatistic):
            edges = list(edges._edges)
        list_edges = []
        iiproj = 0
        for iproj, proj in enumerate(list_projs):
            if iproj in proj_indices:
                if isinstance(edges, list): iedges = edges[iiproj]
                else: iedges = edges
                iiproj += 1
            else:
                iedges = slice(None)
            list_edges.append(iedges)
        matrix = observable._slice_matrix(list_edges, projs=list_projs, normalize=normalize)
        import scipy
        return scipy.linalg.block_diag(*matrix)

    def slice(self, edges=None, axis='o', projs=Ellipsis, select_projs=False):
        """
        Apply selections to the window matrix, slicing for given axis and projections.

        Parameters
        ----------
        slice : slice, default=None
            Slicing to apply, defaults to ``slice(None)``.

        axis : str
            Axis to apply ``slice`` to.
            One of ('o', 'observable') or ('t', 'theory').

        projs : list, default=None
            List of projections to apply ``slice`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : WindowMatrix
        """
        axis, name, observable = self._axis_index(axis=axis)
        observable = observable.slice(edges, projs=projs)
        matrix = self._slice_matrix(edges, axis=axis, projs=projs, normalize=axis == 0)
        value = matrix.dot(self._value) if axis == 0 else self._value.dot(matrix.T)
        if select_projs:
            observable = observable.select(projs=projs, select_projs=True)
            index = self._index(projs=projs, concatenate=True)
            value = np.take(value, index, axis=axis)
        new = self.clone(value=value, attrs=self.attrs, **{name: observable})
        return new

    @classmethod
    def sum(cls, others, axis='o', weights=None):
        if weights is None:
            weights = [1] * len(others)
        new = None
        for other, weight in zip(others, weights):
            if new is None:
                new = other.copy()
                axis, name, observable = new._axis_index(axis=axis)
                setattr(new, '_' + name, observable.sum([other._axis_index(axis=axis)[-1] for other in others], weights=weights))
            else:
                new._value = new._value + weight * other._value
        return new

    @classmethod
    def concatenate(cls, others, axis='o'):
        # No check performed
        new = None
        values = []
        for other in others:
            if new is None:
                new = other.copy()
                axis, name, observable = new._axis_index(axis=axis)
                setattr(new, '_' + name, observable.concatenate([other._axis_index(axis=axis)[-1] for other in others]))
            values.append([other.slice(axis=axis, projs=proj, select_projs=True)._value for proj in observable.projs])
        new._value = np.concatenate([np.concatenate([value[iv] for value in values], axis=axis) for iv in range(len(values[0]))], axis=axis)
        return new

    def _index(self, axis='o', xlim=None, projs=Ellipsis, method='mid', concatenate=True):
        """
        Return indices for given x-limits and projs.

        Parameters
        ----------
        axis : str
            Axis to return indices for.
            One of ('o', 'observable') or ('t', 'theory').

        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        projs : list, default=None
            List of projections to return indices for.
            Defaults to :attr:`projs`.

        concatenate : bool, default=True
            If ``False``, return list of indices, for each input observable and projection.

        Returns
        -------
        indices : array, list
        """
        axis, _, observable = self._axis_index(axis=axis)
        return observable._index(xlim=xlim, projs=projs, method=method, concatenate=concatenate)

    def select(self, xlim=None, axis='o', projs=Ellipsis, select_projs=False, method='mid'):
        """
        Apply selections for given observables and projections.

        Parameters
        ----------
        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        rebin : int, default=1
            Optionally, rebinning factor (after ``xlim`` cut).

        axis : str
            Axis to apply the selection to.
            One of ('o', 'observable') or ('t', 'theory').

        projs : list, default=None
            List of projections to apply ``xlim`` and ``rebin`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : WindowMatrix
        """
        axis, name, observable = self._axis_index(axis=axis)
        if projs is not Ellipsis and not isinstance(projs, list): projs = [projs]
        #print(observable_indices, projs)
        proj_indices = observable._index_projs(projs)
        index = np.concatenate([observable._index(xlim=xlim if iproj in proj_indices else None, projs=proj, method=method, concatenate=True) for iproj, proj in enumerate(observable._projs)])
        observable = observable.select(xlim=xlim, projs=projs, method=method)
        new = self.clone(value=np.take(self._value, index, axis=axis), **{name: observable}, attrs=self.attrs)
        if select_projs:
            index = self._index(axis=axis, projs=projs, concatenate=True)
            observable = observable.select(projs=projs, select_projs=True)
            new = self.clone(value=np.take(new._value, index, axis=axis), **{name: observable}, attrs=self.attrs)
        return new

    def view(self, return_type='nparray'):
        """
        Return window matrix.

        Parameters
        ----------
        return_type : str, default='nparray'
            If 'nparray', return numpy array :attr:`value`.
            Else, return a new :class:`ObservableCovariance`, restricting to ``xlim`` and ``projs``.

        Returns
        -------
        new : array, WindowMatrix
        """
        toret = self.slice()
        if return_type is None:
            return toret
        return toret._value

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = {name[1:] if name.startswith('_') else name: getattr(self, name) for name in self._data_fields + self._meta_fields}  # remove front _
        state.update(kwargs)
        return self.__class__(**state)

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        state = {name: getattr(self, name).copy() for name in self._data_fields + self._meta_fields}
        new.__dict__.update(state)
        return new

    def __getstate__(self):
        state = {name: getattr(self, name) for name in self._data_fields + self._meta_fields}
        for name in ['_theory', '_observable']:
            value = getattr(self, name)
            state[name] = value.__getstate__()
        return state

    def __setstate__(self, state):
        for name in ['_theory', '_observable']:
            state[name] = BinnedStatistic.from_state(state[name])
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        """Save object."""
        state = self.__getstate__()
        #self.log_info('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        np.save(filename, state, allow_pickle=True)

    @classmethod
    def load(cls, filename):
        """Load object."""
        filename = str(filename)
        state = np.load(filename, allow_pickle=True)
        #cls.log_info('Loading {}.'.format(filename))
        state = state[()]
        return cls.from_state(state)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self._data_fields), {name: getattr(self, name) for name in self._meta_fields}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__setstate__(aux_data | {name: value for name, value in zip(cls._data_fields, children)})
        return new

    def __repr__(self):
        """Return string representation of window matrix."""
        return '{}({}, {})'.format(self.__class__.__name__, self._observable, self._theory)

    def __array__(self, *args, **kwargs):
        return np.asarray(self._value, *args, **kwargs)

    @property
    def shape(self):
        """Return window matrix shape."""
        return self._value.shape

    def dot(self, theory, zpt=True, return_type='nparray'):
        """Apply window matrix to input theory."""
        if isinstance(theory, BinnedStatistic):
            theory = theory.view()
        theory = jnp.ravel(theory)
        if zpt:
            toret = self.observable.view() + self._value.dot(theory - self.theory.view())
        else:
            toret = self._value.dot(theory)
        if return_type == 'nparray':
            return toret
        return self.observable.clone(value=toret)

    @plotter
    def plot(self, split_projs=True, **kwargs):
        """
        Plot covariance matrix.

        Parameters
        ----------
        barlabel : str, default=None
            Optionally, label for the color bar.

        figsize : int, tuple, default=None
            Optionally, figure size.

        norm : matplotlib.colors.Normalize, default=None
            Scales the matrix to the canonical colormap range [0, 1] for mapping to colors.
            By default, the matrix range is mapped to the color bar range using linear scaling.

        labelsize : int, default=None
            Optionally, size for labels.

        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least ``len(self._observables) * len(self._observables)`` axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        def _get(axis):
            xlabels, plabels, x, indices = [], [], [], []
            axis, _, observable = self._axis_index(axis=axis)
            for proj in observable._projs:
                indices.append(self._index(projs=proj, axis=axis, concatenate=True))
                x.append(observable.x(projs=proj))
                xlabels.append(observable._label_x)
                plabels.append(observable._get_label_proj(proj))
            return xlabels, plabels, x, indices

        xlabels, plabels, x, indices = zip(*[_get(axis) for axis in [0, 1]])

        if not split_projs:
            indices = [np.concatenate(index) for index in indices]
            x = [[np.concatenate(xx, axis=0) for xx in x]]
            xlabels = [label[0] for label in xlabels]
            plabels = []
        for ilabel, label in enumerate(xlabels):
            kwargs.setdefault(f'xlabel{ilabel + 1:d}', label)
        for ilabel, label in enumerate(plabels):
            kwargs.setdefault(f'label{ilabel + 1:d}', label)
        mat = [[self._value[np.ix_(index1, index2)] for index2 in indices[1]] for index1 in indices[0]]
        return plot_matrix(mat, x1=x[0], x2=x[1], **kwargs)

    def interp(self, new, axis='o', extrap=False):
        """
        Interpolate window matrix.

        Parameters
        ----------
        new : BinnedStatistic
            New observable (axis='o') or theory (axis='t') vector.

        axis : str
            Axis to apply the selection to.
            One of ('o', 'observable') or ('t', 'theory').

        Returns
        -------
        new : WindowMatrix
        """
        axis, name, _ = self._axis_index(axis=axis)
        from scipy import interpolate
        shape = list(self._value.shape)
        shape[axis] = new.size
        value = np.zeros_like(self._value, shape=shape)
        for po in self.observable.projs:
            for pt in self.theory.projs:
                xo, xt = self.observable.x(projs=po), self.theory.x(projs=pt)
                ixo, ixt = self.observable._index(projs=po, concatenate=True), self.theory._index(projs=pt, concatenate=True)
                xxo, xxt = np.meshgrid(xo, xt, indexing='ij')
                # x is observable.x, y is theory.x - x, z is wmat
                x = [xxo, xxt][axis]
                p = [po, pt][axis]
                y = xxt - xxo
                v = self._value[np.ix_(ixo, ixt)]
                x, y, z = x.ravel(), y.ravel(), v.ravel()
                del xxo, xxt
                ixnew = new._index(projs=p, concatenate=True)
                xnew = new.x(projs=p)
                if extrap:
                    xold = [xo, xt][axis]
                    idxs = []
                    if xnew[0] < xold[0]:
                        idxs.append(0)
                    if xnew[-1] > xold[-1]:
                        idxs.append(-1)
                    for idx in idxs:
                        if axis == 0:  # observable
                            x = np.append(x, np.full_like(xt, fill_value=xnew[idx]))
                            y = np.append(y, xt - xo[idx])
                            z = np.append(z, v[idx, :])
                        else:  # theory
                            x = np.append(x, np.full_like(xo, fill_value=xnew[idx]))
                            y = np.append(y, xt[idx] - xo)
                            z = np.append(z, v[:, idx])
                interp = interpolate.LinearNDInterpolator(np.column_stack([x, y]), z, fill_value=0., rescale=False)
                for ix, x in zip(ixnew, xnew):
                    if axis == 0:  # observable
                        xo = x
                        idx = (ix, ixt)
                    else:  # theory
                        xt = x
                        idx = (ixo, ix)
                    value[idx] += interp(x, xt - xo)
        return self.clone(**{name: new, 'value': value})


@plotter
def plot_matrix(matrix, x1=None, x2=None, xlabel1=None, xlabel2=None, barlabel=None, label1=None, label2=None,
                figsize=None, norm=None, labelsize=None, fig=None):

    """
    Plot matrix.

    Parameters
    ----------
    matrix : array, list of lists of arrays
        Matrix, organized per-block.

    x1 : array, list of arrays, default=None
        Optionally, coordinates corresponding to the first axis of the matrix, organized per-block.

    x2 : array, list of arrays, default=None
        Optionally, coordinates corresponding to the second axis of the matrix, organized per-block.

    xlabel1 : str, list of str, default=None
        Optionally, label(s) corresponding to the first axis of the matrix, organized per-block.

    xlabel2 : str, list of str, default=None
        Optionally, label(s) corresponding to the second axis of the matrix, organized per-block.

    barlabel : str, default=None
        Optionally, label for the color bar.

    label1 : str, list of str, default=None
        Optionally, label(s) for the first observable(s) in the matrix, organized per-block.

    label2 : str, list of str, default=None
        Optionally, label(s) for the second observable(s) in the matrix, organized per-block.

    figsize : int, tuple, default=None
        Optionally, figure size.

    norm : matplotlib.colors.Normalize, default=None
        Scales the matrix to the canonical colormap range [0, 1] for mapping to colors.
        By default, the matrix range is mapped to the color bar range using linear scaling.

    labelsize : int, default=None
        Optionally, size for labels.

    fig : matplotlib.figure.Figure, default=None
        Optionally, a figure with at least as many axes as blocks in ``covariance``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize

    def _is_sequence(item):
        return isinstance(item, (tuple, list))

    if not _is_sequence(matrix[0]) or not np.size(matrix[0][0]):
        matrix = [[matrix]]
    mat = matrix
    size1, size2 = [row[0].shape[0] for row in mat], [col.shape[1] for col in mat[0]]

    def _make_list(x, size):
        if not _is_sequence(x):
            x = [x] * size
        return list(x)

    if x2 is None: x2 = x1
    x1, x2 = [_make_list(x, len(size)) for x, size in zip([x1, x2], [size1, size2])]
    if xlabel2 is None: xlabel2 = xlabel1
    xlabel1, xlabel2 = [_make_list(x, len(size)) for x, size in zip([xlabel1, xlabel2], [size1, size2])]
    if label2 is None: label2 = label1
    label1, label2 = [_make_list(x, len(size)) for x, size in zip([label1, label2], [size1, size2])]

    vmin, vmax = min(item.min() for row in mat for item in row), max(item.max() for row in mat for item in row)
    norm = norm or Normalize(vmin=vmin, vmax=vmax)
    nrows, ncols = [len(x) for x in [size2, size1]]
    if fig is None:
        figsize = figsize or tuple(max(n * 3, 6) for n in [ncols, nrows])
        if np.ndim(figsize) == 0: figsize = (figsize,) * 2
        xextend = 0.8
        fig, lax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,
                                figsize=(figsize[0] / xextend, figsize[1]),
                                gridspec_kw={'height_ratios': size2[::-1], 'width_ratios': size1},
                                squeeze=False)
        lax = lax.ravel()
        wspace = hspace = 0.18
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
    else:
        lax = fig.axes
    lax = np.array(lax).reshape((nrows, ncols))
    cmap = plt.get_cmap('jet_r')
    for i in range(ncols):
        for j in range(nrows):
            ax = lax[nrows - j - 1][i]
            xx1, xx2 = x1[i], x2[j]
            if x1[i] is None: xx1 = 1 + np.arange(mat[i][j].shape[0])
            if x2[j] is None: xx2 = 1 + np.arange(mat[i][j].shape[1])
            xx1 = np.append(xx1, xx1[-1] + (1. if xx1.size == 1 else xx1[-1] - xx1[-2]))
            xx2 = np.append(xx2, xx2[-1] + (1. if xx2.size == 1 else xx2[-1] - xx2[-2]))
            mesh = ax.pcolormesh(xx1, xx2, mat[i][j].T, norm=norm, cmap=cmap)
            if i > 0 or x1[i] is None: ax.yaxis.set_visible(False)
            if j == 0 and xlabel1[i]: ax.set_xlabel(xlabel1[i], fontsize=labelsize)
            if j > 0 or x2[j] is None: ax.xaxis.set_visible(False)
            if i == 0 and xlabel2[j]: ax.set_ylabel(xlabel2[j], fontsize=labelsize)
            ax.tick_params()
            if label1[i] is not None or label2[j] is not None:
                text = '{}\nx {}'.format(label1[i], label2[j])
                ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top',\
                        transform=ax.transAxes, color='black')

    fig.subplots_adjust(right=xextend)
    cbar_ax = fig.add_axes([xextend + 0.05, 0.15, 0.03, 0.7])
    cbar_ax.tick_params()
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    if barlabel: cbar.set_label(barlabel, rotation=90)
    return fig


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