import os
import sys
import time
import logging
import traceback

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


def rebin(array: np.ndarray | jax.Array, factor: int | tuple, axis: int | tuple=None, reduction: callable=jnp.sum):
    """
    Rebin input ``array`` by factors ``factor`` along axes ``axis``,
    with reduction operation ``reduction``.
    """
    if axis is None:
        axis = list(range(array.ndim))
    if not np.ndim(axis) == 0:
        axis = [axis]
    if not np.ndim(factor) == 0:
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
        array = reduction(array, axis=-1 * (i + 1))

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


def legendre(ell):
    return globals()['_legendre_{:d}'.format(ell)]
