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


@contextmanager
def set_env(**environ):
    """
    Temporarily set environment variables inside the context.

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

def _format_slice(sl, size):
    if sl is None: sl = slice(None)
    start, stop, step = sl.start, sl.stop, sl.step
    # To handle slice(0, None, 1)
    if start is None: start = 0
    if step is None: step = 1
    if stop is None: stop = size
    if stop < 0: stop = stop + size
    stop = min((size - start) // step * step, stop)
    #start, stop, step = sl.indices(len(self._x[iproj]))
    if step < 0:
        raise IndexError('positive slicing step only supported')
    return slice(start, stop, step)


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
    _data_fields = ['_x', '_edges', '_value', '_weights', '_norm']
    _meta_fields = ['_projs', 'name', 'attrs', '_label_x', '_label_proj', '_label_value']
    _select_x_fields = ['_x', '_value', '_weights', '_edges']
    _select_proj_fields = ['_x', '_value', '_weights', '_edges', '_projs']
    _sum_fields = ['_value', '_norm']
    _init_fields = {'x': '_x', 'edges': '_edges', 'projs': '_projs', 'value': '_value', 'weights': '_weights', 'name': 'name', 'attrs': 'attrs'}

    def __init__(self, x=None, edges=None, projs=None, value=None, norm=1., weights=None, name=None, attrs=None):
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
                edges = [jnp.atleast_2d(xx) for xx in edges]
                state['_x'] = [jnp.mean(edges, axis=-1) for edges in edges]
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
        # Turn everything into tuples
        state['_norm'] = jnp.asarray(norm)
        for name, value in state.items():
            if isinstance(value, list):
                state[name] = tuple(value)
        self.__dict__.update(state)

    @property
    def norm(self):
        """Correlation function normalization."""
        return self._norm

    def _clone_as_binned_statistic(self, **kwargs):
        init_kw = {'x': self.x(), 'value': self.view(), 'weights': self.weights(), 'edges': self.edges()}
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
        return sum(len(edges) for edges in self._edges)

    @property
    def value(self):
        return tuple(vv / self._norm for vv in self._value)

    @classmethod
    def sum(cls, others, weights=None):
        if weights is None:
            weights = np.ones(len(others))
        weights = np.asarray(weights)
        if weights.ndim == 0: weights = np.full(len(others), weights)
        assert weights.shape == (len(others),), 'weights must be of same shape as number of input statistics'
        new = None
        for other, weight in zip(others, weights):
            if new is None:
                new = other.copy()
                states = {name: [[] for i in range(len(getattr(new, name)))] if name in new._select_proj_fields else [] for name in new._sum_fields}
            for name in new._sum_fields:
                value = getattr(other, name)
                if name in new._select_proj_fields:
                    for iproj, value in enumerate(value):
                        states[name][iproj].append(weight * value)
                else:
                    states[name].append(weight * value)
        for name in new._sum_fields:
            if name in new._select_proj_fields:
                value = tuple(sum(values) for values in states[name])
            else:
                value = sum(states[name])
            new.__dict__[name] = value
        return new

    def __add__(self, other):
        return self.sum([self, other])

    def __radd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0: return self.copy()
        return self.__add__(other)

    @classmethod
    def mean(cls, others):
        return cls.sum(others, weights=1. / len(others))

    @classmethod
    def cov(cls, others, return_type=None):
        observables = None
        values = []
        for other in others:
            islist = isinstance(other, (tuple, list))
            if observables is None:
                if islist:
                    observables = tuple(cls.mean([other[i] for other in others]) for i in range(len(others)))
                else:
                    observables = (cls.mean(others),)
            values.append(other.view())
        value = np.cov(values, rowvar=False)
        if return_type == 'nparray':
            return value
        return CovarianceMatrix(observables=observables, value=value)

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
            new.__dict__[name] = value
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
        iprojs = self._index_projs(projs, return_list=True)
        indices = []
        for iproj in iprojs:
            if method == 'mid': xx = (self._edges[iproj][..., 0] + self._edges[iproj][..., 1]) / 2.
            else: xx = self._x[iproj]
            xx = np.asarray(xx)
            if xlim is not None:
                tmp = (xx >= xlim[0]) & (xx <= xlim[1])
                tmp = np.all(tmp, axis=tuple(range(1, tmp.ndim)))
            else:
                tmp = np.ones(xx.shape[0], dtype='?')
                #print(xlim, self._x[iproj].shape, self._edges[iproj].shape)
            tmp = np.flatnonzero(tmp)
            if concatenate: tmp += sum(len(xx) for xx in self._x[:iproj])
            indices.append(tmp)
        if concatenate:
            return np.concatenate(indices, axis=0)
        #if isscalar:
        #    return toret[0]
        return indices

    def _index_projs(self, projs=Ellipsis, return_list=False):
        """Return projs indices."""
        if projs is Ellipsis:
            return list(range(len(self._projs)))

        def _get_index(proj):
            try: return self._projs.index(proj)
            except ValueError:
                raise ValueError('{} could not be found in {}'.format(proj, self._projs))

        if not isinstance(projs, list):
            index = _get_index(projs)
            if return_list: return [index]
            return index
        return [_get_index(proj) for proj in projs]

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

        def get_unique_edges(edges):
            return [jnp.unique(edges[:, axis], axis=0) for axis in range(edges.shape[1])]

        def get_1d_slice(edges, sl):
            sl1 = edges[sl, 0]
            sl2 = edges[sl.start + sl.step - 1:sl.stop + sl.step - 1:sl.step, 1]
            size = min(sl1.shape[0], sl2.shape[0])
            return jnp.column_stack([sl1[:size], sl2[:size]])

        for iproj in iprojs:
            self_edges = self._edges[iproj]
            if isinstance(edges, list): iedges = edges[iproj]
            else: iedges = edges
            ndim = (1 if self_edges.ndim < 3 else self_edges.shape[1])
            if isinstance(iedges, slice):
                iedges = (iedges,) * ndim
            if isinstance(iedges, tuple):
                assert all(isinstance(iedge, slice) for iedge in iedges)
                slices = [_format_slice(iedge, len(self_edges)) for iedge in iedges]
                assert len(slices) == ndim, f'Provided tuple of slices should be of size {ndim:d}, found {len(slices):d}'
                if self_edges.ndim == 2:
                    iedges = get_1d_slice(self_edges, slices[0])
                else:
                    iedges1d = [get_1d_slice(e, s) for e, s in zip(get_unique_edges(self_edges), slices)]

                    def isin2d(array1, array2):
                        assert len(array1) == len(array2)
                        toret = True
                        for a1, a2 in zip(array1, array2): toret &= jnp.isin(a1, a2)
                        return toret

                    # This is to keep the same ordering
                    upedges = self_edges[..., 1][isin2d(self_edges[..., 1].T, [e[..., 1] for e in iedges1d])]
                    lowedges = jnp.column_stack([iedges1d[axis][..., 0][jnp.searchsorted(iedges1d[axis][..., 1], upedges[..., axis])] for axis in range(ndim)])
                    iedges = jnp.concatenate([lowedges[..., None], upedges[..., None]], axis=-1)

            # Broadcast iedges[:, None, :] against edges[None, :, :]
            mask = (self_edges[None, ..., 0] >= iedges[:, None, ..., 0]) & (self_edges[None, ..., 1] <= iedges[:, None, ..., 1])  # (new_size, old_size) or (new_size, old_size, ndim)
            if mask.ndim >= 3:
                mask = mask.all(axis=-1)  # collapse extra dims if needed
            if weighted:
                matrix = jnp.where(mask, self._weights[iproj][None, :], 0.0)
            else:
                matrix = mask.astype(float)

            if normalize:
                matrix /= jnp.where(jnp.all(matrix == 0, axis=-1), 1., jnp.sum(matrix, axis=-1))[:, None]
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
        iprojs = self._index_projs(projs, return_list=True)
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
                    for row in matrix:
                        edge = state[name][iproj][row != 0]
                        _edges.append(jnp.concatenate([edge[0, ..., 0][None, ..., None], edge[-1, ..., 1][None, ..., None]], axis=-1))
                    state[name][iproj] = jnp.concatenate(_edges, axis=0)
                elif name == '_weights':
                    state[name][iproj] = matrix.dot(state[name][iproj])
                else:
                    mask = self._weights[iproj] == 0
                    tmp = jnp.where(mask[(Ellipsis,) + (None,) * (state[name][iproj].ndim - mask.ndim)], 0., state[name][iproj])
                    state[name][iproj] = nwmatrix.dot(tmp)
        state = {name: getattr(self, name) for name in self._data_fields + self._meta_fields} | {name: tuple(value) for name, value in state.items()}
        projs = self._projs
        if select_projs:
            for name in self._select_proj_fields:
                state[name] = tuple(state[name][iproj] for iproj in iprojs)
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(state)
        return new

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
        iprojs = self._index_projs(projs, return_list=True)
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
            x = self._x[iproj]
            xmid = jnp.mean(self._edges[iproj], axis=-1)
            if method == 'mixed':
                return jnp.where(jnp.isnan(x), xmid, x)
            elif method == 'mid':
                return xmid
            return x

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

    def _slice_matrix(self, edges, axis='o', projs=Ellipsis, weighted=True, normalize=True):
        # Return, for a given slice, the corresponding matrix to apply to the data arrays
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
        matrix = observable._slice_matrix(list_edges, projs=list_projs, weighted=weighted, normalize=normalize)
        import scipy
        return scipy.linalg.block_diag(*matrix)

    @classmethod
    def sum(cls, others, axis='o', weights=None):
        if weights is None:
            weights = np.ones(len(others))
        weights = np.asarray(weights)
        if weights.ndim == 0: weights = np.full(len(others), weights)
        assert weights.shape == (len(others),), 'weights must be of same shape as number of input matrices'
        new = None
        values = []
        for other, weight in zip(others, weights):
            if new is None:
                new = other.copy()
                axis, name, observable = new._axis_index(axis=axis)
                new.__dict__['_' + name] = observable.sum([other._axis_index(axis=axis)[-1] for other in others], weights=weights)
            values.append(weight * other._value)
        new._value = sum(values)
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
                new.__dict__['_' + name] = observable.concatenate([other._axis_index(axis=axis)[-1] for other in others])
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
        new = self.select(axis=axis, projs=projs, select_projs=select_projs)
        axis, name, observable = new._axis_index(axis=axis)
        observable = observable.slice(edges, projs=projs)
        matrix = new._slice_matrix(edges, axis=axis, projs=projs, weighted=axis == 0, normalize=axis == 0)
        masks = [jnp.concatenate([weight for weight in observable._weights]) == 0 for observable in (new._observable, new._theory)]
        value = jnp.where(masks[0][..., None] | masks[1], 0., new._value)
        value = matrix.dot(value) if axis == 0 else value.dot(matrix.T)
        return new.clone(value=value, attrs=new.attrs, **{name: observable})

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
        #print(observable_indices, projs)
        proj_indices = observable._index_projs(projs, return_list=True)
        select_iprojs = proj_indices if select_projs else range(len(observable._projs))
        index = np.concatenate([observable._index(xlim=xlim if iproj in proj_indices else None, projs=observable._projs[iproj], method=method, concatenate=True) for iproj in select_iprojs])
        observable = observable.select(xlim=xlim, projs=projs, method=method, select_projs=select_projs)
        value = np.take(self._value, index, axis=axis)
        return self.clone(value=value, **{name: observable}, attrs=self.attrs)

    def view(self, return_type='nparray'):
        """
        Return window matrix.

        Parameters
        ----------
        return_type : str, default='nparray'
            If 'nparray', return numpy array :attr:`value`.
            Else, return a new :class:`WindowMatrix`, restricting to ``xlim`` and ``projs``.

        Returns
        -------
        new : array, WindowMatrix
        """
        if return_type is None:
            return self.copy()
        return jnp.asarray(self._value)

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
        return np.asarray(self.view(), *args, **kwargs)

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
        Plot window matrix.

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

    @plotter
    def plot_slice(self, indices, axis='o', color='C0', label=None, xscale='linear', yscale='log', fig=None):
        from matplotlib import pyplot as plt
        if np.ndim(indices) == 0: indices = [indices]
        indices = np.array(indices)
        alphas = np.linspace(1, 0.2, len(indices))
        fshape = len(self.observable.projs), len(self.theory.projs)
        if fig is None:
            fig, lax = plt.subplots(*fshape, sharey=True, figsize=(8, 6), squeeze=False)
        else:
            lax = np.array(fig.axes).reshape(fshape[::-1])
        axis, _, observable = self._axis_index(axis=axis)
        plotted_observable = [self.observable, self.theory][axis - 1]

        for it, pt in enumerate(self.theory.projs):
            for io, po in enumerate(self.observable.projs):
                value = self._value[np.ix_(self.observable._index(projs=pt, concatenate=True), self.theory._index(projs=po, concatenate=True))]
                for ix, idx in enumerate(indices):
                    ii = [io, it][axis]
                    plotted_ii = [io, it][axis - 1]
                    iidx = idx
                    if np.issubdtype(idx.dtype, np.floating):
                        iidx = np.abs(observable.x()[ii] - idx).argmin()
                    # Indices in approximate window matrix
                    x = plotted_observable._x[plotted_ii]
                    dx = 1.
                    if axis == 0:  # axis = 'o', showing theory, dividing by integration element dx
                        dx = plotted_observable._edges[plotted_ii]
                        dx = dx[..., 1] - dx[..., 0]
                    v = np.take(value, iidx, axis=axis)
                    v = v / dx
                    if yscale == 'log': v = np.abs(v)
                    ax = lax[io][it]
                    ax.plot(x, v, alpha=alphas[ix], color=color, label=label if ix == 0 else None)
                ax.set_title(r'${} \times {}$'.format(pt, po))
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.grid(True)
                if io == len(self.observable.projs) - 1: ax.set_xlabel(plotted_observable._label_x)
                if label and it == io == 0: lax[it][io].legend()

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        return fig

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
                tnew = [self.theory, new][axis]  # new theory axis
                dx = self.theory.edges(projs=pt)[..., 1] - self.theory.edges(projs=pt)[..., 0]
                dxnew = tnew.edges(projs=pt)[..., 1] - tnew.edges(projs=pt)[..., 0]
                ixo, ixt = self.observable._index(projs=po, concatenate=True), self.theory._index(projs=pt, concatenate=True)
                xxo, xxt = np.meshgrid(xo, xt, indexing='ij')
                # x is observable.x, y is theory.x - x, z is wmat
                x = [xxo, xxt][axis]
                p = [po, pt][axis]
                y = xxt - xxo
                v = self._value[np.ix_(ixo, ixt)] / dx
                x, y, z = x.ravel(), y.ravel(), v.ravel()
                del xxo, xxt
                if p not in new.projs: continue
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
                        ddx = dxnew
                    else:  # theory
                        xt = x
                        idx = (ixo, ix)
                        ddx = dxnew[ix]
                    value[idx] += interp(x, xt - xo) * ddx
        return self.clone(**{name: new, 'value': value})



@jax.tree_util.register_pytree_node_class
class CovarianceMatrix(object):
    """
    Class representing a covariance matrix.

    Attributes
    ----------
    _value : array
        Window matrix 2D array.

    _observables : BinnedStatistic
        (Mean) observable corresponding to the window matrix.

    attrs : dict
        Other attributes.
    """
    _data_fields = ['_observables', '_value']
    _meta_fields = ['attrs']

    def __init__(self, observables, value, attrs=None):
        """
        Initialize :class:`CovarianceMatrix`.

        Parameters
        ----------
        observables : BinnedStatistic
            (Mean) observable corresponding to the covariance matrix.

        value : array
            Covariance matrix 2D array.

        attrs : dict, default=None
            Optionally, other attributes, stored in :attr:`attrs`.
        """
        if isinstance(value, self.__class__):
            self.__dict__.update(value.__dict__)
            return
        state = {}
        state['_value'] = np.asarray(value)
        shape = state['_value'].shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Covariance matrix must be square, got shape = {}'.format(shape))
        if not isinstance(observables, (tuple, list)):
            observables = [observables]
        state['_observables'], size = [], 0
        for value in observables:
            value if isinstance(value, BinnedStatistic) else BinnedStatistic(**value)
            state['_observables'].append(value)
            size += value.size
        if size != shape[0]:
            raise ValueError('size = {:d} of input observables must match input matrix shape = {}'.format(size, shape))
        state['_observables'] = tuple(state['_observables'])
        state['attrs'] = dict(attrs or {})
        self.__dict__.update(state)

    def _observable_index(self, observables, return_list=False):
        if observables is None or observables is Ellipsis:
            return tuple(range(len(self._observables)))
        names = [observable.name for observable in self._observables]

        def _get_index(observable):
            name = observable
            if isinstance(observable, BinnedStatistic): name = observable.name
            try:
                idx = names.index(name)
            except ValueError:
                raise ValueError('Observable {} not found in {}'.format(observable, self._observables))
            return idx

        if not isinstance(observables, (tuple, list)):
            index = _get_index(observables)
            if return_list: return [index]
        return [_get_index(observable) for observable in observables]

    def observables(self, observables=None):
        """(Mean) observables."""
        index = self._observable_index(observables)
        if isinstance(index, tuple):
            return tuple(self._observables[idx] for idx in index)
        return self._observables[index]

    def _slice_matrix(self, edges=None, observables=Ellipsis, projs=Ellipsis, normalize=True):
        # Return, for a given slice, the corresponding matrix to apply to the data arrays
        if projs is not Ellipsis and not isinstance(projs, list): projs = [projs]
        obs_indices = self._observable_index(observables, return_list=True)
        if edges is None:
            edges = slice(None)
        if isinstance(edges, BinnedStatistic):
            edges = list(edges._edges)
        matrix = []
        for iobs, observable in enumerate(self._observables):
            if iobs not in obs_indices:
                matrix += observable._slice_matrix(normalize=normalize)
                continue
            observable = self._observables[iobs]
            proj_indices = observable._index_projs(projs, return_list=True)
            list_projs = list(observable._projs)
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
            matrix += observable._slice_matrix(list_edges, projs=list_projs, normalize=normalize)
        import scipy
        return scipy.linalg.block_diag(*matrix)

    @classmethod
    def sum(cls, others, weights=None):
        if weights is None:
            weights = np.ones(len(others))
        weights = np.asarray(weights)
        if weights.ndim == 0: weights = np.full(len(others), weights)
        assert weights.shape == (len(others),), 'weights must be of same shape as number of input matrices'
        new = None
        values = []
        for other, weight in zip(others, weights):
            if new is None:
                new = other.copy()
            values.append(weight**2 * other._value)
        new._value = sum(values)
        return new

    def __add__(self, other):
        """Sum of `self`` + ``other`` covariance matrices."""
        return self.sum([self, other])

    def __radd__(self, other):
        if other == 0: return self.slice()
        return self.__add__(other)

    def _index(self, observables=Ellipsis, xlim=None, projs=Ellipsis, method='mid', concatenate=True):
        """
        Return indices for given x-limits and projs.

        Parameters
        ----------
        observables : BinnedStatistic, str
            Observable(s).

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
        obs_indices = self._observable_index(observables, return_list=True)
        indices = []
        for iobs in obs_indices:
            observable = self._observables[iobs]
            index = observable._index(xlim=xlim, projs=projs, method=method, concatenate=concatenate)
            if concatenate: indices.append(sum(xx.size for xx in self._observables[:iobs]) + index)
            else: indices += index
        if concatenate:
            return np.concatenate(indices, axis=0)
        return indices

    def slice(self, edges=None, observables=Ellipsis, projs=Ellipsis, select_observables=False, select_projs=False):
        """
        Apply selections to the covariance matrix, slicing for given observable and projections.

        Parameters
        ----------
        observables : BinnedStatistic, str
            Observable(s) to apply ``slice`` to.

        edges : slice, default=None
            Slicing to apply, defaults to ``slice(None)``.

        projs : list, default=None
            List of projections to apply ``slice`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : CovarianceMatrix
        """
        new = self.select(observables=observables, projs=projs, select_observables=select_observables, select_projs=select_projs)
        matrix = new._slice_matrix(observables=observables, edges=edges, projs=projs)
        mask = jnp.concatenate([weight for observable in new._observables for weight in observable._weights]) == 0
        value = jnp.where(mask[..., None] | mask, 0., new._value)
        value = matrix.dot(value).dot(matrix.T)
        observables = list(new._observables)
        obs_indices = new._observable_index(observables, return_list=True)
        for iobs in obs_indices:
            observables[iobs] = observables[iobs].slice(edges, projs=projs)
        return new.clone(value=value, observables=observables, attrs=new.attrs)

    def select(self, observables=Ellipsis, xlim=None, projs=Ellipsis, select_observables=False, select_projs=False, method='mid'):
        """
        Apply selections for given observables and projections.

        Parameters
        ----------
       observables : BinnedStatistic, str
            Observable(s).

        xlim : tuple, default=None
            Restrict coordinates to these (min, max) limits.
            Defaults to ``(-np.inf, np.inf)``.

        axis : str
            Axis to apply the selection to.
            One of ('o', 'observable') or ('t', 'theory').

        projs : list, default=None
            List of projections to apply ``xlim`` and ``rebin`` to.
            Defaults to :attr:`projs`.

        Returns
        -------
        new : CovarianceMatrix
        """
        obs_indices = self._observable_index(observables, return_list=True)
        nobs = len(obs_indices) if select_observables else len(self._observables)
        observables, indices = [None] * nobs, [None] * nobs
        for iobs, observable in enumerate(self._observables):
            if iobs in obs_indices:
                observable = observable.select(xlim=xlim, projs=projs, select_projs=select_projs)
            if iobs in obs_indices or not select_observables:
                idx = obs_indices.index(iobs) if select_observables else iobs
                observables[idx] = observable
                proj_indices = observable._index_projs(projs, return_list=True)
                select_iprojs = proj_indices if select_projs else range(len(observable._projs))
                index = np.concatenate([self._index(observables=observable, xlim=xlim if iproj in proj_indices else None, projs=observable._projs[iproj], method=method, concatenate=True) for iproj in select_iprojs])
                indices[idx] = index
        indices = np.concatenate(indices)
        value = self._value[np.ix_(indices, indices)]
        new = self.clone(value=value, observables=observables, attrs=self.attrs)
        return new

    def view(self, return_type='nparray'):
        """
        Return covariance matrix.

        Parameters
        ----------
        return_type : str, default='nparray'
            If 'nparray', return numpy array :attr:`value`.
            Else, return a new :class:`CovarianceMatrix`, restricting to ``xlim`` and ``projs``.

        Returns
        -------
        new : array, CovarianceMatrix
        """
        if return_type is None:
            return self.copy()
        return jnp.asarray(self._value)

    def std(self, return_type='nparray'):
        std = jnp.sqrt(jnp.diag(self._value))
        if return_type is None:
            toret = []
            for observable in self._observables:
                indices = [self._index(observables=observable, projs=proj, concatenate=True) for proj in observable.projs]
                toret.append(observable.clone(value=[std[index] for index in indices]))
            return toret
        return std

    def corrcoef(self, return_type='nparray'):
        std = self.std()
        corrcoef = self._value / (std[..., None] * std)
        if return_type is None:
            return self.clone(value=corrcoef)
        return corrcoef

    def clone(self, **kwargs):
        """Create a new instance, updating some attributes."""
        state = {name[1:] if name.startswith('_') else name: getattr(self, name) for name in self._data_fields + self._meta_fields}  # remove front _
        state.update(kwargs)
        return self.__class__(**state)

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        state = {}
        for name in self._data_fields + self._meta_fields:
            value = getattr(self, name)
            if isinstance(value, tuple):  # observables
                value = tuple(observable.copy() for observable in value)
            else:
                value = value.copy()
            state[name] = value
        new.__dict__.update(state)
        return new

    def __getstate__(self):
        state = {name: getattr(self, name) for name in self._data_fields + self._meta_fields}
        for name in ['_observables']:
            value = getattr(self, name)
            state[name] = tuple(value.__getstate__() for value in value)
        return state

    def __setstate__(self, state):
        for name in ['_observables']:
            state[name] = tuple(BinnedStatistic.from_state(value) for value in state[name])
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
        """Return string representation of covariance matrix."""
        return '{}({})'.format(self.__class__.__name__, self._observables)

    def __array__(self, *args, **kwargs):
        return np.asarray(self._value, *args, **kwargs)

    @property
    def shape(self):
        """Return covariance matrix shape."""
        return self._value.shape

    @plotter
    def plot(self, corrcoef=False, split_observables=True, split_projs=True, **kwargs):
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
        assert split_projs if split_observables else True, 'Cannot split projections without splitting observables.'
        indices, x, xlabels, plabels = [], [], [], []
        for observable in self._observables:
            indices2 = [self._index(observables=observable, projs=proj, concatenate=True) for proj in observable._projs]
            x2 = [observable.x(projs=proj) for proj in observable._projs]
            xlabels2 = [observable._label_x] * len(observable._projs)
            plabels2 = [observable._get_label_proj(proj) for proj in observable._projs]
            if not split_projs:
                indices2 = [np.concatenate(indices2, axis=0)]
                x2 = [np.concatenate(x2, axis=0)]
                xlabels2 = [xlabels2[0]]
                plabels2 = ['']
            indices += indices2
            x += x2
            xlabels += xlabels2
            plabels += plabels2
        if not split_observables:
            x = [np.concatenate(x, axis=0)]
            indices = [np.concatenate(indices, axis=0)]
            xlabels = []
            plabels = []
        for ilabel in range(2):
            kwargs.setdefault(f'xlabel{ilabel + 1:d}', xlabels)
            kwargs.setdefault(f'label{ilabel + 1:d}', plabels)
        value = self._value
        if corrcoef:
            std = np.sqrt(np.diag(value))
            value = value / (std[..., None] * std)
        mat = [[value[np.ix_(index1, index2)] for index2 in indices] for index1 in indices]
        return plot_matrix(mat, x1=x, x2=x, **kwargs)

    @plotter
    def plot_diag(self, offset=0, color='C0', xscale='linear', yscale='linear', ytransform=None, fig=None):
        from matplotlib import pyplot as plt
        offsets = np.atleast_1d(offset)
        alphas = np.linspace(1, 0.2, len(offsets))
        observables_projs = [(observable.select(projs=proj, select_projs=True), proj) for observable in self._observables for proj in observable.projs]
        fshape = (len(observables_projs),) * 2
        if fig is None:
            fig, lax = plt.subplots(*fshape, sharex=False, sharey=False, figsize=(8, 6), squeeze=False)
        else:
            lax = np.array(fig.axes).reshape(fshape[::-1])
        for i1, i2 in itertools.product(*[list(range(s)) for s in fshape]):
            observables, projs = zip(*[observables_projs[i] for i in [i1, i2]])
            value = self._value[np.ix_(*[self._index(observables=observable, projs=proj, concatenate=True) for observable, proj in zip(observables, projs)])]
            for offset, alpha in zip(offsets, alphas):
                index = np.arange(max(min(observable.size - offset for observable in observables), 0))
                flag = int(i2 > i1)
                index1, index2 = index, index + offset
                diag = value[index1, index2]
                x = observables[flag].x()[0][index]
                if ytransform is not None: diag = ytransform(x, diag)
                label = None
                if i1 == i2 == 0: label = r'$\mathrm{{offset}} = {:d}$'.format(offset)
                ax = lax[i2, i1]
                ax.set_title(r'${} \times {}$'.format(*projs))
                ax.plot(x, diag, alpha=alpha, color=color, label=label)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.grid(True)
                if i1 == i2 == 0 and len(offsets) > 1: ax.legend()

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        return fig

    @plotter
    def plot_slice(self, indices, axis='vertical', color='C0', label=None, xscale='linear', yscale='log', fig=None):
        from matplotlib import pyplot as plt
        if np.ndim(indices) == 0: indices = [indices]
        indices = np.array(indices)
        alphas = np.linspace(1, 0.2, len(indices))
        observables_projs = [(observable.select(projs=proj, select_projs=True), proj) for observable in self._observables for proj in observable.projs]
        fshape = (len(observables_projs),) * 2
        if fig is None:
            fig, lax = plt.subplots(*fshape, sharey=True, figsize=(8, 6), squeeze=False)
        else:
            lax = np.array(fig.axes).reshape(fshape[::-1])
        for i1, i2 in itertools.product(*[list(range(s)) for s in fshape]):
            observables, projs = zip(*[observables_projs[i] for i in [i1, i2]])
            value = self._value[np.ix_(*[self._index(observables=observable, projs=proj, concatenate=True) for observable, proj in zip(observables, projs)])]
            ax = lax[i2, i1]
            for ix, idx in enumerate(indices):
                iidx = idx
                if np.issubdtype(idx.dtype, np.floating):
                    iidx = np.abs(observables[0].x()[0] - idx).argmin()
                v = np.take(value, iidx, axis=0)
                if yscale == 'log': value = np.abs(v)
                x = observables[1].x()[0]
                ax.plot(x, v, alpha=alphas[ix], color=color, label=label if ix == 0 else None)
            ax.set_title(r'${} \times {}$'.format(*projs))
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.grid(True)
            if label and i1 == i2 == 0: lax[i1][i2].legend()

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        return fig


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