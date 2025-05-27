from dataclasses import dataclass

import numpy as np
import healpy as hp
import jax
from jax import numpy as jnp

from .mesh import MeshAttrs, staticarray, ParticleField
from .pspec import _format_ells, PowerSpectrumMultipoles
from .utils import get_legendre, get_spherical_jn, BinnedStatistic, plotter



@jax.tree_util.register_pytree_node_class
class CorrelationFunctionMultipoles(BinnedStatistic):

    _label_x = r'$s$ [$\mathrm{Mpc}/h$]'
    _label_proj = r'$\ell$'
    _label_value = r'$\xi_{\ell}(s)$'
    _data_fields = BinnedStatistic._data_fields + ['_norm', '_num_shotnoise', '_num_zero']
    _select_proj_fields = BinnedStatistic._select_proj_fields + ['_num_zero']
    _sum_fields = BinnedStatistic._sum_fields + ['_norm', '_num_shotnoise', '_num_zero']
    _init_fields = {'k': '_x', 'num': '_value', 'nmodes': '_weights', 'edges': '_edges', 'ells': '_projs', 'norm': '_norm',
                    'num_shotnoise': '_num_shotnoise', 'num_zero': '_num_zero', 'name': 'name', 'attrs': 'attrs'}

    def __init__(self, s: np.ndarray, num: jax.Array, ells: tuple, nmodes: np.ndarray=None, edges: np.ndarray=None, norm: jax.Array=1.,
                 num_shotnoise: jax.Array=0., num_zero: jax.Array=None, name: str=None, attrs: dict=None):

        def _tuple(item):
            if item is None:
                return None
            if not isinstance(item, (tuple, list)):
                item = (item,) * len(ells)
            return tuple(item)

        self.__dict__.update(_norm=jnp.asarray(norm), _num_shotnoise=jnp.asarray(num_shotnoise), _num_zero=tuple(jnp.asarray(p) for p in num_zero))
        super().__init__(x=_tuple(s), edges=_tuple(edges), projs=ells, value=num,
                         weights=_tuple(nmodes), name=name, attrs=attrs)

    @property
    def norm(self):
        """Power spectrum normalization."""
        return self._norm

    @property
    def shotnoise(self):
        """Shot noise."""
        return self._num_shotnoise / self._norm

    @property
    def num(self):
        """Correlation function with shot noise *not* subtracted."""
        return self._value

    s = BinnedStatistic.x
    savg = BinnedStatistic.xavg
    nmodes = BinnedStatistic.weights

    @property
    def ells(self):
        return self._projs

    @property
    def value(self):
        """Correlation function estimate."""
        toret = list(self.num)
        for ill, ell in enumerate(self.ells):
            mask_zero = (self._edges[ill][..., 0] <= 0.) & (self._edges[ill][..., 1] >= 0.)
            toret[ill] = toret[ill] - (2 * ell + 1) * self._num_shotnoise * get_legendre(ell)(0.) * mask_zero
        return tuple(tmp / self._norm for tmp in toret)

    def to_power(self, k):
        num = []
        for ill, ell in enumerate(self.ells):
            if isinstance(k, BinnedStatistic):
                kk = k._x[ill]
            elif isinstance(k, (tuple, list)):
                kk = k[ill]
            else:
                kk = k
            def f(kk):
                return (-1)**(ell // 2) * jnp.sum(self.value[ill] * get_spherical_jn(ell)(kk * self._x[ill]), axis=-1)

            num.append(jax.lax.map(f, kk, batch_size=max(min(1000 * 1000 / len(self._x[ill]), len(kk)), 1)))
        num = tuple(num)
        if isinstance(k, BinnedStatistic):
            return k.clone(num=num)
        else:
            return PowerSpectrumMultipoles(k=k, num=num, ells=self.ells, norm=self._norm, num_shotnoise=self._num_shotnoise,
                                            num_zero=self._num_zero, attrs=self.attrs)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot correlation function.

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
            ax.plot(self._x[ill], self._x[ill]**2 * self.value[ill].real, label=self._get_label_proj(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(self._label_x)
        ax.set_ylabel(r'$s^2 \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
        return fig



@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class bin_pcount(object):

    edges: jax.Array = None
    xavg: jax.Array = None
    boxsize: jax.Array = None
    selection: dict = None
    linear: bool = False
    kind: str = 'real'

    def __init__(self, mattrs=None, edges: staticarray | dict | None=None, selection: dict | None=None, kind: str='real'):
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            edges = dict(edges)
            if mattrs is not None:
                edges.setdefault('max', jnp.sqrt(jnp.sum(mattrs.boxsize**2)) if kind == 'real' else mattrs.knyq.max())
            edges = np.arange(edges.get('min', 0.), edges['max'], edges['step'])
        boxsize = getattr(mattrs, 'boxsize', None)
        edges = np.asarray(edges)
        if edges.ndim == 2:
            assert np.allclose(edges[1:, 0], edges[:-1, 1])
            edges = np.append(edges[:-1, 0], edges[-1, 1])
        linear = np.allclose(np.diff(edges), edges[1] - edges[0])
        edges = np.column_stack([edges[:-1], edges[1:]])
        xavg = 3. / 4. * (edges[..., 1]**4 - edges[..., 0]**4) / (edges[..., 1]**3 - edges[..., 0]**3)
        if selection is None:
            selection = {}
        self.__dict__.update(edges=edges, xavg=xavg, selection=selection, linear=linear, kind=kind, boxsize=boxsize)

    def __call__(self, positions1, weights1, positions2, weights2, ells=(0, 2, 4), los='firstpoint'):

        diff = positions2[:, None, :] - positions1[None, :, :]
        if self.boxsize is not None:
            diff = diff % self.boxsize

        #toret = jnp.stack([jnp.zeros(len(self.edges)) for ell in ells])
        #weight = weights1[:, None] * weights2[None, :]
        #toret = toret.at[0].set(jnp.sum(jnp.all(diff == 0., axis=-1) * weight))
        #return toret

        def norm(x, **kwargs):
            return jnp.sqrt(jnp.sum(x**2, axis=-1, **kwargs))

        dist = norm(diff)
        norm1 = norm2 = None
        if not isinstance(los, str):
            pass
        elif los == 'firstpoint':
            norm1 = norm(positions1)
            los = positions1 / norm1[:, None]
        elif los == 'endpoint':
            norm2 = norm(positions2)
            los = positions1 / norm2[:, None]
        elif los == 'midpoint':
            los = positions1 + positions2
            los /= norm(los, keepdims=True)
        else:
            raise NotImplementedError
        mu = jnp.sum(diff * los, axis=-1) / jnp.where(dist == 0., 1., dist)
        weight = weights1[:, None] * weights2[None, :]
        for name, limits in self.selection.items():
            if name == 'theta':
                if norm1 is None: norm1 = norm(positions1)
                if norm2 is None: norm2 = norm(positions2)
                costheta = jnp.sum(positions2[:, None, :] * positions1[None, ...], axis=-1) / (norm1[:, None] * norm2)
                weight *= costheta >= jnp.cos(limits[1])
                if limits[0] != 0.: weight *= (costheta <= jnp.cos(limits[0]))  # numerical precision
            else:
                raise NotImplementedError
        if self.kind == 'real':
            if self.linear:
                idx = jnp.floor((dist - self.edges[0, 0]) / (self.edges[0, 1] - self.edges[0, 0])).astype(jnp.int16)
            else:
                bins = jnp.append(self.edges[:-1, 0], self.edges[-1, 1])
                idx = jnp.digitize(bins, dist, right=False) - 1
            mask = (idx >= 0) & (idx < len(self.edges))
            weight *= mask
            idx = jnp.where(mask, idx, 0)
            return jnp.stack([jnp.zeros(len(self.edges)).at[idx].add((2 * ell + 1) * get_legendre(ell)(mu) * weight) for ell in ells])
        else:
            toret = []
            for ell in ells:
                wleg = (-1)**(ell // 2) * (2 * ell + 1) * get_legendre(ell)(mu) * weight
                tmp = jax.lax.map(lambda x: jnp.sum(get_spherical_jn(ell)(x * dist) * wleg), self.xavg, batch_size=max(min(1000 * 1000 / dist.size, 1), len(self.xavg), 1))
                toret.append(tmp)
            return jnp.stack(toret)

    def tree_flatten(self):
        state = {name: getattr(self, name) for name in self.__annotations__.keys()}
        meta_fields = ['kind', 'linear'] + ['boxsize'] if self.boxsize is None else []
        return tuple(state[name] for name in state if name not in meta_fields), {name: state[name] for name in meta_fields}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(zip(cls.__annotations__.keys(), children))
        new.__dict__.update(aux_data)
        return new


def sort_array(target, array, minlength=None):
    if minlength is None:
        minlength = np.bincount(array, minlength=target.max() + 1).max()
    indices = np.full((len(target), minlength), -1, dtype=array.dtype)
    for it, t in enumerate(target):
        index = np.flatnonzero(array == t)
        indices[it, :len(index)] = index
    return indices


try:
    from numba import njit
    sort_array = njit(sort_array)
except ImportError:
    pass


def compute_particle_pcount(*particles: ParticleField, bin: bin_pcount=None, ells=(0, 2, 4), los='firstpoint'):

    ells = _format_ells(ells)

    if 'theta' in bin.selection:

        theta_max = np.max(bin.selection['theta'])
        nsides = 2**np.arange(10)
        resols = hp.nside2resol(nsides, arcmin=False) * (180. / np.pi)
        margin = 4  # TBC: is it enough?
        nside = nsides[np.argmin(np.abs(resols - margin * theta_max))]
        nest = True

        def _sort_particles(*particles):
            ipix = [hp.vec2pix(nside, *particle.positions.T, nest=nest) for particle in particles]
            ipix_all = np.concatenate(ipix)
            ipix_count = np.bincount(ipix_all, minlength=hp.nside2npix(nside))
            neighbors = hp.get_all_neighbours(nside, np.flatnonzero(ipix_count), nest=nest)

            ipix_all = np.flatnonzero(ipix_count + np.bincount(neighbors.ravel(), minlength=hp.nside2npix(nside)))
            ipix_count_max = ipix_count.max()
            neighbors_all = np.repeat(np.arange(len(ipix_all))[None, ...], 9, axis=0)
            neighbors_all[1:, ipix_count[ipix_all] != 0] = np.searchsorted(ipix_all, neighbors, side='left')  # add neighbors for pixels that are not empty

            toret = [neighbors_all]
            for i, particle in enumerate(particles):
                indices = sort_array(ipix_all, ipix[i], minlength=ipix_count_max)
                mask = indices >= 0
                indices = np.where(mask, indices, 0)
                positions = particle.positions[indices] #* jnp.where(mask, 1., jnp.nan)[..., None]
                weights = particle.weights[indices] * mask
                toret.append((positions, weights))
            #sl = np.flatnonzero(np.isin(ipix_all, ipix[0]))[:10]
            #toret = [toret[0][..., sl]] + [(tmp[0], tmp[1]) for tmp in toret[1:]]
            return toret

    else:
        raise NotImplementedError("Selection by 'theta' is required for particle pair counts.")

    autocorr = len(particles) == 1
    neighbors, *sorted_particles = _sort_particles(*particles)
    num = jnp.zeros((len(ells), len(bin.edges)), dtype=particles[0].positions.dtype)

    def f(carry, particles):
        carry += bin(*particles, ells=ells, los=los)
        return carry, None

    for neighbor in neighbors:
        p1 = sorted_particles[0]
        p2 = p1 if autocorr else sorted_particles[1]
        ps = [p1[0][neighbors[0]], p1[1][neighbors[0]], p2[0][neighbor], p2[1][neighbor]]
        num = jax.lax.scan(f, init=num, xs=ps)[0]

    num_zero = jnp.sum(particles[0].weights) if autocorr else jnp.sum(particles[0].weights) * jnp.sum(particles[1].weights)
    num_zero = jnp.zeros_like(num[..., 0]).at[0].set(num_zero)
    num_shotnoise = 0.
    if autocorr:
        num_shotnoise = jnp.sum(particles[0].weights**2)

    if bin.kind == 'real':
        return CorrelationFunctionMultipoles(s=bin.xavg, num=num, ells=ells, edges=bin.edges,
                                              norm=1., num_shotnoise=num_shotnoise, num_zero=num_zero)
    else:  # 'complex'
        return PowerSpectrumMultipoles(k=bin.xavg, num=num, ells=ells, edges=bin.edges,
                                       norm=1., num_shotnoise=num_shotnoise, num_zero=num_zero)
