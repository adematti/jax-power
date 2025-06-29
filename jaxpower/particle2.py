from dataclasses import dataclass

import os
import numpy as np
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from .mesh import MeshAttrs, staticarray, ParticleField, _identity_fn
from .mesh2 import _format_ells, Spectrum2Poles, Correlation2Poles
from .utils import get_legendre, get_spherical_jn, BinnedStatistic, plotter, set_env


def _compute_dist_mu_weight(positions1, weights1, positions2, weights2, selection=None, boxsize=None, los='x'):
    selection = selection or {}
    diff = positions2[:, None, :] - positions1[None, :, :]
    if boxsize is not None:
        diff = diff % boxsize

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
    for name, limits in selection.items():
        if name == 'theta':
            if norm1 is None: norm1 = norm(positions1)
            if norm2 is None: norm2 = norm(positions2)
            costheta = jnp.sum(positions2[:, None, :] * positions1[None, ...], axis=-1) / (norm1[:, None] * norm2)
            weight *= costheta >= jnp.cos(limits[1])
            if limits[0] != 0.: weight *= (costheta <= jnp.cos(limits[0]))  # numerical precision
        else:
            raise NotImplementedError
    return dist, mu, weight


def _sort_array(target, array, minlength=None):
    if minlength is None:
        minlength = np.bincount(array, minlength=target.max() + 1).max()
    indices = np.full((len(target), minlength), -1, dtype=array.dtype)
    for it, t in enumerate(target):
        index = np.flatnonzero(array == t)
        indices[it, :len(index)] = index
    return indices


try:
    from numba import njit
    _sort_array = njit(_sort_array)
except ImportError:
    pass


def _scan(bin, *particles, los='firstpoint'):
    ells = bin.ells

    if 'theta' in bin.selection:

        import healpy as hp

        theta_max = np.max(bin.selection['theta'])
        nsides = 2**np.arange(10)
        resols = hp.nside2resol(nsides, arcmin=False) * (180. / np.pi)
        margin = 4  # TBC: is it enough?
        nside = nsides[np.argmin(np.abs(resols - margin * theta_max))]
        nest = True

        def _sort_particles(*particles):
            ipix = [hp.vec2pix(nside, *(particle.positions.T / jnp.sqrt(jnp.sum(particle.positions**2, axis=-1))), nest=nest) for particle in particles]
            ipix_all = np.concatenate(ipix)
            ipix_count = np.bincount(ipix_all, minlength=hp.nside2npix(nside))
            neighbors = hp.get_all_neighbours(nside, np.flatnonzero(ipix_count), nest=nest)

            ipix_all = np.flatnonzero(ipix_count + np.bincount(neighbors.ravel(), minlength=hp.nside2npix(nside)))
            ipix_count_max = ipix_count.max()
            neighbors_all = np.repeat(np.arange(len(ipix_all))[None, ...], 9, axis=0)
            neighbors_all[1:, ipix_count[ipix_all] != 0] = np.searchsorted(ipix_all, neighbors, side='left')  # add neighbors for pixels that are not empty

            toret = [neighbors_all]
            for i, particle in enumerate(particles):
                indices = _sort_array(ipix_all, ipix[i], minlength=ipix_count_max)
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

    #@jax.jit
    def f(carry, particles):
        carry += bin._slab(*particles, los=los)
        return carry, None

    for neighbor in neighbors:
        p1 = sorted_particles[0]
        p2 = p1 if autocorr else sorted_particles[1]
        ps = [p1[0][neighbors[0]], p1[1][neighbors[0]], p2[0][neighbor], p2[1][neighbor]]
        num = jax.lax.scan(f, init=num, xs=ps)[0]

    return num


def get_engine(engine='auto'):
    assert engine in ['auto', 'jax', 'cucount']
    if engine == 'auto':
        try:
            from cucountlib import cucount
        except ImportError:
            engine = 'jax'
        else:
            engine = 'cucount'
    return engine


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class BinParticle2Spectrum(object):

    edges: jax.Array = None
    xavg: jax.Array = None
    boxsize: jax.Array = None
    selection: dict = None
    engine: str = None

    def __init__(self, mattrs=None, edges: staticarray | dict | None=None, selection: dict | None=None, ells=0, engine='auto'):
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            edges = dict(edges)
            if mattrs is not None:
                edges.setdefault('max', mattrs.knyq.max())
            edges = np.arange(edges.get('min', 0.), edges['max'], edges['step'])
        boxsize = getattr(mattrs, 'boxsize', None)
        edges = np.asarray(edges)
        if edges.ndim == 2:  # coming from BinnedStatistic
            assert np.allclose(edges[1:, 0], edges[:-1, 1])
            edges = np.append(edges[:, 0], edges[-1, 1])
        edges = np.column_stack([edges[:-1], edges[1:]])
        xavg = 3. / 4. * (edges[..., 1]**4 - edges[..., 0]**4) / (edges[..., 1]**3 - edges[..., 0]**3)
        if selection is None:
            selection = {}
        ells = _format_ells(ells)
        engine = get_engine(engine)
        self.__dict__.update(edges=edges, xavg=xavg, selection=selection, boxsize=boxsize, ells=ells, engine=engine)

    def _slab(self, positions1, weights1, positions2, weights2, los='firstpoint'):
        dist, mu, weight = _compute_dist_mu_weight(positions1, weights1, positions2, weights2, selection=self.selection, boxsize=self.boxsize, los=los)
        toret = []
        for ell in self.ells:
            wleg = (-1)**(ell // 2) * (2 * ell + 1) * get_legendre(ell)(mu) * weight
            tmp = jax.lax.map(lambda x: jnp.sum(get_spherical_jn(ell)(x * dist) * wleg), self.xavg, batch_size=max(min(1000 * 1000 / dist.size, 1), len(self.xavg), 1))
            toret.append(tmp)
        return jnp.stack(toret)

    def __call__(self, *particles, los='firstpoint'):
        if self.engine == 'cucount':
            with set_env(CUDA_VISIBLE_DEVICES=os.environ.get('SLURM_LOCALID', '0')):
                from cucountlib import cucount
                if len(particles) == 1:
                    particles = particles * 2
    
                def get(array):
                    return np.concatenate([_.data for _ in array.addressable_shards], axis=0)
    
                cparticles = [cucount.Particles(get(p.positions), get(p.weights)) for p in particles]
                battrs = cucount.BinAttrs(k=self.xavg, pole=(np.array(self.ells), los))
                sattrs = cucount.SelectionAttrs(**self.selection)
                toret = cucount.count2(*cparticles, battrs=battrs, sattrs=sattrs).T
        else:
            toret = _scan(self, *particles, los=los)
        return toret

    def tree_flatten(self):
        state = {name: getattr(self, name) for name in self.__annotations__.keys()}
        meta_fields = ['selection', 'engine'] + (['boxsize'] if self.boxsize is None else [])
        return tuple(state[name] for name in state if name not in meta_fields), {name: state[name] for name in meta_fields}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(zip(cls.__annotations__.keys(), children))
        new.__dict__.update(aux_data)
        return new


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class BinParticle2Correlation(object):

    edges: jax.Array = None
    xavg: jax.Array = None
    boxsize: jax.Array = None
    selection: dict = None
    linear: bool = False

    def __init__(self, mattrs=None, edges: staticarray | dict | None=None, selection: dict | None=None, ells=0, engine='auto'):
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            edges = dict(edges)
            if mattrs is not None:
                edges.setdefault('max', jnp.sqrt(jnp.sum(mattrs.boxsize**2)))
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
        ells = _format_ells(ells)
        engine = get_engine(engine)
        self.__dict__.update(edges=edges, xavg=xavg, selection=selection, linear=linear, boxsize=boxsize, ells=ells, engine=engine)

    def _slab(self, positions1, weights1, positions2, weights2, los='firstpoint'):
        dist, mu, weight = _compute_dist_mu_weight(positions1, weights1, positions2, weights2, selection=self.selection, boxsize=self.boxsize, los=los)
        if self.linear:
            idx = jnp.floor((dist - self.edges[0, 0]) / (self.edges[0, 1] - self.edges[0, 0])).astype(jnp.int16)
        else:
            bins = jnp.append(self.edges[:, 0], self.edges[-1, 1])
            idx = jnp.digitize(bins, dist, right=False) - 1
        mask = (idx >= 0) & (idx < len(self.edges))
        weight *= mask
        idx = jnp.where(mask, idx, 0)
        return jnp.stack([jnp.zeros(len(self.edges)).at[idx].add((2 * ell + 1) * get_legendre(ell)(mu) * weight) for ell in self.ells])
    
    def __call__(self, *particles, los='firstpoint'):
        if self.engine == 'cucount':
            with set_env(CUDA_VISIBLE_DEVICES=os.environ.get('SLURM_LOCALID', '0')):
                from cucountlib import cucount
                if len(particles) == 1:
                    particles = particles * 2
    
                def get(array):
                    return np.concatenate([_.data for _ in array.addressable_shards], axis=0)
    
                cparticles = [cucount.Particles(get(p.positions), get(p.weights)) for p in particles]
                bins = np.append(self.edges[:, 0], self.edges[-1, 1])
                battrs = cucount.BinAttrs(s=bins, pole=(self.ells[0], self.ells[-1], 2, los))
                sattrs = cucount.SelectionAttrs(**self.selection)
                toret = cucount.count2(*cparticles, battrs=battrs, sattrs=sattrs).T
        else:
            toret = _scan(self, *particles, los=los)
        return toret

    def tree_flatten(self):
        state = {name: getattr(self, name) for name in self.__annotations__.keys()}
        meta_fields = ['selection', 'linear'] + (['boxsize'] if self.boxsize is None else [])
        return tuple(state[name] for name in state if name not in meta_fields), {name: state[name] for name in meta_fields}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(zip(cls.__annotations__.keys(), children))
        new.__dict__.update(aux_data)
        return new


def compute_particle2(*particles: ParticleField, bin: BinParticle2Spectrum | BinParticle2Correlation=None, los='firstpoint'):

    ells = bin.ells
    autocorr = len(particles) == 1
    # Can't be computed easily in general because of the selection
    num_zero = 0.  #jnp.sum(particles[0].weights)**2 if autocorr else jnp.sum(particles[0].weights) * jnp.sum(particles[1].weights)
    num_shotnoise = 0.
    if autocorr:
        num_shotnoise = jnp.sum(particles[0].weights**2)
    particles = list(particles)
    is_distributed = jax.distributed.is_initialized() and (jax.process_count() > 1)
    sharding_mesh = particles[0].attrs.sharding_mesh
    if is_distributed:
        if autocorr: particles = particles * 2
        particle = particles[0]
        sharding = jax.sharding.NamedSharding(sharding_mesh, spec=P())
        p = jax.lax.with_sharding_constraint(particle.positions, sharding)
        w = jax.lax.with_sharding_constraint(particle.weights, sharding)
        particles[0] = ParticleField(p, w, attrs=particle.attrs)

    num = bin(*particles, los=los)
    if is_distributed:
        sharding = jax.sharding.NamedSharding(sharding_mesh, P(sharding_mesh.axis_names,))
        num = jax.make_array_from_process_local_data(sharding, num[None, ...])
        num = num.sum(axis=0)

    num_zero = jnp.zeros_like(num[..., 0]).at[0].set(num_zero)

    if isinstance(bin, BinParticle2Correlation):
        return Correlation2Poles(s=bin.xavg, num=num, ells=ells, edges=bin.edges,
                                 norm=1., num_shotnoise=num_shotnoise, num_zero=num_zero)
    else:  # 'complex'
        return Spectrum2Poles(k=bin.xavg, num=num, ells=ells, edges=bin.edges,
                              norm=1., num_shotnoise=num_shotnoise, num_zero=num_zero)
