import os
from functools import partial
from dataclasses import dataclass

import numpy as np
import jax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .mesh import MeshAttrs, staticarray, ParticleField, _identity_fn
from .mesh2 import _format_ells
from .utils import get_legendre, get_spherical_jn, set_env
from .types import Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles


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


def _scancount(bin, *particles, los='firstpoint'):
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
            ipix = [hp.vec2pix(nside, *(particle[0].T / jnp.sqrt(jnp.sum(particle[0]**2, axis=-1))), nest=nest) for particle in particles]
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
                positions = particle[0][indices] #* jnp.where(mask, 1., jnp.nan)[..., None]
                weights = particle[1][indices] * mask
                toret.append((positions, weights))
            #sl = np.flatnonzero(np.isin(ipix_all, ipix[0]))[:10]
            #toret = [toret[0][..., sl]] + [(tmp[0], tmp[1]) for tmp in toret[1:]]
            return toret

    else:
        raise NotImplementedError("Selection by 'theta' is required for particle pair counts.")

    autocorr = len(particles) == 1
    neighbors, *sorted_particles = _sort_particles(*particles)
    num = jnp.zeros((len(ells), len(bin.edges)), dtype=particles[0][0].dtype)

    if isinstance(bin, BinParticle2SpectrumPoles):
        def _slab(self, positions1, weights1, positions2, weights2, los='firstpoint'):
            dist, mu, weight = _compute_dist_mu_weight(positions1, weights1, positions2, weights2, selection=self.selection, boxsize=self.boxsize, los=los)
            toret = []
            for ell in self.ells:
                wleg = (-1)**(ell // 2) * (2 * ell + 1) * get_legendre(ell)(mu) * weight
                tmp = jax.lax.map(lambda x: jnp.sum(get_spherical_jn(ell)(x * dist) * wleg), self.xavg, batch_size=max(min(1000 * 1000 / dist.size, 1), len(self.xavg), 1))
                toret.append(tmp)
            return jnp.stack(toret)
    else:
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

    def f(carry, particles):
        carry += _slab(bin, *particles, los=los)
        return carry, None

    for neighbor in neighbors:
        p1 = sorted_particles[0]
        p2 = p1 if autocorr else sorted_particles[1]
        ps = [p1[0][neighbors[0]], p1[1][neighbors[0]], p2[0][neighbor], p2[1][neighbor]]
        num = jax.lax.scan(f, init=num, xs=ps)[0]

    return num


def get_engine(engine='auto'):
    """
    Select the computation engine for pair counting.

    Parameters
    ----------
    engine : {'auto', 'jax', 'cucount'}, optional
        Engine selection. 'auto' tries to use cucount if available.

    Returns
    -------
    engine : str
        Selected engine name.
    """
    assert engine in ['auto', 'jax', 'cucount']
    if engine == 'auto':
        try:
            import cucount
        except ImportError:
            engine = 'jax'
        else:
            engine = 'cucount'
    return engine


from .mesh import default_sharding_mesh

@default_sharding_mesh
def _slice_particles(particles, nslices=None, sharding_mesh=None):
    if sharding_mesh.axis_names and sharding_mesh.devices.size > 1:
        if nslices is None: nslices = sharding_mesh.devices.size
        if nslices == 1:
            yield particles
        else:
            for islice in range(nslices):

                def f(array):
                    local_size = array.shape[0]
                    return array[slice(islice * local_size // nslices, (islice + 1) * local_size // nslices)]

                sliced_particles = shard_map(partial(jax.tree_map, f), mesh=sharding_mesh, in_specs=(P(sharding_mesh.axis_names,)), out_specs=P(sharding_mesh.axis_names,))(particles)

                yield sliced_particles
    else:
        yield particles


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class BinParticle2SpectrumPoles(object):
    """
    Binning class for estimating the 2-point power spectrum from particle pairs.

    Parameters
    ----------
    mattrs : MeshAttrs, optional
        Mesh attributes, e.g., boxsize, k-space limits.
    edges : array-like or dict, optional
        Bin edges or binning specification.
    selection : dict, optional
        Selection criteria for pairs.
    ells : int or array-like, optional
        Multipole orders to compute.
    engine : {'auto', 'jax', 'cucount'}, optional
        Computation engine.

    Attributes
    ----------
    edges : ndarray
        Bin edges for pair separation.
    boxsize : array-like
        Size of the periodic box.
    selection : dict
        Selection criteria.
    ells : ndarray
        Multipole orders.
    engine : str
        Computation engine.
    """
    edges: jax.Array = None
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
        if edges.ndim == 2:  # coming from Observable
            assert np.allclose(edges[1:, 0], edges[:-1, 1])
            edges = np.append(edges[:, 0], edges[-1, 1])
        edges = np.column_stack([edges[:-1], edges[1:]])
        xavg = 3. / 4. * (edges[..., 1]**4 - edges[..., 0]**4) / (edges[..., 1]**3 - edges[..., 0]**3)
        if selection is None:
            selection = {}
        ells = _format_ells(ells)
        engine = get_engine(engine)
        self.__dict__.update(edges=edges, xavg=xavg, selection=selection, boxsize=boxsize, ells=ells, engine=engine)

    def __call__(self, *particles, los='firstpoint'):
        """
        Compute the binned power spectrum multipoles from particle pairs.

        Parameters
        ----------
        *particles : ParticleField
            Particle fields to correlate (autocorrelation if one provided).
        los : str, optional
            Line-of-sight definition.

        Returns
        -------
        spectrum : ndarray
            Power spectrum multipoles for each bin.
        """
        sharding_mesh = particles[0].attrs.sharding_mesh
        if len(particles) == 1: particles = particles * 2
        particles = [(particle.positions, particle.weights) for particle in particles]

        if self.engine == 'cucount':

            def _call(*particles):
                from cucount.jax import count2, Particles, BinAttrs, SelectionAttrs, setup_logging
                setup_logging('error')
                particles = [Particles(*particle) for particle in particles]
                battrs = BinAttrs(k=self.xavg, pole=(np.array(self.ells), los))
                sattrs = SelectionAttrs(**self.selection)
                return count2(*particles, battrs=battrs, sattrs=sattrs).T

        else:

            def _call(*particles):
                return _scancount(self, *particles, los=los)

        call = _call

        if sharding_mesh.axis_names:

            call = shard_map(lambda *particles: jax.lax.psum(_call(*particles), sharding_mesh.axis_names), mesh=sharding_mesh, in_specs=(P(sharding_mesh.axis_names,), P(None)), out_specs=P(None))

            # Loop to reduce computational footprint
            nshards = sharding_mesh.devices.size
            particles2 = jax.tree_map(lambda array: array.reshape(nshards, array.shape[0] // nshards, *array.shape[1:]), particles[1])
            init = jnp.zeros((len(self.ells), len(self.edges)), dtype=particles[0][0].dtype)
            return jax.lax.scan(lambda carry, particles2: (carry + call(particles[0], particles2), 0), init, particles2)[0]

            #return sum(call(particles[0], particles2) for particles2 in _slice_particles(particles[1], nslices=None, sharding_mesh=sharding_mesh))

        return call(*particles)

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
class BinParticle2CorrelationPoles(object):
    """
    Binning class for estimating the 2-point correlation function from particle pairs.

    Parameters
    ----------
    mattrs : MeshAttrs, optional
        Mesh attributes, e.g., boxsize.
    edges : array_like or dict, optional
        Bin edges or binning specification.
    selection : dict, optional
        Selection criteria for pairs.
    ells : int or array_like, optional
        Multipole orders to compute.
    engine : {'auto', 'jax', 'cucount'}, optional
        Computation engine.

    Attributes
    ----------
    edges : ndarray
        Bin edges for pair separation.
    boxsize : array-like
        Size of the periodic box.
    selection : dict
        Selection criteria.
    ells : ndarray
        Multipole orders.
    linear : bool
        Whether binning is linear.
    engine : str
        Computation engine.
    """
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

    def __call__(self, *particles, los='firstpoint'):
        """
        Compute the binned correlation function multipoles from particle pairs.

        Parameters
        ----------
        *particles : ParticleField
            Particle fields to correlate (autocorrelation if one provided).
        los : str, optional
            Line-of-sight definition.

        Returns
        -------
        correlation : ndarray
            Correlation function multipoles for each bin.
        """
        sharding_mesh = particles[0].attrs.sharding_mesh
        if len(particles) == 1: particles = particles * 2
        particles = [(particle.positions, particle.weights) for particle in particles]

        if self.engine == 'cucount':

            def _call(*particles):
                from cucount.jax import count2, Particles, BinAttrs, SelectionAttrs, setup_logging
                setup_logging('error')
                particles = [Particles(*particle) for particle in particles]
                bins = np.append(self.edges[:, 0], self.edges[-1, 1])
                battrs = BinAttrs(s=bins, pole=(np.array(self.ells), los))
                sattrs = SelectionAttrs(**self.selection)
                return count2(*particles, battrs=battrs, sattrs=sattrs).T
        else:

            def _call(*particles):
                return _scancount(self, *particles, los=los)

        call = _call

        if sharding_mesh.axis_names:
            # Note about parallel computation:
            # To create a shard array, with same size on all process,
            # we add particles with weight 0 located at the mean position of the data chunk.
            # This creates a lot of repeats at the same location,
            # which would lead to an increased computation time in the pair counting,
            # as it is dominated by the pairs in that spatial bin.
            # So in cucount we remove all particles with weight 0 prior to pair counting.
            call = shard_map(lambda *particles: jax.lax.psum(_call(*particles), sharding_mesh.axis_names), mesh=sharding_mesh, in_specs=(P(sharding_mesh.axis_names,), P(None)), out_specs=P(None))

            # Loop to reduce computational footprint
            nshards = sharding_mesh.devices.size
            particles2 = jax.tree_map(lambda array: array.reshape(nshards, array.shape[0] // nshards, *array.shape[1:]), particles[1])
            init = jnp.zeros((len(self.ells), len(self.edges)), dtype=particles[0][0].dtype)
            return jax.lax.scan(lambda carry, particles2: (carry + call(particles[0], particles2), 0), init, particles2)[0]

            #return sum(call(particles[0], particles2) for particles2 in _slice_particles(particles[1], nslices=None, sharding_mesh=sharding_mesh))

        return call(*particles)

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


def compute_particle2(*particles: ParticleField, bin: BinParticle2SpectrumPoles | BinParticle2CorrelationPoles=None, los='firstpoint'):
    """
    Compute the 2-point spectrum or correlation function from particles.

    Parameters
    ----------
    *particles : ParticleField
        Particles to correlate (autocorrelation if one provided).
    bin : BinParticle2SpectrumPoles or BinParticle2CorrelationPoles
        Binning object specifying edges, selection, and multipoles.
    los : str, optional
        Line-of-sight definition.

    Returns
    -------
    result : Mesh2CorrelationPoles or Mesh2SpectrumPoles
        Resulting spectrum or correlation function object.
    """
    ells = bin.ells
    autocorr = len(particles) == 1
    # Can't be computed easily in general because of the selection
    num_shotnoise = 0.
    particles = list(particles)
    if autocorr:
        particles = particles * 2
        num_shotnoise = jnp.sum(particles[0].weights**2)

    num = bin(*particles, los=los)
    num_shotnoise = [(2 * ell + 1) * get_legendre(ell)(0.) * num_shotnoise for ell in ells]

    if isinstance(bin, BinParticle2CorrelationPoles):
        correlation = []
        for ill, ell in enumerate(ells):
            correlation.append(Mesh2CorrelationPole(s=bin.xavg, edges=bin.edges, num_raw=num[ill], norm=jnp.ones_like(num), num_shotnoise=num_shotnoise[ill], ell=ell))
        return Mesh2CorrelationPoles(correlation)
    else:  # 'complex'
        spectrum = []
        for ill, ell in enumerate(ells):
            spectrum.append(Mesh2SpectrumPole(k=bin.xavg, edges=bin.edges, num_raw=num[ill], edges=bin.edges, norm=jnp.ones_like(num), num_shotnoise=num_shotnoise[ill], ell=ell))
        return Mesh2SpectrumPoles(spectrum)
