import os
from functools import partial
from dataclasses import dataclass

import numpy as np
import jax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .mesh import MeshAttrs, staticarray, ParticleField, get_sharding_mesh
from .mesh2 import FKPField, _format_ells
from .utils import get_legendre, get_spherical_jn, set_env, register_pytree_dataclass
from .types import Particle2SpectrumPole, Particle2SpectrumPoles, Mesh2CorrelationPole, Particle2CorrelationPoles, ObservableLeaf, ObservableTree


def _make_edges2(kind, mattrs, edges: staticarray | dict | None=None, xavg: staticarray | None=None, sattrs: dict | None=None, wattrs: dict | None=None, ells=0):
    from cucount.jax import SelectionAttrs, WeightAttrs
    if kind == 'complex':
        max_ = mattrs.knyq.max()
    else:
        max_ = jnp.sqrt(jnp.sum(mattrs.boxsize**2))
    if edges is None:
        edges = {}
    if isinstance(edges, dict):
        edges = dict(edges)
        if mattrs is not None:
            edges.setdefault('max', max_)
        edges = np.arange(edges.get('min', 0.), edges['max'], edges['step'])
    edges = np.asarray(edges)
    if edges.ndim == 2:
        assert np.allclose(edges[1:, 0], edges[:-1, 1])
        edges = np.append(edges[:, 0], edges[-1, 1])
    edges = np.column_stack([edges[:-1], edges[1:]])
    if xavg is None:
        xavg = 3. / 4. * (edges[..., 1]**4 - edges[..., 0]**4) / (edges[..., 1]**3 - edges[..., 0]**3)
    ells = _format_ells(ells)
    if not isinstance(sattrs, SelectionAttrs):
        sattrs = SelectionAttrs(**(sattrs or {}))
    if not isinstance(wattrs, WeightAttrs):
        wattrs = WeightAttrs(**(wattrs or {}))
    return dict(edges=edges, xavg=xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)


@partial(register_pytree_dataclass, meta_fields=['ells'])
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
    sattrs : dict, cucount.jax.SelectionAttrs, optional
        Selection criteria for pairs.
    wattrs : dict, cucount.jax.WeightAttrs, optional
        Weight attributes.
    ells : int or array-like, optional
        Multipole orders to compute.

    Attributes
    ----------
    edges : ndarray
        Bin edges for pair separation.
    sattrs : cucount.jax.SelectionAttrs
        Selection criteria.
    wattrs : cucount.jax.WeightAttrs
        Weight attributes.
    ells : tuple
        Multipole orders.
    """
    edges: jax.Array = None
    sattrs: dict = None
    wattrs: dict = None

    def __init__(self, mattrs=None, edges: staticarray | dict | None=None, xavg: staticarray | None=None, sattrs: dict | None=None, wattrs: dict | None=None, ells=0):
        kw = _make_edges2('complex', mattrs, edges=edges, xavg=xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)
        self.__dict__.update(kw)

    def __call__(self, *particles, los='firstpoint'):
        """
        Compute the binned power spectrum multipoles from particle pairs.

        Parameters
        ----------
        *particles : cucount.jax.Particles
            Particle fields to correlate (autocorrelation if one provided).
        los : str, optional
            Line-of-sight definition.

        Returns
        -------
        spectrum : ndarray
            Power spectrum multipoles for each bin.
        """
        from cucount.jax import Particles, count2, BinAttrs, setup_logging
        sharding_mesh = get_sharding_mesh()
        particles = [convert_particles(particle) if not isinstance(particle, Particles) else particle for particle in particles]

        if len(particles) == 1: particles = particles * 2

        def call(*particles):
            #setup_logging('error')
            battrs = BinAttrs(k=np.asarray(self.xavg), pole=(self.ells, los))
            return count2(*particles, battrs=battrs, sattrs=self.sattrs, wattrs=self.wattrs)['weight'].T

        # Note about parallel computation:
        # To create a shard array, with same size on all process,
        # we add particles with weight 0 located at the mean position of the data chunk.
        # This creates a lot of repeats at the same location,
        # which would lead to an increased computation time in the pair counting,
        # as it is dominated by the pairs in that spatial bin.
        # So in cucount we remove all particles with weight 0 prior to pair counting.
        # Loop to reduce the memory footprint
        return call(*particles)


@partial(register_pytree_dataclass, meta_fields=['ells'])
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
    sattrs : dict, cucount.jax.SelectionAttrs, optional
        Selection criteria for pairs.
    wattrs : dict, cucount.jax.WeightAttrs, optional
        Weight attributes.
    ells : int or array_like, optional
        Multipole orders to compute.

    Attributes
    ----------
    edges : ndarray
        Bin edges for pair separation.
    sattrs : cucount.jax.SelectionAttrs
        Selection criteria.
    wattrs : cucount.jax.WeightAttrs
        Weight attributes.
    ells : ndarray
        Multipole orders.
    """
    edges: jax.Array = None
    xavg: jax.Array = None
    boxsize: jax.Array = None
    selection: dict = None

    def __init__(self, mattrs=None, edges: staticarray | dict | None=None, sattrs: dict | None=None, wattrs: dict | None=None, ells=0):
        kw = _make_edges2('real', mattrs, edges=edges, sattrs=sattrs, wattrs=wattrs, ells=ells)
        self.__dict__.update(kw)

    def __call__(self, *particles, los='firstpoint'):
        """
        Compute the binned correlation function multipoles from particle pairs.

        Parameters
        ----------
        *particles : cucount.jax.Particles
            Particle fields to correlate (autocorrelation if one provided).
        los : str, optional
            Line-of-sight definition.

        Returns
        -------
        correlation : ndarray
            Correlation function multipoles for each bin.
        """
        from cucount.jax import Particles, count2, BinAttrs, setup_logging
        sharding_mesh = get_sharding_mesh()
        particles = [convert_particles(particle) if not isinstance(particle, Particles) else particle for particle in particles]

        if len(particles) == 1: particles = particles * 2

        def call(*particles):
            #setup_logging('error')
            edges = np.append(self.edges[:, 0], self.edges[-1, 1])
            battrs = BinAttrs(s=edges, pole=(self.ells, los))
            return count2(*particles, battrs=battrs, sattrs=self.sattrs, wattrs=self.wattrs)['weight'].T

        return call(*particles)


def convert_particles(particles: ParticleField, weights=None, exchange_weights: bool=True, index_value: bool=None):
    """Convert :class:`ParticleField` to :class:`cucount.jax.Particles`, optionally updating weights."""
    from cucount.jax import Particles, _make_list_weights
    if isinstance(particles, FKPField):
        particles = particles.particles
    sharding_mesh = particles.attrs.sharding_mesh
    with_sharding = bool(sharding_mesh.axis_names)
    if weights is not None:
        weights = _make_list_weights(weights)
        weights = [jnp.asarray(weight) for weight in weights]
        sharding_mesh = particles.attrs.sharding_mesh
        with_sharding = bool(sharding_mesh.axis_names)
        if with_sharding and exchange_weights and particles.exchange_direct is not None:
            input_is_not_sharded = (getattr(weights[0], 'sharding', None) is not None) and (getattr(weights[0].sharding, 'mesh', None) is None)
            if particles.exchange_direct.backend == 'jax' and input_is_not_sharded:
                from .mesh import make_array_from_process_local_data
                weights = [make_array_from_process_local_data(weight, pad=0) for weight in weights]
            weights = [particles.exchange_direct(weight, pad=0) for weight in weights]
    else:
        weights = particles.weights
    return Particles(particles.positions, weights=weights, positions_type='pos', index_value=index_value, exchange=False, sharding_mesh=particles.attrs.sharding_mesh)



def compute_particle2(*particles: ParticleField, bin: BinParticle2SpectrumPoles | BinParticle2CorrelationPoles=None, los='firstpoint'):
    """
    Compute the 2-point spectrum or correlation function from particles.

    Parameters
    ----------
    *particles : ParticleField or cucount.jax.Particles
        Particles to correlate (autocorrelation if one provided).
    bin : BinParticle2SpectrumPoles or BinParticle2CorrelationPoles
        Binning object specifying edges, selection, and multipoles.
    los : str, optional
        Line-of-sight definition.

    Returns
    -------
    result : Particle2CorrelationPoles or Particle2SpectrumPoles
        Resulting spectrum or correlation function object.
    """
    ells = bin.ells
    particles = list(particles)

    num = bin(*particles, los=los)

    if isinstance(bin, BinParticle2CorrelationPoles):
        correlation = []
        for ill, ell in enumerate(ells):
            correlation.append(Mesh2CorrelationPole(s=bin.xavg, s_edges=bin.edges, num_raw=num[ill], norm=jnp.ones_like(num[ill]), ell=ell))
        return Particle2CorrelationPoles(correlation)
    else:  # 'complex'
        spectrum = []
        for ill, ell in enumerate(ells):
            spectrum.append(Particle2SpectrumPole(k=bin.xavg, k_edges=bin.edges, num_raw=num[ill], norm=jnp.ones_like(num[ill]), ell=ell))
        return Particle2SpectrumPoles(spectrum)


def compute_particle2_shotnoise(*particles: ParticleField, bin: BinParticle2SpectrumPoles | BinParticle2CorrelationPoles=None):
    """
    Compute the shot noise for the particle-based power spectrum and correlation function multipoles.

    Parameters
    ----------
    particles : ParticleField or cucount.jax.Particles
        Particles.
    bin : BinParticle2SpectrumPoles or BinParticle2CorrelationPoles, optional
        Binning operator.

    Returns
    -------
    shotnoise : float, list
    """
    autocorr = len(particles) == 1

    if autocorr:
        wattrs = bin.wattrs
        from cucount.jax import Particles
        particles = [convert_particles(particles[0]) if not isinstance(particles[0], Particles) else particles[0]] * 2
        num_shotnoise = jnp.sum(wattrs(*particles))

    else:
        num_shotnoise = 0.

    ells = bin.ells
    num_shotnoise = [(2 * ell + 1) * get_legendre(ell)(0.) * num_shotnoise for ell in ells]
    if isinstance(bin, BinParticle2CorrelationPoles):
        mask_shotnoise = (bin.edges[..., 0] <= 0.) & (bin.edges[..., 1] >= 0.)
    else:
        mask_shotnoise = jnp.ones_like(bin.xavg)
        num_shotnoise = [sn * (ell == 0) for ell, sn in zip(ells, num_shotnoise)]
    return [num_shotnoise[ill] * mask_shotnoise for ill, ell in enumerate(ells)]