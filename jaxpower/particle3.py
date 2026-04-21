from functools import partial
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from .mesh import staticarray, ParticleField, _make_input_tuple
from .mesh2 import FKPField
from .mesh3 import _format_meshes
from .utils import register_pytree_dataclass
from .types import Particle3SpectrumPole, Particle3SpectrumPoles, Particle3CorrelationPole, Particle3CorrelationPoles, ObservableLeaf, ObservableTree


def _make_edges3(kind, mattrs, edges: staticarray | dict | None=None,
                 sattrs: list | None=None,
                 wattrs: dict | None=None,
                 ells: tuple=(0, 0, 0)):
    from cucount.jax import SelectionAttrs, WeightAttrs
    if kind == 'complex':
        max_ = mattrs.knyq.max()
    else:
        max_ = jnp.sqrt(jnp.sum(mattrs.boxsize**2))
    if edges is None:
        edges = {}
    if not isinstance(battrs, (tuple, list)):
        ndim = 2
        battrs = [battrs] * ndim
    ndim = 2
    if not isinstance(edges, (tuple, list)):
        edges = [edges] * ndim
    edges = list(edges)
    ndim = len(edges)
    assert len(edges) == ndim
    if np.ndim(ells[0]) == 0:
        ells = [ells]
    ells = list(ells)
    if not isinstance(sattrs, (tuple, list)):
        sattrs = [sattrs] * ndim
    assert len(sattrs) == ndim
    sattrs = list(sattrs)
    xavg = [None] * ndim
    for iedge, edge in enumerate(edges):
        if isinstance(edge, dict):
            edge = dict(edge)
            if mattrs is not None:
                edge.setdefault('max', max_)
            edge = np.arange(edge.get('min', 0.), edge['max'], edge['step'])
        else:
            edge = np.asarray(edge)
        if edge.ndim == 2:
            assert np.allclose(edge[1:, 0], edge[:-1, 1])
            edge = np.append(edge[:, 0], edge[-1, 1])
        edge = np.column_stack([edge[:-1], edge[1:]])
        edges[iedge] = edge
        xavg[iedge] = 3. / 4. * (edge[..., 1]**4 - edge[..., 0]**4) / (edge[..., 1]**3 - edge[..., 0]**3)
        if not isinstance(sattrs[iedge], SelectionAttrs):
            sattrs[iedge] = SelectionAttrs(**(sattrs[iedge] or {}))
    if not isinstance(wattrs, WeightAttrs):
        wattrs = WeightAttrs(**(wattrs or {}))
    return dict(edges=[staticarray(edge) for edge in edges],
                xavg=[staticarray(xx) for xx in xavg],
                sattrs=sattrs, wattrs=wattrs, ells=ells)


@partial(register_pytree_dataclass, meta_fields=['edges', 'xavg', 'sattrs', 'wattrs', 'ells'])
@dataclass(init=False, frozen=True)
class BinParticle3CorrelationPoles(object):
    """
    Binning class for estimating the 3-point correlation function from particle triplets.

    Parameters
    ----------
    mattrs : MeshAttrs, optional
        Mesh attributes, e.g., boxsize.
    edges : list of array_like or dict, optional
        Bin edges or binning specification.
    sattrs : list of dict, cucount.jax.SelectionAttrs, optional
        Selection criteria for pairs 12, 13, 23 (optional).
    wattrs : dict, cucount.jax.WeightAttrs, optional
        Weight attributes.
    ells : list, optional
        List of multipole orders to compute.

    Attributes
    ----------
    edges : list
        Bin edges for pair separation.
    sattrs : cucount.jax.SelectionAttrs
        Selection criteria.
    wattrs : cucount.jax.WeightAttrs
        Weight attributes.
    ells : list
        Multipole orders.
    """
    edges: list = None
    xavg: list = None
    sattrs: list = None
    wattrs: dict = None

    def __init__(self, mattrs=None, edges: staticarray | dict | None=None, sattrs: dict | None=None, wattrs: dict | None=None, ells=(0, 0, 0)):
        kw = _make_edges3('real', mattrs, edges=edges, sattrs=sattrs, wattrs=wattrs, ells=ells)
        self.__dict__.update(kw)

    def __call__(self, *particles, cross=('12', '21'), **kwargs):
        """
        Compute the binned correlation function multipoles from particle pairs.

        Parameters
        ----------
        *particles : cucount.jax.Particles
            Particle fields to correlate (autocorrelation if one provided).
        **kwargs : dict
            Other arguments for :func:`cucount.jax.count3close`.

        Returns
        -------
        correlation : ndarray
            Correlation function multipoles for each bin.
        """
        from cucount.jax import Particles, count3close, BinAttrs, triposh_to_poles, triposh_transform_matrix
        particles = _make_input_tuple(*particles)
        particles = [convert_particles(particle) if not isinstance(particle, Particles) else particle for particle in particles]
        particles = _format_meshes(*particles)[0]

        def call(*particles, cross='12'):
            #setup_logging('error')
            ells = triposh_to_poles(self.ells)
            battrs = []
            for edges, ells in zip(self.edges, ells, strict=True):
                edges = np.append(edges[:, 0], edges[-1, 1])
                # los irrelevant here
                battrs.append(BinAttrs(s=edges, pole=(ells, 'firstpoint')))
            sattrs = self.sattrs
            if cross == '13':
                particles = [particles[0], particles[2], particles[1]]
                battrs = [battrs[1], battrs[0]]
                sattrs = [sattrs[1], sattrs[0], sattrs[2]]
                kwargs.setdefault('veto13', True)
            matrix = triposh_transform_matrix(battrs[0], battrs[1], self.ells)
            counts = count3close(*particles, battrs12=battrs[0], battrs13=battrs[1],
                                 sattrs12=sattrs[0], sattrs12=sattrs[1], sattrs23=sattrs[2],
                                 wattrs=self.wattrs, **kwargs)['weight']
            # In sugiyama = triposh basis, multipoles on first axis
            return jnp.moveaxis(counts.dot(matrix), -1, 0)

        return sum(call(*particles, cross=cross) for cross in cross)


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
            weights = [particles.exchange_direct(weight, pad=0) for weight in weights]
    else:
        weights = particles.weights
    return Particles(particles.positions, weights=weights, positions_type='pos', index_value=index_value, exchange=False, sharding_mesh=particles.attrs.sharding_mesh)


def compute_particle3(*particles: ParticleField, bin: BinParticle3CorrelationPoles=None, **kwargs):
    """
    Compute the 3-point correlation function from particles.

    Parameters
    ----------
    *particles : ParticleField or cucount.jax.Particles
        Particles to correlate (autocorrelation if one provided).
    bin : BinParticle3SpectrumPoles or BinParticle3CorrelationPoles
        Binning object specifying edges, selection, and multipoles.
    **kwargs : dict, optional
        Optional arguments for :class:`BinParticle3CorrelationPoles.__call__`.

    Returns
    -------
    result : Particle3CorrelationPoles or Particle3SpectrumPoles
        Resulting spectrum or correlation function object.
    """
    ells = bin.ells
    num = bin(*particles, **kwargs)

    if isinstance(bin, BinParticle3CorrelationPoles):
        correlation = []
        for ill, ell in enumerate(ells):
            correlation.append(Particle3CorrelationPole(s=bin.xavg, s_edges=bin.edges, num_raw=num[ill], norm=jnp.ones_like(num[ill]), ell=ell))
        return Particle3CorrelationPoles(correlation)
    else:
        raise NotImplementedError