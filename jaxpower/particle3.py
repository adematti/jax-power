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
                 wattrs: dict | None=None, sattrs: dict | None=None,
                 ells: tuple=(0, 0, 0)):
    from cucount.jax import SelectionAttrs, WeightAttrs
    if kind == 'complex':
        max_ = mattrs.knyq.max()
    else:
        max_ = jnp.sqrt(jnp.sum(mattrs.boxsize**2))
    if edges is None:
        edges = {}
    ndim = 2
    if not isinstance(edges, (tuple, list)):
        edges = [edges] * ndim
    edges = list(edges)
    ndim = len(edges)
    assert len(edges) == ndim
    if np.ndim(ells[0]) == 0:
        ells = [ells]
    ells = list(ells)
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
    if not isinstance(sattrs, SelectionAttrs):
        sattrs = SelectionAttrs(**(sattrs or {}))
    if not isinstance(wattrs, WeightAttrs):
        wattrs = WeightAttrs(**(wattrs or {}))
    edges1d = [staticarray(edge) for edge in edges]
    xavg1d = [staticarray(xx) for xx in xavg]

    def _cproduct(array):
        grid = np.meshgrid(*array, sparse=False, indexing='ij')
        return np.column_stack([tmp.ravel() for tmp in grid])

    edges = staticarray(np.concatenate([_cproduct([edge[..., 0] for edge in edges1d])[..., None], _cproduct([edge[..., 1] for edge in edges1d])[..., None]], axis=-1))
    xavg = staticarray(_cproduct(xavg1d))
    return dict(edges1d=edges1d, xavg1d=xavg1d, edges=edges, xavg=xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)


@partial(register_pytree_dataclass, meta_fields=['edges1d', 'xavg1d', 'edges', 'xavg', 'sattrs', 'wattrs', 'ells'])
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
    edges1d: list = None
    xavg1d: list = None
    edges: list = None
    xavg: list = None
    sattrs: dict = None
    wattrs: dict = None

    def __init__(self, mattrs=None, edges: staticarray | dict | None=None, sattrs: dict | None=None, wattrs: dict | None=None, ells=(0, 0, 0)):
        kw = _make_edges3('real', mattrs, edges=edges, sattrs=sattrs, wattrs=wattrs, ells=ells)
        self.__dict__.update(kw)

    def __call__(self, *particles, close_pair=((1, 2), (1, 3), (2, 3)), **kwargs):
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
        if np.ndim(close_pair[0]) == 0:
            close_pair = [close_pair]

        def call(*particles, close_pair=(1, 2)):
            #setup_logging('error')
            ells = triposh_to_poles(self.ells)
            battrs = []
            for edges, ells in zip(self.edges1d, ells, strict=True):
                edges = np.append(edges[:, 0], edges[-1, 1])
                # los irrelevant here
                battrs.append(BinAttrs(s=edges, pole=(ells, 'firstpoint')))
                #battrs.append(BinAttrs(s=edges))
            kw = dict(kwargs)
            kw[f'sattrs{close_pair[0]:d}{close_pair[1]:d}'] = self.sattrs
            if close_pair == (1, 3):
                kw.setdefault('veto12', self.sattrs)
            elif close_pair == (2, 3):
                kw.setdefault('veto12', self.sattrs)
                kw.setdefault('veto13', self.sattrs)
            ells, matrix = triposh_transform_matrix(battrs[0], battrs[1], self.ells)
            assert ells == self.ells
            counts = count3close(*particles, close_pair=close_pair, battrs12=battrs[0], battrs13=battrs[1],
                                 wattrs=self.wattrs, **kw)['weight']
            # In sugiyama = triposh basis, multipoles on first axis
            return jnp.dot(matrix, jnp.moveaxis(counts, -1, 0).reshape(counts.shape[-1], -1))

        return sum(call(*particles, close_pair=close_pair) for close_pair in close_pair)


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