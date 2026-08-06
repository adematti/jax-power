"""Shared realistic survey selection function for the window validations.

Lives in scripts/ rather than a scratchpad on purpose: the session scratchpad gets cleaned mid-run
and took an earlier copy of this module (plus two queued diagnostics) with it.

Geometry: octant of the sky (exactly 1/8, x,y,z >= 0) between z = 0.4 and z = 0.6, punched with small
circular holes. Sampled by high-density randoms, painted TSC + interlacing=3 + compensated. Several
INDEPENDENT random realizations are painted so that the window, the normalization AND the field legs
can all be built as CROSSES between them -- no self-pair (randoms shot-noise) term anywhere. A single
mask realization leaves its own shot power (1/n_r ~ 279 here) convolved with the theory, biasing P by
a few per cent.

Box: 1.5x the survey's bounding extent, meshsize 128 => cellsize 18.0, kf 0.00273, kNyq 0.175.
"""
import numpy as np, jax, jax.numpy as jnp
from jax import random
from jaxpower import MeshAttrs, ParticleField, compute_normalization

ZMIN, ZMAX = 0.4, 0.6
NHOLE, HOLE_DEG = 50, 2.0          # small holes: 2 deg radius, ~12% of the octant area
MESHSIZE = 128
PAD = 1.5
NRAND = 5_000_000                  # high density: ~23 randoms per mesh cell inside the survey
KW_PAINT = dict(resampler='tsc', interlacing=3, compensate=True)


def comoving_distances():
    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    return [float(cosmo.comoving_radial_distance(z)) for z in (ZMIN, ZMAX)]


R1, R2 = comoving_distances()


def get_mattrs():
    return MeshAttrs(boxsize=PAD * R2, meshsize=MESHSIZE, boxcenter=[R2 / 2.] * 3)


def hole_centers(seed=1234):
    # fixed across realizations: the holes are part of the SELECTION FUNCTION, not of the sampling
    u = np.asarray(random.normal(random.key(seed), (NHOLE, 3)))
    return np.abs(u) / np.linalg.norm(u, axis=-1, keepdims=True)


def make_particles(seed=0, nrand=NRAND, mattrs=None):
    """Uniform in the octant shell, holes removed. UNPAINTED, so the same catalogue can be painted
    on the several box sizes the stitched window needs."""
    if mattrs is None: mattrs = get_mattrs()
    keys = random.split(random.key(seed), 3)
    u = random.uniform(keys[0], (nrand,))
    r = (R1**3 + u * (R2**3 - R1**3))**(1. / 3.)                      # p(r) ~ r^2 on [R1, R2]
    n = random.normal(keys[1], (nrand, 3))
    n = jnp.abs(n) / jnp.linalg.norm(n, axis=-1, keepdims=True)       # uniform on the octant
    keep = jnp.all(n @ jnp.asarray(hole_centers()).T < np.cos(np.deg2rad(HOLE_DEG)), axis=-1)
    return ParticleField(r[:, None] * n, weights=1. * keep, attrs=mattrs)


def make_randoms(seed=0, nrand=NRAND, mattrs=None):
    return make_particles(seed=seed, nrand=nrand, mattrs=mattrs).paint(**KW_PAINT)


def selection_and_norms(nrand=NRAND, mattrs=None, nreal=3, particles=None):
    if mattrs is None: mattrs = get_mattrs()
    if particles is None:
        particles = [make_particles(seed=s, nrand=nrand, mattrs=mattrs) for s in range(nreal)]
    Ws = [p.clone(attrs=mattrs).paint(**KW_PAINT) for p in particles]
    norm2 = compute_normalization(Ws[0], Ws[1])
    norm3 = compute_normalization(*Ws[:3]) if nreal >= 3 else None
    return Ws[0], norm2, norm3, Ws


def describe():
    m = get_mattrs()
    box, cell = float(m.boxsize[0]), float(m.cellsize[0])
    vol = (1. / 8.) * (4. * np.pi / 3.) * (R2**3 - R1**3)
    return dict(R1=R1, R2=R2, boxsize=box, meshsize=MESHSIZE, cellsize=cell,
                kf=2. * np.pi / box, knyq=np.pi / cell, volume=vol,
                cells_in_survey=vol / cell**3, rand_per_cell=NRAND / (vol / cell**3),
                max_separation=R2 * np.sqrt(2.), s_reach_one_box=box / 4.)


# ---------------------------------------------------------------------------
# Stitched (multi-boxsize) window, following desi-clustering's scheme exactly
# (clustering_statistics/spectrum2_tools.py `method='smooth_mesh'` and
# spectrum3_tools.py `_get_window_edges`): keep the meshsize and scale the
# BOXSIZE, measure each scale only out to boxsize/4, then combine on a common
# s-grid with mutually exclusive masks cut at edges[-3] (the -3 offset leaves
# room for the cubic spline). The octant shell's largest separation is
# R2*sqrt(2) = 2171 while ONE box reaches only boxsize/4 = 575, so a single-box
# window is cut off; scale 4 reaches 2302 and closes it.
SCALES = (1, 4)


def _window_edges3(mattrs, scales=SCALES):
    """desi-clustering's _get_window_edges: 6 bins per cell size, doubling 6 times, then regular
    spacing out to distmax * scale."""
    distmax, cellmin = float(mattrs.boxsize.min()) / 4., float(mattrs.cellsize.min())
    nsizes, cellsizes = [6] * 5 + [None], [cellmin * 2**i for i in range(6)]
    edges = []
    for scale in scales:
        edges_scale, start = [], 0.
        for nsize, cellsize in zip(nsizes, cellsizes):
            cellsize = cellsize * scale
            if nsize is None:
                tmp = np.arange(start, distmax * scale / scales[0] + cellsize, cellsize)
            else:
                tmp = start + np.arange(nsize) * cellsize
            if tmp.size:
                start = tmp[-1] + cellsize
                edges_scale.append(tmp)
        edges_scale = np.concatenate(edges_scale, axis=0)
        edges.append(edges_scale[edges_scale < distmax * scale / scales[0] + cellsize])
    return edges


def _exclusive_weights(masks):
    weights = []
    for mask in masks:
        weights.append(mask & (~weights[-1]) if len(weights) else mask)
    return [jnp.maximum(m, 1e-6) for m in weights]


def stitched_window2(particles, norm, kw_window, los, mattrs=None, coords=None, scales=SCALES):
    from jaxpower import BinMesh2CorrelationPoles, compute_mesh2_correlation, interpolate_window_function
    if mattrs is None: mattrs = get_mattrs()
    if coords is None: coords = jnp.logspace(-3., 4., 2048)
    correlations, list_edges = [], []
    for scale in scales:
        mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
        distmax, cellsize = float(mattrs2.boxsize.min()) / 4., float(mattrs2.cellsize.min())
        edges = np.arange(0., distmax + cellsize, cellsize)
        list_edges.append(edges)
        meshes = [p.clone(attrs=mattrs2).paint(**KW_PAINT) for p in particles]
        sbin = BinMesh2CorrelationPoles(mattrs2, edges=edges, **kw_window, basis='bessel', batch_size=8)
        xi = compute_mesh2_correlation(meshes, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
        correlations.append(interpolate_window_function(xi, coords=coords, order=3))
    masks = [coords < edges[-3] for edges in list_edges[:-1]] + [coords < np.inf]
    return correlations[0].sum(correlations, weights=_exclusive_weights(masks)), list_edges


def stitched_window3(particles, norm, kw_window, los, mattrs=None, coords=None, scales=SCALES,
                     verbose=True):
    import time as _time
    from jaxpower import BinMesh3CorrelationPoles, compute_mesh3_correlation, interpolate_window_function
    if mattrs is None: mattrs = get_mattrs()
    # 512, not the 256 this defaulted to: at 256 the 2-D FFTlog chain is under-resolved and the
    # scoccimarro window matrix acquires a spurious ~15% squeezed-triangle excess (survey A/B:
    # squeezed/rest 1.154 at 256 versus 1.006 at 512, chi2/n 133.6 -> 61.9, everything else fixed).
    # Row sums are identical at both, so nothing about the normalization flags it.
    if coords is None: coords = jnp.logspace(-3., 4., 512)
    list_edges = _window_edges3(mattrs, scales=scales)
    correlations = []
    for scale, edges in zip(scales, list_edges):
        _t = _time.time()
        mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
        meshes = [p.clone(attrs=mattrs2).paint(**KW_PAINT) for p in particles]
        sbin = BinMesh3CorrelationPoles(mattrs2, edges=edges, **kw_window)
        Q = compute_mesh3_correlation(meshes, bin=sbin, los=los).clone(norm=[norm] * len(sbin.ells))
        if verbose:
            print(f'    scale {scale}: {len(edges)} s edges, {len(sbin.ells)} poles, '
                  f'correlation in {_time.time()-_t:.0f}s', flush=True)
        correlations.append(interpolate_window_function(Q.unravel(), coords=coords, order=3))
    c = list(next(iter(correlations[0])).coords().values())
    masks = [(c[0] < edges[-3])[:, None] * (c[1] < edges[-3])[None, :] for edges in list_edges[:-1]]
    masks.append((c[0] < np.inf)[:, None] * (c[1] < np.inf)[None, :])
    return correlations[0].sum(correlations, weights=_exclusive_weights(masks)), list_edges
