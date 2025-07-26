import numpy as np
import jax
from jax import numpy as jnp
from scipy import special

from jaxpower import MeshAttrs, generate_uniform_particles, compute_particle2, BinParticle2Spectrum, BinParticle2Correlation, Spectrum2Poles, Correlation2Poles, utils


@np.vectorize
def spherical_bessel(x, ell=0):

    ABS, SIN, COS = np.abs, np.sin, np.cos

    absx = ABS(x)
    threshold_even = 4e-2
    threshold_odd = 2e-2
    if (ell == 0):
        if (absx < threshold_even):
            x2 = x * x
            return 1 - x2 / 6 + (x2 * x2) / 120
        return SIN(x) / x
    if (ell == 2):
        x2 = x * x
        if (absx < threshold_even): return x2 / 15 - (x2 * x2) / 210
        return (3 / x2 - 1) * SIN(x) / x - 3 * COS(x) / x2
    if (ell == 4):
        x2 = x * x
        x4 = x2 * x2
        if (absx < threshold_even): return x4 / 945
        return 5 * (2 * x2 - 21) * COS(x) / x4 + (x4 - 45 * x2 + 105) * SIN(x) / (x * x4)
    if (ell == 1):
        if (absx < threshold_odd): return x / 3 - x * x * x / 30
        return SIN(x) / (x * x) - COS(x) / x
    if (ell == 3):
        if (absx < threshold_odd): return x * x * x / 105
        x2 = x * x
        return (x2 - 15) * COS(x) / (x * x2) - 3 * (2 * x2 - 5) * SIN(x) / (x2 * x2)


def legendre(x, ell):

    if (ell == 0):
        return 1.
    if (ell == 2):
        x2 = x * x
        return (3 * x2 - 1) / 2
    if (ell == 4):
        x2 = x * x
        return (35 * x2 * x2 - 30 * x2 + 3) / 8
    if (ell == 1):
        return x
    if (ell == 3):
        return (5 * x * x * x - 3 * x) / 2


def test_legendre_bessel():
    mu = np.linspace(-1., 1., 1000)
    x = np.geomspace(1e-9, 100, 1000)
    for ell in range(5):
        assert np.allclose(legendre(mu, ell), special.legendre(ell)(mu), atol=0, rtol=1e-9)
        assert np.allclose(spherical_bessel(x, ell), special.spherical_jn(ell, x, derivative=False), atol=1e-7, rtol=1e-3)


def generate_catalogs(size=100, boxsize=(1000,) * 3, offset=(1000., 0., 0.), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size) * b for o, b in zip(offset, boxsize)]
        weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64 * n_bitwise_weights)], dtype=np.uint64)
        # weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        # weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions + weights)
    return toret


def diff(position1, position2):
    return [p2 - p1 for p1, p2 in zip(position1, position2)]


def midpoint(position1, position2):
    return [p2 + p1 for p1, p2 in zip(position1, position2)]


def norm(position):
    return (sum(p**2 for p in position))**0.5


def dotproduct(position1, position2):
    return sum(x1 * x2 for x1, x2 in zip(position1, position2))


def dotproduct_normalized(position1, position2):
    return dotproduct(position1, position2) / (norm(position1) * norm(position2))


def wiip(weights, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + utils.popcount(*weights)
    mask = denom == 0
    denom[mask] = 1.
    toret = nrealizations / denom
    toret[mask] = default_value
    return toret


def wpip_single(weights1, weights2, nrealizations=None, noffset=1, default_value=0., correction=None):
    denom = noffset + sum(bin(w1 & w2).count('1') for w1, w2 in zip(weights1, weights2))
    if denom == 0:
        weight = default_value
    else:
        weight = nrealizations / denom
        if correction is not None:
            c = tuple(sum(bin(w).count('1') for w in weights) for weights in [weights1, weights2])
            weight /= correction[c]
    return weight


def wiip_single(weights, nrealizations=None, noffset=1, default_value=0.):
    denom = noffset + utils.popcount(*weights)
    return default_value if denom == 0 else nrealizations / denom


def get_weight(xyz1, xyz2, weights1, weights2, n_bitwise_weights=0, twopoint_weights=None, nrealizations=None, noffset=1, default_value=0., correction=None, weight_type='auto'):
    weight = 1
    if nrealizations is not None:
        weight *= wpip_single(weights1[:n_bitwise_weights], weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value, correction=correction)
    if twopoint_weights is not None:
        sep_twopoint_weights = twopoint_weights.sep
        twopoint_weights = twopoint_weights.weight
        costheta = sum(x1 * x2 for x1, x2 in zip(xyz1, xyz2)) / (norm(xyz1) * norm(xyz2))
        if (sep_twopoint_weights[0] <= costheta < sep_twopoint_weights[-1]):
            ind_costheta = np.searchsorted(sep_twopoint_weights, costheta, side='right', sorter=None) - 1
            frac = (costheta - sep_twopoint_weights[ind_costheta]) / (sep_twopoint_weights[ind_costheta + 1] - sep_twopoint_weights[ind_costheta])
            weight *= (1 - frac) * twopoint_weights[ind_costheta] + frac * twopoint_weights[ind_costheta + 1]
    if weight_type == 'inverse_bitwise_minus_individual':
        # print(1./nrealizations * weight, 1./nrealizations * wiip_single(weights1[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)\
        #          * wiip_single(weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value))
        weight -= wiip_single(weights1[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)\
                  * wiip_single(weights2[:n_bitwise_weights], nrealizations=nrealizations, noffset=noffset, default_value=default_value)
    for w1, w2 in zip(weights1[n_bitwise_weights:], weights2[n_bitwise_weights:]):
        weight *= w1 * w2
    return weight


def ref_theta_corr(edges, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), autocorr=False, selection_attrs=None, **kwargs):
    if data2 is None: data2 = data1
    counts = np.zeros(len(edges) - 1, dtype='f8')
    sep = np.zeros(len(edges) - 1, dtype='f8')
    poles = [np.zeros(len(edges) - 1, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    selection_attrs = dict(selection_attrs or {})
    theta_limits = selection_attrs.get('theta', None)
    if theta_limits is not None:
        costheta_limits = np.cos(np.deg2rad(theta_limits)[::-1])
    rp_limits = selection_attrs.get('rp', None)
    npairs = 0
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            if theta_limits is not None:
                theta = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1)))  # min to avoid rounding errors
                if theta < theta_limits[0] or theta >= theta_limits[1]: continue
                #if all(x1 == x2 for x1, x2 in zip(xyz1, xyz2)): costheta = 1.
                #else: costheta = min(dotproduct_normalized(xyz1, xyz2), 1)
                #if costheta <= costheta_limits[0] or costheta > costheta_limits[1]: continue
            dxyz = diff(xyz1, xyz2)
            dist = norm(dxyz)
            npairs += 1
            if dist > 0:
                if los == 'midpoint': mu = dotproduct_normalized(dxyz, midpoint(xyz1, xyz2))
                elif los == 'endpoint': mu = dotproduct_normalized(dxyz, xyz2)
                elif los == 'firstpoint': mu = dotproduct_normalized(dxyz, xyz1)
            else:
                mu = 0.
            if dist < edges[0] or dist >= edges[-1]: continue
            if rp_limits is not None:
                rp2 = (1. - mu**2) * dist**2
                if rp2 < rp_limits[0]**2 or rp2 >= rp_limits[1]**2: continue
            ind = np.searchsorted(edges, dist, side='right', sorter=None) - 1
            weights1, weights2 = xyzw1[3:], xyzw2[3:]
            weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
            counts[ind] += weight
            sep[ind] += weight * dist
            for ill, ell in enumerate(ells):
                poles[ill][ind] += weight * (2 * ell + 1) * legendre[ill](mu)
    return np.asarray(poles), sep / counts


def ref_theta_power(modes, data1, data2=None, boxsize=None, los='midpoint', ells=(0, 2, 4), autocorr=False, selection_attrs=None, **kwargs):
    if data2 is None: data2 = data1
    poles = [np.zeros_like(modes, dtype='c16') for ell in ells]
    legendre = [special.legendre(ell) for ell in ells]
    selection_attrs = dict(selection_attrs or {})
    theta_limits = selection_attrs.get('theta', None)
    rp_limits = selection_attrs.get('rp', None)
    npairs = 0
    for i1, xyzw1 in enumerate(zip(*data1)):
        for i2, xyzw2 in enumerate(zip(*data2)):
            if autocorr and i2 == i1: continue
            xyz1, xyz2 = xyzw1[:3], xyzw2[:3]
            if theta_limits is not None:
                theta = np.rad2deg(np.arccos(min(dotproduct_normalized(xyz1, xyz2), 1)))  # min to avoid rounding errors
                if theta < theta_limits[0] or theta >= theta_limits[1]: continue
            dxyz = diff(xyz1, xyz2)
            dist = norm(dxyz)
            npairs += 1
            if dist > 0:
                if los == 'midpoint': mu = dotproduct_normalized(dxyz, midpoint(xyz1, xyz2))
                elif los == 'endpoint': mu = dotproduct_normalized(dxyz, xyz2)
                elif los == 'firstpoint': mu = dotproduct_normalized(dxyz, xyz1)
            else:
                mu = 0.
            if rp_limits is not None:
                rp2 = (1. - mu**2) * dist**2
                if rp2 < rp_limits[0]**2 or rp2 >= rp_limits[1]**2: continue
            weights1, weights2 = xyzw1[3:], xyzw2[3:]
            weight = get_weight(xyz1, xyz2, weights1, weights2, **kwargs)
            for ill, ell in enumerate(ells):
                poles[ill] += (-1)**(ell // 2) * weight * (2 * ell + 1) * legendre[ill](mu) * special.spherical_jn(ell, modes * dist)
    return np.asarray(poles)


def test_particle2(plot=False):
    import time

    attrs = MeshAttrs(meshsize=(128,) * 3, boxsize=100., boxcenter=1200.)
    size = int(1e-3 * attrs.boxsize.prod())
    data = generate_uniform_particles(attrs, size + 1, seed=42)
    ells = (0, 2, 4)
    kw = dict(ells=ells, selection={'theta': (0., 0.05)})
    bin = BinParticle2Correlation(attrs, edges={'step': 0.1, 'max': 100.}, **kw)
    #with jax.disable_jit():
    t0 = time.time()
    pcount = compute_particle2(data, bin=bin)
    pcount = jax.block_until_ready(pcount)
    print(time.time() - t0)
    assert isinstance(pcount, Correlation2Poles)
    pcount.to_spectrum(jnp.linspace(0.01, 0.1, 20))
    bin = BinParticle2Spectrum(attrs, edges={'step': 0.01, 'max': 0.2}, **kw)
    power = compute_particle2(data, bin=bin)
    assert isinstance(power, Spectrum2Poles)
    power2 = pcount.to_spectrum(power)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for ill, ell in enumerate(power.ells):
            color = f'C{ill}'
            k = power.x(ell)
            ax.plot(k, k * power.view(projs=ell).real, color=color, linestyle='--')
            ax.plot(k, k * power2.view(projs=ell).real, color=color, linestyle='-', label=rf'$\ell = {ell:d}$')
        ax.plot([], [], color='k', linestyle='--', label='real')
        ax.plot([], [], color='k', linestyle='-', label='complex')
        plt.show()

def test():
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec as P

    sharding_mesh = jax.make_mesh((2,), ('x',))

    def custom(x, y):
        x, y = np.array(x), np.array(y)
        return np.sum(x + y)

    def s(x):
        out_type = jax.ShapeDtypeStruct((), float)
        return jax.lax.psum(jax.pure_callback(custom, out_type, x['x'], x['y']), sharding_mesh.axis_names)

    s = shard_map(s, mesh=sharding_mesh, in_specs=P(*sharding_mesh.axis_names), out_specs=P(None))
    x = jnp.ones(10)
    x = jax.device_put(x, jax.sharding.NamedSharding(sharding_mesh, spec=P(*sharding_mesh.axis_names)))
    print(s({'x': x, 'y': x}))


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)
    #test_particle2(plot=True)
    jax.distributed.initialize()
    test()
    jax.distributed.shutdown()