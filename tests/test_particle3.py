import time

import jax

from jaxpower import MeshAttrs, generate_uniform_particles, compute_particle3, BinParticle3CorrelationPoles, Mesh3CorrelationPoles, utils


def test_particle3():
    mattrs = MeshAttrs(meshsize=(128,) * 3, boxsize=100., boxcenter=1200.)
    size = int(1e-3 * mattrs.boxsize.prod())
    data = generate_uniform_particles(mattrs, size + 1, seed=42)
    ells = [(0, 0, 0), (2, 0, 2)]
    kw = dict(ells=ells, sattrs={'theta': (0., 0.05)})
    bin = BinParticle3CorrelationPoles(mattrs, edges={'step': 0.1, 'max': 100.}, **kw)
    #with jax.disable_jit():
    t0 = time.time()
    pcount = compute_particle3(data, bin=bin)
    pcount = jax.block_until_ready(pcount)
    print(time.time() - t0)
    assert isinstance(pcount, Mesh3CorrelationPoles)


if __name__ == '__main__':
    from jax import config
    config.update('jax_enable_x64', True)
    
    test_particle3()
