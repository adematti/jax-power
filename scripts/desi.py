import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp

from mockfactory import Catalog
from jaxpower import (compute_mesh_power, ParticleField, FKPField, compute_fkp_power, MeshAttrs, BinAttrs, compute_fkp_normalization_and_shotnoise, utils, create_sharding_mesh, make_particles_from_local, create_sharded_array, create_sharded_random, setup_logging)


def get_clustering_positions_weights(catalog, zrange=None):
    from cosmoprimo.fiducial import DESI
    fiducial = DESI()
    catalog.get(catalog.columns())
    z = catalog['Z']
    weight = catalog['WEIGHT'] * catalog['WEIGHT_FKP']
    mask = Ellipsis
    if zrange:
        mask = (z >= zrange[0]) & (z <= zrange[1])
    dist = fiducial.comoving_radial_distance(z[mask])
    return np.column_stack([catalog['RA'][mask], catalog['DEC'][mask], dist]), np.asarray(weight[mask])


def get_data_fn(kind='power'):
    base_dir = Path('_tests')
    return base_dir / '{}.npy'.format(kind)


def compute_power_spectrum(data_fn, all_randoms_fn, zrange=(0.4, 1.1), **attrs):
    t0 = time.time()
    data = Catalog.read(data_fn)
    randoms = Catalog.read(all_randoms_fn)
    with create_sharding_mesh() as sharding_mesh:
        #get_clustering_positions_weights(data, zrange=zrange)
        #get_clustering_positions_weights(randoms, zrange=zrange)
        data = get_clustering_positions_weights(data, zrange=zrange)
        randoms = get_clustering_positions_weights(randoms, zrange=zrange)
        print('read', time.time() - t0)
        data = ParticleField(*make_particles_from_local(*data), **attrs)
        randoms = ParticleField(*make_particles_from_local(*randoms), **attrs)
        fkp = FKPField(data, randoms)
        jax.block_until_ready(fkp.randoms.sum())
        print('field', time.time() - t0)
        norm, shotnoise_nonorm = compute_fkp_normalization_and_shotnoise(fkp)
        mesh = fkp.paint(resampler='tsc', interlacing=3, compensate=True, out='real')
        del fkp
        jax.block_until_ready(mesh.std())
        print('painting', time.time() - t0)
        compute = partial(compute_mesh_power, edges={'step': 0.001}, ells=(0, 2, 4), los='firstpoint')
        compute = jax.jit(compute)
        mesh_power = compute(mesh).clone(norm=norm, shotnoise_nonorm=shotnoise_nonorm)
        #power = compute_fkp_power(fkp, resampler='tsc', interlacing=3, edges={'step': 0.001}, ells=(0, 2, 4), los='firstpoint')
        #power.save(get_data_fn(kind='power'))
        jax.block_until_ready(mesh_power)
        print('final', time.time() - t0)


def compute_box_power_spectrum(data_fn, all_randoms_fn, zrange=(0.4, 1.1), **attrs):
    with create_sharding_mesh() as sharding_mesh:
        if True:
            from jaxpower import MeshAttrs, create_sharded_random
            attrs = MeshAttrs(meshsize=np.rint(attrs['boxsize'] / attrs['cellsize']).astype(int), boxsize=attrs['boxsize'], boxcenter=1.1 * attrs['boxsize'])

            mesh = attrs.create(kind='real', fill=create_sharded_random(jax.random.normal, jax.random.key(42), shape=attrs.meshsize)).r2c()
        else:
            
            from jaxpower import MeshAttrs, generate_anisotropic_gaussian_mesh, BinnedStatistic

            def pk(k):
                kp = 0.03
                return 1e4 * (k / kp)**3 * jnp.exp(-k / kp)
        
            f, b = 0.8, 1.5
            beta = f / b
            kinedges = np.linspace(0.001, 0.7, 30)
            kin = (kinedges[:-1] + kinedges[1:]) / 2.
            ells = (0, 2, 4)
            poles = jnp.array([(1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pk(kin),
                                0.9 * (4. / 3. * beta + 4. / 7. * beta ** 2) * pk(kin),
                                8. / 35 * beta ** 2 * pk(kin)])
            theory = BinnedStatistic(x=[kin] * len(ells), edges=[kinedges] * len(ells), value=poles, projs=ells)
            generate = partial(generate_anisotropic_gaussian_mesh, attrs, los='x', unitary_amplitude=True)
            generate = jax.jit(generate)
            mesh = generate(theory, seed=jax.random.key(42))

        compute = partial(compute_mesh_power, edges={'step': 0.001}, ells=(0, 2, 4), los='firstpoint')
        compute = jax.jit(compute)
        mesh_power = compute(mesh)
        #power = compute_fkp_power(fkp, resampler='tsc', interlacing=3, edges={'step': 0.001}, ells=(0, 2, 4), los='firstpoint')
        #power.save(get_data_fn(kind='power'))


if __name__ == '__main__':
    import os
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
    from jax import config
    config.update('jax_enable_x64', True)

    catalog_dir = Path('/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/')
    tracer = 'LRG'
    region = 'NGC'
    data_fn = catalog_dir / f'{tracer}_{region}_clustering.dat.fits'
    all_randoms_fn = [catalog_dir / f'{tracer}_{region}_{iran:d}_clustering.ran.fits' for iran in range(2)]

    jax.distributed.initialize()
    setup_logging()
    #compute_box_power_spectrum(data_fn, all_randoms_fn, zrange=(0.4, 0.6), cellsize=6., boxsize=6. * 1620)  # 650 with 1 process, 1620 with 16
    compute_power_spectrum(data_fn, all_randoms_fn, zrange=(0.4, 0.6), cellsize=6., boxsize=6. * 1620)
    jax.distributed.shutdown()