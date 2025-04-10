import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import random
from jax import numpy as jnp

from jaxpower import (compute_mesh_power, PowerSpectrumMultipoles, generate_gaussian_mesh, generate_anisotropic_gaussian_mesh, generate_uniform_particles, ParticleField, FKPField,
                      compute_fkp_power, BinnedStatistic, WindowMatrix, MeshAttrs, BinAttrs, compute_mean_mesh_power, compute_mesh_window, compute_normalization, utils, create_sharding_mesh, make_particles_from_local)


dirname = Path('_tests')


def test_jaxdecomp():
    import jaxdecomp
    attrs = MeshAttrs(meshsize=[8, 16, 32], boxsize=1000.)
    mesh = attrs.create(kind='real', fill=partial(random.normal, key=random.key(43)))
    delta = mesh.value
    delta_k = jaxdecomp.fft.pfft3d(delta) #.astype(jnp.complex64))
    jax.block_until_ready(delta_k)
    delta_ref = jnp.fft.fftn(delta)
    delta_ref = jnp.transpose(delta_ref, axes=(1, 2, 0))
    diff = jnp.abs(delta_ref - delta_k)
    print(diff.max())


def test_sharding():
    from scipy import special

    def spherical_jn(ell):
        return lambda x: jax.pure_callback(partial(special.spherical_jn, ell), x, x)

    meshsize = (64,) * 3
    with create_sharding_mesh(meshsize=meshsize) as mesh:
        attrs = MeshAttrs(meshsize=meshsize, boxsize=1000.)
        if True:
            t0 = time.time()
            n = 5
            for i in range(n):
                attrs.kcoords(sparse=True)
            print((time.time() - t0) / n)

        if False:
            knorm = sum(xx**2 for xx in attrs.kcoords(sparse=True))
            jax.debug.inspect_array_sharding(knorm, callback=print)
            tmp = spherical_jn(2)(knorm)
            jax.debug.inspect_array_sharding(tmp, callback=print)


def test_mesh_power():

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

    list_los = ['x', 'endpoint']

    meshsize = (32,) * 3
    with create_sharding_mesh(meshsize=meshsize) as mesh:

        attrs = MeshAttrs(meshsize=meshsize, boxsize=1000.)
        edges = {'step': 0.01}
        ellsin = ells

        @partial(jax.jit, static_argnames=['los', 'ells'])
        def mock(seed, los='x', ells=(0, 2, 4)):
            mesh = generate_anisotropic_gaussian_mesh(attrs, theory, seed=seed, los=los, unitary_amplitude=True)
            return compute_mesh_power(mesh, ells=ells, los={'local': 'firstpoint'}.get(los, los), edges=edges)

        #mock(random.key(43), los='x')
        #mock(random.key(43), los='local')

        for flag in ['smooth', 'infinite'][1:]:
            for los, thlos in [('x', None), ('firstpoint', 'firstpoint'), ('firstpoint', 'local')]:
                print(flag, los, thlos, flush=True)
                mean = compute_mean_mesh_power(attrs, theory=(theory, thlos) if thlos is not None else theory, ells=ells, edges=edges, los=los)
                wmatrix = compute_mesh_window(attrs, edgesin=kinedges, ellsin=(ellsin, thlos) if thlos is not None else ellsin, ells=ells, edges=edges, los=los, pbar=True, flags=(flag,))


def test_particles():
    rank = jax.process_index()
    size = 1000
    attrs = MeshAttrs(meshsize=[8, 16, 32], boxsize=1000.)
    per_host_positions = attrs.boxsize * random.uniform(random.key(rank), (size, len(attrs.boxsize))) - attrs.boxsize / 2. + attrs.boxcenter

    mesh = ParticleField(positions=per_host_positions, attrs=attrs).paint(resmpler='tsc', interlacing=3, compensate=True, pexchange=True)
    ells = (0, 2, 4)
    edges = {'step': 0.01}
    compute_mesh_power(mesh, ells=ells, los='firstpoint', edges=edges)


def get_mock_fn(kind='data'):
    base_dir = Path('_tests')
    return base_dir / '{}.npy'.format(kind)


def save_reference_mock():
    attrs = MeshAttrs(meshsize=128, boxsize=1000., boxcenter=1200.)
    size = 128**2
    data_positions = generate_uniform_particles(attrs, size, seed=42).positions
    randoms_positions = generate_uniform_particles(attrs, 4 * size, seed=43).positions

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
    mesh = generate_anisotropic_gaussian_mesh(attrs, theory, seed=random.key(42), los='x', unitary_amplitude=True)
    data_weights = mesh.read(data_positions)

    utils.mkdir(get_mock_fn().parent)
    data = {'positions': data_positions, 'weights': data_weights}
    randoms = {'positions': randoms_positions, 'weights': jnp.ones_like(randoms_positions[..., 0])}
    np.save(get_mock_fn('data'), data, allow_pickle=True)
    np.save(get_mock_fn('randoms'), randoms, allow_pickle=True)

    data = ParticleField(**data, attrs=attrs)
    randoms = ParticleField(**randoms, attrs=attrs)
    fkp = FKPField(data, randoms)
    power = compute_fkp_power(fkp, edges={'step': 0.01}, resampler='tsc', interlacing=3, ells=(0, 2, 4), los='firstpoint')
    power.save(get_mock_fn('power'))


def compute_power_spectrum():

    size = jax.process_count()
    rank = jax.process_index()

    def load(kind='data'):
        catalog = np.load(get_mock_fn(kind=kind), allow_pickle=True)[()]
        total_size = catalog['positions']
        sl = slice(rank * total_size // size, (rank + 1) * total_size // size)
        return {name: catalog[name][sl] for name in catalog}

    ref = PowerSpectrumMultipoles.load(get_mock_fn('power'))
    los = ref.attrs.pop('los')
    attrs = MeshAttrs(**ref.attrs)

    with create_sharding_mesh(meshsize=attrs.meshsize) as mesh:
        data = ParticleField(make_particles_from_local(**load(kind='data')), attrs=attrs)
        randoms = ParticleField(make_particles_from_local(**load(kind='randoms')), attrs=attrs)
        fkp = FKPField(data, randoms)
        power = compute_fkp_power(fkp, edges={'step': 0.01}, resampler='tsc', interlacing=3, ells=(0, 2, 4), los='firstpoint')

    diff = jnp.abs(power.view() - ref.view()).nanmax()
    print(diff)


if __name__ == '__main__':
    # Setting up distributed jax
    jax.distributed.initialize()
    #test_jaxdecomp()
    #test_mesh_power()
    #test_sharding()
    # Closing distributed jax
    jax.distributed.shutdown()