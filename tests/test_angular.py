import os
from pathlib import Path
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp

from jaxpower import (AngularAttrs, AlmField, PixelField, to_alm, to_pixel, alm2map, BinAngular2Spectrum,
                      compute_angular2_spectrum, compute_fkp_angular2_normalization, compute_fkp_angular2_shotnoise,
                      compute_angular2_spectrum_window, compute_angular2_spectrum_mean,
                      BinAngular3Spectrum, compute_angular3_spectrum, compute_fkp_angular3_normalization,
                      compute_fkp_angular3_shotnoise, compute_angular3_spectrum_window, Angular2Spectrum, Angular3Spectrum,
                      MeshAttrs, ParticleField, FKPField, read, utils)


dirname = Path('_tests')


def random_positions(size, seed=42, mask=None):
    """Uniform points on the (masked) sphere, as (N, 3) unit vectors."""
    rng = np.random.RandomState(seed=seed)
    positions = []
    while sum(len(p) for p in positions) < size:
        z = rng.uniform(-1., 1., size)
        phi = rng.uniform(0., 2. * np.pi, size)
        sin = np.sqrt(1. - z**2)
        xyz = np.column_stack([sin * np.cos(phi), sin * np.sin(phi), z])
        if mask is not None:
            xyz = xyz[mask(xyz)]
        positions.append(xyz)
    return np.concatenate(positions)[:size]


def get_mattrs():
    # dummy 3D mesh attributes, only used to carry ParticleField positions
    return MeshAttrs(meshsize=4, boxsize=3., boxcenter=0.)


def test_wigner3j000():
    from jaxpower.angular import _compute_wigner3j000_sq
    from jaxpower.utils import wigner_3j
    for ell1 in range(8):
        for ell2 in range(8):
            for ell3 in range(12):
                ref = wigner_3j(ell1, ell2, ell3, 0, 0, 0)**2
                test = _compute_wigner3j000_sq(ell1, ell2, ell3)
                assert np.allclose(test, ref, rtol=1e-12, atol=1e-15), (ell1, ell2, ell3, test, ref)
    print('test_wigner3j000 OK')


def test_alm_direct():
    # Direct summation vs brute-force scipy spherical harmonics
    try:
        from scipy.special import sph_harm_y
        Ylm = lambda ell, m, theta, phi: sph_harm_y(ell, m, theta, phi)
    except ImportError:
        from scipy.special import sph_harm
        Ylm = lambda ell, m, theta, phi: sph_harm(m, ell, phi, theta)

    ellmax = 8
    size = 100
    positions = random_positions(size, seed=42)
    rng = np.random.RandomState(seed=43)
    weights = rng.uniform(0.5, 1.5, size)
    theta = np.arccos(positions[:, 2])
    phi = np.arctan2(positions[:, 1], positions[:, 0])

    attrs = AngularAttrs(ellmax=ellmax)
    alm = to_alm(jnp.asarray(positions), weights=jnp.asarray(weights), attrs=attrs)
    for ell in range(ellmax + 1):
        for m in range(ell + 1):
            ref = np.sum(weights * np.conj(Ylm(ell, m, theta, phi)))
            assert np.allclose(alm.value[ell, m], ref, rtol=1e-8, atol=1e-12), (ell, m, alm.value[ell, m], ref)

    # batching must not change the result
    alm2 = to_alm(jnp.asarray(positions), weights=jnp.asarray(weights), attrs=attrs, batch_size=7)
    assert np.allclose(alm2.value, alm.value, rtol=1e-10, atol=1e-14)

    # healpy round-trip
    hlm = alm.to_healpy()
    alm3 = AlmField.from_healpy(hlm)
    assert alm3.attrs.ellmax == ellmax
    assert np.allclose(alm3.value, alm.value)

    # jit
    jitted = jax.jit(lambda p, w: to_alm(p, weights=w, attrs=attrs))
    alm4 = jitted(jnp.asarray(positions), jnp.asarray(weights))
    assert np.allclose(alm4.value, alm.value, rtol=1e-10, atol=1e-14)
    print('test_alm_direct OK')


def test_pixel_vs_anafast():
    # Pixel quadrature vs healpy.anafast(iter=0) on a random map
    import healpy as hp
    nside = 32
    ellmax = 2 * nside
    rng = np.random.RandomState(seed=44)
    value = rng.normal(size=12 * nside**2)
    pixel = PixelField(value=jnp.asarray(value), attrs=AngularAttrs(ellmax=ellmax, nside=nside))
    alm = pixel.to_alm()
    ref = hp.map2alm(value, lmax=ellmax, iter=0, use_pixel_weights=False)
    assert np.allclose(alm.to_healpy(), ref, rtol=1e-8, atol=1e-10)

    bin = BinAngular2Spectrum(AngularAttrs(ellmax=ellmax))
    spectrum = compute_angular2_spectrum(pixel, bin=bin)
    cl_ref = hp.anafast(value, lmax=ellmax, iter=0, use_pixel_weights=False)
    assert np.allclose(spectrum.value(), cl_ref, rtol=1e-8, atol=1e-12)
    print('test_pixel_vs_anafast OK')


def test_pixel_backends():
    # 'jax_healpy' (default, FFT over rings) and 'quadrature' (direct summation over pixel centers)
    # must agree with each other and with healpy; and the whole healpix path must be jit-able
    import healpy as hp
    nside, ellmax = 32, 48
    value = np.random.RandomState(seed=50).normal(size=12 * nside**2)
    attrs = AngularAttrs(ellmax=ellmax, nside=nside)
    pixel = PixelField(value=jnp.asarray(value), attrs=attrs)

    ref = hp.map2alm(value, lmax=ellmax, iter=0, use_pixel_weights=False)
    scale = np.max(np.abs(ref))
    alms = {backend: pixel.to_alm(backend=backend) for backend in ['jax_healpy', 'healpy', 'quadrature']}
    for backend, alm in alms.items():
        assert np.max(np.abs(alm.to_healpy() - ref)) / scale < 1e-11, backend
    assert np.max(np.abs(alms['jax_healpy'].value - alms['quadrature'].value)) / scale < 1e-11

    # the host (healpy) backends go through jax.pure_callback, so they must still trace
    base = alms['healpy'].value
    f = jax.jit(lambda m: PixelField(value=m, attrs=attrs).to_alm(backend='healpy').value)
    assert np.allclose(f(jnp.asarray(value)), base)
    g = jax.vmap(lambda m: PixelField(value=m, attrs=attrs).to_alm(backend='healpy').value)
    assert np.allclose(g(jnp.asarray([value, 2. * value]))[1], 2. * base)

    # painting: jax_healpy vec2pix vs healpy vec2pix must give the identical map
    positions = jnp.asarray(random_positions(20000, seed=51))
    particles = ParticleField(positions, attrs=get_mattrs())
    p_jhp = to_pixel(particles, attrs=attrs, backend='jax_healpy')
    p_hp = to_pixel(particles, attrs=attrs, backend='healpy')
    assert np.allclose(p_jhp.value, p_hp.value)

    jf = jax.jit(lambda pos: to_pixel((pos, jnp.ones(pos.shape[0])), attrs=attrs, backend='healpy').value)
    assert np.allclose(jf(positions), p_hp.value)
    vf = jax.vmap(lambda pos: to_pixel((pos, jnp.ones(pos.shape[0])), attrs=attrs, backend='healpy').value)
    half = vf(positions.reshape(2, -1, 3))
    assert np.allclose(half[0] + half[1], p_hp.value)

    # end-to-end jit of particles -> healpix map -> alm (only possible with the jax_healpy backend)
    bin = BinAngular2Spectrum(attrs, edges={'min': 2, 'step': 4})
    f = jax.jit(lambda p: compute_angular2_spectrum(p, bin=bin, method='healpix', backend='jax_healpy').value())
    jitted = f(particles)
    eager = compute_angular2_spectrum(particles, bin=bin, method='healpix', backend='jax_healpy').value()
    assert np.allclose(jitted, eager, rtol=1e-10)

    # ... and it is differentiable
    def loss(weights):
        p = particles.clone(weights=weights)
        return jnp.sum(compute_angular2_spectrum(p, bin=bin, method='healpix', backend='jax_healpy').value())

    grad = jax.grad(loss)(particles.weights)
    assert np.all(np.isfinite(grad)) and np.any(grad != 0.)
    print('test_pixel_backends OK')


def test_gradients():
    # The healpy backends go through jax.pure_callback, which autodiff cannot see through;
    # the transform's gradient is supplied analytically (adjoint = alm2map) and must reproduce
    # the two natively-differentiable backends, and finite differences.
    attrs = AngularAttrs(ellmax=16, nside=16)
    value = jnp.asarray(np.random.RandomState(0).normal(size=attrs.npix))
    # a loss sensitive to the complex phase (|alm|^2 alone is invariant under conjugation,
    # so it would not catch a wrong cotangent convention)
    kre = jnp.asarray(np.random.RandomState(3).normal(size=(attrs.ellmax + 1,) * 2))
    kim = jnp.asarray(np.random.RandomState(4).normal(size=(attrs.ellmax + 1,) * 2))

    def make_loss(backend):
        def loss(m):
            alm = PixelField(value=m, attrs=attrs).to_alm(backend=backend).value
            return jnp.sum(kre * alm.real + kim * alm.imag)
        return loss

    grads = {backend: jax.grad(make_loss(backend))(value) for backend in ['quadrature', 'jax_healpy', 'healpy']}
    ref = grads['quadrature']
    for backend, grad in grads.items():
        assert np.linalg.norm(grad - ref) / np.linalg.norm(ref) < 1e-10, backend

    loss, eps = make_loss('healpy'), 1e-6
    for ipix in [0, 137, 1000]:
        fd = (loss(value.at[ipix].add(eps)) - loss(value.at[ipix].add(-eps))) / (2. * eps)
        assert np.allclose(grads['healpy'][ipix], fd, rtol=1e-6)

    # the analytic vjp must survive jit
    assert np.allclose(jax.jit(jax.grad(make_loss('healpy')))(value), grads['healpy'])

    # end-to-end: particles -> healpix map -> alm -> spectrum, differentiated w.r.t. weights,
    # with the transform on the callback path
    positions = jnp.asarray(random_positions(5000, seed=52))
    particles = ParticleField(positions, attrs=get_mattrs())
    bin = BinAngular2Spectrum(attrs, edges={'min': 2, 'step': 4})

    def spectrum_loss(weights, backend):
        p = particles.clone(weights=weights)
        return jnp.sum(compute_angular2_spectrum(p, bin=bin, method='healpix', backend=backend).value())

    g_hp = jax.grad(partial(spectrum_loss, backend='healpy'))(particles.weights)
    g_jhp = jax.grad(partial(spectrum_loss, backend='jax_healpy'))(particles.weights)
    assert np.all(np.isfinite(g_hp)) and np.any(g_hp != 0.)
    assert np.linalg.norm(g_hp - g_jhp) / np.linalg.norm(g_jhp) < 1e-10
    print('test_gradients OK')


def test_sharded_pixel():
    # Painting must be correct with sharded positions / weights, for both backends:
    # healpy is called through jax.pure_callback, and the scatter-add is pure JAX in both cases.
    # Run with e.g. XLA_FLAGS=--xla_force_host_platform_device_count=4 to get several devices.
    from jax.sharding import NamedSharding, PartitionSpec as P
    ndevices = len(jax.devices())
    if ndevices < 2:
        print(f'test_sharded_pixel SKIPPED (1 device; rerun with XLA_FLAGS=--xla_force_host_platform_device_count=4)')
        return
    attrs = AngularAttrs(ellmax=32, nside=32)
    positions = jnp.asarray(random_positions(20000, seed=51))
    weights = jnp.ones(positions.shape[0])
    ref = to_pixel((positions, weights), attrs=attrs, backend='healpy').value

    mesh = jax.make_mesh((ndevices,), ('i',), axis_types=(jax.sharding.AxisType.Auto,))
    positions = jax.device_put(positions, NamedSharding(mesh, P('i', None)))
    weights = jax.device_put(weights, NamedSharding(mesh, P('i')))
    assert positions.addressable_shards[0].data.shape[0] == 20000 // ndevices
    for backend in ['jax_healpy', 'healpy']:
        f = jax.jit(lambda pos, wts: to_pixel((pos, wts), attrs=attrs, backend=backend).value)
        assert np.allclose(f(positions, weights), ref), backend
    print(f'test_sharded_pixel OK ({ndevices} devices)')


def make_cl(ells, amplitude=1e-2):
    ells = np.asarray(ells, dtype='f8')
    return amplitude / (1. + (ells / 10.)**2)


def test_fkp_spectrum(plot=False):
    # Full-sky Poisson-sampled catalog: FKP estimator (norm + shot noise) vs the realization's anafast
    import healpy as hp
    nside = 64
    ellmax = 32
    ellmax_gen = 64
    rng = np.random.RandomState(seed=45)
    cl_in = make_cl(np.arange(ellmax_gen + 1))
    np.random.seed(46)  # synfast uses global numpy random state
    delta = hp.synfast(cl_in, nside, lmax=ellmax_gen)
    density = np.clip(1. + delta, 0., None)
    delta_eff = density / density.mean() - 1.  # actually sampled field (after clipping)

    npix = 12 * nside**2
    ndata, nrandoms = 200000, 1000000
    prob = density / density.sum()
    counts = rng.multinomial(ndata, prob)
    ipix = np.repeat(np.arange(npix), counts)
    data_positions = np.column_stack(hp.pix2vec(nside, ipix))
    randoms_positions = random_positions(nrandoms, seed=47)

    mattrs = get_mattrs()
    data = ParticleField(jnp.asarray(data_positions), attrs=mattrs)
    randoms = ParticleField(jnp.asarray(randoms_positions), attrs=mattrs)
    fkp = FKPField(data, randoms)

    bin = BinAngular2Spectrum(AngularAttrs(ellmax=ellmax, nside=nside), edges={'min': 2, 'step': 4})
    spectrum = compute_angular2_spectrum(fkp, bin=bin)
    norm = compute_fkp_angular2_normalization(fkp, bin=bin)
    num_shotnoise = compute_fkp_angular2_shotnoise(fkp, bin=bin)
    spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)

    # normalization: data x randoms (default), disjoint random splits, and the single-catalog
    # self-pair-debiased estimate must all agree with the analytic (n / 4 pi)^2
    alpha = data.sum() / randoms.sum()
    norms = {'data x randoms': norm,
             'split': compute_fkp_angular2_normalization(fkp, bin=bin, split=42),
             'randoms only': alpha**2 * compute_fkp_angular2_normalization(randoms, bin=bin)}
    norm_ref = (ndata / (4. * np.pi))**2
    for name, value in norms.items():
        assert np.allclose(value, norm_ref, rtol=0.02), (name, value, norm_ref)

    # reference: pseudo-Cl of the actually-sampled realization (same seed, no cosmic variance)
    cl_ref = hp.anafast(delta_eff, lmax=ellmax, iter=0, use_pixel_weights=False)
    num_ref = (2 * np.arange(ellmax + 1) + 1) * cl_ref
    ref = np.asarray(bin.wbin) @ num_ref / np.asarray(bin.nmodes)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        ax.plot(spectrum.ell, spectrum.value(), label='FKP direct')
        ax.plot(spectrum.ell, ref, label='anafast of sampled map')
        ax.legend()
        plt.show()

    assert np.allclose(spectrum.value(), ref, rtol=0.15, atol=2e-4), np.column_stack([np.asarray(spectrum.value()), ref])

    # I/O round-trip
    fn = dirname / 'angular2_spectrum.h5'
    fn.parent.mkdir(parents=True, exist_ok=True)
    spectrum.write(fn)
    spectrum2 = read(fn)
    assert isinstance(spectrum2, Angular2Spectrum)
    assert np.allclose(spectrum2.value(), spectrum.value())
    print('test_fkp_spectrum OK')


def test_generate_spectrum2_alm(nmocks=150):
    # The generator's convention must match compute_angular2_spectrum: the ensemble mean of the
    # measured C_ell reproduces the input, and unitary_amplitude reproduces it exactly.
    from jaxpower import generate_spectrum2_alm

    ellmax = 32
    attrs = AngularAttrs(ellmax=ellmax, nside=32)
    cl = lambda ell: 1e-3 / (1. + (np.asarray(ell) / 8.)**2)
    bin = BinAngular2Spectrum(attrs, edges={'min': 0, 'step': 4})
    ells, wbin = np.arange(ellmax + 1), np.asarray(bin.wbin)
    # input, averaged over each band with the (2l+1) weights the estimator uses
    ref = np.array([np.sum((2 * ells + 1) * cl(ells) * w) / np.sum((2 * ells + 1) * w) for w in wbin])

    measured = np.array([np.asarray(compute_angular2_spectrum(generate_spectrum2_alm(attrs, cl=cl, seed=imock), bin=bin).value())
                         for imock in range(nmocks)])
    mean, err = measured.mean(axis=0), measured.std(axis=0) / np.sqrt(nmocks)
    nsig = (mean - ref) / err
    assert np.all(np.abs(nsig) < 4.), np.column_stack([ref, mean, err, nsig])

    # no scatter, and exact, when the modulus is fixed
    unitary = np.array([np.asarray(compute_angular2_spectrum(generate_spectrum2_alm(attrs, cl=cl, seed=imock, unitary_amplitude=True), bin=bin).value())
                        for imock in range(3)])
    assert np.allclose(unitary, ref, rtol=1e-10), unitary

    # m > ell must be zero, and a_l0 real
    alm = generate_spectrum2_alm(attrs, cl=cl, seed=7).value
    upper = np.arange(ellmax + 1)[:, None] < np.arange(ellmax + 1)[None, :]
    assert np.all(np.asarray(alm)[upper] == 0.)
    assert np.allclose(np.asarray(alm)[:, 0].imag, 0.)

    # jit, with an explicit key (an int seed cannot be traced, as elsewhere in mock.py)
    f = jax.jit(lambda key: generate_spectrum2_alm(attrs, cl=cl, seed=key).value)
    assert np.allclose(f(jax.random.key(3)), generate_spectrum2_alm(attrs, cl=cl, seed=jax.random.key(3)).value)
    print('test_generate_spectrum2_alm OK (residuals in sigma: {})'.format(', '.join(f'{s:+.1f}' for s in nsig)))


def test_generate_spectrum3(nmocks=120):
    # local construction delta = g + alpha2_local (g^2 - <g^2>) has b = 2 alpha2_local (C1C2 + C2C3 + C3C1),
    # exactly constant for a white C_l. Bin from ell >= 2: bands holding the monopole fall short,
    # since subtracting <g^2> removes the quadratic term's ell = 0 piece by construction.
    from jaxpower import generate_spectrum2_alm, generate_spectrum3_alm, generate_spectrum3_mesh, generate_gaussian_mesh

    ellmax, nside, amplitude, alpha2_local = 8, 32, 1., 0.05
    attrs = AngularAttrs(ellmax=ellmax, nside=nside)
    cl = lambda ell: amplitude + 0. * np.asarray(ell)
    bin3 = BinAngular3Spectrum(attrs, edges={'min': 2, 'step': 3})
    target = 6. * alpha2_local * amplitude**2

    # antisymmetric in alpha2_local: cancels the pure-Gaussian noise and the even orders
    diff = []
    for imock in range(nmocks):
        alms = [generate_spectrum3_alm(attrs, cl=cl, alpha2_local=sign * alpha2_local, seed=imock) for sign in (1, -1)]
        values = [np.asarray(compute_angular3_spectrum(alm, bin=bin3).value()).real for alm in alms]
        diff.append((values[0] - values[1]) / 2.)
    diff = np.array(diff)
    mean, err = diff.mean(axis=0), diff.std(axis=0) / np.sqrt(nmocks)
    nsig = (mean - target) / err
    assert np.all(np.abs(nsig) < 4.), np.column_stack([mean, err, nsig])

    # alpha2_local = 0 must reproduce the Gaussian generators exactly
    assert np.allclose(generate_spectrum3_alm(attrs, cl=cl, alpha2_local=0., seed=3).value,
                       generate_spectrum2_alm(attrs, cl=cl, seed=3).to_pixel().to_alm().value)
    mattrs = MeshAttrs(meshsize=32, boxsize=1000.)
    power = lambda kvec: 1e4 * jnp.exp(-jnp.sqrt(sum(kk**2 for kk in kvec)) / 0.1)
    assert np.allclose(generate_spectrum3_mesh(mattrs, power=power, alpha2_local=0., seed=3).value,
                       generate_gaussian_mesh(mattrs, power=power, seed=3).value)
    # the quadratic term is mean-free, so the mesh mean is unchanged
    gaussian = generate_gaussian_mesh(mattrs, power=power, seed=3)
    mesh = generate_spectrum3_mesh(mattrs, power=power, alpha2_local=1e-3, seed=3)
    assert np.allclose(jnp.mean(mesh.value), jnp.mean(gaussian.value), atol=1e-8 * jnp.std(gaussian.value))
    print('test_generate_spectrum3 OK (residuals in sigma: {})'.format(', '.join(f'{s:+.1f}' for s in nsig)))


def test_inject_spectrum3(nmocks=100):
    # Injecting a bispectrum into one band triplet must be recovered there, and leave the others at zero.
    # A bin-restricted target is separable, so the quadratic kernel B/(3 P1 P2) costs a few transforms.
    from jaxpower import generate_spectrum3_alm, generate_spectrum3_mesh
    from jaxpower import BinMesh3SpectrumPoles, compute_mesh3_spectrum

    def antisymmetric(generate, measure):
        # cancels the pure-Gaussian noise and the even orders in the injected amplitude
        out = []
        for imock in range(nmocks):
            values = [measure(generate(sign, imock)) for sign in (1, -1)]
            out.append((values[0] - values[1]) / 2.)
        out = np.array(out)
        return out.mean(axis=0), out.std(axis=0) / np.sqrt(len(out))

    # --- sphere
    attrs = AngularAttrs(ellmax=8, nside=32)
    edges = {'min': 2, 'step': 3}
    bin3 = BinAngular3Spectrum(attrs, edges=edges)
    cl = lambda ell: 1. + 0. * np.asarray(ell)
    target, amplitude = (0, 1, 1), 0.4
    mean, err = antisymmetric(
        lambda sign, imock: generate_spectrum3_alm(attrs, cl=cl, edges=edges, seed=imock,
                                                   spectrum3={target: sign * amplitude}),
        lambda alm: np.asarray(compute_angular3_spectrum(alm, bin=bin3).value()).real)
    for ibin, bands in enumerate(np.asarray(bin3.ibands)):
        expected = amplitude if tuple(bands) == target else 0.
        assert abs(mean[ibin] - expected) < 4. * err[ibin], (bands, mean[ibin], err[ibin], expected)

    # --- mesh; compute_mesh3_spectrum already returns the bispectrum, so no normalization applies
    # (compute_box3_normalization is for a density field with a mean, not a zero-mean delta)
    mattrs = MeshAttrs(meshsize=48, boxsize=1000.)
    p0 = 1e4
    power = lambda kvec: p0 + 0. * jnp.sqrt(sum(kk**2 for kk in kvec))
    kedges = np.array([0.05, 0.10, 0.15])
    binmesh = BinMesh3SpectrumPoles(mattrs, edges=kedges, basis='scoccimarro', ells=[0])
    amplitude = 1e8  # ~1% of P^2, so the O(amplitude^3) terms stay negligible
    mean, err = antisymmetric(
        lambda sign, imock: generate_spectrum3_mesh(mattrs, power=power, edges=kedges, seed=imock,
                                                    spectrum3={target: sign * amplitude}),
        lambda mesh: np.asarray(compute_mesh3_spectrum(mesh, bin=binmesh, los='z').get(ells=0).value()))
    for ibin in range(len(mean)):
        expected = amplitude if ibin == 2 else 0.  # (0, 1, 1) is the last sorted triplet of 2 bands
        assert abs(mean[ibin] - expected) < max(4. * err[ibin], 0.02 * amplitude), (ibin, mean[ibin], err[ibin], expected)
    print('test_inject_spectrum3 OK')


def test_angular3_gaunt():
    # The filtered-map estimator must reproduce the exact Gaunt sum
    #   num_ijk = sum_{l in bands} sum_{m1m2m3} G^{m1m2m3}_{l1l2l3} a a a,
    # up to the healpix quadrature of the triple product, which must converge as nside grows.
    from jaxpower.utils import wigner_3j

    ellmax = 6
    rng = np.random.RandomState(2)
    alm = np.zeros((ellmax + 1,) * 2, dtype=complex)
    for ell in range(ellmax + 1):
        alm[ell, 0] = rng.normal()
        for m in range(1, ell + 1):
            alm[ell, m] = rng.normal() + 1j * rng.normal()

    def get_alm(ell, m):  # reality condition for m < 0
        return alm[ell, m] if m >= 0 else (-1)**m * np.conj(alm[ell, -m])

    bin = BinAngular3Spectrum(AngularAttrs(ellmax=ellmax, nside=16), edges={'min': 0, 'step': 3})
    wbin, ells = np.asarray(bin.wbin), np.arange(ellmax + 1)
    ref = []
    for i, j, k in np.asarray(bin.ibands):
        total = 0.
        for ell1 in ells[wbin[i]]:
            for ell2 in ells[wbin[j]]:
                for ell3 in ells[wbin[k]]:
                    w000 = wigner_3j(ell1, ell2, ell3, 0, 0, 0)
                    if abs(w000) < 1e-14: continue
                    h = np.sqrt((2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) / (4. * np.pi)) * w000
                    total += h * sum(wigner_3j(ell1, ell2, ell3, m1, m2, -(m1 + m2))
                                     * get_alm(ell1, m1) * get_alm(ell2, m2) * get_alm(ell3, -(m1 + m2))
                                     for m1 in range(-ell1, ell1 + 1) for m2 in range(-ell2, ell2 + 1)
                                     if abs(m1 + m2) <= ell3)
        ref.append(total.real)
    ref = np.array(ref) / np.asarray(bin.nmodes)

    errs = []
    for nside in [16, 32, 64]:
        attrs = AngularAttrs(ellmax=ellmax, nside=nside)
        bin = BinAngular3Spectrum(attrs, edges={'min': 0, 'step': 3})
        est = np.asarray(compute_angular3_spectrum(AlmField(value=jnp.asarray(alm), attrs=attrs), bin=bin).value()).real
        errs.append(np.max(np.abs(est / ref - 1.)))
    assert errs[0] < 2e-2, errs
    # quadrature error must fall off as ~1/nside^2
    for prev, cur in zip(errs[:-1], errs[1:]):
        assert cur < prev / 3., errs
    print('test_angular3_gaunt OK (quadrature error {})'.format(', '.join(f'{e:.1e}' for e in errs)))


def test_angular3_shotnoise():
    # A single particle: only the p = q = r coincidence contributes, so num_raw = w^3 / (4 pi)
    # exactly, which validates both N_ijk = sum h^2 and the S3 (three coincident points) shot noise.
    for nside, ellmax in [(32, 16), (64, 24)]:
        attrs = AngularAttrs(ellmax=ellmax, nside=nside)
        w = 1.7
        particles = ParticleField(jnp.asarray([[0.3, -0.5, 0.81]]), weights=jnp.asarray([w]), attrs=get_mattrs())

        bin3 = BinAngular3Spectrum(attrs, edges={'min': 0, 'step': 6})
        raw = np.asarray(compute_angular3_spectrum(particles, bin=bin3).value()).real
        shot = np.asarray(compute_fkp_angular3_shotnoise(particles, bin=bin3))
        assert np.allclose(raw, w**3 / (4. * np.pi), rtol=1e-3), raw
        assert np.allclose(shot, w**3 / (4. * np.pi), rtol=1e-12), shot

        # same identity at 2 points, where it is exact (no quadrature involved)
        bin2 = BinAngular2Spectrum(attrs, edges={'min': 0, 'step': 6})
        raw2 = np.asarray(compute_angular2_spectrum(particles, bin=bin2).value())
        shot2 = np.asarray(compute_fkp_angular2_shotnoise(particles, bin=bin2))
        assert np.allclose(raw2, w**2 / (4. * np.pi), rtol=1e-10), raw2
        assert np.allclose(shot2, w**2 / (4. * np.pi), rtol=1e-12), shot2

    # normalization: uniform full-sky randoms -> nbar^3
    attrs = AngularAttrs(ellmax=16, nside=32)
    bin3 = BinAngular3Spectrum(attrs, edges={'min': 0, 'step': 6})
    nrandoms = 200000
    randoms = ParticleField(jnp.asarray(random_positions(nrandoms, seed=61)), attrs=get_mattrs())
    norm = compute_fkp_angular3_normalization(randoms, bin=bin3, split=62)
    assert np.allclose(norm, (nrandoms / (4. * np.pi))**3, rtol=0.02), (norm, (nrandoms / (4. * np.pi))**3)
    print('test_angular3_shotnoise OK')


def test_angular3_window(nmocks=40):
    # (a) for a full-sky uniform mask the window must reduce to the identity;
    # (b) a constant mask must too, since numerator and normalization scale alike;
    # (c) against masked mocks with a known bispectrum.
    import healpy as hp
    nside, ellmax = 16, 6
    attrs = AngularAttrs(ellmax=ellmax, nside=nside)
    bin = BinAngular3Spectrum(attrs, edges={'min': 0, 'step': 3})

    for value in [1., 2.5]:
        mask = PixelField(value=value * jnp.ones(attrs.npix), attrs=attrs)
        wmat = np.asarray(compute_angular3_spectrum_window(mask, bin=bin).value())
        assert np.max(np.abs(wmat - np.eye(*wmat.shape))) < 1e-4, (value, wmat)

    # local-type non-Gaussianity: delta = g + f (g^2 - <g^2>) has b = 2f(C1C2 + C2C3 + C3C1),
    # which for a white C_l = A is the constant 6 f A^2, so the window's piecewise-constant
    # theory assumption is exact.
    A, f = 1., 0.05
    vec = np.column_stack(hp.pix2vec(nside, np.arange(attrs.npix)))
    mask = (np.abs(vec[:, 2]) > 0.3).astype('f8')
    wmat = compute_angular3_spectrum_window(PixelField(value=jnp.asarray(mask), attrs=attrs), bin=bin)
    pred = np.asarray(wmat.value()) @ np.full(len(np.asarray(wmat.theory.nmodes)), 6. * f * A**2)
    norm = np.sum(mask**3) * attrs.pixarea / (4. * np.pi)

    def estimator(m):
        return np.asarray(compute_angular3_spectrum(PixelField(value=jnp.asarray(m), attrs=attrs), bin=bin).value()).real / norm

    cl, diff = np.full(ellmax + 1, A), []
    for imock in range(nmocks):
        np.random.seed(2000 + imock)
        g = hp.synfast(cl, nside, lmax=ellmax)
        h = g**2 - np.mean(g**2)
        # antisymmetric in f: cancels both the (large) pure-Gaussian noise and the even-order terms,
        # leaving the O(f) bispectrum the window predicts
        diff.append((estimator(mask * (g + f * h)) - estimator(mask * (g - f * h))) / 2.)
    diff = np.array(diff)
    mean, err = diff.mean(axis=0), diff.std(axis=0) / np.sqrt(nmocks)
    nsig = (mean - pred) / err
    assert np.all(np.abs(nsig) < 4.), np.column_stack([pred, mean, err, nsig])
    print('test_angular3_window OK (mock residuals in sigma: {})'.format(', '.join(f'{s:+.1f}' for s in nsig)))


def test_angular3_window_cross():
    # Window for three (possibly different) masks. The theory is indexed by *ordered* band triplets,
    # merged under the stabilizer of the field triple, so the number of theory bins grows as the
    # fields become distinguishable.
    import healpy as hp
    nside, ellmax = 16, 6
    attrs = AngularAttrs(ellmax=ellmax, nside=nside)
    bin = BinAngular3Spectrum(attrs, edges={'min': 0, 'step': 3})
    vec = np.column_stack(hp.pix2vec(nside, np.arange(attrs.npix)))
    mask = PixelField(value=jnp.asarray((np.abs(vec[:, 2]) > 0.3).astype('f8')), attrs=attrs)

    wauto = compute_angular3_spectrum_window(mask, bin=bin)
    wcross = compute_angular3_spectrum_window(mask, mask, mask, bin=bin, fields=(0, 1, 2))
    wsemi = compute_angular3_spectrum_window(mask, None, mask, bin=bin, fields=(0, 0, 1))
    nbins = [np.asarray(w.value()).shape[1] for w in (wauto, wsemi, wcross)]
    assert nbins == [4, 6, 8], nbins  # sorted / first-two-symmetric / fully ordered

    # the same mask everywhere is a symmetric problem, so a constant (symmetric) theory must give
    # the same prediction however the fields are labelled
    preds = [np.asarray(w.value()) @ np.ones(np.asarray(w.value()).shape[1]) for w in (wauto, wsemi, wcross)]
    for pred in preds[1:]:
        assert np.allclose(pred, preds[0], rtol=1e-10), (pred, preds[0])

    # three genuinely different masks: the estimator symmetrizes over the legs, so relabelling
    # which mask comes first must leave a symmetric theory's prediction unchanged
    mask2 = PixelField(value=jnp.asarray((vec[:, 0] > -0.5).astype('f8')), attrs=attrs)
    mask3 = PixelField(value=jnp.asarray(1. + 0.5 * vec[:, 1]), attrs=attrs)
    preds = []
    for masks in [(mask, mask2, mask3), (mask2, mask3, mask), (mask3, mask2, mask)]:
        wmat = np.asarray(compute_angular3_spectrum_window(*masks, bin=bin, fields=(0, 1, 2)).value())
        preds.append(wmat @ np.ones(wmat.shape[1]))
    for pred in preds[1:]:
        assert np.allclose(pred, preds[0], rtol=1e-10), (pred, preds[0])
    print('test_angular3_window_cross OK (theory bins {})'.format(nbins))


def test_window(plot=False):
    # Mode-coupling matrix on a galactic-cut mask, vs mean pseudo-Cl of masked Gaussian skies
    import healpy as hp
    nside = 64
    ellmax = 32
    ellmax_in = 48
    npix = 12 * nside**2
    vec = np.column_stack(hp.pix2vec(nside, np.arange(npix)))
    mask = (np.abs(vec[:, 2]) > 0.25).astype('f8')

    aattrs = AngularAttrs(ellmax=ellmax, nside=nside)
    bin = BinAngular2Spectrum(aattrs, edges={'min': 2, 'step': 4})
    mask_pixel = PixelField(value=jnp.asarray(mask), attrs=aattrs)
    fsky = np.mean(mask)
    wmat = compute_angular2_spectrum_window(mask_pixel, edgesin={'max': ellmax_in + 1}, bin=bin, norm=fsky)

    cl_in = make_cl(np.arange(ellmax_in + 1))
    mean = compute_angular2_spectrum_mean(wmat, make_cl)

    np.random.seed(49)
    nmocks = 20
    measured = []
    for imock in range(nmocks):
        delta = hp.synfast(cl_in, nside, lmax=ellmax_in)
        pixel = PixelField(value=jnp.asarray(mask * delta), attrs=aattrs)
        spectrum = compute_angular2_spectrum(pixel, bin=bin)
        measured.append(np.asarray(spectrum.clone(norm=fsky).value()))
    measured = np.mean(measured, axis=0)

    if plot:
        from matplotlib import pyplot as plt
        ax = plt.gca()
        ax.plot(mean.ell, mean.value(), label='window @ theory')
        ax.plot(mean.ell, measured, label='mean of masked mocks')
        ax.plot(mean.ell, make_cl(np.asarray(mean.ell)), label='input', ls=':')
        ax.legend()
        plt.show()

    # fsky vs alm-based normalization of the binary mask: W_L-sum = int nbar^2 dOmega / 4pi = fsky
    from jaxpower.angular import _compute_cross_power
    alm_mask = mask_pixel.to_alm(ellmax=ellmax + ellmax_in)
    ellsw = np.arange(ellmax + ellmax_in + 1)
    fsky_alm = np.sum(np.asarray(_compute_cross_power(alm_mask, alm_mask))) / (4. * np.pi)
    assert np.allclose(fsky_alm, fsky, rtol=0.05)
    assert np.allclose(mean.value(), measured, rtol=0.1, atol=2e-4), np.column_stack([np.asarray(mean.value()), measured])
    print('test_window OK')


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)
    utils.mkdir(dirname)

    test_wigner3j000()
    test_alm_direct()
    test_pixel_vs_anafast()
    test_pixel_backends()
    test_gradients()
    test_sharded_pixel()
    test_fkp_spectrum()
    test_window()
    test_generate_spectrum2_alm()
    test_generate_spectrum3()
    test_inject_spectrum3()
    test_angular3_gaunt()
    test_angular3_shotnoise()
    test_angular3_window()
    test_angular3_window_cross()
