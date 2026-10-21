import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from matplotlib import pyplot as plt
from jax import numpy as jnp

from jaxpower.pt import ProjectToPoles, ProjectToSell, compute_spt_matter_1loop, prepare_spectrum2_redshift_tracer, spectrum2_redshift_tracer, spectrum3_redshift_tracer


def test_spectrum2():

    k = jnp.logspace(-2.5, -0.3, 80)
    to_poles = ProjectToPoles(ells=(0, 2, 4))

    def pk_callable(q):
        return jnp.where(q > 0.0, q * jnp.exp(-q / 0.25), 0.0)

    def pknow_callable(q):
        return jnp.where(q > 0.0, q * jnp.exp(-q / 0.28), 0.0)

    from cosmoprimo import Cosmology, PowerSpectrumBAOFilter
    cosmo = Cosmology(h=jnp.array(0.7), engine='eisenstein_hu')
    z = 1.
    pk_interpolator = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    filter = PowerSpectrumBAOFilter(pk_interpolator, engine='wallish2018', cosmo=cosmo, cosmo_fid=cosmo)
    pknow_interpolator = filter.smooth_pk_interpolator()

    kin = jnp.logspace(-4, 1, 130)
    pkin = pk_interpolator(kin)
    pkin_now = pknow_interpolator(kin)
    pk_callable = lambda k: jnp.interp(k, kin, pkin, left=0., right=0.)
    pknow_callable = lambda k: jnp.interp(k, kin, pkin_now, left=0., right=0.)
    f = cosmo.growth_rate(z)

    if False:
        spt = compute_spt_matter_1loop(k, pk_callable)
        fig, ax = plt.subplots()
        for name in ['P11', 'Pdd', 'Pdt', 'Ptt']:
            ax.plot(k, k * spt[name], label=name)
        ax.legend(frameon=False)
        plt.show()

    bias_params = {"a": {"b1": 2.0, "b2": 0.5, "bs": -0.3, "b3nl": 0.1, "X_FoG": 1.},
                   "b": {"b1": 1.7, "b2": 0.2, "bs": -0.2, "b3nl": 0.05, "X_FoG": 1.}}

    if False:
        spectrum = spectrum2_real_tracer(k, pk_callable, bias_params=bias_params)
        fig, ax = plt.subplots()
        for name in ['Pdd_ab', 'Ptt']:
            ax.plot(k, k * spectrum[name], label=name)
        ax.legend(frameon=False)
        plt.show()

    if False:
        table, table_now = prepare_spectrum2_redshift_tracer(k, pk_callable, pknow_callable, kbao=1.0 / 105.)
        spectrum = spectrum2_redshift_tracer(to_poles.mu, table, table_now, f, bias_params)
        poles = to_poles(spectrum)
        fig, ax = plt.subplots()
        for ill, ell in enumerate(to_poles.ells):
            ax.plot(k, k * poles[ill])
        plt.show()

    bias_params = {"a": {"b1": 2.0, "b2": 0.5, "bs": -0.3, "c1": 0.1, "c2": 0.2, 'X_FoG': 2., 'snb0': 0.1, 'sn0': 0.1},
                   "b": {"b1": 1.7, "b2": 0.2, "bs": -0.2, "c1": 0.05, "c2": 0.2, 'X_FoG': 2., 'snb0': 0.1, 'sn0': 0.1}}
    to_Sell = ProjectToSell(ells=[(0, 0, 0), (2, 0, 2)])
    k1norm = k2norm = jnp.linspace(0.01, 0.2, 20)
    k1vec, k2vec = k1norm[:, None, None] * to_Sell.k1hat[None, ...], k2norm[:, None, None] * to_Sell.k2hat[None, ...]
    print(k1vec.shape, k2vec.shape)
    spectrum = spectrum3_redshift_tracer(k1vec, k2vec, pk_callable, pknow_callable, f=f, bias_params=bias_params)
    poles = to_Sell(spectrum)
    fig, ax = plt.subplots()
    for ill, ell in enumerate(to_Sell.ells):
        ax.plot(k1norm, k1norm**2 * poles[ill])
    plt.show()


def test_partition_bias(atol=1e-12, size=200, seed=0):
    """The two-block partition sum must reproduce ``jaxpower.pt``'s own K_2 and K_3.

    Both were validated against Z_2 and Z_3, so they are the reference, and the sum is what continues them into K_4 and K_5.
    Checked term by term -- the local part alone with ``bs = 0``, the tidal part alone with ``b2' = 0``, and the two together
    -- because the two failures the continuation invites cancel in neither: dropping the block factorials leaves the local
    part short by 2/3 at n = 3, and putting F where G belongs leaves the tidal part wrong at n = 3 while both stay exact at
    n = 2.
    """
    import math
    from jaxpower.pt import fg_tables, _set_partitions, _sigma2, BiasModel
    rng = np.random.default_rng(seed)
    b1 = 2.0
    # (b2, bs) with, in turn, no tidal operator, no local one (b2' = b2 + 2 bs / 3 = 0), both
    cases = {'local': (1.0, 0.), 'tidal': (0.8, -1.2), 'both': (1.0, -0.7)}
    for n in (2, 3):
        vs = [jnp.asarray(rng.normal(scale=0.1, size=(size, 3))) for _ in range(n)]
        F, G, ksum = fg_tables(vs)
        mask = (1 << n) - 1
        local = tidal = 0.
        for part in _set_partitions(mask):
            if len(part) != 2:
                continue
            block_a, block_b = part
            weight = float(math.factorial(bin(block_a).count('1')) * math.factorial(bin(block_b).count('1')))
            local = local + weight * np.asarray(F[block_a]) * np.asarray(F[block_b])
            tidal = tidal + weight * np.asarray(_sigma2(ksum[block_a], ksum[block_b])) * np.asarray(G[block_a]) * np.asarray(G[block_b])
        linear = np.asarray(BiasModel(b1=b1)(mask, F, G, ksum, vs))
        for name, (b2, bs) in cases.items():
            reference = np.asarray(BiasModel(b1=b1, b2=b2, bs=bs)(mask, F, G, ksum, vs)) - linear
            prediction = ((b2 + 2. / 3. * bs) * local + bs * tidal) / math.factorial(n)
            # against the sample's typical magnitude: the two sides group the same terms differently, so a per-element
            # ratio would report round-off on configurations that happen to cancel
            err = float(np.max(np.abs(reference - prediction)) / np.median(np.abs(reference)))
            print(f'K_{n} ({name}): {err:.2e}')
            assert err < atol, f'K_{n} ({name}) disagrees with the partition sum by {err:.3e}'


if __name__ == '__main__':

    test_spectrum2()
    test_partition_bias()
