from matplotlib import pyplot as plt
from jax import numpy as jnp

from jaxpower.pt import ProjectToMultipoles, ProjectToSell, compute_spt_matter_1loop, prepare_spectrum2_redshift_tracer, spectrum2_redshift_tracer, spectrum3_redshift_tracer


def test_spectrum2():

    k = jnp.logspace(-2.5, -0.3, 80)
    to_poles = ProjectToMultipoles(ells=(0, 2, 4))

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

    bias_params = {"a": {"b1": 2.0, "b2": 0.5, "bs": -0.3, "c1": 0.1, "c2": 0.2, 'X_FoG': 2., 'Bshot': 0.1, 'Pshot': 0.1},
                "b": {"b1": 1.7, "b2": 0.2, "bs": -0.2, "c1": 0.05, "c2": 0.2, 'X_FoG': 2., 'Bshot': 0.1, 'Pshot': 0.1}}
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


if __name__ == '__main__':

    test_spectrum2()