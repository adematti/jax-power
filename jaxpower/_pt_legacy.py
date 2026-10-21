r"""The retired one-loop tracer trispectrum, kept because nothing replaces it exactly.

``spectrum4_redshift_tracer`` was removed from :mod:`jaxpower.pt` by the cov3 rewrite (24e25c6)
in favour of the labelled-tree construction ``make_spectrum_redshift_tracer(4, ...)``. The two
agree on the deterministic tree trispectrum to machine precision (8.9e-16 on random
configurations), so what only this one carries is:

* the IR resummation of the legs, which the general construction does not apply;
* the ``0.25 (leg_i leg_j)`` stochastic term, dimensionally inconsistent and therefore zeroed
  wherever a covariance uses it, but part of what the campaign fits;
* the EFT counterterms ``c1`` / ``c2`` on the four-point kernel.

``claude_abacus_analytic_cov/theory.py`` already documented it as living here; the module was
simply never written, which left ``claude_bk_cov_new/fitall.py`` unable to import it and so
unrunnable against this checkout. Restored verbatim from ``24e25c6^`` -- not maintained, not
re-derived, and not to be used for new work.
"""

from functools import partial

import numpy as np
import jax
from jax import numpy as jnp

from .pt import *  # noqa: F401,F403
from .pt import _format_bias_params


def spectrum4_redshift_tracer(k1vec, k2vec, k3vec, pk_callable, pknow_callable, f, bias_params, shot=0., sigma2v=None, sigma2=None, sigma2_delta=None, damping='lor'):
    """
    JAX-friendly tree-level cross-trispectrum T^{abcd}(k1, k2, k3, k4),
    with k4 = -k1 - k2 - k3, including the same wiggle damping as in spectrum3.

    Parameters
    ----------
    k1vec, k2vec, k3vec : array_like[..., 3]
        Wavevectors of the first three legs.
    f, sigma2v, sigma2, sigma2_delta : float
        RSD / IR / FoG parameters.
    bias_params : dict
        Mapping field -> parameters.
    pk_callable : callable
        Callable returning the linear power spectrum P(k).
    pknow_callable : callable
        Callable returning the no-wiggle linear power spectrum P_now(k).
    damping : {None, 'lor', 'exp', 'vdg'}, default=None
        Optional phenomenological FoG damping on the full trispectrum.
    shot : float, default=0.
        Constant shot-noise contribution.

    Returns
    -------
    trispectrum : array_like
        Tree-level trispectrum model.
    """
    fields, bias_params = _format_bias_params(bias_params, nfields=4)
    a, b, c, d = fields

    if sigma2v is None:
        sigma2v = compute_sigma2v(pk_callable)
    if sigma2 is None or sigma2_delta is None:
        sigma2, sigma2_delta = compute_sigma2ir(pknow_callable)

    def _get_bias_params(field, names=None):
        pars = bias_params[field]
        if names is None:
            names = ['b1', 'b2', 'bs', 'c1', 'c2', 'snb0', 'sn0', 'X_FoG']
        if isinstance(names, str):
            return pars[names]
        return [pars[name] for name in names]

    def _sn3(field):
        """Third stochastic moment: the amplitude of a three-point coincidence.

        Optional. Defaults to ``snb0**2``, the relation a Poisson process satisfies -- but
        only a Poisson process does. For a halo-occupation tracer the second and third
        factorial moments of N are independent numbers, and measuring them on the mock's own
        halo catalog (see ``claude_abacus_analytic_cov/hod_moments.py``) gives
        ``<N(N-1)(N-2)>/<N> / nbar^2`` for this and ``<N(N-1)>/<N> / nbar`` for ``snb0``;
        for a DESI-like LRG HOD the two differ by ~50%, so the Poisson relation is not a
        harmless default. Pass ``sn3`` in the bias dict to override it.
        """
        pars = bias_params[field]
        if 'sn3' in pars:
            return pars['sn3']
        return pars['snb0']**2

    def _norm(kvec):
        return jnp.sqrt(jnp.sum(kvec**2, axis=-1))

    def _mu(kvec, knorm):
        return jnp.where(knorm > 0., kvec[..., 2] / knorm, 0.)

    def _xcos(kivec, kjvec, ki, kj):
        denom = ki * kj
        return jnp.where(denom > 0., jnp.sum(kivec * kjvec, axis=-1) / denom, 0.)

    def _safe_div(num, denom):
        # Drops the contribution at denom == 0 -- e.g. the trispectrum's
        # "snake" (2-2-1-1) term has a genuine squeezed/collinear pole when
        # an internal momentum q = k_i + k_l vanishes (a real kinematic
        # configuration, not a formula error); this avoids 0/0 -> NaN there.
        return jnp.where(denom > 0., num / jnp.where(denom > 0., denom, 1.), 0.)

    def _Z1(field, mu):
        b1 = _get_bias_params(field, 'b1')
        return b1 + f * mu**2

    def _Z1eft(field, k, mu):
        # Same counterterm basis as spectrum3_redshift_tracer's _Z1eft; see the note there.
        c1, c2 = _get_bias_params(field, ['c1', 'c2'])
        return _Z1(field, mu) - (c1 * mu**2 + c2 * mu**4) * k**2

    def _F2(ki, kj, xij):
        return 5. / 7. + xij / 2. * (_safe_div(ki, kj) + _safe_div(kj, ki)) + 2. / 7. * xij**2

    def _G2(ki, kj, xij):
        return 3. / 7. + xij / 2. * (_safe_div(ki, kj) + _safe_div(kj, ki)) + 4. / 7. * xij**2

    def _alpha(ki, kj, xij):
        return 1. + _safe_div(kj, ki) * xij

    def _beta(ki, kj, xij):
        return _safe_div(xij * (ki**2 + kj**2 + 2. * ki * kj * xij), 2. * ki * kj)

    def _branches3(ki, kj, kk, xij, xik, xjk):
        """The two branches of the n = 3 SPT recursion.

        F_n = sum_{m=1}^{n-1} [G_m/((2n+3)(n-1))] [(2n+1) alpha F_{n-m} + 2 beta G_{n-m}],
        so n = 3 has two contributions, m = 1 (single leg i through G_1 = 1, pair (j, k) through F_2/G_2)
        and m = 2 (pair (i, j) through G_2, single leg k through F_1 = G_1 = 1). Only the
        second was present before; symmetrising over the 6 orderings does not recover the
        first, because `alpha` is not symmetric -- the m = 1 branch needs alpha(k_i, k_jk)
        where a permuted m = 2 term can only supply alpha(k_jk, k_i). At three equal collinear
        legs the analytic value is F3 = G3 = 4.5, which this reproduces and the previous form
        undershot at 5/3.
        """
        # clamp: the folded configuration (x -> -1 with equal legs) drives the
        # argument through zero and roundoff can take it negative -> NaN
        q_ij = jnp.sqrt(jnp.maximum(ki**2 + kj**2 + 2. * ki * kj * xij, 0.))
        x_ij_k = jnp.where(q_ij > 0., (ki * xik + kj * xjk) / jnp.where(q_ij > 0., q_ij, 1.), 0.)
        q_jk = jnp.sqrt(jnp.maximum(kj**2 + kk**2 + 2. * kj * kk * xjk, 0.))
        x_i_jk = jnp.where(q_jk > 0., (kj * xij + kk * xik) / jnp.where(q_jk > 0., q_jk, 1.), 0.)
        return (_alpha(ki, q_jk, x_i_jk), _beta(ki, q_jk, x_i_jk),
                _F2(kj, kk, xjk), _G2(kj, kk, xjk),
                _alpha(q_ij, kk, x_ij_k), _beta(q_ij, kk, x_ij_k), _G2(ki, kj, xij))

    def _F3_unsym(ki, kj, kk, xij, xik, xjk):
        a1, b1_, F2_jk, G2_jk, a2, b2_, G2_ij = _branches3(ki, kj, kk, xij, xik, xjk)
        m1 = 7. * a1 * F2_jk + 2. * b1_ * G2_jk
        m2 = G2_ij * (7. * a2 + 2. * b2_)
        return (m1 + m2) / 18.

    def _G3_unsym(ki, kj, kk, xij, xik, xjk):
        a1, b1_, F2_jk, G2_jk, a2, b2_, G2_ij = _branches3(ki, kj, kk, xij, xik, xjk)
        m1 = 3. * a1 * F2_jk + 6. * b1_ * G2_jk
        m2 = G2_ij * (3. * a2 + 6. * b2_)
        return (m1 + m2) / 18.

    def _F3(ki, kj, kk, xij, xik, xjk):
        t1 = _F3_unsym(ki, kj, kk, xij, xik, xjk)
        t2 = _F3_unsym(ki, kk, kj, xik, xij, xjk)
        t3 = _F3_unsym(kj, ki, kk, xij, xjk, xik)
        t4 = _F3_unsym(kj, kk, ki, xjk, xij, xik)
        t5 = _F3_unsym(kk, ki, kj, xik, xjk, xij)
        t6 = _F3_unsym(kk, kj, ki, xjk, xik, xij)
        return (t1 + t2 + t3 + t4 + t5 + t6) / 6.

    def _G3(ki, kj, kk, xij, xik, xjk):
        t1 = _G3_unsym(ki, kj, kk, xij, xik, xjk)
        t2 = _G3_unsym(ki, kk, kj, xik, xij, xjk)
        t3 = _G3_unsym(kj, ki, kk, xij, xjk, xik)
        t4 = _G3_unsym(kj, kk, ki, xjk, xij, xik)
        t5 = _G3_unsym(kk, ki, kj, xik, xjk, xij)
        t6 = _G3_unsym(kk, kj, ki, xjk, xik, xij)
        return (t1 + t2 + t3 + t4 + t5 + t6) / 6.

    def _Z2(field, ki, kj, xij, mui, muj):
        b1, b2, bs = _get_bias_params(field, ['b1', 'b2', 'bs'])
        km = ki * mui + kj * muj
        term1 = b2 / 2. + bs / 2. * (xij**2 - 1. / 3.)
        term2 = km / 2. * (_safe_div(mui, ki) * f * (b1 + f * muj**2) + _safe_div(muj, kj) * f * (b1 + f * mui**2))
        term3 = b1 * _F2(ki, kj, xij)
        mu2 = _safe_div(km**2, ki**2 + kj**2 + 2. * ki * kj * xij)
        term4 = f * mu2 * _G2(ki, kj, xij)
        return term1 + term2 + term3 + term4

    def _A2(field, ki, kj, xij, mui, muj):
        b1, b2, bs = _get_bias_params(field, ['b1', 'b2', 'bs'])
        q12vec_z = ki * mui + kj * muj
        q12 = jnp.sqrt(ki**2 + kj**2 + 2. * ki * kj * xij)
        mu12 = jnp.where(q12 > 0., q12vec_z / q12, 0.)
        return b1 * _F2(ki, kj, xij) + b2 / 2. + bs / 2. * (xij**2 - 1. / 3.) + f * mu12**2 * _G2(ki, kj, xij)

    def _Z3(field, k1vec, k2vec, k3vec):
        """Third-order redshift-space kernel, symmetrised over the 3! orderings.

        The redshift-space map `e^{-i k_z u_z} (1 + delta)` organises each order into one
        density block times p velocity blocks, with weight `(f mu k)^p / p!`:

            Z3 = K3 + f mu^2 G3
               + (f mu k)     [ (mu12/k12) G2(1,2) K1 + (mu3/k3) K2(1,2) ]
               + (f mu k)^2/2 [ 2 (mu12/k12)(mu3/k3) G2(1,2) + (mu1/k1)(mu2/k2) K1 ]
               + (f mu k)^3/6 (mu1/k1)(mu2/k2)(mu3/k3)

        with K1 = b1 and K2 the bias-only second-order kernel -- its `f mu^2 G2` piece is
        already carried by the explicit `f mu^2 G3` term, and including it (as the previous
        form did, via `_A2`) double counts. Only the first line is symmetric as written, hence
        the average over the six orderings. The previous form also omitted the `(f mu k)^3`
        term entirely.

        Checked against two independent implementations that agree with each other to 1e-6 and
        disagree with the previous form by factors of 2-3: arXiv:1705.02574 Eq. (A3), and
        Kernels.nb of github.com/oliverphilcox/OneLoopBispectrum (whose
        `perm123 = Permutations[{q1,q2,q3}]` and whose `Z3s` is likewise their mean).
        """
        b1, b2, bs = _get_bias_params(field, ['b1', 'b2', 'bs'])
        # Philcox's Z3 omits K1 on this block where his Z4's analogue carries it;
        # JAXPOWER_PT_Z3_K1=0 selects his literal form, for A/B at b1 != 1.
        _k1fac = b1 if int(os.environ.get('JAXPOWER_PT_Z3_K1', '1')) else 1.

        def _one(v1, v2, v3):
            kvec = v1 + v2 + v3
            k = _norm(kvec)
            fmk = f * _mu(kvec, k) * k
            k1, k2, k3 = _norm(v1), _norm(v2), _norm(v3)
            mu1, mu2, mu3 = _mu(v1, k1), _mu(v2, k2), _mu(v3, k3)
            v12 = v1 + v2
            k12 = _norm(v12)
            mu12 = _mu(v12, k12)
            x12 = _xcos(v1, v2, k1, k2)
            G2_12 = _G2(k1, k2, x12)
            K2_12 = b1 * _F2(k1, k2, x12) + b2 / 2. + bs / 2. * (x12**2 - 1. / 3.)
            return (fmk * (_safe_div(mu12, k12) * G2_12 * _k1fac + _safe_div(mu3, k3) * K2_12)
                    + fmk**2 / 2. * (2. * _safe_div(mu12, k12) * _safe_div(mu3, k3) * G2_12
                                     + _safe_div(mu1, k1) * _safe_div(mu2, k2) * b1)
                    + fmk**3 / 6. * _safe_div(mu1, k1) * _safe_div(mu2, k2) * _safe_div(mu3, k3))

        k1, k2, k3 = _norm(k1vec), _norm(k2vec), _norm(k3vec)
        kvec = k1vec + k2vec + k3vec
        mu = _mu(kvec, _norm(kvec))
        x12 = _xcos(k1vec, k2vec, k1, k2)
        x13 = _xcos(k1vec, k3vec, k1, k3)
        x23 = _xcos(k2vec, k3vec, k2, k3)
        # Density block: the full third-order kernel, not b1 F3. Philcox's
        #   K3 = (b3/6 + g2x angK + g3 angL + g21 angK angK)          <- no counterpart here
        #      + (b2 Fn2{q1,q2} + 2 g2 angK[q1, q2+q3] Gn2{q2,q3}) + b1 Fn3
        # with `g2 = bs/2` and `b2_P = b2 + (2/3) bs` (matching his `g2 (x^2 - 1)` onto the
        # `bs/2 (x^2 - 1/3)` used by _A2/_Z2 here). The dropped block is the genuine third-order
        # bias basis: only one combination of it survives at 1-loop in P(k), which is what b3nl
        # is, but the trispectrum resolves the full angular structure, so there is no faithful
        # mapping and it is set to zero. K3 is not symmetric, hence the average over orderings.
        b2p, g2 = b2 + 2. / 3. * bs, bs / 2.

        def _angK(via, vib):
            kia, kib = _norm(via), _norm(vib)
            return _xcos(via, vib, kia, kib)**2 - 1.

        def _k3_one(v1, v2, v3):
            ka, kb, kc = _norm(v1), _norm(v2), _norm(v3)
            return (b2p * _F2(ka, kb, _xcos(v1, v2, ka, kb))
                    + 2. * g2 * _angK(v1, v2 + v3) * _G2(kb, kc, _xcos(v2, v3, kb, kc)))

        vs3 = [k1vec, k2vec, k3vec]
        k3bias = 0.
        for i3 in range(3):
            for j3 in range(3):
                if j3 != i3:
                    k3bias = k3bias + _k3_one(vs3[i3], vs3[j3], vs3[3 - i3 - j3])
        dens = (b1 * _F3(k1, k2, k3, x12, x13, x23) + k3bias / 6.
                + f * mu**2 * _G3(k1, k2, k3, x12, x13, x23))
        vs = [k1vec, k2vec, k3vec]
        rsd = 0.
        for i in range(3):
            for j in range(3):
                if j != i:
                    rsd = rsd + _one(vs[i], vs[j], vs[3 - i - j])
        return dens + rsd / 6.


    def _IR_pk(k, mu):
        pk = pk_callable(k)
        pknw = pknow_callable(k)
        eIR = (1. + f * mu**2 * (2. + f)) * sigma2 + (f * mu)**2 * (mu**2 - 1.) * sigma2_delta
        return pknw + (pk - pknw) * jnp.exp(-eIR * k**2)

    def _t2211_term(fi, fj, fk, fl, kivec, kjvec, kkvec, klvec):
        ki, kj = _norm(kivec), _norm(kjvec)
        mui, muj = _mu(kivec, ki), _mu(kjvec, kj)
        qvec = kivec + klvec
        q = _norm(qvec)
        muq = _mu(qvec, q)
        x_iq = _xcos(-kivec, qvec, ki, q)
        x_jmq = _xcos(-kjvec, -qvec, kj, q)
        Zi = _Z1eft(fi, ki, mui)
        Zj = _Z1eft(fj, kj, muj)
        Z2k = _Z2(fk, ki, q, x_iq, -mui, muq)
        Z2l = _Z2(fl, kj, q, x_jmq, -muj, -muq)
        Pki = _IR_pk(ki, mui)
        Pkj = _IR_pk(kj, muj)
        Pq = _IR_pk(q, muq)
        return Zi * Zj * Z2k * Z2l * Pki * Pkj * Pq

    def _t3111_term(fi, fj, fk, fl, kivec, kjvec, kkvec):
        ki, kj, kk = _norm(kivec), _norm(kjvec), _norm(kkvec)
        mui, muj, muk = _mu(kivec, ki), _mu(kjvec, kj), _mu(kkvec, kk)
        Zi, Zj, Zk = _Z1eft(fi, ki, mui), _Z1eft(fj, kj, muj), _Z1eft(fk, kk, muk)
        Z3l = _Z3(fl, kivec, kjvec, kkvec)
        Pki = _IR_pk(ki, mui)
        Pkj = _IR_pk(kj, muj)
        Pkk = _IR_pk(kk, muk)
        return Zi * Zj * Zk * Z3l * Pki * Pkj * Pkk

    k1vec = jnp.asarray(k1vec)
    k2vec = jnp.asarray(k2vec)
    k3vec = jnp.asarray(k3vec)
    k4vec = -k1vec - k2vec - k3vec

    # --- The parallelogram configuration T(k, -k, k', -k') -------------------------------
    #
    # This is the configuration Cov[P, P] is built from, and it is a degenerate point of the
    # generic permutation sum: four of the twelve (2, 2, 1, 1) terms carry the internal
    # momentum q = k1 + k2, which vanishes there, and their individual 1/q^2 poles cancel
    # only in the sum. Evaluated right at q = 0 the `_safe_div` guards return zero for those four
    # terms, which is not their (finite) limit, so the sum comes out wrong.
    #
    # The limit itself is perfectly well behaved: shifting k2 -> k2 - eps k1 (so that
    # q = -eps k1 and, since k4 is rebuilt from momentum conservation, k3 + k4 = +eps k1) the
    # generic sum is flat to 6 significant figures over 1e-4 > eps > 1e-8 -- four decades --
    # before round-off sets in. So the fix is simply to evaluate it just off the degenerate
    # point rather than to write a special expression for it.
    #
    # Such a special expression did once exist (`_para_channel`, reachable with
    # JAXPOWER_PT_PARA_REDUCED=1). It returns precisely twice the limit, in every configuration
    # tested and in the pure matter limit (b1 = 1, f = 0, no counterterms), so this is not a
    # bias- or RSD-modelling difference. It is a double count: its (3, 1, 1, 1) piece is added
    # in both of the two channel calls, giving 12 x 2 x 2 = 48 units against the generic
    # 6 x 4 = 24, and the same for its (2, 2, 1, 1) piece. `tests/test_pt_trispectrum.py`.
    #
    # One requirement falls on the caller: `pk_callable` must fall to zero below its grid, not clamp to
    # P(k_min) as `jnp.interp` does by default. The four collapsed terms carry P(q) with
    # q -> 0 and are finite only because P(q) kills their 1/q^2 kernels; with a clamped
    # constant they instead contribute a spurious O(1) piece -- 44% of T on one matter
    # configuration -- and the limit acquires a 28% dependence on the direction of approach.
    # With a physical P that direction dependence is 1e-5, i.e. the parallelogram value is
    # perfectly well defined, and this regularization agrees with evaluating at q = 0 exactly
    # to the same 1e-5. It is kept because it makes the limit explicit and costs nothing.
    _use_reduced = bool(int(os.environ.get('JAXPOWER_PT_PARA_REDUCED', '0')))
    _para_eps = float(os.environ.get('JAXPOWER_PT_PARA_EPS', '1e-5'))
    if not _use_reduced:
        _scale2 = jnp.sum(k1vec**2, axis=-1) + jnp.sum(k2vec**2, axis=-1)
        _degen = jnp.sum((k1vec + k2vec)**2, axis=-1) < (1e-10)**2 * jnp.maximum(_scale2, 1e-300)
        k2vec = k2vec - jnp.where(_degen, _para_eps, 0.)[..., None] * k1vec
        k4vec = -k1vec - k2vec - k3vec

    k1, k2, k3, k4 = _norm(k1vec), _norm(k2vec), _norm(k3vec), _norm(k4vec)
    mu1, mu2, mu3, mu4 = _mu(k1vec, k1), _mu(k2vec, k2), _mu(k3vec, k3), _mu(k4vec, k4)

    t2211 = (
        _t2211_term(a, b, c, d, k1vec, k2vec, k3vec, k4vec) +
        _t2211_term(a, b, d, c, k1vec, k2vec, k4vec, k3vec) +
        _t2211_term(a, c, b, d, k1vec, k3vec, k2vec, k4vec) +
        _t2211_term(a, c, d, b, k1vec, k3vec, k4vec, k2vec) +
        _t2211_term(a, d, b, c, k1vec, k4vec, k2vec, k3vec) +
        _t2211_term(a, d, c, b, k1vec, k4vec, k3vec, k2vec) +
        _t2211_term(b, c, a, d, k2vec, k3vec, k1vec, k4vec) +
        _t2211_term(b, c, d, a, k2vec, k3vec, k4vec, k1vec) +
        _t2211_term(b, d, a, c, k2vec, k4vec, k1vec, k3vec) +
        _t2211_term(b, d, c, a, k2vec, k4vec, k3vec, k1vec) +
        _t2211_term(c, d, a, b, k3vec, k4vec, k1vec, k2vec) +
        _t2211_term(c, d, b, a, k3vec, k4vec, k2vec, k1vec)
    )

    t3111 = (
        _t3111_term(a, b, c, d, k1vec, k2vec, k3vec) +
        _t3111_term(a, b, d, c, k1vec, k2vec, k4vec) +
        _t3111_term(a, c, d, b, k1vec, k3vec, k4vec) +
        _t3111_term(b, c, d, a, k2vec, k3vec, k4vec)
    )

    # Legacy reduced expression for the parallelogram, kept only so the change above can be
    # A/B'd: set JAXPOWER_PT_PARA_REDUCED=1. It is wrong (see the note at the regularization).
    para_tol = 1e-10
    is_para = ((_norm(k1vec + k2vec) < para_tol) & (_norm(k3vec + k4vec) < para_tol)
               if _use_reduced else False)

    def _para_channel(rvec, fr, fmr):
        # Channel q = k + r, with r either +k' or -k'.
        r = _norm(rvec)
        mur = _mu(rvec, r)
        mrvec = -rvec

        qvec = k1vec + rvec
        q = _norm(qvec)
        muq = _mu(qvec, q)

        Pk = _IR_pk(k1, mu1)
        Pr = _IR_pk(r, mur)
        Pq = _IR_pk(q, muq)

        Z1_k = _Z1eft(a, k1, mu1)
        Z1_mk = _Z1eft(b, k2, mu2)
        Z1_r = _Z1eft(fr, r, mur)
        Z1_mr = _Z1eft(fmr, r, -mur)

        x_mk_q = _xcos(-k1vec, qvec, k1, q)
        x_mr_q = _xcos(mrvec, qvec, r, q)

        # Z2 arguments correspond to Z2(-k, q) and Z2(-r, q).
        Z2_k = _Z2(a, k1, q, x_mk_q, -mu1, muq)
        Z2_r = _Z2(fr, r, q, x_mr_q, -mur, muq)

        # Parallelogram/reduced snake terms. For identical fields this reduces
        # to the usual compact expression; for cross fields this keeps the
        # external Z1 factors attached to their legs.
        t2211 = (
            8. * Z1_k * Z1_mk * Z2_k**2 * Pk**2 * Pq
            + 8. * Z1_r * Z1_mr * Z2_r**2 * Pr**2 * Pq
            + 16. * Z1_k * Z1_r * Z2_k * Z2_r * Pk * Pr * Pq
        )

        # Star terms with one third-order kernel on either hard pair.
        Z3_kkr = _Z3(a, k1vec, -k1vec, rvec)
        Z3_rrk = _Z3(fr, rvec, -rvec, k1vec)
        t3111 = 12. * (
            Z1_k * Z1_mk * Z1_r * Z3_kkr * Pk**2 * Pr
            + Z1_r * Z1_mr * Z1_k * Z3_rrk * Pr**2 * Pk
        )
        return t2211 + t3111

    # Both physical diagonals: q = k + k' and q = k - k'.
    trispec_tree = 4. * t2211 + 6. * t3111
    if is_para is not False:
        trispec_tree = jnp.where(is_para,
                                 _para_channel(k3vec, c, d) + _para_channel(k4vec, d, c),
                                 trispec_tree)

    X1, X2, X3, X4 = [_get_bias_params(field, 'X_FoG') for field in fields]
    W = fog_damping((k1 * mu1, X1), (k2 * mu2, X2), (k3 * mu3, X3), (k4 * mu4, X4), f=f, sigma2v=sigma2v, damping=damping)

    def _shot_leg(field, k, mu, Z1eft, pkIR):
        b1, snb0, sn0 = _get_bias_params(field, ['b1', 'snb0', 'sn0'])
        # f mu^2 P_eps, not 2 f mu^2 P_eps: the Poisson limit is (1/nbar) sum_i Z1(k_i)^2 P(k_i) + 1/nbar^2 (see jaxpower.pt)
        return (b1 * snb0 + sn0 * f * mu**2) * Z1eft * pkIR

    # ---- Stochastic trispectrum ---------------------------------------------------------
    #
    # Structure, and why it is what it is. For a point process obtained by sampling a
    # continuous field, the discrete n-point spectrum is a sum over set partitions of the n
    # legs: each block B of size m contributes one coincidence factor (amplitude)^(m - 1) and
    # the continuous connected spectrum of the *blocks*, evaluated at the block momenta
    # K_B = sum_{i in B} k_i. For n = 2 and n = 3 this reproduces the familiar
    #
    #     P^(N)  = P + sn
    #     B^(N)  = B + sn [P(k1) + P(k2) + P(k3)] + sn^2
    #
    # -- the latter being exactly what the legs below build for :func:`spectrum3_redshift_tracer`.
    # For n = 4 the partitions are: one pair (6 ways), two disjoint pairs (3), a triple (4),
    # and all four; giving
    #
    #     T^(N) = T
    #           + sn   sum_{6 pairs (i,j)}     B(k_i + k_j, k_k, k_l)
    #           + sn^2 sum_{3 pairings}        P(k_i + k_j)
    #           + sn^2 sum_{4 legs}            P(k_l)
    #           + sn^3
    #
    # Each term has total dimension L^9, as it must: with [P] = L^3, [B] = L^6 and
    # [sn] = L^3, the powers (p, q) of (P, sn) satisfy p + q = 3 term by term.
    #
    # On the previous expression, and why it was wrong. It read
    #
    #     0.25 sum_{i<j} leg_i leg_j  +  0.5 sum_i leg_i sn0_i  +  shot
    #
    # with leg_i = (b1 snb0 + 2 sn0 f mu_i^2) Z1eft_i P_IR_i. `leg` is a *bispectrum*-sized
    # object (sn x P, i.e. L^6), so `leg_i leg_j` is L^12 and cannot be a trispectrum
    # contribution at all -- it is (p, q) = (2, 2). It appears in no partition of four legs.
    # Numerically it dominated everything: at the stochastic amplitude fitted to an
    # AbacusSummit LRG HOD mock set (snb0 = 1.2 / nbar) it was 174x the tree trispectrum at
    # k = 0.03 and 14500x at k = 0.29, and pushed sigma_analytic / sigma_mock from ~1 to 7.0
    # on P0 and 5.9 on B000. The `0.5 leg_i sn0_i` piece had the right dimensions but the
    # wrong normalization (a factor 4 low against the partition expansion, which wants
    # sn x leg_i); and the trailing `+ shot` was L^3 where L^9 is needed (`shot**3` now,
    # matching `shot**2` in spectrum3). No caller in this repository passed `shot`.
    #
    # Conventions, all inherited from spectrum3_redshift_tracer so the two stay consistent:
    #   * the pair-coincidence amplitude is `snb0` (which equals sn2 = 1/nbar in the Poisson
    #     limit); the (snb0, sn0) split, i.e. the mu-dependence of the stochastic amplitude,
    #     is kept only where it multiplies a leg, exactly as in the bispectrum's shot leg;
    #   * the pure constant is `sn0**3`, as spectrum3's is `sn0**2`;
    #   * the stochastic sector is left undamped by FoG, as in spectrum3;
    #   * a coincidence between different tracers vanishes identically.
    #
    # **Poisson is `snb0 = sn0 = 1 / nbar`.** This said `sn0 = sn2 / 2`, which came from matching
    # against `sn sum_i P_g^s(k_i) + sn^2` -- the expansion that ignores the noise being itself
    # transported by the real-to-redshift-space mapping. The redshift-space vertex is
    # `b1 B_eps + 2 f mu^2 P_eps`, whose second piece IS that transport, so at Poisson it reads
    # `(b1 + 2 f mu^2) / nbar`, not `Z1 / nbar`. The tell that `sn0 = 1/(2 nbar)` is wrong: it
    # puts `sn0**2` at a quarter of the three-point Poisson constant `1 / nbar**2`.
    #
    # The zero-momentum guard. Terms carrying a pair momentum q = k_i + k_j are switched off
    # where q vanishes identically. That is not a physical statement about the squeezed limit:
    # it is that P and B are not defined at q = 0 by any tabulated theory, and the covariance
    # calls this at (k, -k, k', -k'), where two of the six pairs have q == 0 exactly. It
    # mirrors `cov3.compute_spectrum3_covariance`'s own `_alive` guard, and with it the six
    # pair terms reduce in that configuration to precisely the four cross pairs, and the three
    # pairings to the two, that cov3 builds for itself from `shotnoise` -- an independent check
    # that the two derivations agree.
    kvecs = (k1vec, k2vec, k3vec, k4vec)
    knorms, munorms = (k1, k2, k3, k4), (mu1, mu2, mu3, mu4)

    def _sn_pair(fi, fj):
        # A coincidence requires the two points to be the same tracer.
        if fi != fj:
            return None
        return _get_bias_params(fi, 'snb0')

    def _Pg(field, kvec):
        """The model galaxy power at an arbitrary momentum (a block momentum, in general)."""
        k = _norm(kvec)
        mu = _mu(kvec, k)
        return _Z1eft(field, k, mu)**2 * _IR_pk(k, mu)

    def _B_tree(fq, fk, fl, kqvec, kkvec):
        """Tree-level B(kq, kk, kl), kl = -kq - kk; same kernels as spectrum3_redshift_tracer."""
        klvec = -kqvec - kkvec
        kq, kk, kl = _norm(kqvec), _norm(kkvec), _norm(klvec)
        muq, muk, mul = _mu(kqvec, kq), _mu(kkvec, kk), _mu(klvec, kl)
        xqk = _xcos(kqvec, kkvec, kq, kk)
        xkl = _xcos(kkvec, klvec, kk, kl)
        xlq = _xcos(klvec, kqvec, kl, kq)
        Zq, Zk, Zl = _Z1eft(fq, kq, muq), _Z1eft(fk, kk, muk), _Z1eft(fl, kl, mul)
        Pq, Pk, Pl = _IR_pk(kq, muq), _IR_pk(kk, muk), _IR_pk(kl, mul)
        # The Z2 kernel carries the field of the leg outside the contracted pair, as in
        # spectrum3_redshift_tracer's B12 / B23 / B31.
        return 2. * (_Z2(fl, kq, kk, xqk, muq, muk) * Zq * Pq * Zk * Pk
                     + _Z2(fq, kk, kl, xkl, muk, mul) * Zk * Pk * Zl * Pl
                     + _Z2(fk, kl, kq, xlq, mul, muq) * Zl * Pl * Zq * Pq)

    def _alive(qvec, kavec, kbvec):
        # The threshold must sit above the parallelogram regularization, never below it.
        # That regularization moves k1 + k2 from exactly 0 to -eps k1, i.e. q^2 / ref = eps^2 /
        # 2 ~ 5e-11 at the default eps = 1e-5. A fixed 1e-12 threshold would let those legs
        # through, and they would be evaluated at a q so small that P(q) and B(q, k, -k) are
        # whatever the theory's k-grid extrapolates to at its lowest node -- typically a
        # constant, where the true answer is zero (P(q) -> 0 as q -> 0, and the tree B in the
        # squeezed limit goes with it). Measured on an LRG-like case that spurious term is
        # ~0.6 of the whole tree trispectrum at k = 0.29. Tying the threshold to eps keeps
        # genuine squeezed configurations (q / k >~ 1e-4) and rejects the regularized zero.
        q2 = jnp.sum(qvec**2, axis=-1)
        ref = jnp.sum(kavec**2, axis=-1) + jnp.sum(kbvec**2, axis=-1)
        return jnp.where(q2 > (10. * _para_eps)**2 * ref, 1., 0.)

    legs = [_shot_leg(fields[i], knorms[i], munorms[i],
                      _Z1eft(fields[i], knorms[i], munorms[i]),
                      _IR_pk(knorms[i], munorms[i])) for i in range(4)]

    stoch = jnp.zeros_like(legs[0])

    # (1) one pair coincides:  sn * B(k_i + k_j, k_k, k_l)
    for i, j, kk_, ll_ in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
                           (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]:
        sn_ij = _sn_pair(fields[i], fields[j])
        if sn_ij is None:
            continue
        qvec = kvecs[i] + kvecs[j]
        stoch = stoch + sn_ij * _alive(qvec, kvecs[i], kvecs[j]) \
            * _B_tree(fields[i], fields[kk_], fields[ll_], qvec, kvecs[kk_])

    # (2) two disjoint pairs coincide:  sn^2 * P(k_i + k_j)
    for (i, j), (kk_, ll_) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        sn_ij, sn_kl = _sn_pair(fields[i], fields[j]), _sn_pair(fields[kk_], fields[ll_])
        if sn_ij is None or sn_kl is None:
            continue
        qvec = kvecs[i] + kvecs[j]
        stoch = stoch + sn_ij * sn_kl * _alive(qvec, kvecs[i], kvecs[j]) \
            * _Pg(fields[i], qvec)

    # (3) a triple coincides, one leg free:  sn3 * P(k_l).
    for l_ in range(4):
        stoch = stoch + _sn3(fields[l_]) * _Pg(fields[l_], kvecs[l_])

    # (4) all four coincide. `P_eps**3`, for the same reason spectrum3's constant is
    # `P_eps**2`: at Poisson `P_eps = 1 / nbar` and the connected four-point self-contraction is
    # `1 / nbar**3`. It used to read `shot**3`, a separate argument no caller ever passed -- the
    # same spurious third stochastic knob spectrum3 carried.
    stoch = stoch + _get_bias_params(fields[0], 'sn0')**3

    return W * trispec_tree + stoch

