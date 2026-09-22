r"""Tests for :mod:`jaxpower.cov3`, the periodic-box power spectrum x bispectrum covariance.

The module was rewritten from arXiv:1908.06234 term by term. The implementation it replaced is
kept at ``jaxpower/_cov3_legacy.py``, and its three test files are archived beside this one as
``_test_cov3_legacy.py``, ``_test_cov3_p5_legacy.py`` and ``_test_cov3_mocks_legacy.py``. They
are not collected and none of them applies as written:

* the FKP-window tests predate the cutsky port and assume the old term set;
* ``_test_cov3_legacy.test_cov3_bb_ties_modesum`` and ``_test_cov3_p5_legacy`` select sub-terms
  through ``COV3_*`` environment switches that the rewrite does not read -- it takes ``terms=``
  instead -- so both runs return the same matrix and the assertions compare it with itself;
* ``_test_cov3_mocks_legacy`` still has a working box path, and it is the only mock-based
  validation there was. It is archived rather than deleted for that reason.

Two things in there are worth recovering when there is reason to: that mock comparison, and the
exact discrete mode-sum reference in ``_test_cov3_legacy.test_cov3_bb_ties_modesum``, which needs
the arm / closure split of ``_bb_tie_terms``' nine leg pairings exposed as a selectable term.

Everything below asserts a closed form or an exact property, never one run of the module against
another run of the module.

What is pinned here:

* :func:`test_gaussian_pp_closed_form` -- the one block with an exact closed form. A constant
  theory makes the shell and angular integrals exact, so ``Cov[P_l, P_l'] = 2 (2l + 1) P^2 / N``
  with ``N`` the continuum mode count, bin-diagonal and multipole-diagonal, holds to quadrature
  precision rather than approximately.
* :func:`test_theory_receives_independent_legs` and :func:`test_partial_theory` -- two regressions from
  moving the module into jaxpower, both of which every caller in ``claude_bk_cov_new`` happens to
  hide. See their docstrings.
* :func:`test_shotnoise_is_pairwise` -- every shot line is a product of disjoint *pair*
  coincidences, so it scales as a power of the pair amplitude and never involves a higher weight
  moment. The exponents are read off by scaling ``sn``.
* :func:`test_isotropic_kills_b000_b202` -- ``Cov[B000, B202]`` is identically zero for an
  isotropic theory, so it measures anisotropy rather than covariance. This is the sharpest cheap
  statement available about the bispectrum blocks' angular structure.
* :func:`test_terms_are_additive` -- the eight terms are a partition of the total.
* :func:`test_calling_conventions` -- ``window3=None``, and the guard that stops it reaching the
  one block that cannot accept it.
* :func:`test_window_gaussian_pp_matches_cov2`, :func:`test_window_periodic_approximation` and
  :func:`test_window_double_closure_tie` -- the survey-window path. The first is a comparison
  against an independent implementation. The second is a limit: a window that *is* a uniform box
  has to reproduce the box covariance, and to stay there as the quadrature tightens. The third is
  the one closed form the doubly-derived ``(2, 2)`` tie has, which is the pairing no other
  implementation covers -- there is nothing else to compare it against.

Run everything with ``python tests/test_cov3.py``; ``--fast`` skips the ones that build a
bispectrum or a windowed covariance and take minutes. The survey window is built from uniform
randoms on first use and cached in ``_tests/``.
"""

import argparse
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp

import lsstypes
from jaxpower import MeshAttrs, pt, types
from jaxpower.cov3 import Spectra, compute_spectrum3_covariance


#: Deliberately tiny. These pin properties and closed forms, not values, so the cheapest
#: configuration that still has more than one bin and more than one multipole is the right one.
BOXSIZE = 2000.
NK, KMIN, DK = 3, 0.05, 0.03
#: Crude on purpose, and only ever compared against something computed at the same settings.
CHEAP = dict(order3=4, order5=3, p6_size=64)


def _edges():
    e = KMIN + DK * np.arange(NK + 1)
    return e, (e[:-1] + e[1:]) / 2.


def _observable(ells2=(0, 2), ells3=((0, 0, 0), (2, 0, 2))):
    """``P_ell(k)`` and ``B_{l1 l2 L}(k, k)`` on one binning, as an ``ObservableTree``."""
    edges, k = _edges()
    e2 = np.stack([edges[:-1], edges[1:]], axis=-1)
    zero = np.zeros(NK)
    branches, fields = [], []
    if ells2:
        branches.append(lsstypes.Mesh2SpectrumPoles(
            [lsstypes.Mesh2SpectrumPole(k=k, k_edges=e2, num_raw=zero, num_shotnoise=zero,
                                        norm=1., nmodes=np.ones(NK), ell=ell) for ell in ells2],
            ells=list(ells2)))
        fields.append((0, 0))
    if ells3:
        k3 = np.stack([k, k], axis=-1)
        e3 = np.stack([e2, e2], axis=1)
        branches.append(lsstypes.Mesh3SpectrumPoles(
            [lsstypes.Mesh3SpectrumPole(k=k3, k_edges=e3, num_raw=zero, num_shotnoise=zero,
                                        norm=1., nmodes=np.ones(NK), ell=tuple(ell),
                                        basis='sugiyama-diagonal') for ell in ells3],
            ells=[tuple(ell) for ell in ells3]))
        fields.append((0, 0, 0))
    return types.ObservableTree(branches, fields=fields)


def _multitracer_observable(tracers):
    """One ``P`` and one ``B`` leaf per tracer, all on the same binning."""
    edges, k = _edges()
    e2 = np.stack([edges[:-1], edges[1:]], axis=-1)
    e3 = np.stack([e2, e2], axis=1)
    k3 = np.stack([k, k], axis=-1)
    zero = np.zeros(NK)
    branches, fields = [], []
    for tracer in tracers:
        branches.append(lsstypes.Mesh2SpectrumPoles(
            [lsstypes.Mesh2SpectrumPole(k=k, k_edges=e2, num_raw=zero, num_shotnoise=zero,
                                        norm=1., nmodes=np.ones(NK), ell=ell) for ell in (0, 2)],
            ells=[0, 2]))
        fields.append((tracer, tracer))
        branches.append(lsstypes.Mesh3SpectrumPoles(
            [lsstypes.Mesh3SpectrumPole(k=k3, k_edges=e3, num_raw=zero, num_shotnoise=zero,
                                        norm=1., nmodes=np.ones(NK), ell=ell,
                                        basis='sugiyama-diagonal')
             for ell in ((0, 0, 0), (2, 0, 2))], ells=[(0, 0, 0), (2, 0, 2)]))
        fields.append((tracer, tracer, tracer))
    return types.ObservableTree(branches, fields=fields)


def test_multitracer_reduces_to_one_tracer():
    """Two tracers that are the same tracer must give back the one-tracer covariance.

    With a label-independent theory the two are indistinguishable *except* through the shot
    noise, which is a coincidence amplitude: two points carrying different labels are never the
    same point, so ``P^{(N)}_ab`` has no constant while ``P^{(N)}_aa`` does. That is physical, not
    an artefact, so the check is split in two.

    At finite shot noise only the auto blocks may agree, and they must agree *exactly* -- the
    single-tracer path has to survive the field plumbing bit for bit. At zero shot noise nothing
    distinguishes the labels at all, and then every block, cross ones included, must agree.

    This also covers a dispatch trap: the leaves here interleave as ``P_a, B_a, P_b, B_b``, so
    ``Cov[B_a, P_b]`` is reached with a bispectrum group *before* a power spectrum one. Handling
    that by skipping and waiting for symmetry to fill it in leaves the block zero, which the
    ratios below would show as -1.
    """
    theory = _tree_theory()
    mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=64)
    one, two = _multitracer_observable(['a']), _multitracer_observable(['a', 'b'])

    for shotnoise in (1995., 0.):
        options = dict(theory=theory, shotnoise=shotnoise, **CHEAP)
        single = np.asarray(compute_spectrum3_covariance(mattrs, None, one, **options).value())
        double = np.asarray(compute_spectrum3_covariance(mattrs, None, two, **options).value())
        size = single.shape[0]
        assert double.shape == (2 * size, 2 * size), double.shape
        nonzero = np.abs(single) > 0
        assert nonzero.any(), 'the single-tracer covariance is empty; nothing is being compared'
        quadrants = {'aa': double[:size, :size], 'bb': double[size:, size:],
                     'ab': double[:size, size:], 'ba': double[size:, :size]}
        for name in ('aa', 'bb'):
            err = np.max(np.abs(quadrants[name][nonzero] / single[nonzero] - 1.))
            print(f'  shotnoise={shotnoise:<7.0f} block {name} vs one tracer: '
                  f'max |ratio - 1| = {err:.3e}')
            assert err == 0., (shotnoise, name, err)
        if shotnoise:
            # ...and the cross blocks must NOT match, or the labels are being ignored.
            err = np.max(np.abs(quadrants['ab'][nonzero] / single[nonzero] - 1.))
            print(f'  shotnoise={shotnoise:<7.0f} block ab differs, as it must: '
                  f'max |ratio - 1| = {err:.3e}')
            assert err > 1e-3, 'a cross-tracer block must lose the shot-noise coincidences'
        else:
            # At zero shot noise the labels are meaningless, so the cross blocks match too --
            # except in Cov[B, B], where the two orientations of a multipole pair are separate
            # quadratures that agree only in the converged limit. `test_bb_multipole_pair_symmetry`
            # is that statement; here the P sector and the P x B blocks must be exact.
            p_sector = np.zeros_like(nonzero)
            p_sector[:2 * NK, :] = p_sector[:, :2 * NK] = True
            mask = nonzero & p_sector
            for name in ('ab', 'ba'):
                err = np.max(np.abs(quadrants[name][mask] / single[mask] - 1.))
                print(f'  shotnoise=0       block {name} vs one tracer, outside Cov[B, B]: '
                      f'max |ratio - 1| = {err:.3e}')
                # Not bit-identical, unlike the auto blocks: `sum_over_terms` groups a permutation
                # sum by field tuple, and `(a, a, b, b)` groups differently from `(a, a, a, a)`,
                # so the trispectrum line is summed in a different order. Round-off only.
                assert err < 1e-9, (name, err)


def test_bb_multipole_pair_symmetry(orders=(8, 16), atol=2e-2):
    r"""``Cov[B_i(k), B_j(k')]`` must equal ``Cov[B_j(k'), B_i(k)]``, and only quadrature separates them.

    The two are separate integrals here -- one is computed with multipole ``i`` on the unprimed
    triangle, the other with ``j`` -- so their agreement is a convergence statement about the
    angular quadrature, not an identity of the code. It is worth asserting because **the
    single-tracer assembly cannot see it**: when a group is paired with itself, ``blocks[i][j]``
    is written once as ``res[i][j]`` and again as ``res[j][i].T``, and the second write wins, so
    one of the two estimates is silently discarded. Two tracers keep both, which is the only
    reason this is visible at all.

    Measured on ``Cov[B000, B202]``, the entry the angular machinery is most delicate about: the
    gap falls from 0.60 at ``order3 = 4`` -- the crude setting these tests otherwise use -- to
    0.012 at 8 and 0.010 at 16.
    """
    mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=64)
    two = _multitracer_observable(['a', 'b'])
    b000, b202 = slice(2 * NK, 3 * NK), slice(3 * NK, 4 * NK)

    previous = None
    for order3 in orders:
        value = np.asarray(compute_spectrum3_covariance(
            mattrs, None, two, theory=_tree_theory(), shotnoise=0., terms=('PPP',),
            order3=order3, order5=3, p6_size=64).value())
        size = value.shape[0] // 2
        cross = value[:size, size:]
        gap = (np.max(np.abs(cross[b000, b202] - cross[b202, b000].T))
               / np.max(np.abs(cross[b000, b202])))
        print(f'  order3={order3:<3} Cov[B000, B202] vs Cov[B202, B000]^T: {gap:.3e}')
        assert gap < atol, (order3, gap)
        if previous is not None:
            assert gap <= previous * 1.5, 'the gap must not grow with quadrature order'
        previous = gap


def test_multitracer_isolates_tracers():
    """A tracer's own block must not depend on what else is in the observable.

    The previous test uses one theory for both labels, so a field tuple that is silently ignored
    still passes it. Here the two tracers have genuinely different bias, and the ``a`` block of
    the joint run has to reproduce a run that knows only about ``a``.
    """
    mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=64)
    per_tracer = {'a': _tree_theory(b1=2.0), 'b': _tree_theory(b1=1.2)}

    def theory(fields):
        # A cross-spectrum between distinct tracers is not modelled by
        # `make_spectra_redshift_tracer`, and for this test it does not need to be:
        # nothing in the `a` block asks for one.
        if len(set(fields)) > 1:
            return None
        return per_tracer[fields[0]](fields)

    options = dict(shotnoise=1995., **CHEAP)
    alone = np.asarray(compute_spectrum3_covariance(
        mattrs, None, _multitracer_observable(['a']), theory=per_tracer['a'], **options).value())
    joint = np.asarray(compute_spectrum3_covariance(
        mattrs, None, _multitracer_observable(['a', 'b']), theory=theory, **options).value())
    size = alone.shape[0]
    nonzero = np.abs(alone) > 0
    err = np.max(np.abs(joint[:size, :size][nonzero] / alone[nonzero] - 1.))
    print(f'  block a of the joint run vs a run knowing only a: max |ratio - 1| = {err:.3e}')
    assert err == 0., err

    # And the b block must differ, since b has a different bias.
    berr = np.max(np.abs(joint[size:, size:][nonzero] / alone[nonzero] - 1.))
    print(f'  block b differs, as it must: max |ratio - 1| = {berr:.3e}')
    assert berr > 1e-3, 'the two tracers have different bias and must not give the same block'


def _const_theory(value):
    """Only ``P``, and constant: every integral the Gaussian block does becomes exact."""
    def theory(fields):
        if len(fields) != 2:
            return None
        return lambda kvec: value * jnp.ones_like(jnp.asarray(kvec)[..., 0])
    return theory


def _tree_theory(f=0.75, b1=2.0, nmax=6):
    """The tree-level n-point of :mod:`jaxpower.pt` on a smooth power law."""
    def pk(k):
        k = jnp.asarray(k)
        return 2e4 * (jnp.maximum(k, 1e-8) / 0.1) ** (-1.5) * jnp.exp(-(k / 0.6) ** 2)

    spectra = pt.make_spectra_redshift_tracer(pk, f=f, bias=dict(b1=b1), nmax=nmax)
    return lambda fields: spectra.get(len(fields), None)


def _legs(n, size=2, seed=0):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.normal(scale=0.1, size=(size, 3))) for _ in range(n)]


# --------------------------------------------------------------------------------------
# the one closed form
# --------------------------------------------------------------------------------------

def test_gaussian_pp_closed_form(rtol=1e-6):
    r"""``Cov[P_l(k_a), P_l'(k_b)] = 2 (2l + 1) P^2 / N_a`` for a constant theory.

    With :math:`P` independent of :math:`k` and :math:`\mu`, the shell average and the angular
    integral are both exact, and

    .. math::
        {\rm Cov} = \frac{2}{N_a} (2l + 1)(2l' + 1) P^2 \int \frac{d\mu}{2} L_l L_{l'}
                  = \frac{2 (2l + 1)}{N_a} P^2 \delta_{ll'}

    so the multipole structure, the bin-diagonality and the mode count are all pinned at once.

    The angular factor is checked as the ratio between multipoles, where :math:`N_a` cancels
    outright, and the normalization separately against ``nmodes='continuum'``, which is Eq. (43)
    verbatim: :math:`4 \pi k^2 \Delta k V / (2\pi)^3` at the bin's own :math:`k`. That is *not*
    the exact shell integral :math:`(4\pi/3)(k_{\rm hi}^3 - k_{\rm lo}^3) V / (2\pi)^3` -- the two
    differ by 1.8% on the bins used here and by 3.7% on a bin starting at :math:`k = 0.02`, which
    is 1.9% on :math:`\sigma(P_0)` in the lowest bin. The difference is the continuum
    approximation itself, and ``nmodes='mesh'`` is the way out of it; the assertion below pins
    which of the two the default computes, so that choice cannot drift silently.
    """
    P, volume = 1.2e4, BOXSIZE**3
    obs = _observable(ells2=(0, 2), ells3=())
    cov = compute_spectrum3_covariance(MeshAttrs(boxsize=BOXSIZE, meshsize=64), None, obs,
                                       theory=_const_theory(P), shotnoise=0., terms=('PP',))
    value = np.asarray(cov.value())

    edges, kmid = _edges()
    nmodes = 4. * np.pi * kmid**2 * DK * volume / (2. * np.pi)**3

    diag = {}
    for i, ell in enumerate((0, 2)):
        block = value[i * NK:(i + 1) * NK, i * NK:(i + 1) * NK]
        diag[ell] = np.diag(block)
        expect = 2. * (2 * ell + 1) * P**2 / nmodes
        err = np.max(np.abs(diag[ell] / expect - 1.))
        print(f'  Cov[P{ell}, P{ell}] vs 2(2l+1)P^2/N_continuum: max |ratio - 1| = {err:.3e}')
        assert err < rtol, (ell, diag[ell], expect)
        off = block - np.diag(diag[ell])
        assert np.all(np.abs(off) <= rtol * np.abs(block).max()), 'Gaussian block is not bin-diagonal'

    # N cancels here, so this is the angular structure alone.
    ratio = diag[2] / diag[0]
    print(f'  Cov[P2, P2] / Cov[P0, P0] = {ratio}   (must be 5)')
    assert np.allclose(ratio, 5., rtol=rtol), ratio

    cross = value[:NK, NK:2 * NK]
    scale = np.abs(value).max()
    print(f'  Cov[P0, P2] / max|Cov| = {np.abs(cross).max() / scale:.3e}   (must vanish)')
    assert np.abs(cross).max() <= rtol * scale, 'P0 x P2 must vanish for an isotropic theory'
    return value


# --------------------------------------------------------------------------------------
# two regressions from the move into jaxpower
# --------------------------------------------------------------------------------------

def test_theory_receives_independent_legs():
    """The theory callable is handed the ``n - 1`` independent legs, not every leg.

    This is ``pt.py``'s own convention -- ``spectrum3_redshift_tracer`` takes two wavevectors and
    ``spectrum4_redshift_tracer`` three. A connected n-point lives on ``sum k_i = 0``, so the
    closing leg is not an independent argument, and handing it over invites the caller's idea of
    it to disagree with the callee's.
    """
    seen = {}

    def theory(fields):
        n = len(fields)

        def fun(*legs):
            seen[n] = len(legs)
            return jnp.zeros_like(jnp.asarray(legs[0])[..., 0])

        return fun

    sp = Spectra(theory=theory)
    v = _legs(5, seed=1)
    sp.power(('a',) * 2, v[0])
    for order in (3, 4, 5, 6):
        sp.connected(('a',) * order, *v[:order - 1])
    print(f'  legs passed per order: {seen}')
    assert seen == {2: 1, 3: 2, 4: 3, 5: 4, 6: 5}, seen


def test_partial_theory():
    """A theory that models only some orders must still give shaped zeros for the rest.

    Returning a bare ``0.`` makes a whole block a rank-0 array that then fails to broadcast
    against the node and bin axes. Nothing in the campaign hits it because
    :func:`jaxpower.pt.make_spectra_redshift_tracer` always supplies all five orders at
    once; a bispectrum-only theory is what exposes it.
    """
    def theory(fields):
        if len(fields) != 3:
            return None
        return lambda k1, k2, k3: jnp.ones_like(jnp.asarray(k1)[..., 0])

    sp = Spectra(theory=theory, shotnoise=0.)
    v = _legs(4, size=7, seed=2)
    assert not sp.has(('a',) * 2) and sp.has(('a',) * 3) and not sp.has(('a',) * 6)
    triangles = ((v[0], v[1], -v[0] - v[1]), (v[2], v[3], -v[2] - v[3]))
    for name, val in [
            ('P', sp.power(('a',) * 2, v[0])),
            ('T', sp.connected(('a',) * 4, v[0], v[1], v[2])),
            ('TN', sp.trispectrum_shot(('a',) * 4, v[0], v[1], v[2], v[3])),
            ('P6N', sp.six_point_shot(('a',) * 6, *triangles))]:
        shape = np.shape(val)
        print(f'  {name:<4} shape {shape}')
        assert shape == (7,), (name, shape)


# --------------------------------------------------------------------------------------
# discreteness
# --------------------------------------------------------------------------------------

def test_shotnoise_is_pairwise():
    r"""Every shot line is a product of disjoint *pair* coincidences, so it scales as a power of
    the pair amplitude.

    The powers look like weight moments and are not. Sugiyama's :math:`i \neq j` estimators
    exclude any coincidence inside one estimator, so a term can only ever tie points across
    estimators, two at a time: Eq. (14)'s :math:`sn^2` is two disjoint cross pairs and Eq. (B6)'s
    :math:`sn^3` is three, not a triple and a quadruple coincidence. The distinction is invisible
    for Poisson weights, where :math:`sn_m = sn_2^{m-1}` anyway, and changes the bispectrum
    covariance by tens of per cent for skewed ones -- so it is pinned here by scaling ``sn`` and
    reading off the exponent, which is exactly the thing that differs between the two readings.

    Each line is isolated by supplying a theory that models exactly one order, so the dressed
    spectrum reduces to that line alone and the exponent is exact.
    """
    sn = 3.

    def only(order, value=1.):
        def theory(fields):
            if len(fields) != order:
                return None
            return lambda *legs: value * jnp.ones_like(jnp.asarray(legs[0])[..., 0])
        return theory

    v = _legs(4, size=3, seed=4)
    # (method, legs, the one order modelled, the exponent of sn the surviving line must carry)
    lines = [('bispectrum_shot', 3, (v[0], v[1], v[2]), 2, 1),          # sn   x (P + P)
             ('trispectrum_shot', 4, (v[0], v[1], v[2], v[3]), 2, 2),   # sn^2 x sum P
             ('five_point_shot', 5, (v[0], v[1], v[2], -v[1] - v[2]), 3, 2),   # sn^2 x sum B
             ('six_point_shot', 6, ((v[0], v[1], -v[0] - v[1]),
                                    (v[2], v[3], -v[2] - v[3])), 3, 3)]  # sn^3 x sum B

    for name, nfields, legs, order, exponent in lines:
        fields = ('a',) * nfields
        one = getattr(Spectra(theory=only(order), shotnoise=sn), name)
        two = getattr(Spectra(theory=only(order), shotnoise=2. * sn), name)
        a, b = np.asarray(one(fields, *legs)), np.asarray(two(fields, *legs))
        assert np.all(a != 0.), f'{name}: the shot line must not vanish, or nothing is tested'
        ratio = np.median(b / a)
        print(f'  {name:<17} (only the {order}-point modelled) doubling sn scales it by '
              f'{ratio:.4f}   (must be 2^{exponent} = {2. ** exponent})')
        assert np.allclose(b / a, 2. ** exponent, rtol=1e-12), (name, ratio, exponent)

    # A bare pair amplitude may be spelled as a dict; the higher moments must be refused rather
    # than silently dropped, because no term here is a triple or quadruple coincidence.
    assert Spectra(theory=only(3), shotnoise={2: sn}).coincidence('a', 'a') == sn
    assert Spectra(theory=only(3), shotnoise=sn).coincidence('a', 'b') == 0., \
        'two points of different tracers are never the same point'
    for bad in ({2: sn, 3: sn**2}, {2: sn, 4: sn**3}):
        try:
            Spectra(theory=only(3), shotnoise=bad)
        except ValueError as exc:
            assert 'do not enter' in str(exc), exc
        else:
            raise AssertionError(f'shotnoise={bad} must raise, not be ignored')
    print('  the pair amplitude may be a dict; higher moments raise')

    # With no shot noise at all, the dressed spectra are the bare ones.
    bare = Spectra(theory=only(3), shotnoise=0.)
    triangles = ((v[0], v[1], -v[0] - v[1]), (v[2], v[3], -v[2] - v[3]))
    assert np.array_equal(np.asarray(bare.six_point_shot(('a',) * 6, *triangles)), np.zeros(3)), \
        'no shot noise must leave the six-point bare (here absent, so zero)'
    print('  zero shot noise leaves the dressed spectra bare')


# --------------------------------------------------------------------------------------
# conventions
# --------------------------------------------------------------------------------------

def test_calling_conventions():
    """``window3=None`` means the box of ``window2``, and ``PPP`` will not accept that."""
    mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=64)
    obs = _observable(ells2=(0,), ells3=())
    theory = _const_theory(1.2e4)
    a = np.asarray(compute_spectrum3_covariance(mattrs, None, obs, theory=theory,
                                                terms=('PP',)).value())
    b = np.asarray(compute_spectrum3_covariance(mattrs, mattrs, obs, theory=theory,
                                                terms=('PP',)).value())
    assert np.array_equal(a, b), 'window3=None must be the same box as window2'
    print('  window3=None matches window3=window2 exactly')

    # Every term now has a windowed form, so nothing refuses on that ground any more. What does
    # refuse is `PPP` without a three-anchor window: it is the one block whose kernel has three
    # anchor pairs, and `window3=None` is legal above, where it means "the box of window2". That
    # makes it an easy thing to pass by accident, and it must say so rather than fail six frames
    # down. The windowed blocks themselves are exercised by the two window tests at the end.
    try:
        compute_spectrum3_covariance(object(), None, obs, theory=theory, terms=('PPP',))
    except ValueError as exc:
        print(f'  windowed PPP without window3 raises ValueError: {str(exc)[:46]}...')
    else:
        raise AssertionError('windowed PPP must refuse window3=None')


# --------------------------------------------------------------------------------------
# the bispectrum blocks
# --------------------------------------------------------------------------------------

def _b000_b202(terms, theory, **options):
    """``max |Cov[B000, B202]|`` relative to the largest entry of the same run.

    Not divided by ``sqrt(var var)``: a single term's diagonal is a contribution, not a variance,
    and the ``P6`` term's own is negative in places, which makes that normalization ``nan``.
    Scaling by the largest block of the run is what ``proj_test.py`` reports and is well defined
    term by term.
    """
    obs = _observable(ells2=(), ells3=((0, 0, 0), (2, 0, 2)))
    cov = compute_spectrum3_covariance(MeshAttrs(boxsize=BOXSIZE, meshsize=64), None, obs,
                                       theory=theory, shotnoise=1995., terms=terms, **options)
    value = np.asarray(cov.value())
    return float(np.max(np.abs(value[:NK, NK:2 * NK])) / np.max(np.abs(value)))


def test_isotropic_kills_b000_b202(atol=1e-10, pt_atol=5e-3, p6_size=1 << 10):
    r"""``Cov[B000, B202]`` vanishes identically for an isotropic theory.

    The :math:`L = 2` weight integrates to zero against an orientation-independent bispectrum, so
    this entry is not a generic covariance element: it is zero unless the theory is anisotropic,
    and its value measures *how* anisotropic each term is. Both variances can be right while it is
    wrong, because they are insensitive to the sign structure it exposes -- which is what makes it
    the sharpest statement available about the angular machinery.

    The theory is a constant at every order, so it is isotropic by construction rather than by an
    argument about redshift-space kernels -- and, more importantly, so that the check needs no
    Monte Carlo convergence. A constant factors out of the integrand, leaving the multipole
    orthogonality of the quadrature itself, which the shared-node design makes exact. That is why
    all four terms reach machine precision here, ``P6`` included, even though its
    seven-dimensional integral is sampled by scrambled Sobol rather than gridded. The same trick
    is what ``proj_test.py`` in the campaign this module came from relies on.

    Do not soften this into a check on a *varying* isotropic theory, such as tree level at
    ``f = 0``. The zero is still analytic there, but ``P6`` then reaches it only as
    :math:`1/\sqrt{N}`: at the cheap 64-point setting the residual is 0.55, which is
    indistinguishable from a broken projection unless one knows to look.

    ``PT`` is the one term that does *not* reach machine precision, and it is the soft-momentum
    mask that stops it -- see ``qmin`` on :func:`jaxpower.cov3._bb_tie_terms`, which the block
    cannot do without. The mask itself is orientation independent, so the argument above should
    still hold exactly; what spoils it is :func:`jaxpower.cov3._frame`, whose azimuth origin for a
    derived leg is chosen by a branch on ``|a_z| < 0.9``. That branch *is* line-of-sight
    dependent, and it costs nothing for a smooth integrand -- the azimuth is integrated over a
    full period, which the midpoint rule does exactly -- but the mask is discontinuous, so the
    azimuthal sum leaves a residue and the residue inherits the frame's line-of-sight dependence.

    It converges away: 2.25e-3 / 1.73e-3 / 2.28e-4 / 8.06e-5 / 5.37e-5 at ``order5`` 6 / 8 / 12 /
    16 / 24, a factor 42 for a factor 4 in order. ``pt_atol`` is set at the order-6 value this
    test runs at, so a regression that breaks the projection outright still fails here.
    """
    def const(value=1.):
        def theory(fields):
            return lambda *legs: value * jnp.ones_like(jnp.asarray(legs[0])[..., 0])
        return theory

    for term in ('PPP', 'BB', 'PT', 'P6'):
        r = _b000_b202((term,), const(), order5=6, order3=6, p6_size=p6_size)
        print(f'  {term:<4} max |Cov[B000, B202]| / max |block| = {r:.3e}')
        assert r < (pt_atol if term == 'PT' else atol), (term, r)


def test_terms_are_additive(rtol=1e-10):
    """The eight terms partition the total, so their covariances must sum to it.

    This catches a term that is silently dropped, double counted, or that leaks into a block it
    does not belong in -- none of which shows up in the total alone.
    """
    from jaxpower.cov3 import TERMS

    mattrs = MeshAttrs(boxsize=BOXSIZE, meshsize=64)
    obs = _observable()
    theory = _tree_theory()
    kw = dict(theory=theory, shotnoise=1995., **CHEAP)
    total = np.asarray(compute_spectrum3_covariance(mattrs, None, obs, **kw).value())
    parts = sum(np.asarray(compute_spectrum3_covariance(mattrs, None, obs, terms=(t,), **kw).value())
                for t in TERMS)
    nz = np.abs(total) > 0
    err = np.max(np.abs(parts[nz] / total[nz] - 1.))
    print(f'  sum of {len(TERMS)} terms vs total: max |ratio - 1| = {err:.3e}')
    assert err < rtol, err


# --------------------------------------------------------------------------------------
# the survey window
# --------------------------------------------------------------------------------------

WINDOW_FILE = Path('_tests') / 'window_fkp2_cov_split.h5'
WINDOW3_FILE = Path('_tests') / 'window_fkp3_cov_split.h5'

#: Seed for the disjoint random subsamples each window factor is painted from. A group of ``n``
#: fields is an integral of ``n`` window fields, and painting them all from the *same* randoms
#: leaves the catalogue's self-pairs in it -- its own discreteness, not the survey's geometry.
#: The damage is worst where the group is largest and where the window is read at zero
#: separation, which together is exactly the effective volume :math:`1 / Q_W(s \to 0)` that
#: normalises every connected term in the survey path and the exact vector tie of
#: :func:`jaxpower.cov3._c22_tie`.
#:
#: Measured on this window, a uniform box where every grouping must give ``V Q_W(0) = 1`` exactly:
#:
#: =========  ======  ======  ======  ======
#: build      2|2     2|3     2|4     3|3
#: =========  ======  ======  ======  ======
#: unsplit    1.1246  1.2052  1.2782  1.3265
#: split      0.9984  1.0122  1.0270  1.0382
#: =========  ======  ======  ======  ======
#:
#: The unsplit excess grows with the field count and falls with the random density -- 3|3 reaches
#: 1.0443 at 100x this density -- which is how a self-pair term behaves and how the survey's
#: geometry does not. Split, it stops depending on the density at all (1.038 / 1.043 / 1.041 over
#: three decades), so what is left is the 64^3 mesh resolving the box's edges; at 128^3 the same
#: numbers are 0.977 / 0.986 / 0.992 / 0.993.
WINDOW_SPLIT = 0


def _survey_window3(boxsize=2000., meshsize=64):
    """The *three*-anchor covariance window of the same uniform box, cached on disk.

    ``PPP`` is the one windowed block whose kernel has three anchor pairs rather than two, so it
    needs this and not just :func:`_survey_window`; passing ``window3=None`` raises.

    Only the ``(0, 0, 0)`` channel is computed, which keeps the build affordable and is what the
    two existing users of this function do. The windowed ``PPP`` block reconstructs its kernel in
    the ``S`` channels of ``_ppp_channels``, so the higher ones are simply absent here; that is a
    truncation of the test, not of the block, and a scan from 8 to 29 channels was measured to move
    the block by 0.1%.
    """
    from jaxpower import generate_uniform_particles, interpolate_window_function
    from jaxpower.cov3 import compute_fkp3_covariance_window

    mattrs = MeshAttrs(boxsize=boxsize, boxcenter=[0., 0., 1200.], meshsize=meshsize)
    if WINDOW3_FILE.exists():
        window = types.read(WINDOW3_FILE)
    else:
        inner = mattrs.clone(boxsize=boxsize / 2., meshsize=meshsize)
        randoms = generate_uniform_particles(inner, int(1e-4 * inner.boxsize.prod()),
                                             seed=32).clone(attrs=mattrs)
        window = compute_fkp3_covariance_window(randoms, edges={'step': 40.}, interlacing=2,
                                               resampler='tsc', los='local', buffer_size=50,
                                               ells=[(0, 0, 0)], split=WINDOW_SPLIT)
        WINDOW3_FILE.parent.mkdir(parents=True, exist_ok=True)
        window.write(WINDOW3_FILE)
    # The three-anchor poles are stored raveled over the (s1, s2) pair and must be unraveled
    # before interpolation, unlike the two-anchor ones.
    window = window.map(lambda pole: pole.unravel())
    return interpolate_window_function(window, coords=jnp.logspace(-3, 4, 1024), order=3)


def _survey_window(boxsize=2000., meshsize=64):
    """The two-anchor covariance window of a uniform box of randoms, cached on disk."""
    from jaxpower import generate_uniform_particles, interpolate_window_function
    from jaxpower.cov3 import compute_fkp2_covariance_window

    mattrs = MeshAttrs(boxsize=boxsize, boxcenter=[0., 0., 1200.], meshsize=meshsize)
    if WINDOW_FILE.exists():
        window = types.read(WINDOW_FILE)
    else:
        inner = mattrs.clone(boxsize=boxsize / 2., meshsize=meshsize)
        randoms = generate_uniform_particles(inner, int(1e-4 * inner.boxsize.prod()),
                                             seed=32).clone(attrs=mattrs)
        window = compute_fkp2_covariance_window(randoms, edges={'step': 40.}, interlacing=2,
                                                resampler='tsc', los='local', group_sizes=(2, 3, 4),
                                                max_total_size=6, ells=[0, 2, 4],
                                                split=WINDOW_SPLIT)
        WINDOW_FILE.parent.mkdir(parents=True, exist_ok=True)
        window.write(WINDOW_FILE)
    return mattrs, interpolate_window_function(window, coords=jnp.logspace(-3, 4, 1024), order=3)


def test_window_gaussian_pp_matches_cov2(rtol=1e-10):
    """The windowed Gaussian ``Cov[P, P]`` against ``cov2``, an independent implementation.

    This is the one windowed block with a reference in the package, and the only cutsky check
    here that is a comparison rather than a limit. ``cov2`` must be called with
    ``flags=['smooth', 'fftlog']``: plain ``'smooth'`` silently picks direct spherical-Bessel
    summation for the window block, while ``compute_QW_AB`` only ever uses the FFTlog route, and
    comparing the two methods gives an ell-dependent mismatch that looks like a normalization bug.
    """
    from jaxpower import BinMesh2SpectrumPoles, Mesh2SpectrumPole, Mesh2SpectrumPoles
    from jaxpower.cov2 import compute_spectrum2_covariance

    mattrs, window = _survey_window()
    theory = _tree_theory()
    shotnoise = 1. / 1e-4
    bin2 = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01, 'min': 0.01}, ells=(0, 2, 4))
    observable = types.ObservableTree([Mesh2SpectrumPoles(
        [Mesh2SpectrumPole(k=bin2.xavg, k_edges=bin2.edges, nmodes=bin2.nmodes,
                           num_raw=jnp.zeros_like(bin2.xavg), ell=ell) for ell in bin2.ells])],
        fields=[(0, 0)])
    mine = np.block(compute_spectrum3_covariance(window, None, observable, theory=theory,
                                                 shotnoise=shotnoise, terms=('PP',),
                                                 cache={}).value())

    # cov2 takes multipole tables, not a callable: project the same P(k, mu).
    power = theory((0, 0))
    mu, weights = np.polynomial.legendre.leggauss(20)
    k = np.asarray(bin2.xavg)

    def multipole(ell):
        grid = jnp.asarray(k)[:, None] * jnp.ones_like(jnp.asarray(mu))[None, :]
        cosine = jnp.ones_like(jnp.asarray(k))[:, None] * jnp.asarray(mu)[None, :]
        kvec = jnp.stack([grid * jnp.sqrt(1 - cosine**2), jnp.zeros_like(grid), grid * cosine],
                         axis=-1)
        out = (2 * ell + 1) / 2. * np.sum(np.asarray(power(kvec))
                                          * np.polynomial.legendre.Legendre.basis(ell)(mu)
                                          * weights, axis=-1)
        return out + shotnoise if ell == 0 else out

    table = Mesh2SpectrumPoles([
        Mesh2SpectrumPole(k=k, k_edges=bin2.edges, nmodes=bin2.nmodes,
                          num_raw=jnp.asarray(multipole(ell)), ell=ell) for ell in bin2.ells])
    # cov2 wants flat (a, b, c, d) field labels; the covariance window stores grouped ones.
    items, fields = [], []
    for label, item in window.items():
        first, second = label['fields1'], label['fields2']
        if len(first) == 2 and len(second) == 2:
            items.append(item)
            fields.append(tuple(first) + tuple(second))
    reference = np.block(compute_spectrum2_covariance(
        types.ObservableTree(items, fields=fields), table, flags=['smooth', 'fftlog']).value())

    nk = len(k)
    err = np.max(np.abs(mine[:nk, :nk] - reference[:nk, :nk])) / np.abs(reference[:nk, :nk]).max()
    print(f'  windowed Cov[P0, P0] vs cov2: max |difference| / max |reference| = {err:.3e}')
    assert err < rtol, err


def test_window_periodic_approximation(atol=0.25, rtol=0.1):
    r"""A window that is a uniform box must reproduce the box covariance.

    The agreement is not exact -- the FKP selection has real boundary effects, the window is
    painted on a 64^3 mesh that resolves the box's edges only so well, and each block carries the
    approximations of ``desi-cov3-notes/covariance.tex``. What is asserted is that each windowed
    block lands near one, and that the blocks whose box limit is a *convergence* statement stay
    put as the quadrature is refined.

    The box reference is recomputed at each order rather than once at the default, which matters:
    the ``(2, 2)`` tie's box term converges only around ``order5`` 16, so a fixed order-8
    reference reads its own drift as the window's. Measured with the theory and binning below, as
    the median over the entries where the box answer exceeds 5% of its maximum:

    ====  ==================  ======================  =========
    term  order               median windowed / box   stable?
    ====  ==================  ======================  =========
    PB    ``order3`` 6, 10    0.909, 0.911            yes
    BB    ``order5`` 6, 10    0.955, 0.983            converging
    PPP   ``order5`` 6, 8     1.102, 1.159            converging
    PT    ``order5`` 6, 10    1.034, 1.050            yes
    ====  ==================  ======================  =========

    ``PT`` is flat because its windowed block *is* the box block with Eq. (47)'s mode count
    taken in the window's effective tie volume: the vector tie is consumed exactly, so almost the
    only difference between the two sides of this ratio is that one number, and 1.027 is that
    number -- the 2|4 grouping's ``V Q_W(s -> 0)`` on this window, see :data:`WINDOW_SPLIT`. It
    reads 1.03-1.05 rather than 1.027 because two other things differ: the closure floor and the
    soft-momentum mask, which each side takes from its own volume, so their thresholds differ by
    :math:`1.027^{1/3}` and the mask's discontinuity converges at its own rate on each. The same
    construction serves ``BB``'s
    ``(2, 2)`` pairing, which lands at 1.010 on its own; the rest of ``BB`` is the channel
    expansion and is what leaves the block at 0.98.

    Note what this test cannot see. Every block here evaluates a power spectrum on a closure leg,
    and where that leg is unconstrained the integral *diverges* -- the box ``BB`` block was still
    growing 18% per step at ``order5`` 32, and so was the windowed one, in step, so their ratio
    looked converged. See ``closure_min`` on :func:`jaxpower.cov3._bb_tie_terms`. A ratio of two
    quantities that share a defect is blind to it; only the absolute convergence scan shows it.

    ``PPP`` converges but has not arrived by order 8, so it is not asserted to be stable; its
    spread also reaches 2.2 on one entry, the ``(2, 0, 2)`` blocks being the slow ones. It used to
    *diverge* here -- 0.969 / 1.073 / 1.179 / 1.236 / 1.283 at ``order5`` 4 to 12, still climbing
    -- and both sides of that ratio did. The cause was the same branch point that
    :func:`jaxpower.cov3._c22_tie` meets: Eq. (B7) puts a power spectrum on the unprimed closure
    leg in every permutation, and where nothing ties that leg to a primed bin it runs to zero,
    where :math:`P(k_3) \sim k_3^{-1.5}` against the measure's :math:`k_3 dk_3` leaves a
    :math:`k_3^{-0.5}` square root. Floored (see ``closure_min`` on
    :func:`jaxpower.cov3._triangle_nodes`) the box block settles by ``order3`` 24 where it had
    been moving 1.7% per step at 48, and on the ``(0, 0, 0)`` block the windowed ratio now runs
    1.027 / 1.064 / 1.081 / 1.084 / 1.085 at ``order5`` 6 / 8 / 12 / 16 / 20.

    What is left is not a resolution problem and does not move with order at all: run with a
    *constant* theory, where no soft spectral edge exists anywhere and the box side is exact at
    any order, the same block gives 0.9180 at orders 6, 8, 12, 16 and 20 alike. That 8% is the
    three-anchor window's own normalisation on a 64^3 mesh, the ``PPP`` counterpart of the 1.027
    that ``PT`` sits at.

    A block that is wrong in principle rather than merely unresolved fails this even though it may
    look fine at a single order. Two that did: summing ``BB``'s doubly-derived ``(2, 2)`` tie
    through the channel expansion, which no finite ``(L, L')`` window expansion can represent,
    takes its ratio to 3.6; and the channel-expanded ``PT``, since removed, ran
    3.89 / 2.27 / 0.99 / 0.77 / 0.45 at ``order5`` 4 / 6 / 8 / 10 / 12, passing through one rather
    than settling there.
    """
    from jaxpower import (BinMesh2SpectrumPoles, BinMesh3SpectrumPoles, Mesh2SpectrumPole,
                          Mesh2SpectrumPoles, Mesh3SpectrumPoles)
    from jaxpower.types import Mesh3SpectrumPole

    mattrs, window = _survey_window()
    box = mattrs.clone(boxsize=mattrs.boxsize[0] / 2., meshsize=64)
    theory = _tree_theory()
    shotnoise = 1. / 1e-4
    bin2 = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.02, 'min': 0.02}, ells=(0, 2))
    bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.02, 'min': 0.02},
                                 ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama-diagonal')
    observable = types.ObservableTree([
        Mesh2SpectrumPoles([Mesh2SpectrumPole(k=bin2.xavg, k_edges=bin2.edges,
                                              nmodes=bin2.nmodes,
                                              num_raw=jnp.zeros_like(bin2.xavg), ell=ell)
                            for ell in bin2.ells]),
        Mesh3SpectrumPoles([Mesh3SpectrumPole(k=bin3.xavg, k_edges=bin3.edges,
                                              nmodes=bin3.nmodes[i],
                                              num_raw=jnp.zeros_like(bin3.xavg[..., 0]),
                                              basis=bin3.basis, ell=ell)
                            for i, ell in enumerate(bin3.ells)])], fields=[(0, 0), (0, 0, 0)])
    n2 = 2 * len(bin2.xavg)
    spectrum, bispectrum = slice(None, n2), slice(n2, None)
    # `PPP` is the only one that needs the three-anchor window, and it is built only for it: the
    # (0, 0, 0)-channel build is the expensive part of this test.
    blocks = {'PB': (spectrum, bispectrum, 'order3', (6, 10), True),
              'BB': (bispectrum, bispectrum, 'order5', (6, 10), False),
              'PPP': (bispectrum, bispectrum, 'order5', (6, 8), False),
              'PT': (bispectrum, bispectrum, 'order5', (6, 10), True)}

    for term, (rows, cols, name, orders, stable) in blocks.items():
        window3 = _survey_window3() if term == 'PPP' else None
        ratios = []
        for value in orders:
            # Same order on both sides: see the note above on the (2, 2) tie's own convergence.
            reference = np.asarray(compute_spectrum3_covariance(
                box, box, observable, theory=theory, shotnoise=shotnoise, terms=(term,),
                **{name: value}).value())[rows, cols]
            keep = np.abs(reference) > 0.05 * np.abs(reference).max()
            windowed = np.asarray(compute_spectrum3_covariance(
                window, window3, observable, theory=theory, shotnoise=shotnoise, terms=(term,),
                cache={}, **{name: value}).value())[rows, cols]
            ratio = windowed[keep] / reference[keep]
            ratios.append(np.median(ratio))
            print(f'  {term} {name}={value:<3} median windowed / box = {ratios[-1]:.3f}'
                  f'  [{ratio.min():.3f}, {ratio.max():.3f}]')
        assert abs(ratios[-1] - 1.) < atol, (term, ratios)
        if stable:
            # Not "improves": these two are not converging towards one from anywhere, they are
            # already where they will stay, and a block that drifts is telling us something.
            assert abs(ratios[-1] / ratios[0] - 1.) < rtol, (term, 'not stable', ratios)


def test_window_double_closure_tie(order=16, rtol=0.025):
    r"""The ``(2, 2)`` tie against a closed form, and against the box under a window.

    This pairing has no reference anywhere else: it is the one the channel expansion cannot
    represent, so there is no second implementation of it to compare with. But with ``B``
    constant, ``S_{000} = 1`` and a diagonal binning it is pure geometry, and can be done by hand.
    Both closure legs are tied, so the term is the overlap of the two triangles' closure-leg mode
    densities. For a shell pair,

    .. math::
        \rho(q) = \int d^3a\, d^3b\, \Theta_1(a) \Theta_2(b)\, \delta^3(a + b - q)
                 = 2\pi k_1 k_2 \Delta k^2 / q

    and the term is :math:`\int d^3q\, \rho(q) \rho'(q)` over the *overlap* of the two ranges
    :math:`q \in [0, 2k]`, divided by the four bins' mode counts. Everything cancels but

    .. math::
        C(a, b) \propto \min(k_a, k_b) / (k_a^2 k_b^2).

    The overall constant is not predicted -- it carries the estimator normalisation -- so what is
    compared is the shape, normalised on the diagonal. It lands within 2.0e-2 at ``order`` 16,
    which is the size of the :math:`O((\Delta k / k)^2)` that evaluating the two binned legs at
    their bin centres leaves behind. This is what pins the measures, the mode count and the
    anchoring average of :func:`jaxpower.cov3._c22_tie`, and it pins them in the geometry where
    the two anchorings provably agree; with a real theory they part by :math:`O(\Delta k / k)`,
    which is the finite bin width and not the quadrature.

    The second half is the windowed block against the box. Since the windowed ``(2, 2)`` is the
    box's with Eq. (47)'s mode count taken in ``1 / Q_W(s -> 0)``, the ratio must be exactly the
    window's own ``V Q_W(0)`` for the 3|3 grouping, at every order and every entry -- which is a
    sharper statement than "near one" and catches anything that leaks in besides the volume.
    """
    from jaxpower import BinMesh3SpectrumPoles, Mesh3SpectrumPoles
    from jaxpower.types import Mesh3SpectrumPole
    from jaxpower.cov3 import (BoxGeometry, Spectra, SurveyGeometry, _c22_tie, _group,
                               _tie_volume)

    def bb_const(value=1.):
        # `Spectra.connected` hands an n-point its n - 1 independent legs: two for a bispectrum.
        def theory(fields):
            if len(fields) != 3:
                return None
            return lambda k1, k2: value * jnp.ones_like(jnp.asarray(k1)[..., 0])
        return theory

    # Narrow bins well away from k = 0: the closed form uses each bin's centre for its two binned
    # legs, so it is a statement about the substitution only where Delta k / k is small.
    mattrs = MeshAttrs(boxsize=1000., meshsize=128)
    bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01, 'min': 0.1, 'max': 0.2},
                                 ells=[(0, 0, 0)], basis='sugiyama-diagonal')
    observable = types.ObservableTree([Mesh3SpectrumPoles(
        [Mesh3SpectrumPole(k=bin3.xavg, k_edges=bin3.edges, nmodes=bin3.nmodes[0],
                           num_raw=jnp.zeros_like(bin3.xavg[..., 0]), basis=bin3.basis,
                           ell=(0, 0, 0))])], fields=[(0, 0, 0)])
    g = _group(observable)[0]
    k = np.asarray(g['k'])
    assert np.allclose(k[0], k[1]), 'the closed form assumes a diagonal binning'
    k = k[0]
    volume = float(np.prod(np.asarray(mattrs.boxsize)))
    sp = Spectra(theory=bb_const(), shotnoise=0.)

    closed = np.minimum(k[:, None], k[None, :]) / (k[:, None]**2 * k[None, :]**2)
    block = np.asarray(_c22_tie(sp, g, g, BoxGeometry(volume), order, 1., 'BB')[0][0])
    shape = block / closed
    shape = shape / np.mean(np.diag(shape))
    error = np.abs(shape - 1.).max()
    print(f'  (2, 2) tie vs closed form, normalized on the diagonal: max |ratio - 1| = '
          f'{error:.3e}')
    assert error < rtol, error

    # And the same term under a window that is a uniform box: the *only* thing that changes is the
    # volume Eq. (47)'s mode count is taken in, so the ratio is that volume and nothing else.
    _, window = _survey_window()
    geometry = SurveyGeometry(window, None, {})
    tie_volume = _tie_volume(geometry)
    expected = volume / tie_volume(g['fields'], g['fields'])
    windowed = np.asarray(_c22_tie(sp, g, g, geometry, order, 1., 'BB',
                                   tie_volume=tie_volume)[0][0])
    ratio = windowed / block
    print(f'  (2, 2) tie windowed / box = [{ratio.min():.4f}, {ratio.max():.4f}], '
          f'V / V_tie = {expected:.4f}')
    assert np.allclose(ratio, expected, rtol=1e-3), (ratio.min(), ratio.max(), expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', action='store_true',
                        help='skip the two tests that build a bispectrum covariance')
    args = parser.parse_args()

    fast = [test_theory_receives_independent_legs, test_partial_theory, test_shotnoise_is_pairwise,
            test_calling_conventions, test_gaussian_pp_closed_form]
    slow = [test_isotropic_kills_b000_b202, test_terms_are_additive,
            test_window_gaussian_pp_matches_cov2, test_window_periodic_approximation,
            test_window_double_closure_tie]

    for test in fast + ([] if args.fast else slow):
        print(f'\n=== {test.__name__} ===', flush=True)
        test()
        print(f'  PASS', flush=True)
    print('\nall done')
