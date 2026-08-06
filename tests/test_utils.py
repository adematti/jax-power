import itertools

import numpy as np

# there is no public wigner_6j wrapper (unlike wigner_3j / wigner_9j); the 9j sums over it, so
# it is worth testing directly rather than only through its caller
from jaxpower.utils import wigner_3j, wigner_9j, _wigner_6j as wigner_6j


def test_wigner_6j():
    """The float Racah 6j against sympy's exact rational one.

    This is the load-bearing piece: `wigner_9j` is a single sum over ~20-30 of these, so any 6j
    error is what would surface there, amplified by the (2x+1) weights.
    """
    from sympy.physics.wigner import wigner_6j as ref
    worst, n = 0., 0
    for args in itertools.product(range(5), repeat=6):
        got, exp = wigner_6j(*args), float(ref(*args))
        worst, n = max(worst, abs(got - exp)), n + 1
    assert n > 10000
    assert worst < 1e-12, f'max |fast - sympy| = {worst:.3e} over {n} 6j symbols'


def test_wigner_9j():
    """The Racah single-sum 9j against sympy's exact one.

    sympy's exact-rational 9j costs 20-200 ms per DISTINCT symbol at the orders the scoccimarro
    window matrix needs, and the calls are already lru_cached so memoisation cannot help -- which
    is what made `ellmax >= 16` impractical before the Racah rewrite. This guards that the
    replacement is numerically equivalent, not merely fast.
    """
    from sympy.physics.wigner import wigner_9j as ref
    # the zero-containing family {0 0 0; p q L; p q L} the analytic window coefficient needs,
    # plus a dense sweep of low orders
    cases = [(0, 0, 0, p, q, L, p, q, L) for p in range(5) for q in range(5) for L in (0, 2, 4)]
    cases += [t for t in itertools.product(range(4), repeat=9)][:600]
    worst, worst_args = 0., None
    for args in cases:
        got, exp = wigner_9j(*args), float(ref(*args))
        if abs(got - exp) > worst: worst, worst_args = abs(got - exp), args
    assert worst < 1e-12, f'max |fast - sympy| = {worst:.3e} at {worst_args}'

    # higher orders, where sympy is slow enough that we only spot-check: these are the symbols
    # ellmax=16 actually reaches
    for args in ((4, 4, 2, 2, 2, 2, 4, 4, 4), (8, 8, 4, 4, 4, 4, 8, 8, 8),
                 (6, 6, 4, 4, 4, 4, 6, 6, 6), (12, 12, 6, 6, 6, 6, 12, 12, 12)):
        assert abs(wigner_9j(*args) - float(ref(*args))) < 1e-12, args


def test_wigner_9j_selection_rules():
    """Every triad must satisfy the triangle condition, else the symbol vanishes identically.

    `_wigner_9j` short-circuits on these before summing, so a broken guard would return a garbage
    partial sum rather than 0 -- and the window matrix sums thousands of such terms, where a
    spurious non-zero would be invisible.
    """
    # rows, columns: all six triads
    for args in ((1, 1, 5, 1, 1, 1, 1, 1, 1),      # row 0 fails: |1-1| <= 5 <= 2 is false
                 (1, 1, 1, 1, 1, 5, 1, 1, 1),      # row 1 fails
                 (1, 1, 1, 1, 1, 1, 1, 1, 5),      # row 2 fails
                 (1, 1, 1, 1, 1, 1, 5, 1, 1),      # column 0 fails
                 (1, 1, 1, 1, 1, 1, 1, 5, 1)):     # column 1 fails
        assert wigner_9j(*args) == 0., args

    # a 9j with a zero entry reduces to a 6j:
    # {a b c; d e f; g h 0} = delta_cf delta_gh (-1)^(b+c+d+g) / sqrt((2c+1)(2g+1)) {a b c; e d g}
    for (a, b, c, d, e) in ((1, 2, 2, 2, 1), (2, 2, 2, 2, 2), (3, 2, 3, 2, 3), (2, 4, 4, 3, 3)):
        for g in range(abs(d - e), d + e + 1):
            got = wigner_9j(a, b, c, d, e, c, g, g, 0)
            exp = ((-1)**(b + c + d + g) / np.sqrt((2 * c + 1.) * (2 * g + 1.))
                   * wigner_6j(a, b, c, e, d, g))
            assert np.allclose(got, exp, atol=1e-12), (a, b, c, d, e, g, got, exp)


def test_wigner_9j_symmetry():
    """The 9j is invariant under transposition, and picks up (-1)^S under an odd row/column swap,
    with S the sum of all nine arguments. Independent of the sympy comparison: it tests the sum
    itself, since the Racah identity treats the nine entries asymmetrically.
    """
    rng = np.random.default_rng(42)
    checked = 0
    while checked < 60:
        m = rng.integers(0, 5, size=(3, 3))
        v = wigner_9j(*m.ravel())
        if v == 0.: continue
        checked += 1
        assert np.allclose(wigner_9j(*m.T.ravel()), v, atol=1e-12), 'transpose'
        S = int(m.sum())
        swapped = m[[1, 0, 2]]                      # one row exchange = odd permutation
        assert np.allclose(wigner_9j(*swapped.ravel()), (-1)**S * v, atol=1e-12), 'row swap'
        swapped = m[:, [0, 2, 1]]                   # one column exchange
        assert np.allclose(wigner_9j(*swapped.ravel()), (-1)**S * v, atol=1e-12), 'column swap'


def test_wigner_3j():
    from sympy.physics.wigner import wigner_3j as ref
    worst = 0.
    for j1, j2, j3 in itertools.product(range(5), repeat=3):
        for m1, m2 in itertools.product(range(-j1, j1 + 1), range(-j2, j2 + 1)):
            m3 = -m1 - m2
            if abs(m3) > j3: continue
            worst = max(worst, abs(wigner_3j(j1, j2, j3, m1, m2, m3) - float(ref(j1, j2, j3, m1, m2, m3))))
    assert worst < 1e-12, f'max |jaxpower - sympy| = {worst:.3e}'


if __name__ == '__main__':

    test_wigner_3j()
    test_wigner_6j()
    test_wigner_9j()
    test_wigner_9j_selection_rules()
    test_wigner_9j_symmetry()
