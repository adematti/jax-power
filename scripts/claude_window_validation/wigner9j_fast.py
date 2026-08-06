r"""Fast Wigner 9j via the Racah single-sum over 6j, as a drop-in for sympy's.

sympy's `wigner_9j` is exact-rational and costs 20-55 ms per distinct symbol at the orders the
scoccimarro window needs, while `wigner_6j` is ~free. That is what makes ellmax=16 expensive: the
cost is (number of DISTINCT 9j symbols) x 20-55 ms, and jaxpower already lru_caches the calls, so
memoisation cannot help further.

The identity (Racah):

    {a b c}
    {d e f}  = sum_x (-1)^(2x) (2x+1) {a b c} {d e f} {g h i}
    {g h i}                           {f i x} {b x h} {x a d}

with x over max(|a-i|, |d-h|, |b-f|) <= x <= min(a+i, d+h, b+f). Each 9j therefore costs ~20-30 6j
evaluations, all of which are cheap.

Validated against sympy below over a wide range, including the zero-containing symbols the analytic
coefficient needs ({0 0 0; p q L; p q L}).
"""
from functools import lru_cache
import numpy as np


from scipy.special import gammaln

_lf = lambda n: gammaln(n + 1.)        # log(n!)


def _delta_log(a, b, c):
    """log of the Racah triangle coefficient Delta(abc); -inf if the triangle fails."""
    if a + b - c < 0 or a - b + c < 0 or -a + b + c < 0:
        return -np.inf
    return 0.5 * (_lf(a + b - c) + _lf(a - b + c) + _lf(-a + b + c) - _lf(a + b + c + 1))


@lru_cache(maxsize=None)
def _w6j(a, b, c, d, e, f):
    """Racah formula for the 6j in FLOATING point, via log-factorials.

    sympy's exact-rational wigner_6j is NOT cheap -- an earlier timing that suggested it was had
    measured 20 calls to the SAME symbol, i.e. sympy's own cache. Doing the 9j as ~30 exact 6j
    evaluations is therefore SLOWER than one exact 9j (measured 0.6x). This float version is what
    makes the Racah route pay.
    """
    dl = (_delta_log(a, b, c) + _delta_log(a, e, f) + _delta_log(d, b, f) + _delta_log(d, e, c))
    if not np.isfinite(dl):
        return 0.
    a1, a2, a3, a4 = a + b + c, a + e + f, d + b + f, d + e + c
    b1, b2, b3 = a + b + d + e, b + c + e + f, a + c + d + f
    lo, hi = int(max(a1, a2, a3, a4)), int(min(b1, b2, b3))
    if lo > hi:
        return 0.
    tot = 0.
    for t in range(lo, hi + 1):
        lg = (_lf(t + 1) - _lf(t - a1) - _lf(t - a2) - _lf(t - a3) - _lf(t - a4)
              - _lf(b1 - t) - _lf(b2 - t) - _lf(b3 - t))
        tot += (-1)**t * np.exp(dl + lg)
    return float(tot)


def _tri(a, b, c):
    return abs(a - b) <= c <= a + b and (a + b + c) % 1 == 0


@lru_cache(maxsize=None)
def wigner_9j_fast(a, b, c, d, e, f, g, h, i):
    """Racah single-sum 9j. Arguments in the same order as sympy's wigner_9j."""
    # rows and columns must each satisfy the triangle condition, else the symbol vanishes
    for (x, y, z) in ((a, b, c), (d, e, f), (g, h, i), (a, d, g), (b, e, h), (c, f, i)):
        if not _tri(x, y, z):
            return 0.
    lo = max(abs(a - i), abs(d - h), abs(b - f))
    hi = min(a + i, d + h, b + f)
    tot = 0.
    for x in range(int(lo), int(hi) + 1):
        t = _w6j(a, b, c, f, i, x)
        if t == 0.: continue
        u = _w6j(d, e, f, b, x, h)
        if u == 0.: continue
        v = _w6j(g, h, i, x, a, d)
        if v == 0.: continue
        tot += (-1)**(2 * x) * (2 * x + 1) * t * u * v
    return tot


if __name__ == '__main__':
    import time, itertools
    from sympy.physics.wigner import wigner_9j as w9_sympy
    print('--- validation against sympy ---')
    worst, n = 0., 0
    cases = [(0, 0, 0, p, q, L, p, q, L) for p in range(5) for q in range(5) for L in (0, 2, 4)]
    cases += [tuple(t) for t in itertools.product((0, 1, 2, 3, 4), repeat=9)][:400]
    for args in cases:
        ref = float(w9_sympy(*args)); got = wigner_9j_fast(*args)
        d = abs(got - ref)
        if d > worst: worst, wa = d, args
        n += 1
    print(f'  {n} symbols checked, max |fast - sympy| = {worst:.3e}'
          f'{"  at " + str(wa) if worst > 0 else ""}')
    print(f'  => {"PASS" if worst < 1e-12 else "FAIL"}')
    print('\n--- timing (uncached, the orders the window needs) ---')
    for args in ((4, 4, 2, 2, 2, 2, 4, 4, 4), (8, 8, 4, 4, 4, 4, 8, 8, 8), (12, 12, 6, 6, 6, 6, 12, 12, 12)):
        wigner_9j_fast.cache_clear(); _w6j.cache_clear()
        t0 = time.time(); v1 = wigner_9j_fast(*args); t1 = time.time() - t0
        t0 = time.time(); v2 = float(w9_sympy(*args)); t2 = time.time() - t0
        print(f'  9j{args}: fast {t1*1e3:7.2f} ms | sympy {t2*1e3:8.2f} ms | speedup {t2/max(t1,1e-9):6.1f}x '
              f'| agree {abs(v1-v2):.1e}')
