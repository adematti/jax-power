"""Validate the TripoSH shape factors for the reference-leg convention.

The bispectrum estimator applies :math:`\\mathcal{L}_L` to the leg that CLOSES the triangle,
``k_3`` (``mesh3.py``: ``meshes[2] * get_legendre(ell3)(mu)`` in the global-LOS branch, and
``Ylms[2][...] * meshes[2]`` in the local-LOS one). The window-matrix derivation, however, was
originally written for :math:`\\mathcal{L}_L(\\hat{q}_1 \\cdot \\hat{x}_3)`. The whole difference
between the two conventions sits in ONE rotational scalar (desi-gqc-notes, sec. "The reference
leg"):

.. math::

    \\mathcal{S}^{(r)}_{\\ell_1\\ell_2L} = \\frac{1}{H_{\\ell_1\\ell_2L}}
    \\sum_{m_1 m_2 M} \\begin{pmatrix}\\ell_1&\\ell_2&L\\\\m_1&m_2&M\\end{pmatrix}
    y_{\\ell_1 m_1}(\\hat{k}_1)\\, y_{\\ell_2 m_2}(\\hat{k}_2)\\, y_{LM}(\\hat{k}_r),

evaluated at the output triangle, and its theory-side analogue :math:`\\Sigma^{(3)}`. For
:math:`r = 1` this collapses to :math:`\\mathcal{L}_{\\ell_2}(\\cos\\theta_{12})` -- what the code
used to apply; for :math:`r = 3` it does not, and is no longer a single Legendre polynomial.

This file checks the closed forms against the definition evaluated in an ARBITRARY frame, which is
the only test that is blind to the algebra used to derive them.
"""

import itertools

import numpy as np
from scipy.special import lpmv


def y(ell, m, theta, phi=0.):
    """Reduced complex harmonic y_lm = sqrt(4 pi / (2 ell + 1)) Y_lm, Condon-Shortley."""
    from scipy.special import factorial
    norm = np.sqrt(factorial(ell - m) / factorial(ell + m))
    return norm * lpmv(m, ell, np.cos(theta)) * np.exp(1j * m * phi)


def wigner_3j(*args):
    from sympy.physics.wigner import wigner_3j as w3j
    return float(w3j(*args))


def angles(vec):
    """(theta, phi) of a cartesian vector."""
    x, yy, z = vec
    return np.arctan2(np.sqrt(x**2 + yy**2), z), np.arctan2(yy, x)


def shape_factor_frame(ell1, ell2, L, k1, k2, kr):
    """The DEFINITION: the full triple sum, in whatever frame the three vectors are given.

    A rotational scalar, so the answer may not depend on the frame -- that invariance is what the
    closed forms below are tested against.
    """
    H = wigner_3j(ell1, ell2, L, 0, 0, 0)
    t1, p1 = angles(k1 / np.linalg.norm(k1))
    t2, p2 = angles(k2 / np.linalg.norm(k2))
    tr, pr = angles(kr / np.linalg.norm(kr))
    toret = 0. + 0.j
    for m1, m2, M in itertools.product(range(-ell1, ell1 + 1), range(-ell2, ell2 + 1),
                                       range(-L, L + 1)):
        if m1 + m2 + M != 0: continue
        toret += (wigner_3j(ell1, ell2, L, m1, m2, M)
                  * y(ell1, m1, t1, p1) * y(ell2, m2, t2, p2) * y(L, M, tr, pr))
    assert abs(toret.imag) < 1e-12, toret
    return toret.real / H


def cos_theta(ka, kb, kc):
    """Interior angle between legs a and b of a closed triangle (opposite leg c)."""
    return (kc**2 - ka**2 - kb**2) / (2. * ka * kb)


def shape_factor_leg1(ell1, ell2, L, k1, k2, k3):
    """Closed form for r = 1: a single Legendre in the 1-2 opening angle."""
    from numpy.polynomial import legendre
    return legendre.legval(cos_theta(k1, k2, k3), [0] * ell2 + [1])


def shape_factor_leg3(ell1, ell2, L, k1, k2, k3):
    """Closed form for r = 3, desi-gqc-notes eq. (shape_factor_leg3).

    In the frame k_3 = z, M = 0 is forced and the two remaining legs sit at opposite azimuths,
    which is where the (-1)^m comes from.
    """
    H = wigner_3j(ell1, ell2, L, 0, 0, 0)
    t31 = np.arccos(np.clip(cos_theta(k3, k1, k2), -1., 1.))
    t32 = np.arccos(np.clip(cos_theta(k2, k3, k1), -1., 1.))
    toret = 0.
    for m in range(-min(ell1, ell2), min(ell1, ell2) + 1):
        toret += ((-1)**m * wigner_3j(ell1, ell2, L, m, -m, 0)
                  * y(ell1, m, t31).real * y(ell2, -m, t32).real)
    return toret / H


def theory_factor_leg3(ell1, ell2, L, M, k1, k2, k3):
    """Closed form for the THEORY-side factor, desi-gqc-notes eq. (theory_factor_leg3).

    Replaces ``3j(l1', l2', L'; 0, -M', M') y_{l2'}^{-M'}(cos theta_12', 0)``. Unlike the output
    side it carries no 1/H (that H cancels against the Sugiyama denominator).
    """
    t31 = np.arccos(np.clip(cos_theta(k3, k1, k2), -1., 1.))
    t32 = np.arccos(np.clip(cos_theta(k2, k3, k1), -1., 1.))
    toret = 0.
    for mu in range(-ell1, ell1 + 1):
        mu2 = -mu - M
        if abs(mu2) > ell2: continue
        toret += ((-1)**mu2 * wigner_3j(ell1, ell2, L, mu, mu2, M)
                  * y(ell1, mu, t31).real * y(ell2, mu2, t32).real)
    return toret


def triangle_vectors(k1, k2, k3, rot=None):
    """A closed triangle k1 + k2 + k3 = 0, optionally in a random orientation."""
    c31 = np.clip(cos_theta(k3, k1, k2), -1., 1.)
    v3 = np.array([0., 0., k3])
    v1 = k1 * np.array([np.sqrt(1. - c31**2), 0., c31])
    v2 = -v3 - v1
    assert np.allclose(np.linalg.norm(v2), k2), (np.linalg.norm(v2), k2)
    if rot is not None: return rot @ v1, rot @ v2, rot @ v3
    return v1, v2, v3


def random_rotation(rng):
    q = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(q)
    return q * np.sign(np.diag(r))


TRIANGLES = [(1., 1., 1.),           # equilateral
             (1., 1., 0.2),          # squeezed isosceles
             (1., 1., 1.98),         # flattened
             (0.5, 0.8, 1.1),        # scalene
             (0.3, 1.0, 1.2)]


def test_shape_factor(ellmax=4):
    """Both closed forms reproduce the frame-independent definition."""
    rng = np.random.RandomState(42)
    worst1 = worst3 = worstt = 0.
    for (k1, k2, k3) in TRIANGLES:
        rot = random_rotation(rng)
        v1, v2, v3 = triangle_vectors(k1, k2, k3, rot=rot)
        for ell1, ell2, L in itertools.product(range(ellmax + 1), repeat=3):
            if abs(wigner_3j(ell1, ell2, L, 0, 0, 0)) < 1e-10: continue
            # r = 1: reference leg is k_1
            ref = shape_factor_frame(ell1, ell2, L, v1, v2, v1)
            worst1 = max(worst1, abs(ref - shape_factor_leg1(ell1, ell2, L, k1, k2, k3)))
            # r = 3: reference leg is k_3
            ref = shape_factor_frame(ell1, ell2, L, v1, v2, v3)
            worst3 = max(worst3, abs(ref - shape_factor_leg3(ell1, ell2, L, k1, k2, k3)))
            # theory-side factor, all M
            for M in range(-L, L + 1):
                s = theory_factor_leg3(ell1, ell2, L, M, k1, k2, k3)
                if M == 0:  # at M = 0 it must be H x S^(3)
                    H = wigner_3j(ell1, ell2, L, 0, 0, 0)
                    worstt = max(worstt, abs(s - H * shape_factor_leg3(ell1, ell2, L, k1, k2, k3)))
    print(f'max |closed - definition|: r=1 {worst1:.2e}, r=3 {worst3:.2e}, theory(M=0) {worstt:.2e}')
    assert worst1 < 1e-11 and worst3 < 1e-11 and worstt < 1e-11


def test_monopole_unchanged(ellmax=4):
    """At L = 0 the two conventions are IDENTICAL -- no monopole result is affected."""
    worst = 0.
    for (k1, k2, k3) in TRIANGLES:
        for ell in range(ellmax + 1):
            s1 = shape_factor_leg1(ell, ell, 0, k1, k2, k3)
            s3 = shape_factor_leg3(ell, ell, 0, k1, k2, k3)
            worst = max(worst, abs(s1 - s3))
    print(f'max |S^(1) - S^(3)| at L = 0: {worst:.2e}')
    assert worst < 1e-12


def test_quadrupole_differs():
    """At L >= 2 they differ by O(1). Reference values from desi-gqc-notes."""
    k = (1., 1., 1.)
    for (ell1, ell2, L), (s1_ref, s3_ref) in [((2, 0, 2), (1., -1. / 8.)),
                                              ((1, 1, 2), (-1. / 2., 5. / 8.))]:
        s1 = shape_factor_leg1(ell1, ell2, L, *k)
        s3 = shape_factor_leg3(ell1, ell2, L, *k)
        print(f'  equilateral ({ell1}{ell2}L={L}): S^(1) = {s1:+.6f} (ref {s1_ref:+.6f}), '
              f'S^(3) = {s3:+.6f} (ref {s3_ref:+.6f})')
        assert abs(s1 - s1_ref) < 1e-12 and abs(s3 - s3_ref) < 1e-12


def test_library_coeffs_match(ellmax=4):
    """jaxpower's host-side coefficients reproduce the closed forms above."""
    from jaxpower.mesh3 import get_reference_leg_shape_coeffs

    def evaluate(coeffs, ell1, ell2, k1, k2, k3):
        t31 = np.arccos(np.clip(cos_theta(k3, k1, k2), -1., 1.))
        t32 = np.arccos(np.clip(cos_theta(k2, k3, k1), -1., 1.))
        return sum(c * y(ell1, mu1, t31).real * y(ell2, mu2, t32).real for mu1, mu2, c in coeffs)

    worst_out = worst_in = 0.
    for (k1, k2, k3) in TRIANGLES:
        for ell1, ell2, L in itertools.product(range(ellmax + 1), repeat=3):
            if abs(wigner_3j(ell1, ell2, L, 0, 0, 0)) < 1e-10: continue
            got = evaluate(get_reference_leg_shape_coeffs((ell1, ell2, L)), ell1, ell2, k1, k2, k3)
            worst_out = max(worst_out, abs(got - shape_factor_leg3(ell1, ell2, L, k1, k2, k3)))
            for M in range(-L, L + 1):
                got = evaluate(get_reference_leg_shape_coeffs((ell1, ell2, L), m=M, normalize=False),
                               ell1, ell2, k1, k2, k3)
                worst_in = max(worst_in, abs(got - theory_factor_leg3(ell1, ell2, L, M, k1, k2, k3)))
    print(f'library coeffs vs closed form: output {worst_out:.2e}, theory {worst_in:.2e}')
    assert worst_out < 1e-11 and worst_in < 1e-11


def test_monopole_block_unchanged(ellmax=6):
    """The L = 0 <- L' = 0 window block is EXACTLY what the old k_1 convention gave.

    Two things moved at once, and only their product is observable: the 3j (l1', l2', L'; 0, -M', M')
    was taken OUT of ``get_scoccimarro_window_convolution_coeffs`` and folded INTO the theory-side
    angular factor. This checks the product is invariant at L' = 0 -- so no monopole result, and in
    particular none of the mock validation, is affected by the convention change.
    """
    from jaxpower.mesh3 import get_scoccimarro_window_convolution_coeffs

    coeffs = get_scoccimarro_window_convolution_coeffs(0, 0, ellmax=ellmax)
    assert coeffs, 'no L = 0 <- L\' = 0 terms to check'
    worst = 0.
    for se, st, wc in coeffs:
        # old: coeff carried an extra H' = 3j(l1', l2', L'; 0, 0, 0), angular factor was L_{l2'}
        Hp = wigner_3j(*st, 0, 0, 0)
        # new: coeff dropped it, angular factor is Sigma^(3) = H' S^(3)
        for (k1, k2, k3) in TRIANGLES:
            old = Hp * shape_factor_leg1(*st, k1, k2, k3)
            new = theory_factor_leg3(*st, 0, k1, k2, k3)
            worst = max(worst, abs(old - new))
            # and the estimator-side weight, where L = 0 forces l1 = l2
            worst = max(worst, abs(shape_factor_leg1(*se, k1, k2, k3)
                                   - shape_factor_leg3(*se, k1, k2, k3)))
    print(f'L=0 <- L\'=0 block, max |new - old| over {len(coeffs)} terms: {worst:.2e}')
    assert worst < 1e-12


def test_leg_exchange_symmetry(ellmax=4):
    """S^(3) is symmetric under (l1, k1) <-> (l2, k2); S^(1) is NOT.

    This is the decisive property: the window matrix enumerates k1' <-> k2' as equivalent for
    every L', which is only legitimate when the reference leg is the one left untouched, k_3.
    """
    worst3 = worst1 = 0.
    for (k1, k2, k3) in TRIANGLES:
        for ell1, ell2, L in itertools.product(range(ellmax + 1), repeat=3):
            if abs(wigner_3j(ell1, ell2, L, 0, 0, 0)) < 1e-10: continue
            worst3 = max(worst3, abs(shape_factor_leg3(ell1, ell2, L, k1, k2, k3)
                                     - shape_factor_leg3(ell2, ell1, L, k2, k1, k3)))
            worst1 = max(worst1, abs(shape_factor_leg1(ell1, ell2, L, k1, k2, k3)
                                     - shape_factor_leg1(ell2, ell1, L, k2, k1, k3)))
    print(f'leg-exchange asymmetry: S^(3) {worst3:.2e} (must vanish), S^(1) {worst1:.2e} (does not)')
    assert worst3 < 1e-12
    assert worst1 > 0.1


if __name__ == '__main__':
    test_shape_factor()
    test_monopole_unchanged()
    test_quadrupole_differs()
    test_leg_exchange_symmetry()
    test_library_coeffs_match()
    test_monopole_block_unchanged()
    print('all shape-factor checks passed')
