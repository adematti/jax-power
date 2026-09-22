r"""Analytic covariance of the power spectrum and bispectrum multipoles in a **periodic box**.

This is arXiv:1908.06234 (Sugiyama, Saito, Beutler & Seo) implemented term by term, for the
geometry the paper's own derivation assumes: no survey window, no super-sample term, a single
volume :math:`V` and a single mean density :math:`\bar n`.

It was written from the paper rather than from the implementation it replaces, and shares nothing
with it but the entry-point signature -- which is why the two are worth comparing. That earlier
implementation is kept at :mod:`jaxpower._cov3_legacy`.

Multitracer is complete: ``theory`` is keyed by a field tuple throughout, and the routes the
paper folds into a leading factor -- the two Wick pairings of ``PP``, the two straddle choices of
``PB``, the per-pairing ties of ``PPP`` -- are written out, because cross-spectra break the
degeneracy that made them equal.

THE CUTSKY PATH, AND WHAT IS LEFT OF IT
---------------------------------------
:class:`SurveyGeometry` supplies the two things a window changes: :math:`1/V^{(n)}`, the integral
of the window fields over the *cross* pairing of the two groups, and :math:`Q_{\mathcal W}`, the
kernel that replaces a radial delta. ``PP`` has its windowed integral
(:func:`_block_pp_gaussian_window`); ``T``, ``P5`` and ``P6`` are purely connected and need
nothing but the normalisation, their angular measure being the box's unchanged.

``PB``, ``PPP``, ``BB`` and ``PT`` remain. ``desi-cov3-notes/covariance.tex`` settles their form,
recorded here because deriving it again is the expensive part. That file is the reference of
record; ``desi-gqc-notes/ms.tex`` holds an older copy of the same sections which is *not* kept in
step with it and still carries corrected errors (the ``Cov^PPP`` prefactor, the ``PT`` tie sign,
the window-multipole normalisation), so do not consult it for these equations.

* Each loses its delta to a window kernel and *gains back* the integral the delta had removed.
  How much that costs differs by block, and the two bullets below say why: ``BB`` and ``PPP`` end
  up on ``P6``'s measure -- seven angles across the two triangles, against the box's five, which
  is the real cost of the cutsky path -- while ``PB`` collapses back to three and ``PT`` keeps its
  tie and stays at five.
* Which kernel: ``BB`` takes :math:`Q_{\mathcal W}^{(2)}` between two 3-field groups, ``PB``
  between a 2-field and a 3-field group, and ``PPP`` the three-anchor
  :math:`Q_{\mathcal W}^{(3)}` of :func:`compute_QW_ABC`, which carries four momentum arguments.
  ``PT`` takes *no* kernel: its tie cannot be expanded in one at all, and it is consumed as a
  vector instead, with the window entering through the effective tie volume alone. So does
  ``BB``'s doubly-derived ``(2, 2)`` pairing. See :func:`_block_bb_pt_window` and
  :func:`_c22_tie`; the reason is approximation 1 below, taken to its limit -- a kernel truncated
  in :math:`(L, L')` constrains two magnitudes and two line-of-sight cosines, and those ties need
  a *direction*.
* ``PB`` is cheaper than the rest. Its :math:`\hat k` integral is pure Legendre orthogonality,
  :math:`\int (d\mu/2) \mathcal{L}_\ell \mathcal{L}_{L_1} = \delta_{\ell L_1}/(2\ell + 1)`, so
  only the window channel :math:`L_1 = \ell` survives, the leading :math:`(2\ell + 1)` cancels,
  and the integral collapses back to the triangle's three dimensions. Nothing is lost by this:
  the spectrum side depends on :math:`\hat k` only through :math:`\mathcal{L}_\ell`, which is
  exactly the condition under which the kernel used here is the exact one (below).
* The estimator weights stay :math:`S_{\ell_1\ell_2L}` on each triangle's **literal** legs, never
  permuted by which leg is contracted; every sign flip and dependent momentum belongs to
  :math:`Q_{\mathcal W}`. Tie signs: ``PPP`` ties :math:`k'_{\sigma(i)}` to :math:`\pm k_i`,
  ``BB`` to :math:`+k_i` and ``PT`` to :math:`-k_i` -- the ``sign`` argument of
  :func:`_bb_tie_terms`. The two triangles' *relative* orientation matters in ``PT``, because it
  enters :math:`T^{(N)}` through sums like :math:`k_r + k'_s`.

One trap, met and reverted once: a tied leg that is a triangle's *closure* leg has a magnitude
that varies with the quadrature node, so its window table carries a node axis and has to be
contracted inside the angular integral. Only a tied leg with its own bin edges gives a table that
can be multiplied outside.

Two approximations the cutsky path makes that the box path does not, neither of them visible in
the equations as written:

1. **The two-anchor kernel is truncated to its** :math:`m = 0` **part.** The exact angular content
   of :math:`\mathcal{Q}_{\mathcal W}(\mathbf k - \mathbf k')` is
   :math:`\sum_{LL'q} Q_{LL'q}(k, k') S_{LL'q}(\hat k, \hat k', \hat n)` -- the same tri-polar
   basis the bispectrum estimator uses, since
   :math:`\int d\hat s\, \mathcal{L}_q \mathcal{L}_L \mathcal{L}_{L'} = 4\pi H^2_{LL'q} S_{LL'q}`.
   :func:`compute_QW_AB` keeps only the :math:`m = 0` term, whose form is
   :math:`\mathcal{L}_L(\mu) \mathcal{L}_{L'}(\mu')`, and sums over :math:`q` first. For ``PP``
   and the spectrum side of ``PB`` that is *exact*, because
   :math:`\int d\Omega_{\hat k} \mathcal{L}_\ell\, y^m_L \propto \delta_{m0}\delta_{\ell L}`
   annihilates the rest. For ``BB`` and ``PT`` it is formally not: both estimator weights depend
   on their own triangle's azimuths, so the :math:`m \neq 0` channels survive and carry the
   *relative* azimuth of the two anchors. **Measured, they are worth** :math:`\sim 10^{-4}`: the
   exact sum was implemented and run against the box limit on a uniform-box window, and the median
   ratio over the eight served tie pairings moved 0.7601 -> 0.7600 (at ``order5 = 6``, five
   multipole triplets, 25 entries; restricting that sum to :math:`m = 0` reproduced this block to
   5e-16, which is what pinned the channel bookkeeping). So the truncation is an excellent
   approximation rather than a structural gap, the exact path was removed again as unused, and the
   seventh quadrature axis of :func:`_block_bb_bb_window` is numerically spent integrating a
   constant. ``desi-cov3-notes/covariance.tex`` carries the derivation and the table.
2. **One window serves both clustering and shot noise.** A single :math:`Q_{\mathcal W}` multiplies
   the whole of :math:`P^{(N)}`, :math:`B^{(N)}`, :math:`T^{(N)}`, and the :math:`1/V^{(n)}` are
   pure products of :math:`W = \bar n w`. That is exact only if the shot-noise window
   :math:`S^{ab} \propto \bar n_{ab} w_a w_b` is proportional to
   :math:`W^{ab} = \bar n_a w_a \bar n_b w_b`, i.e.\ only if the shot-noise level
   :math:`\bar n_{ab}/(\bar n_a \bar n_b)` is constant over the survey. In a box it is, and the
   ratio is the :math:`1/\bar n` that turns :math:`P` into :math:`P^{(N)}`; in a survey it is not.
   :mod:`jaxpower.cov2` carries the split (``WW``/``WS``/``SS``) for the two-point case;
   :func:`compute_fkp2_covariance_window` paints one :math:`W` per tracer and does not.

WHAT IS COMPUTED
----------------
==========================  ======================  =================================
block                       paper                   here
==========================  ======================  =================================
``Cov[P, P]`` Gaussian      Eq. (43)                :func:`_block_pp_gaussian`
``Cov[P, P]`` trispectrum   Eqs. (13), (14), (36)   :func:`_block_pp_trispectrum`
``Cov[P, B]`` ``PB``        Eq. (48)                :func:`_block_pb_unconnected`
``Cov[P, B]`` ``P5``        Eqs. (26), (27), (36)   :func:`_block_pb_p5`
``Cov[B, B]`` ``PPP``       Eq. (B7)                :func:`_block_bb_ppp`
``Cov[B, B]`` ``BB``        Eq. (B8)                :func:`_block_bb_bb`
``Cov[B, B]`` ``PT``        Eq. (B9)                :func:`_block_bb_pt`
``Cov[B, B]`` ``P6``        Eqs. (33), (B4)-(B6)    :func:`_block_bb_p6`
==========================  ======================  =================================

The :math:`n`-point spectra themselves come from ``theory``; :func:`jaxpower.pt.make_spectra_redshift_tracer`
supplies the paper's
tree-level ones. Every discreteness term (:math:`P^{(N)}`, :math:`B^{(N)}`, :math:`T^{(N)}`,
:math:`P_5^{(N)}`, and the three shot lines of Eq. 33) is rebuilt here from the *connected*
spectra and one amplitude, exactly as the paper writes them, so ``theory`` must return connected
spectra and carry no stochastic terms of its own.

THREE PLACES THE PUBLISHED EQUATIONS NEED CORRECTING
----------------------------------------------------
Each is used in the corrected form here, and each is checked against the paper's own source
(arXiv:1908.06234, both versions) rather than assumed.

1. **The** :math:`M` **of Eqs. (B7)-(B9) should carry** :math:`H^2 H'^2`, **not** :math:`H H'`.
   Eq. (34) defines the multipole with :math:`N H^2`, Eq. (36) propagates :math:`N N' H^2 H'^2`,
   and Eq. (49) -- the same :math:`PPP` term in the main text -- writes :math:`H^2 H'^2`. Only the
   appendix's definition of :math:`M` disagrees.

2. **Eq. (A11) is missing a topology**: the six labelled stars :math:`K_{1,5}` with prefactor
   :math:`5!`. It is ``theory`` that decides whether they are present, so the switch is
   :func:`jaxpower.pt.make_spectrum_redshift_tracer`'s ``paper=True``, which reproduces the
   paper's :math:`P_6` instead of the complete one. There is nothing to set here.

3. **Eq. (A9)'s multiplicities are wrong** (40 and 36 against 60 and 60). The topologies are
   right. Again :func:`jaxpower.pt.make_spectrum_redshift_tracer`.

Both counts follow from one fact: a tree-level connected :math:`n`-point function is a sum over
labelled trees on its :math:`n` legs, a leg's kernel order being its degree, and there are
:math:`(n-2)!/\prod_i (d_i - 1)!` labelled trees per degree sequence. For :math:`n = 5` that gives
:math:`60 + 60 + 5 = 125 = 5^3`, for :math:`n = 6` it gives
:math:`360 + 720 + 90 + 120 + 6 = 1296 = 6^4`, and the missing six are exactly the stars.

A fourth item used to be listed here and was **wrong**: Eq. (23) does *not* read :math:`1/V^2`.
The paper prints :math:`2 (2\pi)^3 \delta_D(k + k_1)/V` in both arXiv versions. The spurious
:math:`1/V^2` came from this project's own write-up of :math:`{\rm Cov}^{PPP}`, which did carry an
extra :math:`1/V` and has been corrected there. Do not re-add it.

QUADRATURE
----------
Every block reduces to an angular integral of 3, 5 or 7 dimensions once the radial deltas are
resolved. Two things decide how it is done here.

*The sharp bin windows are integrated exactly, not sampled.* Wherever Eq. (45)'s top-hat
:math:`W(k, k')` acts on a *continuous* leg -- the closure leg :math:`k_3 = |k_1 + k_2|` of a
bispectrum bin, or the :math:`k_\alpha`, :math:`k_\beta` of Eqs. (B8), (B9) -- its argument
depends on exactly one integration variable, through
:math:`k_c^2 = k_a^2 + k_b^2 + 2 k_a k_b x`. The window is therefore an interval in :math:`x`,
computed in closed form by :func:`_window_interval`, and the quadrature is placed inside it. A
top-hat sampled on a fixed grid instead converges as :math:`1/n` and is the single largest source
of error in these blocks.

A survey window has no such interval to solve for, so the same sharpness is handled the other way
round: a tie on a closure leg gets its :math:`Q_{\mathcal W}` *cell-averaged* over each node's own
:math:`x` cell rather than sampled at the node (:func:`_closure_measure`). Making the cells
proportional to the Gauss-Legendre weights is what makes that exact -- for a kernel narrower than
one cell, point-sampling can return zero where the cell average returns the right integral. It is
worth ~0.5% on ``BB`` and nothing measurable on ``PT``, because a leg that carries bin *edges* is
already integrated over its bin by the rebin matrix; it bites only where both tied legs are
continuous.

*The 7-dimensional* :math:`P_6` *block is sampled, not gridded.* Its integrand is a
sign-changing sum over 1296 trees whose structure lives on joint diagonals
:math:`|k_i + k'_j| \to 0` that no tensor lattice resolves; a product grid aliases it badly
(measured: still ~50% wrong at 10 nodes per axis, and wrong-signed at 4). Scrambled Sobol points
shared across every bin pair and every multipole pair converge as :math:`\sim 1/N` and, being
shared, leave the correlation matrix far cleaner than independent sampling would.
"""

from functools import partial

import functools
import itertools
import os

import numpy as np
import jax
from jax import numpy as jnp

from .mesh import MeshAttrs, split_particles
from .mesh2 import get_smooth2_window_bin_attrs
from .mesh3 import (BinMesh3CorrelationPoles, compute_mesh3, FKPField,
                    get_smooth3_window_bin_attrs, get_sugiyama_window_convolution_coeffs)
from .types import CovarianceMatrix, ObservableTree
from .utils import get_legendre, get_S, legendre_product, wigner_3j, wigner_9j
from .cov2 import Correlation2Spectrum, matrix_rebin


TWO_PI3 = (2. * np.pi)**3


# ======================================================================================
# geometry
# ======================================================================================

def _norm(v):
    return jnp.sqrt(jnp.sum(v**2, axis=-1))


def _ahat(mu):
    """Unit vector at line-of-sight cosine ``mu``, azimuth zero. The line of sight is ``z``."""
    s = jnp.sqrt(jnp.clip(1. - mu**2, 0., None))
    return jnp.stack([s, jnp.zeros_like(mu), mu], axis=-1)


def _frame(a):
    """An orthonormal pair perpendicular to the unit vector ``a``.

    Any frame gives the same integral -- the azimuth it references is integrated over a full
    period -- so the branch below only has to be non-degenerate, not continuous. The azimuthal
    rule is the uniform midpoint one, which is exact for the band-limited azimuthal dependence
    these integrands have, and so is insensitive to the frame's origin.
    """
    ref = jnp.where(jnp.abs(a[..., 2:3]) < 0.9,
                    jnp.asarray([0., 0., 1.]), jnp.asarray([1., 0., 0.]))
    e2 = jnp.cross(ref, a)
    e2 = e2 / jnp.clip(_norm(e2), 1e-30, None)[..., None]
    return e2, jnp.cross(a, e2)


def _bhat(ahat, x, phi):
    """Unit vector at ``cos`` angle ``x`` from ``ahat``, azimuth ``phi`` about it."""
    e2, e3 = _frame(ahat)
    sx = jnp.sqrt(jnp.clip(1. - x**2, 0., None))
    return (x[..., None] * ahat
            + sx[..., None] * (jnp.cos(phi)[..., None] * e2 + jnp.sin(phi)[..., None] * e3))


def _window_interval(ka, kb, kt, dk):
    r"""The interval in :math:`x = \hat k_a \cdot \hat k_b` where
    :math:`|k_a + k_b| = \sqrt{k_a^2 + k_b^2 + 2 k_a k_b x}` falls in the bin ``(kt, dk)``.

    Returns ``(lo, hi)`` clipped to ``[-1, 1]``; ``hi <= lo`` means the term does not contribute.
    This is Eq. (45)'s top-hat, solved rather than sampled.
    """
    den = 2. * ka * kb
    lo = (jnp.maximum(kt - dk / 2., 0.)**2 - ka**2 - kb**2) / den
    hi = ((kt + dk / 2.)**2 - ka**2 - kb**2) / den
    return jnp.clip(lo, -1., 1.), jnp.clip(hi, -1., 1.)


def _continuum_count(k, dk, volume):
    r"""Eq. (43)'s mode count, :math:`N_{\rm mode}(k) = 4 \pi k^2 \Delta k V / (2\pi)^3`."""
    return 4. * np.pi * k**2 * dk * volume / TWO_PI3


def _inv_nmode(ka, kb, dk, volume, count=None):
    r""":math:`1 / \tilde N_{\rm mode}(k_a, k_b)`, Eq. (47).

    Eq. (47) is exactly :math:`\sqrt{N_{\rm mode}(k_a) N_{\rm mode}(k_b)}` with Eq. (43)'s
    continuum count -- the geometric mean, which is what lets the same expression serve a leg
    tied to a bin and a closure leg whose magnitude is continuous. Written that way it also says
    how to stop using the continuum: pass ``count``, a function giving the number of grid modes
    in a shell of width ``dk`` about ``k``, and the same geometric mean carries the measured
    count through both cases. See :class:`ModeCount` for why that matters.

    A shell holding no modes returns 0 rather than a division by zero: such a configuration is
    not measurable and contributes nothing, which is the discreteness effect itself.
    """
    if count is None:
        return TWO_PI3 / (4. * np.pi * ka * kb * dk * volume)
    na, nb = count(ka), count(kb)
    den = na * nb
    return jnp.where(den > 0., 1. / jnp.sqrt(jnp.where(den > 0., den, 1.)), 0.)


class ModeCount(object):
    r"""The number of grid modes in a shell of width :math:`\Delta k` about :math:`k`.

    Eq. (43) counts modes as :math:`4 \pi k^2 \Delta k V / (2\pi)^3`, the continuum volume of
    the shell. A real measurement counts the grid modes that actually fall in it, and the two
    part company exactly where the paper's own comparison is worst: a bin of width
    :math:`\Delta k = 0.02` at :math:`k = 0.02` in a 1.76 (Gpc/h)^3 box is 3.8 fundamentals
    wide, so it holds 713 modes by the continuum formula against a discrete count that differs by
    tens of per cent, and its effective wavenumber is 0.023 rather than 0.020. Measured against
    600 Gaussian box mocks, the continuum count leaves the two lowest bins 20-70% high while
    every other bin is within the mock scatter.

    The count is exact, not modelled: the grid magnitudes are sorted once and the shell
    population read off by binary search, then tabulated so that a traced (closure-leg)
    magnitude can be interpolated. Because a bin's centre plus and minus :math:`\Delta k / 2`
    *are* its edges, evaluating this at a bin centre returns that bin's own population.
    """

    def __init__(self, mattrs, dk, nk=8192):
        boxsize = np.asarray(mattrs.boxsize) * np.ones(3)
        meshsize = np.asarray(mattrs.meshsize) * np.ones(3, dtype=int)
        axes = [np.fft.fftfreq(int(n), d=float(l) / int(n)) * 2. * np.pi
                for n, l in zip(meshsize, boxsize)]
        knorm = np.sqrt(sum(a**2 for a in np.meshgrid(*axes, indexing='ij'))).ravel()
        knorm.sort()
        self.knorm, self.dk = knorm, float(dk)
        self.k = np.linspace(0., float(knorm[-1]), nk)
        lo = np.searchsorted(knorm, np.maximum(self.k - dk / 2., 0.), side='left')
        hi = np.searchsorted(knorm, self.k + dk / 2., side='right')
        self.table = (hi - lo).astype(float)
        self._k, self._table = jnp.asarray(self.k), jnp.asarray(self.table)

    def __call__(self, k):
        return jnp.interp(jnp.asarray(k), self._k, self._table, left=0., right=0.)

    def ratio_to_continuum(self, k, volume):
        """Diagnostic: measured / Eq. (43)."""
        return np.asarray(self(k)) / _continuum_count(np.asarray(k), self.dk, volume)

    def exact(self, lo, hi):
        """Exact mode population and mode-weighted mean magnitude of the shell ``[lo, hi]``."""
        if getattr(self, '_cum', None) is None:
            self._cum = np.concatenate([[0.], np.cumsum(self.knorm)])
        a = np.searchsorted(self.knorm, np.asarray(lo), side='left')
        b = np.searchsorted(self.knorm, np.asarray(hi), side='right')
        n = (b - a).astype(float)
        kbar = np.where(n > 0, (self._cum[b] - self._cum[a]) / np.where(n > 0, n, 1.), 0.)
        return n, kbar

    def subbins(self, edges, nsub):
        r"""Split each bin radially into ``nsub`` sub-shells of equal *width*.

        Returns ``(kbar, weight, sub_edges)`` of leading shape ``(nbins, nsub)``: the
        mode-weighted mean magnitude of each sub-shell, its share of the bin's modes, and its
        edges. Equal width rather than equal population, so that every sub-shell of every leg has
        the same :math:`\Delta k` and one mode counter serves them all; a sub-shell holding no
        modes simply gets weight zero, which is the right answer and not a special case.
        """
        edges = np.asarray(edges)
        lo = edges[:, 0][:, None] + np.arange(nsub)[None, :] * (
            (edges[:, 1] - edges[:, 0]) / nsub)[:, None]
        hi = lo + ((edges[:, 1] - edges[:, 0]) / nsub)[:, None]
        n, kbar = self.exact(lo, hi)
        tot = n.sum(axis=-1, keepdims=True)
        w = np.where(tot > 0, n / np.where(tot > 0, tot, 1.), 1. / nsub)
        kbar = np.where(n > 0, kbar, 0.5 * (lo + hi))
        return kbar, w, np.stack([lo, hi], axis=-1)


# ======================================================================================
# quadrature
# ======================================================================================

def quad_nodes(kinds, order, seed=0, size=None):
    """Nodes and weights on the angular variables named by ``kinds``.

    ``kinds`` is a string, one character per axis: ``'m'`` for a cosine axis (Gauss-Legendre on
    ``[-1, 1]``, measure ``dx / 2``), ``'p'`` for an azimuth (uniform midpoint on ``[0, 2 pi)``,
    measure ``dphi / 2 pi``), ``'M'`` for a cosine axis whose range is set per bin pair by a
    window -- returned on ``[0, 1]`` for the caller to map, with the ``dx/2`` Jacobian left out.

    ``size`` switches to scrambled Sobol sampling with that many points, which is what the
    7-dimensional block needs; the measure is then uniform and the weights are ``1 / size``.
    """
    if size is not None:
        from scipy.stats import qmc
        u = qmc.Sobol(d=len(kinds), scramble=True, seed=seed).random(size)
        cols, w = [], np.full(size, 1. / size)
        for i, kind in enumerate(kinds):
            if kind == 'p':
                cols.append(2. * np.pi * u[:, i])
            elif kind == 'M':
                cols.append(u[:, i])
            else:
                cols.append(2. * u[:, i] - 1.)
        return np.stack(cols, axis=-1), w
    axes = []
    for kind in kinds:
        if kind == 'p':
            h = 2. * np.pi / order
            axes.append(((np.arange(order) + 0.5) * h, np.full(order, 1. / order)))
        elif kind == 'M':
            x, w = np.polynomial.legendre.leggauss(order)
            axes.append((0.5 * (x + 1.), 0.5 * w))       # on [0, 1], sum of w = 1
        else:
            x, w = np.polynomial.legendre.leggauss(order)
            axes.append((x, 0.5 * w))                    # dx / 2, sum of w = 1
    grids = np.meshgrid(*[a[0] for a in axes], indexing='ij')
    wgrids = np.meshgrid(*[a[1] for a in axes], indexing='ij')
    w = np.ones(grids[0].size)
    for g in wgrids:
        w = w * g.ravel()
    return np.stack([g.ravel() for g in grids], axis=-1), w


def _batched(fun, nodes, weights, batch_size, shape):
    """``sum_q w_q fun(nodes[q])`` with the node axis chopped into batches."""
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    n = len(weights)
    batch_size = n if batch_size is None else min(batch_size, n)
    out = jnp.zeros(shape)
    vfun = jax.jit(jax.vmap(fun))
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        val = vfun(jnp.asarray(nodes[sl]))
        out = out + jnp.tensordot(jnp.asarray(weights[sl]), val, axes=(0, 0))
    return out


# ======================================================================================
# the spectra, dressed with discreteness
# ======================================================================================

def sum_over_terms(spectrum, terms):
    r"""``sum_t amplitude_t * spectrum(fields_t, *momenta_t)``, traced once per field tuple.

    ``terms`` is a list of ``(amplitude, fields, momenta)``. The permutation sums of Eqs. (14),
    (27) and (B4)-(B6) are the same spectrum evaluated on 4, 6, 9 or 18 relabelled
    configurations; written as a Python loop they trace one copy of the kernel per term, and for
    the 1296-tree :math:`P_6` line that is a graph XLA spends longer compiling than running.
    Stacking the configurations on a new leading axis traces one copy, and the spectra are
    elementwise in their leading axes, so this is exact.

    Field tuples are Python objects and cannot ride that stacking axis, so terms are grouped by
    field tuple and each group is stacked on its own. With one tracer there is a single group and
    this is the original single-trace behaviour; with several, the cost grows with the number of
    distinct cross-spectra rather than with the number of terms. Terms of zero amplitude are
    dropped, so a cross-spectrum that cannot contribute is never traced.
    """
    grouped = {}
    for amplitude, fields, momenta in terms:
        if amplitude == 0.:
            continue
        configurations, amplitudes = grouped.setdefault(tuple(fields), ([], []))
        configurations.append(tuple(momenta))
        amplitudes.append(amplitude)
    total = 0.
    for fields, (configurations, amplitudes) in grouped.items():
        stacked = [jnp.stack(jnp.broadcast_arrays(*leg)) for leg in zip(*configurations)]
        values = spectrum(fields, *stacked)
        amplitudes = jnp.asarray(amplitudes).reshape((-1,) + (1,) * (jnp.ndim(values) - 1))
        total = total + (amplitudes * values).sum(axis=0)
    return total


class Spectra(object):
    r"""``P^{(N)}``, ``B^{(N)}``, ``T^{(N)}``, ``P_5^{(N)}`` and the ``P_6`` shot lines.

    ``theory(fields)`` returns the connected :math:`n`-point for ``n = len(fields)``, as a
    callable of all :math:`n` wavevectors -- one for :math:`P`, whose second leg is
    :math:`-k` -- or ``None`` if that order is not modelled. ``shotnoise`` is the coincidence
    amplitude :math:`1/\bar n`, and is both what is added to :math:`P` to make :math:`P^{(N)}`
    and what every coincidence structure above is built from. A tracer whose stochasticity is not
    Poisson cannot be described by one amplitude at all, and the excess belongs in ``theory``.

    **Only the pair amplitude enters.** ``shotnoise`` may be given as ``{2: sn2}``, but not with
    the higher weight moments :math:`sn_3 = V^2 \sum w^3 / (\sum w)^3` or
    :math:`sn_4 = V^3 \sum w^4 / (\sum w)^4`, because no term here is a triple or quadruple
    coincidence. The powers of ``sn`` below look like moments and are not: they are products of
    *independent, disjoint pair* coincidences, so they stay :math:`sn_2^2` and :math:`sn_2^3`
    whatever the weight distribution. Eq. (14)'s :math:`sn^2` is two disjoint cross pairs, Eq.
    (B6)'s :math:`sn^3` is three of them. Any genuine triple would put two of its points in the
    same estimator, and Sugiyama's :math:`i \neq j` estimators exclude that -- which is why Eqs.
    (14) and (27) carry no :math:`sn_3` at all.

    The higher moments do enter for an FFT estimator whose bispectrum has never had its contact
    terms subtracted; that convention is not implemented here, so supplying them raises rather
    than being silently ignored.
    """

    def __init__(self, theory=None, shotnoise=0.):
        self._theory = theory if theory is not None else (lambda fields: None)
        self._cache = {}
        # A `{order: moment}` dict is the one spelling that is refused rather than used: no term
        # here is a triple or quadruple coincidence, so `{3: ..., 4: ...}` would be silently
        # dropped. A `{(a, b): value}` cross-field dict and a callable both pass through.
        if isinstance(shotnoise, dict) and shotnoise and all(
                isinstance(order, (int, np.integer)) for order in shotnoise):
            above_pairs = sorted(order for order in shotnoise if int(order) != 2)
            if above_pairs:
                raise ValueError(
                    f'shotnoise={shotnoise}: the weight moments {above_pairs} do not enter this '
                    'covariance. Every power of the shot noise here is a product of disjoint '
                    'pair coincidences, not a higher coincidence, because Sugiyama\'s i != j '
                    'estimators exclude any coincidence within an estimator. Pass the pair '
                    'amplitude alone.')
            shotnoise = float(shotnoise[2]) if 2 in shotnoise else 0.
        self._shotnoise = shotnoise

    # ---- theory, keyed by field tuple -------------------------------------------------
    def get(self, fields):
        """The connected ``len(fields)``-point for this field tuple, or ``None``.

        Cached, because the assembly asks for the same cross-spectrum once per term and a theory
        built on ``pt.py`` does real work -- building tables -- when it is constructed.
        """
        fields = tuple(fields)
        if fields not in self._cache:
            self._cache[fields] = self._theory(fields)
        return self._cache[fields]

    def has(self, fields):
        return self.get(fields) is not None

    # ---- shot noise -------------------------------------------------------------------
    def coincidence(self, first, second):
        """The amplitude of a pair of points landing on top of one another.

        This is both the constant added to :math:`P_{ab}` to make :math:`P^{(N)}_{ab}` and what
        every coincidence structure in :math:`B^{(N)}`, :math:`T^{(N)}` and the :math:`P_6` shot
        lines is built from. It is zero across distinct tracers: two points of different types
        are never the same point.
        """
        shotnoise = self._shotnoise
        if callable(shotnoise):
            return shotnoise(first, second)
        if isinstance(shotnoise, dict):
            return shotnoise.get((first, second), shotnoise.get((second, first), 0.))
        return shotnoise if first == second else 0.

    # ---- the connected spectra -------------------------------------------------------
    # `fields` names one tracer per leg and is as long as the configuration: two for P, three for
    # B, and so on. The momenta are the `n - 1` *independent* ones, which is the convention
    # `pt.py` uses throughout -- `spectrum3_redshift_tracer` takes two wavevectors and
    # `spectrum4_redshift_tracer` three. A connected n-point lives on `sum k_i = 0`, so the
    # closing leg is not an independent argument and passing it invites the caller's idea of it
    # to disagree with the callee's.
    #
    # An order the theory does not model gives a zero of the configuration's shape rather than a
    # bare `0.`, which would propagate as a rank-0 array and fail to broadcast against the node
    # and bin axes in `_run`.
    def power(self, fields, k):
        spectrum = self.get(fields)
        return spectrum(k) if spectrum is not None else jnp.zeros_like(jnp.asarray(k)[..., 0])

    def connected(self, fields, *momenta):
        """The connected ``len(fields)``-point on ``len(fields) - 1`` independent momenta."""
        spectrum = self.get(fields)
        if spectrum is None:
            return jnp.zeros_like(jnp.asarray(momenta[0])[..., 0])
        return spectrum(*momenta)

    # ---- Eq. (11) --------------------------------------------------------------------
    def power_shot(self, fields, k):
        r""":math:`P^{(N)}_{ab}(k) = P_{ab}(k) + \delta_{ab} / \bar n`."""
        return self.power(fields, k) + self.coincidence(*fields)

    # ---- Eq. (24) --------------------------------------------------------------------
    def bispectrum_shot(self, fields, k1, k2, k3):
        r""":math:`B^{(N)}_{abc}(k_1, k_2, k_3)`.

        Leg 1 is the one tied to whatever the bispectrum is paired with, and the two surviving
        coincidences are the *cross* pairs (1, 2) and (1, 3): Sugiyama's :math:`i \neq j`
        estimators exclude the within-estimator pair (2, 3), which is why the familiar bispectrum
        shot noise -- all three pairs plus a constant -- never appears here. When points 1 and 2
        coincide the surviving two-point runs between the merged point and leg 3, so it carries
        fields ``(a, c)`` and momentum ``k3``.
        """
        a, b, c = fields
        return (self.connected(fields, k1, k2)
                + self.coincidence(a, b) * self.power((a, c), k3)
                + self.coincidence(a, c) * self.power((a, b), k2))

    # ---- Eq. (14) --------------------------------------------------------------------
    def trispectrum_shot(self, fields, k1, k2, k1p, k2p):
        r""":math:`T^{(N)}` with ``(k1, k2)`` one estimator's pair and ``(k1p, k2p)`` the other's.

        Only cross coincidences survive, giving the four single-cross-pair :math:`B` terms and
        the two double-cross-pair :math:`P` terms. The latter carry :math:`sn^2` because they are
        *two disjoint pairs*, not a triple coincidence -- any triple would contain a
        within-estimator pair. The other two pair sums are minus the ones written, and ``P``,
        ``B`` are even, so the paper's "2 P" is the whole of that line.
        """
        a, b, c, d = fields
        total = self.connected(fields, k1, k2, k1p)
        # One cross pair coincides; the merged point plus the two free legs make a bispectrum.
        total = total + sum_over_terms(self.connected, [
            (self.coincidence(a, c), (a, b, d), (-k1 - k1p, k1)),
            (self.coincidence(a, d), (a, b, c), (-k1 - k2p, k1)),
            (self.coincidence(b, c), (b, a, d), (-k2 - k1p, k2)),
            (self.coincidence(b, d), (b, a, c), (-k2 - k2p, k2))])
        # Two disjoint cross pairs coincide; one power spectrum is left.
        total = total + sum_over_terms(self.power, [
            (self.coincidence(a, c) * self.coincidence(b, d), (a, b), (k1 + k1p,)),
            (self.coincidence(a, d) * self.coincidence(b, c), (a, b), (k1 + k2p,))])
        return total

    # ---- Eq. (27) --------------------------------------------------------------------
    def five_point_shot(self, fields, k, k1, k2, k3):
        r""":math:`P_5^{(N)}(k, -k, k_1, k_2, k_3)` with :math:`k_1 + k_2 + k_3 = 0`.

        ``fields`` names the power spectrum's pair first, then the bispectrum's triangle. The
        power spectrum's two legs sit at :math:`\pm k`; tying one of them to a bispectrum leg
        merges the two momenta and leaves a trispectrum, and tying both to different bispectrum
        legs leaves a bispectrum.
        """
        pair, triangle = fields[:2], fields[2:]
        legs = (k1, k2, k3)
        total = self.connected(fields, k, -k, k1, k2)

        # One cross pair. `tied` indexes the power spectrum leg, `other` the bispectrum leg; the
        # trispectrum runs over the merged point, the free power spectrum leg and the two
        # untied bispectrum legs.
        terms = []
        for tied, sign in enumerate((1., -1.)):
            for other in range(3):
                free = [index for index in range(3) if index != other]
                terms.append((self.coincidence(pair[tied], triangle[other]),
                              (pair[tied], pair[1 - tied], triangle[free[0]], triangle[free[1]]),
                              (sign * k + legs[other], -sign * k, legs[free[0]])))
        total = total + sum_over_terms(self.connected, terms)

        # Two disjoint cross pairs, one per power spectrum leg: a bispectrum is left, on the two
        # merged points and the untied bispectrum leg.
        terms = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            for tied, sign in enumerate((1., -1.)):
                amplitude = (self.coincidence(pair[tied], triangle[first])
                             * self.coincidence(pair[1 - tied], triangle[second]))
                terms.append((amplitude,
                              (triangle[first], triangle[second], triangle[3 - first - second]),
                              (sign * k + legs[first], legs[second] - sign * k)))
        return total + sum_over_terms(self.connected, terms)

    # ---- Eq. (33) with Eqs. (B4)-(B6) ------------------------------------------------
    def six_point_shot(self, fields, unprimed, primed):
        r""":math:`P_6^{(N)}` on the two triangles ``unprimed`` and ``primed``.

        ``fields`` names the unprimed triangle first, then the primed one. Only cross ties between
        the two triangles survive, one, two or three at a time, leaving a five-point, a
        trispectrum and a bispectrum respectively.
        """
        left, right = fields[:3], fields[3:]
        total = self.connected(fields, *unprimed, primed[0], primed[1])

        # Eq. (B4), 9 terms: one cross tie.
        terms = []
        for i in range(3):
            for j in range(3):
                free_left = [index for index in range(3) if index != i]
                free_right = [index for index in range(3) if index != j]
                terms.append((self.coincidence(left[i], right[j]),
                              (left[i], left[free_left[0]], left[free_left[1]],
                               right[free_right[0]], right[free_right[1]]),
                              (unprimed[i] + primed[j], unprimed[free_left[0]],
                               unprimed[free_left[1]], primed[free_right[0]])))
        total = total + sum_over_terms(self.connected, terms)

        # Eq. (B5), 18 terms: two cross ties, leaving one free leg on each side.
        terms = []
        for i, i2 in ((0, 1), (0, 2), (1, 2)):
            for j in range(3):
                for j2 in range(3):
                    if j2 == j:
                        continue
                    amplitude = (self.coincidence(left[i], right[j])
                                 * self.coincidence(left[i2], right[j2]))
                    terms.append((amplitude,
                                  (left[i], left[i2], left[3 - i - i2], right[3 - j - j2]),
                                  (unprimed[i] + primed[j], unprimed[i2] + primed[j2],
                                   unprimed[3 - i - i2])))
        total = total + sum_over_terms(self.connected, terms)

        # Eq. (B6), 6 terms: all three legs tied, leaving a bispectrum on the merged points.
        terms = []
        for j in range(3):
            for j2 in range(3):
                if j2 == j:
                    continue
                amplitude = (self.coincidence(left[0], right[j])
                             * self.coincidence(left[1], right[j2])
                             * self.coincidence(left[2], right[3 - j - j2]))
                terms.append((amplitude, left,
                              (unprimed[0] + primed[j], unprimed[1] + primed[j2])))
        return total + sum_over_terms(self.connected, terms)


# ======================================================================================
# multipole weights
# ======================================================================================

def _NH2(ells):
    l1, l2, L = ells
    return (2 * l1 + 1) * (2 * l2 + 1) * (2 * L + 1) * wigner_3j(l1, l2, L, 0, 0, 0)**2


def _Sfun(ells):
    return get_S(tuple(ells), z3=True)


def _dirhat(mu, phi):
    """Unit vector from its line-of-sight cosine and azimuth."""
    s = jnp.sqrt(jnp.clip(1. - mu**2, 0., None))
    return jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), mu], axis=-1)


def _bc(u, p):
    """Broadcast per-bin arrays of the two observables to a common ``(nu, np)`` shape."""
    return np.asarray(u)[:, None], np.asarray(p)[None, :]


# ======================================================================================
# drivers
# ======================================================================================

def _align(a, ndim):
    """Insert singleton axes just after the node axis so ``a`` broadcasts against the value.

    A projection weight depends on the node and, when the geometry is bin-dependent (a window
    remaps an angle per bin pair), on some of the bin axes; the value always carries all of them.
    Right-aligning the bin axes is what makes both cases one code path.
    """
    extra = ndim - jnp.ndim(a)
    return jnp.reshape(a, jnp.shape(a)[:1] + (1,) * extra + jnp.shape(a)[1:]) if extra > 0 else a


def _run(term_fn, nodes, weights, batch_size, shape, ells_u, ells_p, kind_u, kind_p):
    r"""Integrate one term over the angular nodes, for every multipole pair at once.

    ``term_fn(node)`` returns ``(value, *hats)``: the physics, which does not depend on the
    multipoles, and the unit vectors the projection bases need -- one for a power spectrum
    (:math:`\hat k`), two for a bispectrum (:math:`\hat k_1, \hat k_2`). Evaluating the physics
    once and contracting it against every multipole pair is what makes the expensive blocks
    affordable: the 1296-tree :math:`P_6` is paid once for all ten pairs of ``(B000, B202,
    B110, B220)``.
    """
    nu_ell, np_ell = len(ells_u), len(ells_p)
    proj_u = [get_legendre(e) if kind_u == 2 else _Sfun(e) for e in ells_u]
    proj_p = [get_legendre(e) if kind_p == 2 else _Sfun(e) for e in ells_p]
    pref_u = [(2 * e + 1) if kind_u == 2 else _NH2(e) for e in ells_u]
    pref_p = [(2 * e + 1) if kind_p == 2 else _NH2(e) for e in ells_p]
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    n = len(weights)
    bs = n if batch_size is None else min(batch_size, n)
    out = [[jnp.zeros(shape) for _ in range(np_ell)] for _ in range(nu_ell)]
    nhat_u = 1 if kind_u == 2 else 2

    @jax.jit
    def _batch(nd):
        return jax.vmap(term_fn)(nd)

    for start in range(0, n, bs):
        sl = slice(start, min(start + bs, n))
        res = _batch(jnp.asarray(nodes[sl]))
        val, hats = res[0], res[1:]
        w = jnp.asarray(weights[sl])
        hu, hp = hats[:nhat_u], hats[nhat_u:]
        ndim = 1 + len(shape)
        for i in range(nu_ell):
            su = (proj_u[i](hu[0][..., 2]) if kind_u == 2 else proj_u[i](*hu)) * pref_u[i]
            su = _align(jnp.broadcast_to(su, jnp.shape(val)[:1] + jnp.shape(su)[1:]), ndim) \
                if jnp.ndim(su) else su
            for j in range(np_ell):
                sp = (proj_p[j](hp[0][..., 2]) if kind_p == 2 else proj_p[j](*hp)) * pref_p[j]
                sp = _align(sp, ndim) if jnp.ndim(sp) else sp
                out[i][j] = out[i][j] + jnp.tensordot(w, su * sp * val, axes=(0, 0))
    return out


# ======================================================================================
# Cov[P, P]
# ======================================================================================

class BoxGeometry(object):
    r"""The periodic box, as a covariance block sees it: one volume and one mode count.

    Every place geometry enters a block goes through this object, so that a survey window can be
    dropped in beside it rather than threaded through as a second code path. Two things are
    needed, and they are exactly the two the retired implementation factored out as
    ``inverse_V2``/``inverse_V3`` and ``W2``/``W3``:

    ``inverse_volume``
        the :math:`(2\pi)^3 \delta_D(0) = V` substitution every connected :math:`n`-point carries.
        For a box it is :math:`1/V` whatever the legs are; for a survey it is the window integral
        :math:`1/V^{(n)}`, which depends on which fields sit on which side -- and the pairing is
        the *cross* one, :math:`\int W_a W_b W_c W_d / (I_{ab} I_{cd})`, not the product of each
        spectrum's own normalisation.

    ``pair_weight``
        Eq. (47)'s :math:`1/\tilde N_{\rm mode}(k, k')`. For a box it is diagonal in the bins and
        depends only on the two magnitudes; for a survey it is a dense mixing matrix that also
        depends on the two directions, which is why the cutsky version will need the vectors and
        the bin edges rather than the magnitudes alone.
    """

    #: Whether :meth:`pair_weight` depends on the leg *directions*. A box's does not -- the delta
    #: ties the two directions together and one shared line-of-sight integral survives -- so the
    #: blocks hoist it out of the quadrature. A survey window's does, which is why the windowed
    #: blocks are separate integrals rather than the same one reweighted.
    directional = False

    def __init__(self, volume, count=None):
        self.volume = volume
        self.count = count

    def inverse_volume(self, rows, cols):
        """``1 / V``. The field groups are ignored here and are what a survey window keys on."""
        del rows, cols
        return 1. / self.volume

    def pair_weight(self, ka, kb, dk, count=None):
        return _inv_nmode(ka, kb, dk, self.volume, self.count if count is None else count)


class SurveyGeometry(object):
    r"""A survey window, in the same two methods :class:`BoxGeometry` provides.

    ``window2`` and ``window3`` are what :func:`compute_fkp2_covariance_window` and
    :func:`compute_fkp3_covariance_window` return: configuration-space covariance-window
    multipoles labelled by *groups* of fields, because the mixed terms need
    :math:`Q_W^{(ac)(bde)}` and the like rather than one field per anchor.

    The two replacements for the box's single volume and single mode count, following
    ``desi-cov3-notes/covariance.tex``:

    * :math:`1/V^{(4)}_{ab,cd} = \int W_a W_b W_c W_d / (I_{ab} I_{cd})` is the *monopole* of the
      two-anchor window, and :math:`1/V^{(6)}` the monopole of the three-anchor one. Note the
      cross pairing in the denominator -- not the product of each spectrum's own normalisation.
    * :math:`Q_W(k - k')` expanded in the two line-of-sight angles, which is
      :func:`compute_QW_AB`. It mixes bins and depends on both directions, so it cannot be
      hoisted out of the angular quadrature the way the box's mode count can. That expansion is
      the :math:`m = 0` truncation of the exact kernel, exact for ``PP`` and ``PB`` and not for
      ``BB``/``PT``; see approximation 1 in the module docstring.

    Both of these are built from one window field per tracer, :math:`W = \bar n w`, so the
    shot-noise parts of :math:`P^{(N)}`, :math:`B^{(N)}`, :math:`T^{(N)}` are given the clustering
    window's shape rather than their own -- approximation 2 in the module docstring.
    """

    directional = True

    def __init__(self, window2, window3=None, cache=None):
        self.window2 = window2
        self.window3 = window3
        self.cache = {} if cache is None else cache

    def inverse_volume(self, rows, cols):
        """The window monopole, which *is* :math:`1/V^{(4)}` for two anchors."""
        return self.window2.get(fields1=tuple(rows), fields2=tuple(cols), ells=0).value()[0]

    def inverse_volume3(self, first, second, third):
        """:math:`1/V^{(6)}`, the monopole of the three-anchor window."""
        return self.window3.get(fields1=tuple(first), fields2=tuple(second),
                                fields3=tuple(third), ells=(0, 0, 0)).value()[0]

    def pair_weight(self, kvec, kpvec, edges, edgesp, rows, cols,
                    rows_is_points=False, cols_is_points=False):
        """``Q_W(k - k')``, dense in the bins and dependent on both line-of-sight angles."""
        return compute_QW_AB(self.window2, edges, edgesp, _mu(kvec), _mu(kpvec),
                             fields1=tuple(rows), fields2=tuple(cols), cache=self.cache,
                             k1_is_points=rows_is_points, k2_is_points=cols_is_points).real


def _block_pp_gaussian(sp, u, p, geometry, order, count=None):
    r"""Eq. (43). Diagonal in the bins; a single line-of-sight integral survives.

    The two Wick routes are written out rather than folded into a leading factor 2. They tie
    :math:`\hat k' = +\hat k` and :math:`-\hat k`, pairing the unprimed fields :math:`(a, b)` with
    the primed :math:`(c, d)` as :math:`(ac)(bd)` and :math:`(ad)(bc)`; for one tracer the two are
    the same number and the paper's 2 is recovered, for cross-spectra they are not.
    """
    ku, kp = _bc(u['k'], p['k'])
    (a, b), (c, d) = u['fields'], p['fields']
    same = np.all(np.isclose(u['edges'][:, None, :], p['edges'][None, :, :]), axis=-1)
    pref = same * geometry.pair_weight(ku, np.where(same, kp, 1.), u['dk'][:, None], count)
    nodes, weights = quad_nodes('m', order)

    def term(nd):
        khat = _ahat(nd[0])
        kvec = ku[..., None] * khat
        routes = (sp.power_shot((a, c), kvec) * sp.power_shot((b, d), kvec)
                  + sp.power_shot((a, d), kvec) * sp.power_shot((b, c), kvec))
        # The delta ties khat' = +- khat, so both multipoles are projected on the same direction.
        return routes * pref, khat, khat

    return _run(term, nodes, weights, None, (len(u['k']), len(p['k'])),
                u['ells'], p['ells'], 2, 2)


def _block_pp_gaussian_window(sp, u, p, geometry, order, window_ells=(0, 2, 4)):
    r"""The Gaussian ``Cov[P, P]`` with a survey window (``desi-cov3-notes/covariance.tex``).

    .. math::
        {\rm Cov}^{PP}[P^{ab}_\ell(k), P^{cd}_{\ell'}(k')]
        = (2\ell + 1)(2\ell' + 1)
          \int \frac{d\mu}{2} \mathcal{L}_\ell(\mu) \int \frac{d\mu'}{2} \mathcal{L}_{\ell'}(\mu')
          \Big[ Q_{\mathcal W}^{(ac)(bd)}(k, k') P^{(N)}_{ac}(k) P^{(N)}_{bd}(-k')
              + Q_{\mathcal W}^{(ad)(bc)}(k, -k') P^{(N)}_{ad}(k) P^{(N)}_{bc}(k') \Big]

    Two things make this cheaper than it looks, and both come from the window already being
    expanded in the two line-of-sight angles,
    :math:`Q_{\mathcal W} = \sum_{\ell_1 \ell_2} Q_{\ell_1 \ell_2}(k, k')
    \mathcal{L}_{\ell_1}(\mu) \mathcal{L}_{\ell_2}(\mu')`:

    * there is no azimuth left to integrate, so this is two one-dimensional quadratures rather
      than the three-dimensional one the box's :math:`T` term needs;
    * the integrand is *separable* -- one factor depends on :math:`(k, \mu)` and the other on
      :math:`(k', \mu')` -- so the double integral factorises into a product of one-dimensional
      ones, contracted against :math:`Q_{\ell_1 \ell_2}`. Nothing is evaluated per node pair.

    The second route's window is :math:`Q_{\mathcal W}(k, -k')`, which is
    :math:`\mu' \to -\mu'`; the stored window multipoles are even, so it is numerically the first
    route's and the two differ only by which fields are grouped together. The sign is kept
    explicit below rather than cancelled, because it is a property of the window, not of this
    block.
    """
    (a, b), (c, d) = u['fields'], p['fields']
    ku, kp = np.asarray(u['k']), np.asarray(p['k'])
    nodes, weights = quad_nodes('m', order)
    mu = nodes[:, 0]
    hats = _ahat(jnp.asarray(mu))

    def angular(fields, k, ells, sign):
        """``int dmu/2 L_ell(mu) L_ellw(mu) P^(N)_fields(sign * k)``, keyed by ``(ell, ellw)``."""
        power = sp.power_shot(fields, sign * k[:, None, None] * hats[None, :, :])
        return {(ell, ellw): np.asarray(jnp.tensordot(
                    power, jnp.asarray(weights * get_legendre(ell)(sign * mu)
                                       * get_legendre(ellw)(sign * mu)), axes=(1, 0)))
                for ell in ells for ellw in window_ells}

    out = [[0. for _ in p['ells']] for _ in u['ells']]
    for rows, cols, sign in (((a, c), (b, d), -1.), ((a, d), (b, c), 1.)):
        left = angular(rows, ku, u['ells'], 1.)
        right = angular(cols, kp, p['ells'], sign)
        for ell1 in window_ells:
            for ell2 in window_ells:
                block = compute_spectrum2_covariance_window_block(
                    (geometry.window2, geometry.window2), u['edges'], p['edges'], ell1, ell2,
                    fields1=tuple(rows), fields2=tuple(cols), cache=geometry.cache)
                block = np.asarray(block) * ((2 * ell1 + 1) * (2 * ell2 + 1)
                                             * (-1)**(ell1 // 2) * (-1)**(ell2 // 2))
                for i, ell in enumerate(u['ells']):
                    for j, ellp in enumerate(p['ells']):
                        out[i][j] = out[i][j] + ((2 * ell + 1) * (2 * ellp + 1) * block
                                                 * left[ell, ell1][:, None]
                                                 * right[ellp, ell2][None, :])
    return out


def _block_pp_trispectrum(sp, u, p, geometry, order, batch_size=None):
    """Eqs. (13), (14) projected with Eq. (36): three angles, no delta left to resolve."""
    ku, kp = _bc(u['k'], p['k'])
    fields = tuple(u['fields']) + tuple(p['fields'])
    nodes, weights = quad_nodes('mmp', order)

    def term(nd):
        khat, kphat = _ahat(nd[0]), _dirhat(nd[1], nd[2])
        kv, kpv = ku[..., None] * khat, kp[..., None] * kphat
        return (sp.trispectrum_shot(fields, kv, -kv, kpv, -kpv)
                * geometry.inverse_volume(u['fields'], p['fields'])), khat, kphat

    return _run(term, nodes, weights, batch_size, (len(u['k']), len(p['k'])),
                u['ells'], p['ells'], 2, 2)


# ======================================================================================
# Cov[P, B]
# ======================================================================================

def _triangle(k1, k2, mu, x, phi):
    """The three legs of a bispectrum bin, leg 1 at line-of-sight cosine ``mu``, azimuth zero."""
    a = _ahat(mu)
    b = _bhat(jnp.broadcast_to(a, jnp.shape(x) + (3,)), x, phi)
    K1, K2 = k1[..., None] * a, k2[..., None] * b
    return K1, K2, -K1 - K2


def _block_pb_unconnected(sp, u, p, geometry, order, count=None, closure_min=None):
    r"""Eq. (48). Three terms; the closure-leg one has its top-hat solved for exactly.

    ``closure_min`` floors the bispectrum's closure leg in slots 0 and 1, which tie a binned leg
    and so leave it free while :math:`B^{(N)}`'s shot line evaluates a power spectrum on it. Slot
    2 ties the closure leg itself, into the power spectrum's bin, and is already bounded. See
    ``closure_min`` on :func:`_bb_tie_terms` for the branch point; here it costs convergence
    rather than existence -- one power of :math:`P(k_3)`, not two -- and the block drifts
    1.290e15 to 1.442e15 over ``order3`` 8 to 48 instead of settling.

    The five points split into a power spectrum that straddles the two estimators and a
    bispectrum on what is left. One power spectrum leg pairs with one bispectrum leg -- which is
    what the radial delta enforces -- and the *other* power spectrum leg joins the two untied
    bispectrum legs, at the same momentum as the tied one and so in the same slot 1. Which of the
    two power spectrum legs straddles is a free choice, and the paper's leading 2 is those two
    terms coinciding for one tracer; for cross-spectra they carry different fields.
    """
    k, k1, k2 = u['k'][:, None], p['k'][0][None, :], p['k'][1][None, :]
    dk = u['dk'][:, None]
    pair, triangle = u['fields'], p['fields']
    if closure_min is None:
        closure_min = _closure_floor(1. / geometry.inverse_volume(u['fields'], p['fields']))
    x_floor = _closure_x_floor(k1, k2, closure_min)

    def straddle(slot, tie, others):
        """The two ways to choose the straddling power spectrum leg, summed."""
        untied = tuple(triangle[index] for index in range(3) if index != slot)
        return sum(sp.power_shot((pair[which], triangle[slot]), tie)
                   * sp.bispectrum_shot((pair[1 - which],) + untied, tie, *others)
                   for which in (0, 1))

    out = None
    for slot in (0, 1, 2):
        if slot < 2:
            kt = (k1, k2)[slot]
            sel = np.abs(k - kt) < dk / 2.
            if not sel.any():
                continue
            wn = sel * geometry.pair_weight(k, kt, dk, count)
            nodes, weights = quad_nodes('m{}p'.format('m' if x_floor is None else 'M'), order)

            def term(nd, _s=slot, _wn=wn, _lo=x_floor):
                mu = jnp.broadcast_to(nd[0], k1.shape)
                x, xjac = _map_axis(nd[1], _lo, k1.shape)
                legs = _triangle(k1, k2, mu, x, nd[2])
                tie = legs[_s]
                oth = [legs[i] for i in range(3) if i != _s]
                val = _wn * xjac * straddle(_s, tie, oth)
                a1, a2 = legs[0] / k1[..., None], legs[1] / k2[..., None]
                return val, tie / _norm(tie)[..., None], a1, a2
        else:
            lo, hi = _window_interval(k1, k2, k, dk)
            if not (hi > lo).any():
                continue
            nodes, weights = quad_nodes('mMp', order)

            def term(nd, _lo=lo, _hi=hi):
                mu = jnp.broadcast_to(nd[0], _lo.shape)
                x = _lo + (_hi - _lo) * nd[1]
                jac = jnp.where(_hi > _lo, (_hi - _lo) / 2., 0.)
                legs = _triangle(k1, k2, mu, x, nd[2])
                k3 = _norm(legs[2])
                wn = geometry.pair_weight(k, k3, dk, count)
                val = jac * wn * straddle(2, legs[2], [legs[0], legs[1]])
                a1, a2 = legs[0] / k1[..., None], legs[1] / k2[..., None]
                return val, legs[2] / jnp.clip(k3, 1e-30, None)[..., None], a1, a2

        blk = _run(term, nodes, weights, None, (len(u['k']), p['k'].shape[1]),
                   u['ells'], p['ells'], 2, 3)
        out = blk if out is None else [[a + b for a, b in zip(ra, rb)]
                                       for ra, rb in zip(out, blk)]
    return out


def _block_pb_unconnected_window(sp, u, p, geometry, order, window_ells=(0, 2, 4),
                                 closure_min=None):
    r"""The disconnected ``Cov[P, B]`` with a survey window (``desi-cov3-notes/covariance.tex``).

    The box has a radial delta tying the power spectrum's :math:`k` to one bispectrum leg, which
    removes an integral and confines the block to the diagonal. A window replaces the delta by
    :math:`Q_{\mathcal W}(\pm k, k_i)` and the :math:`\hat k` integral returns -- but it is free:
    the only :math:`\hat k` dependence left is the window's own
    :math:`\mathcal{L}_{L_1}(\hat k \cdot \hat n)`, so

    .. math::
        \int \frac{d\mu}{2} \mathcal{L}_\ell(\mu) \mathcal{L}_{L_1}(\mu)
        = \frac{\delta_{\ell L_1}}{2\ell + 1},

    the estimator's leading :math:`(2\ell + 1)` cancels it, and what is left is the triangle's
    own three-dimensional integral with the window channel :math:`L_1` pinned to the observable
    multipole :math:`\ell`. The two straddle choices differ only by field grouping: their windows
    are :math:`Q_{\mathcal W}(+k, k_i)` and :math:`Q_{\mathcal W}(-k, k_i)`, equal because the
    stored window multipoles are even.

    Since the surviving channel depends on :math:`\ell`, the physics cannot be evaluated once and
    contracted against every multipole pair as :func:`_run` does it, so the loop is explicit.

    The tied leg's window table is node-independent when that leg carries its own bin edges, and
    is then simply multiplied on. When the tied leg is the triangle's *closure* leg its magnitude
    :math:`k_3 = |k_1 + k_2|` varies over the integral and the table has to be contracted inside
    it -- but :math:`k_3` depends only on :math:`x = \hat k_1 \cdot \hat k_2`, so it takes only
    ``order`` distinct values per bin pair rather than one per node, and the nodes sharing an
    :math:`x` are summed before the contraction.
    """
    k = np.asarray(u['k'])
    k1, k2 = np.asarray(p['k'][0]), np.asarray(p['k'][1])
    pair, triangle = u['fields'], p['fields']
    edges = np.asarray(p['edges'])
    leg_edges = [edges[:, 0, :], edges[:, 1, :]]
    # Same free closure leg as the box block's slots 0 and 1, and the same floor -- see
    # `closure_min` on `_block_pb_unconnected`. Slot 2's leg is not tied to a bin here either,
    # the window having replaced that delta, so it is floored along with the rest.
    if closure_min is None:
        closure_min = _closure_floor(1. / geometry.inverse_volume(u['fields'], p['fields']))
    x_floor = _closure_x_floor(jnp.asarray(k1), jnp.asarray(k2), closure_min)
    nodes, weights = quad_nodes('m{}p'.format('m' if x_floor is None else 'M'), order)
    nbin = len(k1)

    x, x_jacobian = _map_axis(jnp.asarray(nodes[:, 1])[:, None],
                              None if x_floor is None else x_floor[None, :],
                              (len(weights), nbin))
    legs = _triangle(jnp.asarray(k1)[None, :], jnp.asarray(k2)[None, :],
                     jnp.asarray(nodes[:, 0])[:, None], x,
                     jnp.asarray(nodes[:, 2])[:, None])
    hats = [leg / jnp.clip(_norm(leg), 1e-30, None)[..., None] for leg in legs]
    # x is the second axis of the 'mmp' grid, which `quad_nodes` builds with indexing='ij'.
    x_nodes = np.polynomial.legendre.leggauss(order)[0]
    if x_floor is not None:
        # The mapped axis: the same nodes, placed in each bin's own admitted range.
        x_nodes = (np.asarray(x_floor)[None, :]
                   + (1. - np.asarray(x_floor))[None, :] * ((x_nodes[:, None] + 1.) / 2.))
    else:
        x_nodes = x_nodes[:, None]
    x_index = (np.arange(len(weights)) // order) % order
    closure = np.sqrt(k1**2 + k2**2 + 2. * k1 * k2 * x_nodes)

    projections = [(_Sfun(ell)(hats[0], hats[1]) * _NH2(ell)) for ell in p['ells']]
    out = [[0. for _ in p['ells']] for _ in u['ells']]

    for slot in range(3):
        untied = tuple(triangle[index] for index in range(3) if index != slot)
        others = [legs[index] for index in range(3) if index != slot]
        tie_cosine = hats[slot][..., 2]
        for which in (0, 1):
            rows = (pair[which], triangle[slot])
            cols = (pair[1 - which],) + untied
            physics = (sp.power_shot(rows, legs[slot])
                       * sp.bispectrum_shot(cols, legs[slot], *others))
            for iell, ell in enumerate(u['ells']):
                if ell not in window_ells:
                    # Orthogonality leaves nothing: the window carries no such channel.
                    continue
                for ellw in window_ells:
                    coefficient = ((2 * ell + 1) * (2 * ellw + 1)
                                   * (-1)**(ell // 2) * (-1)**(ellw // 2))
                    weighted = (jnp.asarray(weights)[:, None] * jnp.asarray(x_jacobian)
                                * get_legendre(ellw)(tie_cosine) * physics)
                    table = compute_spectrum2_covariance_window_block(
                        geometry.window2, u['edges'],
                        leg_edges[slot] if slot < 2 else closure.ravel(), ell, ellw,
                        fields1=rows, fields2=cols, cache=geometry.cache,
                        k2_is_points=slot == 2)
                    table = np.asarray(table)
                    for jell, projection in enumerate(projections):
                        integrand = np.asarray(projection * weighted)
                        if slot < 2:
                            block = table * integrand.sum(axis=0)[None, :]
                        else:
                            grouped = np.zeros((order, nbin))
                            np.add.at(grouped, x_index, integrand)
                            block = np.einsum('kxb,xb->kb',
                                              table.reshape(len(k), order, nbin), grouped)
                        out[iell][jell] = out[iell][jell] + coefficient * block
    return out


def _block_pb_p5(sp, u, p, geometry, order, batch_size=None, size=None, seed=0,
                 closure_min=None):
    """Eqs. (26), (27) projected with Eq. (36): five angles, no window.

    ``closure_min`` floors the bispectrum's closure leg ``K3``, which is derived, unconstrained,
    and carries a power spectrum through :math:`P_5^{(N)}`'s shot lines -- see ``closure_min`` on
    :func:`_bb_tie_terms`. Unfloored the block does not settle: 2.974e24 / 3.091e24 / 3.163e24 /
    3.249e24 at ``order5`` 8 / 12 / 16 / 24, still rising 2.7% per step, against a constant
    theory that gives exactly :math:`1/V` at every order.
    """
    k, k1, k2 = u['k'][:, None], p['k'][0][None, :], p['k'][1][None, :]
    fields = tuple(u['fields']) + tuple(p['fields'])
    if closure_min is None:
        closure_min = _closure_floor(1. / geometry.inverse_volume(u['fields'], p['fields']))
    x_floor = _closure_x_floor(k1, k2, closure_min)
    kinds = 'm{}pmp'.format('m' if x_floor is None else 'M')
    nodes, weights = quad_nodes(kinds, order, size=size, seed=seed)

    def term(nd, _lo=x_floor):
        x, xjac = _map_axis(nd[1], _lo, jnp.shape(k1 * k2))
        a = jnp.broadcast_to(_ahat(nd[0]), jnp.shape(x) + (3,))
        b = _bhat(a, x, nd[2])
        khat = _dirhat(nd[3], nd[4])
        K1, K2 = k1[..., None] * a, k2[..., None] * b
        K3 = -K1 - K2
        kv = k[..., None] * khat
        return (sp.five_point_shot(fields, kv, K1, K2, K3) * xjac
                * geometry.inverse_volume(u['fields'], p['fields'])), khat, a, b

    return _run(term, nodes, weights, batch_size, (len(u['k']), p['k'].shape[1]),
                u['ells'], p['ells'], 2, 3)


# ======================================================================================
# Cov[B, B]
# ======================================================================================

def _block_bb_ppp(sp, u, p, geometry, order, count=None, closure_min=None):
    """Eq. (B7): six ways to pair the two triangles' legs, two radial deltas each.

    ``closure_min`` floors the unprimed closure leg in the two pairings that leave it
    unconstrained -- the ones where ``2`` is neither tied leg. All six evaluate a power spectrum
    on it, but the other four tie it into a primed bin, which already bounds it below. Left to run
    to zero it makes the integrand go as :math:`k_3^{-0.5}`: finite, but a square-root branch
    point, and the block then converges as a slow power of ``order`` rather than settling -- 5.79e20
    to 6.45e20 between ``order3`` 8 and 48, still moving 1.7% per step, where the same block with a
    *constant* theory is exact at order 8. ``None`` takes :func:`_closure_floor` of the volume the
    block is normalised in, which is the fundamental below which the estimator has no modes.
    """
    ku1, ku2 = u['k'][0][:, None], u['k'][1][:, None]
    kp1, kp2 = p['k'][0][None, :], p['k'][1][None, :]
    dkp = p['dk'][None, :]
    unprimed, primed = u['fields'], p['fields']
    shape = (u['k'].shape[1], p['k'].shape[1])
    if closure_min is None:
        closure_min = _closure_floor(1. / geometry.inverse_volume(unprimed, primed))
    out = None
    for a, b in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)):
        free = 2 in (a, b)
        if free:
            # The closure leg carries one of the two windows; the other is a bin selector.
            slot = 0 if a == 2 else 1
            kt = (kp1, kp2)[slot]
            lo, hi = _window_interval(ku1, ku2, kt, dkp)
            other_u, other_p = (b if a == 2 else a), (1 - slot)
            sel = np.abs((ku1, ku2)[other_u] - (kp1, kp2)[other_p]) < dkp / 2.
            if not ((hi > lo) & sel).any():
                continue
            nodes, weights = quad_nodes('mMp', order)
        else:
            sel = ((np.abs((ku1, ku2)[a] - kp1) < dkp / 2.)
                   & (np.abs((ku1, ku2)[b] - kp2) < dkp / 2.))
            if not sel.any():
                continue
            # Nothing ties the closure leg here, and a power spectrum sits on it: place the
            # quadrature above `closure_min` rather than let it run to k_3 = 0. Solved, not
            # masked -- the branch point is exactly where the weight is.
            lo = jnp.clip((closure_min**2 - ku1**2 - ku2**2) / (2. * ku1 * ku2), -1., 1.)
            hi = None
            nodes, weights = quad_nodes('mMp', order)

        def term(nd, _a=a, _b=b, _lo=lo, _hi=hi, _sel=sel, _free=free):
            mu = jnp.broadcast_to(nd[0], shape)
            if _free:
                x = _lo + (_hi - _lo) * nd[1]
                jac = jnp.where(_hi > _lo, (_hi - _lo) / 2., 0.)
            else:
                x = jnp.broadcast_to(_lo + (1. - _lo) * nd[1], shape)
                jac = (1. - _lo) / 2.
            legs = _triangle(ku1, ku2, mu, x, nd[2])
            kk = [_norm(leg) for leg in legs]
            wn = (jac * _sel * geometry.pair_weight(kk[_a], kp1, dkp, count)
                  * geometry.pair_weight(kk[_b], kp2, dkp, count)
                  / geometry.inverse_volume(unprimed, primed))
            # Unprimed leg `_a` is tied to primed leg 0, `_b` to primed leg 1, and whichever is
            # left to primed leg 2; each tie is one cross power spectrum at the unprimed momentum.
            rest = 3 - _a - _b
            val = wn * (sp.power_shot((unprimed[_a], primed[0]), legs[_a])
                        * sp.power_shot((unprimed[_b], primed[1]), legs[_b])
                        * sp.power_shot((unprimed[rest], primed[2]), legs[rest]))
            hats = [leg / jnp.clip(k, 1e-30, None)[..., None] for leg, k in zip(legs, kk)]
            return val, hats[0], hats[1], hats[_a], hats[_b]

        blk = _run(term, nodes, weights, None, shape, u['ells'], p['ells'], 3, 3)
        out = blk if out is None else [[c + d for c, d in zip(ra, rb)]
                                       for ra, rb in zip(out, blk)]
    return out


def _bb_tie_terms(sp, u, p, geometry, order, sign, kind, batch_size=None, count=None,
                  pairs=None, tie_volume=None, closure_min=None, qmin=None, qwidth=0.):
    r"""The nine leg pairings of Eqs. (B8) and (B9), which differ only in ``sign`` and ``value``.

    ``sign = +1`` is ``BB``, whose delta is :math:`\delta_D(k_i - k'_j)`; ``sign = -1`` is ``PT``,
    with :math:`\delta_D(k_i + k'_j)`. Three geometries cover the nine:

    * ``i`` any, ``j in (1, 2)`` -- the unprimed triangle is built in the standard way and the
      primed one hangs off the tied leg. Six terms.
    * ``i in (1, 2)``, ``j = 3`` -- the roles swap, the *primed* triangle is the standard one and
      its closure leg is what the unprimed triangle hangs off. Two terms.
    * ``i = j = 3`` -- both closure legs are tied, so the primed triangle's first leg is forced to
      the paper's :math:`k_\alpha` (or :math:`k_\beta`) and its magnitude carries the window.
      One term.

    Parameters
    ----------
    tie_volume : callable, default=None
        ``tie_volume(first_fields, second_fields) -> V``, the volume Eq. (47)'s mode count is
        taken in. ``None`` is the periodic box, where it is ``geometry``'s own. A survey passes
        the *integrated tie strength* :math:`1 / Q_{\mathcal W}(s \to 0)` of the two field
        groups instead, which is what turns this routine -- unchanged otherwise -- into the
        exact vector-tie treatment of a windowed block: the tie is consumed as a vector, and the
        window enters only through how much volume it leaves the tie. What that neglects is the
        finite width of the window's ridge, which the box makes a delta and no box-limit test can
        therefore see.
    closure_min : float, default=None
        Smallest magnitude an *unconstrained* closure leg may take. Every pairing evaluates a
        power spectrum on at least one closure leg -- through :math:`B^{(N)}`'s shot line for
        ``BB``, through the trispectrum's untied legs for ``PT`` -- and where nothing bounds that
        leg it runs to zero. There :math:`P(k_3) \sim k_3^{-1.5}` meets the ``x``-measure's
        :math:`k_3 dk_3 / (k_1 k_2)`, leaving :math:`k_3^{-0.5}`: integrable once, **divergent
        twice**, and ``(2, 2)`` is the twice -- it locks the two closure legs to each other, so
        both bispectra squeeze on the same soft leg and the integrand carries :math:`P(k_3)^2`.

        Measured on the box ``BB`` block at one bin, over ``order5`` 8 / 16 / 24 / 32:

        ==========  =========  =========  =========  =========
        variant     8          16         24         32
        ==========  =========  =========  =========  =========
        floored     1.348e21   1.418e21   1.428e21   1.426e21
        free        1.695e21   2.301e21   2.850e21   3.376e21
        ==========  =========  =========  =========  =========

        -- the free column still rising 18% per step at order 32 and already 2.4x the floored
        one, the floored column settled to 0.12%. With a *constant* theory both are flat to four
        digits, which is what says this is the spectral edge and not the geometry. A box-limit
        ratio hides all of it, because the windowed block diverges in step with the box one.

        Four legs are already bounded and are left alone: case ``A`` with ``i = 2`` and case ``B``
        tie one into a bin through :func:`_window_interval`. The rest are floored here -- the
        standard triangle's own closure leg, and the *derived* triangle's, whose magnitude
        :math:`k_{\rm tie}^2 + k_o^2 + 2 k_{\rm tie} k_o y` makes the free cosine ``y`` the axis
        to solve on.

        The estimator these stand for is finite because its closure leg is a mesh mode, and that
        is the floor: nothing below the fundamental exists to be averaged. ``None`` derives it
        from the volume the pairing is weighted in -- the box's, or the pairing's own field
        grouping's tie volume under a window -- which is the right one in both paths and is what
        every caller wants. ``0.`` switches the floors off exactly, reproducing the free axis node
        for node, which is how the table above was measured.

        Every floor is an exact interval with the quadrature placed inside, never a mask on a
        fixed grid; a mask would converge as :math:`1/n` on exactly the edge that carries the
        integrand's weight.
    qmin : float, default=None
        ``PT`` only: the floor below which the trispectrum's internal pair and triple sums are
        masked out. The tie fixes :math:`k_i + k'_j` and nothing else, so every *other* internal
        momentum is off shell and free to vanish, taking :math:`P(q)/q^2` with it. Unmasked, the
        block does not converge at all: with the tree theory it runs 3.230e28 / 1.463e28 /
        8.308e27 / 5.349e27 at ``order5`` 8 / 12 / 16 / 20, a clean :math:`{\rm order}^{-2}`,
        while a constant theory is flat to four digits. Masked at the fundamental it settles.

        Unlike the closure floors this *is* a mask, because the excluded region is a union of
        surfaces in five angles and there is no interval to solve for. It therefore converges as
        :math:`1/n` rather than spectrally -- which is still a convergence, against none.

        ``None`` uses the same fundamental the closure legs are floored at, on the same argument:
        an internal momentum is a mesh mode too. ``0.`` masks nothing, which is how the numbers
        above were measured.
    qwidth : float, default=0.
        Width of a smooth edge on that mask, as a fraction of ``qmin``; ``0.`` is a hard cut.
        A hard cut is the honest statement of the physics and is the default. The cost it carries
        is the discontinuity: the block converges as :math:`1/n` rather than spectrally, and the
        residue leaks into the otherwise exact isotropy identity of
        ``test_isotropic_kills_b000_b202`` through :func:`_frame`, whose azimuth origin for a
        derived leg is chosen by a line-of-sight-dependent branch.

        **A taper does not buy them back, and this was measured rather than assumed.** On the box
        block at one bin, over ``order5`` 8 / 12 / 16 / 24:

        ========  ============  =============  ==============
        qwidth    limit         change at 24   isotropy leak
        ========  ============  =============  ==============
        0         2.458e21      0.0117         2.10e-3
        0.5       2.126e21      0.0272         2.42e-3
        1.0       1.840e21      0.0074         2.88e-3
        2.0       1.418e21      0.0014         4.34e-3
        ========  ============  =============  ==============

        The taper costs 13% to 42% of the block, and the leak it was meant to cure gets *worse*
        with it, monotonically. That is not a paradox: widening the ramp gives the frame's
        line-of-sight dependence more support to act on, rather than less. Only the convergence
        rate improves, and only at widths where the bias is already unusable. ``qwidth`` is kept
        so the dead end stays measured instead of being tried again.
    """
    unprimed, primed = u['fields'], p['fields']
    ku = [u['k'][0][:, None], u['k'][1][:, None]]
    kp = [p['k'][0][None, :], p['k'][1][None, :]]
    dku, dkp = u['dk'][:, None], p['dk'][None, :]
    shape = (u['k'].shape[1], p['k'].shape[1])
    out = None

    for i in range(3):
        for j in range(3):
            # `pairs` restricts the nine leg pairings the way `tie_legs` restricts the windowed
            # BB. The block is the sum over all of them; a subset is a diagnostic, not an answer.
            if pairs is not None and (i, j) not in pairs:
                continue
            untied_u = [index for index in range(3) if index != i]
            untied_p = [index for index in range(3) if index != j]
            if kind == 'BB':
                groups = ((unprimed[i],) + tuple(unprimed[index] for index in untied_u),
                          (primed[j],) + tuple(primed[index] for index in untied_p))
            else:
                groups = ((unprimed[i], primed[j]),
                          tuple(unprimed[index] for index in untied_u)
                          + tuple(primed[index] for index in untied_p))
            volume = geometry.volume if tie_volume is None else tie_volume(*groups)
            if tie_volume is None:
                pair_weight = partial(geometry.pair_weight, count=count)
            else:
                pair_weight = partial(_inv_nmode, volume=volume, count=count)
            floor = _closure_floor(volume) if closure_min is None else closure_min
            soft = (_closure_floor(volume) if qmin is None else qmin) if kind != 'BB' else None
            # Axis 1 carries the standard triangle's shape, axis 3 the derived leg's
            # direction. Each is either an exact bin interval, an exact floor on a closure
            # magnitude, or free -- and a mapped axis ('M', on [0, 1]) serves the first two
            # alike. `floor` is None only when a caller switches the floors off.
            lo = hi = sel = None
            if j < 2:                                                            # case A
                if i < 2:
                    sel = np.abs(ku[i] - kp[j]) < dkp / 2.
                    if not sel.any():
                        continue
                    # The unprimed closure leg is free here, and carries a power spectrum.
                    lo = _closure_x_floor(ku[0], ku[1], floor)
                    kinds = 'm{}p{}p'.format('m' if lo is None else 'M',
                                             'm' if floor is None else 'M')
                else:
                    lo, hi = _window_interval(ku[0], ku[1], kp[j], dkp)
                    if not (hi > lo).any():
                        continue
                    kinds = 'mMp{}p'.format('m' if floor is None else 'M')
            elif i < 2:                                                          # case B
                lo, hi = _window_interval(kp[0], kp[1], ku[i], dku)
                if not (hi > lo).any():
                    continue
                kinds = 'mMp{}p'.format('m' if floor is None else 'M')
            else:                                                                # case C
                # Both closure legs are the tie, so flooring the unprimed one floors both.
                lo = _closure_x_floor(ku[0], ku[1], floor)
                kinds = 'm{}pMp'.format('m' if lo is None else 'M')
            nodes, weights = quad_nodes(kinds, order)

            def term(nd, _i=i, _j=j, _lo=lo, _hi=hi, _sel=sel, _w=pair_weight,
                     _uu=untied_u, _up=untied_p, _floor=floor, _qmin=soft, _qw=qwidth):
                mu = jnp.broadcast_to(nd[0], shape)
                if _j < 2:                                                       # case A
                    if _i < 2:
                        x, jac = _map_axis(nd[1], _lo, shape)
                    else:
                        x = _lo + (_hi - _lo) * nd[1]
                        jac = jnp.where(_hi > _lo, (_hi - _lo) / 2., 0.)
                    ul = _triangle(ku[0], ku[1], mu, x, nd[2])
                    tie = ul[_i]
                    ktie = _norm(tie)
                    ahat = sign * tie / jnp.clip(ktie, 1e-30, None)[..., None]
                    o = 1 - _j
                    # The primed closure leg is |tie + po|, so the free cosine is what decides
                    # whether it reaches zero: floor it on the same identity.
                    y, yjac = _map_axis(nd[3], _closure_x_floor(ktie, kp[o], _floor), shape)
                    po = kp[o][..., None] * _bhat(ahat, y, nd[4])
                    pl = [None, None, None]
                    pl[_j], pl[o] = sign * tie, po
                    pl[2] = -pl[0] - pl[1]
                    wn = _w(ktie, kp[_j], dkp) * jac * yjac * (1. if _i == 2 else _sel)
                elif _i < 2:                                                     # case B
                    xp = _lo + (_hi - _lo) * nd[1]
                    jac = jnp.where(_hi > _lo, (_hi - _lo) / 2., 0.)
                    pl = _triangle(kp[0], kp[1], mu, xp, nd[2])
                    tie = pl[2]
                    ktie = _norm(tie)
                    ahat = sign * tie / jnp.clip(ktie, 1e-30, None)[..., None]
                    o = 1 - _i
                    # Mirror of case A: here the *unprimed* closure leg is the derived one.
                    y, yjac = _map_axis(nd[3], _closure_x_floor(ktie, ku[o], _floor), shape)
                    uo = ku[o][..., None] * _bhat(ahat, y, nd[4])
                    ul = [None, None, None]
                    ul[_i], ul[o] = sign * tie, uo
                    ul[2] = -ul[0] - ul[1]
                    wn = _w(ku[_i], ktie, dku) * jac * yjac
                else:                                                            # case C
                    x, xjac = _map_axis(nd[1], _lo, shape)
                    ul = _triangle(ku[0], ku[1], mu, x, nd[2])
                    k3 = _norm(ul[2])
                    ahat = sign * ul[2] / jnp.clip(k3, 1e-30, None)[..., None]
                    ylo, yhi = _window_interval(k3, kp[1], kp[0], dkp)
                    y = ylo + (yhi - ylo) * nd[3]
                    jac = jnp.where(yhi > ylo, (yhi - ylo) / 2., 0.)
                    p2 = kp[1][..., None] * _bhat(ahat, y, nd[4])
                    p1 = -sign * ul[2] - p2
                    pl = [p1, p2, -p1 - p2]
                    wn = _w(_norm(p1), kp[0], dkp) * jac * xjac
                untied_u, untied_p = _uu, _up
                if kind == 'BB':
                    # Eq. (B2): the tied leg leads each bispectrum, and each factor is written on
                    # its *own* triangle with its own triangle's field labels. The Wick
                    # contractions actually produce factors that mix unprimed and primed labels,
                    # e.g. B_{a b b'}(k1, k2, k2'); on the delta support the two agree for one
                    # tracer, so this is exact for an auto-covariance and an approximation
                    # otherwise, differing through the field-dependent shot-noise pieces of
                    # B^(N). See desi-cov3-notes/covariance.tex, "Note the two organizations of the BB
                    # term" -- the own-triangle form is what the multipole boxes there use.
                    inner = (sp.bispectrum_shot(
                                 (unprimed[_i],) + tuple(unprimed[index] for index in untied_u),
                                 ul[_i], *[ul[index] for index in untied_u])
                             * sp.bispectrum_shot(
                                 (primed[_j],) + tuple(primed[index] for index in untied_p),
                                 pl[_j], *[pl[index] for index in untied_p]))
                else:
                    # Eq. (B3): the tie is a cross power spectrum between the two estimators, and
                    # the four untied legs make one trispectrum.
                    quad = ([ul[index] for index in untied_u]
                            + [pl[index] for index in untied_p])
                    inner = (sp.power_shot((unprimed[_i], primed[_j]), ul[_i])
                             * sp.trispectrum_shot(
                                 tuple(unprimed[index] for index in untied_u)
                                 + tuple(primed[index] for index in untied_p), *quad)
                             * _soft_momentum_mask(_qmin, *quad, qwidth=_qw))
                val = wn * inner
                hu = [leg / jnp.clip(_norm(leg), 1e-30, None)[..., None] for leg in ul[:2]]
                hp = [leg / jnp.clip(_norm(leg), 1e-30, None)[..., None] for leg in pl[:2]]
                return val, hu[0], hu[1], hp[0], hp[1]

            blk = _run(term, nodes, weights, batch_size, shape, u['ells'], p['ells'], 3, 3)
            out = blk if out is None else [[c + d for c, d in zip(ra, rb)]
                                           for ra, rb in zip(out, blk)]
    return out


def _closure_x_floor(ka, kb, closure_min):
    r"""The smallest :math:`x = \hat k_a \cdot \hat k_b` with :math:`|k_a + k_b| \ge` ``closure_min``.

    ``None`` when no floor is asked for, which the caller reads as "leave the axis free". This is
    :func:`_window_interval`'s identity with only a lower edge: the closure magnitude obeys
    :math:`k_c^2 = k_a^2 + k_b^2 + 2 k_a k_b x`, so a floor on it is an exact floor on ``x``.
    """
    if closure_min is None:
        return None
    return jnp.clip((closure_min**2 - ka**2 - kb**2)
                    / jnp.clip(2. * ka * kb, 1e-30, None), -1., 1.)


def _soft_momentum_mask(qmin, *legs, qwidth=0.):
    r"""Which configurations keep every internal momentum of a trispectrum above ``qmin``.

    The pair sums :math:`k_i + k_j` are the ``alpha``/``beta``/:math:`Z_2` denominators and the
    squeezed :math:`T \sim P(q)/q^2`; the triple sums are the :math:`F_3` recursion's
    :math:`1/|q_1 + q_2 + q_3|^2`, which on shell equals the fourth leg. Off shell -- which is
    what an untied leg of this quadrature is -- either can vanish. See :func:`_bb_tie_terms`'s
    ``qmin`` for what happens when they are not masked.

    ``qmin = None`` or ``0.`` masks nothing. ``qwidth > 0`` replaces the step by a C1 smoothstep
    ramping from ``qmin`` to ``(1 + qwidth) qmin``, which restores the quadrature's rate at the
    price of removing more than the excluded region -- see :func:`_bb_tie_terms`'s ``qwidth``.
    """
    if not qmin:
        return 1.

    def edge(q):
        if not qwidth:
            return (q >= qmin).astype(legs[0].dtype)
        t = jnp.clip((q - qmin) / (qwidth * qmin), 0., 1.)
        return t**2 * (3. - 2. * t)

    weight = 1.
    for index, first in enumerate(legs):
        for second in legs[index + 1:]:
            weight = weight * edge(_norm(first + second))
    for left in range(len(legs)):
        for middle in range(left + 1, len(legs)):
            for right in range(middle + 1, len(legs)):
                weight = weight * edge(_norm(legs[left] + legs[middle] + legs[right]))
    return weight


def _map_axis(node, lo, shape):
    """Place a mapped (``'M'``) node in ``[lo, 1]``, or pass a free (``'m'``) one through.

    Returns ``(value, jacobian)``. ``lo = None`` is the free axis, whose node already spans
    ``[-1, 1]`` with a ``dx/2`` measure and so needs no Jacobian.
    """
    if lo is None:
        return jnp.broadcast_to(node, shape), 1.
    return jnp.broadcast_to(lo + (1. - lo) * node, shape), (1. - lo) / 2.


def _closure_floor(volume):
    r"""``2 pi / V^(1/3)``: the fundamental of a box of volume ``V``.

    The floor below which a closure leg has no modes to average. See ``closure_min`` on
    :func:`_bb_tie_terms` for why the ``(2, 2)`` pairing needs one at all.
    """
    return 2. * np.pi / volume**(1. / 3.)


def _tie_volume(geometry):
    r"""``V_{\rm tie}(A, B) = 1 / Q_{\mathcal W}^{AB}(s \to 0)``, the volume a window leaves a tie.

    :meth:`SurveyGeometry.inverse_volume` returns exactly the :math:`s = 0` monopole of the
    two-anchor covariance window for the field groups ``A``, ``B``, which is the survey's
    replacement for the box's :math:`V` in Eq. (47). The groups keep their roles: ``BB`` pairs
    two triples, ``PT`` a pair against a quadruple, and a mixed-size lookup must not be swapped.
    """
    def volume(first, second):
        return 1. / geometry.inverse_volume(first, second)
    return volume


def _c22_tie(sp, u, p, geometry, order, sign, kind, batch_size=None, count=None,
             tie_volume=None, closure_min=None):
    r"""The doubly-derived pairing ``(i, j) = (2, 2)``, averaged over which side is anchored.

    Both tied legs are closure legs, so which triangle supplies the free shape integral and which
    one is reconstructed is genuinely ambiguous: the tie :math:`k'_3 = s\,k_3` leaves one triangle
    parametrised by its own orientation nodes and the other solved for, and the two choices are
    different quadratures of the same integral. In the periodic box they agree, which is what
    makes this an identity there and only an ansatz under a window -- nothing in the box limit can
    test the averaging itself.

    The primed-anchored half is the unprimed-anchored one with the two observables exchanged, so
    it is the same routine called the other way round and transposed back; there is no second
    construction to keep in step with the first. ``PT``'s field groups are mixed in size
    (:math:`(f_3 f'_3)` against the four untied legs) and stay in their roles under the exchange,
    so the same ``tie_volume`` serves both halves.
    """
    first = _bb_tie_terms(sp, u, p, geometry, order, sign, kind, batch_size, count,
                          pairs=[(2, 2)], tie_volume=tie_volume, closure_min=closure_min)
    second = _bb_tie_terms(sp, p, u, geometry, order, sign, kind, batch_size, count,
                           pairs=[(2, 2)], tie_volume=tie_volume, closure_min=closure_min)
    return [[0.5 * (first[i][j] + second[j][i].T) for j in range(len(p['ells']))]
            for i in range(len(u['ells']))]


def _ppp_channels(lmax=2):
    """S-basis channels the three-anchor window kernel is reconstructed in.

    These are independent of which multipoles the 3-point window actually stores: the stored
    monopole alone feeds every diagonal channel, with the Parseval weight
    :math:`(2L_1+1)(2L_2+1)(2J+1) H^2 = 1/\\|S_\\ell\\|^2`. Tying the list to the stored
    multipoles instead nearly cancels the Gaussian variance of an anisotropic pole such as
    ``(2, 0, 2)``.

    The list must be closed under exchange of the first two indices, because the primed side's
    :math:`S` is evaluated at permuted legs and a mirror-asymmetric list unbalances the
    six-permutation Wick sum. ``lmax = 2`` covers the standard poles and is the largest
    alias-free order at the default quadrature: by ``lmax = 3`` the channels are no longer
    numerically orthogonal to the lower ones, so raise ``order`` with it.
    """
    channels = []
    for ell1, ell2 in itertools.product(range(lmax + 1), repeat=2):
        for ell in range(abs(ell1 - ell2), ell1 + ell2 + 1, 2):
            if (ell1 + ell2 + ell) % 2 or ell % 2:
                continue
            if abs(wigner_3j(ell1, ell2, ell, 0, 0, 0)) < 1e-12:
                continue
            channels.append((ell1, ell2, ell))
    return channels


def _block_bb_ppp_window(sp, u, p, geometry, order, channel_lmax=2, closure_min=None):
    r"""The Gaussian ``Cov[B, B]`` ``PPP`` term with a survey window.

    .. math::
        {\rm Cov}^{PPP} = \mathfrak{A} \sum_{\sigma \in S_3}
            Q_{\mathcal W}^{(a f'_{\sigma(1)})(b f'_{\sigma(2)})(c f'_{\sigma(3)})}
            (k_1, k'_{\sigma(1)}, k_2, k'_{\sigma(2)})\,
            P^{(N)}_{a f'_{\sigma(1)}}(k_1) P^{(N)}_{b f'_{\sigma(2)}}(k_2)
            P^{(N)}_{c f'_{\sigma(3)}}(k_3)

    Two deltas tie two pairs of legs in the box, which is what leaves the block bin-diagonal
    there. Here they become the **three-anchor** window kernel, whose anchors are *pairs* -- one
    unprimed field and one primed field each -- and which therefore carries four momentum
    arguments rather than two.

    All three power spectra sit on unprimed legs, so the physics depends on one triangle only and
    the primed integral carries none: it reduces to the overlap of the estimator's
    :math:`S_{\ell'}` with the window channel, which is an orthogonality relation. It is computed
    rather than assumed, since the quadrature only makes it orthogonal to the extent it is
    converged.

Each :math:`P^{(N)}` is symmetrised over its two tied momenta, :math:`(P(k_m) + P(k'_m))/2`,
    and the product of the three expands into eight terms weighted :math:`1/8` -- one per way of
    sending each factor to the unprimed or the primed side.

    **Evaluating all three at the unprimed legs instead is not a harmless choice**, though it was
    made here once on the argument that the final ``(value + value.T) / 2`` of
    :func:`compute_spectrum3_covariance` restores what it costs. It does not: putting all the
    physics on one side makes the primed integral a pure orthogonality relation, which collapses
    ``channel_right`` onto the estimator's own multipole and kills every off-diagonal channel pair
    on that side. Measured against :mod:`jaxpower._cov3_legacy` on a varying ``P``, the one-sided
    form is 25% low on ``(2, 0, 2) x (2, 0, 2)`` and 6% low on ``(0, 0, 0) x (0, 0, 0)``. The
    error is proportional to :math:`dP/dk`, hence identically zero for a constant theory -- which
    is how it survived a constant-theory comparison that agreed to 0.1%.

    A permutation that ties the primed *closure* leg selects it through ``paxes`` and evaluates
    the window at that leg's per-node magnitudes, which gives the table an extra node axis.

    The *unprimed* closure leg is floored, in every permutation. All three power spectra sit on
    unprimed legs and one of those is always the closure leg, which nothing here ties to a primed
    bin -- unlike the box block, where four of the six pairings do. See ``closure_min`` on
    :func:`_triangle_nodes` for the branch point that leaves behind, and :func:`_block_bb_ppp`
    for what it costs: measured on the box block, drifting 5.79e20 to 6.45e20 between ``order3``
    8 and 48 instead of settling by 24.

    .. warning::
        **This block is about 8% low against its box limit, and the cause is not known.** Three
        explanations have been measured and refuted; do not spend them again.

        *Not the quadrature.* With a constant theory -- no soft spectral edge anywhere, and the
        box side exact at any order -- the ratio is 0.9180 at ``order5`` 6, 8, 12, 16 and 20
        alike. Four digits across five orders.

        *Not the window's mesh resolution*, which was the obvious suspect: the three-anchor window
        integrates *six* window fields, and on a 64^3 mesh of a 2000 box a uniform box of side
        1000 is 32 cells across. Refining to 128^3, with the binning held fixed so that only the
        window varies, moves the ratio the **wrong way**, 0.918 to 0.892 -- while over the same
        step the two-anchor window's own self-consistency *improves*, ``V Q_W(0)`` for the 3|3
        grouping going 1.038 to 0.980. A resolution effect does not do that.

        *Not the random catalogue's sparsity.* Ten times the density gives 0.925 at mesh 64 and
        0.887 at mesh 128 -- the same numbers, and the same backwards trend.

        That the ratio *worsens* as the window is better resolved is the clue worth following.
        This block reads the interpolated window, so the mesh reaches it only through how well
        small separations are measured, which points at the small-``s`` behaviour of the
        three-anchor path -- :func:`compute_spectrum3_covariance_window_block`, or the channel
        reconstruction above it -- rather than at anything in this function.
    """
    unprimed, primed = u['fields'], p['fields']
    if closure_min is None:
        closure_min = _closure_floor(1. / geometry.inverse_volume(unprimed, primed))
    nodes_u, weights_u = quad_nodes('mmp', order)
    nodes_p, weights_p = quad_nodes('mpmp', order)
    legs_u, hats_u, _, jacobian_u = _triangle_nodes(u['k'][0], u['k'][1], nodes_u, False,
                                                    closure_min)
    # The primed side is not floored: no power spectrum sits on a primed leg at all here, and its
    # closure magnitude enters only through the window, which is finite there.
    legs_p, hats_p, norms_p, _ = _triangle_nodes(p['k'][0], p['k'][1], nodes_p, True)
    # All three primed directions. Which two the window's channel basis is expanded in is decided
    # by the Wick permutation, not fixed at legs 1 and 2 -- see the note on `paxes` below. The
    # third is the closure leg, whose direction depends on the bin.
    hats_p_all = [leg / jnp.clip(norm, 1e-30, None)[..., None]
                  for leg, norm in zip(legs_p, norms_p)]
    channels = _ppp_channels(channel_lmax)

    project_u = [np.asarray(_Sfun(ell)(*hats_u)) * _NH2(ell) for ell in u['ells']]
    # The primed hats carry a bin axis they do not need -- nothing on that side depends on the
    # bin -- and the contractions below want a plain node vector, so take one column.
    project_p = [np.asarray(_Sfun(ell)(*hats_p))[:, 0] * _NH2(ell) for ell in p['ells']]
    channel_u = {ell: np.asarray(_Sfun(ell)(*hats_u)) for ell in channels}
    # Keyed on the permutation's anchored primed legs, as `_cov3_legacy`'s `_Sp_cache` is: the
    # basis is evaluated at `hats_p_all[paxes[0]], hats_p_all[paxes[1]]`, and reusing legs (1, 2)
    # for every permutation is exact for `(0, 0, 0)` -- a constant -- and wrong for every
    # anisotropic channel, which is how this hid behind a validating sigma(B000).
    channel_p_cache = {}

    def channels_primed(paxes):
        if paxes not in channel_p_cache:
            pair = (hats_p_all[paxes[0]], hats_p_all[paxes[1]])
            channel_p_cache[paxes] = {ell: np.asarray(_Sfun(ell)(*pair)) for ell in channels}
        return channel_p_cache[paxes]
    closure_p = np.asarray(norms_p[2])
    # A permutation that ties the primed closure leg sweeps its magnitude across the quadrature,
    # so the window is cell-averaged over each node's own sweep rather than sampled at the node --
    # see `_closure_measure`. x is axis 2 of 'mpmp', of stride `order` in the raveled node grid.
    measure_p = _closure_measure(p['k'][0], p['k'][1], order,
                                 (np.arange(len(weights_p)) // order) % order)

    out = [[0. for _ in p['ells']] for _ in u['ells']]
    for permutation in itertools.permutations(range(3)):
        anchors = [(unprimed[leg], primed[permutation[leg]]) for leg in range(3)]
        # Each factor is (P(k_m) + P(k'_m)) / 2, so it is evaluated at *both* ends of its tie:
        # the unprimed leg m and the primed leg the permutation pairs it with. The product of
        # the three expands into the eight `mask` terms below, each weighted 1/8.
        side_u = [np.asarray(sp.power_shot(anchors[leg], legs_u[leg])) for leg in range(3)]
        side_p = [np.asarray(sp.power_shot(anchors[leg], legs_p[permutation[leg]]))
                  for leg in range(3)]
        paxes = (permutation[0], permutation[1])
        channel_p = channels_primed(paxes)
        points = closure_p if 2 in paxes else None
        for channel_left in channels:
            # The unprimed side depends on the mask only through which factors land on it, and
            # not on `channel_right`, so it is built once per mask here.
            weighted_u = {}
            for mask in range(8):
                physics_u = 1.
                for leg in range(3):
                    if (mask >> leg) & 1:
                        physics_u = physics_u * side_u[leg]
                weighted_u[mask] = [(np.asarray(weights_u)[:, None] * projection
                                     * channel_u[channel_left] * np.asarray(jacobian_u)
                                     * physics_u).sum(axis=0)
                                    for projection in project_u]
            for channel_right in channels:
                table = np.asarray(compute_spectrum3_covariance_window_block(
                    geometry.window3, u['edges'], p['edges'], channel_left, channel_right,
                    fields1=anchors[0], fields2=anchors[1], fields3=anchors[2],
                    cache=geometry.cache, paxes=paxes, kp_points=points,
                    kp_measure=measure_p if points is not None else None)).real
                for mask in range(8):
                    physics_p = 1.
                    for leg in range(3):
                        if not (mask >> leg) & 1:
                            physics_p = physics_p * side_p[leg]
                    for jell, projection in enumerate(project_p):
                        # The estimator's own weight stays at its own legs -- it does not
                        # permute -- while the channel basis does, so this carries the primed
                        # bin axis, and so does the physics that now sits on this side.
                        weighted_p = (np.asarray(weights_p)[:, None] * projection[:, None]
                                      * channel_p[channel_right] * physics_p)
                        for iell, left in enumerate(weighted_u[mask]):
                            # A points axis is appended only when the window block actually
                            # carries one; the primed closure sweep may collapse it.
                            if table.ndim == 2:
                                block = table * left[:, None] * weighted_p.sum(axis=0)[None, :]
                            else:
                                block = np.einsum('bcn,b,nc->bc', table, left, weighted_p)
                            out[iell][jell] = out[iell][jell] + 0.125 * block
    return out


def _triangle_nodes(k1, k2, nodes, free_azimuth, closure_min=None):
    r"""One triangle's three legs, its two unit vectors and its leg magnitudes, at every node.

    ``free_azimuth`` gives the first leg an azimuth of its own. Only one of the two triangles in
    a ``Cov[B, B]`` block needs it: the integrand is invariant under a common rotation about the
    line of sight, so fixing the other triangle's azimuth to zero removes exactly that freedom.

    ``closure_min`` places the ``x = \hat k_1 . \hat k_2`` quadrature above the value that makes
    :math:`|k_1 + k_2| = ` ``closure_min``, instead of letting the closure leg run to zero. Any
    block that evaluates a power spectrum on an *unconstrained* closure leg needs this: with
    :math:`P(k_3) \sim k_3^{-1.5}` against the measure's :math:`k_3 dk_3 / (k_1 k_2)` the
    integrand goes as :math:`k_3^{-0.5}`, a square-root branch point that leaves the block
    converging as a slow power of the quadrature order rather than settling. The floor is the
    fundamental, below which the estimator has no modes to average; see :func:`_closure_floor`.

    Returns ``legs, hats, norms, jacobian``. The ``x`` range is per bin once floored, so the hats
    -- and every projection built from them -- then carry the bin axis that they do not carry
    otherwise, and ``jacobian`` is ``(1 - x_lo) / 2`` per bin rather than 1. Callers that pass a
    floor have to contract accordingly.
    """
    k1, k2 = jnp.asarray(k1), jnp.asarray(k2)
    axis = 2 if free_azimuth else 1
    if free_azimuth:
        first = _dirhat(jnp.asarray(nodes[:, 0]), jnp.asarray(nodes[:, 1]))
    else:
        first = _ahat(jnp.asarray(nodes[:, 0]))
    x, phi = jnp.asarray(nodes[:, axis]), jnp.asarray(nodes[:, axis + 1])
    if closure_min is None:
        second = _bhat(first, x, phi)
        first = jnp.broadcast_to(first[:, None, :], (len(x), len(k1), 3))
        second = jnp.broadcast_to(second[:, None, :], first.shape)
        jacobian = 1.
    else:
        lo = jnp.clip((closure_min**2 - k1**2 - k2**2) / (2. * k1 * k2), -1., 1.)
        jacobian = (1. - lo) / 2.
        # The 'm' axis arrives on [-1, 1] with a dx/2 measure, so the map to [lo, 1] goes
        # through (x + 1) / 2; the Jacobian is the same (1 - lo) / 2 either way.
        x = lo[None, :] + (1. - lo)[None, :] * ((x[:, None] + 1.) / 2.)
        first = jnp.broadcast_to(first[:, None, :], x.shape + (3,))
        second = _bhat(first, x, jnp.broadcast_to(phi[:, None], x.shape))
    leg1 = k1[None, :, None] * first
    leg2 = k2[None, :, None] * second
    legs = [leg1, leg2, -leg1 - leg2]
    return legs, (first, second), [_norm(leg) for leg in legs], jacobian


def _block_bb_bb_window(sp, u, p, geometry, order, window_ells=(0, 2, 4),
                        tie_legs=(0, 1, 2), batch_size=None, double_closure=True,
                        closure_min=None, floor_closure=True):
    r"""The connected ``Cov[B, B]`` ``BB`` term with a survey window.

    .. math::
        {\rm Cov}^{BB} = \mathfrak{A} \sum_{i,j}
            Q_{\mathcal W}^{(f_i f_{r_1} f_{r_2})(f'_j f'_{s_1} f'_{s_2})}(k_i, k'_j)\,
            B^{(N)}_{f_i f_{r_1} f_{r_2}} B^{(N)}_{f'_j f'_{s_1} f'_{s_2}}

    with :math:`\mathfrak{A}` the estimator measure of ``desi-cov3-notes/covariance.tex``: both triangles
    integrated over their own orientations, with :math:`S_{\ell_1\ell_2L}` on each one's literal
    legs.

    .. note::
        The doubly-derived tie ``(i, j) = (2, 2)`` is **not summed in the channel expansion
        below**; it is added by :func:`_c22_tie`, which consumes the vector tie exactly instead.
        It is not a matter of resolution: the double Legendre channels the window is expanded in
        enforce only :math:`|k_3| = |k_3'|`, not the vector tie, so that pair converges to the
        wrong number however fine the quadrature. Summed here it takes the box-limit ratio to
        1.14 / 1.35 / 1.22 at ``order5`` 6 / 8 / 10 with the spread reaching [0.31, 2.90];
        dropped, the same ratio converges smoothly to 0.979 / 0.963 / 0.957 with the spread
        tightening to [0.84, 0.99]. The residual few per cent *is* that missing term, and
        :func:`_c22_tie` is what supplies it.

        It is the truncation in :math:`L, L'` that does this, **not** the :math:`m = 0`
        truncation of the kernel, and that was measured rather than argued: summing the
        :math:`m \neq 0` channels as well, with this pair included, gives a median box-limit ratio
        of 3.5830 against the truncated kernel's 3.5998 -- a factor 3.6 either way. On this
        pairing both tied legs are closure legs, so the kernel is asked to localise a vector
        mismatch rather than a magnitude, and an angular delta has flat Legendre content: no
        finite :math:`(L, L')` sum represents it, exactly as for the ``PT`` tie -- see
        :func:`_block_bb_pt_window`. The
        repair is an exact treatment of this pair, :func:`_c22_tie`, and not a better multipole
        kernel.

    The box ties :math:`k'_j = +k_i` with a radial delta, which removes two integrals. A window
    replaces the delta by :math:`Q_{\mathcal W}`, so both triangles are integrated freely -- eight
    angles at fixed line of sight, less the one common rotation about it, so seven axes in total,
    split three and four between them.

    .. note::
        With the :math:`m = 0` kernel used by default (see the module docstring) the two triangles
        may be rotated about the line of sight *independently*, because nothing in the integrand
        then carries their relative azimuth: the primed triangle's absolute azimuth is a redundant
        quadrature axis, integrating a constant to the :math:`10^{-4}` the :math:`m \neq 0`
        channels are worth, and six axes would do -- ``mmp`` in place of ``mpmp`` is a free factor
        ``order`` on the primed side for anyone who wants it. It is kept as a convergence
        diagnostic.

    Parameters
    ----------
    tie_legs : tuple, default=(0, 1, 2)
        Restricts which legs the tie sum runs over. A subset is a diagnostic, not an answer.
    double_closure : bool, default=True
        Whether to add the ``(2, 2)`` pairing through :func:`_c22_tie`. Off is the state this
        block was in before that routine existed, and is what the few per cent quoted above is
        measured against.

    What makes that affordable is that everything factorises between the two triangles. The
    physics does, because each bispectrum lives on its own triangle; the estimator weights do;
    and :math:`Q_{\mathcal W}` is a table in the two legs' bins. So the two sides are reduced
    separately and contracted at the end, and the node axes only survive for a leg that is a
    triangle's *closure* leg, whose magnitude varies over the integral and which therefore has to
    be interpolated per node rather than rebinned.
    """
    unprimed, primed = u['fields'], p['fields']
    if closure_min is None:
        closure_min = _closure_floor(1. / geometry.inverse_volume(unprimed, primed))
    # Both triangles carry their own B^(N), whose shot line evaluates a power spectrum on that
    # triangle's closure leg, and neither leg is tied here -- both orientations are free. So both
    # need the floor; see `closure_min` on `_triangle_nodes` for what it is worth.
    floor = closure_min if floor_closure else None
    nodes_u, weights_u = quad_nodes('mmp', order)
    nodes_p, weights_p = quad_nodes('mpmp', order)
    legs_u, hats_u, norms_u, jacobian_u = _triangle_nodes(u['k'][0], u['k'][1], nodes_u, False,
                                                          floor)
    legs_p, hats_p, norms_p, jacobian_p = _triangle_nodes(p['k'][0], p['k'][1], nodes_p, True,
                                                          floor)
    edges_u, edges_p = np.asarray(u['edges']), np.asarray(p['edges'])
    nbin_u, nbin_p = edges_u.shape[0], edges_p.shape[0]

    project_u = [_Sfun(ell)(*hats_u) * _NH2(ell) for ell in u['ells']]
    project_p = [_Sfun(ell)(*hats_p) * _NH2(ell) for ell in p['ells']]
    out = [[0. for _ in p['ells']] for _ in u['ells']]

    # A tie on a closure leg sweeps its magnitude across the quadrature, and the window is
    # narrower than a cell is wide, so it is cell-averaged rather than sampled at the node -- see
    # `_closure_measure`. The x axis is axis 1 of 'mmp' for the unprimed triangle and axis 2 of
    # 'mpmp' for the primed one, both of stride `order` in the C-order raveled node grid.
    x_index = (np.arange(len(weights_u)) // order) % order
    x_index_p = (np.arange(len(weights_p)) // order) % order
    measure_u = _closure_measure(u['k'][0], u['k'][1], order, x_index, floor)
    measure_p = _closure_measure(p['k'][0], p['k'][1], order, x_index_p, floor)

    for i in tie_legs:
        rest_u = [index for index in range(3) if index != i]
        fields_u = (unprimed[i],) + tuple(unprimed[index] for index in rest_u)
        value_u = sp.bispectrum_shot(fields_u, legs_u[i], *[legs_u[index] for index in rest_u])
        cosine_u = np.asarray(legs_u[i][..., 2] / jnp.clip(norms_u[i], 1e-30, None))
        spec_u = (edges_u[:, i, :] if i < 2 else np.asarray(norms_u[2]).ravel())
        for j in tie_legs:
            if i == 2 and j == 2:
                # The doubly-derived tie is not representable this way at all. The double
                # Legendre channels the window is expanded in enforce only |k3| = |k3'|, not the
                # vector tie k3 = k3', so summing this pair here does not converge to the right
                # answer however fine the quadrature -- it converges to a different one. It is
                # added below by `_c22_tie`, which consumes the tie as a vector. That also
                # removes what would otherwise be the memory ceiling of the whole block: this is
                # the only pairing whose Q_W table would carry node axes on both sides,
                # (q^3 nbins)^2, tens of GB at a production quadrature. The exact substitution
                # needs the window at s -> 0 alone, which is one number.
                continue
            rest_p = [index for index in range(3) if index != j]
            fields_p = (primed[j],) + tuple(primed[index] for index in rest_p)
            value_p = sp.bispectrum_shot(fields_p, legs_p[j],
                                         *[legs_p[index] for index in rest_p])
            cosine_p = np.asarray(legs_p[j][..., 2] / jnp.clip(norms_p[j], 1e-30, None))
            spec_p = (edges_p[:, j, :] if j < 2 else np.asarray(norms_p[2]).ravel())
            for ell1 in window_ells:
                for ell2 in window_ells:
                    # Same-size field groups: the swap A <-> B is an equivalent relabelling, so
                    # the window is symmetrized by handing the block a pair. Mixed sizes have
                    # fixed roles and must not be swapped -- see `compute_QW_AB`.
                    table = np.asarray(compute_spectrum2_covariance_window_block(
                        (geometry.window2, geometry.window2), spec_u, spec_p, ell1, ell2,
                        fields1=fields_u, fields2=fields_p, cache=geometry.cache,
                        k1_is_points=i == 2, k2_is_points=j == 2,
                        k1_measure=measure_u if i == 2 else None,
                        k2_measure=measure_p if j == 2 else None)).real
                    # A leg with its own bin edges gives a table with no node axis, and its
                    # side of the integral is summed over nodes first; a closure leg keeps one.
                    axes_u = 'nb' if i == 2 else 'b'
                    axes_p = 'mc' if j == 2 else 'c'
                    table = table.reshape(((len(weights_u), nbin_u) if i == 2 else (nbin_u,))
                                          + ((len(weights_p), nbin_p) if j == 2 else (nbin_p,)))
                    coefficient = ((2 * ell1 + 1) * (2 * ell2 + 1)
                                   * (-1)**(ell1 // 2) * (-1)**(ell2 // 2))
                    legendre_u = np.asarray(get_legendre(ell1)(cosine_u))
                    legendre_p = np.asarray(get_legendre(ell2)(cosine_p))
                    contraction = f'{axes_u}{axes_p},{axes_u},{axes_p}->bc'
                    for iell, projection_u in enumerate(project_u):
                        # The projections depend on the triangle's directions alone, which carry
                        # no bin axis -- the bins only set the leg magnitudes -- so they broadcast
                        # in explicitly.
                        side_u = (np.asarray(weights_u)[:, None]
                                  * np.asarray(projection_u) * np.asarray(jacobian_u)
                                  * legendre_u * np.asarray(value_u))
                        if i != 2:
                            side_u = side_u.sum(axis=0)
                        for jell, projection_p in enumerate(project_p):
                            side_p = (np.asarray(weights_p)[:, None]
                                      * np.asarray(projection_p) * np.asarray(jacobian_p)
                                      * legendre_p * np.asarray(value_p))
                            if j != 2:
                                side_p = side_p.sum(axis=0)
                            out[iell][jell] = out[iell][jell] + coefficient * np.einsum(
                                contraction, table, side_u, side_p)
    if double_closure and 2 in tie_legs:
        extra = _c22_tie(sp, u, p, geometry, order, 1., 'BB', batch_size=batch_size,
                         tie_volume=_tie_volume(geometry), closure_min=closure_min)
        out = [[a + np.asarray(b) for a, b in zip(row, add)] for row, add in zip(out, extra)]
    return out


def _block_bb_pt_window(sp, u, p, geometry, order, batch_size=None, pairs=None,
                        closure_min=None):
    r"""The connected ``Cov[B, B]`` ``PT`` term with a survey window, by the exact vector tie.

    .. math::
        {\rm Cov}^{PT} = \mathfrak{A} \sum_{i,j}
            \frac{1}{\tilde N_{\rm mode}(k_i, k'_j)\big|_{V \to V_{\rm tie}}}\,
            P^{(N)}_{f_i f'_j}(k_i)\,
            T^{(N)}(k_{r_1}, k_{r_2}, k'_{s_1}, k'_{s_2})

    This is :func:`_bb_tie_terms` -- the periodic-box block, unchanged -- with one substitution:
    Eq. (47)'s mode count is taken in the window's effective tie volume
    :math:`V_{\rm tie} = 1 / Q_{\mathcal W}(s \to 0)` rather than the box's :math:`V`. The tie
    :math:`k'_j = -k_i` is consumed as a *vector* at every node, exactly as in the box, and the
    partner triangle's residual freedom is solved against its own bins in closed form.

    That is not a stylistic choice over a multipole kernel: it is the only representation this
    term has, and the kernel version was measured and removed
    (``claude_cov3_window/removed_block_bb_pt_window_channels.py`` keeps its account of five
    refuted hypotheses). ``PT`` takes two legs from each triangle into one trispectrum, and those four sum to
    :math:`-(k_i + k'_j)`, which is zero only on the tie -- a connected trispectrum exists nowhere
    else. So the tie cannot be freed, and it cannot be smeared either: :math:`T` is a *peak* at the
    tied orientation, :math:`400\times` its average over a freed one. Localising a direction is
    what the :math:`m = 0` double-Legendre kernel provably cannot do, since it constrains the two
    magnitudes and the two line-of-sight cosines only, and an angular delta has flat Legendre
    content in :math:`L`. The channel block's box limit ran 3.89 / 2.27 / 0.99 / 0.77 / 0.45 at
    ``order5`` 4 / 6 / 8 / 10 / 12, passing through one rather than settling at it.

    What this substitution neglects instead is the *width* of the window's ridge, which the box
    makes a delta. That has a visible consequence and an invisible one. Visibly, the six pairings
    that tie two *binned* legs keep the box's bin-overlap selector, so the block stays banded in
    those bins where a window would couple them; a survey whose window is broad compared with a
    bin will want that coupling back. Invisibly, every pairing's amplitude is off by however much
    the ridge's finite width matters, which the retired implementation put at 20-50% on a single
    tie. **No box-limit test can see either** -- the box is the limit in which both vanish -- so
    the 1.027 below bounds this block's bookkeeping, not its accuracy on a real survey. That needs
    mocks.

    The doubly-derived ``(2, 2)`` pairing goes through :func:`_c22_tie`, which averages the two
    anchorings; see there.

    .. note::
        The trispectrum's soft internal momenta **are** masked, through ``qmin`` on
        :func:`_bb_tie_terms`, and the block does not work without it. The tie fixes
        :math:`k_i + k'_j` alone, so every other internal pair and triple sum is off shell and
        free to vanish, taking :math:`P(q)/q^2` with it. Unmasked, the *box* block is about
        :math:`10^7` times too large and falls as :math:`{\rm order}^{-2}` instead of settling.

        This note previously said the opposite -- that the mask was unnecessary here because the
        tie closes the four untied legs and :mod:`jaxpower.pt`'s ``_ZERO_EDGE`` covers the rest --
        and cited a flat box limit as evidence. The box limit is flat, and it is worthless as
        evidence: the windowed block *is* the box block with one volume swapped, so the two
        diverge together and their ratio converges beautifully onto nothing. Only an absolute
        convergence scan shows it. That mistake is recorded because the reasoning error is the
        reusable part, not the conclusion.
    """
    volume = _tie_volume(geometry)
    selected = [(i, j) for i in range(3) for j in range(3)
                if pairs is None or (i, j) in pairs]
    out = None
    rest = [pair for pair in selected if pair != (2, 2)]
    if rest:
        out = _bb_tie_terms(sp, u, p, geometry, order, -1., 'PT', batch_size, None, rest,
                            tie_volume=volume, closure_min=closure_min)
    if (2, 2) in selected:
        add = _c22_tie(sp, u, p, geometry, order, -1., 'PT', batch_size,
                       tie_volume=volume, closure_min=closure_min)
        out = add if out is None else [[a + b for a, b in zip(ra, rb)]
                                       for ra, rb in zip(out, add)]
    return out




def _block_bb_bb(sp, u, p, geometry, order, batch_size=None, count=None, pairs=None,
                 closure_min=None):
    return _bb_tie_terms(sp, u, p, geometry, order, 1., 'BB', batch_size, count, pairs,
                         closure_min=closure_min)


def _block_bb_pt(sp, u, p, geometry, order, batch_size=None, count=None, pairs=None,
                 closure_min=None, qmin=None, qwidth=0.):
    return _bb_tie_terms(sp, u, p, geometry, order, -1., 'PT', batch_size, count, pairs,
                         closure_min=closure_min, qmin=qmin, qwidth=qwidth)


def _block_bb_p6(sp, u, p, geometry, order, batch_size=None, size=None, seed=0,
                 closure_min=None):
    """Eq. (33) with Eqs. (B4)-(B6): seven angles, no delta and no window left.

    Both triangles' closure legs are derived and unconstrained here, and :math:`P_6`'s shot lines
    put power spectra on them, so both are floored -- see ``closure_min`` on
    :func:`_bb_tie_terms`. Sobol sampling makes this block's own rate hard to read, but it has the
    same two legs as the blocks where the effect was measured.
    """
    ku1, ku2 = u['k'][0][:, None], u['k'][1][:, None]
    kp1, kp2 = p['k'][0][None, :], p['k'][1][None, :]
    fields = tuple(u['fields']) + tuple(p['fields'])
    shape = (u['k'].shape[1], p['k'].shape[1])
    if closure_min is None:
        closure_min = _closure_floor(1. / geometry.inverse_volume(u['fields'], p['fields']))
    u_floor = _closure_x_floor(ku1, ku2, closure_min)
    p_floor = _closure_x_floor(kp1, kp2, closure_min)
    mapped = 'm' if u_floor is None else 'M'
    nodes, weights = quad_nodes('m{}pmp{}p'.format(mapped, mapped), order, size=size, seed=seed)

    # Each triangle carries its own bin axis and no more: the unprimed legs broadcast to
    # (nbin_u, 1), the primed to (1, nbin_p). Using the full block shape here instead costs a
    # factor nbin_p in every intermediate of the 1296-tree six-point sum.
    shape_u, shape_p = jnp.shape(ku1 * ku2), jnp.shape(kp1 * kp2)

    def term(nd, _ulo=u_floor, _plo=p_floor):
        xu, ujac = _map_axis(nd[1], _ulo, shape_u)
        a = jnp.broadcast_to(_ahat(nd[0]), shape_u + (3,))
        b = _bhat(a, xu, nd[2])
        xp, pjac = _map_axis(nd[5], _plo, shape_p)
        ap = jnp.broadcast_to(_dirhat(nd[3], nd[4]), shape_p + (3,))
        bp = _bhat(ap, xp, nd[6])
        U1, U2 = ku1[..., None] * a, ku2[..., None] * b
        P1, P2 = kp1[..., None] * ap, kp2[..., None] * bp
        val = (sp.six_point_shot(fields, [U1, U2, -U1 - U2], [P1, P2, -P1 - P2])
               * ujac * pjac
               * geometry.inverse_volume(u['fields'], p['fields']))
        return val, a, b, ap, bp

    return _run(term, nodes, weights, batch_size, shape, u['ells'], p['ells'], 3, 3)


# ======================================================================================
# entry point
# ======================================================================================

#: Blocks, in the paper's own naming. ``compute_spectrum3_covariance(..., terms=...)`` selects a
#: subset, which is how the per-term curves of the paper's Figures 1-4 are produced.
TERMS = ('PP', 'T', 'PB', 'P5', 'PPP', 'BB', 'PT', 'P6')

#: Quadrature defaults. ``order`` is the number of nodes per angular axis for the gridded blocks;
#: ``p6_size`` and ``p5_size`` switch those blocks to scrambled Sobol with that many points
#: (``None`` grids them instead). Every one of these is scanned in ``test_convergence.py``.
DEFAULTS = dict(order3=16, order5=8, order7=6, p5_size=None, p6_size=1 << 13, seed=0)


def _group(observable):
    """Group the observable's leaves by what they are and how they are binned.

    Multipoles sharing a binning are integrated together: the angular integrand is the same for
    all of them and only the projection weight differs, so the expensive spectra are evaluated
    once per group pair rather than once per multipole pair.
    """
    groups = []
    for idx, (label, obs) in enumerate(observable.items(level=None)):
        nfields = len(label['fields'])
        edges = np.asarray(obs.edges('k'))
        coords = np.asarray(obs.coords('k', center='mid_if_edges_and_nan')).T
        if nfields == 2:
            dk = edges[:, 1] - edges[:, 0]
        elif nfields == 3:
            widths = edges[..., 1] - edges[..., 0]
            dk = widths[:, 0]
            if not np.allclose(widths, dk[:, None]):
                raise ValueError('the paper\'s Eq. (45) top-hat assumes one bin width; this '
                                 'bispectrum binning has different widths on its two legs')
        else:
            raise ValueError(f'unsupported {nfields}-point observable')
        # The field tuple is part of the key, not just carried along: two tracers binned the same
        # way are different observables, and merging them would evaluate one's spectra for the
        # other's covariance.
        fields = tuple(label['fields'])
        key = (fields, edges.shape, edges.tobytes())
        for group in groups:
            if group['key'] == key:
                group['ells'].append(label['ells'])
                group['index'].append(idx)
                break
        else:
            groups.append(dict(key=key, nfields=nfields, fields=fields, k=coords, dk=dk,
                               edges=edges, ells=[label['ells']], index=[idx],
                               size=len(coords.T)))
    return groups


def _subbin_group(g, count, nsub):
    r"""A radially sub-binned copy of a group, and the matrix that contracts back.

    Eq. (43) evaluates the spectra at one representative :math:`k` per bin. What an estimator
    averages is the bin's modes, and for a Gaussian term that is
    :math:`\langle P^2 \rangle`, not :math:`P(\bar k)^2` -- the two differ by the variance of
    :math:`P` across the bin, which is large wherever the bin is wide compared with :math:`k`.
    Splitting each bin into ``nsub`` radial shells and contracting the resulting covariance with
    the mode weights recovers it exactly: sub-shells at different radii are disjoint sets of
    modes, so
    :math:`\sum_r (N_r/N)^2 \, 2 P_r^2/N_r = (2/N) \langle P^2\rangle`.

    For a bispectrum bin the expansion runs over **both** binned legs, so the coarse bin's
    :math:`k_1 \neq k_2` triangles are included -- which is exactly what a coarse
    ``sugiyama-diagonal`` bin contains and what rebinning a finer *diagonal* covariance would
    miss.
    """
    nb, nf = g['size'], g['nfields']
    if nf == 2:
        kbar, w, ee = count.subbins(np.asarray(g['edges']), nsub)
        sub = dict(g, k=kbar.ravel(), edges=ee.reshape(-1, 2),
                   dk=(ee[..., 1] - ee[..., 0]).ravel(), size=nb * nsub)
        W = np.zeros((nb, nb * nsub))
        for i in range(nb):
            W[i, i * nsub:(i + 1) * nsub] = w[i]
        return sub, W
    e = np.asarray(g['edges'])                                   # (nb, 2, 2)
    k1, w1, e1 = count.subbins(e[:, 0], nsub)
    k2, w2, e2 = count.subbins(e[:, 1], nsub)
    kk = np.stack([np.repeat(k1, nsub, axis=1), np.tile(k2, (1, nsub))], axis=0)
    ed = np.stack([np.repeat(e1, nsub, axis=1), np.tile(e2, (1, nsub, 1))], axis=2)
    sub = dict(g, k=kk.reshape(2, -1), edges=ed.reshape(-1, 2, 2),
               dk=np.repeat((e1[..., 1] - e1[..., 0]).ravel(), 1).reshape(nb, nsub).repeat(
                   nsub, axis=1).ravel(), size=nb * nsub * nsub)
    W = np.zeros((nb, nb * nsub * nsub))
    for i in range(nb):
        W[i, i * nsub * nsub:(i + 1) * nsub * nsub] = np.outer(w1[i], w2[i]).ravel()
    return sub, W


def compute_spectrum3_covariance(window2, window3, observable, theory=None, shotnoise: float=0.,
                                 cache=None, batch_size=None,
                                 terms=TERMS, nmodes='continuum', nsub=(4, 2),
                                 **options):
    r"""The periodic-box covariance of arXiv:1908.06234.

    Parameters
    ----------
    window2, window3 : MeshAttrs
        The box. Only ``boxsize`` is used, through :math:`V = \prod {\rm boxsize}`; both
        arguments are accepted, and must agree, so that the call site is the same as
        ``jaxpower.cov3.compute_spectrum3_covariance``. Anything other than a
        :class:`MeshAttrs` is rejected here rather than silently mis-handled: this module has no
        window machinery at all.
    observable : ObservableTree
        Power spectrum and/or bispectrum multipoles, giving the multipole orders and the binning.
        Bispectrum leaves carry ``(k1, k2)`` per bin, with the closure leg free, which is the
        binning Eq. (34) defines.
    theory : callable
        ``theory(fields)`` returns the **connected** ``len(fields)``-point spectrum as a callable
        of ``len(fields) - 1`` wavevectors, or ``None``. All discreteness is added here.
    shotnoise : float
        :math:`1/\bar n`, the coincidence amplitude. The paper is Poisson throughout, so the
        higher coincidences are its powers.
    terms : tuple
        Which blocks to include; see :data:`TERMS`.
    nmodes : str
        ``'continuum'`` uses Eq. (43)'s :math:`4 \pi k^2 \Delta k V / (2\pi)^3`, which is what
        the paper does and what reproducing it needs. ``'mesh'`` counts the grid modes of
        ``window2`` instead, which is what a measurement on that mesh actually averages over;
        see :class:`ModeCount`. Use ``'mesh'`` whenever the covariance is to be compared with
        mocks measured on a known grid, and make sure ``window2.meshsize`` is that grid's.
    **options
        Quadrature settings, see :data:`DEFAULTS`.

    Returns
    -------
    CovarianceMatrix
    """
    # The periodic box is selected by passing a MeshAttrs, exactly as the retired implementation
    # did; anything else is a survey window. This is the `use_window_kernels` branch of
    # `_cov3_legacy.py`, expressed as which geometry object the blocks are handed.
    survey = not isinstance(window2, MeshAttrs)
    if window3 is None and not survey:
        # The box needs no 3-point window, and callers of the retired implementation passed None
        # rather than repeat the mesh. Both spellings work.
        window3 = window2
    if survey:
        # The blocks that are not yet written must refuse rather than fall through and return
        # box numbers for a survey.
        # The three purely connected terms need nothing but the right normalisation: their
        # angular measure is the box's unchanged and the window enters only as 1/V^(n), which
        # `inverse_volume` supplies -- 1/V^(4) for T, 1/V^(5) for P5, 1/V^(6) for P6, each the
        # integral of the window fields over the cross pairing of the two groups. `PP`, `PB`,
        # `PPP` and `BB` each have their own windowed integral, because each carries a radial
        # delta that a window smears into a dense bin coupling.
        # `PPP` is the only windowed block that needs the *three*-anchor window: its kernel has
        # three anchor pairs, not two. Say so here rather than let `window3.get(...)` fail with an
        # AttributeError six frames down -- `window3=None` is legal for a box, where it means the
        # box of `window2`, so it is an easy thing to pass by accident.
        if 'PPP' in terms and window3 is None:
            raise ValueError(
                'the cutsky PPP term needs the three-anchor covariance window: pass window3 from '
                'compute_fkp3_covariance_window. window3=None means "the box of window2", which '
                'is meaningful only for a periodic box.')
        volume = None
    else:
        if not np.allclose(np.asarray(window2.boxsize), np.asarray(window3.boxsize)):
            raise ValueError('window2 and window3 describe different boxes')
        volume = float(np.prod(np.asarray(window2.boxsize)))
    opt = dict(DEFAULTS, **options)
    sp = Spectra(theory=theory, shotnoise=shotnoise)
    groups = _group(observable)
    if nmodes not in ('continuum', 'mesh'):
        raise ValueError(f"nmodes must be 'continuum' or 'mesh', not {nmodes!r}")
    counts = {}
    if nmodes == 'mesh' and survey:
        raise ValueError("nmodes='mesh' counts the grid modes of a periodic box and means "
                         'nothing for a survey window, whose mode coupling is the window itself')
    if nmodes == 'mesh':
        # One counter per distinct bin width; the shell width is what defines the count.
        cache = {} if cache is None else cache
        for g in groups:
            for dk in np.unique(np.round(np.asarray(g['dk']), 12)):
                counts.setdefault(float(dk), cache.setdefault(
                    ('modecount', float(dk), tuple(np.atleast_1d(window2.meshsize).tolist())),
                    ModeCount(window2, float(dk))))
        if any(np.ptp(np.round(np.asarray(g['dk']), 12)) > 0 for g in groups):
            raise ValueError("nmodes='mesh' needs one bin width per observable leaf")
    # Round the lookup the same way the keys were rounded: 0.03 - 0.01 is not 0.02 in binary.
    count = counts.get(round(float(np.asarray(groups[0]['dk']).ravel()[0]), 12),
                       None) if counts else None
    subcount, subgroups = None, None
    if count is not None and max(nsub) > 1 and not survey:
        widths = {float(v) for g in groups for v in np.atleast_1d(np.round(g['dk'], 12)).ravel()}
        if len(widths) > 1:
            raise ValueError("nmodes='mesh' with nsub > 1 needs one bin width for the whole "
                             'observable')
        dk = widths.pop()
        ns = {2: nsub[0], 3: nsub[1]}
        subcount = {m: ModeCount(window2, dk / n) for m, n in ns.items() if n > 1}
        subgroups = [_subbin_group(g, count, ns[g['nfields']]) if ns[g['nfields']] > 1
                     else (g, np.eye(g['size'])) for g in groups]
    # Everything geometric the blocks touch goes through here; a survey window replaces this
    # object rather than adding a branch inside each block.
    geometry = (SurveyGeometry(window2, window3, cache) if survey
                else BoxGeometry(volume, count))
    nitem = sum(len(g['index']) for g in groups)
    blocks = [[None] * nitem for _ in range(nitem)]

    for iu, gu in enumerate(groups):
        for ip, gp in enumerate(groups):
            if ip < iu:
                continue
            # Only the (2, 3) orientation is implemented, so a bispectrum group sitting before a
            # power spectrum group is computed the other way round and transposed. Relying on
            # some later (2, 3) pair to fill this in by symmetry only works while every power
            # spectrum leaf precedes every bispectrum one, which stops being true as soon as the
            # observable holds more than one tracer.
            swapped = (gu['nfields'], gp['nfields']) == (3, 2)
            rows, cols = (gp, gu) if swapped else (gu, gp)
            irows, icols = (ip, iu) if swapped else (iu, ip)
            nu, npp = rows['nfields'], cols['nfields']
            res = None
            if (nu, npp) == (2, 2):
                gaussian = (_block_pp_gaussian_window if geometry.directional
                            else _block_pp_gaussian)
                todo = [('PP', gaussian, dict(order=opt['order3'])
                         if geometry.directional else dict(order=opt['order3'], count=count)),
                        ('T', _block_pp_trispectrum,
                         dict(order=opt['order3'], batch_size=batch_size))]
            elif (nu, npp) == (2, 3):
                unconnected = (_block_pb_unconnected_window if geometry.directional
                               else _block_pb_unconnected)
                todo = [('PB', unconnected, dict(order=opt['order3'])
                         if geometry.directional else dict(order=opt['order3'], count=count)),
                        ('P5', _block_pb_p5, dict(order=opt['order5'], batch_size=batch_size,
                                                  size=opt['p5_size'], seed=opt['seed']))]
            elif (nu, npp) == (3, 3):
                bb = _block_bb_bb_window if geometry.directional else _block_bb_bb
                ppp = _block_bb_ppp_window if geometry.directional else _block_bb_ppp
                todo = [('PPP', ppp, dict(order=opt['order5']) if geometry.directional
                         else dict(order=opt['order3'], count=count)),
                        ('BB', bb, dict(order=opt['order5'], batch_size=batch_size)
                         if geometry.directional
                         else dict(order=opt['order5'], batch_size=batch_size, count=count)),
                        ('PT', _block_bb_pt_window if geometry.directional else _block_bb_pt,
                         dict(order=opt['order5'], batch_size=batch_size)
                         if geometry.directional
                         else dict(order=opt['order5'], batch_size=batch_size, count=count)),
                        ('P6', _block_bb_p6, dict(order=opt['order7'], batch_size=batch_size,
                                                  size=opt['p6_size'], seed=opt['seed']))]
            else:
                todo = []
            for name, fun, kw in todo:
                if name not in terms:
                    continue
                if subgroups is not None and name in ('PP', 'PPP'):
                    su, Wu = subgroups[irows]
                    spp, Wp = subgroups[icols]
                    add = fun(sp, su, spp, geometry,
                              **dict(kw, count=subcount.get(rows['nfields'], count)))
                    add = [[Wu @ np.asarray(b) @ Wp.T for b in row] for row in add]
                else:
                    add = fun(sp, rows, cols, geometry, **kw)
                if add is None:
                    continue
                res = add if res is None else [[a + b for a, b in zip(ra, rb)]
                                               for ra, rb in zip(res, add)]
            if res is None:
                res = [[jnp.zeros((rows['size'], cols['size'])) for _ in cols['ells']]
                       for _ in rows['ells']]
            for a, ia in enumerate(rows['index']):
                for b, ib in enumerate(cols['index']):
                    val = np.asarray(res[a][b])
                    blocks[ia][ib] = val
                    blocks[ib][ia] = val.T

    sizes = [None] * nitem
    for g in groups:
        for i in g['index']:
            sizes[i] = g['size']
    for i in range(nitem):
        for j in range(nitem):
            if blocks[i][j] is None:
                blocks[i][j] = np.zeros((sizes[i], sizes[j]))
    value = np.block(blocks)
    value = (value + value.T) / 2.
    out = CovarianceMatrix(observable=observable, value=value)
    return out


# --------------------------------------------------------------------------------------
# survey-window construction
#
# Carried over unchanged from the retired implementation (`_cov3_legacy.py`). These build the
# covariance window multipoles from catalogues -- catalogue -> mesh -> multipoles, with no
# covariance algebra in them -- so they are independent of how the blocks above are written, and
# they are what the cutsky path will consume once it is ported. `compute_fkp2_covariance_window`
# shares its name with the one in `cov2`, which is the exported one and a different object;
# neither of these is exported.
# --------------------------------------------------------------------------------------
def compute_fkp2_covariance_window(fkps, bin=None, los="local", fields=None, split=None,
                                   group_sizes=(2, 3, 4), max_total_size=6,
                                   group_pairs=None, **kwargs):
    r"""
    Compute the two-anchor covariance-window multipoles used by ``compute_QW_AB``.

    Unlike a plain power-spectrum window, this routine keeps the two window
    factors as grouped field labels.  This is required by the mixed covariance
    terms in the LaTeX formulae, e.g.

        Q_W^{(ac)(bde)}       for PB,
        Q_W^{(bcap)(bpcpa)}   for BB,
        Q_W^{(aap)(bcbpcp)}   for PT.

    Parameters
    ----------
    group_sizes : tuple, default=(2, 3, 4)
        Field multiplicities allowed for each of the two window factors.
    max_total_size : int, default=6
        Maximum total number of elementary FKP fields in the two factors.
    group_pairs : list, optional
        Explicit list of pairs of grouped field labels.  Each element is
        ``(fields1, fields2)``.  When omitted, all combinations with replacement
        of the requested ``group_sizes`` are generated, restricted by
        ``max_total_size``.

    Notes
    -----
    The returned ``ObservableTree`` stores grouped field labels
    ``((...), (...))``.  ``compute_QW_AB`` also understands older flat labels,
    but grouped labels avoid ambiguities such as distinguishing
    ``(ab,cde)`` from ``(abc,de)``.

    One window field is painted per tracer, ``W = alpha * n(x) * w(x)``, and a group is a product
    of those.  There is no separate shot-noise window ``S ~ nbar_ab w_a w_b``: the covariance
    blocks multiply the whole of ``P^(N)``, ``B^(N)``, ``T^(N)`` by kernels built from ``W``
    alone.  That is exact only where ``nbar_ab / (nbar_a nbar_b)`` is constant over the survey --
    approximation 2 in the module docstring.  ``cov2.compute_spectrum2_covariance`` keeps the
    ``WW``/``WS``/``SS`` split for the two-point case and is the template for restoring it here.
    """
    if not isinstance(fkps, (tuple, list)):
        fkps = [fkps]

    if fields is None:
        fields = list(range(len(fkps)))

    fkps = {field: fkp for field, fkp in zip(fields, fkps, strict=True)}
    field_values = tuple(fields)

    try:
        from .mesh2 import BinMesh2CorrelationPoles, compute_mesh2
    except ImportError:
        from .mesh import BinMesh2CorrelationPoles, compute_mesh2

    if bin is None:
        mattrs = next(iter(fkps.values())).attrs
        kw = {"edges": None, "ells": None, "basis": None, "klimit": None, "batch_size": None}
        kw = kw | get_smooth2_window_bin_attrs([0, 2, 4], ellsin=2)

        for name in kw:
            kw[name] = kwargs.pop(name, kw[name])

        edges = kw.pop("edges")
        if edges is None:
            edges = {}

        bin = BinMesh2CorrelationPoles(mattrs, edges=edges, **kw)

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    def get_W(fkp, mask=None):
        randoms = get_randoms(fkp)

        if mask is not None:
            randoms = randoms.clone(weights=randoms.weights * mask)

        alpha = fkp.data.weights.sum() / randoms.weights.sum() if isinstance(fkp, FKPField) else 1.0
        mesh = randoms.paint(**kwargs, out="real")

        return alpha * mesh / mesh.cellsize.prod()

    def canonical_group(group):
        return tuple(sorted(tuple(group)))

    if group_pairs is None:
        groups_by_size = {
            size: tuple(itertools.combinations_with_replacement(field_values, size))
            for size in group_sizes
        }
        group_pairs = []
        for size1 in group_sizes:
            for size2 in group_sizes:
                if size1 > size2 or size1 + size2 > max_total_size:
                    continue
                for group1 in groups_by_size[size1]:
                    for group2 in groups_by_size[size2]:
                        group_pairs.append((group1, group2))
    else:
        group_pairs = [(tuple(group1), tuple(group2)) for group1, group2 in group_pairs]

    splits = None
    if split is not None:
        if not isinstance(split, list):
            split = [split]
        splits = {field: split for field, split in zip(fields, split, strict=True)}

    windows = {}
    compute_mesh2_window = jax.jit(compute_mesh2, static_argnames=["los"])

    for group1, group2 in group_pairs:
        group1, group2 = canonical_group(group1), canonical_group(group2)
        wfield = (group1, group2)

        if wfield in windows:
            continue

        flat_field = group1 + group2
        _fkps = [fkps[field] for field in flat_field]
        masks = [None] * len(_fkps)

        if split is not None:
            seed = list({field: splits[field] for field in flat_field}.values())
            masks = split_particles(*[get_randoms(fkp) for fkp in _fkps], seed=seed, fields=list(flat_field), return_masks=True)

        n1 = len(group1)
        WA = get_W(_fkps[0], mask=masks[0])
        for fkp, mask in zip(_fkps[1:n1], masks[1:n1], strict=True):
            WA = WA * get_W(fkp, mask=mask)

        WB = get_W(_fkps[n1], mask=masks[n1])
        for fkp, mask in zip(_fkps[n1 + 1:], masks[n1 + 1:], strict=True):
            WB = WB * get_W(fkp, mask=mask)

        # Normalization: cross-pair group1's anchor field (fkps[0]) with
        # group1's remaining n1-1 fields with the last fields of group2, not
        # the naive product of each group's own fields (WA.sum()*WB.sum()).
        Ws = [get_W(fkp, mask=mask) for fkp, mask in zip(_fkps, masks, strict=True)]
        normA = Ws[0]
        for w in Ws[n1:2 * n1 - 1]:
            normA = normA * w
        normB = None
        for w in Ws[1:n1] + Ws[2 * n1 - 1:]:
            normB = w if normB is None else normB * w
        if normB is None:
            normB = jnp.ones_like(normA)
        norm = normA.sum() * normB.sum()
        update = dict(norm=[norm * jnp.ones_like(bin.xavg)] * len(bin.ells))
        windows[wfield] = compute_mesh2_window(WA, WB, bin=bin, los=los).clone(**update)
        del WA, WB

    return ObservableTree(
        list(windows.values()),
        fields1=[field_groups[0] for field_groups in windows],
        fields2=[field_groups[1] for field_groups in windows],
    )


def compute_fkp3_covariance_window(fkps, bin=None, los="local", fields=None, split=None, **kwargs):
    r"""
    Compute the WWW 3-point covariance-window multipoles

        Q^{ABC}_{W,lambda1 lambda2 Lambda}(s1, s2)

    in the Sugiyama / TripoSH basis. Here A, B, C are field-pair labels, e.g.

        A = (a, a')
        B = (b, b')
        C = (c, c')

    and the configuration-space window is schematically

        Q_W^{ABC}(s1, s2) ~ < W_A(x) W_B(x + s1) W_C(x + s2) >.

    This function only computes the WWW piece.
    """
    if not isinstance(fkps, (tuple, list)):
        fkps = [fkps]

    if fields is None:
        fields = list(range(len(fkps)))

    fkps = {field: fkp for field, fkp in zip(fields, fkps, strict=True)}

    if bin is None:
        mattrs = next(iter(fkps.values())).attrs
        kw = {"edges": None, "ells": None, "basis": 'sugiyama', "klimit": None, "batch_size": None, 'buffer_size': 0}
        kw = kw | get_smooth3_window_bin_attrs([(0, 0, 0), (2, 0, 2)], ellsin=2)

        for name in kw:
            kw[name] = kwargs.pop(name, kw[name])

        edges = kw.pop("edges")
        if edges is None:
            edges = {}

        bin = BinMesh3CorrelationPoles(mattrs, edges=edges, **kw)

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    def get_W(fkp, mask=None):
        randoms = get_randoms(fkp)

        if mask is not None:
            randoms = randoms.clone(weights=randoms.weights * mask)

        alpha = fkp.data.weights.sum() / randoms.weights.sum() if isinstance(fkp, FKPField) else 1.0
        mesh = randoms.paint(**kwargs, out="real")

        return alpha * mesh / mesh.cellsize.prod()

    splits = None
    if split is not None:
        if not isinstance(split, list):
            split = [split]
        splits = {field: split for field, split in zip(fields, split, strict=True)}

    pairs = tuple(itertools.combinations_with_replacement(tuple(fields), 2))
    # Q_W^{ABC}(s1, s2) ~ <W_A(x) W_B(x+s1) W_C(x+s2)>: positions 1 and 2 (the
    # two "arm" separations s1, s2) are interchangeable (Q_W^{ABC} = Q_W^{BAC}),
    # but position 3 is not (Q_W^{ABC} != Q_W^{ACB} in general) -- so only the
    # first two groups are drawn from an unordered (sorted) pair-of-pairs;
    # the third is chosen independently, not from a fully symmetric 3-way
    # combinations_with_replacement.
    pairs_ab = tuple(itertools.combinations_with_replacement(pairs, 2))
    triplets = tuple((pab[0], pab[1], pc) for pab in pairs_ab for pc in pairs)
    windows = {}
    compute_mesh3_window = jax.jit(compute_mesh3, static_argnames=["los"])

    for triple in triplets:
        wfield = tuple(tuple(sorted(pair)) for pair in triple)
        flat_field = sum(wfield, start=tuple())

        if wfield in windows:
            continue

        _fkps = [fkps[field] for field in flat_field]
        masks = [None] * len(_fkps)

        if split is not None:
            seed = list({field: splits[field] for field in flat_field}.values())
            masks = split_particles(*[get_randoms(fkp) for fkp in _fkps], seed=seed, fields=list(flat_field), return_masks=True)

        Ws = [get_W(_fkps[i], mask=masks[i]) for i in range(6)]
        WA, WB, WC = Ws[0] * Ws[1], Ws[2] * Ws[3], Ws[4] * Ws[5]

        # Normalization: the product of the two bispectrum-estimator
        # normalizations, int(n_a n_b n_c) x int(n_a' n_b' n_c') (and not the
        # product of the three pair integrals int(n n') as for the 2-point
        # covariance window): the window's periodic limit is then
        # (2 pi)^6 delta_D delta_D / V, matching the Sugiyama PPP covariance
        # with no extra volume factor in the assembly. The 1/cellsize makes
        # the norm's total cell-volume power (one per anchor, three) match
        # the numerator (compute_mesh3) conventions: two mesh sums only
        # carry two.
        norm = (Ws[0] * Ws[2] * Ws[4]).sum() * (Ws[1] * Ws[3] * Ws[5]).sum() / WA.cellsize.prod()
        del Ws
        update = dict(norm=[norm * jnp.ones_like(bin.xavg[..., 0])] * len(bin.ells))
        windows[wfield] = compute_mesh3_window(WA, WB, WC, bin=bin, los=los).clone(**update)
        del WA, WB, WC

    return ObservableTree(
        list(windows.values()),
        fields1=[field_groups[0] for field_groups in windows],
        fields2=[field_groups[1] for field_groups in windows],
        fields3=[field_groups[2] for field_groups in windows],
    )



# ======================================================================================
# survey window kernels
#
# Ported unchanged from the retired implementation (`_cov3_legacy.py`), whose algebra is
# `desi-cov3-notes/covariance.tex`, section "Window functions". Nothing above calls these yet: the blocks
# still go through `BoxGeometry`, and wiring a `SurveyGeometry` onto them is what remains. They
# are here rather than left behind because they are the part of the cutsky path that is
# independent of how the blocks are written -- catalogue windows in, Q_W multipole tables out.
#
#   Q_W^{A,B}(k - k') = sum_{l1 l2} Q_{l1 l2}(k, k') L_{l1}(khat . n) L_{l2}(khat' . n)
#
# with Q_{l1 l2} a double spherical-Bessel transform of the stored window multipoles weighted by
# a 3j^2, and the three-point analogue carrying a Wigner 9j coupling in the TripoSH basis.
# ======================================================================================
@functools.lru_cache(maxsize=None)
def get_sugiyama_covariance_window_convolution_coeffs(ell, ellin):
    r"""Window-multipole coefficients for the *covariance* 4-point kernel.

    ``ell`` = (L1, L2, J) indexes the unprimed-side S-basis channel and
    ``ellin`` = (L1', L2', J') the primed-side one (both z3, M = 0). Returns
    the list of (q, coeff) such that the 4-point angular window kernel is

    .. math::
        Q_W(k_1, k_1', k_2, k_2')
        = \sum_{\ell,\ell'} \Big[\sum_q c_q\, \mathrm{Hankel}_{L_1 L_1' L_2 L_2'}[Q_{W,q}]\Big]
          S_\ell(\hat k_1, \hat k_2)\, S_{\ell'}(\hat k_1', \hat k_2'),

    following the :math:`\mathcal C^{\lambda_1\lambda_2\Lambda}_{L_1L_1'L_2L_2'}`
    kernel of ``_cov3_math.tex``, keeping only its N = 0 term (the N != 0
    azimuthal channels vanish identically under the estimators' independent
    per-side orientation averages). This differs from
    :func:`get_sugiyama_window_convolution_coeffs` (the *mean* bispectrum
    window convolution, eq. 63 of arXiv:1803.02132): here the monopole
    window feeds each diagonal channel with the Parseval weight
    :math:`(2L_1+1)(2L_2+1)(2J+1) H_{L_1L_2J}^2 = 1 / \| S_\ell \|^2`.

    The relative phase :math:`(-i)^{L_1+L_2} i^{L_1'+L_2'}` is real
    (:math:`\pm 1`) for every allowed q: the two triangle conditions with
    even-sum q force :math:`(L_1'-L_1)+(L_2'-L_2)` even. It is included here
    because the covariance Hankel matrices drop the transforms'
    :math:`i^\ell` prefactors. Normalization is anchored so that the
    ((0,0,0), (0,0,0)) channel has coeff((0,0,0)) = 1, matching the box
    limit.
    """
    L1, L2, J = ell
    L1p, L2p, Jp = ellin
    HJ = wigner_3j(L1, L2, J, 0, 0, 0)
    HJp = wigner_3j(L1p, L2p, Jp, 0, 0, 0)
    if abs(HJ) < 1e-12 or abs(HJp) < 1e-12:
        return []
    coeffs = []
    for q in itertools.product(range(L1 + L1p + 1), range(L2 + L2p + 1), range(abs(J - Jp), J + Jp + 1)):
        if sum(q) % 2 or q[2] % 2:
            continue
        Hq = wigner_3j(*q, 0, 0, 0)
        if abs(Hq) < 1e-12:
            continue
        coeff = (2 * L1 + 1) * (2 * L1p + 1) * (2 * L2 + 1) * (2 * L2p + 1)
        coeff *= wigner_3j(q[0], L1, L1p, 0, 0, 0) * wigner_3j(q[1], L2, L2p, 0, 0, 0) / Hq
        coeff *= (2 * J + 1) * (2 * Jp + 1) * HJ * HJp
        coeff *= wigner_9j(L1, L2, J, L1p, L2p, Jp, *q)
        coeff *= wigner_3j(J, Jp, q[2], 0, 0, 0)
        if abs(coeff) < 1e-10:
            continue
        coeff *= (-1) ** (((L1p - L1 + L2p - L2) // 2) % 2)
        coeffs.append((tuple(q), coeff))
    return coeffs


def _hankel_matrix(s, ell, cache=None):
    """Explicit ``(n_k, n_s)`` matrix of ``CorrelationToSpectrum(s, ell=ell)``,
    extracted via ``jax.jacfwd`` (same technique ``CorrelationToSpectrum``
    uses, including its prefactor/postfactor handling: the Jacobian is taken
    of the *raw* FFTlog transform with ``ignore_prepostfactor=True``, and the
    physical normalization is re-applied explicitly afterward as a
    per-row scale, since it factorizes as ``scale(k) * scale(k')`` and must
    be split symmetrically between the two sides of a bilinear sandwich).

    Unlike composing two independently-built FFTlog round-trip objects (one
    ``SpectrumToCorrelation``, one ``CorrelationToSpectrum``, which are only
    approximate *inverses* of each other, not *transposes*), sandwiching a
    window between two explicit forward-transform matrices built this way --
    one per side (``ell``/``ellin``) -- gives a bilinear form that is
    symmetric under side-swap by construction.

    Returns ``(k, matrix)``.
    """
    from .fftlog import CorrelationToSpectrum

    cache = {} if cache is None else cache
    key = (tuple(np.ravel(s)), ell)
    if key not in cache:
        fftlog = CorrelationToSpectrum(s=s, ell=ell, lowring=False, minfolds=0, check_level=1).fftlog

        def fwd(fun):
            return fftlog(fun, extrap=False, ignore_prepostfactor=True)[1]

        raw_matrix = jax.jacfwd(fwd)(jnp.zeros_like(jnp.asarray(s)))
        k = fftlog.y
        dlnk = jnp.diff(jnp.log(k)).mean()
        scale = jnp.sqrt(2 * np.pi**2 / dlnk) / k**1.5
        cache[key] = (k, scale[:, None] * raw_matrix)
    return cache[key]


#: Channel pairs already warned about, so a block sweeping many of them says it once each.
_WARNED_WINDOW3_ELLS = set()


def _warn_missing_window3_ells(ell, ellin, missing, fraction):
    """Say which three-anchor window multipoles a channel wanted and did not get.

    Silence here is how a bispectrum covariance ends up with a validating ``sigma(B000)`` and a
    ``sigma(B202)`` 10-36% high: the isotropic channel is the one a monopole-only window serves
    exactly, so the check most people run is the one that cannot see the problem.
    """
    key = (tuple(ell), tuple(ellin))
    if key in _WARNED_WINDOW3_ELLS:
        return
    _WARNED_WINDOW3_ELLS.add(key)
    import warnings
    warnings.warn(
        f'the three-anchor window carries none of {sorted(missing)}, which channel '
        f'{tuple(ell)} x {tuple(ellin)} needs: {fraction:.0%} of its coefficient weight is '
        'being set to zero. Rebuild the window with those multipoles, or accept the truncation '
        'knowingly -- it biases the anisotropic channels and leaves (0, 0, 0) exact, so a '
        'B000-only validation will not show it.', stacklevel=2)


def compute_spectrum3_covariance_window_block(window3, kedges, kpedges,
                                              ell, ellin,
                                              fields1=None, fields2=None, fields3=None,
                                              cache=None, batch_size=None,
                                              paxes=(0, 1), kp_points=None, kp_measure=None):
    """
    Compute one smooth covariance-window block W_{ell,ellin}(k1,k2;k1',k2').

    The unprimed radial axes always carry the (binned) k1, k2 legs. The
    primed radial axes are selected by ``paxes``: entry 0 or 1 rebins over
    the corresponding ``kpedges`` column (which primed *binned* leg pairs
    with each separation axis depends on the Wick permutation), while entry
    2 point-evaluates that axis at the primed *closure*-leg magnitudes
    ``kp_points`` (shape ``(nnodes, nbinsp)``, node- and bin-dependent, as
    in the box path's exact substitution) -- or, when ``kp_measure`` (a
    callable ``measure(fk) -> (nnodes * nbinsp, n_fk)`` cell-average
    weight matrix, see ``_closure_measure``) is given,
    *cell-averages* the transform over each node's own closure sweep, so
    window structure narrower than the node spacing cannot slip between nodes.
    With a points axis the result has shape ``(n, npx, nnodes)`` instead of
    ``(n, npx)``.
    """
    if cache is None:
        cache = {}
    rebin_cache = cache.setdefault("rebin_matrix", [])
    hankel_cache = cache.setdefault("hankel_matrix", {})

    def get_window_field(window, fields1, fields2, fields3):
        if fields1 is None:
            return window
        return window.get(fields1=fields1, fields2=fields2, fields3=fields3)

    def get_w_rect(q):
        transpose = False
        if q not in window3.ells:
            qswap = q[1::-1] + q[2:]
            if qswap in window3.ells:
                q = qswap
                transpose = True
            else:
                # Absent multipoles are dropped, which is a modelling choice and not always the
                # wrong one -- the full set is 29 and an expensive build. It must not be a silent
                # one: `(0,0,0)` is the only channel a monopole-only window serves exactly, and
                # the anisotropic channels lose 36-55% of their coefficient weight this way.
                _missing.add(tuple(q))
                return jnp.zeros(())
        value = window3.get(q).value().real

        if transpose:
            value = jnp.swapaxes(value, 0, 1)

        return value

    def normalize_edges(edges):
        edges = np.asarray(edges)

        if edges.ndim == 1:
            edges = np.column_stack([edges[:-1], edges[1:]])

        return edges

    def unravel_edges(edges):
        """
        Convert input edge specification into a paired-bin array of shape
        (nbins, 2, 2): one (k1, k2) edge pair per bispectrum bin. This
        supports both genuinely paired binnings (e.g. the sugiyama-diagonal
        basis, where bins are a *list* of (k1, k2) pairs, rather than a product
        grid -- reinterpreting them as a sqrt(n) x sqrt(n) product grid
        scrambles the bins) and product grids (tuple of 1D edges, or an
        explicit (N1, N2, 2, 2) array, flattened row-major).
        """
        if isinstance(edges, (tuple, list)) and len(edges) == 2:
            k1edges, k2edges = map(normalize_edges, edges)
            out = np.empty((len(k1edges), len(k2edges), 2, 2), dtype=float)
            out[:, :, 0, :] = k1edges[:, None, :]
            out[:, :, 1, :] = k2edges[None, :, :]
            return out.reshape(-1, 2, 2)

        edges = np.asarray(edges)

        if edges.ndim == 4 and edges.shape[-2:] == (2, 2):
            return edges.reshape(-1, 2, 2)

        if edges.ndim == 3 and edges.shape[-2:] == (2, 2):
            return edges

        raise ValueError("Invalid edge specification.")

    edges = unravel_edges(kedges)
    edgesp = unravel_edges(kpedges)

    n, npx = edges.shape[0], edgesp.shape[0]

    window3 = get_window_field(window3, fields1, fields2, fields3)

    # Covariance-kernel coefficients (the C^{lambda}_{L1L1'L2L2'} pairing of
    # _cov3_math.tex, N = 0) -- NOT the mean-convolution coefficients of
    # get_sugiyama_window_convolution_coeffs: the monopole window must feed
    # each diagonal (ell, ell) channel with the Parseval weight 1/||S_ell||^2.
    wcoeffs = get_sugiyama_covariance_window_convolution_coeffs(ell, ellin)

    if not wcoeffs:
        return np.zeros((n, npx))

    _missing = set()
    Qs = sum(coeff * get_w_rect(q) for q, coeff in wcoeffs)
    if _missing:
        total = sum(abs(coeff) for _, coeff in wcoeffs)
        lost = sum(abs(coeff) for q, coeff in wcoeffs if tuple(q) in _missing)
        _warn_missing_window3_ells(ell, ellin, _missing, lost / total if total else 0.)

    if np.ndim(Qs) == 0:
        return np.zeros((n, npx))

    s = tuple(next(iter(window3)).coords().values())

    k1_fftlog, H1 = _hankel_matrix(s[0], ell[0], cache=hankel_cache)
    k2_fftlog, H2 = _hankel_matrix(s[1], ell[1], cache=hankel_cache)
    k1p_fftlog, H1p = _hankel_matrix(s[0], ellin[0], cache=hankel_cache)
    k2p_fftlog, H2p = _hankel_matrix(s[1], ellin[1], cache=hankel_cache)

    interp_order = 3
    Mk1 = matrix_rebin(edges[:, 0, :], k1_fftlog, wt=k1_fftlog**2, interp_order=interp_order, cache=rebin_cache)
    Mk2 = matrix_rebin(edges[:, 1, :], k2_fftlog, wt=k2_fftlog**2, interp_order=interp_order, cache=rebin_cache)

    # Fuse the bin-rebinning into the forward-transform matrices before
    # contracting against Qs, so the (n_s1, n_s2) ~ 1000x1000 grid is never
    # expanded into a dense 4D tensor. Rows/columns are paired bins: bin a
    # applies its own k1-rebin on the s1 axis and its own k2-rebin on the s2
    # axis.
    R1, R2 = Mk1 @ H1, Mk2 @ H2

    def primed_R(axis, kp_fftlog, Hp):
        spec = paxes[axis]
        if spec == 2:
            if kp_measure is not None:
                # Closure leg, cell-averaged with the exact phi measure over
                # each node's own closure sweep instead of sampled at the
                # node magnitude (see _closure_measure).
                avg = jnp.asarray(kp_measure(np.asarray(kp_fftlog))) @ Hp
                return jnp.reshape(avg, (np.shape(kp_points)[0], npx, -1))  # (n_nodes, npx, n_s)
            # Closure leg: point-evaluate the transform at the node- and
            # bin-dependent magnitudes (n_nodes, npx), as in the box path.
            pts = jnp.reshape(jnp.asarray(kp_points), (-1,))
            # Clamp to the FFTlog grid: k3' -> 0 at antiparallel quadrature
            # nodes, and polynomial spline extrapolation there is unsafe.
            pts = jnp.clip(pts, jnp.min(kp_fftlog), jnp.max(kp_fftlog))
            interp = matrix_spline_interp(kp_fftlog, pts, interp_order, cache=hankel_cache)
            return jnp.reshape(interp @ Hp, (np.shape(kp_points)[0], npx, -1))  # (n_nodes, npx, n_s)
        Mkp = matrix_rebin(edgesp[:, spec, :], kp_fftlog, wt=kp_fftlog**2, interp_order=interp_order, cache=rebin_cache)
        return Mkp @ Hp

    R1p = primed_R(0, k1p_fftlog, H1p)
    R2p = primed_R(1, k2p_fftlog, H2p)

    if np.ndim(R1p) == 3:
        return jnp.einsum('as,at,st,nbs,bt->abn', R1, R2, Qs, R1p, R2p, optimize=True)
    if np.ndim(R2p) == 3:
        return jnp.einsum('as,at,st,bs,nbt->abn', R1, R2, Qs, R1p, R2p, optimize=True)
    return jnp.einsum('as,at,st,bs,bt->ab', R1, R2, Qs, R1p, R2p, optimize=True)



def matrix_spline_interp(xt, xo, interp_order, cache=None):
    """Like ``cov2.matrix_spline_interp(xt, xo, ...)``, but ``xo`` may be a
    traced (e.g. vmapped) array; ``xt`` must be concrete.

    If ``interpax`` is installed, this runs natively in JAX: no host
    callback, transparently batched under ``vmap`` (no special
    ``vmap_method`` needed), and differentiable. ``interpax``'s
    ``method='cubic2'`` (C2, natural-spline-like) is used because it matches
    scipy's ``make_interp_spline(k=3)`` (used by ``matrix_rebin`` elsewhere
    in this module) to ~1e-7 relative -- its plain ``'cubic'`` (C1, local
    splines) is a *different* scheme and differs by ~10%, so is not a
    drop-in match.

    Otherwise, falls back to scipy via ``jax.pure_callback``. The
    cubic-spline basis only depends on ``xt`` (not ``xo``), so it is built
    once and cached -- otherwise every vmapped call would refit it from
    scratch on a ``len(xt)``-sized identity matrix.
    """
    xt = jnp.asarray(xt)
    xo = jnp.asarray(xo)

    try:
        import interpax
    except ImportError:
        interpax = None

    if interpax is not None:
        method = 'linear' if interp_order == 1 else 'cubic2'
        return interpax.interp1d(xo, xt, jnp.eye(xt.shape[-1], dtype=xo.dtype), method=method)

    from scipy.interpolate import make_interp_spline

    spline_cache = {} if cache is None else cache
    key = (tuple(np.ravel(xt)), interp_order)
    if key not in spline_cache:
        xt_arr = np.asarray(xt, dtype=float)
        spline_cache[key] = make_interp_spline(xt_arr, np.eye(len(xt_arr)), k=interp_order, axis=0)
    spl = spline_cache[key]

    def host_fn(xo):
        return jnp.asarray(spl(np.asarray(xo)))

    out_shape = jax.ShapeDtypeStruct((xo.shape[-1], len(xt)), xo.dtype)
    # 'broadcast_all', not 'sequential': scipy's spline evaluation already
    # broadcasts over an extra leading batch axis, so under vmap (e.g. over
    # quadrature points) this calls the host once for the whole batch
    # instead of once per point -- ~50x faster in practice, same result.
    return jax.pure_callback(host_fn, out_shape, xo, vmap_method='broadcast_all')


def _cosine_cell_edges(order):
    r"""Cells tiling :math:`[-1, 1]` with widths *proportional to the Gauss-Legendre weights*.

    This is what makes a cell average exact rather than merely better. Writing the axis integral
    as :math:`\int_{-1}^{1} (dx/2) f(x) \simeq \sum_i w_i f(x_i)` and replacing :math:`f(x_i)` by
    its average over a cell of width :math:`\Delta_i`, the sum is
    :math:`\sum_i (w_i/\Delta_i)\int_{{\rm cell}\,i} f\,dx`; choosing
    :math:`\Delta_i = 2 w_i` makes that :math:`\tfrac12\int_{-1}^{1} f\,dx` identically, for any
    :math:`f`, however narrow. Cells built from node midpoints instead would leave a
    :math:`w_i` versus :math:`\Delta_i/2` mismatch of tens of per cent per cell, which for a
    window narrower than one cell is an O(1) error.

    Returns the ``order + 1`` edges, so cell ``i`` is ``[edges[i], edges[i + 1]]`` and contains
    node ``i`` (Gauss-Legendre nodes crowd the endpoints, where the weights are small, so the
    cells track the nodes).
    """
    _, weights = np.polynomial.legendre.leggauss(order)
    cumulative = np.concatenate([[0.], np.cumsum(weights / 2.)])
    return -1. + 2. * cumulative / cumulative[-1]


def _closure_x_measure_matrix(fk, lo, hi, ksq, k1k2, chunk=2048):
    r"""Cell-average weights for a closure leg swept in :math:`x = \hat k_1 \cdot \hat k_2`.

A row-normalized ``(n_rows, n_fk)`` matrix such that ``M @ table`` is the average of the
    piecewise-linear ``table`` over each row's own sweep. Here
    :math:`k_3 = \sqrt{k_1^2 + k_2^2 + 2 k_1 k_2 x}` with :math:`x` a quadrature variable in its
    own right, so the measure of a :math:`k_3` sub-segment is uniform in :math:`x`,
    :math:`\Delta x = (q_{\rm hi}^2 - q_{\rm lo}^2) / (2 k_1 k_2)`. (:mod:`jaxpower._cov3_legacy`
    parametrises the triangle by :math:`(\mu_1, \mu_2, \phi_2)` instead, where :math:`k_3` depends
    on all three axes and the measure carries an :math:`\arccos` Jacobian; its
    ``_closure_measure_matrix`` is that version, and it does not apply to this parametrisation.
    It was ported here, never wired to anything, and has been removed.)

    Why it is needed: the tie is narrower in :math:`k` than a cell is wide, so point-sampling
    :math:`Q_{\mathcal W}` at the node either hits the window or misses it, and refining the
    quadrature makes the miss *more* likely -- the block then decays monotonically with order
    instead of converging.

    Returns a dense ``(n_rows, n_fk)`` numpy array.
    """
    fk = np.asarray(fk)
    nfk = len(fk)
    lo, hi, ksq, k1k2 = map(np.asarray, (lo, hi, ksq, k1k2))
    n = len(lo)
    out = np.zeros((n, nfk))

    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        # Segment [fk_j, fk_j+1] clipped to the row's sweep [lo, hi].
        seg_lo = np.maximum(fk[None, :-1], lo[sl, None])
        seg_hi = np.minimum(fk[None, 1:], hi[sl, None])
        valid = seg_hi > seg_lo
        seg_lo = np.where(valid, seg_lo, 0.)
        seg_hi = np.where(valid, seg_hi, 0.)
        dx = (seg_hi**2 - seg_lo**2) / (2. * np.maximum(k1k2[sl, None], 1e-300)) * valid
        out[sl, :-1] += 0.5 * dx
        out[sl, 1:] += 0.5 * dx
        rowsum = out[sl].sum(axis=1)
        good = rowsum > 1e-12
        rows = out[sl]
        rows[good] /= rowsum[good, None]
        # A row whose sweep falls entirely outside the table contributes nothing; leaving it zero
        # is the right answer here, unlike the phi version's turning-point rows.
        out[sl] = rows
    return out


def _closure_measure(k1, k2, order, x_index, closure_min=None):
    r"""A ``measure(fk) -> (n_nodes * nbins, n_fk)`` callable for one triangle's closure leg.

    ``x_index`` gives each quadrature node's index along the :math:`x` axis, so the same helper
    serves every node layout: the caller knows which axis of its ``kinds`` string is the one
    :func:`_triangle` receives as ``x``. Rows are raveled ``(node, bin)`` in C order, matching
    ``norms[2].ravel()``.

    ``closure_min`` must match whatever floor the caller gave :func:`_triangle_nodes`, because
    the cells have to tile the axis the nodes actually live on. Floored, that axis is
    ``[x_lo, 1]`` per bin rather than ``[-1, 1]`` shared, so the cell edges are mapped bin by bin.
    Their *weights* are unaffected: the map is affine and the returned matrix is row-normalised,
    so the Jacobian cancels row by row.
    """
    edges = _cosine_cell_edges(order)
    k1, k2 = np.asarray(k1), np.asarray(k2)
    ksq = (k1**2 + k2**2)[None, :]
    k1k2 = (k1 * k2)[None, :]
    if closure_min is None:
        cells = np.broadcast_to(edges[:, None], (len(edges), len(k1)))
    else:
        floor = np.clip((closure_min**2 - ksq[0]) / np.maximum(2. * k1k2[0], 1e-300), -1., 1.)
        cells = floor[None, :] + (1. - floor)[None, :] * ((edges[:, None] + 1.) / 2.)
    x_lo, x_hi = cells[np.asarray(x_index)], cells[np.asarray(x_index) + 1]
    lo = np.sqrt(np.maximum(ksq + 2. * k1k2 * x_lo, 0.))
    hi = np.sqrt(np.maximum(ksq + 2. * k1k2 * x_hi, 0.))
    args = tuple(np.broadcast_to(a, lo.shape).reshape(-1)
                 for a in (lo, hi, ksq, k1k2))
    cache = {}

    def measure(fk):
        fk = np.asarray(fk)
        key = (fk.shape[0], float(fk[0]), float(fk[-1]))
        if key not in cache:
            cache[key] = _closure_x_measure_matrix(fk, *args)
        return cache[key]

    return measure


class _QWSpectrum(object):
    """The (k1, k2) covariance-window table of one (ell1, ell2) pair, kept lazy.

    Every consumer of this table either sandwiches it between two matrices
    (``A @ T @ B.T`` -- bin rebinning, closure-cell measures) or interpolates it at traced
    points. The first form never needs the dense ``T``: both FFTlog transforms are linear, so
    the left matrix can be pushed through them (:meth:`Correlation2Spectrum.contracted`),
    touching only ``(nrow, n_s)`` arrays. At the ``n_s = 8192`` the window must be resampled to
    for the FFTlog to be accurate, dense is 537 MB against 27 MB contracted -- and it is what
    made the Zel'dovich P+B covariance OOM an 80 GB device. Only the interpax path, which
    genuinely interpolates the table, materializes ``.dense`` (once, cached).
    """
    def __init__(self, fftlog, w):
        self._fftlog, self._w, self._dense = fftlog, w, None

    @property
    def k(self):
        return self._fftlog.k

    @property
    def dense(self):
        if self._dense is None:
            self._dense = self._fftlog(self._w)[1]
        return self._dense

    def contract(self, A, B):
        if self._dense is not None:  # already paid for; reuse
            return jnp.asarray(A) @ self._dense @ jnp.asarray(B).T
        # The contraction is a win only when the left operand has few rows: it costs
        # O(nrow * n_s log n_s) against the dense table's one-off O(n_s^2 log n_s) plus a
        # matmul. Binned rows (~400) against a resampled n_s = 8192 grid: contract. Closure
        # measures (nside * nbins rows, ~9k at q = 6 and ~21k at q = 8) against the stored
        # n_s = 1024 grid: the dense table is far cheaper and only 8 MB.
        nrow = np.shape(A)[0]
        if int(os.environ.get('COV3_WINDOW_DENSE', '0')) or nrow > len(self.k) // 4:
            return jnp.asarray(A) @ self.dense @ jnp.asarray(B).T
        return self._fftlog.contracted(self._w, A, B)


def compute_spectrum2_covariance_window_block(window2, k1edges, k2edges, ell1, ell2,
                                              fields1=None, fields2=None, cache=None,
                                              k1_is_points=False, k2_is_points=False,
                                              k1_measure=None, k2_measure=None):
    r"""Return one :math:`(k_1, k_2)` covariance-window block.

    If ``k1_is_points`` (``k2_is_points``) is True, ``k1edges`` (``k2edges``) is
    treated as literal k-values (e.g. a derived/closure leg with no native bin
    edges) at which Q_W is interpolated, rather than bin edges Q_W is rebinned
    into. If additionally ``k1_measure`` (``k2_measure``) is given -- a
    callable ``measure(fk) -> (n, n_fk)`` row-normalized weight matrix (see
    ``_closure_measure``) -- Q_W is *cell-averaged* over each node's own
    sweep instead of point-sampled, so window structure
    narrower than the node spacing cannot slip between nodes.
    """
    if cache is None:
        cache = {}
    rebin_cache = cache.setdefault("rebin_matrix", [])
    spectrum_cache = cache.setdefault("QW_spectrum", {})
    spline_cache = cache.setdefault("spline_basis", {})

    def normalize_edges(edges):
        edges = np.asarray(edges)
        if edges.ndim == 1:
            return np.column_stack([edges[:-1], edges[1:]])
        if edges.ndim >= 2 and edges.shape[-1] == 2:
            return edges.reshape(-1, 2)
        raise ValueError("k edges must be 1D bin edges or explicit per-bin edges with last dimension 2.")

    def get_window_field(window, fields1, fields2):
        if fields1 is None:
            return window
        if isinstance(window, tuple):
            # Symmetrize Q_W^{A,B} with the second window evaluated as Q_W^{B,A}.
            w1 = get_window_field(window[0], fields1, fields2)
            w2 = get_window_field(window[1], fields2, fields1)
            return w1.clone(value=(w1.value() + w2.value()) / 2.)
        return window.get(fields1=fields1, fields2=fields2)

    if k1_is_points:
        # jnp, not np: k1edges may be a traced (e.g. vmapped) array here.
        k1points = jnp.ravel(jnp.asarray(k1edges))
    else:
        k1edges = normalize_edges(k1edges)
    if k2_is_points:
        k2points = jnp.ravel(jnp.asarray(k2edges))
    else:
        k2edges = normalize_edges(k2edges)

    # The FFTlog-transformed window grid only depends on (window2, fields,
    # ell1, ell2) -- not on k1edges/k2edges -- so cache it by that key alone.
    # This keeps it cached even when k1edges/k2edges are traced (e.g. a
    # closure leg varying under jax.vmap), where the outer block_cache in
    # compute_QW_AB can't be used.
    # Key by the id(s) of the *underlying* window object(s): callers like
    # compute_QW_AB pass a freshly-built (window2, window2) symmetrization
    # tuple each call, whose own id changes every time -- keying on that
    # would defeat the cache entirely, re-running the FFTlog + jacfwd
    # construction and appending another dense spectrum grid to the cache
    # for every W2 call (tens of GB leaked over a 6D-quadrature run).
    window2_ids = tuple(id(w) for w in window2) if isinstance(window2, tuple) else id(window2)
    spectrum_key = (window2_ids, fields1, fields2, ell1, ell2)
    if spectrum_key not in spectrum_cache:
        window2_field = get_window_field(window2, fields1, fields2) if fields1 is not None else window2
        w = sum(legendre_product(ell1, ell2, q) * window2_field.get(q).value().real if q in window2_field.ells else jnp.zeros(())
                for q in range(abs(ell1 - ell2), ell1 + ell2 + 1))
        if w.size <= 1:
            spectrum_cache[spectrum_key] = None
        else:
            tmpw = next(iter(window2_field))
            s = tmpw.coords('s')
            fftlog = Correlation2Spectrum(s, (ell1, ell2), check_level=1)
            spectrum_cache[spectrum_key] = (fftlog.k, _QWSpectrum(fftlog, w))

    cached = spectrum_cache[spectrum_key]
    if cached is None:
        n1 = len(k1points) if k1_is_points else len(k1edges)
        n2 = len(k2points) if k2_is_points else len(k2edges)
        return np.zeros((n1, n2))
    fk, spectrum = cached

    interp_order = 3

    if k1_measure is not None or k2_measure is not None:
        # Cell-averaged closure leg(s): average the (concrete, cached)
        # spectrum table with the exact closure phi measure over each node's
        # own cell instead of sampling it at the node value -- see the
        # docstring. The other axis is either rebinned (concrete bin edges)
        # or cell-averaged too.
        if k1_measure is not None and k2_measure is not None:
            W1 = jnp.asarray(k1_measure(fk))
            W2 = jnp.asarray(k2_measure(fk))
            return spectrum.contract(W1, W2)
        if k1_measure is not None:
            My = matrix_rebin(k2edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
            return spectrum.contract(jnp.asarray(k1_measure(fk)), My)
        Mx = matrix_rebin(k1edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
        return spectrum.contract(Mx, jnp.asarray(k2_measure(fk)))

    try:
        import interpax
    except ImportError:
        interpax = None

    if interpax is None or not (k1_is_points or k2_is_points):
        # Both legs binned (concrete): the plain matrix sandwich is cheap and
        # cached upstream by compute_QW_AB. Without interpax, also use the
        # (memory-hungrier) matrix path for traced-points legs.
        if k1_is_points:
            Mx = matrix_spline_interp(fk, k1points, interp_order=interp_order, cache=spline_cache)
        else:
            Mx = matrix_rebin(k1edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
        if k2_is_points:
            My = matrix_spline_interp(fk, k2points, interp_order=interp_order, cache=spline_cache)
        else:
            My = matrix_rebin(k2edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
        return spectrum.contract(Mx, My)

    # At least one traced-points leg: interpolate *values* of the cached
    # spectrum directly instead of building an (npoints, n_fftlog)
    # interpolation matrix from an identity and sandwiching the dense
    # (n_fftlog, n_fftlog) spectrum. Interpolation is linear in the table, so
    # this is mathematically identical to the matrix form, but per vmapped
    # quadrature point it allocates only (npoints, nbins) outputs instead of
    # (npoints, n_fftlog) operators and batched (npoints, n_fftlog) @
    # (n_fftlog, n_fftlog) matmuls -- the difference between a few MB and
    # tens of GB over a 6D-quadrature vmap chunk. 'cubic2' matches the
    # scipy/matrix_rebin spline convention (see matrix_spline_interp).
    method = 'linear' if interp_order == 1 else 'cubic2'
    contracted_cache = cache.setdefault("QW_contracted", {})

    if k1_is_points and k2_is_points:
        # 2D interpolation at the (k1, k2) outer grid; spline coefficients
        # of the (concrete, cached) spectrum are precomputed once.
        key = spectrum_key + ('interp2d',)
        if key not in contracted_cache:
            contracted_cache[key] = interpax.Interpolator2D(fk, fk, spectrum.dense, method=method)
        interp = contracted_cache[key]
        n1, n2 = k1points.shape[0], k2points.shape[0]
        xq = jnp.repeat(k1points, n2)
        yq = jnp.tile(k2points, n1)
        return interp(xq, yq).reshape(n1, n2)

    if k1_is_points:
        # k2 binned (concrete): contract the binned side into the cached
        # spectrum once, then a single 1D interpolation along the traced leg.
        key = spectrum_key + ('right', tuple(np.ravel(k2edges)))
        if key not in contracted_cache:
            My = matrix_rebin(k2edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
            contracted_cache[key] = spectrum.dense @ My.T  # (n_fftlog, n2)
        table = contracted_cache[key]
        return interpax.interp1d(k1points, fk, table, method=method)  # (n1, n2)

    # k2 traced, k1 binned: mirror case.
    key = spectrum_key + ('left', tuple(np.ravel(k1edges)))
    if key not in contracted_cache:
        Mx = matrix_rebin(k1edges, fk, wt=fk**2, interp_order=interp_order, cache=rebin_cache)
        contracted_cache[key] = (Mx @ spectrum.dense).T  # (n_fftlog, n1)
    table = contracted_cache[key]
    return interpax.interp1d(k2points, fk, table, method=method).T  # (n1, n2)


def compute_QW_AB(window2, k1edges, k2edges, khat_dot_n, khatp_dot_n, fields1=None, fields2=None, cache=None, ells=None,
                  k1_is_points=False, k2_is_points=False):
    """
    Reconstruct

        Q_W^{A,B}(k - k') = sum_{ell1,ell2} Q^W_{ell1 ell2}(k,k') L_{ell1}(khat . n) L_{ell2}(khat' . n)

    This is the ``m = 0`` part of the exact kernel, whose angular factor is the tri-polar
    ``S_{ell1 ell2 q}(khat, khat', nhat)`` with ``q`` kept resolved rather than summed inside
    ``compute_spectrum2_covariance_window_block`` -- see approximation 1 in the module docstring.
    Since the two directions enter only through their line-of-sight cosines, the form above cannot
    represent the vector tie ``k' = +- k``, only ``|k'| = |k|`` together with the two cosines. It
    is *exact* wherever the rest of the integrand depends on each direction only through a
    ``L_ell(mu)`` -- the ``PP`` block, and the spectrum side of ``PB`` -- and an approximation in
    ``BB`` and ``PT``, which is why those two carry the caveats they do.

    Uses compute_spectrum2_covariance_window_block.  If ``k1_is_points``
    (``k2_is_points``) is True, ``k1edges`` (``k2edges``) is literal k-values
    (e.g. a derived/closure leg with no native bin edges) at which Q_W is
    interpolated rather than rebinned.
    """
    if cache is None:
        cache = {}
    if ells is None:
        ells = [0, 2, 4]
    block_cache = cache.setdefault("QW_ell_blocks", {})
    # Pass window2 as a (window2, window2) pair, not a single pre-resolved
    # window2.get(fields1=fields1, fields2=fields2): compute_spectrum2_covariance_window_block's
    # own get_window_field only symmetrizes Q_W^{A,B} with Q_W^{B,A} when its
    # window argument is a tuple -- a single resolved field block is used
    # as-is, un-symmetrized. compute_spectrum2_covariance (cov2.py) already
    # relies on this same tuple form for its own WW/WS/SS lookups; compute_QW_AB
    # needs it too, for the PP block where fields1/fields2 are same-size
    # (e.g. (a,a')/(b,b')) groups and the swap is a genuinely equivalent
    # relabeling. For PB/BP, fields1/fields2 are *mixed*-size groups (a
    # 2-field spectrum group and a 3-field bispectrum-derived group) with
    # fixed, non-interchangeable roles -- window2 only stores that one
    # canonical ordering, so swapping is not a valid lookup there (and
    # raises). Only symmetrize when the two groups are the same size.
    window2_pair = window2 if (fields1 is None or len(fields1) != len(fields2)) else (window2, window2)
    # k1edges/k2edges may be traced (e.g. a closure leg under jax.vmap), in
    # which case they cannot be used as a cache key: recompute, uncached.
    cacheable = jax.core.is_concrete(k1edges) and jax.core.is_concrete(k2edges)
    out = None
    for ell1 in ells:
        L1 = get_legendre(ell1)(khat_dot_n)
        for ell2 in ells:
            L2 = get_legendre(ell2)(khatp_dot_n)
            prefactor = (2 * ell1 + 1) * (2 * ell2 + 1) * (-1)**(ell1 // 2) * (-1)**(ell2 // 2)
            if cacheable:
                key = (id(window2), tuple(np.ravel(k1edges)), tuple(np.ravel(k2edges)), ell1, ell2, (fields1, fields2), k1_is_points, k2_is_points)
                if key not in block_cache:
                    block_cache[key] = prefactor * compute_spectrum2_covariance_window_block(window2_pair, k1edges, k2edges, ell1, ell2, fields1=fields1, fields2=fields2, cache=cache, k1_is_points=k1_is_points, k2_is_points=k2_is_points)
                block = block_cache[key]
            else:
                block = prefactor * compute_spectrum2_covariance_window_block(window2_pair, k1edges, k2edges, ell1, ell2, fields1=fields1, fields2=fields2, cache=cache, k1_is_points=k1_is_points, k2_is_points=k2_is_points)

            term = block * L1[:, None] * L2[None, :]
            out = term if out is None else out + term
    return out


def compute_QW_ABC(window3, kedges, kpedges,
                   khat1, khat2, khat1p, khat2p,
                   fields1=None, fields2=None, fields3=None,
                   ells=None, cache=None, batch_size=None):
    """
    Reconstruct

        Q_W^{ABC}(k1,k1',k2,k2')
        =
        sum_{ell,ellin}
        Q^W_{ell,ellin}(k1,k2;k1',k2')
        S_ell(khat1,khat2,n)
        S_ellin(khat1',khat2',n)

    using compute_spectrum3_covariance_window_block.
    """
    if cache is None:
        cache = {}
    if ells is None:
        ells = [(0, 0, 0)]

    block_cache = cache.setdefault("QW_ABC_ell_blocks", {})
    basis_cache = cache.setdefault("QW_ABC_S_basis", {})

    def basis(ell, xhat1, xhat2):
        key = ("S", tuple(ell))
        if key not in basis_cache:
            basis_cache[key] = get_S(ell, z3=True)
        return jnp.ravel(basis_cache[key](xhat1, xhat2))

    Sell = {tuple(ell): basis(ell, khat1, khat2) for ell in ells}
    Sellp = {tuple(ell): basis(ell, khat1p, khat2p) for ell in ells}

    out = None

    for ell in ells:
        ell = tuple(ell)
        S_ell = Sell[ell]
        for ellp in ells:
            ellp = tuple(ellp)
            S_ellp = Sellp[ellp]
            key = (id(window3), tuple(np.ravel(kedges)), tuple(np.ravel(kpedges)), ell, ellp, (fields1, fields2, fields3))
            if key not in block_cache:
                block_cache[key] = compute_spectrum3_covariance_window_block(
                    window3, kedges, kpedges, ell, ellp,
                    fields1=fields1, fields2=fields2, fields3=fields3,
                    cache=cache, batch_size=batch_size,
                )
            term = block_cache[key] * S_ell[:, None] * S_ellp[None, :]
            out = term if out is None else out + term

    return out


