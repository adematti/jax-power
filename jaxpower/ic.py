r"""
Exact window matrices of template deprojection: the radial integral constraint (RIC) of shuffled randoms, the angular mode removal (AMR)
of imaging weights fitted on the catalog (linear regression, or a calibrated harmonic response such as SYSNet), and any integral
constraint of the same form, including the global one removed through :math:`\alpha` (the last template), for the power spectrum on the
sphere with the local line of sight. The field is :math:`W \delta`, that of the plain window convolution: :math:`\Delta P` is all the
integral constraints.

The selection of each region :math:`b` is separable, :math:`W_b(\mathbf{x}) = g_b(r) \Theta_b(\hat{\mathbf{x}})`, :math:`\int d^2\hat{x}\, \Theta_b = 1`.
A *removed field* is a sum of templates :math:`a`,

.. math::

    \delta^{\rm t}(\mathbf{x}) = \sum_a g_a(r) \Theta_a(\hat{\mathbf{x}})\, \delta^{\rm t}_a(r),
    \qquad \delta^{\rm t}_a(r) = \int ds\, \Lambda_a(r, s)\, \bar{\delta}_{U_a}(s),
    \qquad \bar{\delta}_{U_a}(s) = \int d^2\hat{x}\, U_a(\hat{\mathbf{x}})\, \delta(s \hat{\mathbf{x}}),

with :math:`\Theta_a` the template's angular window (possibly signed), :math:`U_a` its mean window and :math:`\Lambda_a` its radial response:
diagonal (shell means), dense, or of rank one (:math:`\Lambda(r, s) = \alpha(r) \lambda(s)`: a regression coefficient over a redshift bin,
a cap mean). The cross power spectrum multipoles of two fields (line of sight on the first one) are

.. math::

    P_\ell^{XY}(k) = \frac{4\pi (2\ell + 1)}{\mathcal{N}} \sum_{L, L'} i^{L' - L} h^\ell_{LL'} \int dr\, dr'\, r^2 r'^2 j_L(kr) j_{L'}(kr') S^{XY}_{L'}(r, r'),

where, for the field :math:`\sum_b W_b \delta` and a removed field,
:math:`S_{L'}(r, r') = \sum_{b, a} g_b(r) g_a(r') \int ds\, K^{ba}_{L'}(r, s) \Lambda_a(r', s)`,
:math:`K^{ba}_{L'} = \xi^{\phi\psi}` for :math:`\phi = \Theta_b \sum_{M'} \Theta_{a, L'M'} Y_{L'M'}`, :math:`\psi = U_a`; for two removed fields,
:math:`S_{L'}(r, r') = \sum_{a, a'} (2L' + 1) C^{\Theta_a \Theta_{a'}}_{L'} g_a(r) g_{a'}(r') [\Lambda_a \xi^{U_a U_{a'}} \Lambda_{a'}^T](r, r')`.
The shell kernels :math:`\xi^{\phi\psi}(r, s) = \int d\theta \sin\theta\, \Pi^{\phi\psi}(\theta)\, \xi(r, s, \theta)`
are one angular integral weighted by the pair-count function :math:`\Pi^{\phi\psi}(\theta) = 2\pi \sum_l (2l + 1) C^{\phi\psi}_l \mathcal{L}_l(\cos\theta)`,
:math:`\xi(r, s, \theta)` the theory correlation function, plane parallel for each pair. All terms are linear in the theory multipoles
:math:`P_{\ell'}(q)`: one evaluation per theory band. A white shot noise :math:`N` adds :math:`N \delta_D(r - s) / r^2 \int \phi \psi` to the kernels.

Regions of the survey and templates are both :class:`Template`: a HEALPix map of the sky pattern (from the randoms) times a radial
profile on distance shells (shared by all, the model's), plus, for templates, the sky area over which their amplitude is measured and
how it follows the data. Footprints and mean windows are normalized to :math:`\int \Theta = 1`, the pixel window deconvolved; products
and cross-spectra of maps take each map from a different random subset, so that no random pairs with itself. healpy is used for the maps
(harmonic transforms, cls); everything else is JAX.

Nomenclature. Everything that depends on the density is a :math:`\delta`; the rest is geometry, fixed by the randoms and the pipeline.

- Density and fields: :math:`\delta(\mathbf{x})` the density contrast (Gaussian, theory correlation function :math:`\xi`, power spectrum
  :math:`P`); :math:`\delta^{\rm f} = W \delta` the field without any removal (the plain window convolution);
  :math:`\bar{\delta}_W = \int d^3x\, W \delta / \int d^3x\, W` its survey mean (removed through :math:`\alpha`); :math:`\delta^{\rm t}` the
  removed field, :math:`\delta^{\rm obs} = \delta^{\rm f} - \delta^{\rm t}`; :math:`\delta^{\rm t}_a(r)` the amplitude of template :math:`a`;
  :math:`\bar{\delta}_U(s)` the mean of :math:`\delta` over the mean window :math:`U` in shell :math:`s`; :math:`\bar{\delta}^{\rm t}` the
  survey mean of the removed field, removed through :math:`\alpha`; :math:`d_R(\hat{x}) = \sum_s \lambda_R(s)\, \delta(s \hat{x})` the
  density projected over a redshift bin, :math:`d_{R, LM}` its harmonic coefficients; :math:`\epsilon_R` the weight pattern of calibrated
  weights (data weighted by :math:`1 - \epsilon_R`), :math:`\epsilon^\perp_R` its part independent of the density.
- Selection: :math:`b` a survey region, :math:`W_b = g_b \Theta_b`, :math:`W = \sum_b W_b`; :math:`g(r)` a radial profile
  (:attr:`Template.radial`, :math:`g \int r^2 dr` the weighted counts per shell of ``redges``); :math:`\Theta(\hat{\mathbf{x}})` a sky map
  (:attr:`Template.angular`, :math:`\int \Theta = 1` if non-negative); :math:`\mathcal{N}` the normalization (``norm``).
- Templates: :math:`a` a template, :math:`W_a = g_a \Theta_a` its pattern; :math:`U_a` its mean window (:attr:`Template.mean`);
  :math:`\Lambda_a(r, s)` its response (:attr:`Template.response`), from the shell means :math:`s` to the template shells :math:`r`: diagonal
  :math:`f(r)`, dense, or rank one :math:`\alpha(r) \lambda(s)`, with :math:`\lambda(s)` the radial weights (fraction of the data of shell
  :math:`s` in the single amplitude) and :math:`\alpha(r)` the profile; harmonic templates (calibrated weights of region :math:`R`): the
  templates :math:`a = (R, L, M)` of window :math:`W_R Y_{LM}`, mean window :math:`Y^\star_{LM}` and response :math:`\kappa_R(L) \lambda_R(s)`,
  :math:`\delta^{\rm t}_a = \kappa_R(L)\, d_{R, LM}`, with :math:`\kappa_R(L)` the removed fraction of degree :math:`L` and :math:`\lambda_R`
  the radial weights; :math:`C^{\perp\perp}_{R, L}` the angular power of :math:`\epsilon^\perp_R`.
- Degrees and functions: :math:`\ell, \ell'` the multipoles of the measured and of the theory power spectra; :math:`L, L'` angular degrees on
  the sky (:math:`L'` that of a pair of points, :math:`L` that of a removed mode); :math:`l` the degree of a map spectrum;
  :math:`\mathcal{L}_l` the Legendre polynomials, :math:`Y_{LM}` the spherical harmonics, :math:`f_{LM} = \int d^2\hat{x}\, f\, Y^\star_{LM}` the harmonic
  coefficients of a map :math:`f` (:math:`\sum_M f_{LM} Y_{LM}` its degree-:math:`L` part); :math:`j_L` the spherical Bessel functions.
- Geometric factors: :math:`C^{\phi\psi}_l` the cross angular power of maps :math:`\phi`, :math:`\psi`; :math:`\Pi^{\phi\psi}(\theta)` their
  pair-count function; :math:`\xi^{\phi\psi}(r, s)` the shell kernel (pairs of :math:`\phi` at :math:`r`, :math:`\psi` at :math:`s`);
  :math:`K^{ba}_{L'}` the field x template kernel; :math:`M^{bR}_{L'L}` the mode-coupling matrix of two footprints;
  :math:`N^{Ra}_{L'L}` the harmonic x regular coupling; :math:`C^{\delta\delta}_L(r, s)` the theory angular power between shells.
- Spectra: :math:`S^{XY}_{L'}(r, r')` the angular kernel of the cross spectrum of fields :math:`X`, :math:`Y` (line of sight on
  :math:`X`); :math:`P^{XY}_\ell`, :math:`X, Y \in \{{\rm f}, {\rm t}\}`; :math:`\Delta P = -P^{\rm ft} - P^{\rm tf} + P^{\rm tt}`.
- In the code: ``IC2Model.removed['t']`` (regular templates and, with ``gic``, the :math:`\alpha` template, :func:`gic_templates`
  :math:`W (\bar{\delta}_W - \bar{\delta}^{\rm t})`), ``['h']`` (harmonic templates); ``IC2Model.pairs`` keyed by pairs of these and ``'f'``
  (the field).

Bispectrum (:func:`compute_smooth3_ic_window`, Sec. V G of the notes). At first order in the removed field, on one leg at a time,
:math:`\Delta B = W^{BB} B^{\rm th} + (W^{BP} + \bar{n}^{-1} W^{BN}) P^{\rm th}` with :math:`W^{BB} = W^{\rm geo} D^{BB} + C^{BB}` and
likewise for BP and BN (Eq. 154). The :math:`D` act on the theory before the geometry window :math:`W^{\rm geo}` (Eq. 141):

- :math:`D^{BB}` (Eqs. 143-144) replaces each leg :math:`\mathbf{k}_i` of the triangle by the modes :math:`\mathbf{q}` the templates draw on,
  weighted by the removal kernel :math:`\tau(k, \mu; q, \mu')` (Eq. 145), from the window matrix of :math:`-P^{\rm ft}` (Eq. 142), i.e.
  :func:`compute_smooth2_ic_window` with output multipoles :math:`\ell \leq 8`. :math:`\mathbf{q}` is taken in the plane of
  :math:`\mathbf{k}_i` and the line of sight, :math:`\tau` being averaged over azimuths. The kernel only depends on the geometry and the
  templates, so :math:`D^{BB}` is a theory-independent sparse matrix on the theory's Sugiyama multipoles on the :math:`(k_1', k_2')` grid
  of the geometry window. When the third leg is replaced, the new triangles are read as the pairs :math:`(\mathbf{q}, \mathbf{k}_2)` and
  :math:`(\mathbf{k}_1, \mathbf{q})`: the small removed leg is then a grid magnitude, and the window's six multipoles nearly suffice (as
  :math:`(-(\mathbf{q} + \mathbf{k}_2), \mathbf{k}_2)` the squeezed triangle sits in the opening angle, and the multipoles converge very slowly).
- :math:`D^{BP}` (Eqs. 148-149) is the Gaussian four-point term of weights :math:`1 + \eta` that follow the density:
  :math:`P^\eta = \tau^\eta \ast P^{\rm th}` (Eq. 147) on one leg times :math:`P^{\rm th}` on the other two. Examples are FKP weights from each
  catalog's own :math:`n(z)` (RIC; :func:`fkp_modulation_templates`, :func:`compute_smooth2_modulation_window`), or
  :math:`P^\eta = -P^{\rm ft} + P^{\rm tt}` (Eq. 161) for imaging weights.
- :math:`D^{BN}` (Eq. 150), the shot noise of the removal, is the :math:`D^{BP}` form with :math:`P^\eta \to -N`, :math:`N` the shot noise of
  :math:`P^{\rm ft}`.

The composition with :math:`W^{\rm geo}` treats the window and the removal as separable, which holds for the binned legs but not for the
closure (third, unbinned) leg at the window scale, reached by the nearly antipodal pairs of close shells. One rule for every term: the
closure leg carrying the removal (or :math:`P^\eta`, :math:`-N`) is composed for :math:`k_3 \geq q_{\rm split}`, and below replaced by its
exact term, computed with the window :math:`W^2` on the field and the normalization :math:`\int W^3` (the subscript :math:`W^2`):
:math:`C^{BB}` from :math:`\tau_{W^2}` (Eq. 152, :func:`compute_smooth3_closure_window`, ``closure_kernel``), :math:`C^{BP}` and :math:`C^{BN}`
from :math:`P^\eta_{W^2}` and :math:`-N_{W^2}` (Eq. 153, ``closure_modulation``, ``closure_shotnoise``, the same function). Any bins, diagonal or not: the exact
terms vanish for shells further apart than :math:`q_{\rm split}`. Terms with the removed field on two or three legs are neglected (< 2% in
the forward tests). A cut of the window-scale modes applied to the catalogs (e.g. the angular modes of the projected density,
:func:`ellcut_templates`) is a removed field like the others. A fixed data / randoms mismatch (weights that do not restore the counts in the
mocks) is a defect of the catalogs, not modelled here.

References to desi-gqc-notes give the equation number in the compiled notes (``ms.pdf`` of 2026-11-18) and its label. Sec. V
(``sections/ric_model.tex``) covers the RIC and the general formalism:

- the removed field, Eq. 122 [eq:ric_template];
- :math:`\Delta P`, Eq. 126 [eq:ric_cross_auto];
- the multipoles, Eq. 130 [eq:ric_poles];
- the leg spectra, Eqs. 132-133 [eq:ric_field_removed, eq:ric_removed_removed];
- the window matrix, Eq. 138 [eq:ric_window_matrix]; the removal kernel from it, Eq. 145 [eq:ric_tau_matrix];
- the bispectrum, Eqs. 141-149 [eq:ric_bispectrum_window, eq:ric_tau_kernel, eq:bispectrum_rem, eq:ric_bispectrum_rem,
  eq:ric_eta_kernel, eq:bispectrum_eta, eq:ric_bispectrum_eta]; the shot noise of the removal, Eq. 150 [eq:ric_bispectrum_shotnoise];
  the closure leg, Eqs. 152-154 [eq:ric_closure_removal, eq:collapsed_multipoles, eq:ric_bispectrum_full].

Sec. VI (``sections/amr_model.tex``) covers the imaging weights: Eq. 157 [eq:amr_regression], Eqs. 159-160 [eq:amr_template]; for
SYSNet (Sec. VI D), Eq. 162 [eq:amr_sysnet_template] and Eq. 166 [eq:amr_sysnet]. If the notes change, the labels stay valid and the
numbers may shift.
"""

import time
import logging
from dataclasses import dataclass

import numpy as np
import jax
from jax import numpy as jnp

from .types import WindowMatrix, Mesh2SpectrumPole, Mesh2SpectrumPoles, ObservableTree
from .utils import BesselIntegral, get_S, wigner_3j


logger = logging.getLogger('IC')


# Special functions at all orders up to L ~ 800 (lax.scan recurrences; utils.get_legendre_recurrence and utils.get_spherical_jn_all unroll
# the recurrence per order, which is impractical there)

def _peak_memory() -> str:
    """Peak resident memory of this process (MPI rank), for the logs."""
    import resource
    return f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB'  # ru_maxrss in kB on Linux


def _legendre_matrix(lmax: int, x) -> jax.Array:
    r"""Legendre polynomials :math:`\mathcal{L}_0 .. \mathcal{L}_{lmax}` at ``x``, ``(lmax + 1, x.size)`` (upward recurrence)."""
    x = jnp.asarray(x, dtype='f8')
    p0, p1 = jnp.ones_like(x), x
    if lmax == 0: return p0[None]
    if lmax == 1: return jnp.stack([p0, p1])

    def step(carry, l):
        pm1, p = carry
        pn = ((2 * l + 1) * x * p - l * pm1) / (l + 1)  # Bonnet: (l + 1) L_{l+1} = (2l + 1) x L_l - l L_{l-1}
        return (p, pn), pn

    _, ps = jax.lax.scan(step, (p0, p1), jnp.arange(1, lmax, dtype='f8'))
    return jnp.concatenate([p0[None], p1[None], ps], axis=0)


def _spherical_jn_all(lmax: int, x) -> jax.Array:
    r"""
    Spherical Bessel functions :math:`j_0 .. j_{lmax}` at all ``x > 0``, ``(lmax + 1, x.size)``. Upwards in :math:`l`, the recurrence
    :math:`j_{l+1} = (2l + 1) / x\, j_l - j_{l-1}` is unstable for :math:`l > x`; downwards it is stable. Run downwards on the ratios
    :math:`r_l = j_l / j_{l-1}`, which never overflow, from :math:`l \gg \max(lmax, x)` where :math:`r_l \simeq 0`; then
    :math:`j_l = j_0 r_1 \cdots r_l`, anchored on the exact :math:`j_0` or :math:`j_1`, whichever is farther from a zero.
    """
    xmax = float(np.max(np.asarray(x)))
    x = jnp.asarray(x, dtype='f8')
    lmax1 = max(lmax, 1)
    lstart = int(max(lmax1, xmax)) + 50 + int(2. * np.sqrt(max(lmax1, xmax)))  # far enough above that the start value does not matter

    def step(r, l):  # r_l = x / (2l + 1 - x r_{l+1}), from j_{l-1} + j_{l+1} = (2l + 1) / x j_l divided by j_l
        r = x / (2 * l + 1 - x * r)
        return r, r

    _, ratios = jax.lax.scan(step, jnp.zeros_like(x), jnp.arange(lstart, 0, -1, dtype='f8'))  # r_lstart, ..., r_1
    ratios = ratios[::-1][:lmax1]  # r_1, ..., r_lmax1
    j0, j1 = jnp.sinc(x / np.pi), (jnp.sin(x) / x - jnp.cos(x)) / x
    j0 = jnp.where(jnp.abs(j0) >= jnp.abs(j1), j0, j1 / ratios[0])  # near a zero of j_0, take it from j_1 = j_0 r_1
    out = j0 * jnp.concatenate([jnp.ones_like(x)[None], jnp.cumprod(ratios, axis=0)], axis=0)  # j_l = j_0 r_1 ... r_l
    return out[:lmax + 1]


# Maps and templates

@dataclass(frozen=True, eq=False)
class Template(object):
    r"""
    A pattern on the sky times a profile in distance: either one region of the survey (geometry), or one template of what the pipeline
    removes, :math:`g_a(r) \Theta_a(\hat{x}) \delta^{\rm t}_a(r)` (:math:`a` labels the template),
    :math:`\delta^{\rm t}_a(r) = \int ds\, \Lambda_a(r, s)\, \bar{\delta}_{U_a}(s)` its amplitude, measured from the data,
    :math:`\bar{\delta}_{U_a}(s)` the mean density contrast over the mean window :math:`U_a` in shell :math:`s`. The amplitude itself is never
    given: it is linear in the density, and the model needs only how it is obtained, :math:`U_a` (``mean``) and :math:`\Lambda_a`
    (``response``). Nothing depends on the data realization.

    Parameters
    ----------
    angular : array, (npix,) or (nsplit, npix)
        HEALPix map(s) (ring ordering) of the sky pattern :math:`\Theta`: for a region, its footprint (randoms with the field weights); for a
        template, its sky pattern (e.g. a region's footprint, or a footprint times an imaging map). Several maps are independent random
        subsets: products of maps take each factor from a different subset, so that a random never pairs with itself. With a single map,
        the randoms' own shot noise leaks into the prediction.
    radial : array (nshell,)
        Radial profile :math:`g(r)`, one value per shell of ``redges``, :math:`g \int r^2 dr` the weighted counts per shell:
        the weighted counts per shell of ``redges`` divided by :math:`\int r^2 dr` over the shell, :math:`(r_{i + 1}^3 - r_i^3) / 3`.
    redges : array (nshell + 1,)
        Edges of the distance shells on which ``radial`` and the responses below are tabulated (widths may vary; thin enough to resolve the
        shell means). All the templates of a model share them.
    mean : array, optional
        HEALPix map(s) (same convention) of the sky area, with the data weights, over which the data are averaged to get the template's
        amplitude, :math:`U`; or a linear combination of maps, ``((map, coefficient), ...)``. ``None`` for a region of the geometry.
    response : tuple, optional
        How the amplitude follows the data, :math:`\Lambda`: in
        :math:`\delta^{\rm t}_a(r) = \sum_s \Lambda(r, s) \bar{\delta}_{U_a}(s)`, how the amplitude at template shell :math:`r` is built from
        the shell means :math:`\bar{\delta}_{U_a}(s)` of the data. ``None`` for a region of the geometry.

        .. list-table::
           :header-rows: 1
           :widths: 18 14 34 34

           * - kind
             - :math:`\Lambda(r, s)`
             - amplitude
             - example
           * - ``('diag', f)``
             - :math:`f(r) \delta_{rs}`
             - each shell uses its own shell mean: :math:`\delta^{\rm t}_a(r) = f(r) \bar{\delta}_{U_a}(r)`
             - RIC (:func:`ric_templates`): shuffled randoms follow the mean density of the data shell by shell; :math:`f` the share
               of the shuffle region's data weight in the shell (1 for a region shuffled on its own)
           * - ``('dense', L)``
             - any matrix ``L`` (nshell, nshell)
             - each shell's amplitude mixes the shell means of several shells
             - a response known only as a matrix (e.g. a smoothed :math:`n(z)` fitted on the data); a RIC written as a full matrix
           * - ``('outer', profile, radial_weights)``
             - :math:`\alpha(r) \lambda(s)`, rank one
             - one number for many shells: :math:`\delta^{\rm t}_a(r) = \alpha(r) \sum_s \lambda(s) \bar{\delta}_{U_a}(s)`;
               ``radial_weights`` :math:`\lambda(s)` the share of the data of shell :math:`s` in that number, ``profile``
               :math:`\alpha(r)` how it spreads over the template's shells
             - imaging regression (:func:`amr_templates`): one coefficient per map fitted over the redshift bin, :math:`\lambda(s)` the
               fraction of the bin's randoms in shell :math:`s`, :math:`\alpha = 1`; the survey means removed through :math:`\alpha`.
               The fast case: :math:`\xi` is integrated against :math:`\lambda` first
           * - ``('harmonic', radial_weights, removed_fraction, uncorrelated_power)``
             - :math:`\kappa_R(L) \lambda(s)`, rank one for each harmonic :math:`(L, M)`
             - one template stands for the templates :math:`a = (R, L, M)` of pattern :math:`\Theta_R Y_{LM}` and mean window
               :math:`Y^\star_{LM}` (``mean`` unused): :math:`\delta^{\rm t}_a = \kappa_R(L) \sum_s \lambda(s) \bar{\delta}_{Y^\star_{LM}}(s)`, a
               fraction ``removed_fraction`` :math:`\kappa_R(L)` of each angular mode of the density projected with ``radial_weights``
               :math:`\lambda(s)`; plus a part independent of the density, of angular power ``uncorrelated_power``
             - weights calibrated on mocks, e.g. SYSNet (:func:`harmonic_templates`): the sums over :math:`(L, M)` done analytically,
               through mode-coupling matrices

        In all of them, :math:`U_a` (``mean``) is the sky area, with the data weights, over which the shell means are taken: the shuffle
        region for the RIC, the regression operator applied to the imaging bins for a regression coefficient. ``L`` and ``radial_weights``
        include the shell width.
    group : hashable, optional
        Templates of the same group share ``radial`` and ``response``: their pair-count functions are summed (faster).
    """
    angular: np.ndarray
    radial: np.ndarray
    redges: np.ndarray
    mean: object = None
    response: tuple = None
    group: object = None

    def __post_init__(self):
        object.__setattr__(self, 'radial', np.asarray(self.radial, dtype='f8'))
        object.__setattr__(self, 'redges', np.asarray(self.redges, dtype='f8'))
        if self.redges.ndim != 1 or self.redges.size < 2 or np.any(np.diff(self.redges) <= 0.):
            raise ValueError('redges must be an increasing 1D array of at least two edges')
        mean = self.mean
        if mean is not None and not (isinstance(mean, tuple) and len(mean) and isinstance(mean[0], tuple)):
            mean = ((mean, 1.),)  # stored as a linear combination of maps
        object.__setattr__(self, 'mean', None if mean is None else tuple((m, float(c)) for m, c in mean))
        if self.response is not None:
            kind = self.response[0]
            if kind not in ('diag', 'dense', 'outer', 'harmonic'): raise ValueError(f'unknown response {kind}')
            object.__setattr__(self, 'response', (kind,) + tuple(np.asarray(x, dtype='f8') for x in self.response[1:]))
        # radial profile and responses tabulated on the shells of redges
        nshell = self.redges.size - 1
        shapes = [('radial', self.radial.shape, (nshell,))]
        if self.kind == 'dense': shapes.append(('response', self.response[1].shape, (nshell, nshell)))
        elif self.kind in ('diag', 'outer', 'harmonic'):
            shapes += [('response', x.shape, (nshell,)) for x in self.response[1:3 if self.kind == 'outer' else 2]]
        for what, shape, expected in shapes:
            if shape != expected: raise ValueError(f'{what} has shape {shape}, expected {expected} for the {nshell} shells of redges')

    @property
    def kind(self):
        return None if self.response is None else self.response[0]


def check_common_redges(**templates) -> np.ndarray:
    """The shell edges shared by all templates (keyword: list of templates), else a ValueError naming the first that differs."""
    redges = None
    for name, ts in templates.items():
        for it, t in enumerate(ts):
            if redges is None: redges, first = t.redges, f'{name}[{it}]'
            elif t.redges.shape != redges.shape or not np.allclose(t.redges, redges, rtol=1e-12, atol=0.):
                raise ValueError(f'{name}[{it}].redges ({t.redges.size - 1} shells, [{t.redges[0]:g}, {t.redges[-1]:g}]) differ from '
                                 f'{first}.redges ({redges.size - 1} shells, [{redges[0]:g}, {redges[-1]:g}])')
    return redges


def ric_templates(geometry: list, shuffle: list) -> list:
    r"""
    Templates of the radial integral constraint (RIC) of randoms whose redshifts are drawn from the data. In each thin distance shell, the
    randoms of a region follow the data's mean density in that shell, over the region's shuffle region: the pipeline thus subtracts, in
    each shell, the mean data density of the shuffle region (Eq. 122 [eq:ric_template]). Each region gets one template per part of its
    shuffle region: the region's footprint and radial profile, times the mean density over the part, weighted by the part's share of
    the data weight in the shell.

    Parameters
    ----------
    geometry : list of Template
        The regions of the cap (randoms with the field weights).
    shuffle : list of Template
        For each region, its shuffle region as a :class:`Template`, with the weights the shuffle averages the data with (e.g.
        :math:`w_{\rm tot}`, without FKP weights): ``angular`` the HEALPix map(s), ``radial`` its weighted counts per shell over :math:`\int r^2 dr`. If the
        shuffle region extends beyond the cap (e.g. S, shuffled over both caps), a list of its parts (in the cap, in the other cap): the
        shares of the parts per shell are read from their ``radial`` (Eq. 123).

    Returns
    -------
    ic : list of Template

    Examples
    --------
    North shuffled on its own and South shuffled over both caps, in the North cap:

    >>> geometry = [Template(map_N, g_N, redges), Template(map_S_ngc, g_S_ngc, redges)]  # randoms with the field weights
    >>> shuffle = [Template(mapd_N, gd_N, redges), [Template(mapd_S_ngc, gd_S_ngc, redges), Template(mapd_S_sgc, gd_S_sgc, redges)]]  # data weights
    >>> ic = ric_templates(geometry, shuffle)
    """
    out = []
    for ireg, (reg, parts) in enumerate(zip(geometry, shuffle)):
        parts = [parts] if isinstance(parts, Template) else list(parts)
        check_common_redges(geometry=[reg], shuffle=parts)
        # share of each part in the data weight per shell: r^2 g(r) dr, the common r^2 dr dropping out of the ratio (f_cap of Eq. 123
        # for S over both caps; 1 for a region shuffled on its own)
        weights = np.array([part.radial for part in parts])
        total = weights.sum(axis=0)
        shares = np.where(total > 0., weights / np.where(total > 0., total, 1.), 0.)
        for ipart, (part, share) in enumerate(zip(parts, shares)):
            # Eq. 122 [eq:ric_template] with Lambda(r, s) = share(r) delta_D(r - s): the shell mean over the part, times its share
            out.append(Template(reg.angular, reg.radial, reg.redges, mean=part.angular, response=('diag', share), group=('ric', ireg, ipart)))
    return out


def fkp_modulation_templates(geometry: list, means: list, beta: list, zbin: np.ndarray) -> list:
    r"""
    Templates of the modulation :math:`W \eta` of the field by weights that follow the data's own radial density, e.g. FKP weights
    :math:`1 / (1 + \bar{n} P_{\rm FKP})` with :math:`\bar{n}` from each catalog's own :math:`n(z)` (the shuffled arm of the RIC mocks;
    notes, Eq. 147 [eq:ric_eta_kernel]): :math:`\eta_a(r) = -\beta_a(r)\, \bar{\delta}_{\rm cap}(r)` on region :math:`a`, with
    :math:`\beta = -d \ln w / d \ln \bar{n}` (:math:`\bar{n} P / (1 + \bar{n} P)` for FKP weights; :math:`(1 + 2\beta_{\rm FKP}) / 3` for the
    bispectrum weights :math:`w_{\rm FKP} (\bar{n} w_{\rm FKP})^{-1/3}`) and :math:`\bar{\delta}_{\rm cap}(r)` the mean density contrast of the
    cap over the redshift bin of :math:`\bar{n}` that contains :math:`r` (all regions of the cap, weighted by their randoms in the bin: the
    :math:`n(z)` is the cap's). :math:`\beta_a` is that of the region's own :math:`\bar{n}` (``NX``, the :math:`n(z)` times the completeness
    of the region), which differs between regions; within a region it is taken constant on the sky.
    The weights multiply the data, so :math:`\eta` is a template field like the removed ones, with a dense response; its power spectrum
    with the observed field, :math:`P^\eta`, is :func:`compute_smooth2_modulation_window`, the input ``modulation`` of
    :func:`compute_smooth3_ic_window` (:math:`D^{BP}`).

    Parameters
    ----------
    geometry : list of Template
        The regions of the cap (randoms with the field weights): the pattern and radial profile of :math:`\eta` on each, and, through
        their randoms per shell, the weights of the regions in the cap mean.
    means : list of Template
        For each region, the sky area over which its data are averaged (``angular``: data weights, as the shuffle's mean windows of
        :func:`ric_templates`), one per region of ``geometry``.
    beta : array (nshell,), list
        :math:`\beta_a(r)` on the shells of ``redges`` (zero outside the data), one array per region of ``geometry``; one array: the same
        for every region.
    zbin : array (nshell,)
        Label of the :math:`\bar{n}` bin of each shell (e.g. ``floor(z / dz)``): the cap mean is taken over the shells of the same label.

    Returns
    -------
    eta : list of Template
        One template per (region of the pattern, region of the mean), response ``('dense', L)``,
        :math:`L(r, s) = -\beta_a(r) [Z(r) = Z(s)]\, w_b(s) / \sum_{s' \in Z(r)} w_{\rm tot}(s')`, :math:`w_b` the randoms of region :math:`b`
        in shell :math:`s` (:math:`\int r^2 dr` over the shell times ``radial``).
    """
    redges = check_common_redges(geometry=geometry, means=list(means))
    if len(means) != len(geometry): raise ValueError('one mean window per region of the geometry')
    zbin = np.asarray(zbin)
    betas = [np.asarray(b, dtype='f8') for b in beta] if isinstance(beta, (list, tuple)) else [np.asarray(beta, dtype='f8')] * len(geometry)
    if len(betas) != len(geometry): raise ValueError('one beta per region of the geometry')
    wr = [(np.diff(np.asarray(redges) ** 3) / 3.) * reg.radial for reg in geometry]
    wtot = sum(wr)
    same = (zbin[:, None] == zbin[None, :]).astype('f8')
    total = same @ wtot  # sum of w_tot over the shells of the bin of each shell r
    out = []
    for ia, (reg, beta) in enumerate(zip(geometry, betas)):
        for ib, (mean, w) in enumerate(zip(means, wr)):
            L = -beta[:, None] * same * w[None, :] / np.maximum(total[:, None], 1e-300)
            out.append(Template(reg.angular, reg.radial, redges, mean=mean.angular, response=('dense', L), group=('fkp modulation', ia, ib)))
    return out


def amr_templates(region: Template, maps: np.ndarray, zbin: np.ndarray, counts: np.ndarray, xbar: np.ndarray=None,
                  nbins: int=10, errors: np.ndarray=None) -> list:
    r"""
    Templates of imaging weights from a linear regression, fitted in one imaging region :math:`R` and redshift bin :math:`Z`: the data are
    weighted by :math:`1 - \sum_j c_j (x_j - \bar{x}_j)`, :math:`x_j` the imaging maps (Eq. 159 [eq:amr_template]). The pipeline thus
    subtracts, for each map, the pattern :math:`x_j - \bar{x}_j` on the region, times a coefficient measured from the data: the fit makes
    :math:`c_j = \sum_{i\nu} G_{j, i\nu} d_{i\nu}` (Eq. 157 [eq:amr_regression]), :math:`d_{i\nu}` the mean density contrast of the data in
    bin :math:`\nu` of map :math:`i`, over the whole redshift bin. This function builds the regression operator :math:`G` from the maps and
    the randoms, and returns one template per map.

    Parameters
    ----------
    region : Template
        The region in the cap (footprint of the randoms with the field weights, radial selection on its shells ``redges``).
    maps : array (nmap, npix)
        Imaging maps as fitted, each rescaled onto [0, 1] by the bin edges of the fit (``nbins`` equal bins over the data range); NaN
        where undefined.
    zbin : array (nshell,)
        Indicator of the redshift bin on the shells.
    counts : array (npix,) or (nsplit, npix)
        Weighted counts per pixel of the randoms of the fit (weights of the fit, region and redshift bin, over the whole area of the fit,
        e.g. both caps for a region spanning them), zero in the pixels the fit sets aside (e.g. the tails of the maps). Several maps are
        independent random subsets, as :attr:`Template.angular`; the regression operator uses their sum.
    xbar : array (nmap,), optional
        Means of the maps over the data of the bin, which the randoms inherit with the data's weights; default: over ``counts``.
    nbins : int, default=10
        Number of bins per map in the fit.
    errors : array (nmap, nbins), optional
        Errors of the bin contrasts in the fit. Default: :math:`\propto 1 / \sqrt{R_{i\nu}}`, :math:`R_{i\nu}` the randoms of the bin, the
        pipeline's error for the expected data counts (:math:`G` does not depend on their overall scale).

    Returns
    -------
    ic : list of Template
    """
    redges = region.redges
    maps = np.atleast_2d(np.asarray(maps, dtype='f8'))
    nmap, npix = maps.shape
    pixarea = 4. * np.pi / npix
    counts = np.atleast_2d(np.asarray(counts, dtype='f8'))
    defined = np.all(np.isfinite(maps), axis=0)
    x = np.where(defined, maps, 0.)
    counts = np.where(defined, counts, 0.)
    ibin = np.clip(np.floor(x * nbins).astype(int), 0, nbins - 1)  # bin nu of each pixel in each map i, (nmap, npix)

    def bin_sum(weights):  # sum over the pixels of each bin (i, nu), (nmap, nbins)
        return np.array([np.bincount(ibin[i], weights=weights, minlength=nbins) for i in range(nmap)])

    # Eq. 157 [eq:amr_regression]: c = G d, G = (D^T Sigma^-1 D)^-1 D^T Sigma^-1, rows of D (1, <x_1>_{i nu}, ..., <x_J>_{i nu}) over the randoms
    total = counts.sum(axis=0)
    nrand = bin_sum(total)
    xmean = np.array([bin_sum(total * x[j]) for j in range(nmap)]) / np.maximum(nrand, 1e-300)[None]  # <x_j>_{i nu}, (nmap j, nmap i, nbins)
    D = np.concatenate([np.ones((nmap * nbins, 1)), xmean.reshape(nmap, -1).T], axis=1)
    if errors is None: errors = 1. / np.sqrt(np.maximum(nrand, 1e-300))
    inv_var = np.where(nrand > 0., 1. / np.asarray(errors, dtype='f8')**2, 0.).ravel()  # empty bins do not enter the fit
    G = np.linalg.solve(D.T @ (inv_var[:, None] * D), D.T * inv_var[None, :])  # (1 + nmap, nmap * nbins)
    if xbar is None: xbar = np.sum(total * x, axis=1) / np.sum(total)
    xbar = np.asarray(xbar, dtype='f8')

    # sky patterns Theta_R (x_j - xbar_j), from the normalized footprint (signed maps are used as given by IC2Model)
    footprint = np.atleast_2d(np.asarray(region.angular, dtype='f8'))
    footprint = footprint / (np.sum(footprint, axis=-1, keepdims=True) * pixarea)
    angular = [footprint * np.where(defined, x[j] - xbar[j], 0.) for j in range(nmap)]
    # mean windows sum_{i nu} G_{j, i nu} U_{i nu}, U_{i nu} the normalized density of the randoms of bin (i, nu), per random subset
    per_bin = [np.maximum(bin_sum(c), 1e-300) for c in counts]  # randoms per bin of each subset, which normalize its U_{i nu}
    mean = []
    for j in range(nmap):
        mean.append(np.array([c * sum(G[1 + j, i * nbins + ibin[i]] / n[i, ibin[i]] for i in range(nmap)) / pixarea for c, n in zip(counts, per_bin)]))
    # delta^t_j = c_j: Lambda(r, s) = lambda_RZ(s), the radial distribution of the randoms of the region in the bin, the same for every r (Eq. 160)
    zbin = np.asarray(zbin, dtype='f8')
    radial_weights = (np.diff(np.asarray(redges) ** 3) / 3.) * region.radial * zbin
    radial_weights = radial_weights / np.sum(radial_weights)
    g = region.radial * zbin
    group = ('amr', id(region), zbin.tobytes())  # the templates of one (R, Z) share radial and response
    return [Template(a, g, redges, mean=m, response=('outer', np.ones_like(g), radial_weights), group=group) for a, m in zip(angular, mean)]


def real_harmonic_maps(lcut: int, nside: int, pixels: np.ndarray=None) -> np.ndarray:
    r"""
    Real spherical harmonics of degree :math:`L < l_{\rm cut}` on HEALPix pixels (ring ordering): :math:`Y_{L0}`, and for :math:`M > 0`,
    :math:`\sqrt{2}\, {\rm Re}\, Y_{LM}` and :math:`\sqrt{2}\, {\rm Im}\, Y_{LM}`, ordered by :math:`L`, then :math:`M` (re, im), :math:`l_{\rm cut}^2`
    maps. A map :math:`\sum_a c_a Y_a` is ``healpy.alm2map`` of the complex coefficients of :func:`real_to_complex_alm`.

    Parameters
    ----------
    lcut : int
        Degrees :math:`0 \leq L < l_{\rm cut}`.
    nside : int
        HEALPix resolution.
    pixels : array, optional
        Pixel indices to return (default all).

    Returns
    -------
    Y : array (lcut**2, npix or len(pixels))
    """
    import healpy as hp
    out = []
    lmax = max(lcut - 1, 0)
    nalm = hp.Alm.getsize(lmax)
    for L in range(lcut):
        for M in range(L + 1):
            for value in ([1.] if M == 0 else [1. / np.sqrt(2.), -1j / np.sqrt(2.)]):
                alm = np.zeros(nalm, dtype='c16')
                alm[hp.Alm.getidx(lmax, L, M)] = value
                m = hp.alm2map(alm, nside, lmax=lmax, pixwin=False)
                out.append(m if pixels is None else m[pixels])
    return np.array(out)


def real_to_complex_alm(coeffs: np.ndarray, lcut: int) -> np.ndarray:
    r"""Complex coefficients (``healpy`` layout, ``lmax = lcut - 1``) of :math:`\sum_a c_a Y_a`, :math:`Y_a` the real harmonics of
    :func:`real_harmonic_maps`: :math:`a_{L0} = c_{L0}`, :math:`a_{LM} = (c^{\rm re}_{LM} - i\, c^{\rm im}_{LM}) / \sqrt{2}`."""
    import healpy as hp
    lmax = max(lcut - 1, 0)
    alm = np.zeros(hp.Alm.getsize(lmax), dtype='c16')
    ia = 0
    for L in range(lcut):
        for M in range(L + 1):
            idx = hp.Alm.getidx(lmax, L, M)
            if M == 0:
                alm[idx] = coeffs[ia]; ia += 1
            else:
                alm[idx] = (coeffs[ia] - 1j * coeffs[ia + 1]) / np.sqrt(2.); ia += 2
    return alm


def ellcut_fit(data_counts: np.ndarray, randoms_counts: np.ndarray, lcut: int, rcond: float=1e-10) -> np.ndarray:
    r"""
    The angular cut :math:`L < l_{\rm cut}` (:func:`ellcut_templates`) on a catalog: the coefficients :math:`c` of the real harmonics
    (:func:`real_harmonic_maps`) fitted by weighted least squares to the projected density contrast :math:`d = D / R - 1` over the pixels
    with randoms, weights :math:`R`: :math:`c = (Y^T R Y)^{-1} Y^T (D - R)`, :math:`D` and :math:`R` the weighted counts per pixel of the data
    and of the randoms, :math:`R` normalized to the total of :math:`D` (:math:`\alpha` applied). The pattern
    :math:`\epsilon = \sum_a c_a Y_a` (``healpy.alm2map(real_to_complex_alm(c, lcut), nside)``) is then removed from the field by weighting the
    randoms by :math:`1 + \epsilon`, which leaves their total weight unchanged (:math:`Y_{00}` is in the fit).

    On a cut sky :math:`Y^T R Y` is numerically singular (combinations of the harmonics that nearly vanish on the footprint: for
    :math:`L < 13` on a DESI cap, 20 of its 169 eigenvalues below :math:`10^{-14}` of the largest), and solving it puts round-off noise of
    40-50% of :math:`\epsilon` on the footprint, with outliers of 10x. The fit is the least-squares solution of
    :math:`R^{1/2} Y^T c = R^{-1/2} (D - R)` by singular value decomposition, the singular directions below ``rcond`` times the largest
    dropped (they carry no signal on the footprint): the projection of the contrast on the harmonics restricted to the footprint.
    """
    import healpy as hp
    D, R = (np.asarray(c, dtype='f8') for c in (data_counts, randoms_counts))
    R = R * D.sum() / R.sum()
    pixels = np.flatnonzero(R > 0.)
    Y = real_harmonic_maps(lcut, hp.npix2nside(R.size), pixels=pixels)
    sq = np.sqrt(R[pixels])
    return np.linalg.lstsq((Y * sq).T, (D[pixels] - R[pixels]) / sq, rcond=rcond)[0]


def ellcut_templates(region: Template, lcut: int, counts: np.ndarray, zbin: np.ndarray=None, orthonormal: bool=True, rcond: float=1e-10) -> list:
    r"""
    Templates of an angular cut: all angular modes :math:`L < l_{\rm cut}` of the density projected over the redshift bin removed from the
    field, on the footprint, as fitted by :func:`ellcut_fit` and applied to the randoms (weights :math:`1 + \epsilon`): an integral
    constraint on the projected density, acting on the field (every leg of an :math:`n`-point function), exactly linear in the density.
    On a cut sky, "the modes :math:`L < l_{\rm cut}`" is the least-squares projection of the projected contrast on the span of the
    :math:`Y_{LM}` restricted to the footprint: one rank-one template per real harmonic :math:`Y_a` (:func:`real_harmonic_maps`),
    pattern the footprint times :math:`Y_a`, mean window the row :math:`a` of the least-squares operator
    :math:`G = (Y^T N Y)^{-1} Y^T N` (:math:`N` the randoms per pixel, the weights of the fit), response
    :math:`\lambda(s)` the radial distribution of the randoms in the bin (as the linear imaging regression, :func:`amr_templates`, with the
    harmonics as maps and no binning). :math:`Y_{00}` is part of the fit: the removed field has zero mean, as the global integral constraint.

    Parameters
    ----------
    region : Template
        The region (footprint of the randoms with the field weights, radial selection on its shells ``redges``).
    lcut : int
        Degrees removed, :math:`0 \leq L < l_{\rm cut}`: :math:`l_{\rm cut}^2` templates.
    counts : array (npix,) or (nsplit, npix)
        Weighted counts per pixel of the randoms of the fit (the weights the fit uses, e.g. the field weights), on the pixels of the
        footprint maps. Several maps are independent random subsets, as :attr:`Template.angular`; the fit uses their sum.
    zbin : array (nshell,), optional
        Indicator of the redshift bin on the shells; default all.
    orthonormal : bool, default=True
        How the removal is split into rank-one templates. The removed pattern is the same either way, the projection of the projected
        contrast on the span of the harmonics :math:`Y` restricted to the footprint, with the projector
        :math:`Y^T (Y^T N Y)^+ Y N`; the two choices differ in numerical conditioning only.

        - ``True``: write the projector in a basis :math:`E` of that span that is orthonormal with the weights :math:`N`
          (:math:`E N E^T = 1`), :math:`Y^T (Y^T N Y)^+ Y N = E^T E N`. Template :math:`k` has the pattern :math:`\Theta E_k` (the footprint
          times the basis map :math:`E_k`) and the mean window :math:`E_k N`. The basis is built in two steps: :math:`E_0` is the constant;
          the others are the singular vectors of the harmonics once the constant is projected out, those above ``rcond``. There are as
          many templates as the rank of the harmonics on the footprint, fewer than the :math:`l_{\rm cut}^2` harmonics, and each mixes
          several degrees :math:`L`. Every template but the first has zero mean over the footprint, so random combinations of them
          (:func:`sketch_templates` with ``blocks=[[0], arange(1, n)]``) never touch the survey-mean mode. That matters because the
          survey-mean mode couples to the field orders of magnitude more strongly than the modes of the cut.
        - ``False``: one template per real harmonic :math:`Y_a`, with the pattern :math:`\Theta Y_a` and the mean window the row :math:`a`
          of :math:`(Y^T N Y)^+ Y N`. On a cut sky :math:`Y^T N Y` is nearly singular, and these mean windows carry weights up to
          :math:`1 / \mathrm{rcond}^2` along the combinations of harmonics that nearly vanish on the footprint. Those large weights cancel
          only in the sum over templates, which :class:`IC2Model` cannot resolve numerically (:math:`P^{\rm tt}` off by orders of magnitude
          on a DESI cap at :math:`L < 13`). Use it only for small, well-conditioned footprints.
    rcond : float, default=1e-10
        Singular directions of :math:`N^{1/2} Y^T` below ``rcond`` times the largest are dropped, as in :func:`ellcut_fit`.

    Returns
    -------
    ic : list of Template
    """
    import healpy as hp
    import scipy.linalg
    redges = region.redges
    counts_ndim = np.ndim(counts)
    counts = np.atleast_2d(np.asarray(counts, dtype='f8'))
    nside = hp.npix2nside(counts.shape[-1])
    pixarea = 4. * np.pi / counts.shape[-1]
    total = counts.sum(axis=0)
    pixels = np.flatnonzero(total > 0.)
    Y = real_harmonic_maps(lcut, nside, pixels=pixels)  # (nbasis, npix in the fit)
    # G = (Y^T N Y)^-1 Y^T N, the scale of N dropping out: rows (Y^T N Y)^-1 Y, times N per random subset (each scaled to the total)
    footprint = np.atleast_2d(np.asarray(region.angular, dtype='f8'))
    footprint = footprint / (np.sum(footprint, axis=-1, keepdims=True) * pixarea)
    # the least-squares operator, stably (as ellcut_fit): singular value decomposition of sqrt(N) Y^T, directions below rcond dropped
    sq = np.sqrt(total[pixels])
    Yw = (Y * sq).T  # (npix in the fit, nbasis)
    if orthonormal:  # an orthonormal basis of the span on the footprint (weights N): the constant first, then the rest orthogonal to it
        q0 = sq / np.linalg.norm(sq)
        U, S, _ = np.linalg.svd(Yw - q0[:, None] * (q0 @ Yw)[None, :], full_matrices=False)
        Q = np.column_stack([q0, U[:, S > rcond * S[0]]])  # (npix in the fit, rank)
        A = Q.T / sq  # E on the fit's pixels; mean windows E N, patterns E (zero off the fit's pixels)
        Yfull = np.zeros((A.shape[0], footprint.shape[-1]))
        Yfull[:, pixels] = A
    else:  # mean windows (Y^T N Y)^+ Y N, the pseudo-inverse from the same decomposition
        _, S, Vt = np.linalg.svd(Yw, full_matrices=False)
        keep = S > rcond * S[0]
        A = (Vt[keep].T / S[keep]**2) @ Vt[keep] @ Y  # (nbasis, npix in the fit)
        Yfull = real_harmonic_maps(lcut, hp.npix2nside(footprint.shape[-1]))  # the patterns, on the footprint's pixels
    zbin = np.ones(redges.size - 1) if zbin is None else np.asarray(zbin, dtype='f8')
    radial_weights = (np.diff(np.asarray(redges) ** 3) / 3.) * region.radial * zbin
    radial_weights = radial_weights / np.sum(radial_weights)
    g = region.radial * zbin
    group = ('ellcut', id(region), int(lcut))
    out = []
    for a in range(A.shape[0]):  # the number of harmonics, or the rank on the footprint (orthonormal)
        mean = np.zeros_like(counts)
        mean[:, pixels] = A[a] * counts[:, pixels] * (total.sum() / counts.sum(axis=-1))[:, None] / pixarea
        angular = footprint * Yfull[a]
        out.append(Template(angular.reshape(np.shape(region.angular)), g, redges, mean=mean if counts_ndim > 1 else mean[0],
                            response=('outer', np.ones_like(g), radial_weights), group=group))
    return out


def sketch_templates(templates: list, nprobe: int, blocks: list=None, method: str='sign', seed: int=42) -> list:
    r"""
    Replace many rank-one templates by a few random combinations of them ("probes"), to make :class:`IC2Model` affordable: its cost and
    memory scale with the number of template patterns (two spherical-harmonic transforms per pattern and degree :math:`L'`). The templates
    must share their radial profile and response, e.g. the :math:`l_{\rm cut}^2` templates of :func:`ellcut_templates`.

    Why it works. Template :math:`a` enters the removal through its pattern :math:`\Theta_a` (where it acts on the sky) and its mean window
    :math:`U_a` (where its amplitude is measured), only through the removal kernel
    :math:`S(\hat{n}, \hat{n}') = \sum_a \Theta_a(\hat{n})\, U_a(\hat{n}')`. Each probe :math:`p` combines all templates with random signs,
    :math:`\Theta_p = \sum_a \xi_{pa} \Theta_a` and :math:`U_p = \sum_a \xi_{pa} U_a`, :math:`\xi_{pa} = \pm 1 / \sqrt{n_{\rm probe}}`. Then
    :math:`\sum_p \Theta_p \otimes U_p = \sum_a \Theta_a \otimes U_a + \sum_{a \neq b} (\sum_p \xi_{pa} \xi_{pb})\, \Theta_a \otimes U_b`,
    and the cross terms (:math:`a \neq b`) average to zero over the signs (Hutchinson's estimator): the probes reproduce :math:`S` on
    average.

    What is unbiased. Everything linear in :math:`S`: :math:`P^{\rm ft}` and :math:`P^{\rm tf}` and their shot noise, the closure kernel
    :math:`\tau_{W^2}` and its shot noise :math:`N_{W^2}`, hence :func:`compute_smooth3_ic_window` and :func:`compute_smooth3_closure_window`,
    including the global integral constraint (whose mean window is also linear in the templates'). Not :math:`P^{\rm tt}`, which is quadratic
    in the templates: for the :math:`\Delta P` of the power spectrum, use all templates.

    Scatter. One draw scatters around the truth through the cross terms. With ``blocks``, the signs are drawn within each block only (e.g.
    one block per degree :math:`L` of an angular cut): there are no cross terms between blocks, at the price of ``nprobe`` probes per block.
    Two sets of probes (two values of ``seed``) estimate the scatter.

    ``method='subsample'`` instead keeps ``nprobe`` templates of each block, drawn without replacement, their patterns scaled by the block
    size over ``nprobe``. It is also unbiased and has no cross terms; its scatter is that of the templates' contributions within a block,
    small when they respond alike (e.g. the harmonics of one degree on a wide footprint).

    Parameters
    ----------
    templates : list of Template
        Rank-one (``'outer'``) templates with the same radial profile, response and group.
    nprobe : int
        Number of probes (per block).
    blocks : list of arrays, optional
        Indices of the templates of each block; default one block.
    method : str, default='sign'
        ``'sign'``: random-sign combinations; ``'subsample'``: random subsets of each block.
    seed : int, default=42
        Random seed of the signs.

    Returns
    -------
    probes : list of Template
        ``nprobe`` templates per block, in the same group; a block of at most ``nprobe`` templates is kept as it is (exact).

    Notes
    -----
    Keep the survey mean out of the probes. A template whose pattern has a non-zero mean over the survey leaks, through the cross terms,
    into the survey-mean mode (the global integral constraint, built from the probes), which couples to the field orders of magnitude more
    strongly than the templates themselves: a single draw is then off by orders of magnitude. So probe a basis where all templates but one
    have zero survey mean, e.g. ``ellcut_templates(..., orthonormal=True)``, and keep the constant (first) template as it is, in its own
    block.

    Validation: on the angular cut :math:`L < 13` of a DESI LRG NGC footprint (169 templates), one block of 32 probes plus the constant
    (33 patterns, 5 times cheaper) gives :math:`W^{\rm geo} D^{BB}` within 0.7-1.8% (:math:`B_{000}`) and 3-4% (:math:`B_{202}`) of its peak
    per draw, against all templates, and within 0.6% and 0.5% for the mean of two draws.

    Examples
    --------
    An angular cut, orthonormal basis, the constant kept, one block of the others:

    >>> templates = ellcut_templates(geometry, lcut, counts, orthonormal=True)
    >>> probes = sketch_templates(templates, nprobe=32, blocks=[[0], np.arange(1, len(templates))])
    """
    t0 = templates[0]
    for t in templates:
        if t.kind != 'outer' or not (np.array_equal(t.radial, t0.radial) and all(np.array_equal(a, b) for a, b in zip(t.response[1:], t0.response[1:]))):
            raise ValueError('sketch_templates: rank-one templates with the same radial profile and response only')

    def normalized(arr):  # the map(s) as IC2Model uses them, before the pixel window: non-negative maps at unit integral (each random subset), else as given
        arr = np.array(arr, dtype='f8')
        for m in np.atleast_2d(arr):
            if np.all(m >= 0.) and np.sum(m) > 0.: m *= m.size / (4. * np.pi) / np.sum(m)
        return arr

    rng = np.random.default_rng(seed)
    blocks = [np.arange(len(templates))] if blocks is None else [np.asarray(b) for b in blocks]
    out = []
    for block in blocks:
        if block.size <= nprobe:  # no cheaper than the templates themselves: kept, exact
            out += [templates[a] for a in block]
            continue
        if method == 'subsample':  # nprobe templates of the block, patterns scaled by size / nprobe
            for a in rng.choice(block, size=nprobe, replace=False):
                t = templates[a]
                if np.all(np.asarray(t.angular) >= 0.): raise ValueError('sketch_templates: subsample a non-negative pattern (IC2Model would undo its scaling)')
                out.append(Template(np.asarray(t.angular, dtype='f8') * (block.size / nprobe), t.radial, t.redges, mean=t.mean, response=t.response, group=t.group))
            continue
        signs = rng.choice([-1., 1.], size=(nprobe, block.size)) / np.sqrt(nprobe)
        # the probes accumulated template by template (memory: the probes, not the block's maps)
        angulars, means = None, None
        for ia, a in enumerate(block):
            # the pattern, and the mean window as one map: the sum of its maps as IC2Model uses them, with their coefficients
            p, m = normalized(templates[a].angular), sum(c * normalized(mm) for mm, c in templates[a].mean)
            if angulars is None: angulars, means = np.zeros((nprobe,) + p.shape), np.zeros((nprobe,) + m.shape)
            angulars += signs[:, ia].reshape((-1,) + (1,) * p.ndim) * p[None]
            means += signs[:, ia].reshape((-1,) + (1,) * m.ndim) * m[None]
        for angular, mean in zip(angulars, means):
            # a non-negative pattern would be renormalized by IC2Model: flip the signs of pattern and mean window (their product is unchanged)
            if np.all(angular >= 0.): angular, mean = -angular, -mean
            # a non-negative combination would be renormalized by IC2Model: pass it at unit integral with its integral as the coefficient
            integral = np.sum(np.atleast_2d(mean), axis=-1) * 4. * np.pi / np.shape(mean)[-1]
            if np.all(mean >= 0.) and np.all(integral > 0.): mean = ((mean / integral[..., None] if np.ndim(mean) > 1 else mean / integral[0], float(np.mean(integral))),)
            if not np.any(angular): raise ValueError('sketch_templates: a probe pattern is zero')
            out.append(Template(angular, t0.radial, t0.redges, mean=mean, response=t0.response, group=t0.group))
    return out


def harmonic_templates(region: Template, zbin: np.ndarray, removed_fraction: np.ndarray, uncorrelated_power: np.ndarray=None) -> list:
    r"""
    Templates of imaging weights calibrated on mocks with a harmonic response (e.g. SYSNet), in one imaging region and redshift bin. As
    every :class:`Template`, they do not hold the weights but describe how they respond to the density (derivation: notes, Eq. 162
    [eq:amr_sysnet_template] and below): the weights remove a fraction ``removed_fraction`` of each angular mode :math:`L` of the density
    projected over the bin, plus a part independent of the density, of angular power ``uncorrelated_power``. Two templates are returned:

    - the harmonic template: footprint and radial profile of the region in the bin, response
      ``('harmonic', radial_weights, removed_fraction, uncorrelated_power)``, ``radial_weights`` the radial distribution of the region's
      randoms in the bin. It stands for one rank-one template per harmonic :math:`Y_{LM}` (pattern the footprint times :math:`Y_{LM}`, mean
      window :math:`Y^\star_{LM}`, amplitude ``removed_fraction`` at :math:`L` times the harmonic coefficient of the projected density),
      summed over :math:`(L, M)` analytically by :class:`IC2Model`;
    - the normalization of the weights to mean 1 over the region, as in the pipeline: footprint of the region, minus its radial profile,
      rank-one response with the same radial weights, and mean window the footprint with each degree :math:`L` multiplied by
      ``removed_fraction`` (the mean of the weights over the region reads the density through this map).

    With the normalization, what the weights remove has zero mean over the region, as for the centred patterns of :func:`amr_templates`:
    the survey mean removed by :math:`\alpha` is zero for these templates (:func:`gic_templates` skips them), and :class:`IC2Model` refuses a
    harmonic template without its normalization. The mean of the part independent of the density, also removed, only changes the additive
    term and is neglected.

    Parameters
    ----------
    region : Template
        The region in the cap (footprint of the randoms with the field weights, radial selection on its shells ``redges``).
    zbin : array (nshell,)
        Indicator of the redshift bin on the shells.
    removed_fraction : array (lmax + 1,)
        :math:`\kappa_R(L)`, the fraction of each angular mode :math:`L` of the projected density removed by the weights (calibrated on
        mocks), zero beyond its last degree.
    uncorrelated_power : array (lmax + 1,), optional
        :math:`C^{\perp\perp}_{R, L}`, added as the power of a pattern independent of the density. In practice it is calibrated on mocks
        as the part of the change of the density contrast caused by the weights that is not correlated with the density at the same
        degree, :math:`(C^{\epsilon\epsilon}_L - (C^{\epsilon d}_L)^2 / C^{dd}_L) / f_{\rm sky}`: the weight pattern independent of the
        density, responses to the density at other degrees, and the weights times the density (a modulation of the field); the weights
        times the data's Poisson noise, white, should be left out (the shot-noise subtraction of the weighted measurement removes it).
        Default: zero.

    Returns
    -------
    ic : list of Template
        The harmonic template and its normalization.
    """
    import healpy as hp
    redges = region.redges
    kappa = np.asarray(removed_fraction, dtype='f8')
    uncorrelated_power = np.zeros_like(kappa) if uncorrelated_power is None else np.asarray(uncorrelated_power, dtype='f8')
    # lambda_RZ(s), the radial distribution of the randoms of the region in the bin, the density being projected with it (Eq. 162)
    zbin = np.asarray(zbin, dtype='f8')
    radial_weights = (np.diff(np.asarray(redges) ** 3) / 3.) * region.radial * zbin
    radial_weights = radial_weights / np.sum(radial_weights)
    g = region.radial * zbin
    harmonic = Template(region.angular, g, redges, response=('harmonic', radial_weights, kappa, uncorrelated_power))
    # Second template: the normalization of the weights to mean 1 over the region. Its amplitude is the mean of the weight pattern over the
    # region, which reads the projected density through one map, U: the region's footprint with each angular degree L multiplied by kappa(L).
    # U is the mean window of this template, built below for each random subset of the footprint.
    U = []
    for m in np.atleast_2d(np.asarray(region.angular, dtype='f8')):
        # the footprint normalized to unit integral, as IC2Model does with every non-negative map
        if np.all(m >= 0.) and np.sum(m) > 0.: m = m / np.sum(m) * m.size / (4. * np.pi)
        nside = hp.npix2nside(m.size)
        lmax = min(kappa.size, 3 * nside) - 1
        # footprint -> harmonic coefficients (map2alm) -> each degree L multiplied by kappa(L), degrees beyond kappa dropped (almxfl) -> map
        # (alm2map). The pixel window is left in: IC2Model removes it from U, as from all maps
        U.append(hp.alm2map(hp.almxfl(hp.map2alm(m, lmax=lmax), kappa[:lmax + 1]), nside, lmax=lmax))
    U, k0 = np.reshape(U, np.shape(region.angular)), float(kappa[0])
    # A mean window is stored as (map, coefficient). IC2Model would rescale U to unit integral if it is non-negative: so pass U divided by
    # its integral, which is kappa(0) (the footprint having unit integral), with kappa(0) as the coefficient; if kappa(0) = 0, U as it is
    mean = ((U / k0, k0),) if k0 != 0. else ((U, 1.),)
    normalization = Template(region.angular, -g, redges, mean=mean, response=('outer', np.ones_like(g), radial_weights),
                             group=('harmonic normalization', id(harmonic)))
    return [harmonic, normalization]


def _is_harmonic_normalization(t: Template) -> bool:
    """Whether ``t`` is the normalization of a harmonic template (:func:`harmonic_templates`), ``group = ('harmonic normalization', id)``."""
    return isinstance(t.group, tuple) and len(t.group) == 2 and t.group[0] == 'harmonic normalization'


def gic_templates(geometry: list, ic: list) -> list:
    r"""
    Templates of the global integral constraint (GIC), what the estimator removes through
    :math:`\alpha = \sum w_{\rm data} / \sum w_{\rm randoms}` (Eqs. 124-125, Sec. V A of the notes, "Integral constraints of both arms").
    :math:`\alpha` is read on the final catalog, after the other templates are removed: it subtracts the survey mean of what they leave,
    :math:`W (\bar{\delta}_W - \bar{\delta}^{\rm t})`, with :math:`\bar{\delta}_W = \int d^3x\, W \delta / \int d^3x\, W` the survey mean of
    the density and :math:`\bar{\delta}^{\rm t} = \int d^3x \sum_a W_a \delta^{\rm t}_a / \int d^3x\, W` that of what the templates remove.
    The amplitude is a single number for the whole survey, linear in the density: rank-one (``'outer'``) templates, of mean windows the
    regions and those of the templates of ``ic``. It depends on all of them, unlike the other templates, whose amplitudes read the density
    itself.

    :class:`IC2Model` adds them by default (``gic=True``). Passed explicitly instead,
    ``IC2Model(geometry, ic + gic_templates(geometry, ic), ..., gic=False)`` is the same model; build them from the final ``ic`` (a
    template added afterwards would be missing from the survey mean).

    Parameters
    ----------
    geometry : list of Template
        The regions of the survey, as passed to :class:`IC2Model`.
    ic : list of Template
        The templates removed before :math:`\alpha`. Harmonic templates and their normalization (:func:`harmonic_templates`) are skipped:
        together they remove :math:`W_R (\epsilon_R - \langle \epsilon_R \rangle)`, of zero mean over the region. With ``ic=[]``, the GIC alone,
        :math:`W \bar{\delta}_W`.

    Returns
    -------
    gic : list of Template
        :math:`-W \bar{\delta}^{\rm t}` (one template per region and group of rank-one templates or other template of ``ic``), then
        :math:`W \bar{\delta}_W` (one per region and region whose mean it reads).
    """
    geometry = list(geometry)
    # harmonic templates and their normalization remove W_R (epsilon_R - <epsilon_R>), of zero mean over the region (exactly): nothing for alpha
    ic = [t for t in ic if t.kind != 'harmonic' and not _is_harmonic_normalization(t)]
    redges = check_common_redges(geometry=geometry, ic=ic)

    out = []
    vol = np.diff(np.asarray(redges) ** 3) / 3.  # int r^2 dr per shell
    wr = [vol * reg.radial for reg in geometry]  # int r^2 dr g_b: the weighted counts of region b per shell
    wtot = sum(np.sum(w) for w in wr)  # int d^3x W, footprints integrating to 1
    ones = np.ones(redges.size - 1)
    # \bar{delta}^t = sum_a [int d^2x Theta_a] [sum_r V_r g_a(r) delta^t_a(r)] / int W (V_r = int r^2 dr), with
    # delta^t_a(r) = sum_s Lambda_a(r, s) \bar{delta}_{U_a}(s): \bar{delta}^t = sum_a sum_s w_a(s) \bar{delta}_{U_a}(s), w_a(s) = [int Theta_a]
    # sum_r V_r g_a(r) Lambda_a(r, s) / int W the weights with which the survey mean reads the shell means of template a: a single number,
    # rank one in the sense of the 'outer' response, with radial weights w_a. Write it as few templates as possible.
    # Recap of Template.response, the map Lambda_a(r, s) from the shell means \bar{delta}_{U_a}(s) to the amplitude delta^t_a(r) at template shell r:
    # - ('diag', f): Lambda_a(r, s) = f(r) delta_rs, f = response[1] (RIC: the share of the shuffle part in shell r; 1 for a region
    #   shuffled on its own);
    # - ('dense', L): Lambda_a(r, s) = L(r, s), L = response[1] a full matrix (shell width included);
    # - ('outer', profile, radial_weights): rank one, Lambda_a(r, s) = alpha_a(r) lambda_a(s). The amplitude is a single number,
    #   c_a = sum_s lambda_a(s) \bar{delta}_{U_a}(s), with lambda_a = radial_weights = response[2] (how much each shell s counts in it, e.g.
    #   the fraction of the randoms of an imaging-regression bin in shell s, summing to 1), spread over the template shells with
    #   alpha_a = profile = response[1] (usually ones: the same coefficient at every r, the redshift bin being in g_a = Template.radial).
    merged, rows = {}, []  # rank one: {group: (radial weights, [(map, coefficient)])}; other responses: [(row of Lambda, mean)]
    for it, t in enumerate(ic):
        # int d^2x Theta_a of the map as IC2Model uses it: 1 if non-negative (normalized), else as given (first random subset)
        m = np.atleast_2d(np.asarray(t.angular, dtype='f8'))[0]
        integral = 1. if np.all(m >= 0.) and np.sum(m) > 0. else np.sum(m) * 4. * np.pi / m.size
        if t.kind == 'outer':
            # Lambda_a(r, s) = alpha_a(r) lambda_a(s), lambda_a the template's own radial weights (e.g. the fraction of the randoms of an
            # imaging-regression bin in shell s): w_a(s) = scale_a lambda_a(s), scale_a = [int Theta_a] sum_r V_r g_a(r) alpha_a(r) / int W.
            # Templates of a group share lambda_a (e.g. the maps of one imaging regression): sum_a scale_a \bar{delta}_{U_a} = \bar{delta}_U for the
            # single mean window U = sum_a scale_a U_a, so the whole group gives one template, of mean window that linear combination
            profile, radial_weights = t.response[1:]  # alpha_a(r), lambda_a(s)
            scale = integral * np.sum(vol * t.radial * profile) / wtot
            key = t.group if t.group is not None else ('_single', it)
            weights0, spec = merged.setdefault(key, (radial_weights, []))
            if not np.array_equal(weights0, radial_weights): raise ValueError(f'templates of group {key} must share their response')
            spec += [(m, scale * c) for m, c in t.mean]  # U += scale_a U_a (U_a itself possibly a combination of maps)
        else:
            # diagonal or dense Lambda_a (e.g. the RIC shell means, Lambda_a(r, s) = share(r) delta_rs): w_a(s) differs from one template to
            # the next (for the RIC, the survey fraction of the weighted randoms of region a in shell s, times the share), so each gives its
            # own rank-one template, of mean window U_a and radial weights w_a
            L = np.diag(t.response[1]) if t.kind == 'diag' else t.response[1]  # Lambda_a(r, s) as a matrix
            rows.append(((vol * t.radial) @ L * integral / wtot, t.mean))
    # -W \bar{delta}^t = -sum_b W_b \bar{delta}^t: on every region b, pattern Theta_b and radial profile -g_b (minus: the estimator adds the mean back to
    # what the templates remove), amplitude \bar{delta}^t the same for all r (profile ones), measured with the radial weights w found above
    for ib, reg in enumerate(geometry):
        for key, (radial_weights, spec) in merged.items():
            out.append(Template(reg.angular, -reg.radial, redges, mean=tuple(spec), response=('outer', ones, radial_weights),
                                group=('mean', ib, key)))
        for ir, (row, spec) in enumerate(rows):
            out.append(Template(reg.angular, -reg.radial, redges, mean=spec, response=('outer', ones, row), group=('mean', ib, ir)))
    # W \bar{delta}_W = sum_b W_b sum_jb int d^3x W_jb delta / int W, the survey mean of the density: on region b, profile +g_b, one
    # template per region jb whose mean it reads, of mean window Theta_jb and radial weights V g_jb / int W (the regions read the
    # same mean with different radial weights, so they are not merged)
    out += [Template(reg.angular, reg.radial, redges, mean=other.angular, response=('outer', ones, wr[jb] / wtot), group=('fieldmean', ib, jb))
            for ib, reg in enumerate(geometry) for jb, other in enumerate(geometry)]
    return out


# Model

@dataclass(frozen=True, init=False)
class IC2Model(object):
    r"""
    Survey geometry needed to predict how a mode-removing step biases the measured power spectrum.

    Context. Some steps of the catalog pipeline subtract from the observed density field quantities measured on the data themselves.
    Randoms whose redshifts are drawn from the data remove, in each thin redshift shell, the mean density of the data; imaging weights
    fitted on the data remove the part of the density that correlates with maps of the observing conditions. In both cases the removed
    part is a sum of *templates*: a fixed pattern on the sky and in distance (e.g. "region N, shell 12"), multiplied by a number measured
    from the data (e.g. the mean density of region N in shell 12). The power spectrum of what is left differs from that of the full field,
    by an amount that is linear in the theory power spectrum and is computed exactly here, for a Gaussian field. Everything that depends
    only on the survey geometry and on the templates is computed once, when the model is built; applying a theory is then cheap
    (:func:`compute_smooth2_ic_window`).

    Parameters
    ----------
    geometry : list of Template
        The survey regions (e.g. North, South): ``angular`` the footprint (randoms with the field weights), ``radial`` the randoms density
        per shell. The selection of a region is :math:`W = g(r) \Theta(\hat{x})`, its amplitude carried by ``radial``: non-negative maps
        (footprints, mean windows) are normalized to :math:`\int \Theta = 1`, each random subset on its own; signed maps (e.g. the
        patterns and mean windows of :func:`amr_templates`) are used as given, and must be built from normalized ones. All templates
        (geometry and ``ic``) must share their shells ``redges``, which are the model's.
    ic : list of Template
        What the pipeline removes (:func:`ric_templates`, :func:`amr_templates`, :func:`harmonic_templates`, or any list of templates).
        Weights calibrated on mocks (e.g. SYSNet) are built with :func:`harmonic_templates`, one per region :math:`R`, with
        ``response=('harmonic', radial_weights, removed_fraction, uncorrelated_power)``: each stands for the templates :math:`a = (R, L, M)`
        of pattern :math:`\Theta_R Y_{LM}`, mean window :math:`Y^\star_{LM}` and response :math:`\kappa_R(L) \lambda_R(s)`
        (``removed_fraction`` (lmax + 1,), ``radial_weights`` (nshell,)), plus a part independent of the density, of angular power
        ``uncorrelated_power`` :math:`C^{\perp\perp}_{R, L}` (lmax + 1,). They can be mixed with the other templates (e.g. SYSNet weights
        and the RIC). :func:`harmonic_templates` adds the normalization of the weights to mean 1 over the region, as in the pipeline (a
        harmonic template without it is refused): their survey mean is then zero.
    norm : float
        Normalization of the measured power spectrum (e.g. its ``norm``), so that :math:`\Delta P` is normalized as the measurement. The
        measurement normalizes the data-weighted field, the model :math:`W \delta`, with :math:`W` the selection of ``geometry``: their
        amplitudes differ by :math:`\alpha = w_{\rm sum} / \int d^3x\, W`, and ``norm`` is divided by :math:`\alpha^2` (what both share is
        :math:`(\sum w)^2 / \mathcal{N}`, the power of the window at :math:`k = 0`).
    wsum : float, optional
        Sum of the weights of the data of the measurement, :math:`w_{\rm sum}`. If not given, ``norm`` is taken as already in the units of
        ``geometry`` (normalizing :math:`W \delta`, e.g. :math:`\int d^3x\, W^2` for a toy).
    lmax : int
        Finest angular scale (spherical harmonic degree) kept for the templates: the largest :math:`k` times the largest distance, plus 20;
        capped, with a warning, at :math:`3 N_{\rm side} - 1`, the largest degree the maps resolve.
    gic : bool, default=True
        Add the global integral constraint (GIC), what the estimator removes through :math:`\alpha`, as the last templates
        (:func:`gic_templates`): the survey mean of what the other templates leave, :math:`W (\bar{\delta}_W - \bar{\delta}^{\rm t})`
        (zero for the harmonic templates, normalized over their region). With ``ic=[]``, the GIC alone. ``gic=False``
        with ``ic + gic_templates(geometry, ic)`` is the same model. :math:`\Delta P` is relative to the field without any removal, :math:`W \delta` (the
        plain window convolution); relative to the field with :math:`\alpha` only (the merged arm of the mocks), subtract the model with
        ``ic=[]``.
    deconvolve : bool, default=True
        Remove the HEALPix pixel window from the maps.
    theta : array, optional
        Angular separations (rad) at which correlations between two points on the sky are integrated; default 2000 points below 0.05 rad,
        2000 above.
    ds : float, default=0.25
        Spacing of the separations at which the theory correlation function is tabulated.
    splits : tuple, default=(0, 1, 2)
        Which independent subsets of randoms are used for (template pattern, region footprint / first mean window, second mean window), so
        that a random is never paired with itself (which would add the randoms' own shot noise to the prediction).
    iter : int, default=0
        Jacobi iterations of the HEALPix analysis (``healpy.map2alm``) of the products of maps in the pair-count functions of the field
        and the templates, one per template pattern, degree and region, the dominant cost of the model: each iteration adds a synthesis and
        an analysis. With 0, the analysis is the sum over the pixels, the same quadrature as the zero-separation pairs (``auto_pairs``); 3
        changes the windows by ~2e-4 (relative) and doubles the time of the model.
    mpicomm : MPI communicator, optional
        Split the costly loops (over the template patterns, and over the mean windows for the harmonic x regular coupling) over the MPI
        ranks, the results summed over them; all ranks get the same model.

    Attributes
    ----------
    redges : array (nshell + 1,)
        The shells of the templates.
    geometry, norm, lmax, theta, ds, splits
        As the parameters.
    separations : array
        Separations, spaced by ``ds``, at which the theory correlation function multipoles are tabulated, from 0 to beyond the largest
        distance between two shells.
    removed : dict
        ``{name: list of Template}``, the removed fields of Eq. 126 [eq:ric_cross_auto], the field being :math:`\delta^{\rm f} = W \delta`.
        ``'t'``: the removed field :math:`\delta^{\rm t}`, the templates of the pipeline and, unless ``gic=False``, the :math:`\alpha`
        template, :math:`W \bar{\delta}_W` (one template per region and region whose mean it reads) and :math:`-W \bar{\delta}^{\rm t}` (one per
        region and group of rank-one templates or other template); ``'h'``: the templates with a harmonic response, also part of
        :math:`\delta^{\rm t}`.
    pairs : dict
        Precomputed geometric factors, one entry per pair of fields whose cross power spectrum is needed (theory independent):

        - ``('f', 't')``: the full field against the removed field. ``cross_pairs`` (nregion, ngroup, lmax + 1, ntheta) counts, at each angular
          separation, the pairs (point of a region, point used to measure a template), resolved by the angular scale of the template's sky
          pattern: it fixes how much of the field correlates with each template. It is only the angular factor of the kernel
          :math:`K^{ba}_{L'}(r, s) = \int d\theta\, \Pi^{ba}_{L'}(\theta)\, \xi(r, s, \theta)` of Eq. 132 [eq:ric_field_removed]; the distances
          (:math:`\xi`, then :math:`g_b(r)`, :math:`g_a(r')`, :math:`\Lambda_a(r', s)`, giving :math:`S^{\rm ft}_{L'}(r, r')`) enter later, in
          :func:`compute_smooth2_ic_window`. The ``lmax + 1`` axis is the degree :math:`L' = 0..\,lmax` of :math:`S_{L'}(r, r')` (Eq. 130
          [eq:ric_poles]): only the degree-:math:`L'` part of the pattern, :math:`\sum_{M'} \Theta_{a, L'M'} Y_{L'M'}`, is kept at the field point; :math:`L'`
          enters the multipoles through :math:`j_{L'}(kr')`, negligible beyond :math:`L' \sim k r'`, hence ``lmax``. The degree of the pair
          counts themselves is summed into ``ntheta``. ``auto_pairs`` (same shape without ``ntheta``) is the same at zero separation, for the shot
          noise. ``groups`` lists the templates that share their radial behaviour and are summed together.
        - ``('t', 't')``: the removed field against itself, one entry per pair of templates (not per group), ``n1 = n2 = len(removed['t'])``:
          the regular templates of ``ic`` plus, if ``gic``, the :math:`\alpha` template (nregion x (nregion + one per group of
          ``'outer'`` templates + one per ``'diag'`` / ``'dense'`` template)). ``cross_pairs`` (n1, n2, ntheta) counts, at each angular separation,
          the pairs of points used to measure two templates: it fixes how the measured template numbers correlate with each other;
          ``auto_pairs`` (n1, n2) is the same at zero separation, for the shot noise; ``C`` (n1, n2, lmax + 1) is the overlap of the two
          templates' sky patterns at each degree :math:`L' = 0..\,lmax`, :math:`C_{L'}[\Theta_{a_1}, \Theta_{a_2}]`.
        - ``('f', 'h')``, ``('h', 'h')``, ``kind='harmonic'``: ``M`` (nregion, nh, lmax + 1, lmax + 1) and (nh, lmax + 1, lmax + 1), how the
          footprints mix the angular modes of the field and of the harmonic templates: last two axes (:math:`L'`, :math:`L`), the degree of
          the pair and that of the removed mode, both ``0..lmax``.
        - ``('h', 't')``, ``kind='harmonic'``: ``N`` (nh, nt, lmax + 1, lmax + 1), how the angular modes the harmonic templates remove
          (degree :math:`L`) reach a regular template, through its mean window, at degree :math:`L'` of the pair:
          :math:`N^{Ra}_{L'L} = C_{L'}[\Theta_R \sum_M U_{a, LM} Y_{LM}, \Theta_a]`, with the degree-:math:`L` part of the mean window.
    """
    redges: np.ndarray
    geometry: list
    removed: dict
    norm: float
    lmax: int
    theta: np.ndarray
    ds: float
    splits: tuple
    separations: np.ndarray
    pairs: dict

    def __init__(self, geometry: list, ic: list, norm: float, lmax: int, wsum: float=None, gic: bool=True,
                 deconvolve: bool=True, theta: np.ndarray=None, ds: float=0.25, splits: tuple=(0, 1, 2), iter: int=0, mpicomm=None):
        import healpy as hp
        t0 = time.time()
        geometry, ic = list(geometry), list(ic)
        redges = check_common_redges(geometry=geometry, ic=ic)  # everything radial is tabulated on these shells
        normalized = {t.group[1] for t in ic if _is_harmonic_normalization(t)}
        if any(t.kind == 'harmonic' and id(t) not in normalized for t in ic):
            raise ValueError('a harmonic template must come with the normalization of the weights to mean 1 over its region: build it with harmonic_templates')
        # Global integral constraint (gic=True, gic_templates): alpha, read on the final catalog, removes the survey mean of what the other
        # templates leave, as rank-one templates added last. The field is then W delta, the plain window convolution: Delta P includes the GIC
        # (with ic=[], it is the GIC alone). Built first: the survey means of harmonic templates have their own mean windows
        alpha = gic_templates(geometry, ic) if gic else []
        # Sky maps ready for the model, {id(array): (alms, maps, integrals)}, one entry per random subset: non-negative maps (footprints, mean
        # windows) normalized to int Theta = 1, signed maps as given, the pixel window deconvolved, the maps resynthesized from the deconvolved
        # harmonics; the integrals are those of the input maps. Each array is processed once
        import healpy as hp
        maps = {}
        for arr in [t.angular for t in geometry + ic + alpha] + [m for t in ic + alpha if t.mean is not None for m, _ in t.mean]:
            if id(arr) in maps: continue
            subsets = np.atleast_2d(np.asarray(arr, dtype='f8'))  # the random subsets, (nsplit, npix)
            nside = hp.npix2nside(subsets.shape[-1])
            pixwin = hp.pixwin(nside, lmax=3 * nside - 1)
            alms, ms, integrals = [], [], []
            for m in subsets:
                # int d^2x Theta = 1 for non-negative maps (footprints, mean windows: the amplitude of W = g Theta is in g, a mean window averages)
                if np.all(m >= 0.) and np.sum(m) > 0.: m = m / np.sum(m) * m.size / (4. * np.pi)
                integrals.append(np.sum(m) * 4. * np.pi / m.size)
                alm = hp.map2alm(m, lmax=3 * nside - 1)
                if deconvolve:  # pixel window (notes, Sec. V E, "Numerical evaluation", maps)
                    alm = hp.almxfl(alm, 1. / pixwin)
                    m = hp.alm2map(alm, nside, lmax=3 * nside - 1)
                alms.append(alm)
                ms.append(m)
            maps[id(arr)] = (alms, ms, integrals)

        def get(arr, split):  # (alm, map) of a random subset of a map (the first one if there are fewer)
            alms, ms, _ = maps[id(arr)]
            split = split if split < len(alms) else 0
            return alms[split], ms[split]

        if theta is None: theta = np.unique(np.concatenate([np.linspace(0., 0.05, 2000), np.linspace(0.05, np.pi, 2000)]))
        theta = np.asarray(theta, dtype='f8')
        # the integrals over the pair separations run over theta, with weights dtheta sin(theta) (np.gradient): it must span [0, pi], increasing
        if not (np.isclose(theta[0], 0.) and np.isclose(theta[-1], np.pi) and np.all(np.diff(theta) > 0.)):
            raise ValueError('theta must increase from 0 to pi')
        nside = hp.npix2nside(get(geometry[0].angular, 0)[1].size)
        if lmax > 3 * nside - 1:  # the maps resolve degrees up to 3 nside - 1 only
            logger.warning(f'lmax = {lmax} capped at 3 nside - 1 = {3 * nside - 1} (nside = {nside} of the maps): the angular modes above are not modelled')
            lmax = 3 * nside - 1
        lm, pixarea = 3 * nside - 1, 4. * np.pi / (12 * nside**2)
        s0, s1, s2 = splits
        # separations at which the theory xi_ell(s) is tabulated: up to the largest distance between two shells, plus margin
        separations = np.arange(0., redges[-1] + redges[-2] + 4. * ds, ds)
        if wsum is not None:  # the measured normalization, of the data-weighted field, in the units of W delta: divided by alpha^2
            integral = sum(np.sum((np.diff(np.asarray(redges) ** 3) / 3.) * reg.radial) for reg in geometry)  # int d^3x W, footprints integrating to 1
            norm = norm * (integral / wsum)**2
        attrs = dict(redges=redges, geometry=geometry, norm=float(norm), lmax=lmax, theta=theta, ds=float(ds), splits=tuple(splits),
                     separations=separations)

        harmonic = [t for t in ic if t.kind == 'harmonic']  # calibrated harmonic responses, handled at the end
        ic = [t for t in ic if t.kind != 'harmonic']

        # Nothing here depends on the data realization. A template is not given by its measured amplitude delta^t_a but by how that amplitude
        # is obtained from the density: delta^t_a(r) = int ds Lambda_a(r, s) \bar{delta}_{U_a}(s), \bar{delta}_{U_a}(s) = int d^2x U_a delta(s x) the
        # mean of the density over the mean window U_a in shell s. This is linear in delta, and the model only needs the linear map
        # (U_a, Lambda_a): for a Gaussian field, the expected power spectrum of what is removed is a quadratic form of that map with the theory
        # correlation function.
        removed, pairs = {'t': ic + alpha}, {}

        # Mean windows. Template.mean stores a mean window U_a as a linear combination of maps, ((map, coefficient), ...): a single map for
        # most templates (e.g. a shuffle region), several for the survey means of a group of rank-one templates built above
        # (U = sum_a scale_a U_a) or for imaging-regression templates. The maps were processed one by one (IC2Model.__init__: normalized if
        # non-negative, pixel window removed), and harmonic transforms are linear, so the harmonic coefficients and the map of the combination
        # are the same combination of those of its maps. A mean window is used by several templates and regions: built once per random subset.
        means = {}  # cache: {(mean key, split): (alm, map)}
        mean_key = lambda spec: tuple((id(m), c) for m, c in spec)  # identifies a mean window: its maps (by identity) and coefficients

        def get_mean(spec, split):  # (alm, map) of the mean window spec = ((map, coefficient), ...) from random subset split
            key = (mean_key(spec), split)
            if key not in means:
                alms, ms = zip(*[get(m, split) for m, _ in spec])  # each map, processed
                means[key] = (sum(c * a for (_, c), a in zip(spec, alms)), sum(c * m for (_, c), m in zip(spec, ms)))  # sum_i c_i map_i
            return means[key]

        # Geometric factors of the regular templates (all but the harmonic ones), theory independent. Each term of Delta P, ('f', 't') and
        # ('t', 't'), correlates the density at the directions n seen by one field with the density at the directions m
        # seen by the other, e.g. a point of region b with a point of the mean window U_a over which a template amplitude is measured. The
        # theory xi only depends on the angle theta between n and m (and on the two distances), so all the sky geometry reduces to one function
        # of theta: the two sky maps phi(n), psi(m) multiplied and integrated over all pairs of directions at angle theta,
        #     Pi^{phi psi}(theta) dtheta = int d^2n d^2m phi(n) psi(m) [angle(n, m) in dtheta],
        # (the pair-count function: with maps of randoms, the weighted number of random pairs at that separation). It follows from the cross
        # angular power of the two maps, C^{phi psi}_l (pair_counts below). The theory enters later, in compute_smooth2_ic_window:
        # xi^{phi psi}(r, s) = int dtheta Pi^{phi psi}(theta) xi(r, s, theta). Skipped if only harmonic templates are removed
        if removed['t']:
            wtheta = jnp.asarray(np.gradient(theta) * np.sin(theta))  # dtheta sin(theta): the measure of the integrals over theta

            # Pi^{phi psi}(theta) dtheta sin(theta): the number of phi--psi pairs at angle theta, 2 pi sum_l (2l + 1) C^{phi psi}_l L_l(cos theta), with
            # which the shell kernels xi^{phi psi}(r, s) are one integral over theta (Eq. 131; Sec. V E, "Numerical evaluation", shell kernels)
            def pair_counts(cls):  # dtheta sin(theta) 2 pi sum_l (2l + 1) C_l L_l(cos theta), (..., lmax + 1) -> (..., ntheta)
                cls = jnp.asarray(cls)
                ell = jnp.arange(cls.shape[-1])
                return np.asarray(wtheta * (2. * np.pi * ((2 * ell + 1) * cls) @ _legendre_matrix(cls.shape[-1] - 1, np.cos(theta))))

            # Field x removed (Eq. 132 [eq:ric_field_removed]). The field in region b correlates with template a through the points of the mean
            # window U_a, where the template's amplitude is measured; at degree L' of the template leg (S_{L'}, Eq. 130), only the degree-L' part
            # of the pattern Theta_a enters. With the two sky maps
            #     phi = Theta_b sum_M' Theta_{a, L'M'} Y_{L'M'},   psi = U_a,
            # the code below computes, for each region b, template a and degree L' (theory independent; summed over the templates of a group):
            #     C^{phi psi}_l = sum_m phi_lm psi*_lm / (2l + 1), their cross angular power  -> pairs['f', 't']['cross_pairs'] = pair_counts(C^{phi psi}),
            #                                                                                   the pair-count function of phi and psi;
            #     int d^2n phi psi, the pairs at zero separation (shot noise)                   -> pairs['f', 't']['auto_pairs'].
            # The theory enters in compute_smooth2_ic_window, which integrates cross_pairs against xi(r, s, theta). One SHT per pattern and degree L'. The
            # three maps come from different random subsets (s0: Theta_a, s1: Theta_b, s2: U_a), so that no random pairs with itself
            needed = {}  # each pattern once: {id(Theta_a): (Theta_a, {(region b, mean window key): mean window})}
            for t in removed['t']:
                entry = needed.setdefault(id(t.angular), (t.angular, {}))
                for ib in range(len(geometry)): entry[1][ib, mean_key(t.mean)] = t.mean
            cls = {}  # {(b, id(Theta_a), mean key): (C^{phi psi}_l for each L' (lmax + 1, lm + 1), int phi psi for each L' (lmax + 1,))}
            for ipat, (pattern, region_means) in enumerate(needed.values()):  # one template pattern Theta_a at a time
                if mpicomm is not None and ipat % mpicomm.size != mpicomm.rank: continue  # another rank's pattern
                alm_a = get(pattern, s0)[0]  # its harmonic coefficients Theta_{a, LM}, from random subset s0
                # output for each (region b, mean window) this pattern is used with: C^{phi psi}_l per L' and the zero-separation term per L'
                for ib, mk in region_means: cls[ib, id(pattern), mk] = (np.zeros((lmax + 1, lm + 1)), np.zeros(lmax + 1))
                for Lp in range(lmax + 1):  # degree L' of S_{L'}(r, r') (Eq. 130), angular factor only; lmax set by the largest k r
                    # sum_M' Theta_{a, L'M'} Y_{L'M'}: the degree-L' part of the pattern, as a map; synthesized up to degree L' only (the same
                    # map, the cost of the synthesis growing with the degree)
                    alm_Lp = hp.almxfl(hp.resize_alm(alm_a, lm, lm, Lp, Lp), np.eye(Lp + 1)[Lp])
                    pattern_Lp = hp.alm2map(alm_Lp, nside, lmax=Lp)
                    for ib in sorted(set(ib for ib, _ in region_means)):  # each region b the field is in
                        # phi = Theta_b sum_M' Theta_{a, L'M'} Y_{L'M'}: footprint of b (random subset s1) times the degree-L' part of the pattern, on the pixels
                        prod = get(geometry[ib].angular, s1)[1] * pattern_Lp
                        alm_prod = hp.map2alm(prod, lmax=lm, iter=iter)  # phi_{lm}, all degrees l (the product mixes degrees)
                        for (jb, mk), spec in region_means.items():  # each mean window U_a used with this pattern (and region b)
                            if jb != ib: continue
                            alm_U, map_U = get_mean(spec, s2)  # psi = U_a, from random subset s2 (a linear combination of maps if merged)
                            # C^{phi psi}_l = sum_m phi_{lm} psi*_{lm} / (2l + 1): turned into the pair-count function of phi and psi at each
                            # angle theta by pair_counts, then into K^{ba}_{L'}(r, s) with the theory xi in compute_smooth2_ic_window
                            cls[ib, id(pattern), mk][0][Lp] = hp.alm2cl(alm_prod, alm_U)
                            # pairs at zero separation, int d^2n phi psi: the shot noise, which adds N delta_D(r - s) / r^2 int phi psi to
                            # K^{ba}_{L'} (Eq. 137, "Shot noise"; the same galaxy in the field and in the mean)
                            cls[ib, id(pattern), mk][1][Lp] = np.sum(prod * map_U) * pixarea
                logger.info(f'pair-count functions: template pattern {ipat + 1} / {len(needed)} ({time.time() - t0:.0f} s)')
            # templates of a group share g_a and Lambda_a: S^ft is linear in K^{ba}, so their cls add
            groups = {}
            for t in removed['t']: groups.setdefault(t.group if t.group is not None else ('_single', id(t)), []).append(t)
            groups = [(members[0], members) for members in groups.values()]
            cl, auto = np.zeros((len(geometry), len(groups), lmax + 1, lm + 1)), np.zeros((len(geometry), len(groups), lmax + 1))
            for ib in range(len(geometry)):
                for ig, (_, members) in enumerate(groups):
                    for t in members:
                        key = (ib, id(t.angular), mean_key(t.mean))
                        if key not in cls: continue  # computed by another rank
                        c, tr = cls[key]
                        cl[ib, ig] += c
                        auto[ib, ig] += tr
            if mpicomm is not None:  # the sums of all ranks' patterns
                cl, auto = mpicomm.allreduce(cl), mpicomm.allreduce(auto)
            pairs['f', 't'] = dict(kind='K', groups=groups, cross_pairs=pair_counts(cl), auto_pairs=auto)
            # Removed x removed, Eq. 133 [eq:ric_removed_removed]: xi^{uu'} = xi^{phi psi} for phi = U_a, psi = U_a' (pair counts of the mean windows),
            # the patterns through sum_M Theta_{a, L'M} Theta*_{a', L'M} = (2L' + 1) C^{Theta_a Theta_a'}_{L'}; int U_a U_a' for the shot noise
            for name1, name2 in [('t', 't')]:
                t1, t2 = removed[name1], removed[name2]
                cl, auto, C = [], [], []
                for a1 in t1:
                    U1, T1 = get_mean(a1.mean, s1), get(a1.angular, s0)[0]
                    for a2 in t2:
                        U2, T2 = get_mean(a2.mean, s2), get(a2.angular, s1)[0]
                        cl.append(hp.alm2cl(U1[0], U2[0]))
                        auto.append(np.sum(U1[1] * U2[1]) * pixarea)
                        C.append(hp.alm2cl(T1, T2)[:lmax + 1])
                n1, n2 = len(t1), len(t2)
                pairs[name1, name2] = dict(kind='M', cross_pairs=pair_counts(np.array(cl)).reshape(n1, n2, -1),
                                           auto_pairs=np.array(auto).reshape(n1, n2),
                                           C=np.array(C).reshape(n1, n2, -1))
                logger.info(f'pair-count functions: {name1} x {name2} done ({time.time() - t0:.0f} s)')
        if harmonic:
            # Calibrated harmonic response (Eq. 162 [eq:amr_sysnet_template]): the sums over the harmonics are analytic, through the mode-coupling
            # matrices M^{bR}_{L'L} of the footprints, Eq. 166 [eq:amr_sysnet] (their survey mean is a regular template, gic_templates)
            removed['h'] = harmonic

            # Mode-coupling matrix of two footprints, M^{bR}_{L'L} (Eq. 165, in Eq. 166 [eq:amr_sysnet]):
            #     M_{L'L} = (2L + 1) / 2 int dx L_L'(x) L_L(x) xi(x),   xi(x) = sum_l (2l + 1) / (4 pi) C_l L_l(x),
            # with C_l the cross angular power of the two footprints, xi their correlation function and L the Legendre polynomials; the integral
            # by Gauss-Legendre quadrature. It satisfies sum_MM' (Theta_b Y_LM)*_{L'M'} (Theta_R Y_LM)_{L'M'} = (2L' + 1) M_{L'L}
            x, wx = np.polynomial.legendre.leggauss(2 * lmax + 10)
            PL = _legendre_matrix(lmax, x)

            def coupling(arr1, arr2):
                # only l <= L + L' <= 2 lmax contributes to M_{L'L}: C_l cut there, so that the quadrature (exact up to degree 4 lmax + 19) is exact
                cl = jnp.asarray(hp.alm2cl(get(arr1, s0)[0], get(arr2, s1)[0]), dtype='f8')[:2 * lmax + 1]
                xi = ((2 * jnp.arange(cl.size) + 1) / (4. * np.pi) * cl) @ _legendre_matrix(cl.size - 1, x)
                return np.asarray((PL * (wx * xi)[None, :]) @ PL.T * (2 * jnp.arange(lmax + 1) + 1)[None, :] / 2.)

            pairs['f', 'h'] = dict(kind='harmonic', M=np.array([[coupling(reg.angular, t.angular) for t in harmonic] for reg in geometry]))
            pairs['h', 'h'] = dict(kind='harmonic', M=np.array([coupling(t.angular, t.angular) for t in harmonic]))
            if removed['t']:
                # Harmonic x regular templates (beyond Eq. 166): <W_R(x) epsilon_R(x) W_a(x') delta^t_a(r')> = g_R(r) g_a(r') Theta_R(n) Theta_a(n')
                # sum_L kappa_R(L) c_L(r') sum_M U_{a, LM} Y_LM(n), the degree-L part of the mean window of a, and
                # c_L(r') = int ds' Lambda_a(r', s') int ds lambda_R(s) C_L(s', s); the product of the two angular factors projects on degree L'
                # as sum_M' (Theta_R sum_M U_{a, LM} Y_LM)_{L'M'} Theta*_{a, L'M'} = (2L' + 1) N^{Ra}_{L'L}. One SHT per mean window, degree and region
                resize = lambda alm: hp.resize_alm(alm, lm, lm, lmax, lmax)
                N = np.zeros((len(harmonic), len(removed['t']), lmax + 1, lmax + 1))
                by_mean = {}
                for ia, a in enumerate(removed['t']): by_mean.setdefault(mean_key(a.mean), (a.mean, []))[1].append(ia)
                theta_a = [resize(get(a.angular, s1)[0]) for a in removed['t']]
                theta_R = [get(t.angular, s0)[1] for t in harmonic]
                for imean, (spec, members) in enumerate(by_mean.values()):
                    if mpicomm is not None and imean % mpicomm.size != mpicomm.rank: continue  # another rank's mean window
                    alm_U = resize(get_mean(spec, s2)[0])
                    for L in range(lmax + 1):
                        fl = np.zeros(lmax + 1)
                        fl[L] = 1.
                        U_L = hp.alm2map(hp.almxfl(alm_U, fl), nside, lmax=lmax)
                        for iR, m_R in enumerate(theta_R):
                            alm_prod = hp.map2alm(m_R * U_L, lmax=lmax)
                            for ia in members: N[iR, ia, :, L] = hp.alm2cl(alm_prod, theta_a[ia])
                    logger.info(f'harmonic x regular templates: mean window {imean + 1} / {len(by_mean)} ({time.time() - t0:.0f} s)')
                if mpicomm is not None: N = mpicomm.allreduce(N)  # each entry computed by one rank, zero elsewhere
                pairs['h', 't'] = dict(kind='harmonic', N=N)
        self.__dict__.update(removed=removed, pairs=pairs, **attrs)


# Theory-dependent part (JAX)

def _correlation_rows(r, shell_centers, theta, separations, xi, ellsin):
    r"""
    Correlation function xi_t(r, s, theta) = sum_l xi_{t, l}(|Delta x|) L_l(mu) of several theories t, between the points at distances
    ``r`` (a few shells) and ``shell_centers`` (all shells s) separated by the angles ``theta``, plane parallel for each pair of points, mu the
    cosine to the bisector of the two lines of sight (Eq. 136); ``xi`` (n_theories, len(ellsin), n_separations) is tabulated at the uniformly
    spaced ``separations``, linearly interpolated in between. Returns (n_theories, r.size, n_shells, n_theta).
    """
    r, s = r[:, None, None], shell_centers[None, :, None]
    cos_theta, cos_half_theta = jnp.cos(theta), jnp.cos(theta / 2.)
    # separation |Delta x|^2 = r^2 + s^2 - 2 r s cos(theta), and mu = (r - s) cos(theta / 2) / |Delta x| (Eq. 136)
    separation = jnp.sqrt(jnp.maximum(r**2 + s**2 - 2. * r * s * cos_theta, 0.))
    mu = (r - s) * cos_half_theta / jnp.maximum(separation, 1e-10)
    legendre = [jnp.ones_like(mu), mu]  # Legendre polynomials L_l(mu), l = 0 .. max(ellsin), upward recurrence
    for ell in range(1, max(ellsin)): legendre.append(((2 * ell + 1) * mu * legendre[ell] - ell * legendre[ell - 1]) / (ell + 1))
    # linear interpolation of xi_{t, l} at |Delta x| in the table of separations, uniformly spaced: the bracketing index and the weight follow
    # from the spacing (a search in the table is ~70 times slower), once for all theories and multipoles; outside the table, the end values
    position = (separation - separations[0]) / (separations[1] - separations[0])
    index = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, separations.size - 2)
    fraction = jnp.clip(position - index, 0., 1.)
    return sum((xi[:, il][:, index] * (1. - fraction) + xi[:, il][:, index + 1] * fraction) * legendre[ell] for il, ell in enumerate(ellsin))


def _angular_kernels(shell_centers, theta, separations, xi, ellsin, full_pair_counts=None, radial_weights=None, rank_one_pair_counts=None,
                     rows_per_chunk: int=8):
    r"""
    Angular integrals of the theory correlation function, for several theories at once:

        full_kernels[t, r, s, c] = sum_theta full_pair_counts[theta, c] xi_t(r, s, theta),
        rank_one_kernels[t, w, r, c] = sum_s radial_weights[w, s] sum_theta rank_one_pair_counts_w[theta, c] xi_t(r, s, theta),

    with xi_t(r, s, theta) the correlation function of theory t between the points at distances r, s (shell centres) separated by the angle
    theta: sum_l xi_{t, l}(|Delta x|) L_l(mu), plane parallel for each pair of points, mu the cosine to the bisector of the two lines of
    sight (Eq. 136). Each column c is one pair-count function of theta.

    Parameters
    ----------
    shell_centers : array (n_shells,)
        Distances r, s.
    theta : array (n_theta,)
        Angles at which the pair-count functions are tabulated.
    separations : array (n_separations,)
        Separations at which ``xi`` is tabulated, uniformly spaced (``IC2Model.separations``); linearly interpolated in between.
    xi : array (n_theories, len(ellsin), n_separations)
        Correlation function multipoles xi_{t, l} of each theory.
    ellsin : tuple
        Multipoles l of ``xi``.
    full_pair_counts : array (n_theta, n_full), optional
        The pair-count functions of the n_full "full" columns.
    radial_weights : array (n_weights, n_shells), optional
        The radial weights lambda_w(s) of the "rank-one" columns.
    rank_one_pair_counts : array (n_theta, n_columns) or list of arrays (n_theta, n_columns_w), optional
        The pair-count functions of the rank-one columns: the same for all radial weights (one array), or one array per radial weight w
        (a list, each its own number of columns n_columns_w; the output is padded with zeros to n_columns, the largest).
    rows_per_chunk : int, default=8
        Number of shells r processed together (memory).

    Returns
    -------
    full_kernels : array (n_theories, n_shells, n_shells, n_full), or None if no full columns
    rank_one_kernels : array (n_theories, n_weights, n_shells, n_columns), or None if no rank-one columns

    xi_t(r, s, theta) is evaluated once per chunk of shells r and contracted with all the columns at once: it is the costly part.
    """
    shell_centers = jnp.asarray(shell_centers)
    n_shells = shell_centers.size
    n_chunks = -(-n_shells // rows_per_chunk)
    # shell indices of each chunk, the last chunk padded with the last shell
    chunk_shells = jnp.concatenate([jnp.arange(n_shells), jnp.full(n_chunks * rows_per_chunk - n_shells, n_shells - 1)]).reshape(n_chunks, rows_per_chunk)
    has_full = full_pair_counts is not None and full_pair_counts.shape[-1] > 0
    has_rank_one = rank_one_pair_counts is not None and radial_weights is not None and radial_weights.shape[0] > 0
    shared_columns = has_rank_one and not isinstance(rank_one_pair_counts, (list, tuple))  # the same columns for all radial weights
    if has_rank_one and not shared_columns:
        if len(rank_one_pair_counts) != radial_weights.shape[0]:
            raise ValueError(f'one block of pair-count columns per radial weight: {len(rank_one_pair_counts)} != {radial_weights.shape[0]}')
        n_rank_one_columns = max(columns.shape[-1] for columns in rank_one_pair_counts)

    def integrate_chunk(ishells):  # a few shells r at a time, all shells s and angles theta
        xi_rs = _correlation_rows(shell_centers[ishells], shell_centers, theta, separations, xi, ellsin)  # (n_theories, rows_per_chunk, n_shells, n_theta)
        out = []
        # full columns: sum over theta of the pair counts times xi(r, s, theta)
        if has_full: out.append(jnp.einsum('trsq,qc->trsc', xi_rs, full_pair_counts))
        # rank-one columns: sum over s against lambda(s) first, then over theta (Sec. VI A, "Numerical evaluation": ~200 times faster)
        if has_rank_one:
            xi_weighted = jnp.einsum('trsq,ws->twrq', xi_rs, radial_weights)  # (n_theories, n_weights, rows_per_chunk, n_theta)
            if shared_columns: out.append(jnp.einsum('twrq,qc->twrc', xi_weighted, rank_one_pair_counts))
            else:  # each radial weight with its own columns, padded to the largest number of columns
                out.append(jnp.stack([jnp.pad(xi_weighted[:, iw] @ columns, ((0, 0), (0, 0), (0, n_rank_one_columns - columns.shape[-1])))
                                      for iw, columns in enumerate(rank_one_pair_counts)], axis=1))
        return tuple(out)

    results = list(jax.lax.map(integrate_chunk, chunk_shells))
    full_kernels = rank_one_kernels = None
    if has_full:  # (n_chunks, n_theories, rows_per_chunk, n_shells, n_full) -> (n_theories, n_shells, n_shells, n_full)
        full_kernels = results.pop(0)
        full_kernels = jnp.moveaxis(full_kernels, 0, 1).reshape(full_kernels.shape[1], n_chunks * rows_per_chunk, n_shells, -1)[:, :n_shells]
    if has_rank_one:  # (n_chunks, n_theories, n_weights, rows_per_chunk, n_columns) -> (n_theories, n_weights, n_shells, n_columns)
        rank_one_kernels = results.pop(0)
        rank_one_kernels = jnp.moveaxis(rank_one_kernels, 0, 2).reshape(rank_one_kernels.shape[1], rank_one_kernels.shape[2],
                                                                         n_chunks * rows_per_chunk, -1)[:, :, :n_shells]
    return full_kernels, rank_one_kernels


def _bessel_projector(redges, lmax: int, norm: float, k, ells):
    r"""
    Function ``project(S, V, transpose=False)`` projecting an :math:`S_{L'}(r, r')` stack ``(nL, nr, nr)`` onto the power spectrum
    multipoles ``ells`` at ``k``, and the table ``V`` of :math:`r^2 dr\, j_L(kr)` it takes (an argument, not a constant of the function:
    XLA compiles large constants slowly):
    :math:`\frac{4\pi (2\ell + 1)}{\mathcal{N}} \sum_{L, L'} i^{L' - L} h^\ell_{LL'} \int dr\, dr'\, r^2 r'^2 j_L(kr) j_{L'}(kr') S_{L'}(r, r')`,
    with :math:`h^\ell_{LL'} = (2L + 1) / 2 \int dx \mathcal{L}_\ell \mathcal{L}_L \mathcal{L}_{L'}` (:math:`\mathcal{L}` the Legendre
    polynomials; Gauss-Legendre, exact); rows of :math:`S` carry the line of sight (``transpose=True``: the columns).
    """
    # Eq. 130 [eq:ric_poles]. L' <= lmax on the template leg, L = L' + d with |d| <= ell on the other (h^ell_{LL'} = 0 otherwise)
    nL, lmaxV = lmax + 1, lmax + max(ells)
    rc = (redges[1:] + redges[:-1]) / 2.
    k = jnp.asarray(k)
    k = jnp.where(k > 0, k, 1e-6)
    V = _spherical_jn_all(lmaxV, (k[:, None] * rc[None, :]).ravel()).reshape(lmaxV + 1, k.size, rc.size)
    V = jnp.transpose(V, (0, 2, 1)) * (np.diff(np.asarray(redges) ** 3) / 3.)[None, :, None]  # int r^2 dr j_L(kr), j_L at the shell centre, (lmaxV + 1, nr, nk)
    # padded with max(ells) zero degrees in front: the degrees L = L' + d of an offset d are the slice V[pad + d:pad + d + nL'], zero for L < 0
    pad = max(ells)
    V = jnp.pad(V, ((pad, 0), (0, 0), (0, 0)))
    offsets = sorted({d for ell in ells for d in range(-ell, ell + 1, 2)})
    coefs = []  # for each ell, {offset d: c^ell_{L' + d, L'}, (nL,)}
    for ell in ells:
        x, w = np.polynomial.legendre.leggauss(lmaxV + ell + 10)
        P = _legendre_matrix(max(lmaxV, ell), x)
        # h^ell_{LL'} = (2L + 1) / 2 int dx L_ell L_L L_L' (Gauss-Legendre, exact for these degrees)
        h = (2 * jnp.arange(lmaxV + 1)[:, None] + 1) / 2. * jnp.einsum('x,ix,jx->ij', w * P[ell], P[:lmaxV + 1], P[:lmaxV + 1])
        Lp = np.arange(nL)
        row = {}
        for d in range(-ell, ell + 1, 2):  # L = L' + d
            L = np.clip(Lp + d, 0, lmaxV)
            valid = (Lp + d >= 0) & (Lp + d <= lmaxV)
            row[d] = jnp.where(valid, (2 * ell + 1) * 4. * np.pi / norm * (-1)**(abs(d) // 2) * h[L, Lp], 0.)  # i^(L' - L)
        coefs.append(row)

    def project(S, V, transpose=False, rows=None):
        # S (nL', n_rows, nr): the rows ``rows`` (indices) of S_L'(r, r'), all rows if None. Returns (len(ells), nk), the contribution of
        # these rows: summed over blocks of rows, the projection of the whole S.
        nL = S.shape[0]
        shifted = lambda d: V[pad + d:pad + d + nL]  # int r^2 dr j_{L' + d}(kr), (nL', nr, nk)
        if rows is None and transpose: S = jnp.swapaxes(S, 1, 2)
        if rows is None or not transpose:
            # Q_L'(r, k) = int dr' r'^2 j_L'(kr') S_L'(r, r'), then sum_{L, L'} c^ell_{LL'} int dr r^2 j_L(kr) Q_L'(r, k), r over the rows
            Q = jnp.einsum('lrs,lsk->lrk', S, shifted(0))
            leg = (lambda d: shifted(d)) if rows is None else (lambda d: shifted(d)[:, rows])
            return jnp.array([sum(jnp.einsum('l,lrk,lrk->k', c, leg(d), Q) for d, c in row.items()) for row in coefs])
        # transpose, by rows: the rows of S are the leg r' of the transposed S; sum_{r' in rows} r'^2 j_L'(kr') sum_r r^2 j_L(kr) S_L'(r', r)
        T = {d: jnp.einsum('lsr,lrk->lsk', S, shifted(d)) for d in offsets}
        leg = shifted(0)[:, rows]
        return jnp.array([sum(jnp.einsum('l,lsk,lsk->k', c, leg, T[d]) for d, c in row.items()) for row in coefs])

    return project, V


# Window matrices

def compute_smooth2_ic_window(model: IC2Model, edgesin: np.ndarray, k: np.ndarray, ells: tuple=(0, 2, 4), ellsin: tuple=(0, 2, 4),
                              terms: tuple=('total',), shotnoise: bool=True, batch_size: int=4, mpicomm=None) -> dict:
    r"""
    Window matrices of the change of the measured power spectrum when the templates are removed:
    :math:`\Delta P_\ell = P_\ell^{\rm obs\,obs} - P_\ell^{\rm ff} = -P^{\rm ft}_\ell - P^{\rm tf}_\ell + P^{\rm tt}_\ell`, with
    :math:`\delta^{\rm f} = W \delta` the field and :math:`\delta^{\rm t}` what the templates remove (with the global integral constraint, see
    :class:`IC2Model`), for a theory piecewise constant in the bands ``edgesin``. Add it to the geometry window, applied to the same theory.

    For harmonic templates (weights calibrated on mocks, see :class:`IC2Model`), the sums over the harmonics are done analytically:

    .. math::

        S^{\rm ft}_{L'}(r, r') = \sum_{b, R} g_b(r) g_R(r') (2L' + 1) \sum_L M^{bR}_{L'L} \kappa_R(L) \int ds \lambda_R(s) C^{\delta\delta}_L(r, s),
        \qquad
        S^{\rm tt}_{L'}(r, r') = \sum_R g_R(r) g_R(r') (2L' + 1) \sum_L M^{RR}_{L'L} [\kappa_R(L)^2 C^{dd}_L + C^{\perp\perp}_{R, L}],

    :math:`C^{\delta\delta}_L(r, s) = 2\pi \int d\theta \sin\theta \mathcal{L}_L(\cos\theta) \xi(r, s, \theta)` (with the white shot noise),
    :math:`C^{dd}_L = \int\int \lambda_R \lambda_R C^{\delta\delta}_L`. Mixed with regular templates :math:`a`, :math:`P^{\rm tt}` also
    gets the two orderings of

    .. math::

        S^{\rm ht}_{L'}(r, r') = \sum_{R, a} g_R(r) g_a(r') (2L' + 1) \sum_L N^{Ra}_{L'L} \kappa_R(L) \int ds' \Lambda_a(r', s')
        \int ds \lambda_R(s) C^{\delta\delta}_L(s', s),

    :math:`N^{Ra}_{L'L}` (see :class:`IC2Model`); :math:`P^{\rm ft}` is the sum of both kinds of templates.

    Parameters
    ----------
    model : IC2Model
    edgesin : array
        Theory bands, ``(nband + 1,)`` or ``(nband, 2)``.
    k : array
        Output wavenumbers (unbinned).
    ells : tuple, default=(0, 2, 4)
        Output multipoles; :math:`\ell \leq 8` for the bispectrum removal kernels.
    ellsin : tuple, default=(0, 2, 4)
        Theory multipoles :math:`\ell'` (any; the correlation function is rebuilt as :math:`\sum_{\ell'} \xi_{\ell'} \mathcal{L}_{\ell'}`).
    terms : tuple, default=('total',)
        Among ``'total'`` (:math:`\Delta P`), ``'ft'`` (:math:`P^{\rm ft}`, line of sight on the field), ``'tf'``, ``'tt'``.
    shotnoise : bool, default=True
        Also return the shot noise removed with the templates, per unit shot noise, entries ``(term, 'shotnoise')``.
    batch_size : int, default=4
        Theory columns evaluated together.
    mpicomm : MPI communicator, optional
        Split the theory columns over the MPI ranks; all ranks get the whole result.

    Returns
    -------
    out : dict
        ``{term: WindowMatrix}``, and arrays (nell, nk) for ``(term, 'shotnoise')``; for harmonic templates, ``(term, 'additive')`` for
        ``'tt'`` and ``'total'``, the template of the part of the weights independent of the density. The prediction for a theory is the
        window applied to the theory multipoles averaged in the bands, plus the shot noise and additive terms.

    Examples
    --------
    RIC of shuffled randoms in a cap with two photometric regions, each shuffled on its own. ``randoms`` lists three independent random
    subsets (so that no random pairs with itself), each with ``ra``, ``dec`` (degrees), comoving distance ``r``, field weight ``w``
    (:math:`w_{\rm tot} w_{\rm FKP}`), data weight ``wd`` (:math:`w_{\rm tot}`, which the shuffle averages with) and region label ``region``.

    >>> redges = np.arange(rmin, rmax + 2., 2.)  # thin distance shells, as the shuffle
    >>> def get_template(name, weight):  # sky maps of the three subsets and radial selection of the randoms of a region
    ...     sel = [rand['region'] == name for rand in randoms]
    ...     angular = np.array([np.bincount(hp.ang2pix(256, rand['ra'][s], rand['dec'][s], lonlat=True), weights=rand[weight][s],
    ...                                     minlength=hp.nside2npix(256)) for rand, s in zip(randoms, sel)])
    ...     counts = np.histogram(randoms[0]['r'][sel[0]], bins=redges, weights=randoms[0][weight][sel[0]])[0]
    ...     return Template(angular, counts / (np.diff(np.asarray(redges) ** 3) / 3.), redges)
    >>> geometry = [get_template(name, 'w') for name in ['N', 'S']]  # the regions, with the field weights
    >>> shuffle = [get_template(name, 'wd') for name in ['N', 'S']]  # each region shuffled on its own, with the data weights
    >>> ic = ric_templates(geometry, shuffle)  # shell means of each shuffle region
    >>> norm, wsum = spectrum.norm, np.sum(data['w'])  # the measured spectrum's normalization and the sum of its data weights
    >>> model = IC2Model(geometry, ic, norm=norm, wsum=wsum, lmax=int(0.2 * rmax) + 20)  # adds the global integral constraint
    >>> out = compute_smooth2_ic_window(model, edgesin=np.arange(0., 0.3, 0.005), k=np.arange(0.005, 0.2, 0.005))
    >>> pband = ...  # theory P_ell'(q) averaged in the bands, (nell', nband)
    >>> dP = (np.asarray(out['total'].value()) @ pband.ravel()).reshape(3, -1) + shotnoise * out['total', 'shotnoise']

    A shuffle region spanning both caps is given to :func:`ric_templates` as the list of its parts (in this cap and in the other one);
    imaging-regression templates come from :func:`amr_templates`; for the bispectrum, pass ``out['ft']``
    computed with ``ells=(0, 2, 4, 6, 8)``, ``terms=('ft',)`` to :func:`compute_smooth3_ic_window`.
    """
    edgesin = np.asarray(edgesin, dtype='f8')
    if edgesin.ndim == 1: edgesin = np.column_stack([edgesin[:-1], edgesin[1:]])
    # the phases i^(L' - L) of the projection (_bessel_projector) and i^ell' of the correlation multipoles are taken real: even multipoles only
    if any(int(ell) % 2 for ell in list(ells) + list(ellsin)): raise ValueError('only even multipoles ells, ellsin are supported')
    # the projection takes j_L(kr) at the centre of each shell (midpoint rule): the shells must be thin compared to 1 / k
    kr_shell = float(np.max(np.asarray(k))) * float(np.max(np.diff(np.asarray(model.redges))))
    if kr_shell > 0.5: logger.warning(f'k max(dr) = {kr_shell:.2f} > 0.5: the shells are too wide for the midpoint rule in j_L(kr)')
    s = jnp.asarray(model.separations)
    # Eq. 138 [eq:ric_window_matrix]: a column of W^PP is the model for P^th_ell'(q) = 1 in one band of q, i.e. the correlation multipoles xi_ell'(s)
    # = i^ell' int q^2 dq / (2 pi^2) P_ell'(q) j_ell'(qs) (Eq. 135), integrated exactly over the band
    nin = len(ellsin)
    xi = jnp.zeros((nin, len(edgesin), nin, s.size))  # (theory multipole of the column, band, multipole of xi, separation)
    for ill, ell in enumerate(ellsin):
        w = BesselIntegral(jnp.asarray(edgesin), s, ell=ell, edges=True, method='exact', mode='backward', volume=True).w  # (ns, nband)
        xi = xi.at[ill, :, ill].set(w.T)
    xi = xi.reshape(nin * len(edgesin), nin, s.size)
    if shotnoise: xi = jnp.concatenate([jnp.zeros((1, nin, s.size)), xi], axis=0)  # W^N of Eq. 138 [eq:ric_window_matrix]: zero correlation function, unit noise
    noise = jnp.zeros(len(xi)).at[:int(shotnoise)].set(1.)

    def map_columns(run, *args):
        # run(xi, noise, *args) over the theory columns, split over the MPI ranks if mpicomm, gathered in order; the memory XLA plans for it
        # is logged before it runs
        index = np.arange(len(xi)) if mpicomm is None else np.array_split(np.arange(len(xi)), mpicomm.size)[mpicomm.rank]
        local = None
        if index.size:
            compiled = run.lower(xi[index], noise[index], *args).compile()
            memory = compiled.memory_analysis()
            if memory is not None:
                logger.info(f'theory columns: XLA plans {memory.temp_size_in_bytes / 1e9:.1f} GB of temporaries, {memory.argument_size_in_bytes / 1e9:.1f} GB '
                            f'of arguments, {memory.output_size_in_bytes / 1e9:.1f} GB of outputs (peak memory so far {_peak_memory()})')
            local = np.asarray(compiled(xi[index], noise[index], *args))
        if mpicomm is None: return local
        return np.concatenate([part for part in mpicomm.allgather(local) if part is not None], axis=0)
    additive = None

    ft = tf = tt = np.zeros((len(xi), len(ells), len(k)))
    if model.removed['t']:
        # Regular templates: P^ft, P^tf and P^tt from the angular kernels S_L'(r, r') of the two pairs of fields, field x template and
        # template x template (notes Eqs. 130-133). The field is W delta; the global integral constraint is among the templates.
        shell_centers = (model.redges[1:] + model.redges[:-1]) / 2.
        # number of shells, of degrees L' = 0..lmax, of angular separations theta at which the pair-count functions are tabulated
        n_shells, n_degrees, n_theta = shell_centers.size, model.lmax + 1, model.theta.size
        templates = model.removed['t']
        field_template = model.pairs['f', 't']  # geometric factors computed by IC2Model
        template_template = model.pairs['t', 't']

        # The theory enters through angular integrals: for each pair-count function c(theta) of the geometric factors (a "column"),
        #     K_c(r, s) = sum_theta c(theta) xi(r, s, theta)   for all pairs of shells (r, s).
        # When a template's response is rank one, Lambda(r', s) = alpha(r') lambda(s), only sum_s K_c(r, s) lambda(s) is needed: such "rank-one"
        # columns are integrated against lambda during the angular integral, which is much cheaper. The other columns are "full": K_c(r, s).
        # Memory: the full kernels of field x template, (n_shells, n_shells, n_degrees) per region and group, are never held for all shells r.
        # The shells r are taken a block at a time (rows_per_chunk): xi(r, s, theta) of the block, then the kernels of its rows K_c(r, .), the
        # rows r of S^ft_L'(r, r') they give, projected at once (P^ft and P^tf are sums over the rows). Only the template x template kernels
        # (one column per pair of templates) are kept for all shells: S^tt needs the whole covariance M(s, s') of the template means; S^tt is
        # then projected a block of rows at a time too.
        # The columns are stored in batches: an array (n, n_theta), n columns, n_degrees (one per L') for field x template, 1 for template x
        # template; separately for the two pairs of fields ('ft', 'tt').
        full_columns = {'ft': [], 'tt': []}  # batches of full columns
        radial_weights_list = []  # the distinct lambda(s), each (n_shells,)
        rank_one_columns = {'ft': {}, 'tt': {}}  # {index in radial_weights_list: list of batches of rank-one columns}

        def add_full_columns(pairs, pair_counts):
            # store the batch pair_counts (n, n_theta) as full columns of pairs ('ft' or 'tt'); return the index of its first column among them
            first = sum(len(columns) for columns in full_columns[pairs])
            full_columns[pairs].append(pair_counts)
            return first

        def add_rank_one_columns(pairs, radial_weights, pair_counts):
            # store the batch pair_counts (n, n_theta) as columns of pairs to integrate against radial_weights; return the index of
            # radial_weights and that of the batch's first column in the block of rank-one kernels of these pairs and radial weights
            iweights = next((i for i, w in enumerate(radial_weights_list) if w.shape == radial_weights.shape and np.array_equal(w, radial_weights)),
                            None)
            if iweights is None:
                radial_weights_list.append(radial_weights)
                iweights = len(radial_weights_list) - 1
            first = sum(len(columns) for columns in rank_one_columns[pairs].get(iweights, []))
            rank_one_columns[pairs].setdefault(iweights, []).append(pair_counts)
            return iweights, first

        # Field x template: one pair-count function per region of the field, group of templates (same radial profile and response, their
        # pair counts summed in IC2Model) and degree L'. Record where the columns of each (region, group) are.
        field_template_terms = []  # (region index, group index, index of lambda or None if full, first column)
        for iregion in range(len(model.geometry)):
            for igroup, (template, _) in enumerate(field_template['groups']):
                pair_counts = field_template['cross_pairs'][iregion, igroup]  # (n_degrees, n_theta)
                if template.kind == 'outer':
                    iweights, first = add_rank_one_columns('ft', template.response[2], pair_counts)
                else:
                    iweights, first = None, add_full_columns('ft', pair_counts)
                field_template_terms.append((iregion, igroup, iweights, first))

        # Template x template: one pair-count function per pair of templates, that of their mean windows, giving the covariance M(s, s') of
        # their shell means. S needs Lambda_1 M Lambda_2^T: a rank-one response on either side is summed over during the angular integral (M is
        # symmetric). Record which side is rank one.
        template_template_terms = []  # (template index 1, index 2, rank-one side 'both' / 'first' / 'second' / None, index of lambda, first column)
        for i1, template1 in enumerate(templates):
            for i2, template2 in enumerate(templates):
                pair_counts = template_template['cross_pairs'][i1, i2][None, :]  # (1, n_theta): the degree L' enters through the overlap C, not here
                if template2.kind == 'outer':  # M lambda_2
                    side = 'both' if template1.kind == 'outer' else 'second'
                    iweights, first = add_rank_one_columns('tt', template2.response[2], pair_counts)
                elif template1.kind == 'outer':  # M lambda_1
                    side = 'first'
                    iweights, first = add_rank_one_columns('tt', template1.response[2], pair_counts)
                else:  # M(s, s')
                    side, iweights, first = None, None, add_full_columns('tt', pair_counts)
                template_template_terms.append((i1, i2, side, iweights, first))

        # the columns as arguments of the jitted function: for each pair of fields, all full columns (n_theta, n_full), and the rank-one
        # columns, one array (n_theta, n) per lambda (each its own width: the cost scales with the number of columns); the lambdas (n_lambda, n_shells)
        stacked_full_columns = {pairs: jnp.asarray(np.concatenate(columns, axis=0).T if columns else np.zeros((n_theta, 0)))
                                for pairs, columns in full_columns.items()}
        stacked_rank_one_columns = {pairs: [jnp.asarray(np.concatenate(columns[i], axis=0).T if i in columns else np.zeros((n_theta, 0)))
                                            for i in range(len(radial_weights_list))] for pairs, columns in rank_one_columns.items()}
        radial_weights_array = jnp.asarray(np.array(radial_weights_list) if radial_weights_list else np.zeros((0, n_shells)))
        # responses that are not rank one, as matrices Lambda(r, s)
        response_matrix = {id(t): np.diag(t.response[1]) if t.kind == 'diag' else t.response[1] for t in templates if t.kind in ('diag', 'dense')}
        shotnoise_kernel = jnp.asarray(1. / (np.diff(np.asarray(model.redges) ** 3) / 3.))  # shot noise per unit amplitude: delta_D(r - s) / r^2 on the shells
        degree_weights = 2 * jnp.arange(n_degrees) + 1  # 2L' + 1
        project, bessel_table = _bessel_projector(model.redges, model.lmax, model.norm, k, ells)
        separations, shell_centers = jnp.asarray(model.separations), jnp.asarray(shell_centers)
        # blocks of rows_per_chunk shells r, the last one padded with the last shell; valid: 0 for the padding, which must not be summed
        rows_per_chunk = 8
        n_chunks = -(-n_shells // rows_per_chunk)
        chunk_shells = np.concatenate([np.arange(n_shells), np.full(n_chunks * rows_per_chunk - n_shells, n_shells - 1)]).reshape(n_chunks, rows_per_chunk)
        chunk_valid = (np.arange(n_chunks * rows_per_chunk) < n_shells).reshape(n_chunks, rows_per_chunk).astype('f8')
        chunks = (jnp.asarray(chunk_shells), jnp.asarray(chunk_valid))

        def rank_one_rows(xi_weighted, columns):
            # sum_s K_c(r, s) lambda_w(s) for each lambda w and its columns c, padded with zeros to the largest number: (n_lambda, n_rows, n_max)
            n_max = max(c.shape[-1] for c in columns)
            return jnp.stack([jnp.pad(xi_weighted[iw] @ c, ((0, 0), (0, n_max - c.shape[-1]))) for iw, c in enumerate(columns)])

        def shotnoise_rows(ishells):  # the rows ishells of diag(shotnoise_kernel), (n_rows, n_shells)
            return jnp.where(ishells[:, None] == jnp.arange(n_shells)[None, :], shotnoise_kernel[None, :], 0.)

        def spectra(args, arrays):
            # P^ft, P^tf and P^tt, (3, len(ells), nk), for one theory: its correlation function multipoles and its shot-noise amplitude noise
            (xi_theory, noise), (full_pair_counts, rank_one_pair_counts, radial_weights_array, bessel_table) = args, arrays
            has_rank_one = radial_weights_array.shape[0] > 0

            def field_template_rows(carry, chunk):
                # the angular integrals of the theory for the rows r of the block, all columns in one pass
                ishells, valid = chunk
                xi_rs = _correlation_rows(shell_centers[ishells], shell_centers, model.theta, separations, xi_theory[None], ellsin)[0]  # (n_rows, n_shells, n_theta)
                kernels = {pairs: jnp.einsum('rsq,qc->rsc', xi_rs, full_pair_counts[pairs]) for pairs in ['ft', 'tt']}  # K_c(r, s), (n_rows, n_shells, n_full)
                rank_one_kernels = {}
                if has_rank_one:  # sum over s against lambda(s) first, then over theta (Sec. VI A, "Numerical evaluation": ~200 times faster)
                    xi_weighted = jnp.einsum('rsq,ws->wrq', xi_rs, radial_weights_array)  # (n_lambda, n_rows, n_theta)
                    rank_one_kernels = {pairs: rank_one_rows(xi_weighted, rank_one_pair_counts[pairs]) for pairs in ['ft', 'tt']}
                # rows r of S^ft_L'(r, r') = sum_{b, a} g_b(r) g_a(r') sum_s K^{ba}_L'(r, s) Lambda_a(r', s), (n_degrees, n_rows, n_shells) (Eq. 132)
                S_ft = jnp.zeros((n_degrees, rows_per_chunk, n_shells))
                for iregion, igroup, iweights, first in field_template_terms:
                    region, template = model.geometry[iregion], field_template['groups'][igroup][0]
                    region_radial = jnp.asarray(region.radial)[ishells]
                    zero_separation = jnp.asarray(field_template['auto_pairs'][iregion, igroup])  # (n_degrees,), for the shot noise
                    if iweights is not None:  # rank one, Lambda_a(r', s) = alpha(r') lambda(s)
                        profile, radial_weights = template.response[1:]
                        # sum_s K(r, s) lambda(s), from the angular integral, plus the shot noise: (n_degrees, n_rows)
                        kernel = rank_one_kernels['ft'][iweights, :, first:first + n_degrees].T
                        kernel = kernel + noise * zero_separation[:, None] * (shotnoise_kernel * radial_weights)[ishells][None, :]
                        S_ft = S_ft + (region_radial[None, :, None] * kernel[:, :, None]) * (profile * template.radial)[None, None, :]
                    else:  # diagonal or dense Lambda_a(r', s)
                        # K(r, s), from the angular integral, plus the shot noise: (n_degrees, n_rows, n_shells)
                        kernel = jnp.moveaxis(kernels['ft'][:, :, first:first + n_degrees], -1, 0)
                        kernel = kernel + noise * zero_separation[:, None, None] * shotnoise_rows(ishells)[None]
                        # sum_s K(r, s) Lambda(r', s)
                        if template.kind == 'diag': kernel = kernel * template.response[1][None, None, :]
                        else: kernel = jnp.einsum('lrs,us->lru', kernel, template.response[1])
                        S_ft = S_ft + region_radial[None, :, None] * kernel * template.radial[None, None, :]
                S_ft = S_ft * valid[None, :, None]
                # P^tf is P^ft with the line of sight on the template, S^tf_L'(r, r') = S^ft_L'(r', r)
                carry = carry + jnp.stack([project(S_ft, bessel_table, rows=ishells), project(S_ft, bessel_table, transpose=True, rows=ishells)])
                return carry, (kernels['tt'], rank_one_kernels.get('tt', None))

            ft_tf, (tt_kernels, tt_rank_one_kernels) = jax.lax.scan(field_template_rows, jnp.zeros((2, len(ells), len(k))), chunks)
            # the template x template kernels of all rows: (n_shells, n_shells, n_full) and (n_lambda, n_shells, n_max)
            tt_kernels = tt_kernels.reshape(n_chunks * rows_per_chunk, n_shells, tt_kernels.shape[-1])[:n_shells]
            if has_rank_one:
                tt_rank_one_kernels = jnp.moveaxis(tt_rank_one_kernels, 0, 1).reshape(len(radial_weights_list), n_chunks * rows_per_chunk, tt_rank_one_kernels.shape[-1])[:, :n_shells]

            # S^tt_L'(r, r') = sum_{a1, a2} (2L' + 1) C_{a1 a2, L'} g_a1(r) g_a2(r') [Lambda_a1 M Lambda_a2^T](r, r'), (Eq. 133), M(s, s') the
            # covariance of the shell means of the two templates, plus the shot noise
            overlap = jnp.asarray(template_template['C'][:, :, :n_degrees])  # C_{a1 a2, L'}: overlap of the two patterns at degree L'
            # both responses rank one, all such pairs at once: the amplitude covariance lambda_a1^T M lambda_a2 times alpha_a1(r) alpha_a2(r')
            both = [term for term in template_template_terms if term[2] == 'both']
            if both:
                i1, i2, _, iweights, first = (np.array(x) for x in zip(*both))
                weights1 = np.array([templates[i].response[2] for i in i1])
                weights2 = np.array([templates[i].response[2] for i in i2])
                # M lambda_a2, from the angular integral, plus the shot noise: (n_pairs, n_shells)
                covariance_weights2 = tt_rank_one_kernels[iweights, :, first]
                covariance_weights2 = covariance_weights2 + noise * template_template['auto_pairs'][i1, i2][:, None] * shotnoise_kernel[None, :] * weights2
                amplitude_covariance = jnp.zeros((len(templates), len(templates))).at[i1, i2].set(jnp.sum(covariance_weights2 * weights1, axis=-1))
                profiles = jnp.asarray(np.array([t.response[1] * t.radial if t.kind == 'outer' else np.zeros(n_shells) for t in templates]))  # alpha_a g_a
                both_columns = jnp.einsum('ijl,ij,js->ils', overlap, amplitude_covariance, profiles)  # (n_templates, n_degrees, n_shells)
            # one rank-one side: [Lambda_a1 M Lambda_a2^T](r, r') = left(r) right(r'), stacked over the pairs, with their degree weights
            # (2L' + 1) C_{a1 a2, L'}; no rank-one side: the matrices g_a1(r) [Lambda_a1 M Lambda_a2^T](r, r') g_a2(r'), one per pair
            left, right, pair_weights, dense_terms = [], [], [], []
            for i1, i2, side, iweights, first in template_template_terms:
                if side == 'both': continue
                template1, template2 = templates[i1], templates[i2]
                zero_separation = template_template['auto_pairs'][i1, i2]
                weights = degree_weights * overlap[i1, i2]
                if side is None:  # M(s, s') from the angular integral, plus the shot noise
                    shell_covariance = tt_kernels[:, :, first] + noise * zero_separation * jnp.diag(shotnoise_kernel)
                    covariance = response_matrix[id(template1)] @ shell_covariance @ response_matrix[id(template2)].T
                    dense_terms.append((weights, template1.radial[:, None] * covariance * template2.radial[None, :]))
                elif side == 'second':  # Lambda_a2 = alpha_a2 lambda_a2: Lambda_a1 (M lambda_a2) times alpha_a2(r')
                    profile2, weights2 = template2.response[1:]
                    covariance_weights2 = tt_rank_one_kernels[iweights, :, first] + noise * zero_separation * shotnoise_kernel * weights2
                    left.append(template1.radial * (response_matrix[id(template1)] @ covariance_weights2))
                    right.append(profile2 * template2.radial)
                    pair_weights.append(weights)
                else:  # Lambda_a1 = alpha_a1 lambda_a1: alpha_a1(r) times Lambda_a2 (M lambda_a1), M symmetric
                    profile1, weights1 = template1.response[1:]
                    covariance_weights1 = tt_rank_one_kernels[iweights, :, first] + noise * zero_separation * shotnoise_kernel * weights1
                    left.append(jnp.asarray(template1.radial * profile1))
                    right.append(template2.radial * (response_matrix[id(template2)] @ covariance_weights1))
                    pair_weights.append(weights)
            if left: left, right, pair_weights = jnp.stack(left), jnp.stack(right), jnp.stack(pair_weights)

            def template_template_rows(carry, chunk):
                # rows r of S^tt_L'(r, r'), (n_degrees, n_rows, n_shells), projected at once
                ishells, valid = chunk
                S_tt = jnp.zeros((n_degrees, rows_per_chunk, n_shells))
                if both: S_tt = S_tt + degree_weights[:, None, None] * jnp.einsum('ir,ils->lrs', profiles[:, ishells], both_columns)
                if len(left): S_tt = S_tt + jnp.einsum('plr,ps->lrs', pair_weights[:, :, None] * left[:, None, ishells], right)
                for weights, matrix in dense_terms: S_tt = S_tt + weights[:, None, None] * matrix[ishells][None]
                return carry + project(S_tt * valid[None, :, None], bessel_table, rows=ishells), None

            tt = jax.lax.scan(template_template_rows, jnp.zeros((len(ells), len(k))), chunks)[0]
            return jnp.concatenate([ft_tf, tt[None]], axis=0)

        t0 = time.time()
        # all theories (the columns of the window matrix) in one jitted map; the large arrays are arguments, not constants of the compiled code
        arrays = (stacked_full_columns, stacked_rank_one_columns, radial_weights_array, bessel_table)
        run = jax.jit(lambda xis, noises, arrays: jax.lax.map(lambda args: spectra(args, arrays), (xis, noises), batch_size=batch_size))
        ft, tf, tt = np.moveaxis(map_columns(run, arrays), 1, 0)  # each (ncol, len(ells), nk)
        logger.info(f'regular templates: {len(xi)} theory columns in {time.time() - t0:.0f} s (including compilation), peak memory {_peak_memory()}')

    if 'h' in model.removed:
        rc = (model.redges[1:] + model.redges[:-1]) / 2.
        templates, nL = model.removed['h'], model.lmax + 1
        Lp = jnp.arange(nL)
        # Eq. 166 [eq:amr_sysnet]. Columns 2 pi dtheta sin(theta) L_L(cos theta): C^{delta delta}_L(r, s), integrated against lambda_R (Eq. 164)
        pw = (2. * np.pi * jnp.asarray(np.gradient(model.theta) * np.sin(model.theta))[None, :] * _legendre_matrix(model.lmax, np.cos(model.theta))).T
        radial_weights = jnp.asarray(np.array([t.response[1] for t in templates]))  # lambda_R(s) ds
        removed_fraction = jnp.asarray(np.array([t.response[2][:nL] for t in templates]))  # kappa_R(L)
        uncorrelated_power = jnp.asarray(np.array([t.response[3][:nL] for t in templates]))
        gR = jnp.asarray(np.array([t.radial for t in templates]))
        gb = jnp.asarray(np.array([reg.radial for reg in model.geometry]))
        Mft, Mtt = jnp.asarray(model.pairs['f', 'h']['M']), jnp.asarray(model.pairs['h', 'h']['M'])  # (nb, nR, nL', nL), (nR, nL', nL)
        project, V = _bessel_projector(model.redges, model.lmax, model.norm, k, ells)
        diag = jnp.asarray(1. / (np.diff(np.asarray(model.redges) ** 3) / 3.))
        cross_t = ('h', 't') in model.pairs
        if cross_t:  # regular templates: radial profiles g_a and responses Lambda_a(r', s') as dense matrices
            regular = model.removed['t']
            dense = {'diag': lambda t: np.diag(t.response[1]), 'dense': lambda t: t.response[1], 'outer': lambda t: np.outer(*t.response[1:])}
            Lam, ga, N = (jnp.asarray(np.array([dense[t.kind](t) for t in regular])), jnp.asarray(np.array([t.radial for t in regular])),
                          jnp.asarray(model.pairs['h', 't']['N']))
        else: Lam = ga = N = None
        arrays = (pw, radial_weights, removed_fraction, uncorrelated_power, gR, gb, Mft, Mtt, V, Lam, ga, N)  # arguments of the jitted functions, not constants

        def column(args, arrays, with_uncorrelated=False):
            (x, nn), (pw, radial_weights, removed_fraction, uncorrelated_power, gR, gb, Mft, Mtt, V, Lam, ga, N) = args, arrays
            # (nR, nr, nL): int ds lambda_R(s) C_L(r, s)
            proj = _angular_kernels(rc, model.theta, jnp.asarray(model.separations), x[None], ellsin, radial_weights=radial_weights,
                                    rank_one_pair_counts=pw)[1][0]
            proj = proj + nn * (diag[None, :, None] * radial_weights[:, :, None])  # shot noise N delta_D(r - s) / r^2 in C^{delta delta}_L
            # Memory: each contraction below has two operands, one matrix product (dot_general); an einsum of three or four operands may be
            # evaluated through an intermediate with all their indices, e.g. (nR, na, nL', nr', nL) for Y below: ~200 GB per MPI rank for the
            # DR2 ELG at z 1.1-1.6
            kappa_proj = removed_fraction[:, None, :] * proj  # kappa_R(L) int ds lambda_R(s) C_L(r, s), (nR, nr, nL)
            cross = jnp.einsum('bRml,Rrl->bRrm', Mft, kappa_proj)  # sum_L M_{L'L} kappa_L proj_L(r), (nb, nR, nr, nL')
            # Eq. 166 [eq:amr_sysnet], first line: S^ft_L'(r, r') = sum_{b, R} g_b(r) g_R(r') (2L' + 1) sum_L M^{bR}_{L'L} kappa_R(L) int ds lambda_R
            # C_L(r, s)
            field_sum = jnp.einsum('br,bRrm->Rmr', gb, cross)  # sum_b g_b(r) cross, (nR, nL', nr)
            S_ft = (2 * Lp + 1)[:, None, None] * jnp.einsum('Rmr,Rs->mrs', field_sum, gR)
            # second line: kappa_R(L)^2 C^dd_L (C^dd_L = int int lambda_R lambda_R C_L), + C^perp_{R, L} for the additive template
            c = removed_fraction**2 * jnp.einsum('Rrl,Rr->Rl', proj, radial_weights) + (uncorrelated_power if with_uncorrelated else 0.)
            template_sum = jnp.einsum('Rml,Rl->Rm', Mtt, c)[:, :, None] * gR[:, None, :]  # (nR, nL', nr)
            S_tt = (2 * Lp + 1)[:, None, None] * jnp.einsum('Rmr,Rs->mrs', template_sum, gR)
            tt = project(S_tt, V)
            if cross_t:
                # S^ht_L'(r, r') = sum_{R, a} g_R(r) g_a(r') (2L' + 1) sum_L N^{Ra}_{L'L} kappa_R(L) c_{Ra, L}(r'), with
                # c_{Ra, L}(r') = sum_s' Lambda_a(r', s') int ds lambda_R(s) C_L(s', s); in P^tt with both orderings, the line of sight on either
                c_ht = jnp.einsum('aus,RsL->RauL', Lam, kappa_proj) * ga[None, :, :, None]  # g_a(r') kappa_R(L) c_{Ra, L}(r'), (nR, na, nr', nL)
                Y = jnp.einsum('RaML,RauL->RMu', N, c_ht)  # sum_a g_a(r') sum_L N_{L'L} kappa_L c_L(r'), (nR, nL', nr'): contracted over (a, L)
                S_ht = (2 * Lp + 1)[:, None, None] * jnp.einsum('Rr,RMu->Mru', gR, Y)
                tt = tt + project(S_ht, V) + project(S_ht, V, transpose=True)
            return jnp.stack([project(S_ft, V), project(S_ft, V, transpose=True), tt])

        t0 = time.time()
        vals = map_columns(jax.jit(lambda xs, ns, arrays: jax.lax.map(lambda args: column(args, arrays), (xs, ns), batch_size=batch_size)), arrays)
        ft, tf, tt = ft + vals[:, 0], tf + vals[:, 1], tt + vals[:, 2]  # (ncol, nell, nk), added to the regular templates'
        additive = np.asarray(jax.jit(lambda x, arrays: column((x, 0.), arrays, with_uncorrelated=True))(jnp.zeros((nin, s.size)), arrays))[2]
        logger.info(f'{len(xi)} theory columns in {time.time() - t0:.0f} s (including compilation), peak memory {_peak_memory()}')
    allterms = {'ft': ft, 'tf': tf, 'tt': tt, 'total': -ft - tf + tt}  # Eq. 126 [eq:ric_cross_auto]

    # observable multipoles at k (unbinned), theory piecewise constant in the bands
    k = np.asarray(k)
    observable = Mesh2SpectrumPoles([Mesh2SpectrumPole(k=k, k_edges=np.column_stack([k - 1e-8, k + 1e-8]), nmodes=np.ones_like(k), num_raw=np.zeros_like(k),
                                                       num_shotnoise=np.zeros_like(k), norm=np.ones_like(k), ell=ell) for ell in ells])
    kin = np.mean(edgesin, axis=-1)
    theory = ObservableTree([Mesh2SpectrumPole(k=kin, k_edges=edgesin, num_raw=np.zeros_like(kin)) for _ in ellsin], ells=list(ellsin))
    out = {}
    for term in terms:
        val = allterms[term]  # (ncol, nell, nk)
        W = val[int(shotnoise):].reshape(len(ellsin), len(edgesin), len(ells), len(k))
        out[term] = WindowMatrix(observable=observable, theory=theory, value=np.transpose(W, (2, 3, 0, 1)).reshape(len(ells) * len(k), -1))
        if shotnoise: out[term, 'shotnoise'] = val[0]
        if additive is not None and term in ('tt', 'total'): out[term, 'additive'] = additive
    return out


def compute_smooth2_modulation_window(geometry: list, eta: list, removed: list, norm: float, lmax: int, edgesin: np.ndarray, k: np.ndarray,
                                      ells: tuple=(0, 2, 4), ellsin: tuple=(0, 2, 4), wsum: float=None, shotnoise: bool=True, batch_size: int=4,
                                      mpicomm=None) -> dict:
    r"""
    Window matrix :math:`\tau^\eta` of :math:`P^\eta = \tau^\eta \ast P^{\rm th}` (notes, Eq. 147 [eq:ric_eta_kernel]), the input of
    :math:`D^{BP}`.

    :math:`P^\eta` is the cross power spectrum of the weight modulation :math:`W \eta` (weights that follow the density,
    :func:`fkp_modulation_templates`) with the observed field :math:`\delta^{\rm obs} = \delta^{\rm f} - \delta^{\rm r}`: :math:`\delta^{\rm f} = W \delta`
    the field, :math:`\delta^{\rm r}` what the templates ``removed`` take out (e.g. :func:`ric_templates` plus their :func:`gic_templates`, for
    the shuffled arm). It has two parts:

    - :math:`\langle W \eta\, \delta^{\rm f} \rangle`: :math:`W \eta` in the place of a removed field, crossed with the field, the window
      ``'ft'`` of an :class:`IC2Model` whose templates are ``eta``;
    - :math:`\langle W \eta\, \delta^{\rm r} \rangle`: the cross of two template fields. :class:`IC2Model` gives only the auto power
      :math:`P^{\rm tt}` of all its templates together, so the cross follows by polarization:
      :math:`P^{\rm tt}[{\rm removed} + \eta] - P^{\rm tt}[{\rm removed}] - P^{\rm tt}[\eta]` is twice the cross (both orderings averaged).

    Hence, from three :class:`IC2Model` windows without automatic global constraint (``removed`` gives it explicitly):

    .. math::

        P^\eta = P^{\rm ft}[\eta] - \tfrac{1}{2} \left( P^{\rm tt}[{\rm removed} + \eta] - P^{\rm tt}[{\rm removed}] - P^{\rm tt}[\eta] \right).

    Uses:

    - applied to the fiducial theory (plus the shot noise): the ``modulation`` of :func:`compute_smooth3_ic_window`, for :math:`D^{BP}`
      (Eq. 149 [eq:ric_bispectrum_eta]);
    - with the geometry :math:`W^2` and ``norm`` :math:`\int W^3` (as the closure kernel of :func:`compute_smooth3_closure_window`):
      :math:`\tau^\eta_{W^2}`, giving :math:`P^\eta_{W^2}`, the ``closure_modulation``, for :math:`C^{BP}` (Eq. 153 [eq:collapsed_multipoles]).

    Not needed for imaging weights fitted on the data: :math:`\eta = -\epsilon` is then the removed field itself, and
    :math:`P^\eta = -P^{\rm ft} + P^{\rm tt}` of the regression templates (Eq. 161).

    Parameters
    ----------
    geometry : list of Template
    eta : list of Template
        Templates of :math:`W \eta`.
    removed : list of Template
        Templates removed from the observed field (with the global integral constraint given explicitly if it applies).
    norm, lmax, wsum : as :class:`IC2Model`.
    edgesin, k, ells, ellsin, shotnoise, batch_size, mpicomm : as :func:`compute_smooth2_ic_window`.

    Returns
    -------
    out : dict
        ``'eta'``: :class:`WindowMatrix` of :math:`P^\eta`; ``('eta', 'shotnoise')`` (nell, nk) per unit shot noise if ``shotnoise``.
    """
    kw = dict(norm=norm, lmax=lmax, wsum=wsum, gic=False, mpicomm=mpicomm)
    wkw = dict(edgesin=edgesin, k=k, ells=ells, ellsin=ellsin, shotnoise=shotnoise, batch_size=batch_size, mpicomm=mpicomm)
    w_eta = compute_smooth2_ic_window(IC2Model(geometry, list(eta), **kw), terms=('ft', 'tt'), **wkw)
    w_rem = compute_smooth2_ic_window(IC2Model(geometry, list(removed), **kw), terms=('tt',), **wkw)
    w_both = compute_smooth2_ic_window(IC2Model(geometry, list(removed) + list(eta), **kw), terms=('tt',), **wkw)
    value = lambda w, t: np.asarray(w[t].value())
    out = {'eta': w_eta['ft'].clone(value=value(w_eta, 'ft') - 0.5 * (value(w_both, 'tt') - value(w_rem, 'tt') - value(w_eta, 'tt')))}
    if shotnoise:
        sn = lambda w, t: np.asarray(w[t, 'shotnoise'])
        out['eta', 'shotnoise'] = sn(w_eta, 'ft') - 0.5 * (sn(w_both, 'tt') - sn(w_rem, 'tt') - sn(w_eta, 'tt'))
    return out


def compute_smooth3_ic_window(ft: WindowMatrix, window: WindowMatrix=None, kin: np.ndarray=None, ells: tuple=None, ellsin: tuple=None,
                              closure_kernel: WindowMatrix=None, modulation: ObservableTree=None, closure_modulation: ObservableTree=None,
                              shotnoise: ObservableTree=None, closure_shotnoise: ObservableTree=None, qsplit: float=0.02, pin: ObservableTree=None,
                              nmu: int=16, nc: int=64, npsi: int=16, nx: int=6, batch_size: int=8) -> dict:
    r"""
    The window matrices of the change of the measured bispectrum when the templates are removed (notes, Sec. V G), at first order in the
    removed field, on one leg at a time, linear in the theory:

    .. math::

        \Delta B = W^{BB} B^{\rm th} + \left(W^{BP} + \bar{n}^{-1} W^{BN}\right) P^{\rm th},
        \qquad W^{BB} = W^{\rm geo} D^{BB} + C^{BB}, \quad W^{BP} = W^{\rm geo} D^{BP} + C^{BP}, \quad W^{BN} = W^{\rm geo} D^{BN} + C^{BN}

    (Eq. 154 [eq:ric_bispectrum_full]), the outputs ``'BB'``, ``'BP'``, ``'BN'``. :math:`W^{\rm geo}` is the geometry window (``window``);
    the :math:`D` act on the theory before it, the :math:`C` are the exact terms of the closure leg below :math:`q_{\rm split}`.

    - :math:`D^{BB}` (Eq. 144 [eq:ric_bispectrum_rem]), the removal: each leg :math:`\mathbf{k}_i` of the triangle is replaced by the modes
      :math:`\mathbf{q}` the templates draw on, weighted by the removal kernel :math:`\tau` (Eq. 145 [eq:ric_tau_matrix]),

      .. math::

          D^{BB}[B](k_1, k_2) = \sum_{i=1}^3 \left\langle w \int dq\, \frac{d\mu'}{2}\, \tau(k_i, \mu_i; q, \mu')\, B|_{\mathbf{k}_i \to \mathbf{q}} \right\rangle,
          \qquad
          \tau(k, \mu; q, \mu') = -\sum_{\ell \ell'} W^{\rm ft}_{\ell \ell'}(k, q)\, \mathcal{L}_\ell(\mu)\, (2\ell' + 1)\, \mathcal{L}_{\ell'}(\mu'),

      :math:`W^{\rm ft}` the window of :math:`P^{\rm ft}` (``ft``; with ``gic=True`` it includes the global integral constraint, so that
      :math:`\Delta B` is relative to the bispectrum of :math:`W \delta`). :math:`\mathbf{q}` has the norm of the band centre and lies in the
      plane of :math:`\mathbf{k}_i` and the line of sight (:math:`\tau` is averaged over azimuths). When the closure leg is replaced, either
      binned leg absorbs the change, half each, the triangles read from the theory as the pairs :math:`(\mathbf{q}, \mathbf{k}_2)` and
      :math:`(\mathbf{k}_1, \mathbf{q})`. Legs below the smallest :math:`k` of :math:`W^{\rm ft}` take its kernel there, legs above the
      largest are not removed.
    - :math:`D^{BP}` (Eq. 149 [eq:ric_bispectrum_eta]), the modulation: data weights that follow the density, the observed field
      :math:`(1 + \eta)\, \delta^{\rm obs}` (FKP weights from the data's own :math:`n(z)`, :math:`\eta = -\beta \bar{\delta}_{\rm cap}`,
      :func:`fkp_modulation_templates`; imaging weights fitted on the data, :math:`\eta = -\epsilon`), one leg carrying
      :math:`P^\eta = \tau^\eta \ast P^{\rm th}` (Eq. 147 [eq:ric_eta_kernel], ``modulation``, at the fiducial theory) and the two others the
      theory :math:`P`:

      .. math::

          D^{BP}[P](k_1, k_2) = \sum_{j=1}^3 \left\langle w\, P^\eta(k_j, \mu_j) \left[P(k_i, \mu_i) + P(k_l, \mu_l)\right] \right\rangle.

    - :math:`D^{BN}` (Eq. 150 [eq:ric_bispectrum_shotnoise]), the shot noise of the removal: the removed field is measured on the data and
      carries their shot noise; the estimator subtracts the coincidences of two legs, not those of a leg's particle with the points the
      templates are measured on (leg :math:`i` carries the removal of the noise of the particle at leg :math:`j`, which clusters with leg
      :math:`l`). It is :math:`D^{BP}` with :math:`P^\eta \to -N`, :math:`N` the shot noise of :math:`P^{\rm ft}` per unit shot noise
      (``shotnoise``), :math:`\Delta B^N = \bar{n}^{-1} D^{BN} P`.
    - :math:`C^{BB}` (Eq. 152 [eq:ric_closure_removal], ``closure_kernel``, :func:`compute_smooth3_closure_window`), :math:`C^{BP}` and
      :math:`C^{BN}` (Eq. 153 [eq:collapsed_multipoles], ``closure_modulation``, ``closure_shotnoise``, the same function): the
      composition with :math:`W^{\rm geo}` treats the window and the removal as separable, which holds for legs well above the window scale.
      The binned legs always are; the closure leg is not when the two shells are close and the pairs nearly antipodal (the diagonal bins and
      the adjacent ones), where the composition undercounts the removal of angular templates by 25% to a factor 2-3. One rule for every
      term: the closure leg carrying the removal (or :math:`P^\eta`, :math:`-N`) is composed for :math:`k_3 \geq q_{\rm split}` (``qsplit``),
      and below replaced by its exact term, computed with the window :math:`W^2` on the field and the normalization :math:`\int W^3`:
      :math:`\tau_{W^2}`, :math:`P^\eta_{W^2}`, :math:`N_{W^2}`. Without an exact term, its closure leg is composed at all :math:`k_3`. The
      closure leg carrying the theory :math:`P` while a binned leg carries the removal stays composed.

    The theory bispectrum is given by its Sugiyama multipoles :math:`B_{\ell_1 \ell_2 L}(k_1', k_2')` on a grid of :math:`(k_1', k_2')`
    nodes, the theory power spectrum by its multipoles on the grid of ``pin``. One rule for every function tabulated in :math:`k` (the theory
    on its nodes, the removal kernel, :math:`P^\eta`, :math:`N`): linear interpolation between the nodes, the first value below them, zero
    above them. The corrected multipoles are projected with the weights :math:`w = \Phi / \langle \Phi^2 \rangle` (Eqs. 139-140,
    :func:`utils.get_S`).

    Parameters
    ----------
    ft : WindowMatrix
        Window of :math:`P^{\rm ft}`, ``compute_smooth2_ic_window(..., ells=(0, 2, 4, 6, 8), terms=('ft',))['ft']``, on a ``k`` grid
        covering the legs to remove (e.g. up to 0.2), finer than the variation of the kernel.
    window : WindowMatrix, optional
        Geometry window of the bispectrum (Sugiyama basis, e.g. from :func:`mesh3.compute_smooth3_spectrum_window`), whose theory
        gives the grid ``kin`` and the multipoles ``ells`` (all its theory multipoles on the same grid).
    kin : array, optional
        Theory grid ``(npoint, 2)``, all pairs of the nodes in :math:`k_1'` and :math:`k_2'`, if ``window`` is not given (else it must be
        that of ``window``).
    ells : tuple, optional
        Output Sugiyama multipoles; default: those of ``window``'s theory, else ``(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)``.
    ellsin : tuple, optional
        Theory Sugiyama multipoles, default ``ells``. More multipoles than the geometry window takes can be given: the removal moves
        power between multipoles. Multipoles :math:`(\ell_2, \ell_1, L)` not listed are taken from :math:`(\ell_1, \ell_2, L)` at
        :math:`(k_2', k_1')`, as in the geometry window. For the RIC of BGS (tree level), adding :math:`(0, 4, 4), (2, 2, 4), (2, 4, 2)` to
        the window's six brings :math:`W^{\rm geo} D^{BB}` within 1% of the full theory for :math:`k < 0.11` (the six alone: 3% on
        :math:`B_{000}`, 7% on :math:`B_{202}`).
    closure_kernel : WindowMatrix, optional
        :math:`W^{\rm ft}_{W^2}`, the window of :math:`P^{\rm ft}` with the field leg :math:`W^2`, giving :math:`\tau_{W^2}` and :math:`C^{BB}`
        below ``qsplit``, see :func:`compute_smooth3_closure_window`. Requires ``window``, and ``ellsin`` equal to its theory multipoles.
    modulation : ObservableTree, optional
        :math:`P^\eta`, for weights that depend on the density (see above), as lsstypes multipoles (e.g. :class:`Mesh2SpectrumPoles`, its
        multipoles on the same ``k``). :math:`P^\eta` is the cross power spectrum
        of :math:`W \eta` with the observed field :math:`\delta^{\rm obs}`, at the fiducial theory. :math:`\eta` is linear in the density,
        so it is a template field: for the FKP example, shell means of the cap (as :func:`ric_templates`) with response
        :math:`-\beta(r)`; for imaging weights, :math:`W \eta = -\delta^{\rm t}` and :math:`P^\eta = -P^{\rm ft} + P^{\rm tt}`, the windows
        ``'ft'`` and ``'tt'`` of :func:`compute_smooth2_ic_window` applied to the fiducial theory (with its shot noise). If given, also
        return :math:`D^{BP}`.
    closure_modulation : ObservableTree, optional
        With ``modulation`` and ``window``: :math:`C^{BP}` below ``qsplit`` (Eq. 153), from :math:`P^\eta_{W^2}`, the :math:`P^\eta` of
        ``modulation`` with the window :math:`W^2` on the field and the normalization :math:`\int W^3` (the window of
        :func:`compute_smooth2_modulation_window` with the geometry :math:`W^2`, as the closure kernel, applied to the fiducial theory), as
        lsstypes multipoles at points ``q`` fine at the window scale (see :func:`compute_smooth3_closure_window`).
    shotnoise : ObservableTree, optional
        :math:`N`, the shot noise removed with the templates, as lsstypes multipoles: ``compute_smooth2_ic_window(..., terms=('ft',),
        shotnoise=True)['ft', 'shotnoise']`` on the observable of ``['ft']``, relative to the reference as ``ft`` (the model with the global
        integral constraint minus that of :func:`gic_templates` alone). If given, also return :math:`D^{BN}`, per unit shot noise:
        :math:`\Delta B^N = \bar{n}^{-1} D^{BN} P`.
    closure_shotnoise : ObservableTree, optional
        With ``shotnoise`` and ``window``: :math:`C^{BN}` below ``qsplit`` (Eq. 153 with :math:`P^\eta_{W^2} \to -N_{W^2}`), from
        :math:`N_{W^2}`, the shot noise of the closure kernel (an
        :class:`IC2Model` with the field leg :math:`W^2`, see :func:`compute_smooth3_closure_window`) per unit shot noise, in the
        plane-parallel limit: the angle between the line of sight of the pair (the noise point) and that of the estimator (the closure leg)
        is neglected, which overcounts the anisotropic part (:math:`B_{202}` 2.5x on a synthetic angular cut, :math:`B_{000}` unaffected). As
        lsstypes multipoles: the ``'shotnoise'`` entry of the kernel's :func:`compute_smooth2_ic_window` (``shotnoise=True``) on the points
        ``q`` of the kernel, relative to :math:`\alpha` as the kernel.
    qsplit : float, default=0.02
        :math:`q_{\rm split}`: the closure leg is composed with the window above ``qsplit``, and given by the exact terms below (when
        given): a few times the window scale, above which the composition holds on the closure leg as on the binned legs.
    pin : ObservableTree, optional
        With ``modulation`` or ``shotnoise``: the grid (``k`` and multipoles; values ignored) on which :math:`D^{BP}` (:math:`D^{BN}`) takes the
        theory power spectrum :math:`P`, e.g. the fiducial power spectrum itself, so that ``out['BP'].dot(pin)`` applies it; default the
        multipoles 0, 2, 4 at the :math:`k_1'` nodes.
    nmu, nc, npsi : int, default=16, 64, 16
        Directions: Gauss-Legendre in :math:`\mu_1` and in the cosine of the angle between :math:`\hat{k}_1` and :math:`\hat{k}_2`, on
        sub-intervals clustered at :math:`\cos\theta_{12} = -1` (which resolves the squeezed third leg down to :math:`k_3 / k \sim 0.03`),
        uniform in the azimuth of :math:`\hat{k}_2` around :math:`\hat{k}_1`.
    nx : int, default=6
        Gauss-Legendre nodes in :math:`\mu'`.
    batch_size : int, default=8
        Grid points evaluated together.

    Returns
    -------
    out : dict
        ``'DBB'``, :math:`D^{BB}` as a ``scipy.sparse.csr_matrix`` ``(len(ells) * npoint, len(ellsin) * npoint)`` (multipole, then grid
        point, as the theory of the window), and with ``modulation``, ``'DBP'``, :math:`D^{BP}` ``(len(ells) * npoint, pin.size)``;
        with ``shotnoise``, ``'DBN'``, the same shape (the closure leg at :math:`k_3 \geq` ``qsplit`` only, with the exact term).
        With ``window``, also ``'BB'``, ``'BP'``, ``'BN'``: :class:`WindowMatrix`, the geometry window applied (``'BB'`` including
        :math:`C^{BB}` with ``closure_kernel``, also returned alone, ``'CBB'``; ``'BP'``, ``'BN'`` including the collapsed terms :math:`C^{BP}`,
        :math:`C^{BN}` of the closure leg, also returned alone, ``'CBP'``, ``'CBN'``).

    Examples
    --------
    >>> ft = compute_smooth2_ic_window(model, edgesin, k=np.arange(0.001, 0.2, 0.001), ells=(0, 2, 4, 6, 8), terms=('ft',), shotnoise=False)['ft']
    >>> out = compute_smooth3_ic_window(ft, window=wgeo)  # wgeo: read('window_mesh3_spectrum_sugiyama-diagonal_poles_*.h5')
    >>> dB = out['BB'].value() @ bth.ravel()  # bth: theory multipoles on the grid of wgeo's theory, (len(ells), npoint)

    With the exact closure leg, ``wft_w2`` the window of :math:`P^{\rm ft}` with the field leg :math:`W^2` (see
    :func:`compute_smooth3_closure_window`):

    >>> out = compute_smooth3_ic_window(ft, window=wgeo, closure_kernel=wft_w2)

    With the shot noise of the removal, ``N`` and ``n_w2`` the ``'shotnoise'`` entries of ``ft`` and ``wft_w2`` on their observables,
    ``sn`` the shot noise of the power spectrum, ``pth`` its theory multipoles (lsstypes):

    >>> out = compute_smooth3_ic_window(ft, window=wgeo, closure_kernel=wft_w2, shotnoise=ft.observable.clone(value=N.ravel()),
    ...                                 closure_shotnoise=wft_w2.observable.clone(value=n_w2.ravel()), pin=pth)
    >>> dB = out['BB'].value() @ bth.ravel() + sn * out['BN'].dot(pth)
    """
    if closure_kernel is not None and window is None: raise ValueError('closure_kernel requires the geometry window')
    if closure_shotnoise is not None and (window is None or shotnoise is None): raise ValueError('closure_shotnoise requires window and shotnoise')
    if closure_modulation is not None and (window is None or modulation is None): raise ValueError('closure_modulation requires window and modulation')
    # the closure leg: composed with the window (in D^BB, D^BP, D^BN) for k3 >= qsplit, and below qsplit too unless its exact term is given
    if window is not None:  # the theory grid and multipoles of the window (all its theory multipoles on the same grid)
        pole = next(iter(window.theory))
        kin_window, ells_window = np.asarray(pole.coords('k')), [tuple(int(l) for l in ell) for ell in window.theory.ells]
        if not all(np.shape(other.coords('k')) == kin_window.shape and np.allclose(other.coords('k'), kin_window) for other in window.theory):
            raise ValueError('the theory multipoles of window must share their grid')
        if kin is not None and not (np.shape(kin) == kin_window.shape and np.allclose(kin, kin_window)): raise ValueError('kin differs from the theory grid of window')
        if ells is not None and [tuple(ell) for ell in ells] != ells_window: raise ValueError('ells differs from the theory multipoles of window')
        kin, ells = kin_window, ells_window
    if ells is None: ells = [(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)]
    ells = [tuple(ell) for ell in ells]
    ellsin = ells if ellsin is None else [tuple(ell) for ell in ellsin]
    # The theory grid: npoint points (k1', k2'), all pairs of the nodes k1n (n1 values) and k2n (n2 values); i1, i2 the node indices of each
    # point, index[i1, i2] the point at given nodes
    kin = np.asarray(kin, dtype='f8')
    npoint = len(kin)
    k1n, i1 = np.unique(kin[:, 0], return_inverse=True)
    k2n, i2 = np.unique(kin[:, 1], return_inverse=True)
    index = np.full((k1n.size, k2n.size), -1)
    index[i1, i2] = np.arange(npoint)
    if np.any(index < 0): raise ValueError('kin must hold all pairs of the k1\' and k2\' nodes')
    if pin is None:  # the grid of the theory power spectrum of D^BP, D^BN: default its multipoles 0, 2, 4 at the k1' nodes
        pin = ObservableTree([Mesh2SpectrumPole(k=k1n, k_edges=np.column_stack([k1n - 1e-8, k1n + 1e-8]), num_raw=np.zeros(k1n.size)) for _ in range(3)], ells=[0, 2, 4])
    kpin, ellspin, _ = _spectrum_poles(pin, 'pin')
    # multipoles (l2, l1, L) not given are those of (l1, l2, L) at (k2', k1'), B being symmetric in its two legs (as in the geometry window)
    # ellsfull: the theory multipoles the integrand needs, ellsin and these swapped ones; source[f] = (index in ellsin, swapped or not)
    ellsfull, source = list(ellsin), [(ill, False) for ill in range(len(ellsin))]
    for ill, ell in enumerate(ellsin):
        if ell[0] != ell[1] and (ell[1], ell[0], ell[2]) not in ellsin:
            ellsfull.append((ell[1], ell[0], ell[2]))
            source.append((ill, True))
    if len(ellsfull) > len(ellsin) and not (k1n.size == k2n.size and np.allclose(k1n, k2n)):
        raise ValueError('the k1\' and k2\' nodes must be the same to take the multipoles (l2, l1, L) from (l1, l2, L)')

    # directions of k1^, k2^ about the line of sight z, and the weights projecting on the Sugiyama basis, <S_ell f> / <S_ell^2>: the multipoles
    # average over the orientation and the opening angle theta_12, uniform in cos(theta_12), with w = Phi / <Phi^2> (Eqs. 139-140)
    # mu1 = cos(k1^, line of sight): Gauss-Legendre nodes and weights
    mu1_nodes, wmu1_nodes = np.polynomial.legendre.leggauss(nmu)
    # c12 = cos(k1^, k2^): nodes and weights (summing to 2), Gauss-Legendre on the sub-intervals 1 + c12 in [0, 1e-3], [1e-3, 1e-2],
    # [1e-2, 1e-1], [1e-1, 2], with nc / 8, nc / 8, nc / 4 and the rest of the nodes, which resolves the squeezed third leg
    # (1 + c12 ~ k3^2 / (2 k1 k2)) down to k3 / k ~ 0.03
    c12_edges = np.array([0., 1e-3, 1e-2, 1e-1, 2.]) - 1.
    nsub = [max(nc // 8, 1), max(nc // 8, 1), max(nc // 4, 1)]
    nsub.append(nc - sum(nsub))
    if nsub[-1] < 1: raise ValueError('nc too small for the clustered nodes')
    c12_nodes, wc12_nodes = [], []
    for lo, hi, nn in zip(c12_edges[:-1], c12_edges[1:], nsub):
        nodes, weights = np.polynomial.legendre.leggauss(nn)
        c12_nodes.append(lo + (hi - lo) * (nodes + 1.) / 2.)
        wc12_nodes.append(weights * (hi - lo) / 2.)
    c12_nodes, wc12_nodes = np.concatenate(c12_nodes), np.concatenate(wc12_nodes)
    psi_nodes = (np.arange(npsi) + 0.5) * 2. * np.pi / npsi  # azimuth of k2^ around k1^: uniform
    # all ndir = nmu * nc * npsi directions, flattened, and their weights, summing to 1
    mu1, c12, psi = (x.ravel() for x in np.meshgrid(mu1_nodes, c12_nodes, psi_nodes, indexing='ij'))
    wdir = np.prod(np.meshgrid(wmu1_nodes, wc12_nodes, np.ones(npsi), indexing='ij'), axis=0).ravel()
    wdir = wdir / wdir.sum()
    sin1, sin12, zeros = np.sqrt(1. - mu1**2), np.sqrt(1. - c12**2), np.zeros_like(mu1)
    khat1 = np.stack([sin1, zeros, mu1], axis=-1)  # unit vector k1^, in the x-z plane, (ndir, 3)
    ehat1, ehat2 = np.stack([mu1, zeros, -sin1], axis=-1), np.stack([zeros, zeros + 1., zeros], axis=-1)  # orthonormal to k1^
    khat2 = c12[:, None] * khat1 + sin12[:, None] * (np.cos(psi)[:, None] * ehat1 + np.sin(psi)[:, None] * ehat2)  # unit vector k2^, (ndir, 3)
    khat1, khat2 = jnp.asarray(khat1), jnp.asarray(khat2)
    zhat = jnp.array([0., 0., 1.])  # the line of sight

    def sugiyama(ells, xhat1, xhat2):  # (len(ells), ...)
        xhat1, xhat2 = jnp.broadcast_arrays(xhat1, xhat2)
        return jnp.stack([jnp.broadcast_to(get_S(ell, z3=True)(xhat1, xhat2), xhat1.shape[:-1]) for ell in ells])

    # projection weights: sum over the directions of Sw[ill] f gives the multipole ells[ill] of f, (nell, ndir), with S_ell(k1^, k2^) and
    # <S_ell^2> analytic (_sugiyama_norm)
    Sw = sugiyama(ells, khat1, khat2) * wdir / jnp.asarray(_sugiyama_norm(ells))[:, None]

    # removal kernel, Eq. 142 [eq:ric_tau_kernel]: -P^ft(k, mu) = int dq dmu' / 2 tau(k, mu; q, mu') P^th(q, mu'), i.e. tau the window of -P^ft with
    # the Legendre factors, Eq. 145 [eq:ric_tau_matrix]: tau(k, mu; q_b, mu'_j) = sum_l L_l(mu) T[k, l, b, j], the Gauss-Legendre weights of mu'_j included (P^ft for all legs;
    # the line of sight on the removed field, P^tf, differs by 0.3%)
    kft, ellsft, _ = _spectrum_poles(ft.observable, 'ft (observable)')  # k and multipoles l of W^ft
    ellsq = [int(ell) for ell in ft.theory.ells]  # its theory multipoles l', on the bands in q (the same for all, checked)
    qedges, qcoords = np.asarray(next(iter(ft.theory)).edges('k')), np.asarray(next(iter(ft.theory)).coords('k'))
    if not all(np.shape(pole.edges('k')) == qedges.shape and np.allclose(pole.edges('k'), qedges) for pole in ft.theory):
        raise ValueError('the theory multipoles of ft must share their bands')
    wft = np.asarray(ft.value()).reshape(len(ellsft), kft.size, len(ellsq), len(qedges))
    band = np.diff(qedges, axis=-1)[:, 0] > 0.  # non-empty bands
    qmid = jnp.asarray(qcoords[band])  # band coordinates q_b
    mup, wmup = np.polynomial.legendre.leggauss(nx)  # mu'_j and their Gauss-Legendre weights
    # tau(k, mu; q_b, mu'_j) = sum_l L_l(mu) tau[k, l, b, j], (nk, nl, nb, nx), with (2l' + 1) L_l'(mu'_j) w_j / 2
    tau = jnp.asarray(-np.einsum('lkpb,pj->klbj', wft[..., band], np.asarray(_legendre(ellsq, mup)) * (2 * np.array(ellsq) + 1)[:, None] * wmup / 2.))

    def tau_at(k):  # tau at the wavenumbers k (k.size, nl, nb, nx), linear within the table, its first value below it, zero above
        return jnp.tensordot(_interp_matrix(k, kft), tau, axes=1)

    def rotate(leg):  # unit vectors at angle arccos(mu'_j) to the line of sight, in the plane of the leg and the line of sight: (ndir, nx, 3)
        perp = leg.at[..., 2].set(0.)
        nperp = jnp.sqrt(jnp.sum(perp**2, axis=-1))[:, None]
        ehat = jnp.where(nperp > 1e-12, perp / jnp.maximum(nperp, 1e-12), jnp.array([1., 0., 0.]))
        return np.sqrt(1. - mup**2)[None, :, None] * ehat[:, None, :] + mup[None, :, None] * zhat

    # Eq. 144 [eq:ric_bispectrum_rem]: sum_i < w int dq dmu' / 2 tau(k_i, mu_i; q, mu') B^th |_{k_i -> q} >, q in the plane of k_i and the line of
    # sight (tau being averaged over azimuths), the theory B from its multipoles, B = sum_ell B_ell(k1', k2') S_ell, interpolated on the nodes. Legs 1
    # and 2: the replaced leg q is on a grid axis and its direction does not depend on (k1, k2): the sum over directions is done once
    nell, nellin, nellfull, grid = len(ells), len(ellsin), len(ellsfull), (nmu, nc, npsi)
    # leg 1 replaced by q: for output multipole e, kernel multipole l, theory multipole m and mu'_j, the sum over directions of
    # Sw_e L_l(mu1) S_m(q^, k2^), q^ the rotated k1^: (nell, nl, nellfull, nx); proj2 the same for leg 2
    proj1 = jnp.einsum('ed,ld,mdj->elmj', Sw, _legendre(ellsft, khat1[:, 2]), sugiyama(ellsfull, rotate(khat1), khat2[:, None, :]))
    proj2 = jnp.einsum('ed,ld,mdj->elmj', Sw, _legendre(ellsft, khat2[:, 2]), sugiyama(ellsfull, khat1[:, None, :], rotate(khat2)))
    k1n, k2n, c12_nodes = jnp.asarray(k1n), jnp.asarray(k2n), jnp.asarray(c12_nodes)
    qmid_on_k1n, qmid_on_k2n = _interp_matrix(qmid, k1n), _interp_matrix(qmid, k2n)  # band centres on the nodes, (nb, n1), (nb, n2)
    if float(qmid[0]) < float(k1n[0]):
        logger.warning(f'the bands of q below the first theory node k = {float(k1n[0]):.4f} take the theory at that node')
    # the terms of the D^BP form, sum over legs i of P^eta(k_i, mu_i) [P(k_j) + P(k_l)]: name, k, multipoles, values (nell_eta, nk), whether
    # the closure leg below qsplit is left to its exact term
    etas = []
    if modulation is not None:  # D^BP, Eq. 149 [eq:ric_bispectrum_eta]; its closure leg below qsplit is C^BP with closure_modulation
        kmod, ellseta, peta = _spectrum_poles(modulation, 'modulation')
        etas.append(('DBP', jnp.asarray(kmod), ellseta, jnp.asarray(peta), closure_modulation is not None))
    if shotnoise is not None:  # D^BN, Eq. 150 [eq:ric_bispectrum_shotnoise], P^eta -> -N; its closure leg below qsplit is C^BN with closure_shotnoise
        kmod, ellseta, noise = _spectrum_poles(shotnoise, 'shotnoise')
        etas.append(('DBN', jnp.asarray(kmod), ellseta, -jnp.asarray(noise), closure_shotnoise is not None))
    kpin = jnp.asarray(kpin)

    def one_point(inodes):
        # The entries of one grid point (k1', k2') = (k1n[inodes[0]], k2n[inodes[1]]) of D^BB: row, on the theory points (any k1' node, this
        # k2'), and col, on (this k1', any k2' node), each (nell, nellfull, n_nodes); and of D^BP, bp (nell, len(ellspin), nkp)
        k1, k2 = k1n[inodes[0]], k2n[inodes[1]]
        kvec3 = -(k1 * khat1 + k2 * khat2)  # third leg, closing the triangle, for each direction (ndir, 3)
        knorm3 = jnp.sqrt(jnp.sum(kvec3**2, axis=-1))
        mu3 = kvec3[:, 2] / jnp.maximum(knorm3, 1e-10)  # its cosine to the line of sight
        knorm3_c12 = jnp.sqrt(k1**2 + k2**2 + 2. * k1 * k2 * c12_nodes)  # its norm, a function of c12 only, (nc,)
        # first leg replaced: B(q, k2), nodes (., k2); second: B(k1, q), nodes (k1, .)
        row = jnp.einsum('elmj,lbj->emb', proj1, tau_at(k1)[0]) @ qmid_on_k1n  # (nell, nellfull, n1)
        col = jnp.einsum('elmj,lbj->emb', proj2, tau_at(k2)[0]) @ qmid_on_k2n  # (nell, nellfull, n2)
        # third leg replaced (Eq. 143 [eq:bispectrum_rem]: "one side absorbs the change, averaged over the two choices"): the triangles {q, k2, -(q +
        # k2)} and {k1, q, -(k1 + q)}, half each, taken as the pairs (q, k2) and (k1, q), so that q, small where the kernel is largest, is a grid
        # magnitude (as the pairs (-(q + k2), k2), (k1, -(k1 + q)), the multipoles converge slowly: the squeezed triangle is then in the opening
        # angle). The kernel at k3 is taken per c12.
        qhat3 = rotate(kvec3)  # directions of q replacing the third leg, (ndir, nx, 3)
        Sw_legendre3 = (Sw[:, None] * _legendre(ellsft, mu3)[None]).reshape((nell, len(ellsft)) + grid)  # Sw_e L_l(mu3), (e, l, mu1, c12, psi)
        tau3 = 0.5 * tau_at(knorm3_c12)  # the kernel at the third leg, half for each of the two choices, (nc, nl, nb, nx)
        if closure_kernel is not None: tau3 = tau3 * (knorm3_c12 >= qsplit)[:, None, None, None]  # below qsplit: C^BB instead (Eq. 154)
        # the triangle read as (q, k2), on the k1' axis (added to row), then as (k1, q), on the k2' axis (added to col)
        for sugiyama3, qmid_on_nodes, on_k1n in [(sugiyama(ellsfull, qhat3, khat2[:, None, :]), qmid_on_k1n, True),
                                                 (sugiyama(ellsfull, khat1[:, None, :], qhat3), qmid_on_k2n, False)]:
            # sum over mu1 and psi at each c12 (the kernel depends on c12 through k3), then contracted with the kernel and put on the nodes
            proj3 = jnp.einsum('elacp,macpj->celmj', Sw_legendre3, sugiyama3.reshape((nellfull,) + grid + (nx,)))
            entries = jnp.einsum('celmj,clbj->emb', proj3, tau3) @ qmid_on_nodes
            if on_k1n: row = row + entries
            else: col = col + entries
        if not etas: return row, col
        # modulation, Eq. 149 [eq:ric_bispectrum_eta]: sum_j < w P^eta(k_j, mu_j) [P^th(k_i, mu_i) + P^th(k_l, mu_l)] > (Eq. 148 [eq:bispectrum_eta],
        # the Gaussian four-point term of the weights 1 + eta), P^eta = tau^eta * P^th at the fiducial theory (Eq. 147 [eq:ric_eta_kernel]), the
        # theory P interpolated on the nodes kpin; the shot noise of the removal has the same form
        # the three legs: norms per direction, norms per c12 and cosines to the line of sight
        knorms = [k1, k2, knorm3]
        knorms_c12 = [jnp.broadcast_to(k1, c12_nodes.shape), jnp.broadcast_to(k2, c12_nodes.shape), knorm3_c12]
        mus = [khat1[:, 2], khat2[:, 2], mu3]
        bps = []
        for _, kmod, ellseta, peta, cut3 in etas:
            bp = 0.
            for i in range(3):  # leg i carries P^eta
                # P^eta(k_i, mu_i), from its multipoles, its first value below kmod, zero above
                peta_i = jnp.sum((_interp_matrix(knorms[i], kmod) @ peta.T).T * _legendre(ellseta, mus[i]), axis=0)
                if i == 2 and cut3: peta_i = peta_i * (knorms[2] >= qsplit)  # the closure leg below qsplit: C^BP or C^BN instead (Eq. 154)
                for j in range(3):  # each other leg j carries P
                    if j == i: continue
                    # multipole e of P^eta(k_i, mu_i) L_l'(mu_j), per c12 (summed over mu1 and psi: the third leg's norm depends on c12 only),
                    # (nell, len(ellspin), nc); P_l'(k_j) interpolated on the nodes kpin
                    proj = (Sw[:, None] * (peta_i * _legendre(ellspin, mus[j]))[None]).reshape((nell, len(ellspin)) + grid).sum(axis=(-3, -1))
                    bp = bp + jnp.einsum('epc,cn->epn', proj, _interp_matrix(knorms_c12[j], kpin))
            bps.append(bp)
        return row, col, jnp.stack(bps)

    t0 = time.time()
    # all grid points: res = (row, col[, bp]), each with a leading axis npoint (lax.map compiles one_point)
    res = jax.lax.map(one_point, jnp.asarray(np.column_stack([i1, i2])), batch_size=batch_size)
    res = jax.block_until_ready(res)
    logger.info(f'{npoint} grid points in {time.time() - t0:.1f} s (including compilation)')

    # D^BB in CSR: each row (ell, point) holds, for each theory multipole, the points of the same k2' node (first leg replaced), then
    # of the same k1' node (second), or the transposed points; the point itself appears more than once (summed in products)
    t0 = time.time()
    import scipy.sparse
    src, transposed = np.array([s for s, _ in source]), np.array([t for _, t in source])[None, :, None]  # source of each multipole of ellsfull
    # theory point of each entry: for each output point and multipole of ellsfull, the n1 points of its row then the n2 of its column (with
    # (k1', k2') swapped for the swapped multipoles), (npoint, nellfull, n1 + n2)
    cols = np.concatenate([np.where(transposed, index[i2, :][:, None, :], index[:, i2].T[:, None, :]),
                           np.where(transposed, index[:, i1].T[:, None, :], index[i1, :][:, None, :])], axis=-1)
    nrow = nellfull * cols.shape[-1]  # entries per row of D^BB
    itype = 'i4' if nell * npoint * nrow < 2**31 - 1 else 'i8'
    cols = (cols + src[None, :, None] * npoint).astype(itype)  # column of D^BB: (theory multipole in ellsin, point)
    data = np.concatenate([np.asarray(res[0]), np.asarray(res[1])], axis=-1)  # the values, (npoint, nell, nellfull, n1 + n2)
    DBB = scipy.sparse.csr_matrix((np.transpose(data, (1, 0, 2, 3)).ravel(), np.broadcast_to(cols, (nell,) + cols.shape).ravel(),
                                   np.arange(nell * npoint + 1, dtype=itype) * nrow), shape=(nell * npoint, nellin * npoint))
    out = {'DBB': DBB}
    for ieta, (name, *_) in enumerate(etas):
        out[name] = np.transpose(np.asarray(res[2][:, ieta]), (1, 0, 2, 3)).reshape(nell * npoint, -1)  # (ell, point), (ell', node)
    logger.info(f'D^BB with {DBB.nnz} entries assembled in {time.time() - t0:.1f} s')
    if window is None: return out

    # Eqs. 141 [eq:ric_bispectrum_window] and 154 [eq:ric_bispectrum_full], W^BB = W^geo D^BB (+ C^BB below), on the device, on the grid of
    # points: sum over the output multipoles and the points sharing the k2' (k1') node
    t0 = time.time()
    wgeo = jnp.asarray(np.asarray(window.value()).reshape(-1, nell, npoint)[..., index])  # the geometry window on the grid, (nobs, nell, n1, n2)
    wgeo_row = jnp.einsum('oeab,abefn->ofnb', wgeo, res[0][index])  # (nobs, nellfull, n1, n2): theory points (node, k2' node of the row)
    wgeo_col = jnp.einsum('oeab,abefn->ofan', wgeo, res[1][index])  # (nobs, nellfull, n1, n2): theory points (k1' node of the row, node)
    # summed into the theory multipoles ellsin, the swapped ones transposed back, (nobs, nellin, n1, n2)
    wgeo_dbb = jnp.zeros((wgeo.shape[0], nellin) + index.shape)
    for ifull, (ill, swapped) in enumerate(source):
        tmp = wgeo_row[:, ifull] + wgeo_col[:, ifull]
        wgeo_dbb = wgeo_dbb.at[:, ill].add(jnp.swapaxes(tmp, -1, -2) if swapped else tmp)
    value = np.zeros((wgeo.shape[0], nellin, npoint))
    value[..., index.ravel()] = np.asarray(wgeo_dbb).reshape(wgeo.shape[0], nellin, -1)
    theory = window.theory
    if ellsin != ells:
        theory = ObservableTree([pole] * nellin, ells=ellsin)
    out['BB'] = WindowMatrix(observable=window.observable, theory=theory, value=value.reshape(wgeo.shape[0], -1))
    for name, *_ in etas:  # the theory power spectrum on the grid of pin
        out[name[1:]] = WindowMatrix(observable=window.observable, theory=pin, value=np.asarray(window.value()) @ out[name])
    logger.info(f'composed with the geometry window in {time.time() - t0:.1f} s')
    # the closure leg below qsplit, Eqs. 152-153: C^BB added to 'BB', C^BP to 'BP', C^BN to 'BN'
    if closure_kernel is not None and [tuple(int(l) for l in ell) for ell in window.theory.ells] != list(ellsin):
        raise ValueError('closure_kernel: ellsin must be the multipoles of the window theory')
    if any(term is not None for term in [closure_kernel, closure_modulation, closure_shotnoise]):
        t0 = time.time()
        closure = compute_smooth3_closure_window(window, kernel=closure_kernel, modulation=closure_modulation, shotnoise=closure_shotnoise,
                                                 pin=pin, qmax=qsplit)
        for name, value in closure.items():
            out[name] = out[name[1:]].clone(value=np.asarray(value.value()))
            out[name[1:]] = out[name[1:]].clone(value=np.asarray(out[name[1:]].value()) + np.asarray(value.value()))
        logger.info(f'closure leg below qsplit (exact) in {time.time() - t0:.1f} s')
    return out


def _legendre(ells, x):
    """Legendre polynomials :math:`\\mathcal{L}_\\ell(x)` for :math:`\\ell` in ``ells``, ``(len(ells),) + x.shape``."""
    return _legendre_matrix(max(ells), x)[np.array(ells)]


def _interp_matrix(x, xp):
    """
    Weights ``(x.size, xp.size)`` of the linear interpolation at ``x`` on the strictly increasing nodes ``xp`` (checked), with the one rule of
    this module outside the nodes: the first value below ``xp[0]``, zero above ``xp[-1]``; ``_interp_matrix(x, xp) @ fp`` is
    ``np.interp(np.maximum(x, xp[0]), xp, fp, right=0.)``.
    """
    xp = jnp.asarray(xp, dtype='f8')
    try:
        nodes = np.asarray(xp)
    except jax.errors.TracerArrayConversionError:  # traced nodes: not checked
        nodes = None
    if nodes is not None and np.any(np.diff(nodes) <= 0.): raise ValueError('the nodes must be strictly increasing')
    x = jnp.atleast_1d(jnp.asarray(x, dtype='f8'))
    below_last = x <= xp[-1]
    if xp.size == 1: return below_last[:, None] * 1.  # a single node: its value at and below it
    ileft = jnp.clip(jnp.searchsorted(xp, x) - 1, 0, xp.size - 2)
    frac = jnp.clip((x - xp[ileft]) / (xp[ileft + 1] - xp[ileft]), 0., 1.)
    rows = jnp.arange(x.size)
    return jnp.zeros((x.size, xp.size)).at[rows, ileft].add(below_last * (1. - frac)).at[rows, ileft + 1].add(below_last * frac)


def _spectrum_poles(spectrum, name='spectrum'):
    """
    ``(k, ells, values)`` of a power spectrum given as lsstypes multipoles (e.g. :class:`Mesh2SpectrumPoles`), whose multipoles share the same
    strictly increasing ``k`` (checked): ``k`` ``(nk,)``, the multipoles (int), the values ``(len(ells), nk)``.
    """
    ells = [int(ell) for ell in spectrum.ells]
    ks = [np.asarray(spectrum.get(ells=ell).coords('k'), dtype='f8').ravel() for ell in ells]
    if not all(k.shape == ks[0].shape and np.allclose(k, ks[0]) for k in ks): raise ValueError(f'{name}: the multipoles must share their k')
    if np.any(np.diff(ks[0]) <= 0.): raise ValueError(f'{name}: k must be strictly increasing')
    return ks[0], ells, np.array([np.asarray(spectrum.get(ells=ell).value(), dtype='f8').ravel() for ell in ells])


def _sugiyama_norm(ells):
    r"""
    :math:`\langle \Phi_e^2 \rangle` of the Sugiyama multipoles ``ells`` (third leg on the line of sight, :func:`utils.get_S` with
    ``z3=True``), over independent directions of :math:`\hat{\mathbf{k}}_1`, :math:`\hat{\mathbf{k}}_2`: the projection weight is
    :math:`w_e = \Phi_e / \langle \Phi_e^2 \rangle` (notes, Eqs. 139-140). With
    :math:`\Phi_{\ell_1 \ell_2 L} = \sum_m [(\ell_1\, \ell_2\, L; m\, {-m}\, 0) / H]\, y_{\ell_1}^m(\hat{\mathbf{k}}_1)\, y_{\ell_2}^{-m}(\hat{\mathbf{k}}_2)`,
    :math:`y_\ell^m = \sqrt{4\pi / (2\ell + 1)}\, Y_\ell^m`, :math:`H = (\ell_1\, \ell_2\, L; 0\, 0\, 0)`, the terms of different :math:`m` average
    to zero, :math:`\langle |y_\ell^m|^2 \rangle = 1 / (2\ell + 1)` and :math:`\sum_m (\ell_1\, \ell_2\, L; m\, {-m}\, 0)^2 = 1 / (2L + 1)`, so
    that :math:`\langle \Phi^2 \rangle = 1 / [(2\ell_1 + 1)(2\ell_2 + 1)(2L + 1) H^2]` (infinite if :math:`H = 0`, where :math:`\Phi = 0`).
    """
    out = []
    for ell1, ell2, ell3 in ells:
        H = float(wigner_3j(ell1, ell2, ell3, 0, 0, 0))
        out.append(1. / ((2 * ell1 + 1) * (2 * ell2 + 1) * (2 * ell3 + 1) * H**2) if abs(H) > 1e-12 else np.inf)
    return np.array(out)


def compute_smooth3_closure_window(window: WindowMatrix=None, kernel: WindowMatrix=None, modulation: ObservableTree=None,
                                   shotnoise: ObservableTree=None, k: np.ndarray=None, kedges: np.ndarray=None, ells: tuple=None,
                                   pin: ObservableTree=None, qmax: float=0.02, nmu: int=32, nmuq: int=16, nphi: int=32, nx: int=6) -> dict:
    r"""
    The closure (third, unbinned) leg of the Sugiyama bispectrum where it is at the window scale, :math:`q < q_{\rm split}` (``qmax``): the
    exact terms :math:`C^{BB}`, :math:`C^{BP}`, :math:`C^{BN}` of Eq. 154 [eq:ric_bispectrum_full], with the window acting on the binned legs;
    the exact counterparts of that part of the closure leg of :math:`W^{\rm geo} D^{BB}`, :math:`W^{\rm geo} D^{BP}`, :math:`W^{\rm geo} D^{BN}`
    (:func:`compute_smooth3_ic_window`, which calls it with ``closure_kernel``, ``closure_modulation``, ``closure_shotnoise`` and
    ``qmax = qsplit``). With :math:`\delta^{\rm t} = \sum_a W_a c_a` the removed field (Eq. 151), the windows of the two binned legs combine into
    that of :math:`W^2`, and the observed closure :math:`\mathbf{q} = -(\mathbf{k}_1 + \mathbf{k}_2)` sees the true long mode :math:`\mathbf{p}`
    through :math:`\tau_{W^2}`, the removal kernel :math:`\tau` (Eq. 145 [eq:ric_tau_matrix]) with the window :math:`W^2` on the field and the
    normalization :math:`\int W^3` (Eq. 152 [eq:ric_closure_removal]):

    .. math::

        C^{BB}_e[B](k_1, k_2) = \frac{1}{V_{k_1} V_{k_2}} \int_{S_{k_1}} d^3k_1 \int_{S_{k_2}} d^3k_2\, \Theta(q_{\rm split} - q)\,
        w_e(\hat{\mathbf{k}}_1, \hat{\mathbf{k}}_2) \int dp\, \frac{d\mu'}{2}\, \tau_{W^2}(q, \mu_q; p, \mu')\,
        \frac{B(p \hat{\mathbf{p}}, \mathbf{k}_1) + B(p \hat{\mathbf{p}}, \mathbf{k}_2)}{2},

    :math:`w_e` the projection weight of the multipole :math:`e`, :math:`\hat{\mathbf{p}}` at cosine :math:`\mu'` to the line of sight in the
    plane of :math:`\hat{\mathbf{q}}` and the line of sight (:math:`\tau_{W^2}` being averaged over azimuths, as :math:`\tau` in
    :math:`D^{BB}`); the closure leg of the terms of the :math:`D^{BP}` form carries :math:`P^\eta_{W^2}` (:math:`P^\eta` with the window
    :math:`W^2` on the field and the normalization :math:`\int W^3`), the two binned legs the theory :math:`P` (Eq. 153
    [eq:collapsed_multipoles]):

    .. math::

        C^{BP}_e[P](k_1, k_2) = \frac{1}{V_{k_1} V_{k_2}} \int_{S_{k_1}} d^3k_1 \int_{S_{k_2}} d^3k_2\, \Theta(q_{\rm split} - q)\,
        w_e(\hat{\mathbf{k}}_1, \hat{\mathbf{k}}_2) \left[P(k_1, \mu_1) + P(k_2, \mu_2)\right] P^\eta_{W^2}(q, \mu_q),

    and :math:`C^{BN}` (per unit shot noise) the same with :math:`P^\eta_{W^2} \to -N_{W^2}`. All are evaluated alike: one binned leg at its
    bin centre, integrated over its directions, the other leg :math:`-(\mathbf{k} + \mathbf{q})` through the shell overlap
    :math:`\Lambda_{k_1 k_2}`, half for each choice of the leg at its bin centre; :math:`B` read from the theory
    multipoles at the pair (long mode, that leg), bilinear on the nodes, each :math:`P` at the bin centre of its leg on the grid of ``pin``, with
    the one rule of :func:`compute_smooth3_ic_window` (linear between the nodes, the first value below, zero above). The shell overlap is
    exact for :math:`q \leq k_{\rm lo}` the lower edge of the reference shell, approximate (a warning) in the bins where the closure range
    reaches it. Observed-level terms (the observed shells), not composed with the geometry window; linear in the theory, nothing depends on a
    theory spectrum.

    Parameters
    ----------
    window : WindowMatrix, optional
        Geometry window of the bispectrum: its observable gives the bins (diagonal or not) and multipoles, its theory the grid and the theory
        multipoles of :math:`C^{BB}` (``(l2, l1, L)`` not given are taken from ``(l1, l2, L)`` at ``(k2', k1')``). Required with ``kernel``.
    kernel : WindowMatrix, optional
        For :math:`C^{BB}`: :math:`W^{\rm ft}_{W^2}(q; p)`, the window of :math:`P^{\rm ft}` with the field leg :math:`W^2` (:math:`\tau_{W^2}`
        is minus it, with the Legendre factors of Eq. 145):
        ``compute_smooth2_ic_window(model, edgesin, q, ells=(0, 2, 4), terms=('ft',), shotnoise=False)['ft']`` of an :class:`IC2Model` whose
        field leg is :math:`W^2`: ``geometry`` of patterns :math:`\Theta^2` built from random subsets disjoint from those of the templates (else
        the randoms' self-pairs bias it), radial :math:`g^2 \int \Theta^2`, ``norm`` :math:`\int W^3`. The removal is relative to the reference
        measurement, which has the global integral constraint too: the window of the templates with their :func:`gic_templates` minus that of
        :func:`gic_templates` alone, both passed explicitly with ``gic=False`` and built from the geometry :math:`W` (the :math:`\alpha`
        template has the pattern :math:`W`, not :math:`W^2`); each alone holds the cross with the survey-mean mode, orders of magnitude larger
        at the window scale. Output points ``q`` fine at the window scale (e.g. steps of 0.001 to 0.05), theory bands ``edgesin`` fine there
        too.

        >>> removed = templates + gic_templates([geometry], templates)  # geometry: W; geometry2: W^2 (disjoint random subsets)
        >>> ft = lambda ic: compute_smooth2_ic_window(IC2Model([geometry2], ic, norm=w3, lmax=lmax, gic=False), edgesin, q, terms=('ft',), shotnoise=False)['ft']
        >>> with_templates, alpha_only = ft(removed), ft(gic_templates([geometry], []))
        >>> kernel = with_templates.clone(value=with_templates.value() - alpha_only.value())
    modulation : ObservableTree, optional
        For :math:`C^{BP}`: :math:`P^\eta_{W^2}` as lsstypes multipoles at points ``q`` (e.g. :class:`Mesh2SpectrumPoles`).
    shotnoise : ObservableTree, optional
        For :math:`C^{BN}`: the same with :math:`N_{W^2}` (the ``'shotnoise'`` entry of the window of ``kernel``, on its observable).
    k, kedges : array, optional
        Without ``window``: the bins, centres ``(nk,)`` and edges ``(nk, 2)`` (diagonal), or ``(nk, 2)`` and ``(nk, 2, 2)`` (one per leg).
    ells : tuple, optional
        Output Sugiyama multipoles; default those of ``window.observable``, else ``(0, 0, 0), (2, 0, 2)``.
    pin : ObservableTree, optional
        For :math:`C^{BP}`, :math:`C^{BN}`: the grid (``k`` and multipoles; values ignored) of the theory power spectrum; default the
        multipoles 0, 2, 4 at the bin centres.
    qmax : float, default=0.02
        :math:`q_{\rm split}`: largest observed closure of the exact terms. The points ``q`` of the kernel, :math:`P^\eta_{W^2}`,
        :math:`N_{W^2}` are given at points (their observables carry no bands): the band of each extends half-way to its neighbours (the
        first one down to 0 at most), cut at ``qmax``, so that the exact terms and the composed closure leg (:math:`k_3 \geq q_{\rm split}`)
        partition the closure range.
    nmu, nmuq, nphi, nx : int, default=32, 16, 32, 6
        Quadrature: Gauss-Legendre in the cosines of the reference leg and of :math:`\hat{\mathbf{q}}` to the line of sight, uniform in their
        relative azimuth; Gauss-Legendre in the cosine :math:`\mu'` of the true long mode.

    Returns
    -------
    out : dict
        ``'CBB'`` (with ``kernel``), ``'CBP'`` (with ``modulation``), ``'CBN'`` (with ``shotnoise``): with ``window``, :class:`WindowMatrix`
        with its observable, and as theory that of ``window`` (``'CBB'``) or ``pin``; else arrays ``(sum of the observed bins over the
        multipoles, pin.size)``.
    """
    if kernel is not None and window is None: raise ValueError('kernel requires window (the theory grid of the bispectrum)')

    def closure_bands(q):  # the points q whose band, extending half-way to the neighbours (the first one down to 0 at most) and cut at qmax,
        # is not empty; d^3q / (2 pi)^3 of their bands; the largest band edge
        mid = 0.5 * (q[1:] + q[:-1])
        qlo, qhi = np.concatenate([[max(2. * q[0] - mid[0], 0.)], mid]), np.minimum(np.concatenate([mid, [2. * q[-1] - mid[-1]]]), qmax)
        keep = qhi > qlo
        return keep, (q[keep], (qhi[keep]**3 - qlo[keep]**3) / (6. * np.pi**2), qhi[keep].max(initial=0.))

    # observed bins, per output multipole: the centres (nk_e, 2) and edges (nk_e, 2, 2) of the two binned legs (one leg given: diagonal bins)
    if window is not None:
        observable = window.observable
        if ells is None: ells = [tuple(int(l) for l in ell) for ell in observable.ells]
        poles = [observable.get(ells=ell) for ell in ells]
        bins = [(np.asarray(pole.coords('k'), dtype='f8'), np.asarray(pole.edges('k'), dtype='f8')) for pole in poles]
    else:
        if ells is None: ells = [(0, 0, 0), (2, 0, 2)]
        bins = [(np.asarray(k, dtype='f8'), np.asarray(kedges, dtype='f8'))] * len(ells)
    bins = [(np.column_stack([kbin, kbin]), np.stack([kbin_edges, kbin_edges], axis=1)) if kbin.ndim == 1 else (kbin, kbin_edges)
            for kbin, kbin_edges in bins]
    ells = [tuple(ell) for ell in ells]
    norm = _sugiyama_norm(ells)

    # directions: the reference binned leg k^ at cosine mu to the line of sight z, in the x-z plane (the integrands being invariant about z),
    # Gauss-Legendre (nmu, 3); the observed closure q^, Gauss-Legendre in mu_q and uniform in its azimuth phi relative to k^, (nmuq, nphi, 3)
    mu, wmu = np.polynomial.legendre.leggauss(nmu)
    khat = np.stack([np.sqrt(1. - mu**2), np.zeros_like(mu), mu], axis=-1)
    muq, wmuq = np.polynomial.legendre.leggauss(nmuq)
    phi = (np.arange(nphi) + 0.5) * 2. * np.pi / nphi
    sinq = np.sqrt(1. - muq**2)
    qhat = np.stack([sinq[:, None] * np.cos(phi), sinq[:, None] * np.sin(phi), np.broadcast_to(muq[:, None], (nmuq, nphi))], axis=-1)
    cosqk = np.einsum('ix,jpx->ijp', khat, qhat)  # q^ . k^, (nmu, nmuq, nphi)

    terms = {}  # name: (the points q, the volumes of their bands, the largest band edge)
    if kernel is not None:  # C^BB
        # the observed closure q: the kernel's output points; the true long-mode bands p: its theory (the same for all multipoles, checked)
        qkernel, ellsq, _ = _spectrum_poles(kernel.observable, 'kernel (observable)')
        keep, terms['CBB'] = closure_bands(qkernel)
        ellsp = [int(ell) for ell in kernel.theory.ells]
        pedges, pmid = np.asarray(next(iter(kernel.theory)).edges('k'), dtype='f8'), np.asarray(next(iter(kernel.theory)).coords('k'), dtype='f8')
        if not all(np.shape(pole.edges('k')) == pedges.shape and np.allclose(pole.edges('k'), pedges) for pole in kernel.theory):
            raise ValueError('the theory multipoles of kernel must share their bands')
        wft_w2 = np.nan_to_num(np.asarray(kernel.value(), dtype='f8')).reshape(len(ellsq), qkernel.size, len(ellsp), pmid.size)[:, keep]
        mup, wmup = np.polynomial.legendre.leggauss(nx)  # mu', the cosine of the true long mode to the line of sight, and its weights
        # Eq. 145 [eq:ric_tau_matrix] with the field leg W^2: tau_W2(q, mu_q; p, mu'_j) = -sum_{l, l'} W^ft_W2_{l l'}(q; p) L_l(mu_q) (2 l' + 1) L_l'(mu'_j),
        # times the Gauss-Legendre weight w_j / 2 of the mu' average, all the input multipoles: (nq, nmuq, np, nx)
        tau_w2 = -np.einsum('lbsp,lm,sj->bmpj', wft_w2, _legendre(ellsq, muq), _legendre(ellsp, mup) * (2 * np.array(ellsp) + 1)[:, None] * wmup / 2.)
        # the true long mode p^, at cosine mu'_j to the line of sight in the plane of q^ (azimuth phi) and the line of sight: (nphi, nx, 3)
        sinp = np.sqrt(1. - mup**2)
        phat = np.stack([sinp * np.cos(phi)[:, None], sinp * np.sin(phi)[:, None], np.broadcast_to(mup, (nphi, nx))], axis=-1)
        # theory grid and multipoles, with the swapped ones (l2, l1, L) at (k2', k1'): full = (index in ellsin, multipole, swapped)
        kin = np.asarray(next(iter(window.theory)).coords('k'))
        if not all(np.shape(pole.coords('k')) == kin.shape and np.allclose(pole.coords('k'), kin) for pole in window.theory):
            raise ValueError('the theory multipoles of window must share their grid')
        ellsin = [tuple(int(l) for l in ell) for ell in window.theory.ells]
        nodes, i1 = np.unique(kin[:, 0], return_inverse=True)
        nodes2, i2 = np.unique(kin[:, 1], return_inverse=True)
        if not np.allclose(nodes, nodes2): raise ValueError('the theory grid must have the same nodes on both legs')
        npoint = len(kin)
        index = np.full((nodes.size, nodes.size), -1)
        index[i1, i2] = np.arange(npoint)
        if np.any(index < 0): raise ValueError('the theory grid must hold all pairs of its nodes')
        full = [(ill, ell, False) for ill, ell in enumerate(ellsin)]
        for ill, ell in enumerate(ellsin):
            if ell[0] != ell[1] and (ell[1], ell[0], ell[2]) not in ellsin: full.append((ill, (ell[1], ell[0], ell[2]), True))
        # the theory basis S_m(p^, k^) of each full multipole: B read at the pair (p, k), the long mode and the reference leg, (nfull, nmu, nphi, nx)
        shape = (nmu, nphi, nx, 3)
        phat_b, khat_b = jnp.asarray(np.broadcast_to(phat[None], shape)), jnp.asarray(np.broadcast_to(khat[:, None, None, :], shape))
        sugiyama_pk = np.array([np.asarray(get_S(ell, z3=True)(phat_b, khat_b)) for _, ell, _ in full])
        pmid_on_nodes = np.asarray(_interp_matrix(pmid, nodes))  # the long-mode bands on the nodes, (np, nnodes)

    peta_q = {}  # name: P^eta_W2 (or -N_W2) at (q_b, mu_q), (nq, nmuq)
    for name, term, sign in [('CBP', modulation, 1.), ('CBN', shotnoise, -1.)]:
        if term is None: continue
        q, ellseta, peta = _spectrum_poles(term, name)
        keep, terms[name] = closure_bands(q)
        peta_q[name] = np.einsum('lb,lj->bj', sign * np.nan_to_num(peta)[:, keep], _legendre(ellseta, muq))
    if peta_q:
        if pin is None:  # default: the multipoles 0, 2, 4 at the bin centres
            kpin = np.unique(np.concatenate([kbin.ravel() for kbin, _ in bins]))
            pin = ObservableTree([Mesh2SpectrumPole(k=kpin, k_edges=np.column_stack([kpin - 1e-8, kpin + 1e-8]), num_raw=np.zeros(kpin.size)) for _ in range(3)], ells=[0, 2, 4])
        kpin, ellspin, _ = _spectrum_poles(pin, 'pin')
        legendre_ref = _legendre(ellspin, mu)  # P on the reference leg, (nellp, nmu)

    # the shell overlap is exact for q <= k_lo of the reference shell: count the bins where the closure range reaches it
    qrange = max([qhi for _, _, qhi in terms.values()], default=0.)
    approximate = sum(int(min(kbin_edges[ik, 0, 0], kbin_edges[ik, 1, 0]) < qrange and max(kbin_edges[ik, 1, 0] - kbin_edges[ik, 0, 1], kbin_edges[ik, 0, 0] - kbin_edges[ik, 1, 1]) < qrange)
                      for _, kbin_edges in bins for ik in range(len(kbin_edges)))
    if approximate: logger.warning(f'{approximate:d} bins with a lower edge below the closure range {qrange:.4f}: their shell overlap is approximate')

    rows = {name: [] for name in terms}
    for ill, ell in enumerate(ells):
        kbin, kbin_edges = bins[ill]
        value = {name: np.zeros((len(kbin), len(ellsin), npoint) if name == 'CBB' else (len(kbin), len(ellspin), kpin.size)) for name in terms}
        for ik in range(len(kbin)):
            (klo1, khi1), (klo2, khi2) = kbin_edges[ik]
            gap = max(klo2 - khi1, klo1 - khi2)  # the smallest closure of the two shells
            # each binned leg in turn the reference (at its bin centre, integrated over its directions), the other one -(k_ref + q), half each,
            # which keeps the terms symmetric in the two legs
            for kref, (klo, khi), kother, (klo_o, khi_o), first in [(kbin[ik, 0], (klo1, khi1), kbin[ik, 1], (klo2, khi2), True),
                                                                    (kbin[ik, 1], (klo2, khi2), kbin[ik, 0], (klo1, khi1), False)]:
                for name, (q, vol, qhi) in terms.items():
                    if gap >= qhi: continue  # shells further apart than the closure range
                    # The shell overlap Lambda_{ref, other}(q, c), c = q^ . k^ (notes, below Eq. 153), with which the pair integrals are evaluated:
                    # changing variables from the other leg to the closure q = -(k_ref + k_other), with f evaluated at k_ref at its bin centre,
                    # int_{S_ref} d^3k_ref int_{S_other} d^3k_other f = int d^3q int_{S_ref} d^3k_ref 1[|k_ref + q| in S_other] f
                    # ~ V_ref int d^3q < Lambda_{ref, other}(q, q^ . k_ref^) f >_{k_ref^}, Lambda_{ref, other} the fraction of the reference shell
                    # [klo, khi] (measure k^2 dk) whose |k + q| lies in the other shell [klo_o, khi_o]. |k + q| increases with k when q <= k, so
                    # these k are an interval, [g(klo_o), g(khi_o)] cut to the reference shell, g(K) = -q c + sqrt(K^2 - q^2 (1 - c^2)) (no k
                    # reaches K < q sqrt(1 - c^2)): exact for q <= klo. Either leg can be the reference, V_ref Lambda_{ref, other} =
                    # V_other Lambda_{other, ref} (the pairs of the two shells with closure q): half of each
                    qb, c = q[:, None, None, None], cosqk[None]
                    qperp2 = qb**2 * (1. - c**2)  # the minimum of |k + q|^2 over k
                    g = lambda K: -qb * c + np.sqrt(np.clip(K**2 - qperp2, 0., None))
                    lo = np.where(klo_o**2 > qperp2, np.maximum(g(klo_o), klo), klo)  # all k reach klo_o if klo_o < q sqrt(1 - c^2)
                    hi = np.where(khi_o**2 > qperp2, np.minimum(g(khi_o), khi), -np.inf)  # none stays below khi_o if khi_o < q sqrt(1 - c^2)
                    lam = (np.maximum(hi, lo)**3 - lo**3) / (khi**3 - klo**3)  # (nq, nmu, nmuq, nphi)
                    if not np.any(lam): continue
                    kvec_other = -(kref * khat[None, :, None, None, :] + q[:, None, None, None, None] * qhat[None, None])  # (nq, nmu, nmuq, nphi, 3)
                    khat_other = kvec_other / np.sqrt(np.sum(kvec_other**2, axis=-1))[..., None]
                    khat_ref = np.broadcast_to(khat[None, :, None, None, :], khat_other.shape)
                    xhats = (khat_ref, khat_other) if first else (khat_other, khat_ref)
                    # Eqs. 152-153: pairs V_ref Lambda_{ref, other} over V_{k1} V_{k2}, half for this choice of the reference leg, times the
                    # projection weight w_e and the quadrature weights, (nq, nmu, nmuq, nphi)
                    weight = (0.5 * (2. * np.pi)**3 / (4. * np.pi / 3. * (khi_o**3 - klo_o**3)) * (wmu / 2.)[None, :, None, None]
                              * vol[:, None, None, None] * (wmuq / 2.)[None, None, :, None] / nphi * lam
                              * np.asarray(get_S(ell, z3=True)(*(jnp.asarray(xhat) for xhat in xhats))) / norm[ill])
                    if name == 'CBB':
                        # contracted with tau_W2 over q and mu_q, then with the theory basis over the reference leg's directions and mu': (nfull, np)
                        contracted = np.einsum('ifpj,nifj->np', np.einsum('bimf,bmpj->ifpj', weight, tau_w2, optimize=True), sugiyama_pk, optimize=True)
                        kref_on_nodes = np.asarray(_interp_matrix(kref, nodes))[0]
                        for (illin, _, swapped), entries in zip(full, contracted):
                            on_grid = np.einsum('p,pa,b->ab', entries, pmid_on_nodes, kref_on_nodes)  # (long-mode node, reference node)
                            value[name][ik, illin][index] += on_grid.T if swapped else on_grid
                    else:
                        # [P(k1, mu1) + P(k2, mu2)] P^eta_W2(q, mu_q): the reference leg and the other one, each P at the bin centre of its leg
                        weight = weight * peta_q[name][:, None, :, None]
                        p_ref = np.einsum('bimf,pi->p', weight, legendre_ref)
                        p_other = np.einsum('bimf,pbimf->p', weight, _legendre(ellspin, khat_other[..., 2]))
                        value[name][ik] += p_ref[:, None] * np.asarray(_interp_matrix(kref, kpin))[0] + p_other[:, None] * np.asarray(_interp_matrix(kother, kpin))[0]
        for name in terms: rows[name].append(value[name].reshape(len(kbin), -1))
    out = {name: np.concatenate(rows[name], axis=0) for name in terms}
    if window is not None:
        out = {name: WindowMatrix(observable=window.observable, theory=window.theory if name == 'CBB' else pin, value=value)
               for name, value in out.items()}
    return out
