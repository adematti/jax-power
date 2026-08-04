"""Comparison figure: windowed bispectrum, predicted vs measured, per triangle.

Four panels sharing the triangle index on x:
  1. k1 k2 k3 B(k1,k2,k3)  -- measured (points + error bars) vs predicted (line)
  2. dB / sigma for ell = 0
  3. dB / sigma for ell = 2
  4. the k1, k2, k3 values of each triangle

  python plot_window_vs_mocks.py --meshsize 64 --ellmax 2 --ninsub 16 --exact-box-limit
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import validate_scocc_window_mocks as V

# Reference-palette categorical slots 1-3 (light). Documented all-pairs result:
# worst-pair CVD dE 9.2, normal-vision 24.0. Identity is never carried by colour
# alone: measured vs predicted differ by marker vs line, and the k-panel series
# are directly labelled (slot 3 sits below 3:1 on the light surface -> relief rule).
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, INK3 = '#0b0b0b', '#52514e', '#8a8880'
SERIES = {0: BLUE, 2: ORANGE}


def make_figure(d, fn):
    ells = list(d['ells'])
    valid = d['valid']
    idx = np.where(valid)[0]
    x = np.arange(len(idx))
    k = d['xavg'][idx]                      # (ntri, 3)
    kprod = k.prod(axis=-1)

    nres = len(ells)
    fig, axes = plt.subplots(2 + nres, 1, figsize=(max(9., 0.20 * len(x) + 4.), 4.2 + 1.9 * nres),
                             sharex=True, height_ratios=[3.0] + [1.5] * nres + [1.6],
                             gridspec_kw=dict(hspace=0.12))
    for ax in axes:
        ax.grid(True, color='#e6e5e1', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(INK3)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)

    # ---- panel 1: k1 k2 k3 B ----
    ax = axes[0]
    for ell in ells:
        ill = ells.index(ell)
        c = SERIES[ell]
        ax.errorbar(x, kprod * d['wind_mean'][ill][idx], yerr=kprod * d['wind_err'][ill][idx],
                    fmt='o', ms=4.5, mfc='white', mec=c, mew=1.4, ecolor=c, elinewidth=1.0,
                    capsize=0, zorder=3, label=f'measured  $\\ell={ell}$')
        ax.plot(x, kprod * d['pred'][ill][idx], '-', color=c, lw=1.8, zorder=2,
                label=f'full window  $\\ell={ell}$')
        if 'pred_ident' in d:
            ax.plot(x, kprod * d['pred_ident'][ill][idx], '--', color=c, lw=1.3, alpha=0.85, zorder=2,
                    label=f'identity $\\times Q^\\infty$  $\\ell={ell}$')
    ax.axhline(0., color=INK3, lw=0.8, zorder=1)
    ax.set_ylabel(r'$k_1 k_2 k_3\, B_\ell(k_1,k_2,k_3)$', color=INK, fontsize=10)
    leg = ax.legend(ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper left')
    ttl = (f"windowed bispectrum: predicted (theory $\\times$ window) vs measured   "
           f"[{d['geometry']}, {d['nmocks']} mocks, mesh ${d['meshsize']}^3$, "
           f"$\\bar n$={d['nbar']:g}, $\\ell_{{\\max}}$={d['ellmax']}, "
           f"ninsub={d['ninsub']}, split={'on' if d['exact_box_limit'] else 'off'}]")
    ax.set_title(ttl, color=INK, fontsize=10.5, loc='left', pad=10)

    # ---- panels 2..: residuals in sigma, one per multipole ----
    for j, ell in enumerate(ells):
        ill = ells.index(ell)
        ax = axes[1 + j]
        c = SERIES[ell]
        err = d['wind_err'][ill][idx]
        nsig = np.where(err > 0, (d['pred'][ill][idx] - d['wind_mean'][ill][idx]) / np.where(err > 0, err, 1.), np.nan)
        for lev, ls in [(1., ':'), (3., '--')]:
            for sgn in (-1., 1.):
                ax.axhline(sgn * lev, color=INK3, lw=0.7, ls=ls, zorder=1)
        ax.axhline(0., color=INK3, lw=0.8, zorder=1)
        if 'pred_ident' in d:
            nsig_d = np.where(err > 0, (d['pred_ident'][ill][idx] - d['wind_mean'][ill][idx]) / np.where(err > 0, err, 1.), np.nan)
            ax.plot(x, nsig_d, '--', color=c, lw=1.2, alpha=0.7, zorder=2)
        ax.plot(x, nsig, 'o-', color=c, ms=4., lw=1.1, mfc='white', mec=c, mew=1.2, zorder=3)
        ax.set_ylabel(r'$\Delta B / \sigma$' + f'\n$\\ell={ell}$', color=INK, fontsize=10)
        lim = max(3.6, np.nanmax(np.abs(nsig)) * 1.12,
                  min(np.nanmax(np.abs(nsig_d)) * 1.12 if 'pred_ident' in d else 0., 30.))
        ax.set_ylim(-lim, lim)
        med = np.nanmedian(np.abs(nsig))
        lbl = f'median $|\\Delta B/\\sigma|$ = {med:.2f}'
        if 'pred_ident' in d:
            lbl += f'   (identity $\\times Q^\\infty$: {np.nanmedian(np.abs(nsig_d)):.2f})'
        ax.text(0.006, 0.93, lbl, transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=INK2)
        if j == 0:  # band key once, not on every panel
            ax.text(0.994, 0.93, r'dotted $1\sigma$   dashed $3\sigma$', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8.5, color=INK3)

    # ---- last panel: the k1, k2, k3 values ----
    ax = axes[-1]
    for leg_i, (c, lab, ls) in enumerate(zip((BLUE, ORANGE, AQUA), ('$k_1$', '$k_2$', '$k_3$'),
                                             ('-', '-', '-'))):
        ax.plot(x, k[:, leg_i], ls, color=c, lw=1.6, zorder=3 + leg_i, label=lab)
    # legend above the axes: the k lines sweep the full panel, so any in-axes
    # placement collides with one of them
    ax.legend(ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, loc='lower left',
              bbox_to_anchor=(0., 1.005), handlelength=1.6, columnspacing=1.6)
    ax.set_ylabel(r'$k$  [$h\,\mathrm{Mpc}^{-1}$]', color=INK, fontsize=10)
    ax.set_xlabel('triangle index', color=INK, fontsize=10)
    ax.set_xlim(-0.8, len(x) - 0.2)

    fig.savefig(fn, dpi=160, bbox_inches='tight', facecolor='#fcfcfb')
    print('wrote', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--meshsize', type=int, default=64)
    ap.add_argument('--boxsize', type=float, default=None, help='box size [Mpc/h]; 500 gives sigma=6.4 cells and kNyq=0.40, vs 1.6 cells and kmax AT Nyquist at 1000')
    ap.add_argument('--geometry', choices=['cutsky', 'box', 'periodic'], default='cutsky')
    ap.add_argument('--ellmax', type=int, default=2)
    ap.add_argument('--ellwmax', type=int, default=5)
    ap.add_argument('--ninsub', type=int, default=16)
    ap.add_argument('--worder', type=int, default=3)
    ap.add_argument('--exact-box-limit', action='store_true')
    ap.add_argument('--nmocks', type=int, default=16)
    ap.add_argument('--mask-sigma', type=float, default=None, help='selection width [Mpc/h]; default boxsize/6')
    ap.add_argument('--nbar', type=float, default=None, help='tracer density; N_eff = nbar (2pi)^1.5 sigma^3')
    ap.add_argument('--rfac', type=float, default=None, help='randoms oversampling (default 10)')
    ap.add_argument('--theory-los', choices=['local', 'z'], default='local')
    ap.add_argument('--los', choices=['local', 'z'], default='local')
    ap.add_argument('--dk', type=float, default=None)
    ap.add_argument('--kmax', type=float, default=None)
    ap.add_argument('--buffer-size', type=int, default=0, help='band-powers at once in the estimator')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # V.<name> works here (unlike inside the harness run as a script, where the
    # module is __main__ and importing it by name binds a second copy)
    if args.nbar is not None: V.nbar = args.nbar
    if args.rfac is not None: V.RFAC = args.rfac
    if args.dk is not None: V.DK = args.dk
    if args.kmax is not None: V.KMAX = args.kmax
    V.set_geometry(args.meshsize, args.mask_sigma, boxsize=args.boxsize)
    d = V.run(nmocks=args.nmocks, geometry=args.geometry, ellmax=args.ellmax, ellwmax=args.ellwmax,
              ninsub=args.ninsub, worder=args.worder, exact_box_limit=args.exact_box_limit,
              buffer_size=args.buffer_size, batch_size=args.batch_size, theory_los=args.theory_los, los=args.los,
              from_cache=True)
    out = args.out or (V.CACHE_DIR / f"window_vs_mocks_{args.geometry}{V._box_suffix()}_mesh{args.meshsize}_n{args.nmocks}"
                       f"_ellmax{args.ellmax}{'_split' if args.exact_box_limit else ''}.png")
    make_figure(d, out)
