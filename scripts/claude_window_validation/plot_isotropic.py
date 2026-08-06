"""Isotropic Gaussian: pipeline ellmax scan vs the EXACT analytic matrix and the mocks.

The analytic reference here is GROUND TRUTH (Q000 analytic, separable expansion exact to ~1e-16 in 14
terms), so measured/analytic is a real accuracy statement and pipeline/analytic isolates the
pipeline's own error. Two widths give the ellmax-vs-extent lever arm.
argv: sigma [sigma ...]
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', '_tests')
SIGS = [int(a) for a in sys.argv[1:]] or [200, 350]
CM = plt.get_cmap('viridis')
fig, axes = plt.subplots(3, len(SIGS), figsize=(max(8.6, 7.2 * len(SIGS)), 9.6), squeeze=False,
                         gridspec_kw=dict(height_ratios=[3, 1.8, 1.3], hspace=0.13, wspace=0.2))
for c, sg in enumerate(SIGS):
    fn = os.path.join(OUT, f'iso{sg}_run.npz')
    if not os.path.exists(fn):
        print(f'  missing {fn}'); continue
    z = np.load(fn)
    xa, valid, bm, be = z['xavg'], z['valid'], z['meas'], z['err']
    EMS = [int(e) for e in z['ellmaxes']]; LMS = [int(l) for l in z['lmaxes']]
    iv = np.where(valid)[0]; idx = np.arange(len(iv)); legs = xa[iv]
    sig = np.abs(bm[0][iv]) > 5. * be[0][iv]
    sq = xa[iv, 0] / np.maximum(xa[iv, 2], 1e-12) < 0.55
    # prefer the REBINNED analytic (output-side k^2-weighted bin average, NOUT=4) if present:
    # without it the analytic evaluates the output at the bin representative and is 2.46x off at
    # sigma=200 / 12.4x at sigma=350. With it, 1.05.
    # take the HIGHEST available NOUT: the analytic turns out to be converged in it already at 4
    # (median meas/pred 1.0508 / 1.0488 / 1.0463 at NOUT 4 / 8 / 16, worst bins unmoved), so this
    # is a labelling correction rather than a numerical one -- but the label should not claim 4.
    import glob as _glob, re as _re
    _rebs = {int(_re.search(r'_ana_nout(\d+)\.npy', f).group(1)): f
             for f in _glob.glob(os.path.join(OUT, f'iso{sg}_ana_nout*.npy'))}
    if _rebs:
        _n = max(_rebs)
        ana = np.load(_rebs[_n]); ana_lab = f'analytic, REBINNED output (NOUT={_n})'
        ana_old = z[f'ana_{max(LMS)}']
    else:
        ana = z[f'ana_{max(LMS)}']; ana_lab = f'analytic, NO output rebin (LMAX={max(LMS)})'
        ana_old = None
    cols = [CM(i / max(len(EMS) - 1., 1)) for i in range(len(EMS))]
    # Finer window s-grids (pipeline_wcoords_scan.py). The run npz was built at the stitched_window3
    # default wcoords=256; refining it is a LARGER effect than ellmax, so show it on the same axes.
    import glob, re
    WCS = []
    for f in sorted(glob.glob(os.path.join(OUT, f'iso{sg}_wc*_em*_r-3_4.npy'))):
        mm = re.search(r'_wc(\d+)_em(\d+)_', os.path.basename(f))
        WCS.append((int(mm.group(1)), int(mm.group(2)), np.load(f)))
    WCS.sort()
    wcol = {512: '#0b7285', 1024: '#c2255c', 2048: '#5f3dc4'}
    ax, axr, axk = axes[0, c], axes[1, c], axes[2, c]
    for a in (ax, axr, axk):
        for j in np.where(sq)[0]:
            a.axvspan(j - .5, j + .5, color='#ffd27f', alpha=.35, lw=0, zorder=0)
        a.grid(alpha=.22, lw=.6); a.set_xlim(-0.6, len(iv) - 0.4)
    ax.errorbar(idx, np.abs(bm[0][iv]), be[0][iv], fmt='o-', ms=5, color='k', lw=1.4, capsize=0,
                zorder=6, label=f'measured $B_{{000}}$ ({int(z["nmocks"])} pairs)')
    for j, em in enumerate(EMS):
        ax.plot(idx, np.abs(z[f'pred_{em}'][0][iv]), 's-', ms=4.4, mfc='none', color=cols[j], lw=1.3,
                label=f'pipeline $\\ell_\\mathrm{{max}}={em}$' + (', $w=256$' if WCS else ''))
    for wc, em, pr in WCS:
        ax.plot(idx, np.abs(pr[iv]), '^-', ms=5, mfc='none', color=wcol.get(wc, '#555'), lw=1.5,
                label=f'pipeline $\\ell_\\mathrm{{max}}={em}$, $w={wc}$')
    ax.plot(idx, np.abs(ana), 'D--', ms=4.6, mfc='none', color='#a33', lw=1.8, label=f'EXACT {ana_lab}')
    if ana_old is not None:
        ax.plot(idx, np.abs(ana_old), 'v:', ms=4, mfc='none', color='#c99', lw=1.1,
                label='analytic, output at bin centre (old)')
    ax.set_yscale('log'); ax.set_xticklabels([]); ax.set_ylim(1e1, 1e6)
    if c == 0: ax.set_ylabel(r'$|B_{000}|$  $[(h^{-1}\mathrm{Mpc})^6]$')
    ax.legend(fontsize=7.8, frameon=False, loc='lower left', ncol=2)
    ax.set_title(f'isotropic Gaussian, $\\sigma={sg}$   (window Q000 EXACT, no fit)\n'
                 f'measured Q vs analytic: rel {float(z["qrel"]):.1e}', fontsize=10)
    # error bars are the MEASUREMENT's, propagated: sigma(meas)/|pred|. They decide whether a
    # departure from 1 means anything -- the medians quoted elsewhere hide them.
    for j, em in enumerate(EMS):
        pr = z[f'pred_{em}'][0][iv]
        r, er = bm[0][iv] / pr, be[0][iv] / np.abs(pr)
        axr.errorbar(idx[sig], r[sig], er[sig], fmt='s-', ms=5, color=cols[j], lw=1.3, capsize=2,
                     elinewidth=1., label=f'$\\ell_\\mathrm{{max}}={em}$')
        axr.errorbar(idx[~sig], r[~sig], er[~sig], fmt='s', ms=3, color=cols[j], alpha=.22,
                     elinewidth=.8, capsize=1)
    for wc, em, pr in WCS:
        r, er = bm[0][iv] / pr[iv], be[0][iv] / np.abs(pr[iv])
        axr.errorbar(idx[sig], r[sig], er[sig], fmt='^-', ms=5.4, color=wcol.get(wc, '#555'), lw=1.5,
                     capsize=2, elinewidth=1., label=f'$\\ell_\\mathrm{{max}}={em}$, $w={wc}$')
        axr.errorbar(idx[~sig], r[~sig], er[~sig], fmt='^', ms=3, color=wcol.get(wc, '#555'),
                     alpha=.22, elinewidth=.8, capsize=1)
    if ana_old is not None:
        rq_ = bm[0][iv] / ana_old
        axr.plot(idx[sig], rq_[sig], 'v:', ms=4, color='#c99', lw=1.,
                 label='analytic, no output rebin')
    ra, era = bm[0][iv] / ana, be[0][iv] / np.abs(ana)
    axr.errorbar(idx[sig], ra[sig], era[sig], fmt='D--', ms=5, color='#a33', lw=1.4, capsize=2,
                 elinewidth=1., label='EXACT analytic')
    axr.errorbar(idx[~sig], ra[~sig], era[~sig], fmt='D', ms=3, color='#a33', alpha=.22,
                 elinewidth=.8, capsize=1)
    axr.axhline(1., color='k', lw=.9); axr.axhspan(.95, 1.05, color='k', alpha=.08, lw=0)
    axr.set_ylim(0.4, 2.4); axr.set_xticklabels([])
    if c == 0: axr.set_ylabel('measured / predicted')
    axr.legend(fontsize=7.4, frameon=False, ncol=2, loc='upper right')
    axr.text(.012, .04, f'only {int(sig.sum())} of {len(iv)} bins are detected at $>5\\sigma$ '
             f'({int(z["nmocks"])} pairs, large symbols); bars = $\\sigma$(meas)/|pred|',
             transform=axr.transAxes, fontsize=7.4, color='#a33')
    for j, (lab, cc, mk) in enumerate((('$k_1$', '#1f4e79', 'o'), ('$k_2$', '#d95f02', 's'),
                                       ('$k_3$', '#7570b3', '^'))):
        axk.plot(idx, legs[:, j], mk + '-', ms=4, color=cc, lw=1.1, label=lab)
    axk.set_xlabel('bispectrum bin index')
    if c == 0: axk.set_ylabel(r'$k_i$  $[h\,\mathrm{Mpc}^{-1}]$')
    axk.legend(fontsize=8, frameon=False, ncol=3, loc='upper left')
    print(f'  sigma={sg}: {int(sig.sum())}/{len(iv)} bins >5sigma')
    for em in EMS:
        print(f'    pipeline ellmax={em:2d}: meas/pred {np.median((bm[0][iv]/z[f"pred_{em}"][0][iv])[sig]):.4f} '
              f'| pipeline/EXACT {np.median((z[f"pred_{em}"][0][iv]/ana)[sig]):.4f}')
    for wc, em, pr in WCS:
        ns = (bm[0][iv][sig] - pr[iv][sig]) / be[0][iv][sig]
        print(f'    pipeline ellmax={em:2d} wcoords={wc:5d}: meas/pred '
              f'{np.median((bm[0][iv]/pr[iv])[sig]):.4f} | pipeline/EXACT '
              f'{np.median((pr[iv]/ana)[sig]):.4f} | chi2/n {np.sum(ns**2)/sig.sum():.1f} '
              f'| max|nsig| {np.max(np.abs(ns)):.1f}')
    print(f'    EXACT analytic ({ana_lab}): meas/pred {np.median((bm[0][iv]/ana)[sig]):.4f}')
    for L in LMS:
        print(f'      [no output rebin, LMAX={L:2d}]: {np.median((bm[0][iv]/z[f"ana_{L}"])[sig]):.4f}')
fig.suptitle('Isotropic Gaussian selection: pipeline vs an EXACT analytic window matrix', fontsize=12.5, y=0.965)
fig.subplots_adjust(left=0.115 if len(SIGS) == 1 else 0.075, right=0.985, top=0.9, bottom=0.06)
_fn = os.path.join(OUT, 'iso_ellmax_' + '_'.join(str(x) for x in SIGS) + '.png')
fig.savefig(_fn, dpi=140)
print('wrote', _fn)
