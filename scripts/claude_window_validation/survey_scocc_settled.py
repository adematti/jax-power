"""The settled configuration -- ellmax=16, wcoords=512, noutsub=8 -- on the REALISTIC survey.

Fixed on the isotropic Gaussian test, where at these settings the pipeline reproduces the exact
analytic window matrix to 0.3% in the median (chi2/n 7.0, max|nsig| 5.0 against the mocks, i.e.
matching the analytic's own 7.0). This applies the same settings to the octant-with-holes survey
selection, which the analytic arm cannot adjudicate.

Note the survey window Q must be REBUILT: the cached p3_scocc_Q_ellwmax0.h5 was made with
stitched_window3's default coords = logspace(-3, 4, 256), and 256 is exactly the under-resolution
this scan established. Cached separately as ..._w512.h5 so the old runs stay reproducible.

Metric is per-bin, not a median: chi2/n over the >5sigma bins and max|nsig|. Medians hid per-bin
structure repeatedly in this investigation.
argv: [ellmax] [wcoords] [noutsub]
"""
import os, sys, time, itertools
import numpy as np, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import survey_window_geom as SG
from survey_window_geom import get_mattrs, make_particles, stitched_window3, KW_PAINT
from jaxpower import (BinMesh3SpectrumPoles, compute_smooth3_spectrum_window,
                      get_smooth3_window_bin_attrs, compute_normalization)
from lsstypes import read
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', '_tests')

ELLMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 16
WC = int(sys.argv[2]) if len(sys.argv) > 2 else 512
NOUTSUB = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ELLS = ELLSIN = (0, 2)
DK, KMAX, LOS = 0.015, 0.09, 'local'

mattrs = get_mattrs()
d = np.load(os.path.join(OUT, 'surv_p3_scocc_inject.npz'))
bm, be, AMP = d['meas'], d['err'], float(d['amp'])
bands = tuple(int(b) for b in d['bands']); kt_ref, ITH = d['ktheory'], int(d['ith'])
bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': DK, 'max': KMAX}, basis='scoccimarro', ells=ELLS)
xa = np.asarray(bin3.xavg)
valid = (xa[:, 2] >= np.abs(xa[:, 0] - xa[:, 1])) & (xa[:, 2] <= xa[:, 0] + xa[:, 1])
tgt = np.all(np.abs(xa - kt_ref[ITH]) < DK / 2., axis=1)
edges1d = np.arange(0., 2. * KMAX + DK, DK)

QFN = os.path.join(OUT, f'p3_scocc_Q_ellwmax0_w{WC}.h5')
if not os.path.exists(QFN):
    t0 = time.time()
    parts = [make_particles(seed=s_) for s_ in range(3)]
    norm3 = compute_normalization(*[p.clone(attrs=mattrs).paint(**KW_PAINT) for p in parts])
    kw = get_smooth3_window_bin_attrs(ELLS, ellsin=ELLSIN, basis='scoccimarro', ellmax=ELLMAX)
    kw['ells'] = [q for q in kw['ells'] if max(np.atleast_1d(q).ravel()) <= 0]   # ellwmax=0, W000 only
    kw.setdefault('mask_edges', '')
    Q, _ = stitched_window3(parts, norm3, kw, LOS, mattrs=mattrs,
                            coords=jnp.logspace(-3., 4., WC))
    Q.write(QFN); print(f'  Q rebuilt at wcoords={WC} in {time.time()-t0:.0f}s', flush=True)
Q = read(QFN)

print(f'survey octant+holes: {int(valid.sum())} valid output triangles | '
      f'ellmax={ELLMAX}, wcoords={WC}, noutsub={NOUTSUB}, ellwmax=0\n', flush=True)
t0 = time.time()
wm = compute_smooth3_spectrum_window(Q, edgesin=edges1d, ellsin=ELLSIN, bin=bin3,
                                     flags=('fftlog',), batch_size=4, ellmax=ELLMAX, noutsub=NOUTSUB)
M = np.asarray(wm.value()).real
nin = M.shape[1] // len(ELLSIN)
eth = np.asarray(next(iter(wm.theory)).edges('k'))
want = np.array([[edges1d[b], edges1d[b + 1]] for b in bands])
# the spike must sit on ALL 6 leg permutations: B is symmetric, and placing it on the sorted
# representative alone is the ordered-octant bug that cost 45% (see SCOCCIMARRO_WINDOW_STATUS 3.1)
cols = sorted({int(h) for pm in itertools.permutations(range(3))
               for h in np.where(np.all(np.abs(eth - want[list(pm)][None]) < 1e-9, axis=(1, 2)))[0]})
thv = np.zeros((len(ELLSIN), nin)); thv[0, cols] = AMP
rs = M.reshape(len(ELLS), len(xa), len(ELLSIN), nin)[0].sum(axis=(1, 2))
pred = (M @ thv.reshape(-1)).reshape(len(ELLS), len(xa))
dt = time.time() - t0

iv = np.where(valid)[0]
m, e, p = bm[0][iv], be[0][iv], pred[0][iv]
sig = np.abs(m) > 5. * e
ns = (m[sig] - p[sig]) / e[sig]
sq = xa[iv, 0] / np.maximum(xa[iv, 2], 1e-12) < 0.55
print(f'  time {dt:.0f}s | row sums {rs[valid].mean():.4f} (Qinf = 0.999997)')
print(f'  B0 at injected bin      {float(bm[0][tgt] / pred[0][tgt]):.4f}')
print(f'  detected bins           {int(sig.sum())} of {len(iv)} at >5sigma')
print(f'  chi2/n vs mocks         {np.sum(ns**2) / max(sig.sum(), 1):.1f}')
print(f'  max|nsig|               {np.max(np.abs(ns)):.1f}')
print(f'  median meas/pred        {np.median((m / p)[sig]):.4f}')
print(f'    squeezed  k1/k3<0.55  {np.median((m / p)[sig & sq]) if (sig & sq).any() else float("nan"):.4f}')
print(f'    the rest              {np.median((m / p)[sig & ~sq]) if (sig & ~sq).any() else float("nan"):.4f}')
np.savez(os.path.join(OUT, f'scocc_settled_em{ELLMAX}_w{WC}_n{NOUTSUB}.npz'),
         xavg=xa, valid=valid, meas=bm, err=be, tgt=tgt, pred=pred, rs=rs,
         ellmax=ELLMAX, wcoords=WC, noutsub=NOUTSUB)
print(f'\nsaved scocc_settled_em{ELLMAX}_w{WC}_n{NOUTSUB}.npz')
