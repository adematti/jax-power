r"""Isotropic-Gaussian ellmax test: pipeline vs an EXACT analytic reference.

For W = exp(-d^2/2sigma^2), Q000 is analytic (eq:gaussian_q000_exact) and its separable expansion
sum_m h_m(s1) h_m(s2) is exact to machine precision in ~14 terms. So the resummed analytic window
matrix here is GROUND TRUTH -- no Laguerre fit, none of the ~20% offset that stopped the analytic arm
adjudicating on the octant window. ellmax convergence can be measured against truth.

Two widths, because the survey window has a Gaussian-like CORE (half-max 284, matching sigma=200 to
2%) but a much heavier TAIL (measure-weighted sqrt(<s^2>) = 959, wanting sigma~450). Required ellmax
should scale with extent, so sigma = 200 and 350 give a lever arm. sigma=450 would be clipped by the
box (3 sigma = 1350 > half-box 1151).

argv: <sigma> <mock_budget_seconds>
"""
import os, sys, time, itertools
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import special
import isotropic_geom as IG
from jaxpower import (BinMesh3SpectrumPoles, BinMesh3CorrelationPoles, compute_mesh3_spectrum,
                      compute_mesh3_correlation, compute_smooth3_spectrum_window, compute_normalization,
                      get_smooth3_window_bin_attrs, generate_spectrum3_mesh, interpolate_window_function)
from lsstypes import ObservableLeaf, ObservableTree
from jaxpower.utils import wigner_3j, wigner_9j

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', '_tests')
SIGMA = float(sys.argv[1]) if len(sys.argv) > 1 else 200.
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 900.
ELLS = ELLSIN = (0, 2)
DK, KMAX, LOS, NOUTSUB = 0.015, 0.09, 'z', 8      # global LOS: the cleanest isotropic configuration
ELLMAXES = [4, 8, 16]
BANDS = (2, 3, 4)
P0, KREF = 2.0e4, 0.05
tag = f'iso{int(SIGMA)}'
print(f'=== isotropic Gaussian selection, sigma={SIGMA:.0f}, los={LOS} ===', flush=True)
mattrs = IG.get_mattrs()
print(f'  box {float(mattrs.boxsize[0]):.0f}, mesh {int(mattrs.meshsize[0])}, '
      f'3 sigma = {3*SIGMA:.0f} vs half-box {float(mattrs.boxsize[0])/2:.0f}', flush=True)
pk_np = lambda k: P0 / (1. + (np.asarray(k) / KREF)**2)
pk_jax = lambda kv: P0 / (1. + (jnp.sqrt(sum(k**2 for k in kv)) / KREF)**2)

t0 = time.time()
parts, Ws, norm3 = IG.selection(SIGMA, nreal=3)
print(f'  selection painted (3 realizations) in {time.time()-t0:.0f}s, norm3={float(norm3):.4e}', flush=True)
bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': DK, 'max': KMAX}, basis='scoccimarro', ells=ELLS)
xa = np.asarray(bin3.xavg)
valid = (xa[:, 2] >= np.abs(xa[:, 0] - xa[:, 1])) & (xa[:, 2] <= xa[:, 0] + xa[:, 1])
edges1d = np.arange(0., 2. * KMAX + DK, DK)
inj_edges = np.arange(0., 2. * KMAX + DK, DK)
sg = float(jnp.std(generate_spectrum3_mesh(mattrs, power=pk_jax, seed=0).value))
AMP = 0.02 * pk_np(np.array([inj_edges[b:b+2].mean() for b in BANDS])).prod()**(2./3.)
print(f'  {int(valid.sum())} valid output triangles; injected bands {BANDS}, amp {AMP:.3e}', flush=True)

# ---------- measured window Q (stitched), for the pipeline ----------
t0 = time.time()
kw = get_smooth3_window_bin_attrs(ELLS, ellsin=ELLSIN, basis='scoccimarro', ellmax=max(ELLMAXES))
kw['ells'] = [q for q in kw['ells'] if max(np.atleast_1d(q).ravel()) <= 0]
kw.setdefault('mask_edges', '')
import survey_window_geom as SG
Q, _ = SG.stitched_window3(parts, norm3, kw, LOS, mattrs=mattrs)
print(f'  window Q (stitched) in {time.time()-t0:.0f}s', flush=True)
# how well does the MEASURED Q match the exact analytic one?
pole = Q.get(ells=(0, 0, 0)); cds = list(pole.coords().values())
s1 = np.asarray(cds[0]); s2 = np.asarray(cds[1])
Vm = np.asarray(pole.value()).real
Va = IG.analytic_q000(SIGMA, s1, s2)
m = (s1 < 6. * SIGMA)
d1 = np.gradient(s1)
mw = (d1 * s1**2)[:, None] * (d1 * s2**2)[None, :]
rel = np.sqrt((mw[np.ix_(m, m)] * (Vm - Va)[np.ix_(m, m)]**2).sum()
              / (mw[np.ix_(m, m)] * Va[np.ix_(m, m)]**2).sum())
print(f'  measured Q000 vs EXACT analytic: measure-weighted rel diff {rel:.3e} '
      f'(peak {Vm[0,0]:.4f} vs {Va[0,0]:.4f})', flush=True)

# ---------- mocks ----------
_c3 = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])
def measure(mesh):
    s3 = _c3(*[mesh * w.value for w in Ws], bin=bin3, los=LOS).map(lambda p: p.clone(norm=norm3))
    return np.stack([np.asarray(s3.get(ells=e).value()).real for e in ELLS])
box, t0, i = [], time.time(), 0
while time.time() - t0 < BUDGET:
    v = [measure(generate_spectrum3_mesh(mattrs, power=pk_jax, edges=inj_edges,
                                         spectrum3={BANDS: s * AMP}, seed=i)) for s in (1, -1)]
    box.append((v[0] - v[1]) / 2.); i += 1
    if i == 1: print(f'  first pair (incl. compile) {time.time()-t0:.0f}s', flush=True)
box = np.array(box); NM = len(box); bm, be = box.mean(0), box.std(0) / np.sqrt(NM)
print(f'  {NM} mock pairs in {time.time()-t0:.0f}s', flush=True)

# ---------- pipeline window matrices ----------
preds, rss = {}, {}
for em in ELLMAXES:
    t0 = time.time()
    wm = compute_smooth3_spectrum_window(Q, edgesin=edges1d, ellsin=ELLSIN, bin=bin3, flags=('fftlog',),
                                         batch_size=4, ellmax=em, noutsub=NOUTSUB)
    M = np.asarray(wm.value()).real
    nin = M.shape[1] // len(ELLSIN)
    if em == ELLMAXES[0]:
        eth = np.asarray(next(iter(wm.theory)).edges('k'))
        want = np.array([[inj_edges[b], inj_edges[b + 1]] for b in BANDS])
        cols = sorted({int(h) for pm in itertools.permutations(range(3))
                       for h in np.where(np.all(np.abs(eth - want[list(pm)][None]) < 1e-9, axis=(1, 2)))[0]})
        thv = np.zeros((len(ELLSIN), nin)); thv[0, cols] = AMP
    preds[em] = (M @ thv.reshape(-1)).reshape(len(ELLS), len(xa))
    rss[em] = M.reshape(len(ELLS), len(xa), len(ELLSIN), nin)[0].sum(axis=(1, 2))
    print(f'  pipeline ellmax={em:2d} ({time.time()-t0:.0f}s): row sums {rss[em][valid].mean():.4f}', flush=True)

# ---------- EXACT resummed analytic matrix (h_m exact, no fit) ----------
t0 = time.time()
NM_H, NQ, LMAX = 14, 4, 24
H = lambda a, b, c: wigner_3j(a, b, c, 0, 0, 0)
def Ccoef(L, p, q):
    if abs(H(p, q, L)) < 1e-12: return 0.
    N = (2*p+1) * (2*q+1) * (2*L+1)
    v = ((4*np.pi)**2 / np.sqrt(4*np.pi*(2*L+1))) * np.sqrt(4*np.pi/(2*L+1))
    v *= N * N * H(p, q, L) * wigner_9j(0, 0, 0, p, q, L, p, q, L) * H(p, q, L) * H(p, p, 0) * H(q, q, 0) * H(L, L, 0)
    return float(v)
costh = lambda a, b, c: (c**2 - a**2 - b**2) / (2.*a*b)
out = xa[valid]
boxes = sorted(set(itertools.permutations(BANDS)))
edg = np.array([[[inj_edges[b], inj_edges[b+1]] for b in pm] for pm in boxes])
u, wu = np.polynomial.legendre.leggauss(NQ)
nodes, wts, mu_t = [], [], []
for e in edg:
    lo, hi = e[:, 0], e[:, 1]
    g = [0.5*(hi[j]-lo[j])*u + 0.5*(hi[j]+lo[j]) for j in range(3)]
    wq = [0.5*(hi[j]-lo[j])*wu for j in range(3)]
    P = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
    W3 = np.einsum('i,j,k->ijk', *wq).ravel()
    cc = costh(P[:,0], P[:,1], P[:,2]); ok = np.abs(cc) <= 1.
    I0 = np.where(ok, np.pi**2 / (P[:,0]*P[:,1]*P[:,2]), 0.)
    nodes.append(P); wts.append(W3 * np.prod(P**2, axis=1) / (2.*np.pi**2)**3 * I0)
    mu_t.append(np.clip(cc, -1., 1.))
nodes, wts, mu_t = np.array(nodes), np.array(wts), np.array(mu_t)
nbx, nn = nodes.shape[:2]
kv = np.unique(np.round(np.concatenate([out[:,0], out[:,1]]), 9))
ik1 = np.searchsorted(kv, np.round(out[:,0], 9)); ik2 = np.searchsorted(kv, np.round(out[:,1], 9))
rq = np.geomspace(1e-2, max(12.*SIGMA, 3000.), 12000); wr = np.gradient(rq)
HM = np.array([IG.hm(mm, rq, SIGMA) for mm in range(NM_H)])
p1, p2 = nodes[..., 0].ravel(), nodes[..., 1].ravel()
muo = np.clip(costh(out[:,0], out[:,1], out[:,2]), -1., 1.)
ana = {}
for LM in (8, 16, LMAX):
    pred = np.zeros(len(out))
    for p in range(LM + 1):
        c = Ccoef(0, p, p)
        if abs(c) < 1e-12: continue
        A = special.spherical_jn(p, np.outer(kv, rq))
        C_ = (HM[:, None, :] * (wr * rq**2 * A)[None, :, :]).reshape(NM_H*len(kv), -1)
        G1 = (C_ @ special.spherical_jn(p, np.outer(p1, rq)).T).reshape(NM_H, len(kv), nbx, nn)
        G2 = (C_ @ special.spherical_jn(p, np.outer(p2, rq)).T).reshape(NM_H, len(kv), nbx, nn)
        rad = np.einsum('mobn,mobn->obn', G1[:, ik1], G2[:, ik2])
        pred += c * special.eval_legendre(p, muo) * np.sum(wts[None] * special.eval_legendre(p, mu_t)[None] * rad, axis=(1, 2))
    ana[LM] = pred * AMP
    print(f'  EXACT analytic LMAX={LM:2d}: sum|pred| = {np.abs(ana[LM]).sum():.4e}', flush=True)
print(f'  analytic total {time.time()-t0:.0f}s', flush=True)

np.savez(os.path.join(OUT, f'{tag}_run.npz'), xavg=xa, valid=valid, meas=bm, err=be, sigma=SIGMA,
         nmocks=NM, amp=AMP, ellmaxes=np.array(ELLMAXES), lmaxes=np.array(sorted(ana)),
         **{f'pred_{e}': preds[e] for e in preds}, **{f'rs_{e}': rss[e] for e in rss},
         **{f'ana_{L}': ana[L] for L in ana}, qrel=rel)
iv = np.where(valid)[0]
sig = np.abs(bm[0][iv]) > 5. * be[0][iv]
print(f'\n  {int(sig.sum())} of {len(iv)} triangles detected >5sigma')
print('  ---- median measured/predicted ----')
for em in ELLMAXES:
    print(f'    pipeline ellmax={em:2d}: {np.median((bm[0][iv]/preds[em][0][iv])[sig]):.4f}')
for L in sorted(ana):
    print(f'    EXACT analytic LMAX={L:2d}: {np.median((bm[0][iv]/ana[L])[sig]):.4f}')
print('  ---- pipeline / EXACT analytic (LMAX=24) ----')
for em in ELLMAXES:
    print(f'    ellmax={em:2d}: {np.median((preds[em][0][iv]/ana[LMAX])[sig]):.4f}')
print(f'\nsaved {tag}_run.npz', flush=True)
