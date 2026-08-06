r"""Analytic window matrix WITH both sub-binnings, so it is like-for-like with the pipeline.

Two asymmetries made the earlier comparison unfair, and neither is inherent to the analytic form:
  - THEORY side (ninsub): the analytic used NQ=4 Gauss-Legendre nodes per theory-bin axis against the
    pipeline's ninsub=16. Raised here, and scanned.
  - OUTPUT side (noutsub): the analytic evaluated at the bin representative (k1i, k2i, mu_i) with NO
    bin average at all, against the pipeline's noutsub=8. ms.tex flags this as up to 35% on squeezed
    triangles, and within the pipeline noutsub=1 vs 8 was measured to move squeezed bins by ~1.7x.
    Implemented here: the output triangle is sub-binned on the same k1^2 k2^2 k3^2 * Theta measure
    the pipeline uses, so mu_i and the radial kernels are averaged over the bin rather than sampled.
argv: sigma [NOUT] [NQ]
"""
import os, sys, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import special
import isotropic_geom as IG
from jaxpower.utils import wigner_3j, wigner_9j
from jaxpower import BinMesh3SpectrumPoles
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', '_tests')
SIGMA = float(sys.argv[1]) if len(sys.argv) > 1 else 200.
NOUTS = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1', '2', '4'])]
NQ = int(sys.argv[3]) if len(sys.argv) > 3 else 6
DK, KMAX, NM_H, LMAX = 0.015, 0.09, 14, 24
BANDS = (2, 3, 4)
H = lambda a, b, c: wigner_3j(a, b, c, 0, 0, 0)
def C0(p):
    N = (2*p+1)**2 * 1
    v = ((4*np.pi)**2 / np.sqrt(4*np.pi)) * np.sqrt(4*np.pi)
    return float(v * N * N * H(p, p, 0)**2 * wigner_9j(0,0,0,p,p,0,p,p,0) * H(p,p,0)**2 * H(0,0,0))
costh = lambda a, b, c: (c**2 - a**2 - b**2) / (2.*a*b)
z = np.load(os.path.join(OUT, f'iso{int(SIGMA)}_run.npz'))
bm, be, AMP = z['meas'], z['err'], float(z['amp'])
mattrs = IG.get_mattrs()
bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': DK, 'max': KMAX}, basis='scoccimarro', ells=(0, 2))
xa = np.asarray(bin3.xavg); oe = np.asarray(bin3.edges)
valid = (xa[:,2] >= np.abs(xa[:,0]-xa[:,1])) & (xa[:,2] <= xa[:,0]+xa[:,1])
iv = np.where(valid)[0]; sig = np.abs(bm[0][iv]) > 5.*be[0][iv]
inj = np.arange(0., 2.*KMAX + DK, DK)
boxes = sorted(set(itertools.permutations(BANDS)))
edg = np.array([[[inj[b], inj[b+1]] for b in pm] for pm in boxes])
rq = np.geomspace(1e-2, max(12.*SIGMA, 3000.), 12000); wr = np.gradient(rq)
HM = np.array([IG.hm(m, rq, SIGMA) for m in range(NM_H)])
# theory-side nodes (ninsub equivalent)
u, wu = np.polynomial.legendre.leggauss(NQ)
tn, tw, tmu = [], [], []
for e in edg:
    lo, hi = e[:,0], e[:,1]
    g = [0.5*(hi[j]-lo[j])*u + 0.5*(hi[j]+lo[j]) for j in range(3)]
    wq = [0.5*(hi[j]-lo[j])*wu for j in range(3)]
    P = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
    W3 = np.einsum('i,j,k->ijk', *wq).ravel()
    cc = costh(P[:,0], P[:,1], P[:,2]); ok = np.abs(cc) <= 1.
    I0 = np.where(ok, np.pi**2/(P[:,0]*P[:,1]*P[:,2]), 0.)
    tn.append(P); tw.append(W3*np.prod(P**2, axis=1)/(2.*np.pi**2)**3*I0); tmu.append(np.clip(cc,-1.,1.))
tn, tw, tmu = np.array(tn), np.array(tw), np.array(tmu)
nbx, nn = tn.shape[:2]
print(f'sigma={SIGMA:.0f}, NQ(theory)={NQ}; {int(sig.sum())} of {len(iv)} bins >5sigma\n', flush=True)
print('  NOUT | output sub-nodes |  time | median meas/pred | worst 4 bins')
for NOUT in NOUTS:
    t0 = time.time()
    uo, wo = np.polynomial.legendre.leggauss(NOUT)
    O, OW, OMU = [], [], []
    for i in iv:
        lo, hi = oe[i][:, 0], oe[i][:, 1]
        if NOUT == 1:                      # bin representative: what the analytic did before
            P = xa[i][None, :]; W = np.array([1.]); cc = np.array([costh(*xa[i])])
        else:
            g = [0.5*(hi[j]-lo[j])*uo + 0.5*(hi[j]+lo[j]) for j in range(3)]
            wq = [0.5*(hi[j]-lo[j])*wo for j in range(3)]
            P = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
            W3 = np.einsum('i,j,k->ijk', *wq).ravel()
            cc = costh(P[:,0], P[:,1], P[:,2]); ok = np.abs(cc) <= 1.
            W = W3 * np.prod(P**2, axis=1) * ok          # k1^2k2^2k3^2 * Theta, as the pipeline
        O.append(P); OW.append(W); OMU.append(np.clip(cc, -1., 1.))
    O, OW, OMU = np.array(O), np.array(OW), np.array(OMU)
    kv = np.unique(np.round(np.concatenate([O[..., 0].ravel(), O[..., 1].ravel()]), 9))
    j1 = np.searchsorted(kv, np.round(O[..., 0], 9)); j2 = np.searchsorted(kv, np.round(O[..., 1], 9))
    p1, p2 = tn[..., 0].ravel(), tn[..., 1].ravel()
    pred = np.zeros(len(iv))
    for p in range(LMAX + 1):
        c = C0(p)
        if abs(c) < 1e-12: continue
        A = special.spherical_jn(p, np.outer(kv, rq))
        C_ = (HM[:, None, :] * (wr*rq**2*A)[None, :, :]).reshape(NM_H*len(kv), -1)
        G1 = (C_ @ special.spherical_jn(p, np.outer(p1, rq)).T).reshape(NM_H, len(kv), nbx, nn)
        G2 = (C_ @ special.spherical_jn(p, np.outer(p2, rq)).T).reshape(NM_H, len(kv), nbx, nn)
        # theory sum, per OUTPUT SUB-NODE, then averaged over the output bin
        # chunk over output sub-nodes: the (nodes, NM_H, nbox, ntheory) tensor is 12.2 GiB at NOUT=16
        jr1, jr2 = j1.ravel(), j2.ravel()
        vflat = np.empty(len(jr1))
        CH = max(1, int(3e7 // max(G1.shape[0] * nbx * nn, 1)))
        tws = tw[None] * special.eval_legendre(p, tmu)[None]
        for st in range(0, len(jr1), CH):
            sl = slice(st, st + CH)
            rad = np.einsum('mobn,mobn->obn', G1[:, jr1[sl]], G2[:, jr2[sl]])
            vflat[sl] = np.sum(tws * rad, axis=(1, 2))
        val = vflat.reshape(O.shape[:2])
        pred += c * np.sum(OW * special.eval_legendre(p, OMU) * val, axis=1) / np.sum(OW, axis=1)
    pred *= AMP
    r = (bm[0][iv] / pred)[sig]
    w4 = np.sort(np.abs(r - 1.))[-4:]
    print(f'  {NOUT:4d} | {O.shape[1]:16d} | {time.time()-t0:4.0f}s | {np.median(r):16.4f} | '
          f'{", ".join(f"{1+x:.2f}" for x in w4)}', flush=True)
    np.save(os.path.join(OUT, f'iso{int(SIGMA)}_ana_nout{NOUT}.npy'), pred)
print('\ndone', flush=True)
