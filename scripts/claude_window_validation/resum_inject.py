"""Fully-resummed analytic window matrix applied to the INJECTED (exactly known) bispectrum.

The theory is a delta at the 6 distinct permutations of the target band box, so the prediction is
just those 6 columns of the window matrix times the amplitude. Here the matrix is evaluated from
the closed-form Gaussian kernel

    F_ll(k,k') = int r^2 dr e^{-r^2/2sigma^2} j_l(kr) j_l(k'r) = sqrt(pi/2) sigma^3 e^{-sigma^2(k^2+k'^2)/2} i_l(sigma^2 k k')

expanded in the (exact, 14-term) Gaussian series, with the ell sum carried to LMAX rather than
truncated at ellmax=2 as the pipeline does. Convergence in LMAX is printed, which is what makes it
"fully resummed": no ellmax truncation and no FFTlog noise floor.
"""
import sys, time
import numpy as np
from scipy import special
sys.path.insert(0, '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/jax-power')
sys.argv = ['x']
from jaxpower.utils import wigner_3j, wigner_9j

T = '/local/home/adematti/Bureau/DESI/NERSC/cosmodesi/jax-power/tests/_tests/'
d = np.load(T + 'win_inject_2-3-4.npz')
xa, oi, amp, TARGET, kedges = d['xavg'], d['oidx'], float(d['amp']), tuple(d['target']), d['kedges']
SIG, NM, NQ = 50., 14, 6
ALPHA = 1. / (3. * SIG**2)
H = lambda a, b, c: wigner_3j(a, b, c, 0, 0, 0)
def Ccoef(L, p, q):
    if abs(H(p, q, L)) < 1e-12: return 0.
    N = (2*p+1) * (2*q+1) * (2*L+1)
    v = ((4*np.pi)**2 / np.sqrt(4*np.pi*(2*L+1))) * np.sqrt(4*np.pi/(2*L+1))
    v *= N * N * H(p, q, L) * wigner_9j(0, 0, 0, p, q, L, p, q, L) * H(p, q, L) * H(p, p, 0) * H(q, q, 0) * H(L, L, 0)
    return float(v)
hfun = lambda m, r: np.exp(m*np.log(ALPHA) - 0.5*special.gammaln(2*m+2) + 2*m*np.log(np.maximum(r, 1e-30)) - ALPHA*r**2)
costh = lambda a, b, c: (c**2 - a**2 - b**2) / (2.*a*b)

# the 6 distinct permuted boxes of the target band triplet
import itertools
boxes = sorted(set(itertools.permutations(TARGET)))
edg = np.array([[[kedges[b], kedges[b+1]] for b in pm] for pm in boxes])     # (nbox, 3, 2)
cen = edg.mean(-1)
print(f'target bands {TARGET} -> {len(boxes)} distinct permuted theory boxes; amp={amp:.3e}', flush=True)

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
nb, nn = nodes.shape[:2]
out = xa[oi]
kv = np.unique(np.round(np.concatenate([out[:,0], out[:,1]]), 9))
ik1 = np.searchsorted(kv, np.round(out[:,0], 9)); ik2 = np.searchsorted(kv, np.round(out[:,1], 9))
rq = np.geomspace(1e-2, 900., 3000); wr = np.gradient(rq)
HM = np.array([hfun(m, rq) for m in range(NM)])
p1, p2 = nodes[..., 0].ravel(), nodes[..., 1].ravel()
muo = np.clip(costh(out[:,0], out[:,1], out[:,2]), -1., 1.)
print(f'{len(out)} outputs, {len(kv)} distinct output k, {nb} boxes x {nn} nodes', flush=True)

for LMAX in (4, 8, 16, 24):
    t0 = time.time()
    pred = np.zeros(len(out))
    for p in range(LMAX + 1):
        c = Ccoef(0, p, p)
        if abs(c) < 1e-12: continue
        A = special.spherical_jn(p, np.outer(kv, rq))
        C_ = (HM[:, None, :] * (wr * rq**2 * A)[None, :, :]).reshape(NM*len(kv), -1)
        G1 = (C_ @ special.spherical_jn(p, np.outer(p1, rq)).T).reshape(NM, len(kv), nb, nn)
        G2 = (C_ @ special.spherical_jn(p, np.outer(p2, rq)).T).reshape(NM, len(kv), nb, nn)
        Lp_t = special.eval_legendre(p, mu_t)
        rad = np.einsum('mobn,mobn->obn', G1[:, ik1], G2[:, ik2])       # (nout, nbox, nnode)
        pred += c * special.eval_legendre(p, muo) * np.sum(wts[None] * Lp_t[None] * rad, axis=(1, 2))
    pred = pred * amp
    print(f'  LMAX={LMAX:2d}: {time.time()-t0:5.1f}s  sum |pred|/amp = {np.sum(np.abs(pred))/amp:.6f}', flush=True)
    np.save(T + f'resum_inject_pred_LMAX{LMAX}.npy', pred)
np.savez(T + 'resum_inject.npz', oidx=oi, xavg=out, pred=pred, amp=amp, boxes=np.array(boxes), sigma=SIG, lmax=LMAX)
print('  saved -> _tests/resum_inject.npz', flush=True)