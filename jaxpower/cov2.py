import itertools

import numpy as np

from .mesh2 import BinMesh2Correlation, compute_mesh2, FKPField
from .utils import CovarianceMatrix, legendre_product, BinnedStatistic
from .utils import get_spherical_jn_scipy as get_spherical_jn


def compute_fkp2_window(fkps, bin=None, los='local', **kwargs):
    if not isinstance(fkps, (tuple, list)): fkps = [fkps]
    fkps = fkps[0].same_mesh(*fkps)
    WW, WS, SS = {}, {}, {}
    if bin is None:
        kw = {'edges': None, 'ells': tuple(range(0, 9, 2))}
        for name in kw: kw[name] = kwargs.pop(name, kw[name])
        bin = BinMesh2Correlation(fkps[0].attrs, **kw)

    def get_alpha(fkp):
        return fkp.data.sum() / fkp.randoms.sum() if isinstance(fkp, FKPField) else 1.0

    def get_W(fkp):
        alpha = get_alpha(fkp)
        randoms = fkp.randoms if isinstance(fkp, FKPField) else fkp
        return alpha * randoms.paint(**kwargs, out='complex')

    def get_S(fkp):
        alpha = get_alpha(fkp)
        randoms = fkp.randoms if isinstance(fkp, FKPField) else fkp
        return alpha * randoms.clone(weights=randoms.weights**2).paint(**kwargs, out='complex')

    Ws = [get_W(fkp) for fkp in fkps]
    Ss = [get_S(fkp) for fkp in fkps]
    i2pts = tuple(itertools.combinations_with_replacement(tuple(range(len(Ws))), 2))  # pairs of fields for each P(k)
    for iW in itertools.combinations_with_replacement(i2pts, 2):  # then choose two pairs of fields
        iW = sum(iW, start=tuple())
        if iW not in WW:
            W = [Ws[iW[0]] * Ws[iW[1]]]
            if iW[2:] != iW[:2]:
                W.append(Ws[iW[2]] * Ws[iW[3]])
            norm = (W[0] * W[-1]).sum() * W[0].cellsize.prod()
            WW[iW] = compute_mesh2(*W, bin=bin, los=los).clone(norm=norm)
            if iW[3] == iW[2]:
                WS[iW] = compute_mesh2(W[0], Ss[iW[2]], bin=bin, los=los).clone(norm=norm)
            if iW[1] == iW[0]:
                WS[iW[2:] + iW[:2]] = compute_mesh2(W[-1], Ss[iW[0]], bin=bin, los=los).clone(norm=norm)
            if iW[1] == iW[0] and iW[3] == iW[2]:
                S = [Ss[iW[0]]]
                if iW[2] != iW[0]:
                    S.append(Ss[iW[2]])
                SS[iW] = compute_mesh2(*S, bin=bin, los=los).clone(norm=norm)
    return WW, WS, SS


def compute_fkp2_spectrum_covariance(window2, poles, delta=None):
    # TODO: check for multiple fields
    WW, WS, SS = window2
    keys = []
    for key in WW:
        key = key[:2]
        if key not in keys:
            keys.append(key)

    if not isinstance(poles, dict):
        assert len(WW) == 1, 'non-dict poles can be provided if only one window (field)'
        for key in WW: break
        poles = {key[::2]: poles}
    for key, pole in poles.items():
        assert isinstance(pole, BinnedStatistic), 'input poles must be BinnedStatistic'
    for key in keys:
        assert key in poles, f'input pole {key} is required'

    def get_wj(ww, k, q1, q2, q):
        k1, k2 = np.meshgrid(k, k, indexing='ij', sparse=False)
        shape = k1.shape
        k1, k2 = k1.ravel(), k2.ravel()
        kmask = Ellipsis
        if delta is not None:
            kmask = np.abs(k2 - k1) <= delta
            k1, k2 = k1[kmask], k2[kmask]
        if q not in ww.projs:
            return 0.
        vol = ww.weights(q)
        s = ww.x(q)
        w = ww.view(projs=q).real
        w, s = np.where(vol == 0, 1, w), np.where(vol == 0, 1, s)
        nbatch = int(min(max((k1.size * s.size) / 1e7, 1), s.size))
        tmp = 0.
        for ibatch in range(nbatch):
            sl = slice(ibatch * s.size // nbatch, (ibatch + 1) * s.size // nbatch)
            tmp += np.sum(vol[sl] * w[sl] * get_spherical_jn(q1)(k1[..., None] * s[sl]) * get_spherical_jn(q2)(k2[..., None] * s[sl]), axis=-1)
        toret = np.zeros(shape)
        toret.flat[kmask] = tmp
        return toret

    cov_WW, cov_WS, cov_SS = {}, {}, {}
    for key in WW:
        pole1, pole2 = poles[key[:2]], poles[key[2:]]
        assert pole1.projs == pole2.projs
        ills = list(range(len(pole1.projs)))

        def init():
            return [[np.zeros((len(pole1.x(pole1.projs[ill])),) * 2) for ill in ills] for ill in ills]

        cov_WW[key], cov_WS[key], cov_SS[key] = init(), init(), init()
        cache_WW, cache_WS1, cache_WS2 = {}, {}, {}
        for ill1, ill2 in itertools.product(ills, ills):
            ell1, ell2 = pole1.projs[ill1], pole2.projs[ill2]
            for p1, p2 in itertools.product(pole1.projs, pole2.projs):
                q1 = list(range(abs(ell1 - p1), ell1 + p1 + 1))
                q2 = list(range(abs(ell2 - p2), ell2 + p2 + 1))
                # WW
                for q1, q2 in itertools.product(q1, q2):
                    coeff1 = legendre_product(ell1, p1, q1) * legendre_product(ell2, p2, q2)
                    if coeff1 == 0.: continue
                    q = list(range(abs(q1 - q2), q1 + q2 + 1))
                    if (q1, q2) in cache_WW:
                        tmp = cache_WW[q1, q2]
                    else:
                        tmp = 0.
                        for q in q:
                            coeff2 = (-1)**((q1 - q2) // 2) * (2 * q1 + 1) * (2 * q2 + 1) * legendre_product(q1, q2, q)
                            if coeff2 != 0: tmp += coeff2 * get_wj(WW[key], pole1.x(projs=ell1), q1, q2, q)
                        cache_WW[q1, q2] = tmp
                    cov_WW[key][ill1][ill2] += coeff1 * tmp * pole1.view(projs=ell1) * pole2.view(projs=ell2)[None, ...]
                # WS
                q1 = list(range(abs(ell1 - p1), ell1 + p1 + 1))
                for q1 in q1:
                    q2 = q1
                    coeff1 = legendre_product(ell1, p1, q1)
                    if coeff1 == 0.: continue
                    q = list(range(abs(q1 - q2), q1 + q2 + 1))
                    if (q1, ell2) in cache_WS1:
                        tmp = cache_WS1[q1, ell2]
                    else:
                        tmp = 0.
                        for q in q:
                            coeff2 = (-1)**((q1 - ell2) // 2) * (2 * q1 + 1) * legendre_product(q1, q2, q)
                            if coeff2 != 0: tmp += coeff2 * get_wj(WS[key], pole1.x(projs=ell1), q1, ell2, q)
                        cache_WS1[q1, ell2] = tmp
                    cov_WS[key][ill1][ill2] += coeff1 * tmp * pole1.view(projs=ell1)
                # WS swap
                q2 = list(range(abs(ell2 - p2), ell2 + p2 + 1))
                for q2 in q2:
                    q1 = q2
                    coeff1 = legendre_product(ell2, p2, q2)
                    if coeff1 == 0.: continue
                    q = list(range(abs(q1 - q2), q1 + q2 + 1))
                    if (q2, ell1) in cache_WS2:
                        tmp = cache_WS2[q2, ell1]
                    else:
                        tmp = 0.
                        for q in q:
                            coeff2 = (-1)**((q2 - ell1) // 2) * (2 * q2 + 1) * legendre_product(q2, q1, q)
                            if coeff2 != 0: tmp += coeff2 * get_wj(WS[key], pole2.x(projs=ell2), q2, ell1, q)
                        cache_WS2[q2, ell1] = tmp
                    cov_WS[key][ill1][ill2] += coeff1 * tmp * pole2.view(projs=ell1)
                # SS
                q = list(range(abs(ell1 - ell2), ell1 + ell2 + 1))
                for q in q:
                    coeff1 = (-1)**((ell1 - ell2) // 2) *legendre_product(ell1, ell2, q)
                    if coeff1 == 0.: continue
                    cov_SS[key][ill1][ill2] += coeff1 * get_wj(SS[key], pole2.x(projs=ell2), ell1, ell2, q)

    def finalize(cov):
        observables = [poles[key].clone() for key in keys]
        names = [observable.name for observable in observables]
        if len(set(names)) < len(keys):
            for iobs, observable in enumerate(observables):
                observables[iobs] = observable.clone(name=observable.name + f'{key[0]:d}_{key[1]:d}')  # set different names
        nobs = len(observables)
        value = [[None for i in range(nobs)] for i in range(nobs)]
        for iobs1, iobs2 in itertools.combinations_with_replacement(range(nobs), 2):
            key = keys[iobs1] + keys[iobs2]
            value[iobs1][iobs2] = np.block(cov[key])
            if iobs2 > iobs1:
                value[iobs2][iobs1] = value[iobs1][iobs2].T
        value = np.block(value)
        return CovarianceMatrix(observables=observables, value=value)

    covs = tuple(map(finalize, (cov_WW, cov_WS, cov_SS)))
    return covs
