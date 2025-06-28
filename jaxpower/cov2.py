import itertools

import numpy as np
import jax
from jax import numpy as jnp

from .mesh import MeshAttrs, RealMeshField
from .mesh2 import BinMesh2Spectrum, BinMesh2Correlation, compute_mesh2, FKPField
from .utils import CovarianceMatrix, legendre_product, BinnedStatistic
from .utils import get_spherical_jn


class Correlation2Spectrum(object):

    def __init__(self, k, ells):
        from .fftlog import PowerToCorrelation
        fftlog = PowerToCorrelation(k, ell=ells[0], lowring=False, minfolds=False)
        self._H = jax.jacfwd(lambda fun: fftlog(fun, extrap=False, ignore_prepostfactor=True)[1])(jnp.zeros_like(k))
        self._fftlog = PowerToCorrelation(k, ell=ells[1], lowring=False, minfolds=False)
        dlnk = jnp.diff(jnp.log(k)).mean()
        self._postfactor = 2 * np.pi**2 / dlnk / (k[..., None] * k)**1.5
        self.k = k
        self.s = fftlog.y

    def __call__(self, fun):
        fun = self._H * fun
        _, fun = self._fftlog(fun, extrap=False, ignore_prepostfactor=True)
        return self.k, self._postfactor * fun


def compute_fkp2_covariance_window(fkps, bin=None, los='local', **kwargs):
    if not isinstance(fkps, (tuple, list)): fkps = [fkps]
    fkps = fkps[0].same_mesh(*fkps)
    WW, WS, SS = {}, {}, {}
    if bin is None:
        mattrs = fkps[0].attrs
        kw = {'edges': None, 'ells': tuple(range(0, 9, 2))}
        for name in kw: kw[name] = kwargs.pop(name, kw[name])
        edges = kw.pop('edges', None) or {}
        if isinstance(edges, dict):
            edges.setdefault('max', np.sqrt(np.sum((mattrs.boxsize / 2)**2)))
        bin = BinMesh2Correlation(mattrs, edges=edges, **kw)

    def get_alpha(fkp):
        return fkp.data.sum() / fkp.randoms.sum() if isinstance(fkp, FKPField) else 1.

    def get_W(fkp):
        alpha = get_alpha(fkp)
        randoms = fkp.randoms if isinstance(fkp, FKPField) else fkp
        mesh = randoms.paint(**kwargs, out='real')
        return alpha * mesh / mesh.cellsize.prod()

    def get_S(fkp):
        alpha = get_alpha(fkp)
        randoms = fkp.randoms if isinstance(fkp, FKPField) else fkp
        mesh = randoms.clone(weights=randoms.weights**2).paint(**kwargs, out='real')
        return alpha * mesh / mesh.cellsize.prod()

    Ws = [get_W(fkp) for fkp in fkps]
    Ss = [get_S(fkp) for fkp in fkps]
    i2pts = tuple(itertools.combinations_with_replacement(tuple(range(len(Ws))), 2))  # pairs of fields for each P(k)
    for iW in itertools.combinations_with_replacement(i2pts, 2):  # then choose two pairs of fields
        iW = iW[0] + iW[1]
        if iW not in WW:
            W = [Ws[iW[0]] * Ws[iW[1]]]
            if iW[2:] != iW[:2]:
                W.append(Ws[iW[2]] * Ws[iW[3]])
            mattrs = W[0].attrs
            norm = W[0].sum() * W[-1].sum() * mattrs.cellsize.prod()**2  # sum(cellsize * W^2) *  sum(cellsize * W^2)
            # compute_mesh2 computes ~ sum(cellsize * W^2 * W^2) / cellsize^2, so correct norm:
            norm = norm / mattrs.cellsize.prod()**2
            update = dict(norm=norm, attrs={'mattrs': dict(mattrs)})
            WW[iW] = compute_mesh2(*W, bin=bin, los=los).clone(**update)
            if iW[3] == iW[2]:
                WS[iW] = compute_mesh2(W[0], Ss[iW[2]], bin=bin, los=los).clone(**update)
            if iW[1] == iW[0]:
                WS[iW[2:] + iW[:2]] = compute_mesh2(W[-1], Ss[iW[0]], bin=bin, los=los).clone(**update)
            if iW[1] == iW[0] and iW[3] == iW[2]:
                S = [Ss[iW[0]]]
                if iW[2] != iW[0]:
                    S.append(Ss[iW[2]])
                SS[iW] = compute_mesh2(*S, bin=bin, los=los).clone(**update)
    return WW, WS, SS


def compute_mesh2_covariance_window(meshs, bin=None, los='local', **kwargs):
    if not isinstance(meshs, (tuple, list)): meshs = [meshs]

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    meshs = [_2r(mesh) for mesh in meshs]
    WW = {}
    if bin is None:
        mattrs = meshs[0].attrs
        kw = {'edges': None, 'ells': tuple(range(0, 9, 2))}
        for name in kw: kw[name] = kwargs.pop(name, kw[name])
        edges = kw.pop('edges', None) or {}
        if isinstance(edges, dict):
            edges.setdefault('max', np.sqrt(np.sum((mattrs.boxsize / 2)**2)))
        bin = BinMesh2Correlation(mattrs, edges=edges, **kw)

    def get_W(mesh):
        return mesh / mesh.cellsize.prod()

    Ws = [get_W(mesh) for mesh in meshs]
    del meshs
    i2pts = tuple(itertools.combinations_with_replacement(tuple(range(len(Ws))), 2))  # pairs of fields for each P(k)
    for iW in itertools.combinations_with_replacement(i2pts, 2):  # then choose two pairs of fields
        iW = iW[0] + iW[1]
        if iW not in WW:
            W = [Ws[iW[0]] * Ws[iW[1]]]
            if iW[2:] != iW[:2]:
                W.append(Ws[iW[2]] * Ws[iW[3]])
            mattrs = W[0].attrs
            norm = W[0].sum() * W[-1].sum() * mattrs.cellsize.prod()**2  # sum(cellsize * W^2) *  sum(cellsize * W^2)
            # compute_mesh2 computes ~ sum(cellsize * W^2 * W^2) / cellsize^2, so correct norm:
            norm = norm / mattrs.cellsize.prod()**2
            update = dict(norm=norm, attrs={'mattrs': dict(mattrs)})
            WW[iW] = compute_mesh2(*W, bin=bin, los=los).clone(**update)
    return WW


def compute_spectrum2_covariance(window2, poles, delta=None, flags=('smooth',)):
    # TODO: check for multiple fields

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

    if isinstance(window2, MeshAttrs):
        mattrs = window2
        if not isinstance(poles, dict):
            poles = {(0, 0): poles}
        keys = []
        for key in poles:
            if key not in keys:
                keys.append(key)

        cov = {}
        for key in itertools.combinations_with_replacement(keys, 2):
            key = key[0] + key[1]
            pole1, pole2 = poles[key[:2]], poles[key[2:]]
            assert pole1.projs == pole2.projs
            ills = list(range(len(pole1.projs)))

            def init():
                return [[np.zeros((len(pole1.x(pole1.projs[ill])),) * 2) for ill in ills] for ill in ills]

            cov[key] = init()

            from scipy import special
            if 'smooth' not in flags:
                bin = BinMesh2Spectrum(mattrs, edges=pole1.edges()[0], ells=(0,))
                kvec = mattrs.kcoords(sparse=True)
                vlos = (0, 0, 1)  # doesn't matter
                mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)

            for ill1, ill2 in itertools.product(ills, ills):
                ell1, ell2 = pole1.projs[ill1], pole2.projs[ill2]
                for p1, p2 in itertools.product(pole1.projs, pole2.projs):
                    p12 = 1. / mattrs.boxsize.prod() * 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * pole1.view(projs=p1) * pole2.view(projs=p2)
                    if 'smooth' in flags:
                        coeff = sum((2 * q + 1) * legendre_product(ell1, p1, q) * legendre_product(ell2, p2, q) for q in range(abs(ell1 - p1), ell1 + p1 + 1))
                        vol = 4. / 3. * np.pi * np.diff(pole1.edges(projs=ell1)**3, axis=-1)[..., 0]
                    else:
                        poly = 1.
                        for ell in [ell1, ell2, p1, p2]: poly *= special.legendre(ell)
                        coeff = bin(poly(mu))
                        vol = bin.nmodes * mattrs.kfun.prod()
                    cov[key][ill1][ill2] += np.diag((2. * np.pi)**3 / vol * coeff * p12)

        return finalize(cov)

    else:
        assert 'smooth' in flags, 'only "smooth" approximation is implemented'
        has_shotnoise = len(window2) > 1
        if has_shotnoise:
            WW, WS, SS = window2
        else:
            WW = window2
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

        def get_wj(ww, k, q1, q2):
            shape = k.shape * 2
            ww = ww.clone(num_zero=None)
            w = sum(legendre_product(q1, q2, q) * ww.view(projs=q).real if q in ww.projs else jnp.zeros(()) for q in list(range(abs(q1 - q2), q1 + q2 + 1)))
            if w.size <= 1:
                return np.zeros(shape)
            s = ww.x()[0]

            if 'fftlog' in flags:
                fftlog = Correlation2Spectrum(np.logspace(-6, 2, num=3645, endpoint=False), (q1, q2))
                from scipy.interpolate import UnivariateSpline, RectBivariateSpline
                if s[0] > 0: s, w = np.insert(s, 0, 0.), np.insert(w, 0, w[0])
                w = UnivariateSpline(s, w, k=1, s=0, ext='zeros')(fftlog.s)
                _, tmp = fftlog(w)
                toret = RectBivariateSpline(fftlog.k, fftlog.k, tmp, kx=1, ky=1)(k, k, grid=True)
                return toret

            else:
                k1, k2 = np.meshgrid(k, k, indexing='ij', sparse=False)
                k1, k2 = k1.ravel(), k2.ravel()
                kmask = Ellipsis
                if delta is not None:
                    kmask = np.abs(k2 - k1) <= delta
                    k1, k2 = k1[kmask], k2[kmask]

                vol = ww.weights()[0] * np.prod(ww.attrs['mattrs']['boxsize'] / ww.attrs['mattrs']['meshsize'])
                #vol = 4. / 3. * np.pi * np.diff(ww.edges(projs=q1)**3, axis=-1)[..., 0]
                w, s = np.where(vol == 0, 1, w), np.where(vol == 0, 0, s)
                w *= vol

                def f(k12):
                    k1, k2 = k12
                    return jnp.sum(w * get_spherical_jn(q1)(k1 * s) * get_spherical_jn(q2)(k2 * s))

                import jax
                batch_size = int(min(max(1e7 / (k1.size * s.size), 1), k1.size))
                tmp = jax.lax.map(f, (k1, k2), batch_size=batch_size)
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

            cov_WW[key] = init()
            if has_shotnoise: cov_WS[key], cov_SS[key] = init(), init()
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
                        if (q1, q2) in cache_WW:
                            tmp = cache_WW[q1, q2]
                        else:
                            tmp = (-1)**((q1 - q2) // 2) * (2 * q1 + 1) * (2 * q2 + 1) * get_wj(WW[key], pole1.x(projs=p1), q1, q2)
                            cache_WW[q1, q2] = tmp
                        cov_WW[key][ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * coeff1 * tmp * pole1.view(projs=p1)[..., None] * pole2.view(projs=p2)
                if not has_shotnoise: continue
                # WS
                for p1 in pole1.projs:
                    q1 = list(range(abs(ell1 - p1), ell1 + p1 + 1))
                    for q1 in q1:
                        q2 = q1
                        coeff1 = legendre_product(ell1, p1, q1)
                        if coeff1 == 0.: continue
                        if (q1, ell2) in cache_WS1:
                            tmp = cache_WS1[q1, ell2]
                        else:
                            tmp = (-1)**((q1 - ell2) // 2) * (2 * q1 + 1) * get_wj(WS[key], pole1.x(projs=ell1), q1, ell2)
                            cache_WS1[q1, ell2] = tmp
                        cov_WS[key][ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * coeff1 * tmp * pole1.view(projs=p1)
                # WS swap
                for p2 in pole2.projs:
                    q2 = list(range(abs(ell2 - p2), ell2 + p2 + 1))
                    for q2 in q2:
                        q1 = q2
                        coeff1 = legendre_product(ell2, p2, q2)
                        if coeff1 == 0.: continue
                        if (q2, ell1) in cache_WS2:
                            tmp = cache_WS2[q2, ell1]
                        else:
                            tmp = (-1)**((q2 - ell1) // 2) * (2 * q2 + 1) * get_wj(WS[key], pole2.x(projs=ell2), q2, ell1)
                            cache_WS2[q2, ell1] = tmp
                        cov_WS[key][ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * coeff1 * tmp * pole2.view(projs=p2)
                # SS
                cov_SS[key][ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * (-1)**((ell1 - ell2) // 2) * get_wj(SS[key], pole1.x(projs=ell1), ell1, ell2)

        if has_shotnoise:
            covs = tuple(map(finalize, (cov_WW, cov_WS, cov_SS)))
        else:
            covs = finalize(cov_WW)
        return covs
