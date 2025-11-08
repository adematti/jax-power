import itertools

import numpy as np
import jax
from jax import numpy as jnp

from .mesh import MeshAttrs, RealMeshField
from .mesh2 import BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2, FKPField
from .types import CovarianceMatrix, Mesh2SpectrumPoles, ObservableTree
from .utils import legendre_product, get_spherical_jn


class Correlation2Spectrum(object):
    """
    Transform a function from s-space to 2D k-space using (2D) FFTLog.

    Parameters
    ----------
    k : array-like
        Wavenumber array.
    ells : tuple
        Tuple of two multipole orders (ell1, ell2).

    Attributes
    ----------
    s : array-like
        Separation array.
    """
    def __init__(self, k, ells, check_level=0):
        from .fftlog import SpectrumToCorrelation
        fftlog = SpectrumToCorrelation(k, ell=ells[0], lowring=False, check_level=check_level, minfolds=0).fftlog
        self._H = jax.jacfwd(lambda fun: fftlog(fun, extrap=False, ignore_prepostfactor=True)[1])(jnp.zeros_like(k))
        self._fftlog = SpectrumToCorrelation(k, ell=ells[1], lowring=False, minfolds=0).fftlog
        dlnk = jnp.diff(jnp.log(k)).mean()
        self._postfactor = 2 * np.pi**2 / dlnk / (k[..., None] * k)**1.5

    @property
    def k(self):
        return self._fftlog.x

    @property
    def s(self):
        return self._fftlog.y

    def __call__(self, fun):
        """
        Apply the transformation to a 1D function in s-space.

        Parameters
        ----------
        fun : array-like
            Function values in s-space.

        Returns
        -------
        k : array-like
            Wavenumber array.
        transformed : array-like, 2D
            Transformed function in k-space.
        """
        fun = self._H * fun
        _, fun = self._fftlog(fun, extrap=False, ignore_prepostfactor=True)
        return self.k, self._postfactor * fun


def compute_fkp2_covariance_window(fkps, bin=None, alpha=None, los='local', fields=None, **kwargs):
    """
    Compute the window matrices (WW, WS, SS) for the FKP 2-point covariance.

    Parameters
    ----------
    fkps : (list of) FKPField, ParticleField
        (List of) FKP or Particle fields.
    bin : BinMesh2CorrelationPoles, optional
        Binning operator.
    los : str, optional
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.
    alpha : float, optional
        Normalization factor for the randoms: sum(data_weights) / sum(randoms_weights).
        If ``None``, measured from input FKP particules.
    fields : list of str, optional
        List of field names (default: [0, 1, 2, ...]).
    kwargs : dict
        Additional arguments for mesh painting, see :meth:`ParticleField.paint`.

    Returns
    -------
    WW : dict
        Dictionary of window matrices for pairs of fields.
    WS : dict
        Dictionary of window matrices for field-shotnoise cross terms.
    SS : dict
        Dictionary of window matrices for shotnoise terms.
    """
    if not isinstance(fkps, (tuple, list)): fkps = [fkps]
    WW, WS, SS = {}, {}, {}
    if bin is None:
        mattrs = fkps[0].attrs
        kw = {'edges': None, 'ells': tuple(range(0, 9, 2)), 'basis': None, 'klimit': None, 'batch_size': None}
        for name in kw: kw[name] = kwargs.pop(name, kw[name])
        edges = kw.pop('edges', None)
        if edges is None: edges = {}
        if isinstance(edges, dict):
            edges.setdefault('max', np.sqrt(np.sum((mattrs.boxsize / 2)**2)))
        bin = BinMesh2CorrelationPoles(mattrs, edges=edges, **kw)

    def get_alpha(fkp):
        if alpha is None:
            return fkp.data.sum() / fkp.randoms.sum() if isinstance(fkp, FKPField) else 1.
        return alpha

    def get_W(fkp):
        randoms = fkp.randoms if isinstance(fkp, FKPField) else fkp
        mesh = randoms.paint(**kwargs, out='real')
        return get_alpha(fkp) * mesh / mesh.cellsize.prod()

    def get_S(fkp):
        randoms = fkp.randoms if isinstance(fkp, FKPField) else fkp
        mesh = get_alpha(fkp) * randoms.clone(weights=randoms.weights**2).paint(**kwargs, out='real')
        return mesh / mesh.cellsize.prod()

    Ws = [get_W(fkp) for fkp in fkps]  # for several fields
    Ss = [get_S(fkp) for fkp in fkps]
    ipairs = tuple(itertools.combinations_with_replacement(tuple(range(len(Ws))), 2))  # pairs of fields for each P(k)
    for iW in itertools.combinations_with_replacement(ipairs, 2):  # then choose two pairs of fields
        iW = iW[0] + iW[1]
        if iW not in WW:
            W = ws = [Ws[iW[0]] * Ws[iW[1]], Ws[iW[2]] * Ws[iW[3]]]
            mattrs = W[0].attrs
            # norm is sum(cellsize * W^2) * sum(cellsize * W^2)
            # compute_mesh2 computes ~ sum(cellsize * W^2) / cellsize^2, so correct norm by cellsize^2
            #norm = ws[0].sum() * ws[-1].sum() #* mattrs.cellsize.prod()**2 / mattrs.cellsize.prod()**2
            # We could have used the normalization from the power spectrum estimation,
            # but this is anyway degenerate with the approximation we're making that W * W ~ W^2
            update = dict(norm=[ws[0].sum() * ws[1].sum() * jnp.ones_like(bin.xavg)] * len(bin.ells))
            WW[iW] = compute_mesh2(*ws, bin=bin, los=los).clone(**update)
            if iW[3] == iW[2]:
                ws = [W[0], Ss[iW[2]]]
                WS[iW] = compute_mesh2(*ws, bin=bin, los=los).clone(**update)
            if iW[1] == iW[0]:
                ws = [W[1], Ss[iW[0]]]
                WS[iW[2:] + iW[:2]] = compute_mesh2(*ws, bin=bin, los=los).clone(**update)
            if iW[1] == iW[0] and iW[3] == iW[2]:
                ws = [Ss[iW[0]], Ss[iW[2]]]
                SS[iW] = compute_mesh2(*ws, bin=bin, los=los).clone(**update)
    if fields is None: fields = list(range(len(fkps)))
    tuple_fields = [tuple(fields[ii] for ii in iW) for iW in WW]
    WW, WS, SS = [ObservableTree(list(_.values()), fields=tuple_fields) for _ in (WW, WS, SS)]
    return WW, WS, SS


def compute_mesh2_covariance_window(meshes, bin=None, los='local', fields=None, **kwargs):
    """
    Compute the window matrices (WW) for the 2-point covariance from mesh fields.

    Parameters
    ----------
    meshes : (list of) RealMeshField
        (List of) mesh fields.
    bin : BinMesh2CorrelationPoles, optional
        Binning operator.
    los : str, optional
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.
    kwargs : dict
        Additional arguments for mesh painting, see :meth:`ParticleField.paint`.

    Returns
    -------
    WW : dict
        Dictionary of window matrices for pairs of fields.
    """
    if not isinstance(meshes, (tuple, list)): meshes = [meshes]

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    meshes = [_2r(mesh) for mesh in meshes]
    WW = {}
    if bin is None:
        mattrs = meshes[0].attrs
        kw = {'edges': None, 'ells': tuple(range(0, 9, 2)), 'basis': None, 'klimit': None, 'batch_size': None}
        for name in kw: kw[name] = kwargs.pop(name, kw[name])
        edges = kw.pop('edges', None)
        if edges is None: edges = {}
        if isinstance(edges, dict):
            edges.setdefault('max', np.sqrt(np.sum((mattrs.boxsize / 2)**2)))
        bin = BinMesh2CorrelationPoles(mattrs, edges=edges, **kw)

    def get_W(mesh):
        return mesh / mesh.cellsize.prod()

    Ws = [get_W(mesh) for mesh in meshes]
    ipairs = tuple(itertools.combinations_with_replacement(tuple(range(len(Ws))), 2))  # pairs of fields for each P(k)
    for iW in itertools.combinations_with_replacement(ipairs, 2):  # then choose two pairs of fields
        iW = iW[0] + iW[1]
        if iW not in WW:
            W = [Ws[iW[0]] * Ws[iW[1]]]
            if iW[2:] != iW[:2]:
                W.append(Ws[iW[2]] * Ws[iW[3]])
            mattrs = W[0].attrs
            #norm = W[0].sum() * W[-1].sum() * mattrs.cellsize.prod()**2  # sum(cellsize * W^2) *  sum(cellsize * W^2)
            # compute_mesh2 computes ~ sum(cellsize * W^2 * W^2) / cellsize^2, so correct norm:
            #norm = norm / mattrs.cellsize.prod()**2
            WW[iW] = compute_mesh2(*W, bin=bin, los=los).clone(norm=[W[0].sum() * W[1].sum() * jnp.ones_like(bin.xavg)] * len(bin.ells))
    if fields is None: fields = list(range(len(meshes)))
    tuple_fields = [tuple(fields[ii] for ii in iW) for iW in WW]
    WW = ObservableTree(list(WW.values()), fields=tuple_fields)
    return WW


def compute_spectrum2_covariance(window2, poles, delta=None, flags=('smooth',)):
    """
    Compute the covariance matrix for the 2-point power spectrum, given window matrices and poles.

    Parameters
    ----------
    window2 : Obser or MeshAttrs
        Window matrices (WW, WS, SS).
        A :class:`MeshAttrs` instance can be directly provided instead, in case the selection function is trivial (constant).
    poles : ObservableTree or Mesh2SpectrumPoles
        Dictionary of Mesh2SpectrumPoles objects for each observable, or a single :class:`Mesh2SpectrumPoles` instance.
    delta : float, optional
        Maximum :math:`|k_i - k_j|` (in :math:`k`-bin units) for which to compute the covariance.
    flags : tuple, optional
        Flags controlling the computation:
        - 'smooth' (default for non-trivial selection function): no binning effects
        - 'fftlog': same as 'smooth', but using 2D fftlog instead of naive spherical Bessel integration.

    Returns
    -------
    cov : CovarianceMatrix or tuple of CovarianceMatrix
        Covariance matrix (or tuple of matrices: PP, PS and SS if WS and SS are input).
    """
    # TODO: check for multiple fields
    single_field = False

    def finalize(cov):
        nfields = len(fields)
        value = [[None for i in range(nfields)] for i in range(nfields)]
        for iobs1, iobs2 in itertools.combinations_with_replacement(range(nfields), 2):
            field = fields[iobs1] + fields[iobs2]
            value[iobs1][iobs2] = np.block(cov[field])
            if iobs2 > iobs1:
                value[iobs2][iobs1] = value[iobs1][iobs2].T
        value = np.block(value)
        if single_field:
            observable = next(iter(poles))
        else:
            observable = ObservableTree([poles.get(field) for field in fields], fields=fields)
        return CovarianceMatrix(observable=observable, value=value)

    if isinstance(window2, MeshAttrs):
        mattrs = window2
        if isinstance(poles, Mesh2SpectrumPoles):
            single_field = True
            poles = ObservableTree([poles], fields=[(0, 0)])
        fields = []
        for field in poles.fields:
            if field not in fields:
                fields.append(field)

        cov = {}
        for field in itertools.combinations_with_replacement(fields, 2):
            field = field[0] + field[1]
            pole1, pole2 = poles.get(fields=field[:2]), poles.get(fields=field[2:])
            assert pole1.ells == pole2.ells
            ills = list(range(len(pole1.ells)))

            def init():
                return [[np.zeros((len(pole1.get(pole1.ells[ill]).coords('k')),) * 2) for ill in ills] for ill in ills]

            cov[field] = init()

            from scipy import special
            if 'smooth' not in flags:
                bin = BinMesh2SpectrumPoles(mattrs, edges=next(iter(pole1)).edges('k'), ells=(0,))
                kvec = mattrs.kcoords(sparse=True)
                vlos = (0, 0, 1)  # doesn't matter
                mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)

            for ill1, ill2 in itertools.product(ills, ills):
                ell1, ell2 = pole1.ells[ill1], pole2.ells[ill2]
                for p1, p2 in itertools.product(pole1.ells, pole2.ells):
                    p12 = 1. / mattrs.boxsize.prod() * 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * pole1.get(p1).value() * pole2.get(p2).value()
                    if 'smooth' in flags:
                        coeff = sum((2 * q + 1) * legendre_product(ell1, p1, q) * legendre_product(ell2, p2, q) for q in range(abs(ell1 - p1), ell1 + p1 + 1))
                        vol = 4. / 3. * np.pi * np.diff(pole1.get(ell1).edges('k')**3, axis=-1)[..., 0]
                    else:
                        poly = 1.
                        for ell in [ell1, ell2, p1, p2]: poly *= special.legendre(ell)
                        coeff = bin(poly(mu))
                        vol = bin.nmodes * mattrs.kfun.prod()
                    cov[field][ill1][ill2] += np.diag((2. * np.pi)**3 / vol * coeff * p12)

        return finalize(cov)

    else:
        assert 'smooth' in flags, 'only "smooth" approximation is implemented'
        has_shotnoise = not isinstance(window2, ObservableTree)
        if has_shotnoise:
            WW, WS, SS = window2
        else:
            WW = window2
        fields = []
        for field in WW.fields:
            field = field[:2]
            if field not in fields:
                fields.append(field)

        if isinstance(poles, Mesh2SpectrumPoles):
            single_field = True
            assert len(WW.fields) == 1, 'single poles can be provided if only one window (field)'
            field = WW.fields[0]
            poles = ObservableTree([poles], fields=[field[::2]])
        for pole in poles:
            assert isinstance(pole, Mesh2SpectrumPoles), 'input poles must be Mesh2SpectrumPoles'
        for field in fields:
            assert field in poles.fields, f'input pole {field} is required'

        def get_wj(ww, k, q1, q2):
            shape = k.shape * 2
            w = sum(legendre_product(q1, q2, q) * ww.get(q).value().real if q in ww.ells else jnp.zeros(()) for q in list(range(abs(q1 - q2), q1 + q2 + 1)))
            if w.size <= 1:
                return np.zeros(shape)
            tmpw = next(iter(ww))
            s = tmpw.coords('s')

            if 'fftlog' in flags:
                s = tmpw.coords('s')
                fftlog = Correlation2Spectrum(s, (q1, q2), check_level=1)
                from scipy.interpolate import RectBivariateSpline
                tmp = fftlog(w)[1]
                toret = RectBivariateSpline(fftlog.k, fftlog.k, tmp, kx=1, ky=1)(k, k, grid=True)
                return toret

            else:
                k1, k2 = np.meshgrid(k, k, indexing='ij', sparse=False)
                k1, k2 = k1.ravel(), k2.ravel()
                kmask = Ellipsis
                if delta is not None:
                    kmask = np.abs(k2 - k1) <= delta
                    k1, k2 = k1[kmask], k2[kmask]

                if 'volume' in tmpw.values():
                    vol = tmpw.values('volume')
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
        for field in WW.fields:
            pole1, pole2 = poles.get(field[:2]), poles.get(field[2:])
            assert pole1.ells == pole2.ells
            ills = list(range(len(pole1.ells)))

            def init():
                return [[np.zeros((len(pole1.get(pole1.ells[ill]).coords('k')),) * 2) for ill in ills] for ill in ills]

            cov_WW[field] = init()
            if has_shotnoise: cov_WS[field], cov_SS[field] = init(), init()
            cache_WW, cache_WS1, cache_WS2 = {}, {}, {}
            for ill1, ill2 in itertools.product(ills, ills):
                ell1, ell2 = pole1.ells[ill1], pole2.ells[ill2]
                for p1, p2 in itertools.product(pole1.ells, pole2.ells):
                    q1 = list(range(abs(ell1 - p1), ell1 + p1 + 1))
                    q2 = list(range(abs(ell2 - p2), ell2 + p2 + 1))
                    # WW
                    for q1, q2 in itertools.product(q1, q2):
                        coeff1 = legendre_product(ell1, p1, q1) * legendre_product(ell2, p2, q2)
                        if coeff1 == 0.: continue
                        if (q1, q2) in cache_WW:
                            tmp = cache_WW[q1, q2]
                        else:
                            tmp = (-1)**((q1 - q2) // 2) * (2 * q1 + 1) * (2 * q2 + 1) * get_wj(WW.get(field), pole1.get(p1).coords('k', center='mid_if_edges_and_nan'), q1, q2)
                            cache_WW[q1, q2] = tmp
                        cov_WW[field][ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * coeff1 * tmp * pole1.get(p1).value()[..., None] * pole2.get(p2).value()
                if not has_shotnoise: continue
                # WS
                for p1 in pole1.ells:
                    q1 = list(range(abs(ell1 - p1), ell1 + p1 + 1))
                    for q1 in q1:
                        q2 = q1
                        coeff1 = legendre_product(ell1, p1, q1)
                        if coeff1 == 0.: continue
                        if (q1, ell2) in cache_WS1:
                            tmp = cache_WS1[q1, ell2]
                        else:
                            tmp = (-1)**((q1 - ell2) // 2) * (2 * q1 + 1) * get_wj(WS.get(field), pole1.get(ell1).coords('k', center='mid_if_edges_and_nan'), q1, ell2)
                            cache_WS1[q1, ell2] = tmp
                        cov_WS[field][ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * coeff1 * tmp * pole1.get(p1).value()
                # WS swap
                for p2 in pole2.ells:
                    q2 = list(range(abs(ell2 - p2), ell2 + p2 + 1))
                    for q2 in q2:
                        q1 = q2
                        coeff1 = legendre_product(ell2, p2, q2)
                        if coeff1 == 0.: continue
                        if (q2, ell1) in cache_WS2:
                            tmp = cache_WS2[q2, ell1]
                        else:
                            tmp = (-1)**((q2 - ell1) // 2) * (2 * q2 + 1) * get_wj(WS.get(field), pole2.get(ell2).coords('k', center='mid_if_edges_and_nan'), q2, ell1)
                            cache_WS2[q2, ell1] = tmp
                        cov_WS[field][ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * coeff1 * tmp * pole2.get(p2).value()
                # SS
                cov_SS[field][ill1][ill2] += 2 * (2 * ell1 + 1) * (2 * ell2 + 1) * (-1)**((ell1 - ell2) // 2) * get_wj(SS.get(field), pole1.get(ell1).coords('k', center='mid_if_edges_and_nan'), ell1, ell2)

        if has_shotnoise:
            covs = tuple(map(finalize, (cov_WW, cov_WS, cov_SS)))
        else:
            covs = finalize(cov_WW)
        return covs
