import itertools

import numpy as np
import jax
from jax import numpy as jnp

from .mesh import MeshAttrs, RealMeshField, split_particles, compute_normalization
from .mesh2 import BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2, FKPField, prod
from .types import CovarianceMatrix, Mesh2SpectrumPoles, ObservableTree
from .utils import legendre_product, get_spherical_jn


class Correlation2Spectrum(object):
    """
    Transform a function from s-space to 2D k-space using (2D) FFTLog.

    Parameters
    ----------
    s : array-like
        Separation array.
    ells : tuple
        Tuple of two multipole orders (ell1, ell2).
    """
    def __init__(self, s, ells, check_level=0):
        from .fftlog import CorrelationToSpectrum
        fftlog = CorrelationToSpectrum(s, ell=ells[0], lowring=False, minfolds=False, check_level=check_level).fftlog
        self._H = jax.jacfwd(lambda fun: fftlog(fun, extrap=False, ignore_prepostfactor=True)[1])(jnp.zeros_like(s))
        self._fftlog = CorrelationToSpectrum(s, ell=ells[1], lowring=False, minfolds=False).fftlog
        k = self.k
        dlnk = jnp.diff(jnp.log(k)).mean()
        self._postfactor = 2 * np.pi**2 / dlnk / (k[..., None] * k)**1.5

    @property
    def k(self):
        return self._fftlog.y

    @property
    def s(self):
        return self._fftlog.x

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


def compute_fkp2_covariance_window(fkps, bin=None, los='local', fields=None, split=None, **kwargs):
    r"""
    Compute the window matrices (WW, WS, SS) for the FKP 2-point covariance.

    Reference
    ---------
    https://arxiv.org/pdf/1811.05714
    Windows are already divided by :math:`\mathcal{W}_0^2`, such that the limit of WW(s->0) is the 1 / Veff
    with Veff the volume of the survey, and the limit of SS(s->0) is 1 / (Veff * density^2).

    Parameters
    ----------
    fkps : (list of) FKPField, ParticleField
        (List of) FKPField or ParticleField.
    bin : BinMesh2CorrelationPoles, optional
        Binning operator.
    los : str, optional
        If ``los`` is 'firstpoint' or 'local' (resp. 'endpoint'), use local (varying) first point (resp. end point) line-of-sight.
        Else, may be 'x', 'y' or 'z', for one of the Cartesian axes.
        Else, a 3-vector.
    fields : list of int or str, optional
        List of field identifiers (default: [0, 1, 2, ...]).
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
        bin = BinMesh2CorrelationPoles(mattrs, edges=edges, **kw)

    def get_randoms(fkp):
        return fkp.randoms if isinstance(fkp, FKPField) else fkp

    def get_W(fkp, mask):
        randoms = get_randoms(fkp)
        if mask is not None: randoms = randoms.clone(weights=randoms.weights * mask)
        alpha = fkp.data.weights.sum() / randoms.weights.sum() if isinstance(fkp, FKPField) else 1.
        mesh = randoms.paint(**kwargs, out='real')
        return alpha * mesh / mesh.cellsize.prod()

    def get_S(fkps, mask):
        randoms = [get_randoms(fkp) for fkp in fkps]
        if mask is not None:
            randoms = [randoms.clone(weights=randoms.weights * mask) for randoms in randoms]
        alpha = 1.
        if isinstance(fkps[0], FKPField):
            alpha *= prod(fkp.data.weights for fkp in fkps).sum() / prod(randoms.weights for randoms in randoms).sum()
        mesh = randoms[0].clone(weights=randoms[0].weights * randoms[1].weights).paint(**kwargs, out='real')
        return alpha * mesh / mesh.cellsize.prod()

    def split_masks(fkps, split, fields):
        unique_fields = []
        for field in fields:
            if field not in unique_fields:
                unique_fields.append(field)
        seed = [split[fields.index(ufield)] for ufield in unique_fields]
        return split_particles(*[get_randoms(fkp) for fkp in fkps], seed=seed, fields=fields, return_masks=True)

    if fields is None:
        fields = list(range(len(fkps)))
    if split is not None and not isinstance(split, list):
        assert len(split) == len(fields), 'provide as many seeds as fields'
        split = [split]
    fkps = {field: fkp for field, fkp in zip(fields, fkps, strict=True)}
    windows = {name: {} for name in ['WW', 'WS', 'SW', 'SS']}
    pairs = tuple(itertools.combinations_with_replacement(tuple(fields), 2))  # pairs of fields for each inner P(k)
    compute_mesh2_window = jax.jit(compute_mesh2, static_argnames=['los'])

    for pair in itertools.product(pairs, pairs):
        wfield = sum(pair, start=tuple())
        if wfield not in windows['WW']:
            _fkps = [fkps[field] for field in wfield]
            masks = [None] * len(_fkps)
            if split is not None:
                masks = split_masks(_fkps, split=split, fields=list(wfield))
            W0, W1 = [get_W(_fkps[0], mask=masks[0]) * get_W(_fkps[1], mask=masks[1]),
                      get_W(_fkps[2], mask=masks[2]) * get_W(_fkps[3], mask=masks[3])]
            # mattrs = W[0].attrs
            # norm is sum(cellsize * W^2) * sum(cellsize * W^2)
            # compute_mesh2 computes ~ sum(cellsize * W^2) / cellsize^2, so correct norm by cellsize^2
            #norm = ws[0].sum() * ws[-1].sum() #* mattrs.cellsize.prod()**2 / mattrs.cellsize.prod()**2
            # We could have used the normalization from the power spectrum estimation,
            # but this is anyway degenerate with the approximation we're making that W * W ~ W^2
            update = dict(norm=[W0.sum() * W1.sum() * jnp.ones_like(bin.xavg)] * len(bin.ells))
            windows['WW'][wfield] = compute_mesh2_window(W0, W1, bin=bin, los=los).clone(**update)
            if wfield[3] == wfield[2]:
                windows['WS'][wfield] = compute_mesh2_window(W0, get_S(_fkps[2:], masks[2]), bin=bin, los=los).clone(**update)
            del W0
            if wfield[1] == wfield[0]:
                windows['SW'][wfield] = compute_mesh2_window(get_S(_fkps[:2], masks[0]), W1, bin=bin, los=los).clone(**update)
            del W1
            if wfield[1] == wfield[0] and wfield[3] == wfield[2]:
                windows['SS'][wfield] = compute_mesh2_window(get_S(_fkps[:2], mask=masks[0]), get_S(_fkps[2:], mask=masks[2]),
                                                             bin=bin, los=los).clone(**update)
    return ObservableTree([ObservableTree(list(windows[name].values()), fields=[tuple(wfield) for wfield in windows[name]]) for name in windows], types=list(windows))


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

    if bin is None:
        mattrs = meshes[0].attrs
        kw = {'edges': None, 'ells': tuple(range(0, 9, 2)), 'basis': None, 'klimit': None, 'batch_size': None}
        for name in kw: kw[name] = kwargs.pop(name, kw[name])
        edges = kw.pop('edges', None)
        if edges is None: edges = {}
        if isinstance(edges, dict):
            edges.setdefault('max', np.sqrt(np.sum((mattrs.boxsize / 2)**2)))
        bin = BinMesh2CorrelationPoles(mattrs, edges=edges, **kw)

    def get_W(field):
        mesh = meshes[field]
        return mesh / mesh.cellsize.prod()

    meshes = [_2r(mesh) for mesh in meshes]
    if fields is None: fields = list(range(len(meshes)))
    meshes = {field: mesh for field, mesh in zip(fields, meshes, strict=True)}

    pairs = tuple(itertools.combinations_with_replacement(tuple(fields), 2))  # pairs of fields for each P(k)
    window = {}
    compute_mesh2_window = jax.jit(compute_mesh2, static_argnames=['los'])
    for pair in itertools.product(pairs, pairs):  # then choose two pairs of fields
        wfield = sum(pair, start=tuple())
        if wfield not in window:
            ws = [get_W(wfield[0]) * get_W(wfield[1])]
            if wfield[2:] != wfield[:2]:
                ws.append(get_W(wfield[2]) * get_W(wfield[3]))
            #mattrs = W[0].attrs
            #norm = W[0].sum() * W[-1].sum() * mattrs.cellsize.prod()**2  # sum(cellsize * W^2) *  sum(cellsize * W^2)
            # compute_mesh2 computes ~ sum(cellsize * W^2 * W^2) / cellsize^2, so correct norm:
            #norm = norm / mattrs.cellsize.prod()**2
            norm = [ws[0].sum() * ws[-1].sum() * jnp.ones_like(bin.xavg)] * len(bin.ells)
            window[wfield] = compute_mesh2_window(*ws, bin=bin, los=los).clone(norm=norm)
    return ObservableTree(list(window.values()), fields=[tuple(wfield) for wfield in window])


def compute_spectrum2_covariance(window2, poles, delta=None, return_type=None, flags=('smooth',)):
    r"""
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

    Notes
    -----
    Most accurate option: 'fftlog' in ``flags``, compute windows (:func:`compute_fkp2_covariance_window`) with arguments ``basis='bessel'``,
    and interpolate with :func:`interpolate_window_function`.
    Else, use fine :math:`s`-binning (smaller than the :math:`pi / k_\mathrm{max}`, where :math:`k_\mathrm{max}` is the maximum wavenumber of the covariance matrix).

    Reference
    ---------
    https://arxiv.org/pdf/1811.0571

    Returns
    -------
    cov : CovarianceMatrix
        Covariance matrix.
    """
    # TODO: check for multiple fields
    single_field = False

    def with_fields(poles):
        return 'fields' in poles.labels(return_type='keys')

    def finalize(cov):
        nfields = len(fields)
        value = [[None for i in range(nfields)] for i in range(nfields)]
        for ifield1, ifield2 in itertools.combinations_with_replacement(range(nfields), 2):
            field = fields[ifield1] + fields[ifield2]
            value[ifield1][ifield2] = np.block(cov[field])
            if ifield2 > ifield1:
                value[ifield2][ifield1] = value[ifield1][ifield2].T
        value = np.block(value)
        if single_field:
            observable = next(iter(poles))
        else:
            observable = ObservableTree([poles.get(field) for field in fields], fields=fields)
        return CovarianceMatrix(observable=observable, value=value)

    if isinstance(window2, MeshAttrs):
        mattrs = window2
        if not with_fields(poles):
            single_field = True
            poles = ObservableTree([poles], fields=[(0, 0)])
        fields = []
        for field1 in poles.fields:
            for field2 in poles.fields:
                field = (field1[0], field2[0])
                if field not in fields:
                    fields.append(field)

        cov = {}
        for field in itertools.combinations_with_replacement(fields, 2):
            field = sum(field, start=tuple())
            cross_fields = [(field[0], field[2]), (field[1], field[3])]
            cross_fields_swap = [(field[0], field[3]), (field[1], field[2])]
            pole1, pole2 = poles.get(cross_fields[0]), poles.get(cross_fields[1])
            assert pole1.ells == pole2.ells
            ells = pole1.ells
            ills = list(range(len(ells)))

            def init():
                return [[np.zeros((len(pole1.get(ell).coords('k')),) * 2) for ell in ells] for ell in ells]

            cov[field] = init()

            from scipy import special
            if 'smooth' not in flags:
                bin = BinMesh2SpectrumPoles(mattrs, edges=next(iter(pole1)).edges('k'), ells=(0,))
                kvec = mattrs.kcoords(sparse=True)
                vlos = (0, 0, 1)  # doesn't matter
                mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)

            for ill1, ill2 in itertools.product(ills, ills):
                ell1, ell2 = ells[ill1], ells[ill2]
                for p1, p2 in itertools.product(ells, ells):
                    pk12 = pole1.get(p1).value() * (-1)**(p2 % 2) * pole2.get(p2).value()
                    if field[1] != field[0] or field[3] != field[2]:
                        pk12 += poles.get(cross_fields_swap[0]).get(p1).value() * (-1)**(p2 % 2) * poles.get(cross_fields_swap[1]).get(p2).value()
                    else:
                        pk12 *= 2
                    pk12 *= 1. / mattrs.boxsize.prod() * (2 * ell1 + 1) * (2 * ell2 + 1)
                    if 'smooth' in flags:
                        coeff = sum((2 * q + 1) * legendre_product(ell1, p1, q) * legendre_product(ell2, p2, q) for q in range(abs(ell1 - p1), ell1 + p1 + 1))
                        vol = 4. / 3. * np.pi * np.diff(pole1.get(ell1).edges('k')**3, axis=-1)[..., 0]
                    else:
                        poly = 1.
                        for ell in [ell1, ell2, p1, p2]: poly *= special.legendre(ell)
                        coeff = bin(poly(mu))
                        vol = bin.nmodes * mattrs.kfun.prod()
                    cov[field][ill1][ill2] += np.diag((2. * np.pi)**3 / vol * coeff * pk12)

        return finalize(cov)

    else:
        assert 'smooth' in flags, 'only "smooth" approximation is implemented'
        has_shotnoise = 'types' in window2.labels(return_type='keys')
        if has_shotnoise:
            WW, WS, SW, SS = [window2.get(name) for name in ['WW', 'WS', 'SW', 'SS']]
        else:
            WW = window2

        fields = []
        for field in WW.fields:
            field = field[::2]
            if field not in fields:
                fields.append(field)

        if not with_fields(poles):
            single_field = True
            assert len(WW.fields) == 1, 'single poles can be provided if only one window (field)'
            field = WW.fields[0]
            poles = ObservableTree([poles], fields=[field[::2]])
        for field in fields:
            assert field in poles.fields, f'input pole {field} is required'

        cache_rebin = []
        def _edges_from_centers(x):
            """Return bin edges from bin centers."""
            x = np.asarray(x)
            if x.ndim != 1 or np.any(x[1:] <= x[:-1]):
                raise ValueError("x must be 1D and strictly increasing")
            edges = np.empty(len(x) + 1, dtype=x.dtype)
            edges[1:-1] = 0.5 * (x[:-1] + x[1:])
            edges[0] = np.maximum(x[0] - 0.5 * (x[1] - x[0]), 0)
            edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
            return edges

        def get_wj(ww, k1, k2, k1edges, k2edges, q1, q2):
            shape = k1.shape + k2.shape
            w = sum(legendre_product(q1, q2, q) *  ww.get(q).value().real if q in ww.ells else jnp.zeros(()) for q in list(range(abs(q1 - q2), q1 + q2 + 1)))
            if w.size <= 1:
                return np.zeros(shape)
            tmpw = next(iter(ww))
            s = tmpw.coords('s')

            def rebin2d(xedges, yedges, xp, yp, fp, cache=cache_rebin):
                """Rebin fp sampled on (xp, yp) to bins centered on (x, y)."""
                interp_order = 3
                Mx = matrix_rebin(xedges, xp, wt=xp**2, interp_order=interp_order, cache=cache)
                My = matrix_rebin(yedges, yp, wt=yp**2, interp_order=interp_order, cache=cache)
                return Mx @ fp @ My.T

            if 'fftlog' in flags:
                s = tmpw.coords('s')
                fftlog = Correlation2Spectrum(s, (q1, q2), check_level=1)
                tmp = fftlog(w)[1]
                #from scipy.interpolate import RectBivariateSpline
                #toret = RectBivariateSpline(fftlog.k, fftlog.k, tmp, kx=1, ky=1)(k, k, grid=True)
                #print(ell1, ell2, q1, q2, np.diag(tmp)[:4], np.diag(toret)[:4])
                toret = rebin2d(k1edges, k2edges, fftlog.k, fftlog.k, tmp)
                #toret = toret.at[(k <= 0.)[:, None] * (k <= 0.)[None, :]].set(0.)
                #toret[(k <= 0.)[:, None] * (k <= 0.)[None, :]] = 0.
                return toret

            else:
                k1, k2 = np.meshgrid(k1, k2, indexing='ij', sparse=False)
                k1, k2 = k1.ravel(), k2.ravel()
                kmask = Ellipsis
                if delta is not None:
                    kmask = np.abs(k2 - k1) <= delta
                    k1, k2 = k1[kmask], k2[kmask]

                if 'volume' in tmpw.values():
                    vol = tmpw.values('volume')
                    #vol = 4. / 3. * np.pi * np.diff(tmpw.edges('s')**3, axis=-1)[..., 0]
                    #print(vol / vol2)
                    w, s = np.where(vol == 0, 1, w), np.where(vol == 0, 0, s)
                    w *= vol

                def f(k12):
                    k1, k2 = k12
                    return jnp.sum(w * get_spherical_jn(q1)(k1 * s) * get_spherical_jn(q2)(k2 * s))

                batch_size = int(min(max(1e7 / (k1.size * s.size), 1), k1.size))
                tmp = jax.lax.map(f, (k1, k2), batch_size=batch_size)
                toret = np.zeros(shape)
                toret.flat[kmask] = tmp
                return toret

        def get_window_field(window, fields, test=False):
            def sort(field):
                return tuple(sorted(field[:2])) + tuple(sorted(field[2:]))
            wfields = [sort(field) for field in window.fields]
            fields = sort(fields)
            if fields in wfields:
                if test: return True
                return window.get(window.fields[wfields.index(fields)])
            if test: return False
            raise ValueError(f'{fields} not found in {window}')

        cov_WW, cov_WS, cov_SS = {}, {}, {}
        for field in itertools.combinations_with_replacement(fields, 2):
            field = sum(field, start=tuple())
            cross_fields = [(field[0], field[2]), (field[1], field[3])]
            cross_fields_swap = [(field[0], field[3]), (field[1], field[2])]
            pole1, pole2 = poles.get(cross_fields[0]), poles.get(cross_fields[1])
            assert pole1.ells == pole2.ells
            ells = pole1.ells
            ills = list(range(len(ells)))

            def init():
                return [[np.zeros((len(pole1.get(ell).coords('k')), len(pole2.get(ell).coords('k')))) for ell in ells] for ell in ells]

            cov_WW[field] = init()
            if has_shotnoise: cov_WS[field], cov_SS[field] = init(), init()

            # multifield follows https://arxiv.org/pdf/1811.05714, eq. A. 10
            if field[1] != field[0] or field[3] != field[2]:
                cross_fields = [(1, False, cross_fields), (1, True, cross_fields_swap)]
            else:
                cross_fields = [(2, False, cross_fields)]

            for factor, swap, cross_fields in cross_fields:
                window_fields = sum(cross_fields, start=tuple())
                window_fields_swap = sum(cross_fields[::-1], start=tuple())
                pole1, pole2 = poles.get(cross_fields[0]), poles.get(cross_fields[1])

                cache_WW, cache_WS1, cache_WS2 = {}, {}, {}
                center = 'mid_if_edges_and_nan'
                for ill1, ill2 in itertools.product(ills, ills):
                    ell1, ell2 = ells[ill1], ells[ill2]
                    k1 = pole1.get(ell1).coords('k', center=center)
                    k1edges = pole1.get(ell1).edges('k', default=_edges_from_centers(k1))
                    k2 = pole2.get(ell2).coords('k', center=center)
                    k2edges = pole1.get(ell1).edges('k', default=_edges_from_centers(k2))
                    for p1, p2 in itertools.product(ells, ells):
                        q1 = list(range(abs(ell1 - p1), ell1 + p1 + 1))
                        q2 = list(range(abs(ell2 - p2), ell2 + p2 + 1))
                        # WW
                        for q1, q2 in itertools.product(q1, q2):
                            coeff = legendre_product(ell1, p1, q1) * legendre_product(ell2, p2, q2)
                            if coeff == 0.: continue
                            coeff *= factor * (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * q1 + 1) * (2 * q2 + 1)
                            if (q1, q2) in cache_WW:
                                wj = cache_WW[q1, q2]
                            else:
                                wj = get_wj(get_window_field(WW, window_fields), k1, k2, k1edges, k2edges, q1, q2)
                                cache_WW[q1, q2] = wj
                            if swap:
                                pk12 = pole1.get(p1).value()[:, None] * pole2.get(p2).value() * wj
                                # k <=> k'
                                pk12 += (-1)**((p1 % 2) + (p2 % 2)) * pole2.get(p2).value()[:, None] * pole1.get(p1).value() * wj.T
                                coeff *= (-1)**((q1 + q2) // 2)
                            else:
                                pk12 = pole1.get(p1).value()[:, None] * (-1)**(p2 % 2) * pole2.get(p2).value() * wj
                                # k <=> k'
                                pk12 += (-1)**(p1 % 2) * pole2.get(p2).value()[:, None] * pole1.get(p1).value() * wj.T
                                coeff *= (-1)**((q1 - q2) // 2)
                            cov_WW[field][ill1][ill2] += coeff * pk12 / 2.
                    if not has_shotnoise: continue
                    if get_window_field(WS, window_fields, test=True):
                        # WS
                        for p1 in ells:
                            q1 = list(range(abs(ell1 - p1), ell1 + p1 + 1))
                            for q1 in q1:
                                q2 = q1
                                coeff = legendre_product(ell1, p1, q1)
                                if coeff == 0.: continue
                                coeff *= factor * (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * q1 + 1)
                                if (q1, ell2) in cache_WS1:
                                    wj = cache_WS1[q1, ell2]
                                else:
                                    wj1 = wj2 = get_wj(get_window_field(WS, window_fields), k1, k2, k1edges, k2edges, q1, ell2)
                                    if window_fields_swap != window_fields:
                                        # conjugate * in ref paper: actually SW
                                        wj2 = get_wj(get_window_field(SW, window_fields_swap), k1, k2, k1edges, k2edges, q1, ell2)
                                    wj = (wj1, wj2)
                                    cache_WS1[q1, ell2] = wj
                                if swap:
                                    pk1 = pole1.get(p1).value()[:, None]
                                    pk2 = (-1)**(p1 % 2) * pole2.get(p1).value()[:, None]
                                    coeff *= (-1)**((q1 + ell2) // 2)  #-k2 -> +k2 in window
                                else:
                                    # P^ac, P^bd in k
                                    pk1 = pole1.get(p1).value()[:, None]
                                    pk2 = (-1)**(p1 % 2) * pole2.get(p1).value()[:, None]
                                    coeff *= (-1)**((q1 - ell2) // 2)
                                cov_WS[field][ill1][ill2] += coeff * (wj[0] * pk1 + wj[1] * pk2) / 2.

                        # WS swap
                        for p2 in ells:
                            q2 = list(range(abs(ell2 - p2), ell2 + p2 + 1))
                            for q2 in q2:
                                coeff = legendre_product(ell2, p2, q2)
                                if coeff == 0.: continue
                                coeff *= factor * (2 * ell1 + 1) * (2 * ell2 + 1) * (2 * q2 + 1)
                                if (ell1, q2) in cache_WS2:
                                    wj = cache_WS2[ell1, q2]
                                else:
                                    wj1 = wj2 = get_wj(get_window_field(WS, window_fields), k1, k2, k1edges, k2edges, ell1, q2)
                                    if window_fields_swap != window_fields:
                                        # conjugate * in ref paper: actually SW
                                        wj2 = get_wj(get_window_field(SW, window_fields_swap), k1, k2, k1edges, k2edges, ell1, q2)
                                    wj = (wj1, wj2)
                                    cache_WS2[ell1, q2] = wj
                                if swap:
                                    pk1 = (-1)**(p2 % 2) * pole1.get(p2).value()[None, :]
                                    pk2 = pole2.get(p2).value()[None, :]
                                    coeff *= (-1)**((ell1 + q2) // 2)  #-k2 -> +k2 in window
                                else:
                                    # P^ac, P^bd in k'
                                    pk1 = pole1.get(p2).value()[None, :]
                                    pk2 = (-1)**(p2 % 2) * pole2.get(p2).value()[None, :]
                                    coeff *= (-1)**((ell1 - q2) // 2)
                                cov_WS[field][ill1][ill2] += coeff * (wj[0] * pk1 + wj[1] * pk2) / 2.
                    if get_window_field(SS, window_fields, test=True):
                        # SS
                        coeff = factor * (2 * ell1 + 1) * (2 * ell2 + 1)
                        if swap:
                            coeff *= (-1)**((ell1 + ell2) // 2)  #-k2 -> +k2 in window
                        else:
                            coeff *= (-1)**((ell1 - ell2) // 2)
                        cov_SS[field][ill1][ill2] += coeff * get_wj(get_window_field(SS, window_fields), k1, k2, k1edges, k2edges, ell1, ell2)

        if has_shotnoise:
            covs = tuple(map(finalize, (cov_WW, cov_WS, cov_SS)))
            if return_type != 'list':
                covs = covs[0].clone(value=sum(cov.value() for cov in covs))
            return covs
        else:
            covs = finalize(cov_WW)
        return covs


def matrix_spline_interp(xt, xo, deriv: int=0, interp_order: int=3):
    r"""
    Matrix for spline interpolation from samples on `xt` to evaluations at `xo`.

    Returns matrix A such that, for a 1D array y sampled on `xt`,
        y_interp = A @ y
    approximates y evaluated at `xo`.
    """
    from scipy.interpolate import make_interp_spline

    xt = np.asarray(xt)
    xo = np.asarray(xo)

    if xt.ndim != 1 or np.any(xt[1:] <= xt[:-1]):
        raise ValueError("xt must be 1D and strictly increasing")
    if xo.ndim != 1:
        raise ValueError("xo must be 1D")

    I = np.eye(len(xt), dtype=float)
    spl = make_interp_spline(xt, I, k=interp_order, axis=0)
    if deriv == -1:
        spl = spl.antiderivative()
    elif deriv == 1:
        spl = spl.derivative()
    elif deriv != 0:
        raise NotImplementedError(f"deriv={deriv} not implemented")

    return jnp.asarray(spl(xo))


def matrix_rebin(xedges, xt, wt=None, interp_order=3, cache=None):
    r"""
    Rebin samples on `xt` into bins `xedges`.

    Parameters
    ----------
    xedges : array_like
        Either:
          - 1D array of shape (nedges,), defining consecutive bins
          - 2D array of shape (nbins, 2), where each row is [xmin, xmax]
    xt : array_like, shape (nt,)
        Sample positions.
    wt : array_like, shape (nt,), optional
        Weights associated with samples on `xt`.
        If None, uses ones.
    interp_order : int, optional
        Spline order.
    cache : list, optional
        Cache list storing previously computed matrices.

    Returns
    -------
    M : jax array, shape (nbins, nt)
        Matrix such that for sampled values `ft`,
            rebinned = M @ ft
        gives the weighted bin average.
    """
    cache = [] if cache is None else cache

    xt = np.asarray(xt, dtype=float)
    xedges = np.asarray(xedges, dtype=float)

    if xt.ndim != 1 or np.any(xt[1:] <= xt[:-1]):
        raise ValueError("xt must be 1D and strictly increasing")

    if wt is None:
        wt = np.ones_like(xt, dtype=float)
    else:
        wt = np.asarray(wt, dtype=float)
        if wt.shape != xt.shape:
            raise ValueError("wt must have same shape as xt")

    # Normalize xedges to a (nbins, 2) array
    if xedges.ndim == 1:
        if len(xedges) < 2:
            raise ValueError("1D xedges must have at least 2 entries")
        bins = np.column_stack([xedges[:-1], xedges[1:]])
    elif xedges.ndim == 2 and xedges.shape[1] == 2:
        bins = xedges
    else:
        raise ValueError("xedges must be either 1D (nedges,) or 2D (nbins, 2)")

    if np.any(bins[:, 1] <= bins[:, 0]):
        raise ValueError("Each bin must satisfy xmax > xmin")

    for (_bins, _xt, _wt, M) in cache:
        if (
            _bins.shape == bins.shape
            and np.allclose(_bins, bins)
            and np.allclose(_xt, xt)
            and np.allclose(_wt, wt)
        ):
            return M

    # Antiderivative interpolation matrix evaluated at all bin endpoints
    endpoints = np.concatenate([bins[:, 0], bins[:, 1]])   # shape (2*nbins,)
    Aint = matrix_spline_interp(xt, endpoints, deriv=-1, interp_order=interp_order)
    nbins = bins.shape[0]

    # Bin integral operator: F(xmax) - F(xmin)
    B = Aint[nbins:, :] - Aint[:nbins, :]   # shape (nbins, len(xt))

    wt = np.asarray(wt)
    # Weighted average in each bin
    W = B @ wt                 # shape (nbins,)
    M = (B * wt[None, :]) / W[:, None]

    cache.append((bins.copy(), xt.copy(), wt.copy(), M))
    return M


def matrix_project_to_spectrum(edges, theory, interp_order=3):
    if edges.ndim < 2:
        edges = np.column_stack((edges[:-1], edges[1:]))
    matrices, cache = [], []
    for pole in theory.flatten(level=1):
        kin = pole.coords('k')
        #edgesin = pole.edges('k')
        w = matrix_rebin(edges, kin, wt=kin**2, interp_order=interp_order, cache=cache)
        matrices.append(w)
    from scipy.linalg import block_diag
    matrix = block_diag(*matrices)
    return matrix


from .utils import get_spherical_jn_tophat_integral


def matrix_project_to_correlation(edges, theory, window_deconvolution=None):
    """Return matrix to project spectrum covariance to correlation multipoles."""
    # Window_deconvolution: convolved correlation multipoles -> deconvolved correlation multipoles
    if edges.ndim < 2:
        edges = np.column_stack((edges[:-1], edges[1:]))
    matrices = []
    for label, pole in theory.items(level=1):
        k = pole.coords('k')
        kedges = pole.edges('k')
        ell = label['ells']
        norm = (-1)**((ell + 1) // 2) / (2 * np.pi)**3
        norm *= 4. / 3. * np.pi * np.diff(kedges**3, axis=-1).T  # (1, k.size)
        if False:
            s = np.mean(edges, axis=-1)
            w = norm * get_spherical_jn(ell)(k[None, :] * s[:, None])
        else:
            norm = norm / ((4. * np.pi) / 3. * np.diff(edges**3, axis=-1))  # (s.size, 1)
            w = norm * get_spherical_jn_tophat_integral(ell)(k, edges).T
        matrices.append(w)
    from scipy.linalg import block_diag
    matrix = block_diag(*matrices)
    if window_deconvolution is not None:
        matrix = window_deconvolution @ matrix
    return matrix