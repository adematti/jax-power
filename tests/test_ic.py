import numpy as np
import jax
from jax import numpy as jnp

jax.config.update('jax_enable_x64', True)


def get_poles(k, values, ells=None):
    """lsstypes power spectrum multipoles ``values`` (nell, nk) at the points ``k`` (multipoles 0, 2, ... by default)."""
    from jaxpower.ic import Mesh2SpectrumPole, ObservableTree
    k, values = np.asarray(k, dtype='f8'), np.atleast_2d(np.asarray(values, dtype='f8'))
    if ells is None: ells = list(2 * np.arange(len(values)))
    poles = ObservableTree([Mesh2SpectrumPole(k=k, k_edges=np.column_stack([k - 1e-8, k + 1e-8]), num_raw=np.zeros(k.size)) for _ in ells], ells=list(ells))
    return poles.clone(value=values.ravel())


def shell_overlap(q, c, klo, khi):
    """Fraction of the shell [klo, khi] (measure k^2 dk) whose |k + q| stays in it, at the cosine c of q to k^ (exact for q <= klo)."""
    qperp2 = q**2 * (1. - c**2)
    g = lambda K: -q * c + np.sqrt(np.clip(K**2 - qperp2, 0., None))
    lo = np.where(klo**2 > qperp2, np.maximum(g(klo), klo), klo)
    hi = np.maximum(np.where(khi**2 > qperp2, np.minimum(g(khi), khi), -np.inf), lo)
    return (hi**3 - lo**3) / (khi**3 - klo**3)


def angular_map(ra, dec, weights=None, nside=256):
    """Weighted counts per HEALPix pixel (ring ordering) of points at ``ra``, ``dec`` (degrees)."""
    import healpy as hp
    pix = hp.ang2pix(nside, np.radians(90. - np.asarray(dec)), np.radians(np.asarray(ra)))
    return np.bincount(pix, weights=weights, minlength=hp.nside2npix(nside)).astype('f8')


def radial_selection(redges, r, weights=None):
    """g(r) such that g int r^2 dr (int r^2 dr per shell) are the weighted counts per shell (the last one includes its upper edge)."""
    from jaxpower import ic
    return np.histogram(r, bins=redges, weights=weights)[0] / (np.diff(np.asarray(redges) ** 3) / 3.)


def get_toy_geometry(nside=32, nr=16, seed=42, redges=None):
    """Two regions (two halves of a spherical cap), shells of 8 Mpc/h at r ~ 1000 Mpc/h (or ``redges``); footprints, data-weight maps (3 random subsets)."""
    from jaxpower import ic
    rng = np.random.RandomState(seed)
    if redges is None: redges = np.linspace(1000., 1000. + 8. * nr, nr + 1)
    footprint, mean, radial = {}, {}, {}
    for split in range(3):
        n = 200000
        ra, dec = rng.uniform(0., 60., n), np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(10.)), np.sin(np.radians(60.)), n)))
        w = 1. + 0.2 * np.cos(np.radians(3. * ra))
        r = rng.uniform(redges[0], redges[-1], n)
        for name, sel in [('A', ra < 30.), ('B', ra >= 30.)]:
            footprint.setdefault(name, []).append(angular_map(ra[sel], dec[sel], weights=w[sel], nside=nside))
            mean.setdefault(name, []).append(angular_map(ra[sel], dec[sel], nside=nside))
            if split == 0: radial[name] = radial_selection(redges, r[sel], w[sel])
    geometry = [ic.Template(np.array(footprint[name]), radial[name], redges) for name in radial]
    norm = sum(np.sum((np.diff(np.asarray(redges) ** 3) / 3.) * g**2) for g in radial.values()) / (4. * np.pi * 0.2)
    return redges, geometry, [np.array(mean[name]) for name in radial], norm


def get_toy_model(lmax=30, dense=False, gic=True):
    """A shuffle-like RIC (and the cap means) on the toy geometry."""
    from jaxpower import ic
    redges, geometry, mean, norm = get_toy_geometry()
    templates = ic.ric_templates(geometry, [ic.Template(m, reg.radial, redges) for m, reg in zip(mean, geometry)])  # each region shuffled on its own
    if dense:
        templates = [ic.Template(t.angular, t.radial, t.redges, mean=t.mean, response=('dense', np.diag(t.response[1])), group=t.group) for t in templates]
    return ic.IC2Model(geometry, templates, norm=norm, lmax=lmax, gic=gic)


def get_theory(q):
    pk = 1e4 * (q / 0.02) / (1. + (q / 0.02)**2.5) * np.exp(-(q / 0.15)**2)
    return np.array([pk, 0.5 * pk, 0.1 * pk])


def apply_window(out, term, edges, nells):
    """Prediction of the window for the theory of :func:`get_theory`, averaged in the bands."""
    q = np.linspace(edges[0], edges[-1], 30000)
    pells = get_theory(q)
    pband = np.array([[np.mean(p[(q >= lo) & (q < hi)]) for lo, hi in zip(edges[:-1], edges[1:])] for p in pells])
    return (np.asarray(out[term].value()) @ pband.ravel()).reshape(nells, -1)


def get_random_points(n=200000, seed=7):
    """Points uniform on the sphere, plus the poles and the boundaries of the equatorial belt; distances and weights."""
    rng = np.random.RandomState(seed)
    ra, dec = rng.uniform(0., 360., n), np.degrees(np.arcsin(rng.uniform(-1., 1., n)))
    zcap = np.degrees(np.arcsin(2. / 3.))
    ra = np.concatenate([ra, [0., 359.999999, 90., 180., 45.]])
    dec = np.concatenate([dec, [90., -90., zcap, -zcap, 0.]])
    r = rng.uniform(1000., 1128., ra.size)
    r[:2] = [1000., 1128.]  # the edges of the shells
    return ra, dec, r, rng.uniform(0.5, 1.5, ra.size)


def test_amr_templates():
    """For a density linear in the maps, delta = c0 + sum_k c_k x_k, the bin contrasts are exactly c0 + sum_k c_k <x_k>_{i nu}: the
    regression returns c, i.e. each template's mean window applied to delta gives c_j."""
    import healpy as hp
    from jaxpower import ic
    redges, geometry, _, _ = get_toy_geometry()
    region = geometry[0]
    npix = region.angular.shape[-1]
    theta, phi = hp.pix2ang(hp.npix2nside(npix), np.arange(npix))
    maps = np.array([(np.cos(theta) + 1.) / 2., (np.sin(4. * phi) + 1.) / 2., np.sin(theta)**2])  # in [0, 1]
    c = np.array([0.03, -0.05, 0.02])
    delta = 0.01 + c @ maps
    templates = ic.amr_templates(region, maps, np.ones(redges.size - 1), region.angular[:1])  # one random subset: exact
    measured = np.array([np.sum(np.asarray(t.mean[0][0])[0] * delta) * 4. * np.pi / npix for t in templates])
    print('regression of a density linear in the maps: c', c, 'recovered', measured)
    assert np.allclose(measured, c, rtol=1e-10, atol=1e-14)
    assert all(np.isclose(np.sum(t.response[2]), 1.) for t in templates)


def test_shell_widths():
    """Shells of unequal widths: an imaging-regression template fitted over all shells does not depend on how they are cut, so uniform and
    unequal shells (same range) give the same window, to the accuracy of the radial discretization."""
    import healpy as hp
    from jaxpower import ic
    k, edges, pred = np.linspace(0.005, 0.15, 10), np.linspace(0., 0.3, 61), {}
    widths = np.tile([5., 11.], 8)
    for name, redges in [('uniform', np.linspace(1000., 1128., 17)), ('unequal', 1000. + np.concatenate([[0.], np.cumsum(widths)]))]:
        redges, geometry, _, norm = get_toy_geometry(redges=redges)
        npix = geometry[0].angular.shape[-1]
        theta, phi = hp.pix2ang(hp.npix2nside(npix), np.arange(npix))
        maps = np.array([(np.cos(theta) + 1.) / 2., (np.sin(4. * phi) + 1.) / 2.])
        templates = sum((ic.amr_templates(reg, maps, np.ones(redges.size - 1), reg.angular) for reg in geometry), [])
        model = ic.IC2Model(geometry, templates, norm=norm, lmax=30)
        pred[name] = apply_window(ic.compute_smooth2_ic_window(model, edges, k, ells=(0, 2), terms=('total',)), 'total', edges, 2)
    rel = np.max(np.abs(pred['unequal'] - pred['uniform'])) / np.max(np.abs(pred['uniform']))
    print('uniform vs unequal shells: max |diff| / max', rel, 'prediction', pred['uniform'][0, :3])
    assert rel < 0.02


def test_radial_shells():
    """Templates carry their shells: a radial profile or response not on them, or templates of one model on different shells (even
    with the same number of shells) are reported."""
    from jaxpower import ic
    redges, geometry, mean, norm = get_toy_geometry()
    reg = geometry[0]
    for kwargs in [dict(), dict(mean=mean[0], response=('diag', np.ones(redges.size)))]:
        try: ic.Template(reg.angular, reg.radial[:-1] if not kwargs else reg.radial, redges, **kwargs)
        except ValueError as exc: print('mismatch reported:', exc)
        else: raise AssertionError('profile or response not on the shells, but no error')
    templates = ic.ric_templates(geometry, [ic.Template(m, reg.radial, redges) for m, reg in zip(mean, geometry)])
    shifted = [ic.Template(reg.angular, reg.radial, redges + 1.) for reg in geometry]
    for call in [lambda: ic.IC2Model(shifted, templates, norm=norm, lmax=30),
                 lambda: ic.ric_templates(geometry, [ic.Template(m, reg.radial, redges + 1.) for m, reg in zip(mean, geometry)])]:
        try: call()
        except ValueError as exc: print('mismatch reported:', exc)
        else: raise AssertionError('templates on different shells, but no error')
    model = ic.IC2Model(geometry, templates, norm=norm, lmax=30)
    assert np.array_equal(model.redges, redges)
    # a measured normalization, of a field alpha times larger (data weights wsum = alpha int W), is divided by alpha^2
    integral = sum(np.sum((np.diff(np.asarray(redges) ** 3) / 3.) * reg.radial) for reg in geometry)
    measured = ic.IC2Model(geometry, [], norm=9. * norm, wsum=3. * integral, lmax=30, gic=False)
    assert np.isclose(measured.norm, norm, rtol=1e-12, atol=0.)


def test_window_convergence():
    from jaxpower import ic
    model = get_toy_model()
    k = np.linspace(0.005, 0.15, 12)
    pred = {}
    for nband in [60, 300]:
        edges = np.linspace(0., 0.3, nband + 1)
        out = ic.compute_smooth2_ic_window(model, edges, k, ells=(0, 2), terms=('total', 'ft', 'tt'))
        pred[nband] = {term: apply_window(out, term, edges, 2) for term in ['total', 'ft', 'tt']}
        assert all(np.all(np.isfinite(out[term, 'shotnoise'])) for term in ['total', 'ft', 'tt'])
    for term in ['total', 'ft', 'tt']:
        rel = np.max(np.abs(pred[60][term] - pred[300][term])) / np.max(np.abs(pred[300][term]))
        print(term, 'coarse vs fine theory bands: max |diff| / max', rel, 'prediction', pred[300][term][0, :4])
        assert rel < 0.02


def test_response_kinds():
    from jaxpower import ic
    edges, k = np.linspace(0., 0.3, 31), np.linspace(0.005, 0.15, 8)
    outs = [ic.compute_smooth2_ic_window(get_toy_model(dense=dense), edges, k, ells=(0, 2), terms=('ft', 'tf', 'tt')) for dense in [False, True]]
    for term in ['ft', 'tf', 'tt']:
        for key in [term, (term, 'shotnoise')]:
            v0, v1 = (np.asarray(out[key].value()) if key == term else out[key] for out in outs)
            d = np.max(np.abs(v1 - v0)) / np.max(np.abs(v0))
            print(key, 'relative difference between diagonal and dense responses', d)
            assert d < 1e-8


def test_theory_multipoles():
    """Any theory multipoles ellsin, in any order: the window columns of l' = 0 and 2 with ellsin=(2, 0) are those with (0, 2, 4)."""
    from jaxpower import ic
    model = get_toy_model()
    edges, k = np.linspace(0., 0.2, 21), np.linspace(0.005, 0.15, 6)
    nband = edges.size - 1
    W = {}
    for ellsin in [(0, 2, 4), (2, 0)]:
        out = ic.compute_smooth2_ic_window(model, edges, k, ells=(0, 2), ellsin=ellsin, terms=('total',))
        W[ellsin] = np.asarray(out['total'].value()).reshape(2 * k.size, len(ellsin), nband)
    for ell in (0, 2):
        a, b = W[0, 2, 4][:, (0, 2, 4).index(ell)], W[2, 0][:, (2, 0).index(ell)]
        rel = np.max(np.abs(a - b)) / np.max(np.abs(a))
        print(f"theory multipole l' = {ell}: ellsin (2, 0) vs (0, 2, 4), max |diff| / max {rel:.1e}")
        assert rel < 1e-10


def test_no_gic():
    from jaxpower import ic
    model = get_toy_model(gic=False)
    assert set(model.removed) == {'t'}
    edges = np.linspace(0., 0.3, 31)
    out = ic.compute_smooth2_ic_window(model, edges, np.linspace(0.005, 0.15, 6), ells=(0, 2))
    pred = apply_window(out, 'total', edges, 2)
    print('without the global integral constraint: prediction', pred[0, :3])
    assert np.all(np.isfinite(pred))


def test_gic():
    # alpha is the last template: the observed field integrates to zero over the survey, so its monopole vanishes at k -> 0,
    # P^obs_0(0) = 0, i.e. Delta P_0(0) = -P^{W delta W delta}_0(0) whatever the other templates. Alone (ic=[]), its amplitude is
    # \bar{delta}_W and P^ft_0(0) = P^tf_0(0) = P^tt_0(0) = P^{W delta W delta}_0(0). Exact with one map per region and mean window (no
    # random subsets) and no pixel window
    from jaxpower import ic
    redges, geometry, mean, norm = get_toy_geometry()
    geometry = [ic.Template(np.mean(reg.angular, axis=0), reg.radial, redges) for reg in geometry]
    ric = ic.ric_templates(geometry, [ic.Template(np.mean(m, axis=0), reg.radial, redges) for m, reg in zip(mean, geometry)])
    kw = dict(norm=norm, lmax=30, deconvolve=False)
    alone = ic.IC2Model(geometry, [], **kw)
    assert set(alone.removed) == {'t'} and len(alone.removed['t']) == len(geometry)**2
    edges, k = np.linspace(0., 0.3, 61), np.array([1e-5, 0.01, 0.05])
    terms = ('total', 'ft', 'tf', 'tt')
    pred = {}
    for name, model in [('alone', alone), ('ric', ic.IC2Model(geometry, ric, **kw))]:
        out = ic.compute_smooth2_ic_window(model, edges, k, ells=(0, 2), terms=terms)
        pred[name] = {term: apply_window(out, term, edges, 2) for term in terms}
    a = pred['alone']
    for term in ['ft', 'tf']:
        rel = abs(a[term][0, 0] / a['tt'][0, 0] - 1.)
        print(f'alpha alone, k -> 0: P^{term}_0 / P^tt_0 - 1 = {rel:.1e}')
        assert rel < 1e-3
    assert a['total'][0, 0] < 0.  # the global integral constraint removes the largest scales
    rel = abs(pred['ric']['total'][0, 0] / a['total'][0, 0] - 1.)
    print(f'k -> 0: Delta P_0(ric + alpha) / Delta P_0(alpha) - 1 = {rel:.1e}; at k = 0.05: {pred["ric"]["total"][0, 2]:.3e} vs {a["total"][0, 2]:.3e}')
    assert rel < 1e-3
    # the same templates passed explicitly: the same model
    explicit = ic.IC2Model(geometry, ric + ic.gic_templates(geometry, ric), **{**kw, 'gic': False})
    out = ic.compute_smooth2_ic_window(explicit, edges, k, ells=(0, 2), terms=terms)
    for term in terms:
        d = np.max(np.abs(apply_window(out, term, edges, 2) - pred['ric'][term])) / np.max(np.abs(pred['ric'][term]))
        print(f'gic_templates passed explicitly, {term}: relative difference {d:.1e}')
        assert d < 1e-12


def test_harmonic():
    from jaxpower import ic
    redges, geometry, _, norm = get_toy_geometry()
    L = np.arange(3 * 32)
    kappa, uncorrelated_power = 0.4 / (1. + L / 5.), 1e-6 * np.ones(L.size)
    templates = []
    for reg in geometry:
        templates += ic.harmonic_templates(reg, np.ones(redges.size - 1), kappa, uncorrelated_power)
    model = ic.IC2Model(geometry, templates, norm=norm, lmax=30)
    k = np.linspace(0.005, 0.15, 12)
    pred = {}
    for nband in [60, 300]:
        edges = np.linspace(0., 0.3, nband + 1)
        out = ic.compute_smooth2_ic_window(model, edges, k, ells=(0, 2), terms=('total', 'tt'))
        pred[nband] = apply_window(out, 'total', edges, 2)
    rel = np.max(np.abs(pred[60] - pred[300])) / np.max(np.abs(pred[300]))
    print('harmonic: coarse vs fine theory bands: max |diff| / max', rel, 'additive', out['total', 'additive'][0, :3])
    assert rel < 0.02
    assert np.all(np.isfinite(out['total', 'additive'])) and np.all(np.isfinite(out['total', 'shotnoise']))


def get_toy_ft(kmax=1., c=0.1):
    """Window of P^ft such that the removal kernel is tau = c for q in the band [0.05, 0.1] (whatever mu, mu'), and k <= kmax."""
    from jaxpower.types import WindowMatrix, Mesh2SpectrumPole, Mesh2SpectrumPoles, ObservableTree
    k, ells, ellsin, edges = np.linspace(0., kmax, 11), (0, 2, 4, 6, 8), (0, 2, 4), np.array([[0., 0.05], [0.05, 0.1]])
    value = np.zeros((len(ells), k.size, len(ellsin), len(edges)))
    value[0, :, 0, 1] = -c
    observable = Mesh2SpectrumPoles([Mesh2SpectrumPole(k=k, k_edges=np.column_stack([k - 1e-8, k + 1e-8]), nmodes=np.ones_like(k), num_raw=np.zeros_like(k),
                                                       num_shotnoise=np.zeros_like(k), norm=np.ones_like(k), ell=ell) for ell in ells])
    theory = ObservableTree([Mesh2SpectrumPole(k=edges.mean(axis=-1), k_edges=edges, num_raw=np.zeros(len(edges))) for _ in ellsin], ells=list(ellsin))
    return WindowMatrix(observable=observable, theory=theory, value=value.reshape(len(ells) * k.size, -1))


def get_harmonic_toy():
    """Toy for the harmonic templates: RIC on the toy geometry (one map per region), a harmonic template on region 0 removing all of
    degree L0 = 2 with its normalization (harmonic_templates), and the same as rank-one templates of pattern Theta_R Y_L0M and mean window
    Y_L0M (real harmonics), normalized by -Theta_R [int Theta_R Y_L0M] g_R."""
    import healpy as hp
    from jaxpower import ic
    redges, geometry, mean, norm = get_toy_geometry()
    geometry = [ic.Template(np.mean(reg.angular, axis=0), reg.radial, redges) for reg in geometry]  # one map: all subsets are the same
    ric = ic.ric_templates(geometry, [ic.Template(np.mean(m, axis=0), reg.radial, redges) for m, reg in zip(mean, geometry)])
    reg = geometry[0]
    npix = reg.angular.size
    nside, L0, lmax = hp.npix2nside(npix), 2, 30
    radial_weights = reg.radial * (np.diff(np.asarray(redges) ** 3) / 3.)
    radial_weights = radial_weights / radial_weights.sum()
    kappa = np.zeros(3 * nside)
    kappa[L0] = 1.
    zbin = np.ones(redges.size - 1)
    harmonic = ic.harmonic_templates(reg, zbin, kappa)  # the template and the normalization of the weights to mean 1 over the region
    assert len(harmonic) == 2 and np.allclose(harmonic[0].response[1], radial_weights, rtol=1e-14, atol=0.)
    footprint = reg.angular / (np.sum(reg.angular) * 4. * np.pi / npix)  # normalized, as IC2Model does
    real_harmonics = []
    for M in range(L0 + 1):  # real orthonormal harmonics of degree L0: Y_L00, sqrt(2) Re Y_L0M, sqrt(2) Im Y_L0M
        for coeff in ([1.] if M == 0 else [1. / np.sqrt(2.), -1j / np.sqrt(2.)]):
            alm = np.zeros(hp.Alm.getsize(3 * nside - 1), dtype='c16')
            alm[hp.Alm.getidx(3 * nside - 1, L0, M)] = coeff
            real_harmonics.append(hp.alm2map(alm, nside, lmax=3 * nside - 1))
    print('real harmonics: int Y^2 =', [round(np.sum(y**2) * 4. * np.pi / npix, 4) for y in real_harmonics])
    # Theta_R Y_L0M is non-negative on this footprint for most M, and IC2Model would normalize it as a footprint: split it into
    # Theta_R (Y_L0M - c) (non-positive, used as given) + c Theta_R (the footprint, normalized, with c in the radial profile)
    c = 1. + np.max(np.abs(real_harmonics))
    response = ('outer', np.ones(redges.size - 1), radial_weights)
    rank_one = [ic.Template(footprint * (y - c), reg.radial, redges, mean=y, response=response, group=('Y', i, 0)) for i, y in enumerate(real_harmonics)]
    rank_one += [ic.Template(reg.angular, c * reg.radial, redges, mean=y, response=response, group=('Y', i, 1)) for i, y in enumerate(real_harmonics)]
    # the normalization of the weights over the region: -Theta_R [int Theta_R Y_L0M] g_R, of mean window Y_L0M
    pixarea = 4. * np.pi / npix
    rank_one += [ic.Template(reg.angular, -np.sum(footprint * y) * pixarea * reg.radial, redges, mean=y, response=response, group=('Yn', i))
                 for i, y in enumerate(real_harmonics)]
    return dict(redges=redges, geometry=geometry, norm=norm, ric=ric, reg=reg, footprint=footprint, lmax=lmax, harmonic=harmonic,
                real_harmonics=real_harmonics, response=response, rank_one=rank_one, zbin=zbin, kappa=kappa)


def test_harmonic_mixed():
    """Harmonic templates (harmonic_templates, with the normalization of the weights to mean 1 over the region) mixed with regular ones
    (RIC): a harmonic template removing all of one degree L0 of the projected density is the sum over M of the rank-one templates of pattern
    Theta_R Y_L0M and mean window Y_L0M (real harmonics), normalized by -Theta_R [int Theta_R Y_L0M] g_R. On smooth maps (one random subset),
    without pixel window, both mixed models agree, including the harmonic x regular cross terms; also with the global integral constraint,
    which skips the harmonic templates (zero survey mean), while the rank-one route sums survey means that cancel."""
    from jaxpower import ic
    toy = get_harmonic_toy()
    geometry, norm, ric, lmax, harmonic, rank_one = (toy[name] for name in ['geometry', 'norm', 'ric', 'lmax', 'harmonic', 'rank_one'])
    try:  # a harmonic template without its normalization is refused
        ic.IC2Model(geometry, ric + harmonic[:1], norm=norm, lmax=lmax)
    except ValueError as exc:
        print('harmonic template without its normalization:', exc)
    else:
        raise AssertionError('IC2Model accepted a harmonic template without its normalization')
    edges, k = np.linspace(0., 0.3, 61), np.linspace(0.005, 0.15, 10)
    terms = ('ft', 'tt', 'total')
    for gic in [False, True]:
        kw = dict(norm=norm, lmax=lmax, gic=gic, deconvolve=False)
        cases = [('harmonic', ric + harmonic), ('rank-one', ric + rank_one)]
        if not gic: cases += [('harmonic only', harmonic), ('ric only', ric)]
        outs = {name: ic.compute_smooth2_ic_window(ic.IC2Model(geometry, t, **kw), edges, k, ells=(0, 2), terms=terms) for name, t in cases}
        for term in terms:
            pred = {name: apply_window(out, term, edges, 2) for name, out in outs.items()}
            sn = {name: out[term, 'shotnoise'] for name, out in outs.items()}
            rel = np.max(np.abs(pred['harmonic'] - pred['rank-one'])) / np.max(np.abs(pred['rank-one']))
            rel_sn = np.max(np.abs(sn['harmonic'] - sn['rank-one'])) / np.max(np.abs(sn['rank-one']))
            msg = f'gic={gic}, {term}: harmonic vs rank-one, max |diff| / max {rel:.2e} (shot noise {rel_sn:.2e})'
            if not gic:  # the harmonic x regular cross terms: what the mixed model adds to the sum of the two separate ones
                cross = pred['harmonic'] - pred['harmonic only'] - pred['ric only']
                msg += f'; cross terms max |.| / max {np.max(np.abs(cross)) / np.max(np.abs(pred["rank-one"])):.2e}'
            print(msg)
            assert rel < 1e-4 and rel_sn < 1e-4  # exact up to the angular and radial quadratures


def test_sugiyama_norm():
    """<Phi_e^2> of the Sugiyama basis over independent directions: the analytic 1 / [(2 l1 + 1)(2 l2 + 1)(2 L + 1) H^2] against a fine quadrature."""
    import jax.numpy as jnp
    from jaxpower import ic
    from jaxpower.utils import get_S
    ells = [(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2), (2, 0, 2), (4, 0, 4)]
    mu, wmu = np.polynomial.legendre.leggauss(24)
    c, wc = np.polynomial.legendre.leggauss(24)
    psi = (np.arange(24) + 0.5) * 2. * np.pi / 24
    M1, C12, PS = (x.ravel() for x in np.meshgrid(mu, c, psi, indexing='ij'))
    wd = (wmu[:, None, None] * wc[None, :, None] * np.ones(psi.size)).ravel()
    wd = wd / wd.sum()
    s1, sc, zeros = np.sqrt(1. - M1**2), np.sqrt(1. - C12**2), np.zeros_like(M1)
    h1 = np.stack([s1, zeros, M1], axis=-1)
    e1, e2 = np.stack([M1, zeros, -s1], axis=-1), np.stack([zeros, zeros + 1., zeros], axis=-1)
    h2 = C12[:, None] * h1 + sc[:, None] * (np.cos(PS)[:, None] * e1 + np.sin(PS)[:, None] * e2)
    quad = np.array([np.sum(wd * np.asarray(get_S(ell, z3=True)(jnp.asarray(h1), jnp.asarray(h2)))**2) for ell in ells])
    analytic = ic._sugiyama_norm(ells)
    print('<Phi^2>: analytic / quadrature - 1', analytic / quad - 1.)
    assert np.allclose(analytic, quad, rtol=1e-12, atol=0.)


def test_smooth3():
    """
    Constant removal kernel tau = c on the band q in [0.05, 0.1]: each leg is replaced by q, so a bispectrum linear in (k1', k2') gives
    Delta B000 = c [B(q, k2) + B(k1, q)] (1 + 1 / 2); constant P^eta = e and theory P = P0 give Delta B000 = 6 e P0.
    """
    from jaxpower import ic
    c, e, p0 = 0.1, 0.3, 2.
    nodes = np.linspace(0., 1., 21)
    kin = np.column_stack([x.ravel() for x in np.meshgrid(nodes, nodes, indexing='ij')])
    ells = [(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)]
    out = ic.compute_smooth3_ic_window(get_toy_ft(c=c), kin=kin, ells=ells, modulation=get_poles([0., 2.], [[e, e]]),
                                       nmu=8, nc=16, npsi=8, nx=4)
    npoint = len(kin)
    sel = np.all(kin <= 0.4, axis=-1)  # all legs and replaced legs inside the nodes and the kernel
    # constant B: Delta B000 = 3 c B0, no other multipole
    bth = np.zeros((len(ells), npoint))
    bth[0] = 1.
    dB = (out['DBB'] @ bth.ravel()).reshape(len(ells), npoint)
    print('constant B: max |Delta B000 / (3 c) - 1|', np.max(np.abs(dB[0, sel] / (3. * c) - 1.)), 'other multipoles', np.max(np.abs(dB[1:, sel])))
    assert np.allclose(dB[0, sel], 3. * c, rtol=1e-10, atol=0.) and np.allclose(dB[1:, sel], 0., atol=1e-10)
    # B000 = k1' + 2 k2': each leg replaced by q, the third one through the triangles (q, k2) and (k1, q), half each
    qc = 0.075
    bth[0] = kin[:, 0] + 2. * kin[:, 1]
    dB = (out['DBB'] @ bth.ravel()).reshape(len(ells), npoint)
    k1, k2 = kin[sel, 0], kin[sel, 1]
    expected = 1.5 * c * ((qc + 2. * k2) + (k1 + 2. * qc))
    print('linear B000: max |Delta B000 / expected - 1|', np.max(np.abs(dB[0, sel] / expected - 1.)))
    assert np.allclose(dB[0, sel], expected, rtol=1e-10, atol=0.)
    # same, k1' <-> k2': legs 1 and 2 swap
    bswap = np.zeros_like(bth)
    bswap[0] = 2. * kin[:, 0] + kin[:, 1]
    dBs = (out['DBB'] @ bswap.ravel()).reshape(len(ells), npoint)
    swap = np.array([np.flatnonzero(np.all(kin == kk[::-1], axis=-1))[0] for kk in kin])
    print('k1 <-> k2 symmetry of B000', np.max(np.abs(dBs[0, swap] - dB[0])))
    assert np.allclose(dBs[0, swap][sel], dB[0][sel], rtol=1e-10, atol=0.)
    # modulation
    pth = np.zeros((3, nodes.size))
    pth[0] = p0
    dB = (out['DBP'] @ pth.ravel()).reshape(len(ells), npoint)
    print('modulation: max |Delta B000 / (6 e P0) - 1|', np.max(np.abs(dB[0, sel] / (6. * e * p0) - 1.)))
    assert np.allclose(dB[0, sel], 6. * e * p0, rtol=1e-10, atol=0.) and np.allclose(dB[1:, sel], 0., atol=1e-10)


def test_fkp_modulation_templates():
    """
    FKP-weight modulation eta = -beta deltabar_cap: the responses L_b(r, s) of the templates (pattern a, mean window b) vanish outside the
    n(z) bin of r, and a density contrast constant over the cap gives eta(r) = -beta(r): sum_b sum_s L_b(r, s) = -beta(r).
    """
    from jaxpower import ic
    rng = np.random.default_rng(42)
    npix, redges = 12 * 8**2, np.linspace(100., 200., 11)
    nr = redges.size - 1
    geometry = [ic.Template(rng.uniform(size=npix), rng.uniform(1., 2., size=nr), redges) for _ in range(2)]
    means = [ic.Template(rng.uniform(size=npix), np.ones(nr), redges) for _ in range(2)]
    beta = rng.uniform(0.2, 0.8, size=nr)
    zbin = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    eta = ic.fkp_modulation_templates(geometry, means, beta, zbin)
    assert len(eta) == 4 and all(t.kind == 'dense' for t in eta)
    for ia in range(2):
        L = [t.response[1] for t in eta if t.group[1] == ia]
        assert np.allclose(sum(L).sum(axis=1), -beta, rtol=1e-12, atol=0.)
        assert all(np.all(LL[zbin[:, None] != zbin[None, :]] == 0.) for LL in L)
    # one beta per region of the pattern: the responses of region a follow beta_a, the cap mean is unchanged
    betas = [beta, rng.uniform(0.2, 0.8, size=nr)]
    eta = ic.fkp_modulation_templates(geometry, means, betas, zbin)
    for ia in range(2):
        L = [t.response[1] for t in eta if t.group[1] == ia]
        assert np.allclose(sum(L).sum(axis=1), -betas[ia], rtol=1e-12, atol=0.)


def get_toy_window3(k, dk, nodes):
    """Sugiyama-diagonal geometry window of zero value: observable the diagonal bins k (width dk), multipoles 000, 202; theory all pairs of
    the nodes, multipole 000."""
    from jaxpower.types import WindowMatrix, Mesh3SpectrumPole, Mesh3SpectrumPoles, ObservableTree
    kk = np.column_stack([k, k])
    edges = np.stack([np.column_stack([k - dk / 2., k + dk / 2.])] * 2, axis=1)
    observable = Mesh3SpectrumPoles([Mesh3SpectrumPole(k=kk, k_edges=edges, nmodes=np.ones(k.size), num_raw=np.zeros(k.size), basis='sugiyama-diagonal', ell=ell)
                                     for ell in [(0, 0, 0), (2, 0, 2)]])
    kin = np.column_stack([x.ravel() for x in np.meshgrid(nodes, nodes, indexing='ij')])
    dn = nodes[1] - nodes[0]
    kin_edges = np.stack([np.column_stack([kin[:, i] - dn / 2., kin[:, i] + dn / 2.]) for i in range(2)], axis=1)
    theory = ObservableTree([Mesh3SpectrumPole(k=kin, k_edges=kin_edges, num_raw=np.zeros(len(kin)), basis='sugiyama')], ells=[(0, 0, 0)])
    return WindowMatrix(observable=observable, theory=theory, value=np.zeros((2 * k.size, len(kin))))


def get_toy_closure_kernel(q, pedges, value=1.):
    """Closure kernel K(q; p): ``value`` from the isotropic input band containing q to the monopole at q, zero otherwise."""
    from jaxpower.types import WindowMatrix, Mesh2SpectrumPole, Mesh2SpectrumPoles, ObservableTree
    observable = Mesh2SpectrumPoles([Mesh2SpectrumPole(k=q, k_edges=np.column_stack([q - 1e-8, q + 1e-8]), nmodes=np.ones_like(q), num_raw=np.zeros_like(q),
                                                       num_shotnoise=np.zeros_like(q), norm=np.ones_like(q), ell=ell) for ell in (0, 2, 4)])
    pc = pedges.mean(axis=-1)
    theory = ObservableTree([Mesh2SpectrumPole(k=pc, k_edges=pedges, num_raw=np.zeros_like(pc)) for _ in (0, 2, 4)], ells=[0, 2, 4])
    K = np.zeros((3, q.size, 3, pc.size))
    K[0, np.arange(q.size), 0, np.searchsorted(pedges[:, 1], q)] = value
    return WindowMatrix(observable=observable, theory=theory, value=K.reshape(3 * q.size, -1))


def test_smooth3_closure_kernel():
    """
    Closure-leg term with a unit kernel (the observed long mode q is the true one, isotropic) and a constant B000 = B0: Delta B000 =
    -(2 pi)^3 / V_k G_k B0, G_k = int_0^qmax 4 pi q^2 dq / (2 pi)^3 w_k(q), w_k the shell overlap averaged over the directions of q (as
    test_smooth3_collapsed_analytic); Delta B202 = 0. Through compute_smooth3_ic_window(closure_kernel=...), with a zero geometry window,
    'BB' is the closure term alone.
    """
    from jaxpower import ic
    b0, qmax, dk = 2., 0.004, 0.005
    q = np.arange(0.0005, qmax, 0.001)
    pedges = np.column_stack([np.arange(0., qmax, 0.001), np.arange(0.001, qmax + 1e-9, 0.001)])
    k = np.arange(0.0225, 0.1, dk)
    nodes = np.linspace(0., 0.2, 81)
    window = get_toy_window3(k, dk, nodes)
    CBB = ic.compute_smooth3_closure_window(window, kernel=get_toy_closure_kernel(q, pedges), nmuq=24, nphi=32)['CBB']
    dB = (np.asarray(CBB.value()) @ np.full(nodes.size**2, b0)).reshape(2, k.size)
    vk = 4. * np.pi / 3. * ((k + dk / 2.)**3 - (k - dk / 2.)**3)
    qq = np.linspace(0., qmax, 801)
    cc = np.linspace(-1., 1., 4001)
    w = np.array([[np.trapezoid(shell_overlap(qx, cc, kx - dk / 2., kx + dk / 2.), cc) / 2. for qx in qq] for kx in k])
    expected = -(2. * np.pi)**3 / vk * np.trapezoid(qq**2 * w, qq, axis=-1) / (2. * np.pi**2) * b0
    print('closure kernel, constant B: max |Delta B000 / expected - 1|', np.max(np.abs(dB[0] / expected - 1.)), 'B202', np.max(np.abs(dB[1] / expected)))
    assert np.allclose(dB[0], expected, rtol=1e-2, atol=0.) and np.all(np.abs(dB[1]) < 1e-2 * np.abs(expected))
    # merged into compute_smooth3_ic_window (qsplit = inf: the exact term at all q, the closure leg not in D^BB),
    # 'BB' = W^geo D^BB_12 + C^BB = C^BB for a zero geometry window
    out = ic.compute_smooth3_ic_window(get_toy_ft(c=0.1), window=window, closure_kernel=get_toy_closure_kernel(q, pedges), qsplit=np.inf,
                                       nmu=4, nc=16, npsi=4, nx=2)
    assert np.allclose(np.asarray(out['BB'].value()), np.asarray(out['CBB'].value()), rtol=0., atol=0.)


def test_smooth3_shotnoise():
    """
    Shot noise of the removal, Delta B^N = sn D^BN P, D^BN the D^BP form with P^eta = -N: a constant N = n0 and theory P = P0 give
    Delta B000 = -6 n0 P0 (three legs, two P each), identical to modulation = -N. With closure_shotnoise the closure leg leaves D^BN (-4 n0 P0,
    the binned legs) for C^BN, the closure term of P^eta_W2 = -N_W2 (Eq. 153), 'BN' (with a zero geometry window, C^BN alone).
    """
    from jaxpower import ic
    n0, p0 = 0.3, 2.
    nodes = np.linspace(0., 1., 21)
    kin = np.column_stack([x.ravel() for x in np.meshgrid(nodes, nodes, indexing='ij')])
    ells = [(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)]
    kn, N = np.array([0., 2.]), np.array([[n0, n0]])
    kw = dict(kin=kin, ells=ells, nmu=8, nc=16, npsi=8, nx=4)
    out = ic.compute_smooth3_ic_window(get_toy_ft(c=0.1), shotnoise=get_poles(kn, N), **kw)
    mod = ic.compute_smooth3_ic_window(get_toy_ft(c=0.1), modulation=get_poles(kn, -N), **kw)
    assert np.array_equal(out['DBN'], mod['DBP'])
    npoint = len(kin)
    sel = np.all(kin <= 0.4, axis=-1)
    pth = np.zeros((3, nodes.size))
    pth[0] = p0
    dB = (out['DBN'] @ pth.ravel()).reshape(len(ells), npoint)
    print('shot noise: max |Delta B000 / (-6 n0 P0) - 1|', np.max(np.abs(dB[0, sel] / (-6. * n0 * p0) - 1.)))
    assert np.allclose(dB[0, sel], -6. * n0 * p0, rtol=1e-10, atol=0.) and np.allclose(dB[1:, sel], 0., atol=1e-10)
    # exact closure leg
    qmax, dk = 0.004, 0.005
    q = np.arange(0.0005, qmax, 0.001)
    pedges = np.column_stack([np.arange(0., qmax, 0.001), np.arange(0.001, qmax + 1e-9, 0.001)])
    k = np.arange(0.0225, 0.1, dk)
    nodes = np.linspace(0., 0.2, 81)
    window = get_toy_window3(k, dk, nodes)
    n_w2 = np.array([np.full(q.size, 1.5), np.zeros(q.size), np.zeros(q.size)])
    out = ic.compute_smooth3_ic_window(get_toy_ft(c=0.1), window=window, closure_kernel=get_toy_closure_kernel(q, pedges), shotnoise=get_poles(kn, N),
                                       closure_shotnoise=get_poles(q, n_w2), qsplit=np.inf, pin=get_poles(nodes, np.zeros((3, nodes.size))),
                                       nmu=4, nc=16, npsi=4, nx=2)
    kin = np.asarray(window.theory.get(ells=(0, 0, 0)).coords('k'))
    sel = np.all(kin <= 0.1, axis=-1) & np.all(kin > 0., axis=-1)  # the third leg |k1 + k2| inside the P nodes (<= 0.2)
    pth = np.zeros((3, nodes.size))
    pth[0] = p0
    dB = (out['DBN'] @ pth.ravel()).reshape(-1, len(kin))[0]
    print('shot noise, binned legs only: max |Delta B000 / (-4 n0 P0) - 1|', np.max(np.abs(dB[sel] / (-4. * n0 * p0) - 1.)))
    assert np.allclose(dB[sel], -4. * n0 * p0, rtol=1e-10, atol=0.)
    collapsed = np.asarray(ic.compute_smooth3_closure_window(window, shotnoise=get_poles(q, n_w2), pin=get_poles(nodes, np.zeros((3, nodes.size))), qmax=np.inf)['CBN'].value())
    assert np.allclose(np.asarray(out['BN'].value()), collapsed, rtol=1e-12, atol=0.) and np.any(collapsed != 0.)


def test_sketch_templates():
    """
    Random-sign probes of the angular-cut templates (sketch_templates, one block per degree L): the 'ft' window (with its shot noise and
    the global IC, relative to the GIC alone) averaged over sign draws converges to that of all the templates (unbiased, within 4 sigma of
    the draws' scatter); blocks of at most nprobe templates are kept as they are, so nprobe >= 2 lcut - 1 is exact.
    """
    from jaxpower import ic
    redges, geometry, _, norm = get_toy_geometry()
    region, lcut = geometry[0], 4
    templates = ic.ellcut_templates(region, lcut, region.angular)
    blocks = [np.arange(L**2, (L + 1)**2) for L in range(lcut)]
    edges, k = np.linspace(0., 0.2, 21), np.linspace(0.01, 0.1, 10)

    def ft(tpl):  # P^ft and its shot noise at the toy theory, templates with the GIC minus the GIC alone
        out = []
        for t in [tpl, []]:
            w = ic.compute_smooth2_ic_window(ic.IC2Model([region], t, norm=norm, lmax=30), edges, k, terms=('ft',))
            out.append(apply_window(w, 'ft', edges, 3) + 1e3 * np.asarray(w['ft', 'shotnoise']))
        return out[0] - out[1]

    exact = ft(templates)
    assert np.array_equal(ft(ic.sketch_templates(templates, nprobe=2 * lcut - 1, blocks=blocks)), exact)
    draws = np.array([ft(ic.sketch_templates(templates, nprobe=2, blocks=blocks, seed=seed)) for seed in range(16)])
    mean, err = draws.mean(0), draws.std(0, ddof=1) / np.sqrt(len(draws))
    scale = np.max(np.abs(exact[0]))
    print('sketch: max |mean of draws - exact| / max |exact|', np.max(np.abs(mean - exact)[0]) / scale, '; one draw scatter / max |exact|', np.max(draws.std(0)[0]) / scale,
          '; pull', np.max(np.abs((mean - exact) / err)[0]))
    assert np.all(np.abs(mean - exact)[:2] < 4. * err[:2] + 1e-10 * scale)


def test_ellcut_rank_deficient():
    """Angular cut on a footprint too small for its harmonics (rank below lcut^2): the orthonormal templates number the rank, and their
    removal, sum_a Theta_a (U_a . d), is the footprint times epsilon of ellcut_fit, for any contrast d."""
    from jaxpower import ic
    import healpy as hp
    nside, lcut = 8, 6
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    rng = np.random.default_rng(5)
    R = np.where(theta < 0.25, 1. + 0.3 * rng.uniform(size=npix), 0.)
    pixels = np.flatnonzero(R > 0.)
    rank = np.linalg.matrix_rank(ic.real_harmonic_maps(lcut, nside, pixels=pixels) * np.sqrt(R[pixels]), tol=1e-10 * lcut)
    assert len(pixels) < lcut**2 or rank < lcut**2
    counts = np.array([0.5 * R, 0.5 * R])
    region = ic.Template(counts, np.array([1., 2.]), np.array([1000., 1100., 1200.]))
    templates = ic.ellcut_templates(region, lcut, counts)
    pixarea = 4. * np.pi / npix
    D = R * (1. + 0.1 * rng.normal(size=npix))
    D = D * R.sum() / D.sum()
    c = ic.ellcut_fit(D, R, lcut)
    eps = c @ ic.real_harmonic_maps(lcut, nside)
    d = np.where(R > 0., D / R - 1., 0.)
    got = sum(t.angular[0] * np.sum(t.mean[0][0][0] * d) * pixarea * t.mean[0][1] for t in templates)
    foot = counts[0] / (counts[0].sum() * pixarea)
    print(f'ellcut rank deficient: {len(pixels)} pixels, {lcut**2} harmonics, {len(templates)} templates; max |diff| / max',
          np.max(np.abs(got - foot * eps)) / np.max(np.abs(foot * eps)))
    assert len(templates) <= min(len(pixels), lcut**2)
    assert np.allclose(got, foot * eps, rtol=0., atol=1e-8 * np.max(np.abs(foot * eps)))


def get_toy_window3_pairs(pairs, dk, nodes):
    """As get_toy_window3, with the bins given as pairs of centres (k1, k2), diagonal or not."""
    from jaxpower.types import WindowMatrix, Mesh3SpectrumPole, Mesh3SpectrumPoles, ObservableTree
    kk = np.asarray(pairs, dtype='f8')
    edges = np.stack([np.column_stack([kk[:, i] - dk / 2., kk[:, i] + dk / 2.]) for i in range(2)], axis=1)
    observable = Mesh3SpectrumPoles([Mesh3SpectrumPole(k=kk, k_edges=edges, nmodes=np.ones(len(kk)), num_raw=np.zeros(len(kk)), basis='sugiyama', ell=ell)
                                     for ell in [(0, 0, 0), (2, 0, 2)]])
    kin = np.column_stack([x.ravel() for x in np.meshgrid(nodes, nodes, indexing='ij')])
    dn = nodes[1] - nodes[0]
    kin_edges = np.stack([np.column_stack([kin[:, i] - dn / 2., kin[:, i] + dn / 2.]) for i in range(2)], axis=1)
    theory = ObservableTree([Mesh3SpectrumPole(k=kin, k_edges=kin_edges, num_raw=np.zeros(len(kin)), basis='sugiyama')], ells=[(0, 0, 0)])
    return WindowMatrix(observable=observable, theory=theory, value=np.zeros((2 * len(kk), len(kin))))


def test_smooth3_closure_split():
    """
    One rule for the closure leg of every term: composed for k3 >= qsplit, exact below. qsplit = 0: the exact terms vanish and D^BB, D^BP,
    D^BN are the plain composition (no exact terms). The modulation with P^eta = -N and its closure term with P^eta_W2 = -N_W2 are
    the shot noise of the removal. Off-diagonal bins: the shell overlap of two shells, V1 Lambda_12 = V2 Lambda_21 (the term is symmetric
    in the two legs), the diagonal recovered, zero for shells further apart than the closure range.
    """
    from jaxpower import ic
    qmax, dk, n0 = 0.004, 0.005, 0.3
    q = np.arange(0.0005, qmax, 0.001)
    pedges = np.column_stack([np.arange(0., qmax, 0.001), np.arange(0.001, qmax + 1e-9, 0.001)])
    k = np.arange(0.0225, 0.1, dk)
    nodes = np.linspace(0., 0.2, 81)
    window = get_toy_window3(k, dk, nodes)
    kn, N = np.array([0., 2.]), np.array([[n0, n0]])
    n_w2 = np.array([np.full(q.size, 1.5), np.zeros(q.size), np.zeros(q.size)])
    K = get_toy_closure_kernel(q, pedges)
    kw = dict(window=window, pin=get_poles(nodes, np.zeros((3, nodes.size))), nmu=4, nc=16, npsi=4, nx=2)
    plain = ic.compute_smooth3_ic_window(get_toy_ft(c=0.1), modulation=get_poles(kn, -N), shotnoise=get_poles(kn, N), **kw)
    split0 = ic.compute_smooth3_ic_window(get_toy_ft(c=0.1), closure_kernel=K, modulation=get_poles(kn, -N), closure_modulation=get_poles(q, -n_w2),
                                          shotnoise=get_poles(kn, N), closure_shotnoise=get_poles(q, n_w2), qsplit=0., **kw)
    assert np.allclose(split0['DBB'].toarray(), plain['DBB'].toarray(), rtol=0., atol=0.)
    for name in ['DBP', 'DBN']: assert np.array_equal(split0[name], plain[name])
    assert np.all(np.asarray(split0['CBB'].value()) == 0.)
    for name in ['BP', 'BN']: assert np.allclose(np.asarray(split0[name].value()), np.asarray(plain[name].value()), rtol=0., atol=0.)
    # all exact (qsplit = inf): the modulation P^eta = -N with P^eta_W2 = -N_W2 is the shot noise of the removal
    inf = ic.compute_smooth3_ic_window(get_toy_ft(c=0.1), modulation=get_poles(kn, -N), closure_modulation=get_poles(q, -n_w2), shotnoise=get_poles(kn, N),
                                       closure_shotnoise=get_poles(q, n_w2), qsplit=np.inf, **kw)
    assert np.array_equal(inf['DBP'], inf['DBN'])
    assert np.allclose(np.asarray(inf['BP'].value()), np.asarray(inf['BN'].value()), rtol=1e-12, atol=0.) and np.any(np.asarray(inf['BP'].value()) != 0.)
    # off-diagonal bins: collapsed term and C^BB
    pairs = [(0.05, 0.05), (0.05, 0.055), (0.055, 0.05), (0.05, 0.07)]
    win = get_toy_window3_pairs(pairs, dk, nodes)
    pth = np.zeros((3, nodes.size))
    pth[0] = 2.
    for name, value in [('collapsed', np.asarray(ic.compute_smooth3_closure_window(win, shotnoise=get_poles(q, n_w2), pin=get_poles(nodes, np.zeros((3, nodes.size))), qmax=np.inf)['CBN'].value()) @ pth.ravel()),
                        ('CBB', np.asarray(ic.compute_smooth3_closure_window(win, kernel=K, nmuq=24, nphi=32)['CBB'].value()) @ np.full(nodes.size**2, 2.))]:
        v = value.reshape(2, len(pairs))[0]
        diag = (ic.compute_smooth3_closure_window(k=np.array([0.05]), kedges=np.array([[0.0475, 0.0525]]), shotnoise=get_poles(q, n_w2), pin=get_poles(nodes, np.zeros((3, nodes.size))), qmax=np.inf)['CBN'] @ pth.ravel())[0] \
            if name == 'collapsed' else v[0]
        print(f'{name}: diagonal {v[0]:.4e}, adjacent {v[1]:.4e} / {v[2]:.4e}, far {v[3]:.1e}')
        assert np.isclose(v[0], diag, rtol=1e-12) and np.isclose(v[1], v[2], rtol=2e-2) and 0. < abs(v[1]) < abs(v[0]) and v[3] == 0.


def test_smooth3_closure_exact():
    """
    The closure terms beyond the squeezed limit, against a direct sum over the mode pairs: k1 in the first shell, q in the ball q < qmax,
    |k1 + q| in the second, weighted by the projection weight at the true k2 = -(k1 + q) and [P(k1, mu1) + P(k2, mu2)] / 2, for X constant
    (Delta B = (2 pi)^3 / (V1 V2) int d^3k1 d^3q / (2 pi)^3 w X (P1 + P2) / 2), with q up to the bin centre, diagonal and adjacent bins, B000
    and B202, P0 and P2. And C^BB with all the input multipoles of the kernel, as tau in D^BB: an input quadrupole only (the l' = 2 column of
    K), on an isotropic B, gives zero.
    """
    import jax.numpy as jnp
    from jaxpower import ic
    from jaxpower.utils import get_S
    qmax, dk, x0 = 0.02, 0.005, 2.5
    qedges = np.column_stack([np.arange(0., qmax, 5e-4), np.arange(5e-4, qmax + 1e-9, 5e-4)])
    X = np.array([np.full(len(qedges), x0)])
    pairs = np.array([(0.025, 0.025), (0.025, 0.03), (0.06, 0.065)])
    edges = np.stack([np.stack([pairs[:, i] - dk / 2., pairs[:, i] + dk / 2.], axis=-1) for i in range(2)], axis=1)
    ells = [(0, 0, 0), (2, 0, 2)]
    nodes = np.linspace(0., 0.2, 401)
    out = ic.compute_smooth3_closure_window(modulation=get_poles(qedges.mean(axis=-1), X / 2.), k=pairs, kedges=edges, ells=ells, pin=get_poles(nodes, np.zeros((2, nodes.size))), nmu=32, nmuq=32, nphi=48)['CBP']
    # projection normalization <S_e^2>, as in the code
    mu, wmu = np.polynomial.legendre.leggauss(16)
    c, wc = np.polynomial.legendre.leggauss(64)
    psi = (np.arange(16) + 0.5) * 2. * np.pi / 16
    M1, C12, PS = (x.ravel() for x in np.meshgrid(mu, c, psi, indexing='ij'))
    wd = (wmu[:, None, None] * wc[None, :, None] * np.ones(psi.size)).ravel()
    wd = wd / wd.sum()
    s1, sc, zeros = np.sqrt(1. - M1**2), np.sqrt(1. - C12**2), np.zeros_like(M1)
    h1 = np.stack([s1, zeros, M1], axis=-1)
    e1, e2 = np.stack([M1, zeros, -s1], axis=-1), np.stack([zeros, zeros + 1., zeros], axis=-1)
    h2 = C12[:, None] * h1 + sc[:, None] * (np.cos(PS)[:, None] * e1 + np.sin(PS)[:, None] * e2)
    S2norm = [float(np.sum(wd * np.asarray(get_S(ell, z3=True)(jnp.asarray(h1), jnp.asarray(h2)))**2)) for ell in ells]
    rng = np.random.default_rng(42)
    n = 4000000
    unit = lambda: (lambda v: v / np.linalg.norm(v, axis=-1)[:, None])(rng.normal(size=(n, 3)))
    L2 = lambda x: 1.5 * x**2 - 0.5
    for ip, Ppoles in enumerate([(1., 0.), (0., 1.)]):  # P0 = 1, then P2 = 1
        pth = np.zeros((2, nodes.size))
        pth[ip] = 1.
        model = (out @ pth.ravel()).reshape(len(ells), len(pairs))
        for ib, ((klo1, khi1), (klo2, khi2)) in enumerate(edges):
            r1 = (klo1**3 + rng.uniform(size=n) * (khi1**3 - klo1**3))**(1. / 3.)
            k1 = r1[:, None] * unit()
            q = qmax * rng.uniform(size=n)[:, None]**(1. / 3.) * unit()
            k2 = -(k1 + q)
            r2 = np.linalg.norm(k2, axis=-1)
            inside = (r2 >= klo2) & (r2 < khi2)
            h1_, h2_ = k1 / r1[:, None], k2 / r2[:, None]
            P = 0.5 * (Ppoles[0] * 2. + Ppoles[1] * (L2(h1_[:, 2]) + L2(h2_[:, 2])))
            v2, vq = 4. * np.pi / 3. * (khi2**3 - klo2**3), 4. * np.pi / 3. * qmax**3
            for ie, ell in enumerate(ells):
                w = np.asarray(get_S(ell, z3=True)(jnp.asarray(h1_), jnp.asarray(h2_))) / S2norm[ie]
                direct = vq / v2 * np.mean(inside * w * x0 * P)
                err = vq / v2 * np.std(inside * w * x0 * P) / np.sqrt(n)
                print(f'collapsed, bins {pairs[ib]}, P{2 * ip}, B{"".join(map(str, ell))}: model {model[ie, ib]:.5e}, direct {direct:.5e} +- {err:.1e}')
                assert abs(model[ie, ib] - direct) < 4. * err + 1e-2 * abs(direct)
    # symmetric in the two binned legs: B_{l1 l2 L}(k1, k2) = B_{l2 l1 L}(k2, k1)
    swap = ic.compute_smooth3_closure_window(modulation=get_poles(qedges.mean(axis=-1), X / 2.), k=pairs[[1, 2]][:, ::-1], kedges=edges[[1, 2]][:, ::-1],
                                             ells=[(0, 0, 0), (0, 2, 2)], pin=get_poles(nodes, np.zeros((2, nodes.size))), nmu=16, nmuq=16, nphi=16)['CBP']
    direct = ic.compute_smooth3_closure_window(modulation=get_poles(qedges.mean(axis=-1), X / 2.), k=pairs[[1, 2]], kedges=edges[[1, 2]],
                                               ells=[(0, 0, 0), (2, 0, 2)], pin=get_poles(nodes, np.zeros((2, nodes.size))), nmu=16, nmuq=16, nphi=16)['CBP']
    print('collapsed: max |B_{l1 l2 L}(k1, k2) - B_{l2 l1 L}(k2, k1)| / max', np.max(np.abs(swap - direct)) / np.max(np.abs(direct)))
    assert np.allclose(swap, direct, rtol=0., atol=1e-12 * np.max(np.abs(direct)))
    # C^BB: an input quadrupole only, on an isotropic B: zero
    q = np.arange(0.0005, 0.004, 0.001)
    pedges = np.column_stack([np.arange(0., 0.004, 0.001), np.arange(0.001, 0.004 + 1e-9, 0.001)])
    K = get_toy_closure_kernel(q, pedges)
    value = np.asarray(K.value()).reshape(3, q.size, 3, -1)
    value = np.roll(value, 1, axis=2)  # the unit entries moved from the input monopole column to the quadrupole
    K = K.clone(value=value.reshape(3 * q.size, -1))
    window = get_toy_window3(np.arange(0.0225, 0.1, dk), dk, nodes[::5])
    dB = np.asarray(ic.compute_smooth3_closure_window(window, kernel=K)['CBB'].value()) @ np.full(nodes[::5].size**2, 2.)
    print('closure kernel with an input quadrupole only, isotropic B: max |Delta B|', np.max(np.abs(dB)))
    assert np.allclose(dB, 0., atol=1e-12)


def test_smooth3_collapsed_analytic():
    """
    Closure term C^BP with P^eta_W2 = x0 / 2 constant (X = 2 P^eta_W2 = x0), for q < qmax: G_k = x0 int_0^qmax 4 pi q^2 dq / (2 pi)^3 w_k(q), w_k the shell overlap averaged
    over the directions of q, 1/2 int dc Lambda_k(q, c) (here on a fine 1D grid; the code averages over k^ and q^ in 3D), Delta B000 =
    (2 pi)^3 / V_k G_k P0, Delta B202 = (2 pi)^3 / V_k G_k P2. Also the exact overlap against its first order,
    1 - q / (2 dk) averaged, for q << k: within 3% at qmax = 0.004 from k = 0.05.
    """
    from jaxpower import ic
    x0, qmax, dk = 2.5, 0.004, 0.005
    qedges = np.column_stack([np.arange(0., qmax, 1e-4), np.arange(1e-4, qmax + 1e-5, 1e-4)])
    X = np.array([np.full(len(qedges), x0), np.zeros(len(qedges))])
    kedges = np.column_stack([np.arange(0.02, 0.1, dk), np.arange(0.02, 0.1, dk) + dk])
    k = kedges.mean(axis=-1)
    vk = 4. * np.pi / 3. * (kedges[:, 1]**3 - kedges[:, 0]**3)
    pth = np.zeros((3, k.size))
    pth[0], pth[1] = 3. + k, -1. + 2. * k
    out = ic.compute_smooth3_closure_window(modulation=get_poles(qedges.mean(axis=-1), X / 2.), k=k, kedges=kedges, ells=[(0, 0, 0), (2, 0, 2)])
    dB = (out['CBP'] @ pth.ravel()).reshape(2, k.size)
    qq = np.linspace(0., qmax, 801)
    cc = np.linspace(-1., 1., 4001)
    w = np.array([[np.trapezoid(shell_overlap(q, cc, lo, hi), cc) / 2. for q in qq] for lo, hi in kedges])  # (nk, nq)
    G = x0 / (2. * np.pi**2) * np.trapezoid(qq**2 * w, qq, axis=-1)
    Glin = x0 / (2. * np.pi**2) * (qmax**3 / 3. - qmax**4 / (8. * dk))
    print('collapsed: exact / first-order overlap ' + ' '.join(f'{x:.3f}' for x in (G / Glin)[::3]) + ' at k ' + ' '.join(f'{x:.3f}' for x in k[::3]))
    assert np.all(np.abs(G / Glin - 1.)[k > 0.05] < 0.03)
    for ill, ip in enumerate([0, 1]):
        expected = (2. * np.pi)**3 / vk * G * pth[ip]
        print(f'collapsed, B{"000" if ill == 0 else "202"}: max |Delta B / expected - 1|', np.max(np.abs(dB[ill] / expected - 1.)))
        # the expectation is the squeezed limit (k2 = -k1); the code takes the true k2 = -(k1 + q), which leaves B000 unchanged and moves
        # B202 by up to 1% here (test_smooth3_closure_exact checks it against a direct sum over the mode pairs)
        assert np.allclose(dB[ill], expected, rtol=1.5e-2 if ill else 2e-3, atol=0.)


def test_ellcut():
    """Angular cut: the fit recovers a low-degree pattern exactly, its mean windows reproduce the fit, the randoms keep their total weight."""
    from jaxpower import ic
    import healpy as hp
    nside, lcut = 32, 5
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    rng = np.random.default_rng(0)
    R = np.where(theta < 1., 1. + 0.3 * rng.uniform(size=npix), 0.)  # a cap, non-uniform randoms
    c_true = 0.05 * rng.normal(size=lcut**2)
    Y = ic.real_harmonic_maps(lcut, nside)
    D = R * (1. + c_true @ Y)
    c_fit = ic.ellcut_fit(D, R * 3.7, lcut)  # the scale of R does not matter
    # R is normalized to the total of D (alpha), which absorbs the mean m of the pattern over the randoms: R' = R (1 + m), and the fit
    # of (D - R') / R' gives (c - m / Y_00 on Y_00) / (1 + m)
    m = np.sum(R * (c_true @ Y)) / np.sum(R)
    c_expected = c_true.copy()
    c_expected[0] -= m / Y[0, 0]
    c_expected /= 1. + m
    assert np.allclose(c_fit, c_expected, rtol=0., atol=1e-7), np.max(np.abs(c_fit - c_expected))
    eps = hp.alm2map(ic.real_to_complex_alm(c_fit, lcut), nside, lmax=lcut - 1, pixwin=False)
    assert np.allclose(eps, c_expected @ Y, atol=1e-7)
    # any contrast: the removed pattern is orthogonal to every harmonic on the footprint (weights R), and keeps the randoms' total weight
    D = R * (1. + 0.2 * rng.normal(size=npix))
    D = D * R.sum() / D.sum()
    c = ic.ellcut_fit(D, R, lcut)
    eps = c @ Y
    resid = (D - R) - R * eps
    assert np.max(np.abs(Y[:, R > 0] @ resid[R > 0])) < 1e-7 * np.sum(np.abs(D - R))
    assert abs(np.sum(R * eps)) < 1e-7 * R.sum()
    # the templates: mean windows read the fit coefficients from the projected contrast, with two random subsets
    redges = np.array([1000., 1100., 1200.])
    counts = np.array([0.4 * R, 0.6 * R])
    region = ic.Template(counts, np.array([1., 2.]), redges)
    templates = ic.ellcut_templates(region, lcut, counts, orthonormal=False)  # one template per harmonic
    assert len(templates) == lcut**2
    d = np.where(R > 0., D / R - 1., 0.)
    pixarea = 4. * np.pi / npix
    for a, t in enumerate(templates):
        for isub in range(2):
            assert np.isclose(np.sum(t.mean[0][0][isub] * d) * pixarea * t.mean[0][1], c[a], rtol=1e-8, atol=1e-12)
    norm = counts / (counts.sum(axis=-1, keepdims=True) * pixarea)
    assert np.allclose(templates[3].angular, norm * Y[3][None])
    # orthonormal basis: the same removal operator, sum_a Theta_a (U_a . v) for any v, and every template but the first of zero survey mean
    ortho = ic.ellcut_templates(region, lcut, counts, orthonormal=True)
    v = rng.normal(size=npix)
    apply = lambda tpl: sum(t.angular * np.sum(t.mean[0][0] * v, axis=-1, keepdims=True) * t.mean[0][1] for t in tpl)
    assert np.allclose(apply(ortho), apply(templates), rtol=0., atol=1e-6 * np.max(np.abs(apply(templates))))  # cut sky: Y^T N Y ill-conditioned
    assert all(np.max(np.abs(np.sum(t.mean[0][0], axis=-1))) < 1e-8 * np.max(np.abs(np.sum(ortho[0].mean[0][0], axis=-1))) for t in ortho[1:])


if __name__ == '__main__':
    test_amr_templates()
    test_shell_widths()
    test_radial_shells()
    test_window_convergence()
    test_response_kinds()
    test_theory_multipoles()
    test_no_gic()
    test_gic()
    test_harmonic()
    test_harmonic_mixed()
    test_sugiyama_norm()
    test_smooth3()
    test_fkp_modulation_templates()
    test_smooth3_closure_kernel()
    test_smooth3_shotnoise()
    test_smooth3_closure_split()
    test_smooth3_closure_exact()
    test_sketch_templates()
    test_smooth3_collapsed_analytic()
    test_ellcut()
    test_ellcut_rank_deficient()
