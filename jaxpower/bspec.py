import itertools
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from dataclasses import dataclass

from .mesh import BaseMeshField, MeshAttrs, RealMeshField, ComplexMeshField, staticarray, get_sharding_mesh, _get_hermitian_weights, _find_unique_edges, _get_bin_attrs, _bincount
from .pspec import get_real_Ylm, _get_los_vector, _get_zero
from .utils import real_gaunt, BinnedStatistic, get_legendre, get_spherical_jn, plotter


@jax.tree_util.register_pytree_node_class
@dataclass(init=False, frozen=True)
class bin_bspec(object):

    edges: jax.Array = None
    nmodes: jax.Array = None
    xavg: jax.Array = None
    ibin: jax.Array = None
    wmodes: jax.Array = None
    _iedges: jax.Array = None
    _buffer_iedges: tuple = None
    _nmodes1d: jax.Array = None
    basis: str = 'sugiyama'
    batch_size: int = 1
    buffer_size: int = 0

    def __init__(self, mattrs: MeshAttrs | BaseMeshField, edges: staticarray | dict | None=None, basis='sugyama', batch_size=1, buffer_size=0):
        kind = 'complex'
        if not isinstance(mattrs, MeshAttrs):
            kind = 'complex' if 'complex' in mattrs.__class__.__name__.lower() else 'real'
            mattrs = mattrs.attrs
        hermitian = mattrs.hermitian
        if kind == 'real':
            vec = mattrs.xcoords(kind='separation', sparse=True)
            vec0 = mattrs.cellsize.min()
        else:
            vec = mattrs.kcoords(kind='separation', sparse=True)
            vec0 = mattrs.kfun.min()
        wmodes = None
        if hermitian:
            wmodes = _get_hermitian_weights(vec, sharding_mesh=None)
        if edges is None:
            edges = {}
        if isinstance(edges, dict):
            step = edges.get('step', None)
            if step is None:
                edges = _find_unique_edges(vec, vec0, xmin=edges.get('min', 0.), xmax=edges.get('max', np.inf))[0]
            else:
                edges = np.arange(edges.get('min', 0.), edges.get('max', vec0 * np.min(mattrs.meshsize) / 2.), step)

        assert basis in ['sugiyama', 'sugiyama-diagonal', 'scoccimarro', 'scoccimarro-equilateral']

        ibin, nmodes, xsum = [], 0, 0
        coords = jnp.sqrt(sum(xx**2 for xx in vec))
        ibin, nmodes, xsum = _get_bin_attrs(coords, edges, wmodes)
        ibin = ibin.reshape(coords.shape)
        xsum /= nmodes
        nmodes1d = nmodes
        if 'scoccimarro' in basis: ndim = 3
        else: ndim = 2

        def _product(array):
            if not isinstance(array, (tuple, list)):
                array = [array] * ndim
            if 'diagonal' in basis or 'equilateral' in basis:
                grid = jnp.array(array)
            else:
                grid = jnp.meshgrid(*array, sparse=False, indexing='ij')
            return jnp.column_stack([tmp.ravel() for tmp in grid])

        xmid = _product((edges[:-1] + edges[1:]) / 2.)
        mask = True
        for i in range(xmid.shape[1] - 1): mask &= xmid[:, i] <= xmid[:, i + 1]  # select k1 <= k2 <= k3...

        edges = jnp.concatenate([_product(edges[:-1])[..., None], _product(edges[1:])[..., None]], axis=-1)[mask]
        xavg = _product(xsum)[mask]
        nmodes = jnp.prod(_product(nmodes)[mask], axis=-1)
        iedges = _product(jnp.arange(len(xsum)))[mask]

        if 'scoccimarro' in basis:
            xmid = jnp.mean(edges, axis=-1).T
            mask = (xmid[2] >= jnp.abs(xmid[0] - xmid[1])) & (xmid[2] <= jnp.abs(xmid[0] + xmid[1]))
            edges, xavg, nmodes, iedges = edges[mask], xavg[mask], nmodes[mask], iedges[mask]

        if buffer_size >= 1:
            iuedges = jnp.arange(len(xsum))
            iuedges_size = (len(iuedges) + buffer_size - 1) // buffer_size * buffer_size
            iuedges.append(jnp.pad(iuedges, pad_width=(0, iuedges_size - len(iuedges)), mode='edge').reshape(-1, buffer_size))
            _buffer_iedges, _buffer_iuedges = [], []
            for biuedges in itertools.product(*iuedges):
                _buffer_iedges.append(_product(biuedges))
                _buffer_iuedges.append(jnp.stack(biuedges))
            _buffer_iedges = jnp.stack(_buffer_iedges), jnp.stack(_buffer_iuedges)
            _buffer_sort = jnp.array([jnp.flatnonzero(jnp.all(iedge == _buffer_iedges.reshape(-1, iedges.shape[1]), axis=1))[0] for iedge in iedges])
            _buffer_iedges =_buffer_iedges + (_buffer_sort,)
        else:
            _buffer_iedges = None

        if 'scoccimarro' in basis:
            symfactor = jnp.ones_like(xmid[0])
            symfactor = jnp.where((xmid[1] == xmid[0]) | (xmid[2] == xmid[0]) | (xmid[2] == xmid[1]), 2, symfactor)
            symfactor = jnp.where((xmid[1] == xmid[0]) & (xmid[2] == xmid[0]), 6, symfactor)

            def f(_ibin):
                mesh_prod = 1.
                for ivalue in range(ndim):
                    mask = ibin == _ibin[ivalue]
                    mesh_prod = mesh_prod * mattrs.c2r(mask.astype(mattrs.dtype))
                return mesh_prod.sum()

            nmodes = jax.lax.map(f, iedges) * symfactor

        self.__dict__.update(edges=edges, xavg=xavg, nmodes=nmodes, ibin=ibin, wmodes=wmodes, mattrs=mattrs, basis=basis, batch_size=batch_size, buffer_size=buffer_size,
                             _iedges=iedges, _buffer_iedges=_buffer_iedges, _nmodes1d=nmodes1d)

    @property
    def sharding_mesh(self):
        return get_sharding_mesh()

    def __call__(self, *meshs, remove_zero=False):
        values = []
        for imesh, mesh in enumerate(meshs):
            value = mesh.value if isinstance(mesh, BaseMeshField) else mesh
            if remove_zero and imesh < self.xavg.shape[1]:  # 0, 1 for sugiyama, 0, 1, 2 for scoccimaro
                value = value.at[(0,) * value.ndim].set(0.)
            values.append(value)

        if self._buffer_iedges is None:
            def f(ibin):
                if 'scoccimarro' in self.basis:
                    mesh_prod = 1.
                else:
                    mesh_prod = values[2]
                for ivalue, value in enumerate(values[:self.xavg.shape[1]]):  # values if 'scoccimarro' else values[:2]
                    mesh_prod = mesh_prod * self.mattrs.c2r(value * (self.ibin == ibin[ivalue]))
                return mesh_prod.sum()

            return jax.lax.map(f, self._iedges, batch_size=self.batch_size) / self.nmodes

        else:
            def f(iedges, uedges):

                def f_bin(value, ibin):
                    return self.mattrs.c2r(value * (self.ibin == ibin))

                iter_binned_values = [jax.vmap(partial(f_bin, value))(edge) for value, edge in zip(values[:self.xavg.shape[1]], uedges)]

                def f_prod(index):
                    if 'scoccimarro' in self.basis: mesh_prod = 1.
                    else: mesh_prod = values[2]
                    for ivalue, value in enumerate(iter_binned_values):
                        mesh_prod *= value[index[ivalue]]
                    return mesh_prod.sum()

                return jax.vmap(f_prod)(iedges)

            return jax.lax.map(f, *self._buffer_iedges[:2]).ravel()[self._buffer_iedges[2]] / self.nmodes

    def tree_flatten(self):
        state = {name: getattr(self, name) for name in self.__annotations__.keys()}
        meta_fields = ['basis', 'batch_size']
        return tuple(state[name] for name in state if name not in meta_fields), {name: state[name] for name in meta_fields}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new = cls.__new__(cls)
        new.__dict__.update(zip(cls.__annotations__.keys(), children))
        new.__dict__.update(aux_data)
        return new


@jax.tree_util.register_pytree_node_class
class BispectrumMultipoles(BinnedStatistic):

    _data_fields = BinnedStatistic._data_fields + ['_norm', '_num_shotnoise', '_num_zero']
    _select_x_fields = BinnedStatistic._select_proj_fields + ['_num_shotnoise']
    _select_proj_fields = BinnedStatistic._select_proj_fields + ['_num_shotnoise', '_num_zero']
    _sum_fields = BinnedStatistic._sum_fields + ['_norm', '_num_shotnoise', '_num_zero']
    _meta_fields = BinnedStatistic._meta_fields + ['basis']
    _init_fields = BinnedStatistic._init_fields | {'k': '_x', 'num': '_value', 'nmodes': '_weights', 'edges': '_edges', 'ells': '_projs', 'norm': '_norm',
                                                   'num_shotnoise': '_num_shotnoise', 'num_zero': '_num_zero', 'name': 'name', 'attrs': 'attrs'}


    def __init__(self, k: np.ndarray, num: jax.Array, ells: tuple, nmodes: np.ndarray=None, edges: np.ndarray=None, norm: jax.Array=1.,
                 num_shotnoise: jax.Array=None, num_zero: jax.Array=None, name: str=None, basis: str=None, attrs: dict=None):

        def _tuple(item):
            if item is None:
                return None
            if not isinstance(item, (tuple, list)):
                item = (item,) * len(ells)
            return tuple(item)

        num = tuple(num)
        num_zero = tuple(num_zero)
        if num_shotnoise is None:
            num_shotnoise = tuple(jnp.zeros_like(num) for num in num)

        kw = dict()
        if 'scoccimarro' in basis:
            kw.update(_label_x=r'$k_1, k_2, k_3$ [$h/\mathrm{Mpc}$]', _label_proj=r'$\ell_3$', _label_value=r'$B_{\ell_3}(k_3)$ [$(\mathrm{Mpc}/h)^{6}$]')
        else:
            kw.update(_label_x=r'$k_1, k_2$ [$h/\mathrm{Mpc}$]', _label_proj=r'$\ell_1, \ell_2, \ell_3$', _label_value=r'$B_{\ell_1, \ell_2, \ell_3}(k_1, k_2)$ [$(\mathrm{Mpc}/h)^{6}$]')

        self.__dict__.update(_norm=jnp.asarray(norm), _num_shotnoise=jnp.asarray(num_shotnoise), _num_zero=tuple(jnp.asarray(p) for p in num_zero), basis=basis, **kw)
        super().__init__(x=_tuple(k), edges=_tuple(edges), projs=ells, value=num, weights=_tuple(nmodes), name=name, attrs=attrs)

    @property
    def norm(self):
        """Power spectrum normalization."""
        return self._norm

    @property
    def shotnoise(self):
        """Shot noise."""
        return self._num_shotnoise / self._norm

    @property
    def num(self):
        """Power spectrum without shot noise."""
        return self._value

    k = BinnedStatistic.x
    kavg = BinnedStatistic.xavg
    nmodes = BinnedStatistic.weights

    @property
    def ells(self):
        return self._projs

    @property
    def value(self):
        """Power spectrum estimate."""
        toret = []
        for ill, ells in enumerate(self._projs):
            toret.append(self.num[ill] - self._num_shotnoise[ill])
            dig_zero = jnp.all((self._edges[ill][..., 0] <= 0.) & (self._edges[ill][..., 1] >= 0.), axis=1)
            toret[ill] = toret[ill].at[dig_zero].add(- self._num_zero[ill] / self._weights[ill][dig_zero])
        return tuple(tmp / self._norm for tmp in toret)

    @plotter
    def plot(self, fig=None):
        r"""
        Plot bispectrum.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally, a figure with at least 1 axis.

        fn : str, Path, default=None
            Optionally, path where to save figure.
            If not provided, figure is not saved.

        kw_save : dict, default=None
            Optionally, arguments for :meth:`matplotlib.figure.Figure.savefig`.

        show : bool, default=False
            If ``True``, show figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from matplotlib import pyplot as plt
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        for ill, ell in enumerate(self.ells):
            ax.plot(np.arange(len(self._x[ill])), self._x[ill].prod(axis=-1) * self.value[ill].real, label=self._get_label_proj(ell))
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('bin index')
        if 'scoccimarro' in self.basis:
            ax.set_ylabel(r'$k_1 k_2 k_3 B_{\ell_1 \ell_2 \ell_3}(k_1, k_2, k_3)$ [$(\mathrm{Mpc}/h)^{6}$]')
        else:
            ax.set_ylabel(r'$k_1 k_2 B_{\ell_1 \ell_2 \ell_3}(k_1, k_2)$ [$(\mathrm{Mpc}/h)^{4}$]')
        return fig


import functools
import operator


prod = functools.partial(functools.reduce, operator.mul)


def _format_meshs(*meshs):
    meshs = list(meshs)
    meshs = meshs + [None] * (3 - len(meshs))
    same = [0]
    for mesh in meshs[1:]: same.append(same[-1] if mesh is None else same[-1] + 1)
    for imesh, mesh in enumerate(meshs):
        if mesh is None:
            meshs[imesh] = meshs[imesh - 1]
    return meshs, tuple(same)


def _format_ells(ells, basis: str='sugiyama'):
    if 'scoccimarro' in basis:
        if np.ndim(ells) == 0:
            ells = [ells]
        ells = list(ells)
    else:
        msg = 'ells must be (a list of) (ell1, ell2, L)'
        assert np.ndim(ells) != 0, msg
        if np.ndim(ells[0]) == 0:
            assert len(ells) == 3, msg
            ells = [ells]
        ells = list(ells)
    return ells


def _format_los(los, ndim=3):
    vlos, swap = None, False
    if isinstance(los, str) and los in ['local']:
        pass
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    return los, vlos


def compute_mesh_bspec(*meshs: RealMeshField | ComplexMeshField, bin: bin_bspec=None,
                        ells: int | tuple=0, los: str | np.ndarray='x'):

    meshs, same = _format_meshs(*meshs)
    rdtype = meshs[0].real.dtype
    mattrs = meshs[0].attrs
    norm = mattrs.meshsize.prod(dtype=rdtype) / jnp.prod(mattrs.cellsize, dtype=rdtype)**2

    los, vlos = _format_los(los, ndim=mattrs.ndim)
    attrs = dict(los=vlos if vlos is not None else los)
    ells = _format_ells(ells, basis=bin.basis)

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    # The real-space grid
    xvec = mattrs.rcoords(sparse=True)
    # The Fourier-space grid
    kvec = mattrs.kcoords(sparse=True)

    num, num_zero = [], []
    if 'scoccimarro' in bin.basis:

        if vlos is None:
            meshs = [_2c(mesh) for mesh in meshs[:2]] + [_2r(meshs[2])]

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += _2c(meshs[2] * jax.lax.switch(im, Ylm, *xvec)) * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            for ell3 in ells:
                Ylms = [get_real_Ylm(ell3, m) for m in range(-ell3, ell3 + 1)]
                xs = np.arange(len(Ylms))
                tmp = tuple(meshs[i] for i in range(2)) + (jax.lax.scan(partial(f, Ylms), init=meshs[0].clone(value=jnp.zeros_like(meshs[0].value)), xs=xs)[0],)
                num.append((4. * np.pi) * bin(*tmp))
                num_zero.append(jnp.real(prod(map(_get_zero, meshs))) if ell3 == 0 else 0.)

        else:

            meshs = [_2c(mesh) for mesh in meshs]
            mu = sum(kk * ll for kk, ll in zip(kvec, vlos)) / jnp.sqrt(sum(kk**2 for kk in kvec)).at[(0,) * mattrs.ndim].set(1.)

            for ell3 in ells:
                tmp = meshs[:2] + [meshs[2] * get_legendre(ell3)(mu)]
                tmp = (2 * ell3 + 1) * bin(*tmp)
                num.append(tmp)
                num_zero.append(jnp.real(prod(map(_get_zero, meshs))) if ell3 == 0 else 0.)

    else:

        meshs = [_2c(mesh) for mesh in meshs[:2]] + [_2r(meshs[2])]

        @partial(jax.checkpoint, static_argnums=0)
        def f(Ylm, carry, im):
            tmp = tuple(meshs[i] * jax.lax.switch(im[i], Ylm[i], *kvec) for i in range(2))
            los = xvec if vlos is None else vlos
            tmp += (jax.lax.switch(im[2], Ylm[2], *los) * meshs[2],)
            carry += (4. * np.pi)**2 * im[3] * bin(*tmp)
            return carry, im

        for ell1, ell2, ell3 in ells:
            Ylms = [[get_real_Ylm(ell, m) for m in range(-ell, ell + 1)] for ell in (ell1, ell2, ell3)]
            if los != 'z':
                xs = [(im1, im2, im3, real_gaunt.get(((ell1, im1 - ell1), (ell2, im2 - ell2), (ell3, im3 - ell3)), 0.)) for im1, im2, im3 in itertools.product(*[np.arange(2 * ell + 1) for ell in (ell1, ell2, ell3)])]
            else:
                xs = [(im1, im2, im3, real_gaunt.get(((ell1, im1 - ell1), (ell2, im2 - ell2), (ell3, im3 - ell3)), 0.)) for im1, im2, im3 in itertools.product(*[np.arange(2 * ell + 1) for ell in (ell1, ell2)] + [[ell3]])]
            xs = [jnp.array(xx) for xx in zip(*[xx for xx in xs if xx[-1]])]
            num.append(jax.lax.scan(partial(f, Ylms), init=jnp.zeros(len(bin.edges), dtype=mattrs.dtype), xs=xs)[0])
            num_zero.append(jnp.real(prod(map(_get_zero, meshs[:2])) * meshs[2].sum()) if (ell1, ell2, ell3) == (0, 0, 0) else 0.)

    return BispectrumMultipoles(k=bin.xavg, num=num, nmodes=bin.nmodes, edges=bin.edges, ells=ells, norm=norm, num_zero=num_zero, attrs=attrs, basis=bin.basis)


from .pspec import compute_normalization


def compute_fkp_bspec_normalization(*fkps, cellsize=10.):
    fkps, same = _format_meshs(*fkps)
    alpha = prod(map(lambda fkp: fkp.data.sum() / fkp.randoms.sum(), fkps))
    norm = alpha * compute_normalization(*[fkp.randoms for fkp in fkps], cellsize=cellsize)
    return norm


def compute_fkp_bspec_shotnoise(*fkps, bin=None, ells=None, los: str | np.ndarray='x', **kwargs):
    fkps, same = _format_meshs(*fkps)
    ells = _format_ells(ells, basis=bin.basis)
    shotnoise = [jnp.ones_like(bin.xavg[..., 0]) for ill in range(len(ells))]

    def bin_pspec(mesh):
        return _bincount(bin.ibin, mesh.value, weights=bin.wmodes, length=len(bin.xavg)) / bin.nmodes1d

    if same[2] == same[1] + 1 == same[0] + 2:
        return tuple(shotnoise)

    particles = []
    for fkp, s in zip(fkp, same):
        if s < len(particles): particles.append(particles[s])
        else: particles.append(fkp.data - fkp.data.sum() / fkp.randoms.sum() * fkp.randoms)


    mattrs = particles.attrs
    los, vlos = _format_los(los, ndim=mattrs.ndim)
    # The real-space grid
    xvec = mattrs.rcoords(sparse=True)
    # The Fourier-space grid
    kvec = mattrs.kcoords(sparse=True)

    if 'scoccimaro' in bin.basis:

        # Eq. 56 of https://arxiv.org/pdf/1506.02729, 1 => 3
        raise NotImplementedError('Scoccimarro bispectrum shot noise not implemented yet')

    else:
        # Eq. 45 - 46 of https://arxiv.org/pdf/1803.02132

        def compute_S111(particles, ells):
            if ells == [(0, 0)]:
                return [jnp.sum(particles[0].weights**3) / jnp.sqrt(4. * jnp.pi)]
            rmesh = particles[0].clone(weights=particles[0].weights**3).paint(**kwargs, out='real')
            s111 = [jnp.sum(rmesh.value * get_real_Ylm(ell, m)) for ell, m in ells]
            return s111

        def compute_S122(particles, ells):  # 1 == 2
            rmesh = particles[1].clone(weights=particles[1].weights**2).paint(**kwargs, out='real')

            @partial(jax.checkpoint, static_argnums=0)
            def f(Ylm, carry, im):
                carry += (rmesh * jax.lax.switch(im, Ylm, *xvec)).r2c().conj() * jax.lax.switch(im, Ylm, *kvec)
                return carry, im

            cmesh = particles[0].paint(**kwargs, out='complex')

            for ell in ells:
                Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
                s122.append(4. * jnp.pi * bin_pspec(jax.lax.scan(partial(f, Ylms), init=cmesh.clone(value=jnp.zeros_like(cmesh.value)), xs=xs)[0] * cmesh) - s111 * (ell == 0))
            return s122

        def compute_S113(particles, ells):

            ells3 = sorted(list(set(ell[2] for ell in ells)))
            ells3 = [(ell, m) for m in range(-ell, ell + 1) for ell in ells3]
            s111 = compute_S111(particles, ells3)

            rmesh = particles[2].paint(**kwargs, out='real')
            cmesh = particles[0].clone(weights=particles[0].weights**2).paint(**kwargs, out='complex')

            @partial(jax.checkpoint, static_argnums=(0, 1))
            def f(Ylm, jl, carry, im):
                los = xvec if vlos is None else vlos
                tmp = im[3] * (jax.lax.switch(im[2], Ylm, *los) * rmesh).r2c() * cmesh - im[4].c2r()
                xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
                tmp *= jax.lax.switch(im[0], Ylm, *xvec) * jax.lax.switch(im[1], Ylm, *xvec)

                def fk(k):
                    return tmp * jl[0](xnorm * k[0]) * jl[1](xnorm * [1])

                carry += (4. * np.pi)**2 * jax.lax.map(fk, bin.xavg)
                return carry, im

            s113 = []
            for ell1, ell2, ell3 in ells:
                Ylms = [get_real_Ylm(ell, m) for m in range(-ell, ell + 1)]
                if los != 'z':
                    xs = [(im1, im2, im3, real_gaunt.get(((ell1, im1 - ell1), (ell2, im2 - ell2), (ell3, im3 - ell3)), 0.), s111[ells3.index((ell3, m3))]) for im1, im2, im3 in itertools.product(*[np.arange(2 * ell + 1) for ell in (ell1, ell2, ell3)])]
                else:
                    xs = [(im1, im2, im3, real_gaunt.get(((ell1, im1 - ell1), (ell2, im2 - ell2), (ell3, im3 - ell3)), 0.), s111[ells3.index((ell3, m3))]) for im1, im2, im3 in itertools.product(*[np.arange(2 * ell + 1) for ell in (ell1, ell2)] + [[ell3]])]
                xs = [jnp.array(xx) for xx in zip(*[xx for xx in xs if xx[-1]])]
                sign = (-1)**((ell1 + ell2) // 2)
                s113.append(sign * jax.lax.scan(partial(f, Ylms, [get_spherical_jn(ell1), get_spherical_jn(ell2)]), init=A0.clone(value=jnp.zeros_like(A0.value)), xs=xs)[0])

            return s113

        s111 = 0.
        if same[0] == same[1] == same[2]:
            s111 = jnp.sqrt(4. * jnp.pi) * compute_S111(particles, [(0, 0)])[0]
            shotnoise[ells.index((0, 0, 0))] += s111

        if same[1] == same[2]:
            ells1 = [ell[0] for ell in ells if ell[2] == ell[0] and ell[1] == 0]
            particles01 = particles
            s122 = compute_S122(particles01, ells1)

            for ill, ell in enumerate(ells):
                if ell[0] in ells1:
                    idx = ells1.index(ell[0])
                    shotnoise[ill] += s122[idx][bin._iedges[0]]

        if same[0] == same[2]:
            ells2 = [ell[1] for ell in ells if ell[2] == ell[1] and ell[0] == 0]
            particles01 = [particles[1], particles[0]]
            if same[0] == same[1]:  # re-use what we can from the previous calculation
                ells2 = [ell for ell in ells2 if ell not in ells1]
                if ells2:
                    s121 = s122 + compute_S122(particles01, ells2)
                    ells2 = ells1 + ells2
                else:
                    s121 = s122
                    ells2 = ells1
            else:
                s121 = compute_S122(particles01, ells2)
            for ill, ell in enumerate(ells):
                if ell[1] in ells2:
                    idx = ells2.index(ell[1])
                    shotnoise[ill] += s121[idx][bin._iedges[1]]

        if same[0] == same[1]:
            s113 = compute_S113(particles, ells)
            for ill, ell in enumerate(ells):
                shotnoise[ill] += s113[ill]

    return tuple(shotnoise)
