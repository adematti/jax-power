import os
import logging

import numpy as np
import jax
from jax import numpy as jnp

from .utils import WindowMatrix, CovarianceMatrix, BinnedStatistic, mkdir, plotter


class BaseWindowRotation(object):

    logger = logging.getLogger('WindowRotation')
    _meta_fields = ['observable', 'theory', 'xohalf', 'wmatrix', 'covmatrix', 'mmatrix', 'state', 'attrs']

    def __init__(self, wmatrix, covmatrix, attrs=None, **kwargs):
        self.set_wmatrix(wmatrix, **kwargs)
        self.set_covmatrix(covmatrix)
        self.attrs = dict(attrs or {})
        self.clear()
        self.init = self.loss = None

    def set_wmatrix(self, wmatrix, xpivot=None):
        self.observable, self.theory = wmatrix.observable, wmatrix.theory
        self.wmatrix = jnp.array(wmatrix.view())

        wm00 = self.wmatrix[np.ix_(self.mask_oprojs[0], self.mask_tprojs[0])]
        if xpivot is not None:
            idx = np.argmin(np.abs(self.ox[0] - xpivot))
        else:
            idx = len(self.ox[0]) // 2
        wm00 = wm00[idx]
        height = np.max(wm00)  # peak height
        xmax = self.tx[0][np.argmin(np.abs(wm00 - height))]  # k at maximum
        # Measuring bandwidth
        masks = (self.tx[0] < xmax, self.tx[0] > xmax)
        xx = tuple(self.tx[0][mask][np.argmin(np.abs(wm00[mask] - height / 2.))] for mask in masks)
        self.xbandwidth = np.abs(xx[1] - xx[0]) #/ (2. * np.sqrt(2. * np.log(2)))
        wm00 = self.wmatrix[np.ix_(self.mask_oprojs[0], self.mask_tprojs[0])]
        tdx = self.theory.edges(projs=self.tprojs[0])
        tdx = tdx[..., 1] - tdx[..., 0]
        mask_half = (wm00 / tdx) / np.max(wm00 / tdx, axis=1)[:, None] > 0.5
        self.xohalf = np.sum(wm00 * mask_half * self.tx[0], axis=1) / np.sum(wm00 * mask_half, axis=1)
        self.xohalf = [self.xohalf] * len(self.oprojs)

    @property
    def oprojs(self):
        return self.observable.projs

    @property
    def tprojs(self):
        return self.theory.projs

    @property
    def ox(self):
        return self.observable.x()

    @property
    def tx(self):
        return self.theory.x()

    @property
    def mask_oprojs(self):
        observable = self.observable
        mask = []
        for proj in observable.projs:
            tmp = np.zeros(observable.size, dtype='?')
            tmp[observable._index(projs=proj, concatenate=True)] = True
            mask.append(tmp)
        return mask

    @property
    def mask_tprojs(self):
        observable = self.theory
        mask = []
        for proj in observable.projs:
            tmp = np.zeros(observable.size, dtype='?')
            tmp[observable._index(projs=proj, concatenate=True)] = True
            mask.append(tmp)
        return mask

    def set_covmatrix(self, covmatrix):
        if isinstance(covmatrix, CovarianceMatrix):
            covmatrix = covmatrix.view()
        self.covmatrix = jnp.array(covmatrix)

    def clear(self):
        self.mmatrix = np.eye(self.covmatrix.shape[0])  # default M matrix

    def setup(self):
        raise NotImplementedError

    def rotate(self):
        raise NotImplementedError

    def fit(self, state=None, **kwargs):
        """Fit."""
        import optax
        if getattr(self, 'loss', None) is None or getattr(self, 'init', None) is None:
            raise ValueError('call "setup" to set loss function and init')

        def fit(theta, loss, init_learning_rate=1e-5, meta_learning_rate=1e-4, nsteps=100000, state=None, meta_state=None):

            self.logger.info(f'Will do {nsteps} steps')
            optimizer = optax.inject_hyperparams(optax.adabelief)(learning_rate=init_learning_rate)
            meta_opt = optax.adam(learning_rate=meta_learning_rate)

            @jax.jit
            def step(theta, state):
                grads = jax.grad(loss)(theta)
                updates, state = optimizer.update(grads, state)
                theta = optax.apply_updates(theta, updates)
                return theta, state

            @jax.jit
            def outer_loss(eta, theta, state):
                # Apparently this is what inject_hyperparams allows us to do
                state.hyperparams['learning_rate'] = jnp.exp(eta)
                theta, state = step(theta, state)
                return loss(theta), (theta, state)

            # Only this jit actually matters
            @jax.jit
            def outer_step(eta, theta, meta_state, state):
                #has_aux says we're going to return the 2nd part, extra info
                grad, (theta, state) = jax.grad(outer_loss, has_aux=True)(eta, theta, state)
                meta_updates, meta_state = meta_opt.update(grad, meta_state)
                eta = optax.apply_updates(eta, meta_updates)
                return eta, theta, meta_state, state

            if state is None: state = optimizer.init(theta)
            eta = jnp.log(init_learning_rate)
            if meta_state is None: meta_state = meta_opt.init(eta)
            printstep = max(nsteps // 20, 1)
            self.logger.info(f'Initial loss: {loss(theta):.3g}')
            for i in range(nsteps):
                eta, theta, meta_state, state = outer_step(eta, theta, meta_state, state)
                if i < 2 or nsteps - i < 4 or i % printstep == 0:
                    self.logger.info(f'step {i}, loss: {loss(theta):.3g}, lr: {jnp.exp(eta):.3g}')
            return theta, (jnp.exp(eta), meta_state, state)

        if state is None:
            self.mmatrix, self.state = fit(self.init, self.loss, **kwargs)
        else:
            self.mmatrix, self.state = fit(self.init, self.loss, init_learning_rate=state[0], state=state[2], meta_state=state[1], **kwargs)
        return self.mmatrix, self.state

    @plotter
    def plot_wmatrix_slice(self, indices):
        wmatrix = WindowMatrix(observable=self.observable, theory=self.theory, value=self.wmatrix)
        wmatrix_rotated = WindowMatrix(observable=self.observable, theory=self.theory, value=self.rotate()[0])
        fig = wmatrix.plot_slice(indices, color='C0', label='$W$', yscale='log')
        return wmatrix_rotated.plot_slice(indices, color='C1', label=r'$W^{\prime}$', yscale='log', fig=fig)

    @plotter
    def plot_compactness(self, frac=0.95, xlim=None, projs=None):
        from matplotlib import pyplot as plt
        if projs is None:
            projs = self.oprojs
            projs = [projs[:i + 1] for i in range(len(projs))]

        wmatrix_rotated = self.rotate()[0]
        alphas = np.linspace(0.4, 1., len(projs))
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), squeeze=True)

        def compactness(wm, projs, frac):
            weights_bf = sum(np.cumsum(np.abs(wm[np.ix_(self.mask_oprojs[self.oprojs.index(oproj)], self.mask_tprojs[self.tprojs.index(tproj)])]), axis=-1) for tproj in projs for oproj in projs)
            weights_tot = weights_bf[:, -1]
            itxmax = np.argmax(weights_bf / weights_tot[:, None] >= frac, axis=-1)
            return self.tx[0][itxmax]

        for iproj, projs in enumerate(projs):
            label = str(projs)
            if self.observable._label_proj:
                label = '{} = {}'.format(self.observable._label_proj, label)
            ax.plot(self.ox[0], compactness(self.wmatrix, projs=projs, frac=frac), color='C0', alpha=alphas[iproj], label=label)
            ax.plot(self.ox[0], compactness(wmatrix_rotated, projs=projs, frac=frac), color='C1', alpha=alphas[iproj])

        for xx in (xlim or []): ax.axvline(xx, ls=':', color='k')

        ax.set_xlabel(self.observable._label_x)
        ax.set_ylabel(self.theory._label_x)
        ax.legend()
        return fig

    def __getstate__(self):
        state = {name: getattr(self, name) for name in self._meta_fields if hasattr(self, name)}
        for name in ['observable', 'theory']:
            state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        for name in ['theory', 'observable']:
            state[name] = BinnedStatistic.from_state(state[name])
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        """Save object."""
        state = self.__getstate__()
        #self.log_info('Saving {}.'.format(filename))
        mkdir(os.path.dirname(filename))
        np.save(filename, state, allow_pickle=True)

    @classmethod
    def load(cls, filename):
        """Load object."""
        filename = str(filename)
        state = np.load(filename, allow_pickle=True)
        #cls.log_info('Loading {}.'.format(filename))
        state = state[()]
        return cls.from_state(state)


class WindowRotationSpectrum2(BaseWindowRotation):

    _meta_fields = BaseWindowRotation._meta_fields + ['marg_precmatrix', 'marg_prior_mo', 'marg_theory_offset']

    def setup(self, Mtarg=None, Minit='momt', max_sigma_W=5, max_sigma_R=5, factor_diff_proj=10):
        tx = np.concatenate(list(self.tx))
        ox = np.concatenate(list(self.xohalf))
        if Mtarg is None:
            Mtarg = jnp.eye(len(ox))

        if Minit in [None, 'momt']:
            with_momt = Minit == 'momt'
            Minit = jnp.identity(len(ox))
            if with_momt:
                mo, mt = [], []
                txcut = self.tx[0][len(self.tx[0]) // 2]
                oidx = np.argmin(np.abs(self.ox[0] - txcut)) // 2
                #txcut, oidx = 0.20, 20
                for omask, oproj in zip(self.mask_oprojs, self.oprojs):
                    trow = self.wmatrix[omask, :][oidx, :]
                    mt.append(trow * (tx >= txcut))  # choose vector that captures non-diagonal behavior
                    tmp = jnp.zeros(len(omask))
                    if oproj in self.tprojs:
                        tmask = self.mask_tprojs[self.tprojs.index(oproj)]
                        if trow[tmask][-1] != 0 and oproj in self.oprojs:
                            omask = self.mask_oprojs[self.oprojs.index(oproj)]
                            tmp = tmp.at[omask].set(self.wmatrix[np.ix_(omask, tmask)][..., -1] / trow[tmask][-1])
                    mo.append(tmp)
                Minit = (Minit, mo, mt)
        with_momt = isinstance(Minit, tuple)

        weights_wmatrix = np.empty_like(self.wmatrix)
        weights_covmatrix = np.empty_like(self.covmatrix)
        if weights_covmatrix.shape[0] != weights_wmatrix.shape[0]:
            raise ValueError(f'shapes of covariance and window matrix must match; found {weights_covmatrix.shape} and {weights_wmatrix.shape}')
        for omask, oproj in zip(self.mask_oprojs, self.oprojs):
            weights_wmatrix[omask, :] = np.minimum(((tx - ox[omask, None]) / self.xbandwidth)**2, max_sigma_W**2)
            mask = np.ones_like(weights_wmatrix[omask, :])
            if oproj in self.tprojs:
                mask[..., self.mask_tprojs[self.tprojs.index(oproj)]] = False
            weights_wmatrix[omask, :] += factor_diff_proj * mask  # off-diagonal blocks
            weights_covmatrix[omask, :] = np.minimum(((ox - ox[omask, None]) / self.xbandwidth)**2, max_sigma_R**2)
            weights_covmatrix[omask, :] += factor_diff_proj * (~omask[None, ...])  # off-diagonal blocks

        def softabs(x):
            return jnp.sqrt(x**2 + 1e-37)

        def RfromC(C):
            sig = jnp.sqrt(jnp.diag(C))
            denom = jnp.outer(sig, sig)
            return C / denom

        def loss(mmatrix):
            Wp, Cp = self.rotate(mmatrix=mmatrix, prior_cov=False)
            if with_momt: mmatrix = mmatrix[0]
            loss_W = jnp.sum(softabs(Wp * weights_wmatrix)) / jnp.sum(softabs(Wp) * (weights_wmatrix > 0))
            #loss_W = jnp.sum(softabs(Wp * weights_wmatrix)) / jnp.sum(softabs(Wp) * weights_wmatrix_denom)
            Rp = RfromC(Cp)
            loss_C = jnp.sum(softabs(Rp * weights_covmatrix)) / jnp.sum(softabs(Rp) * (weights_covmatrix > 0))
            #loss_C = jnp.sum(softabs(Rp * weights_covmatrix)) / jnp.sum(softabs(Rp) * weights_covmatrix_denom)
            loss_M = 10 * jnp.sum((jnp.sum(mmatrix, axis=1) - 1.)**2)
            #print(loss_W, loss_C, weights_wmatrix.sum(), weights_covmatrix.sum(), weights_wmatrix.shape, weights_covmatrix.shape)
            #print(loss_W, loss_C, loss_M)
            return loss_W + loss_C + loss_M

        self.init = Minit
        self.loss = loss

    @property
    def with_momt(self):
        return isinstance(self.mmatrix, tuple)

    def rotate(self, mmatrix=None, covmatrix=None, data=None, mask_cov=None, prior_data=True, prior_cov=True):
        """Return prior and precmatrix if input theory."""
        if mmatrix is None: mmatrix = self.mmatrix
        input_covmatrix = covmatrix is not None
        if not input_covmatrix: covmatrix = self.covmatrix
        with_momt = isinstance(mmatrix, tuple)
        if with_momt:
            Wsub = jnp.zeros(self.wmatrix.shape)
            mmatrix, mo, mt = mmatrix
            Csub = 0
            for mmo, mmt, omask in zip(mo, mt, self.mask_oprojs):
                mask_mo = omask * mmo
                Wsub += jnp.outer(mask_mo, mmt)
            mo = jnp.array(mo)
        else:
            Wsub = Csub = 0.

        #print('WC', Wsub.sum(), Csub.sum())
        wmatrix_rotated = jnp.matmul(mmatrix, self.wmatrix) - Wsub

        if mask_cov is not None:
            tmpmmatrix = np.eye(covmatrix.shape[0])
            tmpmmatrix[np.ix_(mask_cov, mask_cov)] = mmatrix
            mmatrix = tmpmmatrix
        covmatrix_rotated = mmatrix.dot(covmatrix).dot(mmatrix.T) - Csub
        if with_momt and prior_cov:
            covmatrix_rotated += mo.T.dot(jnp.diag(self.marg_prior_mo)).dot(mo)

        if data is not None:
            data = np.asarray(data).real.ravel()
            #data_rotated = np.matmul(mmatrix, data + shotnoise * self.mask_ellsout[0]) - shotnoise * self.mask_ellsout[0]
            data_rotated = np.matmul(mmatrix, data)
            if with_momt and prior_data:
                data_rotated -= self.marg_prior_mo.dot(mo)
            return wmatrix_rotated, covmatrix_rotated, data_rotated

        return wmatrix_rotated, covmatrix_rotated

    def set_prior(self, data, theory, covmatrix=None, xlim=None):
        if not self.with_momt:
            self.logger.info(f'I did not use momt parameters -- no prior set')
        wmatrix_rotated, covmatrix_rotated, data_rotated = self.rotate(covmatrix=covmatrix, data=data, prior_data=False, prior_cov=False)
        mmatrix, mo, mt = self.mmatrix
        if xlim is not None:
            mask_xout = self.observable._index(xlim=xlim, concatenate=True)
            wmatrix_rotated = wmatrix_rotated[mask_xout, :]
            covmatrix_rotated = covmatrix_rotated[np.ix_(mask_xout, mask_xout)]
            data_rotated = data_rotated[mask_xout]
            if self.with_momt: mo = [mmo[mask_xout] for mmo in mo]
        theory = np.asarray(theory).real.ravel()
        precmatrix = np.linalg.inv(covmatrix_rotated)
        deriv = np.array(mo)
        derivp = deriv.dot(precmatrix)
        fisher = derivp.dot(deriv.T)
        self.marg_prior_mo = np.linalg.solve(fisher, derivp.dot(data_rotated - np.matmul(wmatrix_rotated, theory)))

    @plotter
    def plot_rotated(self, data, shotnoise=0., xlim=None):
        from matplotlib import pyplot as plt
        oprojs = self.oprojs
        data_rotated = self.rotate(data=data, shotnoise=shotnoise)[2]
        fig, lax = plt.subplots(1, len(oprojs), figsize=(8, 3), sharey=False, squeeze=False)
        lax = lax.ravel()

        for iproj, proj in enumerate(oprojs):
            ax = lax[iproj]
            ax.plot(self.ox[iproj], self.ox[iproj] * data[self.mask_oprojs[iproj]], color='C0', label=r'$P_{\mathrm{o}}(k)$')
            ax.plot(self.ox[iproj], self.ox[iproj] * data_rotated[self.mask_oprojs[iproj]], color='C1', label=r'$P_{\mathrm{o}}^{\prime}(k)$')
            ax.set_title(r'$\ell = {}$'.format(proj))
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            if xlim is not None: ax.set_xlim(xlim)

        lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.legend()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        return fig

    @plotter
    def plot_validation(self, data, theory, xlim=None, covmatrix=None, shotnoise=0., marg_shotnoise=False, nobs=1):
        from matplotlib import pyplot as plt
        oprojs = self.oprojs

        fig, lax = plt.subplots(2, len(oprojs), figsize=(3 * len(oprojs), 4), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)
        rotate = self.rotate(data=data, theory=theory, covmatrix=covmatrix, xlim=xlim, shotnoise=shotnoise)
        precmatrix_rotated = None
        try:
            wmatrix_rotated, covmatrix_rotated, data_rotated = rotate
            offset = 0.
        except ValueError:
            wmatrix_rotated, covmatrix_rotated, data_rotated, m, offset, precmatrix_rotated = rotate

        #np.save('precm.npy', precmatrix_rotated)
        std = np.sqrt(np.diag(covmatrix_rotated))
        if precmatrix_rotated is not None:
            std = np.sqrt(np.diag(np.linalg.inv(precmatrix_rotated)))
        std /= np.sqrt(nobs)
        ox = np.concatenate(list(self.ox))
        if xlim is not None:
            mask_ox = (ox >= xlim[0]) & (ox <= xlim[-1])
        else:
            mask_ox = np.ones(len(ox), dtype='?')
        data = np.asarray(data).real.ravel()
        if xlim is not None:
            data = data[mask_ox]
        data_rotated -= offset
        theory_rotated = np.matmul(wmatrix_rotated, theory + shotnoise * self.mask_tprojs[self.tprojs.index(0)]) - shotnoise * self.mask_oprojs[self.oprojs.index(0)][mask_ox]
        #theory_rotated = np.matmul(wmatrix_rotated, theory)
        #theory_rotated = theory

        if marg_shotnoise:  # shotnoise free
            precmatrix = np.linalg.inv(covmatrix_rotated)
            deriv = np.matmul(wmatrix_rotated, self.mask_tprojs[self.tprojs.index(0)])[None, :]
            derivp = deriv.dot(precmatrix)
            fisher = derivp.dot(deriv.T)
            shotnoise_value = np.linalg.solve(fisher, derivp.dot(data_rotated - theory_rotated))
            theory_rotated_shotnoise = theory_rotated + shotnoise_value.dot(deriv)

        for iproj, proj in enumerate(oprojs):
            color = 'C{}'.format(iproj)
            ax = lax[0][iproj]
            xx = ox[self.mask_oprojs[iproj] & mask_ox]
            mask = self.mask_oprojs[iproj][mask_ox]
            ax.errorbar(xx, xx * data_rotated[mask], xx * std[mask], color=color, marker='.', ls='', label=r'$P_{\mathrm{o}}(k)$')
            ax.plot(xx, xx * np.interp(xx, self.tx[self.tprojs.index(proj)], theory[self.mask_tprojs[self.tprojs.index(proj)]]), color=color, ls=':', label=r'$P_{\mathrm{t}}(k)$')
            ax.plot(xx, xx * theory_rotated[mask], color=color, label=r'$W(k, k^{\prime}) P_{\mathrm{t}}(k^{\prime})$')
            if marg_shotnoise:
                ax.plot(xx, xx * theory_rotated_shotnoise[mask], color=color, ls='--', label=r'$W(k, k^{\prime}) (P_{\mathrm{t}}(k^{\prime}) + N)$')
            ax.set_title(r'$\ell = {}$'.format(proj))
            ax.set_xlim(xlim)
            ax.grid(True)
            ax = lax[1][iproj]
            ax.plot(xx, ((theory_rotated_shotnoise if marg_shotnoise else theory_rotated)[mask] - data_rotated[mask]) / std[mask], color=color)
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            ax.set_ylim(-2., 2.)
            ax.grid(True)

        lax[0][0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[0][0].legend()
        lax[1][0].set_ylabel(r'$\Delta P_{\ell}(k) / \sigma$')
        fig.align_ylabels()
        return fig