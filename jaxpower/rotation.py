import os
import logging

import numpy as np
import jax
from jax import numpy as jnp

from .types import WindowMatrix, CovarianceMatrix, ObservableTree
from .utils import mkdir, plotter


def _get_coords(observable):

    def _meshgrid(*args):
        m = jnp.meshgrid(*args, indexing='ij')
        return jnp.column_stack([mm.ravel() for mm in m])

    def _get_leaf_coords(leaf):
        coords = list(leaf.coords().values())
        if len(coords) == 1:
            return coords[0]
        else:
            if coords.ndim == 1:
                return _meshgrid(*coords)
            return jax.vmap(_meshgrid, in_axes=-1, out_axes=-1)(*coords)

    return [_get_leaf_coords(observable.get(**label)) for label in observable.labels()]


def _get_leaf_edges(leaf):
    edges = list(leaf.edges().values())

    def _meshgrid(*args):
        m = jnp.meshgrid(*args, indexing='ij')
        return jnp.column_stack([mm.ravel() for mm in m])

    if len(edges) == 1:
        return edges[0]
    else:
        if edges.ndim == 2:
            return jax.vmap(_meshgrid, in_axes=-1, out_axes=-1)(*edges)
        else:
            return jax.vmap(jax.vmap(_meshgrid, in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1)(*edges)


def _get_masks(observable):
    mask = []
    start = 0
    for label in observable.labels():
        tmp = np.zeros(observable.size, dtype='?')
        leaf = observable.get(**label)
        tmp[start:start + leaf.size] = True
        start += leaf.size
        mask.append(tmp)
    return mask


class BaseWindowRotation(object):
    """
    Base class for window matrix (:class:`WindowMatrix`) rotation: make it more diagonal in a likelihood-informed way.
    Routines are provided to correspondingly rotate the data vector (:class:`BinnedStatistic`) and covariance matrix (:class:`CovarianceMatrix`).

    Reference
    ---------
    https://arxiv.org/abs/2406.04804

    Parameters
    ----------
    window : WindowMatrix
        Window matrix object containing observable and theory information.
    covariance : CovarianceMatrix or array_like
        Covariance matrix for the observable.
    attrs : dict, optional
        Additional attributes.
    **kwargs
        Additional keyword arguments for window matrix setup.

    Attributes
    ----------
    observable : BinnedStatistic
        Observable object.
    theory : BinnedStatistic
        Theory object.
    window : ndarray
        Window matrix.
    covariance : ndarray
        Covariance matrix.
    mmatrix : ndarray
        Transformation matrix (default identity).
    xohalf : list
        Characteristic scale for each observable.
    xbandwidth : float
        Bandwidth of the window matrix.
    attrs : dict
        Additional attributes.
    state : object
        Fitting state.
    init : object
        Initial fit parameters.
    loss : callable
        Loss function for fitting.
    """

    logger = logging.getLogger('WindowRotation')
    _meta_fields = ['observable', 'theory', 'xohalf', 'window', 'covariance', 'mmatrix', 'state', 'attrs']

    def __init__(self, window: WindowMatrix, covariance: CovarianceMatrix, attrs: dict=None, **kwargs):
        """
        Initialize the rotation.

        Parameters
        ----------
        window : WindowMatrix
            Window matrix object.
        covariance : CovarianceMatrix or array-like
            Covariance matrix.
        attrs : dict, optional
            Additional attributes.
        **kwargs
            Additional keyword arguments for window matrix setup.
        """
        self.set_window(window, **kwargs)
        self.set_covariance(covariance)
        self.attrs = dict(attrs or {})
        self.clear()
        self.init = self.loss = None

    def set_window(self, window: WindowMatrix, xpivot: float=None):
        """
        Set the window matrix and compute characteristic scales.

        Parameters
        ----------
        window : WindowMatrix
            Window matrix object.
        xpivot : float, optional
            Pivot value for characteristic scale calculation.
            Default to the middle of the observed data vector.
        """
        self.observable, self.theory = window.observable, window.theory
        self.window = jnp.array(window.value())

        wm00 = self.window[np.ix_(self.mask_olabels[0], self.mask_tlabels[0])]
        if xpivot is not None:
            idx = np.argmin(np.abs(self.ocoords[0] - xpivot))
        else:
            idx = len(self.ocoords[0]) // 2
        wm00 = wm00[idx]
        height = np.max(wm00)  # peak height
        xmax = self.tcoords[0][np.argmin(np.abs(wm00 - height))]  # k at maximum
        # Measuring bandwidth
        masks = (self.tcoords[0] < xmax, self.tcoords[0] > xmax)
        xx = tuple(self.tcoords[0][mask][np.argmin(np.abs(wm00[mask] - height / 2.))] for mask in masks)
        self.xbandwidth = np.abs(xx[1] - xx[0]) #/ (2. * np.sqrt(2. * np.log(2)))
        wm00 = self.window[np.ix_(self.mask_olabels[0], self.mask_tlabels[0])]
        tdx = _get_leaf_edges(self.theory.get(**next(iter(self.theory.labels()))))
        tdx = tdx[..., 1] - tdx[..., 0]
        mask_half = (wm00 / tdx) / np.max(wm00 / tdx, axis=1)[:, None] > 0.5
        self.xohalf = np.sum(wm00 * mask_half * self.tcoords[0], axis=1) / np.sum(wm00 * mask_half, axis=1)
        self.xohalf = [self.xohalf] * len(self.olabels)

    @property
    def olabels(self):
        """Return list of observable projections."""
        return self.observable.labels(only='values')

    @property
    def tlabels(self):
        """Return list of theory projections."""
        return self.theory.labels(only='values')

    @property
    def ocoords(self):
        """Return list of observable coordinates."""
        return _get_coords(self.observable)

    @property
    def tcoords(self):
        """Return list of theory coordinates."""
        return _get_coords(self.theory)

    @property
    def mask_olabels(self):
        return _get_masks(self.observable)

    @property
    def mask_tlabels(self):
        return _get_masks(self.theory)

    def set_covariance(self, covariance: CovarianceMatrix | jax.Array):
        """
        Set the covariance matrix.

        Parameters
        ----------
        covariance : CovarianceMatrix or array-like
            Covariance matrix to set.
        """
        if isinstance(covariance, CovarianceMatrix):
            covariance = covariance.value()
        self.covariance = jnp.array(covariance)

    def clear(self):
        """Reset the transformation matrix to identity."""
        self.mmatrix = np.eye(self.covariance.shape[0])  # default M matrix

    def setup(self):
        """
        Abstract method to set up loss function and initial parameters for fitting.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        raise NotImplementedError

    def rotate(self):
        """
        Abstract method to perform window matrix rotation.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        raise NotImplementedError

    def fit(self, state=None, **kwargs):
        """
        Fit the transformation matrix using gradient-based optimization.

        Parameters
        ----------
        state : tuple, optional
            Previous fitting state.
        **kwargs
            Additional arguments for optimizer.

        Returns
        -------
        mmatrix : jax.Array
            Fitted transformation matrix.
        state : tuple
            Final optimizer state.
        """
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
    def plot_window_slice(self, indices):
        """
        Plot a slice of the window matrix before and after rotation.

        Parameters
        ----------
        indices : array-like
            Indices for which to take slices.
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
        fig : matplotlib.figure.Figure
            Figure object.
        """
        window = WindowMatrix(observable=self.observable, theory=self.theory, value=self.window)
        window_rotated = WindowMatrix(observable=self.observable, theory=self.theory, value=self.rotate()[0])
        fig = window.plot_slice(indices, color='C0', label='$W$', yscale='log')
        return window_rotated.plot_slice(indices, color='C1', label=r'$W^{\prime}$', yscale='log', fig=fig)

    @plotter
    def plot_compactness(self, frac=0.95, xlim=None, labels=None):
        """
        Plot the compactness of the window matrix for different projections.

        Reference
        ---------
        Fig. 14 of https://arxiv.org/pdf/2406.04804

        Parameters
        ----------
        frac : float, default=0.95
            At each observed point, plot the theory coordinate for which the
            fraction of the sum of the window matrix along the theory is ``frac``.
        xlim : tuple, optional
            x-axis limits.
        labels : list of dict, optional
            Labels to plot.
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
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        if labels is None:
            labels = self.olabels
            labels = [labels[:i + 1] for i in range(len(labels))]

        window_rotated = self.rotate()[0]
        alphas = np.linspace(0.4, 1., len(labels))
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), squeeze=True)

        def compactness(wm, labels, frac):
            weights_bf = sum(np.cumsum(np.abs(wm[np.ix_(self.mask_olabels[self.olabels.index(olabel)], self.mask_tlabels[self.tlabels.index(tlabel)])]), axis=-1) for tlabel in labels for olabel in labels)
            weights_tot = weights_bf[:, -1]
            itxmax = np.argmax(weights_bf / weights_tot[:, None] >= frac, axis=-1)
            return self.tcoords[0][itxmax]

        for ilabel, labels in enumerate(labels):
            label = str(labels)
            ax.plot(self.ocoords[0], compactness(self.window, labels=labels, frac=frac), color='C0', alpha=alphas[ilabel], label=label)
            ax.plot(self.ocoords[0], compactness(window_rotated, labels=labels, frac=frac), color='C1', alpha=alphas[ilabel])

        for xx in (xlim or []): ax.axvline(xx, ls=':', color='k')

        ax.legend()
        return fig

    def __getstate__(self):
        state = {name: getattr(self, name) for name in self._meta_fields if hasattr(self, name)}
        for name in ['observable', 'theory']:
            state[name] = getattr(self, name).__getstate__()
        return state

    def __setstate__(self, state):
        for name in ['theory', 'observable']:
            state[name] = ObservableTree.from_state(state[name])
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
    """
    Window rotation class for 2-point spectrum analysis.

    Extends BaseWindowRotation to support marginalization and advanced setup for spectrum rotation.

    Attributes
    ----------
    marg_precision : ndarray
        Marginalized precision matrix.
    marg_prior_mo : ndarray
        Marginalized prior offset.
    marg_theory_offset : ndarray
        Marginalized theory offset.
    """
    _meta_fields = BaseWindowRotation._meta_fields + ['marg_precision', 'marg_prior_mo', 'marg_theory_offset']

    def setup(self, Mtarg=None, Minit='momt', max_sigma_W=5, max_sigma_R=5, factor_diff_label=10):
        """
        Set up the loss function and initial parameters for rotation fitting.

        Parameters
        ----------
        Mtarg : ndarray, optional
            Target transformation matrix.
        Minit : str or tuple, optional
            Initialization method or tuple of initial values.
        max_sigma_W : float, optional
            Maximum sigma for window matrix weighting.
        max_sigma_R : float, optional
            Maximum sigma for covariance matrix weighting.
        factor_diff_label : float, optional
            Weighting factor for off-diagonal blocks.
        """
        tcoords = np.concatenate(list(self.tcoords))
        ocoords = np.concatenate(list(self.xohalf))
        if Mtarg is None:
            Mtarg = jnp.eye(len(ocoords))

        if Minit in [None, 'momt']:
            with_momt = Minit == 'momt'
            Minit = jnp.identity(len(ocoords))
            if with_momt:
                mo, mt = [], []
                txcut = self.tcoords[0][len(self.tcoords[0]) // 2]
                oidx = np.argmin(np.abs(self.ocoords[0] - txcut)) // 2
                #txcut, oidx = 0.20, 20
                for omask, olabel in zip(self.mask_olabels, self.olabels):
                    trow = self.window[omask, :][oidx, :]
                    mt.append(trow * (tcoords >= txcut))  # choose vector that captures non-diagonal behavior
                    tmp = jnp.zeros(len(omask))
                    if olabel in self.tlabels:
                        tmask = self.mask_tlabels[self.tlabels.index(olabel)]
                        if trow[tmask][-1] != 0 and olabel in self.olabels:
                            omask = self.mask_olabels[self.olabels.index(olabel)]
                            tmp = tmp.at[omask].set(self.window[np.ix_(omask, tmask)][..., -1] / trow[tmask][-1])
                    mo.append(tmp)
                Minit = (Minit, mo, mt)
        with_momt = isinstance(Minit, tuple)

        weights_window = np.empty_like(self.window)
        weights_covariance = np.empty_like(self.covariance)
        if weights_covariance.shape[0] != weights_window.shape[0]:
            raise ValueError(f'shapes of covariance and window matrix must match; found {weights_covariance.shape} and {weights_window.shape}')
        for omask, olabel in zip(self.mask_olabels, self.olabels):
            weights_window[omask, :] = np.minimum(((tcoords - ocoords[omask, None]) / self.xbandwidth)**2, max_sigma_W**2)
            mask = np.ones_like(weights_window[omask, :])
            if olabel in self.tlabels:
                mask[..., self.mask_tlabels[self.tlabels.index(olabel)]] = False
            weights_window[omask, :] += factor_diff_label * mask  # off-diagonal blocks
            weights_covariance[omask, :] = np.minimum(((ocoords - ocoords[omask, None]) / self.xbandwidth)**2, max_sigma_R**2)
            weights_covariance[omask, :] += factor_diff_label * (~omask[None, ...])  # off-diagonal blocks

        def softabs(x):
            return jnp.sqrt(x**2 + 1e-37)

        def RfromC(C):
            sig = jnp.sqrt(jnp.diag(C))
            denom = jnp.outer(sig, sig)
            return C / denom

        def loss(mmatrix):
            Wp, Cp = self.rotate(mmatrix=mmatrix, prior_cov=False)
            if with_momt: mmatrix = mmatrix[0]
            loss_W = jnp.sum(softabs(Wp * weights_window)) / jnp.sum(softabs(Wp) * (weights_window > 0))
            #loss_W = jnp.sum(softabs(Wp * weights_window)) / jnp.sum(softabs(Wp) * weights_window_denom)
            Rp = RfromC(Cp)
            loss_C = jnp.sum(softabs(Rp * weights_covariance)) / jnp.sum(softabs(Rp) * (weights_covariance > 0))
            #loss_C = jnp.sum(softabs(Rp * weights_covariance)) / jnp.sum(softabs(Rp) * weights_covariance_denom)
            loss_M = 10 * jnp.sum((jnp.sum(mmatrix, axis=1) - 1.)**2)
            #print(loss_W, loss_C, weights_window.sum(), weights_covariance.sum(), weights_window.shape, weights_covariance.shape)
            #print(loss_W, loss_C, loss_M)
            return loss_W + loss_C + loss_M

        self.init = Minit
        self.loss = loss

    @property
    def with_momt(self):
        return isinstance(self.mmatrix, tuple)

    def rotate(self, mmatrix=None, covariance=None, data=None, mask_cov=None, prior_data=True, prior_cov=True):
        """
        Rotate the window and covariance matrices, optionally transforming data.

        Parameters
        ----------
        mmatrix : ndarray or tuple, optional
            Transformation matrix or tuple for marginalization.
        covariance : ndarray, optional
            Covariance matrix to rotate.
        data : array_like, optional
            Data vector to transform.
        mask_cov : array_like, optional
            Mask for covariance matrix.
        prior_data : bool, optional
            Whether to apply prior to data.
        prior_cov : bool, optional
            Whether to apply prior to covariance.

        Returns
        -------
        window_rotated : ndarray
            Rotated window matrix.
        covariance_rotated : ndarray
            Rotated covariance matrix.
        data_rotated : ndarray, optional
            Rotated data vector (if data is provided).
        """
        if mmatrix is None: mmatrix = self.mmatrix
        input_covariance = covariance is not None
        if not input_covariance: covariance = self.covariance
        with_momt = isinstance(mmatrix, tuple)
        if with_momt:
            Wsub = jnp.zeros(self.window.shape)
            mmatrix, mo, mt = mmatrix
            Csub = 0
            for mmo, mmt, omask in zip(mo, mt, self.mask_olabels):
                mask_mo = omask * mmo
                Wsub += jnp.outer(mask_mo, mmt)
            mo = jnp.array(mo)
        else:
            Wsub = Csub = 0.

        #print('WC', Wsub.sum(), Csub.sum())
        window_rotated = jnp.matmul(mmatrix, self.window) - Wsub

        if mask_cov is not None:
            tmpmmatrix = np.eye(covariance.shape[0])
            tmpmmatrix[np.ix_(mask_cov, mask_cov)] = mmatrix
            mmatrix = tmpmmatrix
        covariance_rotated = mmatrix.dot(covariance).dot(mmatrix.T) - Csub
        if with_momt and prior_cov:
            covariance_rotated += mo.T.dot(jnp.diag(self.marg_prior_mo)).dot(mo)

        if data is not None:
            data = np.asarray(data).real.ravel()
            #data_rotated = np.matmul(mmatrix, data + shotnoise * self.mask_ellsout[0]) - shotnoise * self.mask_ellsout[0]
            data_rotated = np.matmul(mmatrix, data)
            if with_momt and prior_data:
                data_rotated -= self.marg_prior_mo.dot(mo)
            return window_rotated, covariance_rotated, data_rotated

        return window_rotated, covariance_rotated

    def set_prior(self, data, theory, covariance=None, xlim=None):
        r"""
        Set the prior for window rotation parameters :math:`s`

        Reference
        ---------
        Eq. 5.4 of https://arxiv.org/pdf/2406.04804

        Parameters
        ----------
        data : array-like
            Data vector.
        theory : array-like
            Theory vector.
        covariance : ndarray, optional
            Covariance matrix.
        xlim : tuple, optional
            X-axis limits.
        """
        if not self.with_momt:
            self.logger.info(f'I did not use momt parameters -- no prior set')
        window_rotated, covariance_rotated, data_rotated = self.rotate(covariance=covariance, data=data, prior_data=False, prior_cov=False)
        mmatrix, mo, mt = self.mmatrix
        if xlim is not None:
            mask_xout = self.observable._index(xlim=xlim, concatenate=True)
            window_rotated = window_rotated[mask_xout, :]
            covariance_rotated = covariance_rotated[np.ix_(mask_xout, mask_xout)]
            data_rotated = data_rotated[mask_xout]
            if self.with_momt: mo = [mmo[mask_xout] for mmo in mo]
        theory = np.asarray(theory).real.ravel()
        precmatrix = np.linalg.inv(covariance_rotated)
        deriv = np.array(mo)
        derivp = deriv.dot(precmatrix)
        fisher = derivp.dot(deriv.T)
        self.marg_prior_mo = np.linalg.solve(fisher, derivp.dot(data_rotated - np.matmul(window_rotated, theory)))

    @plotter
    def plot_rotated(self, data, shotnoise=0., xlim=None):
        """
        Plot the rotated data and original data for each projection.

        Parameters
        ----------
        data : array-like
            Data vector.
        shotnoise : float, optional
            Shot noise value.
        xlim : tuple, optional
            X-axis limits.
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
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        oells = self.olabels
        data_rotated = self.rotate(data=data, shotnoise=shotnoise)[2]
        fig, lax = plt.subplots(1, len(oells), figsize=(8, 3), sharey=False, squeeze=False)
        lax = lax.ravel()

        for ill, ell in enumerate(oells):
            ax = lax[ill]
            ax.plot(self.ocoords[ill], self.ocoords[ill] * data[self.mask_oells[ill]], color='C0', ell=r'$P_{\mathrm{o}}(k)$')
            ax.plot(self.ocoords[ill], self.ocoords[ill] * data_rotated[self.mask_oells[ill]], color='C1', ell=r'$P_{\mathrm{o}}^{\prime}(k)$')
            ax.set_title(r'$\ell = {}$'.format(ell))
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            if xlim is not None: ax.set_xlim(xlim)

        lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.legend()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        return fig

    @plotter
    def plot_validation(self, data, theory, xlim=None, covariance=None, shotnoise=0., marg_shotnoise=False, nobs=1):
        """
        Validation plot of the window matrix and rotation, including error bars and residuals.

        Parameters
        ----------
        data : array-like
            Data vector.
        theory : array-like
            Theory vector.
        xlim : tuple, optional
            x-axis limits.
        covariance : ndarray, optional
            Covariance matrix.
        shotnoise : float, optional
            Shot noise value.
        marg_shotnoise : bool, optional
            Whether to marginalize shot noise.
        nobs : int, optional
            Number of observations for error scaling.
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
        fig : matplotlib.figure.Figure
            Figure object.
        """
        from matplotlib import pyplot as plt
        oells = self.olabels

        fig, lax = plt.subplots(2, len(oells), figsize=(3 * len(oells), 4), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)
        rotate = self.rotate(data=data, theory=theory, covariance=covariance, xlim=xlim, shotnoise=shotnoise)
        precmatrix_rotated = None
        try:
            window_rotated, covariance_rotated, data_rotated = rotate
            offset = 0.
        except ValueError:
            window_rotated, covariance_rotated, data_rotated, m, offset, precmatrix_rotated = rotate

        #np.save('precm.npy', precmatrix_rotated)
        std = np.sqrt(np.diag(covariance_rotated))
        if precmatrix_rotated is not None:
            std = np.sqrt(np.diag(np.linalg.inv(precmatrix_rotated)))
        std /= np.sqrt(nobs)
        ocoords = np.concatenate(list(self.ocoords))
        if xlim is not None:
            mask_ox = (ocoords >= xlim[0]) & (ocoords <= xlim[-1])
        else:
            mask_ox = np.ones(len(ocoords), dtype='?')
        data = np.asarray(data).real.ravel()
        if xlim is not None:
            data = data[mask_ox]
        data_rotated -= offset
        theory_rotated = np.matmul(window_rotated, theory + shotnoise * self.mask_tlabels[self.tlabels.index(0)]) - shotnoise * self.mask_olabels[self.olabels.index(0)][mask_ox]
        #theory_rotated = np.matmul(window_rotated, theory)
        #theory_rotated = theory

        if marg_shotnoise:  # shotnoise free
            precmatrix = np.linalg.inv(covariance_rotated)
            deriv = np.matmul(window_rotated, self.mask_tlabels[self.tlabels.index(0)])[None, :]
            derivp = deriv.dot(precmatrix)
            fisher = derivp.dot(deriv.T)
            shotnoise_value = np.linalg.solve(fisher, derivp.dot(data_rotated - theory_rotated))
            theory_rotated_shotnoise = theory_rotated + shotnoise_value.dot(deriv)

        for ill, ell in enumerate(oells):
            color = 'C{}'.format(ill)
            ax = lax[0][ill]
            xx = ocoords[self.mask_olabels[ill] & mask_ox]
            mask = self.mask_olabels[ill][mask_ox]
            ax.errorbar(xx, xx * data_rotated[mask], xx * std[mask], color=color, marker='.', ls='', label=r'$P_{\mathrm{o}}(k)$')
            ax.plot(xx, xx * np.interp(xx, self.tcoords[self.tlabels.index(ell)], theory[self.mask_tlabels[self.tlabels.index(ell)]]), color=color, ls=':', label=r'$P_{\mathrm{t}}(k)$')
            ax.plot(xx, xx * theory_rotated[mask], color=color, label=r'$W(k, k^{\prime}) P_{\mathrm{t}}(k^{\prime})$')
            if marg_shotnoise:
                ax.plot(xx, xx * theory_rotated_shotnoise[mask], color=color, ls='--', label=r'$W(k, k^{\prime}) (P_{\mathrm{t}}(k^{\prime}) + N)$')
            ax.set_title(r'$\ell = {}$'.format(ell))
            ax.set_xlim(xlim)
            ax.grid(True)
            ax = lax[1][ill]
            ax.plot(xx, ((theory_rotated_shotnoise if marg_shotnoise else theory_rotated)[mask] - data_rotated[mask]) / std[mask], color=color)
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            ax.set_ylim(-2., 2.)
            ax.grid(True)

        lax[0][0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[0][0].legend()
        lax[1][0].set_ylabel(r'$\Delta P_{\ell}(k) / \sigma$')
        fig.align_ylabels()
        return fig