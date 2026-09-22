r"""Reproduce the AbacusSummit LRG covariance of ``claude_bk_cov_new`` with :mod:`jaxpower.cov3`.

That campaign's best matrix is ``claude_bk_cov_new/out/baseline_cov.npz``, plotted as
``cov_best_final.png``: ``P0, P2, P4, B000, B202`` on nine bins over ``0.02 < k < 0.20``, against
500 AbacusSummit small-box mocks. It is ``run_baseline.sh`` that says so -- step 4 writes that
file and step 5 plots it. ``fig_cov_final.py`` defaults to a different file, ``abacus_f6exp.npz``,
which is not the one in the figure. It was produced by that directory's own ``covbox.py``, which
:mod:`jaxpower.cov3` was ported from and has since been rewritten around -- multitracer field
tuples, a geometry seam for survey windows, a single shot-noise amplitude, and the ``n - 1``
leg convention. This script checks that none of that moved the answer.

What it does, in order:

1. Rebuilds the inputs with the campaign's own code -- ``data.load`` for the observable and the
   mocks, ``abacus.build_theory`` for the theory callables -- so the covariance modules are the
   only thing that differs. Nothing in ``claude_bk_cov_new`` is modified or imported for its
   covariance.
2. Computes the eight terms with :mod:`jaxpower.cov3`.
3. Compares three ways:

   * **against the pinned reference**, term by term, at machine precision. ``covbox`` used to
     play this role and can no longer: :mod:`jaxpower.cov3` now departs from it deliberately,
     the ``(2, 2)`` closure floor being a fix for a pairing that diverges with quadrature order
     in ``covbox`` (1.27e23 to 7.17e23 between orders 6 and 24, still climbing). Against that
     implementation an intended fix reads as a 0.8% failure in ``BB`` and ``PT``. So the
     reference is this script's own output, pinned with ``--pin`` and re-pinned deliberately
     whenever a change is meant. It records the hash of ``cov3.py`` it was made with, so a diff
     always says whether the code moved too.
   * **against the stored terms** of ``baseline_cov.npz``. That file does not record every flag
     it was made with -- ``fog_damping`` and ``tree_bias`` in particular -- so this one tests the
     guesses in ``CONFIGURATION`` below. A mismatch here with agreement above means the flags are
     wrong, not the code.
   * **against the 500 mocks**: the ratio of analytic to mock standard deviation per block, the
     mean off-diagonal correlation, and the smallest eigenvalue of the correlation matrix, which
     is where a covariance that matches every variance can still be wrong. This is the only arm
     that can say the answer is *right* rather than *unchanged*, and it resolves a few per cent
     at best -- which is why the pinned reference earns its keep beside it.

4. Writes a figure: the two correlation matrices, their difference, and the diagonal ratios.

Run it inside an allocation::

    srun --jobid=$JID -n 1 --overlap bash claude_bk_cov_new/gpu2.sh \
        jax-power/scripts/reproduce_abacus_covariance.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

#: The bispectrum-covariance campaign, which holds ``abacus.py`` and its ``covbox.py``.
CAMPAIGN = Path('/global/u2/a/adematti/cosmodesi/claude_bk_cov_new')
#: The earlier campaign it draws the mocks and the tracer model from.
MOCKS = Path('/global/u2/a/adematti/cosmodesi/claude_abacus_analytic_cov')
#: What ``run_baseline.sh`` step 4 writes and step 5 turns into ``cov_best_final.png``.
REFERENCE = CAMPAIGN / 'out' / 'baseline_cov.npz'
#: Step 3's output, the deterministic theory plus coincidence amplitudes the covariance wants.
COVARIANCE_INPUT = CAMPAIGN / 'out' / 'baseline_cov_in.npz'

#: ``run_baseline.sh`` step 4, which is the pipeline of record and states each flag with its
#: reason. One caveat that the stored file cannot tell you and this script cannot assume:
#: ``out/baseline_cov.npz`` was written before ``treepk.py`` gained ``above3`` and before
#: ``run_baseline.sh`` was edited to pass ``--tree-bias local+tidal``, so the *stored* P5 and P6
#: carry no tidal b_K^2 in K_4 and K_5 while these do. Nothing in the file records the flag, so
#: the only way to see it is the file times. ``--tree-bias`` below overrides this for that test.
CONFIGURATION = dict(krange=(0.02, 0.20), rebin2=5, meshsize=256, nmax=6, shot_bias='b',
                     ct_t=None, tree_bias='local+tidal', fog_full=False, fog_damping='exp',
                     order3=16, order5=6, p6_size=8192, seed=0, nsub=(4, 2))

#: Steps 2 and 3: the joint fit, and its translation into covariance input. ``--mode excess``
#: keeps the coincidence amplitudes Poisson and puts the tracer's non-Poisson deviation into the
#: theory, so nothing is counted twice against what ``Spectra`` rebuilds itself.
FIT_COMMAND = ['fitall.py', '--model', 'minimal', '--kmax', '0.2', '--alpha-prior', '0.2',
               '--ct-prior', '50', '--xfog', '1.0112', '--free-xfog', '--fog-damping', 'exp',
               '--b-source', 'sug', '--b-meas', 'out/bmult_sug10.npz', '--nkeep-b', '2000',
               '--hessian', '--output']
TRANSLATE_COMMAND = ['minimal_cov_input.py', '--mode', 'excess']


#: The pinned regression reference: this script's own eight term matrices, at machine precision.
#: Small enough (eight 45 x 45 float64 blocks) to live beside the script and be diffed in review.
#: Not to be confused with ``REFERENCE`` above, which is the campaign's stored baseline.
PINNED = Path(__file__).parent / '_reference' / 'abacus_cov3_terms.npz'


def _cov3_md5():
    """The hash of the ``cov3.py`` actually imported, so a diff can say whether the code moved."""
    import hashlib
    import jaxpower.cov3
    return hashlib.md5(Path(jaxpower.cov3.__file__).read_bytes()).hexdigest()


def _read_reference():
    return np.load(PINNED, allow_pickle=True) if PINNED.exists() else None


def _write_reference(blocks, k, stored, terms):
    """Pin the current output. Deliberate: ``--pin`` is the only way to move the reference.

    The configuration and the ``cov3.py`` hash go in beside the matrices, because a reference
    whose provenance is not recorded answers "did anything change" and not "should it have".
    """
    import datetime
    PINNED.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PINNED, k=k, terms=list(terms), cov3_md5=_cov3_md5(),
             pinned=datetime.date.today().isoformat(), configuration=repr(CONFIGURATION),
             xfog=float(stored['xfog']), damping_method=str(stored['damping_method']),
             nmodes=str(stored['nmodes']), shotnoise=float(stored['shotnoise']),
             **{f'term_{term}': block for term, block in blocks.items()})
    print(f'pinned {len(blocks)} terms to {PINNED} with cov3.py {_cov3_md5()[:8]}')


def run_fit(destination):
    """Steps 2 and 3 of ``run_baseline.sh``: fit the mocks, then translate for the covariance.

    The fit is over P + B(Scoccimarro) + S_4 on diagonal Gaussian errors, with the bispectrum
    measured in six Sugiyama multipoles (``out/bmult_sug10.npz``). Restricting it to (0,0,0) and
    (2,0,2) drives b_K^2 to an unphysical -5.8 anti-correlated with alpha0, so the six are what
    make that parameter measurable rather than railed.

    ``X_FoG`` is seeded at 1.0112 and freed. It cannot be seeded at zero: the damping goes as
    X^2, so dW/dX vanishes there and a freed X_FoG started at zero can never leave it.
    """
    import subprocess
    fit = destination.with_name(destination.stem + '_fit.npz')
    for command in ([sys.executable] + FIT_COMMAND + [str(fit)],
                    [sys.executable] + TRANSLATE_COMMAND[:1] + [str(fit), str(destination)]
                    + TRANSLATE_COMMAND[1:]):
        print('  ' + ' '.join(command[1:]), flush=True)
        subprocess.run(command, cwd=CAMPAIGN, check=True)
    return destination


def main(output, terms=None, quick=False, fit=False, tree_bias=None, pin=False):
    sys.path.insert(0, str(MOCKS))
    sys.path.insert(0, str(CAMPAIGN))
    import data
    import abacus
    from jaxpower import MeshAttrs
    from jaxpower.cov3 import compute_spectrum3_covariance

    covariance_input = COVARIANCE_INPUT
    if fit:
        print('refitting (run_baseline.sh steps 2 and 3)')
        covariance_input = run_fit(Path('/pscratch/sd/a/adematti/claude/cov3/refit_cov_in.npz'))
    reference = np.load(REFERENCE, allow_pickle=True)
    stored = {name: reference[name] for name in reference.files}
    terms = tuple(terms or str(stored['terms']).split(','))
    shotnoise = float(stored['shotnoise'])
    print(f'reference {REFERENCE.name}: fit {stored["fit"]}, nmodes {stored["nmodes"]}, '
          f'damping {stored["damping_method"]}, X_FoG {float(stored["xfog"]):.4f}, '
          f'coincidence amplitude {shotnoise:.1f}')
    print(f'terms {terms}, tree_bias {tree_bias or CONFIGURATION["tree_bias"]}, '
          f'cov3.py md5 {_cov3_md5()[:8]}')

    loaded = data.load(rebin2=CONFIGURATION['rebin2'], krange=CONFIGURATION['krange'])
    observable, k = loaded['observable'], np.asarray(loaded['k2'])
    assert np.allclose(k, stored['k'], rtol=1e-6), \
        'the binning does not match the stored file; check CONFIGURATION["krange"]'
    mocks = np.cov(np.asarray(loaded['X']), rowvar=False)
    assert np.allclose(mocks, stored['mock'], rtol=1e-8), \
        'the mock covariance does not match the stored one; the binning or cache differs'
    mattrs = MeshAttrs(boxsize=data.BOXSIZE, meshsize=CONFIGURATION['meshsize'],
                       boxcenter=data.BOXSIZE / 2.)
    theory, info = abacus.build_theory(
        fit=str(covariance_input), nmax=CONFIGURATION['nmax'],
        shot_bias=CONFIGURATION['shot_bias'], ct_t=CONFIGURATION['ct_t'],
        fog_tree=bool(float(stored['fog_tree'])), fog_damping=CONFIGURATION['fog_damping'],
        xfog=float(stored['xfog']), damping_method=str(stored['damping_method']),
        tree_bias=tree_bias or CONFIGURATION['tree_bias'],
        fog_full=CONFIGURATION['fog_full'])

    shared = dict(theory=theory, order3=CONFIGURATION['order3'], order5=CONFIGURATION['order5'],
                  p6_size=CONFIGURATION['p6_size'], seed=CONFIGURATION['seed'],
                  nmodes=str(stored['nmodes']), nsub=tuple(CONFIGURATION['nsub']))
    if quick:
        shared.update(order3=6, order5=3, p6_size=256, nmodes='continuum', nsub=(1, 1))
        print('QUICK: crude quadrature, so only new-vs-old is meaningful')

    new = {}
    for term in terms:
        start = time.time()
        new[term] = np.asarray(compute_spectrum3_covariance(
            mattrs, mattrs, observable, terms=(term,), shotnoise=shotnoise, **shared).value())
        print(f'  {term:<4} {time.time() - start:5.0f}s', flush=True)

    if pin:
        _write_reference(new, k, stored, terms)
    print('\nterm      vs reference    vs stored')
    reference = _read_reference()
    for term in terms:
        line = f'  {term:<6}'
        if reference is not None and 'term_' + term in reference:
            pinned = np.asarray(reference['term_' + term])
            line += f'  {np.abs(new[term] - pinned).max() / max(np.abs(pinned).max(), 1e-300):.3e}'
        else:
            line += '  (not pinned)'
        if 'term_' + term in stored:
            reference_term = np.asarray(stored['term_' + term])
            line += f'    {np.abs(new[term] - reference_term).max() / max(np.abs(reference_term).max(), 1e-300):.3e}'
        print(line)
    if reference is not None:
        same = str(reference['cov3_md5']) == _cov3_md5()
        print(f'  reference pinned {str(reference["pinned"])} with cov3.py '
              f'{"unchanged since" if same else "DIFFERENT from"} the one just run')

    analytic = sum(new.values())
    if not quick:
        _report_against_mocks(analytic, mocks, list(stored['labels']),
                              len(k), int(stored['nmock']))
    _figure(k, list(stored['labels']), analytic, mocks, len(k), output)
    print(f'\nwrote {output}')
    return analytic


def _correlation(matrix):
    deviation = np.sqrt(np.abs(np.diag(matrix)))
    deviation = np.where(deviation > 0., deviation, np.inf)
    return matrix / np.outer(deviation, deviation)


def _report_against_mocks(analytic, mock, labels, nbin, nmock):
    print(f'\nagainst {nmock} mocks:')
    print('  block   sigma_analytic / sigma_mock')
    for index, label in enumerate(labels):
        block = slice(index * nbin, (index + 1) * nbin)
        ratio = np.sqrt(np.diag(analytic)[block] / np.diag(mock)[block])
        print(f'  {str(label):<6}  {np.median(ratio):.3f}')
    for name, matrix in (('analytic', analytic), ('mocks', mock)):
        correlation = _correlation(matrix)
        off = ~np.eye(len(correlation), dtype=bool)
        print(f'  {name:<9} mean |off-diagonal r| {np.abs(correlation[off]).mean():.4f}, '
              f'min eigenvalue {np.linalg.eigvalsh(correlation).min():+.4f}')


def _figure(k, labels, analytic, mock, nbin, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    analytic_correlation, mock_correlation = _correlation(analytic), _correlation(mock)
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for axis, matrix, title in ((axes[0], mock_correlation, f'mocks'),
                                (axes[1], analytic_correlation, 'jaxpower.cov3'),
                                (axes[2], analytic_correlation - mock_correlation,
                                 'analytic - mocks')):
        limit = 1. if 'mock' in title or 'cov3' in title else 0.2
        image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap='RdBu_r', origin='lower')
        axis.set_title(title)
        for edge in range(1, len(labels)):
            axis.axhline(edge * nbin - 0.5, color='k', lw=0.5)
            axis.axvline(edge * nbin - 0.5, color='k', lw=0.5)
        axis.set_xticks([(index + 0.5) * nbin - 0.5 for index in range(len(labels))])
        axis.set_yticks([(index + 0.5) * nbin - 0.5 for index in range(len(labels))])
        axis.set_xticklabels([str(label) for label in labels])
        axis.set_yticklabels([str(label) for label in labels])
        figure.colorbar(image, ax=axis, fraction=0.046)
    for index, label in enumerate(labels):
        block = slice(index * nbin, (index + 1) * nbin)
        axes[3].plot(k, np.sqrt(np.diag(analytic)[block] / np.diag(mock)[block]),
                     marker='o', ms=3, label=str(label))
    axes[3].axhline(1., color='k', ls=':')
    axes[3].set_xlabel('$k$ [$h$/Mpc]')
    axes[3].set_ylabel(r'$\sigma_{\rm analytic} / \sigma_{\rm mock}$')
    axes[3].set_ylim(0.6, 1.3)
    axes[3].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=110)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default='reproduce_abacus_covariance.png')
    parser.add_argument('--terms', default=None,
                        help='comma-separated subset, e.g. PP,T (default: all eight)')
    parser.add_argument('--fit', action='store_true',
                        help='redo the fit (run_baseline.sh steps 2 and 3) instead of using the '
                             'stored covariance input; adds the expensive part of the pipeline')
    parser.add_argument('--tree-bias', default=None,
                        help='override CONFIGURATION["tree_bias"]; the stored file records no '
                             'such flag, so this is how its value is identified')
    parser.add_argument('--pin', action='store_true',
                        help='overwrite the pinned regression reference with this run; the only '
                             'way it moves, so that an intended change is a deliberate re-pin')
    parser.add_argument('--quick', action='store_true',
                        help='crude quadrature; only the new-vs-old comparison stays meaningful')
    arguments = parser.parse_args()
    main(arguments.output, terms=arguments.terms.split(',') if arguments.terms else None,
         quick=arguments.quick, fit=arguments.fit, tree_bias=arguments.tree_bias,
         pin=arguments.pin)
