"""Task 6 -- every run of the learnable-aggregator probe.

Source: outputs/probe22b_learnalpha_resnet50_imagenette.json, written by
scripts/probe22b_learnalpha.py.  AutoPool head (scripts/probe22_autopool.py):
z_k = sum_t softmax_t(alpha_k * s_tk) * s_tk with per-cell class scores s_tk, one
alpha per class; alpha = 0 is the mean (GAP + FC), alpha > 0 moves toward max
(disjunctive), alpha < 0 toward min (conjunctive).  Frozen ResNet-50 features,
Imagenette, 8 epochs, own learning rate for alpha, 5 seeds.  The reported alpha
of a run is the mean over the 10 classes; the accuracy change is paired with the
alpha-frozen-at-0 run of the same seed.  Accuracy in percentage points.
"""
import os
import json
import numpy as np
from scipy import stats
import llib as L

SRC = os.path.join(L.ROOT, 'outputs', 'probe22b_learnalpha_resnet50_imagenette.json')


def tci(x):
    x = np.asarray(x, float)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return [float(m), float(m - h), float(m + h)]


def main():
    j = json.load(open(SRC))
    meta = j['meta']
    base = j['frozen0']['per_seed']
    runs, cfg = [], []
    for alr in meta['alr']:
        for a0 in meta['inits']:
            k = 'alr%s_init%s' % (alr, a0)
            r = j[k]
            al = np.asarray(r['per_seed_alpha'], float)
            ac = np.asarray(r['per_seed_acc'], float)
            d = 100.0 * (ac - np.asarray(base, float))
            assert abs(np.mean(d) / 100.0 - r['dacc']) < 1e-12, k
            for s, (x, y, z) in enumerate(zip(al, ac, d)):
                runs.append(dict(alr=alr, init=a0, seed=meta['seeds'][s], alpha_end=float(x),
                                 drift=float(x - a0), acc=float(y), dacc_pts=float(z)))
            cfg.append(dict(alr=alr, init=a0, alpha_end_mean=float(al.mean()),
                            alpha_end_min=float(al.min()), alpha_end_max=float(al.max()),
                            drift_mean=float(al.mean() - a0),
                            toward_mean=int(np.sum(np.abs(al) < abs(a0))) if a0 != 0 else None,
                            dacc_pts_t=tci(d), dacc_pts_min=float(d.min()),
                            dacc_pts_max=float(d.max()), PR=r['PR'],
                            acc_mean=float(ac.mean())))
    allr = np.array([r['dacc_pts'] for r in runs])
    by_init = {}
    for a0 in meta['inits']:
        sub = [r for r in runs if r['init'] == a0]
        al = np.array([r['alpha_end'] for r in sub]); dd = np.array([r['dacc_pts'] for r in sub])
        by_init[str(a0)] = dict(n=len(sub), alpha_end_min=float(al.min()),
                                alpha_end_max=float(al.max()), dacc_mean=float(dd.mean()),
                                dacc_min=float(dd.min()), dacc_max=float(dd.max()),
                                n_dacc_positive=int((dd > 0).sum()),
                                n_moved_positive=int((al > a0).sum()),
                                n_toward_zero=int((np.abs(al) < abs(a0)).sum()) if a0 != 0 else None)
    # correlation of |alpha_end| with accuracy change over the init-0 runs
    z = [r for r in runs if r['init'] == 0.0]
    rho = stats.spearmanr([r['alpha_end'] for r in z], [r['dacc_pts'] for r in z])
    res = dict(source=os.path.relpath(SRC, L.ROOT), meta=meta,
               frozen0=dict(acc=j['frozen0']['acc'], per_seed=base),
               configs=cfg, runs=runs, by_init=by_init,
               all45=dict(n=len(runs), dacc_mean=float(allr.mean()), dacc_min=float(allr.min()),
                          dacc_max=float(allr.max()), n_positive=int((allr > 0).sum())),
               init0_spearman_alpha_vs_dacc=dict(rho=float(rho.statistic), p=float(rho.pvalue),
                                                 n=len(z)))
    print('frozen at 0 (mean pooling): acc %.4f  per seed %s' % (j['frozen0']['acc'], base))
    print('%-8s %-6s %-30s %-34s %s' % ('alpha lr', 'init', 'alpha_end mean [min, max]',
                                        'd acc pts mean [t95] (min, max)', 'PR'))
    for c in cfg:
        print('%-8g %+-6.1f %+.3f [%+.3f, %+.3f]         %+.3f [%+.3f, %+.3f] (%+.3f, %+.3f)   %.3f'
              % (c['alr'], c['init'], c['alpha_end_mean'], c['alpha_end_min'], c['alpha_end_max'],
                 *c['dacc_pts_t'], c['dacc_pts_min'], c['dacc_pts_max'], c['PR']))
    print('by init:', json.dumps(by_init, indent=1))
    print('all 45 runs: mean d acc %+.3f pts, range [%+.3f, %+.3f], %d of %d positive'
          % (allr.mean(), allr.min(), allr.max(), (allr > 0).sum(), len(allr)))
    print('init-0 runs: Spearman(alpha_end, d acc) = %+.3f (p %.2g, n %d)'
          % (rho.statistic, rho.pvalue, len(z)))
    L.save('l6_learnalpha.json', res)

    def m3(x):
        return ('%+.3f' % x).replace('-', '$-$')
    rows = ['%g & %s & %s [%s, %s] & %s [%s, %s] & %s, %s & %.3f'
            % (c['alr'], ('%+.1f' % c['init']).replace('-', '$-$'), m3(c['alpha_end_mean']),
               m3(c['alpha_end_min']), m3(c['alpha_end_max']), m3(c['dacc_pts_t'][0]),
               m3(c['dacc_pts_t'][1]), m3(c['dacc_pts_t'][2]), m3(c['dacc_pts_min']),
               m3(c['dacc_pts_max']), c['PR']) for c in cfg]
    L.longtable(os.path.join(L.OUT, 'tab_l6_learnalpha.tex'), 'lrlllr',
                'Learning rate of $\\alpha$ & Initial $\\alpha$ & Final $\\alpha$, mean [min, max] & '
                '$\\Delta$acc (points), mean [$t$ 95\\% CI] & $\\Delta$acc min, max & PR', rows,
                'All 45 runs of the learnable aggregator (ResNet-50, Imagenette, frozen features).',
                'tab:learnalpha',
                'Five seeds per row; $\\Delta$acc is paired with the run of the same seed whose '
                'aggregator is frozen at the mean ($\\alpha = 0$, accuracy %.2f\\%%). PR: '
                'participation ratio of the pooling weights (1 = mean, 1/49 = maximum).'
                % (100 * j['frozen0']['acc']))


if __name__ == '__main__':
    main()
