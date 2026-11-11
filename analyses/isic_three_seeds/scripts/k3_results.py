"""K3: collect the three per-seed grids into out/k3_grid_all_seeds.json and write RESULTS.md."""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
K3 = os.path.abspath(os.path.join(HERE, '..'))
OUTD = os.path.join(K3, 'out')
SEEDS = (0, 1, 2)
ORIGINAL = dict(diff=-0.13577468718005914, lo=-0.15385485721311198, hi=-0.116981621234093,
                p_wilcoxon=3.309912651921741e-36, n=735, source='outputs/medical/m5_isic.json')


def f(c, p='P'):
    return '%+.3f [%+.3f, %+.3f], P = %.2g' % (c['diff'], c['lo'], c['hi'], c[p])


def main():
    R = {s: json.load(open(os.path.join(OUTD, 'k3_seed%d.json' % s))) for s in SEEDS}
    keys = sorted(k for k in R[0]['grid']['closed_form'] if k.startswith(('H1', 'H2')))
    extra = sorted(k for k in R[0]['grid']['closed_form'] if k.startswith('Sminus'))
    allg = dict(original_first_run_primary=ORIGINAL,
                seeds={s: dict(auc=R[s]['auc'], n_scored=R[s]['n_scored'], primary=R[s]['primary'],
                               grid=R[s]['grid'], dice_q_chosen_per_fold=R[s]['dice_q_chosen_per_fold'],
                               fd_vs_cf=R[s]['fd_vs_cf'], peak_gpu_gb=R[s]['peak_gpu_gb'])
                       for s in SEEDS},
                prespecified_keys=keys, extra_keys=extra)
    json.dump(allg, open(os.path.join(OUTD, 'k3_grid_all_seeds.json'), 'w'), indent=1)

    L = ['# K3: ISIC arm, three seeds, closed-form and forward-difference maps', '',
         'Contrast: necessity map minus comparison map, paired over validation items (n = %d scored of %d).'
         % (R[0]['n_scored'], R[0]['n_val']),
         'Interval: 95% item bootstrap, 10,000 resamples. P: Wilcoxon signed-rank (Dice, AP); '
         'exact McNemar for hit (Wilcoxon P also in the JSON).', '',
         '## Primary: H1, T = +8, 1 x 1, Dice', '',
         '| Seed | Macro AUC | Closed form | Forward difference |', '|---|---|---|---|']
    for s in SEEDS:
        p = R[s]['primary']
        L.append('| %d | %.3f | %s | %s |' % (s, R[s]['auc']['macro'], f(p['closed_form']),
                                               f(p['forward_difference'])))
    L.append('| first run | 0.834 | - | %s |' % f(ORIGINAL, 'p_wilcoxon'))
    L += ['', 'Mean Dice of the two maps (necessity / collapsed):', '']
    for s in SEEDS:
        p = R[s]['primary']
        L.append('- seed %d: closed form %.3f / %.3f; forward difference %.3f / %.3f'
                 % (s, p['closed_form']['mean_a'], p['closed_form']['mean_b'],
                    p['forward_difference']['mean_a'], p['forward_difference']['mean_b']))
    L += ['', '## Pre-specified grid (difference, interval, P)', '',
          '| Cell | ' + ' | '.join('Seed %d CF | Seed %d FD' % (s, s) for s in SEEDS) + ' |',
          '|---|' + '---|' * (2 * len(SEEDS))]
    for k in keys:
        row = []
        for s in SEEDS:
            for est in ('closed_form', 'forward_difference'):
                c = R[s]['grid'][est][k]
                row.append('%+.3f [%+.3f, %+.3f] %.2g' % (c['diff'], c['lo'], c['hi'], c['P']))
        L.append('| %s | %s |' % (k, ' | '.join(row)))
    L += ['', 'CF: closed form. FD: forward difference. H1: necessity minus collapsed. '
          'H2: necessity minus sufficiency. The collapsed map is the T = 0 map of the same '
          'estimator.', '',
          '## Agreement of the two estimators (seed 0, 1, 2)', '',
          '| Map | Cells exactly zero, FD | Max relative difference | Argmax agreement |',
          '|---|---|---|---|']
    for k in sorted(R[0]['fd_vs_cf']):
        L.append('| %s | %s | %s | %s |' % (
            k, ' / '.join('%.4f' % R[s]['fd_vs_cf'][k]['frac_exact_zero_fd'] for s in SEEDS),
            ' / '.join('%.1e' % R[s]['fd_vs_cf'][k]['max_rel_abs_diff'] for s in SEEDS),
            ' / '.join('%.3f' % R[s]['fd_vs_cf'][k]['argmax_agree'] for s in SEEDS)))
    L += ['', '## Departures from the protocol', '',
          '- ImageNet initialisation: torchvision resnet50-0676ba61.pth (classifier re-initialised), '
          'because the timm default ResNet-50 weights used by the first run are not available offline.',
          '- GPU nondeterminism: seeds fix the initialisation of the classifier, data order and '
          'augmentation; cuDNN kernels were not forced to be deterministic, as in the first run.', '']
    open(os.path.join(K3, 'RESULTS.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
