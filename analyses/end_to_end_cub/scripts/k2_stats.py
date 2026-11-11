"""K2 contrasts and RESULTS.md from the per-image arrays of the 12 models.

    python k2_stats.py [--dir OUTDIR]

Primary family (protocol): for H in {H1,H2,H3} and map in {N,S}:
    D = mean over seeds s of [MaxBoxAccV2(map_{H,s}) - MaxBoxAccV2(collapsed_{H0,s})]
  95% percentile bootstrap over test images (1,000 resamples, one image resample shared by all
  seeds and maps, i.e. seeds pooled by image), P = 2 min(P*(b <= 0), P*(b >= 0)) clipped to
  [1/1000, 1] (the convention of wlib.boot_mbav2_diff), Holm (wlib.holm) over the 6 at 0.05.
Secondary families, same bootstrap and P: the 6 contrasts on strict pointing and on best IoU;
the 6 maps against plain CAM of H0 (per metric); within-model N vs S (3 per metric);
H0 collapsed vs H0 CAM (1 per metric); top-1 differences H1-H3 minus H0 (t interval over the
3 seed-matched differences, df = 2; Holm over 3).
"""
import os, sys, json, argparse
import numpy as np
from scipy import stats as ST
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2lib as L
W = L.W

NB = 1000
DELTAS = tuple(W.PROTOCOL['deltas'])
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=L.OUT)
    a = ap.parse_args()
    D = {}; E = {}
    for h in L.HEADS:
        for s in L.SEEDS:
            D[h, s] = dict(np.load(os.path.join(a.dir, 'per_image', '%s.npz' % L.tag(h, s))))
            E[h, s] = json.load(open(os.path.join(a.dir, 'eval_%s.json' % L.tag(h, s))))
    n = len(D['H0', 0]['correct'])
    assert all(len(v['correct']) == n for v in D.values())
    MAPS = [('H0', 'collapsed'), ('H0', 'cam')] + [(h, m) for h in ('H1', 'H2', 'H3') for m in ('N', 'S')]
    key = lambda h, m: '%s_%s' % (h, m)                                  # noqa: E731

    # ---- MaxBoxAccV2 through indicator matrices; equal to wlib.maxboxacc_v2 (checked below)
    IND = {}
    for h, m in MAPS:
        for s in L.SEEDS:
            c = D[h, s]['curve_' + m]
            IND[h, m, s] = np.concatenate([(c >= d) for d in DELTAS], 1).astype(np.float64)  # (n, 3*100)

    def mba_w(h, m, s, w):
        v = (w @ IND[h, m, s]) / w.sum()
        return float(v.reshape(len(DELTAS), -1).max(1).mean())

    rs = np.random.RandomState(W.PROTOCOL['seed'])
    IDX = rs.randint(0, n, (NB, n))
    WTS = np.stack([np.bincount(k, minlength=n) for k in IDX]).astype(np.float64)
    one = np.ones(n)
    # check against wlib on three resamples
    chk = 0.0
    for b in range(3):
        for (h, m) in MAPS[:3]:
            chk = max(chk, abs(mba_w(h, m, 0, WTS[b]) -
                               W.maxboxacc_v2(D[h, 0]['curve_' + m], DELTAS, IDX[b])[0]))
            chk = max(chk, abs(mba_w(h, m, 0, one) - W.maxboxacc_v2(D[h, 0]['curve_' + m], DELTAS)[0]))
    assert chk < 1e-12, chk

    def per_img(h, m, s, metric):
        if metric == 'pointing_strict':
            return D[h, s]['pt_strict_' + m].astype(np.float64)
        return D[h, s]['best_iou_' + m].astype(np.float64)

    def contrast(A, B, metric):
        """mean over seeds of metric(A) - metric(B); A, B = (head, map)."""
        if metric == 'MaxBoxAccV2':
            f = lambda w: np.mean([mba_w(*A, s, w) - mba_w(*B, s, w) for s in L.SEEDS])  # noqa: E731
            d0 = f(one)
            bs = np.array([f(WTS[b]) for b in range(NB)])
            per_seed = [mba_w(*A, s, one) - mba_w(*B, s, one) for s in L.SEEDS]
        else:
            dimg = np.mean([per_img(*A, s, metric) - per_img(*B, s, metric) for s in L.SEEDS], 0)
            d0 = float(dimg.mean())
            bs = (WTS @ dimg) / n
            per_seed = [float((per_img(*A, s, metric) - per_img(*B, s, metric)).mean()) for s in L.SEEDS]
        p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
        return dict(a=key(*A), b=key(*B), metric=metric, diff=float(d0),
                    ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    p=float(min(1.0, max(p, 1.0 / NB))), diff_per_seed=[float(x) for x in per_seed])

    def family(name, rows):
        adj = W.holm([r['p'] for r in rows])
        for r, q in zip(rows, adj):
            r['p_holm'] = q
            r['verdict'] = ('no difference' if q >= ALPHA else
                            ('better' if r['diff'] > 0 else 'worse'))
            r['family'] = name
            r['ci_excludes_0'] = bool(r['ci'][0] > 0 or r['ci'][1] < 0)
        return rows

    SEP = [(h, m) for h in ('H1', 'H2', 'H3') for m in ('N', 'S')]
    out = dict(n_test=n, n_boot=NB, seeds=L.SEEDS, alpha=ALPHA, mba_wlib_check_max_abs=chk)
    out['primary'] = family('primary: MaxBoxAccV2 vs H0 collapsed',
                            [contrast(x, ('H0', 'collapsed'), 'MaxBoxAccV2') for x in SEP])
    sec = {}
    for met in ('pointing_strict', 'best_iou'):
        sec['vs_collapsed_' + met] = family('secondary: %s vs H0 collapsed' % met,
                                            [contrast(x, ('H0', 'collapsed'), met) for x in SEP])
    for met in ('MaxBoxAccV2', 'pointing_strict', 'best_iou'):
        sec['vs_cam_' + met] = family('secondary: %s vs H0 CAM' % met,
                                      [contrast(x, ('H0', 'cam'), met) for x in SEP])
        sec['N_vs_S_' + met] = family('secondary: %s within-model N vs S' % met,
                                      [contrast((h, 'N'), (h, 'S'), met) for h in ('H1', 'H2', 'H3')])
        sec['H0_collapsed_vs_cam_' + met] = family('secondary: %s H0 collapsed vs CAM' % met,
                                                   [contrast(('H0', 'collapsed'), ('H0', 'cam'), met)])
    top = []
    for h in ('H1', 'H2', 'H3'):
        d = np.array([E[h, s]['top1'] - E['H0', s]['top1'] for s in L.SEEDS])
        se = d.std(ddof=1) / np.sqrt(len(d)); tq = ST.t.ppf(0.975, len(d) - 1)
        p = float(2 * ST.t.sf(abs(d.mean() / se), len(d) - 1)) if se > 0 else 1.0
        top.append(dict(a=h, b='H0', metric='top1', diff=float(d.mean()),
                        ci=[float(d.mean() - tq * se), float(d.mean() + tq * se)], p=p,
                        diff_per_seed=d.tolist()))
    sec['top1'] = family('secondary: top-1 vs H0 (t over seeds)', top)
    out['secondary'] = sec

    # ---- descriptives
    desc = {}
    for h in L.HEADS:
        t1 = [E[h, s]['top1'] for s in L.SEEDS]
        sp = {k: [E[h, s]['separation'][k] for s in L.SEEDS] for k in E[h, 0]['separation']}
        mp = {}
        for m in E[h, 0]['maps']:
            mp[m] = {met: [E[h, s]['per_map'][m][met] for s in L.SEEDS]
                     for met in ('MaxBoxAccV2', 'pointing_strict', 'pointing_tol15', 'best_iou')}
            for met in list(mp[m]):
                mp[m][met + '_mean'] = float(np.mean(mp[m][met]))
        chk_w = {f: max(E[h, s]['check_worst'][f] for s in L.SEEDS) for f in E[h, 0]['check_worst']}
        desc[h] = dict(desc=L.HEAD_DESC[h], top1=t1, top1_mean=float(np.mean(t1)),
                       top1_sd=float(np.std(t1, ddof=1)),
                       top1_float64_head=[E[h, s]['top1_float64_head'] for s in L.SEEDS],
                       separation=sp, separation_mean={k: float(np.mean(v)) for k, v in sp.items()},
                       maps=mp, checks_worst=chk_w,
                       checks=[E[h, s]['checks'] for s in L.SEEDS],
                       micro_batch=[E[h, s]['micro'] for s in L.SEEDS])
    out['models'] = desc
    L.save_json(os.path.join(a.dir, 'k2_contrasts.json'), out)
    write_md(a.dir, out)


LBL = {'H0': 'H0 (average pooling, degenerate)', 'H1': 'H1 (log-sum-exp, T = +1)',
       'H2': 'H2 (gated attention)', 'H3': 'H3 (average pooling + LayerNorm)'}
MLBL = {'H0_collapsed': 'H0 collapsed', 'H0_cam': 'H0 CAM'}


def ml(k):
    if k in MLBL:
        return MLBL[k]
    h, m = k.split('_')
    return '%s %s' % (h, {'N': 'necessity', 'S': 'sufficiency'}[m])


def fmt(r):
    return ('| %s | %s | %+.4f | [%+.4f, %+.4f] | %s | %.3g | %s |'
            % (ml(r['a']) if '_' in r['a'] else r['a'], ml(r['b']) if '_' in r['b'] else r['b'],
               r['diff'], r['ci'][0], r['ci'][1],
               ', '.join('%+.4f' % x for x in r['diff_per_seed']), r['p_holm'], r['verdict']))


def write_md(d, o):
    M = o['models']
    L_ = ['# K2: localisation of separable heads trained end to end (CUB-200-2011)', '',
          'ResNet-50, 448 x 448, 3 seeds per head; %d test images; maps for the ground-truth class '
          'from 1 x 1 cell substitution in float64, averaged over 8 training-split references. '
          'Differences are A minus B, mean over seeds; intervals are 95%% percentile bootstrap over '
          'test images (%d resamples); P is Holm-adjusted within each family at alpha = 0.05.'
          % (o['n_test'], o['n_boot']), '',
          '## Accuracy and separation', '',
          '| head | top-1 (mean of 3) | top-1 per seed | separation, 1 reference | rank separation, 1 reference | separation, 8 references | rank separation, 8 references |',
          '|---|---|---|---|---|---|---|']
    for h in L.HEADS:
        s = M[h]['separation_mean']
        L_.append('| %s | %.4f | %s | %.4f | %.4f | %.4f | %.4f |'
                  % (LBL[h], M[h]['top1_mean'], ', '.join('%.4f' % x for x in M[h]['top1']),
                     s['sep1_pearson'], s['sep1_spearman'], s['sep8_pearson'], s['sep8_spearman']))
    L_ += ['', 'Separation = 1 - Pearson correlation of the necessity and sufficiency maps; rank '
           'separation = 1 - Spearman correlation; per image, mean over images and seeds.', '',
           '## Localisation of every map (mean of 3 seeds)', '',
           '| map | MaxBoxAccV2 | per seed | strict pointing | best IoU | pointing, 15 px |',
           '|---|---|---|---|---|---|']
    for h in L.HEADS:
        for m, v in M[h]['maps'].items():
            L_.append('| %s | %.4f | %s | %.4f | %.4f | %.4f |'
                      % (ml('%s_%s' % (h, m)), v['MaxBoxAccV2_mean'],
                         ', '.join('%.4f' % x for x in v['MaxBoxAccV2']), v['pointing_strict_mean'],
                         v['best_iou_mean'], v['pointing_tol15_mean']))
    hdr = ['| A | B | difference | 95% interval | per seed | P (Holm) | verdict |',
           '|---|---|---|---|---|---|---|']
    L_ += ['', '## Primary: MaxBoxAccV2 against the collapsed map of H0 (6 contrasts)', ''] + hdr
    L_ += [fmt(r) for r in o['primary']]
    S = o['secondary']
    for k, title in [('vs_collapsed_pointing_strict', 'Strict pointing against H0 collapsed'),
                     ('vs_collapsed_best_iou', 'Best IoU against H0 collapsed'),
                     ('vs_cam_MaxBoxAccV2', 'MaxBoxAccV2 against H0 CAM'),
                     ('vs_cam_pointing_strict', 'Strict pointing against H0 CAM'),
                     ('vs_cam_best_iou', 'Best IoU against H0 CAM'),
                     ('N_vs_S_MaxBoxAccV2', 'Necessity vs sufficiency, MaxBoxAccV2'),
                     ('N_vs_S_pointing_strict', 'Necessity vs sufficiency, strict pointing'),
                     ('N_vs_S_best_iou', 'Necessity vs sufficiency, best IoU'),
                     ('H0_collapsed_vs_cam_MaxBoxAccV2', 'H0 collapsed vs CAM, MaxBoxAccV2'),
                     ('H0_collapsed_vs_cam_pointing_strict', 'H0 collapsed vs CAM, strict pointing'),
                     ('H0_collapsed_vs_cam_best_iou', 'H0 collapsed vs CAM, best IoU'),
                     ('top1', 'Top-1 against H0 (t interval over 3 seed-matched differences)')]:
        L_ += ['', '## Secondary: %s' % title, ''] + hdr + [fmt(r) for r in S[k]]
    L_ += ['', '## Map checks (largest value over 3 images x 2 references x 3 seeds)', '',
           '| head | necessity, map vs literal substitution | sufficiency, map vs literal substitution | literal necessity minus sufficiency (relative) | same, absolute |',
           '|---|---|---|---|---|']
    for h in L.HEADS:
        c = M[h]['checks_worst']
        L_.append('| %s | %.1e | %.1e | %.1e | %.1e |'
                  % (h, c['N_closed_vs_literal_rel'], c['S_closed_vs_literal_rel'],
                     c['N_vs_S_literal_rel'], c['N_vs_S_literal_abs']))
    L_ += ['', 'Relative values are divided by the largest absolute map value of that image and reference.']
    dev = os.path.join(d, 'DEVIATIONS.md')
    if os.path.exists(dev):
        L_ += ['', open(dev, encoding='utf-8').read().strip()]
    open(os.path.join(os.path.dirname(d), 'RESULTS.md'), 'w', encoding='utf-8').write('\n'.join(L_) + '\n')


if __name__ == '__main__':
    main()
