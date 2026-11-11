"""C3 tasks 5 and 7 -- summary of the GPU pass (c3_gpu_refit.py) plus the
archived certificates.

Task 5: separation (1 - corr(N,S)) per cell
  * as archived: the certificate's corr, min over 3 probe pairs of fit-split
    images (class 0, reference = the partner image), for edited_lse_tau1
    (T=+1) and edited_lnpost;
  * per pair (re-computed, reproduces the archived minimum);
  * on the evaluated maps: per-record corr(N,S) on the evaluation records
    (bank reference), median [IQR] and mean.
Task 7: the refitted-LSE-head arm.  N_rf / S_rf against the collapsed map and
  the centre prior with w3's tests and Holm inside {N_rf,S_rf} x {collapsed,
  center} per (dataset, backbone, metric); reproduced reused-head contrasts
  alongside as a check of the pipeline.
"""
import os
import json
from collections import Counter
import numpy as np
from localisation_reanalysis_lib import W, WO, OUT, DATASETS, BACKBONES, METRICS, MNAME, EXTENT, holm, verdict, save

G = os.path.join(OUT, 'gpu')


def vec(A, metric, mp):
    if metric == 'MaxBoxAccV2':
        return A['curve_' + mp]
    if metric == 'best_iou':
        return A['curve_' + mp].max(1).astype(np.float64)
    if metric.startswith('pointing'):
        return A['pt_' + mp][0 if metric == 'pointing_strict' else 1].astype(np.float64)
    return A['iou_' + mp]


def tst(A, metric, a, b):
    xa, xb = vec(A, metric, a), vec(A, metric, b)
    if metric == 'MaxBoxAccV2':
        d, lo, hi, p = W.boot_mbav2_diff(xa, xb, n=1000, seed=0)
        return dict(va=W.maxboxacc_v2(xa)[0], vb=W.maxboxacc_v2(xb)[0], diff=d, ci=[lo, hi], p=p)
    d, lo, hi = W.boot_diff(xa, xb, n=2000, seed=0)
    p = (W.mcnemar(xa.astype(bool), xb.astype(bool))['p'] if metric.startswith('pointing')
         else W.wilcoxon(xa, xb)['p'])
    return dict(va=float(xa.mean()), vb=float(xb.mean()), diff=d, ci=[lo, hi], p=p)


def main():
    sep_rows, t7_rows, cls_rows, cert_T = [], [], [], []
    for ds in DATASETS:
        for bk in BACKBONES:
            arch = json.load(open(os.path.join(WO, 'w2_%s_%s.json' % (ds, bk))))
            ce = arch['certificate']
            row = dict(dataset=ds, backbone=bk, n_records=arch['n_records'],
                       cert_sep_lse=1 - ce['edited_lse_tau1']['corrNS_recorded_never_used'],
                       cert_n_pairs=ce['edited_lse_tau1']['n_pairs'],
                       cert_sep_ln=(1 - ce['edited_lnpost']['corrNS_recorded_never_used']
                                    if 'edited_lnpost' in ce else None))
            rp = os.path.join(G, 'res_%s_%s.json' % (ds, bk))
            if os.path.exists(rp):
                R = json.load(open(rp))
                c = R['certificate']
                row['cert_corr_per_pair_lse'] = c['T']['1.0']['corr_per_pair']
                row['cert_corr_per_pair_ln'] = c.get('lnpost', {}).get('corr_per_pair')
                row['cert_reproduced'] = abs(c['T']['1.0']['corr_min'] - c['archived_corr_T1']) < 1e-6
                row['cert_sep_refit'] = c['refit_T1']['sep']
                row['class0_score_range'] = c['class0_score_range_per_image']
                for k, v in R['eval_separation'].items():
                    row['eval_sep_' + k] = v
                row['reproduction'] = R['reproduction']
                cls_rows.append(dict(dataset=ds, backbone=bk, **{k: list(v.values())[0] for k, v in R['classification'].items()},
                                     train_loss=R['train_loss']))
                cert_T.append(dict(dataset=ds, backbone=bk,
                                   **{('sep_T%s' % T): v['sep'] for T, v in c['T'].items()},
                                   **{('ess_T%s' % T): float(np.median(v['effective_cells_per_image'])) for T, v in c['T'].items()},
                                   score_range_median=float(np.median(c['class0_score_range_per_image']))))
                A = np.load(os.path.join(G, 'arr_%s_%s.npz' % (ds, bk)))
                for m in METRICS[ds]:
                    for arm, (na, sa) in (('refit', ('N_rf', 'S_rf')), ('reused', ('N_lse', 'S_lse'))):
                        fam = []
                        for a in (na, sa):
                            for b in ('collapsed', 'center'):
                                r = tst(A, m, a, b)
                                r.update(dataset=ds, backbone=bk, metric=m, a=a, b=b, head=arm)
                                fam.append(r)
                        for r, q in zip(fam, holm([r['p'] for r in fam])):
                            r['p_holm'] = q; r['sig'] = verdict(r['diff'], q)
                        t7_rows += fam
            sep_rows.append(row)

    print('Task 5 -- separation 1-corr(N,S)')
    print('%-9s %-9s %8s %-26s %8s %-22s %-22s %-22s' % ('dataset', 'backbone', 'cert', 'per pair (T=+1)', 'cert_ln',
                                                       'eval lse med[IQR]', 'eval ln med[IQR]', 'eval refit med[IQR]'))
    for r in sep_rows:
        f = lambda k: ('%.3f [%.3f,%.3f]' % (r[k]['median_sep'], r[k]['q25'], r[k]['q75'])) if k in r else '--'
        print('%-9s %-9s %8.3f %-26s %8s %-22s %-22s %-22s'
              % (r['dataset'], r['backbone'], r['cert_sep_lse'],
                 str([round(1 - x, 3) for x in r.get('cert_corr_per_pair_lse', [])]),
                 '%.3f' % r['cert_sep_ln'] if r['cert_sep_ln'] is not None else '--',
                 f('eval_sep_lse'), f('eval_sep_ln'), f('eval_sep_rf')))
    if cert_T:
        print('\ncertificate separation vs T (GAP weights reused), and median effective #cells under LSE weights')
        for r in cert_T:
            print('  %-9s %-9s range %.1f ' % (r['dataset'], r['backbone'], r['score_range_median']) +
                  ' '.join('T%s:%.3f/%.1f' % (T.replace('.0', ''), r['sep_T' + T], r['ess_T' + T])
                           for T in ('-8.0', '-2.0', '-0.5', '0.5', '1.0', '2.0', '8.0')))
    if cls_rows:
        print('\nclassification of the three heads (top-1 for CUB, mAP otherwise)')
        for r in cls_rows:
            print('  %-9s %-9s gap %.4f  lse(reused) %.4f  lse(refit) %.4f  train loss %s'
                  % (r['dataset'], r['backbone'], r['gap'], r['lse_reused'], r['lse_refit'],
                     {k: round(v, 4) for k, v in r['train_loss'].items()}))
    if t7_rows:
        print('\nTask 7 -- refitted LSE head (T=+1) vs collapsed / centre')
        for head in ('reused', 'refit'):
            for comp in ('collapsed', 'center'):
                sub = [r for r in t7_rows if r['head'] == head and r['b'] == comp]
                c = Counter(r['sig'] for r in sub)
                ce = Counter(r['sig'] for r in sub if r['metric'] in EXTENT)
                print('  %-7s vs %-9s n=%2d win %2d loss %2d tie %2d   extent: %s' % (head, comp, len(sub), c['win'], c['loss'], c['tie'], dict(ce)))
                for bk in BACKBONES:
                    s2 = [r for r in sub if r['backbone'] == bk]
                    if s2:
                        c2 = Counter(r['sig'] for r in s2)
                        print('      %-9s n=%2d win %2d loss %2d tie %2d' % (bk, len(s2), c2['win'], c2['loss'], c2['tie']))
        print('  refit rows vs collapsed:')
        for r in t7_rows:
            if r['head'] == 'refit' and r['b'] == 'collapsed':
                print('    %-8s %-8s %-15s %-5s %.4f vs %.4f %+.4f [%+.4f,%+.4f] %s'
                      % (r['dataset'], r['backbone'], MNAME[r['metric']], r['a'], r['va'], r['vb'],
                         r['diff'], r['ci'][0], r['ci'][1], r['sig']))
        # check: reused-head verdicts reproduce w9
        w9 = json.load(open(os.path.join(WO, 'w9_percell.json')))['rows']
        ref = {(x['dataset'], x['backbone'], x['metric'], x['pair']): x['sig'] for x in w9}
        agree = tot = 0
        for r in t7_rows:
            if r['head'] != 'reused':
                continue
            pair = '%s vs %s' % ('necessity' if r['a'].startswith('N') else 'sufficiency',
                                 'collapsed' if r['b'] == 'collapsed' else 'centre prior')
            k = (r['dataset'], r['backbone'], MNAME[r['metric']], pair)
            if k in ref:
                tot += 1; agree += (ref[k] == r['sig'])
        print('  reused-head verdicts reproduced from the re-run: %d of %d' % (agree, tot))
    save('t5_t7_separation_refit.json', dict(separation=sep_rows, cert_vs_T=cert_T,
                                             classification=cls_rows, refit_rows=t7_rows))


if __name__ == '__main__':
    main()
