"""Tasks 1 and 5.

Task 1 -- within-model necessity vs sufficiency (N_lse vs S_lse; N_ln vs S_ln)
for every dataset x backbone x metric, with the test w3_stats.py uses for the
primary contrasts (paired bootstrap for MaxBoxAccV2, exact McNemar for the
pointing games, Wilcoxon signed-rank for best IoU / peak-mask IoU), the
unadjusted 95% percentile bootstrap CI of the difference, and Holm.

Holm families (both reported):
  EXT  (main)  the primary family of w3_stats -- one (dataset, backbone, metric)
               cell -- enlarged by its N-vs-S rows: 4 + 1 = 5 contrasts on a CNN
               cell, 8 + 2 = 10 on a ViT cell.  The primary rows are re-adjusted
               in the enlarged family too, so the cost to the 192 primary
               verdicts is reported.
  NS           the N-vs-S rows of a cell alone: 1 contrast (CNN), 2 (ViT).

Task 5 -- primary contrasts whose Holm verdict and unadjusted bootstrap CI
disagree (significant with a CI covering 0, or not significant with a CI
excluding 0).  Also counted for the N-vs-S rows.
"""
import os
from collections import Counter, defaultdict
import numpy as np
import llib as L

NS_PAIRS = [('N_lse', 'S_lse'), ('N_ln', 'S_ln')]


def main():
    w3 = L.load_w3()
    prim, ns = [], []
    rep_d, rep_p, rep_ph, rep_ns_d, rep_ns_p = [], [], [], [], []
    for ds in L.DATASETS:
        for bk in L.BACKBONES:
            cell = '%s|%s' % (ds, bk)
            for metric in L.METRICS[ds]:
                w3rows = {(r['a'], r['b']): r for r in w3[cell]['contrasts'][metric]}
                fam_p, fam_n = [], []
                for a, b in L.PRIMARY_PAIRS:
                    if (a, b) not in w3rows:
                        continue
                    t = L.test(ds, bk, metric, a, b)
                    r = dict(dataset=ds, backbone=bk, metric=metric, a=a, b=b, **t)
                    o = w3rows[(a, b)]
                    rep_d.append(abs(r['diff'] - o['diff'])); rep_p.append(abs(r['p'] - o['p']))
                    r['p_holm_w3'] = o['p_holm']
                    fam_p.append(r)
                for a, b in NS_PAIRS:
                    if (a, b) not in w3rows:
                        continue
                    t = L.test(ds, bk, metric, a, b)
                    r = dict(dataset=ds, backbone=bk, metric=metric, a=a, b=b,
                             arm=L.arm_of(a), mclass=L.mclass(metric), **t)
                    o = w3rows[(a, b)]
                    rep_ns_d.append(abs(r['diff'] - o['diff']))
                    rep_ns_p.append(abs(r['p'] - o['p']))
                    fam_n.append(r)
                # primary-only Holm (must reproduce w3)
                adj = L.holm([r['p'] for r in fam_p])
                for r, q in zip(fam_p, adj):
                    r['p_holm'] = q
                    rep_ph.append(abs(q - r['p_holm_w3']))
                # EXT: primary + NS rows
                allr = fam_p + fam_n
                adj = L.holm([r['p'] for r in allr])
                for r, q in zip(allr, adj):
                    r['p_holm_ext'] = q
                # NS-only
                adj = L.holm([r['p'] for r in fam_n])
                for r, q in zip(fam_n, adj):
                    r['p_holm_ns'] = q
                    r['family_size_ext'] = len(allr)
                    r['family_size_ns'] = len(fam_n)
                prim += fam_p
                ns += fam_n

    rep = dict(n_primary=len(prim), n_ns=len(ns),
               max_abs_diff_primary=max(rep_d), max_abs_p_primary=max(rep_p),
               max_abs_pholm_primary=max(rep_ph),
               max_abs_diff_ns=max(rep_ns_d), max_abs_p_ns=max(rep_ns_p))
    print('reproduction vs w3_stats.json:', rep)

    # ---------------- verdicts
    def nsv(r, key):
        if r[key] >= L.ALPHA:
            return 'no difference'
        return 'N better' if r['diff'] > 0 else 'S better'
    for r in ns:
        r['verdict_ext'] = nsv(r, 'p_holm_ext')
        r['verdict_ns'] = nsv(r, 'p_holm_ns')
        r['verdict_raw'] = nsv(r, 'p')
        r['ci_disagree_ext'] = bool(L.ci_disagrees(r, 'p_holm_ext'))
    for r in prim:
        r['verdict'] = L.verdict(r['diff'], r['p_holm'])
        r['verdict_ext'] = L.verdict(r['diff'], r['p_holm_ext'])
        r['ci_disagree'] = bool(L.ci_disagrees(r, 'p_holm'))

    # ---------------- tallies
    tallies = {}
    for key in ('verdict_ext', 'verdict_ns', 'verdict_raw'):
        t = defaultdict(Counter)
        for r in ns:
            t['%s|%s' % (r['arm'], r['mclass'])][r[key]] += 1
            t['%s|all' % r['arm']][r[key]] += 1
            t['all|%s' % r['mclass']][r[key]] += 1
            t['all|all'][r[key]] += 1
            t['%s|%s' % (r['arm'], r['metric'])][r[key]] += 1
        tallies[key] = {k: dict(v) for k, v in sorted(t.items())}
    n_ext_vs_ns = sum(r['verdict_ext'] != r['verdict_ns'] for r in ns)
    n_ext_vs_raw = sum(r['verdict_ext'] != r['verdict_raw'] for r in ns)
    prim_changed = [r for r in prim if r['verdict'] != r['verdict_ext']]

    # by backbone within the aggregator arm (useful for the text)
    by_bk = defaultdict(Counter)
    for r in ns:
        by_bk['%s|%s|%s' % (r['arm'], r['backbone'], r['mclass'])][r['verdict_ext']] += 1

    # ---------------- task 5
    dis_prim = [r for r in prim if r['ci_disagree']]
    dis_ns = [r for r in ns if r['ci_disagree_ext']]

    # ---------------- print
    print('\nTask 1: %d within-model N-vs-S contrasts (aggregator %d, relocation %d)'
          % (len(ns), sum(r['arm'] == 'aggregator' for r in ns),
             sum(r['arm'] == 'relocation' for r in ns)))
    for key, lab in (('verdict_ext', 'Holm, EXT family (primary + N-vs-S, 5/10)'),
                     ('verdict_ns', 'Holm, N-vs-S-only family (1/2)'),
                     ('verdict_raw', 'raw P < 0.05 (no Holm)')):
        print('  %s' % lab)
        for k in ('aggregator|maxloc', 'aggregator|extent', 'aggregator|all',
                  'relocation|maxloc', 'relocation|extent', 'relocation|all',
                  'all|maxloc', 'all|extent', 'all|all'):
            c = tallies[key].get(k, {})
            print('    %-20s N better %2d  S better %2d  no difference %2d  (n=%d)'
                  % (k, c.get('N better', 0), c.get('S better', 0),
                     c.get('no difference', 0), sum(c.values())))
    print('  verdicts differing EXT vs NS-only family: %d; EXT vs raw: %d'
          % (n_ext_vs_ns, n_ext_vs_raw))
    print('  primary verdicts changed by enlarging the family to EXT: %d' % len(prim_changed))
    for r in prim_changed:
        print('    %s %s %s %s vs %s  p_holm %.4g -> %.4g  %s -> %s'
              % (r['dataset'], r['backbone'], r['metric'], r['a'], r['b'],
                 r['p_holm'], r['p_holm_ext'], r['verdict'], r['verdict_ext']))
    print('  by arm x backbone x class (EXT):')
    for k in sorted(by_bk):
        print('    %-32s %s' % (k, dict(by_bk[k])))
    print('\n  all N-vs-S rows:')
    for r in ns:
        print('    %-8s %-8s %-15s %-6s N %.4f S %.4f d %+.4f [%+.4f, %+.4f] %-8s p %.3g'
              ' holm_ext %.3g holm_ns %.3g  %s%s'
              % (r['dataset'], r['backbone'], r['metric'], r['a'], r['va'], r['vb'],
                 r['diff'], r['ci'][0], r['ci'][1], r['test'], r['p'], r['p_holm_ext'],
                 r['p_holm_ns'], r['verdict_ext'],
                 '  [CI disagrees]' if r['ci_disagree_ext'] else ''))

    print('\nTask 5: primary contrasts whose Holm verdict and unadjusted CI disagree: %d of %d'
          % (len(dis_prim), len(prim)))
    for r in dis_prim:
        print('    %-8s %-8s %-15s %-6s vs %-9s d %+.4f [%+.4f, %+.4f] %-16s p %.4g p_holm %.4g -> %s'
              % (r['dataset'], r['backbone'], r['metric'], r['a'], r['b'], r['diff'],
                 r['ci'][0], r['ci'][1], r['test'], r['p'], r['p_holm'], r['verdict']))
    print('  N-vs-S rows with the same disagreement (EXT Holm): %d of %d' % (len(dis_ns), len(ns)))

    L.save('l1_ns_contrasts.json', dict(
        design=__doc__, reproduction=rep, ns_rows=ns, primary_rows=prim,
        tallies=tallies, by_arm_backbone_class={k: dict(v) for k, v in by_bk.items()},
        n_verdict_ext_vs_ns=n_ext_vs_ns, n_verdict_ext_vs_raw=n_ext_vs_raw,
        primary_changed_by_ext=[{k: r[k] for k in ('dataset', 'backbone', 'metric', 'a', 'b',
                                                   'p_holm', 'p_holm_ext', 'verdict',
                                                   'verdict_ext')} for r in prim_changed],
        task5_primary_disagreements=dis_prim, task5_ns_disagreements=dis_ns))

    # ---------------- LaTeX
    rows = []
    for r in ns:
        rows.append(' & '.join([
            L.DSNAME[r['dataset']], L.BKNAME[r['backbone']],
            'aggregator' if r['arm'] == 'aggregator' else 'relocation',
            L.MNAME[r['metric']], L.f4(r['va']), L.f4(r['vb']),
            '%s [%s, %s]' % (L.fs(r['diff']), L.fs(r['ci'][0]), L.fs(r['ci'][1])),
            L.fp(r['p_holm_ext']),
            r['verdict_ext'] + ('$^{\\dagger}$' if r['ci_disagree_ext'] else '')]))
    note = ('N and S are the necessity and sufficiency maps of the same model (aggregator arm: '
            'log-sum-exp pooling at $T=+1$ with the reused head; relocation arm: ViT-B/16 with '
            'the layer normalisation after pooling and a refitted head). The difference is N '
            '$-$ S with its unadjusted 95\\%% percentile bootstrap interval (%d resamples; %d '
            'for MaxBoxAccV2). Tests as for the primary contrasts: paired image bootstrap '
            '(MaxBoxAccV2), exact McNemar (pointing), Wilcoxon signed-rank (best IoU, peak-mask '
            'IoU). Holm within each dataset, backbone and metric family, here enlarged from the '
            'four (CNN) or eight (ViT) primary contrasts by the one or two N-vs-S contrasts. '
            '$^{\\dagger}$Holm verdict and interval disagree.'
            % (L.W.PROTOCOL['n_boot'], L.W.PROTOCOL['n_boot_mbav2']))
    L.longtable(os.path.join(L.OUT, 'tab_l1_ns_contrasts.tex'), 'llllrrlrl',
                'Dataset & Backbone & Arm & Metric & N & S & N $-$ S [95\\% CI] & '
                '$P_\\mathrm{Holm}$ & Verdict', rows,
                'Within-model necessity versus sufficiency contrasts on the localisation '
                'benchmarks.', 'tab:ns_within', note)

    # tally table
    t = tallies['verdict_ext']
    trow = []
    for arm in ('aggregator', 'relocation', 'all'):
        for cl, cname in (('maxloc', 'maximum location'), ('extent', 'extent'), ('all', 'all')):
            c = t.get('%s|%s' % (arm, cl), {})
            if not c:
                continue
            trow.append('%s & %s & %d & %d & %d & %d' % (
                arm, cname, sum(c.values()), c.get('N better', 0), c.get('S better', 0),
                c.get('no difference', 0)))
    L.longtable(os.path.join(L.OUT, 'tab_l1_ns_tally.tex'), 'llrrrr',
                'Arm & Metric class & $n$ & N better & S better & No difference', trow,
                'Tally of the within-model necessity versus sufficiency contrasts '
                '(Holm, $\\alpha = 0.05$).', 'tab:ns_within_tally',
                'Maximum location: the two pointing games; extent: MaxBoxAccV2, best IoU, '
                'peak-mask IoU. Counts over dependent contrasts, descriptive only; '
                '``no difference'' means not significant after Holm, not equivalence.')


if __name__ == '__main__':
    main()
