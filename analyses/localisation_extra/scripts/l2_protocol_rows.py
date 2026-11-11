"""Task 2 -- protocol-required localisation rows that the article does not report.

The written protocol (protocols/wsol_protocol.md
section 2) names: both perturbation scales "reported everywhere" (1x1 and 3x3
blocks), the trivial controls center, actnorm (activation magnitude ||x_t||_2) and
random, the question "does either beat the trivial controls", and a second
reference basis (the dataset-mean cell broadcast) as a robustness row.

All tests are the w3_stats.py tests (llib.test), Holm within each
(dataset, backbone, metric) family over exactly the rows of the set named below,
alpha = 0.05, verdict = sign of the difference (map a minus comparator b).

SET B3   3x3-block analogue of the primary set, mirrored one-to-one:
         (N_lse_b3, collapsed_b3) (S_lse_b3, collapsed_b3) (N_lse_b3, center)
         (S_lse_b3, center) and on ViT-B/16 (N_ln_b3, collapsed_b3)
         (S_ln_b3, collapsed_b3) (N_ln_b3, center) (S_ln_b3, center).
         Family = the 4 (CNN) or 8 (ViT) rows of one cell -> 192 rows.
         Within-model N_b3 vs S_b3 (1 or 2 rows per cell, own family) -> 48 rows.
SET CTL  separated 1x1 maps vs the two secondary trivial controls:
         {N_lse, S_lse (, N_ln, S_ln)} x {actnorm, random}.
         Family = the 4 (CNN) or 8 (ViT) rows of one cell -> 192 rows.
         Sensitivity: family = all three trivial controls, i.e. the CTL rows plus
         the primary centre rows (6 / 12 per cell).
         Context: collapsed vs actnorm and vs random (family of 2 per cell).
SET B3CTL the same at the 3x3 scale: {N_lse_b3, S_lse_b3 (, N_ln_b3, S_ln_b3)} x
         {actnorm, random}, family of 4/8 per cell-metric; context collapsed_b3
         vs actnorm / random (family of 2).  (The controls have no 3x3 version.)
SET MC   dataset-mean-cell basis (run only for cub|resnet50 and voc12seg|resnet50):
         the 4 primary pairs (family of 4 per cell-metric), their 3x3 analogue
         (family of 4) and N_lse vs S_lse (family of 1), with the verdict of the
         same row under the bank basis beside it.
"""
import os
from collections import Counter, defaultdict
import numpy as np
import llib as L

B3_PAIRS = [('N_lse_b3', 'collapsed_b3'), ('S_lse_b3', 'collapsed_b3'),
            ('N_lse_b3', 'center'), ('S_lse_b3', 'center'),
            ('N_ln_b3', 'collapsed_b3'), ('S_ln_b3', 'collapsed_b3'),
            ('N_ln_b3', 'center'), ('S_ln_b3', 'center')]
B3_NS = [('N_lse_b3', 'S_lse_b3'), ('N_ln_b3', 'S_ln_b3')]
CTL_PAIRS = [(m, c) for c in ('actnorm', 'random') for m in ('N_lse', 'S_lse', 'N_ln', 'S_ln')]
CTR_PAIRS = [(m, 'center') for m in ('N_lse', 'S_lse', 'N_ln', 'S_ln')]
CTX_PAIRS = [('collapsed', 'actnorm'), ('collapsed', 'random')]
B3CTL_PAIRS = [(m, c) for c in ('actnorm', 'random')
               for m in ('N_lse_b3', 'S_lse_b3', 'N_ln_b3', 'S_ln_b3')]
B3CTX_PAIRS = [('collapsed_b3', 'actnorm'), ('collapsed_b3', 'random')]
MC_PRIM = [('N_lse', 'collapsed'), ('S_lse', 'collapsed'), ('N_lse', 'center'), ('S_lse', 'center')]
MC_B3 = [('N_lse_b3', 'collapsed_b3'), ('S_lse_b3', 'collapsed_b3'),
         ('N_lse_b3', 'center'), ('S_lse_b3', 'center')]
MC_NS = [('N_lse', 'S_lse')]
MC_CELLS = [('cub', 'resnet50'), ('voc12seg', 'resnet50')]


def run_set(pairs, cells, suf='', key='p_holm', cache=None):
    rows = []
    for ds, bk in cells:
        for metric in L.METRICS[ds]:
            fam = []
            for a, b in pairs:
                ck = (ds, bk, suf, metric, a, b)
                if cache is not None and ck in cache:
                    t = cache[ck]
                else:
                    t = L.test(ds, bk, metric, a, b, suf)
                    if cache is not None:
                        cache[ck] = t
                if t is None:
                    continue
                fam.append(dict(dataset=ds, backbone=bk, basis=suf.strip('_') or 'bank',
                                metric=metric, mclass=L.mclass(metric), a=a, b=b,
                                arm=L.arm_of(a), **t))
            adj = L.holm([r['p'] for r in fam]) if fam else []
            for r, q in zip(fam, adj):
                r[key] = q
                r['family_size'] = len(fam)
                r['verdict'] = L.verdict(r['diff'], q)
                r['verdict_raw'] = L.verdict(r['diff'], r['p'])
                r['ci_disagree'] = bool(L.ci_disagrees(r, key))
            rows += fam
    return rows


def tally(rows, keyf, vkey='verdict'):
    t = defaultdict(Counter)
    for r in rows:
        for k in keyf(r):
            t[k][r[vkey]] += 1
    return {k: dict(better=v.get('better', 0), worse=v.get('worse', 0),
                    no_difference=v.get('no difference', 0), n=sum(v.values()))
            for k, v in sorted(t.items())}


def show(title, tl):
    print(title)
    for k, v in tl.items():
        print('    %-40s n=%-4d better %-4d worse %-4d no difference %-4d'
              % (k, v['n'], v['better'], v['worse'], v['no_difference']))


def half(a):
    return a.split('_')[0]           # 'N' or 'S'


def tex_rows(rows, with_basis_ref=False):
    out = []
    for r in rows:
        cells = [L.DSNAME[r['dataset']], L.BKNAME[r['backbone']], L.MNAME[r['metric']],
                 '\\texttt{%s}' % L.tex_escape(r['a']), '\\texttt{%s}' % L.tex_escape(r['b']),
                 L.f4(r['va']), L.f4(r['vb']),
                 '%s [%s, %s]' % (L.fs(r['diff']), L.fs(r['ci'][0]), L.fs(r['ci'][1])),
                 L.fp(r['p_holm']),
                 r['verdict'] + ('$^{\\dagger}$' if r['ci_disagree'] else '')]
        if with_basis_ref:
            cells.append(r.get('verdict_bank', '--'))
        out.append(' & '.join(cells))
    return out


def main():
    cells = [(ds, bk) for ds in L.DATASETS for bk in L.BACKBONES]
    cache = {}
    w3 = L.load_w3()

    # ------------------------------------------------------------------ B3
    b3 = run_set(B3_PAIRS, cells, cache=cache)
    b3ns = run_set(B3_NS, cells, cache=cache)
    # reproduction of the rows w3 already holds (raw p)
    rep = []
    for r in b3 + b3ns:
        for o in w3['%s|%s' % (r['dataset'], r['backbone'])]['contrasts'][r['metric']]:
            if (o['a'], o['b']) == (r['a'], r['b']):
                rep.append((abs(o['diff'] - r['diff']), abs(o['p'] - r['p'])))
    print('B3: %d rows (+%d N-vs-S); %d of them also in w3_stats.json, max |diff| %.1e, '
          'max |p| %.1e' % (len(b3), len(b3ns), len(rep), max(x for x, _ in rep),
                            max(y for _, y in rep)))
    t_b3 = tally(b3, lambda r: ['all', 'vs %s' % r['b'],
                                '%s | vs %s' % (r['arm'], r['b']),
                                '%s | vs %s | %s' % (r['a'], r['b'], r['mclass']),
                                '%s | %s' % (r['a'], r['b'])])
    show('  SET B3 (Holm, family = 4/8 rows per cell-metric)', t_b3)
    t_b3_raw = tally(b3, lambda r: ['%s | %s' % (r['a'], r['b'])], 'verdict_raw')
    show('  SET B3, raw P < 0.05 (for comparison with the reader count)', t_b3_raw)
    t_b3ns = tally(b3ns, lambda r: ['%s | %s' % (r['arm'], r['mclass']), r['arm']])
    show('  SET B3 within-model N_b3 vs S_b3 (better = N better)', t_b3ns)
    n_lse_b3 = sum(1 for r in b3 + b3ns if '_lse_b3' in r['a'] or '_lse_b3' in r['b'])

    # ------------------------------------------------------------------ CTL
    ctl = run_set(CTL_PAIRS, cells, cache=cache)
    ctl_sens = [r for r in run_set(CTL_PAIRS + CTR_PAIRS, cells, key='p_holm', cache=cache)
                if r['b'] != 'center']
    for r, s in zip(ctl, ctl_sens):
        assert (r['dataset'], r['backbone'], r['metric'], r['a'], r['b']) == \
            (s['dataset'], s['backbone'], s['metric'], s['a'], s['b'])
        r['p_holm_sens'] = s['p_holm']
        r['verdict_sens'] = s['verdict']
    ctx = run_set(CTX_PAIRS, cells, cache=cache)
    t_ctl = tally(ctl, lambda r: ['all', 'vs %s' % r['b'], '%s | vs %s' % (r['a'], r['b']),
                                  '%s | vs %s | %s' % (r['arm'], r['b'], r['mclass'])])
    show('\n  SET CTL (Holm, family = 4/8 rows per cell-metric)', t_ctl)
    n_sens = sum(r['verdict'] != r['verdict_sens'] for r in ctl)
    print('  CTL verdicts that change when the family also holds the centre rows: %d' % n_sens)
    t_ctx = tally(ctx, lambda r: ['%s | %s' % (r['a'], r['b']),
                                  '%s | %s | %s' % (r['a'], r['b'], r['mclass'])])
    show('  context: collapsed vs actnorm / random (family of 2)', t_ctx)
    b3ctl = run_set(B3CTL_PAIRS, cells, cache=cache)
    b3ctx = run_set(B3CTX_PAIRS, cells, cache=cache)
    t_b3ctl = tally(b3ctl, lambda r: ['all', 'vs %s' % r['b'], '%s | vs %s' % (r['a'], r['b']),
                                      '%s | vs %s | %s' % (r['arm'], r['b'], r['mclass'])])
    show('  SET B3CTL (3x3 maps vs actnorm / random; family 4/8)', t_b3ctl)
    t_b3ctx = tally(b3ctx, lambda r: ['%s | %s' % (r['a'], r['b'])])
    show('  context: collapsed_b3 vs actnorm / random (family of 2)', t_b3ctx)
    worse = [r for r in ctl if r['verdict'] != 'better']
    print('  CTL rows that are NOT "better":')
    for r in worse:
        print('    %-8s %-8s %-15s %-6s vs %-8s %.4f vs %.4f d %+.4f [%+.4f, %+.4f] p_holm %.3g %s'
              % (r['dataset'], r['backbone'], r['metric'], r['a'], r['b'], r['va'], r['vb'],
                 r['diff'], r['ci'][0], r['ci'][1], r['p_holm'], r['verdict']))

    # ------------------------------------------------------------------ MC
    mc = run_set(MC_PRIM, MC_CELLS, suf='_meancell', cache=cache)
    mcb3 = run_set(MC_B3, MC_CELLS, suf='_meancell', cache=cache)
    mcns = run_set(MC_NS, MC_CELLS, suf='_meancell', cache=cache)
    # same rows under the bank basis
    bank_prim = {(r['dataset'], r['backbone'], r['metric'], r['a'], r['b']): r
                 for r in run_set(MC_PRIM, MC_CELLS, cache=cache)}
    bank_b3 = {(r['dataset'], r['backbone'], r['metric'], r['a'], r['b']): r
               for r in run_set(MC_B3, MC_CELLS, cache=cache)}
    bank_ns = {(r['dataset'], r['backbone'], r['metric'], r['a'], r['b']): r
               for r in run_set(MC_NS, MC_CELLS, cache=cache)}
    for rows, bank in ((mc, bank_prim), (mcb3, bank_b3), (mcns, bank_ns)):
        for r in rows:
            o = bank[(r['dataset'], r['backbone'], r['metric'], r['a'], r['b'])]
            r['verdict_bank'] = o['verdict']; r['diff_bank'] = o['diff']
            r['va_bank'] = o['va']; r['vb_bank'] = o['vb']
    t_mc = tally(mc, lambda r: ['all', 'vs %s' % r['b'], '%s | %s' % (r['dataset'], r['b'])])
    show('\n  SET MC primary analogue (meancell basis, family of 4)', t_mc)
    agree = sum(r['verdict'] == r['verdict_bank'] for r in mc)
    print('  MC verdict == bank verdict: %d of %d' % (agree, len(mc)))
    for r in mc:
        print('    %-8s %-15s %-6s vs %-10s mc %.4f vs %.4f d %+.4f [%+.4f,%+.4f] p_holm %.3g %-13s | bank d %+.4f %s'
              % (r['dataset'], r['metric'], r['a'], r['b'], r['va'], r['vb'], r['diff'],
                 r['ci'][0], r['ci'][1], r['p_holm'], r['verdict'], r['diff_bank'],
                 r['verdict_bank']))
    t_mcb3 = tally(mcb3, lambda r: ['all', 'vs %s' % r['b']])
    show('  SET MC 3x3 analogue (family of 4)', t_mcb3)
    print('  MC b3 verdict == bank verdict: %d of %d'
          % (sum(r['verdict'] == r['verdict_bank'] for r in mcb3), len(mcb3)))
    show('  SET MC N_lse vs S_lse (better = N better)', tally(mcns, lambda r: ['all']))
    for r in mcns:
        print('    %-8s %-15s N %.4f S %.4f d %+.4f [%+.4f,%+.4f] p %.3g %s | bank %s'
              % (r['dataset'], r['metric'], r['va'], r['vb'], r['diff'], r['ci'][0],
                 r['ci'][1], r['p'], r['verdict'], r['verdict_bank']))
    # the collapsed map under the meancell basis vs plain CAM (identity check)
    cc = []
    for ds, bk in MC_CELLS:
        for metric in L.METRICS[ds]:
            xa = L.per_record(ds, bk, metric, 'collapsed', '_meancell')
            xb = L.per_record(ds, bk, metric, 'cam', '_meancell')
            cc.append(float(np.max(np.abs(np.asarray(xa, float) - np.asarray(xb, float)))))
    print('  meancell collapsed vs cam, max |per-record difference| over the 7 metric '
          'arrays: %.3g' % max(cc))

    dis = dict(B3=sum(r['ci_disagree'] for r in b3), B3_NS=sum(r['ci_disagree'] for r in b3ns),
               CTL=sum(r['ci_disagree'] for r in ctl),
               B3CTL=sum(r['ci_disagree'] for r in b3ctl), MC=sum(r['ci_disagree'] for r in mc),
               MC_B3=sum(r['ci_disagree'] for r in mcb3))
    print('\n  Holm verdict vs CI disagreements per set:', dis)

    L.save('l2_protocol_rows.json', dict(
        design=__doc__, b3=b3, b3_ns=b3ns, ctl=ctl, ctx=ctx, b3ctl=b3ctl, b3ctx=b3ctx,
        tally_b3ctl=t_b3ctl, tally_b3ctx=t_b3ctx, mc=mc, mc_b3=mcb3, mc_ns=mcns,
        tally_b3=t_b3, tally_b3_raw=t_b3_raw, tally_b3_ns=t_b3ns, tally_ctl=t_ctl,
        tally_ctx=t_ctx, tally_mc=t_mc, tally_mc_b3=t_mcb3, n_ctl_sens_changed=n_sens,
        n_rows_using_lse_b3=n_lse_b3, mc_agree_bank=agree,
        meancell_collapsed_eq_cam_maxdiff=max(cc), ci_disagreements=dis,
        reproduction_w3=dict(n=len(rep), max_abs_diff=max(x for x, _ in rep),
                             max_abs_p=max(y for _, y in rep))))

    # ------------------------------------------------------------------ LaTeX
    hdr = ('Dataset & Backbone & Metric & Map & Comparator & Map & Comp. & '
           'Difference [95\\% CI] & $P_\\mathrm{Holm}$ & Verdict')
    cs = 'lllllrrlrl'
    nb = (L.W.PROTOCOL['n_boot'], L.W.PROTOCOL['n_boot_mbav2'])
    common = ('Tests as for the primary contrasts (paired image bootstrap for MaxBoxAccV2, exact '
              'McNemar for pointing, Wilcoxon signed-rank for best IoU and peak-mask IoU); '
              'unadjusted 95\\%% percentile bootstrap intervals (%d resamples, %d for '
              'MaxBoxAccV2); verdict = sign of map $-$ comparator when $P_\\mathrm{Holm}<0.05$. '
              '$^{\\dagger}$Holm verdict and interval disagree. ' % nb)
    L.longtable(os.path.join(L.OUT, 'tab_l2_b3.tex'), cs, hdr, tex_rows(b3 + b3ns),
                'Protocol rows at the $3\\times3$-block perturbation scale.', 'tab:loc_b3',
                common + 'Holm within each dataset, backbone and metric family over the four '
                '(CNN) or eight (ViT) rows that mirror the primary set; the within-model '
                '\\texttt{N\\_b3} vs \\texttt{S\\_b3} rows form their own family. The '
                'log-sum-exp $3\\times3$ maps (\\texttt{N\\_lse\\_b3}, \\texttt{S\\_lse\\_b3}) were '
                'computed with an integral-image block sum of exponentials that is numerically '
                'unstable; their rows are reported as archived and are indicative only (Section 2.11).')
    L.longtable(os.path.join(L.OUT, 'tab_l2_controls.tex'), cs, hdr, tex_rows(ctl + ctx + b3ctl + b3ctx),
                'Separated maps against the secondary trivial controls (activation magnitude, '
                'random).', 'tab:loc_controls',
                common + 'Holm within each dataset, backbone and metric family over the four '
                '(CNN) or eight (ViT) rows \\{separated maps\\} $\\times$ \\{activation '
                'magnitude, random\\}; the two collapsed-map context rows form their own family.')
    L.longtable(os.path.join(L.OUT, 'tab_l2_meancell.tex'), cs + 'l', hdr + ' & Bank basis',
                tex_rows(mc + mcb3 + mcns, True),
                'Robustness rows with the dataset-mean cell as the reference '
                '(ResNet-50; CUB and VOC masks only).', 'tab:loc_meancell',
                common + 'Holm within each dataset, metric and row-group family (four primary '
                'analogues; four $3\\times3$ analogues; one N-vs-S row). The last column gives '
                'the verdict of the same row with the eight-image reference bank.')

    # compact tally table for the three sets
    trow = []
    def add(setname, rows, keyf, labf):
        for k, v in tally(rows, keyf).items():
            trow.append('%s & %s & %d & %d & %d & %d' % (setname, labf(k), v['n'], v['better'],
                                                       v['worse'], v['no_difference']))
    add('$3\\times3$', b3, lambda r: ['%s|%s' % (r['a'], r['b'])],
        lambda k: '\\texttt{%s} vs \\texttt{%s}' % tuple(L.tex_escape(x) for x in k.split('|')))
    add('$3\\times3$, N vs S', b3ns, lambda r: ['%s|%s' % (r['a'], r['b'])],
        lambda k: '\\texttt{%s} vs \\texttt{%s}' % tuple(L.tex_escape(x) for x in k.split('|')))
    add('controls', ctl, lambda r: ['%s|%s' % (r['a'], r['b'])],
        lambda k: '\\texttt{%s} vs \\texttt{%s}' % tuple(L.tex_escape(x) for x in k.split('|')))
    add('context', ctx, lambda r: ['%s|%s' % (r['a'], r['b'])],
        lambda k: '\\texttt{%s} vs \\texttt{%s}' % tuple(L.tex_escape(x) for x in k.split('|')))
    add('$3\times3$ controls', b3ctl, lambda r: ['%s|%s' % (r['a'], r['b'])],
        lambda k: '\texttt{%s} vs \texttt{%s}' % tuple(L.tex_escape(x) for x in k.split('|')))
    add('$3\times3$ context', b3ctx, lambda r: ['%s|%s' % (r['a'], r['b'])],
        lambda k: '\texttt{%s} vs \texttt{%s}' % tuple(L.tex_escape(x) for x in k.split('|')))
    add('mean cell', mc + mcb3 + mcns, lambda r: ['%s|%s' % (r['a'], r['b'])],
        lambda k: '\\texttt{%s} vs \\texttt{%s}' % tuple(L.tex_escape(x) for x in k.split('|')))
    L.longtable(os.path.join(L.OUT, 'tab_l2_tally.tex'), 'llrrrr',
                'Set & Contrast & $n$ & Better & Worse & No difference', trow,
                'Tally of the protocol rows not reported in the article.', 'tab:loc_protocol_tally',
                'Better / worse: the first map minus the comparator is positive / negative with '
                '$P_\\mathrm{Holm} < 0.05$ (for N vs S rows: better = necessity higher). Counts '
                'over dependent contrasts, descriptive only.')


if __name__ == '__main__':
    main()
