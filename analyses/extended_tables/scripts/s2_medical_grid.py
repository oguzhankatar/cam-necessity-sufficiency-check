"""S2 -- every medical grid cell for supplementary Table S7, and the counts the
main text quotes.

Sources (read only):
  outputs/medical/{m4_stats,m10_condB,m6_own,m11_rsna,m5_isic}.json  'prespecified'
      -> H1 (N - collapsed) and H2 (N - S) cells of the original runs
  analyses/medical_baselines/out/t6_score.json  'grid'
      -> Sminus_lse8.0_* (S(+8) - collapsed) for every arm;
         every cell of the cross-fitted disjoint RSNA arm;
         the replicate-run value of every cell of nih_m10, nih_m6 and isic.
Holm over the 48 H1 cells of the five original arms, with the P each cell's test
gives (exact McNemar for hit, Wilcoxon signed-rank otherwise; for the hit cells of
the first ISIC run, whose per-image outcomes were not stored, the exact McNemar P
is computed from the discordant counts that the stored Wilcoxon P and mean
difference determine, see mcnemar_from_wilcoxon).
Outputs: ../out/s2_medical_grid.json, ../out/s7_rows.tex
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *['..'] * 3))
MED = os.path.join(ROOT, 'outputs', 'medical')
T6 = os.path.join(ROOT, 'analyses', 'medical_baselines', 'out', 't6_score.json')
OUT = os.path.abspath(os.path.join(HERE, '..', 'out'))
os.makedirs(OUT, exist_ok=True)

# arm key in t6_score, label, archived json, status, source of S(+8)-collapsed cells
ARMS = [('nih_m4', 'NIH, distributed (reused) head', 'm4_stats.json', 'pre-specified', 'M'),
        ('rsna_xrv', 'RSNA, distributed model', 'm11_rsna.json', 'pre-specified', 'R'),
        ('isic', 'ISIC', 'm5_isic.json', 'pre-specified', 'P'),
        ('nih_m10', 'NIH, refitted head', 'm10_condB.json', 'post hoc', 'P'),
        ('nih_m6', 'NIH, model trained in this study', 'm6_own.json', 'post hoc', 'P'),
        ('rsna_disjoint', 'RSNA, disjoint models', None, 'post hoc', 'D')]
REPL = {'nih_m10', 'nih_m6', 'isic'}
METRICS = {'nih_m4': ['hit', 'ap', 'iou'], 'nih_m10': ['hit', 'ap'], 'nih_m6': ['hit', 'ap'],
           'rsna_xrv': ['hit', 'ap'], 'rsna_disjoint': ['hit', 'ap'], 'isic': ['hit', 'ap', 'dice']}
MLAB = {'hit': 'hit', 'ap': 'AP', 'iou': 'IoU', 'dice': 'Dice'}


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    out = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, run)
    return out


def mcnemar_from_wilcoxon(c):
    """Exact McNemar P for a paired binary cell whose archive kept only the Wilcoxon P.

    For per-item hits, a - b lies in {-1, 0, +1}.  The Wilcoxon signed-rank test
    (zero_method='wilcox', no continuity correction) drops the zeros and, all
    remaining |d| being tied, reduces to z = (b - c) / sqrt(b + c), where b and c
    are the discordant counts.  b - c = diff * n is known, so b + c is the unique
    integer that reproduces the archived P.  The reconstruction is checked by
    re-running scipy's Wilcoxon on the rebuilt difference vector."""
    from scipy.stats import norm, wilcoxon, binomtest
    n, p = c['n'], c['p_wilcoxon']
    net = int(round(c['diff'] * n))
    if net == 0:
        return 1.0, None
    z = norm.isf(p / 2)
    tot = int(round((net / z) ** 2))
    b, cc = (tot + net) // 2, (tot - net) // 2
    assert b + cc == tot and b - cc == net and 0 <= cc and tot <= n
    d = np.r_[np.ones(b), -np.ones(cc), np.zeros(n - tot)]
    p_back = float(wilcoxon(d, zero_method='wilcox').pvalue)
    assert abs(np.log(p_back) - np.log(p)) < 1e-6, (p, p_back)
    return float(binomtest(min(b, cc), tot, 0.5).pvalue), (b, cc)


def pval(c):
    return c.get('p_mcnemar', c.get('p_wilcoxon'))


def verdict(c):
    return 'above' if c['lo'] > 0 else ('below' if c['hi'] < 0 else 'interval spans 0')


def main():
    R = json.load(open(T6))
    rows = []
    for arm, lab, js, status, ssrc in ARMS:
        A = json.load(open(os.path.join(MED, js)))['prespecified'] if js else None
        g = R[arm]['grid']
        for hyp in ('H1', 'H2', 'S'):
            for temp in ('lse8.0', 'lse-8.0'):
                if hyp == 'S' and temp != 'lse8.0':
                    continue
                for unit in ('1x1', '3x3'):
                    for mt in METRICS[arm]:
                        if hyp == 'S':
                            k = 'Sminus_%s_%s_%s' % (temp, unit, mt)
                            c, src = g[k], ssrc
                        else:
                            k = '%s_%s_%s_%s' % (hyp, temp, unit, mt)
                            if A is not None:
                                c, src = A[k], 'O'
                                if mt == 'hit' and 'p_mcnemar' not in c:
                                    c = dict(c)
                                    c['p_mcnemar'], c['discordant'] = mcnemar_from_wilcoxon(c)
                            else:
                                c, src = g[k], 'D'
                        rep = g[k]['diff'] if (arm in REPL and hyp != 'S') else None
                        rows.append(dict(arm=arm, label=lab, status=status, hyp=hyp,
                                         T=('+8' if temp == 'lse8.0' else '-8'), unit=unit, metric=mt,
                                         diff=c['diff'], lo=c['lo'], hi=c['hi'], p=pval(c),
                                         test='exact McNemar' if 'p_mcnemar' in c else 'Wilcoxon',
                                         n=c['n'], source=src, replicate_diff=rep, key=k,
                                         verdict=verdict(c), discordant=c.get('discordant'),
                                         p_wilcoxon_archived=(c['p_wilcoxon'] if c.get('discordant')
                                                              else None)))
    # Holm over the 48 H1 cells of the five original arms, in the order of the arms above
    h1 = []
    for arm, lab, js, status, ssrc in ARMS:
        if js:
            for k in json.load(open(os.path.join(MED, js)))['prespecified']:
                if k.startswith('H1'):
                    h1.append((arm, k))
    ph = holm([next(r['p'] for r in rows if r['arm'] == a and r['key'] == k) for a, k in h1])
    for (a, k), p in zip(h1, ph):
        r = next(r for r in rows if r['arm'] == a and r['key'] == k)
        r['p_holm48'] = float(p)
    # ---------------------------------------------------------------- counts
    C = {}
    H1 = [r for r in rows if 'p_holm48' in r]
    C['n_H1_48'] = len(H1)
    C['H1_favour_necessity'] = sum(r['lo'] > 0 for r in H1)
    C['H1_favour_all_at_T-8'] = all(r['T'] == '-8' for r in H1 if r['lo'] > 0)
    C['H1_favour_and_holm'] = sum(r['lo'] > 0 and r['p_holm48'] < 0.05 for r in H1)
    C['H1_below'] = sum(r['hi'] < 0 for r in H1)
    tp8 = [r for r in H1 if r['T'] == '+8']
    C['H1_T+8_n'] = len(tp8)
    C['H1_T+8_below'] = sum(r['hi'] < 0 for r in tp8)
    C['H1_T+8_diff_negative'] = sum(r['diff'] < 0 for r in tp8)
    tm8 = [r for r in H1 if r['T'] == '-8']
    C['H1_T-8_n'] = len(tm8)
    C['H1_T-8_above/span/below'] = [sum(r['lo'] > 0 for r in tm8),
                                    sum(r['lo'] <= 0 <= r['hi'] for r in tm8),
                                    sum(r['hi'] < 0 for r in tm8)]
    S = [r for r in rows if r['hyp'] == 'S']
    C['S_n'] = len(S)
    C['S_below'] = sum(r['hi'] < 0 for r in S)
    C['S_not_below'] = ['%s %s %s %+.3f [%+.3f, %+.3f]' % (r['arm'], r['unit'], r['metric'], r['diff'],
                                                            r['lo'], r['hi']) for r in S if not r['hi'] < 0]
    for part, st in (('prespecified_arms', 'pre-specified'), ('posthoc_arms', 'post hoc')):
        P = [r for r in H1 if r['status'] == st]
        ph_in = holm([r['p'] for r in P])
        C[part] = dict(arms=sorted(set(r['label'] for r in P)), n=len(P),
                       favour=sum(r['lo'] > 0 for r in P),
                       favour_holm48=sum(r['lo'] > 0 and r['p_holm48'] < 0.05 for r in P),
                       favour_holm_within_part=int(sum((r['lo'] > 0) and (q < 0.05) for r, q in zip(P, ph_in))),
                       below=sum(r['hi'] < 0 for r in P),
                       T_plus8_n=sum(r['T'] == '+8' for r in P),
                       T_minus8_n=sum(r['T'] == '-8' for r in P),
                       favour_cells=['%s %s %s %s %+.3f [%+.3f, %+.3f] Holm48 %.2g' % (
                           r['label'], r['T'], r['unit'], r['metric'], r['diff'], r['lo'], r['hi'],
                           r['p_holm48']) for r in P if r['lo'] > 0])
    D = [r for r in rows if r['arm'] == 'rsna_disjoint']
    C['disjoint'] = dict(
        n_cells=len(D),
        H1_Tminus8=['%s %s %+.3f [%+.3f, %+.3f]' % (r['unit'], r['metric'], r['diff'], r['lo'], r['hi'])
                    for r in D if r['hyp'] == 'H1' and r['T'] == '-8'])
    C['per_arm_cells'] = {lab: dict(H1=sum(r['hyp'] == 'H1' for r in rows if r['label'] == lab),
                                    H2=sum(r['hyp'] == 'H2' for r in rows if r['label'] == lab),
                                    S=sum(r['hyp'] == 'S' for r in rows if r['label'] == lab),
                                    metrics=METRICS[a])
                          for a, lab, *_ in ARMS}
    C['n_rows'] = len(rows)
    # replicate movements quoted in the text
    C['isic_repl'] = {r['key']: [r['diff'], r['replicate_diff']] for r in rows
                      if r['arm'] == 'isic' and r['hyp'] == 'H1' and r['T'] == '-8'}
    C['m6_repl'] = {r['key']: [r['diff'], r['replicate_diff']] for r in rows
                    if r['arm'] == 'nih_m6' and r['hyp'] == 'H1'}
    json.dump(dict(rows=rows, counts=C), open(os.path.join(OUT, 's2_medical_grid.json'), 'w'),
              indent=1, default=float)
    print(json.dumps(C, indent=1, default=float))
    # ---------------------------------------------------------------- LaTeX rows
    f3 = lambda x: ('+%.3f' % x if x >= 0 else '$-$%.3f' % -x)

    def fp(p):
        if p is None:
            return ''
        if p < 1e-300:
            return '$<$1e-300'
        s = '%.2g' % p
        if s == '0.05':                       # rounding would hide the side of 0.05
            s = '%.4f' % p
        elif p < 1e-3 and 'e' not in s and len(s.replace('0.', '', 1).lstrip('0')) < 2:
            s = '%.1e' % p                     # keep two significant figures
        if 'e' in s:
            m, e = s.split('e')
            s = '%se%s%02d' % (m, '-' if e[0] == '-' else '+', abs(int(e)))
            s = s.replace('e+', 'e')
        return s
    HY = {'H1': 'N vs collapsed', 'H2': 'N vs S', 'S': 'S vs collapsed'}
    ML = {'nih_m4': 'hit, AP, IoU', 'isic': 'hit, AP, Dice'}
    lines = []
    last = None
    n_dag = 0
    for r in rows:
        if r['arm'] != last:
            if last is not None:
                lines.append(r'\midrule')
            lines.append(r'\multicolumn{9}{@{}l}{\textit{%s} (%s; %s; \textit{n} = %s)} \\' % (
                r['label'], r['status'], ML.get(r['arm'], 'hit, AP'), '{:,}'.format(r['n'])))
        last = r['arm']
        holm_s = fp(r['p_holm48']) if 'p_holm48' in r else ''
        if 'p_holm48' in r and r['p_holm48'] < 0.05:
            holm_s = r'\textbf{%s}' % holm_s
        rep = f3(r['replicate_diff']) if r['replicate_diff'] is not None else ''
        T = '$+8$' if r['T'] == '+8' else '$-8$'
        # dagger: the bootstrap interval and the uncorrected test disagree
        dag = r'$^{\dagger}$' if ((r['lo'] > 0 or r['hi'] < 0) != (r['p'] < 0.05)) else ''
        n_dag += bool(dag)
        lines.append(' & '.join([HY[r['hyp']], T,
                                 r['unit'].replace('x', r'$\times$'), MLAB[r['metric']],
                                 '%s [%s, %s]%s' % (f3(r['diff']), f3(r['lo']), f3(r['hi']), dag),
                                 fp(r['p']), holm_s, r['source'], rep]) + r' \\')
    open(os.path.join(OUT, 's7_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    print('rows', len(rows), 'daggers', n_dag)
    print('ISIC first-run hit cells, exact McNemar from the Wilcoxon statistic:')
    for r in rows:
        if r.get('discordant') or r.get('p_wilcoxon_archived') is not None:
            print('  %s  discordant %s  Wilcoxon %.4g  exact McNemar %.4g' % (
                r['key'], r['discordant'], r['p_wilcoxon_archived'], r['p']))


if __name__ == '__main__':
    main()
