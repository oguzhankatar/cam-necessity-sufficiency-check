"""Task 3 -- H2 (necessity map beats sufficiency map, same T and unit).

The 32 pre-specified H2 cells are those of the three pre-specified arms:
NIH distributed head (2 T x 2 units x hit/AP/IoU = 12, outputs/medical/m4_stats.json),
RSNA distributed model (2 x 2 x hit/AP = 8, m11_rsna.json) and ISIC
(2 x 2 x hit/AP/Dice = 12, m5_isic.json), 'prespecified' blocks (= Table S7, Src O).
A cell favours N when its 95% bootstrap interval of N - S lies above zero, S when
it lies below, and spans zero otherwise; P (McNemar for hit, Wilcoxon otherwise)
is counted at 0.05 uncorrected as a second view.  Post hoc arms (NIH refitted head,
NIH own model: original runs; RSNA cross-fitted disjoint: C2 t6_score.json) are
tallied separately.  If out/t1_*_cf.json exist, the same tally is repeated with
the closed-form cells (NIH, RSNA) and the r4 ISIC replicate (forward differences
and closed form).
Output: out/t3_h2.json
"""
import os
import medical_extra_lib as R

PRE = [('NIH, distributed head', 'm4_stats.json', ['hit', 'ap', 'iou'], 'hit'),
       ('RSNA, distributed model', 'm11_rsna.json', ['hit', 'ap'], 'hit'),
       ('ISIC', 'm5_isic.json', ['hit', 'ap', 'dice'], 'dice')]
POST = [('NIH, refitted head', 'm10_condB.json', ['hit', 'ap'], 'hit'),
        ('NIH, own model', 'm6_own.json', ['hit', 'ap'], 'hit')]


def cls(c):
    return 'N' if c['lo'] > 0 else ('S' if c['hi'] < 0 else 'span')


def pv(c):
    return c.get('P', c.get('p_mcnemar', c.get('p_wilcoxon')))


def tally(cells):
    out = {}
    for tk, tl in (('all', None), ('T+8', 'lse8.0'), ('T-8', 'lse-8.0')):
        cc = [c for c in cells if tl is None or c['temp'] == tl]
        out[tk] = dict(n=len(cc), favour_N=sum(cls(c) == 'N' for c in cc),
                       favour_S=sum(cls(c) == 'S' for c in cc),
                       span=sum(cls(c) == 'span' for c in cc),
                       P_lt_05_N=sum(pv(c) < 0.05 and c['diff'] > 0 for c in cc),
                       P_lt_05_S=sum(pv(c) < 0.05 and c['diff'] < 0 for c in cc))
    return out


def cells_of(arm, pre, metrics):
    out = []
    for temp in R.OFF:
        for unit in ('1x1', '3x3'):
            for mt in metrics:
                k = 'H2_%s_%s_%s' % (temp, unit, mt)
                c = dict(pre[k])
                c.update(arm=arm, temp=temp, unit=unit, metric=mt, key=k,
                         P=c['P'] if 'P' in c else R.paper_p(c, k))
                out.append(c)
    return out


def main():
    res = {}
    pre_cells, prim = [], {}
    for arm, fn, mets, pm in PRE:
        A = R.jload(os.path.join(R.MOUT, fn))
        cc = cells_of(arm, A['prespecified'], mets)
        pre_cells += cc
        prim[arm] = {mt: R.brief(dict(A['prespecified']['H2_lse8.0_1x1_%s' % mt],
                                      P=R.paper_p(A['prespecified']['H2_lse8.0_1x1_%s' % mt],
                                                  'H2_lse8.0_1x1_%s' % mt)))
                     for mt in mets}
        prim[arm]['primary_metric'] = pm
        res.setdefault('by_arm', {})[arm] = tally(cc)
    assert len(pre_cells) == 32
    res['prespecified_32'] = tally(pre_cells)
    res['h2_at_primary_cell'] = prim
    res['prespecified_cells'] = [dict(arm=c['arm'], key=c['key'], diff=c['diff'], lo=c['lo'], hi=c['hi'],
                                      P=c['P'], verdict=cls(c)) for c in pre_cells]
    post = []
    for arm, fn, mets, pm in POST:
        A = R.jload(os.path.join(R.MOUT, fn))
        cc = cells_of(arm, A['prespecified'], mets)
        post += cc
        res.setdefault('h2_at_primary_cell_posthoc', {})[arm] = R.brief(dict(
            A['prespecified']['H2_lse8.0_1x1_hit'], P=pv(A['prespecified']['H2_lse8.0_1x1_hit'])))
    T6 = R.jload(os.path.join(R.C2O, 't6_score.json'))['rsna_disjoint']['grid']
    dj = []
    for temp in R.OFF:
        for unit in ('1x1', '3x3'):
            for mt in ('hit', 'ap'):
                c = dict(T6['H2_%s_%s_%s' % (temp, unit, mt)])
                c.update(arm='RSNA, disjoint', temp=temp, unit=unit, metric=mt, P=pv(c))
                dj.append(c)
    res['h2_at_primary_cell_posthoc']['RSNA, disjoint models'] = R.brief(
        dict(T6['H2_lse8.0_1x1_hit'], P=pv(T6['H2_lse8.0_1x1_hit'])))
    res['posthoc_nih_16'] = tally(post)
    res['posthoc_rsna_disjoint_8'] = tally(dj)
    # closed-form view
    cf = {}
    for fn, arm, mets, key in (('t1_nih_cf.json', 'NIH, distributed head', ['hit', 'ap', 'iou'], 'closed_form'),
                               ('t1_rsna_cf.json', 'RSNA, distributed model', ['hit', 'ap'], 'closed_form'),
                               ('t1_isic_cf.json', 'ISIC r4 replicate FD', ['hit', 'ap', 'dice'], 'replicate_r4_fd'),
                               ('t1_isic_cf.json', 'ISIC r4 replicate CF', ['hit', 'ap', 'dice'], 'replicate_r4_closed_form')):
        p = os.path.join(R.OUT, fn)
        if not os.path.exists(p):
            continue
        J = R.jload(p)['cells']
        cc = cells_of(arm, {k: v[key] for k, v in J.items()}, mets)
        cf[arm] = dict(tally=tally(cc), primary={mt: J['H2_lse8.0_1x1_%s' % mt][key] for mt in mets})
    if len(cf) == 4:
        cc = []
        for arm, mets, fn, key in (('NIH', ['hit', 'ap', 'iou'], 't1_nih_cf.json', 'closed_form'),
                                   ('RSNA', ['hit', 'ap'], 't1_rsna_cf.json', 'closed_form'),
                                   ('ISIC', ['hit', 'ap', 'dice'], 't1_isic_cf.json', 'replicate_r4_closed_form')):
            J = R.jload(os.path.join(R.OUT, fn))['cells']
            cc += cells_of(arm, {k: v[key] for k, v in J.items()}, mets)
        cf['prespecified_32_closed_form (NIH, RSNA closed form; ISIC r4 replicate closed form)'] = tally(cc)
    res['closed_form_view'] = cf
    print('32 pre-specified H2 cells:', res['prespecified_32'])
    for a, v in res['by_arm'].items():
        print('  ', a, v['all'])
    for a, v in prim.items():
        print('H2 primary', a, {k: (R.fmt(x) if isinstance(x, dict) else x) for k, x in v.items()})
    for a, v in res['h2_at_primary_cell_posthoc'].items():
        print('H2 primary (post hoc)', a, R.fmt(v))
    print('post hoc NIH 16:', res['posthoc_nih_16']['all'], '| disjoint 8:', res['posthoc_rsna_disjoint_8']['all'])
    for a, v in cf.items():
        print('CF', a, v.get('tally', v)['all'] if 'tally' in v else v['all'])
    R.save('t3_h2.json', res)


if __name__ == '__main__':
    main()
