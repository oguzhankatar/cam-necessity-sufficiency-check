"""Task 2 -- conjunctive sufficiency: S at T = -8 minus the collapsed map, every arm.

Sources (read only):
  medical_baselines/out/t6_score.json [arm]['grid']['Sminus_lse-8.0_<unit>_<metric>']: paired mean
     difference, 10,000-resample item bootstrap 95% CI (mlib.paired, seed 0),
     exact McNemar (hit) / Wilcoxon (AP, IoU, Dice).  Map source per arm as in
     Table S7: nih_m4 M (archived maps), rsna_xrv R (regenerated, same weights),
     nih_m10 / nih_m6 / isic P (replicate run), rsna_disjoint D.
  outputs/medical/{m4_stats,m10_condB,m6_own,m11_rsna,m5_isic}.json 'summary': the
     ORIGINAL runs' mean hit (Dice) of S_lse-8.0 and collapsed (no per-item vectors
     are stored for the original post hoc / ISIC runs, so no interval).
  RSNA held-out 283 studies (medical_baselines/out/rsna_local_xrv_heldout.json) and NIH held-out 182
     items (nih_boxed_xrv_heldout.json): per-item vectors from cache/t1_*_items.npz
     (the archive's maps re-scored by t1_*_cf.py, and the closed-form maps).
  Closed form: out/t1_nih_cf.json, out/t1_rsna_cf.json, out/t1_isic_cf.json.
Output: out/t2_sminus8.json
"""
import os
import json
import numpy as np
import medical_extra_lib as R

ARMS = [('nih_m4', 'NIH, distributed head (pre-specified)', 'M', 'm4_stats.json'),
        ('nih_m10', 'NIH, refitted head (post hoc)', 'P', 'm10_condB.json'),
        ('nih_m6', 'NIH, own model (post hoc)', 'P', 'm6_own.json'),
        ('rsna_xrv', 'RSNA, distributed model (pre-specified)', 'R', 'm11_rsna.json'),
        ('rsna_disjoint', 'RSNA, cross-fitted disjoint models (post hoc)', 'D', None),
        ('isic', 'ISIC (pre-specified)', 'P', 'm5_isic.json')]


def main():
    T6 = R.jload(os.path.join(R.C2O, 't6_score.json'))
    res = {}
    for arm, lab, src, fn in ARMS:
        r = T6[arm]
        cells = {}
        for unit in ('1x1', '3x3'):
            for mt in r['metrics']:
                c = dict(r['grid']['Sminus_lse-8.0_%s_%s' % (unit, mt)])
                c['P'] = c['p_mcnemar'] if mt == 'hit' else c['p_wilcoxon']
                cells['%s_%s' % (unit, mt)] = R.brief(c)
        row = dict(label=lab, source=src, n=r['n'], cells=cells, strongest_model_map=r['strongest_model_map'],
                   strongest_hit=r['comparators_hit'][r['strongest_model_map']])
        if fn:
            O = R.jload(os.path.join(R.MOUT, fn))['summary']
            row['original_run_mean_diff'] = {
                '%s_%s' % (u, mt): O['S_lse-8.0_%s' % u][mt]['mean'] - O['collapsed_%s' % u][mt]['mean']
                for u in ('1x1', '3x3') for mt in ('hit', 'dice', 'ap', 'iou') if mt in O['collapsed_%s' % u]}
            row['original_run_means_1x1'] = {
                mt: dict(S_minus8=O['S_lse-8.0_1x1'][mt]['mean'], collapsed=O['collapsed_1x1'][mt]['mean'])
                for mt in O['collapsed_1x1'] if mt in ('hit', 'dice')}
        res[arm] = row
    # held-out subsets (training-overlap free for the distributed TorchXRayVision models)
    sub = {}
    for arm, npzn, heldf, pre in (('rsna_xrv', 't1_rsna_items.npz', 'rsna_local_xrv_heldout.json', 'R'),
                                  ('nih_m4', 't1_nih_items.npz', 'nih_boxed_xrv_heldout.json', 'A')):
        p = os.path.join(R.CACHE, npzn)
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ids = z['items'] if 'items' in z.files else z['images']
        held = set(json.load(open(os.path.join(R.C2O, heldf))))
        hm = np.array([i in held for i in ids])
        d = dict(n=int(hm.sum()))
        for tag, lab in ((pre, 'stored_maps'), ('CF', 'closed_form')):
            for u in ('1x1', '3x3'):
                a = z['%s__S_lse-8.0_%s__hit' % (tag, u)][hm]
                b = z['%s__collapsed_%s__hit' % (pre, u)][hm]
                d['%s_%s_hit' % (lab, u)] = R.brief(R.cmp(a, b, True))
        sub[arm] = d
    res['heldout_subsets'] = sub
    # closed form, full arms
    cf = {}
    for fn, key, mets in (('t1_nih_cf.json', 'closed_form', ['hit', 'ap', 'iou']),
                          ('t1_rsna_cf.json', 'closed_form', ['hit', 'ap']),
                          ('t1_isic_cf.json', 'replicate_r4_closed_form', ['hit', 'ap', 'dice']),
                          ('t1_isic_cf.json', 'replicate_r4_fd', ['hit', 'ap', 'dice'])):
        p = os.path.join(R.OUT, fn)
        if os.path.exists(p):
            J = R.jload(p)['cells']
            cf['%s:%s' % (fn, key)] = {'%s_%s' % (u, mt): J['Sminus_lse-8.0_%s_%s' % (u, mt)][key]
                                       for u in ('1x1', '3x3') for mt in mets}
    res['closed_form'] = cf
    for arm, *_ in ARMS:
        r = res[arm]
        print('%-14s %s n=%d 1x1 hit %s | orig-run mean diff %s' % (
            arm, r['source'], r['n'], R.fmt(r['cells']['1x1_hit']),
            {k: round(v, 4) for k, v in r.get('original_run_mean_diff', {}).items() if k.startswith('1x1')}))
        if arm == 'isic':
            print('               1x1 dice %s' % R.fmt(r['cells']['1x1_dice']))
    for a, d in sub.items():
        print('held-out', a, d['n'], {k: R.fmt(v) for k, v in d.items() if k != 'n'})
    for k, v in cf.items():
        print('CF', k, R.fmt(v['1x1_hit']), R.fmt(v['1x1_dice']) if '1x1_dice' in v else '')
    R.save('t2_sminus8.json', res)


if __name__ == '__main__':
    main()
