"""Task 5 -- Grad-CAM on NIH (distributed head) scored three ways.

Maps: medical_baselines/out/maps_nih_m4.npz (= the archived m3_maps.json maps), 'gradcam_gap'
(Grad-CAM at the aggregation site, ReLU'd; identical for the GAP and both LSE heads
under condition A) and 'collapsed_1x1'.  Ground truth m4_stats.gt_mask.

  (a) as in the paper: mlib.pointing_hit = np.argmax of the bilinearly upsampled
      map; an all-zero map puts the argmax at pixel (0,0), an automatic miss.
  (b) all-zero maps as 'no prediction': those items are dropped; hit rates and the
      paired McNemar test on the remaining items.
  (c) the average-box tie rule (C2 t6_score.plateau_point): on a plateau of equal
      maxima of the upsampled map take the plateau pixel nearest its centroid; an
      all-zero map then points at the image centre.  Applied to BOTH maps.
The same is reported for the other arms that have Grad-CAM maps (their all-zero
fractions are near 0, so they are a check that (c) does not move other maps).
Output: out/t5_gradcam_ties.json
"""
import os
import numpy as np
import medical_extra_lib as R
from medical_extra_lib import L, mlib, m4


def hit_plateau(mp, g):
    import t6_score as t6
    u = mlib.upsample(mp, 7, 224)
    y, x = t6.plateau_point(u)
    return float(g[y, x] > 0)


def arm_gts(arm, meta):
    import t6_score as t6
    gt, _ = t6.gts(arm, meta)
    return gt


def one(arm, npz):
    M, meta = L.load_maps(npz)
    gts = arm_gts(arm, meta)
    keep = [i for i, g in enumerate(gts) if 0 < g.sum() < g.size]
    gc = M['gradcam_gap'][keep]
    co = M['collapsed_1x1'][keep]
    gk = [gts[i] for i in keep]
    zero = np.all(gc <= 0, 1)
    exact_zero = np.all(gc == 0, 1)
    ha = np.array([mlib.pointing_hit(m, 7, g, 224) for m, g in zip(gc, gk)])
    ca = np.array([mlib.pointing_hit(m, 7, g, 224) for m, g in zip(co, gk)])
    hc = np.array([hit_plateau(m, g) for m, g in zip(gc, gk)])
    cc = np.array([hit_plateau(m, g) for m, g in zip(co, gk)])
    nz = ~zero
    res = dict(n=int(len(keep)), n_allzero=int(zero.sum()), frac_allzero=float(zero.mean()),
               n_exactly_zero=int(exact_zero.sum()))
    res['a_as_paper'] = dict(gradcam=mlib.boot_ci(ha), collapsed=mlib.boot_ci(ca),
                             collapsed_minus_gradcam=R.cmp(ca, ha, True))
    res['b_allzero_no_prediction'] = dict(
        n=int(nz.sum()), gradcam=mlib.boot_ci(ha[nz]), collapsed=mlib.boot_ci(ca[nz]),
        collapsed_minus_gradcam=R.cmp(ca[nz], ha[nz], True),
        collapsed_hit_on_allzero_items=float(ca[zero].mean()) if zero.any() else None)
    res['c_centroid_tie_rule_both_maps'] = dict(
        gradcam=mlib.boot_ci(hc), collapsed=mlib.boot_ci(cc),
        collapsed_minus_gradcam=R.cmp(cc, hc, True),
        gradcam_hit_on_allzero_items=float(hc[zero].mean()) if zero.any() else None,
        items_changed_gradcam=int((hc != ha).sum()), items_changed_collapsed=int((cc != ca).sum()))
    return res


def main():
    out = {}
    for arm, npz in (('nih_m4', 'maps_nih_m4.npz'), ('nih_m10', 'maps_nih_m10.npz'),
                     ('nih_m6', 'maps_nih_m6.npz'), ('rsna_xrv', 'maps_rsna_xrv.npz'),
                     ('rsna_disjoint', 'maps_rsna_disjoint.npz')):
        out[arm] = one(arm, npz)
        r = out[arm]
        print('%-14s allzero %d/%d | (a) GC %.4f C %.4f %s | (b) n=%d GC %.4f C %.4f %s | (c) GC %.4f C %.4f %s'
              % (arm, r['n_allzero'], r['n'],
                 r['a_as_paper']['gradcam']['mean'], r['a_as_paper']['collapsed']['mean'],
                 R.fmt(r['a_as_paper']['collapsed_minus_gradcam']),
                 r['b_allzero_no_prediction']['n'], r['b_allzero_no_prediction']['gradcam']['mean'],
                 r['b_allzero_no_prediction']['collapsed']['mean'],
                 R.fmt(r['b_allzero_no_prediction']['collapsed_minus_gradcam']),
                 r['c_centroid_tie_rule_both_maps']['gradcam']['mean'],
                 r['c_centroid_tie_rule_both_maps']['collapsed']['mean'],
                 R.fmt(r['c_centroid_tie_rule_both_maps']['collapsed_minus_gradcam'])), flush=True)
    O = R.jload(os.path.join(R.MOUT, 'm4_stats.json'))
    out['nih_m4_archived_check'] = dict(
        gradcam_gap_hit=O['summary']['gradcam_gap']['hit']['mean'],
        collapsed_1x1_hit=O['summary']['collapsed_1x1']['hit']['mean'],
        collapsed_vs_gradcam_gap_hit=O['extra']['collapsed_vs_gradcam_gap_hit'])
    T6 = R.jload(os.path.join(R.C2O, 't6_score.json'))
    out['nih_m4_avgbox_hit_t6'] = T6['nih_m4']['comparators_hit'].get('avgbox_full')
    R.save('t5_gradcam_ties.json', out)


if __name__ == '__main__':
    main()
