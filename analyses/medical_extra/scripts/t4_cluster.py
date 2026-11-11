"""Task 4 -- NIH contrasts with the image as the resampling unit.

NIH has 984 scored boxes (image x finding items) on 880 images.  The archive's
intervals resample items (mlib.paired: 10,000 item resamples, seed 0) and test with
exact McNemar / Wilcoxon, treating boxes as independent.  Here, for the same per-item
scores (cache/t1_nih_items.npz, written by t1_nih_cf.py: the archived maps re-scored
exactly as m4_stats.py, and the closed-form maps):
  * image-level (cluster) bootstrap: 10,000 resamples of the 880 images with
    replacement (RandomState(0)), all items of a drawn image kept; the statistic is
    the ratio mean  sum(d) / number of items;
  * cluster-robust z test of the mean paired difference (ratio-estimator variance
    over images, the clustered-McNemar form of Obuchowski 1998 for binary scores);
  * image-level sign-flip permutation test of the per-image summed differences
    (100,000 flips, RandomState(1)); its P cannot fall below about 1e-5;
  * the design effect (cluster variance / item variance of the mean).
Also the primary hit cell of the two post hoc NIH arms from their stored replicate
maps (medical_baselines/out/maps_nih_m10.npz, maps_nih_m6.npz).
Output: out/t4_cluster.json
"""
import os
import numpy as np
from scipy.stats import norm
import medical_extra_lib as R
from medical_extra_lib import L, mlib, m4

B = 10000


def cluster_stats(d, groups, seed=0):
    d = np.asarray(d, float)
    ok = ~np.isnan(d)
    d, groups = d[ok], np.asarray(groups)[ok]
    ug, inv = np.unique(groups, return_inverse=True)
    K, N = len(ug), len(d)
    s = np.bincount(inv, weights=d, minlength=K)
    m = np.bincount(inv, minlength=K).astype(float)
    mean = s.sum() / N
    rs = np.random.RandomState(seed)
    bs = np.empty(B)
    for i in range(0, B, 1000):
        idx = rs.randint(0, K, (1000, K))
        bs[i:i + 1000] = s[idx].sum(1) / m[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    r = s - mean * m
    var_c = K / (K - 1) * np.sum(r ** 2) / N ** 2
    var_i = d.var(ddof=1) / N
    z = mean / np.sqrt(var_c) if var_c > 0 else np.nan
    rs2 = np.random.RandomState(1)
    obs = abs(s.sum())
    cnt = 0
    for _ in range(100):
        sg = rs2.randint(0, 2, (1000, K)) * 2 - 1
        cnt += int((np.abs((sg * s).sum(1)) >= obs - 1e-12).sum())
    return dict(diff=float(mean), lo=float(lo), hi=float(hi), n_items=int(N), n_images=int(K),
                z_cluster=float(z), P_cluster_z=float(2 * norm.sf(abs(z))),
                P_signflip_images=float((cnt + 1) / (100000 + 1)),
                design_effect=float(var_c / var_i) if var_i > 0 else np.nan,
                se_cluster=float(np.sqrt(var_c)), se_item=float(np.sqrt(var_i)))


def main():
    z = np.load(os.path.join(R.CACHE, 't1_nih_items.npz'))
    imgs = z['images']
    res = dict(n_items=int(len(imgs)), n_images=int(len(np.unique(imgs))),
               items_per_image=dict(zip(*[x.tolist() for x in np.unique(np.bincount(np.unique(imgs, return_inverse=True)[1]), return_counts=True)])))
    cells = {}
    for tag, lab in (('A', 'archived_maps'), ('CF', 'closed_form')):
        for unit in ('1x1', '3x3'):
            for mt in ('hit', 'ap', 'iou'):
                for temp in R.OFF:
                    for h, a_, b_ in (('H1', 'N', 'collapsed'), ('H2', 'N', 'S'), ('Sminus', 'S', 'collapsed')):
                        an = '%s_%s_%s' % (a_, temp, unit)
                        bn = 'collapsed_%s' % unit if b_ == 'collapsed' else '%s_%s_%s' % (b_, temp, unit)
                        ta = tag
                        tb = 'A' if b_ == 'collapsed' else tag
                        a = z['%s__%s__%s' % (ta, an, mt)]
                        b = z['%s__%s__%s' % (tb, bn, mt)]
                        key = '%s_%s_%s_%s' % (h, temp, unit, mt)
                        item = R.cmp(a, b, mt == 'hit')
                        cells.setdefault(lab, {})[key] = dict(item_level=R.brief(item),
                                                              image_level=cluster_stats(a - b, imgs))
    res['cells'] = cells
    P = cells['archived_maps']['H1_lse8.0_1x1_hit']
    print('NIH primary H1 (archived maps): item %s | image %+.4f [%+.4f, %+.4f] Pz=%.3g Pflip=%.3g deff=%.2f'
          % (R.fmt(P['item_level']), P['image_level']['diff'], P['image_level']['lo'], P['image_level']['hi'],
             P['image_level']['P_cluster_z'], P['image_level']['P_signflip_images'], P['image_level']['design_effect']))
    for key in ('H2_lse8.0_1x1_hit', 'Sminus_lse-8.0_1x1_hit', 'H1_lse8.0_1x1_ap', 'H1_lse8.0_1x1_iou'):
        Q = cells['archived_maps'][key]
        print('%-24s item %s | image [%+.4f, %+.4f] Pz=%.3g deff=%.2f' % (
            key, R.fmt(Q['item_level']), Q['image_level']['lo'], Q['image_level']['hi'],
            Q['image_level']['P_cluster_z'], Q['image_level']['design_effect']))
    # verdict changes: does any interval change side of zero?
    flips = []
    for lab, cc in cells.items():
        for k, v in cc.items():
            it, im = v['item_level'], v['image_level']
            si = 'above' if it['lo'] > 0 else ('below' if it['hi'] < 0 else 'span')
            sc = 'above' if im['lo'] > 0 else ('below' if im['hi'] < 0 else 'span')
            if si != sc:
                flips.append(dict(maps=lab, cell=k, item=si, image=sc, item_ci=[it['lo'], it['hi']],
                                  image_ci=[im['lo'], im['hi']]))
    res['interval_side_changes'] = flips
    res['design_effect_range'] = [min(v['image_level']['design_effect'] for cc in cells.values() for v in cc.values()),
                                  max(v['image_level']['design_effect'] for cc in cells.values() for v in cc.values())]
    print('interval side changes item->image:', flips)
    print('design effect range', res['design_effect_range'])
    # post hoc arms, primary hit, replicate maps
    boxes = mlib.nih_boxes()
    ph = {}
    for arm, npz in (('nih_m10', 'maps_nih_m10.npz'), ('nih_m6', 'maps_nih_m6.npz')):
        M, meta = L.load_maps(npz)
        gts = [m4.gt_mask(boxes[n], f) for n, f in meta['items']]
        keep = [i for i, g in enumerate(gts) if 0 < g.sum() < g.size]
        a = np.array([mlib.pointing_hit(M['N_lse8.0_1x1'][i], 7, gts[i], 224) for i in keep])
        b = np.array([mlib.pointing_hit(M['collapsed_1x1'][i], 7, gts[i], 224) for i in keep])
        ims = np.array([meta['items'][i][0] for i in keep])
        ph[arm] = dict(item_level=R.brief(R.cmp(a, b, True)), image_level=cluster_stats(a - b, ims),
                       note='C2 replicate maps (Table S7 Src P), not the original run')
        print(arm, 'item', R.fmt(ph[arm]['item_level']), 'image [%+.4f, %+.4f] Pz=%.3g' % (
            ph[arm]['image_level']['lo'], ph[arm]['image_level']['hi'], ph[arm]['image_level']['P_cluster_z']))
    res['posthoc_primary_hit_replicate_maps'] = ph
    R.save('t4_cluster.json', res)


if __name__ == '__main__':
    main()
