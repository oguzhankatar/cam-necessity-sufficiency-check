"""S1 -- per-finding NIH hit rates for supplementary Table S4.

Recomputes, without any model, the two average-annotation-box baselines of
medical_baselines/scripts/t6_score.py (leave-one-patient-out `avgbox_full` and the 5-fold
patient-level cross-fit `avgbox_cf5_full`) per item on the reused-head arm
(maps_nih_m4.npz), and reports them per finding next to plain CAM and the
18 maps of outputs/medical/m4_stats.json (per_finding_hit).

Checks: overall means must equal t6_score.json nih_m4 comparators_hit
(0.397 LOPO, 0.385 cross-fit); per-finding LOPO and CAM must equal
t6_score.json nih_m4 per_finding_hit.
Output: ../out/s1_nih_perfinding.json
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C2S = os.path.abspath(os.path.join(HERE, '..', '..', 'medical_baselines', 'scripts'))
sys.path.insert(0, C2S)
import medical_baselines_lib as L                     # noqa: E402
import t6_score as T                  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, '..', 'out'))
os.makedirs(OUT, exist_ok=True)


def main():
    M, meta = L.load_maps(T.NPZ['nih_m4'])
    gt, fnd = T.gts('nih_m4', meta)
    keep = [i for i, g in enumerate(gt) if 0 < g.sum() < g.size]
    hit_full, _, extra = T.avgbox_maps('nih_m4', meta, gt, fnd)
    hit_cf = extra['avgbox_cf5_hit']
    kept = np.array(keep)
    hf, hc = hit_full[kept], hit_cf[kept]
    fk = np.array([fnd[i] for i in keep])
    # plain CAM per item, scored as in t6_score
    from medical_baselines_lib import mlib
    cam = np.array([mlib.pointing_hit(M['cam'][i], T.G, gt[i], 224) for i in keep])

    R6 = json.load(open(os.path.join(L.OUT, 't6_score.json')))['nih_m4']
    A = L.load_json('m4_stats.json')
    res = dict(n=int(len(kept)), overall=dict(avgbox_lopo=float(hf.mean()), avgbox_cf5=float(hc.mean()),
                                               cam=float(cam.mean())),
               check_overall=dict(t6_lopo=R6['comparators_hit']['avgbox_full'],
                                  t6_cf5=R6['comparators_hit']['avgbox_cf5_full'],
                                  t6_cam=R6['comparators_hit']['cam']),
               per_finding={})
    PF = A['per_finding_hit']
    maps18 = [k for k in PF['Effusion'] if k not in ('n', 'gt_frac')]
    nonmodel = {'center', 'random', 'actnorm'}
    model15 = [k for k in maps18 if k not in nonmodel]
    for f in sorted(set(fk)):
        s = fk == f
        pf = PF[f]
        best_model = max(model15, key=lambda k: pf[k])
        ties = [k for k in model15 if abs(pf[k] - pf[best_model]) < 1e-12]
        row = dict(n=int(s.sum()), gt_frac=pf['gt_frac'],
                   avgbox_cf5=float(hc[s].mean()), avgbox_lopo=float(hf[s].mean()),
                   cam=float(cam[s].mean()),
                   best_model_map=best_model, best_model_ties=ties, best_model_hit=pf[best_model],
                   center=pf['center'], random=pf['random'], actnorm=pf['actnorm'],
                   gradcam_gap=pf['gradcam_gap'], collapsed_1x1=pf['collapsed_1x1'],
                   check_t6_lopo=R6['per_finding_hit'][f]['avgbox_full'],
                   check_t6_cam=R6['per_finding_hit'][f]['cam'],
                   check_n=R6['per_finding_hit'][f]['n'])
        # best of all 18 archived maps, and of all 20 columns now shown
        all18 = max(maps18, key=lambda k: pf[k])
        row['best_of_18'] = all18
        cand = {k: pf[k] for k in maps18}
        cand['avgbox_cf5'] = row['avgbox_cf5']
        cand['cam'] = row['cam']
        row['best_of_all'] = max(cand, key=cand.get)
        row['best_of_all_value'] = cand[row['best_of_all']]
        res['per_finding'][f] = row
        print('%-13s n=%3d gt=%.3f cf5=%.3f lopo=%.3f (t6 %.3f) cam=%.3f (t6 %.3f) best=%s %.3f ties=%s '
              'centre=%.3f random=%.3f actnorm=%.3f best_of_all=%s' % (
                  f, row['n'], row['gt_frac'], row['avgbox_cf5'], row['avgbox_lopo'], row['check_t6_lopo'],
                  row['cam'], row['check_t6_cam'], best_model, pf[best_model], ties, pf['center'],
                  pf['random'], pf['actnorm'], row['best_of_all']))
    print('overall', res['overall'], res['check_overall'])
    json.dump(res, open(os.path.join(OUT, 's1_nih_perfinding.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
