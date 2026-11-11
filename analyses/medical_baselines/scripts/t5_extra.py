"""C2 task 5 extra: RSNA average-box point vs the archived Fig. 6d strongest map
(N at T=-8, 1x1) for both RSNA arms, exact McNemar.  Output out/t5_extra.json."""
import numpy as np
import medical_baselines_lib as L
from medical_baselines_lib import mlib
import t6_score as T6
bx = T6.rsna_boxes()
res = {}
for arm, nm in (('rsna_xrv', 'maps_rsna_xrv.npz'), ('rsna_disjoint', 'maps_rsna_disjoint.npz')):
    M, meta = L.load_maps(nm)
    gts = [L.box_mask_224(bx[n]) for n in meta['items']]
    ab = np.array([float(g[121, 70] > 0) for g in gts])
    for mn in ('N_lse-8.0_1x1', 'collapsed_1x1'):
        h = np.array([mlib.pointing_hit(M[mn][i], 7, gts[i], 224) for i in range(len(gts))])
        res['%s/avgbox_vs_%s' % (arm, mn)] = L.cmp(ab, h, True)
        print(arm, mn, {k: round(v, 4) if isinstance(v, float) else v for k, v in res['%s/avgbox_vs_%s' % (arm, mn)].items()})
L.save('t5_extra.json', res)
