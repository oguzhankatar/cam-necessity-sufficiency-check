"""T4 -- are the medical N / S maps (averaged over 8 'No Finding' references,
float64 forward differences, m3_maps.py / C2 maps_nih_m4.npz) affected?

For every one of the 984 NIH box items: rebuild N and S of the distributed
DenseNet-121 head under LSE T = +8 and T = -8, 1x1 and 3x3 units, with the
closed form (closed_form_lib.ns_exact_units) against the SAME 8 references, average,
and compare with the archived maps:
  * argmax cell agreement, pointing-game hit (mlib.pointing_hit, the paper's
    scorer) archived vs exact, and the primary contrast N(+8) - collapsed;
  * the genuine rank difference between averaged N and averaged S in exact
    arithmetic (Spearman per image).
"""
import os, sys, json, numpy as np, torch
import closed_form_lib as C
sys.path.insert(0, os.path.join(C.ROOT, 'analyses', 'medical_baselines', 'scripts'))
import medical_baselines_lib as L2
import mlib
import m4_stats as m4
from t5_nih import nih_names


def main():
    import torchxrayvision as xrv
    xm = xrv.models.DenseNet(weights='densenet121-res224-nih')
    W = xm.classifier.weight.detach().double().cpu().numpy()
    b = xm.classifier.bias.detach().double().cpu().numpy()
    vn, vY, _, _ = nih_names()
    idx = {n: i for i, n in enumerate(vn)}
    Fa = np.load(os.path.join(L2.CACHE, 'F_nih_xrv_paper.npy'), mmap_mode='r')
    R = json.load(open(os.path.join(C.ROOT, 'outputs', 'medical', 'm3_maps.json')))
    arch, meta = L2.load_maps('maps_nih_m4.npz')
    items = meta['items']
    assert len(items) == len(R['items'])
    bankX = [np.asarray(Fa[idx[n]], np.float64) for n in R['bank']]        # (T, C)
    G = 7
    u1 = [[t] for t in range(49)]
    u3 = [list(map(int, x)) for x in mlib.blocks_of(G, 3)]
    boxes = mlib.nih_boxes()
    out = {}
    UN = {'1x1': u1, '3x3': u3}
    keys = [(T, un) for T in (8.0, -8.0) for un in ('1x1', '3x3')]
    hit = {('arch', k): [] for k in keys} | {('ex', k): [] for k in keys}
    hitS = {('arch', k): [] for k in keys} | {('ex', k): [] for k in keys}
    agreeN, agreeS, spNS, spNS_arch, zeros = ({k: [] for k in keys} for _ in range(5))
    hit_coll = []
    for k_, it in enumerate(R['items']):
        n, f, c = it['image'], it['finding'], it['xrv_index']
        XA = np.asarray(Fa[idx[n]], np.float64)
        sa = XA @ W[c] + b[c]
        SB = [x @ W[c] + b[c] for x in bankX]
        g = m4.gt_mask(boxes[n], f)
        ok = g.sum() > 0
        if ok:
            hit_coll.append(mlib.pointing_hit(arch['collapsed_1x1'][k_], G, g, 224))
        for key in keys:
            T, un = key; us = UN[un]
            NS = [C.ns_exact_units(sa, sb, T, us) for sb in SB]
            Ne = np.mean([x[0] for x in NS], 0); Se = np.mean([x[1] for x in NS], 0)
            tag = 'lse%s_%s' % (T, un)
            Na, Sa = arch['N_' + tag][k_], arch['S_' + tag][k_]
            agreeN[key].append(int(np.argmax(Na) == np.argmax(Ne)))
            agreeS[key].append(int(np.argmax(Sa) == np.argmax(Se)))
            spNS[key].append(C.sp(Ne, Se)); spNS_arch[key].append(C.sp(Na, Sa))
            zeros[key].append(float((Na == 0).mean()))
            if ok:
                hit[('arch', key)].append(mlib.pointing_hit(Na, G, g, 224))
                hit[('ex', key)].append(mlib.pointing_hit(Ne, G, g, 224))
                hitS[('arch', key)].append(mlib.pointing_hit(Sa, G, g, 224))
                hitS[('ex', key)].append(mlib.pointing_hit(Se, G, g, 224))
    for key in keys:
        T, un = key
        tag = 'T%+g_%s' % (T, un)
        ha, he = np.array(hit[('arch', key)]), np.array(hit[('ex', key)])
        sa_, se_ = np.array(hitS[('arch', key)]), np.array(hitS[('ex', key)])
        out[tag] = dict(
            argmax_agree_N=float(np.mean(agreeN[key])), argmax_agree_S=float(np.mean(agreeS[key])),
            hit_N_archived=float(ha.mean()), hit_N_exact=float(he.mean()), hit_N_items_changed=int((ha != he).sum()),
            hit_S_archived=float(sa_.mean()), hit_S_exact=float(se_.mean()), hit_S_items_changed=int((sa_ != se_).sum()),
            N_minus_collapsed_archived=float(ha.mean() - np.mean(hit_coll)),
            N_minus_collapsed_exact=float(he.mean() - np.mean(hit_coll)),
            spearman_Nbar_Sbar_exact=C.boot(spNS[key]), spearman_Nbar_Sbar_archived=C.boot(spNS_arch[key]),
            frac_exact_zero_in_archived_N=float(np.mean(zeros[key])), n_scored=int(len(ha)))
        print(tag, json.dumps(out[tag]), flush=True)
    out['hit_collapsed'] = float(np.mean(hit_coll))
    out['meta'] = dict(items=len(R['items']), bank=R['bank'], source='medical_baselines/out/maps_nih_m4.npz (m3_maps.json)')
    json.dump(out, open(os.path.join(C.OUTD, 't4_medical_nih.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
