"""Task 1 (NIH, distributed TorchXRayVision DenseNet-121 head reused under LSE_T).

Rebuild every N and S map of the pre-specified NIH grid (T = +8 / -8, 1x1 / 3x3)
with the closed form of Eq. 4 (closed_form_lib.ns_exact_units: float64, log1p / expm1, no
difference of nearly equal logits), against the same 8 'No Finding' references
(m3_maps.json 'bank'), and score hit, AP and cross-tuned IoU exactly as the archive
(m4_stats.py).  Compare with
  O  the archived pre-specified cells (outputs/medical/m4_stats.json),
  A  the archived maps re-scored here (medical_baselines/out/maps_nih_m4.npz = m3_maps.json),
  CF the closed-form maps.
Features: the 880 boxed images and the 8 references, rebuilt from data/nih with
C2's pipeline (medical_baselines_lib._pil224 -> u8_to_xrv -> relu(features)), as C5/t4 did.
Outputs: out/t1_nih_cf.json, cache/t1_nih_items.npz (per-item vectors, for task 4).
"""
import os
import numpy as np
import torch
import medical_extra_lib as R
from medical_extra_lib import L, mlib, m4

IMG = os.path.join(R.ROOT, 'data', 'nih', 'images')


def features(names):
    fp = os.path.join(R.CACHE, 'F_nih_needed.npz')
    if os.path.exists(fp):
        z = np.load(fp)
        if list(z['names']) == list(names):
            return z['F']
    import torchxrayvision as xrv
    import m1_model as m1
    xm = xrv.models.DenseNet(weights='densenet121-res224-nih')
    body = m1.XrvBody(xm).eval().to(L.dev)
    U = np.stack([L._pil224(os.path.join(IMG, n)) for n in names])
    F = L.feats(body, L.Lazy(U, L.u8_to_xrv))
    np.savez(fp, F=F, names=np.array(names))
    return F


def main():
    import torchxrayvision as xrv
    xm = xrv.models.DenseNet(weights='densenet121-res224-nih')
    W = xm.classifier.weight.detach().double().cpu().numpy()
    b = xm.classifier.bias.detach().double().cpu().numpy()
    Rm = R.jload(os.path.join(R.MOUT, 'm3_maps.json'))
    arch, meta = L.load_maps('maps_nih_m4.npz')
    items = [(it['image'], it['finding'], it['xrv_index']) for it in Rm['items']]
    assert [list(x[:2]) for x in items] == meta['items']
    need = sorted(set([n for n, f, c in items]) | set(Rm['bank']))
    F = features(need).astype(np.float64)
    ix = {n: i for i, n in enumerate(need)}
    bankX = [F[ix[n]] for n in Rm['bank']]
    cfg = [(T, u) for T in (8.0, -8.0) for u in ('1x1', '3x3')]
    cf = {}
    for T, u in cfg:
        for p in 'NS':
            cf['%s_lse%s_%s' % (p, T, u)] = np.zeros((len(items), 49))
    coll_cf = {u: np.zeros((len(items), 49)) for u in ('1x1', '3x3')}
    for k, (n, f, c) in enumerate(items):
        sa = F[ix[n]] @ W[c] + b[c]
        SB = [x @ W[c] + b[c] for x in bankX]
        for T, u in cfg:
            Nm, Sm = R.cf_maps(sa, SB, T, u)
            cf['N_lse%s_%s' % (T, u)][k] = Nm
            cf['S_lse%s_%s' % (T, u)][k] = Sm
        for u in ('1x1', '3x3'):
            coll_cf[u][k] = R.cf_collapsed(sa, SB, u)
        if k % 200 == 0:
            print('cf item', k, flush=True)
    # ---- diagnostics: archived (forward differences, float64) vs closed form
    diag = {}
    for key in cf:
        A, E = arch[key], cf[key]
        scale = np.abs(E).max(1) + 1e-300
        diag[key] = dict(
            frac_exact_zero_archived=float((A == 0).mean()),
            frac_exact_zero_cf=float((E == 0).mean()),
            argmax_agree=float(np.mean(A.argmax(1) == E.argmax(1))),
            max_rel_abs_diff=float((np.abs(A - E).max(1) / scale).max()),
            median_rel_abs_diff=float(np.median(np.abs(A - E).max(1) / scale)),
            spearman_arch_cf=C_sp(A, E))
    for u in ('1x1', '3x3'):
        A, E = arch['collapsed_%s' % u], coll_cf[u]
        diag['collapsed_%s' % u] = dict(max_rel_abs_diff=float(
            (np.abs(A - E).max(1) / (np.abs(E).max(1) + 1e-300)).max()))
    for k_, v in diag.items():
        print(k_, v, flush=True)
    # ---- scoring
    boxes = mlib.nih_boxes()
    gts = [m4.gt_mask(boxes[n], f) for n, f, c in items]
    mets = ['hit', 'ap', 'iou']
    nm_arch = ['collapsed_1x1', 'collapsed_3x3'] + sorted(cf)
    S_a, keep, ch_a = R.score(arch, gts, mets, nm_arch)
    S_c, keep2, ch_c = R.score(cf, gts, mets, sorted(cf))
    assert (keep == keep2).all()
    for u in ('1x1', '3x3'):
        S_c['collapsed_%s' % u] = S_a['collapsed_%s' % u]
    g_a, g_c = R.grid(S_a, mets), R.grid(S_c, mets)
    O = R.jload(os.path.join(R.MOUT, 'm4_stats.json'))
    cells = {}
    for key in g_a:
        row = dict(rescored_archived=R.brief(g_a[key]), closed_form=R.brief(g_c[key]))
        if key in O['prespecified']:
            row['original'] = R.brief(R.orig_cell(O, key))
            row['rescore_minus_original_diff'] = g_a[key]['diff'] - O['prespecified'][key]['diff']
        for mt in mets:
            if key.endswith('_' + mt):
                h, tp, un, _ = key.split('_')
                mn = ('S_%s_%s' if h == 'Sminus' else 'N_%s_%s') % (tp, un)
                a, e = np.asarray(S_a[mn][mt]), np.asarray(S_c[mn][mt])
                row['items_changed_%s' % mn] = int(np.sum(np.abs(a - e) > 1e-12))
        cells[key] = row
        print('%-26s O %s | A %s | CF %s' % (
            key, R.fmt(row['original']) if 'original' in row else '-', R.fmt(g_a[key]), R.fmt(g_c[key])),
            flush=True)
    ch = {'archived': {k: ch_a[k] for k in ch_a if k.startswith(('N_', 'S_', 'coll'))},
          'closed_form': ch_c}
    R.save('t1_nih_cf.json', dict(
        arm='NIH, distributed head reused (pre-specified)', n_items=len(items), n_scored=int(len(keep)),
        bank=Rm['bank'], diagnostics=diag, cells=cells, iou_q_chosen_per_fold=ch,
        primary=cells['H1_lse8.0_1x1_hit'], h2_primary=cells['H2_lse8.0_1x1_hit'],
        sources=dict(original='outputs/medical/m4_stats.json',
                     archived_maps='analyses/medical_baselines/out/maps_nih_m4.npz',
                     closed_form='analyses/closed_form_maps/scripts/closed_form_lib.ns_exact_units')))
    np.savez_compressed(os.path.join(R.CACHE, 't1_nih_items.npz'),
                        images=np.array([items[i][0] for i in keep]),
                        findings=np.array([items[i][1] for i in keep]),
                        **{'A__%s__%s' % (mn, mt): S_a[mn][mt] for mn in S_a for mt in mets},
                        **{'CF__%s__%s' % (mn, mt): S_c[mn][mt] for mn in S_c for mt in mets})


def C_sp(A, E):
    v = [R.C.sp(a, e) for a, e in zip(A, E)]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


if __name__ == '__main__':
    main()
