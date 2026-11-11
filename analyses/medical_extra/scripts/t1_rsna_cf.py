"""Task 1 (RSNA, distributed TorchXRayVision densenet121-res224-rsna, class 8 'Pneumonia').

Closed-form (Eq. 4) N and S maps of the pre-specified RSNA grid for the 1,385 boxed
studies against the same 8 'no pneumonia' references (medical_baselines/out/maps_rsna_xrv.npz meta
'bank', = m11_rsna.py's RandomState(0) draw), scored by hit and AP as m11_rsna.py.
Compared with
  O  outputs/medical/m11_rsna.json 'prespecified' (the paper's cells),
  R  medical_baselines/out/maps_rsna_xrv.npz (forward differences, float64, regenerated from the
     same weights; 2 of 1,385 items differ from O by arg-max ties) re-scored here,
  CF the closed form on features rebuilt here from data/rsna/dcm.
Outputs: out/t1_rsna_cf.json, cache/t1_rsna_items.npz.
"""
import os
import numpy as np
import medical_extra_lib as R
from medical_extra_lib import L

DCM = os.path.join(R.ROOT, 'data', 'rsna', 'dcm')
CLS = 8


def features(names):
    fp = os.path.join(R.CACHE, 'F_rsna_needed.npz')
    if os.path.exists(fp):
        z = np.load(fp)
        if list(z['names']) == list(names):
            return z['F']
    import torchxrayvision as xrv
    import m1_model as m1
    xm = xrv.models.DenseNet(weights='densenet121-res224-rsna')
    body = m1.XrvBody(xm).eval().to(L.dev)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(12) as ex:
        U = np.stack(list(ex.map(L._dcm224, [os.path.join(DCM, n + '.dcm') for n in names])))
    F = L.feats(body, L.Lazy(U, L.u8_to_xrv))
    np.savez(fp, F=F, names=np.array(names))
    return F


def main():
    import torchxrayvision as xrv
    import t6_score as t6
    import t2_rsna as t2
    xm = xrv.models.DenseNet(weights='densenet121-res224-rsna')
    W = xm.classifier.weight.detach().double().cpu().numpy()
    b = xm.classifier.bias.detach().double().cpu().numpy()
    arch, meta = L.load_maps('maps_rsna_xrv.npz')
    items, bank = meta['items'], meta['bank']
    have, target, boxes = t2.frame()
    negs = [i for i, n in enumerate(have) if target[n] == 0]
    bank_chk = [have[i] for i in sorted(np.random.RandomState(0).choice(negs, 8, replace=False).tolist())]
    assert bank_chk == bank, (bank_chk, bank)
    need = sorted(set(items) | set(bank))
    F = features(need).astype(np.float64)
    ix = {n: i for i, n in enumerate(need)}
    bankX = [F[ix[n]] for n in bank]
    cfg = [(T, u) for T in (8.0, -8.0) for u in ('1x1', '3x3')]
    cf = {('%s_lse%s_%s' % (p, T, u)): np.zeros((len(items), 49)) for T, u in cfg for p in 'NS'}
    coll = {u: np.zeros((len(items), 49)) for u in ('1x1', '3x3')}
    for k, n in enumerate(items):
        sa = F[ix[n]] @ W[CLS] + b[CLS]
        SB = [x @ W[CLS] + b[CLS] for x in bankX]
        for T, u in cfg:
            Nm, Sm = R.cf_maps(sa, SB, T, u)
            cf['N_lse%s_%s' % (T, u)][k] = Nm
            cf['S_lse%s_%s' % (T, u)][k] = Sm
        for u in ('1x1', '3x3'):
            coll[u][k] = R.cf_collapsed(sa, SB, u)
        if k % 250 == 0:
            print('cf item', k, flush=True)
    diag = {}
    for key in cf:
        A, E = arch[key], cf[key]
        sc = np.abs(E).max(1) + 1e-300
        diag[key] = dict(frac_exact_zero_archived=float((A == 0).mean()),
                         frac_exact_zero_cf=float((E == 0).mean()),
                         argmax_agree=float(np.mean(A.argmax(1) == E.argmax(1))),
                         max_rel_abs_diff=float((np.abs(A - E).max(1) / sc).max()),
                         median_rel_abs_diff=float(np.median(np.abs(A - E).max(1) / sc)))
    for u in ('1x1', '3x3'):
        A, E = arch['collapsed_%s' % u], coll[u]
        diag['collapsed_%s' % u] = dict(max_rel_abs_diff=float(
            (np.abs(A - E).max(1) / (np.abs(E).max(1) + 1e-300)).max()))
    for k_, v in diag.items():
        print(k_, v, flush=True)
    bx = t6.rsna_boxes()
    gts = [L.box_mask_224(bx[n]) for n in items]
    mets = ['hit', 'ap']
    S_a, keep, _ = R.score(arch, gts, mets, ['collapsed_1x1', 'collapsed_3x3'] + sorted(cf))
    S_c, keep2, _ = R.score(cf, gts, mets, sorted(cf))
    assert (keep == keep2).all()
    for u in ('1x1', '3x3'):
        S_c['collapsed_%s' % u] = S_a['collapsed_%s' % u]
    g_a, g_c = R.grid(S_a, mets), R.grid(S_c, mets)
    O = R.jload(os.path.join(R.MOUT, 'm11_rsna.json'))
    cells = {}
    for key in g_a:
        row = dict(regenerated_fd=R.brief(g_a[key]), closed_form=R.brief(g_c[key]))
        if key in O['prespecified']:
            row['original'] = R.brief(R.orig_cell(O, key))
        h, tp, un, mt = key.split('_')
        mn = ('S_%s_%s' if h == 'Sminus' else 'N_%s_%s') % (tp, un)
        row['items_changed_%s' % mn] = int(np.sum(np.abs(np.asarray(S_a[mn][mt]) - np.asarray(S_c[mn][mt])) > 1e-12))
        cells[key] = row
        print('%-26s O %s | R %s | CF %s' % (key, R.fmt(row['original']) if 'original' in row else '-',
                                              R.fmt(g_a[key]), R.fmt(g_c[key])), flush=True)
    R.save('t1_rsna_cf.json', dict(
        arm='RSNA, distributed model (pre-specified)', n_items=len(items), n_scored=int(len(keep)),
        bank=bank, diagnostics=diag, cells=cells, primary=cells['H1_lse8.0_1x1_hit'],
        h2_primary=cells['H2_lse8.0_1x1_hit'],
        sources=dict(original='outputs/medical/m11_rsna.json',
                     regenerated_maps='analyses/medical_baselines/out/maps_rsna_xrv.npz',
                     closed_form='analyses/closed_form_maps/scripts/closed_form_lib.ns_exact_units')))
    np.savez_compressed(os.path.join(R.CACHE, 't1_rsna_items.npz'), items=np.array([items[i] for i in keep]),
                        **{'R__%s__%s' % (mn, mt): S_a[mn][mt] for mn in S_a for mt in mets},
                        **{'CF__%s__%s' % (mn, mt): S_c[mn][mt] for mn in S_c for mt in mets})


if __name__ == '__main__':
    main()
