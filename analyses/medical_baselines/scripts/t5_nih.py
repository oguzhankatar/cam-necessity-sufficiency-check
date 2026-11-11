"""C2 tasks 3, 4, 6 -- the NIH arms (and ISIC) with per-item maps.

  stage xrv   features of densenet121-res224-nih on the 5,720 test images under
              (a) the paper's preprocessing (m1.xrv_load) and (b) torchxrayvision's
              own (load_image -> XRayCenterCrop -> XRayResizer(224)); AUC per finding,
              on the whole sample and on the xrv-held-out subset.
  stage m4    the archived m3_maps.json per-item maps, plus plain CAM computed from
              (a) features; collapsed map recomputed for a check.
  stage m10   condition B re-run (m10_condB.fit_head imported, same seed) -> maps.
  stage m6    own DenseNet-121 re-trained with m6_own.train (imported) -> maps.
  stage isic  ResNet-50 re-trained with m5_isic.train (imported) -> maps.
The re-runs of m6 / m10 / isic are replicates (GPU training is not bit-exact);
each is checked against its archived JSON before its per-item maps are used.
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medical_baselines_lib as L
from medical_baselines_lib import mlib, dev
from mlib import NIH_FINDINGS

IMG = os.path.join(L.ROOT, 'data', 'nih', 'images')


def nih_names():
    names, Y, split = mlib.nih_frame()
    te, tr = split == 'test', split == 'train'
    return ([n for n, t in zip(names, te) if t], Y[te],
            [n for n, t in zip(names, tr) if t], Y[tr])


def xrv_pp_cache(names):
    fp = os.path.join(L.CACHE, 'xrvpp_test.npy')
    if os.path.exists(fp):
        return np.load(fp, mmap_mode='r')
    import torchxrayvision as xrv
    from concurrent.futures import ThreadPoolExecutor
    tf = [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]

    def one(n):
        a = xrv.utils.load_image(os.path.join(IMG, n))
        for t in tf:
            a = t(a)
        return a[0].astype(np.float32)
    with ThreadPoolExecutor(12) as ex:
        A = np.stack(list(ex.map(one, names)))
    np.save(fp, A)
    return np.load(fp, mmap_mode='r')


def aucs(Z, Y, idx):
    from sklearn.metrics import roc_auc_score
    a = {}
    for f, j in idx.items():
        y = Y[:, NIH_FINDINGS.index(f)]
        if 0 < y.sum() < len(y):
            a[f] = float(roc_auc_score(y, Z[:, j]))
    return dict(auc=a, mean_auc=float(np.mean(list(a.values()))),
                n=int(len(Y)), n_pos={f: int(Y[:, NIH_FINDINGS.index(f)].sum()) for f in idx})


def stage_xrv(vn, vY):
    import torchxrayvision as xrv
    import m1_model as m1
    xm = xrv.models.DenseNet(weights='densenet121-res224-nih')
    body = m1.XrvBody(xm).eval().to(dev)
    targets = list(xm.targets)
    W = xm.classifier.weight.detach().double().cpu()
    b = xm.classifier.bias.detach().double().cpu()
    U = L.u8_cache([os.path.join(IMG, n) for n in vn], 'nih_test')
    fpa = os.path.join(L.CACHE, 'F_nih_xrv_paper.npy')
    if not os.path.exists(fpa):
        np.save(fpa, L.feats(body, L.Lazy(U, L.u8_to_xrv)))
    Fa = np.load(fpa, mmap_mode='r')
    P = xrv_pp_cache(vn)
    fpb = os.path.join(L.CACHE, 'F_nih_xrv_xrvpp.npy')
    if not os.path.exists(fpb):
        np.save(fpb, L.feats(body, L.Lazy(P, lambda a: torch.from_numpy(np.asarray(a))[:, None])))
    Fb = np.load(fpb, mmap_mode='r')
    # the model's own forward on xrv preprocessing, for a check of the cached head
    with torch.no_grad():
        zf = xm.to(dev)(torch.from_numpy(np.asarray(P[:16]))[:, None].to(dev)).cpu().numpy()
    idx = {f: targets.index(f) for f in NIH_FINDINGS if f in targets}
    Za = np.asarray(Fa).mean(1) @ W.numpy().T + b.numpy()
    Zb = np.asarray(Fb).mean(1) @ W.numpy().T + b.numpy()
    held = set(json.load(open(os.path.join(L.OUT, 'nih_xrv_heldout_images.json'))))
    hm = np.array([n in held for n in vn])
    boxed = set(mlib.nih_boxes())
    bm = np.array([n in boxed for n in vn])
    res = dict(
        paper_pp=aucs(Za, vY, idx), xrv_pp=aucs(Zb, vY, idx),
        paper_pp_xrv_heldout=aucs(Za[hm], vY[hm], idx),
        xrv_pp_xrv_heldout=aucs(Zb[hm], vY[hm], idx),
        paper_pp_unboxed4840=aucs(Za[~bm], vY[~bm], idx),
        xrv_pp_unboxed4840=aucs(Zb[~bm], vY[~bm], idx),
        xrv_pp_unboxed_heldout=aucs(Zb[~bm & hm], vY[~bm & hm], idx),
        model_forward_vs_cached_head_maxabs=dict(
            as_logit=float(np.abs(Zb[:16] - zf).max()),
            as_sigmoid=float(np.abs(1 / (1 + np.exp(-Zb[:16])) - zf).max())),
        input_diff=dict(mean_abs=float(np.mean(np.abs(np.asarray(P[:200]) -
                                                      L.u8_to_xrv(U[:200])[:, 0].numpy()))),
                        scale='xrv units, range [-1024,1024]'),
        n_heldout=int(hm.sum()), n_boxed=int(bm.sum()),
        sample=dict(n=len(vn), boxed=int(bm.sum()), unboxed=int((~bm).sum()),
                    no_finding=int((vY.sum(1) == 0).sum()),
                    prevalence={f: float(vY[:, NIH_FINDINGS.index(f)].mean()) for f in NIH_FINDINGS},
                    prevalence_unboxed={f: float(vY[~bm][:, NIH_FINDINGS.index(f)].mean())
                                        for f in NIH_FINDINGS}))
    m1j = L.load_json('m1_model.json')
    res['check_vs_m1_mean_auc'] = res['paper_pp']['mean_auc'] - m1j['condA_xrv']['gap']['mean_auc']
    print('paper pp mean AUC %.5f (m1 %.5f) | xrv pp %.5f | held-out: paper %.4f xrv %.4f'
          % (res['paper_pp']['mean_auc'], m1j['condA_xrv']['gap']['mean_auc'],
             res['xrv_pp']['mean_auc'], res['paper_pp_xrv_heldout']['mean_auc'],
             res['xrv_pp_xrv_heldout']['mean_auc']), flush=True)
    L.save('t6_nih_auc.json', res)
    return Fa, W, b, targets


def items_nih(vn, findings_index):
    idx = {n: i for i, n in enumerate(vn)}
    boxes = mlib.nih_boxes()
    return [(n, f, findings_index(f)) for n, bs_ in sorted(boxes.items()) if n in idx
            for f in sorted(set(x[0] for x in bs_)) if findings_index(f) is not None], idx


def stage_m4(vn, Fa, W, targets):
    R = json.load(open(os.path.join(L.MOUT, 'm3_maps.json')))
    idx = {n: i for i, n in enumerate(vn)}
    names = sorted(R['items'][0]['maps'].keys())
    recs, meta = [], dict(items=[], G=R['G'], bank=R['bank'])
    bank = [torch.from_numpy(np.ascontiguousarray(Fa[idx[n]])).double().to(dev) for n in R['bank']]
    import m3_maps as m3
    u1, _ = L.unit_masks(49)
    h = mlib.head_of('gap', W, torch.zeros(W.shape[0], dtype=torch.float64)).double().to(dev)
    chk = 0.
    for k, it in enumerate(R['items']):
        XA = torch.from_numpy(np.ascontiguousarray(Fa[idx[it['image']]])).double().to(dev)
        mp = {mn: np.asarray(it['maps'][mn], float) for mn in names}
        mp['cam'] = L.plain_cam(W.to(dev), XA, it['xrv_index'])
        if k < 40:
            c = np.mean([m3.NS_batched(h, XA, XB, it['xrv_index'], u1)[0] for XB in bank], 0)
            ref = mp['collapsed_1x1']
            chk = max(chk, float(np.abs(c - ref).max() / (np.abs(ref).max() + 1e-12)))
        recs.append(mp)
        meta['items'].append([it['image'], it['finding']])
    meta['collapsed_recomputed_rel_maxdiff_first40'] = chk
    print('m4: collapsed recomputed vs archived, rel max diff %.2e' % chk, flush=True)
    L.save_maps('maps_nih_m4.npz', recs, meta)


def nih_score_summary(recs, items):
    import m4_stats as m4
    boxes = mlib.nih_boxes()
    out = {}
    for mn in recs[0]:
        h = []
        for r, (n, f, c) in zip(recs, items):
            g = m4.gt_mask(boxes[n], f)
            if g.sum() == 0:
                continue
            h.append(mlib.pointing_hit(r[mn], 7, g, 224))
        out[mn] = float(np.mean(h))
    return out


def stage_m10(vn, vY, tn, tY, Fa):
    import torchxrayvision as xrv
    import m1_model as m1
    import m10_condB as m10
    xm = xrv.models.DenseNet(weights='densenet121-res224-nih')
    body = m1.XrvBody(xm).eval().to(dev)
    Ut = L.u8_cache([os.path.join(IMG, n) for n in tn], 'nih_train')
    fp = os.path.join(L.CACHE, 'F_nih_xrv_train.npy')
    if not os.path.exists(fp):
        np.save(fp, L.feats(body, L.Lazy(Ut, L.u8_to_xrv)))
    Ftr = np.load(fp, mmap_mode='r')
    K = len(NIH_FINDINGS)
    heads, Wd = {}, {}
    from sklearn.metrics import roc_auc_score
    auc = {}
    for kind in ['gap'] + L.OFF:
        W, b = m10.fit_head(kind, Ftr, tY, Ftr.shape[2], K)
        Wd[kind] = (W.cpu(), b.cpu())
        heads[kind] = mlib.head_of(kind, W, b).double().to(dev)
        with torch.no_grad():
            Z = np.concatenate([heads[kind](torch.from_numpy(np.ascontiguousarray(Fa[i:i + 256])).double().to(dev)).cpu().numpy()
                                for i in range(0, len(Fa), 256)])
        a = [roc_auc_score(vY[:, j], Z[:, j]) for j in range(K) if 0 < vY[:, j].sum() < len(vY)]
        auc[kind] = float(np.mean(a))
    ref = L.load_json('m10_condB.json')
    print('m10 replicate mean AUC', auc, 'archived',
          {k: ref['condB'][k]['mean_auc'] for k in auc}, flush=True)
    items, idx = items_nih(vn, lambda f: NIH_FINDINGS.index(f) if f in NIH_FINDINGS else None)
    nofind = [i for i, n in enumerate(vn) if vY[i].sum() == 0 and n not in mlib.nih_boxes()]
    bank = sorted(np.random.RandomState(0).choice(nofind, 8, replace=False).tolist())
    Xb = [torch.from_numpy(np.ascontiguousarray(Fa[i])).double().to(dev) for i in bank]
    u1, u3 = L.unit_masks(49)
    rmap = np.random.RandomState(7)
    recs = []
    for (n, f, c) in items:
        XA = torch.from_numpy(np.ascontiguousarray(Fa[idx[n]])).double().to(dev)
        recs.append(L.maps_for(heads, XA, Xb, c, u1, u3, Wd['gap'][0], rmap))
    s = nih_score_summary(recs, items)
    diff = {mn: s[mn] - ref['summary'][mn]['hit']['mean'] for mn in s if mn in ref['summary']}
    print('m10 replicate hit vs archived: max |diff| %.4f' % max(abs(v) for v in diff.values()), flush=True)
    L.save_maps('maps_nih_m10.npz', recs, dict(items=[[n, f] for n, f, c in items],
                                               auc_replicate=auc, check_vs_m10=diff))


def stage_m6(vn, vY, tn, tY):
    import timm
    import m6_own as m6
    ck = os.path.join(L.CACHE, 'own_d121_replicate.pt')
    if os.path.exists(ck):
        m = timm.create_model('densenet121', pretrained=False, num_classes=len(NIH_FINDINGS)).to(dev)
        m.load_state_dict(torch.load(ck))
    else:
        m = m6.train(tn, tY, vn, vY)
        torch.save(m.state_dict(), ck)
    m.eval()
    U = L.u8_cache([os.path.join(IMG, n) for n in vn], 'nih_test')
    F = L.feats(lambda x: m.forward_features(x), L.Lazy(U, L.u8_to_imnet))
    W = m.classifier.weight.detach().double().cpu()
    b = m.classifier.bias.detach().double().cpu()
    from sklearn.metrics import roc_auc_score
    Z = F.mean(1) @ W.numpy().T + b.numpy()
    a = {f: float(roc_auc_score(vY[:, j], Z[:, j])) for j, f in enumerate(NIH_FINDINGS)
         if 0 < vY[:, j].sum() < len(vY)}
    ref = L.load_json('m6_own.json')
    print('m6 replicate mean AUC %.4f (archived %.4f)' % (np.mean(list(a.values())),
                                                          ref['condA']['gap']['mean_auc']), flush=True)
    heads = L.make_heads(W, b)
    items, idx = items_nih(vn, lambda f: NIH_FINDINGS.index(f) if f in NIH_FINDINGS else None)
    nofind = [i for i, n in enumerate(vn) if vY[i].sum() == 0 and n not in mlib.nih_boxes()]
    bank = sorted(np.random.RandomState(0).choice(nofind, 8, replace=False).tolist())
    Xb = [torch.from_numpy(np.ascontiguousarray(F[i])).double().to(dev) for i in bank]
    u1, u3 = L.unit_masks(49)
    rmap = np.random.RandomState(7)
    recs = [L.maps_for(heads, torch.from_numpy(F[idx[n]]).double().to(dev), Xb, c, u1, u3, W, rmap)
            for (n, f, c) in items]
    s = nih_score_summary(recs, items)
    diff = {mn: s[mn] - ref['summary'][mn]['hit']['mean'] for mn in s if mn in ref['summary']}
    print('m6 replicate hit vs archived: max |diff| %.4f' % max(abs(v) for v in diff.values()), flush=True)
    L.save_maps('maps_nih_m6.npz', recs, dict(items=[[n, f] for n, f, c in items],
                                              auc_replicate=a, mean_auc_replicate=float(np.mean(list(a.values()))),
                                              check_vs_m6=diff))


def stage_isic():
    import timm
    import m5_isic as m5
    names, y = mlib.isic_frame()
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(names))
    ntr = int(.7 * len(names))
    tr, va = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    ck = os.path.join(L.CACHE, 'isic_r50_replicate.pt')
    if os.path.exists(ck):
        m = timm.create_model('resnet50', pretrained=False, num_classes=3).to(dev)
        m.load_state_dict(torch.load(ck))
    else:
        m, _, _ = m5.train(names, y, tr, va)
        torch.save(m.state_dict(), ck)
    m.eval()
    vnames = [names[i] for i in va]
    X = torch.stack([mlib.load_rgb(os.path.join(m5.IMG, n + '.jpg'), 224) for n in vnames])
    F = L.feats(lambda x: m.forward_features(x), X)
    W = m.fc.weight.detach().double().cpu()
    b = m.fc.bias.detach().double().cpu()
    Z = F.mean(1) @ W.numpy().T + b.numpy()
    a = m5.aucs(Z, y[va])
    ref = L.load_json('m5_isic.json')
    print('isic replicate', a, 'archived macro', ref['gap_val']['macro'], flush=True)
    heads = L.make_heads(W, b)
    rsb = np.random.RandomState(0)
    bankidx = {}
    for c in range(3):
        cand = [i for i in range(len(vnames)) if y[va][i] != c]
        bankidx[c] = sorted(rsb.choice(cand, 8, replace=False).tolist())
    u1, u3 = L.unit_masks(49)
    rmap = np.random.RandomState(7)
    recs = []
    for i, n in enumerate(vnames):
        c = int(y[va][i])
        Xb = [torch.from_numpy(F[j]).double().to(dev) for j in bankidx[c]]
        recs.append(L.maps_for(heads, torch.from_numpy(F[i]).double().to(dev), Xb, c, u1, u3, W, rmap))
    L.save_maps('maps_isic.npz', recs, dict(items=[[n, int(y[va][i])] for i, n in enumerate(vnames)],
                                            auc_replicate=a, train=int(len(tr)), val=int(len(va))))


def main():
    vn, vY, tn, tY = nih_names()
    print('NIH test %d train %d' % (len(vn), len(tn)), flush=True)
    st = os.environ.get('STAGES', 'xrv,m4,m10,m6,isic').split(',')
    Fa = W = None
    if 'xrv' in st or 'm4' in st or 'm10' in st:
        Fa, W, b, targets = stage_xrv(vn, vY)
    if 'm4' in st:
        stage_m4(vn, Fa, W, targets)
    if 'm10' in st:
        stage_m10(vn, vY, tn, tY, Fa)
    if 'm6' in st:
        stage_m6(vn, vY, tn, tY)
    if 'isic' in st:
        stage_isic()


if __name__ == '__main__':
    main()
