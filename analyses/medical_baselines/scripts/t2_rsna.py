"""C2 task 2 (+ inputs for 3, 4, 5) -- RSNA.

Part A  re-run m11 exactly (torchxrayvision densenet121-res224-rsna, the same
        1,385 boxed studies, the same 8-image "no pneumonia" bank, the same map
        loop) and store every per-item map, adding plain CAM.  Checked against
        outputs/medical/m11_rsna.json.
Part B  a DenseNet-121 that has never seen the study it explains.  All 3,884
        local DICOMs are stage_2_train patients: the 1,385 boxed (positive)
        studies and 2,499 negatives.  The non-evaluation remainder has no
        positive, so a single disjoint model cannot be trained from local data.
        We therefore cross-fit: 5 patient-level stratified folds; for each fold a
        DenseNet-121 (timm, ImageNet init, GAP + linear head) is trained on the
        other four folds with m6_own's recipe (AdamW 1e-4, wd 1e-5, cosine,
        batch 32, horizontal flip, BCE with pos_weight, 8 epochs, 224 px,
        mlib.load_gray preprocessing) and explains only its own held-out fold.
        Every scored study is thus explained by a model that did not train on
        it.  The reference bank is 8 negatives drawn from the same held-out fold.
"""
import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medical_baselines_lib as L
from medical_baselines_lib import mlib, dev

D = os.path.join(L.ROOT, 'data', 'rsna')
DCM = os.path.join(D, 'dcm')
CLS_XRV = 8
EPOCHS = int(os.environ.get('EPOCHS', 8))
NFOLD = 5


def frame():
    rows = list(csv.DictReader(open(os.path.join(D, 'stage_2_train_labels.csv'))))
    boxes, target = {}, {}
    for r in rows:
        target[r['patientId']] = max(target.get(r['patientId'], 0), int(r['Target']))
        if r['Target'] == '1' and r['x']:
            boxes.setdefault(r['patientId'], []).append(
                (float(r['x']), float(r['y']), float(r['width']), float(r['height'])))
    have = sorted(n[:-4] for n in os.listdir(DCM))
    return have, target, boxes


def run_maps(F_of, W, b, items, idx, bank_of, cls, tag):
    """F_of(i) -> (T,C) double tensor on dev for image index i."""
    T = F_of(0).shape[0]
    u1, u3 = L.unit_masks(T)
    rmap = np.random.RandomState(7)
    recs = []
    t0 = time.time()
    heads_cache = {}
    for k, n in enumerate(items):
        Wk, bk = (W, b) if not callable(W) else (W(n), b(n))
        key = id(Wk)
        if key not in heads_cache:
            heads_cache[key] = L.make_heads(Wk, bk)
        heads = heads_cache[key]
        XA = F_of(idx[n])
        Xb = [F_of(i) for i in bank_of(n)]
        recs.append(L.maps_for(heads, XA, Xb, cls, u1, u3, Wk, rmap))
        if k % 250 == 0:
            print('  %s map %d/%d %.0fs' % (tag, k, len(items), time.time() - t0), flush=True)
    return recs


def score_check(recs, items, boxes, ref):
    """hit summary per map vs the archived m11 summary."""
    G = 7
    out = {}
    for mn in recs[0]:
        h = [mlib.pointing_hit(r[mn], G, L.box_mask_224(boxes[n]), 224)
             for r, n in zip(recs, items)]
        out[mn] = float(np.mean(h))
    diff = {mn: out[mn] - ref['summary'][mn]['hit']['mean']
            for mn in out if mn in ref['summary']}
    return out, diff


def part_a(have, target, boxes, U):
    import torchxrayvision as xrv
    import m1_model as m1
    xm = xrv.models.DenseNet(weights='densenet121-res224-rsna')
    body = m1.XrvBody(xm).eval().to(dev)
    W = xm.classifier.weight.detach().double().cpu()
    b = xm.classifier.bias.detach().double().cpu()
    fp = os.path.join(L.CACHE, 'F_rsna_xrv.npy')
    if not os.path.exists(fp):
        np.save(fp, L.feats(body, L.Lazy(U, L.u8_to_xrv)))
    F = np.load(fp, mmap_mode='r')
    from sklearn.metrics import roc_auc_score
    y = np.array([target[n] for n in have])
    Z = (np.asarray(F).mean(1) @ W.numpy().T + b.numpy())[:, CLS_XRV]
    auc = float(roc_auc_score(y, Z))
    idx = {n: i for i, n in enumerate(have)}
    items = [n for n in have if n in boxes]
    negs = [i for i, n in enumerate(have) if target[n] == 0]
    bank = sorted(np.random.RandomState(0).choice(negs, 8, replace=False).tolist())
    Fd = lambda i: torch.from_numpy(np.ascontiguousarray(F[i])).double().to(dev)
    recs = run_maps(Fd, W, b, items, idx, lambda n: bank, CLS_XRV, 'xrv')
    ref = L.load_json('m11_rsna.json')
    hits, diff = score_check(recs, items, boxes, ref)
    print('xrv AUC %.5f (archived %.5f); max |hit - archived| %.2e'
          % (auc, ref['condA']['gap']['auc_pneumonia'], max(abs(v) for v in diff.values())),
          flush=True)
    L.save_maps('maps_rsna_xrv.npz', recs, dict(items=items, cls=CLS_XRV, auc=auc,
                                                 bank=[have[i] for i in bank],
                                                 check_vs_m11=diff))
    return auc


def train_fold(U, y, tr, seed):
    import timm
    torch.manual_seed(seed)
    m = timm.create_model('densenet121', pretrained=True, num_classes=1).to(dev)
    bs = 32
    steps = len(tr) // bs
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * steps)
    pos = y[tr].sum() + 1
    pw = torch.tensor([(len(tr) - pos) / pos], dtype=torch.float32, device=dev)
    rs = np.random.RandomState(seed)
    for ep in range(EPOCHS):
        m.train()
        t0 = time.time()
        perm = rs.permutation(tr)
        tot = 0.
        for s in range(steps):
            j = np.sort(perm[s * bs:(s + 1) * bs])
            x = L.u8_to_imnet(U[j]).to(dev)
            flip = torch.from_numpy(rs.rand(len(j)) < .5).to(dev)
            x = torch.where(flip[:, None, None, None], torch.flip(x, [3]), x)
            t = torch.from_numpy(y[j].astype(np.float32))[:, None].to(dev)
            loss = Fn.binary_cross_entropy_with_logits(m(x), t, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tot += float(loss)
        print('    ep%d loss %.4f %.0fs' % (ep, tot / steps, time.time() - t0), flush=True)
    m.eval()
    return m


def part_b(have, target, boxes, U):
    import timm
    import frozen_rule
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    y = np.array([target[n] for n in have])
    skf = StratifiedKFold(NFOLD, shuffle=True, random_state=0)
    fold = np.zeros(len(have), int)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[te] = k
    Foof = np.zeros((len(have), 49, 1024), np.float32)
    Ws, bs_, info = {}, {}, []
    for k in range(NFOLD):
        tr = np.where(fold != k)[0]
        te = np.where(fold == k)[0]
        ck = os.path.join(L.CACHE, 'rsna_d121_fold%d_e%d.pt' % (k, EPOCHS))
        print('fold %d: train %d (pos %d), held-out %d (pos %d)'
              % (k, len(tr), y[tr].sum(), len(te), y[te].sum()), flush=True)
        if os.path.exists(ck):
            m = timm.create_model('densenet121', pretrained=False, num_classes=1).to(dev)
            m.load_state_dict(torch.load(ck))
            m.eval()
        else:
            m = train_fold(U, y, tr, seed=k)
            torch.save(m.state_dict(), ck)
        with torch.no_grad():
            fr = frozen_rule.structural(m, m.forward_features(L.u8_to_imnet(U[te[:1]]).to(dev)))
        Foof[te] = L.feats(lambda x: m.forward_features(x), L.Lazy(U[te], L.u8_to_imnet))
        Ws[k] = m.classifier.weight.detach().double().cpu()
        bs_[k] = m.classifier.bias.detach().double().cpu()
        z = Foof[te].mean(1) @ Ws[k].numpy().T[:, 0] + float(bs_[k][0])
        with torch.no_grad():
            zfull = torch.cat([m(L.u8_to_imnet(U[te[i:i + 64]]).to(dev)).cpu()
                               for i in range(0, len(te), 64)])[:, 0].numpy()
        info.append(dict(fold=k, n_train=int(len(tr)), n_train_pos=int(y[tr].sum()),
                         n_heldout=int(len(te)), n_heldout_pos=int(y[te].sum()),
                         auc_heldout=float(roc_auc_score(y[te], z)),
                         max_abs_cached_vs_forward=float(np.abs(z - zfull).max()),
                         frozen_rule=dict(verdict=fr['verdict'], agg=str(fr['agg']))))
        print('  fold %d held-out AUC %.4f  frozen rule %s' % (k, info[-1]['auc_heldout'],
                                                            fr['verdict']), flush=True)
        del m
        torch.cuda.empty_cache()
    Z = np.array([Foof[i].mean(0) @ Ws[fold[i]].numpy()[0] + float(bs_[fold[i]][0])
                  for i in range(len(have))])
    auc = float(roc_auc_score(y, Z))
    aucs_lse = {}
    for kind in L.OFF:
        zz = []
        for i in range(len(have)):
            h = mlib.head_of(kind, Ws[fold[i]], bs_[fold[i]]).double()
            with torch.no_grad():
                zz.append(float(h(torch.from_numpy(Foof[i][None]).double())[0, 0]))
        aucs_lse[kind] = float(roc_auc_score(y, zz))
    print('cross-fitted pooled out-of-fold AUC %.4f' % auc, flush=True)

    # certificates: one pair per fold, rtlib's literal float64 loop
    cert = []
    for k in range(NFOLD):
        te = np.where(fold == k)[0]
        for kind in ['gap'] + L.OFF:
            XA = torch.from_numpy(Foof[te[0]]).double()
            XB = torch.from_numpy(Foof[te[-1]]).double()
            dd = mlib.certify(mlib.head_of(kind, Ws[k], bs_[k]).double(), XA, XB, 0,
                              np.random.RandomState(0))
            cert.append(dict(fold=k, head=kind, maxNS=dd['maxNS'], additive=dd['additive']))

    idx = {n: i for i, n in enumerate(have)}
    items = [n for n in have if n in boxes]
    banks = {}
    for k in range(NFOLD):
        negs = [i for i in np.where(fold == k)[0] if y[i] == 0]
        banks[k] = sorted(np.random.RandomState(0).choice(negs, 8, replace=False).tolist())
    Fd = lambda i: torch.from_numpy(Foof[i]).double().to(dev)
    recs = run_maps(Fd, lambda n: Ws[fold[idx[n]]], lambda n: bs_[fold[idx[n]]],
                    items, idx, lambda n: banks[fold[idx[n]]], 0, 'disjoint')
    L.save_maps('maps_rsna_disjoint.npz', recs,
                dict(items=items, cls=0, auc_oof=auc, auc_oof_lse=aucs_lse, folds=info,
                     fold_of_item=[int(fold[idx[n]]) for n in items],
                     banks={k: [have[i] for i in v] for k, v in banks.items()},
                     certificates=cert, epochs=EPOCHS))
    L.save('t2_rsna_disjoint_model.json',
           dict(n_images=len(have), n_positive=int(y.sum()), folds=info,
                auc_oof_pooled=auc, auc_oof_lse=aucs_lse, epochs=EPOCHS,
                certificates=cert))


def main():
    have, target, boxes = frame()
    U = L.u8_cache([os.path.join(DCM, n + '.dcm') for n in have], 'rsna', kind='dcm')
    print('RSNA cached', U.shape, flush=True)
    if os.environ.get('SKIP_A') != '1':
        part_a(have, target, boxes, U)
    if os.environ.get('SKIP_B') != '1':
        part_b(have, target, boxes, U)


if __name__ == '__main__':
    main()
