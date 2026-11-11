"""m5 -- the same experiment on ISIC-2018 Task 1 (dermatologist lesion
segmentations) with ISIC-2017 Part 3 diagnosis labels, ResNet-50.

Stages: unpack -> train -> cache -> certify -> maps -> score.
The pre-specification is the one in protocols/medical_protocol.md; only the ground
truth (a pixel mask instead of a box) and the primary metric (Dice at a
cross-tuned threshold, the lesion-segmentation convention) differ.
"""
import os
import sys
import json
import time
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib
from mlib import ROOT, SCRATCH, dev
import frozen_rule
import rtlib

D = os.path.join(ROOT, 'data', 'isic2018')
IMG = os.path.join(D, 'images')
MSK = os.path.join(D, 'masks')
SIZE = 224
QGRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
OFF = ['lse8.0', 'lse-8.0']
CLASSES = ['nevus', 'melanoma', 'seborrheic_keratosis']


def unpack():
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(MSK, exist_ok=True)
    z = os.path.join(D, 'ISIC2018_Task1-2_Training_Input.zip')
    if os.path.exists(z) and len(os.listdir(IMG)) < 100:
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                if n.lower().endswith('.jpg'):
                    with open(os.path.join(IMG, os.path.basename(n)), 'wb') as o:
                        o.write(f.read(n))
    zm = os.path.join(D, 'ISIC2018_Task1_Training_GroundTruth.zip')
    if os.path.exists(zm) and len(os.listdir(MSK)) < 100:
        with zipfile.ZipFile(zm) as f:
            for n in f.namelist():
                if n.lower().endswith('.png'):
                    with open(os.path.join(MSK, os.path.basename(n)), 'wb') as o:
                        o.write(f.read(n))
    print('images %d, masks %d' % (len(os.listdir(IMG)), len(os.listdir(MSK))), flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(s, names, y, aug=False):
        s.n, s.y, s.aug = names, y, aug

    def __len__(s):
        return len(s.n)

    def __getitem__(s, i):
        x = mlib.load_rgb(os.path.join(IMG, s.n[i] + '.jpg'), SIZE)
        if s.aug:
            if np.random.rand() < .5:
                x = torch.flip(x, [2])
            if np.random.rand() < .5:
                x = torch.flip(x, [1])
        return x, int(s.y[i])


def train(names, y, tr, va, epochs=12, bs=32, seed=0):
    import timm
    torch.manual_seed(seed)
    m = timm.create_model('resnet50', pretrained=True, num_classes=3).to(dev)
    ld = torch.utils.data.DataLoader(
        DS([names[i] for i in tr], y[tr], aug=True), batch_size=bs,
        shuffle=True, num_workers=4, drop_last=True)
    vl = torch.utils.data.DataLoader(
        DS([names[i] for i in va], y[va]), batch_size=bs, num_workers=4)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(ld))
    w = torch.tensor(len(y[tr]) / (3 * np.bincount(y[tr], minlength=3) + 1e-9),
                     dtype=torch.float32, device=dev)
    for ep in range(epochs):
        m.train()
        for x, t in ld:
            loss = Fn.cross_entropy(m(x.to(dev)), t.to(dev), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        m.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, t in vl:
                P.append(m(x.to(dev)).cpu().numpy())
                Y.append(t.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        acc = float((P.argmax(1) == Y).mean())
        print('  ep%d val acc %.4f' % (ep, acc), flush=True)
    return m, P, Y


def aucs(P, Y):
    from sklearn.metrics import roc_auc_score
    from scipy.special import softmax
    S = softmax(P, 1)
    out = {}
    for k in range(3):
        yk = (Y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            out[CLASSES[k]] = float(roc_auc_score(yk, S[:, k]))
    out['macro'] = float(np.mean([v for k, v in out.items()]))
    out['accuracy'] = float((P.argmax(1) == Y).mean())
    return out


@torch.no_grad()
def cache(m, names, tag, bs=32):
    fp = os.path.join(SCRATCH, 'F_%s.npy' % tag)
    if os.path.exists(fp):
        return np.load(fp, mmap_mode='r')
    ld = torch.utils.data.DataLoader(DS(names, np.zeros(len(names), int)),
                                     batch_size=bs, num_workers=0)
    out = []
    for x, _ in ld:
        f = m.forward_features(x.to(dev))
        out.append(f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2)
                   .float().cpu().numpy())
    F = np.concatenate(out)
    assert len(F) == len(names), (tag, len(F), len(names))
    np.save(fp, F)
    return np.load(fp, mmap_mode='r')


def gt_of(name):
    from PIL import Image
    p = os.path.join(MSK, name + '_segmentation.png')
    g = Image.open(p).convert('L').resize((SIZE, SIZE), Image.NEAREST)
    return (np.asarray(g) > 127).astype(np.uint8)


@torch.no_grad()
def NS_batched(head, XA, XB, c, mask):
    """Same definition as mlib.NS_unit, one batched head call per direction.

    `mask` is a (M, T) boolean: row i is True on the cells of unit i.  Verified
    against rtlib's literal loop in `verify()`; no algebra is used, the head is
    run on literally perturbed cell matrices.
    """
    zA = head(XA[None])[0, c]
    zB = head(XB[None])[0, c]
    mk = mask[:, :, None]
    Xn = torch.where(mk, XB[None], XA[None])
    Xs = torch.where(mk, XA[None], XB[None])
    N = (zA - head(Xn)[:, c]).cpu().numpy().astype(np.float64)
    S = (head(Xs)[:, c] - zB).cpu().numpy().astype(np.float64)
    return N, S


def unit_mask(units, T, device):
    m = torch.zeros(len(units), T, dtype=torch.bool, device=device)
    for i, u in enumerate(units):
        m[i, torch.as_tensor(u, device=device)] = True
    return m


def verify(head, XA, XB, c, T):
    """batched == rtlib literal loop, exactly."""
    N, S, _, _ = rtlib.NS_literal(head, XA, XB, c)
    mk = unit_mask([np.array([t]) for t in range(T)], T, XA.device)
    Nb, Sb = NS_batched(head, XA, XB, c, mk)
    return float(np.abs(N - Nb).max()), float(np.abs(S - Sb).max())


def main():
    unpack()
    names, y = mlib.isic_frame()
    print('labelled ISIC images with masks: %d  %s' %
          (len(names), np.bincount(y, minlength=3).tolist()), flush=True)
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(names))
    ntr = int(.7 * len(names))
    tr, va = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    res = dict(n=len(names), n_train=len(tr), n_val=len(va),
               class_counts=np.bincount(y, minlength=3).tolist())

    ck = os.path.join(SCRATCH, 'isic_r50.pt')
    import timm
    if os.path.exists(ck):
        m = timm.create_model('resnet50', pretrained=False, num_classes=3).to(dev)
        m.load_state_dict(torch.load(ck))
        m.eval()
        vl = torch.utils.data.DataLoader(DS([names[i] for i in va], y[va]),
                                         batch_size=32, num_workers=4)
        P, Y = [], []
        with torch.no_grad():
            for x, t in vl:
                P.append(m(x.to(dev)).cpu().numpy())
                Y.append(t.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
    else:
        m, P, Y = train(names, y, tr, va)
        torch.save(m.state_dict(), ck)
    m.eval()
    res['gap_val'] = aucs(P, Y)
    print('GAP val:', res['gap_val'], flush=True)

    # scope rule on the trained model, real image, never at random init
    x = mlib.load_rgb(os.path.join(IMG, names[va[0]] + '.jpg'), SIZE)[None].to(dev)
    with torch.no_grad():
        feat = m.forward_features(x)
    d = frozen_rule.structural(m, feat)
    d['post'] = [list(t) for t in d['post']]
    res['frozen_rule'] = d
    print('frozen rule:', d['verdict'], d['agg'], flush=True)

    vnames = [names[i] for i in va]
    F = cache(m, vnames, 'isic_val')
    Ftr = cache(m, [names[i] for i in tr], 'isic_train')
    T, C = F.shape[1], F.shape[2]
    G = int(round(T ** .5))
    res['T'], res['C'], res['G'] = T, C, G
    W = m.fc.weight.detach().double() if hasattr(m, 'fc') else None
    if W is None:
        W = m.get_classifier().weight.detach().double()
        b = m.get_classifier().bias.detach().double()
    else:
        b = m.fc.bias.detach().double()

    # off-degenerate, condition A: aggregator swapped, weights untouched
    heads = {'gap': mlib.head_of('gap', W, b).double().to(dev)}
    for k in OFF:
        heads[k] = mlib.head_of(k, W, b).double().to(dev)
    condA = {}
    for k, h in heads.items():
        Z = []
        with torch.no_grad():
            for i in range(0, len(F), 128):
                X = torch.from_numpy(np.ascontiguousarray(F[i:i + 128])).double().to(dev)
                Z.append(h(X).cpu().numpy())
        condA[k] = aucs(np.concatenate(Z), y[va])
        print('  condA %-9s %s' % (k, condA[k]), flush=True)
    res['condA'] = condA

    # condition B: head re-fit at that aggregator, same body
    condB = {}
    for k in ['gap'] + OFF:
        hh = mlib.head_of(k, W.float().clone(), b.float().clone()).to(dev)
        hh.W.requires_grad_(True)
        hh.b.requires_grad_(True)
        opt = torch.optim.AdamW([hh.W, hh.b], lr=1e-3, weight_decay=1e-4)
        Xtr = torch.from_numpy(np.asarray(Ftr)).float()
        ytr = torch.from_numpy(y[tr]).long()
        for ep in range(30):
            pm = torch.randperm(len(ytr))
            for i in range(0, len(ytr) - 63, 64):
                j = pm[i:i + 64]
                loss = Fn.cross_entropy(hh(Xtr[j].to(dev)), ytr[j].to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
        Z = []
        with torch.no_grad():
            for i in range(0, len(F), 128):
                X = torch.from_numpy(np.ascontiguousarray(F[i:i + 128])).float().to(dev)
                Z.append(hh(X).cpu().numpy())
        condB[k] = aucs(np.concatenate(Z), y[va])
        print('  condB %-9s %s' % (k, condB[k]), flush=True)
    res['condB'] = condB

    # ---- certificates -----------------------------------------------------
    cert = {}
    pairs = [(0, 100), (5, 250), (17, 400)]
    for k, h in heads.items():
        rows = []
        for ia, ib in pairs:
            XA = torch.from_numpy(np.ascontiguousarray(F[ia])).double()
            XB = torch.from_numpy(np.ascontiguousarray(F[ib])).double()
            hc = mlib.head_of(k, W.cpu(), b.cpu()).double()
            dd = mlib.certify(hc, XA, XB, 0, np.random.RandomState(0))
            rows.append(dict(maxNS=dd['maxNS'], Rhalf=dd['Rhalf'], I2=dd['I2'],
                             corrNS=dd['corr_pearson'], additive=dd['additive']))
        cert[k] = dict(pairs=rows,
                       additive_all=bool(all(r['additive'] for r in rows)),
                       worst_maxNS=max(r['maxNS'] for r in rows),
                       worst_Rhalf=max(r['Rhalf'] for r in rows),
                       worst_I2=max(r['I2'] for r in rows))
        print('  cert %-9s additive=%s worst maxNS %.2e' %
              (k, cert[k]['additive_all'], cert[k]['worst_maxNS']), flush=True)
    res['certificate'] = cert

    # ---- maps -------------------------------------------------------------
    units1 = unit_mask([np.array([t]) for t in range(T)], T, dev)
    units3 = unit_mask(mlib.blocks_of(G, 3), T, dev)
    _XA = torch.from_numpy(np.ascontiguousarray(F[0])).double().to(dev)
    _XB = torch.from_numpy(np.ascontiguousarray(F[1])).double().to(dev)
    res['batched_vs_literal'] = {k: verify(h, _XA, _XB, 0, T)
                                 for k, h in heads.items()}
    print('batched-vs-literal:', res['batched_vs_literal'], flush=True)
    # baseline bank: 8 validation images of a DIFFERENT class than the target
    rsb = np.random.RandomState(0)
    bankidx = {}
    for c in range(3):
        cand = [i for i in range(len(vnames)) if y[va][i] != c]
        bankidx[c] = sorted(rsb.choice(cand, 8, replace=False).tolist())
    rmap = np.random.RandomState(7)

    sc = {}
    recs = []
    t0 = time.time()
    for i, n in enumerate(vnames):
        c = int(y[va][i])
        XA = torch.from_numpy(np.ascontiguousarray(F[i])).double().to(dev)
        Xb = [torch.from_numpy(np.ascontiguousarray(F[j])).double().to(dev)
              for j in bankidx[c]]
        mp = {}
        for hk, h in heads.items():
            for uname, units in (('1x1', units1), ('3x3', units3)):
                Ns, Ss = [], []
                for XB in Xb:
                    N, S = NS_batched(h, XA, XB, c, units)
                    Ns.append(N)
                    Ss.append(S)
                if hk == 'gap':
                    mp['collapsed_%s' % uname] = np.mean(Ns, 0)
                else:
                    mp['N_%s_%s' % (hk, uname)] = np.mean(Ns, 0)
                    mp['S_%s_%s' % (hk, uname)] = np.mean(Ss, 0)
            mp['gradcam_%s' % hk] = mlib.gradcam(h, XA, c)
        mp['actnorm'] = mlib.actnorm(XA)
        mp['center'] = mlib.center_prior(G)
        mp['random'] = rmap.rand(T)
        recs.append((n, c, {k: v.tolist() for k, v in mp.items()}))
        if i % 50 == 0:
            print('  map %d/%d %.0fs' % (i, len(vnames), time.time() - t0), flush=True)
    res['n_scored'] = len(recs)
    res['map_names'] = sorted(recs[0][2].keys())
    json.dump(dict(items=[dict(name=a, cls=b_, maps=c_) for a, b_, c_ in recs],
                   G=G, T=T), open(os.path.join(mlib.OUT, 'm5_isic_maps.json'), 'w'))

    # ---- scoring ----------------------------------------------------------
    mapnames = res['map_names']
    S = {mn: dict(hit=[], ap=[], dice={q: [] for q in QGRID}) for mn in mapnames}
    gfrac = []
    for (n, c, mp) in recs:
        g = gt_of(n)
        gfrac.append(float(g.mean()))
        for mn in mapnames:
            S[mn]['hit'].append(mlib.pointing_hit(mp[mn], G, g, SIZE))
            S[mn]['ap'].append(mlib.pixel_ap(mp[mn], G, g, SIZE))
            for q in QGRID:
                S[mn]['dice'][q].append(mlib.dice_at(mp[mn], G, g, SIZE, q))
    res['gt_fraction_mean'] = float(np.mean(gfrac))

    nn_ = len(recs)
    fold = np.random.RandomState(0).randint(0, 2, nn_)
    dice_ct = {}
    for mn in mapnames:
        v = np.full(nn_, np.nan)
        for f in (0, 1):
            trm, tem = fold != f, fold == f
            best, bq = -1, QGRID[0]
            for q in QGRID:
                a = np.asarray(S[mn]['dice'][q], float)[trm]
                if np.nanmean(a) > best:
                    best, bq = np.nanmean(a), q
            v[tem] = np.asarray(S[mn]['dice'][bq], float)[tem]
        dice_ct[mn] = v
    res['summary'] = {mn: dict(hit=mlib.boot_ci(S[mn]['hit']),
                               ap=mlib.boot_ci(S[mn]['ap']),
                               dice=mlib.boot_ci(dice_ct[mn]))
                      for mn in mapnames}

    def cmp(A, B, metric):
        a = np.asarray(S[A][metric], float) if metric in ('hit', 'ap') else dice_ct[A]
        bb = np.asarray(S[B][metric], float) if metric in ('hit', 'ap') else dice_ct[B]
        d = mlib.paired(a, bb)
        d.update(A=A, B=B, metric=metric, mean_a=float(np.nanmean(a)),
                 mean_b=float(np.nanmean(bb)))
        return d

    pre = {}
    for unit in ('1x1', '3x3'):
        for temp in OFF:
            for metric in ('dice', 'hit', 'ap'):
                pre['H1_%s_%s_%s' % (temp, unit, metric)] = cmp(
                    'N_%s_%s' % (temp, unit), 'collapsed_%s' % unit, metric)
                pre['H2_%s_%s_%s' % (temp, unit, metric)] = cmp(
                    'N_%s_%s' % (temp, unit), 'S_%s_%s' % (temp, unit), metric)
    res['prespecified'] = pre
    res['extra'] = {('%s_vs_center_%s' % (a, mt)): cmp(a, 'center', mt)
                    for mt in ('dice', 'hit', 'ap')
                    for a in ('collapsed_1x1', 'N_lse8.0_1x1', 'S_lse8.0_1x1',
                              'gradcam_gap')}
    mlib.save('m5_isic.json', res)
    P1 = pre['H1_lse8.0_1x1_dice']
    P2 = pre['H2_lse8.0_1x1_dice']
    print('\nISIC PRIMARY (Dice, 1x1, T=+8), n=%d' % nn_)
    print('  N %.4f  collapsed %.4f  diff %+.4f [%+.4f,%+.4f] p=%.3g'
          % (P1['mean_a'], P1['mean_b'], P1['diff'], P1['lo'], P1['hi'], P1['p_wilcoxon']))
    print('  N %.4f  S         %.4f  diff %+.4f [%+.4f,%+.4f] p=%.3g'
          % (P2['mean_a'], P2['mean_b'], P2['diff'], P2['lo'], P2['hi'], P2['p_wilcoxon']))


if __name__ == '__main__':
    main()
