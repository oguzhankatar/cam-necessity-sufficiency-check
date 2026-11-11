"""m6 -- robustness check: the same comparison with a DenseNet-121 we train
ourselves on the official NIH train_val split, so that the held-out test images
(which carry all 880 boxes) are genuinely unseen.

Same pre-specification as m3/m4; only the weights change.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib
from mlib import ROOT, SCRATCH, dev, NIH_FINDINGS
import frozen_rule
import m3_maps as m3
import m4_stats as m4

IMG = os.path.join(ROOT, 'data', 'nih', 'images')
SIZE = 224
OFF = ['lse8.0', 'lse-8.0']


class DS(torch.utils.data.Dataset):
    def __init__(s, names, Y, aug=False):
        s.n, s.Y, s.aug = names, Y, aug

    def __len__(s):
        return len(s.n)

    def __getitem__(s, i):
        x = mlib.load_gray(os.path.join(IMG, s.n[i]), SIZE)
        if s.aug and np.random.rand() < .5:
            x = torch.flip(x, [2])
        return x, torch.from_numpy(s.Y[i])


def train(tn, tY, vn, vY, epochs=8, bs=32):
    import timm
    torch.manual_seed(0)
    m = timm.create_model('densenet121', pretrained=True,
                          num_classes=len(NIH_FINDINGS)).to(dev)
    ld = torch.utils.data.DataLoader(DS(tn, tY, aug=True), batch_size=bs,
                                     shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(ld))
    pos = tY.sum(0) + 1
    w = torch.tensor((len(tY) - pos) / pos, dtype=torch.float32, device=dev)
    for ep in range(epochs):
        m.train()
        t0 = time.time()
        for x, y in ld:
            loss = Fn.binary_cross_entropy_with_logits(m(x.to(dev)), y.to(dev),
                                                       pos_weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        print('  ep%d %.0fs' % (ep, time.time() - t0), flush=True)
    m.eval()
    return m


@torch.no_grad()
def cache(m, names, tag, bs=32):
    fp = os.path.join(SCRATCH, 'F_%s.npy' % tag)
    if os.path.exists(fp):
        return np.load(fp, mmap_mode='r')
    ld = torch.utils.data.DataLoader(DS(names, np.zeros((len(names), 1), np.float32)),
                                     batch_size=bs, num_workers=4)
    out = []
    for x, _ in ld:
        f = m.forward_features(x.to(dev))
        out.append(f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2)
                   .float().cpu().numpy())
    F = np.concatenate(out)
    np.save(fp, F)
    return np.load(fp, mmap_mode='r')


def main():
    names, Y, split = mlib.nih_frame()
    tn = [n for n, s in zip(names, split) if s == 'train']
    tY = Y[split == 'train']
    vn = [n for n, s in zip(names, split) if s == 'test']
    vY = Y[split == 'test']
    print('train %d, test %d' % (len(tn), len(vn)), flush=True)
    res = dict(n_train=len(tn), n_test=len(vn))

    ck = os.path.join(SCRATCH, 'own_d121.pt')
    import timm
    if os.path.exists(ck):
        m = timm.create_model('densenet121', pretrained=False,
                              num_classes=len(NIH_FINDINGS)).to(dev)
        m.load_state_dict(torch.load(ck))
        m.eval()
    else:
        m = train(tn, tY, vn, vY)
        torch.save(m.state_dict(), ck)

    x = mlib.load_gray(os.path.join(IMG, vn[0]), SIZE)[None].to(dev)
    with torch.no_grad():
        feat = m.forward_features(x)
    d = frozen_rule.structural(m, feat)
    d['post'] = [list(t) for t in d['post']]
    res['frozen_rule'] = d
    print('frozen rule:', d['verdict'], d['agg'], flush=True)

    F = cache(m, vn, 'own_test')
    T, C = F.shape[1], F.shape[2]
    G = int(round(T ** .5))
    W = m.classifier.weight.detach().double()
    b = m.classifier.bias.detach().double()
    heads = {'gap': mlib.head_of('gap', W, b).double().to(dev)}
    for k in OFF:
        heads[k] = mlib.head_of(k, W, b).double().to(dev)

    from sklearn.metrics import roc_auc_score
    condA = {}
    for k, h in heads.items():
        Z = []
        with torch.no_grad():
            for i in range(0, len(F), 128):
                X = torch.from_numpy(np.ascontiguousarray(F[i:i + 128])).double().to(dev)
                Z.append(h(X).cpu().numpy())
        Z = np.concatenate(Z)
        a = {f: float(roc_auc_score(vY[:, j], Z[:, j]))
             for j, f in enumerate(NIH_FINDINGS)
             if 0 < vY[:, j].sum() < len(vY)}
        condA[k] = dict(auc=a, mean_auc=float(np.mean(list(a.values()))))
        print('  %-9s mean AUC %.4f' % (k, condA[k]['mean_auc']), flush=True)
    res['condA'] = condA

    cert = {}
    for k, h in heads.items():
        rows = []
        for ia, ib in [(0, 500), (11, 1200), (37, 2000)]:
            XA = torch.from_numpy(np.ascontiguousarray(F[ia])).double()
            XB = torch.from_numpy(np.ascontiguousarray(F[ib])).double()
            hc = mlib.head_of(k, W.cpu(), b.cpu()).double()
            dd = mlib.certify(hc, XA, XB, 0, np.random.RandomState(0))
            rows.append(dict(maxNS=dd['maxNS'], Rhalf=dd['Rhalf'], I2=dd['I2'],
                             corrNS=dd['corr_pearson'], additive=dd['additive']))
        cert[k] = dict(pairs=rows, additive_all=bool(all(r['additive'] for r in rows)),
                       worst_maxNS=max(r['maxNS'] for r in rows))
        print('  cert %-9s additive=%s worst maxNS %.2e'
              % (k, cert[k]['additive_all'], cert[k]['worst_maxNS']), flush=True)
    res['certificate'] = cert

    # ---- maps on the box items -------------------------------------------
    idx = {n: i for i, n in enumerate(vn)}
    boxes = mlib.nih_boxes()
    items = []
    for n, bs_ in sorted(boxes.items()):
        if n in idx:
            for f in sorted(set(x[0] for x in bs_)):
                if f in NIH_FINDINGS:
                    items.append((n, f, NIH_FINDINGS.index(f)))
    nofind = [i for i, n in enumerate(vn) if vY[i].sum() == 0 and n not in boxes]
    bank = sorted(np.random.RandomState(0).choice(nofind, 8, replace=False).tolist())
    Xb = [torch.from_numpy(np.ascontiguousarray(F[i])).double().to(dev) for i in bank]
    u1 = m3.unit_mask([np.array([t]) for t in range(T)], T, dev)
    u3 = m3.unit_mask(mlib.blocks_of(G, 3), T, dev)
    res['batched_vs_literal'] = {k: m3.verify(h, Xb[0], Xb[1], 0, T)
                                 for k, h in heads.items()}

    sc = {}
    rmap = np.random.RandomState(7)
    recs = []
    t0 = time.time()
    for i, (n, f, c) in enumerate(items):
        XA = torch.from_numpy(np.ascontiguousarray(F[idx[n]])).double().to(dev)
        mp = {}
        for hk, h in heads.items():
            for uname, um in (('1x1', u1), ('3x3', u3)):
                Ns, Ss = [], []
                for XB in Xb:
                    N, S = m3.NS_batched(h, XA, XB, c, um)
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
        recs.append((n, f, mp))
        if i % 100 == 0:
            print('  map %d/%d %.0fs' % (i, len(items), time.time() - t0), flush=True)

    mapnames = sorted(recs[0][2].keys())
    S = {mn: dict(hit=[], ap=[]) for mn in mapnames}
    for (n, f, mp) in recs:
        g = m4.gt_mask(boxes[n], f)
        if g.sum() == 0:
            continue
        for mn in mapnames:
            S[mn]['hit'].append(mlib.pointing_hit(mp[mn], G, g, m4.SIZE))
            S[mn]['ap'].append(mlib.pixel_ap(mp[mn], G, g, m4.SIZE))
    res['n_scored'] = len(S[mapnames[0]]['hit'])
    res['summary'] = {mn: dict(hit=mlib.boot_ci(S[mn]['hit']),
                               ap=mlib.boot_ci(S[mn]['ap'])) for mn in mapnames}

    def cmp(A, B, metric):
        a, bb = np.asarray(S[A][metric], float), np.asarray(S[B][metric], float)
        d = mlib.paired(a, bb)
        if metric == 'hit':
            d.update(m4.mcnemar(a, bb))
        d.update(A=A, B=B, metric=metric, mean_a=float(np.nanmean(a)),
                 mean_b=float(np.nanmean(bb)))
        return d

    pre = {}
    for unit in ('1x1', '3x3'):
        for temp in OFF:
            for metric in ('hit', 'ap'):
                pre['H1_%s_%s_%s' % (temp, unit, metric)] = cmp(
                    'N_%s_%s' % (temp, unit), 'collapsed_%s' % unit, metric)
                pre['H2_%s_%s_%s' % (temp, unit, metric)] = cmp(
                    'N_%s_%s' % (temp, unit), 'S_%s_%s' % (temp, unit), metric)
    res['prespecified'] = pre
    res['extra'] = {'%s_vs_center_%s' % (a, mt): cmp(a, 'center', mt)
                    for mt in ('hit', 'ap')
                    for a in ('collapsed_1x1', 'N_lse8.0_1x1', 'S_lse8.0_1x1',
                              'gradcam_gap')}
    mlib.save('m6_own.json', res)
    P = pre['H1_lse8.0_1x1_hit']
    Q = pre['H2_lse8.0_1x1_hit']
    print('\nOWN-MODEL PRIMARY (hit, 1x1, T=+8), n=%d' % res['n_scored'])
    print('  N %.4f  collapsed %.4f  diff %+.4f [%+.4f,%+.4f] p=%.3g'
          % (P['mean_a'], P['mean_b'], P['diff'], P['lo'], P['hi'], P['p_mcnemar']))
    print('  N %.4f  S         %.4f  diff %+.4f [%+.4f,%+.4f] p=%.3g'
          % (Q['mean_a'], Q['mean_b'], Q['diff'], Q['lo'], Q['hi'], Q['p_mcnemar']))


if __name__ == '__main__':
    main()
