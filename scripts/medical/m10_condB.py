"""m10 -- condition B on NIH: the head is RE-FITTED at each aggregator on the
official train split, on top of the same frozen torchxrayvision body.  This
answers the obvious objection to condition A -- that the necessity map of the
LSE readout looks bad only because the LSE readout is a worse classifier.
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
import m1_model as m1
import m3_maps as m3
import m4_stats as m4

IMG = os.path.join(ROOT, 'data', 'nih', 'images')
OFF = ['lse8.0', 'lse-8.0']


def fit_head(kind, Ftr, Ytr, C, K, epochs=25, bs=128, seed=0):
    torch.manual_seed(seed)
    W = torch.zeros(K, C, device=dev)
    torch.nn.init.normal_(W, std=0.01)
    b = torch.zeros(K, device=dev)
    W.requires_grad_(True)
    b.requires_grad_(True)
    h = mlib.head_of(kind, W, b).to(dev)
    h.W = torch.nn.Parameter(W)
    h.b = torch.nn.Parameter(b)
    opt = torch.optim.AdamW([h.W, h.b], lr=3e-3, weight_decay=1e-4)
    yt = torch.from_numpy(Ytr).to(dev)
    pos = Ytr.sum(0) + 1
    pw = torch.tensor((len(Ytr) - pos) / pos, dtype=torch.float32, device=dev)
    n = len(Ytr)
    rs = np.random.RandomState(seed)
    for ep in range(epochs):
        perm = rs.permutation(n)
        for i in range(0, n - bs + 1, bs):
            j = np.sort(perm[i:i + bs])
            X = torch.from_numpy(np.ascontiguousarray(Ftr[j])).to(dev).float()
            loss = Fn.binary_cross_entropy_with_logits(h(X), yt[j], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return h.W.detach().double(), h.b.detach().double()


def main():
    import torchxrayvision as xrv
    xm = xrv.models.DenseNet(weights='densenet121-res224-nih')
    body = m1.XrvBody(xm).eval().to(dev)

    names, Y, split = mlib.nih_frame()
    tn = [n for n, s in zip(names, split) if s == 'train']
    tY = Y[split == 'train']
    Ftr = m1.cache(body, tn, tY, m1.xrv_load, 'xrv_train')
    F = np.load(os.path.join(SCRATCH, 'F_xrv_test.npy'), mmap_mode='r')
    vn = list(np.load(os.path.join(SCRATCH, 'names_test.npy')))
    vY = np.load(os.path.join(SCRATCH, 'Y_test.npy'))
    T, C = F.shape[1], F.shape[2]
    G = int(round(T ** .5))
    K = len(NIH_FINDINGS)
    res = dict(n_train=len(tn), n_test=len(vn), T=T, C=C)

    from sklearn.metrics import roc_auc_score
    heads, condB, cert = {}, {}, {}
    for kind in ['gap'] + OFF:
        W, b = fit_head(kind, Ftr, tY, C, K)
        h = mlib.head_of(kind, W, b).double().to(dev)
        heads[kind] = h
        Z = []
        with torch.no_grad():
            for i in range(0, len(F), 128):
                X = torch.from_numpy(np.ascontiguousarray(F[i:i + 128])).double().to(dev)
                Z.append(h(X).cpu().numpy())
        Z = np.concatenate(Z)
        a = {f: float(roc_auc_score(vY[:, j], Z[:, j]))
             for j, f in enumerate(NIH_FINDINGS) if 0 < vY[:, j].sum() < len(vY)}
        condB[kind] = dict(auc=a, mean_auc=float(np.mean(list(a.values()))))
        rows = []
        for ia, ib in [(0, 1000), (7, 2500), (33, 4000)]:
            XA = torch.from_numpy(np.ascontiguousarray(F[ia])).double()
            XB = torch.from_numpy(np.ascontiguousarray(F[ib])).double()
            hc = mlib.head_of(kind, W.cpu(), b.cpu()).double()
            dd = mlib.certify(hc, XA, XB, 0, np.random.RandomState(0))
            rows.append(dict(maxNS=dd['maxNS'], Rhalf=dd['Rhalf'], I2=dd['I2'],
                             corrNS=dd['corr_pearson'], additive=dd['additive']))
        cert[kind] = dict(pairs=rows, additive_all=bool(all(r['additive'] for r in rows)),
                          worst_maxNS=max(r['maxNS'] for r in rows),
                          worst_Rhalf=max(r['Rhalf'] for r in rows),
                          worst_I2=max(r['I2'] for r in rows))
        print('  condB %-9s mean AUC %.4f  additive=%s worst maxNS %.2e'
              % (kind, condB[kind]['mean_auc'], cert[kind]['additive_all'],
                 cert[kind]['worst_maxNS']), flush=True)
    res['condB'] = condB
    res['certificate'] = cert

    idx = {n: i for i, n in enumerate(vn)}
    boxes = mlib.nih_boxes()
    items = [(n, f, NIH_FINDINGS.index(f))
             for n, bs_ in sorted(boxes.items()) if n in idx
             for f in sorted(set(x[0] for x in bs_)) if f in NIH_FINDINGS]
    nofind = [i for i, n in enumerate(vn) if vY[i].sum() == 0 and n not in boxes]
    bank = sorted(np.random.RandomState(0).choice(nofind, 8, replace=False).tolist())
    Xb = [torch.from_numpy(np.ascontiguousarray(F[i])).double().to(dev) for i in bank]
    u1 = m3.unit_mask([np.array([t]) for t in range(T)], T, dev)
    u3 = m3.unit_mask(mlib.blocks_of(G, 3), T, dev)
    res['batched_vs_literal'] = {k: m3.verify(h, Xb[0], Xb[1], 0, T)
                                 for k, h in heads.items()}

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
        if i % 200 == 0:
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
    mlib.save('m10_condB.json', res)
    P = pre['H1_lse8.0_1x1_hit']
    Q = pre['H2_lse8.0_1x1_hit']
    print('\nCOND-B PRIMARY (hit, 1x1, T=+8), n=%d' % res['n_scored'])
    print('  N %.4f  collapsed %.4f  diff %+.4f [%+.4f,%+.4f] p=%.3g'
          % (P['mean_a'], P['mean_b'], P['diff'], P['lo'], P['hi'], P['p_mcnemar']))
    print('  N %.4f  S         %.4f  diff %+.4f [%+.4f,%+.4f] p=%.3g'
          % (Q['mean_a'], Q['mean_b'], Q['diff'], Q['lo'], Q['hi'], Q['p_mcnemar']))


if __name__ == '__main__':
    main()
