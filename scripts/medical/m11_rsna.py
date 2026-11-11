"""m11 -- third dataset: RSNA Pneumonia Detection Challenge, radiologist-drawn
boxes, with torchxrayvision's own `densenet121-res224-rsna` DenseNet-121.

Same pre-specification as NIH: primary metric = pointing hit rate, unit 1x1,
T = +8; comparators, certificates and controls identical.
"""
import os
import sys
import csv
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib
from mlib import ROOT, SCRATCH, dev
import m1_model as m1
import m3_maps as m3
import m4_stats as m4

D = os.path.join(ROOT, 'data', 'rsna')
DCM = os.path.join(D, 'dcm')
SIZE = 224
ORIG = 1024.
OFF = ['lse8.0', 'lse-8.0']
CLS = 8                                    # xrv output index 'Pneumonia'


def load_dcm(path, size=SIZE):
    import pydicom
    a = pydicom.dcmread(path).pixel_array.astype(np.float32)
    from PIL import Image
    im = Image.fromarray(a.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    v = np.asarray(im, dtype=np.float32)
    return torch.from_numpy((2 * (v / 255.) - 1.) * 1024.)[None]


class DS(torch.utils.data.Dataset):
    def __init__(s, ids):
        s.ids = ids

    def __len__(s):
        return len(s.ids)

    def __getitem__(s, i):
        return load_dcm(os.path.join(DCM, s.ids[i] + '.dcm')), 0


def gt_mask(boxes):
    g = np.zeros((SIZE, SIZE), np.uint8)
    for (x, y, w, h) in boxes:
        sfac = SIZE / ORIG
        g[max(0, int(round(y * sfac))):min(SIZE, int(round((y + h) * sfac))),
          max(0, int(round(x * sfac))):min(SIZE, int(round((x + w) * sfac)))] = 1
    return g


def main():
    import torchxrayvision as xrv
    xm = xrv.models.DenseNet(weights='densenet121-res224-rsna')
    body = m1.XrvBody(xm).eval().to(dev)
    W = xm.classifier.weight.detach().double().cpu()
    b = xm.classifier.bias.detach().double().cpu()

    rows = list(csv.DictReader(open(os.path.join(D, 'stage_2_train_labels.csv'))))
    boxes = {}
    target = {}
    for r in rows:
        target[r['patientId']] = max(target.get(r['patientId'], 0), int(r['Target']))
        if r['Target'] == '1' and r['x']:
            boxes.setdefault(r['patientId'], []).append(
                (float(r['x']), float(r['y']), float(r['width']), float(r['height'])))
    have = sorted(n[:-4] for n in os.listdir(DCM))
    y = np.array([target[n] for n in have], dtype=np.float32)
    print('RSNA images on disk %d (positive %d)' % (len(have), int(y.sum())), flush=True)
    res = dict(n=len(have), n_positive=int(y.sum()), cls_index=CLS,
               targets=list(xm.targets))

    fp = os.path.join(SCRATCH, 'F_rsna.npy')
    if os.path.exists(fp):
        F = np.load(fp, mmap_mode='r')
    else:
        ld = torch.utils.data.DataLoader(DS(have), batch_size=32, num_workers=4)
        out = []
        t0 = time.time()
        with torch.no_grad():
            for i, (x, _) in enumerate(ld):
                f = body(x.to(dev))
                out.append(f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2)
                           .float().cpu().numpy())
                if i % 40 == 0:
                    print('  cache %d/%d %.0fs' % (i * 32, len(have), time.time() - t0), flush=True)
        np.save(fp, np.concatenate(out))
        F = np.load(fp, mmap_mode='r')
    T, C = F.shape[1], F.shape[2]
    G = int(round(T ** .5))
    res['T'], res['C'] = T, C

    from sklearn.metrics import roc_auc_score
    heads, condA, cert = {}, {}, {}
    for k in ['gap'] + OFF:
        h = mlib.head_of(k, W, b).double().to(dev)
        heads[k] = h
        Z = []
        with torch.no_grad():
            for i in range(0, len(F), 128):
                X = torch.from_numpy(np.ascontiguousarray(F[i:i + 128])).double().to(dev)
                Z.append(h(X).cpu().numpy())
        Z = np.concatenate(Z)
        condA[k] = dict(auc_pneumonia=float(roc_auc_score(y, Z[:, CLS])),
                        auc_lung_opacity=float(roc_auc_score(y, Z[:, 16])))
        rows_ = []
        for ia, ib in [(0, 1000), (7, 2000), (33, 3000)]:
            XA = torch.from_numpy(np.ascontiguousarray(F[ia])).double()
            XB = torch.from_numpy(np.ascontiguousarray(F[ib])).double()
            hc = mlib.head_of(k, W, b).double()
            dd = mlib.certify(hc, XA, XB, CLS, np.random.RandomState(0))
            rows_.append(dict(maxNS=dd['maxNS'], Rhalf=dd['Rhalf'], I2=dd['I2'],
                              corrNS=dd['corr_pearson'], additive=dd['additive']))
        cert[k] = dict(pairs=rows_, additive_all=bool(all(r['additive'] for r in rows_)),
                       worst_maxNS=max(r['maxNS'] for r in rows_),
                       worst_Rhalf=max(r['Rhalf'] for r in rows_),
                       worst_I2=max(r['I2'] for r in rows_))
        print('  %-9s AUC(pneumonia) %.4f  additive=%s worst maxNS %.2e'
              % (k, condA[k]['auc_pneumonia'], cert[k]['additive_all'],
                 cert[k]['worst_maxNS']), flush=True)
    res['condA'] = condA
    res['certificate'] = cert

    idx = {n: i for i, n in enumerate(have)}
    items = [n for n in have if n in boxes]
    negs = [i for i, n in enumerate(have) if target[n] == 0]
    bank = sorted(np.random.RandomState(0).choice(negs, 8, replace=False).tolist())
    Xb = [torch.from_numpy(np.ascontiguousarray(F[i])).double().to(dev) for i in bank]
    u1 = m3.unit_mask([np.array([t]) for t in range(T)], T, dev)
    u3 = m3.unit_mask(mlib.blocks_of(G, 3), T, dev)
    res['batched_vs_literal'] = {k: m3.verify(h, Xb[0], Xb[1], CLS, T)
                                 for k, h in heads.items()}
    print('scored items %d' % len(items), flush=True)

    rmap = np.random.RandomState(7)
    recs = []
    t0 = time.time()
    for i, n in enumerate(items):
        XA = torch.from_numpy(np.ascontiguousarray(F[idx[n]])).double().to(dev)
        mp = {}
        for hk, h in heads.items():
            for uname, um in (('1x1', u1), ('3x3', u3)):
                Ns, Ss = [], []
                for XB in Xb:
                    N, S = m3.NS_batched(h, XA, XB, CLS, um)
                    Ns.append(N)
                    Ss.append(S)
                if hk == 'gap':
                    mp['collapsed_%s' % uname] = np.mean(Ns, 0)
                else:
                    mp['N_%s_%s' % (hk, uname)] = np.mean(Ns, 0)
                    mp['S_%s_%s' % (hk, uname)] = np.mean(Ss, 0)
            mp['gradcam_%s' % hk] = mlib.gradcam(h, XA, CLS)
        mp['actnorm'] = mlib.actnorm(XA)
        mp['center'] = mlib.center_prior(G)
        mp['random'] = rmap.rand(T)
        recs.append((n, mp))
        if i % 250 == 0:
            print('  map %d/%d %.0fs' % (i, len(items), time.time() - t0), flush=True)

    mapnames = sorted(recs[0][1].keys())
    S = {mn: dict(hit=[], ap=[]) for mn in mapnames}
    gf = []
    for (n, mp) in recs:
        g = gt_mask(boxes[n])
        if g.sum() == 0:
            continue
        gf.append(float(g.mean()))
        for mn in mapnames:
            S[mn]['hit'].append(mlib.pointing_hit(mp[mn], G, g, SIZE))
            S[mn]['ap'].append(mlib.pixel_ap(mp[mn], G, g, SIZE))
    res['n_scored'] = len(gf)
    res['gt_fraction_mean'] = float(np.mean(gf))
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
    res['extra'] = {'%s_vs_%s_%s' % (a, b_, mt): cmp(a, b_, mt)
                    for mt in ('hit', 'ap')
                    for a, b_ in (('collapsed_1x1', 'center'),
                                  ('collapsed_1x1', 'actnorm'),
                                  ('collapsed_1x1', 'gradcam_gap'),
                                  ('N_lse8.0_1x1', 'center'))}
    mlib.save('m11_rsna.json', res)
    P = pre['H1_lse8.0_1x1_hit']
    Q = pre['H2_lse8.0_1x1_hit']
    print('\nRSNA PRIMARY (hit, 1x1, T=+8), n=%d' % res['n_scored'])
    print('  N %.4f  collapsed %.4f  diff %+.4f [%+.4f,%+.4f] p=%.3g'
          % (P['mean_a'], P['mean_b'], P['diff'], P['lo'], P['hi'], P['p_mcnemar']))
    print('  N %.4f  S         %.4f  diff %+.4f [%+.4f,%+.4f] p=%.3g'
          % (Q['mean_a'], Q['mean_b'], Q['diff'], Q['lo'], Q['hi'], Q['p_mcnemar']))


if __name__ == '__main__':
    main()
