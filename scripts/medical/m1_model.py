"""m1 -- obtain / train the CXR classifier, cache its feature maps, and report
held-out AUC for the degenerate (GAP) readout and for the off-degenerate (LSE_T)
readouts under condition A (weights untouched, aggregator swapped).

Primary model: torchxrayvision `densenet121-res224-nih` -- a DenseNet-121 with
`fc(GAP(relu(features)))`, i.e. the Saporta / Arun model class, obtained rather
than fitted.  Secondary model: our own DenseNet-121 trained on the official
NIH train_val split (clean split, reported as a robustness check).
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib
from mlib import ROOT, SCRATCH, dev, NIH_FINDINGS

IMG = os.path.join(ROOT, 'data', 'nih', 'images')


# ------------------------------------------------------------------ models
def get_xrv():
    import torchxrayvision as xrv
    m = xrv.models.DenseNet(weights='densenet121-res224-nih')
    m.eval().to(dev)
    return m, list(m.targets)


class XrvBody(nn.Module):
    """image -> (B,1024,7,7), exactly the tensor xrv pools over."""

    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, x):
        return Fn.relu(s.m.features(x))


def xrv_load(path, size=224):
    from PIL import Image
    im = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    a = (2 * (a / 255.) - 1.) * 1024.                 # xrv.utils.normalize(.,255)
    return torch.from_numpy(a)[None]                  # (1,H,W)


class DS(torch.utils.data.Dataset):
    def __init__(s, names, Y, loader):
        s.n, s.Y, s.l = names, Y, loader

    def __len__(s):
        return len(s.n)

    def __getitem__(s, i):
        return s.l(os.path.join(IMG, s.n[i])), torch.from_numpy(s.Y[i])


@torch.no_grad()
def cache(body, names, Y, loader, tag, bs=32):
    fp = os.path.join(SCRATCH, 'F_%s.npy' % tag)
    if os.path.exists(fp):
        return np.load(fp, mmap_mode='r')
    ld = torch.utils.data.DataLoader(DS(names, Y, loader), batch_size=bs,
                                     num_workers=4)
    out = []
    t0 = time.time()
    for i, (x, _) in enumerate(ld):
        f = body(x.to(dev))                            # (B,C,H,W)
        out.append(f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2)
                   .float().cpu().numpy())             # (B,T,C)
        if i % 40 == 0:
            print('  cache %s %d/%d %.0fs' % (tag, i * bs, len(names),
                                              time.time() - t0), flush=True)
    F = np.concatenate(out)
    np.save(fp, F)
    print('  cached', tag, F.shape, flush=True)
    return np.load(fp, mmap_mode='r')


def aucs(Z, Y):
    from sklearn.metrics import roc_auc_score
    out = {}
    for k in range(Y.shape[1]):
        y = Y[:, k]
        if 0 < y.sum() < len(y):
            out[k] = float(roc_auc_score(y, Z[:, k]))
    return out


@torch.no_grad()
def logits(head, F, bs=256):
    Z = []
    for i in range(0, len(F), bs):
        X = torch.from_numpy(np.ascontiguousarray(F[i:i + bs])).to(dev).float()
        Z.append(head(X).cpu().numpy())
    return np.concatenate(Z)


def main():
    names, Y, split = mlib.nih_frame()
    te = split == 'test'
    tr = split == 'train'
    print('images on disk: %d (test %d, train %d)' % (len(names), te.sum(), tr.sum()),
          flush=True)
    res = dict(n_total=int(len(names)), n_test=int(te.sum()), n_train=int(tr.sum()))

    # ---- primary: torchxrayvision densenet121-res224-nih ------------------
    m, targets = get_xrv()
    res['xrv_targets'] = targets
    body = XrvBody(m).eval()
    Fte = cache(body, [n for n, t in zip(names, te) if t], Y[te], xrv_load, 'xrv_test')
    np.save(os.path.join(SCRATCH, 'names_test.npy'),
            np.array([n for n, t in zip(names, te) if t]))
    np.save(os.path.join(SCRATCH, 'Y_test.npy'), Y[te])

    W = m.classifier.weight.detach().double()
    b = m.classifier.bias.detach().double()
    res['C'] = int(W.shape[1])
    res['T'] = int(Fte.shape[1])

    # map NIH finding -> xrv output index
    idx = {}
    for f in NIH_FINDINGS:
        if f in targets:
            idx[f] = targets.index(f)
    res['finding_to_xrv_index'] = idx

    Yte = Y[te]
    cond = {}
    for kind in ('gap', 'lse8.0', 'lse-8.0', 'lse2.0', 'lse-2.0'):
        h = mlib.head_of(kind, W.float(), b.float()).to(dev).eval()
        Z = logits(h, Fte)
        a = {}
        for f, j in idx.items():
            y = Yte[:, NIH_FINDINGS.index(f)]
            if 0 < y.sum() < len(y):
                from sklearn.metrics import roc_auc_score
                a[f] = float(roc_auc_score(y, Z[:, j]))
        cond[kind] = dict(auc=a, mean_auc=float(np.mean(list(a.values()))))
        print('  %-9s mean AUC %.4f' % (kind, cond[kind]['mean_auc']), flush=True)
    res['condA_xrv'] = cond
    mlib.save('m1_model.json', res)


if __name__ == '__main__':
    main()
