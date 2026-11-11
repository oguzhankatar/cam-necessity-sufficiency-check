"""m3 -- generate every map for every box-annotated NIH image, at both
perturbation units.  Scoring and statistics are m4; this script only produces
the 7x7 maps, so that nothing about the ground truth can influence how a map
was built.

Maps
  collapsed      N(=S) of the degenerate GAP readout
  N_<h>, S_<h>   necessity / sufficiency of the off-degenerate readout h
  gradcam_<h>    Grad-CAM at the aggregation site of readout h
  actnorm        ||a_t||_2            (class-independent control)
  center         -(distance to centre)^2  (image-independent control)
  random         uniform noise

N and S share the baseline (the paired counterfactual convention), and the
baseline is a bank of reference images -- never a summary of the image itself.
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib
from mlib import SCRATCH, dev, NIH_FINDINGS
import rtlib

OFF = ['lse8.0', 'lse-8.0']
NBASE = 8


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
    import torchxrayvision as xrv
    m = xrv.models.DenseNet(weights='densenet121-res224-nih')
    targets = list(m.targets)
    W = m.classifier.weight.detach().double()
    b = m.classifier.bias.detach().double()

    F = np.load(os.path.join(SCRATCH, 'F_xrv_test.npy'), mmap_mode='r')
    names = list(np.load(os.path.join(SCRATCH, 'names_test.npy')))
    Y = np.load(os.path.join(SCRATCH, 'Y_test.npy'))
    idx = {n: i for i, n in enumerate(names)}
    T, C = F.shape[1], F.shape[2]
    G = int(round(T ** .5))

    boxes = mlib.nih_boxes()
    items = []
    for n, bs in sorted(boxes.items()):
        if n not in idx:
            continue
        for f in sorted(set(x[0] for x in bs)):
            if f in targets:
                items.append((n, f, targets.index(f)))
    print('box items available: %d (images %d)' %
          (len(items), len(set(i[0] for i in items))), flush=True)

    # baseline bank: 8 "No Finding" test images that carry no box
    nofind = [i for i, n in enumerate(names)
              if Y[i].sum() == 0 and n not in boxes]
    mode = os.environ.get('BANKMODE', 'nofinding')
    if mode == 'sameclass':
        # baselines drawn from images that CARRY the finding -- the reader's
        # baseline, and the one stage 5 found moves `post` corrNS the most
        nofind = [i for i, n in enumerate(names) if Y[i].sum() > 0 and n not in boxes]
    rs = np.random.RandomState(int(os.environ.get('BANKSEED', 0)))
    bank = sorted(rs.choice(nofind, NBASE, replace=False).tolist())
    print('baseline bank (No Finding, no box):', [names[i] for i in bank], flush=True)

    heads = {'gap': mlib.head_of('gap', W, b).double().to(dev)}
    for k in OFF:
        heads[k] = mlib.head_of(k, W, b).double().to(dev)

    units1 = unit_mask([np.array([t]) for t in range(T)], T, dev)
    units3 = unit_mask(mlib.blocks_of(G, 3), T, dev)

    # ---- verification that the batched loop is the literal loop ----------
    XA = torch.from_numpy(np.ascontiguousarray(F[idx[items[0][0]]])).double().to(dev)
    XB = torch.from_numpy(np.ascontiguousarray(F[bank[0]])).double().to(dev)
    ver = {k: verify(heads[k], XA, XB, items[0][2], T) for k in heads}
    print('batched-vs-literal max abs diff:', ver, flush=True)

    Xbank = [torch.from_numpy(np.ascontiguousarray(F[i])).double().to(dev) for i in bank]

    out = []
    rmap = np.random.RandomState(7)
    t0 = time.time()
    for k, (n, f, c) in enumerate(items):
        XA = torch.from_numpy(np.ascontiguousarray(F[idx[n]])).double().to(dev)
        rec = dict(image=n, finding=f, xrv_index=int(c),
                   label_positive=int(Y[idx[n], NIH_FINDINGS.index(f)])
                   if f in NIH_FINDINGS else -1,
                   maps={})
        for hk, h in heads.items():
            for uname, units in (('1x1', units1), ('3x3', units3)):
                Ns, Ss = [], []
                for XB in Xbank:
                    N, S = NS_batched(h, XA, XB, c, units)
                    Ns.append(N)
                    Ss.append(S)
                N = np.mean(Ns, 0)
                S = np.mean(Ss, 0)
                if hk == 'gap':
                    rec['maps']['collapsed_%s' % uname] = N.tolist()
                    rec['maps']['collapsedS_%s' % uname] = S.tolist()
                else:
                    rec['maps']['N_%s_%s' % (hk, uname)] = N.tolist()
                    rec['maps']['S_%s_%s' % (hk, uname)] = S.tolist()
            rec['maps']['gradcam_%s' % hk] = mlib.gradcam(h, XA, c).tolist()
        rec['maps']['actnorm'] = mlib.actnorm(XA).tolist()
        rec['maps']['center'] = mlib.center_prior(G).tolist()
        rec['maps']['random'] = rmap.rand(T).tolist()
        # the model's own score for this finding, for the subgroup analyses
        with torch.no_grad():
            rec['logit_gap'] = float(heads['gap'](XA[None])[0, c])
        out.append(rec)
        if k % 50 == 0:
            print('  %d/%d  %.0fs' % (k, len(items), time.time() - t0), flush=True)

    res = dict(n_items=len(out), T=T, C=C, G=G, nbase=NBASE,
               bank=[names[i] for i in bank], targets=targets,
               batched_vs_literal=ver, items=out)
    res['bank_mode'] = os.environ.get('BANKMODE', 'nofinding')
    res['bank_seed'] = int(os.environ.get('BANKSEED', 0))
    mlib.save(os.environ.get('OUTNAME', 'm3_maps.json'), res)


if __name__ == '__main__':
    main()
