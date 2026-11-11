"""m2 -- certify that the CXR classifier sits at the degenerate point, and that
the edit moves it off.

Two independent instruments, exactly as the binding rules require:
  * the FROZEN stage-5/6 scope rule (`scripts/heldout/frozen_rule.py`), run on
    shipped weights and real images, never at random init;
  * `scripts/crosscheck/rtlib.py`'s LITERAL float64 per-cell loop.  ADDITIVE iff
    maxNS, Rhalf and I2 are all < 1e-9.  `corrNS` is recorded and never used as
    a criterion.
Both perturbation units (1x1 cells and 3x3 blocks) are reported.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib
from mlib import SCRATCH, dev
import frozen_rule
import rtlib

HEADS = ['gap', 'lse8.0', 'lse-8.0', 'lse2.0', 'lse-2.0']


def scope_rule_on_densenet121():
    """The frozen rule, on timm's shipped densenet121 and a real val image."""
    import timm
    from PIL import Image
    m = timm.create_model('densenet121', pretrained=True).eval()
    p = os.path.join(mlib.ROOT, 'data', 'in1k', 'val')
    cls = sorted(os.listdir(p))[0]
    f = os.path.join(p, cls, sorted(os.listdir(os.path.join(p, cls)))[0])
    im = Image.open(f).convert('RGB').resize((224, 224), Image.BILINEAR)
    a = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.
    for c in range(3):
        a[c] = (a[c] - mlib.IMNET_MEAN[c]) / mlib.IMNET_STD[c]
    x = torch.from_numpy(a)[None]
    with torch.no_grad():
        feat = m.forward_features(x)
    d = frozen_rule.structural(m, feat)
    d['post'] = [list(t) for t in d['post']]
    d['image'] = f
    d['feat_shape'] = list(feat.shape)
    return d


def block_units(G, K):
    return mlib.blocks_of(G, K)


def cert_pair(head, XA, XB, c, rs, G):
    out = {}
    d = mlib.certify(head, XA, XB, c, rs)
    out['unit1x1'] = dict(maxNS=d['maxNS'], Rhalf=d['Rhalf'], I2=d['I2'],
                          corrNS=d['corr_pearson'], additive=d['additive'],
                          sumN_over_dz=d['sumN_over_zA_minus_zB'])
    N, S = mlib.NS_unit(head, XA, XB, c, [np.array([t]) for t in range(XA.shape[0])])
    sc = max(abs(N).max(), abs(S).max()) + 1e-300
    out['unit1x1']['maxNS_recomputed'] = float(np.abs(N - S).max() / sc)
    out['unit1x1']['maxNS_abs'] = float(np.abs(N - S).max())
    Nb, Sb = mlib.NS_unit(head, XA, XB, c, block_units(G, 3))
    scb = max(abs(Nb).max(), abs(Sb).max()) + 1e-300
    out['unit3x3'] = dict(maxNS=float(np.abs(Nb - Sb).max() / scb),
                          maxNS_abs=float(np.abs(Nb - Sb).max()),
                          corrNS=float(np.corrcoef(Nb, Sb)[0, 1])
                          if Nb.std() > 0 and Sb.std() > 0 else np.nan)
    return out


def main():
    res = {}
    res['frozen_rule_densenet121'] = scope_rule_on_densenet121()
    print('frozen rule on timm densenet121:', res['frozen_rule_densenet121']['verdict'],
          res['frozen_rule_densenet121']['agg'], flush=True)

    import torchxrayvision as xrv
    m = xrv.models.DenseNet(weights='densenet121-res224-nih')
    W = m.classifier.weight.detach().double()
    b = m.classifier.bias.detach().double()

    F = np.load(os.path.join(SCRATCH, 'F_xrv_test.npy'), mmap_mode='r')
    names = np.load(os.path.join(SCRATCH, 'names_test.npy'))
    T, C = F.shape[1], F.shape[2]
    G = int(round(T ** .5))
    res['T'], res['C'], res['G'] = T, C, G

    # the readout under test must reproduce the model on its own feature map
    import m1_model as m1
    from PIL import Image                                        # noqa: F401
    body = m1.XrvBody(m).eval().to(dev)
    xs = torch.stack([m1.xrv_load(os.path.join(mlib.ROOT, 'data', 'nih', 'images', n))
                      for n in names[:8]]).to(dev)
    with torch.no_grad():
        f = body(xs)
        zref = m.to(dev).classifier(m.features2(xs)).double().cpu()
    Fh = f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2).double().cpu()
    h0 = mlib.head_of('gap', W, b).double()
    rep = float((h0(Fh) - zref).abs().max())
    res['head_reproduces_model_abs'] = rep
    res['head_reproduces_model_rel'] = float(rep / zref.abs().max())
    print('head reproduces model to %.3e abs' % rep, flush=True)

    rs = np.random.RandomState(0)
    pairs = [(0, 1000), (7, 2500), (33, 4000)]
    cls = 0                                    # xrv index 0 = Atelectasis
    res['probe_pairs'] = [[int(a), int(bx)] for a, bx in pairs]
    res['probe_class_xrv_index'] = cls
    res['probe_images'] = [[str(names[a]), str(names[bx])] for a, bx in pairs]

    cert = {}
    for kind in HEADS:
        h = mlib.head_of(kind, W, b).double()
        rows = []
        for (ia, ib) in pairs:
            XA = torch.from_numpy(np.ascontiguousarray(F[ia])).double()
            XB = torch.from_numpy(np.ascontiguousarray(F[ib])).double()
            rows.append(cert_pair(h, XA, XB, cls, np.random.RandomState(0), G))
        add = all(r['unit1x1']['additive'] for r in rows)
        cert[kind] = dict(pairs=rows, additive_all_pairs=bool(add),
                          worst_maxNS=max(r['unit1x1']['maxNS'] for r in rows),
                          worst_Rhalf=max(r['unit1x1']['Rhalf'] for r in rows),
                          worst_I2=max(r['unit1x1']['I2'] for r in rows))
        print('  %-9s additive=%s  worst maxNS %.3e Rhalf %.3e I2 %.3e' %
              (kind, add, cert[kind]['worst_maxNS'], cert[kind]['worst_Rhalf'],
               cert[kind]['worst_I2']), flush=True)
    res['certificate'] = cert

    # ---- the control: N == S elementwise at the degenerate point ----------
    h = mlib.head_of('gap', W, b).double()
    rsx = np.random.RandomState(1)
    ia = rsx.choice(len(F), 40, replace=False)
    ib = rsx.choice(len(F), 40, replace=False)
    worst_rel, worst_abs, worstb_rel = 0., 0., 0.
    for a, bx in zip(ia, ib):
        if a == bx:
            continue
        XA = torch.from_numpy(np.ascontiguousarray(F[a])).double()
        XB = torch.from_numpy(np.ascontiguousarray(F[bx])).double()
        N, S, _, _ = rtlib.NS_literal(h, XA, XB, cls)
        sc = max(abs(N).max(), abs(S).max()) + 1e-300
        worst_rel = max(worst_rel, float(np.abs(N - S).max() / sc))
        worst_abs = max(worst_abs, float(np.abs(N - S).max()))
        Nb, Sb = mlib.NS_unit(h, XA, XB, cls, block_units(G, 3))
        scb = max(abs(Nb).max(), abs(Sb).max()) + 1e-300
        worstb_rel = max(worstb_rel, float(np.abs(Nb - Sb).max() / scb))
    res['control_N_equals_S_at_GAP'] = dict(
        n_pairs=40, worst_rel_1x1=worst_rel, worst_abs_1x1=worst_abs,
        worst_rel_3x3=worstb_rel)
    print('control: worst |N-S|/scale = %.3e (1x1), %.3e (3x3)'
          % (worst_rel, worstb_rel), flush=True)

    mlib.save('m2_certify.json', res)


if __name__ == '__main__':
    main()
