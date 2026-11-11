"""cross-check ATTACK A1/A2 on a real shipped model, with my own adapter, my own
literal estimator, and float64 arithmetic in the head.

  * independent check of the "recomposed head reproduces the shipped logits"
    claim (bit-for-bit, not rel-err < 2e-4)
  * post / pre / none corrNS, computed with a LITERAL per-cell loop
  * Pearson vs Spearman vs Kendall
  * BASELINE SENSITIVITY: bank image, interpolated baselines A + s(B-A),
    zeros, dataset-mean cells, Gaussian, spatially-shuffled A
  * the affine-parameter confound: gamma=1, beta=0
  * the attribution statistic: sum N / (zA - zB)  vs  sum|N| / |zA|
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtlib as R

dev = 'cuda'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get('MODEL', 'vit_betwixt_patch16_reg4_gap_256.sbb_in1k')
NIMG = int(os.environ.get('NIMG', 32))
NB = int(os.environ.get('NB', 16))


# ---------------------------------------------------------------- my adapter
def build_site(name):
    import timm
    m = timm.create_model(name, pretrained=True).eval().to(dev)
    cfg = m.pretrained_cfg
    fam = type(m).__name__
    if fam in ('VisionTransformer', 'Eva', 'Beit', 'NaFlexVit'):
        npre = m.num_prefix_tokens
        ln, clf, shipped = m.fc_norm, m.head, 'post'
        assert isinstance(m.norm, nn.Identity)
        cells = lambda x: m.forward_features(x)[:, npre:]
    elif fam == 'SwinTransformer':
        ln, clf, shipped = m.norm, m.head.fc, 'pre'
        cells = lambda x: m.layers(m.patch_embed(x)).flatten(1, 2)
    elif fam == 'ConvNeXt':
        ln, clf, shipped = m.head.norm, m.head.fc, 'post'
        assert isinstance(m.head.pre_logits, nn.Identity)
        cells = lambda x: m.forward_features(x).flatten(2).transpose(1, 2)
    else:
        raise ValueError(fam)
    return m, cfg, fam, shipped, cells, ln, clf


def make_head(ln, clf, C, dtype=torch.float64, affine=True):
    w = ln.weight.detach().to(dtype) if affine else torch.ones(C, device=dev, dtype=dtype)
    b = (ln.bias.detach().to(dtype) if (affine and ln.bias is not None)
         else torch.zeros(C, device=dev, dtype=dtype))
    eps = ln.eps
    W = clf.weight.detach().to(dtype); B = clf.bias.detach().to(dtype)

    def LN(v): return Fn.layer_norm(v, (C,), w, b, eps)

    def h(X, mode):
        if mode == 'post':
            return LN(X.mean(-2)) @ W.T + B
        if mode == 'pre':
            return LN(X).mean(-2) @ W.T + B
        return X.mean(-2) @ W.T + B
    return h


def transform(cfg):
    from torchvision import transforms
    isz = cfg['input_size'][-1]
    ip = {'bicubic': transforms.InterpolationMode.BICUBIC,
          'bilinear': transforms.InterpolationMode.BILINEAR}[cfg.get('interpolation', 'bicubic')]
    return transforms.Compose([
        transforms.Resize(int(isz / cfg.get('crop_pct', 0.9)), interpolation=ip),
        transforms.CenterCrop(isz), transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std'])])


def main():
    from torchvision import datasets
    m, cfg, fam, shipped, cells, ln, clf = build_site(MODEL)
    C = ln.weight.numel()
    ds = datasets.ImageFolder(os.path.join(ROOT, 'data', 'in1k', 'val'), transform(cfg))
    rs = np.random.RandomState(0); perm = rs.permutation(len(ds))
    print(f'[{MODEL}] fam={fam} ships={shipped} C={C} T?')

    # ---- CHECK 1: is the recomposition really bit-for-bit?
    x0 = torch.stack([ds[i][0] for i in perm[:8]]).to(dev)
    h32 = make_head(ln, clf, C, torch.float32)
    with torch.no_grad():
        ref = m(x0); X0 = cells(x0); ours = h32(X0.float(), shipped)
    exact = bool(torch.equal(ref, ours))
    print(f'  recompose: bitwise-identical={exact}  max abs diff={float((ref-ours).abs().max()):.3e}'
          f'  rel={float((ref-ours).abs().max()/ref.abs().max()):.3e}  T={X0.shape[1]}')

    h = make_head(ln, clf, C, torch.float64)
    T = X0.shape[1]

    # ---- probe images + bank (disjoint), exactly the project's protocol
    sel = perm[:NIMG + NB]
    with torch.no_grad():
        Xs = torch.cat([cells(torch.stack([ds[i][0] for i in sel[q:q+8]]).to(dev))
                        for q in range(0, NIMG, 8)]).double()
        XBk = cells(torch.stack([ds[i][0] for i in sel[NIMG:NIMG+NB]]).to(dev)).double()
        pred = (h(Xs, shipped)).argmax(1).cpu().numpy()
        # dataset-mean cells over 256 images
        Xm = torch.cat([cells(torch.stack([ds[i][0] for i in perm[500+q:500+q+8]]).to(dev))
                        for q in range(0, 128, 8)]).double().mean(0)

    res = {'meta': dict(model=MODEL, family=fam, shipped=shipped, C=C, T=T,
                        nimg=NIMG, recompose_bitwise=exact)}

    # ---- CHECK 2/3: the three orderings, literal estimator, float64
    for mode in ('post', 'pre', 'none'):
        hh = lambda X, _m=mode: h(X, _m)
        acc = {}
        for q in range(NIMG):
            XA, XB = Xs[q], XBk[np.random.RandomState(1 + q).randint(NB)]
            d = R.report(mode, hh, XA, XB, int(pred[q]), np.random.RandomState(q))
            for k, v in d.items():
                if k != 'tag':
                    acc.setdefault(k, []).append(v)
        r = {k: float(np.nanmean(v)) for k, v in acc.items()}
        res[mode] = r
        print(f"  {mode:5s} r={r['corr_pearson']:+.8f} rho={r['corr_spearman']:+.6f} "
              f"tau={r['corr_kendall']:+.6f} maxdiff={r['max_abs_diff_rel']:.2e} "
              f"Rh={r['Rhalf']:.2e} I2={r['I2']:.2e} slope={r['slope']:+.4f} "
              f"sumN/(zA-zB)={r['sumN_over_zA_minus_zB']:+.4f} "
              f"sum|N|/|zA|={r['sumAbsN_over_absZA']:.4f}")

    # ---- CHECK 4: affine parameters of the LN are not the cause
    hna = make_head(ln, clf, C, torch.float64, affine=False)
    for mode in ('post', 'pre'):
        v = []
        for q in range(min(NIMG, 16)):
            XA, XB = Xs[q], XBk[np.random.RandomState(1 + q).randint(NB)]
            N, S, _, _ = R.NS_literal(lambda X, _m=mode: hna(X, _m), XA, XB, int(pred[q]))
            v.append(R.stats(N, S)['corr_pearson'])
        res[f'{mode}_noaffine_corr'] = float(np.mean(v))
        print(f'  {mode} with gamma=1,beta=0 (LN affine removed): r={np.mean(v):+.8f}')

    # ---- CHECK 5: BASELINE SENSITIVITY (post ordering)
    print('\n  baseline sensitivity, post ordering (16 images):')
    bl = {}
    NS_ = min(NIMG, 16)
    for s in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        v, vr = [], []
        for q in range(NS_):
            XA = Xs[q]; XB0 = XBk[np.random.RandomState(1 + q).randint(NB)]
            XB = XA + s * (XB0 - XA)
            N, S, _, _ = R.NS_literal(lambda X: h(X, 'post'), XA, XB, int(pred[q]))
            st = R.stats(N, S)
            v.append(st['corr_pearson'])
            vr.append(R.additivity(lambda X: h(X, 'post'), XA, XB, int(pred[q]), N,
                                   np.random.RandomState(q), nsub=16, npair=64)[0])
        bl[f'interp_s={s}'] = (float(np.mean(v)), float(np.mean(vr)))
        print(f'    B = A + {s:5.2f}(B0-A):  corrNS={np.mean(v):+.8f}  Rhalf={np.mean(vr):.2e}')
    for nm in ('zeros', 'datasetmean', 'gaussian', 'shuffleA', 'bank'):
        v, vr = [], []
        for q in range(NS_):
            XA = Xs[q]
            g = torch.Generator(device=dev).manual_seed(q)
            if nm == 'zeros':
                XB = torch.zeros_like(XA)
            elif nm == 'datasetmean':
                XB = Xm.clone()
            elif nm == 'gaussian':
                XB = (torch.randn(XA.shape, generator=g, device=dev, dtype=XA.dtype)
                      * XA.std() + XA.mean())
            elif nm == 'shuffleA':
                XB = XA[torch.randperm(T, generator=g, device=dev)]
            else:
                XB = XBk[np.random.RandomState(1 + q).randint(NB)]
            N, S, _, _ = R.NS_literal(lambda X: h(X, 'post'), XA, XB, int(pred[q]))
            v.append(R.stats(N, S)['corr_pearson'])
            vr.append(R.additivity(lambda X: h(X, 'post'), XA, XB, int(pred[q]), N,
                                   np.random.RandomState(q), nsub=16, npair=64)[0])
        bl[nm] = (float(np.nanmean(v)), float(np.nanmean(vr)))
        print(f'    B = {nm:12s}      corrNS={np.nanmean(v):+.8f}  Rhalf={np.nanmean(vr):.2e}')
    # and the same alternative baselines under `pre` -- must stay exactly 1
    for nm in ('zeros', 'datasetmean', 'gaussian'):
        v = []
        for q in range(NS_):
            XA = Xs[q]
            g = torch.Generator(device=dev).manual_seed(q)
            XB = (torch.zeros_like(XA) if nm == 'zeros' else Xm.clone() if nm == 'datasetmean'
                  else torch.randn(XA.shape, generator=g, device=dev, dtype=XA.dtype))
            N, S, _, _ = R.NS_literal(lambda X: h(X, 'pre'), XA, XB, int(pred[q]))
            v.append(R.stats(N, S)['corr_pearson'])
        bl['pre_' + nm] = (float(np.nanmean(v)), None)
        print(f'    PRE with B = {nm:12s}  corrNS={np.nanmean(v):+.10f}')
    res['baselines'] = bl
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f'rt2_{MODEL.split(".")[0]}.json'), 'w'), indent=1)


main()
