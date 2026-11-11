"""TASK 6 -- does the nec/suf correlation fall with reference distance?

The archived sweep (scripts/crosscheck/rt2_*.json, Supplementary Table S5) stores,
for each reference, the pair (corrNS, Rhalf) -- see rt2_real.py, where
`bl[...] = (mean corr, mean Rhalf)`. It records NO distance. This script
re-runs the same sweep (same models, images, references, `post` ordering,
float64 head, literal per-cell loop from rtlib) and records three distances
between reference B and probed input A, averaged over the 16 images:
    d_act  = ||B - A||_F / ||A||_F          over the (T, C) cell activations
    d_pool = ||mean_t B - mean_t A|| / ||mean_t A||
    d_ln   = ||LN(mean B) - LN(mean A)|| / ||LN(mean A)||   (the head's own input)
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L
sys.path.insert(0, os.path.join(L.SCRIPTS, 'crosscheck'))
import rtlib as R   # noqa: E402

dev = 'cuda'
MODELS = ['convnext_tiny.fb_in1k', 'swin_tiny_patch4_window7_224.ms_in1k',
          'vit_betwixt_patch16_reg4_gap_256.sbb_in1k']
NIMG, NB, NS_ = 32, 16, 16


def build_site(name):                       # verbatim from rt2_real.py
    import timm
    m = timm.create_model(name, pretrained=True).eval().to(dev)
    cfg = m.pretrained_cfg
    fam = type(m).__name__
    if fam in ('VisionTransformer', 'Eva', 'Beit', 'NaFlexVit'):
        npre = m.num_prefix_tokens
        ln, clf, shipped = m.fc_norm, m.head, 'post'
        cells = lambda x: m.forward_features(x)[:, npre:]
    elif fam == 'SwinTransformer':
        ln, clf, shipped = m.norm, m.head.fc, 'pre'
        cells = lambda x: m.layers(m.patch_embed(x)).flatten(1, 2)
    elif fam == 'ConvNeXt':
        ln, clf, shipped = m.head.norm, m.head.fc, 'post'
        cells = lambda x: m.forward_features(x).flatten(2).transpose(1, 2)
    return m, cfg, fam, shipped, cells, ln, clf


def make_head(ln, clf, C, dtype=torch.float64):
    w = ln.weight.detach().to(dtype); b = ln.bias.detach().to(dtype); eps = ln.eps
    W = clf.weight.detach().to(dtype); B = clf.bias.detach().to(dtype)
    LN = lambda v: Fn.layer_norm(v, (C,), w, b, eps)

    def h(X, mode):
        if mode == 'post':
            return LN(X.mean(-2)) @ W.T + B
        if mode == 'pre':
            return LN(X).mean(-2) @ W.T + B
        return X.mean(-2) @ W.T + B
    return h, LN


def transform(cfg):
    from torchvision import transforms
    isz = cfg['input_size'][-1]
    ip = {'bicubic': transforms.InterpolationMode.BICUBIC,
          'bilinear': transforms.InterpolationMode.BILINEAR}[cfg.get('interpolation', 'bicubic')]
    return transforms.Compose([
        transforms.Resize(int(isz / cfg.get('crop_pct', 0.9)), interpolation=ip),
        transforms.CenterCrop(isz), transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std'])])


def run(name):
    from torchvision import datasets
    m, cfg, fam, shipped, cells, ln, clf = build_site(name)
    C = ln.weight.numel()
    ds = datasets.ImageFolder(os.path.join(L.ROOT, 'data', 'in1k', 'val'), transform(cfg))
    perm = np.random.RandomState(0).permutation(len(ds))
    h, LN = make_head(ln, clf, C)
    sel = perm[:NIMG + NB]
    with torch.no_grad():
        Xs = torch.cat([cells(torch.stack([ds[i][0] for i in sel[q:q + 8]]).to(dev))
                        for q in range(0, NIMG, 8)]).double()
        XBk = cells(torch.stack([ds[i][0] for i in sel[NIMG:NIMG + NB]]).to(dev)).double()
        pred = h(Xs, shipped).argmax(1).cpu().numpy()
        Xm = torch.cat([cells(torch.stack([ds[i][0] for i in perm[500 + q:500 + q + 8]]).to(dev))
                        for q in range(0, 128, 8)]).double().mean(0)
    T = Xs.shape[1]

    def refB(nm, q):
        XA = Xs[q]; XB0 = XBk[np.random.RandomState(1 + q).randint(NB)]
        if nm.startswith('interp_s='):
            s = float(nm.split('=')[1]); return XA + s * (XB0 - XA)
        g = torch.Generator(device=dev).manual_seed(q)
        if nm == 'zeros':
            return torch.zeros_like(XA)
        if nm == 'datasetmean':
            return Xm.clone()
        if nm == 'gaussian':
            return torch.randn(XA.shape, generator=g, device=dev, dtype=XA.dtype) * XA.std() + XA.mean()
        if nm == 'shuffleA':
            return XA[torch.randperm(T, generator=g, device=dev)]
        return XB0

    names = [f'interp_s={s}' for s in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)] + \
        ['zeros', 'datasetmean', 'gaussian', 'shuffleA', 'bank']
    out = {}
    with torch.no_grad():
        for nm in names:
            cs, da, dp, dl = [], [], [], []
            for q in range(NS_):
                XA = Xs[q]; XB = refB(nm, q)
                N, S, _, _ = R.NS_literal(lambda X: h(X, 'post'), XA, XB, int(pred[q]))
                cs.append(R.stats(N, S)['corr_pearson'])
                da.append(float((XB - XA).norm() / XA.norm()))
                mA, mB = XA.mean(0), XB.mean(0)
                dp.append(float((mB - mA).norm() / mA.norm()))
                dl.append(float((LN(mB) - LN(mA)).norm() / LN(mA).norm()))
            out[nm] = dict(corr=float(np.nanmean(cs)), d_act=float(np.mean(da)),
                           d_pool=float(np.mean(dp)), d_ln=float(np.mean(dl)))
            print(f"  {name.split('.')[0]:28s} {nm:16s} corr={out[nm]['corr']:.4f} d_act={out[nm]['d_act']:.3f} "
                  f"d_pool={out[nm]['d_pool']:.3f} d_ln={out[nm]['d_ln']:.3f}", flush=True)
    del m; torch.cuda.empty_cache()
    return dict(shipped=shipped, refs=out)


def violations(refs, dkey):
    """Pairs in which the nearer reference has the LOWER correlation."""
    ks = [k for k in refs if k != 'bank']          # bank == interp_s=1.0
    v = []
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            ra, rb = refs[a], refs[b]
            near, far = (a, b) if ra[dkey] < rb[dkey] else (b, a)
            if refs[near]['corr'] < refs[far]['corr']:
                v.append(dict(nearer=near, farther=far, d_near=refs[near][dkey], d_far=refs[far][dkey],
                              corr_near=refs[near]['corr'], corr_far=refs[far]['corr'],
                              gap=refs[far]['corr'] - refs[near]['corr'],
                              on_series=a.startswith('interp') and b.startswith('interp')))
    d = np.array([refs[k][dkey] for k in ks]); c = np.array([refs[k]['corr'] for k in ks])
    return dict(spearman_dist_corr=float(spearmanr(d, c).statistic), n_refs=len(ks),
                n_pairs=len(ks) * (len(ks) - 1) // 2, n_violations=len(v), violations=v)


def main():
    arch = {}
    for f in MODELS:
        j = json.load(open(os.path.join(L.SCRIPTS, 'crosscheck', f'rt2_{f.split(".")[0]}.json')))
        arch[f] = j['baselines']
    res = {'note': 'archived rt2 entries are (corr, Rhalf), not (corr, distance)', 'models': {}}
    for name in MODELS:
        r = run(name)
        r['archived_corr_match'] = {k: dict(new=v['corr'], archived=arch[name][k][0],
                                            archived_second_entry_is_Rhalf=arch[name][k][1])
                                    for k, v in r['refs'].items()}
        r['monotonicity'] = {dk: violations(r['refs'], dk) for dk in ('d_act', 'd_pool', 'd_ln')}
        res['models'][name] = r
        for dk, mv in r['monotonicity'].items():
            print(f"  {name.split('.')[0]} {dk}: spearman(dist,corr)={mv['spearman_dist_corr']:+.3f} "
                  f"violations {mv['n_violations']}/{mv['n_pairs']}", flush=True)
        L.save('t6_refdist.json', res)


if __name__ == '__main__':
    main()
