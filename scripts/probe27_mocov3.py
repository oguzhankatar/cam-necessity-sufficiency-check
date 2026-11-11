"""PROBE 27 (P2) -- RECONCILING OUR ~1 POINT WITH MoCo v3's 6.6 POINTS.

THE NUMBER WE MUST EXPLAIN.  Chen, Xie & He, "An Empirical Study of Training
Self-Supervised Vision Transformers", ICCV 2021 (arXiv:2104.02057), Sec. 6.2,
"Class token", ViT-B/16, 300-ep MoCo v3, ImageNet LINEAR PROBING:

      w/ [CLS]   76.5     w/o [CLS]; LN+pool   69.7     w/o [CLS]; pool   76.3

and their own words: "ViT has an extra LayerNorm (LN) after the final block, and
if we keep this LN and remove [CLS], the result is much worse (69.7%). But if we
remove this LN and [CLS], the result is nearly unchanged (76.3%)."

Their `LN+pool` is LN applied to the TOKENS and then averaged = our `pre`.
Their `pool` is our `none`.  They never ran `post` (LN after the mean) at all --
which is the ordering every timm ViT ships.  So the axis is the same and the
comparable delta is  pre - none : theirs -6.6 points, ours ~-1 point.

FOUR CANDIDATE EXPLANATIONS, and what distinguishes them:
  (a) from-scratch-with-that-head vs relocate-in-a-trained-model  -> probe27b
  (b) linear-probe-on-frozen-features vs supervised end-to-end    -> EXP 2 here
  (c) their LN is trained inside that architecture, ours carries
      weights trained for a different placement                   -> EXP 1 here
  (d) head refit vs head weight reuse                             -> EXP 1 here

EXP 1 -- REFIT vs REUSE, and why (c) is ANALYTICALLY DEAD.
Write LN(v) = gamma * vhat + beta with vhat = (v - mean_c v)/std_c v parameter
free.  Then
      pre  : mean_t LN(x_t) = gamma * (mean_t xhat_t) + beta
      post : LN(mean_t x_t)  = gamma * mhat + beta
so in BOTH orderings the LayerNorm's affine parameters act as a fixed affine map
applied immediately before a LINEAR classifier.  A refit linear head absorbs
them exactly.  Therefore, under any linear-probe protocol, the LN's trained
gamma/beta CANNOT change the accuracy at all -- fresh-init LN and trained LN are
the same model class.  Explanation (c) is dead for linear probing, and the only
thing that survives is WHICH POOLED SUMMARY the head sees:

      none : m        = mean_t x_t
      post : mhat     = normalise(mean_t x_t)
      pre  : mbar     = mean_t normalise(x_t)

We verify this numerically (fresh vs frozen LN must agree to the optimiser's
noise) and then measure the three summaries under a real linear probe.

EXP 2 -- REGIME.  The same three summaries, same protocol, on backbones from
four regimes: supervised+GAP-trained (timm sbb), supervised+CLS-trained
(augreg), self-supervised contrastive/distillation (DINO) and masked
auto-encoding (MAE).  MoCo v3's regime is "self-supervised, CLS-trained,
linear probe"; if the big gap is a regime effect it must appear there.

EXP 3 -- MECHANISM.  `pre` replaces each token by a unit-RMS version of itself,
so it DELETES the token-norm weighting that `none` pooling applies implicitly.
We therefore measure the coefficient of variation of the token norms and test
whether it predicts the pre - none gap across backbones.

DATA.  Real ILSVRC-2012 validation images, split by within-class index:
i < 40 -> probe-training (40 000 images), i >= 40 -> evaluation (10 000).
The two are disjoint by construction.  This is 3% of the ImageNet train set, so
ABSOLUTE accuracies are far below MoCo v3's; only the BETWEEN-ORDERING deltas
are being compared, and the protocol is identical across rows.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L

dev = 'cuda'
NTR = int(os.environ.get('IN1K_NTR', 40))
EPOCHS = int(os.environ.get('EPOCHS', 60))
CACHE = L.CACHE

MODELS = os.environ.get('MODELS', ','.join([
    'vit_betwixt_patch16_reg4_gap_256.sbb_in1k',      # supervised, GAP-trained
    'vit_medium_patch16_reg4_gap_256.sbb_in1k',       # supervised, GAP-trained
    'vit_base_patch16_224.augreg2_in21k_ft_in1k',     # supervised, CLS-trained
    'deit3_small_patch16_224.fb_in1k',                # supervised, CLS-trained
    'vit_base_patch16_224.dino',                      # SSL, CLS-trained
    'vit_base_patch16_224.mae',                       # SSL (MIM), CLS-trained
])).split(',')


# -------------------------------------------------------------- token access
@torch.no_grad()
def prenorm_tokens(m, x):
    """Patch tokens at the last block, BEFORE the model's final LayerNorm."""
    z = m.patch_embed(x); z = m._pos_embed(z); z = m.norm_pre(z)
    for b in m.blocks:
        z = b(z)
    return z[:, m.num_prefix_tokens:]


def pool_adjacent_ln(m):
    if not isinstance(m.norm, nn.Identity):
        return m.norm
    return m.fc_norm


def summaries(t, eps):
    """t: (B,T,C) raw tokens -> the three pooled summaries, parameter free."""
    m_ = t.mean(1)
    mhat = (m_ - m_.mean(-1, keepdim=True)) / (m_.var(-1, keepdim=True,
                                                      unbiased=False) + eps).sqrt()
    th = (t - t.mean(-1, keepdim=True)) / (t.var(-1, keepdim=True,
                                                 unbiased=False) + eps).sqrt()
    mbar = th.mean(1)
    tn = t.norm(dim=-1)                                     # (B,T) token norms
    cv = (tn.std(1) / (tn.mean(1) + 1e-12))
    return m_, mhat, mbar, cv


def cache(name):
    tag = os.path.join(CACHE, f'p27_{name.replace("/", "_")}.npz')
    if os.path.exists(tag):
        d = np.load(tag)
        return {k: d[k] for k in d.files}
    import timm
    from torchvision import datasets
    from torch.utils.data import DataLoader
    m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(dev)
    cfg = m.pretrained_cfg
    ds = datasets.ImageFolder(os.path.join(L.ROOT, 'data', 'in1k', 'val'),
                              _tf(cfg))
    idx = np.array([int(os.path.basename(f)[:5]) for f, _ in ds.samples])
    ld = DataLoader(ds, batch_size=64, num_workers=8, pin_memory=True)
    ln = pool_adjacent_ln(m)
    eps = ln.eps if hasattr(ln, 'eps') else 1e-6
    O = {k: [] for k in ('m', 'mhat', 'mbar', 'cv', 'y')}
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(ld):
            t = prenorm_tokens(m, x.to(dev, non_blocking=True)).float()
            a, b, c, cv = summaries(t, eps)
            for k, v in (('m', a), ('mhat', b), ('mbar', c), ('cv', cv)):
                O[k].append(v.cpu().numpy().astype(np.float16 if k != 'cv'
                                                   else np.float32))
            O['y'].append(y.numpy())
            if i % 100 == 0:
                print(f'   {name} {i*64}/{len(ds)} ({time.time()-t0:.0f}s)',
                      flush=True)
    O = {k: np.concatenate(v) for k, v in O.items()}
    O['idx'] = idx
    O['lnw'] = ln.weight.detach().cpu().numpy() if hasattr(ln, 'weight') \
        else np.ones(O['m'].shape[1], np.float32)
    O['lnb'] = ln.bias.detach().cpu().numpy() if getattr(ln, 'bias', None) \
        is not None else np.zeros(O['m'].shape[1], np.float32)
    np.savez(tag, **O)
    del m; torch.cuda.empty_cache()
    return O


def _tf(cfg):
    from torchvision import transforms
    isz = cfg['input_size'][-1]
    ip = {'bicubic': transforms.InterpolationMode.BICUBIC,
          'bilinear': transforms.InterpolationMode.BILINEAR}[
              cfg.get('interpolation', 'bicubic')]
    return transforms.Compose([
        transforms.Resize(int(isz / cfg.get('crop_pct', 0.9)), interpolation=ip),
        transforms.CenterCrop(isz), transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std'])])


# ------------------------------------------------------------- linear probe
def probe(Xtr, Ytr, Xva, Yva, std=True, epochs=None, seed=0, wd=1e-4):
    epochs = epochs or EPOCHS
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(Xtr).float().to(dev); Xva = torch.from_numpy(Xva).float().to(dev)
    if std:
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    yt = torch.from_numpy(Ytr).long().to(dev); yv = torch.from_numpy(Yva).long().to(dev)
    K = int(max(Ytr.max(), Yva.max())) + 1
    h = nn.Linear(Xtr.shape[1], K).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=3e-3, weight_decay=wd)
    n, bs = len(yt), 1024
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (n // bs + 1))
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(epochs):
        p = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            j = p[i:i + bs]
            loss = Fn.cross_entropy(h(Xtr[j]), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        acc = (h(Xva).argmax(1) == yv).float().mean().item()
    return acc


def main():
    res = {'meta': dict(ntr=NTR, epochs=EPOCHS,
                        mocov3=dict(cls=76.5, ln_pool=69.7, pool=76.3,
                                    delta_pre_minus_none=-6.6))}
    print(f"{'backbone':>44s}{'regime':>16s}{'none':>8s}{'post':>8s}"
          f"{'pre':>8s} | {'pre-none':>9s}{'post-none':>10s}{'tokenCV':>9s}",
          flush=True)
    for name in MODELS:
        O = cache(name)
        tr = O['idx'] < NTR; va = ~tr
        row = {}
        for k in ('m', 'mhat', 'mbar'):
            row[k] = probe(O[k][tr], O['y'][tr], O[k][va], O['y'][va])
        # (c) is analytically dead: applying the model's TRAINED LN affine to
        # the summary is a fixed affine before a refit linear head.  Verify.
        w, b = O['lnw'], O['lnb']
        Xa = O['mhat'].astype(np.float32) * w + b
        row['mhat_trainedaffine'] = probe(Xa[tr], O['y'][tr], Xa[va], O['y'][va])
        Xb = O['mbar'].astype(np.float32) * w + b
        row['mbar_trainedaffine'] = probe(Xb[tr], O['y'][tr], Xb[va], O['y'][va])
        row['tokenCV'] = float(O['cv'].mean())
        reg = ('SSL-CLS' if name.endswith(('.dino', '.mae')) else
               'sup-GAP' if 'gap' in name else 'sup-CLS')
        row['regime'] = reg
        res[name] = row
        print(f"{name:>44s}{reg:>16s}{row['m']:8.4f}{row['mhat']:8.4f}"
              f"{row['mbar']:8.4f} | {100*(row['mbar']-row['m']):+9.2f}"
              f"{100*(row['mhat']-row['m']):+10.2f}{row['tokenCV']:9.3f}",
              flush=True)
        print(f"{'':>44s}{'  [LN affine check]':>16s} post fresh/trained = "
              f"{row['mhat']:.4f}/{row['mhat_trainedaffine']:.4f}   pre "
              f"fresh/trained = {row['mbar']:.4f}/{row['mbar_trainedaffine']:.4f}",
              flush=True)
        L.save('probe27_mocov3.json', res)
    # ---- does token-norm dispersion predict the pre-none gap?
    a = np.array([[v['tokenCV'], 100 * (v['mbar'] - v['m'])]
                  for k, v in res.items() if isinstance(v, dict)
                  and 'tokenCV' in v])
    if len(a) > 2:
        res['cv_vs_gap_r'] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        print(f"\ntoken-norm CV vs (pre - none) gap over {len(a)} backbones: "
              f"r = {res['cv_vs_gap_r']:+.4f}")
    L.save('probe27_mocov3.json', res)


if __name__ == '__main__':
    main()
