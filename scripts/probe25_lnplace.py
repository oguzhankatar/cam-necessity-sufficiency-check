"""PROBE 25 (P1) -- HARDENING THE LAYERNORM-PLACEMENT HEADLINE.

Stage 3 measured the headline on TWELVE images of ONE model.  This probe
re-measures it with n >= 200 images, bootstrap CIs over images, accuracy on the
FULL 50 000-image ILSVRC-2012 validation set, and a replication sweep over
models that ship *different* orderings of the same two operations.

THE OBJECT UNDER TEST.  Every model here ends in

        logits = clf( POOL_t ( cells ) )  with ONE LayerNorm adjacent to POOL

and the only thing we change is which side of POOL that LayerNorm is written:

    post :  clf( LN( mean_t x_t ) )      (timm ViT `fc_norm`, ConvNeXt head)
    pre  :  clf( mean_t LN( x_t ) )      (Swin `norm`, ships this way)
    none :  clf( mean_t x_t )            (no LayerNorm at all)

`none` doubles as the MANDATORY INTERNAL CONTROL: it is GAP + a linear map,
i.e. algebraically LSE at T = 0, so it must return corrNS = +1.000000 and
Rhalf = 0 to float precision.  If it does not, the estimator is broken and no
other row in the table means anything.

ESTIMATORS.  Exact finite differences over CELLS (patch tokens / spatial
positions) at the aggregation point, baselines always BANK-SAMPLED from other
images (never a per-image channel mean -- that has the same pool as the image
and makes every difference identically zero).

    N(t) = z(A) - z(A with cell t <- B)          necessity
    S(t) = z(B with cell t <- A) - z(B)          sufficiency
    corrNS = Pearson_t(N, S);  Rhalf = E|D(H) - sum_{t in H} N(t)| / E|D(H)|
    I2 = E|D(t1,t2) - N(t1) - N(t2)| / E|D(t1,t2)|

FAST PATH + BRUTE-FORCE AUDIT.  Because the intervention is at the aggregation
point, the pooled vector under any cell subset H is an exact rank-1 update

    post/none :  m_A + sum_{t in H} (x^B_t - x^A_t)/T
    pre       :  n_A + sum_{t in H} (LN(x^B_t) - LN(x^A_t))/T

so every perturbation is computed in one batched matmul.  This is an algebraic
identity, not an approximation, but --AUDIT re-does the first images with the
literal "build the whole perturbed cell matrix and re-run the head" loop and
asserts agreement; the audit must print PASS.

Usage:
  MODEL=vit_betwixt_patch16_reg4_gap_256.sbb_in1k NIMG=200 NEVAL=50000 \
      python scripts/probe25_lnplace.py
"""
import os, sys, time, json, numpy as np, torch, torch.nn as nn
import torch.nn.functional as Fn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L

dev = 'cuda'
MODEL = os.environ.get('MODEL', 'vit_betwixt_patch16_reg4_gap_256.sbb_in1k')
NIMG = int(os.environ.get('NIMG', 200))
NEVAL = int(os.environ.get('NEVAL', 50000))
NBANK = int(os.environ.get('NBANK', 16))
AUDIT = int(os.environ.get('AUDIT', 3))
MODES = ('post', 'pre', 'none')


# ------------------------------------------------------------------ adapters
class Site:
    """Uniform view of a model's aggregation point.

    cells(x) -> (B, T, C) the per-cell representation immediately BEFORE the
                pool-adjacent LayerNorm (prefix/register tokens dropped)
    lnp      -> (weight, bias, eps) of that LayerNorm, over the channel dim
    clf(v)   -> logits from a pooled (B, C) vector
    shipped  -> 'post' | 'pre' : the ordering the checkpoint actually ships
    """

    def __init__(s, name):
        import timm
        m = timm.create_model(name, pretrained=True).eval().to(dev)
        s.m, s.name = m, name
        s.cfg = m.pretrained_cfg
        fam = type(m).__name__
        s.family = fam
        if fam in ('VisionTransformer', 'Eva', 'Beit', 'NaFlexVit'):
            assert str(m.global_pool) == 'avg', f'{name}: pool={m.global_pool}'
            assert isinstance(m.norm, nn.Identity), f'{name}: norm is not Identity'
            ln, s.shipped = m.fc_norm, 'post'
            npre = m.num_prefix_tokens

            def cells(x):
                z = m.forward_features(x)          # norm == Identity here
                return z[:, npre:]
            s.clf = lambda v: m.head(v)
        elif fam == 'SwinTransformer':
            ln, s.shipped = m.norm, 'pre'          # <-- ships the PRE ordering

            def cells(x):
                z = m.patch_embed(x); z = m.layers(z)     # (B,H,W,C), pre-norm
                return z.reshape(z.shape[0], -1, z.shape[-1])
            s.clf = lambda v: m.head.fc(v)
        elif fam == 'ConvNeXt':
            h = m.head
            assert isinstance(h.pre_logits, nn.Identity)
            ln, s.shipped = h.norm, 'post'

            def cells(x):
                z = m.forward_features(x)          # (B,C,H,W)
                return z.flatten(2).transpose(1, 2)
            s.clf = lambda v: h.fc(v)
        elif fam == 'MaxxVit':
            h = m.head                             # NormMlpClassifierHead
            ln, s.shipped = h.norm, 'post'

            def cells(x):
                z = m.forward_features(x)
                return z.flatten(2).transpose(1, 2)
            s.clf = lambda v: h.fc(h.pre_logits(v))
        else:
            raise ValueError(f'no adapter for {fam} ({name})')
        s.cells = cells
        s.w = ln.weight.detach().clone()
        s.b = ln.bias.detach().clone() if ln.bias is not None else None
        s.eps = ln.eps
        s.C = s.w.numel()

    def ln(s, v):
        return Fn.layer_norm(v, (s.C,), s.w, s.b, s.eps)

    def head(s, X, mode):
        """X: (..., T, C) cells -> logits. The one line under test."""
        if mode == 'post':
            return s.clf(s.ln(X.mean(-2)))
        if mode == 'pre':
            return s.clf(s.ln(X).mean(-2))
        return s.clf(X.mean(-2))

    def pooled(s, X, mode):
        return s.ln(X).mean(-2) if mode == 'pre' else X.mean(-2)

    def frompooled(s, p, mode):
        return s.clf(s.ln(p)) if mode == 'post' else s.clf(p)


def transform(cfg):
    from torchvision import transforms
    isz = cfg['input_size'][-1]
    ip = {'bicubic': transforms.InterpolationMode.BICUBIC,
          'bilinear': transforms.InterpolationMode.BILINEAR,
          'nearest': transforms.InterpolationMode.NEAREST}[
              cfg.get('interpolation', 'bicubic')]
    return transforms.Compose([
        transforms.Resize(int(isz / cfg.get('crop_pct', 0.9)),
                          interpolation=ip),
        transforms.CenterCrop(isz), transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std'])])


# ------------------------------------------------------- exact cell metrics
@torch.no_grad()
def metrics(st, XA, XB, c, mode, rs, npair=512, nsub=64):
    """Exact N, S, corrNS, spNS, Rhalf, I2 at the aggregation point.

    Fast path: every cell subset gives an exact rank-1 update of the pooled
    vector, so all T single-cell swaps are one batched head call.
    """
    T = XA.shape[0]
    pA, pB = st.pooled(XA, mode), st.pooled(XB, mode)
    dA = (st.ln(XA) if mode == 'pre' else XA)       # per-cell contributions
    dB = (st.ln(XB) if mode == 'pre' else XB)
    d = (dB - dA) / T                               # (T,C) swap A->B increment
    zA = st.frompooled(pA[None], mode)[0, c]
    zB = st.frompooled(pB[None], mode)[0, c]
    N = zA - st.frompooled(pA[None] + d, mode)[:, c]
    S = st.frompooled(pB[None] - d, mode)[:, c] - zB
    n, s = N.cpu().numpy().astype(np.float64), S.cpu().numpy().astype(np.float64)
    cr = (float(np.corrcoef(n, s)[0, 1])
          if n.std() > 1e-14 and s.std() > 1e-14 else np.nan)
    sp = (float(spearmanr(n, s).statistic)
          if n.std() > 1e-14 and s.std() > 1e-14 else np.nan)
    # Rhalf : random half of the cells swapped
    H = torch.zeros(nsub, T, device=XA.device)
    for i in range(nsub):
        H[i, torch.from_numpy(rs.choice(T, T // 2, replace=False)).to(XA.device)] = 1
    D = zA - st.frompooled(pA[None] + H @ d, mode)[:, c]
    Rh = float(((D - H @ N).abs().mean() / (D.abs().mean() + 1e-12)).item())
    # I2 : pairs
    p = rs.randint(0, T, (npair, 2)); p = p[p[:, 0] != p[:, 1]]
    P2 = torch.zeros(len(p), T, device=XA.device)
    P2[torch.arange(len(p)), torch.from_numpy(p[:, 0]).to(XA.device)] = 1
    P2[torch.arange(len(p)), torch.from_numpy(p[:, 1]).to(XA.device)] = 1
    D2 = zA - st.frompooled(pA[None] + P2 @ d, mode)[:, c]
    I2 = float(((D2 - P2 @ N).abs().mean() / (D2.abs().mean() + 1e-12)).item())
    return dict(corrNS=cr, spNS=sp, Rhalf=Rh, I2=I2,
                Nmag=float(N.abs().sum() / (abs(float(zA)) + 1e-12)),
                Smag=float(S.abs().sum() / (abs(float(zA)) + 1e-12)))


@torch.no_grad()
def brute(st, XA, XB, c, mode, chunk=64):
    """Literal loop: rebuild each perturbed CELL MATRIX and re-run the head."""
    T = XA.shape[0]
    zA = st.head(XA[None], mode)[0, c]; zB = st.head(XB[None], mode)[0, c]
    Nl, Sl = [], []
    for s0 in range(0, T, chunk):
        k = min(chunk, T - s0)
        X = XA[None].expand(k, -1, -1).clone()
        Y = XB[None].expand(k, -1, -1).clone()
        ar = torch.arange(k, device=XA.device); ix = ar + s0
        X[ar, ix] = XB[ix]; Y[ar, ix] = XA[ix]
        Nl.append(zA - st.head(X, mode)[:, c]); Sl.append(st.head(Y, mode)[:, c] - zB)
    return torch.cat(Nl), torch.cat(Sl)


def main():
    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    t0 = time.time()
    st = Site(MODEL)
    ds = datasets.ImageFolder(os.path.join(L.ROOT, 'data', 'in1k', 'val'),
                              transform(st.cfg))
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(ds))
    print(f'[{MODEL}] family={st.family} shipped={st.shipped} C={st.C} '
          f'input={st.cfg["input_size"][-1]} val={len(ds)}', flush=True)

    # ---- sanity 0: the composed shipped head reproduces the model exactly
    x0 = torch.stack([ds[i][0] for i in perm[:8]]).to(dev)
    with torch.no_grad():
        ref = st.m(x0)
        ours = st.head(st.cells(x0), st.shipped)
    err = (ref - ours).abs().max().item() / (ref.abs().max().item() + 1e-12)
    print(f'  sanity: composed head vs model forward, rel max err = {err:.2e} '
          f'{"PASS" if err < 2e-4 else "FAIL"}', flush=True)
    assert err < 2e-4, 'adapter does not reproduce the shipped model'

    res = {'meta': dict(model=MODEL, family=st.family, shipped=st.shipped,
                        C=st.C, n_img=NIMG, n_bank=NBANK,
                        input=st.cfg['input_size'][-1], compose_err=err)}

    # ---- accuracy for all three orderings, one backbone pass over the val set
    nev = min(NEVAL, len(ds))
    ev = np.sort(perm[:nev]) if nev < len(ds) else np.arange(len(ds))
    ld = DataLoader(Subset(ds, ev.tolist()), batch_size=64, num_workers=8,
                    pin_memory=True)
    hit = {k: 0 for k in MODES}; ntot = 0
    with torch.no_grad():
        for x, y in ld:
            X = st.cells(x.to(dev, non_blocking=True))
            for k in MODES:
                hit[k] += (st.head(X, k).argmax(1).cpu() == y).sum().item()
            ntot += len(y)
            if ntot % 6400 == 0:
                print(f'   acc {ntot}/{nev}  ' + ' '.join(
                    f'{k}={hit[k]/ntot:.4f}' for k in MODES), flush=True)
    acc = {k: hit[k] / ntot for k in MODES}
    res['acc'] = acc; res['n_eval'] = ntot
    print(f'  top-1 on {ntot} val images: ' +
          '  '.join(f'{k}={v:.4f}' for k, v in acc.items())
          + f'   ({time.time()-t0:.0f}s)', flush=True)

    # ---- probe images and baseline bank (disjoint)
    sel = perm[:NIMG + NBANK]
    xs = torch.stack([ds[i][0] for i in sel[:NIMG]])
    bk = torch.stack([ds[i][0] for i in sel[NIMG:NIMG + NBANK]]).to(dev)
    with torch.no_grad():
        XB_all = st.cells(bk)
        Xs = torch.cat([st.cells(xs[i:i + 16].to(dev)) for i in
                        range(0, NIMG, 16)])
        pred = st.head(Xs, st.shipped).argmax(1).cpu().numpy()

    print(f"\n{'ordering':>34s}{'corrNS':>13s}{'[95% CI]':>22s}"
          f"{'Rhalf':>11s}{'spNS':>9s}{'I2':>9s}{'|N|/|z|':>10s}", flush=True)
    for mode in MODES:
        rs2 = np.random.RandomState(1)
        acc_m = {k: [] for k in ('corrNS', 'spNS', 'Rhalf', 'I2', 'Nmag', 'Smag')}
        for q in range(NIMG):
            XA = Xs[q]; XB = XB_all[rs2.randint(NBANK)]
            c = int(pred[q])
            mt = metrics(st, XA, XB, c, mode, rs2)
            for k in acc_m:
                acc_m[k].append(mt[k])
            if q < AUDIT:                              # brute-force audit
                Nb, Sb = brute(st, XA, XB, c, mode)
                pA = st.pooled(XA, mode); pB = st.pooled(XB, mode)
                dA = st.ln(XA) if mode == 'pre' else XA
                dB = st.ln(XB) if mode == 'pre' else XB
                d = (dB - dA) / XA.shape[0]
                zA = st.frompooled(pA[None], mode)[0, c]
                Nf = zA - st.frompooled(pA[None] + d, mode)[:, c]
                e = (Nb - Nf).abs().max().item() / (Nb.abs().max().item() + 1e-12)
                assert e < 1e-3, f'fast path != brute force ({e:.2e})'
                if q == 0:
                    print(f'  [audit {mode}] fast vs brute-force rel err '
                          f'{e:.2e} PASS', flush=True)
        r = {}
        for k, v in acc_m.items():
            v = [z for z in v if np.isfinite(z)]
            mu, lo, hi = L.boot_ci(v)
            r[k] = mu; r[k + '_ci'] = [lo, hi]; r[k + '_n'] = len(v)
        r['acc'] = acc[mode]
        res[mode] = r
        tag = mode + ('  <-- SHIPPED' if mode == st.shipped else '')
        print(f"{tag:>34s}{r['corrNS']:+13.6f}"
              f"  [{r['corrNS_ci'][0]:+.6f},{r['corrNS_ci'][1]:+.6f}]"
              f"{r['Rhalf']:11.2e}{r['spNS']:+9.4f}{r['I2']:9.2e}"
              f"{r['Nmag']:10.3f}", flush=True)

    ctl = res['none']
    ok = abs(ctl['corrNS'] - 1) < 1e-5 and ctl['Rhalf'] < 1e-3
    res['control_none_exact'] = bool(ok)
    res['pre_exact'] = bool(abs(res['pre']['corrNS'] - 1) < 1e-5
                            and res['pre']['Rhalf'] < 1e-3)
    res['post_broken'] = bool(res['post']['corrNS_ci'][1] < 0.999)
    print(f"\nINTERNAL CONTROL (mean+linear == LSE T=0): "
          f"{'PASS' if ok else 'FAIL'}  corrNS={ctl['corrNS']:.6f} "
          f"Rhalf={ctl['Rhalf']:.2e}")
    print(f"PATTERN: pre exact = {res['pre_exact']}, post broken = "
          f"{res['post_broken']}  -> "
          f"{'REPLICATES' if res['pre_exact'] and res['post_broken'] else '*** FAILS ***'}")
    print(f"acc post={acc['post']:.4f} pre={acc['pre']:.4f} none={acc['none']:.4f}"
          f"   (pre-none = {100*(acc['pre']-acc['none']):+.2f} pts, "
          f"shipped-pre = {100*(acc[st.shipped]-acc['pre']):+.2f} pts)")
    L.save(f"probe25_lnplace_{MODEL.split('.')[0]}.json", res)


if __name__ == '__main__':
    main()
