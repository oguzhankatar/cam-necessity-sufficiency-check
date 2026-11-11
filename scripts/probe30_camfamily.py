"""PROBE 30 -- IS THE INVARIANCE A PROPERTY OF GRAD-CAM, OR OF THE WHOLE FAMILY?

The article reports that a gradient-weighted map is invariant to the pooling
operator: at fixed weights it is the same map at every temperature, so it cannot
be an estimate of necessity or of sufficiency, because it does not respond to
the operator that decides whether those two are different quantities at all.

That is stated for Grad-CAM. The obvious question -- and the first one a reader
will ask -- is whether it holds for the rest of the family. It does not, and the
split is informative.

Write the readout at temperature T as
    s_t = w_c . a_t + b_c,        z = (1/T) log mean_t exp(T s_t),
so the per-cell gradient is
    dz/da_ct = w_c * p_t,         p = softmax(T s),  sum_t p_t = 1.

  * Grad-CAM averages the gradient over cells first: mean_t dz/da_ct
    = w_c * mean_t p_t = w_c / N, EXACTLY, at every T. The temperature is gone
    before the map is formed. Invariance is algebraic, not approximate.

  * Every method that multiplies the gradient by the activation elementwise --
    HiResCAM, LayerCAM, XGrad-CAM, Grad-CAM++ -- keeps the factor p_t, which
    is where the whole temperature dependence lives. Those maps do move.

  * Score-CAM and Ablation-CAM read the head's output rather than its gradient,
    so they see the operator directly.

  * Eigen-CAM reads only the activations and never the head, so it is invariant
    for a different and much weaker reason: it does not know the class either.

This probe measures all of that on the distributed ResNet-50 classifier with its
own fc weights dropped into each member of the pooling family, no parameter
trained, so the only thing that differs between conditions is the operator.

Mandatory internal control: T = 0 is algebraically GAP+FC, so corr(N, S) there
must be 1.000000 and every map's agreement with N and with S must be equal.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L
from heads import LSE

BODY, DS = 'resnet50', 'in1k'
NIMG = int(os.environ.get('NIMG', 128))
TS = [float(t) for t in os.environ.get('TS', '-8,-4,-2,-1,0,1,2,4,8').split(',')]
CHUNK = int(os.environ.get('CHUNK', 128))
dev = 'cuda'
EPS = 1e-12


# ----------------------------------------------------------------- the maps
def _grad(h, A, c):
    """dz/da for the class logit, shape (C, T)."""
    X = A[None].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        g, = torch.autograd.grad(h(X)[0, c], X)
    return g[0].detach()


def cam_gradcam(h, A, c, **kw):
    g = _grad(h, A, c)
    return Fn.relu((g.mean(1, keepdim=True) * A).sum(0))


def cam_hirescam(h, A, c, **kw):
    return Fn.relu((_grad(h, A, c) * A).sum(0))


def cam_layercam(h, A, c, **kw):
    return Fn.relu((Fn.relu(_grad(h, A, c)) * A).sum(0))


def cam_xgradcam(h, A, c, **kw):
    g = _grad(h, A, c)
    w = (g * A / (A.sum(1, keepdim=True) + EPS)).sum(1, keepdim=True)
    return Fn.relu((w * A).sum(0))


def cam_gradcampp(h, A, c, **kw):
    g = _grad(h, A, c)
    g2, g3 = g * g, g * g * g
    denom = 2 * g2 + (A.sum(1, keepdim=True) * g3)
    alpha = torch.where(denom.abs() > EPS, g2 / denom, torch.zeros_like(g2))
    w = (alpha * Fn.relu(g)).sum(1, keepdim=True)
    return Fn.relu((w * A).sum(0))


@torch.no_grad()
def cam_scorecam(h, A, c, **kw):
    """Channel-wise increase of confidence: mask the activation tensor with each
    channel's own normalised spatial map and read the softmax score."""
    C, T = A.shape
    m = A - A.min(1, keepdim=True).values
    m = m / (m.max(1, keepdim=True).values + EPS)           # (C, T) in [0, 1]
    sc = torch.empty(C, device=A.device)
    for i in range(0, C, CHUNK):
        M = m[i:i + CHUNK]                                   # (B, T)
        X = A[None] * M[:, None, :]                          # (B, C, T)
        sc[i:i + CHUNK] = torch.softmax(h(X), 1)[:, c]
    w = torch.softmax(sc, 0)
    return Fn.relu((w[:, None] * A).sum(0))


@torch.no_grad()
def cam_ablationcam(h, A, c, **kw):
    """Weight each channel by the relative drop in the class logit when that
    channel is zeroed."""
    C, T = A.shape
    z0 = h(A[None])[0, c]
    w = torch.empty(C, device=A.device)
    for i in range(0, C, CHUNK):
        n = min(CHUNK, C - i)
        X = A[None].expand(n, C, T).clone()
        X[torch.arange(n), torch.arange(i, i + n)] = 0.0
        w[i:i + n] = (z0 - h(X)[:, c]) / (z0.abs() + EPS)
    return Fn.relu((w[:, None] * A).sum(0))


@torch.no_grad()
def cam_eigencam(h, A, c, **kw):
    """First principal component of the activations. Uses no class information
    and never touches the head, so it is included as the trivial control."""
    X = A - A.mean(1, keepdim=True)
    U, S, V = torch.linalg.svd(X.float(), full_matrices=False)
    v = V[0]
    if v.sum() < 0:
        v = -v
    return Fn.relu(v)


CAMS = [
    ('grad-cam', cam_gradcam, 'gradient'),
    ('grad-cam++', cam_gradcampp, 'gradient'),
    ('hirescam', cam_hirescam, 'gradient'),
    ('layercam', cam_layercam, 'gradient'),
    ('xgrad-cam', cam_xgradcam, 'gradient'),
    ('score-cam', cam_scorecam, 'perturbation'),
    ('ablation-cam', cam_ablationcam, 'perturbation'),
    ('eigen-cam', cam_eigencam, 'activation only'),
]


def norm(m):
    """A CAM is displayed after normalisation, so the comparison is made on the
    normalised map -- that is what a reader actually sees."""
    m = m - m.min()
    return m / (m.max() + EPS)


# ----------------------------------------------------------------------- run
def main():
    fva, yva, C, T = L.cache_feats(BODY, DS, 'val', size=224, bs=64, workers=8)
    Fva, Yva = L.load_cached(fva, yva)
    from torchvision import models
    tv = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    W, b = tv.fc.weight.data.clone(), tv.fc.bias.data.clone()
    print(f'[camfamily] C={C} cells={T} val={len(Yva)} images={NIMG} '
          f'temperatures={TS}', flush=True)

    rs = np.random.RandomState(0)
    idx = np.sort(rs.choice(len(Yva), NIMG, replace=False))
    bank_idx = np.sort(rs.choice(len(Yva), 16, replace=False))
    Xi = L.batch(Fva, idx)
    bank = L.batch(Fva, bank_idx)

    heads = {}
    for t in TS:
        h = LSE(k=1000, T=t, c=C).to(dev)
        h.fc.weight.data.copy_(W.to(dev)); h.fc.bias.data.copy_(b.to(dev))
        heads[t] = h.eval()

    # the class each model predicts is fixed at T = 0 so that every temperature
    # explains the same class, which is what makes the maps comparable
    with torch.no_grad():
        cls = heads[0.0](Xi).argmax(1).cpu().numpy()

    rs2 = np.random.RandomState(1)
    acc = {(nm, t): {'devN': [], 'devS': [], 'd0': [], 'sp08': []}
           for nm, _, _ in CAMS for t in TS}
    ns = {t: [] for t in TS}
    t0 = time.time()

    for k in range(NIMG):
        A = Xi[k]
        c = int(cls[k])
        B = bank[rs2.randint(bank.shape[0])]
        m0 = {}
        maps_by_T = {}
        for t in TS:
            h = heads[t]
            N, S = L.maps(h, A, B, c)
            n_, s_ = N.cpu().numpy(), S.cpu().numpy()
            if n_.std() > 1e-12 and s_.std() > 1e-12:
                ns[t].append(float(np.corrcoef(n_, s_)[0, 1]))
            for nm, fn, _ in CAMS:
                m = norm(fn(h, A, c))
                maps_by_T[(nm, t)] = m
                mm = m.cpu().numpy()
                if mm.std() > 1e-12:
                    if n_.std() > 1e-12:
                        acc[(nm, t)]['devN'].append(spearmanr(mm, n_).statistic)
                    if s_.std() > 1e-12:
                        acc[(nm, t)]['devS'].append(spearmanr(mm, s_).statistic)
                if t == 0.0:
                    m0[nm] = m
        for nm, _, _ in CAMS:
            for t in TS:
                d = float((maps_by_T[(nm, t)] - m0[nm]).abs().max())
                acc[(nm, t)]['d0'].append(d)
            a = maps_by_T[(nm, min(TS))].cpu().numpy()
            z = maps_by_T[(nm, max(TS))].cpu().numpy()
            if a.std() > 1e-12 and z.std() > 1e-12:
                for t in TS:
                    acc[(nm, t)]['sp08'].append(spearmanr(a, z).statistic)
        if (k + 1) % 16 == 0:
            print(f'  {k + 1}/{NIMG}  {time.time() - t0:.0f}s', flush=True)

    out = {'meta': dict(body=BODY, ds=DS, C=C, cells=T, n_images=NIMG,
                        temperatures=TS, weights='shipped torchvision fc, '
                        'no parameter trained', chunk=CHUNK)}
    out['corrNS'] = {str(t): float(np.mean(ns[t])) for t in TS if ns[t]}
    rows = {}
    for nm, _, kind in CAMS:
        r = {'kind': kind, 'by_T': {}}
        for t in TS:
            a = acc[(nm, t)]
            r['by_T'][str(t)] = dict(
                max_dev_from_T0=float(np.max(a['d0'])) if a['d0'] else None,
                mean_dev_from_T0=float(np.mean(a['d0'])) if a['d0'] else None,
                rho_with_necessity=float(np.mean(a['devN'])) if a['devN'] else None,
                rho_with_sufficiency=float(np.mean(a['devS'])) if a['devS'] else None)
        r['max_dev_over_all_T'] = float(max(
            v['max_dev_from_T0'] for v in r['by_T'].values()
            if v['max_dev_from_T0'] is not None))
        sp = acc[(nm, TS[0])]['sp08']
        r['rho_between_the_two_extremes'] = float(np.mean(sp)) if sp else None
        rows[nm] = r
    out['cams'] = rows

    L.save('probe30_camfamily.json', out)

    # ------------------------------------------------------------ the table
    print()
    print(f"{'method':14s}{'kind':16s}{'max|m(T)-m(0)|':>16s}"
          f"{'rho(-8,+8)':>12s}{'rho(N) T=0':>12s}{'rho(N) T=8':>12s}"
          f"{'rho(S) T=8':>12s}")
    print('-' * 94)
    for nm, _, kind in CAMS:
        r = rows[nm]
        hi = str(max(TS))
        print(f"{nm:14s}{kind:16s}{r['max_dev_over_all_T']:16.2e}"
              f"{r['rho_between_the_two_extremes']:12.3f}"
              f"{r['by_T']['0.0']['rho_with_necessity']:12.3f}"
              f"{r['by_T'][hi]['rho_with_necessity']:12.3f}"
              f"{r['by_T'][hi]['rho_with_sufficiency']:12.3f}")
    print('-' * 94)
    print('control: corr(N, S) by temperature ->',
          {k: round(v, 6) for k, v in out['corrNS'].items()})


if __name__ == '__main__':
    main()
