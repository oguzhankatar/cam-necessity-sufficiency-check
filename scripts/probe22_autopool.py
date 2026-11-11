"""PROBE 22 (P4) -- A THIRD, *LEARNED* OPERATOR FAMILY: AutoPool.

LSE-T and k-of-N both reproduce the AND/OR fan, but in both the operator's
position on the axis is a hyper-parameter the experimenter sets.  AutoPool
(McFee, Salamon & Bello, IEEE TASLP 26(11), 2018) is structurally different on
both counts:

    z_k = sum_t softmax_t(alpha_k * z_kt) * z_kt          z_kt = w_k . a_t + b_k

It is a *self-attention-weighted average of the per-token logits*, not a
log-sum-exp; at alpha = 0 the weights are uniform and it is exactly GAP + FC; as
alpha -> +inf it is max, as alpha -> -inf it is min.  And `alpha` is a
PARAMETER, so gradient descent chooses the position on the axis.

Three conditions:
  A  FAN      alpha frozen at 11 values          -> does a third family fan?
  B  LEARNED  alpha free, several inits          -> where does training put it?
  C  ATTNMIL  gated attention pooling (Ilse et al. ICML 2018), a learned
              aggregator with no axis parameter at all -> where does it land?

A scale caveat that matters and is reported: alpha is NOT identifiable on its
own, because alpha * z is invariant under (alpha, w) -> (alpha/s, s*w).  The
scale-free summary of where an aggregator sits is the PARTICIPATION RATIO of its
pooling weights, PR = 1 / (T * sum_t w_t^2): PR = 1 is a uniform average (GAP),
PR = 1/T is a hard max/min.  Both are reported.

Internal control: alpha = 0 frozen is algebraically GAP + FC, so it must give
I2 = Rhalf = 0 and corrNS = +1.000000.
"""
import os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L

BODY = os.environ.get('BODY', 'resnet50')
DS = os.environ.get('DS', 'imagenette')
NIMG = int(os.environ.get('NIMG', 32))
EPOCHS = int(os.environ.get('EPOCHS', 8))
ALPHAS = [float(a) for a in os.environ.get(
    'ALPHAS', '-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8').split(',')]
INITS = [float(a) for a in os.environ.get('INITS', '-4,-1,0,1,4').split(',')]
dev = 'cuda'


class AutoPool(nn.Module):
    def __init__(s, k=10, c=2048, alpha=0.0, learn=False, shared=False):
        super().__init__()
        s.fc = nn.Linear(c, k)
        a = torch.full((1,) if shared else (k,), float(alpha))
        s.alpha = nn.Parameter(a) if learn else nn.Parameter(a,
                                                             requires_grad=False)
        s.frozen_zero = (not learn) and alpha == 0.0

    def weights(s, A):
        z = s.fc(A.transpose(1, 2))                      # (B,T,K)
        return torch.softmax(s.alpha[None, None, :] * z, 1), z

    def forward(s, A):
        w, z = s.weights(A)
        if s.frozen_zero:                                # exact GAP+FC branch
            return z.mean(1)
        return (w * z).sum(1)


def part_ratio(h, A):
    """PR of the pooling weights for the predicted class, averaged over images."""
    with torch.no_grad():
        w, z = h.weights(A)
        c = h(A).argmax(1)
        wk = w[torch.arange(len(c)), :, c]               # (B,T)
        T = wk.shape[1]
        return float((1.0 / (T * (wk ** 2).sum(1))).mean())


def train(h, Ftr, Ytr, Fva, Yva, epochs, bs=96, seed=0, lr=1e-3):
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Ytr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs*(n//bs)))
    yt = torch.from_numpy(Ytr).to(dev); rs = np.random.RandomState(seed)
    for ep in range(epochs):
        perm = rs.permutation(n)
        for i in range(0, n - bs + 1, bs):
            j = np.sort(perm[i:i + bs])
            loss = Fn.cross_entropy(h(L.batch(Ftr, j)), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    h.eval()
    with torch.no_grad():
        ok = 0
        for i in range(0, len(Yva), 128):
            j = np.arange(i, min(i + 128, len(Yva)))
            ok += (h(L.batch(Fva, j)).argmax(1).cpu().numpy() == Yva[j]).sum()
    return h, float(ok / len(Yva))


def measure(h, Xi, bank):
    rs2 = np.random.RandomState(1)
    I2s, Rhs, cs, sns, sgn, sgs = [], [], [], [], [], []
    D = {k: [] for k in ('N', 'S', 'G', 'rand')}
    I = {k: [] for k in ('N', 'S', 'G', 'rand')}
    for k in range(Xi.shape[0]):
        A = Xi[k]
        with torch.no_grad():
            c = int(h(A[None]).argmax(1))
        B = bank[rs2.randint(bank.shape[0])]
        i2, rh, cc, mg, zm = L.interaction(h, A, B, c, rs2)
        I2s.append(i2); Rhs.append(rh)
        if not np.isnan(cc):
            cs.append(cc)
        N, S = L.maps(h, A, B, c); G = L.gradcam_head(h, A, c)
        n, s, g = [v.cpu().numpy() for v in (N, S, G)]
        if n.std() > 1e-9 and s.std() > 1e-9: sns.append(spearmanr(n, s).statistic)
        if n.std() > 1e-9 and g.std() > 1e-9: sgn.append(spearmanr(g, n).statistic)
        if s.std() > 1e-9 and g.std() > 1e-9: sgs.append(spearmanr(g, s).statistic)
        rnd = torch.randn(A.shape[1], device=A.device)
        for tag, o in (('N', N), ('S', S), ('G', G), ('rand', rnd)):
            D[tag].append(L.auc(h, A, B, c, o, 'del'))
            I[tag].append(L.auc(h, A, B, c, o, 'ins'))
    r = dict(I2=float(np.mean(I2s)), Rhalf=float(np.mean(Rhs)),
             corrNS=float(np.mean(cs)) if cs else float('nan'),
             spNS=float(np.mean(sns)), spGN=float(np.mean(sgn)),
             spGS=float(np.mean(sgs)))
    for t in D:
        r[f'del_{t}'] = float(np.mean(D[t])); r[f'ins_{t}'] = float(np.mean(I[t]))
    return r


HDR = (f"{'cond':>14s}{'acc':>8s}{'alpha':>9s}{'PR':>7s}{'I2':>8s}{'Rhalf':>8s}"
       f"{'corrNS':>9s}{'spNS':>8s}{'spGN':>8s}{'spGS':>8s} | "
       f"{'delN':>6s}{'delS':>6s}{'delG':>6s}{'delR':>6s} | "
       f"{'insN':>6s}{'insS':>6s}{'insG':>6s}{'insR':>6s}")


def show(tag, r):
    print(f"{tag:>14s}{r['acc']:8.4f}{r['alpha_mean']:+9.3f}{r['PR']:7.3f}"
          f"{r['I2']:8.4f}{r['Rhalf']:8.4f}{r['corrNS']:+9.4f}{r['spNS']:+8.3f}"
          f"{r['spGN']:+8.3f}{r['spGS']:+8.3f} | {r['del_N']:6.3f}"
          f"{r['del_S']:6.3f}{r['del_G']:6.3f}{r['del_rand']:6.3f} | "
          f"{r['ins_N']:6.3f}{r['ins_S']:6.3f}{r['ins_G']:6.3f}"
          f"{r['ins_rand']:6.3f}", flush=True)


def main():
    mp_ = int(os.environ['MAXPC']) if os.environ.get('MAXPC') else None
    sz_ = int(os.environ['SIZE']) if os.environ.get('SIZE') else None
    ftr, ytr, C, T = L.cache_feats(BODY, DS, 'train', size=sz_, max_per_class=mp_)
    fva, yva, _, _ = L.cache_feats(BODY, DS, 'val', size=sz_)
    Ftr, Ytr = L.load_cached(ftr, ytr); Fva, Yva = L.load_cached(fva, yva)
    ncl = int(Ytr.max()) + 1
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:NIMG + 48]
    Xi = L.batch(Fva, np.sort(sel[:NIMG]))
    bank = L.batch(Fva, np.sort(sel[NIMG + 16:NIMG + 48]))
    res = {'meta': dict(body=BODY, ds=DS, cells=T, nclass=ncl, n_img=NIMG,
                        epochs=EPOCHS)}
    print(f'[{BODY}/{DS}] AutoPool, {T} cells, {ncl} classes', flush=True)

    print('\n=== A. AutoPool with alpha FROZEN: does a third family fan? ===')
    print(HDR, flush=True)
    for a in ALPHAS:
        torch.manual_seed(0)
        h = AutoPool(ncl, C, alpha=a, learn=False).to(dev)
        h, acc = train(h, Ftr, Ytr, Fva, Yva, EPOCHS)
        r = measure(h, Xi, bank)
        r.update(acc=acc, alpha_mean=a, PR=part_ratio(h, Xi))
        res[f'ap{a}'] = r; show(f'ap a={a:+.1f}', r)
        L.save(f'probe22_autopool_{BODY}_{DS}.json', res)

    print('\n=== B. AutoPool with alpha LEARNED (per class): where does '
          'training put it? ===')
    print(HDR, flush=True)
    for a0 in INITS:
        torch.manual_seed(0)
        h = AutoPool(ncl, C, alpha=a0, learn=True).to(dev)
        h, acc = train(h, Ftr, Ytr, Fva, Yva, EPOCHS)
        al = h.alpha.detach().cpu().numpy()
        r = measure(h, Xi, bank)
        r.update(acc=acc, alpha_mean=float(al.mean()), PR=part_ratio(h, Xi),
                 alpha_init=a0, alpha_min=float(al.min()),
                 alpha_max=float(al.max()), alpha_med=float(np.median(al)),
                 alpha_all=[float(x) for x in al])
        res[f'learn{a0}'] = r; show(f'learn a0={a0:+.1f}', r)
        print(f"{'':>14s}  alpha after training: min {r['alpha_min']:+.3f} "
              f"med {r['alpha_med']:+.3f} max {r['alpha_max']:+.3f}", flush=True)
        L.save(f'probe22_autopool_{BODY}_{DS}.json', res)

    print('\n=== C. gated attention-MIL pooling (no axis parameter at all) ===')
    print(HDR, flush=True)
    from heads import Attn
    torch.manual_seed(0)
    h = Attn(ncl, c=C).to(dev)
    h, acc = train(h, Ftr, Ytr, Fva, Yva, EPOCHS)
    with torch.no_grad():
        X = Xi.transpose(1, 2)
        al = torch.softmax(h.w(torch.tanh(h.V(X)) * torch.sigmoid(h.U(X))), 1)[..., 0]
        pr = float((1.0 / (al.shape[1] * (al ** 2).sum(1))).mean())
    r = measure(h, Xi, bank); r.update(acc=acc, alpha_mean=float('nan'), PR=pr)
    res['attnmil'] = r; show('attn-MIL', r)
    L.save(f'probe22_autopool_{BODY}_{DS}.json', res)

    z = res.get('ap0.0')
    if z is not None:
        ok = (abs(z['I2']) < 1e-4 and abs(z['Rhalf']) < 1e-4
              and abs(z['corrNS'] - 1) < 1e-6
              and abs(z['del_N'] - z['del_S']) < 1e-9)
        res['control_alpha0'] = bool(ok)
        print(f"\nCONTROL alpha=0 (= GAP+FC): {'PASS' if ok else 'FAIL'}  "
              f"I2={z['I2']:.2e} Rhalf={z['Rhalf']:.2e} "
              f"corrNS={z['corrNS']:.6f} delN-delS={z['del_N']-z['del_S']:.2e}",
              flush=True)
    L.save(f'probe22_autopool_{BODY}_{DS}.json', res)


if __name__ == '__main__':
    main()
