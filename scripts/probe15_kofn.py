"""PROBE 15 -- the fan with a COMPLETELY DIFFERENT AND/OR parameterisation.

Every AND/OR result so far -- stage 1 and stage-2 Probes 10a/b/c -- uses the
same operator family, LSE with temperature T.  A reader's fair objection is
that the fan might be a property of log-sum-exp (smooth, exponential, one
scalar) rather than of the AND--OR axis itself.

k-of-N pooling is the Boolean threshold family and shares nothing with LSE
except the axis: it is piecewise linear, has no exponentials, no temperature,
and a DISCRETE knob.  Per class, pool the k largest per-cell logits (top-k mean)
or the k smallest (bottom-k mean):

    top-k,   k = 1   -> max   = OR    ("at least one cell fires")
    top-k,   k = T   -> mean  = GAP   (exactly additive; the control)
    bottom-k,k = 1   -> min   = AND   ("every cell must fire")

If the fan is about the axis, it must appear here too, with the SAME
Grad-CAM-tracks-sufficiency signature.  If it is about LSE, it must not.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L

BODY = os.environ.get('BODY', 'resnet50')
DS = os.environ.get('DS', 'imagenette')
NIMG = int(os.environ.get('NIMG', 32))
EPOCHS = int(os.environ.get('EPOCHS', 8))
KS = [int(k) for k in os.environ.get('KS', '1,2,4,8,16,32,49').split(',')]


class SortPool(nn.Module):
    """k-of-N pooling. side=+1 top-k (OR-ward), side=-1 bottom-k (AND-ward)."""
    def __init__(s, k=10, c=2048, kk=1, side=1):
        super().__init__(); s.fc = nn.Linear(c, k); s.kk = kk; s.side = side

    def forward(s, A):
        z = s.fc(A.transpose(1, 2))                       # (B,T,K)
        T = z.shape[1]
        if s.kk >= T:
            return z.mean(1)                              # exactly GAP + FC
        v, _ = torch.sort(z, dim=1, descending=(s.side > 0))
        return v[:, :s.kk].mean(1)


def train(kk, side, Ftr, Ytr, Fva, Yva, ncl, C, epochs, seed=0):
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    h = SortPool(ncl, C, kk, side).to(L.dev)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Ytr); bs = 96
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (n // bs))
    yt = torch.from_numpy(Ytr).to(L.dev); rs = np.random.RandomState(seed)
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


def main():
    ftr, ytr, C, T = L.cache_feats(BODY, DS, 'train')
    fva, yva, _, _ = L.cache_feats(BODY, DS, 'val')
    Ftr, Ytr = L.load_cached(ftr, ytr); Fva, Yva = L.load_cached(fva, yva)
    ncl = int(Ytr.max()) + 1
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:NIMG + 48]
    Xi = L.batch(Fva, np.sort(sel[:NIMG]))
    bank = L.batch(Fva, np.sort(sel[NIMG + 16:NIMG + 48]))
    print(f'[{BODY}/{DS}] C={C} cells={T} classes={ncl}', flush=True)
    res = {'meta': dict(body=BODY, ds=DS, cells=T, nclass=ncl, ks=KS)}
    print(f"{'pool':>10s}{'acc':>8s}{'I2':>8s}{'Rhalf':>8s}{'corrNS':>9s}"
          f"{'spNS':>8s}{'spGN':>8s}{'spGS':>8s} | {'delN':>6s}{'delS':>6s}"
          f"{'delG':>6s}{'delR':>6s} | {'insN':>6s}{'insS':>6s}{'insG':>6s}"
          f"{'insR':>6s}", flush=True)
    combos = [(k, -1) for k in sorted(KS)] + [(T, 0)] + \
             [(k, +1) for k in sorted(KS, reverse=True)]
    seen = set()
    for kk, side in combos:
        lab = ('GAP(k=N)' if side == 0 or kk >= T else
               (f'bot-{kk}(AND)' if side < 0 else f'top-{kk}(OR)'))
        if lab in seen:
            continue
        seen.add(lab)
        t0 = time.time()
        h, acc = train(kk, 1 if side == 0 else side, Ftr, Ytr, Fva, Yva, ncl, C,
                       EPOCHS)
        rs2 = np.random.RandomState(1)
        I2s, Rhs, cs, sns, sgn, sgs = [], [], [], [], [], []
        D = {k: [] for k in ('N', 'S', 'G', 'rand')}
        I = {k: [] for k in ('N', 'S', 'G', 'rand')}
        for q in range(Xi.shape[0]):
            A = Xi[q]
            with torch.no_grad():
                c = int(h(A[None]).argmax(1))
            B = bank[rs2.randint(bank.shape[0])]
            i2, rh, cc, _, _ = L.interaction(h, A, B, c, rs2)
            I2s.append(i2); Rhs.append(rh)
            if not np.isnan(cc):
                cs.append(cc)
            N, S = L.maps(h, A, B, c); Gm = L.gradcam_head(h, A, c)
            n, s, g = [v.cpu().numpy() for v in (N, S, Gm)]
            if n.std() > 1e-9 and s.std() > 1e-9: sns.append(spearmanr(n, s).statistic)
            if n.std() > 1e-9 and g.std() > 1e-9: sgn.append(spearmanr(g, n).statistic)
            if s.std() > 1e-9 and g.std() > 1e-9: sgs.append(spearmanr(g, s).statistic)
            rnd = torch.randn(A.shape[1], device=A.device)
            for tag, o in (('N', N), ('S', S), ('G', Gm), ('rand', rnd)):
                D[tag].append(L.auc(h, A, B, c, o, 'del'))
                I[tag].append(L.auc(h, A, B, c, o, 'ins'))
        r = dict(pool=lab, k=kk, side=side, acc=acc, I2=float(np.mean(I2s)),
                 Rhalf=float(np.mean(Rhs)), corrNS=float(np.mean(cs)),
                 spNS=float(np.mean(sns)), spGN=float(np.mean(sgn)),
                 spGS=float(np.mean(sgs)), sec=time.time() - t0)
        for t in D:
            r[f'del_{t}'] = float(np.mean(D[t])); r[f'ins_{t}'] = float(np.mean(I[t]))
        res[lab] = r
        print(f"{lab:>10s}{acc:8.4f}{r['I2']:8.4f}{r['Rhalf']:8.4f}"
              f"{r['corrNS']:+9.4f}{r['spNS']:+8.3f}{r['spGN']:+8.3f}"
              f"{r['spGS']:+8.3f} | {r['del_N']:6.3f}{r['del_S']:6.3f}"
              f"{r['del_G']:6.3f}{r['del_rand']:6.3f} | {r['ins_N']:6.3f}"
              f"{r['ins_S']:6.3f}{r['ins_G']:6.3f}{r['ins_rand']:6.3f}"
              f"  ({r['sec']:.0f}s)", flush=True)
        L.save(f'probe15_kofn_{BODY}_{DS}.json', res)
    z = res.get('GAP(k=N)')
    if z is not None:
        ok = (abs(z['I2']) < 1e-4 and abs(z['Rhalf']) < 1e-4
              and abs(z['corrNS'] - 1) < 1e-6)
        res['control_GAP_exact'] = bool(ok)
        print(f"\nCONTROL k=N (=GAP) exact degeneracy: {'PASS' if ok else 'FAIL'} "
              f"(I2={z['I2']:.2e} Rhalf={z['Rhalf']:.2e} corrNS={z['corrNS']:.6f})")
        L.save(f'probe15_kofn_{BODY}_{DS}.json', res)


if __name__ == '__main__':
    main()
