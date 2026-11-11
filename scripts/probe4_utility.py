"""PROBE 4 -- does the restored interaction BUY anything explanatory?

Probe 1 showed a non-additive aggregator separates necessity from sufficiency
at essentially zero accuracy cost.  But separation is only interesting if the
two maps are USEFUL and if the cheap gradient proxy stops working.

Per aggregator head, over 49 layer4 cells:
  N(t)  exact necessity   z(A) - z(A with cell t -> baseline)
  S(t)  exact sufficiency z(B with cell t from A) - z(B)
  G(t)  Grad-CAM          ReLU(sum_c dz/dA * A)
Measured:
  sp_GN   spearman(G, N)   -- does the gradient proxy still track necessity?
  sp_NS   spearman(N, S)   -- do the two maps actually rank cells differently?
  del/ins AUC in feature space (softmax prob) for each ranking
  del_opt AUC of the ORACLE ranking (sorted exact N) -- for a GAP head the
          logit deletion curve is exactly the cumsum of N, so this is the
          well-posed optimum and any map's gap from it is pure ranking error.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as Fn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from heads import build, GROUP
from probe1_aggregator import load, batch, train, OUT, HEADS

dev = 'cuda'


@torch.no_grad()
def maps(h, A, B, c):
    T = A.shape[1]
    eye = torch.eye(T, dtype=torch.bool, device=A.device)

    def rep(base, other, mk):
        X = base[None].expand(mk.shape[0], *base.shape).clone()
        m = mk[:, None, :].expand(-1, base.shape[0], -1)
        X[m] = other[None].expand(mk.shape[0], *base.shape)[m]
        return torch.cat([h(X[s:s + 64])[:, c] for s in range(0, X.shape[0], 64)])

    N = h(A[None])[0, c] - rep(A, B, eye)
    S = rep(B, A, eye) - h(B[None])[0, c]
    return N, S


def gradcam_head(h, A, c):
    A = A[None].detach().requires_grad_(True)
    with torch.enable_grad():
        g, = torch.autograd.grad(h(A)[0, c], A)
    w = g.mean(dim=2, keepdim=True)
    return Fn.relu((w * A).sum(1))[0].detach()


@torch.no_grad()
def auc(h, A, B, c, order, mode, steps=49):
    T = A.shape[1]
    rank = torch.argsort(order, descending=True)
    start, fin = (A, B) if mode == 'del' else (B, A)
    cur = start.clone(); ps = [torch.softmax(h(cur[None]), 1)[0, c].item()]
    for i in range(steps):
        cur[:, rank[i]] = fin[:, rank[i]]
        ps.append(torch.softmax(h(cur[None]), 1)[0, c].item())
    p = np.array(ps)
    return float((p.sum() - p[0] / 2 - p[-1] / 2) / (len(p) - 1))


def main():
    Ftr, Ytr = load('train'); Fva, Yva = load('val')
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:96]
    Xi = batch(Fva, np.sort(sel[:32])); bank = batch(Fva, np.sort(sel[48:80]))
    res = {}
    for name in HEADS:
        h, acc = train(name, Ftr, Ytr, Fva, Yva)
        rs2 = np.random.RandomState(1)
        sgn, sns, D, I = [], [], {k: [] for k in ('N', 'S', 'G', 'rand')}, \
            {k: [] for k in ('N', 'S', 'G', 'rand')}
        for k in range(Xi.shape[0]):
            A = Xi[k]
            with torch.no_grad():
                c = int(h(A[None]).argmax(1))
            B = bank[rs2.randint(bank.shape[0])]
            N, S = maps(h, A, B, c)
            G = gradcam_head(h, A, c)
            n, s, g = [v.cpu().numpy() for v in (N, S, G)]
            if n.std() > 1e-9 and g.std() > 1e-9:
                sgn.append(spearmanr(g, n).statistic)
            if n.std() > 1e-9 and s.std() > 1e-9:
                sns.append(spearmanr(n, s).statistic)
            rnd = torch.randn(A.shape[1], device=A.device)
            for tag, o in (('N', N), ('S', S), ('G', G), ('rand', rnd)):
                D[tag].append(auc(h, A, B, c, o, 'del'))
                I[tag].append(auc(h, A, B, c, o, 'ins'))
        r = dict(group=GROUP[name], acc=acc,
                 sp_GradCAM_vs_necessity=float(np.mean(sgn)),
                 sp_necessity_vs_sufficiency=float(np.mean(sns)),
                 **{f'del_{t}': float(np.mean(D[t])) for t in D},
                 **{f'ins_{t}': float(np.mean(I[t])) for t in I})
        res[name] = r
        print(f"{name:9s} {r['group']:13s} acc={acc:.4f} spGN={r['sp_GradCAM_vs_necessity']:+.3f} "
              f"spNS={r['sp_necessity_vs_sufficiency']:+.3f} | del N={r['del_N']:.3f} "
              f"S={r['del_S']:.3f} G={r['del_G']:.3f} rnd={r['del_rand']:.3f} | "
              f"ins N={r['ins_N']:.3f} S={r['ins_S']:.3f} G={r['ins_G']:.3f}", flush=True)
    json.dump(res, open(os.path.join(OUT, 'probe4_utility.json'), 'w'), indent=1)
    print('\nwrote outputs/probe4_utility.json')


if __name__ == '__main__':
    main()
