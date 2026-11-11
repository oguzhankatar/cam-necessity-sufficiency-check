"""PROBE 1 -- is the collapse caused by the LINEAR AGGREGATOR or by the head?

Backbone frozen (ResNet50 layer4, cached).  Only the spatial aggregator varies.
Each head is trained to comparable accuracy, so any difference in additivity is
NOT a difference in what the model knows.

METRICS (all with a bank-sampled baseline -- never a per-image channel mean):
  D(t)    = z(A) - z(A with spatial cell t swapped to the baseline)
  I2      = E|D(t1,t2) - D(t1) - D(t2)| / E|D(t1,t2)|       pairwise interaction
  Rhalf   = E|D(S) - sum_t D(t)| / E|D(S)|, |S|=24          subset interaction
  corrNS  = corr over cells of necessity D(t) vs sufficiency
            [z(B with cell t from A) - z(B)]
These are EXACT finite differences: no variance estimator, no independence
assumption, so none of the Sobol traps apply.  I2 = Rhalf = 0 and corrNS = 1
exactly iff the readout is additive over spatial cells.
"""
import os, sys, json, time, numpy as np, torch, torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(__file__))
from heads import build, GROUP

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'outputs')
dev = 'cuda'
HEADS = ['gap_fc', 'gap_mlp', 'ln_gap', 'tok_mlp',
         'lse0.25', 'lse0.5', 'lse1.0', 'lse2.0', 'lse4.0', 'lse8.0',
         'maxpool', 'attn', 'bilin']


def load(split):
    F = np.load(os.path.join(OUT, f'feat_{split}.npy'), mmap_mode='r')
    Y = np.load(os.path.join(OUT, f'lab_{split}.npy'))
    return F, Y


def batch(F, idx):
    a = torch.from_numpy(np.ascontiguousarray(F[idx])).to(dev).float()
    return a.reshape(a.shape[0], a.shape[1], -1)            # (B,C,T)


def train(name, Ftr, Ytr, Fva, Yva, epochs=8, bs=96, seed=0):
    torch.manual_seed(seed)
    h = build(name).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Ytr); steps = epochs * (n // bs)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    yt = torch.from_numpy(Ytr).to(dev)
    rs = np.random.RandomState(seed)
    for ep in range(epochs):
        perm = rs.permutation(n)
        for i in range(0, n - bs + 1, bs):
            j = np.sort(perm[i:i + bs])
            loss = Fn.cross_entropy(h(batch(Ftr, j)), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    h.eval()
    with torch.no_grad():
        ok = 0
        for i in range(0, len(Yva), 128):
            j = np.arange(i, min(i + 128, len(Yva)))
            ok += (h(batch(Fva, j)).argmax(1).cpu().numpy() == Yva[j]).sum()
    return h, float(ok / len(Yva))


# ---------------------------------------------------------------- metrics
@torch.no_grad()
def interaction(h, A, B, c, npair=256, nsub=64, sub=24, rs=None):
    """A,B: (C,T) image and baseline feature maps. Returns I2, Rhalf, corrNS."""
    Cc, T = A.shape
    zA = h(A[None])[0, c]; zB = h(B[None])[0, c]

    def rep(masks):                      # masks (M,T) bool: True -> take B
        M = masks.shape[0]
        X = A[None].expand(M, -1, -1).clone()
        X[masks[:, None, :].expand(-1, Cc, -1)] = \
            B[None].expand(M, -1, -1)[masks[:, None, :].expand(-1, Cc, -1)]
        out = []
        for s in range(0, M, 128):
            out.append(h(X[s:s + 128])[:, c])
        return torch.cat(out)

    eye = torch.eye(T, dtype=torch.bool, device=A.device)
    D1 = zA - rep(eye)                                        # necessity D(t)
    # sufficiency: start from B, insert cell t from A  == delete all but t
    S1 = rep(~eye) - zB
    pn, ps = D1.cpu().numpy(), S1.cpu().numpy()
    corr = float(np.corrcoef(pn, ps)[0, 1]) if pn.std() > 1e-12 and ps.std() > 1e-12 else np.nan

    p = rs.randint(0, T, (npair, 2)); p = p[p[:, 0] != p[:, 1]]
    mk = torch.zeros(len(p), T, dtype=torch.bool, device=A.device)
    mk[torch.arange(len(p)), torch.from_numpy(p[:, 0]).to(A.device)] = True
    mk[torch.arange(len(p)), torch.from_numpy(p[:, 1]).to(A.device)] = True
    D2 = zA - rep(mk)
    add = D1[torch.from_numpy(p[:, 0]).to(A.device)] + D1[torch.from_numpy(p[:, 1]).to(A.device)]
    I2 = float((D2 - add).abs().mean() / (D2.abs().mean() + 1e-12))

    ms = torch.zeros(nsub, T, dtype=torch.bool, device=A.device)
    for i in range(nsub):
        ms[i, torch.from_numpy(rs.choice(T, sub, replace=False)).to(A.device)] = True
    DS = zA - rep(ms)
    addS = (ms.float() * D1[None]).sum(1)
    Rh = float((DS - addS).abs().mean() / (DS.abs().mean() + 1e-12))
    return I2, Rh, corr, float(D1.abs().mean())


@torch.no_grad()
def sobol(h, A, bankC, U, res, c, NS=8192, R=8, CH=256):
    """Valid construction: independent Bernoulli switches over a DECORRELATED
    (SVD) basis, coefficients swapped with those of bank images."""
    cs = U.T @ A                                        # (R,T)

    def ev(u, cb):
        o = []
        for s in range(0, u.shape[0], CH):
            uu = u[s:s + CH]
            cc = uu[:, :, None] * cs[None] + (1 - uu[:, :, None]) * cb[s:s + CH]
            o.append(h(torch.einsum('cr,nrt->nct', U, cc) + res[None])[:, c])
        return torch.cat(o)

    idxA = torch.randint(0, bankC.shape[0], (NS,), device=A.device)
    idxB = torch.randint(0, bankC.shape[0], (NS,), device=A.device)
    cbA, cbB = bankC[idxA], bankC[idxB]
    uA = (torch.rand(NS, R, device=A.device) < .5).float()
    uB = (torch.rand(NS, R, device=A.device) < .5).float()
    fA, fB = ev(uA, cbA), ev(uB, cbB)
    V = torch.var(torch.cat([fA, fB]))
    Si, ST = np.zeros(R), np.zeros(R)
    for i in range(R):
        uAB = uA.clone(); uAB[:, i] = uB[:, i]
        cAB = cbA.clone(); cAB[:, i] = cbB[:, i]
        fAB = ev(uAB, cAB)
        Si[i] = ((fB * (fAB - fA)).mean() / V).item()
        ST[i] = ((.5 * ((fA - fAB) ** 2).mean()) / V).item()
    return Si, ST


def main():
    Ftr, Ytr = load('train'); Fva, Yva = load('val')
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:96]
    imgs = sel[:48]; bankidx = sel[48:80]
    bank = batch(Fva, np.sort(bankidx))                      # (Nb,C,T)
    Xi = batch(Fva, np.sort(imgs))

    # shared SVD basis for the Sobol probe
    R = 8
    U = torch.linalg.svd(bank.permute(1, 0, 2).reshape(bank.shape[1], -1),
                         full_matrices=False)[0][:, :R]
    bankC = torch.einsum('cr,nct->nrt', U, bank)

    res = {}
    for name in HEADS:
        t0 = time.time()
        h, acc = train(name, Ftr, Ytr, Fva, Yva)
        rs2 = np.random.RandomState(1)
        I2s, Rhs, cs_, mags = [], [], [], []
        with torch.no_grad():
            for k in range(Xi.shape[0]):
                A = Xi[k]
                c = int(h(A[None]).argmax(1))
                for b in range(3):
                    B = bank[rs2.randint(bank.shape[0])]
                    i2, rh, cc, mg = interaction(h, A, B, c, rs=rs2)
                    I2s.append(i2); Rhs.append(rh); mags.append(mg)
                    if not np.isnan(cc): cs_.append(cc)
            # Sobol on 6 images
            S1, ST1 = [], []
            for k in range(6):
                A = Xi[k]; c = int(h(A[None]).argmax(1))
                r_ = A - U @ (U.T @ A)
                Si, STi = sobol(h, A, bankC, U, r_, c, NS=8192, R=R)
                S1.append(Si); ST1.append(STi)
            S1 = np.array(S1); ST1 = np.array(ST1)

        res[name] = dict(group=GROUP[name], acc=acc,
                         I2=float(np.mean(I2s)), I2_sd=float(np.std(I2s)),
                         Rhalf=float(np.mean(Rhs)),
                         corrNS=float(np.mean(cs_)), Dmag=float(np.mean(mags)),
                         sumS=float(S1.sum(1).mean()), sumST=float(ST1.sum(1).mean()),
                         inter=float((ST1.sum(1) - S1.sum(1)).mean()),
                         minS=float(S1.min()), sec=time.time() - t0)
        r = res[name]
        print(f"{name:9s} {r['group']:13s} acc={acc:.4f}  I2={r['I2']:.4f}  "
              f"Rhalf={r['Rhalf']:.4f}  corrNS={r['corrNS']:+.4f}  "
              f"sumS={r['sumS']:.3f} sumST={r['sumST']:.3f} "
              f"int={r['inter']:+.4f} minS={r['minS']:+.4f} ({r['sec']:.0f}s)",
              flush=True)

    json.dump(res, open(os.path.join(OUT, 'probe1_aggregator.json'), 'w'), indent=1)
    print('\nwrote outputs/probe1_aggregator.json')


if __name__ == '__main__':
    main()
