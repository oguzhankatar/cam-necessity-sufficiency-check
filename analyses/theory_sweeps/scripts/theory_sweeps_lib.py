"""C4 additional analyses -- shared machinery.

Read-only use of the project: imports scripts/heads.py (head definitions) and the
cached Imagenette ResNet-50 layer4 features in outputs/. Nothing under scripts/
or outputs/ is written. Every head here is fitted with the recipe of
scripts/probe1_aggregator.py:train (AdamW lr 1e-3, wd 1e-4, cosine, batch 96,
8 epochs, seed 0, sorted mini-batch indices), so a head named like a probe-1
head is the same model up to GPU non-determinism.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
C4 = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(C4, '..', '..'))
SCRIPTS = os.path.join(ROOT, 'scripts')
OUTP = os.path.join(ROOT, 'outputs')
OUT = os.path.join(C4, 'out')
HEADDIR = os.path.join(OUT, 'heads')
sys.path.insert(0, SCRIPTS)
from heads import build as _build  # noqa: E402

dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class LNPost(nn.Module):
    """Post-mean LayerNorm head: fc(LN(mean_t a_t)) -- the ordering every timm
    ViT ships (fc_norm after the mean). Not in scripts/heads.py; added here."""
    def __init__(s, k=10, c=2048):
        super().__init__(); s.ln = nn.LayerNorm(c); s.fc = nn.Linear(c, k)

    def forward(s, A): return s.fc(s.ln(A.mean(-1)))


def build(name, k=10, c=2048):
    if name == 'ln_post':
        return LNPost(k, c)
    return _build(name, k, c)


_F = {}


def feats(split):
    """Whole split on the GPU as fp16 (B,C,T); identical values to probe1.batch."""
    if split not in _F:
        F = np.load(os.path.join(OUTP, f'feat_{split}.npy'), mmap_mode='r')
        Y = np.load(os.path.join(OUTP, f'lab_{split}.npy'))
        t = torch.from_numpy(np.ascontiguousarray(F)).to(dev)
        _F[split] = (t.reshape(t.shape[0], t.shape[1], -1), Y)
    return _F[split]


def batch(split, idx):
    F, _ = feats(split)
    return F[torch.as_tensor(np.asarray(idx), device=dev)].float()


def train(name, epochs=8, bs=96, seed=0):
    """probe1_aggregator.train, verbatim recipe."""
    Ftr, Ytr = feats('train'); Fva, Yva = feats('val')
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
            loss = Fn.cross_entropy(h(batch('train', j)), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    h.eval()
    return h, evaluate(h)


@torch.no_grad()
def evaluate(h):
    Fva, Yva = feats('val'); ok = 0
    for i in range(0, len(Yva), 128):
        j = np.arange(i, min(i + 128, len(Yva)))
        ok += (h(batch('val', j)).argmax(1).cpu().numpy() == Yva[j]).sum()
    return float(ok / len(Yva))


def get_head(name, double=False):
    """Fit once, cache the state dict under out/heads/."""
    os.makedirs(HEADDIR, exist_ok=True)
    p = os.path.join(HEADDIR, name + '.pt'); pj = os.path.join(HEADDIR, name + '.json')
    if os.path.exists(p):
        h = build(name).to(dev); h.load_state_dict(torch.load(p)); h.eval()
        acc = json.load(open(pj))['acc']
    else:
        h, acc = train(name)
        torch.save(h.state_dict(), p); json.dump({'acc': acc}, open(pj, 'w'))
    if double:
        h = h.double()
    return h, acc


# ------------------------------------------------------------ probe-1 sample
def probe1_sample():
    """probe1_aggregator.main: 48 images, bank of 32, rs=RandomState(0)."""
    Fva, Yva = feats('val')
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:96]
    return np.sort(sel[:48]), np.sort(sel[48:80]), sel


def probe1_refs(nimg=48, nref=3, T=49, npair=256, nsub=64, sub=24):
    """Bank indices drawn exactly as probe1 draws them: rs2=RandomState(1),
    one randint per (image, ref) followed by the RNG consumption of
    probe1.interaction (randint pairs, then nsub choices)."""
    rs2 = np.random.RandomState(1); out = []
    for k in range(nimg):
        row = []
        for b in range(nref):
            row.append(rs2.randint(32))
            rs2.randint(0, T, (npair, 2))
            for _ in range(nsub):
                rs2.choice(T, sub, replace=False)
        out.append(row)
    return out


def probe7_refs(nimg=32, T=49):
    """probe7_andor: images sel[:32], one ref per image, same RNG consumption."""
    return [r[0] for r in probe1_refs(nimg=nimg, nref=1, T=T)]


# ------------------------------------------------------------ estimators
@torch.no_grad()
def region_ns(f, A, B, masks, ch=256):
    """f: (M,C,T)->(M,) scalar readout. masks (R,T) bool. Returns nec, suf (R,)."""
    R = masks.shape[0]; C = A.shape[0]
    fA = f(A[None])[0]; fB = f(B[None])[0]
    nec, suf = [], []
    for s in range(0, R, ch):
        mk = masks[s:s + ch][:, None, :].expand(-1, C, -1)
        XA = torch.where(mk, B[None], A[None])     # A with region <- B
        XB = torch.where(mk, A[None], B[None])     # B with region <- A
        nec.append(fA - f(XA)); suf.append(f(XB) - fB)
    return torch.cat(nec), torch.cat(suf)


def sep_stats(n, s):
    n = np.asarray(n, np.float64); s = np.asarray(s, np.float64)
    from scipy.stats import spearmanr
    ok = n.std() > 0 and s.std() > 0
    corr = float(np.corrcoef(n, s)[0, 1]) if ok else float('nan')
    rho = float(spearmanr(n, s).statistic) if ok else float('nan')
    sc = max(np.abs(n).max(), np.abs(s).max())
    eps1 = float(np.abs(n - s).max() / sc) if sc > 0 else 0.0
    return corr, rho, eps1


def cell_masks(T=49):
    return torch.eye(T, dtype=torch.bool, device=dev)


def window_masks(G=7, K=3):
    ms = []
    for i in range(G - K + 1):
        for j in range(G - K + 1):
            m = torch.zeros(G, G, dtype=torch.bool, device=dev)
            m[i:i + K, j:j + K] = True
            ms.append(m.flatten())
    return torch.stack(ms)


def half_masks(T=49, n=64, seed=0):
    rs = np.random.RandomState(seed)
    ms = torch.zeros(n, T, dtype=torch.bool, device=dev)
    for i in range(n):
        ms[i, torch.from_numpy(rs.choice(T, T // 2, replace=False)).to(dev)] = True
    return ms


def tci(v, conf=0.95):
    """Student-t interval for the mean of a small sample."""
    from scipy import stats
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.5 + conf / 2, n - 1)
    return float(m), float(m - q * se), float(m + q * se), float(se), int(n)


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=float)
    print('wrote', p, flush=True)
