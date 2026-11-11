"""Stage-2 shared machinery: generic backbone/dataset feature caching plus the
EXACT finite-difference estimators from stage 1, verbatim in definition.

Everything here is deliberately a re-implementation of the stage-1 estimators
with the hard-wired (2048, 7, 7) / imagenette assumptions removed, so that the
AND/OR fan can be re-measured on a different BODY and a different DATASET with
*identical* estimator code.

Estimators (all exact finite differences over SPATIAL CELLS -- the basis CAM
actually perturbs; see stage-1 Probe 8 on basis dependence):
  D(t)   = z(A) - z(A with cell t -> baseline B)            necessity
  S(t)   = z(B with cell t <- A) - z(B)                     sufficiency
  I2     = E|D(t1,t2) - D(t1) - D(t2)| / E|D(t1,t2)|        pairwise
  Rhalf  = E|D(S) - sum_{t in S} D(t)| / E|D(S)|, |S|=T/2   half-map scale
  corrNS = Pearson corr over cells of D vs S
Additive readout  <=>  I2 = Rhalf = 0 and corrNS = +1, exactly.
Baselines are always BANK-SAMPLED (never a per-image channel mean: that has the
same GAP as the image and makes every GAP-head difference exactly zero).
"""
import os, sys, json, tempfile, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(__file__))
from heads import build, LSE                                    # noqa: F401

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'outputs')
CACHE = os.environ.get(
    'R2CACHE', os.path.join(tempfile.gettempdir(), 'nscam_cache', 'r2cache'))
dev = 'cuda'
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ bodies
def get_body(name):
    """Returns (nn.Module mapping image -> (B,C,H,W), channels, input size)."""
    from torchvision import models
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                             m.layer1, m.layer2, m.layer3, m.layer4)
        return body.eval(), 2048, 224
    if name == 'vgg16':
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        return m.features.eval(), 512, 224          # (512,7,7) after last pool
    if name == 'convnext_t':
        import timm
        m = timm.create_model('convnext_tiny', pretrained=True)

        class FF(nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x): return s.m.forward_features(x)
        return FF(m).eval(), 768, 224
    if name.startswith('vitL'):
        # ViT-B/16 truncated after block L: the 196 PATCH tokens are returned as
        # a (B,768,14,14) map, so every downstream estimator sees the same
        # (B,C,H,W) contract as a convnet.  The CLS token is dropped -- the point
        # of the exercise is that a ViT probe is well posed exactly where an
        # additive aggregator over patch tokens is the readout.
        import timm
        Lb = int(name[4:])
        m = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k',
                              pretrained=True)

        class VT(nn.Module):
            def __init__(s, m, Lb):
                super().__init__(); s.m = m; s.Lb = Lb

            def forward(s, x):
                z = s.m.patch_embed(x)
                z = s.m._pos_embed(z)
                z = s.m.norm_pre(z)
                for b in s.m.blocks[:s.Lb + 1]:
                    z = b(z)
                z = s.m.norm(z)
                np_ = s.m.patch_embed.grid_size
                t = z[:, s.m.num_prefix_tokens:]
                return t.transpose(1, 2).reshape(z.shape[0], -1, *np_)
        return VT(m, Lb).eval(), 768, 224
    if name == 'swin_t':
        import timm
        m = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k',
                              pretrained=True)

        class SW(nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x):
                z = s.m.forward_features(x)          # (B,7,7,768) channels-last
                return z.permute(0, 3, 1, 2).contiguous()
        return SW(m).eval(), 768, 224
    raise ValueError(name)


# ---------------------------------------------------------------- datasets
def get_dataset(name, split, size):
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if name in ('imagenette', 'imagewoof'):
        d = {'imagenette': 'imagenette2-160', 'imagewoof': 'imagewoof2-160'}[name]
        p = os.path.join(ROOT, 'data', d, split)
        base = transforms.Compose([
            transforms.Resize(160), transforms.CenterCrop(160),
            transforms.Resize(size), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
        return datasets.ImageFolder(p, base), 10
    if name == 'cifar100':
        return datasets.CIFAR100(os.path.join(ROOT, 'data'),
                                 train=(split == 'train'), transform=tf), 100
    if name.startswith('in1k'):
        # Real ILSVRC-2012 validation images (50 000, 1000 classes), fetched by
        # scripts/get_in1k.py.  File <cls>/<i>.jpg with i in 0..49; i < NTR goes
        # to the head-training split and the rest to the evaluation split, so the
        # two are disjoint by construction.  `in1kNNN` restricts to the first NNN
        # classes.
        from torch.utils.data import Subset
        p = os.path.join(ROOT, 'data', 'in1k', 'val')
        tf2 = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(size),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        base = datasets.ImageFolder(p, tf2)
        ncls = int(name[4:]) if len(name) > 4 else 1000
        ntr = int(os.environ.get('IN1K_NTR', 40))
        keep = [i for i, (f, y) in enumerate(base.samples)
                if y < ncls and ((int(os.path.basename(f)[:5]) < ntr)
                                 == (split == 'train'))]
        return Subset(base, keep), ncls
    raise ValueError(name)


def cache_feats(body_name, ds_name, split, size=None, max_per_class=None, bs=24,
                workers=0):
    """Cache (N,C,H,W) fp16 feature maps. Returns (path_F, path_Y, C, T)."""
    from torch.utils.data import DataLoader, Subset
    os.makedirs(CACHE, exist_ok=True)
    tag = f'{body_name}_{ds_name}{"" if max_per_class is None else f"_m{max_per_class}"}'
    if size is not None:
        tag += f'_s{size}'
    # reuse stage 1's cache verbatim for the one setting it already covers
    if (body_name, ds_name, size, max_per_class) == ('resnet50', 'imagenette',
                                                     None, None):
        fp = os.path.join(OUT, f'feat_{split}.npy')
        yp = os.path.join(OUT, f'lab_{split}.npy')
        if os.path.exists(fp) and os.path.exists(yp):
            F = np.load(fp, mmap_mode='r')
            return fp, yp, F.shape[1], F.shape[2] * F.shape[3]
    fp = os.path.join(CACHE, f'F_{tag}_{split}.npy')
    yp = os.path.join(CACHE, f'Y_{tag}_{split}.npy')
    if os.path.exists(fp) and os.path.exists(yp):
        F = np.load(fp, mmap_mode='r')
        return fp, yp, F.shape[1], F.shape[2] * F.shape[3]
    body, C, sz = get_body(body_name)
    if size is None:
        size = sz
    body = body.to(dev)
    ds, _ = get_dataset(ds_name, split, size)
    if max_per_class is not None:
        lab = np.array(ds.targets)
        keep = np.concatenate([np.where(lab == k)[0][:max_per_class]
                               for k in np.unique(lab)])
        keep.sort(); ds = Subset(ds, keep.tolist())
    ld = DataLoader(ds, batch_size=bs, num_workers=workers)
    Fl, Yl = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(ld):
            Fl.append(body(x.to(dev)).half().cpu().numpy()); Yl.append(y.numpy())
            if i % 100 == 0:
                print(f'  cache {tag} {split} {i*bs}', flush=True)
    F = np.concatenate(Fl); Y = np.concatenate(Yl)
    np.save(fp, F); np.save(yp, Y)
    del body; torch.cuda.empty_cache()
    print(f'  cached {tag} {split} {F.shape}', flush=True)
    return fp, yp, F.shape[1], F.shape[2] * F.shape[3]


def load_cached(fp, yp):
    return np.load(fp, mmap_mode='r'), np.load(yp)


def batch(F, idx):
    a = torch.from_numpy(np.ascontiguousarray(F[idx])).to(dev).float()
    return a.reshape(a.shape[0], a.shape[1], -1)                 # (B,C,T)


# ------------------------------------------------------------------ training
def train_head(name, Ftr, Ytr, Fva, Yva, nclass, C, epochs=8, bs=96, seed=0,
               lr=1e-3):
    torch.manual_seed(seed)
    h = build(name, k=nclass, c=C).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Ytr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, max(1, epochs * (n // bs)))
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
        ok, P, L = 0, [], []
        for i in range(0, len(Yva), 128):
            j = np.arange(i, min(i + 128, len(Yva)))
            z = h(batch(Fva, j))
            ok += (z.argmax(1).cpu().numpy() == Yva[j]).sum()
            P.append(torch.softmax(z, 1).cpu().numpy()); L.append(z.cpu().numpy())
    return h, float(ok / len(Yva)), np.concatenate(P), np.concatenate(L)


# ------------------------------------------------------------- exact metrics
@torch.no_grad()
def _rep(h, base, other, mk, c, ch=128):
    """Replace cells where mk is True in `base` by the same cells of `other`."""
    M = mk.shape[0]
    X = base[None].expand(M, *base.shape).clone()
    m = mk[:, None, :].expand(-1, base.shape[0], -1)
    X[m] = other[None].expand(M, *base.shape)[m]
    return torch.cat([h(X[s:s + ch])[:, c] for s in range(0, M, ch)])


@torch.no_grad()
def maps(h, A, B, c):
    """Exact necessity and sufficiency over spatial cells."""
    T = A.shape[1]
    eye = torch.eye(T, dtype=torch.bool, device=A.device)
    N = h(A[None])[0, c] - _rep(h, A, B, eye, c)
    S = _rep(h, B, A, eye, c) - h(B[None])[0, c]
    return N, S


@torch.no_grad()
def block_masks(G, K=3, device='cuda'):
    """(G*G, G*G) bool: row t is the KxK block of cells centred on cell t."""
    T = G * G
    m = torch.zeros(T, T, dtype=torch.bool, device=device)
    h_ = K // 2
    for t in range(T):
        r, c = t // G, t % G
        for dr in range(-h_, h_ + 1):
            for dc in range(-h_, h_ + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < G and 0 <= cc < G:
                    m[t, rr * G + cc] = True
    return m


@torch.no_grad()
def maps_block(h, A, B, c, G, K=3):
    """Necessity / sufficiency of a KxK BLOCK of cells, attributed to its centre.

    Single-cell perturbation understates the importance of a feature that is
    represented redundantly across neighbouring cells (a convnet's receptive
    fields overlap, so one object occupies several cells and no single one of
    them is necessary).  The perturbation unit has to be at least as large as
    the redundancy scale; this is the same scale-dependence stage 1 found
    between I2 and Rhalf, at the level of a single object.
    """
    mk = block_masks(G, K, A.device)
    N = h(A[None])[0, c] - _rep(h, A, B, mk, c)
    S = _rep(h, B, A, mk, c) - h(B[None])[0, c]
    return N, S


@torch.no_grad()
def interaction(h, A, B, c, rs, npair=256, nsub=64):
    """Exact I2, Rhalf, corrNS.  Returns also mean |N| for the vacuity check."""
    T = A.shape[1]
    zA = h(A[None])[0, c]
    eye = torch.eye(T, dtype=torch.bool, device=A.device)
    D1 = zA - _rep(h, A, B, eye, c)
    S1 = _rep(h, B, A, eye, c) - h(B[None])[0, c]
    n_, s_ = D1.cpu().numpy(), S1.cpu().numpy()
    corr = (float(np.corrcoef(n_, s_)[0, 1])
            if n_.std() > 1e-12 and s_.std() > 1e-12 else np.nan)

    p = rs.randint(0, T, (npair, 2)); p = p[p[:, 0] != p[:, 1]]
    mk = torch.zeros(len(p), T, dtype=torch.bool, device=A.device)
    ar = torch.arange(len(p))
    mk[ar, torch.from_numpy(p[:, 0]).to(A.device)] = True
    mk[ar, torch.from_numpy(p[:, 1]).to(A.device)] = True
    D2 = zA - _rep(h, A, B, mk, c)
    add = D1[torch.from_numpy(p[:, 0]).to(A.device)] + \
        D1[torch.from_numpy(p[:, 1]).to(A.device)]
    I2 = float((D2 - add).abs().mean() / (D2.abs().mean() + 1e-12))

    ms = torch.zeros(nsub, T, dtype=torch.bool, device=A.device)
    for i in range(nsub):
        ms[i, torch.from_numpy(rs.choice(T, T // 2, replace=False)).to(A.device)] = True
    DS = zA - _rep(h, A, B, ms, c)
    Rh = float((DS - (ms.float() * D1[None]).sum(1)).abs().mean()
               / (DS.abs().mean() + 1e-12))
    return I2, Rh, corr, float(D1.abs().mean()), float(zA.abs())


def gradcam_head(h, A, c):
    A = A[None].detach().requires_grad_(True)
    with torch.enable_grad():
        g, = torch.autograd.grad(h(A)[0, c], A)
    w = g.mean(dim=2, keepdim=True)
    return Fn.relu((w * A).sum(1))[0].detach()


@torch.no_grad()
def auc(h, A, B, c, order, mode):
    """Feature-space deletion / insertion AUC over cells (softmax prob)."""
    T = A.shape[1]
    rank = torch.argsort(order, descending=True)
    start, fin = (A, B) if mode == 'del' else (B, A)
    cur = start.clone(); ps = [torch.softmax(h(cur[None]), 1)[0, c].item()]
    for i in range(T):
        cur[:, rank[i]] = fin[:, rank[i]]
        ps.append(torch.softmax(h(cur[None]), 1)[0, c].item())
    p = np.array(ps)
    return float((p.sum() - p[0] / 2 - p[-1] / 2) / (len(p) - 1))


# ------------------------------------------------------------- calibration
def ece(P, Y, bins=15):
    conf = P.max(1); pred = P.argmax(1); acc = (pred == Y).astype(float)
    e, n = 0.0, len(Y)
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def auroc(pos, neg):
    """AUROC of score `pos` (in-dist) over `neg` (OOD); higher = better OOD det."""
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)),
                                                        np.zeros(len(neg))])
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    import scipy.stats as st
    r = st.rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, dtype=float); rs = np.random.RandomState(seed)
    if len(v) < 2:
        return float(v.mean()), float(v.mean()), float(v.mean())
    b = np.array([v[rs.randint(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1)
    print('wrote', p, flush=True)
