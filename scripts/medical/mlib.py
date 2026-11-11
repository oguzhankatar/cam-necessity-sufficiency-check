"""Stage 7-B shared machinery: medical datasets with pixel/box ground truth,
standard classifiers, and the necessity / sufficiency maps scored against the
radiologist / dermatologist annotation.

Conventions kept from earlier rounds (and from the binding method rules):
  * every certificate of additivity is `scripts/crosscheck/rtlib.py`'s LITERAL
    float64 per-cell loop (maxNS, Rhalf, I2 < 1e-9).  `corrNS` is recorded and
    never used as a criterion.
  * the scope verdict is `scripts/heldout/frozen_rule.py`'s frozen rule, run on
    SHIPPED / TRAINED weights and real images, never at random init.
  * baselines are BANK-SAMPLED reference images, never a per-image summary whose
    pooled vector equals the image's.
  * two perturbation units are always reported: 1x1 cells and 3x3 blocks.
"""
import os
import tempfile
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'crosscheck'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'heldout'))
import rtlib                                                     # noqa: E402
import frozen_rule                                               # noqa: E402

OUT = os.path.join(ROOT, 'outputs', 'medical')
SCRATCH = os.environ.get(
    'MEDCACHE', os.path.join(tempfile.gettempdir(), 'nscam_cache', 'medcache'))
os.makedirs(OUT, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

NIH_FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                'Pleural_Thickening', 'Hernia']
# the 8 findings for which NIH ships radiologist-drawn boxes
NIH_BOX8 = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
BOX2LABEL = {'Infiltrate': 'Infiltration'}


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=float)
    print('wrote', p, flush=True)
    return p


# --------------------------------------------------------------- datasets
def nih_frame():
    """(names, Y[n,14], meta) for the images actually on disk."""
    import csv
    D = os.path.join(ROOT, 'data', 'nih')
    have = set(os.listdir(os.path.join(D, 'images')))
    lab = {}
    for r in csv.DictReader(open(os.path.join(D, 'Data_Entry_2017_v2020.csv'))):
        lab[r['Image Index']] = r['Finding Labels'].split('|')
    tr = set(l.strip() for l in open(os.path.join(D, 'split_train.txt')) if l.strip())
    te = set(l.strip() for l in open(os.path.join(D, 'split_test.txt')) if l.strip())
    names = sorted(have & (tr | te))
    Y = np.zeros((len(names), len(NIH_FINDINGS)), dtype=np.float32)
    for i, n in enumerate(names):
        for f in lab[n]:
            if f in NIH_FINDINGS:
                Y[i, NIH_FINDINGS.index(f)] = 1
    split = np.array(['train' if n in tr else 'test' for n in names])
    return names, Y, split


def nih_boxes():
    """{image: [(label, x, y, w, h)]} in the 1024x1024 image frame."""
    import csv
    D = os.path.join(ROOT, 'data', 'nih')
    out = {}
    with open(os.path.join(D, 'BBox_List_2017.csv')) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0].strip():
                continue
            lab = row[1].strip()
            lab = BOX2LABEL.get(lab, lab)
            out.setdefault(row[0].strip(), []).append(
                (lab, float(row[2]), float(row[3]), float(row[4]), float(row[5])))
    return out


def isic_frame():
    """(names, y[n] in 0..2, ) for ISIC-2018 Task1 images with 2017 labels."""
    import csv
    D = os.path.join(ROOT, 'data', 'isic2018')
    lab = {}
    for f in ('ISIC-2017_Training_Part3_GroundTruth.csv',
              'ISIC-2017_Validation_Part3_GroundTruth.csv',
              'ISIC-2017_Test_v2_Part3_GroundTruth.csv'):
        p = os.path.join(D, f)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            mel, sk = float(r['melanoma']), float(r['seborrheic_keratosis'])
            lab[r['image_id']] = 1 if mel else (2 if sk else 0)
    img = os.path.join(D, 'images')
    msk = os.path.join(D, 'masks')
    have = set(n[:-4] for n in os.listdir(img)) if os.path.isdir(img) else set()
    hm = set(n.replace('_segmentation.png', '')
             for n in os.listdir(msk)) if os.path.isdir(msk) else set()
    names = sorted(have & hm & set(lab))
    y = np.array([lab[n] for n in names], dtype=np.int64)
    return names, y


# ---------------------------------------------------------------- loading
def load_gray(path, size=224):
    from PIL import Image
    im = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.
    a = np.stack([a, a, a], 0)
    for c in range(3):
        a[c] = (a[c] - IMNET_MEAN[c]) / IMNET_STD[c]
    return torch.from_numpy(a)


def load_rgb(path, size=224):
    from PIL import Image
    im = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.
    for c in range(3):
        a[c] = (a[c] - IMNET_MEAN[c]) / IMNET_STD[c]
    return torch.from_numpy(a)


class ImgDS(torch.utils.data.Dataset):
    def __init__(s, paths, Y, gray=True, size=224, aug=False):
        s.p, s.Y, s.gray, s.size, s.aug = paths, Y, gray, size, aug

    def __len__(s):
        return len(s.p)

    def __getitem__(s, i):
        x = (load_gray if s.gray else load_rgb)(s.p[i], s.size)
        if s.aug:
            if np.random.rand() < .5:
                x = torch.flip(x, [2])
        return x, torch.from_numpy(np.asarray(s.Y[i]))


# ----------------------------------------------------------------- models
def densenet121(nclass, pretrained=True):
    import timm
    m = timm.create_model('densenet121', pretrained=pretrained, num_classes=nclass)
    return m


def resnet50(nclass, pretrained=True):
    import timm
    return timm.create_model('resnet50', pretrained=pretrained, num_classes=nclass)


class Body(nn.Module):
    """image -> (B, C, H, W) feature map, exactly the model's own."""

    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, x):
        return s.m.forward_features(x)


# ------------------------------------------------------------------ heads
# every head maps (B, T, C) -> (B, K), which is rtlib's contract.
class GAP(nn.Module):
    """the degenerate point: fc(mean_t a_t)"""

    def __init__(s, W, b):
        super().__init__()
        s.W = nn.Parameter(W.clone(), requires_grad=False)
        s.b = nn.Parameter(b.clone(), requires_grad=False)

    def forward(s, X):
        return X.mean(1) @ s.W.t() + s.b


class LSE(nn.Module):
    """LSE_T over per-cell logits.  T -> 0 is algebraically GAP + fc."""

    def __init__(s, W, b, T):
        super().__init__()
        s.W = nn.Parameter(W.clone(), requires_grad=False)
        s.b = nn.Parameter(b.clone(), requires_grad=False)
        s.T = float(T)

    def forward(s, X):
        z = X @ s.W.t() + s.b                                   # (B,T,K)
        if s.T == 0:
            return z.mean(1)
        n = z.shape[1]
        return (torch.logsumexp(s.T * z, 1)
                - torch.log(torch.tensor(float(n), dtype=z.dtype,
                                         device=z.device))) / s.T


def head_of(kind, W, b):
    if kind == 'gap':
        return GAP(W, b)
    if kind.startswith('lse'):
        return LSE(W, b, float(kind[3:]))
    raise ValueError(kind)


# ------------------------------------------------- necessity / sufficiency
def blocks_of(G, K):
    """list of index arrays: the KxK block of cells centred on each cell."""
    out = []
    h = K // 2
    for t in range(G * G):
        r, c = t // G, t % G
        idx = [rr * G + cc
               for rr in range(max(0, r - h), min(G, r + h + 1))
               for cc in range(max(0, c - h), min(G, c + h + 1))]
        out.append(np.array(idx, dtype=np.int64))
    return out


@torch.no_grad()
def NS_unit(head, XA, XB, c, units):
    """LITERAL loop, one perturbation per unit (a unit is a list of cells).

    N(u) = z(A) - z(A with cells u <- B);  S(u) = z(B with cells u <- A) - z(B).
    Identical in definition to rtlib.NS_literal, generalised to blocks.
    """
    zA = float(head(XA[None])[0, c])
    zB = float(head(XB[None])[0, c])
    N = np.empty(len(units), dtype=np.float64)
    S = np.empty(len(units), dtype=np.float64)
    for i, u in enumerate(units):
        Xn = XA.clone()
        Xn[u] = XB[u]
        Xs = XB.clone()
        Xs[u] = XA[u]
        N[i] = zA - float(head(Xn[None])[0, c])
        S[i] = float(head(Xs[None])[0, c]) - zB
    return N, S


def certify(head, XA, XB, c, rs):
    """rtlib's own literal float64 certificate.  Returns the dict + verdict."""
    d = rtlib.report('cert', head, XA, XB, c, rs, do_add=True)
    d['maxNS'] = d['max_abs_diff_rel']
    d['additive'] = bool(d['maxNS'] < 1e-9 and d['Rhalf'] < 1e-9 and d['I2'] < 1e-9)
    return d


# -------------------------------------------------------------- other maps
def gradcam(head, XA, c):
    """Grad-CAM at the aggregation site: relu(sum_c mean_t(dz/da) * a)."""
    X = XA[None].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        g, = torch.autograd.grad(head(X)[0, c], X)
    w = g[0].mean(0, keepdim=True)                              # (1,C)
    return Fn.relu((w * XA).sum(1)).detach().cpu().numpy().astype(np.float64)


def actnorm(XA):
    return XA.norm(dim=1).detach().cpu().numpy().astype(np.float64)


def center_prior(G):
    r = np.arange(G) - (G - 1) / 2.
    d2 = r[:, None] ** 2 + r[None, :] ** 2
    return (-d2).reshape(-1).astype(np.float64)                  # peak at centre


# ------------------------------------------------------------- map scoring
def upsample(map_flat, G, size):
    m = torch.from_numpy(np.asarray(map_flat, dtype=np.float32)).reshape(1, 1, G, G)
    return Fn.interpolate(m, size=(size, size), mode='bilinear',
                          align_corners=False)[0, 0].numpy()


def pointing_hit(map_flat, G, gt, size):
    """argmax of the upsampled map inside the GT mask (the pointing game)."""
    u = upsample(map_flat, G, size)
    i = int(np.argmax(u))
    return float(gt.reshape(-1)[i] > 0)


def pixel_ap(map_flat, G, gt, size):
    """threshold-free: average precision of the map ranking GT pixels."""
    from sklearn.metrics import average_precision_score
    u = upsample(map_flat, G, size).reshape(-1)
    g = (gt.reshape(-1) > 0).astype(np.int8)
    if g.sum() == 0 or g.sum() == len(g):
        return np.nan
    return float(average_precision_score(g, u))


def iou_at(map_flat, G, gt, size, q):
    """IoU of the top-q fraction of the upsampled map against GT."""
    u = upsample(map_flat, G, size).reshape(-1)
    g = gt.reshape(-1) > 0
    k = max(1, int(round(q * len(u))))
    thr = np.partition(u, -k)[-k]
    p = u >= thr
    inter = float((p & g).sum())
    union = float((p | g).sum())
    return inter / union if union > 0 else np.nan


def dice_at(map_flat, G, gt, size, q):
    u = upsample(map_flat, G, size).reshape(-1)
    g = gt.reshape(-1) > 0
    k = max(1, int(round(q * len(u))))
    thr = np.partition(u, -k)[-k]
    p = u >= thr
    s = float(p.sum() + g.sum())
    return 2 * float((p & g).sum()) / s if s > 0 else np.nan


# ----------------------------------------------------------- stats helpers
def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else np.nan, lo=np.nan, hi=np.nan, n=len(v))
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(v), (n, len(v)))
    b = v[idx].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)),
                hi=float(np.percentile(b, 97.5)), n=int(len(v)))


def paired(a, b, n=10000, seed=0):
    """paired difference a - b: bootstrap CI + Wilcoxon signed rank."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    d = a - b
    ci = boot_ci(d, n, seed)
    try:
        w = wilcoxon(a, b, zero_method='wilcox')
        p = float(w.pvalue)
    except Exception:
        p = np.nan
    sd = d.std(ddof=1)
    return dict(diff=ci['mean'], lo=ci['lo'], hi=ci['hi'], n=int(len(d)),
                p_wilcoxon=p, cohen_dz=float(d.mean() / sd) if sd > 0 else np.nan)
