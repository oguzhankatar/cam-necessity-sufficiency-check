"""C2 shared helpers.  Reuses scripts/medical (read-only, by import) and adds
image caches, plain CAM, the average-box baseline and paired tests.

Nothing here writes under scripts/ or outputs/.  mlib's import side effect is
only os.makedirs on directories that already exist / a temp cache dir.
"""
import os
import sys
import json
import tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *['..'] * 3))
MED = os.path.join(ROOT, 'scripts', 'medical')
sys.path.insert(0, MED)
import mlib                                                    # noqa: E402
from mlib import dev                                           # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, '..', 'out'))
CACHE = os.environ.get('C2CACHE', os.path.join(tempfile.gettempdir(), 'nscam_cache', 'c2cache'))
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
MOUT = mlib.OUT                     # outputs/medical, read only


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=float)
    print('wrote', p, flush=True)
    return p


def load_json(name):
    return json.load(open(os.path.join(MOUT, name)))


# ------------------------------------------------------------ image caches
def _pil224(path):
    """the uint8 image both mlib.load_gray and m1.xrv_load start from."""
    from PIL import Image
    im = Image.open(path).convert('L').resize((224, 224), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _dcm224(path):
    """the uint8 image m11.load_dcm starts from."""
    import pydicom
    from PIL import Image
    a = pydicom.dcmread(path).pixel_array.astype(np.float32)
    im = Image.fromarray(a.astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def u8_cache(paths, tag, kind='png'):
    fp = os.path.join(CACHE, 'u8_%s.npy' % tag)
    if os.path.exists(fp):
        return np.load(fp)
    from concurrent.futures import ThreadPoolExecutor
    f = _pil224 if kind == 'png' else _dcm224
    with ThreadPoolExecutor(12) as ex:
        A = np.stack(list(ex.map(f, paths)))
    np.save(fp, A)
    return A


def u8_to_xrv(a):
    """m1.xrv_load / m11.load_dcm tensor from the cached uint8."""
    v = a.astype(np.float32)
    return torch.from_numpy((2 * (v / 255.) - 1.) * 1024.)[:, None]


IM_M = torch.tensor(mlib.IMNET_MEAN).view(1, 3, 1, 1)
IM_S = torch.tensor(mlib.IMNET_STD).view(1, 3, 1, 1)


def u8_to_imnet(a):
    """mlib.load_gray tensor (3 identical channels, ImageNet norm) from uint8."""
    v = torch.from_numpy(a.astype(np.float32) / 255.)[:, None].repeat(1, 3, 1, 1)
    return (v - IM_M) / IM_S


@torch.no_grad()
def feats(body, X, bs=64):
    """X: callable(i0,i1)->tensor batch or a tensor.  Returns (n,T,C) float32."""
    out = []
    n = X.shape[0] if torch.is_tensor(X) else X.n
    for i in range(0, n, bs):
        x = X[i:i + bs] if torch.is_tensor(X) else X(i, min(n, i + bs))
        f = body(x.to(dev))
        out.append(f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2).float().cpu().numpy())
    return np.concatenate(out)


class Lazy:
    def __init__(s, A, fn):
        s.A, s.fn, s.n = A, fn, len(A)

    def __call__(s, i0, i1):
        return s.fn(s.A[i0:i1])


# ------------------------------------------------------------ maps
def plain_cam(W, XA, c):
    """per-cell class logit w_c . a_t (CAM without reference subtraction).
    The bias is constant over cells and cannot move the argmax."""
    return (XA @ W[c]).detach().cpu().numpy().astype(np.float64)


def box_mask_224(boxes, orig=1024.):
    """union of (x,y,w,h) boxes, same rounding as m4.gt_mask / m11.gt_mask."""
    g = np.zeros((224, 224), np.uint8)
    s = 224 / orig
    for (x, y, w, h) in boxes:
        x0, y0 = int(round(x * s)), int(round(y * s))
        x1, y1 = int(round((x + w) * s)), int(round((y + h) * s))
        g[max(0, y0):min(224, y1), max(0, x0):min(224, x1)] = 1
    return g


def hit_full(mapfull, gt):
    """pointing hit for a full-resolution (224x224) map: argmax inside GT."""
    return float(gt.reshape(-1)[int(np.argmax(mapfull))] > 0)


# ------------------------------------------------------------ stats
def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    p = float(binomtest(n10, n, 0.5).pvalue) if n else 1.0
    return dict(a_only=n10, b_only=n01, p_mcnemar=p)


def cmp(a, b, binary):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = mlib.paired(a, b)
    if binary:
        d.update(mcnemar(a, b))
    d.update(mean_a=float(np.nanmean(a)), mean_b=float(np.nanmean(b)))
    return d


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    out = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, run)
    return out


# ------------------------------------------------------------ the map loop
OFF = ['lse8.0', 'lse-8.0']


def make_heads(W, b):
    heads = {'gap': mlib.head_of('gap', W, b).double().to(dev)}
    for k in OFF:
        heads[k] = mlib.head_of(k, W, b).double().to(dev)
    return heads


def unit_masks(T):
    import m3_maps as m3
    G = int(round(T ** .5))
    return (m3.unit_mask([np.array([t]) for t in range(T)], T, dev),
            m3.unit_mask(mlib.blocks_of(G, 3), T, dev))


def maps_for(heads, XA, Xb, c, u1, u3, W, rmap):
    """exactly the per-item loop of m3/m6/m10/m11 (same map names), plus
    'cam' = plain CAM (per-cell class logit, no reference subtraction)."""
    import m3_maps as m3
    G = int(round(XA.shape[0] ** .5))
    mp = {}
    for hk, h in heads.items():
        for uname, um in (('1x1', u1), ('3x3', u3)):
            Ns, Ss = [], []
            for XB in Xb:
                N, S = m3.NS_batched(h, XA, XB, c, um)
                Ns.append(N)
                Ss.append(S)
            if hk == 'gap':
                mp['collapsed_%s' % uname] = np.mean(Ns, 0)
            else:
                mp['N_%s_%s' % (hk, uname)] = np.mean(Ns, 0)
                mp['S_%s_%s' % (hk, uname)] = np.mean(Ss, 0)
        mp['gradcam_%s' % hk] = mlib.gradcam(h, XA, c)
    mp['actnorm'] = mlib.actnorm(XA)
    mp['center'] = mlib.center_prior(G)
    mp['random'] = rmap.rand(XA.shape[0])
    mp['cam'] = plain_cam(W.to(XA.device).double(), XA, c)
    return mp


def save_maps(name, recs, meta):
    """recs: list of dicts name->(T,) ; meta: dict of lists."""
    keys = sorted(recs[0].keys())
    arr = {k: np.stack([r[k] for r in recs]).astype(np.float64) for k in keys}
    np.savez_compressed(os.path.join(OUT, name), **arr,
                        __meta__=np.array(json.dumps(meta)))
    print('wrote maps', name, len(recs), flush=True)


def load_maps(name):
    z = np.load(os.path.join(OUT, name))
    meta = json.loads(str(z['__meta__']))
    return {k: z[k] for k in z.files if k != '__meta__'}, meta
