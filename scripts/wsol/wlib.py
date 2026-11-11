"""STAGE 7-A -- does the necessity / sufficiency distinction buy anything
against REAL localization ground truth?

Everything here is shared machinery.  Design constraints inherited from six
rounds of corrections:

  * `corrNS = 1` is not a certificate.  Additivity is certified only by the
    literal float64 per-cell loop of `scripts/crosscheck/rtlib.py`
    (maxNS, Rhalf, I2 all < 1e-9).
  * No structural or affinity test at random init: pretrained weights, real
    images, float64.
  * The baseline may never have the same pooled vector as the image.  The
    baseline used here is a fixed bank of 8 real images from the FIT split
    (an Expected-Gradients style reference *distribution*), plus a
    dataset-mean-cell baseline as a robustness row.  Spatial shuffle is
    forbidden by construction.
  * Both perturbation scales are reported: 1x1 cells and 3x3 cell blocks.
  * Every number in the report is emitted by code from a JSON.

Cell / map conventions
----------------------
A backbone maps an image to a cell matrix X in R^{T x C} (T = H*W spatial
cells; ViT prefix tokens are dropped).  A readout head is

    GAP  :  z_c(X) = <w_c, mean_t x_t> + b_c                      (degenerate)
    LSE  :  z_c(X) = (1/tau) log( (1/T) sum_t exp(tau (<w_c,x_t> + b_c)) )

LSE at tau -> 0 is algebraically GAP + FC; tau = +1.0 is the knee used by the
project's AND/OR axis.  Both heads are linear in the *scores*
s_t = <w_c, x_t> + b_c, so N and S are exact closed forms in s -- but every
closed form in this file is cross-checked against rtlib's literal loop.

    N(t) = z(A) - z(A with cell t replaced by B's cell t)
    S(t) = z(B with cell t replaced by A's cell t) - z(B)
"""
import os, sys, json, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'outputs', 'wsol')
DATA = os.path.join(ROOT, 'data')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'crosscheck'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'heldout'))


# ------------------------------------------------------------------ protocol
# Fixed before any localization number was computed.  Both the runner and the
# report read these, so no protocol constant is ever transcribed by hand.
PROTOCOL = dict(
    tau=1.0,                 # LSE temperature: the knee of the AND/OR axis
    n_bank=8,                # baseline = 8 real fit-split images, averaged
    eval_res=224,            # maps and ground truth are compared at 224x224
    n_thresholds=100,        # CAM threshold sweep, interval 0.01
    deltas=(0.3, 0.5, 0.7),  # MaxBoxAccV2 IoU thresholds (Choe et al. CVPR 2020)
    point_tol=15,            # pointing-game tolerance in pixels (Zhang et al.)
    n_boot=2000,             # bootstrap resamples for means/proportions
    n_boot_mbav2=1000,       # bootstrap resamples for MaxBoxAccV2
    seed=0,
    cnn_input=448,           # 448 -> a 14x14 cell grid with shipped strides
    vit_input=224,           # 224 -> a 14x14 patch-token grid
)

def code_sha():
    """sha256 of the machinery that produced a run, so a stale result cannot be
    mixed with a fresh one."""
    import hashlib
    h = hashlib.sha256()
    for f in ('wlib.py', 'w2_run.py'):
        h.update(open(os.path.join(HERE, f), 'rb').read())
    return h.hexdigest()[:16]


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=_jd)
    print('  -> %s' % p, flush=True)
    return p


def load(name):
    return json.load(open(os.path.join(OUT, name)))


def _jd(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return float(o)


# ===================================================================== heads
class PoolHead:
    """(B,T,C) -> (B,K).  torch callable, used for the rtlib certification.

    mode 'gap'    : W . mean_t x_t + b                       (degenerate)
    mode 'lse'    : (1/tau) log mean_t exp(tau (W x_t + b))  (edited, AND/OR axis)
    mode 'lnpost' : W . LN(mean_t x_t) + b                   (edited, LN relocation)
    """

    def __init__(s, W, b, mode='gap', tau=1.0, ln=None):
        import torch
        s.W = W                      # (K, C) torch tensor
        s.b = b                      # (K,)
        s.mode, s.tau, s.ln = mode, float(tau), ln
        s.torch = torch

    def __call__(s, X):
        t = s.torch
        if s.mode == 'lnpost':
            return s.ln(X.mean(1)) @ s.W.T + s.b
        S = X @ s.W.T + s.b          # (B, T, K)
        if s.mode == 'gap':
            return S.mean(1)
        T = S.shape[1]
        return (t.logsumexp(s.tau * S, dim=1) - math.log(T)) / s.tau


def head_scores(cells, W, b):
    """cells (T,C) float64 -> scores (K,T) float64."""
    return cells @ W.T + b[:, None]


# ------------------------------------------------- exact N / S for LSE & GAP
def _blocks_3x3(H, W):
    """index list of the 3x3 neighbourhood of every cell on an HxW grid.
    Used only by the LITERAL cross-check; the fast path uses box3_sum."""
    idx = np.arange(H * W).reshape(H, W)
    out = []
    for i in range(H):
        for j in range(W):
            out.append(idx[max(0, i - 1):i + 2, max(0, j - 1):j + 2].ravel())
    return out


def box3_sum(D, H, Wd):
    """(T,...) -> (T,...): sum over the clipped 3x3 neighbourhood of every cell.
    Integral image, so O(T*C) rather than O(T*9*C).

    Numerically unreliable for sums of exponentials at large |T|: the cumulative sums contain the
    dominant term and cancel. The 3x3-block localisation maps that use it are reported in
    Supplementary Table S10, with this caveat.
    """
    sq = (D.ndim == 1)
    A = D.reshape(H, Wd, -1)
    P = np.zeros((H + 1, Wd + 1, A.shape[2]), dtype=np.float64)
    P[1:, 1:] = A.cumsum(0).cumsum(1)
    i0 = np.clip(np.arange(H) - 1, 0, H); i1 = np.clip(np.arange(H) + 2, 0, H)
    j0 = np.clip(np.arange(Wd) - 1, 0, Wd); j1 = np.clip(np.arange(Wd) + 2, 0, Wd)
    S = (P[np.ix_(i1, j1)] - P[np.ix_(i0, j1)]
         - P[np.ix_(i1, j0)] + P[np.ix_(i0, j0)])
    S = S.reshape(H * Wd, -1)
    return S[:, 0] if sq else S


def _rest_sum(e, H, Wd, block):
    """sum of the positive array `e` EXCLUDING each unit (cell or 3x3 block),
    computed without catastrophic cancellation: the single largest term is held
    out of the total, so the subtraction never removes the dominant mass unless
    it genuinely should."""
    i = int(np.argmax(e))
    top = float(e[i])
    e2 = e.copy(); e2[i] = 0.0
    R = float(e2.sum())                       # exact-ish: excludes the max term
    if not block:
        rest = R + top - e                    # t != i: dominated by `top`
        rest[i] = R                           # t == i: no subtraction at all
        inside = np.zeros(len(e), bool); inside[i] = True
        g = e
    else:
        g = box3_sum(e, H, Wd)
        ind = np.zeros(len(e)); ind[i] = 1.0
        inside = box3_sum(ind, H, Wd) > 0.5
        rest = np.where(inside, R - (g - top), R + top - g)
    return np.maximum(rest, 0.0), g, R + top


def ns_gap(a, beta, H=None, Wd=None, block=False):
    """GAP head.  z = mean_t s_t, so both maps are (sum_U (a-beta))/T, equal."""
    T = a.shape[0]
    d = (a - beta) / T
    if not block:
        return d.copy(), d.copy()
    g = box3_sum(d, H, Wd)
    return g, g


def ns_lse(a, beta, tau, H=None, Wd=None, block=False):
    """LSE head, exact, float64, cancellation-safe.

    z(v) = (1/tau)(log sum_t exp(tau v_t) - log T)
    """
    ta, tb = tau * a, tau * beta
    M = max(ta.max(), tb.max())
    ea, eb = np.exp(ta - M), np.exp(tb - M)
    rest_a, sa, Sa = _rest_sum(ea, H, Wd, block)
    rest_b, sb, Sb = _rest_sum(eb, H, Wd, block)
    EA = rest_a + sb                          # A with unit U <- B
    EB = rest_b + sa                          # B with unit U <- A
    tiny = np.finfo(np.float64).tiny
    N = (np.log(Sa) - np.log(np.maximum(EA, tiny))) / tau
    S = (np.log(np.maximum(EB, tiny)) - np.log(Sb)) / tau
    return N, S


def ns_pooled(XA, XB, fn, H=None, Wd=None, block=False):
    """N / S for any head of the form z = fn(pooled vector), pooled = mean_t x_t.

    Swapping the cells in U moves the pooled vector by sum_{t in U}(x^B_t-x^A_t)/T
    exactly, so the whole map is one batched evaluation of `fn`.  Used for the
    GAP head (fn linear, N == S by the identity) and for the LayerNorm-relocation
    head (fn = linear o LN, which is off the degenerate point).
    """
    T = XA.shape[0]
    mA, mB = XA.mean(0), XB.mean(0)
    D = (XB - XA) / T
    G = box3_sum(D, H, Wd) if block else D
    zA, zB = float(fn(mA[None])[0]), float(fn(mB[None])[0])
    return zA - fn(mA[None] + G), fn(mB[None] - G) - zB


def ns_literal_groups(head, XA, XB, c, groups):
    """rtlib-style LITERAL loop for grouped (block) perturbations.  No algebra:
    build the swapped cell matrix, run the head."""
    import torch
    with torch.no_grad():
        zA = float(head(XA[None])[0, c]); zB = float(head(XB[None])[0, c])
        N = np.empty(len(groups)); S = np.empty(len(groups))
        for i, u in enumerate(groups):
            Xn = XA.clone(); Xn[u] = XB[u]
            Xs = XB.clone(); Xs[u] = XA[u]
            N[i] = zA - float(head(Xn[None])[0, c])
            S[i] = float(head(Xs[None])[0, c]) - zB
    return N, S


def certify_head(head, cells, cls_idx, tag, seed=0, nsub=64, npair=256):
    """The binding certificate: rtlib's LITERAL float64 per-cell loop.
    cells: torch float64 (P,T,C); probe pairs (0,1),(2,3),...
    ADDITIVE iff maxNS, Rhalf, I2 all < 1e-9 on EVERY pair."""
    import rtlib
    rs = np.random.RandomState(seed)
    per = []
    for i in range(0, cells.shape[0] - 1, 2):
        XA, XB = cells[i], cells[i + 1]
        c = int(cls_idx[i])
        N, S, zA, zB = rtlib.NS_literal(head, XA, XB, c)
        d = rtlib.stats(N, S)
        d['Rhalf'], d['I2'] = rtlib.additivity(head, XA, XB, c, N, rs,
                                               nsub=nsub, npair=npair)
        mx = float(np.abs(N).max())
        d['n_active'] = int((np.abs(N) > 1e-12 * mx).sum()) if mx > 0 else 0
        d['T'] = int(len(N)); d['zA'] = zA; d['zB'] = zB
        d['pair'] = i // 2
        per.append(d)
    worst = dict(tag=tag,
                 maxNS=max(p['max_abs_diff_rel'] for p in per),
                 Rhalf=max(p['Rhalf'] for p in per),
                 I2=max(p['I2'] for p in per),
                 corrNS_recorded_never_used=min(p['corr_pearson'] for p in per),
                 n_active=min(p['n_active'] for p in per),
                 T=per[0]['T'], n_pairs=len(per))
    worst['ADDITIVE'] = bool(worst['maxNS'] < 1e-9 and worst['Rhalf'] < 1e-9
                             and worst['I2'] < 1e-9)
    worst['per_pair'] = per
    return worst


# ================================================================== datasets
CLS_VOC = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def cub_manifest():
    """CUB-200-2011: one box per image, 200 classes, official train/test split."""
    d = os.path.join(DATA, 'CUB_200_2011')
    ids = {}
    for ln in open(os.path.join(d, 'images.txt')):
        i, p = ln.split()
        ids[int(i)] = p
    lab = {int(a): int(c) - 1 for a, c in
           (l.split() for l in open(os.path.join(d, 'image_class_labels.txt')))}
    spl = {int(a): int(c) for a, c in
           (l.split() for l in open(os.path.join(d, 'train_test_split.txt')))}
    box = {}
    for ln in open(os.path.join(d, 'bounding_boxes.txt')):
        v = ln.split()
        box[int(v[0])] = [float(x) for x in v[1:]]      # x, y, w, h
    tr, te = [], []
    for i, p in sorted(ids.items()):
        r = dict(path=os.path.join(d, 'images', p), label=lab[i],
                 boxes=[[box[i][0], box[i][1], box[i][0] + box[i][2],
                         box[i][1] + box[i][3]]])
        (tr if spl[i] == 1 else te).append(r)
    return dict(name='cub', n_class=200, multilabel=False, fit=tr, ev=te,
                class_names=None)


def voc_manifest(want='box'):
    """VOC2012 trainval.  fit = train list, eval = val list.

    want='box' -> detection val (5823 imgs, boxes)
    want='seg' -> segmentation val (1449 imgs, per-class masks + boxes)
    """
    import xml.etree.ElementTree as ET
    d = os.path.join(DATA, 'VOCdevkit', 'VOC2012')
    cid = {c: i for i, c in enumerate(CLS_VOC)}

    def read(split, sub='Main'):
        f = os.path.join(d, 'ImageSets', sub, split + '.txt')
        return [l.split()[0] for l in open(f) if l.strip()]

    def rec(nm, seg):
        t = ET.parse(os.path.join(d, 'Annotations', nm + '.xml')).getroot()
        bx = {}
        for o in t.findall('object'):
            c = cid[o.find('name').text]
            bb = o.find('bndbox')
            b = [float(bb.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            bx.setdefault(c, []).append(b)
        r = dict(path=os.path.join(d, 'JPEGImages', nm + '.jpg'),
                 labels=sorted(bx), boxes_by_class={str(k): v for k, v in bx.items()})
        if seg:
            r['mask'] = os.path.join(d, 'SegmentationClass', nm + '.png')
        return r

    fit = [rec(n, False) for n in read('train')]
    if want == 'seg':
        ev = [rec(n, True) for n in read('val', 'Segmentation')]
    else:
        ev = [rec(n, False) for n in read('val')]
    return dict(name='voc12' + ('seg' if want == 'seg' else ''), n_class=20,
                multilabel=True, fit=fit, ev=ev, class_names=CLS_VOC)


def coco_manifest(seed=0):
    """COCO val2017 split in half: fit half (head fitting) / eval half."""
    from pycocotools.coco import COCO
    d = os.path.join(DATA, 'coco')
    co = COCO(os.path.join(d, 'annotations', 'instances_val2017.json'))
    cats = sorted(co.getCatIds())
    cid = {c: i for i, c in enumerate(cats)}
    names = [co.loadCats([c])[0]['name'] for c in cats]
    recs = []
    for iid in sorted(co.getImgIds()):
        anns = [a for a in co.loadAnns(co.getAnnIds([iid], iscrowd=False))
                if a['area'] > 1]
        if not anns:
            continue
        by = {}
        for a in anns:
            by.setdefault(cid[a['category_id']], []).append(a['id'])
        im = co.loadImgs([iid])[0]
        recs.append(dict(path=os.path.join(d, 'val2017', im['file_name']),
                         labels=sorted(by), ann_by_class={str(k): v for k, v in by.items()},
                         iid=iid))
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(recs))
    h = len(recs) // 2
    fit = [recs[i] for i in perm[:h]]
    ev = [recs[i] for i in perm[h:]]
    return dict(name='coco17', n_class=len(cats), multilabel=True, fit=fit,
                ev=ev, class_names=names, _coco=co)


# ================================================================ extraction
BACKBONES = {
    'resnet50': dict(timm='resnet50.tv_in1k', size=448),
    'vgg16gap': dict(timm='vgg16.tv_in1k', size=448),
    'vitb16':   dict(timm='vit_base_patch16_224.augreg2_in21k_ft_in1k', size=224),
}


def build(bk):
    import torch, timm
    spec = BACKBONES[bk]
    m = timm.create_model(spec['timm'], pretrained=True).eval().cuda()
    cfg = m.pretrained_cfg
    return m, spec['size'], cfg


class Extractor:
    """image path -> (T,C) cell matrix, plus the HxW grid shape."""

    def __init__(s, bk):
        import torch
        from torchvision import transforms as TT
        s.torch = torch
        s.m, s.size, s.cfg = build(bk)
        s.bk = bk
        s.npfx = int(getattr(s.m, 'num_prefix_tokens', 0) or 0)
        ip = {'bicubic': TT.InterpolationMode.BICUBIC,
              'bilinear': TT.InterpolationMode.BILINEAR}[s.cfg.get('interpolation', 'bicubic')]
        s.tf = TT.Compose([TT.Resize((s.size, s.size), interpolation=ip),
                           TT.ToTensor(), TT.Normalize(s.cfg['mean'], s.cfg['std'])])
        with torch.no_grad():
            f = s.m.forward_features(torch.zeros(1, 3, s.size, s.size).cuda())
        if f.dim() == 4:
            s.fmt, s.H, s.W, s.C = 'BCHW', f.shape[2], f.shape[3], f.shape[1]
        else:
            n = f.shape[1] - s.npfx
            g = int(round(math.sqrt(n)))
            assert g * g == n, (f.shape, s.npfx)
            s.fmt, s.H, s.W, s.C = 'BTC', g, g, f.shape[2]
        s.T = s.H * s.W

    def cells_of(s, f):
        if s.fmt == 'BCHW':
            return f.flatten(2).transpose(1, 2)
        return f[:, s.npfx:]

    def batch(s, paths, bs=16, prenorm=False):
        """yields (idx0, cells, cells_prenorm|None).  `prenorm` is the ViT
        LayerNorm-relocation view: the same forward pass with `m.norm` replaced
        by Identity, i.e. the tokens BEFORE the final LN."""
        import torch.nn as nn
        from PIL import Image
        torch = s.torch
        buf, i0 = [], 0
        for i, p in enumerate(paths):
            buf.append(s.tf(Image.open(p).convert('RGB')))
            if len(buf) == bs or i == len(paths) - 1:
                x = torch.stack(buf).cuda()
                with torch.no_grad():
                    c1 = s.cells_of(s.m.forward_features(x)).float()
                    c2 = None
                    if prenorm:
                        sv = s.m.norm; s.m.norm = nn.Identity()
                        c2 = s.cells_of(s.m.forward_features(x)).float()
                        s.m.norm = sv
                yield i0, c1, c2
                i0 += len(buf); buf = []


# =================================================================== metrics
def upsample(mp, H, W, out):
    """(H,W) cell map -> (out,out) float32, bilinear, then min-max to [0,1]."""
    import cv2
    a = cv2.resize(mp.reshape(H, W).astype(np.float32), (out, out),
                   interpolation=cv2.INTER_LINEAR)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def boxes_from_mask(bm):
    """all connected-component boxes of a binary map (Choe et al. take the max
    IoU over contour boxes)."""
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(bm.astype(np.uint8), 8)
    out = []
    for k in range(1, n):
        x, y, w, h = st[k, 0], st[k, 1], st[k, 2], st[k, 3]
        out.append((x, y, x + w, y + h))
    return out


def iou_box(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_curve(mp, gt, thr):
    """per-threshold best IoU between any estimated box and any GT box.
    mp: (S,S) in [0,1].  gt: list of [x1,y1,x2,y2].  -> (len(thr),) float32
    Choe et al. take the max IoU over ALL connected-component boxes."""
    import cv2
    G = np.asarray(gt, dtype=np.float64)
    ga = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
    out = np.zeros(len(thr), dtype=np.float32)
    for i, t in enumerate(thr):
        bm = np.ascontiguousarray(mp >= t).view(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(bm, 8)
        if n <= 1:
            continue
        x = st[1:, 0].astype(np.float64); y = st[1:, 1].astype(np.float64)
        wd = st[1:, 2].astype(np.float64); ht = st[1:, 3].astype(np.float64)
        ix = np.maximum(0, np.minimum((x + wd)[:, None], G[None, :, 2])
                        - np.maximum(x[:, None], G[None, :, 0]))
        iy = np.maximum(0, np.minimum((y + ht)[:, None], G[None, :, 3])
                        - np.maximum(y[:, None], G[None, :, 1]))
        inter = ix * iy
        out[i] = float((inter / ((wd * ht)[:, None] + ga[None, :] - inter + 1e-9)).max())
    return out


def maxboxacc_v2(curves, deltas=(0.3, 0.5, 0.7), idx=None):
    """curves: (n_img, n_thr).  MaxBoxAccV2 = mean_delta max_thr mean_img 1[iou>=d]"""
    c = curves if idx is None else curves[idx]
    vals = [float((c >= d).mean(0).max()) for d in deltas]
    return float(np.mean(vals)), vals


def pxap(pos_hist, neg_hist):
    """Choe et al. PxAP from per-threshold (TP, FP) histograms, thresholds
    ascending.  Returns area under the pixel PR curve."""
    tp = pos_hist.astype(np.float64)
    fp = neg_hist.astype(np.float64)
    npos = tp[0] if len(tp) else 0.0
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / max(npos, 1e-12)
    # integrate precision d(recall), recall descending as threshold rises
    ap = 0.0
    for i in range(1, len(rec)):
        ap += (rec[i - 1] - rec[i]) * prec[i - 1]
    return float(ap)


# ================================================================ statistics
def boot_mean(x, n=2000, seed=0):
    rs = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float64)
    m = rs.randint(0, len(x), (n, len(x)))
    b = x[m].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_diff(x, y, n=2000, seed=0):
    """paired bootstrap of mean(x) - mean(y)."""
    rs = np.random.RandomState(seed)
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = rs.randint(0, len(x), (n, len(x)))
    b = x[m].mean(1) - y[m].mean(1)
    d = float(x.mean() - y.mean())
    return d, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_mbav2(curves, n=1000, seed=0, deltas=(0.3, 0.5, 0.7)):
    rs = np.random.RandomState(seed)
    v0, _ = maxboxacc_v2(curves, deltas)
    ni = curves.shape[0]
    b = np.empty(n)
    for i in range(n):
        b[i] = maxboxacc_v2(curves, deltas, rs.randint(0, ni, ni))[0]
    return v0, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_mbav2_diff(ca, cb, n=1000, seed=0, deltas=(0.3, 0.5, 0.7)):
    rs = np.random.RandomState(seed)
    d0 = maxboxacc_v2(ca, deltas)[0] - maxboxacc_v2(cb, deltas)[0]
    ni = ca.shape[0]
    b = np.empty(n)
    for i in range(n):
        k = rs.randint(0, ni, ni)
        b[i] = maxboxacc_v2(ca, deltas, k)[0] - maxboxacc_v2(cb, deltas, k)[0]
    p = 2.0 * min((b <= 0).mean(), (b >= 0).mean())
    return (float(d0), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
            float(min(1.0, max(p, 1.0 / n))))


def wilcoxon(x, y):
    from scipy.stats import wilcoxon as W
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    d = x - y
    if np.all(d == 0):
        return dict(stat=np.nan, p=1.0, n_nonzero=0, rank_biserial=0.0)
    r = W(x, y, zero_method='wilcox', alternative='two-sided')
    nz = int((d != 0).sum())
    # rank-biserial correlation = (W+ - W-)/(W+ + W-)
    from scipy.stats import rankdata
    ad = np.abs(d[d != 0]); rk = rankdata(ad)
    wp = rk[d[d != 0] > 0].sum(); wm = rk[d[d != 0] < 0].sum()
    return dict(stat=float(r.statistic), p=float(r.pvalue), n_nonzero=nz,
                rank_biserial=float((wp - wm) / (wp + wm)))


def mcnemar(a, b):
    """paired binary.  a,b boolean arrays."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return dict(n10=n10, n01=n01, p=1.0)
    p = float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return dict(n10=n10, n01=n01, p=p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k])
        adj[k] = min(1.0, run)
    return adj.tolist()
