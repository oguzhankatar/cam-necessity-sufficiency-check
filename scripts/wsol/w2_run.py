"""W2 -- the pre-specified comparison, one (dataset, backbone) pair per call.

    python w2_run.py <cub|voc12|voc12seg|coco17> <resnet50|vgg16gap|vitb16> [nmax]

Protocol (fixed before any number was looked at; see protocols/wsol_protocol.md):

  models
    collapsed : GAP aggregator + linear head   -- the degenerate point.
                Certified ADDITIVE by rtlib's literal fp64 loop; N == S there,
                so the "collapsed map" is a single map, not two.
    edited-LSE: same weights, LSE head at tau = +1.0 (the knee).  Condition A
                = zero refit of W (only the pooling operator changes).
    edited-LN : ViT only.  head(LN(mean_t z_t)) instead of head(mean_t LN(z_t)).
                Zero refit of the LN; W refit because the pooled representation
                it reads is a different vector.
  maps (all at the model's own cell grid, bilinearly upsampled, min-max scaled)
    collapsed, N_lse, S_lse, N_ln, S_ln          (+ 3x3-block versions)
    cam, gradcam                                  (published references)
    center, actnorm, random                       (the trivial controls)
  baseline for N and S: a fixed bank of 8 real images from the FIT split,
    averaged (an Expected-Gradients reference distribution).  A baseline whose
    pooled vector equals the image's is forbidden and none is used.
  metric: the dataset's own -- MaxBoxAccV2 for CUB, pointing game (+ PxAP and
    peak mask IoU) for VOC / COCO.
"""
import os, sys, json, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W

DS = sys.argv[1]
BK = sys.argv[2]
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
BASIS = sys.argv[4] if len(sys.argv) > 4 else 'bank'     # 'bank' | 'meancell'
NBANK = 8 if BASIS == 'bank' else 1
TAU = float(sys.argv[5]) if len(sys.argv) > 5 else W.PROTOCOL['tau']
EVAL_RES = 224
NTHR = 100
THR = np.arange(NTHR) / NTHR             # 100 CAM thresholds (Choe et al. sweep)
DELTAS = (0.3, 0.5, 0.7)
SEED = 0
t00 = time.time()
SUF = (('' if not NMAX else '_n%d' % NMAX)
       + ('' if BASIS == 'bank' else '_' + BASIS)
       + ('' if TAU == W.PROTOCOL['tau'] else '_tau%g' % TAU))


def log(*a):
    print('[%6.0fs]' % (time.time() - t00), *a, flush=True)


man = (W.cub_manifest() if DS == 'cub' else
       W.voc_manifest('box') if DS == 'voc12' else
       W.voc_manifest('seg') if DS == 'voc12seg' else
       W.coco_manifest())
log('dataset %s  fit=%d eval=%d  K=%d multilabel=%s'
    % (man['name'], len(man['fit']), len(man['ev']), man['n_class'], man['multilabel']))
if NMAX:
    man['ev'] = man['ev'][:NMAX]

ex = W.Extractor(BK)
log('backbone %s  input=%d  grid=%dx%d  C=%d  npfx=%d'
    % (BK, ex.size, ex.H, ex.W, ex.C, ex.npfx))
IS_VIT = (ex.fmt == 'BTC')
K = man['n_class']



# --------------------------------------------------------- head fitting
def pooled(recs, prenorm=False):
    P = np.zeros((len(recs), ex.C), dtype=np.float32)
    Q = np.zeros((len(recs), ex.C), dtype=np.float32) if prenorm else None
    for i0, c1, c2 in ex.batch([r['path'] for r in recs], bs=24, prenorm=prenorm):
        P[i0:i0 + c1.shape[0]] = c1.mean(1).cpu().numpy()
        if prenorm:
            Q[i0:i0 + c1.shape[0]] = c2.mean(1).cpu().numpy()
        if i0 % 2400 == 0:
            log('   pooled %d/%d' % (i0, len(recs)))
    return P, Q


def fit_head(Pfit, recs, multilabel, K, epochs=400, wd=1e-4, lr=3e-3, tag=''):
    X = torch.from_numpy(Pfit).cuda()
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    lin = nn.Linear(ex.C, K).cuda()
    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
    if multilabel:
        Y = torch.zeros(len(recs), K).cuda()
        for i, r in enumerate(recs):
            Y[i, r['labels']] = 1.0
        lf = nn.BCEWithLogitsLoss()
    else:
        Y = torch.tensor([r['label'] for r in recs]).cuda()
        lf = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for e in range(epochs):
        opt.zero_grad(); loss = lf(lin(Xn), Y); loss.backward(); opt.step(); sch.step()
    Wt = (lin.weight.data / sd)
    bt = lin.bias.data - (lin.weight.data * mu / sd).sum(1)
    log('  fit %-8s final train loss %.4f' % (tag, float(loss)))
    return Wt.double(), bt.double()


log('pass 1: pooled features of the fit split')
Pfit, Ppre = pooled(man['fit'], prenorm=IS_VIT)
MEANCELL = Pfit.mean(0).astype(np.float64)
MEANCELL_PRE = Ppre.mean(0).astype(np.float64) if IS_VIT else None
Wg, bg = fit_head(Pfit, man['fit'], man['multilabel'], K, tag='gap')
heads = {'gap': (Wg, bg, 'post')}
# a DETACHED float64 copy: .double() on the model's own module would
# silently convert the running model and break the float32 forward pass.
LND = copy.deepcopy(ex.m.norm).double() if IS_VIT else None
if IS_VIT:
    with torch.no_grad():
        Pln = LND(torch.from_numpy(Ppre).cuda().double()).float().cpu().numpy()
    Wl, bl = fit_head(Pln, man['fit'], man['multilabel'], K, tag='lnpost')
    heads['lnpost'] = (Wl, bl, 'pre')
torch.cuda.empty_cache()

WG = heads['gap'][0].cpu().numpy(); BG = heads['gap'][1].cpu().numpy()
if IS_VIT:
    WL = heads['lnpost'][0]; BL = heads['lnpost'][1]

# ------------------------------------------------------------- baseline bank
rs = np.random.RandomState(SEED)
if BASIS == 'bank':
    bank_idx = sorted(rs.choice(len(man['fit']), NBANK, replace=False).tolist())
    bank_paths = [man['fit'][i]['path'] for i in bank_idx]
    for _, c1, c2 in ex.batch(bank_paths, bs=NBANK, prenorm=IS_VIT):
        BANK_POST = c1.double().cpu().numpy()
        BANK_PRE = c2.double().cpu().numpy() if IS_VIT else None
    log('baseline basis=bank: %d fit-split images %s' % (NBANK, bank_idx))
else:
    # dataset-mean cell, broadcast to every cell.  Its pooled vector is the
    # dataset mean, never the image's own -- the forbidden baseline (pooled
    # vector equal to the image's, e.g. a spatial shuffle) is not used anywhere.
    bank_idx = []
    BANK_POST = np.repeat(MEANCELL[None, None, :], ex.T, axis=1)
    BANK_PRE = (np.repeat(MEANCELL_PRE[None, None, :], ex.T, axis=1)
                if IS_VIT else None)
    log('baseline basis=meancell (dataset-mean cell broadcast to all %d cells)' % ex.T)
del Pfit, Ppre
BS_GAP = np.einsum('btc,kc->bkt', BANK_POST, WG) + BG[None, :, None]

# ---------------------------------------------------------------- certificate
log('certificate on EVAL-resolution cells with the FITTED head, float64')
# always real images, never the mean-cell baseline: no structural or affinity
# test is ever run on a synthetic feature map.
cpaths = [man['fit'][i]['path'] for i in
          sorted(np.random.RandomState(SEED + 1).choice(len(man['fit']), 6, False).tolist())]
for _, c1, c2 in ex.batch(cpaths, bs=6, prenorm=IS_VIT):
    CERT_POST = c1.double().cpu().numpy()
    CERT_PRE = c2.double().cpu().numpy() if IS_VIT else None
cert_cells = torch.from_numpy(CERT_POST).cuda()
cert_cls = [0] * 6
cert = {}
Wc, bc = heads['gap'][0].cuda(), heads['gap'][1].cuda()
for nm, hd in [('collapsed_gap', W.PoolHead(Wc, bc, 'gap')),
               ('edited_lse_tau1', W.PoolHead(Wc, bc, 'lse', tau=TAU))]:
    c = W.certify_head(hd, cert_cells, cert_cls, nm); c.pop('per_pair')
    cert[nm] = c
    log('  %-16s ADDITIVE=%-5s maxNS=%.2e Rhalf=%.2e I2=%.2e'
        % (nm, c['ADDITIVE'], c['maxNS'], c['Rhalf'], c['I2']))
if IS_VIT:
    c = W.certify_head(W.PoolHead(WL.cuda(), BL.cuda(), 'lnpost', ln=LND),
                       torch.from_numpy(CERT_PRE).cuda(), cert_cls, 'edited_lnpost')
    c.pop('per_pair'); cert['edited_lnpost'] = c
    log('  %-16s ADDITIVE=%-5s maxNS=%.2e Rhalf=%.2e I2=%.2e'
        % ('edited_lnpost', c['ADDITIVE'], c['maxNS'], c['Rhalf'], c['I2']))

# ------------------------------------------------------------ static controls
yy, xx = np.mgrid[0:ex.H, 0:ex.W].astype(np.float64)
CENTER = np.exp(-(((yy - (ex.H - 1) / 2.) ** 2 + (xx - (ex.W - 1) / 2.) ** 2))
                / (2 * (0.25 * ex.H) ** 2)).ravel()

MAPS = ['collapsed', 'collapsed_b3', 'N_lse', 'S_lse', 'N_lse_b3', 'S_lse_b3',
        'cam', 'gradcam', 'center', 'actnorm', 'random']
if IS_VIT:
    MAPS += ['N_ln', 'S_ln', 'N_ln_b3', 'S_ln_b3']


def ln_fn(c):
    def fn(V):
        with torch.no_grad():
            v = torch.from_numpy(np.ascontiguousarray(V)).cuda().double()
            return (LND(v) @ WL.T.cuda() + BL.cuda())[:, c].cpu().numpy()
    return fn


def make_maps(Xp, Xpre, c, ridx):
    a = Xp @ WG[c] + BG[c]
    T = len(a)
    out = {k: np.zeros(T) for k in
           ('collapsed', 'collapsed_b3', 'N_lse', 'S_lse', 'N_lse_b3', 'S_lse_b3')}
    for k in range(NBANK):
        be = BS_GAP[k, c]
        n, _ = W.ns_gap(a, be); out['collapsed'] += n
        n, s = W.ns_lse(a, be, TAU, ex.H, ex.W); out['N_lse'] += n; out['S_lse'] += s
        n, _ = W.ns_gap(a, be, ex.H, ex.W, True); out['collapsed_b3'] += n
        n, s = W.ns_lse(a, be, TAU, ex.H, ex.W, True)
        out['N_lse_b3'] += n; out['S_lse_b3'] += s
    for k in out:
        out[k] /= NBANK
    out['cam'] = Xp @ WG[c]
    out['gradcam'] = np.maximum(out['cam'], 0.0)
    out['center'] = CENTER
    out['actnorm'] = np.linalg.norm(Xp, axis=1)
    out['random'] = np.random.RandomState(1000 + ridx).rand(T)
    if IS_VIT:
        fn = ln_fn(c)
        for tag, blk in [('', False), ('_b3', True)]:
            n = np.zeros(T); s = np.zeros(T)
            for k in range(NBANK):
                n1, s1 = W.ns_pooled(Xpre, BANK_PRE[k], fn, ex.H, ex.W, blk)
                n += n1; s += s1
            out['N_ln' + tag] = n / NBANK
            out['S_ln' + tag] = s / NBANK
    return out


# ------------------------------------------------------------------- GT utils
def scale_boxes(bs, w0, h0):
    sx, sy = EVAL_RES / w0, EVAL_RES / h0
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in bs]


def gt_mask(r, cls):
    if DS == 'voc12seg':
        seg = np.array(Image.open(r['mask']))
        m = (seg == cls + 1).astype(np.uint8); void = (seg == 255).astype(np.uint8)
    else:
        co = man['_coco']; m = None
        for aid in r['ann_by_class'][str(cls)]:
            mm = co.annToMask(co.loadAnns([aid])[0])
            m = mm if m is None else np.maximum(m, mm)
        void = np.zeros_like(m)
    m = cv2.resize(m, (EVAL_RES, EVAL_RES), interpolation=cv2.INTER_NEAREST).astype(bool)
    void = cv2.resize(void, (EVAL_RES, EVAL_RES), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m, void


DIL = np.ones((31, 31), np.uint8)       # 15 px pointing-game tolerance

# --------------------------------------------------------------- evaluation
recs = man['ev']
HAS_MASK = DS in ('voc12seg', 'coco17')
curves = {k: [] for k in MAPS}
pt_s = {k: [] for k in MAPS}
pt_t = {k: [] for k in MAPS}
peak = {k: [] for k in MAPS}
pos_h = {k: np.zeros(NTHR) for k in MAPS}
neg_h = {k: np.zeros(NTHR) for k in MAPS}
rec_img, acc = [], {k: [] for k in heads}
# map geometry: how concentrated is each map, and how far is it from the
# collapsed map in rank order?  This is what separates "a different map" from
# "the same map with a different scale".
conc = {k: [] for k in MAPS}          # (participation ratio, top-1 mass, H/logT)
rho_coll = {k: [] for k in MAPS}      # Spearman vs the collapsed map, per record
from scipy.stats import rankdata


def concentration(v):
    a = np.abs(v); tot = a.sum() + 1e-300
    pr = (tot ** 2) / (len(a) * (v ** 2).sum() + 1e-300)
    p = a / tot
    h = -(p * np.log(p + 1e-300)).sum() / np.log(len(a))
    return [float(pr), float(a.max() / tot), float(h)]


def spear(x, y):
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))         if sx > 0 and sy > 0 else float('nan')


n_rec = 0
POOL = ThreadPoolExecutor(8)
log('pass 2: evaluation over %d images' % len(recs))

for i0, c1, c2 in ex.batch([r['path'] for r in recs], bs=24, prenorm=IS_VIT):
    Xb = c1.double().cpu().numpy()
    Xpb = c2.double().cpu().numpy() if IS_VIT else None
    for j in range(Xb.shape[0]):
        ridx = i0 + j
        r = recs[ridx]
        Xp, Xpre = Xb[j], (Xpb[j] if IS_VIT else None)
        w0, h0 = Image.open(r['path']).size
        tgts = [r['label']] if not man['multilabel'] else r['labels']
        for hk, (Wt, bt, sp) in heads.items():
            m_ = (Xpre if sp == 'pre' else Xp).mean(0)
            if sp == 'pre':
                with torch.no_grad():
                    z = (LND(torch.from_numpy(m_[None]).cuda().double())
                         @ Wt.T.cuda() + bt.cuda())[0].cpu().numpy()
            else:
                z = m_ @ Wt.cpu().numpy().T + bt.cpu().numpy()
            if man['multilabel']:
                acc[hk].append(z.astype(np.float32))
            else:
                acc[hk].append(int(np.argmax(z) == r['label']))
        for c in tgts:
            mp = make_maps(Xp, Xpre, c, n_rec)
            gb = None
            if not HAS_MASK:
                gb = scale_boxes(r['boxes'] if 'boxes' in r
                                 else r['boxes_by_class'][str(c)], w0, h0)
            if HAS_MASK:
                gm, gv = gt_mask(r, c)
                valid = ~gv
                gmd = cv2.dilate(gm.astype(np.uint8), DIL).astype(bool)
                gm_v = gm[valid]; npos = int(gm_v.sum())
            U = {k: W.upsample(mp[k], ex.H, ex.W, EVAL_RES) for k in MAPS}
            if DS == 'cub':
                # cv2 releases the GIL, so the threshold sweep threads cleanly
                for k, cvv in zip(MAPS, POOL.map(
                        lambda kk: W.box_curve(U[kk], gb, THR), MAPS)):
                    curves[k].append(cvv)
            for k in MAPS:
                conc[k].append(concentration(mp[k]))
                rho_coll[k].append(spear(mp[k], mp['collapsed']))
                u = U[k]
                pk = np.unravel_index(np.argmax(u), u.shape)
                if HAS_MASK:
                    pt_s[k].append(bool(gm[pk])); pt_t[k].append(bool(gmd[pk]))
                    b = np.clip((u[valid] * NTHR).astype(np.int64), 0, NTHR - 1)
                    hp = np.bincount(b[gm_v], minlength=NTHR)
                    hn = np.bincount(b[~gm_v], minlength=NTHR)
                    tp = hp[::-1].cumsum()[::-1].astype(np.float64)
                    fp = hn[::-1].cumsum()[::-1].astype(np.float64)
                    pos_h[k] += tp; neg_h[k] += fp
                    peak[k].append(float((tp / np.maximum(npos + fp, 1e-9)).max())
                                   if npos else 0.0)
                else:
                    pt_s[k].append(any(b[0] <= pk[1] <= b[2] and b[1] <= pk[0] <= b[3]
                                       for b in gb))
                    pt_t[k].append(any(b[0] - 15 <= pk[1] <= b[2] + 15 and
                                       b[1] - 15 <= pk[0] <= b[3] + 15 for b in gb))
            rec_img.append(ridx); n_rec += 1
    if (i0 // 24) % 25 == 0:
        log('  %d/%d images, %d records' % (i0 + Xb.shape[0], len(recs), n_rec))

log('evaluation done: %d records' % n_rec)

# ------------------------------------------------------------------- summary
out = dict(dataset=man['name'], backbone=BK, timm=W.BACKBONES[BK]['timm'],
           input=ex.size, grid=[ex.H, ex.W], C=ex.C, eval_res=EVAL_RES,
           n_fit=len(man['fit']), n_eval=len(recs), n_records=n_rec,
           n_bank=NBANK, bank_idx=bank_idx, basis=BASIS, tau=TAU, n_thresholds=NTHR,
           deltas=list(DELTAS), certificate=cert, maps=MAPS,
           code_sha=W.code_sha(),
           refit=('frozen backbone; a linear head fitted on GAP-pooled cells. '
                  'The LSE row reuses those weights unchanged (Condition A): '
                  'the ONLY difference between the collapsed and edited-LSE '
                  'models is the pooling operator.  The lnpost row refits the '
                  'linear head on LN(mean_t z_t) because that is a different '
                  'pooled vector; the LayerNorm itself is the shipped one.'))

clsq = {}
for hk in heads:
    if man['multilabel']:
        from sklearn.metrics import average_precision_score
        Z = np.stack(acc[hk])
        Y = np.zeros_like(Z)
        for i, r in enumerate(recs):
            Y[i, r['labels']] = 1
        ap = [average_precision_score(Y[:, k], Z[:, k]) for k in range(K)
              if Y[:, k].sum() > 0]
        clsq[hk] = dict(mAP=float(np.mean(ap)), n_class_eval=len(ap))
    else:
        clsq[hk] = dict(top1=float(np.mean(acc[hk])))
out['classification'] = clsq
log('classification: %s' % json.dumps(clsq))

per = {}
for k in MAPS:
    e = {}
    if DS == 'cub':
        cv = np.stack(curves[k]).astype(np.float32)
        v, lo, hi = W.boot_mbav2(cv, n=1000, seed=SEED, deltas=DELTAS)
        _, byd = W.maxboxacc_v2(cv, DELTAS)
        e['MaxBoxAccV2'] = dict(value=v, ci=[lo, hi],
                                per_delta={str(d): byd[i] for i, d in enumerate(DELTAS)})
        e['best_iou_mean'] = W.boot_mean(cv.max(1), seed=SEED)
        np.save(os.path.join(W.OUT, 'curve_%s_%s%s_%s.npy' % (DS, BK, SUF, k)), cv)
    e['pointing_strict'] = W.boot_mean(np.array(pt_s[k], float), seed=SEED)
    e['pointing_tol15'] = W.boot_mean(np.array(pt_t[k], float), seed=SEED)
    np.save(os.path.join(W.OUT, 'pt_%s_%s%s_%s.npy' % (DS, BK, SUF, k)),
            np.stack([np.array(pt_s[k]), np.array(pt_t[k])]))
    if HAS_MASK:
        e['PxAP'] = W.pxap(pos_h[k], neg_h[k])
        e['peak_mask_iou'] = W.boot_mean(np.array(peak[k], float), seed=SEED)
        np.save(os.path.join(W.OUT, 'iou_%s_%s%s_%s.npy' % (DS, BK, SUF, k)),
                np.array(peak[k]))
    c = np.array(conc[k])
    e['concentration'] = dict(participation_ratio=float(c[:, 0].mean()),
                              top1_mass_fraction=float(c[:, 1].mean()),
                              entropy_over_logT=float(c[:, 2].mean()))
    rr = np.array(rho_coll[k], float)
    e['spearman_vs_collapsed'] = float(np.nanmean(rr))
    per[k] = e
out['per_map'] = per
np.save(os.path.join(W.OUT, 'recidx_%s_%s%s.npy' % (DS, BK, SUF)), np.array(rec_img))
W.save('w2_%s_%s%s.json' % (DS, BK, SUF), out)
for k in MAPS:
    s = per[k]
    bits = []
    for n, v in s.items():
        if n == 'MaxBoxAccV2':
            bits.append('MaxBoxAccV2=%.4f' % v['value'])
        elif n == 'concentration':
            bits.append('PR=%.3f' % v['participation_ratio'])
        elif isinstance(v, (list, tuple)):
            bits.append('%s=%.4f' % (n, v[0]))
        elif isinstance(v, (int, float)):
            bits.append('%s=%.4f' % (n, v))
    log('%-14s %s' % (k, ' '.join(bits)))
