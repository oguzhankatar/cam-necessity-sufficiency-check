"""C3 tasks 4b, 5b and 7 -- one GPU pass per (dataset, backbone).

    python c3_gpu_refit.py <cub|voc12|voc12seg|coco17> <resnet50|vgg16gap|vitb16> [nmax]

Re-builds the w2_run.py pipeline from wlib (unchanged recipe, same seeds, same
bank, same certificate images) and adds three things w2 never recorded:

  (a) certificate decomposition: for the GAP-fitted weights reused under LSE
      at T in {-8,-2,-0.5,+0.5,+1,+2,+8}, the per-pair Pearson corr(N,S) of
      the literal rtlib loop on w2's three certificate pairs (class 0), plus
      the spread of the class-0 cell scores and the effective number of cells
      the LSE weights at each T -- the explanation of the near-constant CUB
      correlation across T = 1, 2, 8;
  (b) the separation of the maps actually scored: per-record Pearson
      corr(N_lse, S_lse) on the evaluation records (bank reference, 1x1),
      and corr(N_ln, S_ln) on the ViT (on a 1-in-4 subsample of records);
  (c) a REFITTED-HEAD arm: a linear head trained directly under LSE pooling
      at T = +1 (same recipe as the GAP head: zero init, AdamW lr 3e-3,
      wd 1e-4, cosine, 400 full-batch epochs, per-channel standardisation),
      whose N/S maps are scored on every metric of the dataset next to the
      reproduced collapsed, N_lse, S_lse (reused-head) and centre maps.

Nothing under scripts/ or outputs/ is written; results go to localisation_reanalysis/out/gpu/.
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
from localisation_reanalysis_lib import W, WO, OUT  # noqa: E402
import rtlib  # noqa: E402  (path inserted by wlib)

DS, BK = sys.argv[1], sys.argv[2]
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TAU = 1.0
NBANK = 8
EVAL_RES = 224
NTHR = 100
THR = np.arange(NTHR) / NTHR
DELTAS = (0.3, 0.5, 0.7)
SEED = 0
TLIST = [-8.0, -2.0, -0.5, 0.5, 1.0, 2.0, 8.0]
GOUT = os.path.join(OUT, 'gpu')
os.makedirs(GOUT, exist_ok=True)
TAG = '%s_%s%s' % (DS, BK, '_n%d' % NMAX if NMAX else '')
t00 = time.time()


def log(*a):
    print('[%6.0fs]' % (time.time() - t00), *a, flush=True)


man = (W.cub_manifest() if DS == 'cub' else W.voc_manifest('box') if DS == 'voc12' else
       W.voc_manifest('seg') if DS == 'voc12seg' else W.coco_manifest())
if NMAX:
    man['ev'] = man['ev'][:NMAX]
ex = W.Extractor(BK)
IS_VIT = (ex.fmt == 'BTC')
K = man['n_class']
log('%s %s fit=%d eval=%d K=%d grid=%dx%d C=%d' % (DS, BK, len(man['fit']), len(man['ev']), K, ex.H, ex.W, ex.C))

# ------------------------------------------------------------ fit features
NF = len(man['fit'])
CELLS = torch.empty((NF, ex.T, ex.C), dtype=torch.float16, device='cuda')
P = np.zeros((NF, ex.C), np.float32)
Q = np.zeros((NF, ex.C), np.float32) if IS_VIT else None
for i0, c1, c2 in ex.batch([r['path'] for r in man['fit']], bs=24, prenorm=IS_VIT):
    n = c1.shape[0]
    CELLS[i0:i0 + n] = c1.half()
    P[i0:i0 + n] = c1.mean(1).cpu().numpy()
    if IS_VIT:
        Q[i0:i0 + n] = c2.mean(1).cpu().numpy()
log('fit features extracted')


def targets(recs):
    if man['multilabel']:
        Y = torch.zeros(len(recs), K, device='cuda')
        for i, r in enumerate(recs):
            Y[i, r['labels']] = 1.0
        return Y, nn.BCEWithLogitsLoss()
    return torch.tensor([r['label'] for r in recs], device='cuda'), nn.CrossEntropyLoss()


def fit_head(Pfit, recs, epochs=400, wd=1e-4, lr=3e-3, tag=''):
    """verbatim copy of w2_run.fit_head."""
    X = torch.from_numpy(Pfit).cuda()
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    lin = nn.Linear(ex.C, K).cuda()
    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
    Y, lf = targets(recs)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for e in range(epochs):
        opt.zero_grad(); loss = lf(lin(Xn), Y); loss.backward(); opt.step(); sch.step()
    Wt = (lin.weight.data / sd)
    bt = lin.bias.data - (lin.weight.data * mu / sd).sum(1)
    log('  fit %-8s final train loss %.4f' % (tag, float(loss)))
    return Wt.double(), bt.double(), float(loss)


def fit_lse_head(recs, tau, epochs=400, wd=1e-4, lr=3e-3, chunk=256):
    """the same recipe, but the loss is taken through LSE pooling of the
    per-cell scores; standardisation folded into the weights, full-batch
    gradient accumulated over chunks."""
    X = torch.from_numpy(P).cuda()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lin = nn.Linear(ex.C, K).cuda()
    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
    Y, lf = targets(recs)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lT = math.log(ex.T)
    for e in range(epochs):
        opt.zero_grad()
        tot = 0.0
        for i in range(0, NF, chunk):
            Wf = lin.weight / sd
            bf = lin.bias - (lin.weight * mu / sd).sum(1)
            Xc = CELLS[i:i + chunk].float()
            S = Xc @ Wf.T + bf
            z = (torch.logsumexp(tau * S, dim=1) - lT) / tau
            loss = lf(z, Y[i:i + chunk]) * (Xc.shape[0] / NF)
            loss.backward()
            tot += float(loss)
        opt.step(); sch.step()
    with torch.no_grad():
        Wt = lin.weight / sd
        bt = lin.bias - (lin.weight * mu / sd).sum(1)
    log('  fit lse-refit T=%g final train loss %.4f' % (tau, tot))
    return Wt.double(), bt.double(), tot


Wg, bg, loss_gap = fit_head(P, man['fit'], tag='gap')
LND = copy.deepcopy(ex.m.norm).double() if IS_VIT else None
if IS_VIT:
    with torch.no_grad():
        Pln = LND(torch.from_numpy(Q).cuda().double()).float().cpu().numpy()
    Wl, bl, loss_ln = fit_head(Pln, man['fit'], tag='lnpost')
Wr, br, loss_rf = fit_lse_head(man['fit'], TAU)
del CELLS
torch.cuda.empty_cache()
WG, BG = Wg.cpu().numpy(), bg.cpu().numpy()
WR, BR = Wr.cpu().numpy(), br.cpu().numpy()

# ------------------------------------------------------------ bank (as w2)
rs = np.random.RandomState(SEED)
bank_idx = sorted(rs.choice(len(man['fit']), NBANK, replace=False).tolist())
for _, c1, c2 in ex.batch([man['fit'][i]['path'] for i in bank_idx], bs=NBANK, prenorm=IS_VIT):
    BANK_POST = c1.double().cpu().numpy()
    BANK_PRE = c2.double().cpu().numpy() if IS_VIT else None
BS_GAP = np.einsum('btc,kc->bkt', BANK_POST, WG) + BG[None, :, None]
BS_RF = np.einsum('btc,kc->bkt', BANK_POST, WR) + BR[None, :, None]

# ------------------------------------------------- (a) certificate decomposition
cpaths = [man['fit'][i]['path'] for i in
          sorted(np.random.RandomState(SEED + 1).choice(len(man['fit']), 6, False).tolist())]
for _, c1, c2 in ex.batch(cpaths, bs=6, prenorm=IS_VIT):
    CERT_POST = torch.from_numpy(c1.double().cpu().numpy()).cuda()
    CERT_PRE = torch.from_numpy(c2.double().cpu().numpy()).cuda() if IS_VIT else None
cert = dict(pairs=[], T={})
Wc, bc = Wg.cuda(), bg.cuda()
with torch.no_grad():
    s0 = (CERT_POST @ Wc[0] + bc[0]).cpu().numpy()        # (6, T) class-0 scores
cert['class0_score_range_per_image'] = (s0.max(1) - s0.min(1)).tolist()
cert['class0_score_std_per_image'] = s0.std(1).tolist()
for T in TLIST:
    hd = W.PoolHead(Wc, bc, 'lse', tau=T)
    per = []
    for i in range(0, 6, 2):
        N, S, zA, zB = rtlib.NS_literal(hd, CERT_POST[i], CERT_POST[i + 1], 0)
        per.append(float(np.corrcoef(N, S)[0, 1]))
    ess = []
    for i in range(6):
        p = np.exp(T * (s0[i] - s0[i].max())) if T > 0 else np.exp(T * (s0[i] - s0[i].min()))
        p = p / p.sum()
        ess.append(float(1.0 / (p ** 2).sum()))
    cert['T'][str(T)] = dict(corr_per_pair=per, corr_min=min(per), sep=1 - min(per),
                             effective_cells_per_image=ess)
    log('  cert T=%+g corr per pair %s  effective cells %s' % (T, np.round(per, 4).tolist(), np.round(ess, 1).tolist()))
hd = W.PoolHead(Wr.cuda(), br.cuda(), 'lse', tau=TAU)
per = []
for i in range(0, 6, 2):
    N, S, _, _ = rtlib.NS_literal(hd, CERT_POST[i], CERT_POST[i + 1], 0)
    per.append(float(np.corrcoef(N, S)[0, 1]))
cert['refit_T1'] = dict(corr_per_pair=per, corr_min=min(per), sep=1 - min(per))
if IS_VIT:
    hl = W.PoolHead(Wl.cuda(), bl.cuda(), 'lnpost', ln=LND)
    per = []
    for i in range(0, 6, 2):
        N, S, _, _ = rtlib.NS_literal(hl, CERT_PRE[i], CERT_PRE[i + 1], 0)
        per.append(float(np.corrcoef(N, S)[0, 1]))
    cert['lnpost'] = dict(corr_per_pair=per, corr_min=min(per), sep=1 - min(per))
arch = json.load(open(os.path.join(WO, 'w2_%s_%s.json' % (DS, BK))))
cert['archived_corr_T1'] = arch['certificate']['edited_lse_tau1']['corrNS_recorded_never_used']
log('  certificate T=1 corr reproduced %.6f vs archived %.6f'
    % (cert['T']['1.0']['corr_min'], cert['archived_corr_T1']))

# ------------------------------------------------------------ evaluation
yy, xx = np.mgrid[0:ex.H, 0:ex.W].astype(np.float64)
CENTER = np.exp(-(((yy - (ex.H - 1) / 2.) ** 2 + (xx - (ex.W - 1) / 2.) ** 2))
                / (2 * (0.25 * ex.H) ** 2)).ravel()
EV = ['collapsed', 'N_lse', 'S_lse', 'N_rf', 'S_rf', 'center']


def ln_fn(c):
    Wlc, blc = Wl.cuda(), bl.cuda()

    def fn(V):
        with torch.no_grad():
            v = torch.from_numpy(np.ascontiguousarray(V)).cuda().double()
            return (LND(v) @ Wlc.T + blc)[:, c].cpu().numpy()
    return fn


def pear(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float('nan')


def make_maps(Xp, c):
    a = Xp @ WG[c] + BG[c]
    ar = Xp @ WR[c] + BR[c]
    out = {k: np.zeros(len(a)) for k in ('collapsed', 'N_lse', 'S_lse', 'N_rf', 'S_rf')}
    for k in range(NBANK):
        n, _ = W.ns_gap(a, BS_GAP[k, c]); out['collapsed'] += n
        n, s = W.ns_lse(a, BS_GAP[k, c], TAU, ex.H, ex.W); out['N_lse'] += n; out['S_lse'] += s
        n, s = W.ns_lse(ar, BS_RF[k, c], TAU, ex.H, ex.W); out['N_rf'] += n; out['S_rf'] += s
    for k in out:
        out[k] /= NBANK
    out['center'] = CENTER
    return out


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


DIL = np.ones((31, 31), np.uint8)
recs = man['ev']
HAS_MASK = DS in ('voc12seg', 'coco17')
curves = {k: [] for k in EV}
pt_s = {k: [] for k in EV}; pt_t = {k: [] for k in EV}; peak = {k: [] for k in EV}
corr = dict(lse=[], rf=[], ln=[], ln_rec=[])
zs = dict(gap=[], lse_reused=[], lse_refit=[])
rec_img = []
POOL = ThreadPoolExecutor(8)
n_rec = 0
lT = math.log(ex.T)
for i0, c1, c2 in ex.batch([r['path'] for r in recs], bs=24, prenorm=IS_VIT):
    Xb = c1.double().cpu().numpy()
    Xpb = c2.double().cpu().numpy() if IS_VIT else None
    for j in range(Xb.shape[0]):
        ridx = i0 + j
        r = recs[ridx]
        Xp = Xb[j]
        w0, h0 = Image.open(r['path']).size
        Sg = Xp @ WG.T + BG; Sr = Xp @ WR.T + BR            # (T, K)
        zg = Sg.mean(0)
        zl = (np.logaddexp.reduce(TAU * Sg, axis=0) - lT) / TAU
        zr = (np.logaddexp.reduce(TAU * Sr, axis=0) - lT) / TAU
        for kk, z in (('gap', zg), ('lse_reused', zl), ('lse_refit', zr)):
            zs[kk].append(z.astype(np.float32) if man['multilabel'] else int(np.argmax(z) == r['label']))
        tgts = [r['label']] if not man['multilabel'] else r['labels']
        for c in tgts:
            mp = make_maps(Xp, c)
            corr['lse'].append(pear(mp['N_lse'], mp['S_lse']))
            corr['rf'].append(pear(mp['N_rf'], mp['S_rf']))
            if IS_VIT and n_rec % 4 == 0:
                fn = ln_fn(c)
                n_ = np.zeros(ex.T); s_ = np.zeros(ex.T)
                for k in range(NBANK):
                    n1, s1 = W.ns_pooled(Xpb[j], BANK_PRE[k], fn)
                    n_ += n1; s_ += s1
                corr['ln'].append(pear(n_, s_)); corr['ln_rec'].append(n_rec)
            gb = None
            if not HAS_MASK:
                gb = scale_boxes(r['boxes'] if 'boxes' in r else r['boxes_by_class'][str(c)], w0, h0)
            else:
                gm, gv = gt_mask(r, c)
                valid = ~gv
                gmd = cv2.dilate(gm.astype(np.uint8), DIL).astype(bool)
                gm_v = gm[valid]; npos = int(gm_v.sum())
            U = {k: W.upsample(mp[k], ex.H, ex.W, EVAL_RES) for k in EV}
            if DS == 'cub':
                for k, cvv in zip(EV, POOL.map(lambda kk: W.box_curve(U[kk], gb, THR), EV)):
                    curves[k].append(cvv)
            for k in EV:
                u = U[k]
                pk = np.unravel_index(np.argmax(u), u.shape)
                if HAS_MASK:
                    pt_s[k].append(bool(gm[pk])); pt_t[k].append(bool(gmd[pk]))
                    b = np.clip((u[valid] * NTHR).astype(np.int64), 0, NTHR - 1)
                    hp = np.bincount(b[gm_v], minlength=NTHR)
                    hn = np.bincount(b[~gm_v], minlength=NTHR)
                    tp = hp[::-1].cumsum()[::-1].astype(np.float64)
                    fp = hn[::-1].cumsum()[::-1].astype(np.float64)
                    peak[k].append(float((tp / np.maximum(npos + fp, 1e-9)).max()) if npos else 0.0)
                else:
                    pt_s[k].append(any(b[0] <= pk[1] <= b[2] and b[1] <= pk[0] <= b[3] for b in gb))
                    pt_t[k].append(any(b[0] - 15 <= pk[1] <= b[2] + 15 and
                                       b[1] - 15 <= pk[0] <= b[3] + 15 for b in gb))
            rec_img.append(ridx); n_rec += 1
    if (i0 // 24) % 40 == 0:
        log('  %d/%d images, %d records' % (i0 + Xb.shape[0], len(recs), n_rec))
log('evaluation done: %d records' % n_rec)

# ------------------------------------------------------------ save
arrs = dict(recidx=np.array(rec_img))
for k in EV:
    arrs['pt_' + k] = np.stack([np.array(pt_s[k]), np.array(pt_t[k])])
    if HAS_MASK:
        arrs['iou_' + k] = np.array(peak[k])
    if DS == 'cub':
        arrs['curve_' + k] = np.stack(curves[k]).astype(np.float32)
for k, v in corr.items():
    arrs['corr_' + k] = np.array(v)
np.savez_compressed(os.path.join(GOUT, 'arr_%s.npz' % TAG), **arrs)

clsq = {}
for kk, v in zs.items():
    if man['multilabel']:
        from sklearn.metrics import average_precision_score
        Z = np.stack(v); Y = np.zeros_like(Z)
        for i, r in enumerate(recs):
            Y[i, r['labels']] = 1
        clsq[kk] = dict(mAP=float(np.mean([average_precision_score(Y[:, k], Z[:, k])
                                           for k in range(K) if Y[:, k].sum() > 0])))
    else:
        clsq[kk] = dict(top1=float(np.mean(v)))
# reproduction of the archived arrays
repro = {}
if not NMAX:
    for k in ('collapsed', 'N_lse', 'S_lse', 'center'):
        a = np.load(os.path.join(WO, 'pt_%s_%s_%s.npy' % (DS, BK, k)))
        repro['pt_' + k + '_agree'] = float((a == arrs['pt_' + k]).mean())
        if HAS_MASK:
            b = np.load(os.path.join(WO, 'iou_%s_%s_%s.npy' % (DS, BK, k)))
            repro['iou_' + k + '_maxabs'] = float(np.abs(b - arrs['iou_' + k]).max())
        if DS == 'cub':
            b = np.load(os.path.join(WO, 'curve_%s_%s_%s.npy' % (DS, BK, k)))
            repro['curve_' + k + '_maxabs'] = float(np.abs(b - arrs['curve_' + k]).max())
log('reproduction:', json.dumps(repro))
log('classification:', json.dumps(clsq))
cs = {k: dict(n=int(np.isfinite(v).sum()), mean_sep=float(np.nanmean(1 - np.array(v))),
              median_sep=float(np.nanmedian(1 - np.array(v))),
              q25=float(np.nanpercentile(1 - np.array(v), 25)),
              q75=float(np.nanpercentile(1 - np.array(v), 75)))
      for k, v in corr.items() if k != 'ln_rec' and len(v)}
log('eval-record separation (1-corr):', json.dumps(cs))
json.dump(dict(dataset=DS, backbone=BK, n_records=n_rec, n_eval=len(recs), tau=TAU,
               bank_idx=bank_idx, train_loss=dict(gap=loss_gap, lse_refit=loss_rf,
                                                  **({'lnpost': loss_ln} if IS_VIT else {})),
               certificate=cert, classification=clsq, eval_separation=cs,
               reproduction=repro, elapsed_s=time.time() - t00),
          open(os.path.join(GOUT, 'res_%s.json' % TAG), 'w'), indent=1, default=W._jd)
log('saved', TAG)
