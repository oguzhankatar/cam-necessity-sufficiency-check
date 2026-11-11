"""W6 -- the control that decides how to read the ViT result.

    python w6_prenorm.py <cub|voc12seg|voc12|coco17>

On the ViT the LayerNorm-relocation model beats the collapsed model by a wide
margin.  Two things changed at once:

  (a) the model left the degenerate point, so N and S became two different maps;
  (b) the maps are now drawn on the PRE-norm tokens instead of the post-norm ones.

This script isolates (b).  It builds the third ordering the project already
names, `none`:

    z_c = <w_c, mean_t z_t> + b_c        on the PRE-norm tokens, no LayerNorm

which is **additive** -- certified here by rtlib's literal float64 loop -- and
therefore has exactly ONE map, `collapsed_pre`, with N == S.  If `collapsed_pre`
is as good as `N_ln` / `S_ln`, the ViT gain is (b), the features, and the
necessity / sufficiency split contributed nothing.

Nothing is shared with the main run except `wlib`; the metrics are computed by
the same `wlib` functions on the same 224x224 frame with the same thresholds.
"""
import os, sys, math, time, copy
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
P = W.PROTOCOL
R, NTHR = P['eval_res'], P['n_thresholds']
THR = np.arange(NTHR) / NTHR
t00 = time.time()


def log(*a):
    print('[%6.0fs]' % (time.time() - t00), *a, flush=True)


man = (W.cub_manifest() if DS == 'cub' else
       W.voc_manifest('box') if DS == 'voc12' else
       W.voc_manifest('seg') if DS == 'voc12seg' else W.coco_manifest())
ex = W.Extractor('vitb16')
assert ex.fmt == 'BTC'
K = man['n_class']
log('%s / vitb16 pre-norm control: fit=%d eval=%d K=%d'
    % (man['name'], len(man['fit']), len(man['ev']), K))

# ---- fit the `none` head on mean_t z_t of the PRE-norm tokens
Pq = np.zeros((len(man['fit']), ex.C), dtype=np.float32)
for i0, c1, c2 in ex.batch([r['path'] for r in man['fit']], bs=24, prenorm=True):
    Pq[i0:i0 + c1.shape[0]] = c2.mean(1).cpu().numpy()
X = torch.from_numpy(Pq).cuda()
mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
lin = nn.Linear(ex.C, K).cuda()
nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
if man['multilabel']:
    Y = torch.zeros(len(man['fit']), K).cuda()
    for i, r in enumerate(man['fit']):
        Y[i, r['labels']] = 1.0
    lf = nn.BCEWithLogitsLoss()
else:
    Y = torch.tensor([r['label'] for r in man['fit']]).cuda()
    lf = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400)
for e in range(400):
    opt.zero_grad(); loss = lf(lin((X - mu) / sd), Y); loss.backward(); opt.step(); sch.step()
Wt = (lin.weight.data / sd).double()
bt = (lin.bias.data - (lin.weight.data * mu / sd).sum(1)).double()
WG, BG = Wt.cpu().numpy(), bt.cpu().numpy()
log('fitted `none` head, final train loss %.4f' % float(loss))
del X, Pq

# ---- baseline bank: the SAME fit-split images as the main run
rs = np.random.RandomState(P['seed'])
bank_idx = sorted(rs.choice(len(man['fit']), P['n_bank'], replace=False).tolist())
for _, c1, c2 in ex.batch([man['fit'][i]['path'] for i in bank_idx],
                          bs=P['n_bank'], prenorm=True):
    BANK = c2.double().cpu().numpy()
BS = np.einsum('btc,kc->bkt', BANK, WG) + BG[None, :, None]

# ---- certificate: the `none` ordering must be ADDITIVE
cp = [man['fit'][i]['path'] for i in
      sorted(np.random.RandomState(P['seed'] + 1).choice(len(man['fit']), 6, False).tolist())]
for _, c1, c2 in ex.batch(cp, bs=6, prenorm=True):
    CERT = c2.double().cpu().numpy()
cert = W.certify_head(W.PoolHead(Wt.cuda(), bt.cuda(), 'gap'),
                      torch.from_numpy(CERT).cuda(), [0] * 6, 'collapsed_pre')
cert.pop('per_pair')
log('certificate collapsed_pre: ADDITIVE=%s maxNS=%.2e Rhalf=%.2e I2=%.2e'
    % (cert['ADDITIVE'], cert['maxNS'], cert['Rhalf'], cert['I2']))

MAPS = ['collapsed_pre', 'collapsed_pre_b3', 'cam_pre']
HAS_MASK = DS in ('voc12seg', 'coco17')
curves = {k: [] for k in MAPS}
pt_s = {k: [] for k in MAPS}; pt_t = {k: [] for k in MAPS}
peak = {k: [] for k in MAPS}
pos_h = {k: np.zeros(NTHR) for k in MAPS}; neg_h = {k: np.zeros(NTHR) for k in MAPS}
acc = []
DIL = np.ones((2 * P['point_tol'] + 1, 2 * P['point_tol'] + 1), np.uint8)
POOL = ThreadPoolExecutor(8)
n_rec = 0
log('evaluation over %d images' % len(man['ev']))

for i0, c1, c2 in ex.batch([r['path'] for r in man['ev']], bs=24, prenorm=True):
    Xb = c2.double().cpu().numpy()
    for j in range(Xb.shape[0]):
        r = man['ev'][i0 + j]
        Xp = Xb[j]
        w0, h0 = Image.open(r['path']).size
        m_ = Xp.mean(0)
        z = m_ @ WG.T + BG
        tgts = [r['label']] if not man['multilabel'] else r['labels']
        acc.append(z.astype(np.float32) if man['multilabel']
                   else int(np.argmax(z) == r['label']))
        for c in tgts:
            a = Xp @ WG[c] + BG[c]
            mp = {'collapsed_pre': np.zeros(len(a)), 'collapsed_pre_b3': np.zeros(len(a))}
            for k in range(P['n_bank']):
                mp['collapsed_pre'] += W.ns_gap(a, BS[k, c])[0]
                mp['collapsed_pre_b3'] += W.ns_gap(a, BS[k, c], ex.H, ex.W, True)[0]
            mp['collapsed_pre'] /= P['n_bank']; mp['collapsed_pre_b3'] /= P['n_bank']
            mp['cam_pre'] = Xp @ WG[c]
            U = {k: W.upsample(mp[k], ex.H, ex.W, R) for k in MAPS}
            gb = None
            if not HAS_MASK:
                bs_ = r['boxes'] if 'boxes' in r else r['boxes_by_class'][str(c)]
                sx, sy = R / w0, R / h0
                gb = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in bs_]
            else:
                if DS == 'voc12seg':
                    seg = np.array(Image.open(r['mask']))
                    gm0 = (seg == c + 1).astype(np.uint8); gv0 = (seg == 255).astype(np.uint8)
                else:
                    co = man['_coco']; gm0 = None
                    for aid in r['ann_by_class'][str(c)]:
                        mm = co.annToMask(co.loadAnns([aid])[0])
                        gm0 = mm if gm0 is None else np.maximum(gm0, mm)
                    gv0 = np.zeros_like(gm0)
                gm = cv2.resize(gm0, (R, R), interpolation=cv2.INTER_NEAREST).astype(bool)
                gv = cv2.resize(gv0, (R, R), interpolation=cv2.INTER_NEAREST).astype(bool)
                valid = ~gv; gmd = cv2.dilate(gm.astype(np.uint8), DIL).astype(bool)
                gm_v = gm[valid]; npos = int(gm_v.sum())
            if DS == 'cub':
                for k, cvv in zip(MAPS, POOL.map(
                        lambda kk: W.box_curve(U[kk], gb, THR), MAPS)):
                    curves[k].append(cvv)
            for k in MAPS:
                u = U[k]
                pk = np.unravel_index(np.argmax(u), u.shape)
                if HAS_MASK:
                    pt_s[k].append(bool(gm[pk])); pt_t[k].append(bool(gmd[pk]))
                    b = np.clip((u[valid] * NTHR).astype(np.int64), 0, NTHR - 1)
                    tp = np.bincount(b[gm_v], minlength=NTHR)[::-1].cumsum()[::-1].astype(float)
                    fp = np.bincount(b[~gm_v], minlength=NTHR)[::-1].cumsum()[::-1].astype(float)
                    pos_h[k] += tp; neg_h[k] += fp
                    peak[k].append(float((tp / np.maximum(npos + fp, 1e-9)).max())
                                   if npos else 0.0)
                else:
                    pt_s[k].append(any(bb[0] <= pk[1] <= bb[2] and bb[1] <= pk[0] <= bb[3]
                                       for bb in gb))
                    pt_t[k].append(any(bb[0] - 15 <= pk[1] <= bb[2] + 15 and
                                       bb[1] - 15 <= pk[0] <= bb[3] + 15 for bb in gb))
            n_rec += 1
    if (i0 // 24) % 40 == 0:
        log('  %d/%d images' % (i0 + Xb.shape[0], len(man['ev'])))

out = dict(dataset=man['name'], backbone='vitb16_prenorm_none', ordering='none',
           n_eval=len(man['ev']), n_records=n_rec, certificate=dict(collapsed_pre=cert),
           maps=MAPS, code_sha=W.code_sha(),
           note=('GAP over the PRE-norm tokens with no LayerNorm at all (the '
                 'project `none` ordering).  Additive, so N == S and there is a '
                 'single map.  This isolates the feature change from the '
                 'necessity/sufficiency split.'))
if man['multilabel']:
    from sklearn.metrics import average_precision_score
    Z = np.stack(acc); Yv = np.zeros_like(Z)
    for i, r in enumerate(man['ev']):
        Yv[i, r['labels']] = 1
    out['classification'] = dict(mAP=float(np.mean(
        [average_precision_score(Yv[:, k], Z[:, k]) for k in range(K)
         if Yv[:, k].sum() > 0])))
else:
    out['classification'] = dict(top1=float(np.mean(acc)))
per = {}
for k in MAPS:
    e = {}
    if DS == 'cub':
        cv = np.stack(curves[k]).astype(np.float32)
        v, lo, hi = W.boot_mbav2(cv, n=P['n_boot_mbav2'], seed=P['seed'],
                                 deltas=tuple(P['deltas']))
        e['MaxBoxAccV2'] = dict(value=v, ci=[lo, hi])
        e['best_iou_mean'] = W.boot_mean(cv.max(1), seed=P['seed'])
    e['pointing_strict'] = W.boot_mean(np.array(pt_s[k], float), seed=P['seed'])
    e['pointing_tol15'] = W.boot_mean(np.array(pt_t[k], float), seed=P['seed'])
    if HAS_MASK:
        e['PxAP'] = W.pxap(pos_h[k], neg_h[k])
        e['peak_mask_iou'] = W.boot_mean(np.array(peak[k], float), seed=P['seed'])
    per[k] = e
    # the per-record arrays, under the names and the record order the main
    # run uses, so the control can be paired image by image against the
    # maps it is compared with
    np.save(os.path.join(W.OUT, 'pt_%s_vitb16_prenorm_%s.npy' % (DS, k)),
            np.stack([np.asarray(pt_s[k], float),
                      np.asarray(pt_t[k], float)]))
    if DS == 'cub':
        np.save(os.path.join(W.OUT, 'curve_%s_vitb16_prenorm_%s.npy' % (DS, k)),
                np.stack(curves[k]).astype(np.float32))
    if HAS_MASK:
        np.save(os.path.join(W.OUT, 'iou_%s_vitb16_prenorm_%s.npy' % (DS, k)),
                np.asarray(peak[k], float))
out['per_map'] = per
out['per_record_saved'] = sorted(MAPS)
np.save(os.path.join(W.OUT, 'recidx_%s_vitb16_prenorm.npy' % DS),
        np.array([r['img'] if isinstance(r, dict) and 'img' in r else i
                  for i, r in enumerate(man['ev'])]))
W.save('w6_prenorm_%s.json' % DS, out)
log('classification %s' % out['classification'])
for k in MAPS:
    log('%-18s %s' % (k, ' '.join(
        '%s=%.4f' % (n, v['value'] if isinstance(v, dict) else
                     v[0] if isinstance(v, (list, tuple)) else v)
        for n, v in per[k].items())))
