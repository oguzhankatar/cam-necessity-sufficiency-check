"""H5 -- OUT-OF-SAMPLE test of the price regression found in H4.

H4 searched 1- and 2-subsets of ten structural features on stage 5's 13 blocked
architectures and found `f_nnorm + f_expand` at LOO-R2 = +0.806 (permutation
p = 0.005).  A search at n = 13 is exactly the thing a reader distrusts, so
this script does the only convincing test: FIT on those 13, PREDICT the price of
the blocked architectures discovered in the held-out splits, PRINT the
predictions, and only then read the measured prices.

Models: the 9 mean-pool blocked architectures found in splits A and B.
`volo_d1_224` is excluded because it is AGG-BLOCKED -- no head edit reaches it,
so it has no price.  Prices are measured by stage 5's own
`scripts/scope/s3_edits.py` on the full 50 000-image val set (tag
h5_edits_heldout), not re-implemented here.
"""
import os, sys, json
import numpy as np
import torch
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlib as H

SC = os.path.join(H.ROOT, 'outputs', 'scope')
P4 = json.load(open(os.path.join(H.OUT, 'h4_price.json')))
NEW = json.load(open(os.path.join(SC, 'h5_edits_heldout.json')))
H1 = {r['model']: r for s in 'AB'
      for r in json.load(open(os.path.join(H.OUT, 'h1_%s.json' % s)))
      if 'agree' in r}

PAIR = ['f_nnorm', 'f_expand']          # pre-specified from H4
ALT = ['m_linr2']                       # the cheap measured alternative

# ---------------------------------------------------------------- fit on 13
R = P4['rows']
y = np.array([r['y'] for r in R])


def fit(feats):
    X = np.column_stack([np.ones(len(R))] + [[r[f] for r in R] for f in feats])
    m = ~np.isnan(X).any(1)
    b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
    return b, int(m.sum())


B_PAIR, nfit = fit(PAIR)
B_ALT, nfit2 = fit(ALT)
print('fitted on %d stage-5 blocked models' % nfit)
print('  price ~ %+.3f %+.3f * n_post_pool_norms %+.3f * log2(expansion)'
      % (B_PAIR[0], B_PAIR[1], B_PAIR[2]))
print('  price ~ %+.3f %+.3f * linact_r2   (fitted on %d)'
      % (B_ALT[0], B_ALT[1], nfit2))

# ------------------------------------------------- features for the new models
WIDE = {}
for r in NEW:
    nm = r['model']
    if 'cfgs' not in r:
        continue
    m = timm.create_model(nm, pretrained=True).eval()
    mods = dict(m.named_modules())
    ws = []
    for n, cls, kind in H1[nm]['post']:
        mo = mods.get(n)
        w = getattr(mo, 'weight', None)
        if w is not None and w.dim() >= 2 and int(w.shape[0]) != 1000:
            ws.append(int(w.shape[0]))
    WIDE[nm] = ws
    del m

rows = []
for r in NEW:
    nm = r['model']
    if 'cfgs' not in r:
        print('  SKIP %s: %s' % (nm, r.get('error', '?')[:80]))
        continue
    h = H1[nm]
    C = r['C']
    hid = max(WIDE[nm]) if WIDE[nm] else C
    d = dict(model=nm, f_nnorm=float(sum(1 for p in h['post'] if p[2] == 'NORM')),
             f_expand=float(np.log2(hid / C)), C=C, T=r['T'],
             m_linr2=float(min(r['linact_r2'].values())) if r.get('linact_r2') else np.nan,
             blocker=h['blocker'])
    d['pred_pair'] = float(B_PAIR[0] + B_PAIR[1] * d['f_nnorm']
                           + B_PAIR[2] * d['f_expand'])
    d['pred_alt'] = float(B_ALT[0] + B_ALT[1] * d['m_linr2'])
    rows.append(d)

print('\nPREDICTIONS (made before reading any measured price):')
print('%-24s %6s %8s %8s %10s %10s' % ('model', 'nnorm', 'expand', 'linr2',
                                       'pred(pair)', 'pred(linr2)'))
for d in rows:
    print('%-24s %6d %+8.2f %8.3f %+10.2f %+10.2f'
          % (d['model'], int(d['f_nnorm']), d['f_expand'], d['m_linr2'],
             d['pred_pair'], d['pred_alt']))

# ------------------------------------------------------------- now measure
NEWD = {r['model']: r for r in NEW}
for d in rows:
    r = NEWD[d['model']]
    reach = r.get('reaching_edits') or []
    d['reaching_edits'] = reach
    d['best_edit'] = r.get('best_edit')
    d['y'] = (float(r['best_cost_pts']) if r.get('best_cost_pts') is not None
              else (max(r['cfgs'][e]['dacc_pts'] for e in reach) if reach else np.nan))
    d['acc_shipped'] = r['cfgs']['shipped']['acc']

yy = np.array([d['y'] for d in rows])
print('\nMEASURED, and the comparison:')
print('%-24s %10s %10s %10s %10s' % ('model', 'pred(pair)', 'pred(linr2)',
                                     'measured', 'err(pair)'))
for d in rows:
    print('%-24s %+10.2f %+10.2f %+10.2f %+10.2f'
          % (d['model'], d['pred_pair'], d['pred_alt'], d['y'],
             d['pred_pair'] - d['y']))


def score(pred, obs):
    pred, obs = np.asarray(pred), np.asarray(obs)
    m = ~(np.isnan(pred) | np.isnan(obs))
    pred, obs = pred[m], obs[m]
    ss = float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
    ra = np.argsort(np.argsort(pred)).astype(float)
    rb = np.argsort(np.argsort(obs)).astype(float)
    return dict(n=int(m.sum()), r2_oos=ss,
                pearson=float(np.corrcoef(pred, obs)[0, 1]),
                spearman=float(np.corrcoef(ra, rb)[0, 1]),
                mae=float(np.abs(pred - obs).mean()),
                rmse=float(np.sqrt(np.mean((pred - obs) ** 2))))


res = dict(fit_n=nfit, coef_pair=B_PAIR.tolist(), coef_alt=B_ALT.tolist(),
           pair=PAIR, alt=ALT, rows=rows,
           score_pair=score([d['pred_pair'] for d in rows], yy),
           score_alt=score([d['pred_alt'] for d in rows], yy),
           score_const=score([float(np.mean(y))] * len(rows), yy))
for k in ('score_pair', 'score_alt', 'score_const'):
    s = res[k]
    print('%-12s n=%d  out-of-sample R2 = %+.3f  Pearson %+.3f  Spearman %+.3f  '
          'MAE %.2f pts' % (k, s['n'], s['r2_oos'], s['pearson'], s['spearman'],
                            s['mae']))
H.save('h5_predict.json', res)
