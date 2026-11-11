"""H4 (P4) -- does ANYTHING predict the repair price?

Stage 5: "the price runs from -0.06 to -18.52 points, with no feature of the
architecture that predicts which."  That is an assertion about a regression
nobody ran.  This script runs it.

TARGET  y = `best_cost_pts`, the top-1 change of the cheapest edit that actually
reaches additivity, for the 13 architectures stage 5 found blocked
(outputs/scope/s3_edits.json + s3v_vgg16.json), recomputed here from those JSONs.

PREDICTORS, in two blocks, because they are different claims:
  (S) STRUCTURAL -- readable off the architecture with no data:
      T, log2 C, number of post-pool leaves, number of activations, number of
      norms, the relative POSITION of the first blocking activation in the
      post-pool op list, expansion ratio of the post-pool hidden layer,
      aggregator = DENSE, blocker saturating, blocker is a gate.
  (M) CHEAP MEASURED -- one forward pass over a few thousand images:
      the R^2 of the per-channel least-squares affine fit to the blocking
      activation itself (`linact_r2`, already in the stage-5 JSONs), the shipped
      Rhalf / maxNS / corrNS, the shipped top-1, and the affine-surrogate gap
      from s8_linfit.

METHOD, for n = 13:
  * univariate Pearson and Spearman with exact-ish permutation p-values
    (20 000 label permutations) and Benjamini-Hochberg over the whole family;
  * leave-one-out cross-validated R^2 against the intercept-only model, for
    every single predictor and every pair;
  * a permutation test on the BEST LOO-R^2 found, which is the only honest way
    to report a search over predictors at this n.
Everything is written to outputs/heldout/h4_price.json.
"""
import os, sys, json, itertools, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlib as H

ROOT = H.ROOT
SC = os.path.join(ROOT, 'outputs', 'scope')
S3 = json.load(open(os.path.join(SC, 's3_edits.json')))
S3V = json.load(open(os.path.join(SC, 's3v_vgg16.json')))
S4 = {r['model']: r for r in json.load(open(os.path.join(SC, 's4_decide.json')))}
S8 = {r['model']: r for r in json.load(open(os.path.join(SC, 's8_linfit.json')))
      if 'res' in r}
S8X = os.path.join(H.OUT, 'h4_linfit_extra.json')
if os.path.exists(S8X):
    for r in json.load(open(S8X)):
        if 'res' in r:
            S8[r['model']] = r

SAT = ('Tanh', 'Hardswish', 'HardSwish', 'Sigmoid', 'Hardsigmoid',
       'HardSigmoid', 'Hardtanh')
GATEWORDS = ('gate', 'se.', '.se', 'squeeze', 'attn')

# ---------------------------------------------------- post-pool hidden widths
WID = os.path.join(H.OUT, 'h4_widths.json')
if not os.path.exists(WID):
    import torch, timm
    widths = {}
    names = [r['model'] for r in S3] + ['vgg16']
    for nm in names:
        m = timm.create_model(nm, pretrained=True).eval()
        mods = dict(m.named_modules())
        post = [p[0] for p in S4[nm]['post']]
        ws = []
        for n in post:
            mo = mods.get(n)
            w = getattr(mo, 'weight', None)
            if w is not None and w.dim() >= 2:
                ws.append(int(w.shape[0]))
        widths[nm] = ws
        del m
    json.dump(widths, open(WID, 'w'), indent=1)
    print('-> %s' % WID)
WIDTHS = json.load(open(WID))


def row_of(r, dense=False):
    nm = r['model']
    cfg = r['cfgs']['shipped']
    post = r.get('post_leaves') or [p[0] for p in S4[nm]['post']]
    acts = r['acts']
    ipos = min(post.index(a) for a in acts if a in post) if acts else 0
    C = r['C']
    ws = [w for w in WIDTHS.get(nm, []) if w != 1000]
    hid = max(ws) if ws else C
    blk = str(S4[nm]['blocker'] or '')
    bcls = blk.split('(')[-1].rstrip(')') if '(' in blk else ''
    y = (r.get('best_cost_pts') if 'best_cost_pts' in r else
         max(r['cfgs'][e]['dacc_pts'] for e in r['reaching_edits']))
    lin = S8.get(nm)
    d = dict(
        model=nm, y=float(y), C=int(C),
        blocker=blk, blocker_cls=bcls,
        f_T=float(r['T']), f_logC=float(np.log2(C)),
        f_npost=float(len(post)), f_nact=float(len(acts)),
        f_nnorm=float(len(r.get('norms', []))),
        f_pos=float(ipos) / max(1.0, len(post) - 1),
        f_expand=float(np.log2(hid / C)),
        f_dense=1.0 if dense else 0.0,
        f_sat=1.0 if bcls in SAT else 0.0,
        f_gate=1.0 if any(g in blk.lower() for g in GATEWORDS) else 0.0,
        m_linr2=float(min(r['linact_r2'].values())) if r.get('linact_r2') else np.nan,
        m_logRhalf=float(np.log10(cfg['Rhalf'])),
        m_maxNS=float(cfg.get('max_abs_diff_rel', cfg.get('maxNS'))),
        m_corrmin=float(cfg['corrNS_min']),
        m_acc=float(cfg['acc']),
        m_lingap=float(lin['res']['holdout']['dacc_pts']) if lin else np.nan,
    )
    return d


ROWS = [row_of(r) for r in S3] + [row_of(S3V[0], dense=True)]
ROWS.sort(key=lambda d: d['y'])
FEATS = [k for k in ROWS[0] if k.startswith(('f_', 'm_'))]
y = np.array([d['y'] for d in ROWS])
print('n = %d   y range %.2f .. %.2f  mean %.2f' % (len(y), y.min(), y.max(), y.mean()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def loo_r2(X, y):
    """LOO-CV R^2 of OLS with intercept, relative to the LOO intercept-only model."""
    n = len(y)
    X1 = np.column_stack([np.ones(n), X])
    err, err0 = [], []
    for i in range(n):
        k = np.arange(n) != i
        try:
            b = np.linalg.lstsq(X1[k], y[k], rcond=None)[0]
        except Exception:
            return -np.inf
        err.append((X1[i] @ b - y[i]) ** 2)
        err0.append((y[k].mean() - y[i]) ** 2)
    return float(1.0 - np.sum(err) / np.sum(err0))


rs = np.random.RandomState(0)
NPERM = 20000
uni = []
for f in FEATS:
    v = np.array([d[f] for d in ROWS])
    m = ~np.isnan(v)
    vv, yy = v[m], y[m]
    if vv.std() == 0:
        uni.append(dict(feat=f, n=int(m.sum()), constant=True))
        continue
    r = float(np.corrcoef(vv, yy)[0, 1])
    rho = spearman(vv, yy)
    pr = pp = 0
    for _ in range(NPERM):
        z = rs.permutation(yy)
        pr += abs(float(np.corrcoef(vv, z)[0, 1])) >= abs(r) - 1e-15
        pp += abs(spearman(vv, z)) >= abs(rho) - 1e-15
    uni.append(dict(feat=f, n=int(m.sum()), pearson=r, p_pearson=(pr + 1) / (NPERM + 1),
                    spearman=rho, p_spearman=(pp + 1) / (NPERM + 1),
                    loo_r2=loo_r2(vv[:, None], yy), constant=False))

# Benjamini-Hochberg over the univariate Spearman p-values
live = [u for u in uni if not u.get('constant')]
ps = sorted((u['p_spearman'], u['feat']) for u in live)
mB = len(ps)
crit = {}
for i, (p, f) in enumerate(ps, 1):
    crit[f] = p * mB / i
run = np.inf
for i in range(len(ps) - 1, -1, -1):
    run = min(run, crit[ps[i][1]])
    crit[ps[i][1]] = min(1.0, run)
for u in live:
    u['q_bh'] = crit[u['feat']]

print('\n%-12s %4s %8s %8s %8s %8s %8s %8s' %
      ('feature', 'n', 'pearson', 'p', 'spearman', 'p', 'q(BH)', 'LOO-R2'))
for u in sorted(live, key=lambda d: d['p_spearman']):
    print('%-12s %4d %+8.3f %8.4f %+8.3f %8.4f %8.4f %+8.3f'
          % (u['feat'], u['n'], u['pearson'], u['p_pearson'], u['spearman'],
             u['p_spearman'], u['q_bh'], u['loo_r2']))

# ---------------------------------------------------- pairs, structural only
def subsets(names):
    out = []
    for k in (1, 2):
        for cb in itertools.combinations(names, k):
            V = np.column_stack([[d[f] for d in ROWS] for f in cb])
            m = ~np.isnan(V).any(1)
            if m.sum() < 8 or any(V[m][:, j].std() == 0 for j in range(V.shape[1])):
                continue
            out.append((cb, loo_r2(V[m], y[m]), int(m.sum())))
    out.sort(key=lambda t: -t[1])
    return out


SF = [f for f in FEATS if f.startswith('f_')]
MF = [f for f in FEATS if f.startswith('m_')]
best_S = subsets(SF)
best_M = subsets(FEATS)
print('\nbest LOO-R2, STRUCTURAL-only predictors (1 or 2 at a time):')
for cb, r2, n in best_S[:6]:
    print('  %-28s n=%2d  LOO-R2 = %+.3f' % ('+'.join(cb), n, r2))
print('best LOO-R2, ALL predictors:')
for cb, r2, n in best_M[:6]:
    print('  %-28s n=%2d  LOO-R2 = %+.3f' % ('+'.join(cb), n, r2))

# permutation test on the BEST searched LOO-R2 (accounts for the search)
def best_over(names, yv, idx):
    b = -np.inf
    for k in (1, 2):
        for cb in itertools.combinations(names, k):
            V = np.column_stack([[ROWS[i][f] for i in idx] for f in cb])
            if np.isnan(V).any() or any(V[:, j].std() == 0 for j in range(V.shape[1])):
                continue
            b = max(b, loo_r2(V, yv))
    return b


perm = {}
for tag, names in (('structural', SF), ('all', FEATS)):
    V = np.column_stack([[d[f] for d in ROWS] for f in names])
    m = ~np.isnan(V).any(1)
    idx = np.where(m)[0]
    obs = best_over(names, y[idx], idx)
    rs2 = np.random.RandomState(1)
    NP = 400
    cnt = sum(best_over(names, rs2.permutation(y[idx]), idx) >= obs for _ in range(NP))
    perm[tag] = dict(n=int(len(idx)), best_loo_r2=obs, nperm=NP,
                     p=(cnt + 1) / (NP + 1))
    print('permutation test (%s, n=%d): best searched LOO-R2 = %+.3f, p = %.4f'
          % (tag, len(idx), obs, perm[tag]['p']))

H.save('h4_price.json', dict(rows=ROWS, feats=FEATS, univariate=uni,
                             best_structural=[dict(feats=list(c), loo_r2=r, n=n)
                                              for c, r, n in best_S[:12]],
                             best_all=[dict(feats=list(c), loo_r2=r, n=n)
                                       for c, r, n in best_M[:12]],
                             perm=perm, nperm_univariate=NPERM))
