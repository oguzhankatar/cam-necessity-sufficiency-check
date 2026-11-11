"""S4 -- A DECISION PROCEDURE FOR "IS THIS MODEL IN SCOPE?", validated on 65 nets.

THE RULE (structural half).  Trace the leaf modules executed by `forward_head`
on a real feature map and find the operation that destroys the cell axis -- the
AGGREGATOR.  Then classify every leaf that runs after it:

    AFFINE     Linear, Conv2d (on a 1x1 map), BatchNorm (eval), Identity,
               Flatten, Dropout (eval), LearnableAffineBlock, avg-pool of 1 cell
    NORM       LayerNorm / RmsNorm / GroupNorm family -- nonlinear, but it is a
               PER-CELL map, so it can be moved to the other side of the mean
               (`pre`) or deleted (`none`); it never needs the pool moved
    NONLINEAR  anything else: a pointwise activation, an MLP, a gate

and the aggregator itself:

    MEAN       avg pool over the cells, or x.mean(dim) with global_pool='avg'
    DENSE      a linear map that reads all cells at once (VGG's 7x7 fc6): still
               additive by itself, since it is linear in the cells
    TOKEN      a class-token select (global_pool='token'): the readout does not
               read the cells at all, so the identity holds VACUOUSLY
    OTHER      max / catavgmax / anything not linear in the cells

    VERDICT
      AGG-BLOCKED     aggregator is OTHER                      (no head edit helps)
      DEGENERATE      aggregator is TOKEN                      (vacuously exact)
      IN-SCOPE        aggregator MEAN or DENSE, all post-ops AFFINE
      NORM-BLOCKED    ... + at least one NORM, no NONLINEAR     (one-line fix: pre/none)
      NONLIN-BLOCKED  ... + at least one NONLINEAR              (needs the pool moved)

THE VERIFICATION (numerical half), float64, pretrained weights, real images:
      dev0 = segment deviation of forward_head as shipped
      dev1 = the same after every NORM inside the head is replaced by Identity
    prediction: IN-SCOPE/DEGENERATE <=> dev0 = 0 ;  NORM-BLOCKED <=> dev0 > 0, dev1 = 0 ;
                NONLIN/AGG-BLOCKED  <=> dev0 > 0, dev1 > 0.
Every disagreement is printed.
"""
import os, sys, json, time, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib
import timm

TOL = 1e-11
NAMES = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'names.txt')).read().split()

AFFINE_CLS = ('Linear', 'Identity', 'Flatten', 'Dropout', 'BatchNorm1d',
              'BatchNorm2d', 'LearnableAffineBlock', 'Conv2d', 'Conv1d')
POOL_AVG = ('SelectAdaptivePool2d', 'AdaptiveAvgPool2d', 'AvgPool2d',
            'FastAdaptiveAvgPool', 'AdaptiveAvgPool1d')


def cells_of(t):
    if not torch.is_tensor(t):
        return None
    if t.dim() == 4:
        return t.shape[2] * t.shape[3]
    if t.dim() == 3:
        return t.shape[1]
    return 1 if t.dim() == 2 else None


def trace_head(m, f):
    """leaf modules executed by forward_head, with their cell counts"""
    ev, hs = [], []

    def mk(nm, mod):
        def fn(mod, inp, out):
            if len(inp) and torch.is_tensor(inp[0]) and torch.is_tensor(out):
                ev.append(dict(name=nm, cls=type(mod).__name__,
                               ic=cells_of(inp[0]), oc=cells_of(out)))
        return fn
    for nm, mod in m.named_modules():
        if len(list(mod.children())) == 0:
            hs.append(mod.register_forward_hook(mk(nm, mod)))
    with torch.no_grad():
        m.forward_head(f)
    for h in hs:
        h.remove()
    return ev


def kind(mod):
    c = type(mod).__name__
    if slib.is_norm(mod):
        return 'NORM'
    if slib.is_act(mod):
        return 'NONLINEAR'
    if c in AFFINE_CLS or c in POOL_AVG:
        return 'AFFINE'
    return 'NONLINEAR'


def structural(m, f):
    ev = trace_head(m, f)
    mods = dict(m.named_modules())
    agg_i = None
    for i, e in enumerate(ev):
        if e['ic'] and e['oc'] == 1 and e['ic'] > 1:
            agg_i = i
    if agg_i is None:
        # the pool is a tensor op (x.mean(1) / x.mean((1,2)) / x[:, 0]).  Look for
        # a declared pool type on the model or on its head.
        gp = None
        for owner in (m, getattr(m, 'head', None)):
            if owner is None:
                continue
            v = getattr(owner, 'global_pool', None)
            if isinstance(v, str) and v:
                gp = v
                break
            pt = getattr(v, 'pool_type', None)
            if isinstance(pt, str) and pt:
                gp = pt
                break
        agg = ('TOKEN' if isinstance(gp, str) and 'token' in gp else
               'OTHER' if isinstance(gp, str) and ('max' in gp or 'cat' in gp)
               else 'MEAN')          # default: timm's functional pool is a mean
        agg_name = 'functional(global_pool=%s)' % gp
        post = ev
    else:
        e = ev[agg_i]
        c = e['cls']
        agg = ('MEAN' if c in POOL_AVG else
               'DENSE' if c in ('Conv2d', 'Linear', 'Conv1d') else 'OTHER')
        if c in POOL_AVG:
            pt = getattr(mods[e['name']], 'pool_type', 'avg')
            if isinstance(pt, str) and ('max' in pt or 'cat' in pt):
                agg = 'OTHER'
        agg_name = '%s(%s)' % (e['name'], c)
        post = ev[agg_i + 1:]
    kinds = [(p['name'], p['cls'], kind(mods[p['name']])) for p in post]
    nl = [k for k in kinds if k[2] == 'NONLINEAR']
    nm_ = [k for k in kinds if k[2] == 'NORM']
    if agg == 'OTHER':
        v, blk = 'AGG-BLOCKED', agg_name
    elif agg == 'TOKEN':
        v, blk = 'DEGENERATE', None
    elif nl:
        v, blk = 'NONLIN-BLOCKED', '%s(%s)' % (nl[0][0], nl[0][1])
    elif nm_:
        v, blk = 'NORM-BLOCKED', '%s(%s)' % (nm_[0][0], nm_[0][1])
    else:
        v, blk = 'IN-SCOPE', None
    return dict(agg=agg, agg_name=agg_name, post=kinds, verdict=v, blocker=blk,
                n_norm=len(nm_), n_nonlin=len(nl))


def set_mod(m, n, new):
    parent = m
    parts = n.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def numeric(m, x, post_names):
    """d0 = shipped; d1 = after the POST-AGGREGATION norms only are removed.

    Killing every Norm in the whole model (rt4_scope's `kill_norms` on the head
    plus the fallback) can change or destroy the body, which is not the
    intervention under test and returns NaN on several models.
    """
    with torch.no_grad():
        f = m.forward_features(x)
    d0, _ = slib.seg_dev(lambda v: m.forward_head(v), f[0:1], f[1:2])
    mods = dict(m.named_modules())
    killed = [n for n in post_names if n in mods and slib.is_norm(mods[n])]
    for n in killed:
        set_mod(m, n, nn.Identity())
    with torch.no_grad():
        f = m.forward_features(x)
    d1, _ = slib.seg_dev(lambda v: m.forward_head(v), f[0:1], f[1:2])
    for n in killed:
        set_mod(m, n, mods[n])
    return d0, d1, killed


def predict_from_numeric(d0, d1):
    if d0 < TOL:
        return 'AFFINE-SHIPPED'
    return 'NORM-ONLY' if d1 < TOL else 'HARD-BLOCKED'


EXPECT = {'IN-SCOPE': 'AFFINE-SHIPPED', 'DEGENERATE': 'AFFINE-SHIPPED',
          'NORM-BLOCKED': 'NORM-ONLY', 'NONLIN-BLOCKED': 'HARD-BLOCKED',
          'AGG-BLOCKED': 'HARD-BLOCKED'}

rows = []
for nmm in NAMES:
    rec = dict(model=nmm)
    try:
        m = timm.create_model(nmm, pretrained=True).eval().cuda()
        cfg = m.pretrained_cfg
        x = slib.val_images(cfg, [7, 4211]).cuda()
        with torch.no_grad():
            f = m.forward_features(x)
        st = structural(m, f)
        rec.update(st)
        try:
            m = m.double()
            x = x.double()
        except Exception:
            pass
        d0, d1, killed = numeric(m, x, [q[0] for q in st['post']])
        rec.update(dev_shipped=d0, dev_nonorm=d1, norms_killed=killed,
                   dtype=str(next(m.parameters()).dtype))
        rec['numeric'] = predict_from_numeric(d0, d1)
        rec['agree'] = bool(EXPECT[st['verdict']] == rec['numeric'])
        print('%-38s %-15s blocker=%-34s d0=%.2e d1=%.2e  %-14s %s'
              % (nmm, st['verdict'], str(st['blocker'])[:34], d0, d1,
                 rec['numeric'], 'OK' if rec['agree'] else '*** DISAGREE ***'),
              flush=True)
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        rec['error'] = '%s: %s' % (type(e).__name__, e)
        rec['tb'] = traceback.format_exc()[-1200:]
        print('%-38s SKIP %s' % (nmm, rec['error'][:90]), flush=True)
        print(rec['tb'], flush=True)
    rows.append(rec)

ok = [r for r in rows if 'agree' in r]
print('\nAGREEMENT: %d/%d' % (sum(r['agree'] for r in ok), len(ok)))
from collections import Counter
for k, v in Counter(r['verdict'] for r in ok).most_common():
    print('  %3d  %s' % (v, k))
print('\nDISAGREEMENTS:')
for r in ok:
    if not r['agree']:
        print('  %s: structural=%s numeric=%s blocker=%s d0=%.2e d1=%.2e'
              % (r['model'], r['verdict'], r['numeric'], r['blocker'],
                 r['dev_shipped'], r['dev_nonorm']))
slib.save('s4_decide.json', rows)
