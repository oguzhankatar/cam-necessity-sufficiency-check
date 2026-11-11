"""FROZEN round-5 decision procedure.  DO NOT EDIT.

Auto-extracted verbatim by scripts/heldout/h0_freeze.py from
scripts/scope/slib.py and scripts/scope/s4_decide.py.  Every block below is the
exact source text of the corresponding round-5 object; the per-block sha256 is
in outputs/heldout/h0_freeze.json.  The only lines this file adds are the
imports and this docstring.

Round-5 result on the 66-architecture development set: 65/65 agreement.
"""
import sys as _sys
import torch
import torch.nn as nn


NORMISH = ('LayerNorm', 'LayerNorm2d', 'LayerNormExp2d', 'RmsNorm', 'RmsNorm2d',
           'GroupNorm', 'InstanceNorm2d', 'LayerNormFp32', 'RmsNormFp32')

ACTISH = ('ReLU', 'ReLU6', 'GELU', 'QuickGELU', 'SiLU', 'Swish', 'Hardswish',
          'HardSwish', 'Hardsigmoid', 'HardSigmoid', 'Tanh', 'Sigmoid', 'Mish',
          'ELU', 'LeakyReLU', 'PReLU', 'SquaredReLU', 'StarReLU', 'GELUTanh',
          'Hardtanh', 'SELU', 'CELU', 'Softplus')

def is_norm(mod):
    n = type(mod).__name__
    return n in NORMISH or (n.endswith('Norm') and not n.startswith('Batch')) \
        or n.startswith('RmsNorm') or n.startswith('LayerNorm')

def is_act(mod):
    return type(mod).__name__ in ACTISH

slib = _sys.modules[__name__]

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
