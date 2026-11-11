"""K1: conservative (sound) structural check, implemented as a wrapper.

Protocol: protocols/K1_sound_check_protocol.md (fixed before any K1 verdict).
Nothing here edits scripts/heldout/frozen_rule.py, scripts/heldout/rule_v2.py or
scripts/check_model.py; all three are imported read-only.

check_model_sound(model, x) -> dict with 'verdict' in
    degenerate | normalisation-blocked | nonlinearity-blocked | aggregator-blocked |
    class-token | unknown
Degenerate is returned only if conditions 1-5 of the protocol all hold:
  1  a hooked module performs the spatial reduction, or the operator trace holds
     exactly one reduction over the spatial axes and it is a mean;
  2  the aggregator module's class is in the additive-pool whitelist, or the
     operator-level reduction is a plain mean;
  3  every module the head executes before the aggregator is a whitelisted per-cell
     module (runtime constraints: kernel 1 / stride 1 / no padding for convolutions,
     running statistics in eval mode for batch norm, normalised and linear axes equal
     to the channel axis for per-token norms and linear layers); a non-leaf module that
     completes before the aggregator is accepted only when it executes no operator of
     its own (pure container), so its children are what is checked;
  4  the reduced axes are the spatial axes of the feature map (from tensor shapes in
     the operator trace);
  5  the module-level verdict (check_model, audited) and the operator-level verdict
     (rule_v2 v2b) agree on degenerate.
Otherwise: the module-level verdict when it is not degenerate, else 'unknown'.

Reading of condition 1 used here (recorded in the outputs): "the spatial reduction"
is read as the single spatial reduction; in both branches the operator trace must
contain exactly one operator that removes the spatial axes, and in the module branch
that operator must run inside the aggregator module.

A post-hoc variant (op_guard=True), NOT part of the protocol and reported separately,
adds two operator-level conditions: (6) every non-view operator before the spatial
reduction runs inside a whitelisted per-cell leaf module; (7) every operator after it
is affine (fixed op list; a product or quotient of two batch-dependent tensors is not).
"""
import os
import sys
import json
import time
import warnings

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'heldout'))
import frozen_rule as FR            # noqa: E402  read-only
import rule_v2 as RV2               # noqa: E402  read-only
from check_model import check_model, PAPER_TERM   # noqa: E402  read-only

WL_PATH = os.path.join(HERE, '..', 'out', 'k1_whitelist.json')
_WL = None


def whitelists():
    global _WL
    if _WL is None:
        d = json.load(open(WL_PATH))
        _WL = dict(pool=set(d['pool_whitelist']), percell=set(d['percell_whitelist']),
                   container=set(d['container_whitelist']))
    return _WL


# ------------------------------------------------------------------ operator / module trace
VIEW = ('aten.view', 'aten._unsafe_view', 'aten.reshape', 'aten.clone', 'aten.flatten', 'aten.contiguous',
        'aten.expand', 'aten.permute', 'aten.transpose', 'aten.squeeze', 'aten.unsqueeze', 'aten.t.',
        'aten.detach', 'aten._to_copy', 'aten.to.', 'aten.alias', 'aten.lift_fresh', 'aten.copy_',
        'aten.empty', 'aten.zeros', 'aten.ones', 'aten.full', 'aten.scalar_tensor', 'aten.arange',
        'aten._reshape_alias', 'aten.as_strided', 'aten.split', 'aten.chunk', 'aten.view_as')
MEAN_OPS = ('aten.mean.dim', 'aten.avg_pool1d', 'aten.avg_pool2d', 'aten.adaptive_avg_pool1d',
            'aten.adaptive_avg_pool2d', 'aten._adaptive_avg_pool2d')
# post-hoc op guard: operators accepted as affine after the reduction
AFFINE_OPS = ('aten.addmm', 'aten.mm', 'aten.linear', 'aten.matmul', 'aten.bmm', 'aten.add.', 'aten.add_.',
              'aten.sub.', 'aten.mul.', 'aten.div.', 'aten.convolution', 'aten.cat', 'aten.stack',
              'aten._native_batch_norm_legit_no_training', 'aten.native_batch_norm', 'aten.batch_norm',
              'aten.cudnn_batch_norm', 'aten.neg', 'aten.mv', 'aten.baddbmm', 'aten.addmv', 'aten.dropout',
              'aten.native_dropout', 'aten.sum.dim', 'aten.mean.dim', 'aten.select', 'aten.slice')


def _shapes(xs):
    out = []
    for a in xs:
        if isinstance(a, torch.Tensor):
            out.append(tuple(a.shape))
        elif isinstance(a, (list, tuple)):
            out += [tuple(t.shape) for t in a if isinstance(t, torch.Tensor)]
    return out


def _dims(args):
    """integer / integer-list arguments after the first tensor (reduction dims)."""
    ds = []
    for a in args[1:]:
        if isinstance(a, bool):
            continue
        if isinstance(a, int):
            ds.append([a])
        elif isinstance(a, (list, tuple)) and a and all(isinstance(i, int) and not isinstance(i, bool) for i in a):
            ds.append(list(a))
    return ds


class Tracer(TorchDispatchMode):
    def __init__(s):
        super().__init__()
        s.ops, s.stack = [], []

    def __torch_dispatch__(s, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        s.ops.append(dict(i=len(s.ops), op=str(func), ins=_shapes(args), outs=_shapes([out]),
                          dims=_dims(args), mod=s.stack[-1] if s.stack else None,
                          stack=list(s.stack)))
        return out


def trace(m, f):
    """operator trace of forward_head, each op tagged with the module stack, plus module records."""
    tr = Tracer()
    recs, hs, open_ = [], [], {}
    for nm, mod in m.named_modules():
        if nm == '':
            continue
        leaf = len(list(mod.children())) == 0

        def pre(mod, inp, nm=nm, leaf=leaf):
            open_[nm] = len(recs)
            recs.append(dict(name=nm, cls=type(mod).__name__, leaf=leaf, start=len(tr.ops), end=None,
                             ins=_shapes(inp), outs=None))
            tr.stack.append(nm)

        def post(mod, inp, out, nm=nm):
            r = recs[open_.pop(nm)]
            r['end'] = len(tr.ops)
            r['outs'] = _shapes([out])
            if tr.stack and tr.stack[-1] == nm:
                tr.stack.pop()
        hs.append(mod.register_forward_pre_hook(pre))
        hs.append(mod.register_forward_hook(post))
    try:
        with torch.no_grad(), tr:
            m.forward_head(f)
    finally:
        for h in hs:
            h.remove()
    return tr.ops, recs


# ------------------------------------------------------------------ spatial signature
def spatial_patterns(m, f):
    """consecutive dim-size runs that identify the spatial cell axes of the feature map."""
    pats = set()
    ts = list(f) if isinstance(f, (list, tuple)) else [f]
    C = int(getattr(m, 'num_features', 0) or 0) or None
    npfx = int(getattr(m, 'num_prefix_tokens', 0) or 0)
    layout = None
    for t in ts:
        if t.dim() == 4:
            if C is not None and t.shape[3] == C and t.shape[1] != C:
                H, W = t.shape[1], t.shape[2]; layout = 'BHWC'
            else:
                H, W = t.shape[2], t.shape[3]; layout = 'BCHW'
            if H * W > 1:
                pats.add((H, W)); pats.add((H * W,))
        elif t.dim() == 3:
            T = t.shape[1]; layout = 'BTC' if layout is None else layout
            p = npfx if not isinstance(f, (list, tuple)) else 1
            for n in {T, T - p}:
                if n > 1:
                    pats.add((n,))
                    r = int(round(n ** 0.5))
                    if r * r == n:
                        pats.add((r, r))
    return pats, layout


def _find(shape, pats):
    """positions (dim indices, excluding the batch dim 0) at which a pattern occurs."""
    hits = []
    for p in pats:
        k = len(p)
        for i in range(1, len(shape) - k + 1):
            if tuple(shape[i:i + k]) == p:
                hits.append(tuple(range(i, i + k)))
    return hits


def has_cells(shape, pats):
    return bool(_find(shape, pats))


def is_view(op):
    return any(v in op for v in VIEW)


def spatial_reductions(ops, pats):
    """non-view operators whose tensor input carries the spatial axes and whose outputs do not."""
    red = []
    for o in ops:
        if is_view(o['op']) or not o['ins'] or not o['outs']:
            continue
        if any(has_cells(s, pats) for s in o['ins']) and not any(has_cells(s, pats) for s in o['outs']):
            red.append(o)
    return red


def _mean_over_spatial(o, pats):
    """condition 4 for an operator: a mean whose reduced dims are exactly a spatial run."""
    if not any(k in o['op'] for k in MEAN_OPS):
        return False
    x = o['ins'][0]
    hits = _find(x, pats)
    if not hits:
        return False
    if 'aten.mean.dim' in o['op']:
        if not o['dims']:
            return False
        d = sorted(i % len(x) for i in o['dims'][0])
        return any(tuple(d) == h for h in hits)
    # avg / adaptive-avg pooling: input spatial run is the trailing dims and the output is 1 cell
    out = o['outs'][0]
    tail = tuple(range(len(x) - 2, len(x))) if '2d' in o['op'] else (len(x) - 1,)
    return any(h == tail for h in hits) and all(out[i] == 1 for i in tail)


def percell_ok(mod, rec, pats, wl):
    """condition 3 for one module that completes before the aggregator."""
    c = type(mod).__name__
    if not rec['leaf']:
        if c in wl['container']:
            return True, 'container'
        return False, 'non-leaf module %s' % c
    if c not in wl['percell']:
        return False, 'class %s not in per-cell whitelist' % c
    ins = rec['ins'][0] if rec['ins'] else None
    outs = rec['outs'][0] if rec['outs'] else None
    if ins is None or outs is None:
        return False, 'no tensor input/output'
    if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
        ks = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size,)
        st = mod.stride if isinstance(mod.stride, tuple) else (mod.stride,)
        pd = mod.padding
        pd_ok = pd == 'valid' or (isinstance(pd, tuple) and all(p == 0 for p in pd)) or pd == 0
        ok = all(k == 1 for k in ks) and all(s == 1 for s in st) and pd_ok
        # the cells must be on the convolved (trailing) axes
        tail = tuple(range(len(ins) - len(ks), len(ins)))
        ok = ok and any(h == tail or set(h) <= set(tail) for h in _find(ins, pats))
        return ok, 'pointwise conv' if ok else 'conv not pointwise over the cell axes'
    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
        ok = (not mod.training) and mod.running_mean is not None
        return ok, 'batch norm eval' if ok else 'batch norm uses batch statistics'
    if isinstance(mod, nn.Linear):
        ok = ins[-1] == mod.in_features and has_cells(ins[:-1], pats)
        return ok, 'per-token linear' if ok else 'linear not over the channel axis'
    ns = getattr(mod, 'normalized_shape', None)
    if 'Norm' in c:
        if c.endswith('2d'):                      # LayerNorm2d / RmsNorm2d: normalise over dim 1 of BCHW
            ok = len(ins) == 4 and any(h == (2, 3) or h == (2,) for h in _find(ins, pats))
            return ok, 'channel norm (2d)' if ok else 'norm2d not on a BCHW cell map'
        if ns is not None:
            ns = tuple(ns) if isinstance(ns, (tuple, list)) else (ns,)
            ok = len(ns) == 1 and ins[-1] == ns[0] and has_cells(ins[:-1], pats)
            return ok, 'per-token norm' if ok else 'norm not over the channel axis only'
        return False, 'norm without normalized_shape'
    # activations, dropout, identity: elementwise
    ok = outs == ins
    return ok, 'elementwise' if ok else 'shape-changing elementwise module'


def _code_of(verdict):
    return {v: k for k, v in PAPER_TERM.items()}[verdict]


def check_model_sound(model, x, op_guard=False, return_timing=False):
    """Conservative structural check (protocol K1). See the module docstring."""
    wl = whitelists()
    was = model.training
    model.eval()
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            f = model.forward_features(x)
        # module-level check (audited); forward_features replaced by the cached output for this call
        had = 'forward_features' in model.__dict__
        old = model.__dict__.get('forward_features')
        model.forward_features = lambda _x, _f=f: _f
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cm = check_model(model, x=x, warn=False)
        finally:
            if had:
                model.forward_features = old
            else:
                del model.forward_features
        st2 = RV2.structural_v2(model, f)            # operator-level verdict (v2b); calls frozen_rule unchanged
        model.eval()
        ops, recs = trace(model, f)
        pats, layout = spatial_patterns(model, f)
    finally:
        model.train(was)
    mods = dict(model.named_modules())
    red = spatial_reductions(ops, pats)
    agg_name = cm['aggregator']['op']
    module_branch = not agg_name.startswith('functional(')
    cond = {}
    why = {}

    # condition 5
    cond[5] = cm['verdict'] == 'degenerate' and st2['v2b'] == 'IN-SCOPE'
    why[5] = 'module %s / operator %s' % (cm['verdict'], PAPER_TERM[st2['v2b']])

    agg_rec, agg_op = None, None
    if module_branch:
        nm = agg_name.split('(')[0]
        cand = [r for r in recs if r['name'] == nm and r['leaf']]
        agg_rec = cand[-1] if cand else None
        inside = [o for o in red if agg_rec is not None and agg_rec['start'] <= o['i'] < agg_rec['end']]
        cond[1] = agg_rec is not None and len(red) == 1 and len(inside) == 1
        why[1] = 'module %s; %d spatial reduction(s) in the trace, %d inside it' % (nm, len(red), len(inside))
        cond[2] = agg_rec is not None and agg_rec['cls'] in wl['pool']
        why[2] = 'aggregator class %s' % (agg_rec['cls'] if agg_rec else None)
        cond[4] = (agg_rec is not None and agg_rec['ins'] and agg_rec['outs'] and has_cells(agg_rec['ins'][0], pats)
                   and not has_cells(agg_rec['outs'][0], pats) and len(inside) == 1)
        why[4] = 'aggregator %s -> %s' % ((agg_rec['ins'][:1], agg_rec['outs'][:1]) if agg_rec else (None, None))
        start = agg_rec['start'] if agg_rec else None
        agg_op = inside[0] if inside else None
    else:
        agg_op = red[0] if len(red) == 1 else None
        cond[1] = agg_op is not None and any(k in agg_op['op'] for k in MEAN_OPS)
        why[1] = '%d spatial reduction(s): %s' % (len(red), [o['op'] for o in red][:4])
        cont = [r for r in recs if r['leaf'] and agg_op is not None and r['start'] <= agg_op['i'] < r['end']]
        cond[2] = cond[1] and 'aten.mean.dim' in agg_op['op'] and all(r['cls'] in wl['pool'] for r in cont)
        why[2] = 'operator %s%s' % (agg_op['op'] if agg_op else None,
                                    ' inside %s' % [r['cls'] for r in cont] if cont else '')
        cond[4] = agg_op is not None and _mean_over_spatial(agg_op, pats)
        why[4] = 'reduced %s of %s -> %s' % ((agg_op['dims'][:1], agg_op['ins'][:1], agg_op['outs'][:1])
                                            if agg_op else (None, None, None))
        start = agg_op['i'] if agg_op else None
    # condition 3
    bad = []
    if start is None:
        cond[3] = False
        why[3] = 'no aggregator located'
    else:
        before = [r for r in recs if r['end'] is not None and r['end'] <= start]
        for r in before:
            ok, msg = percell_ok(mods[r['name']], r, pats, wl)
            if r['leaf'] is False and ok:
                own = [o for o in ops if r['start'] <= o['i'] < r['end'] and o['mod'] == r['name'] and not is_view(o['op'])]
                if own:
                    ok, msg = False, 'container %s executes %d operator(s) of its own' % (r['cls'], len(own))
            if not ok:
                bad.append('%s(%s): %s' % (r['name'], r['cls'], msg))
        cond[3] = not bad
        why[3] = '%d module(s) before the aggregator; rejected: %s' % (len(before), bad[:3] or 'none')

    # post-hoc operator guard (not part of the protocol)
    if op_guard and start is not None and agg_op is not None:
        leafs = {r['name'] for r in recs if r['leaf']}
        okleaf = {r['name'] for r in recs if r['leaf'] and r['end'] is not None and r['end'] <= start
                  and percell_ok(mods[r['name']], r, pats, wl)[0]}
        pre_bad = [o['op'] for o in ops if o['i'] < start and not is_view(o['op'])
                   and not any(s in okleaf for s in o['stack'])]
        B = x.shape[0]

        def data(s):
            return len(s) >= 2 and s[0] == B
        post_bad = []
        for o in ops[agg_op['i'] + 1:]:
            if is_view(o['op']) or 'aten.dropout' in o['op']:
                continue
            if any(s in okleaf for s in o['stack']):
                continue
            if not any(k in o['op'] for k in AFFINE_OPS):
                post_bad.append(o['op'])
            elif any(k in o['op'] for k in ('aten.mul.', 'aten.div.', 'aten.bmm', 'aten.matmul', 'aten.mm')) \
                    and sum(data(s) for s in o['ins']) >= 2:
                post_bad.append(o['op'] + ' (data x data)')
        cond[6] = not pre_bad
        why[6] = 'operators before the reduction outside per-cell modules: %s' % (pre_bad[:4] or 'none')
        cond[7] = not post_bad
        why[7] = 'non-affine operators after the reduction: %s' % (post_bad[:4] or 'none')

    need = [1, 2, 3, 4, 5] + ([6, 7] if op_guard else [])
    all_ok = all(cond.get(k, False) for k in need)
    if cm['verdict'] != 'degenerate':
        verdict = cm['verdict']
    elif all_ok:
        verdict = 'degenerate'
    else:
        verdict = 'unknown'
    secs = time.perf_counter() - t0
    out = dict(verdict=verdict, module_verdict=cm['verdict'], module_verdict_check=cm['verdict_check'],
               operator_verdict=PAPER_TERM[st2['v2b']], conditions={str(k): bool(cond.get(k, False)) for k in need},
               failed=[k for k in need if not cond.get(k, False)], why={str(k): why.get(k) for k in need},
               layout=layout, patterns=sorted(pats), n_spatial_reductions=len(red),
               spatial_reduction_ops=[o['op'] for o in red][:6], aggregator=agg_name,
               branch='module' if module_branch else 'operator', op_guard=op_guard)
    if return_timing:
        out['secs'] = secs
    return out
