"""THE REPAIR (P2).  Stage-5 rule + an aten-level resolution of the aggregator.

WHAT FAILED, exactly.  `frozen_rule.structural` finds the aggregator by hooking
leaf MODULES and taking the last one whose input has >1 cell and whose output has
1.  An aggregation performed by a TENSOR OP is invisible to that trace, and when
no module collapses the cell axis stage 5 fell back to reading the model's
declared `global_pool` string.  That string names ONE path.  `volo_d1_224`'s
`forward_head` is

    out = head(x[:, 0])  +  0.5 * aux_head(x[:, 1:]).max(1)[0]

-- a token select AND a max over the 196 patch tokens, in parallel.  The module
trace sees `head_drop(197->197)`, `head(1->1)`, `aux_head(196->196)`: nothing
collapses, so the rule read `global_pool='token'` and returned DEGENERATE.  The
ground truth is maxNS = 4.69e-01: not additive, and not vacuous either.

WHAT IS ADDED.  Nothing is removed and no stage-5 line is edited.  The aten-level
op trace of `forward_head` is recorded and used to override the aggregator:

  v2a  MINIMAL -- if the trace contains a NON-LINEAR CELL-AXIS REDUCTION
       (max.dim / amax / min.dim / amin / topk / sort / median / max_pool /
       adaptive_max_pool applied to a tensor with ndim >= 3), the aggregator is
       OTHER and the verdict is AGG-BLOCKED, whatever `global_pool` says.
       This is the smallest change that fixes the observed failure.

  v2b  PRINCIPLED -- when no leaf module collapses the cell axis, the aggregator
       is read off the aten trace instead of off `global_pool`: every executed op
       that reduces a >=3-dim tensor by one axis is classified MEAN / TOKEN /
       DENSE / OTHER from a fixed table, and the verdict takes the worst
       (OTHER > MEAN/DENSE > TOKEN).  v2b implies v2a.

Both are reported.  The stage-5 module-level logic is reused verbatim through
`frozen_rule`.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import frozen_rule as FR

# op-name -> aggregator class.  Matched as a substring of str(func).
RED_OTHER = ('aten.max.dim', 'aten.amax', 'aten.min.dim', 'aten.amin',
             'aten.topk', 'aten.sort', 'aten.median', 'aten.max_pool',
             'aten.adaptive_max_pool', 'aten.logsumexp')
RED_MEAN = ('aten.mean', 'aten.sum', 'aten.avg_pool', 'aten.adaptive_avg_pool')
RED_TOKEN = ('aten.select.int', 'aten.index_select')
RED_DENSE = ('aten.mm', 'aten.addmm', 'aten.bmm', 'aten.matmul',
             'aten.convolution', 'aten.linear')
# reshape-like ops that change ndim without reducing anything
VIEWLIKE = ('view', 'reshape', 'clone', 'flatten', 'contiguous', 'expand',
            'permute', 'transpose', 'squeeze', 'unsqueeze', 'slice', 'aten.t.',
            'detach', '_to_copy', 'aten.to.', 'alias', 'unbind', 'split')


class OpLog(TorchDispatchMode):
    def __init__(s):
        s.ops = []

    def __torch_dispatch__(s, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ins = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        if isinstance(out, torch.Tensor):
            outs = [tuple(out.shape)]
        elif isinstance(out, (list, tuple)):
            outs = [tuple(o.shape) for o in out if isinstance(o, torch.Tensor)]
        else:
            outs = []
        s.ops.append(dict(op=str(func), ins=ins, outs=outs))
        return out


def aten_trace(m, f):
    log = OpLog()
    with torch.no_grad(), log:
        m.forward_head(f)
    return log.ops


def _reduces(o):
    """does this op take a tensor with ndim>=3 and return one with fewer dims?"""
    if not o['ins'] or not o['outs']:
        return False
    ni = max(len(s) for s in o['ins'])
    no = max(len(s) for s in o['outs'])
    return ni >= 3 and no < ni


def classify_trace(ops):
    """-> (set of aggregator classes found, the op strings that produced them)"""
    found, why = set(), []
    for o in ops:
        nm = o['op']
        if any(v in nm for v in VIEWLIKE):
            continue
        if any(k in nm for k in RED_OTHER) and _reduces(o):
            found.add('OTHER'); why.append(('OTHER', nm, o['ins'][:1], o['outs'][:1]))
        elif any(k in nm for k in RED_MEAN) and _reduces(o):
            found.add('MEAN'); why.append(('MEAN', nm, o['ins'][:1], o['outs'][:1]))
        elif any(k in nm for k in RED_TOKEN) and _reduces(o):
            found.add('TOKEN'); why.append(('TOKEN', nm, o['ins'][:1], o['outs'][:1]))
        elif any(k in nm for k in RED_DENSE) and _reduces(o):
            found.add('DENSE'); why.append(('DENSE', nm, o['ins'][:1], o['outs'][:1]))
        elif _reduces(o):
            why.append(('UNKNOWN', nm, o['ins'][:1], o['outs'][:1]))
    return found, why


VERDICT_OF = {'OTHER': 'AGG-BLOCKED', 'TOKEN': 'DEGENERATE'}


def _verdict_from_agg(agg, st):
    """stage-5's own verdict table, with the aggregator replaced."""
    if agg == 'OTHER':
        return 'AGG-BLOCKED'
    if agg == 'TOKEN':
        return 'DEGENERATE'
    if st['n_nonlin']:
        return 'NONLIN-BLOCKED'
    if st['n_norm']:
        return 'NORM-BLOCKED'
    return 'IN-SCOPE'


def structural_v2(m, f):
    """Returns the stage-5 record plus v2a / v2b verdicts and the evidence."""
    st = FR.structural(m, f)
    ops = aten_trace(m, f)
    found, why = classify_trace(ops)
    st['aten_classes'] = sorted(found)
    st['aten_evidence'] = [list(w) for w in why]
    st['v1'] = st['verdict']

    # --- v2a: minimal.  a non-linear cell-axis reduction anywhere -> OTHER
    if 'OTHER' in found:
        st['v2a'] = 'AGG-BLOCKED'
        st['v2a_agg'] = 'OTHER'
        st['v2a_blocker'] = [w[1] for w in why if w[0] == 'OTHER'][0]
    else:
        st['v2a'] = st['v1']
        st['v2a_agg'] = st['agg']
        st['v2a_blocker'] = st['blocker']

    # --- v2b: principled.  when no MODULE collapsed the cells, read the
    #     aggregator off the aten trace instead of off `global_pool`.
    module_pool = not st['agg_name'].startswith('functional(')
    if 'OTHER' in found:
        agg2 = 'OTHER'
    elif module_pool:
        agg2 = st['agg']
    elif 'MEAN' in found:
        agg2 = 'MEAN'
    elif 'DENSE' in found:
        agg2 = 'DENSE'
    elif 'TOKEN' in found:
        agg2 = 'TOKEN'
    else:
        agg2 = st['agg']          # nothing resolvable: keep stage 5's answer
    st['v2b_agg'] = agg2
    st['v2b'] = _verdict_from_agg(agg2, st)
    st['module_pool'] = module_pool
    return st
