"""H2 -- diagnose every held-out disagreement, one model at a time.

For each failing architecture this prints
  * what the frozen rule saw (the module-level trace it walked, the aggregator it
    chose and why),
  * the aten-level op trace of `forward_head`, which is what it did NOT see,
  * an ablation that isolates the responsible operation, certified with the
    literal float64 loop.
"""
import os, sys, json, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlib as H
import frozen_rule as FR
import rtlib
import timm
from torch.utils._python_dispatch import TorchDispatchMode

IDX = [7, 4211, 15123, 3007, 40555, 21001]


class OpLog(TorchDispatchMode):
    def __init__(s):
        s.ops = []

    def __torch_dispatch__(s, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ins = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        outs = ([tuple(out.shape)] if isinstance(out, torch.Tensor) else
                [tuple(o.shape) for o in out if isinstance(o, torch.Tensor)]
                if isinstance(out, (list, tuple)) else [])
        s.ops.append(dict(op=str(func), ins=ins, outs=outs))
        return out


def aten_trace(m, f):
    log = OpLog()
    with torch.no_grad(), log:
        m.forward_head(f)
    return log.ops


def load(nm):
    m = timm.create_model(nm, pretrained=True).eval().cuda()
    x = H.val_images(m.pretrained_cfg, IDX).cuda()
    m = m.double(); x = x.double()
    with torch.no_grad():
        f = m.forward_features(x)
    return m, x, f


def cert(m, x, f):
    cv = H.CellView(m, f)
    head = H.make_head(m, cv)
    with torch.no_grad():
        z = m(x)
    return H.certify(head, cv.cells(f), z.argmax(1).tolist(), cv.npfx), cv


out = {}
FAILS = json.load(open(os.path.join(H.OUT, 'h1_A.json')))
FAILS = [r['model'] for r in FAILS if 'agree' in r and not r['agree']]
print('disagreements on split A: %s\n' % FAILS)

for nm in FAILS:
    print('=' * 78)
    print(nm)
    rec = dict(model=nm)
    m, x, f = load(nm)
    st = FR.structural(m, f)
    rec['frozen'] = dict(agg=st['agg'], agg_name=st['agg_name'],
                         verdict=st['verdict'], post=[list(p) for p in st['post']])
    print('  frozen rule: aggregator = %s (%s), verdict = %s'
          % (st['agg'], st['agg_name'], st['verdict']))
    print('  module-level post-agg ops: %s' % [p[1] for p in st['post']])
    ev = FR.trace_head(m, f)
    rec['module_trace'] = ev
    print('  module-level leaves executed by forward_head (name/cls/in-cells/out-cells):')
    for e in ev:
        print('     %-28s %-16s %s -> %s' % (e['name'][:28], e['cls'], e['ic'], e['oc']))
    ops = aten_trace(m, f)
    rec['aten_trace'] = ops
    print('  aten-level ops executed by forward_head:')
    for o in ops:
        print('     %-42s %s -> %s' % (o['op'][:42], o['ins'][:2], o['outs'][:2]))
    red = [o for o in ops if any(k in o['op'] for k in
                                 ('max.dim', 'amax', 'min.dim', 'amin', 'topk',
                                  'sort', 'median', 'max_pool'))]
    rec['nonlinear_reductions'] = red
    print('  NON-LINEAR REDUCTIONS in the aten trace: %s' % [o['op'] for o in red])

    sh, cv = cert(m, x, f)
    rec['shipped'] = sh
    print('  shipped: maxNS=%.3e Rhalf=%.3e I2=%.3e  additive=%s'
          % (sh['maxNS'], sh['Rhalf'], sh['I2'], sh['additive']))

    # ---- ablation: remove the auxiliary max path if the model has one
    if getattr(m, 'aux_head', None) is not None:
        keep = m.aux_head
        m.aux_head = None
        with torch.no_grad():
            f2 = m.forward_features(x)
        ab, _ = cert(m, x, f2)
        rec['ablate_aux_head'] = ab
        print('  aux_head removed (the .max(1) path): maxNS=%.3e Rhalf=%.3e I2=%.3e '
              'additive=%s spatial_mass=%.2e'
              % (ab['maxNS'], ab['Rhalf'], ab['I2'], ab['additive'], ab['spatial_mass']))
        m.aux_head = keep
    out[nm] = rec
    del m
    torch.cuda.empty_cache()

H.save('h2_diagnose.json', out)
