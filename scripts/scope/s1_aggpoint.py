"""S1 -- locate each model's AGGREGATION POINT and list the ops that run after it.

The cross-check's rt4_scope.py tested affinity of `forward_head` along the segment
between two *forward_features* outputs.  That conflates two different things:

  * a nonlinearity that sits AFTER the spatial pool   -> really blocks
  * a nonlinearity that sits BEFORE the pool but after forward_features
    (VGG's ConvMlp pre_logits, for instance)          -> irrelevant, it is body

The project's claim is about additivity over CELLS AT THE POOL, so the test has
to start at the pool.  This script finds the pool by running the model with
hooks on every leaf module and looking for the first module whose output loses
the spatial/token axes, then records the execution order of everything after it.
Nothing is imported from the project.
"""
import os, sys, json, warnings, torch, torch.nn as nn
warnings.filterwarnings('ignore')
import timm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'outputs', 'scope')
NAMES = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'names.txt')).read().split()


def spatial_numel(t):
    """number of cells: product of all dims except batch and channel"""
    if not torch.is_tensor(t):
        return None
    if t.dim() == 4:
        return t.shape[2] * t.shape[3]
    if t.dim() == 3:
        return t.shape[1]          # (B,T,C) tokens
    if t.dim() == 2:
        return 1
    return None


def trace(m, x):
    ev = []
    hs = []

    def mk(name, mod):
        def f(mod, inp, out):
            i = inp[0] if len(inp) and torch.is_tensor(inp[0]) else None
            ev.append(dict(name=name, cls=type(mod).__name__,
                           in_shape=list(i.shape) if i is not None else None,
                           out_shape=list(out.shape) if torch.is_tensor(out) else None,
                           in_cells=spatial_numel(i),
                           out_cells=spatial_numel(out) if torch.is_tensor(out) else None))
        return f
    for name, mod in m.named_modules():
        if len(list(mod.children())) == 0:      # leaf
            hs.append(mod.register_forward_hook(mk(name, mod)))
    with torch.no_grad():
        m(x)
    for h in hs:
        h.remove()
    return ev


AFFINE = ('Linear', 'Conv2d', 'Conv1d', 'Identity', 'Flatten', 'Dropout',
          'BatchNorm1d', 'BatchNorm2d', 'SelectAdaptivePool2d',
          'AdaptiveAvgPool2d', 'AvgPool2d')
NORMISH = ('LayerNorm', 'LayerNorm2d', 'RmsNorm', 'RmsNorm2d', 'GroupNorm',
           'LayerNormExp2d', 'RmsNorm2d')


def classify(cls):
    if cls in NORMISH or (cls.endswith('Norm') and cls not in ('BatchNorm1d', 'BatchNorm2d')) \
            or cls.startswith('RmsNorm') or cls.startswith('LayerNorm'):
        return 'NORM'
    if cls in AFFINE:
        return 'AFFINE'
    return 'OTHER'


rows = []
for nm in NAMES:
    try:
        m = timm.create_model(nm, pretrained=False).eval().cuda()
        sz = m.pretrained_cfg.get('input_size', (3, 224, 224))
        x = torch.randn(1, *sz, device='cuda')
        ev = trace(m, x)
        # the aggregation point: LAST event whose input has >1 cell and output has 1
        idx = None
        for i, e in enumerate(ev):
            if e['in_cells'] is not None and e['out_cells'] is not None \
                    and e['in_cells'] > 1 and e['out_cells'] == 1:
                idx = i
        post = ev[idx + 1:] if idx is not None else []
        pool_ev = ev[idx] if idx is not None else None
        # is there a functional (non-module) pool?  detect: no module collapses,
        # yet the final logits are 2-D -> pooling done by a tensor op
        rows.append(dict(model=nm, n_ev=len(ev),
                         pool=None if pool_ev is None else
                         dict(name=pool_ev['name'], cls=pool_ev['cls'],
                              cells=pool_ev['in_cells']),
                         post=[dict(name=e['name'], cls=e['cls'],
                                    kind=classify(e['cls'])) for e in post],
                         last_cells=[e['in_cells'] for e in ev[-6:]]))
        p = rows[-1]
        print(f"{nm:38s} pool={'-' if pool_ev is None else pool_ev['cls']+'/'+str(pool_ev['in_cells'])+'cells'}"
              f"  post=[{' '.join(e['cls'] for e in post)}]", flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f'{nm:38s} SKIP {type(e).__name__}: {str(e)[:70]}', flush=True)
        rows.append(dict(model=nm, error=f'{type(e).__name__}: {e}'))

os.makedirs(OUT, exist_ok=True)
json.dump(rows, open(os.path.join(OUT, 's1_aggpoint.json'), 'w'), indent=1)
print('\nwrote s1_aggpoint.json')
