"""K1 step 0: whitelists from the module classes of the 66 development architectures only.

Run before any K1 verdict. Every module class of the 66 development models
(scripts/heldout/names_D.txt; timm 1.0.27, random weights, structure only) is listed;
the whitelists keep the classes of those lists that fall in the protocol's categories:
  pool      adaptive / global average pooling and their timm wrappers
  per-cell  pointwise convolutions (plain Conv1d/Conv2d; kernel checked at run time),
            batch normalisation (checked to be in eval mode with running statistics),
            activations, dropout, identity, per-token norms (LayerNorm, LayerNorm2d,
            RmsNorm, RmsNorm2d; axis checked at run time), per-token linear layers
  container plain nn.Sequential (accepted only when it runs no operator of its own)
Output: ../out/k1_whitelist.json
"""
import os, sys, json, re, collections
import timm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(HERE, '..', 'out'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'heldout'))
import frozen_rule as FR   # read-only, for the activation name table

names = open(os.path.join(ROOT, 'scripts', 'heldout', 'names_D.txt')).read().split()
assert len(names) == 66
cls_models = collections.defaultdict(set)
leafness = {}
for n in names:
    m = timm.create_model(n, pretrained=False)
    for _, mod in m.named_modules():
        c = type(mod).__name__
        cls_models[c].add(n)
        leafness.setdefault(c, set()).add(len(list(mod.children())) == 0)
    del m
dev = sorted(cls_models)

POOL_RE = re.compile(r'^(AdaptiveAvgPool[12]d|FastAdaptiveAvgPool|SelectAdaptivePool2d|GlobalAvgPool\w*)$')
CONV = {'Conv1d', 'Conv2d'}
BN = {'BatchNorm1d', 'BatchNorm2d'}
ACT = set(FR.ACTISH)
DROP = {'Dropout', 'Dropout1d', 'Dropout2d'}
IDENT = {'Identity'}
TOKNORM = {'LayerNorm', 'LayerNorm2d', 'RmsNorm', 'RmsNorm2d'}
LIN = {'Linear'}
pool = [c for c in dev if POOL_RE.match(c)]
percell = [c for c in dev if c in CONV | BN | ACT | DROP | IDENT | TOKNORM | LIN]
container = [c for c in dev if c == 'Sequential']
res = dict(source='scripts/heldout/names_D.txt (66 development architectures), timm %s' % timm.__version__,
           dev_module_classes=dev, n_dev_classes=len(dev),
           pool_whitelist=pool, percell_whitelist=percell, container_whitelist=container,
           models_per_class={c: len(cls_models[c]) for c in pool + percell + container})
json.dump(res, open(os.path.join(OUT, 'k1_whitelist.json'), 'w'), indent=1)
print('dev classes', len(dev))
print('pool', pool)
print('percell', percell)
print('container', container)
