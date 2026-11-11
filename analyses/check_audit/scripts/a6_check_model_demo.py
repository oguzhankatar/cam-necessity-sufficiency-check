"""Run scripts/check_model.py (audited verdict + pattern rules) on the
named timm models with random weights, CPU, and record the outputs.  Also reports the
footprint of the max-over-tokens and unlisted-normalisation rules on the archived census records (timm 1.0.27), which store
the operator-trace classes and the labelled tails.

usage: python a6_check_model_demo.py   ->  ../out/check_model_demo.log
"""
import os, sys, json, warnings, collections, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import torch, timm
from check_model import check_model

MODELS = ['coat_tiny', 'volo_d1_224', 'convnext_zepto_rms', 'resnet50', 'convnext_tiny',
          'vit_base_patch16_224', 'vgg11']
torch.manual_seed(0)
lines = ['check_model.py demo  (timm %s, torch %s, CPU, random weights)' % (timm.__version__, torch.__version__),
         'vgg11 is an extra negative control for the class-token-stack rule (dense aggregator on a single feature map).', '']
rows = {}
for name in MODELS:
    t0 = time.time()
    m = timm.create_model(name, pretrained=False)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        r = check_model(m)
    r['python_warnings'] = [str(w.message) for w in W]
    rows[name] = r
    lines.append('%-22s verdict_check %-22s verdict %-22s rules %s  (%.0fs)'
                 % (name, r['verdict_check'], r['verdict'], r['rules_fired'] or '-', time.time() - t0))
    lines.append('    aggregator %s [%s]; blocking_op %s' % (r['aggregator']['op'], r['aggregator']['category'], r['blocking_op']))
    for w in r['python_warnings']:
        lines.append('    WARNING: ' + w)
    del m

# footprint on the census records
C1 = os.path.join(ROOT, 'analyses', 'check_census', 'out')
R = json.load(open(os.path.join(C1, 'c5_census.json')))['records']
ok = {a: r for a, r in R.items() if 'verdict' in r}
r1 = sorted(a for a, r in ok.items() if 'OTHER' in r.get('aten_classes', []) and r['verdict'] != 'AGG-BLOCKED')
r3 = sorted(a for a, r in ok.items() if r['verdict'] == 'NONLIN-BLOCKED'
            and all('Norm' in p[1] and not p[1].startswith('Batch') for p in r['post'] if p[2] == 'NONLINEAR'))
dense = sorted(a for a, r in ok.items() if r['agg'] == 'DENSE')
lines += ['', 'Footprint on the %d census records (timm 1.0.27):' % len(ok),
          '  max-over-tokens rule (non-linear cell-axis reduction in the operator trace, verdict not aggregator-blocked): %d  %s' % (len(r1), r1),
          '  unlisted-normalisation rule (all nonlinear-labelled tail ops are unlisted normalisations): %d  %s' % (len(r3), r3),
          '  class-token-stack rule candidates (dense aggregator): %d  %s; the class-token-stack rule also needs a list of token sequences from forward_features, '
          'which only the CoaT models return (vgg11 above: not fired).' % (len(dense), dense)]
txt = '\n'.join(lines) + '\n'
print(txt)
OUT = os.path.join(HERE, '..', 'out')
open(os.path.join(OUT, 'check_model_demo.log'), 'w', encoding='utf-8').write(txt)
json.dump(dict(rows=rows, footprint=dict(max_over_tokens=r1, unlisted_normalisation=r3, dense_aggregator=dense)),
          open(os.path.join(OUT, 'check_model_demo.json'), 'w'), indent=1, default=str)
