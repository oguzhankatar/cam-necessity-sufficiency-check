"""The original, audited and conservative structural checks of one model, side by side.

    python scripts/check_conservative.py resnet50                   # timm name, random weights
    python scripts/check_conservative.py resnet18 --torchvision     # torchvision name, random weights
    python scripts/check_conservative.py convnext_tiny --certify    # also the float64 certificate

    import sys; sys.path.insert(0, 'scripts')
    import timm
    from check_conservative import check_conservative
    check_conservative(timm.create_model('convnext_tiny'))
    # -> {'original': 'normalisation-blocked', 'audited': 'normalisation-blocked',
    #     'conservative': 'normalisation-blocked', 'operation': 'head.norm(LayerNorm2d)', ...}

Three verdicts, in the article's terms:

  original      verdict_check of check_model.py: the check as fixed
                (scripts/heldout/frozen_rule.py, imported unchanged)
  audited       verdict of check_model.py: the original verdict after the three pattern rules
                derived from the census audit and the one held-out disagreement (VOLO-D1)
  conservative  verdict of k1lib.check_model_sound (protocol protocols/K1_sound_check_protocol.md),
                imported unchanged from the folder of the conservative-check analysis (K1DIR
                below): the audited verdict when it is not degenerate; degenerate only when the
                five conditions of the protocol all hold; otherwise 'unknown'

The conservative verdict is the one to rely on. 'unknown' means: do not infer from the
structure, measure instead. certify() (or --certify) runs the float64 additivity certificate
of the article on the model's head, check_audit_lib.measure (C6DIR below), whose per-cell loop is
rtlib.NS_literal (scripts/crosscheck/rtlib.py), as the census
certification did: random-weight features and a unit-scale N(0,1) feature map of the same
shape, the unit-scale label where the two differ.

The model must expose forward_features and forward_head (the timm convention). A torchvision
classifier is exposed that way by torchvision_model() (k1_tv.TV of the same analysis folder);
pass an instance (e.g. with pretrained weights) to wrap it instead of a random one. The
verdicts depend on the module structure only, so random weights give the same verdicts.
Requirements: scripts/requirements-check.txt (CPU suffices; torchvision for --torchvision).
"""
import copy
import os
import sys
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
K1DIR = os.path.join(ROOT, 'analyses', 'conservative_check_census', 'scripts')   # k1lib.py, k1_tv.py
C6DIR = os.path.join(ROOT, 'analyses', 'check_audit', 'scripts')    # check_audit_lib.py (certificate)
for _d in (C6DIR, K1DIR, HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)
from check_model import check_model   # noqa: E402  read-only
import k1lib as K1                    # noqa: E402  read-only; imports check_model, frozen_rule, rule_v2

# measured label -> the verdict it is expected for
EXPECTED = {'ADDITIVE': 'degenerate', 'VACUOUS': 'class-token', 'NORM-ONLY': 'normalisation-blocked',
            'HARD-BLOCKED': 'nonlinearity- or aggregator-blocked', 'ZERO-READOUT': None}


def torchvision_model(name, model=None):
    """torchvision classifier `name` with timm's forward_features / forward_head split (k1_tv.TV).
    Random weights unless `model`, an instance of the same architecture, is given."""
    import k1_tv
    torch.manual_seed(0)
    tv = k1_tv.TV(name)
    if model is not None:
        tv.m = model
    return tv


def _input(model, batch, dtype=None, seed=0):
    cfg = getattr(model, 'pretrained_cfg', None) or {}
    size = tuple(cfg.get('input_size', (3, 224, 224)))
    p = next(iter(model.parameters()), None)
    dtype = dtype or (p.dtype if p is not None else torch.float32)
    x = torch.randn(batch, *size, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    return x.to(p.device) if p is not None else x


def check_conservative(model, x=None):
    """The three verdicts of `model` (an nn.Module with forward_features / forward_head).

    x: input batch for forward_features; by default two N(0,1) images at the model's default
       input size (as in the census run). Returns a dict:
      original, audited, conservative   the three verdicts ('unknown' only for conservative)
      operation          the operation the verdict names: the first blocking operation for a
                         blocked audited verdict, else the aggregator
      operation_role     'blocking' or 'aggregator'
      aggregator         the aggregator the check located, and its category
      rules_fired        pattern rules that changed the audited verdict
      operator_verdict   the operator-level verdict used by condition 5
      conditions_failed  protocol conditions that failed (why: one line per condition)
    """
    if x is None:
        x = _input(model, 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cm = check_model(model, x=x, warn=False)
        s = K1.check_model_sound(model, x)
    blk = cm['blocking_op']
    return dict(original=cm['verdict_check'], audited=cm['verdict'], conservative=s['verdict'],
                operation=blk if blk else cm['aggregator']['op'],
                operation_role='blocking' if blk else 'aggregator',
                aggregator=cm['aggregator'], rules_fired=cm['rules_fired'],
                operator_verdict=s['operator_verdict'], conditions_failed=s['failed'],
                why=s['why'], tail=cm['tail'], notes=cm['notes'])


def certify(model, tail=None):
    """float64 certificate of the head (check_audit_lib.measure), on a deep copy of `model`: six N(0,1) images
    (three pairs), eps1-eps3 < 1e-9 for additive; once on the model's own features and once on a
    unit-scale N(0,1) feature map of the same shape. label is the unit-scale label where the two
    differ (the census rule). A readout that is identically zero (classifier initialised to zero)
    is re-certified after drawing the all-zero weight tensors from N(0, 0.02^2), as in the census."""
    import check_audit_lib as L
    if tail is None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tail = check_model(model, warn=False)['tail']
    m = copy.deepcopy(model).eval().double()
    mods = dict(m.named_modules())
    norms = [p[0] for p in tail if p[0] in mods and L.is_norm_name(mods[p[0]])]
    x = _input(m, 6, dtype=torch.float64)
    out = dict(norms_deleted=norms)
    for attempt in range(2):
        with torch.no_grad():
            f = m.forward_features(x)
        cv = L.cells_for_timm(m, f)
        ship, nonorm, lab = L.measure(m, f, cv, norms)
        g = torch.Generator().manual_seed(1)
        fu = ([torch.randn(t.shape, generator=g, dtype=t.dtype) for t in f] if isinstance(f, (list, tuple))
              else torch.randn(f.shape, generator=g, dtype=f.dtype))
        ship_u, nonorm_u, lab_u = L.measure(m, fu, cv, norms)
        out.update(label_own=lab, label_unit=lab_u, label=lab_u,
                   own=dict(eps1=ship['eps1'], eps2=ship['eps2'], eps3=ship['eps3'], absN_max=ship['absN_max']),
                   unit=dict(eps1=ship_u['eps1'], eps2=ship_u['eps2'], eps3=ship_u['eps3'],
                             absN_max=ship_u['absN_max']))
        if not (ship['absN_max'] == 0.0 and ship_u['absN_max'] == 0.0):
            break
        out['label'] = 'ZERO-READOUT'
        g0 = torch.Generator().manual_seed(0)
        rz = []
        with torch.no_grad():
            for pn, pp in m.named_parameters():
                if pp.dim() >= 2 and not bool(pp.any()):
                    pp.copy_(torch.randn(pp.shape, generator=g0, dtype=pp.dtype) * 0.02)
                    rz.append(pn)
        if not rz:
            break
        out['reinitialised'] = rz
    out['expected_verdict'] = EXPECTED.get(out['label'])
    return out


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser(description='original, audited and conservative structural checks of one model')
    ap.add_argument('model', nargs='?', default='resnet50', help='timm model name (torchvision name with --torchvision)')
    ap.add_argument('--torchvision', action='store_true', help='a torchvision classifier, random weights')
    ap.add_argument('--pretrained', action='store_true',
                    help='timm: load pretrained weights (not needed: the verdicts are structural)')
    ap.add_argument('--certify', action='store_true', help='also run the float64 certificate (slower)')
    ap.add_argument('--json', action='store_true', help='print the full result as JSON')
    a = ap.parse_args()
    if a.torchvision:
        m, src = torchvision_model(a.model), 'torchvision, random weights'
    else:
        import timm
        torch.manual_seed(0)
        m = timm.create_model(a.model, pretrained=a.pretrained)
        src = 'timm, %s weights' % ('pretrained' if a.pretrained else 'random')
    r = check_conservative(m.eval())
    if a.certify:
        r['certificate'] = certify(m, r['tail'])
    if a.json:
        print(json.dumps(dict(model=a.model, source=src, **r), indent=1, default=str))
        sys.exit(0)
    print('%s (%s)' % (a.model, src))
    print('  original      (verdict_check)      %s' % r['original'])
    print('  audited       (verdict)            %s%s' % (r['audited'], '   rules fired: %s' % ', '.join(r['rules_fired'])
                                                        if r['rules_fired'] else ''))
    print('  conservative  (check_model_sound)  %s' % r['conservative'])
    print('  operation     (%s)%s %s' % (r['operation_role'], ' ' * (18 - len(r['operation_role'])), r['operation']))
    if r['conservative'] == 'unknown':
        print('  unknown: the structure does not decide; measure instead (--certify)')
        for k in map(str, r['conditions_failed']):
            print('    condition %s failed: %s' % (k, r['why'][k]))
    if a.certify:
        c = r['certificate']
        print('  certificate   (float64)            %s (own features %s, unit-scale %s); expected for: %s'
              % (c['label'], c['label_own'], c['label_unit'], c['expected_verdict']))
        print('                eps1 / eps2 / eps3 own %.1e / %.1e / %.1e, unit-scale %.1e / %.1e / %.1e%s'
              % (c['own']['eps1'], c['own']['eps2'], c['own']['eps3'], c['unit']['eps1'], c['unit']['eps2'],
                 c['unit']['eps3'], '; re-drawn zero weights: %s' % c['reinitialised'] if 'reinitialised' in c else ''))
