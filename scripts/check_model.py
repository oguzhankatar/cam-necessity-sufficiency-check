"""Apply the structural check to one model and report the verdict in the article's terms.

    python scripts/check_model.py resnet50                 # timm name, random weights
    python scripts/check_model.py convnext_tiny --pretrained

    import sys; sys.path.insert(0, 'scripts')
    import timm
    from check_model import check_model
    check_model(timm.create_model('vit_base_patch16_224'))
    # -> {'verdict': 'class-token', 'verdict_check': 'class-token', 'rules_fired': [], ...}

Two verdicts are returned.

  verdict_check  the check's own verdict: scripts/heldout/frozen_rule.structural, imported
                 unchanged (its SHA-256 is recorded in outputs/heldout/h0_freeze.json),
                 translated to the article's terms. This is the verdict evaluated in the
                 article (held-out architectures, census "as returned").
  verdict        the audited verdict: verdict_check after three documented pattern rules
                 for the check's known blind spots that occur in timm (below). Where a rule
                 fires, a warning names it; where none fires, verdict == verdict_check.

Code translation of the frozen check:

    frozen code       article's verdict
    IN-SCOPE          degenerate              (necessity equals sufficiency as distributed)
    NORM-BLOCKED      normalisation-blocked
    NONLIN-BLOCKED    nonlinearity-blocked
    AGG-BLOCKED       aggregator-blocked
    DEGENERATE        class-token             (NOT the article's "degenerate")

"degenerate" is a mathematical term here, not a defect: it means the head cannot separate
the two quantities, as for a global average pool followed by a linear layer.

Pattern rules (audited verdict). Each was established by measurement in the census audit
(analyses/check_audit/out/a1_audit.log; VOLO-D1 in analyses/check_census) and, on the timm 1.0.27 census
records, fires on exactly the architectures named:

  max-over-tokens rule
      A non-linear reduction over the cell/token axis (max, amax, min, topk, sort, median,
      max-pool, logsumexp on a tensor of rank >= 3) appears in the operator trace of
      forward_head while the check returned another verdict -> aggregator-blocked. VOLO adds
      0.5 * aux_head(x[:, 1:]).max(1) to the class-token logit, a tensor expression the
      module-level check does not see (VOLO-D1 measured not additive, eps1 = 0.469).
      Census: the 11 VOLO models.
  class-token-stack rule
      forward_features returns a list of k token sequences, the class token (index 0) of
      each is selected and stacked, and the layer the check took as the aggregator reduces
      that k-row stack, not the spatial cells -> class-token. CoaT's Conv1d(3, 1, 1) over
      three class tokens; no spatial cell reaches the logit (measured spatial mass 0).
      Census: coat_tiny, coat_mini, coat_small.
  unlisted-normalisation rule
      Every operation the check labelled nonlinear after the aggregator is a normalisation
      module absent from its tables (class name containing "Norm", not BatchNorm, e.g.
      SimpleNorm2d, an RMS normalisation) -> normalisation-blocked; deleting it makes the
      head additive (measured). Census: convnext_zepto_rms, convnext_zepto_rms_ols.

The rules widen what the wrapper recognises; they do not make the check complete. It still
sees leaf modules and, through the first two rules, the operator trace of forward_head only: a
cell-mixing operation applied before the aggregator, or an aggregation outside
forward_head, is not inspected (analyses/check_audit/out/a3_probes.md gives the measured error
envelope on constructed heads). The keys `declared_pool_branch` and `notes` flag the cases
in which the verdict rests on that envelope.

Requirements: scripts/requirements-check.txt (CPU suffices). The model must expose
forward_features and forward_head, as timm models do. The verdict depends only on the module
structure, so random weights give the same verdict as trained ones (the census in
analyses/check_census used random weights).
"""
import os
import re
import sys
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'heldout'))
import frozen_rule as FR            # noqa: E402  (read-only; never edited)
import rule_v2 as RV2               # noqa: E402  (operator trace; calls frozen_rule unchanged)

PAPER_TERM = {
    'IN-SCOPE': 'degenerate',
    'NORM-BLOCKED': 'normalisation-blocked',
    'NONLIN-BLOCKED': 'nonlinearity-blocked',
    'AGG-BLOCKED': 'aggregator-blocked',
    'DEGENERATE': 'class-token',
}

_NORM_NAME = re.compile(r'Norm')


def _default_input(model, batch=1):
    cfg = getattr(model, 'pretrained_cfg', None) or {}
    size = tuple(cfg.get('input_size', (3, 224, 224)))
    return torch.randn(batch, *size)


def _rule_aggregator(code, why):
    """max-over-tokens rule: a non-linear cell-axis reduction in the operator trace of forward_head."""
    other = [w for w in why if w[0] == 'OTHER']
    if not other or code == 'AGG-BLOCKED':
        return None
    op, ins, outs = other[0][1], other[0][2], other[0][3]
    return dict(rule='max-over-tokens rule', verdict='AGG-BLOCKED', blocking_op=op,
                warning='max-over-tokens rule fired: forward_head contains a non-linear reduction over the '
                        'cell/token axis (%s, %s -> %s) that the module-level check does not '
                        'see (VOLO pattern); audited verdict aggregator-blocked, the check '
                        'returned %s.' % (op, ins, outs, PAPER_TERM[code]))


def _rule_token_stack(m, f, st, ops):
    """class-token-stack rule: the 'aggregator' reduces a stack of class tokens, one per returned sequence."""
    if not isinstance(f, (list, tuple)) or st['agg'] != 'DENSE':
        return None
    k = len(f)
    cells = [FR.cells_of(t) for t in f]
    if k < 2 or any(c is None or c <= k for c in cells):
        return None
    ev = FR.trace_head(m, f)
    name = st['agg_name'].split('(')[0]
    ic = [e['ic'] for e in ev if e['name'] == name]
    if not ic or ic[-1] != k:
        return None
    sel = [o for o in ops if 'aten.select.int' in o['op'] and o['ins'] and o['outs']
           and len(o['ins'][0]) == 3 and len(o['outs'][0]) == 2 and o['ins'][0][1] > 1]
    means = [o for o in ops if any(r in o['op'] for r in RV2.RED_MEAN) and RV2._reduces(o)]
    if len(sel) < k or means:
        return None
    return dict(rule='class-token-stack rule', verdict='DEGENERATE', blocking_op=None,
                warning='class-token-stack rule fired: forward_features returned %d token sequences and the layer '
                        'taken as the aggregator (%s) reduces a stack of their %d class '
                        'tokens, not the spatial cells (CoaT pattern); audited verdict '
                        'class-token, the check returned %s.'
                        % (k, st['agg_name'], k, PAPER_TERM[st['verdict']]))


def _rule_norm(st):
    """unlisted-normalisation rule: every 'nonlinear' tail operation is an unlisted normalisation module."""
    if st['verdict'] != 'NONLIN-BLOCKED':
        return None
    nl = [p for p in st['post'] if p[2] == 'NONLINEAR']
    if not nl or not all(_NORM_NAME.search(p[1]) and not p[1].startswith('Batch') for p in nl):
        return None
    blk = '%s(%s)' % (nl[0][0], nl[0][1])
    return dict(rule='unlisted-normalisation rule', verdict='NORM-BLOCKED', blocking_op=blk,
                warning='unlisted-normalisation rule fired: the operation(s) the check labelled nonlinear (%s) are '
                        'normalisation modules missing from its tables (SimpleNorm/RMS '
                        'pattern); audited verdict normalisation-blocked, the check returned '
                        'nonlinearity-blocked.' % ', '.join('%s(%s)' % (p[0], p[1]) for p in nl))


def check_model(model, x=None, warn=True):
    """Structural check of `model`'s classification head.

    model: an nn.Module with forward_features(x) and forward_head(features) (timm convention).
    x:     an input batch for forward_features; a random image at the model's default input
           size when None. Only the module structure matters, not the values.
    warn:  emit a Python warning when a pattern rule changes the verdict.

    Returns a dict:
      verdict        audited verdict: 'degenerate' | 'normalisation-blocked' |
                     'nonlinearity-blocked' | 'aggregator-blocked' | 'class-token'
      verdict_check  the frozen check's own verdict, in the same terms
      rules_fired    names of the pattern rules that changed the verdict ([] if none)
      warnings       one sentence per fired rule
      blocking_op    the first operation that blocks under the audited verdict, the
                     aggregator for aggregator-blocked, None for degenerate and class-token
      raw_code       the frozen check's own code (see the module docstring)
      aggregator     the operation the check took as the aggregator, and its category
                     (MEAN, DENSE, TOKEN, OTHER)
      tail           [(module name, class, AFFINE|NORM|NONLINEAR)] after the aggregator,
                     as labelled by the check
      declared_pool_branch  True when no module reduced the cells and the aggregator was
                     read from the model's declared global_pool (default: a mean)
      notes          cases in which the verdict rests on the check's known blind spots
    """
    was_training = model.training
    model.eval()
    try:
        if x is None:
            p = next(iter(model.parameters()), None)
            x = _default_input(model)
            if p is not None:
                x = x.to(device=p.device, dtype=p.dtype)
        with torch.no_grad():
            f = model.forward_features(x)
        st = FR.structural(model, f)
        ops = RV2.aten_trace(model, f)
        _, why = RV2.classify_trace(ops)
        fired = [r for r in (_rule_aggregator(st['verdict'], why),
                             _rule_token_stack(model, f, st, ops),
                             _rule_norm(st)) if r is not None]
    finally:
        model.train(was_training)
    code = st['verdict']
    blocking = st['blocker'] if code in ('NORM-BLOCKED', 'NONLIN-BLOCKED', 'AGG-BLOCKED') else None
    audited, audited_blk = code, blocking
    if fired:                       # precedence: max-over-tokens > class-token stack > unlisted normalisation
        audited, audited_blk = fired[0]['verdict'], fired[0]['blocking_op']
    if warn:
        for r in fired:
            warnings.warn(r['warning'], stacklevel=2)
    declared = st['agg_name'].startswith('functional(')
    notes = []
    if declared:
        notes.append('no module reduced the cells; the aggregator was read from the declared '
                     'pool type (%s). An aggregation written as a tensor expression in '
                     'forward_head is inspected only through the max-over-tokens rule.' % st['agg_name'])
    if isinstance(f, (list, tuple)):
        notes.append('forward_features returned a list of %d tensors (several token sequences); '
                     'the reduced axis may be a stack of class tokens rather than spatial cells '
                     '(the class-token-stack rule covers the CoaT form).' % len(f))
    if st['agg'] == 'DENSE':
        notes.append('the aggregator is a dense layer (%s); check that the axis it reduces is '
                     'the spatial one.' % st['agg_name'])
    return dict(verdict=PAPER_TERM[audited], verdict_check=PAPER_TERM[code],
                rules_fired=[r['rule'] for r in fired], warnings=[r['warning'] for r in fired],
                blocking_op=audited_blk, raw_code=code,
                aggregator=dict(op=st['agg_name'], category=st['agg']),
                tail=[tuple(p) for p in st['post']], declared_pool_branch=declared,
                notes=notes)


if __name__ == '__main__':
    import argparse
    import json
    import timm
    ap = argparse.ArgumentParser(description='structural check of a timm model head')
    ap.add_argument('model', nargs='?', default='resnet50', help='timm model name')
    ap.add_argument('--pretrained', action='store_true',
                    help='load pretrained weights (not needed: the verdict is structural)')
    a = ap.parse_args()
    m = timm.create_model(a.model, pretrained=a.pretrained)
    r = check_model(m, warn=False)
    for w in r['warnings']:
        print('WARNING: ' + w, file=sys.stderr)
    print(json.dumps(dict(model=a.model, **r), indent=1))
