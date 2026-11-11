"""Census with every audit correction applied.

Read-only re-tally of the archived census records (analyses/check_census/out/
c5_census.json, timm 1.0.27, random weights); nothing is re-run.  Extends
a4_census.py, which applied the CoaT and the ConvNeXt-Zepto-RMS corrections, by the
VOLO family.

Corrections (each supported by a measurement read from the archived outputs, and
checked below before it is applied):
  COAT  coat_tiny / coat_mini / coat_small        degenerate -> class-token.
        CoaT.forward_head stacks the class tokens of three stages into [B, 3, C] and
        reduces them with Conv1d(3, 1, 1); no spatial cell reaches the logit.
        Evidence: a1_audit.json, measured VACUOUS (spatial mass 0) on all three.
  ZEPTO convnext_zepto_rms / convnext_zepto_rms_ols  nonlinearity-blocked ->
        normalisation-blocked.  SimpleNorm2d is an RMS normalisation; deleting it
        makes the head additive.  Evidence: a1_audit.json, measured NORM-ONLY.
  VOLO  all 11 volo_* checkpoints                  class-token -> aggregator-blocked.
        VOLO's head takes a maximum over the patch tokens (aux_head) in a tensor
        expression, outside any pooling module.  Evidence: VOLO-D1 measured
        non-additive (eps1 read from c3_misc.json, ground truth HARD-BLOCKED); the 11
        checkpoints share one head trace; the operator-level variant (v2b, stored in
        every census record) returns AGG-BLOCKED on all 11.

Codes: the census records carry the frozen rule's legacy codes.  Article terms:
IN-SCOPE = degenerate, DEGENERATE = class-token, NORM-BLOCKED = normalisation-blocked,
NONLIN-BLOCKED = nonlinearity-blocked, AGG-BLOCKED = aggregator-blocked.
"""
import os, json, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
C1 = os.path.join(ROOT, 'analyses', 'check_census', 'out')
OUT = os.path.abspath(os.path.join(HERE, '..', 'out'))

R = json.load(open(os.path.join(C1, 'c5_census.json')))['records']
pop = json.load(open(os.path.join(C1, 'c5_population.json')))
misc = json.load(open(os.path.join(C1, 'c3_misc.json')))
audit = json.load(open(os.path.join(OUT, 'a1_audit.json')))
arow = {r['arch']: r for r in audit['rows']}

ORDER = ['IN-SCOPE', 'NORM-BLOCKED', 'NONLIN-BLOCKED', 'DEGENERATE', 'AGG-BLOCKED']
PAPER = {'IN-SCOPE': 'degenerate', 'NORM-BLOCKED': 'normalisation-blocked',
         'NONLIN-BLOCKED': 'nonlinearity-blocked', 'DEGENERATE': 'class-token',
         'AGG-BLOCKED': 'aggregator-blocked'}

COAT = {'coat_tiny', 'coat_mini', 'coat_small'}
ZEPTO = {'convnext_zepto_rms', 'convnext_zepto_rms_ols'}
VOLO = {a for a in R if a.startswith('volo_')}

aggcls = lambda n: re.search(r'\((.*)\)$', n).group(1)
key = lambda r: (aggcls(r['agg_name']), tuple((p[1], p[2]) for p in r['post']))

# ---- evidence checks (fail loudly if an archived measurement does not support a correction)
ev = {}
for a in sorted(COAT):
    r = arow[a]
    assert r['frozen'] == 'IN-SCOPE' and r['measured'] == 'VACUOUS' and r['measured_unit'] == 'VACUOUS', a
    ev[a] = dict(frozen=r['frozen'], measured=r['measured'], spatial_mass=r['shipped']['spatial_mass'],
                 n_active=r['shipped']['n_active'], T=r['shipped']['T'])
for a in sorted(ZEPTO):
    r = arow[a]
    assert r['frozen'] == 'NONLIN-BLOCKED' and r['measured'] == 'NORM-ONLY' and r['measured_unit'] == 'NORM-ONLY', a
    ev[a] = dict(frozen=r['frozen'], measured=r['measured'], eps1=r['shipped']['eps1'],
                 eps1_unit=r['shipped_unit']['eps1'])
vm = misc['task4_provenance']['volo']
assert vm['v1'] == 'DEGENERATE' and vm['gt'] == 'HARD-BLOCKED' and vm['v2b'] == 'AGG-BLOCKED'
volo_traces = {key(R[a]) for a in VOLO}
assert len(volo_traces) == 1
assert all(R[a]['verdict'] == 'DEGENERATE' and R[a]['v2b'] == 'AGG-BLOCKED' for a in VOLO)
volo_trace_members = sorted(a for a in R if 'verdict' in R[a] and key(R[a]) in volo_traces)
assert set(volo_trace_members) == VOLO, volo_trace_members
ev['volo_family'] = dict(n_checkpoints=len(VOLO), checkpoints=sorted(VOLO),
                         frozen='DEGENERATE', operator_level_variant=sorted({R[a]['v2b'] for a in VOLO}),
                         volo_d1_224_measured=vm['gt'], volo_d1_224_eps1=vm['maxNS'],
                         volo_d1_224_corrNS_min=vm['corrNS_min'], volo_d1_224_n_active=vm['n_active'],
                         shared_trace=dict(agg=list(volo_traces)[0][0], tail=[list(t) for t in list(volo_traces)[0][1]]),
                         other_archs_on_trace=[])


def audited(r):
    a = r['arch']
    if a in COAT:
        return 'DEGENERATE'
    if a in ZEPTO:
        return 'NORM-BLOCKED'
    if a in VOLO:
        return 'AGG-BLOCKED'
    return r['verdict']


def tally(ok, fn):
    n = len(ok)
    c = collections.Counter(fn(r) for r in ok)
    counts = {k: c.get(k, 0) for k in ORDER}
    shares = {k: counts[k] / n for k in ORDER}
    tv = collections.defaultdict(set)
    for r in ok:
        tv[key(r)].add(fn(r))
    mixed = [k for k, v in tv.items() if len(v) > 1]
    ct = collections.Counter(next(iter(v)) for v in tv.values() if len(v) == 1)
    tr = {k: ct.get(k, 0) for k in ORDER}
    return dict(n=n, counts=counts, shares=shares,
                counts_paper={PAPER[k]: counts[k] for k in ORDER},
                shares_paper={PAPER[k]: shares[k] for k in ORDER},
                traces=dict(n=len(tv), by_verdict=tr, by_verdict_paper={PAPER[k]: tr[k] for k in ORDER},
                            n_traces_with_mixed_verdict=len(mixed)))


res = dict(corrections=dict(coat=sorted(COAT), zepto=sorted(ZEPTO), volo=sorted(VOLO)), evidence=ev, code_to_article=PAPER)
lines = ['Census, all audit corrections (CoaT -> class-token; ConvNeXt-Zepto-RMS -> normalisation-blocked;',
         'VOLO x%d -> aggregator-blocked).  Source: c5_census.json (timm 1.0.27, random weights).' % len(VOLO),
         'VOLO-D1 evidence: measured %s, eps1 %.3f; operator-level variant on all %d: %s; one shared trace.'
         % (vm['gt'], vm['maxNS'], len(VOLO), ev['volo_family']['operator_level_variant']), '']


def fmt(t):
    s = '  '.join('%s %d (%.1f%%)' % (PAPER[k], t['counts'][k], 100 * t['shares'][k]) for k in ORDER)
    tr = ' / '.join('%s %d' % (PAPER[k], t['traces']['by_verdict'][k]) for k in ORDER)
    return '%s   [n=%d]\n      traces n=%d: %s; mixed %d' % (s, t['n'], t['traces']['n'], tr,
                                                         t['traces']['n_traces_with_mixed_verdict'])


for p in ('p1', 'p2'):
    ok = [R[a] for a in pop[p] if a in R and 'verdict' in R[a]]
    nt = [r for r in ok if not r['arch'].startswith('test_')]
    out = dict(n_population=len(pop[p]), n_classified=len(ok), n_test_models=len(ok) - len(nt),
               test_models=sorted(r['arch'] for r in ok if r['arch'].startswith('test_')))
    for tag, fn in (('frozen', lambda r: r['verdict']), ('audited', audited)):
        out[tag] = tally(ok, fn)
        out[tag + '_without_test_models'] = tally(nt, fn)
    out['changed'] = sorted('%s:%s->%s' % (r['arch'], r['verdict'], audited(r)) for r in ok if audited(r) != r['verdict'])
    res[p] = out
    lines.append('== %s: %d classified of %d (test models %d)' % (p, len(ok), len(pop[p]), out['n_test_models']))
    for tag in ('frozen', 'audited'):
        lines.append('  %-8s all          %s' % (tag, fmt(out[tag])))
        lines.append('  %-8s w/o test     %s' % (tag, fmt(out[tag + '_without_test_models'])))
    lines.append('  changed verdicts: %d' % len(out['changed']))
    lines.append('')

print('\n'.join(lines))
json.dump(res, open(os.path.join(OUT, 'a5_census_full.json'), 'w'), indent=1)
open(os.path.join(OUT, 'a5_census_full.log'), 'w').write('\n'.join(lines) + '\n')
