"""C6 tasks 1 and 4.  Census counts after the audit corrections, from the archived
census records (analyses/check_census/out/c5_census.json, timm 1.0.27, random
weights).  Read-only; nothing is re-run here.

Corrections applied (each established by a1_audit.py measurement and by reading timm):
  COAT : coat_tiny / coat_mini / coat_small  IN-SCOPE -> DEGENERATE (class-token).
         CoaT.forward_head stacks the class tokens of three stages, x[:, 0], into
         [B, 3, C] and reduces them with Conv1d(3, 1, 1); no spatial cell reaches the
         logit (measured spatial mass 0, 3 active cells of 1032).
  ZEPTO: convnext_zepto_rms / _ols  NONLIN-BLOCKED -> NORM-BLOCKED (SimpleNorm2d is an
         RMS normalisation; deleting it makes the head additive).  Reported separately.
"""
import os, json, re, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
C1 = os.path.join(ROOT, 'analyses', 'check_census', 'out')
OUT = os.path.join(HERE, '..', 'out')
d = json.load(open(os.path.join(C1, 'c5_census.json')))
R = d['records']
pop = json.load(open(os.path.join(C1, 'c5_population.json')))

COAT = {'coat_tiny', 'coat_mini', 'coat_small'}
ZEPTO = {'convnext_zepto_rms', 'convnext_zepto_rms_ols'}
aggcls = lambda n: re.search(r'\((.*)\)$', n).group(1)
key = lambda r: (aggcls(r['agg_name']), tuple((p[1], p[2]) for p in r['post']))
ORDER = ['IN-SCOPE', 'NORM-BLOCKED', 'NONLIN-BLOCKED', 'DEGENERATE', 'AGG-BLOCKED']
PAPER = {'IN-SCOPE': 'degenerate', 'DEGENERATE': 'class-token', 'NORM-BLOCKED': 'normalisation-blocked',
         'NONLIN-BLOCKED': 'nonlinearity-blocked', 'AGG-BLOCKED': 'aggregator-blocked'}


def corrected(r, zepto=False):
    v = r['verdict']
    if r['arch'] in COAT:
        return 'DEGENERATE'
    if zepto and r['arch'] in ZEPTO:
        return 'NORM-BLOCKED'
    return v


def declared(r):
    m = re.match(r'functional\(global_pool=(.*)\)$', r['agg_name'])
    return None if m is None else m.group(1)


res = {}
lines = []
for p in ('p1', 'p2'):
    ok = [R[a] for a in pop[p] if a in R and 'verdict' in R[a]]
    n = len(ok)
    tests = [r for r in ok if r['arch'].startswith('test_')]
    dense = sorted(r['arch'] for r in ok if r['agg'] == 'DENSE')
    out = dict(n_classified=n, n_population=len(pop[p]), dense_aggregator_archs=dense,
               coat_in_population=sorted(a for a in COAT if a in pop[p]))
    for tag, fn in (('frozen', lambda r: r['verdict']), ('coat_corrected', corrected),
                    ('coat_and_zepto_corrected', lambda r: corrected(r, True))):
        c = collections.Counter(fn(r) for r in ok)
        out[tag] = {k: c.get(k, 0) for k in ORDER}
        out[tag + '_share'] = {k: c.get(k, 0) / n for k in ORDER}
        nt = [r for r in ok if not r['arch'].startswith('test_')]
        out[tag + '_degenerate_without_test_models'] = [sum(fn(r) == 'IN-SCOPE' for r in nt), len(nt)]
        # design level: one verdict per distinct trace
        tv = collections.defaultdict(set)
        for r in ok:
            tv[key(r)].add(fn(r))
        mixed = [k for k, v in tv.items() if len(v) > 1]
        ct = collections.Counter(next(iter(v)) for v in tv.values() if len(v) == 1)
        out[tag + '_traces'] = dict(n=len(tv), by_verdict={k: ct.get(k, 0) for k in ORDER},
                                    n_traces_with_mixed_verdict=len(mixed))
    # declared-pool (functional) branch
    fun = [r for r in ok if declared(r) is not None]
    out['declared_branch_total'] = len(fun)
    out['declared_branch_by_verdict'] = dict(collections.Counter(r['verdict'] for r in fun))
    deg = [r for r in fun if r['verdict'] == 'IN-SCOPE']
    out['declared_branch_degenerate'] = len(deg)
    out['declared_branch_degenerate_by_declared_string'] = dict(collections.Counter(declared(r) for r in deg))
    out['declared_branch_degenerate_v2b'] = dict(collections.Counter(r['v2b'] for r in deg))
    out['declared_branch_degenerate_aten'] = dict(collections.Counter(','.join(r['aten_classes']) for r in deg))
    out['declared_branch_degenerate_archs_no_avg_string'] = sorted(r['arch'] for r in deg if declared(r) != 'avg')
    out['declared_branch_by_string'] = dict(collections.Counter(declared(r) for r in fun))
    out['declared_branch_string_x_verdict'] = {'%s|%s' % k: v for k, v in
                                               collections.Counter((declared(r), r['verdict']) for r in fun).items()}
    out['declared_branch_v2b_x_v1'] = {'%s->%s' % k: v for k, v in
                                       collections.Counter((r['verdict'], r['v2b']) for r in fun).items()}
    out['n_test_models'] = len(tests)
    out['test_models_degenerate'] = sum(r['verdict'] == 'IN-SCOPE' for r in tests)
    res[p] = out
    lines.append('== %s: %d classified of %d' % (p, n, len(pop[p])))
    for tag in ('frozen', 'coat_corrected', 'coat_and_zepto_corrected'):
        lines.append('  %-26s %s  degenerate share %.4f  w/o test models %s  traces %s'
                     % (tag, out[tag], out[tag + '_share']['IN-SCOPE'],
                        out[tag + '_degenerate_without_test_models'], out[tag + '_traces']))
    lines.append('  DENSE aggregator archs: %s' % dense)
    lines.append('  declared-pool branch: %d verdicts %s; degenerate %d by string %s; v2b on those %s; aten %s'
                 % (len(fun), out['declared_branch_by_verdict'], len(deg),
                    out['declared_branch_degenerate_by_declared_string'], out['declared_branch_degenerate_v2b'],
                    out['declared_branch_degenerate_aten']))
    lines.append('  degenerate on declared branch without an avg string: %s'
                 % out['declared_branch_degenerate_archs_no_avg_string'])
    lines.append('  declared strings over the branch: %s' % out['declared_branch_by_string'])
    lines.append('  string x verdict: %s' % out['declared_branch_string_x_verdict'])
    lines.append('  v1 -> v2b on the branch: %s' % out['declared_branch_v2b_x_v1'])
print('\n'.join(lines))
json.dump(res, open(os.path.join(OUT, 'a4_census.json'), 'w'), indent=1)
open(os.path.join(OUT, 'a4_census.log'), 'w').write('\n'.join(lines) + '\n')
