"""C1 task 1 -- head-trace novelty, per-trace agreement, nearest-trace baseline,
three-way score.  Reads outputs/heldout/h1_{D,A,B}.json only (read-only).

Trace definition: aggregator class + ordered tuple of
(tail module class, frozen label).  The aggregator class is the module class in
`agg_name` (e.g. 'AdaptiveAvgPool2d', 'SelectAdaptivePool2d') or, for the
functional fallback, the string 'functional(global_pool=<x>)'.  A second key
using the coarse aggregator category (MEAN/TOKEN/DENSE/OTHER) is reported too.
"""
import json, os, re, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
GT_OF = {'IN-SCOPE': 'ADDITIVE', 'DEGENERATE': 'VACUOUS', 'NORM-BLOCKED': 'NORM-ONLY',
         'NONLIN-BLOCKED': 'HARD-BLOCKED', 'AGG-BLOCKED': 'HARD-BLOCKED'}
THREE = {'ADDITIVE': 'ADDITIVE', 'VACUOUS': 'VACUOUS', 'NORM-ONLY': 'NOT-ADDITIVE',
         'HARD-BLOCKED': 'NOT-ADDITIVE'}

rows = []
for s in 'DAB':
    rows += json.load(open(os.path.join(ROOT, 'outputs', 'heldout', 'h1_%s.json' % s)))


def aggcls(r):
    return re.search(r'\((.*)\)$', r['agg_name']).group(1)


KEYS = {
    'module_class': lambda r: (aggcls(r), tuple((p[1], p[2]) for p in r['post'])),
    'agg_category': lambda r: (r['agg'], tuple((p[1], p[2]) for p in r['post'])),
    'kinds_only': lambda r: (r['agg'], tuple(p[2] for p in r['post'])),
}

res = {}
for kname, key in KEYS.items():
    for r in rows:
        r['_k'] = key(r)
    allc = collections.Counter(r['_k'] for r in rows)
    dev = [r for r in rows if r['split'] == 'D']
    ho = [r for r in rows if r['split'] in 'AB']
    dev_meas = [r for r in dev if 'gt' in r]
    dev_keys = set(r['_k'] for r in dev)               # structural: crossvit included
    dev_gt = collections.defaultdict(list)
    for r in dev_meas:
        dev_gt[r['_k']].append(r['gt'])
    seen = [r for r in ho if r['_k'] in dev_keys]
    novel = [r for r in ho if r['_k'] not in dev_keys]
    ho_keys = collections.Counter(r['_k'] for r in ho)
    novel_keys = set(r['_k'] for r in novel)

    # majority label: the commonest ground-truth label in the DEVELOPMENT set
    maj = collections.Counter(r['gt'] for r in dev_meas).most_common(1)[0][0]
    maj_ho = collections.Counter(r['gt'] for r in ho).most_common(1)[0][0]

    def nearest(r, three=False):
        lab = dev_gt.get(r['_k'])
        if lab:
            p = collections.Counter(lab).most_common(1)[0][0]
        else:
            p = maj
        return THREE[p] if three else p

    def score(sub, three=False):
        f = (lambda x: THREE[x]) if three else (lambda x: x)
        return dict(
            n=len(sub),
            rule=sum(f(GT_OF[r['verdict']]) == f(r['gt']) for r in sub),
            rule_v2b=sum(f(GT_OF[r['v2b']]) == f(r['gt']) for r in sub),
            majority=sum(f(maj) == f(r['gt']) for r in sub),
            nearest_trace=sum(nearest(r, three) == f(r['gt']) for r in sub))

    # per distinct held-out trace: does the rule agree on every member?
    per_trace = []
    for k, n in ho_keys.most_common():
        mem = [r for r in ho if r['_k'] == k]
        per_trace.append(dict(trace=[k[0], [list(x) if isinstance(x, tuple) else x for x in k[1]]],
                              n=n, in_dev=k in dev_keys,
                              models=[r['model'] for r in mem],
                              gts=sorted(set(r['gt'] for r in mem)),
                              verdicts=sorted(set(r['verdict'] for r in mem)),
                              all_agree=all(r['agree'] for r in mem),
                              majority_all_right=all(maj == r['gt'] for r in mem),
                              nearest_all_right=all(nearest(r) == r['gt'] for r in mem)))
    # do traces seen in dev ever disagree in ground truth between dev and held-out?
    conflicts = [(r['model'], r['gt'], dev_gt.get(r['_k'])) for r in seen
                 if r['_k'] in dev_gt and r['gt'] not in dev_gt[r['_k']]]
    res[kname] = dict(
        n_distinct_all=len(allc), commonest_all=allc.most_common(1)[0][1],
        commonest_models=[r['model'] for r in rows if r['_k'] == allc.most_common(1)[0][0]][:40],
        n_distinct_dev=len(dev_keys), n_distinct_heldout=len(ho_keys),
        n_heldout_seen_in_dev=len(seen), n_heldout_novel=len(novel),
        n_novel_traces=len(novel_keys),
        novel_models=[r['model'] for r in novel],
        novel_detail=[dict(model=r['model'], verdict=r['verdict'], gt=r['gt'],
                           agree=r['agree'], nearest=nearest(r)) for r in novel],
        majority_label_dev=maj, majority_label_heldout=maj_ho,
        four_way=dict(all_heldout=score(ho), seen=score(seen), novel=score(novel),
                      heldout_minus_vacuous=score([r for r in ho if r['gt'] != 'VACUOUS'])),
        three_way=dict(all_heldout=score(ho, True), seen=score(seen, True),
                       novel=score(novel, True), dev=score(dev_meas, True)),
        per_trace_heldout=dict(
            n_traces=len(ho_keys),
            n_traces_all_agree=sum(t['all_agree'] for t in per_trace),
            n_novel_traces=len(novel_keys),
            n_novel_traces_all_agree=sum(t['all_agree'] for t in per_trace if not t['in_dev']),
            n_traces_majority_right=sum(t['majority_all_right'] for t in per_trace),
            n_traces_nearest_right=sum(t['nearest_all_right'] for t in per_trace),
            n_novel_traces_majority_right=sum(t['majority_all_right'] for t in per_trace if not t['in_dev']),
            traces=per_trace),
        gt_conflicts_seen_vs_dev=conflicts,
    )

json.dump(res, open(os.path.join(OUT, 'c1_trace.json'), 'w'), indent=1)
for k, v in res.items():
    print('==', k)
    for kk in ('n_distinct_all', 'commonest_all', 'n_distinct_dev', 'n_distinct_heldout',
               'n_heldout_seen_in_dev', 'n_heldout_novel', 'n_novel_traces'):
        print('  %-24s %s' % (kk, v[kk]))
    print('  novel:', v['novel_models'])
    print('  4-way', v['four_way'])
    print('  3-way', v['three_way'])
    pt = v['per_trace_heldout']
    print('  per-trace: %d/%d traces all-agree; novel %d/%d; majority %d/%d (novel %d); nearest %d/%d' % (
        pt['n_traces_all_agree'], pt['n_traces'], pt['n_novel_traces_all_agree'], pt['n_novel_traces'],
        pt['n_traces_majority_right'], pt['n_traces'], pt['n_novel_traces_majority_right'],
        pt['n_traces_nearest_right'], pt['n_traces']))
    print('  gt conflicts seen-vs-dev:', v['gt_conflicts_seen_vs_dev'])
    for d in v['novel_detail']:
        print('    ', d)
