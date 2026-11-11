"""Summarise the c5 census (pass 1 + retry pass) over populations P1 and P2."""
import os, json, glob, re, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
WD = os.path.join(OUT, 'c5_work')
pop = json.load(open(os.path.join(OUT, 'c5_population.json')))

recs = {}
for pat in ('pass1_res_*.jsonl', 'res_*.jsonl'):          # later pass overrides
    for f in sorted(glob.glob(os.path.join(WD, pat))):
        for line in open(f):
            d = json.loads(line)
            a = d['arch']
            if a in recs and 'verdict' in recs[a] and 'verdict' not in d:
                continue
            recs[a] = d

h1 = []
for s in 'DAB':
    h1 += json.load(open(os.path.join(ROOT, 'outputs', 'heldout', 'h1_%s.json' % s)))
H1 = {r['model']: r for r in h1}


def aggcls(agg_name):
    return re.search(r'\((.*)\)$', agg_name).group(1)


def key(d):
    return (aggcls(d['agg_name']), tuple((p[1], p[2]) for p in d['post']))


summary = {}
for pname in ('p1', 'p2'):
    P = pop[pname]
    rs = [recs.get(a, dict(arch=a, error='no record')) for a in P]
    ok = [d for d in rs if 'verdict' in d]
    skipped = [d['arch'] for d in rs if 'skipped' in d]
    errs = [(d['arch'], d['error'][:160]) for d in rs if 'error' in d]
    v1 = collections.Counter(d['verdict'] for d in ok)
    v2b = collections.Counter(d['v2b'] for d in ok)
    traces = collections.Counter(key(d) for d in ok)
    top = traces.most_common(5)
    in148 = [a for a in P if a in H1]
    # structure-only verdict vs the archived pretrained-weights verdict
    same = [a for a in in148 if a in recs and 'verdict' in recs[a]]
    mism = [(a, recs[a]['verdict'], H1[a]['verdict']) for a in same if recs[a]['verdict'] != H1[a]['verdict']]
    tr148 = set(key(H1[a]) for a in H1)
    summary[pname] = dict(
        n_population=len(P), n_classified=len(ok), n_skipped_over_1e9=len(skipped), skipped=skipped,
        n_failed=len(errs), failed=errs,
        verdicts_frozen_v1=dict(v1), verdicts_v2b=dict(v2b),
        shares_v1={k: v / len(ok) for k, v in v1.items()},
        n_distinct_traces=len(traces),
        commonest_traces=[dict(n=n, agg=k[0], tail=[list(x) for x in k[1]]) for k, n in top],
        n_traces_singletons=sum(1 for n in traces.values() if n == 1),
        n_of_148_in_population=len(in148), of_148_not_in_population=sorted(set(H1) - set(P)),
        n_of_148_classified_here=len(same), n_verdict_mismatch_vs_archived=len(mism), mismatches=mism,
        n_population_traces_seen_in_148=sum(1 for k in traces if k in tr148),
        n_population_archs_with_trace_in_148=sum(n for k, n in traces.items() if k in tr148),
        verdicts_v1_outside_148=dict(collections.Counter(d['verdict'] for d in ok if d['arch'] not in H1)),
        module_pool_false=sum(not d['module_pool'] for d in ok),
        v1_ne_v2b=[(d['arch'], d['verdict'], d['v2b']) for d in ok if d['verdict'] != d['v2b']],
    )
    print('==', pname, json.dumps({k: v for k, v in summary[pname].items()
                                   if k not in ('failed', 'skipped', 'commonest_traces', 'v1_ne_v2b',
                                                'mismatches', 'of_148_not_in_population')}, indent=1))
    print('  failed:', errs[:40])
    print('  skipped:', skipped)
    print('  not-in-pop of 148:', summary[pname]['of_148_not_in_population'])
    print('  mismatches:', mism)
    print('  v1!=v2b: n=%d' % len(summary[pname]['v1_ne_v2b']), summary[pname]['v1_ne_v2b'][:60])
    for t in summary[pname]['commonest_traces']:
        print('  top trace', t['n'], t['agg'], t['tail'])
json.dump(dict(summary=summary, records=recs), open(os.path.join(OUT, 'c5_census.json'), 'w'), indent=1)
