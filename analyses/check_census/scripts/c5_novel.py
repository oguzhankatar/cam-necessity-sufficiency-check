"""C1 task 5 addendum -- the census traces NOT present among the 148 (P1 and P2):
which architectures carry them and what verdict the frozen rule gives.  Read-only."""
import os, json, re, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
recs = json.load(open(os.path.join(OUT, 'c5_census.json')))['records']
pop = json.load(open(os.path.join(OUT, 'c5_population.json')))
h1 = []
for s in 'DAB':
    h1 += json.load(open(os.path.join(ROOT, 'outputs', 'heldout', 'h1_%s.json' % s)))
aggcls = lambda n: re.search(r'\((.*)\)$', n).group(1)
key = lambda d: (aggcls(d['agg_name']), tuple((p[1], p[2]) for p in d['post']))
tr148 = set(key(r) for r in h1)
res = {}
for p in ('p1', 'p2'):
    ok = [recs[a] for a in pop[p] if a in recs and 'verdict' in recs[a]]
    nov = collections.defaultdict(list)
    for d in ok:
        k = key(d)
        if k not in tr148:
            nov[k].append((d['arch'], d['verdict'], d['v2b']))
    res[p] = dict(n_novel_traces=len(nov), n_archs=sum(len(v) for v in nov.values()),
                  verdicts_v1=dict(collections.Counter(x[1] for v in nov.values() for x in v)),
                  verdicts_v2b=dict(collections.Counter(x[2] for v in nov.values() for x in v)),
                  traces=[dict(agg=k[0], tail=[list(t) for t in k[1]], archs=v) for k, v in
                          sorted(nov.items(), key=lambda kv: -len(kv[1]))])
    print('==', p, res[p]['n_novel_traces'], 'traces', res[p]['n_archs'], 'archs', res[p]['verdicts_v1'], res[p]['verdicts_v2b'])
    for t in res[p]['traces']:
        print('  %2d %s %s | %s' % (len(t['archs']), t['agg'], [tuple(x) for x in t['tail']],
                                   [a[0] + ':' + a[1] for a in t['archs']][:6]))
json.dump(res, open(os.path.join(OUT, 'c5_novel_traces.json'), 'w'), indent=1)
