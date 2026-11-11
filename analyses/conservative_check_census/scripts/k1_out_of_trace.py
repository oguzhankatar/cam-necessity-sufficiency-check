"""POST HOC: agreement of the original check on census entries by whether a development
architecture shares their head trace (entry-weighted out-of-trace accuracy).

    python k1_out_of_trace.py

Reads (never writes) K1/out/k1_run_timm_*.jsonl and K1/out/k1_verdicts.json; writes
K1/out/k1_out_of_trace.json.

Head trace = (aggregator category, aggregator operation, ordered tuple of tail module classes).
This definition reproduces the 43 census traces and the 33 traces of the 148 checked
architectures covering 865 of the 895 census entries (Note S2.6). Labels: the census labels of
k1_verdicts.json (unit-scale rule); the 18 zero-initialised entries take the label certified after
re-drawing (re-drawn label = the agreement already counted in 879/895). Majority-verdict baseline:
the commonest measured label of the development architectures (degenerate).
"""
import os, re, json, glob, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'out')
LAB = {'ADDITIVE': 'degenerate', 'NORM-ONLY': 'normalisation-blocked', 'HARD-BLOCKED': 'hard',
       'VACUOUS': 'class-token'}
OV = {'nonlinearity-blocked': 'hard', 'aggregator-blocked': 'hard'}
BLOCKED = ('hard', 'normalisation-blocked')


def trace(r):
    a = r.get('aggregator') or {}
    op = re.sub(r'^.*\(', '(', str(a.get('op')))
    return (a.get('category'), op, tuple(t[1] for t in (r.get('tail') or [])))


def main():
    R = {}
    for f in sorted(glob.glob(os.path.join(OUT, 'k1_run_timm_*.jsonl'))):
        for line in open(f, encoding='utf-8'):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'aggregator' in r:
                R[r['name']] = r
    V = json.load(open(os.path.join(OUT, 'k1_verdicts.json'), encoding='utf-8'))
    cen = {e['name']: e for e in V['verdicts']['census']}
    assert all(n in R for n in cen)
    dev = {trace(r) for r in R.values() if r['set'] == 'arch148_D'}
    checked = {trace(r) for r in R.values() if r['set'].startswith('arch148')}
    devlab = collections.Counter(LAB.get(cen[n]['label']) for n in R
                                 if R[n]['set'] == 'arch148_D' and n in cen)
    rows = []
    for n, e in cen.items():
        rows.append(dict(name=n, label=e['label'], verdict=OV.get(e['original'], e['original']),
                         dev_trace=trace(R[n]) in dev))
    direct = [x for x in rows if x['label'] != 'ZERO-READOUT']
    zero = [x for x in rows if x['label'] == 'ZERO-READOUT']
    unseen = [x for x in direct if not x['dev_trace']]
    seen = [x for x in direct if x['dev_trace']]
    maj = 'degenerate'
    res = dict(
        post_hoc=True,
        n_census=len(cen), n_census_traces=len({trace(R[n]) for n in cen}),
        n_checked_traces=len(checked),
        census_entries_covered_by_checked_traces=sum(trace(R[n]) in checked for n in cen),
        n_zero_readout=len(zero), zero_readout_on_dev_traces=sum(x['dev_trace'] for x in zero),
        dev_trace_direct=dict(n=len(seen), agree=sum(LAB[x['label']] == x['verdict'] for x in seen)),
        no_dev_trace=dict(
            n=len(unseen),
            agree=sum(LAB[x['label']] == x['verdict'] for x in unseen),
            binary_agree=sum((LAB[x['label']] in BLOCKED) == (x['verdict'] in BLOCKED) for x in unseen),
            majority_verdict=maj, majority_agree=sum(LAB[x['label']] == maj for x in unseen),
            labels=dict(collections.Counter(LAB[x['label']] for x in unseen)),
            errors=[x['name'] for x in unseen if LAB[x['label']] != x['verdict']]),
        development_label_counts={str(k): v for k, v in devlab.items()})
    with open(os.path.join(OUT, 'k1_out_of_trace.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'no_dev_trace'}, indent=1))
    u = res['no_dev_trace']
    print('no development trace: %d entries, check right on %d (binary %d), majority verdict %d; errors %s'
          % (u['n'], u['agree'], u['binary_agree'], u['majority_agree'], u['errors']))


if __name__ == '__main__':
    main()
