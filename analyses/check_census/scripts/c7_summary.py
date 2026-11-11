"""C1 task 7 summary -- tabulate out/c3_misc.json task7_prices (read-only)."""
import os, json, collections
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
t7 = json.load(open(os.path.join(OUT, 'c3_misc.json')))['task7_prices']
ok = [d for d in t7 if 'error' not in d]
res = {}
for grp, f in (('dev13', lambda d: d['source'] != 'h5_edits_heldout'),
               ('heldout', lambda d: d['source'] == 'h5_edits_heldout'), ('all', lambda d: True)):
    s = [d for d in ok if f(d)]
    best = collections.Counter(d['best_edit'] for d in s)
    costs = [d['best_cost_pts'] for d in s]
    lo = max(s, key=lambda d: d['best_cost_pts']); hi = min(s, key=lambda d: d['best_cost_pts'])
    delact_fail = [d['model'] for d in s if d['delact_alone_additive'] is False]
    res[grp] = dict(n=len(s), cheapest_edit_counts=dict(best),
                    cheapest_cost_min=dict(model=lo['model'], edit=lo['best_edit'], pts=lo['best_cost_pts']),
                    cheapest_cost_max=dict(model=hi['model'], edit=hi['best_edit'], pts=hi['best_cost_pts']),
                    cheapest_cost_median=sorted(costs)[len(costs) // 2] if len(costs) % 2 else
                    0.5 * (sorted(costs)[len(costs) // 2 - 1] + sorted(costs)[len(costs) // 2]),
                    delact_alone_not_additive=delact_fail,
                    delact_fail_all_have_norm=all(len(d['norms']) > 0 for d in s if d['delact_alone_additive'] is False),
                    delact_ok_all_normfree=all(len(d['norms']) == 0 for d in s if d['delact_alone_additive']),
                    per_model=[(d['model'], d['best_edit'], round(d['best_cost_pts'], 3),
                                d['delact_alone_additive'], d['norms']) for d in s])
res['errors'] = [(d['model'], d['error'][:120]) for d in t7 if 'error' in d]
res['h4_y_equals_cheapest'] = all(abs(d['h4_price_y'] - d['best_cost_pts']) < 1e-9 for d in ok if d['h4_price_y'] is not None)
json.dump(res, open(os.path.join(OUT, 'c7_summary.json'), 'w'), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != 'per_model'}) for k, v in res.items()}, indent=1))
