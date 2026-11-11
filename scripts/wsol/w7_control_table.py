"""W7 -- read the pre-norm control against the maps it was built to test.

w6 fits the `none` ordering on the ViT's pre-norm tokens: a mean over cells and
a linear map, with no LayerNorm anywhere. That readout is additive, certified by
the literal float64 loop, so necessity and sufficiency are provably the same
array and there is exactly ONE map, `collapsed_pre`.

The question it settles: the separated maps on the LayerNorm-relocated ViT beat
the collapsed map by a wide margin, but two things changed at once -- the model
left the degenerate point, and the map moved onto the pre-norm tokens. If a
single map drawn on the pre-norm tokens, which cannot possibly encode a
necessity/sufficiency distinction, recovers the same performance, then the
second change is doing the work and the separation contributed nothing.

This prints the comparison and emits it as a file, so the article can quote
it without anything being transcribed by hand.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W

METRICS = ['MaxBoxAccV2', 'best_iou', 'pointing_strict', 'pointing_tol15',
           'peak_mask_iou']
# w6 stores a point estimate either bare, as a [value, lo, hi] triple, or under
# a 'value' key with its interval beside it; w3 stores bare floats
ALIAS = {'best_iou': 'best_iou_mean'}


def point(x):
    if isinstance(x, dict):
        return x.get('value')
    if isinstance(x, (list, tuple)):
        return x[0]
    return x
SEPARATED = ['N_ln', 'S_ln']
COMPARATORS = ['collapsed', 'center']


def main():
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(W.OUT, 'w6_prenorm_*.json'))):
        w = json.load(open(f))
        ds = w['dataset']
        ctrl = w['per_map']['collapsed_pre']
        cell = f'{ds}|vitb16'
        try:
            main_run = json.load(open(os.path.join(W.OUT, 'w3_stats.json')))[cell]
        except (KeyError, FileNotFoundError):
            continue

        # the values the main run scored, read off the contrast rows
        vals = {}
        for metric, rows in main_run['contrasts'].items():
            for r in rows:
                vals.setdefault(r['a'], {})[metric] = r['va']
                vals.setdefault(r['b'], {})[metric] = r['vb']

        key = lambda m: ALIAS.get(m, m)
        metrics = [m for m in METRICS
                   if key(m) in ctrl and any(m in vals.get(k, {}) for k in SEPARATED)]
        row = {'n_eval': w['n_eval'], 'classification': w['classification'],
               'certificate': w['certificate'], 'metrics': {}}
        for m in metrics:
            cv = point(ctrl[key(m)])
            e = {'control_collapsed_pre': cv}
            for k in SEPARATED + COMPARATORS:
                if m in vals.get(k, {}):
                    e[k] = point(vals[k][m])
            sep = [e[k] for k in SEPARATED if k in e]
            e['control_beats_best_separated'] = bool(sep and cv >= max(sep))
            e['control_minus_best_separated'] = (cv - max(sep)) if sep else None
            row['metrics'][m] = e
        out[ds] = row

    if not out:
        sys.exit('no w6_prenorm_*.json found; run w6_prenorm.py first')

    print(f"{'dataset':10s}{'metric':18s}{'control':>9s}{'N_ln':>9s}"
          f"{'S_ln':>9s}{'collapsed':>11s}{'centre':>9s}{'ctrl-best':>11s}")
    print('-' * 86)
    wins = tot = 0
    for ds, row in sorted(out.items()):
        for m, e in row['metrics'].items():
            tot += 1
            wins += e['control_beats_best_separated']
            g = lambda k: f"{e[k]:9.4f}" if k in e and e[k] is not None else f"{'-':>9s}"
            d = e['control_minus_best_separated']
            print(f"{ds:10s}{m:18s}{g('control_collapsed_pre')}{g('N_ln')}{g('S_ln')}"
                  f"{g('collapsed'):>11s}{g('center')}"
                  f"{('' if d is None else f'{d:+11.4f}')}")
    print('-' * 86)
    print(f'  a single additive map on the pre-norm tokens matches or beats the '
          f'better separated map in {wins} of {tot} cells')
    out['summary'] = dict(cells=tot, control_matches_or_beats=wins)
    W.save('w7_control_table.json', out)


if __name__ == '__main__':
    main()
