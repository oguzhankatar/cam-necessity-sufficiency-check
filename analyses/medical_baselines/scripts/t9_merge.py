"""merge the per-arm t6_score_<arm>.json files into out/t6_score.json (C2)."""
import os, json, glob
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out')
R = {}
for fn in sorted(glob.glob(os.path.join(OUT, 't6_score_*.json'))):
    R.update(json.load(open(fn)))
json.dump(R, open(os.path.join(OUT, 't6_score.json'), 'w'), indent=1)
print('merged', sorted(R))
