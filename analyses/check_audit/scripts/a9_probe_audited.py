"""Stage 3, task 1.  scripts/check_model.check_model() (the audited verdict: frozen check +
three pattern rules) on the 32 constructed heads of a3_probes.py, scored against the
measured labels stored in out/a3_probes.json.

Each probe is built exactly as a3_probes.run_one builds it (seed 0, eval, declared
global_pool where the probe declares one) and fed its own feature map (a3_probes.make_feat);
forward_features is the identity, so check_model sees the same f the frozen check saw.
The measured labels are read from a3_probes.json, not recomputed.

Scoring as in a3_probes: audited verdict -> frozen code -> check_audit_lib.GT_OF -> label, compared
with the measured label; for an agreeing hard-blocked verdict, the cause is right when
aggregator-blocked coincides with a probe designed to block at the aggregator.

usage: python a9_probe_audited.py  ->  ../out/a9_probe_audited.json, ../out/a9_probe_audited.log
"""
import os, sys, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import torch
import timm
import a3_probes as A3                     # sets float64 default dtype; defines PROBES
import check_audit_lib as L
from check_model import check_model, PAPER_TERM

CODE_OF = {v: k for k, v in PAPER_TERM.items()}
OUT = os.path.join(HERE, '..', 'out')
meas = {r['id']: r for r in json.load(open(os.path.join(OUT, 'a3_probes.json')))['rows']}


def identity(x):
    return x


rows = []
for P in A3.PROBES:
    torch.manual_seed(0)
    m = P().eval()
    if P.gp is not None:
        m.global_pool = P.gp
    m.forward_features = identity
    f = A3.make_feat(P)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        r = check_model(m, x=f, warn=False)
    M = meas[P.pname]
    lab = M['measured']
    code_a, code_c = CODE_OF[r['verdict']], r['raw_code']
    agree_a = L.GT_OF[code_a] == lab
    agree_c = L.GT_OF[code_c] == lab
    if lab == 'HARD-BLOCKED' and agree_a:
        reason_a = (code_a == 'AGG-BLOCKED') == P.design.startswith('aggregator')
    else:
        reason_a = agree_a
    rows.append(dict(id=P.pname, probe=P.desc, design=P.design, measured=lab,
                     verdict_check=r['verdict_check'], verdict=r['verdict'], rules_fired=r['rules_fired'],
                     frozen_code_stored=M['frozen'], frozen_code_now=code_c,
                     agree_check=bool(agree_c), agree_audited=bool(agree_a), reason_ok_audited=bool(reason_a),
                     notes=r['notes']))

n = len(rows)
summ = dict(
    n=n,
    agree_check=sum(r['agree_check'] for r in rows),
    agree_check_stored=sum(meas[r['id']]['agree'] for r in rows),
    frozen_code_reproduced=all(r['frozen_code_now'] == r['frozen_code_stored'] for r in rows),
    agree_audited=sum(r['agree_audited'] for r in rows),
    agree_audited_with_reason=sum(r['reason_ok_audited'] for r in rows),
    repaired=[r['id'] for r in rows if r['agree_audited'] and not r['agree_check']],
    broken=[r['id'] for r in rows if r['agree_check'] and not r['agree_audited']],
    still_wrong=[r['id'] for r in rows if not r['agree_audited']],
    right_label_wrong_reason=[r['id'] for r in rows if r['agree_audited'] and not r['reason_ok_audited']],
    rules_fired={r['id']: r['rules_fired'] for r in rows if r['rules_fired']},
    timm=timm.__version__, torch=torch.__version__)

lines = ['check_model() audited verdict on the %d constructed heads of a3_probes.py (timm %s, torch %s, float64, CPU)'
         % (n, timm.__version__, torch.__version__), '']
for r in rows:
    lines.append('%s check %-22s audited %-22s rules %-32s measured %-12s %s%s | %s'
                 % (r['id'], r['verdict_check'], r['verdict'], ','.join(r['rules_fired']) or '-', r['measured'],
                    'agree' if r['agree_audited'] else 'DISAGREE',
                    '' if r['reason_ok_audited'] else ' (wrong cause)' if r['agree_audited'] else '', r['probe']))
lines += ['', json.dumps(summ)]
txt = '\n'.join(lines) + '\n'
print(txt)
open(os.path.join(OUT, 'a9_probe_audited.log'), 'w', encoding='utf-8').write(txt)
json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, 'a9_probe_audited.json'), 'w'), indent=1)
