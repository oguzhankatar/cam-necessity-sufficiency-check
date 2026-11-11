"""K1 on the 32 constructed heads (analyses/check_audit/scripts/a3_probes.py), scored against their stored
float64 certificate labels (analyses/check_audit/out/a3_probes.json).

Each probe is built exactly as a3_probes.run_one / a9_probe_audited build it (seed 0,
eval, declared global_pool where the probe declares one, forward_features = identity),
and the checks see the first two maps of the probe's own feature batch.
Output: ../out/k1_probes.json, ../out/k1_probes.log
"""
import os, sys, json, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k1lib as K
R4 = os.path.join(K.ROOT, 'analyses', 'check_audit')
sys.path.insert(0, os.path.join(R4, 'scripts'))
import torch
import a3_probes as A3            # sets float64 default dtype
from check_model import check_model

meas = {r['id']: r for r in json.load(open(os.path.join(R4, 'out', 'a3_probes.json')))['rows']}
CAN = {'ADDITIVE': False, 'VACUOUS': False, 'NORM-ONLY': True, 'HARD-BLOCKED': True}
CANNOT_V = {'degenerate', 'class-token'}


def identity(x):
    return x


rows = []
for P in A3.PROBES:
    torch.manual_seed(0)
    m = P().eval()
    if P.gp is not None:
        m.global_pool = P.gp
    m.forward_features = identity
    if P.fmt == 'BTC':
        m.num_prefix_tokens = len(P.prefix)
    f = A3.make_feat(P)[:2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter(); cm = check_model(m, x=f, warn=False); t_orig = time.perf_counter() - t0
    s = K.check_model_sound(m, f, return_timing=True)
    g = K.check_model_sound(m, f, op_guard=True)
    lab = meas[P.pname]['measured']
    rows.append(dict(id=P.pname, probe=P.desc, measured=lab, can_differ=CAN[lab],
                     original=cm['verdict_check'], audited=cm['verdict'], sound=s['verdict'],
                     sound_failed=s['failed'], sound_why=s['why'], sound_branch=s['branch'],
                     posthoc_opguard=g['verdict'], posthoc_failed=g['failed'], posthoc_why=g['why'],
                     secs_original=t_orig, secs_sound=s['secs']))


def score(key):
    fd = [r['id'] for r in rows if r[key] == 'degenerate' and r['measured'] in ('NORM-ONLY', 'HARD-BLOCKED')]
    dec = [r for r in rows if r[key] != 'unknown']
    acc = [((r[key] in CANNOT_V) != r['can_differ']) for r in dec]
    return dict(false_degenerate=len(fd), false_degenerate_ids=fd, decided=len(dec), n=len(rows),
                coverage=len(dec) / len(rows), binary_correct=sum(acc),
                binary_accuracy_decided=sum(acc) / len(dec) if dec else None,
                binary_accuracy_all=sum(acc) / len(rows),
                degenerate_returned=sum(r[key] == 'degenerate' for r in rows),
                degenerate_on_vacuous=[r['id'] for r in rows if r[key] == 'degenerate' and r['measured'] == 'VACUOUS'],
                unknown_ids=[r['id'] for r in rows if r[key] == 'unknown'])


summ = {k: score(k) for k in ('original', 'audited', 'sound', 'posthoc_opguard')}
lines = ['K1 on the 32 constructed heads (float64 CPU); labels from analyses/check_audit/out/a3_probes.json', '']
for r in rows:
    lines.append('%s measured %-12s original %-22s audited %-22s sound %-22s failed %-10s post-hoc %-12s | %s'
                 % (r['id'], r['measured'], r['original'], r['audited'], r['sound'], r['sound_failed'] or '-',
                    r['posthoc_opguard'], r['probe']))
lines += ['']
for k, v in summ.items():
    lines.append('%-16s %s' % (k, json.dumps(v)))
txt = '\n'.join(lines) + '\n'
print(txt)
OUT = os.path.join(HERE, '..', 'out')
open(os.path.join(OUT, 'k1_probes.log'), 'w', encoding='utf-8').write(txt)
json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, 'k1_probes.json'), 'w'), indent=1)
