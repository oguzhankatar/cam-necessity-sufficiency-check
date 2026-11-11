"""Stage 3, task 2.  Effective number of cells of the necessity (N) and sufficiency (S) maps,
from what the archive already holds (nothing is re-run on images).

Participation ratio on the absolute map values, PR = (sum_t |m_t|)^2 / sum_t m_t^2, a number
of cells between 1 and T.

  WSOL   outputs/wsol/w2_*.json  per_map/<map>/concentration/participation_ratio is stored
         divided by T (scripts/wsol/w2_run.py concentration(): (sum|v|)^2 / (T sum v^2)),
         averaged over the evaluation records.  Effective cells = stored value * T, with
         T = grid[0] * grid[1] from the same file.  Arms: aggregator = N_lse / S_lse (LSE
         pool, the run's tau), relocation = N_ln / S_ln (ViT LayerNorm moved outside the
         mean, vitb16 runs only).  The 3x3-block (_b3) maps are reported alongside.
  Medical outputs/medical/m3_maps.json  stores the per-item maps; PR is computed here per
         item and averaged (m7_diag.json stores exp(entropy), a different effective number).
         Arm: aggregator at T = +8 and T = -8 (N_lse8.0 / S_lse8.0, N_lse-8.0 / S_lse-8.0).

Non-finite entries are listed per run and map.  A mechanism check calls scripts/wsol/wlib.ns_lse
on a constructed pair whose tau-scaled score maxima differ by more than the float64 exp underflow limit.

usage: python a10_participation.py  ->  ../out/a10_participation.json, ../out/a10_participation.log
"""
import os, sys, glob, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'wsol'))
OUT = os.path.join(HERE, '..', 'out')
ARMS = {'aggregator': ('N_lse', 'S_lse'), 'aggregator_b3': ('N_lse_b3', 'S_lse_b3'),
        'relocation': ('N_ln', 'S_ln'), 'relocation_b3': ('N_ln_b3', 'S_ln_b3')}


def spreads(n, s):
    if not (math.isfinite(n) and math.isfinite(s)):
        return 'non-finite'
    return 'sufficiency' if s > n else ('necessity' if n > s else 'equal')


res = dict(wsol={}, wsol_nonfinite={}, medical={}, medical_nonfinite={})
for f in sorted(glob.glob(os.path.join(ROOT, 'outputs', 'wsol', 'w2_*.json'))):
    run = os.path.basename(f)[3:-5]
    w = json.load(open(f))
    T = int(w['grid'][0] * w['grid'][1])
    pm = w['per_map']
    bad = sorted(k for k, v in pm.items() if not all(math.isfinite(x) for x in v['concentration'].values()))
    if bad:
        res['wsol_nonfinite'][run] = dict(tau=w['tau'], maps=bad)
    for arm, (kn, ks) in ARMS.items():
        if kn not in pm:
            continue
        n = pm[kn]['concentration']['participation_ratio'] * T
        s = pm[ks]['concentration']['participation_ratio'] * T
        res['wsol'].setdefault(run, {})[arm] = dict(tau=w['tau'], T=T, N_eff=n, S_eff=s, spreads_more=spreads(n, s))

M = json.load(open(os.path.join(ROOT, 'outputs', 'medical', 'm3_maps.json')))
Tm = int(M['T'])
for tau in ('8.0', '-8.0'):
    for blk in ('1x1', '3x3'):
        arm = 'aggregator T=%s %s' % (tau, blk)
        vals, nonfin, zero = {}, {}, {}
        for side in ('N', 'S'):
            key = '%s_lse%s_%s' % (side, tau, blk)
            pr, nf, z = [], 0, 0
            for it in M['items']:
                v = np.asarray(it['maps'][key], float)
                if not np.all(np.isfinite(v)):
                    nf += 1; continue
                a = np.abs(v); q = (v ** 2).sum()
                if q <= 0:
                    z += 1; continue
                pr.append(a.sum() ** 2 / q)
            vals[side] = float(np.mean(pr)); nonfin[side] = nf; zero[side] = z
        res['medical'][arm] = dict(T=Tm, n_items=len(M['items']), N_eff=vals['N'], S_eff=vals['S'],
                                   spreads_more=spreads(vals['N'], vals['S']),
                                   nonfinite_items=nonfin, zero_items=zero)
        if any(nonfin.values()):
            res['medical_nonfinite'][arm] = nonfin

# mechanism check: log(Sa) in ns_lse when every exp(tau*a - M) underflows
import wlib
lim = float(-np.log(np.nextafter(0.0, 1.0)))   # exp(-x) is 0 in float64 beyond about this x
demo = {}
for tau in (8.0, -8.0):
    gap = 1.01 * lim / abs(tau)                # tau-scaled maxima differ by 1.01 x the underflow limit
    a = np.zeros(4); b = np.zeros(4); b[0] = gap * np.sign(tau)
    with np.errstate(divide='ignore', invalid='ignore'):
        Nab, Sab = wlib.ns_lse(a, b, tau)
        Nba, Sba = wlib.ns_lse(b, a, tau)
    demo['tau=%g' % tau] = dict(score_gap=gap, N_finite_A_dominated=bool(np.all(np.isfinite(Nab))),
                                S_finite_A_dominated=bool(np.all(np.isfinite(Sab))),
                                N_finite_B_dominated=bool(np.all(np.isfinite(Nba))),
                                S_finite_B_dominated=bool(np.all(np.isfinite(Sba))))
res['ns_lse_underflow_demo'] = dict(
    exp_underflow_limit=lim, cases=demo,
    note='ns_lse: M = max(tau*a, tau*beta); Sa = sum exp(tau*a - M) and Sb likewise. When the '
         'other map holds the maximum by more than the exp underflow limit, every term underflows to 0 '
         'and np.log(Sa) / np.log(Sb) is -inf (EA and EB are guarded by np.finfo.tiny, Sa and '
         'Sb are not). One such record makes the per-record mean of the concentration NaN.')

lines = ['Effective number of cells (participation ratio on |m|), N vs S, per run and arm', '']
for run, arms in res['wsol'].items():
    for arm, d in arms.items():
        lines.append('wsol %-28s %-14s tau %5g  T %d  N_eff %8.3f  S_eff %8.3f  more spread: %s'
                     % (run, arm, d['tau'], d['T'], d['N_eff'], d['S_eff'], d['spreads_more']))
for arm, d in res['medical'].items():
    lines.append('medical %-26s T %d  n %d  N_eff %8.3f  S_eff %8.3f  more spread: %s  non-finite %s  zero %s'
                 % (arm, d['T'], d['n_items'], d['N_eff'], d['S_eff'], d['spreads_more'],
                    d['nonfinite_items'], d['zero_items']))
lines += ['', 'non-finite WSOL concentration: ' + json.dumps(res['wsol_nonfinite']),
          'non-finite medical maps: ' + json.dumps(res['medical_nonfinite']),
          'ns_lse underflow demo: ' + json.dumps(res['ns_lse_underflow_demo'])]
txt = '\n'.join(lines) + '\n'
print(txt)
open(os.path.join(OUT, 'a10_participation.log'), 'w', encoding='utf-8').write(txt)
json.dump(res, open(os.path.join(OUT, 'a10_participation.json'), 'w'), indent=1)
