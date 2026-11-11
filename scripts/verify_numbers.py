"""Re-derive the numbers the article asserts, straight from the raw result
files, and print them next to nothing else, so that each number in the text can
be compared with the data it came from.
"""
import json, glob, os, sys, collections
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'outputs')
L = lambda p: json.load(open(p))


def head(t):
    print('\n' + t)
    print('-' * len(t))


# ---------------------------------------------------------------- architectures
head('architecture census')
D = L(os.path.join(OUT, 'heldout', 'h1_D.json'))
HO = L(os.path.join(OUT, 'heldout', 'h1_A.json')) + L(os.path.join(OUT, 'heldout', 'h1_B.json'))
ALL = D + HO
names = {r['model'] for r in ALL}
print(f'development {len(D)}, held out {len(HO)}, unique total {len(names)}')
for tag, rows in (('development', D), ('all tested', ALL)):
    c = collections.Counter(r['verdict'] for r in rows)
    print(f'  {tag:12s} degenerate {c["IN-SCOPE"]}, norm-blocked {c["NORM-BLOCKED"]}, '
          f'nonlin-blocked {c["NONLIN-BLOCKED"]}, class-token {c["DEGENERATE"]}')
a1 = sum(1 for r in HO if r.get('agree') in (True, 1))
a2 = sum(1 for r in HO if r.get('agree_v2b') in (True, 1))
print(f'  held-out agreement: frozen {a1}/{len(HO)} ({a1/len(HO):.1%}), '
      f'after one repair {a2}/{len(HO)}')
# the frozen verdict is stored as 'v1' and the operator-trace verdict as 'v2b'
flip = [r['model'] for r in ALL if r.get('v1') != r.get('v2b')]
print(f'  verdicts changed by the operator-trace repair, over all {len(ALL)} '
      f'runs: {len(flip)} {flip}')
# Head diagnostics recorded by scripts/heldout/h1_run.py, not the certification
# residuals: seg_d0 / seg_d1 are the relative deviation of the logits from
# linearity along the segment between two feature maps (shipped head, and head
# with post-pool normalisations removed), perm_dev the relative logit change
# when the cells are permuted. The certification residuals eps1, eps2, eps3 of
# the article are stored per record under 'shipped' (maxNS, Rhalf, I2).
add = [max(r.get('seg_d0', 0), r.get('seg_d1', 0), r.get('perm_dev', 0))
       for r in ALL if r.get('gt') == 'ADDITIVE']
print(f'  worst head diagnostic (max of segment-linearity deviation and permutation '
      f'deviation; not eps1) over the {len(add)} certified-additive '
      f'architectures: {max(add):.2e}')

# ---------------------------------------------------------------- the edit
head('the normalisation edit, shipped ImageNet models')
rows = []
for f in sorted(glob.glob(os.path.join(OUT, 'probe25_lnplace_*.json'))):
    d = L(f)
    sh = d['meta']['shipped']
    ok = abs(d['none']['corrNS'] - 1) < 1e-3 and abs(d['pre']['corrNS'] - 1) < 1e-3
    # two one-line edits reach the degenerate point and they cost different
    # amounts; report both, never their maximum under one name
    if sh == 'post':
        c0, c1 = d['post']['corrNS'], d['pre']['corrNS']
        a0, a_rel, a_del = d['acc']['post'], d['acc']['pre'], d['acc']['none']
    else:
        c0, c1 = d['pre']['corrNS'], d['post']['corrNS']
        a0, a_rel = d['acc']['pre'], d['acc']['post']
        a_del = None          # deletion leaves this model degenerate already
    rows.append((d['meta']['model'].split('.')[0], sh, c0, c1,
                 (a_rel - a0) * 100,
                 None if a_del is None else (a_del - a0) * 100, ok))
inn = [r for r in rows if r[6]]
rel = np.array([r[4] for r in inn])
dele = np.array([r[5] for r in inn if r[5] is not None])
cheap = np.array([r[4] if r[5] is None else max(r[4], r[5]) for r in inn])
onto = [r for r in inn if r[1] == 'post']
off = [r for r in inn if r[1] == 'pre']
print(f'  {len(inn)} of {len(rows)} in scope')
print(f'  ship the normalisation AFTER the mean (not degenerate): {len(onto)}, '
      f'corr {min(r[2] for r in onto):.2f} to {max(r[2] for r in onto):.2f} '
      f'-> 1.000000 after the edit')
for r in off:
    print(f'  ship it INSIDE the mean (degenerate): {r[0]} '
          f'{r[2]:.6f} -> {r[3]:.3f} after the edit')
for nm, v in (('relocate across the mean', rel),
              ('delete the normalisation', dele),
              ('cheaper available edit', cheap)):
    print(f'  {nm:26s} worst {v.min():+6.2f} pp  mean {v.mean():+6.2f} pp  '
          f'equal or better in {(v >= 0).sum()} of {len(v)}')
# each column must read the key its header names
for r in inn:
    assert r[5] is None or r[1] == 'post', ('deletion price on a pre-shipped '
                                            'model: ' + r[0])
print(f'  out of scope: {[r[0] for r in rows if not r[6]]}')

# ---------------------------------------------------------------- repair cost
head('repair cost where a nonlinearity blocks (top-1 change of the cheapest edit)')
hard = np.array(sorted(r['y'] for r in L(os.path.join(OUT, 'heldout', 'h4_price.json'))['rows']
                       if r.get('y') is not None))
print(f'  development subset (h4_price): n = {len(hard)}, span {hard.min():+.2f} to '
      f'{hard.max():+.2f} pp, median {np.median(hard):+.2f}')
# the article's set: development and held-out nonlinearity-blocked
# architectures priced together (written by analyses/check_census/scripts/c7_summary.py)
_c7 = L(os.path.join(ROOT, 'analyses', 'check_census', 'out', 'c7_summary.json'))['all']
print(f'  all priced, as reported (C1 c7_summary): n = {_c7["n"]}, span '
      f'{_c7["cheapest_cost_max"]["pts"]:+.2f} to {_c7["cheapest_cost_min"]["pts"]:+.2f} pp, '
      f'median {_c7["cheapest_cost_median"]:+.2f}  (cost '
      f'{-_c7["cheapest_cost_min"]["pts"]:.2f} to {-_c7["cheapest_cost_max"]["pts"]:.2f} '
      f'points, median {-_c7["cheapest_cost_median"]:.2f})')

# ---------------------------------------------------------------- shipped 1k model
head('shipped ResNet-50 classifier, no parameter trained')
d = L(os.path.join(OUT, 'probe29_fan200.json'))   # 224 images, with intervals
r = {v['T']: v for k, v in d.items() if k.startswith('ship_')}
# Under log-sum-exp pooling against one reference per image, necessity and sufficiency
# order the cells identically (closed form; analyses/closed_form_maps/scripts/t1_algebra.py). The
# article's rank-based numbers are the closed-form values written by
# analyses/closed_form_maps/scripts/t2_in1k_fan.py, on the same 224 images and references as probe29.
# The Pearson correlation and the random-order deletion are read from probe29: the
# closed form does not change them.
C5 = L(os.path.join(ROOT, 'analyses', 'closed_form_maps', 'out', 't2_in1k_fan.json'))
c5 = {v['T']: v for k, v in C5.items() if k.startswith('ship_')}
ci = lambda x: f'{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]'
print("  the article's numbers (closed form, float64; analyses/closed_form_maps t2_in1k_fan), "
      "mean [95% CI] over 224 images:")
for T in sorted(r):
    if T in (-8.0, 0.0, 8.0):
        v, e = r[T], c5[T]
        # one ordering: the necessity and sufficiency entries coincide exactly
        assert e['spGN_ex'] == e['spGS_ex'] and e['delN_ex'] == e['delS_ex'], T
        print(f'  T = {T:+5.1f}  corr {v["corrNS"]:.6f}  rank corr(N, S) {e["spNS_ex"][0]:.6f}  '
              f'rho(grad, N) = rho(grad, S) {ci(e["spGN_ex"])}  '
              f'del_N = del_S {ci(e["delN_ex"])}  del_rand {v["del_rand"]:.3f}')
print('  float32 forward differences (superseded by the closed form; see analyses/closed_form_maps), '
      'probe29:')
for T in sorted(r):
    if T in (-8.0, 0.0, 8.0):
        v = r[T]
        print(f'    T = {T:+5.1f}  rho(grad, S) {v["spGS"]:.3f}  rho(grad, N) {v["spGN"]:.3f}  '
              f'del_S {v["del_S"]:.3f}  del_N {v["del_N"]:.3f}')
# the accuracies are stored inside each temperature's own record
print('  top-1 by temperature: ' + ',  '.join(
    '%s %.2f%%' % (k, v['acc'] * 100)
    for k, v in sorted(d.items())
    if isinstance(v, dict) and 'acc' in v and k.startswith('ship_')))

# ---------------------------------------------------------------- localization
head('localization, primary contrasts')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'wsol'))
from primary_set import select
P = select(L(os.path.join(OUT, 'wsol', 'w3_stats.json')))


def tally(rs):
    w = sum(1 for c in rs if c['p_holm'] < .05 and c['diff'] > 0)
    l = sum(1 for c in rs if c['p_holm'] < .05 and c['diff'] <= 0)
    return w, l, len(rs) - w - l


print(f'  {len(P)} primary contrasts')
for b in ('center', 'collapsed'):
    rs = [c for c in P if c['b'] == b]
    w, l, t = tally(rs)
    print(f'  vs {b:10s} win {w}, lose {l}, no difference {t}, of {len(rs)}')
# the per-backbone tallies below pool BOTH comparators (centre prior and collapsed map),
# i.e. all 192 contrasts; they are not a breakdown of the "vs collapsed" line above
print(f'  by backbone, both comparators pooled (all {len(P)} contrasts)')
for m in sorted({c['backbone'] for c in P}):
    rs = [c for c in P if c['backbone'] == m]
    w, l, t = tally(rs)
    print(f'    {m:10s} n {len(rs):3d}  win {w}, lose {l}, no difference {t}')
conv = [c for c in P if c['backbone'] != 'vitb16']
w, l, t = tally(conv)
print(f'    convolutional backbones together, both comparators  n {len(conv)}  win {w}, lose {l}, no difference {t}')

# ---------------------------------------------------------------- medical
head('medical, pre-specified primary cells')
m = L(os.path.join(OUT, 'medical', 'm4_stats.json'))
pre = m['prespecified']
k = [x for x in pre if x.startswith('H1_') and x.endswith('_hit')][0]
cells = [('NIH ChestX-ray14', pre[k])]
for f in ('m11_rsna.json', 'm5_isic.json'):
    d = L(os.path.join(OUT, 'medical', f))
    p = d.get('prespecified', d)
    kk = [x for x in p if str(x).startswith('H1_')]
    if kk:
        cells.append((f, p[kk[0]]))
for nm, c in cells:
    # a binary hit rate is tested with McNemar, a continuous score with Wilcoxon;
    # both are printed where both are stored
    tests = ' '.join(f'{k.split("_", 1)[1]} {c[k]:.3g}'
                     for k in ('p_mcnemar', 'p_wilcoxon') if k in c)
    print(f'  {nm:18s} metric {c.get("metric", "?"):5s} diff {c["diff"]:+.3f}  '
          f'95% CI [{c["lo"]:+.3f}, {c["hi"]:+.3f}]  n {c.get("n", "?")}  {tests}')
mm = m['summary']
for key in ('center', 'actnorm', 'collapsed_1x1', 'N_lse8.0_1x1'):
    if key in mm and 'hit' in mm[key]:
        print(f'  hit rate {key:16s} {mm[key]["hit"]["mean"]:.3f}')
print(f'  maps summarised: {len(mm)}')

# ---------------------------------------------------------------- baselines
head('class-independent baseline, paired test (m13)')
try:
    t = L(os.path.join(OUT, 'medical', 'm13_actnorm_test.json'))
    for k in ('all_findings', 'no_cardiomegaly'):
        a = t[k]
        print(f'  {a["tag"]:24s} n={a["n_items"]:4d}  activation magnitude {a["actnorm_hit"]:.3f}  '
              f'above all {a["n_model_derived"]} model maps: {a["n_below_point_estimate"] == a["n_model_derived"]}  '
              f'significant after Holm in {a["n_below_significant"]} of {a["n_model_derived"]}')
        top = max((r for r in a['rows'] if r['model_derived']), key=lambda r: r['other_rate'])
        # both P values, labelled: uncorrected and Holm-corrected
        print(f'    strongest model map: {top["other"]} {top["other_rate"]:.3f}, '
              f'diff {top["diff"]:+.3f}, P {top["p_mcnemar"]:.2g} uncorrected, '
              f'{top["p_holm"]:.2g} after Holm')
except FileNotFoundError:
    print('  m13_actnorm_test.json missing -- run scripts/medical/m13_actnorm_test.py')

# --------------------------------------------- numbers quoted in the running text
# Each of these is quoted in the running text, so each is re-derived here.
head('the held-out validation, against a constant predictor')
_ho = (L(os.path.join(OUT, 'heldout', 'h1_A.json'))
       + L(os.path.join(OUT, 'heldout', 'h1_B.json')))
_agree = sum(1 for r in _ho if r['agree'])
_maj = max(collections.Counter(r['verdict'] for r in _ho).values())
print(f'  agreement {_agree} of {len(_ho)} = {100 * _agree / len(_ho):.1f}%;  '
      f'majority verdict {_maj} of {len(_ho)} = {100 * _maj / len(_ho):.1f}%')

head('the localisation arms, effect sizes rather than counts (w9)')
_w9 = L(os.path.join(OUT, 'wsol', 'w9_percell.json'))
for _s in _w9['summary']:
    if _s['backbone'] != 'all':
        continue
    _sub = [r for r in _w9['rows'] if r['arm'] == _s['arm']
            and r['pair'].endswith(_s['comparator'])]
    print('  %-24s vs %-12s n=%3d  win %2d lose %2d tie %2d   '
          'median all %+.4f  median win %s  median loss %s'
          % (_s['arm'], _s['comparator'], _s['n'], _s['win'], _s['loss'],
             _s['tie'], _s['median_all'],
             '%+.4f' % _s['median_win'] if _s['median_win'] is not None else '   --  ',
             '%+.4f' % _s['median_loss'] if _s['median_loss'] is not None else '   --  '))
for _s in _w9['summary']:
    if _s['backbone'] != 'all' and _s['comparator'] == 'collapsed':
        print('    %-22s %-10s win %2d lose %2d tie %2d'
              % (_s['arm'], _s['backbone'], _s['win'], _s['loss'], _s['tie']))
_cl = collections.Counter(r['dataset'] for r in _w9['rows']
                          if r['pair'].endswith('centre prior') and r['sig'] == 'loss')
print('  centre-prior losses by dataset: %s' % dict(_cl))
_cub = {r['metric']: r['vb'] for r in _w9['rows']
        if r['dataset'] == 'cub' and r['pair'].endswith('centre prior')}
print('  the centre prior on CUB: ' + ',  '.join(
    '%s %.4f' % (k, v) for k, v in sorted(_cub.items(), key=lambda kv: -kv[1])))

head('class-token readouts at the site where the probe means anything')
# The two runs of this probe do not supersede one another: h3_cls_A.json holds
# the corrected part_a, and h3_cls.json is the only run that carries part_b.
# See outputs/heldout/README-h3.md.
_pb = L(os.path.join(OUT, 'heldout', 'h3_cls.json'))['part_b']
_sites = [(r['model'], t) for r in _pb for t in r.get('sites', []) if 'maxNS' in t]
_mx = [t['maxNS'] for _, t in _sites]
print('  %d models, %d sites; smallest residual %.4f, largest %.3f; '
      'additive at any site: %s'
      % (len({k for k, _ in _sites}), len(_sites), min(_mx), max(_mx),
         any(t.get('additive') for _, t in _sites)))

head('the reference sensitivity sweep (rt2)')
_rt = os.path.join(ROOT, 'scripts', 'crosscheck')
for _f in sorted(glob.glob(os.path.join(_rt, 'rt2_*.json'))):
    _j = L(_f)
    _b = _j['baselines']
    _sh = {k: v[0] for k, v in _b.items() if not k.startswith('pre_')}
    _above = sum(1 for v in _sh.values() if v > 0.92)
    print('  %-34s bank %.4f  shuffleA %.4f  above 0.92: %d of %d references'
          % (_j['meta']['model'].split('.')[0], _b['bank'][0], _b['shuffleA'][0],
             _above, len(_sh)))

head('Pearson against Spearman on the 1,000-class sweep (probe29; Spearman from analyses/closed_form_maps)')
# Pearson from probe29 (the closed form leaves it unchanged); Spearman from the closed
# form (analyses/closed_form_maps t2_in1k_fan), which is 1 at every temperature
print("  the article's numbers:")
_old = []
for _k, _v in L(os.path.join(OUT, 'probe29_fan200.json')).items():
    if not (isinstance(_v, dict) and 'corrNS' in _v and _k.startswith('ship_')):
        continue
    print('  %-16s Pearson %.6f   Spearman %.6f'
          % (_k, _v['corrNS'], C5[_k]['spNS_ex'][0]))
    if 'spNS' in _v:
        _old.append('%s %.6f' % (_k, _v['spNS']))
print('  float32 forward differences (superseded by the closed form; see analyses/closed_form_maps), '
      'Spearman:')
print('    ' + ',  '.join(_old))

head('Table S15\'s correlation column, with and without the lowest row')
_on = []
for _f in sorted(glob.glob(os.path.join(OUT, 'probe25_lnplace_*.json'))):
    _d = L(_f)
    _ok = (abs(_d['none']['corrNS'] - 1) < 1e-3
           and abs(_d['pre']['corrNS'] - 1) < 1e-3)
    if _ok and _d['meta']['shipped'] == 'post':
        _on.append((_d['post']['corrNS'], _d['meta']['model'].split('.')[0]))
_on.sort()
print('  %d models ship it after the mean: %.4f (%s) to %.4f (%s)'
      % (len(_on), _on[0][0], _on[0][1], _on[-1][0], _on[-1][1]))
print('  without the lowest: %.4f to %.4f' % (_on[1][0], _on[-1][0]))

# ------------------------------------------------- the aggregator replacements
head('non-additive aggregator replacements (probe1)')
_a = {}
def _walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, dict) and 'acc' in v and 'corrNS' in v:
                _a[k] = v
            else:
                _walk(v)
    elif isinstance(o, list):
        for v in o:
            _walk(v)
_walk(L(os.path.join(OUT, 'probe1_aggregator.json')))
_base = _a['gap_fc']                      # the averaged readout the text compares against
assert abs(1 - _base['corrNS']) < 5e-4, 'the baseline is not degenerate'
_rep = {k: v for k, v in _a.items()
        if 1 - v['corrNS'] > 5e-3 and k != 'gap_mlp'}   # gap_mlp is a tail change
_sep = sorted((1 - v['corrNS'], k) for k, v in _rep.items())
_pp = sorted(((v['acc'] - _base['acc']) * 100, k) for k, v in _rep.items())
_nagg = sum(1 for k in _rep if k != 'bilin')    # bilin keeps the mean: a tail change
print(f'  {_nagg} aggregators plus the bilinear tail ({len(_rep)} replacements); separation {_sep[0][0]:.3f} ({_sep[0][1]})'
      f' to {_sep[-1][0]:.3f} ({_sep[-1][1]})')
print(f'  top-1 vs the averaged readout: {_pp[0][0]:+.3f} pp ({_pp[0][1]})'
      f' to {_pp[-1][0]:+.3f} pp ({_pp[-1][1]}); at most'
      f' {max(abs(_pp[0][0]), abs(_pp[-1][0])):.2f} pp')
print('  Fig. S2(d) and Fig. 2(b) trio: ' + ', '.join(
    f'{k} {1 - _a[k]["corrNS"]:.3f}' for k in ('lse1.0', 'attn', 'bilin')))

# ---------------------------------------------------------------- localisation arms
head('localisation, by how the separation was obtained')
for tag, suffix in (('relocating the normalisation', '_ln'),
                    ('replacing the aggregator', '_lse')):
    rs = [c for c in P if c['b'] == 'collapsed' and c['a'].endswith(suffix)]
    w = sum(1 for c in rs if c['p_holm'] < .05 and c['diff'] > 0)
    l = sum(1 for c in rs if c['p_holm'] < .05 and c['diff'] <= 0)
    bk = sorted({c['backbone'] for c in rs})
    print(f'  {tag:30s} n={len(rs):3d}  win {w:3d}  lose {l:3d}  tie {len(rs)-w-l:3d}   backbones: {bk}')

head('the pre-normalisation control (w7)')
try:
    t = L(os.path.join(OUT, 'wsol', 'w7_control_table.json'))
    g = [e['control_minus_best_separated']
         for ds, r in t.items() if ds != 'summary' for e in r['metrics'].values()]
    c = [e['control_collapsed_pre'] - e['collapsed']
         for ds, r in t.items() if ds != 'summary' for e in r['metrics'].values()
         if 'collapsed' in e]
    print(f'  cells {len(g)};  control matches or beats the better separated map in '
          f'{t["summary"]["control_matches_or_beats"]}')
    print(f'  control minus better separated: worst {min(g):+.4f}, best {max(g):+.4f}')
    print(f'  control minus collapsed       : worst {min(c):+.4f}, best {max(c):+.4f}, '
          f'above it in {sum(1 for v in c if v > 0)} of {len(c)}')
except FileNotFoundError:
    print('  w7_control_table.json missing -- run scripts/wsol/w7_control_table.py')

head('the control, paired tests (w8)')
try:
    t = L(os.path.join(OUT, 'wsol', 'w8_control_test.json'))
    for k, v in t['summary'].items():
        print(f"  {k:16s} of {v['n']} paired comparisons: better {v['better']}, "
              f"worse {v['worse']}, indistinguishable {v['no_difference']}")
except FileNotFoundError:
    print('  w8_control_test.json missing -- run scripts/wsol/w8_control_test.py')

# ------------------------------------------- conservative check, measured census (analyses/conservative_check_census)
# read from the stored per-entry rows of analyses/conservative_check_census/out/k1_verdicts.json (written by
# k1_report.py from the k1_run_*.jsonl records); nothing is recomputed
K1OUT = os.path.join(ROOT, 'analyses', 'conservative_check_census', 'out')      # analyses/conservative_check_census/out
K2DIR = os.path.join(ROOT, 'analyses', 'end_to_end_cub')             # analyses/end_to_end_cub
K3OUT = os.path.join(ROOT, 'analyses', 'isic_three_seeds', 'out')      # analyses/isic_three_seeds/out
_K1 = L(os.path.join(K1OUT, 'k1_verdicts.json'))
_CAN = {'ADDITIVE': False, 'VACUOUS': False, 'NORM-ONLY': True, 'HARD-BLOCKED': True}
_LAB_OF = {'degenerate': 'ADDITIVE', 'class-token': 'VACUOUS', 'normalisation-blocked': 'NORM-ONLY',
           'nonlinearity-blocked': 'HARD-BLOCKED', 'aggregator-blocked': 'HARD-BLOCKED'}
_CHECKS = (('original', 'original'), ('audited', 'audited'), ('conservative', 'sound'))

head('measured census of 895 entries (Table 2, Measured column; Note S2.6)')
_rows = _K1['verdicts']['census']
_rein = _K1['census']['zero_readout']['reinit_labels']
_lab = {r['name']: (_rein.get(r['name']) if r['label'] == 'ZERO-READOUT' else r['label']) for r in _rows}
_cnt = collections.Counter(_lab.values())
print(f'  {len(_rows)} entries ({len(_rows) - len(_rein)} certified directly, {len(_rein)} after re-drawing '
      f'zero-initialised weights): additive {_cnt["ADDITIVE"]}, normalisation-only {_cnt["NORM-ONLY"]}, '
      f'hard-blocked {_cnt["HARD-BLOCKED"]}, vacuous {_cnt["VACUOUS"]}')
for nm, k in _CHECKS:
    _ag = sum(1 for r in _rows if _LAB_OF.get(r[k]) == _lab[r['name']])
    _bi = sum(1 for r in _rows if r[k] != 'unknown' and (r[k] in ('degenerate', 'class-token')) != _CAN[_lab[r['name']]])
    _fd = sum(1 for r in _rows if r[k] == 'degenerate' and _lab[r['name']] in ('NORM-ONLY', 'HARD-BLOCKED'))
    print(f'  {nm:12s} agreement {_ag}/{len(_rows)}, binary {_bi}/{len(_rows)}, false degenerate {_fd}'
          + ('  (in-sample)' if nm == 'conservative' else ''))
_dif = [r for r in _rows if r['label_own'] != r['label_unit'] and r['label'] != 'ZERO-READOUT']
print(f'  own-feature and unit-scale labels differ on {len(_dif)} entries (the unit-scale label is used)')
for nm, k in _CHECKS:
    _fo = sum(1 for r in _rows if r[k] == 'degenerate' and r['label_own'] in ('NORM-ONLY', 'HARD-BLOCKED'))
    print(f'  scored against own-feature labels, {nm:12s} false degenerate {_fo}')

_OT = L(os.path.join(K1OUT, 'k1_out_of_trace.json'))
_nd = _OT['no_dev_trace']
print(f'  original check, entries whose head trace is not a development trace (post hoc): {_nd["agree"]} of '
      f'{_nd["n"]} (majority verdict {_nd["majority_verdict"]}: {_nd["majority_agree"]}); the other '
      f'{_OT["n_census"] - _nd["n"]} entries share a development trace and are all correct '
      f'({_OT["dev_trace_direct"]["agree"]} of {_OT["dev_trace_direct"]["n"]} certified directly, '
      f'{_OT["zero_readout_on_dev_traces"]} zero-readout)')

head('the three checks on constructed heads and torchvision (Table S16; Note S2.6)')
_pr = _K1['probes']
for nm, k in _CHECKS:
    p = _pr[k]
    print(f'  {len(_K1["verdicts"]["probes"])} constructed heads, {nm:12s} false degenerate {p["false_degenerate"]}, '
          f'decided {p["decided"]}, decided correct {p["binary_correct"]} of {p["decided"]}')
_tv = _K1['torchvision']
for nm, k in _CHECKS:
    t, tr = _tv[k], _tv['with_reinit'][k]
    print(f'  torchvision, {nm:12s} false degenerate {t["false_degenerate"]} of {_tv["n_labelled"]} labelled; '
          f'decided {tr["decided"]} of {_tv["with_reinit"]["n"]} classifiers')

# ------------------------------------------- heads trained end to end (analyses/end_to_end_cub)
head('heads trained end to end on CUB, primary contrasts vs the H0 collapsed map (Table S18; Note S2.8)')
_K2 = L(os.path.join(K2DIR, 'out', 'k2_contrasts.json'))
for r in _K2['primary']:
    print(f'  {r["a"]:6s} {r["metric"]} {r["diff"]:+.3f} [{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}]  '
          f'per seed {", ".join("%+.3f" % v for v in r["diff_per_seed"])}  P_Holm {r["p_holm"]:.3f}  {r["verdict"]}')
print(f'  better {sum(r["verdict"] == "better" for r in _K2["primary"])}, '
      f'worse {sum(r["verdict"] == "worse" for r in _K2["primary"])} of {len(_K2["primary"])}')

head('post hoc block of Table S18: natural map of each head vs the H0 collapsed map (Note S2.8)')
_NAT = L(os.path.join(K2DIR, 'post', 'out', 'k2_natural.json'))
for r in _NAT['natural_vs_collapsed_MaxBoxAccV2']:
    print(f'  {r["a_label"]:32s} {r["diff"]:+.4f} [{r["ci"][0]:+.4f}, {r["ci"][1]:+.4f}]  '
          f'P_Holm {r["p_holm"]:.3f}  {r["verdict"]}')
_HP = L(os.path.join(K2DIR, 'post', 'out', 'k2_h3_pair.json'))['raw_MaxBoxAccV2']
print(f'  H3 CAM (LayerNorm ignored) minus H3 sufficiency, MaxBoxAccV2 {_HP["diff"]:+.4f} '
      f'[{_HP["ci"][0]:+.4f}, {_HP["ci"][1]:+.4f}]  P {_HP["p"]:.3f} (single contrast)')

head('post hoc: MaxBoxAccV2 on within-image ranks of every 14x14 map, vs the H0 collapsed map (Note S2.8)')
_RK = {r['a']: r for r in L(os.path.join(K2DIR, 'post', 'out', 'k2_rank.json'))['contrasts']['rank']
       if r['b'] == 'H0_collapsed'}
for a, lab in (('H1_N', 'H1 necessity'), ('H3_S', 'H3 sufficiency'), ('H2_attn', 'H2 attention weights'),
               ('H2_attn_ev', 'H2 attention x class evidence')):
    r = _RK[a]
    print(f'  {lab:30s} {r["diff"]:+.4f} [{r["ci"][0]:+.4f}, {r["ci"][1]:+.4f}]  P_Holm {r["p_holm"]:.3f}  {r["verdict"]}')

# ------------------------------------------- ISIC, three retrained classifiers (analyses/isic_three_seeds)
head('ISIC, three retrained classifiers: primary Dice contrast, closed form (Table 3; Table S19; Note S2.9)')
for s in range(3):
    _p = L(os.path.join(K3OUT, 'k3_seed%d.json' % s))['primary']
    c = _p['closed_form']
    print(f'  seed {s}  {_p["key"]}  {c["diff"]:+.3f} [{c["lo"]:+.3f}, {c["hi"]:+.3f}]  n {c["n"]}')
