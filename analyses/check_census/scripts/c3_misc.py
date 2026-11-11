"""C1 tasks 3, 4, 7, 8 -- read-only analysis of archived outputs.

 3  separation measured on every blocked architecture (h1_*.json shipped):
    1 - corrNS (min and mean over the 3 probe pairs), maxNS, Rhalf, I2.
 4  provenance: hashes (raw and CRLF-normalised) vs h0_freeze.json, file mtimes,
    the import of rule_v2 in h1_run, crossvit, volo, verdicts changed by repair.
 7  prices: every edit tried in scope/s3_edits.json, scope/s3v_vgg16.json and
    scope/h5_edits_heldout.json; which reach additivity and at what cost.
 8  residuals over the 89 certified-additive architectures.
"""
import os, json, hashlib, re, datetime, collections
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
J = lambda *p: json.load(open(os.path.join(ROOT, *p)))

rows = []
for s in 'DAB':
    rows += J('outputs', 'heldout', 'h1_%s.json' % s)
meas = [r for r in rows if 'gt' in r]
res = {}


def dist(v):
    v = np.asarray(v, float)
    return dict(n=len(v), min=float(v.min()), median=float(np.median(v)), max=float(v.max()))


# ------------------------------------------------------------------ task 3
blk = [r for r in meas if r['verdict'] in ('NORM-BLOCKED', 'NONLIN-BLOCKED', 'AGG-BLOCKED')
       or r['gt'] in ('NORM-ONLY', 'HARD-BLOCKED')]
t3rows = []
for r in blk:
    sh = r['shipped']
    cm = float(np.mean([p['corr_pearson'] for p in sh['pairs']]))
    t3rows.append(dict(model=r['model'], split='dev' if r['split'] == 'D' else 'heldout',
                       verdict=r['verdict'], gt=r['gt'],
                       one_minus_corr_min=1 - sh['corrNS_min'], one_minus_corr_mean=1 - cm,
                       maxNS=sh['maxNS'], Rhalf=sh['Rhalf'], I2=sh['I2']))
t3 = dict(rows=sorted(t3rows, key=lambda d: d['maxNS']))
for grp, f in (('all', lambda d: True),
               ('dev', lambda d: d['split'] == 'dev'), ('heldout', lambda d: d['split'] == 'heldout'),
               ('NORM-ONLY', lambda d: d['gt'] == 'NORM-ONLY'),
               ('HARD-BLOCKED', lambda d: d['gt'] == 'HARD-BLOCKED'),
               ('heldout_NORM-ONLY', lambda d: d['split'] == 'heldout' and d['gt'] == 'NORM-ONLY'),
               ('heldout_HARD-BLOCKED', lambda d: d['split'] == 'heldout' and d['gt'] == 'HARD-BLOCKED')):
    sub = [d for d in t3rows if f(d)]
    t3[grp] = {k: dist([d[k] for d in sub]) for k in
               ('one_minus_corr_min', 'one_minus_corr_mean', 'maxNS', 'Rhalf', 'I2')}
    t3[grp]['argmin_maxNS'] = min(sub, key=lambda d: d['maxNS'])['model']
    t3[grp]['argmin_1mcorr_mean'] = min(sub, key=lambda d: d['one_minus_corr_mean'])['model']
res['task3_separation_blocked'] = t3

# ------------------------------------------------------------------ task 8
add = [r for r in meas if r['gt'] == 'ADDITIVE']
t8 = dict(n=len(add))
for k in ('maxNS', 'Rhalf', 'I2'):
    r = max(add, key=lambda r: r['shipped'][k])
    t8[k] = dict(max=r['shipped'][k], model=r['model'])
r = max(add, key=lambda r: max(r['shipped'][k] for k in ('maxNS', 'Rhalf', 'I2')))
t8['any_of_three'] = dict(max=max(r['shipped'][k] for k in ('maxNS', 'Rhalf', 'I2')), model=r['model'])
r = max(add, key=lambda r: max(r['seg_d0'], r['seg_d1'], r['perm_dev']))
t8['verify_numbers_quantity_seg_d0_d1_perm_dev'] = dict(
    max=max(r['seg_d0'], r['seg_d1'], r['perm_dev']), model=r['model'])
vac = [r for r in meas if r['gt'] == 'VACUOUS']
t8['vacuous_n'] = len(vac)
t8['vacuous_max_any'] = max(max(r['shipped'][k] for k in ('maxNS', 'Rhalf', 'I2')) for r in vac)
hb = [r for r in meas if r['gt'] in ('NORM-ONLY', 'HARD-BLOCKED')]
mn = min(hb, key=lambda r: r['shipped']['maxNS'])
t8['smallest_maxNS_among_failing'] = dict(value=mn['shipped']['maxNS'], model=mn['model'])
allres = [max(r['shipped'][k] for k in ('maxNS', 'Rhalf', 'I2')) for r in meas]
t8['n_in_gap_1e-12_to_1e-4'] = int(sum(1e-12 < v < 1e-4 for v in allres))
res['task8_residuals_89'] = t8

# ------------------------------------------------------------------ task 4
def sha(p, crlf=False):
    b = open(p, 'rb').read()
    if crlf:
        b = b.replace(b'\r\n', b'\n')
    return hashlib.sha256(b).hexdigest()


H0 = J('outputs', 'heldout', 'h0_freeze.json')
fr = os.path.join(ROOT, 'scripts', 'heldout', 'frozen_rule.py')
s4 = os.path.join(ROOT, 'scripts', 'scope', 's4_decide.py')
sl = os.path.join(ROOT, 'scripts', 'scope', 'slib.py')
rawfr = open(fr, 'rb').read()
t4 = dict(
    frozen_rule_sha_raw=sha(fr), frozen_rule_sha_crlf_to_lf=sha(fr, True),
    frozen_rule_recorded=H0['sha256_frozen_file'],
    frozen_rule_has_crlf=b'\r\n' in rawfr,
    frozen_rule_match_raw=sha(fr) == H0['sha256_frozen_file'],
    frozen_rule_match_lf=sha(fr, True) == H0['sha256_frozen_file'],
    s4_decide_match=sha(s4) == H0['sha256_s4_decide'],
    slib_match=sha(sl) == H0['sha256_slib'],
    s4_decide_has_crlf=b'\r\n' in open(s4, 'rb').read(),
    slib_has_crlf=b'\r\n' in open(sl, 'rb').read(),
)
# per-block hashes against the AST segments (same extraction as h0_freeze)
import ast
def blocks(path, names):
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    got = {}
    for node in tree.body:
        nm = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in names:
            got[nm] = ast.get_source_segment(src, node)
    return [(n, got[n]) for n in names]
parts = blocks(sl, ['NORMISH', 'ACTISH', 'is_norm', 'is_act']) + \
    blocks(s4, ['AFFINE_CLS', 'POOL_AVG', 'cells_of', 'trace_head', 'kind', 'structural'])
t4['block_hashes_match'] = sum(hashlib.sha256(s.encode()).hexdigest() == b['sha256']
                               for (n, s), b in zip(parts, H0['blocks']))
t4['n_blocks'] = len(H0['blocks'])
mt = {}
for p in ('scripts/scope/slib.py', 'scripts/scope/s4_decide.py', 'outputs/scope/s4_decide.json',
          'scripts/heldout/h0_freeze.py', 'scripts/heldout/frozen_rule.py', 'outputs/heldout/h0_freeze.json',
          'outputs/heldout/h2_diagnose.json', 'scripts/heldout/rule_v2.py', 'scripts/heldout/h1_run.py',
          'scripts/heldout/hlib.py', 'outputs/heldout/h1_D.json', 'outputs/heldout/h1_B.json',
          'outputs/heldout/h1_A.json'):
    mt[p] = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(ROOT, p))).isoformat(' ', 'seconds')
t4['file_mtimes_local'] = mt
h1src = open(os.path.join(ROOT, 'scripts', 'heldout', 'h1_run.py'), encoding='utf-8').read().splitlines()
t4['h1_run_rule_v2_lines'] = [(i + 1, l.strip()) for i, l in enumerate(h1src) if 'rule_v2' in l or 'R2.' in l]
cv = [r for r in rows if r['model'] == 'crossvit_9_240'][0]
t4['crossvit'] = dict(error=cv.get('error'), v1=cv['v1'], v2a=cv['v2a'], v2b=cv['v2b'], v2b_agg=cv['v2b_agg'])
vo = [r for r in rows if r['model'] == 'volo_d1_224'][0]
t4['volo'] = dict(v1=vo['v1'], v2a=vo['v2a'], v2b=vo['v2b'], gt=vo['gt'], agg_name=vo['agg_name'],
                  maxNS=vo['shipped']['maxNS'], Rhalf=vo['shipped']['Rhalf'], I2=vo['shipped']['I2'],
                  corrNS_min=vo['shipped']['corrNS_min'], n_active=vo['shipped']['n_active'],
                  spatial_mass=vo['shipped']['spatial_mass'], module_pool=vo['module_pool'])
t4['repair_changes_v1_to_v2b'] = [(r['model'], r['split'], r['v1'], r['v2b']) for r in rows if r['v1'] != r['v2b']]
t4['repair_changes_v1_to_v2a'] = [(r['model'], r['split'], r['v1'], r['v2a']) for r in rows if r['v1'] != r['v2a']]
t4['counts'] = dict(dev_listed=sum(r['split'] == 'D' for r in rows),
                    dev_measured=sum(r['split'] == 'D' and 'gt' in r for r in rows),
                    heldout=sum(r['split'] in 'AB' for r in rows),
                    A=sum(r['split'] == 'A' for r in rows), B=sum(r['split'] == 'B' for r in rows))
t4['verdicts_v1_all148'] = dict(collections.Counter(r['verdict'] for r in rows))
t4['verdicts_v1_dev66'] = dict(collections.Counter(r['verdict'] for r in rows if r['split'] == 'D'))
t4['module_pool_false'] = sum(not r['module_pool'] for r in rows)
t4['module_pool_false_heldout'] = sum((not r['module_pool']) and r['split'] in 'AB' for r in rows)
t4['agree'] = dict(dev=sum(r['agree'] for r in meas if r['split'] == 'D'),
                   heldout=sum(r['agree'] for r in meas if r['split'] in 'AB'))
t4['class_token_verdicts'] = [(r['model'], r['split'], r.get('gt')) for r in rows if r['verdict'] == 'DEGENERATE']
names_dev = open(os.path.join(ROOT, 'scripts', 'heldout', 'names_D.txt')).read().split()
names_dev2 = open(os.path.join(ROOT, 'scripts', 'scope', 'names.txt')).read().split()
t4['names_D_equals_scope_names'] = names_dev == names_dev2
res['task4_provenance'] = t4

# ------------------------------------------------------------------ task 7
S3 = J('outputs', 'scope', 's3_edits.json')
S3V = J('outputs', 'scope', 's3v_vgg16.json')
H5 = J('outputs', 'scope', 'h5_edits_heldout.json')
H4 = {r['model']: r['y'] for r in J('outputs', 'heldout', 'h4_price.json')['rows']}
t7 = []
for src, lst in (('s3_edits', S3), ('s3v_vgg16', S3V), ('h5_edits_heldout', H5)):
    for r in lst:
        if 'cfgs' not in r:
            t7.append(dict(source=src, model=r['model'], error=r.get('error', '')[:200]))
            continue
        ed = {}
        for e, c in r['cfgs'].items():
            if e == 'shipped':
                continue
            ed[e] = dict(additive=c['additive'], dacc_pts=c['dacc_pts'],
                         maxNS=c.get('max_abs_diff_rel', c.get('maxNS')),
                         corrNS_mean=c.get('corrNS_mean'))
        reach = {e: v for e, v in ed.items() if v['additive']}
        best = max(reach, key=lambda e: reach[e]['dacc_pts']) if reach else None
        flag = None
        if best in ('delboth',):
            flag = 'cheapest reaching edit deletes the normalisation AND the activation(s); delact alone additive=%s' % (
                ed.get('delact', {}).get('additive'))
        elif best == 'poollast':
            flag = 'cheapest reaching edit moves the whole post-pool map inside the mean (no deletion)'
        elif best == 'linact':
            flag = 'cheapest reaching edit replaces the activation by its fitted affine part'
        t7.append(dict(source=src, model=r['model'], acts=r.get('acts'), norms=r.get('norms', []),
                       shipped_acc=r['cfgs']['shipped']['acc'], edits=ed,
                       reaching=sorted(reach), best_edit=best,
                       best_cost_pts=reach[best]['dacc_pts'] if best else None,
                       delact_alone_additive=ed.get('delact', {}).get('additive'),
                       delact_cost=ed.get('delact', {}).get('dacc_pts'),
                       h4_price_y=H4.get(r['model']), flag=flag))
res['task7_prices'] = t7

json.dump(res, open(os.path.join(OUT, 'c3_misc.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------ print
print('== task 3 (blocked separation)')
for g in ('all', 'dev', 'heldout', 'NORM-ONLY', 'HARD-BLOCKED', 'heldout_NORM-ONLY', 'heldout_HARD-BLOCKED'):
    d = t3[g]
    print('  %-22s n=%d  1-corr(mean) %s | 1-corr(min) %s | maxNS %s | Rhalf %s | argmin maxNS %s' % (
        g, d['maxNS']['n'],
        '%.3f/%.3f/%.3f' % tuple(d['one_minus_corr_mean'][k] for k in ('min', 'median', 'max')),
        '%.3f/%.3f/%.3f' % tuple(d['one_minus_corr_min'][k] for k in ('min', 'median', 'max')),
        '%.3f/%.3f/%.3f' % tuple(d['maxNS'][k] for k in ('min', 'median', 'max')),
        '%.2e/%.2e/%.2e' % tuple(d['Rhalf'][k] for k in ('min', 'median', 'max')), d['argmin_maxNS']))
for d in t3['rows']:
    print('    %-36s %-7s %-14s %-12s 1-c(mean)=%.3f 1-c(min)=%.3f maxNS=%.3f Rh=%.3g I2=%.3g' % (
        d['model'], d['split'], d['verdict'], d['gt'], d['one_minus_corr_mean'], d['one_minus_corr_min'],
        d['maxNS'], d['Rhalf'], d['I2']))
print('== task 8', json.dumps(t8, indent=1))
print('== task 4', json.dumps({k: v for k, v in t4.items()}, indent=1, default=str))
print('== task 7')
for d in t7:
    if 'error' in d:
        print('  %-18s %-26s ERROR %s' % (d['source'], d['model'], d['error'][:90])); continue
    print('  %-18s %-26s best=%-9s cost=%s delact_add=%s delact_cost=%s y=%s reach=%s | %s' % (
        d['source'], d['model'], d['best_edit'],
        None if d['best_cost_pts'] is None else '%.3f' % d['best_cost_pts'],
        d['delact_alone_additive'], None if d['delact_cost'] is None else '%.3f' % d['delact_cost'],
        d['h4_price_y'], d['reaching'], d['flag']))
