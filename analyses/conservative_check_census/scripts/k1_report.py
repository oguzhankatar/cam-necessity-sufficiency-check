"""K1 / K1b tally: every verdict in one JSON, a log, and RESULTS.md.

Inputs (all under ../out unless noted): k1_whitelist.json, k1_probes.json,
k1_run_timm_*.jsonl, k1_run_tv_*.jsonl, k1_tv_verify.json, k1_lse.json;
stored labels of the 148 architectures: outputs/heldout/h1_{D,A,B}.json (field gt);
census population: analyses/check_census/out/c5_population.json, c5_census.json;
audited labels of the 30 unmeasured census traces: analyses/check_audit/out/a1_audit.json.
"""
import os, sys, json, glob, hashlib, statistics, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(HERE, '..', 'out')
J = lambda *p: json.load(open(os.path.join(*p)))

CANNOT = {'degenerate', 'class-token'}
LAB_OF = {'degenerate': 'ADDITIVE', 'class-token': 'VACUOUS', 'normalisation-blocked': 'NORM-ONLY',
          'nonlinearity-blocked': 'HARD-BLOCKED', 'aggregator-blocked': 'HARD-BLOCKED', 'unknown': None}
CAN = {'ADDITIVE': False, 'VACUOUS': False, 'NORM-ONLY': True, 'HARD-BLOCKED': True}
CHECKS = ('original', 'audited', 'sound', 'posthoc_opguard')


def load_runs(pat):
    rows = {}
    for p in sorted(glob.glob(os.path.join(OUT, pat))):
        for l in open(p):
            r = json.loads(l)
            if 'shipped' in r and r['shipped']['absN_max'] == 0.0 and r['shipped_unit']['absN_max'] == 0.0:
                r['label_raw'] = r['label']
                r['label'] = 'ZERO-READOUT'        # readout identically zero at initialisation
            elif 'shipped' in r and r['name'].startswith('pit_') and r['name'].endswith('_distilled_224') and r['T'] <= 2:
                # class-token by inspection: PiT-distilled forward_features returns only its class and
                # distillation tokens (T = 2), so the certificate's "cells" contain no spatial cell
                r['label_raw'] = r['label']
                r['label'] = 'VACUOUS'
                r['label_by_inspection'] = True
            rows[r['name']] = r
    return rows


def with_reinit(rows, rein):
    """copies of rows whose ZERO-READOUT label is replaced by the label after re-drawing the zero weights."""
    out = []
    for r in rows:
        r = dict(r)
        if r.get('label') == 'ZERO-READOUT' and r['name'] in rein and rein[r['name']].get('label') in CAN:
            r['label'] = rein[r['name']]['label']
        out.append(r)
    return out


def score(rows, labkey):
    """rows: list of dicts with the check verdicts and a label under labkey."""
    lab = [r for r in rows if r.get(labkey) in CAN]
    res = dict(n=len(rows), n_labelled=len(lab))
    for k in CHECKS:
        v = [r for r in rows if k in r]
        dec = [r for r in v if r[k] != 'unknown']
        ldec = [r for r in lab if r[k] != 'unknown']
        fd = [r['name'] for r in lab if r[k] == 'degenerate' and r[labkey] in ('NORM-ONLY', 'HARD-BLOCKED')]
        corr = [r for r in ldec if (r[k] in CANNOT) != CAN[r[labkey]]]
        exact = [r for r in ldec if LAB_OF[r[k]] == r[labkey]]
        res[k] = dict(false_degenerate=len(fd), false_degenerate_names=fd,
                      degenerate_on_vacuous=[r['name'] for r in lab if r[k] == 'degenerate' and r[labkey] == 'VACUOUS'],
                      degenerate_returned=sum(r[k] == 'degenerate' for r in v),
                      decided=len(dec), coverage=len(dec) / len(v) if v else None,
                      unknown=len(v) - len(dec),
                      binary_correct=len(corr), binary_accuracy_decided=len(corr) / len(ldec) if ldec else None,
                      binary_accuracy_all=len(corr) / len(lab) if lab else None,
                      label_agree=len(exact), label_agreement_all=len(exact) / len(lab) if lab else None,
                      verdicts=dict(collections.Counter(r[k] for r in v)))
    return res


def secs(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not v:
        return None
    return dict(n=len(v), median=statistics.median(v), mean=sum(v) / len(v), max=max(v), total=sum(v))


# ------------------------------------------------------------------ inputs
probes = J(OUT, 'k1_probes.json')
lse = J(OUT, 'k1_lse.json')
wl = J(OUT, 'k1_whitelist.json')
timm_runs = load_runs('k1_run_timm_*.jsonl')
tv_runs = load_runs('k1_run_tv_*.jsonl')
rein_timm = load_runs('k1_reinit_timm.jsonl')
rein_tv = load_runs('k1_reinit_tv.jsonl')
tvv = J(OUT, 'k1_tv_verify.json') if os.path.exists(os.path.join(OUT, 'k1_tv_verify.json')) else None

gt148 = {}
split = {}
for s in 'DAB':
    for r in J(ROOT, 'outputs', 'heldout', 'h1_%s.json' % s):
        gt148[r['model']] = r.get('gt')
        split[r['model']] = s
C1 = os.path.join(ROOT, 'analyses', 'check_census', 'out')
R = J(C1, 'c5_census.json')['records']
pop = J(C1, 'c5_population.json')
census = [n for n in pop['p1'] if n in R and 'verdict' in R[n]]
a1 = J(ROOT, 'analyses', 'check_audit', 'out', 'a1_audit.json')
a1lab = {r['arch']: (r['measured_unit'] if r.get('measured_unit') and r['measured_unit'] != r.get('measured') else r.get('measured'))
         for r in a1['rows'] if 'measured' in r}
PAPER = {'IN-SCOPE': 'degenerate', 'DEGENERATE': 'class-token', 'NORM-BLOCKED': 'normalisation-blocked',
         'NONLIN-BLOCKED': 'nonlinearity-blocked', 'AGG-BLOCKED': 'aggregator-blocked'}

# ------------------------------------------------------------------ the 148 architectures
rows148 = []
for n, g in gt148.items():
    r = dict(timm_runs.get(n, {'name': n, 'error': 'not run'}))
    r['stored_label'] = g
    r['split'] = split[n]
    rows148.append(r)
ok148 = [r for r in rows148 if 'sound' in r]
s148 = score(ok148, 'stored_label')
s148['n_listed'] = len(rows148)
s148['n_run'] = len(ok148)
s148['errors'] = {r['name']: r['error'] for r in rows148 if 'sound' not in r}
s148['n_measured_stored'] = sum(r['stored_label'] in CAN for r in rows148)
s148['random_weight_label_vs_stored'] = dict(
    n=sum(1 for r in ok148 if r['stored_label'] in CAN and r.get('label')),
    agree=sum(1 for r in ok148 if r['stored_label'] in CAN and r.get('label') == r['stored_label']),
    differ=[(r['name'], r['stored_label'], r.get('label')) for r in ok148
            if r['stored_label'] in CAN and r.get('label') and r.get('label') != r['stored_label']])
s148['by_split'] = {s: score([r for r in ok148 if r['split'] == s], 'stored_label') for s in 'DAB'}
s148['secs_original'] = secs(ok148, 'secs_original')
s148['secs_sound'] = secs(ok148, 'secs_sound')
s148['secs_certificate'] = secs(ok148, 'secs_certificate')

# ------------------------------------------------------------------ census (K1 coverage, K1b measured)
rc = [timm_runs.get(n, {'name': n, 'error': 'not run'}) for n in census]
okc = [r for r in rc if 'sound' in r]
cert_ok = [r for r in okc if r.get('label') in CAN]
sc = score(cert_ok, 'label')
zr = [r for r in okc if r.get('label') == 'ZERO-READOUT']
sc['zero_readout'] = dict(n=len(zr), names=[r['name'] for r in zr],
                          reinit_labels={r['name']: rein_timm.get(r['name'], {}).get('label') for r in zr})
okc_re = with_reinit(okc, rein_timm)
sc['with_reinit'] = score([r for r in okc_re if r.get('label') in CAN], 'label')
sc['n_census'] = len(census)
sc['n_checked'] = len(okc)
sc['n_certified'] = len(cert_ok)
sc['not_run'] = {r['name']: r.get('error', 'certificate missing') for r in rc
                 if r.get('label') not in CAN and r.get('label') != 'ZERO-READOUT'}
sc['sound_coverage_all_checked'] = sum(r['sound'] != 'unknown' for r in okc) / len(okc) if okc else None
sc['sound_unknown_names'] = [r['name'] for r in okc if r['sound'] == 'unknown']
sc['posthoc_unknown_names'] = [r['name'] for r in okc if r.get('posthoc_opguard') == 'unknown']
sc['sound_failed_conditions'] = dict(collections.Counter(
    ','.join(map(str, r['sound_failed'])) for r in okc if r['sound'] == 'unknown'))
sc['measured_distribution'] = dict(collections.Counter(r['label'] for r in cert_ok))
sc['own_vs_unit_label_differ'] = [(r['name'], r['label_own'], r['label_unit']) for r in cert_ok
                                  if r['label_own'] != r['label_unit']]
sc['original_matches_census_record'] = dict(
    n=len(okc), agree=sum(PAPER[R[r['name']]['verdict']] == r['original'] for r in okc),
    differ=[(r['name'], PAPER[R[r['name']]['verdict']], r['original']) for r in okc
            if PAPER[R[r['name']]['verdict']] != r['original']])
sc['a1_audited_traces'] = dict(
    n=sum(1 for r in okc if r['name'] in a1lab),
    label_agree=sum(1 for r in okc if r['name'] in a1lab and r.get('label') == a1lab[r['name']]),
    sound_false_degenerate=[r['name'] for r in okc if r['name'] in a1lab and r['sound'] == 'degenerate'
                            and a1lab[r['name']] in ('NORM-ONLY', 'HARD-BLOCKED')],
    sound_unknown=sum(1 for r in okc if r['name'] in a1lab and r['sound'] == 'unknown'))
sc['head_roundtrip_max'] = max((r.get('head_roundtrip', 0) for r in cert_ok), default=None)
for k in ('secs_original', 'secs_sound', 'secs_certificate', 'secs_total'):
    sc[k] = secs(okc if k != 'secs_certificate' else cert_ok, k)
sc['by_trace'] = None

# ------------------------------------------------------------------ torchvision
rt = list(tv_runs.values())
okt = [r for r in rt if r.get('label') in CAN]
st = score(okt, 'label')
st['zero_readout'] = [r['name'] for r in rt if r.get('label') == 'ZERO-READOUT']
st['with_reinit'] = score([r for r in with_reinit(rt, rein_tv) if r.get('label') in CAN], 'label')
st['n_models'] = len(rt)
st['errors'] = {r['name']: r.get('error') for r in rt if r.get('label') not in CAN and r.get('label') != 'ZERO-READOUT'}
st['measured_distribution'] = dict(collections.Counter(r['label'] for r in okt))
st['wrapper_forward_matches_torchvision'] = tvv
for k in ('secs_original', 'secs_sound', 'secs_certificate'):
    st[k] = secs(okt, k)

# ------------------------------------------------------------------ frozen file hash
fr = os.path.join(ROOT, 'scripts', 'heldout', 'frozen_rule.py')
sha = hashlib.sha256(open(fr, 'rb').read()).hexdigest()
h0 = J(ROOT, 'outputs', 'heldout', 'h0_freeze.json')
sha_lf = hashlib.sha256(open(fr, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()

DEVIATIONS = [
    'Condition 1 is read as "the spatial reduction is the only one": in both branches the operator trace must hold '
    'exactly one operator that removes the spatial axes, and in the module branch it must run inside the aggregator module.',
    'Condition 3 covers non-leaf modules too: one that completes before the aggregator is accepted only when it runs no '
    'operator of its own (plain nn.Sequential, the only container class in the development set); its leaf children are '
    'checked against the per-cell whitelist.',
    'Whitelists were taken from the module classes of the whole 66 development models (not only their heads), filtered '
    'to the protocol categories (k1_whitelist.json).',
    'The protocol-as-written check returned degenerate on constructed head P23 (generalised mean written as tensor '
    'operations, measured hard-blocked): the target of zero false degenerate verdicts is not met on the constructed '
    'heads. A post-hoc variant with two operator-level conditions (only per-cell operators before the reduction, only '
    'affine operators after it) is reported separately; it is not part of the protocol.',
    'False degenerate = degenerate returned where the certificate says not additive (normalisation-only or hard-blocked); '
    'degenerate on a vacuous (additive, zero spatial mass) head is listed separately.',
    'Binary accuracy of the conservative check is given over decided entries and over all entries (unknown counted wrong).',
    'The 148 architectures are scored against their stored labels (pretrained weights, real images); the checks and the '
    'certificate were run here on random weights.',
    'All checks and certificates ran on the CPU (the shared GPU slowed the per-cell loop); costs are CPU wall-clock, '
    'with 4 to 6 threads per process and several processes in parallel.',
    'Measured census label: the label on the model\'s own random-weight features, replaced by the label on a unit-scale '
    'N(0,1) feature map where the two differ (rule of analyses/check_audit/scripts/a1_audit.py).',
    'Entries whose readout is identically zero at initialisation (classifier weights initialised to zero; |N| = 0 on '
    'every cell) carry no structural information at random weights; they are labelled zero-readout and left out of the '
    'measured comparisons. As a supplement they were re-certified after drawing the all-zero weight tensors from '
    'N(0, 0.02^2), and the comparisons are repeated with those labels.',
    'Class-token by inspection: the four PiT-distilled models return only their class and distillation tokens from '
    'forward_features (no spatial cell reaches the head); the certificate on those two tokens reads additive, and they '
    'are labelled vacuous (class-token).',
    'ConViT (3 models): the certificate uses a model converted to float64 before its first forward (ConViT caches '
    'float32 buffers on its first forward); the two CrossViT token sequences of unequal width '
    'were zero-padded to a common width in the cell matrix; for the six mixed-RoPE ViTs, which force float32 inside '
    'the trunk, the trunk of the certificate ran in float32 and the head and all residuals in float64.',
    'torchvision models were wrapped with a forward_features / forward_head split at the last trunk feature map '
    '(k1_tv.py); GoogLeNet and Inception-v3 were built without auxiliary heads, which are unused in evaluation.',
    'HuggingFace part not run: the transformers library is not installed in the environment and nothing was downloaded.',
    'Channel-wise pooling check: Spearman correlations are computed from the exact one-cell change evaluated without '
    'subtracting two logits; the literal per-cell loop agrees to within 1e-5 of the map scale but rounds changes below '
    'float64 resolution to zero (ties), which lowers its score-then-pool value below 1; both are reported.',
]

res = dict(protocol='protocols/K1_sound_check_protocol.md + K1b_measured_census_addendum.md',
           frozen_rule_sha256_raw=sha, frozen_rule_sha256_lf=sha_lf,
           frozen_rule_unchanged=sha_lf == h0['sha256_frozen_file'],
           whitelist=dict(pool=wl['pool_whitelist'], percell=wl['percell_whitelist'], container=wl['container_whitelist']),
           probes=probes['summary'], arch148=s148, census=sc, torchvision=st,
           huggingface='not run: transformers not installed; nothing downloaded',
           lse={h['head'] + '_' + str(h['param']): {k: h[k] for k in h if k != 'rho'} for h in lse['heads']},
           deviations=DEVIATIONS,
           verdicts=dict(probes=probes['rows'],
                         arch148=[{k: r.get(k) for k in ('name', 'split', 'stored_label', 'label', 'original', 'audited',
                                                         'sound', 'sound_failed', 'posthoc_opguard', 'error')}
                                  for r in rows148],
                         census=[{k: r.get(k) for k in ('name', 'label', 'label_own', 'label_unit', 'original', 'audited',
                                                        'sound', 'sound_failed', 'posthoc_opguard', 'secs_original',
                                                        'secs_sound', 'secs_certificate', 'error')} for r in rc],
                         torchvision=[{k: r.get(k) for k in ('name', 'label', 'original', 'audited', 'sound',
                                                             'sound_failed', 'sound_why', 'posthoc_opguard', 'error')}
                                      for r in rt]))
json.dump(res, open(os.path.join(OUT, 'k1_verdicts.json'), 'w'), indent=1, default=float)


# ------------------------------------------------------------------ log + RESULTS.md
def pct(x):
    return 'n/a' if x is None else '%.1f%%' % (100 * x)


def line(name, s, k):
    d = s[k]
    return '| %s | %s | %d | %s | %s | %s |' % (
        name, {'sound': 'conservative', 'posthoc_opguard': 'post-hoc variant'}.get(k, k), d['false_degenerate'], pct(d['coverage']),
        pct(d['binary_accuracy_decided']), pct(d['binary_accuracy_all']))


def f2(d):
    return 'n/a' if not d else '%.3f s median, %.3f s mean' % (d['median'], d['mean'])


L = []
L.append('# K1 and K1b results')
L.append('')
L.append('Verdicts: original (frozen rule), audited (frozen rule plus three pattern rules), conservative (K1, '
         'protocol as written), post-hoc variant (K1 plus two operator-level conditions; not in the protocol). '
         'False degenerate: degenerate where the float64 certificate is not additive. Coverage: share not unknown. '
         'Binary accuracy: "can the two maps differ?" over decided entries / over all entries.')
L.append('')
L.append('## False degenerate verdicts, coverage, binary accuracy')
L.append('')
L.append('| Set | Check | False degenerate | Coverage | Binary acc. (decided) | Binary acc. (all) |')
L.append('|---|---|---|---|---|---|')
for nm, s in (('32 constructed heads', probes['summary']), ('148 architectures (%d labelled)' % s148['n_labelled'], s148),
              ('census, measured (%d)' % sc['n_certified'], sc), ('torchvision 0.21 (%d)' % len(okt), st)):
    for k in CHECKS:
        if k in s:
            L.append(line(nm, s, k))
L.append('')
L.append('- Constructed heads, false degenerate: original %s; audited %s; conservative %s; post-hoc %s.'
         % tuple(', '.join(probes['summary'][k]['false_degenerate_ids']) or 'none' for k in CHECKS))
L.append('- Constructed heads left unknown by the conservative check: %s.' % ', '.join(probes['summary']['sound']['unknown_ids']))
NM = {'original': 'original', 'audited': 'audited', 'sound': 'conservative', 'posthoc_opguard': 'post-hoc'}
L.append('- 148 architectures, false degenerate: %s.' % '; '.join('%s %s' % (NM[k], ', '.join(s148[k]['false_degenerate_names']) or 'none') for k in CHECKS))
L.append('- 148 architectures, conservative unknown: %d; errors: %s.' % (s148['sound']['unknown'], s148['errors'] or 'none'))
L.append('- Census, conservative coverage over the %d checked entries: %s (%d unknown). Failed conditions among unknown: %s.'
         % (sc['n_checked'], pct(sc['sound_coverage_all_checked']), len(sc['sound_unknown_names']), sc['sound_failed_conditions']))
L.append('- Census, false degenerate: %s.' % '; '.join('%s %d' % (NM[k], sc[k]['false_degenerate']) for k in CHECKS))
L.append('- Census, degenerate on vacuous heads: %s.' % '; '.join('%s %d' % (NM[k], len(sc[k]['degenerate_on_vacuous'])) for k in CHECKS))
L.append('- torchvision, false degenerate: %s; conservative unknown: %s.' % (
    '; '.join('%s %d' % (NM[k], st[k]['false_degenerate']) for k in CHECKS),
    ', '.join(r['name'] for r in rt if r.get('sound') == 'unknown')))
L.append('- torchvision with the zero-initialised ViT heads re-drawn (%d labelled): degenerate on a class-token (vacuous) '
         'readout %s; coverage %s; five-class agreement %s.' % (
    st['with_reinit']['n_labelled'],
    '; '.join('%s %d' % (NM[k], len(st['with_reinit'][k]['degenerate_on_vacuous'])) for k in CHECKS),
    '; '.join('%s %s' % (NM[k], pct(st['with_reinit'][k]['coverage'])) for k in CHECKS),
    '; '.join('%s %d' % (NM[k], st['with_reinit'][k]['label_agree']) for k in CHECKS)))
L.append('- Census entries left unknown by the post-hoc variant: %s.' % ', '.join(sc['posthoc_unknown_names']))
if tvv:
    L.append('- torchvision split versus torchvision forward: largest relative output difference %s over %d models.'
             % (tvv['max_rel_diff'], tvv['n']))
L.append('')
L.append('## Measured census (random weights, float64)')
L.append('')
L.append('- Certified with a structural label: %d of %d (%d zero readout); not certified: %s.' % (sc['n_certified'], sc['n_census'], sc['zero_readout']['n'], sc['not_run'] or 'none'))
L.append('- Labels: %s.' % sc['measured_distribution'])
L.append('- Agreement with the measured label (five classes; unknown counted as disagreement): original %d (%s), '
         'audited %d (%s), conservative %d (%s), post-hoc %d (%s).' % tuple(
             x for k in CHECKS for x in (sc[k]['label_agree'], pct(sc[k]['label_agreement_all']))))
L.append('- Zero readout at initialisation: %d census entries (%s); torchvision: %s.' % (
    sc['zero_readout']['n'], ', '.join(sc['zero_readout']['names']), ', '.join(st['zero_readout']) or 'none'))
L.append('- With those entries re-certified after re-drawing the zero weights (labels %s): false degenerate %s; '
         'agreement %s.' % (sc['zero_readout']['reinit_labels'],
                            '; '.join('%s %d' % (NM[k], sc['with_reinit'][k]['false_degenerate']) for k in CHECKS),
                            '; '.join('%s %d/%d' % (NM[k], sc['with_reinit'][k]['label_agree'], sc['with_reinit']['n_labelled'])
                                      for k in CHECKS)))
L.append('- Own-feature and unit-scale labels differ on %d entries: %s.' % (len(sc['own_vs_unit_label_differ']), sc['own_vs_unit_label_differ'][:10]))
L.append('- Original verdict re-run vs stored census record: %d of %d equal; differ: %s.'
         % (sc['original_matches_census_record']['agree'], sc['original_matches_census_record']['n'],
            sc['original_matches_census_record']['differ'][:10]))
L.append('- Entries of the 30 audited traces: %d; measured label equal to the audited label: %d; conservative false '
         'degenerate: %s; unknown: %d.' % (sc['a1_audited_traces']['n'], sc['a1_audited_traces']['label_agree'],
                                            sc['a1_audited_traces']['sound_false_degenerate'] or 'none',
                                            sc['a1_audited_traces']['sound_unknown']))
L.append('- 148 architectures: random-weight label equals stored label on %d of %d; differ: %s.'
         % (s148['random_weight_label_vs_stored']['agree'], s148['random_weight_label_vs_stored']['n'],
            s148['random_weight_label_vs_stored']['differ']))
L.append('')
L.append('## Cost per model (CPU wall-clock)')
L.append('')
for nm, s in (('148 architectures', s148), ('census', sc), ('torchvision', st)):
    L.append('- %s: original check %s; conservative check %s; certificate %s.'
             % (nm, f2(s['secs_original']), f2(s['secs_sound']), f2(s['secs_certificate'])))
L.append('')
L.append('## Channel-wise pooling followed by a linear layer (ResNet-50 layer4, 64 images, 49 cells)')
L.append('')
L.append('| Head | T or p | Spearman(N, S) mean | min | literal loop mean | literal loop min |')
L.append('|---|---|---|---|---|---|')
for h in lse['heads']:
    L.append('| %s | %g | %.4f | %.4f | %.4f | %.4f |' % (h['head'], h['param'], h['rho_mean'], h['rho_min'],
                                                         h['literal_rho_mean'], h['literal_rho_min']))
L.append('')
L.append('ch-LSE: channel-wise log-sum-exp then the fc layer; ch-GeM: channel-wise generalised mean then the fc layer; '
         'stp-LSE: per-cell class score then log-sum-exp (closed form: Spearman = 1).')
L.append('')
L.append('## Deviations')
L.append('')
for d in DEVIATIONS:
    L.append('- ' + d)
L.append('')
L.append('frozen_rule.py SHA-256 (LF-normalised) %s, equal to the recorded hash: %s.' % (sha_lf, res['frozen_rule_unchanged']))
txt = '\n'.join(L) + '\n'
open(os.path.join(HERE, '..', 'RESULTS.md'), 'w', encoding='utf-8').write(txt)
open(os.path.join(OUT, 'k1.log'), 'w', encoding='utf-8').write(txt + '\n' + json.dumps(
    {k: res[k] for k in ('probes', 'arch148', 'census', 'torchvision', 'lse')}, indent=1, default=float))
print(txt)
