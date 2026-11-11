"""C6 tasks 1-2.  Audit of every census head trace never validated against
measurement (analyses/check_census/out/c5_novel_traces.json, P1 and P2), with a
direct float64 additivity measurement on EVERY architecture carrying one.

Weights: random (timm.create_model(pretrained=False)), float64, eval mode; input:
six N(0,1) images at the default tag's input size (three probe pairs).  Why random
weights suffice here: the question audited is structural -- is the readout from the
feature map to the logit additive over spatial cells, and if not, is the obstruction
removable by deleting the post-aggregator normalisations.  Additivity of an affine
head holds for every weight value; a class-token readout puts zero necessity on every
spatial cell for every weight value; and a nonlinearity or normalisation fed by a
generic random input of unit scale is active on the reachable set.  The converse
risk is a nonlinearity left in its linear regime, which makes a blocked head look
additive.  It occurs when a random-init body returns near-zero features
(mobilenetv4_hybrid_large: feature std ~1e-17), so every architecture is measured
twice: on its own forward_features output, and on a unit-scale N(0,1) feature map of
the same shape ('unit' mode).  The audited verdict uses the unit mode where the two
differ.  No pretrained weights were downloaded.

usage:  python a1_audit.py [arch,arch,...]
"""
import os, sys, json, time, re, traceback
import torch, timm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_audit_lib as L
import rule_v2 as R2                      # read-only; calls frozen_rule.structural
from timm.models import get_pretrained_cfg

C1 = os.path.join(L.ROOT, 'analyses', 'check_census', 'out')
OUT = os.path.join(HERE, '..', 'out'); os.makedirs(OUT, exist_ok=True)
nov = json.load(open(os.path.join(C1, 'c5_novel_traces.json')))
pop = json.load(open(os.path.join(C1, 'c5_population.json')))

traces = []
seen = set()
for p in ('p2', 'p1'):
    for t in nov[p]['traces']:
        k = (t['agg'], json.dumps(t['tail']))
        if k in seen:
            continue
        seen.add(k)
        traces.append(t)
todo = [(ti, a[0], a[1]) for ti, t in enumerate(traces) for a in t['archs']]
if len(sys.argv) > 1:
    only = set(sys.argv[1].split(','))
    todo = [x for x in todo if x[1] in only]

rows = []
for ti, arch, v_census in todo:
    rec = dict(trace=ti, arch=arch, census_verdict=v_census,
               in_p1=arch in pop['p1'], in_p2=arch in pop['p2'],
               trace_agg=traces[ti]['agg'], trace_tail=traces[ti]['tail'])
    t0 = time.time()
    try:
        torch.manual_seed(0)
        cfg = get_pretrained_cfg(arch)
        m = timm.create_model(arch, pretrained=False).eval().double().cuda()
        x = torch.randn(6, *cfg.input_size, dtype=torch.float64, device='cuda')
        with torch.no_grad():
            f = m.forward_features(x)
            z = m(x)
        st = R2.structural_v2(m, f)
        rec.update(frozen=st['verdict'], frozen_paper=L.PAPER[st['verdict']], agg=st['agg'],
                   agg_name=st['agg_name'], blocker=st['blocker'], v2b=st['v2b'],
                   aten_classes=st['aten_classes'])
        mods = dict(m.named_modules())
        norms = [p[0] for p in st['post'] if L.is_norm_name(mods[p[0]])]
        cv = L.cells_for_timm(m, f)
        rec.update(fmt=cv.fmt, T=cv.T, C=cv.C, prefix=cv.prefix[:8], norms_deleted=norms)
        with torch.no_grad():
            zh = m.forward_head(cv.feat(cv.cells(f)))
        rec['head_reproduces_model'] = float((zh - z).abs().max() / z.abs().max())
        ship, nonorm, lab = L.measure(m, f, cv, norms)
        rec['feat_std'] = float(torch.cat([t.flatten() for t in (f if isinstance(f, (list, tuple)) else [f])]).std())
        # second mode: unit-scale N(0,1) feature map of the same shape.  Random-init
        # bodies can return near-zero features (mobilenetv4_hybrid_large: std ~1e-17),
        # on which every tail nonlinearity is inside its linear regime.
        g = torch.Generator(device='cuda').manual_seed(1)
        fu = ([torch.randn(t.shape, generator=g, dtype=t.dtype, device=t.device) for t in f]
              if isinstance(f, (list, tuple)) else
              torch.randn(f.shape, generator=g, dtype=f.dtype, device=f.device))
        ship_u, nonorm_u, lab_u = L.measure(m, fu, cv, norms)
        rec.update(shipped=ship, nonorm=nonorm, measured=lab,
                   shipped_unit=ship_u, nonorm_unit=nonorm_u, measured_unit=lab_u,
                   agree=bool(L.GT_OF[st['verdict']] == lab),
                   agree_unit=bool(L.GT_OF[st['verdict']] == lab_u))
        print('%-36s frozen %-14s measured %-12s eps1 %.2e eps2 %.2e eps3 %.2e spm %.1e nact %d/%d %s'
              ' | unit-scale: %-12s eps1 %.2e eps2 %.2e %s | fstd %.1e (%.0fs)'
              % (arch, st['verdict'], lab, ship['eps1'], ship['eps2'], ship['eps3'],
                 ship['spatial_mass'], ship['n_active'], ship['T'],
                 'OK' if rec['agree'] else '*** DISAGREE ***', lab_u, ship_u['eps1'], ship_u['eps2'],
                 'OK' if rec['agree_unit'] else '*** DISAGREE ***', rec['feat_std'], time.time() - t0), flush=True)
        del m, f, x, z
        torch.cuda.empty_cache()
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, str(e)[:300])
        rec['tb'] = traceback.format_exc()[-1200:]
        print('%-36s ERROR %s' % (arch, rec['error'][:150]), flush=True)
    rows.append(rec)

tag = '' if len(sys.argv) == 1 else '_partial'
json.dump(dict(traces=traces, rows=rows), open(os.path.join(OUT, 'a1_audit%s.json' % tag), 'w'),
          indent=1, default=float)
