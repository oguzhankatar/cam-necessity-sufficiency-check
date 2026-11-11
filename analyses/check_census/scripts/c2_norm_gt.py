"""C1 task 2 -- a NORM-ONLY ground truth that does not use frozen_rule.is_norm.

For every architecture whose shipped head fails the float64 certificate
(gt NORM-ONLY or HARD-BLOCKED in outputs/heldout/h1_*.json), re-measure with the
SAME certificate as h1_run (hlib.certify, read-only import) after deletions
chosen WITHOUT the predictor:

  (i)  INDEPENDENT NORM SET.  Our own forward hooks on every leaf module that
       forward_head executes (not frozen_rule.trace_head).  A module is
       normalisation-like if EITHER its class name contains 'norm'
       (case-insensitive; this also catches BatchNorm) OR it is numerically
       scale-invariant on its real input: ||mod(2x) - mod(x)|| / ||mod(x)|| < 1e-3
       and the same for 0.5x.  Every shape-preserving normalisation-like module is
       replaced by Identity jointly, then the certificate is re-run.
  (ii) DELETE-ONE.  Every shape-preserving leaf module of forward_head is replaced
       by Identity alone (one at a time) and the certificate re-run.

Independent label:  shipped additive -> (not in this set);
                    (i) additive     -> NORM-ONLY*
                    otherwise        -> HARD-BLOCKED*
Compared with the archived gt.  Same probe images as h1_run (IDX), float64, GPU.
"""
import os, sys, json, time, traceback
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'heldout'))
import hlib as H            # read-only import: CellView, make_head, certify, val_images, set_mod
import timm

IDX = [7, 4211, 15123, 3007, 40555, 21001]          # identical to h1_run.py
ONLY = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None

rows = []
for s in 'DAB':
    rows += json.load(open(os.path.join(ROOT, 'outputs', 'heldout', 'h1_%s.json' % s)))
todo = [r for r in rows if r.get('gt') in ('NORM-ONLY', 'HARD-BLOCKED')]
if ONLY:
    todo = [r for r in todo if r['model'] in ONLY]


def head_leaves(m, f):
    """our own trace: every leaf module forward_head executes, with its input."""
    ev, hs = [], []
    for nm, mod in m.named_modules():
        if len(list(mod.children())) == 0:
            def fn(mod, inp, out, nm=nm):
                if len(inp) and torch.is_tensor(inp[0]) and torch.is_tensor(out):
                    ev.append(dict(name=nm, cls=type(mod).__name__, x=inp[0].detach().clone(),
                                   ishape=list(inp[0].shape), oshape=list(out.shape)))
            hs.append(mod.register_forward_hook(fn))
    with torch.no_grad():
        m.forward_head(f)
    for h in hs:
        h.remove()
    # one entry per module name (head_drop can be called twice)
    seen, out = set(), []
    for e in ev:
        if e['name'] not in seen:
            seen.add(e['name']); out.append(e)
    return out


@torch.no_grad()
def scale_invariance(mod, x):
    y = mod(x)
    den = y.norm() + 1e-300
    return max(float((mod(2 * x) - y).norm() / den), float((mod(0.5 * x) - y).norm() / den))


def certify_now(m, x, cv):
    with torch.no_grad():
        f1 = m.forward_features(x)
        z1 = m(x)
    head = H.make_head(m, cv)
    c = H.certify(head, cv.cells(f1), z1.argmax(1).tolist(), cv.npfx)
    return dict(additive=c['additive'], maxNS=c['maxNS'], Rhalf=c['Rhalf'], I2=c['I2'],
                corrNS_min=c['corrNS_min'])


res = []
t00 = time.time()
for r in todo:
    nm = r['model']
    rec = dict(model=nm, split=r['split'], gt_archived=r['gt'], verdict=r['verdict'],
               norms_killed_archived=r['norms_killed'])
    t0 = time.time()
    try:
        m = timm.create_model(nm, pretrained=True).eval().cuda()
        x = H.val_images(m.pretrained_cfg, IDX).cuda()
        m = m.double(); x = x.double()
        with torch.no_grad():
            f = m.forward_features(x)
        cv = H.CellView(m, f)
        mods = dict(m.named_modules())
        leaves = head_leaves(m, f)
        info = []
        for e in leaves:
            mod = mods[e['name']]
            sp = e['ishape'] == e['oshape']
            si = scale_invariance(mod, e['x']) if sp else None
            namehit = 'norm' in e['cls'].lower()
            info.append(dict(name=e['name'], cls=e['cls'], ishape=e['ishape'], oshape=e['oshape'],
                             shape_preserving=sp, scale_inv=si, name_has_norm=namehit,
                             norm_like=bool(sp and (namehit or (si is not None and si < 1e-3)))))
        rec['leaves'] = info
        rec['shipped'] = certify_now(m, x, cv)
        # (i) independent joint norm deletion
        ind = [d['name'] for d in info if d['norm_like']]
        rec['independent_norms'] = ind
        for n in ind:
            H.set_mod(m, n, nn.Identity())
        rec['after_independent_norm_deletion'] = certify_now(m, x, cv) if ind else None
        for n in ind:
            H.set_mod(m, n, mods[n])
        # (ii) delete-one
        d1 = []
        for d in info:
            if not d['shape_preserving'] or d['cls'] in ('Identity',):
                continue
            H.set_mod(m, d['name'], nn.Identity())
            try:
                c = certify_now(m, x, cv)
            except Exception as e:
                c = dict(error=str(e)[:200])
            H.set_mod(m, d['name'], mods[d['name']])
            d1.append(dict(name=d['name'], cls=d['cls'], **c))
        rec['delete_one'] = d1
        a = rec['after_independent_norm_deletion']
        rec['gt_independent'] = ('NORM-ONLY' if (a is not None and a['additive']) else 'HARD-BLOCKED')
        rec['shipped_reproduces_fail'] = not rec['shipped']['additive']
        rec['label_changed'] = rec['gt_independent'] != r['gt']
        rec['single_deletions_reaching_additivity'] = [d['name'] + '(' + d['cls'] + ')' for d in d1
                                                       if d.get('additive')]
        print('%-34s %-12s indep=%-12s %s ind_norms=%s single_ok=%s (%.0fs)' % (
            nm, r['gt'], rec['gt_independent'], 'CHANGED' if rec['label_changed'] else 'same',
            ind, rec['single_deletions_reaching_additivity'], time.time() - t0), flush=True)
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, e)
        rec['tb'] = traceback.format_exc()[-1200:]
        print('%-34s ERROR %s' % (nm, rec['error'][:150]), flush=True)
    res.append(rec)

ok = [q for q in res if 'error' not in q]
summ = dict(n=len(res), n_ok=len(ok), errors=[q['model'] for q in res if 'error' in q],
            n_label_changed=sum(q['label_changed'] for q in ok),
            changed=[q['model'] for q in ok if q['label_changed']],
            n_heldout=sum(q['split'] in 'AB' for q in ok),
            n_heldout_changed=sum(q['label_changed'] for q in ok if q['split'] in 'AB'),
            shipped_fail_reproduced=sum(q['shipped_reproduces_fail'] for q in ok),
            independent_set_equals_archived=sum(
                sorted(q['independent_norms']) == sorted(q['norms_killed_archived']) for q in ok),
            independent_set_differs=[(q['model'], q['independent_norms'], q['norms_killed_archived'])
                                     for q in ok if sorted(q['independent_norms']) != sorted(q['norms_killed_archived'])],
            hard_blocked_with_single_deletion_additive=[(q['model'], q['single_deletions_reaching_additivity'])
                                                        for q in ok if q['gt_archived'] == 'HARD-BLOCKED'
                                                        and q['single_deletions_reaching_additivity']],
            seconds=time.time() - t00)
print(json.dumps(summ, indent=1, default=str))
tag = '' if not ONLY else '_partial'
json.dump(dict(summary=summ, rows=res), open(os.path.join(OUT, 'c2_norm_gt%s.json' % tag), 'w'),
          indent=1, default=float)
