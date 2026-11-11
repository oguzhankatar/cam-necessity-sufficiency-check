"""H1 -- BLIND held-out validation of the FROZEN stage-5 decision procedure.

usage:  h1_run.py <split-letter>          e.g.  h1_run.py A

For every architecture, in this order:
  1. the frozen rule's STRUCTURAL PREDICTION is computed and printed;
  2. only then is the float64 ground truth measured;
  3. agreement is recorded.

Ground truth = the literal per-cell loop of scripts/crosscheck/rtlib.py in float64
on pretrained weights and real ImageNet val images (3 probe pairs per model):
ADDITIVE iff maxNS, Rhalf and I2 are all < 1e-9 on every pair.  `corrNS` is
recorded and never used as a criterion.  Stage 5's own weaker seg_dev criterion
is recorded alongside, so the run also answers "would stage 5's test have
agreed?".
"""
import os, sys, time, json, traceback
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlib as H
import frozen_rule as FR
import rule_v2 as R2
import timm

SPLIT = (sys.argv[1] if len(sys.argv) > 1 else 'A').upper()
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = open(os.path.join(HERE, 'names_%s.txt' % SPLIT)).read().split()
IDX = [7, 4211, 15123, 3007, 40555, 21001]        # 6 val images = 3 probe pairs
ONLY = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else None
if ONLY:
    NAMES = [n for n in NAMES if n in ONLY]

rows = []
t00 = time.time()
for nm in NAMES:
    rec = dict(model=nm, split=SPLIT)
    t0 = time.time()
    try:
        m = timm.create_model(nm, pretrained=True).eval().cuda()
        cfg = m.pretrained_cfg
        x = H.val_images(cfg, IDX).cuda()
        try:
            m = m.double(); x = x.double()
        except Exception as e:
            rec['double_failed'] = str(e)[:200]
        rec['dtype'] = str(next(m.parameters()).dtype)
        with torch.no_grad():
            f = m.forward_features(x)

        # ---------------- 1. PREDICTION (frozen rule, no numbers seen yet)
        st = R2.structural_v2(m, f)          # calls frozen_rule.structural first
        rec.update(agg=st['agg'], agg_name=st['agg_name'], verdict=st['verdict'],
                   blocker=st['blocker'], n_norm=st['n_norm'],
                   n_nonlin=st['n_nonlin'], post=[list(p) for p in st['post']],
                   v1=st['v1'], v2a=st['v2a'], v2b=st['v2b'],
                   v2b_agg=st['v2b_agg'], module_pool=st['module_pool'],
                   aten_classes=st['aten_classes'],
                   aten_evidence=st['aten_evidence'])
        print('%-26s PRED %-15s blocker=%-30s' % (nm, st['verdict'],
                                                  str(st['blocker'])[:30]),
              end='', flush=True)

        # ---------------- 2. GROUND TRUTH
        cv = H.CellView(m, f)
        rec.update(fmt=cv.fmt, T=cv.T, C=cv.C, npfx=cv.npfx,
                   feat_shape=list(cv.shape), layout_fallback=cv.fallback)
        head = H.make_head(m, cv)
        cells = cv.cells(f)
        with torch.no_grad():
            z_ref = m(x)
            z_head = head(cells)
        rec['head_reproduces_model'] = float(
            (z_ref - z_head).abs().max() / (z_ref.abs().max() + 1e-300))
        # permutation diagnostic: is the readout invariant to a shuffle of the
        # chosen cell axis?  0 => mean-like aggregator on the right axis.
        g = torch.Generator(device='cpu').manual_seed(0)
        perm = torch.randperm(cv.T, generator=g).cuda()
        with torch.no_grad():
            z_perm = head(cells[:, perm])
        rec['perm_dev'] = float((z_perm - z_head).abs().max()
                                / (z_head.abs().max() + 1e-300))

        cls_idx = z_ref.argmax(1).tolist()
        ship = H.certify(head, cells, cls_idx, cv.npfx)
        d0, sc0 = H.seg_dev(lambda v: m.forward_head(v), f[0:1], f[1:2])

        killed, mods = H.kill_norms(m, [p[0] for p in st['post']])
        nonorm = None
        d1 = d0
        if killed:
            with torch.no_grad():
                f1 = m.forward_features(x)
                z1 = m(x)
            head1 = H.make_head(m, cv)
            nonorm = H.certify(head1, cv.cells(f1), z1.argmax(1).tolist(), cv.npfx)
            d1, _ = H.seg_dev(lambda v: m.forward_head(v), f1[0:1], f1[1:2])
        H.restore(m, killed, mods)
        rec.update(norms_killed=killed, shipped=ship, nonorm=nonorm,
                   seg_d0=d0, seg_d1=d1, seg_scale=sc0)

        # ---------------- 3. AGREEMENT
        rec['gt'] = H.gt_label(ship, nonorm)
        rec['agree'] = bool(H.GT_OF[st['verdict']] == rec['gt'])
        rec['agree_v2a'] = bool(H.GT_OF[st['v2a']] == rec['gt'])
        rec['agree_v2b'] = bool(H.GT_OF[st['v2b']] == rec['gt'])
        r5gt = ('AFFINE-SHIPPED' if d0 < H.TOL_SEG else
                'NORM-ONLY' if d1 < H.TOL_SEG else 'HARD-BLOCKED')
        rec['gt_seg'] = r5gt
        rec['agree_seg'] = bool(H.R5_OF[st['verdict']] == r5gt)
        print(' | GT %-13s maxNS=%.1e Rh=%.1e I2=%.1e spm=%.2e d0=%.1e d1=%.1e %s (%.0fs)'
              % (rec['gt'], ship['maxNS'], ship['Rhalf'], ship['I2'],
                 ship['spatial_mass'], d0, d1,
                 'OK' if rec['agree'] else '*** DISAGREE ***', time.time() - t0),
              flush=True)
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        rec['error'] = '%s: %s' % (type(e).__name__, e)
        rec['tb'] = traceback.format_exc()[-1500:]
        print(' SKIP %s' % rec['error'][:110], flush=True)
    rows.append(rec)

ok = [r for r in rows if 'agree' in r]
n_ag = sum(r['agree'] for r in ok)
print('\nSPLIT %s  v1 (FROZEN stage-5)  STRICT fp64 literal loop: %d/%d'
      % (SPLIT, n_ag, len(ok)))
print('SPLIT %s  v2a (minimal repair)  STRICT                 : %d/%d'
      % (SPLIT, sum(r['agree_v2a'] for r in ok), len(ok)))
print('SPLIT %s  v2b (principled repair) STRICT               : %d/%d'
      % (SPLIT, sum(r['agree_v2b'] for r in ok), len(ok)))
print('SPLIT %s  v1 under stage-5\'s own seg_dev criterion     : %d/%d'
      % (SPLIT, sum(r['agree_seg'] for r in ok), len(ok)))
from collections import Counter
for k, v in Counter(r['verdict'] for r in ok).most_common():
    print('  pred %-15s %3d' % (k, v))
for k, v in Counter(r['gt'] for r in ok).most_common():
    print('  gt   %-15s %3d' % (k, v))
print('\nDISAGREEMENTS (any of v1/v2a/v2b):')
for r in ok:
    if not (r['agree'] and r['agree_v2a'] and r['agree_v2b']):
        print('  %-26s v1=%-14s v2a=%-14s v2b=%-14s gt=%-12s maxNS=%.2e Rh=%.2e '
              'spm=%.2e nact=%d/%d aten=%s'
              % (r['model'], r['v1'], r['v2a'], r['v2b'], r['gt'],
                 r['shipped']['maxNS'], r['shipped']['Rhalf'],
                 r['shipped']['spatial_mass'], r['shipped']['n_active'],
                 r['shipped']['T'], r['aten_classes']))
print('errors: %s' % [r['model'] for r in rows if 'error' in r])
print('total %.0f s' % (time.time() - t00))
H.save('h1_%s.json' % SPLIT, rows)
