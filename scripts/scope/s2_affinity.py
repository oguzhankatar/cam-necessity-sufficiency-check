"""S2 -- the affinity survey redone with PRETRAINED weights, REAL images, fp64.

rt4_scope.py ran the segment test at RANDOM INIT.  Affinity is an architectural
property *of the composition*, but a piecewise-linear activation evaluated at
random init can sit entirely inside one linear piece, so random init produces
FALSE NEGATIVES (a model that looks affine only because Hardswish never left its
x>3 branch).  This re-runs the identical criterion on the shipped checkpoints
with two real ImageNet validation images, in float64.

dev = max_lam |z(lam) - lerp| / |z(1) - z(0)|   over  z(lam) = forward_head(f_A + lam (f_B - f_A))
"""
import os, sys, json, time, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib
import timm

NAMES = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'names.txt')).read().split()
TOL = 1e-11
rows = []
for nm in NAMES:
    t0 = time.time()
    rec = dict(model=nm)
    try:
        m = timm.create_model(nm, pretrained=True).eval().cuda()
        cfg = m.pretrained_cfg
        x = slib.val_images(cfg, [7, 4211]).cuda()
        rec['family'] = type(m).__name__
        # --- fp32 reference (also catches models that will not cast)
        with torch.no_grad():
            f32 = m.forward_features(x)
        d32, sc32 = slib.seg_dev(lambda v: m.forward_head(v), f32[0:1], f32[1:2])
        rec['dev_fp32'] = d32
        # --- fp64
        try:
            m = m.double()
            with torch.no_grad():
                f64 = m.forward_features(x.double())
            d64, sc64 = slib.seg_dev(lambda v: m.forward_head(v), f64[0:1], f64[1:2])
            rec['dev_fp64'] = d64; rec['scale_fp64'] = sc64
        except Exception as e:
            rec['dev_fp64'] = None; rec['fp64_err'] = f'{type(e).__name__}: {e}'[:120]
        d = rec['dev_fp64'] if rec['dev_fp64'] is not None else rec['dev_fp32']
        rec['affine_shipped'] = bool(d < TOL)
        f64s = '   n/a   ' if rec['dev_fp64'] is None else f"{rec['dev_fp64']:.3e}"
        print(f"{nm:38s} fp32={rec['dev_fp32']:.3e}  fp64={f64s}"
              f"  affine={rec['affine_shipped']}  ({time.time()-t0:.0f}s)", flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'[:160]
        print(f'{nm:38s} SKIP {rec["error"]}', flush=True)
    rows.append(rec)
slib.save('s2_affinity.json', rows)
