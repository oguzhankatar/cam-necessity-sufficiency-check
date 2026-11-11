"""Run the 12 K2 training runs sequentially (seed-major order), one log per run.
A run whose checkpoint already exists is skipped.

    python k2_driver.py
"""
import os, sys, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2lib as L

PY = sys.executable
order = [(h, s) for s in L.SEEDS for h in L.HEADS]
for h, s in order:
    ck = L.ckpt_path(h, s)
    if os.path.exists(ck):
        print('skip %s (exists)' % L.tag(h, s), flush=True)
        continue
    lg = os.path.join(L.LOGS, 'train_%s.log' % L.tag(h, s))
    t0 = time.time()
    print('%s start %s' % (time.strftime('%H:%M:%S'), L.tag(h, s)), flush=True)
    with open(lg, 'w') as f:
        r = subprocess.run([PY, '-u', os.path.join(L.HERE, 'k2_train.py'), h, str(s), '--workers', '6'],
                           stdout=f, stderr=subprocess.STDOUT)
    print('%s done %s rc=%d  %.1f min' % (time.strftime('%H:%M:%S'), L.tag(h, s), r.returncode,
                                          (time.time() - t0) / 60), flush=True)
print('all training runs finished', flush=True)
