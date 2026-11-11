"""Evaluate every K2 checkpoint as soon as its training run has finished (sequential, one log per
model). Polls until all 12 models are evaluated.

    python k2_eval_driver.py
"""
import os, sys, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2lib as L

PY = sys.executable
order = [(h, s) for s in L.SEEDS for h in L.HEADS]
while True:
    left = 0
    for h, s in order:
        tg = L.tag(h, s)
        if os.path.exists(os.path.join(L.OUT, 'eval_%s.json' % tg)):
            continue
        left += 1
        if not os.path.exists(L.ckpt_path(h, s)[:-3] + '_train.json'):
            continue
        t0 = time.time()
        print('%s eval %s' % (time.strftime('%H:%M:%S'), tg), flush=True)
        with open(os.path.join(L.LOGS, 'eval_%s.log' % tg), 'w') as f:
            r = subprocess.run([PY, '-u', os.path.join(L.HERE, 'k2_eval.py'), h, str(s)],
                               stdout=f, stderr=subprocess.STDOUT)
        print('%s done %s rc=%d %.1f min' % (time.strftime('%H:%M:%S'), tg, r.returncode,
                                             (time.time() - t0) / 60), flush=True)
        if r.returncode != 0:
            sys.exit('eval failed: %s' % tg)
        left -= 1
    if left == 0:
        break
    time.sleep(120)
print('all evaluations finished', flush=True)
