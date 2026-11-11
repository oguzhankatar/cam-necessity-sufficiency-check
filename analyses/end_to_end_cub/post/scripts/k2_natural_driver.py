"""POST HOC: run k2_natural.py for the 12 K2 models, NPAR at a time (one log per model).

    python k2_natural_driver.py [NPAR]
"""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(os.path.dirname(HERE), 'out', 'logs')
os.makedirs(LOGS, exist_ok=True)
NPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 6
jobs = [(h, s) for s in (0, 1, 2) for h in ('H0', 'H1', 'H2', 'H3')
        if not os.path.exists(os.path.join(os.path.dirname(LOGS), 'natural_%s_s%d.json' % (h, s)))]
env = dict(os.environ, PYTHONIOENCODING='utf-8')
run = []
t0 = time.time()
while jobs or run:
    while jobs and len(run) < NPAR:
        h, s = jobs.pop(0)
        f = open(os.path.join(LOGS, 'natural_%s_s%d.log' % (h, s)), 'w')
        p = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'k2_natural.py'), h, str(s)],
                             stdout=f, stderr=subprocess.STDOUT, env=env)
        run.append((h, s, p, f))
        print('%7.0fs start %s_s%d' % (time.time() - t0, h, s), flush=True)
    for it in list(run):
        h, s, p, f = it
        if p.poll() is not None:
            f.close(); run.remove(it)
            print('%7.0fs done %s_s%d rc=%d' % (time.time() - t0, h, s, p.returncode), flush=True)
            if p.returncode != 0:
                sys.exit('failed %s_s%d' % (h, s))
    time.sleep(5)
print('all done %.1f min' % ((time.time() - t0) / 60), flush=True)
