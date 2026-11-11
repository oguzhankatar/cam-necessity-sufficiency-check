import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rzip

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
D = os.path.join(ROOT, 'data', 'nih')
OUT = os.path.join(D, 'images')
BASE = 'https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/'

want = set(l.strip() for l in open(os.path.join(D, 'want.txt')) if l.strip())
have = set(os.listdir(OUT)) if os.path.isdir(OUT) else set()
want -= have
print('need %d (already have %d)' % (len(want), len(have)), flush=True)

for k in range(1, 13):
    if not want:
        break
    url = BASE + 'images_%03d.zip' % k
    t0 = time.time()
    print('[%s]' % url, flush=True)
    got = rzip.fetch(url, want, OUT, nthread=int(os.environ.get('NTHREAD', 64)))
    names = set(os.path.basename(g) for g in got)
    want -= names
    print('  +%d in %.0fs, %d left' % (len(names), time.time() - t0, len(want)), flush=True)

print('done, remaining %d' % len(want))
