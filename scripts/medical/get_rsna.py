"""Fetch a subset of the RSNA Pneumonia Detection Challenge training DICOMs
(radiologist-drawn boxes) from the open HuggingFace mirror."""
import os
import csv
import time
import random
import threading

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
D = os.path.join(ROOT, 'data', 'rsna')
OUT = os.path.join(D, 'dcm')
RES = 'https://huggingface.co/datasets/Baldezo313/rsna-pneumonia-dataset/resolve/main/'
API = 'https://huggingface.co/api/datasets/Baldezo313/rsna-pneumonia-dataset/tree/main/'
MAP = os.path.join(D, 'flatmap.json')


def build_map():
    import json as _j
    import urllib.request as _u
    if os.path.exists(MAP):
        return _j.load(open(MAP))
    m = {}
    for k in (0, 1, 2):
        u = API + 'stage_2_train_images_%d?limit=1000' % k
        while u:
            for a in range(10):
                try:
                    r = _u.urlopen(_u.Request(u), timeout=60)
                    body, hdr = r.read(), r.headers
                    break
                except Exception:
                    if a == 9:
                        raise
                    time.sleep(min(45., 1.5 * 2 ** a) * (.5 + random.random()))
            for e in _j.loads(body):
                m[os.path.basename(e['path'])[:-4]] = e['path']
            link = hdr.get('Link', '')
            u = link.split('<')[1].split('>')[0] if 'rel="next"' in link else None
        print('  zone %d: %d' % (k, len(m)), flush=True)
    _j.dump(m, open(MAP, 'w'))
    return m
NPOS = int(os.environ.get('NPOS', 2500))
NNEG = int(os.environ.get('NNEG', 2500))
os.makedirs(OUT, exist_ok=True)


def main():
    fm = build_map()
    rows = list(csv.DictReader(open(os.path.join(D, 'stage_2_train_labels.csv'))))
    pos = sorted(set(r['patientId'] for r in rows if r['Target'] == '1'))
    neg = sorted(set(r['patientId'] for r in rows if r['Target'] == '0') - set(pos))
    rs = random.Random(0)
    want = rs.sample(pos, min(NPOS, len(pos))) + rs.sample(neg, min(NNEG, len(neg)))
    open(os.path.join(D, 'want.txt'), 'w').write('\n'.join(want))
    have = set(n[:-4] for n in os.listdir(OUT))
    todo = [w for w in want if w not in have and w in fm]
    print('want %d, todo %d' % (len(want), len(todo)), flush=True)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def worker():
        s = requests.Session()
        while True:
            with lock:
                if not todo:
                    return
                n = todo.pop()
            for a in range(10):
                try:
                    r = s.get(RES + fm[n], timeout=120)
                    r.raise_for_status()
                    body = r.content
                    break
                except Exception:
                    if a == 9:
                        raise
                    time.sleep(min(45., 1.5 * 2 ** a) * (.5 + random.random()))
            with open(os.path.join(OUT, n + '.dcm'), 'wb') as f:
                f.write(body)
            with lock:
                done[0] += 1
                if done[0] % 250 == 0:
                    print('  %d  %.2f/s' % (done[0], done[0] / (time.time() - t0)), flush=True)

    ts = [threading.Thread(target=worker) for _ in range(int(os.environ.get('NTHREAD', 8)))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print('done %d in %.0fs' % (done[0], time.time() - t0))


if __name__ == '__main__':
    main()
