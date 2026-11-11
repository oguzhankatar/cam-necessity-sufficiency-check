"""Fetch the needed NIH ChestX-ray14 PNGs one file at a time from the flat
HuggingFace mirror (iraqigold/nih-chest-xray-14-flat), which keeps the original
file names -- the resized mirrors drop them, so they cannot be matched to
BBox_List_2017.csv.

Priority order: the 880 box-annotated images first, then the held-out test
sample, then the training sample, so the pipeline can start before the tail
of the download finishes.
"""
import os
import csv
import json
import time
import random
import threading
import urllib.request

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
D = os.path.join(ROOT, 'data', 'nih')
OUT = os.path.join(D, 'images')
REPO = 'iraqigold/nih-chest-xray-14-flat'
API = 'https://huggingface.co/api/datasets/%s/tree/main/' % REPO
RESOLVE = 'https://huggingface.co/datasets/%s/resolve/main/' % REPO
os.makedirs(OUT, exist_ok=True)
MAP = os.path.join(D, 'flatmap.json')


def _get(u, timeout=60):
    for a in range(10):
        try:
            r = urllib.request.urlopen(urllib.request.Request(u), timeout=timeout)
            return r.read(), r.headers
        except Exception:
            if a == 9:
                raise
            time.sleep(min(45., 1.5 * 2 ** a) * (0.5 + random.random()))


def build_map():
    if os.path.exists(MAP):
        return json.load(open(MAP))
    m = {}
    for k in range(1, 13):
        u = API + 'images_%03d/images?limit=1000' % k
        n = 0
        while u:
            body, hdr = _get(u)
            d = json.loads(body)
            for e in d:
                m[os.path.basename(e['path'])] = e['path']
            n += len(d)
            link = hdr.get('Link', '')
            u = None
            if 'rel="next"' in link:
                u = link.split('<')[1].split('>')[0]
        print('  images_%03d: %d' % (k, n), flush=True)
    json.dump(m, open(MAP, 'w'))
    return m


def priority():
    bbox = set()
    with open(os.path.join(D, 'BBox_List_2017.csv')) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row and row[0].strip():
                bbox.add(row[0].strip())
    te = [l.strip() for l in open(os.path.join(D, 'split_test.txt')) if l.strip()]
    tr = [l.strip() for l in open(os.path.join(D, 'split_train.txt')) if l.strip()]
    order, seen = [], set()
    for grp in (sorted(bbox), te, tr):
        for n in grp:
            if n not in seen:
                seen.add(n)
                order.append(n)
    return order


def main():
    m = build_map()
    want = priority()
    have = set(os.listdir(OUT))
    todo = [w for w in want if w not in have and w in m]
    todo.reverse()                     # workers pop() from the end
    print('map %d, want %d, todo %d' % (len(m), len(want), len(todo)), flush=True)
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
                    r = s.get(RESOLVE + m[n], timeout=120)
                    r.raise_for_status()
                    body = r.content
                    break
                except Exception:
                    if a == 9:
                        raise
                    time.sleep(min(45., 1.5 * 2 ** a) * (0.5 + random.random()))
            with open(os.path.join(OUT, n), 'wb') as f:
                f.write(body)
            with lock:
                done[0] += 1
                if done[0] % 200 == 0:
                    print('  %d  %.2f/s  eta %.0f min' %
                          (done[0], done[0] / (time.time() - t0),
                           len(todo) / max(1e-9, done[0] / (time.time() - t0)) / 60),
                          flush=True)

    ts = [threading.Thread(target=worker) for _ in range(int(os.environ.get('NTHREAD', 8)))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print('done %d in %.0fs' % (done[0], time.time() - t0))


if __name__ == '__main__':
    main()
