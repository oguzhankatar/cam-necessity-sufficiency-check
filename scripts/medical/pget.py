"""Parallel ranged downloader (resumable): N connections, fixed-size chunks."""
import os
import sys
import time
import random
import threading

import requests


def pget(url, dst, nconn=6, chunk=8 << 20):
    r = requests.head(url, allow_redirects=True, timeout=60)
    size = int(r.headers['Content-Length'])
    part = dst + '.part'
    if not os.path.exists(part) or os.path.getsize(part) != size:
        with open(part, 'wb') as f:
            f.truncate(size)
    done_path = dst + '.done'
    done = set()
    if os.path.exists(done_path):
        done = set(int(x) for x in open(done_path).read().split() if x)
    todo = [i for i in range((size + chunk - 1) // chunk) if i not in done]
    lock = threading.Lock()
    t0 = time.time()
    nb = [0]
    print('%s  %.2f GB, %d chunks, %d left' % (os.path.basename(dst),
                                               size / 2**30,
                                               (size + chunk - 1) // chunk, len(todo)),
          flush=True)

    def worker():
        s = requests.Session()
        fh = open(part, 'r+b')
        while True:
            with lock:
                if not todo:
                    fh.close()
                    return
                i = todo.pop()
            a, b = i * chunk, min(size, (i + 1) * chunk) - 1
            for k in range(10):
                try:
                    resp = s.get(url, headers={'Range': 'bytes=%d-%d' % (a, b)},
                                 timeout=180)
                    resp.raise_for_status()
                    data = resp.content
                    assert len(data) == b - a + 1
                    break
                except Exception:
                    if k == 9:
                        raise
                    time.sleep(min(45., 1.5 * 2 ** k) * (.5 + random.random()))
            fh.seek(a)
            fh.write(data)
            fh.flush()
            with lock:
                done.add(i)
                nb[0] += len(data)
                open(done_path, 'a').write('%d\n' % i)
                el = time.time() - t0
                if len(done) % 20 == 0:
                    print('  %d/%d  %.1f MB/s' % (len(done), (size + chunk - 1) // chunk,
                                                  nb[0] / 2**20 / max(el, 1e-9)), flush=True)

    ts = [threading.Thread(target=worker) for _ in range(nconn)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    os.replace(part, dst)
    if os.path.exists(done_path):
        os.remove(done_path)
    print('done %s %.0fs' % (dst, time.time() - t0), flush=True)


if __name__ == '__main__':
    pget(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
