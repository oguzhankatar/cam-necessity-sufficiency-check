"""Extract selected members from a remote ZIP over HTTP range requests.

Used to pull only the NIH ChestX-ray14 images we need out of the 12 x ~2 GB
zips of the HuggingFace mirror, instead of downloading ~42 GB.
"""
import io
import os
import sys
import zipfile
import threading
import time
import random
import urllib.request


class HttpFile(io.RawIOBase):
    def __init__(self, url, size=None):
        self.url = url
        self.pos = 0
        self._size = size if size is not None else self._head()

    def _head(self):
        req = urllib.request.Request(self.url, method='HEAD')
        for attempt in range(12):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return int(r.headers['Content-Length'])
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(60.0, 2.0 * (2 ** attempt)) * (0.5 + random.random()))

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self._size + off
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self.pos
        if n == 0:
            return b''
        end = min(self.pos + n, self._size) - 1
        if end < self.pos:
            return b''
        req = urllib.request.Request(self.url, headers={'Range': 'bytes=%d-%d' % (self.pos, end)})
        for attempt in range(12):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(60.0, 1.5 * (2 ** attempt)) * (0.5 + random.random()))
        self.pos += len(data)
        return data


def fetch(url, names, outdir, nthread=12, verbose=True):
    """names: set of basenames wanted.  Returns list of written paths."""
    os.makedirs(outdir, exist_ok=True)
    hf = HttpFile(url)
    zf = zipfile.ZipFile(hf)
    infos = [i for i in zf.infolist() if os.path.basename(i.filename) in names]
    if verbose:
        print('  %d / %d members matched' % (len(infos), len(zf.infolist())), flush=True)
    written = []
    lock = threading.Lock()
    todo = list(infos)

    def worker():
        h = HttpFile(url, size=hf._size)
        z = zipfile.ZipFile(h)
        while True:
            with lock:
                if not todo:
                    return
                info = todo.pop()
            dst = os.path.join(outdir, os.path.basename(info.filename))
            if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                with lock:
                    written.append(dst)
                continue
            data = z.read(info)
            with open(dst, 'wb') as f:
                f.write(data)
            with lock:
                written.append(dst)
                if verbose and len(written) % 200 == 0:
                    print('    %d' % len(written), flush=True)

    ts = [threading.Thread(target=worker) for _ in range(nthread)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return written


if __name__ == '__main__':
    url, listfile, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    names = set(l.strip() for l in open(listfile) if l.strip())
    got = fetch(url, names, outdir)
    print('wrote %d' % len(got))
