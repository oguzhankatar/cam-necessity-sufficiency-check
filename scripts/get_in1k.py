"""Fetch real ImageNet-1k images for the P1 scale replication.

Source: HuggingFace `evanarlian/imagenet_1k_resized_256` -- an UNGATED mirror of
ILSVRC-2012 with the shorter side resized to 256 px.  Labels are the standard
0..999 ImageNet class indices in the canonical (synset-sorted) order, which is
exactly the order torchvision's pretrained ResNet50 `fc` uses, so the shipped
1000-way head is directly comparable.

Two jobs:
  val   -> the 50 000-image ILSVRC-2012 validation set (2 shards, ~2.5 GB)
  train -> the first NSHARD train shards.  The train split is ordered by class,
           so K shards give roughly K*1000/52 contiguous classes with the full
           ~1300 images each (used for the 200-class control run).

Images are written to an ImageFolder tree `data/in1k/<split>/<label:04d>/`.
"""
import os, sys, io, json, urllib.request, socket, time
import tempfile

socket.setdefaulttimeout(120)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BASE = ('https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256'
        '/resolve/main/data/')
VAL = ['val-00000-of-00002-b5248be478d25e41.parquet',
       'val-00001-of-00002-85f3d9c8fa1edb63.parquet']
DL = os.environ.get(
    'IN1K_PARQUET', os.path.join(tempfile.gettempdir(), 'nscam_cache', 'in1k_parq'))


def fetch(fn):
    os.makedirs(DL, exist_ok=True)
    p = os.path.join(DL, fn)
    if os.path.exists(p) and os.path.getsize(p) > 1 << 20:
        return p
    t = time.time()
    with urllib.request.urlopen(BASE + fn) as r, open(p + '.part', 'wb') as f:
        n = 0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b); n += len(b)
            if n % (1 << 28) == 0:
                print(f'   {fn} {n/1e9:.2f} GB {n/1e6/(time.time()-t):.1f} MB/s',
                      flush=True)
    os.replace(p + '.part', p)
    print(f'  got {fn} {os.path.getsize(p)/1e9:.2f} GB in {time.time()-t:.0f}s',
          flush=True)
    return p


def explode(pq, split, per_class=None, counts=None):
    import pyarrow.parquet as pqt
    from PIL import Image
    f = pqt.ParquetFile(pq)
    out = os.path.join(ROOT, 'data', 'in1k', split)
    counts = {} if counts is None else counts
    n = 0
    for batch in f.iter_batches(batch_size=256):
        d = batch.to_pydict()
        for img, lab in zip(d['image'], d['label']):
            lab = int(lab)
            if per_class is not None and counts.get(lab, 0) >= per_class:
                continue
            dd = os.path.join(out, f'{lab:04d}')
            os.makedirs(dd, exist_ok=True)
            i = counts.get(lab, 0)
            fp = os.path.join(dd, f'{i:05d}.jpg')
            if not os.path.exists(fp):
                im = Image.open(io.BytesIO(img['bytes'])).convert('RGB')
                im.save(fp, quality=92)
            counts[lab] = i + 1; n += 1
        if n % 5000 < 256:
            print(f'   {split} {n} imgs, {len(counts)} classes', flush=True)
    return counts


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else 'val'
    if job == 'val':
        c = {}
        for fn in VAL:
            c = explode(fetch(fn), 'val', per_class=None, counts=c)
        print('val classes', len(c), 'imgs', sum(c.values()))
    else:
        nsh = int(os.environ.get('NSHARD', 11))
        per = int(os.environ.get('PERCLASS', 400))
        import urllib.request as ur, json as js
        meta = js.loads(ur.urlopen(
            'https://huggingface.co/api/datasets/evanarlian/'
            'imagenet_1k_resized_256?full=true').read())
        names = sorted(s['rfilename'].split('/')[-1]
                       for s in meta['siblings']
                       if s['rfilename'].startswith('data/train-'))
        c = {}
        for fn in names[:nsh]:
            c = explode(fetch(fn), 'train', per_class=per, counts=c)
            print(f'  after {fn}: {len(c)} classes, {sum(c.values())} imgs',
                  flush=True)
            os.remove(os.path.join(DL, fn))
        print('train classes', len(c), 'imgs', sum(c.values()))
        json.dump({str(k): v for k, v in c.items()},
                  open(os.path.join(ROOT, 'data', 'in1k', 'train_counts.json'), 'w'))


if __name__ == '__main__':
    main()
