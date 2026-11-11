"""W0 -- inventory of the datasets actually acquired, emitted as JSON so the
report never states a count that was not counted."""
import os, sys, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W

inv = {}


def sz(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    n = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            n += os.path.getsize(os.path.join(r, f))
    return n


def sha_head(p, nbytes=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


# ---- CUB-200-2011
try:
    m = W.cub_manifest()
    d = os.path.join(W.DATA, 'CUB_200_2011')
    inv['CUB_200_2011'] = dict(
        obtained=True, source='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz',
        n_train=len(m['fit']), n_test=len(m['ev']), n_class=m['n_class'],
        boxes_per_image=1, bytes_on_disk=sz(d),
        archive_bytes=sz(os.path.join(W.DATA, 'dl', 'CUB_200_2011.tgz')),
        archive_sha256_first_1MB=sha_head(os.path.join(W.DATA, 'dl', 'CUB_200_2011.tgz')),
        role='MaxBoxAccV2 (Choe et al. CVPR 2020 WSOL benchmark)')
except Exception as e:
    inv['CUB_200_2011'] = dict(obtained=False, error=str(e)[:200])

# ---- PASCAL VOC 2012
try:
    mb = W.voc_manifest('box')
    ms = W.voc_manifest('seg')
    d = os.path.join(W.DATA, 'VOCdevkit', 'VOC2012')
    nbox = sum(len(v) for r in mb['ev'] for v in r['boxes_by_class'].values())
    inv['VOC2012'] = dict(
        obtained=True, source='http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',
        n_train=len(mb['fit']), n_val_det=len(mb['ev']), n_val_seg=len(ms['ev']),
        n_class=mb['n_class'], n_val_boxes=nbox,
        n_records_det=sum(len(r['labels']) for r in mb['ev']),
        n_records_seg=sum(len(r['labels']) for r in ms['ev']),
        bytes_on_disk=sz(d),
        archive_bytes=sz(os.path.join(W.DATA, 'dl', 'VOCtrainval_11-May-2012.tar')),
        role='pointing game (boxes on det-val; per-class masks + PxAP on seg-val)')
except Exception as e:
    inv['VOC2012'] = dict(obtained=False, error=str(e)[:200])

# ---- MS-COCO val2017
try:
    m = W.coco_manifest()
    d = os.path.join(W.DATA, 'coco')
    inv['COCO_val2017'] = dict(
        obtained=True, source='http://images.cocodataset.org/zips/val2017.zip',
        n_images_with_annotations=len(m['fit']) + len(m['ev']),
        n_fit_half=len(m['fit']), n_eval_half=len(m['ev']), n_class=m['n_class'],
        n_records_eval=sum(len(r['labels']) for r in m['ev']),
        bytes_on_disk=sz(d),
        role='pointing game + PxAP on instance masks (RISE / D-RISE protocol)')
except Exception as e:
    inv['COCO_val2017'] = dict(obtained=False, error=str(e)[:200])

# ---- OpenImages30k (WSOL subset)
inv['OpenImages30k'] = dict(
    obtained=False,
    reason=('clovaai/wsolevaluation distributes it only through OneDrive / Google '
            'Drive consent pages (prepare_openimages.sh points at '
            'onedrive.live.com/download?cid=B7111B95B80CCC66...).  A direct GET '
            'returns HTTP 403 with an HTML consent page, and the OpenImages V5 '
            'originals would have to be re-cropped with their metadata to '
            'reconstruct the benchmark.  SKIPPED, as instructed, rather than '
            'approximated.'),
    probe_http_status=403)

# ---- ILSVRC-2012 val bounding boxes
inv['ILSVRC2012_val_boxes'] = dict(
    obtained=False,
    reason=('the project ImageNet mirror (data/in1k) is the ungated HuggingFace '
            '`evanarlian/imagenet_1k_resized_256` re-encode, which discards the '
            'original ILSVRC2012_val_%08d filenames and reorders images inside '
            'each class directory.  The bbox XMLs are keyed by those filenames, '
            'so they cannot be joined to the mirror without re-downloading the '
            'gated original val set.  SKIPPED rather than guessed.'))

W.save('w0_data.json', inv)
for k, v in inv.items():
    print('%-24s %s' % (k, 'OK' if v.get('obtained') else 'SKIPPED'))
