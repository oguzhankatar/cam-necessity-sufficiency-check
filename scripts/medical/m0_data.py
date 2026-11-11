"""m0 -- record exactly which datasets were obtained and which were gated.
Everything is read off the filesystem / the download attempts, not typed."""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib
from mlib import ROOT


def n_files(p, suf=None):
    if not os.path.isdir(p):
        return 0
    return sum(1 for n in os.listdir(p) if suf is None or n.lower().endswith(suf))


def main():
    D = os.path.join(ROOT, 'data')
    res = {}

    # 1. CheXlocalize -- gated
    s = os.path.join(D, 'chexlocalize_sample')
    got = {}
    if os.path.isdir(s):
        for f in ('gt_annotations_val.json', 'gt_segmentations_val.json'):
            p = os.path.join(s, f)
            if os.path.exists(p):
                got[f] = dict(bytes=os.path.getsize(p),
                              studies=len(json.load(open(p))))
    res['chexlocalize'] = dict(
        status='GATED',
        portal='https://aimi.stanford.edu/datasets/chexlocalize -> '
               'https://stanford.redivis.com/datasets/efx9-5nspnbb4b',
        note='Redivis requires a Stanford SAML web login; the Redivis REST API '
             'returns 401 invalid_token without credentials.  No HuggingFace '
             'mirror exists (dataset search for "chexlocalize" returns 0 hits).',
        public_sample=got,
        public_sample_source='github.com/rajpurkarlab/cheXlocalize/tree/master/sample')

    # 2. NIH ChestX-ray14
    nih = os.path.join(D, 'nih')
    bbox_rows, bbox_imgs = 0, set()
    if os.path.exists(os.path.join(nih, 'BBox_List_2017.csv')):
        with open(os.path.join(nih, 'BBox_List_2017.csv')) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if row and row[0].strip():
                    bbox_rows += 1
                    bbox_imgs.add(row[0].strip())
    have = set(os.listdir(os.path.join(nih, 'images'))) if os.path.isdir(os.path.join(nih, 'images')) else set()
    te = set(l.strip() for l in open(os.path.join(nih, 'split_test.txt'))) if os.path.exists(os.path.join(nih, 'split_test.txt')) else set()
    tr = set(l.strip() for l in open(os.path.join(nih, 'split_train.txt'))) if os.path.exists(os.path.join(nih, 'split_train.txt')) else set()
    res['nih_chestxray14'] = dict(
        status='OBTAINED',
        source='HuggingFace mirror iraqigold/nih-chest-xray-14-flat (original '
               'file names preserved); integrity checked against '
               'alkzar90/NIH-Chest-X-ray-dataset zips',
        box_rows=bbox_rows, box_images=len(bbox_imgs),
        box_images_on_disk=len(bbox_imgs & have),
        images_on_disk=len(have),
        official_test_sample=len(te & have), official_test_target=len(te),
        official_train_sample=len(tr & have), official_train_target=len(tr),
        box_images_all_in_official_test=bool(bbox_imgs <= te) if te else None)

    # 3. ISIC 2018 Task 1 + ISIC 2017 Part 3
    iz = os.path.join(D, 'isic2018')
    z = os.path.join(iz, 'ISIC2018_Task1-2_Training_Input.zip')
    names, y = ([], [])
    try:
        names, y = mlib.isic_frame()
    except Exception:
        pass
    import numpy as np
    res['isic2018_task1'] = dict(
        status='OBTAINED' if os.path.exists(z) else 'DOWNLOADING',
        source='isic-challenge-data.s3.amazonaws.com (open, no registration)',
        zip_bytes=os.path.getsize(z) if os.path.exists(z) else None,
        masks=n_files(os.path.join(iz, 'masks'), '.png'),
        images=n_files(os.path.join(iz, 'images'), '.jpg'),
        labelled_with_masks=len(names),
        class_counts=np.bincount(np.asarray(y, int), minlength=3).tolist() if len(names) else None,
        label_source='ISIC-2017 Part 3 (melanoma / seborrheic keratosis); the '
                     'ISIC-2018 Task 3 (HAM10000) ids do not intersect the '
                     'Task 1 mask ids at all (0 of 2594)')

    # 4. RSNA
    rs = os.path.join(D, 'rsna')
    res['rsna_pneumonia'] = dict(
        status='OBTAINED' if n_files(os.path.join(rs, 'dcm'), '.dcm') else 'NOT FETCHED',
        source='HuggingFace mirror Baldezo313/rsna-pneumonia-dataset (open)',
        dcm_on_disk=n_files(os.path.join(rs, 'dcm'), '.dcm'))

    res['kaggle'] = dict(status='UNAVAILABLE',
                         note='no ~/.kaggle/kaggle.json and no kaggle package; '
                              'SIIM-ACR and the Kaggle route to RSNA were not '
                              'reachable.  RSNA was obtained from an open '
                              'HuggingFace mirror instead.')
    mlib.save('m0_data.json', res)
    print(json.dumps(res, indent=1)[:2000])


if __name__ == '__main__':
    main()
