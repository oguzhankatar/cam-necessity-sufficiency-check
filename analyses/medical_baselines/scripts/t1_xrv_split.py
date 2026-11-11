"""C2 task 1 -- reconstruct the torchxrayvision train/held-out patient split for
`densenet121-res224-nih` and `densenet121-res224-rsna` and intersect it with our
evaluation images.

Source of the recipe (torchxrayvision repo, scripts/train_model.py, master):
    dataset = xrv.datasets.NIH_Dataset(imgpath=..., unique_patients=False, views=["PA","AP"])
    dataset = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=.../stage_2_train_images_jpg,
                                                  unique_patients=False, views=["PA","AP"])
    gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8, test_size=0.2,
                                                    random_state=cfg.seed)   # seed default 0
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
scripts/train_utils.py: the 20% part is the `valid_dataset`, and the saved
`*-best.pt` is the epoch with the highest mean AUC on it.

The reconstruction uses the CSVs shipped with the installed torchxrayvision
(1.5.4).  GroupShuffleSplit permutes the sorted unique groups, so the result
depends only on the set of patient ids, not on row order or label version.
It is a reconstruction: the exact CSV used in 2019-2020 cannot be checked.
"""
import os
import csv
import json
import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit
import torchxrayvision as xrv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *['..'] * 3))
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')


def split(ds):
    gss = GroupShuffleSplit(train_size=0.8, test_size=0.2, random_state=0)
    tr, te = next(gss.split(X=range(len(ds)), groups=ds.csv.patientid))
    return tr, te


def main():
    res = dict(xrv_version=xrv.__version__, sklearn_version=sklearn.__version__)
    # ---------------- NIH
    nih = xrv.datasets.NIH_Dataset(imgpath=os.path.join(ROOT, 'data', 'nih', 'images'),
                                   unique_patients=False, views=["PA", "AP"])
    tr, te = split(nih)
    names = nih.csv['Image Index'].values
    trn, ten = set(names[tr]), set(names[te])
    D = os.path.join(ROOT, 'data', 'nih')
    boxed = set()
    for r in csv.reader(open(os.path.join(D, 'BBox_List_2017.csv'))):
        if r and r[0].strip() and r[0] != 'Image Index':
            boxed.add(r[0].strip())
    test = [l.strip() for l in open(os.path.join(D, 'split_test.txt')) if l.strip()]
    train = [l.strip() for l in open(os.path.join(D, 'split_train.txt')) if l.strip()]
    official_test = set(l.strip() for l in open(os.path.join(D, 'test_list.txt')) if l.strip())
    allnames = set(names)
    pid = dict(zip(names, nih.csv.patientid.values))
    bx = sorted(boxed)
    res['nih'] = dict(
        n_images_in_xrv_dataset=int(len(nih)), n_patients=int(nih.csv.patientid.nunique()),
        n_xrv_train_images=len(trn), n_xrv_heldout_images=len(ten),
        boxed_images=len(bx),
        boxed_in_xrv_train=sum(n in trn for n in bx),
        boxed_in_xrv_heldout=sum(n in ten for n in bx),
        boxed_not_in_xrv_dataset=sum(n not in allnames for n in bx),
        boxed_patients=len(set(pid[n] for n in bx if n in pid)),
        test5720_in_xrv_train=sum(n in trn for n in test),
        test5720_in_xrv_heldout=sum(n in ten for n in test),
        test5720_n=len(test),
        train7000_in_xrv_heldout=sum(n in ten for n in train),
        official_test_list_in_xrv_train=sum(n in trn for n in official_test),
        official_test_list_n=len(official_test),
        boxed_all_in_official_test=all(n in official_test for n in bx))
    json.dump(sorted(n for n in bx if n in ten),
              open(os.path.join(OUT, 'nih_boxed_xrv_heldout.json'), 'w'))
    json.dump(sorted(ten), open(os.path.join(OUT, 'nih_xrv_heldout_images.json'), 'w'))

    # ---------------- RSNA
    rs = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=os.path.join(ROOT, 'data', 'rsna', 'dcm'),
        views=["PA", "AP"])  # this xrv version has no unique_patients arg; one row per patient anyway
    tr, te = split(rs)
    pids = rs.csv.patientid.values
    trp, tep = set(pids[tr]), set(pids[te])
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'rsna', 'stage_2_train_labels.csv'))))
    tgt = {}
    for r in rows:
        tgt[r['patientId']] = max(tgt.get(r['patientId'], 0), int(r['Target']))
    have = sorted(n[:-4] for n in os.listdir(os.path.join(ROOT, 'data', 'rsna', 'dcm')))
    evalp = [n for n in have if tgt[n] == 1]
    res['rsna'] = dict(
        n_images_in_xrv_dataset=int(len(rs)), n_patients=int(len(set(pids))),
        n_xrv_train_patients=len(trp), n_xrv_heldout_patients=len(tep),
        local_dicoms=len(have), eval_positive=len(evalp),
        eval_in_xrv_train=sum(n in trp for n in evalp),
        eval_in_xrv_heldout=sum(n in tep for n in evalp),
        eval_not_in_xrv_dataset=sum(n not in trp and n not in tep for n in evalp),
        local_all_in_xrv_train=sum(n in trp for n in have),
        local_all_in_xrv_heldout=sum(n in tep for n in have),
        stage2_train_label_patients=len(tgt))
    json.dump(sorted(n for n in have if n in tep),
              open(os.path.join(OUT, 'rsna_local_xrv_heldout.json'), 'w'))
    json.dump(res, open(os.path.join(OUT, 't1_xrv_split.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
