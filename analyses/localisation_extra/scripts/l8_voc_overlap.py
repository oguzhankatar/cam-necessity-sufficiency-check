"""Task 8 -- is the VOC 2012 segmentation-val evaluation set a subset of the VOC
2012 (detection/Main) val evaluation set used for the box runs, and how large is
the overlap?

The image lists and records are rebuilt with wlib.voc_manifest itself (the code
the runs used: fit = ImageSets/Main/train, box eval = ImageSets/Main/val, mask
eval = ImageSets/Segmentation/val; one record per (image, annotated class)), and
checked against n_eval / n_records / n_fit / bank_idx stored in every
outputs/wsol/w2_voc12*_*.json, so the comparison is about these runs.
"""
import os
import glob
import json
import llib as L


def main():
    box = L.W.voc_manifest('box')
    seg = L.W.voc_manifest('seg')
    name = lambda r: os.path.splitext(os.path.basename(r['path']))[0]   # noqa: E731
    ev_box = [name(r) for r in box['ev']]
    ev_seg = [name(r) for r in seg['ev']]
    fit_box = [name(r) for r in box['fit']]
    fit_seg = [name(r) for r in seg['fit']]
    rec_box = {(name(r), c) for r in box['ev'] for c in r['labels']}
    rec_seg = {(name(r), c) for r in seg['ev'] for c in r['labels']}
    sb, ss = set(ev_box), set(ev_seg)
    # consistency with the stored runs
    chk = []
    for f in sorted(glob.glob(os.path.join(L.WO, 'w2_voc12*_*.json'))):
        j = json.load(open(f))
        ds = j['dataset']
        ev = ev_seg if ds == 'voc12seg' else ev_box
        recs = rec_seg if ds == 'voc12seg' else rec_box
        chk.append(dict(file=os.path.basename(f), dataset=ds, n_eval=j['n_eval'],
                        n_records=j['n_records'], n_fit=j['n_fit'], bank_idx=j['bank_idx'],
                        match=(j['n_eval'] == len(ev) and j['n_records'] == len(recs)
                               and j['n_fit'] == len(fit_box))))
    same_fit = fit_box == fit_seg
    banks = {tuple(c['bank_idx']) for c in chk if c['bank_idx']}
    d = os.path.join(L.W.DATA, 'VOCdevkit', 'VOC2012', 'ImageSets')
    rd = lambda f: {l.split()[0] for l in open(os.path.join(d, f)) if l.strip()}   # noqa: E731
    main_tv = rd(os.path.join('Main', 'trainval.txt'))
    outside = sorted(ss - sb)
    years = {}
    for n in outside:
        years[n[:4]] = years.get(n[:4], 0) + 1
    res = dict(
        n_box_eval_images=len(ev_box), n_seg_eval_images=len(ev_seg),
        n_overlap_images=len(sb & ss), seg_subset_of_box=ss <= sb,
        n_seg_not_in_box=len(ss - sb), frac_seg_in_box=len(sb & ss) / len(ss),
        frac_box_eval_shared=len(sb & ss) / len(sb),
        n_box_records=len(rec_box), n_seg_records=len(rec_seg),
        n_overlap_records=len(rec_box & rec_seg), seg_records_subset=rec_seg <= rec_box,
        n_fit_images=len(fit_box), same_fit_split=same_fit,
        n_seg_eval_in_fit=len(ss & set(fit_box)), n_box_eval_in_fit=len(sb & set(fit_box)),
        same_bank_all_bank_basis_runs=len(banks) == 1,
        n_seg_eval_outside_main_trainval=len(ss - main_tv), n_main_trainval=len(main_tv),
        seg_not_in_box_by_year=years, run_checks=chk)
    print(json.dumps({k: v for k, v in res.items() if k != 'run_checks'}, indent=1))
    for c in chk:
        print('  %-30s n_eval %-5d n_records %-5d n_fit %-5d match %s'
              % (c['file'], c['n_eval'], c['n_records'], c['n_fit'], c['match']))
    L.save('l8_voc_overlap.json', res)


if __name__ == '__main__':
    main()
