"""C2 task 8 -- the 7x7 resolution ceiling per NIH finding.

A 7x7 map over a 224-px input has cells of 32 px, i.e. 1024/7 = 146.3 px in
the 1024-px frame of the boxes.  For each finding we report
  * area_lt_cell   fraction of boxes whose area is below one cell (146.3^2 px^2)
  * both_lt_cell   fraction of boxes narrower AND shorter than one cell
  * gt_frac        mean fraction of the image the box covers
  * ceiling_hit    fraction of scored items (finding x image, union of boxes, at
                   224 px exactly as m4.gt_mask) whose mask contains at least
                   one pixel at which the argmax of a bilinearly upsampled 7x7
                   map (mlib.pointing_hit) can land -- the best hit rate any
                   7x7 map could reach under the protocol.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medical_baselines_lib as L
from medical_baselines_lib import mlib
import m4_stats as m4
import t6_score as T6


def main():
    bx = mlib.nih_boxes()
    cell = 1024. / 7
    rows = [(lab, w, h) for bs in bx.values() for (lab, x, y, w, h) in bs]
    reach = T6.reachable_pixels()
    out = {}
    for f in sorted(set(r[0] for r in rows)):
        R = [(w, h) for lab, w, h in rows if lab == f]
        A = np.array([w * h for w, h in R])
        items = [(n, f) for n, bs in bx.items() if any(b[0] == f for b in bs)]
        ceil_ = [float(any(m4.gt_mask(bx[n], f)[y, x] for y, x in reach)) for n, _ in items]
        out[f] = dict(n_boxes=len(R), n_items=len(items),
                      area_lt_cell=float(np.mean(A < cell ** 2)),
                      both_lt_cell=float(np.mean([(w < cell) and (h < cell) for w, h in R])),
                      either_lt_cell=float(np.mean([(w < cell) or (h < cell) for w, h in R])),
                      gt_frac=float(np.mean(A / 1024. ** 2)),
                      median_side_px1024=float(np.median([np.sqrt(w * h) for w, h in R])),
                      ceiling_hit=float(np.mean(ceil_)))
    A = np.array([w * h for _, w, h in rows])
    out['ALL'] = dict(n_boxes=len(rows), area_lt_cell=float(np.mean(A < cell ** 2)),
                      both_lt_cell=float(np.mean([(w < cell) and (h < cell) for _, w, h in rows])))
    out['reachable_pixels'] = len(reach)
    out['cell_px_1024'] = cell
    L.save('t8_resolution.json', out)
    for f, d in out.items():
        print(f, d)


if __name__ == '__main__':
    main()
