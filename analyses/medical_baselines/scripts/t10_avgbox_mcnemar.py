# -*- coding: utf-8 -*-
"""Exact McNemar of the cross-fitted average box against each of the six
other NIH maps of panel (d), Holm across the six, from the per-item hits that the scorer of this folder
(t6_score.score_arm) recomputes from out/maps_nih_m4.npz.
out/ is read only except for the result, out/t10_avgbox_mcnemar.json (Figure 4(d)).

    python analyses/medical_baselines/scripts/t10_avgbox_mcnemar.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import medical_baselines_lib as L  # noqa: E402
import t6_score as T6  # noqa: E402

OTHERS = ["actnorm", "center", "N_lse-8.0_1x1", "collapsed_1x1", "cam", "gradcam_gap"]
REF = "avgbox_cf5_full"


def main():
    S = T6.score_arm("nih_m4")[0]
    a = np.asarray(S[REF]["hit"], float)
    out = {"reference": REF, "n": int(len(a)), "rate": {REF: float(np.mean(a))}, "tests": {}}
    ps = []
    for o in OTHERS:
        b = np.asarray(S[o]["hit"], float)
        t = L.cmp(a, b, True)
        out["rate"][o] = float(np.mean(b))
        out["tests"][o] = {k: t[k] for k in ("diff", "a_only", "b_only", "p_mcnemar")}
        ps.append(t["p_mcnemar"])
    for o, ph in zip(OTHERS, L.holm(ps)):
        out["tests"][o]["p_holm"] = float(ph)
    # agreement with the stored comparator tests of t6_score.py
    ref = json.load(open(os.path.join(HERE, "..", "out", "t6_score.json")))["nih_m4"]
    chk = {}
    for o in OTHERS:
        k = "%s_vs_%s" % (REF, o)
        if k in ref["comparator_tests"]:
            chk[o] = ref["comparator_tests"][k]["p_mcnemar"] - out["tests"][o]["p_mcnemar"]
    out["max_abs_dev_from_C2"] = max(abs(v) for v in chk.values())
    json.dump(out, open(os.path.join(HERE, "..", "out", "t10_avgbox_mcnemar.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
