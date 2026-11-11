"""W4 -- the trivial floors of the box metrics, computed from ground truth alone.

Two numbers a WSOL reader always asks for:

  whole_image_box   the score of predicting the WHOLE IMAGE as the box.  Any
                    min-max normalised map reaches this at threshold 0, so it is
                    the floor of MaxBoxAccV2: a map that carries no information
                    at all still scores this much.
  gt_prior_box      the score of the single fixed box that maximises the metric
                    over the evaluation set (a centred box of the best size),
                    i.e. the best image-independent predictor.

No model, no features: ground truth only.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W
from PIL import Image

R = W.PROTOCOL['eval_res']
D = tuple(W.PROTOCOL['deltas'])
out = {}

man = W.cub_manifest()
G = []
for r in man['ev']:
    w0, h0 = Image.open(r['path']).size
    b = r['boxes'][0]
    G.append([b[0] * R / w0, b[1] * R / h0, b[2] * R / w0, b[3] * R / h0])
G = np.array(G, float)
ga = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])


def score(E):
    """E: (n,4) predicted boxes, one per image -> MaxBoxAccV2-style mean over deltas."""
    ix = np.maximum(0, np.minimum(E[:, 2], G[:, 2]) - np.maximum(E[:, 0], G[:, 0]))
    iy = np.maximum(0, np.minimum(E[:, 3], G[:, 3]) - np.maximum(E[:, 1], G[:, 1]))
    inter = ix * iy
    ea = (E[:, 2] - E[:, 0]) * (E[:, 3] - E[:, 1])
    iou = inter / (ea + ga - inter + 1e-9)
    return float(np.mean([(iou >= d).mean() for d in D])), iou


full = np.tile([0.0, 0.0, R, R], (len(G), 1))
v_full, iou_full = score(full)

best = (-1, None)
for frac in np.arange(0.20, 1.001, 0.02):
    s = R * frac
    o = (R - s) / 2.0
    E = np.tile([o, o, o + s, o + s], (len(G), 1))
    v, _ = score(E)
    if v > best[0]:
        best = (v, float(frac))

out['cub'] = dict(n=len(G), gt_box_area_fraction_mean=float((ga / (R * R)).mean()),
                  whole_image_box=v_full,
                  whole_image_box_mean_iou=float(iou_full.mean()),
                  best_fixed_centred_box=best[0],
                  best_fixed_centred_box_side_fraction=best[1],
                  deltas=list(D))
W.save('w4_floor.json', out)
print(json.dumps(out, indent=1))
