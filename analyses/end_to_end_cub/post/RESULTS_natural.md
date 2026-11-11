# K2 post hoc: the natural map of each head against the H0 collapsed map

**Post hoc analysis, run after K2 was complete; not part of the K2 protocol (protocols/K2_end_to_end_protocol.md). The pre-specified K2 results in analyses/end_to_end_cub/RESULTS.md are unchanged.**

Same 12 checkpoints, same 5794 CUB-200-2011 test images, same preprocessing (whole image resized to 448 x 448, no crop), ground-truth class, box rescaling, map up-sampling and min-max scaling, MaxBoxAccV2 (0.3/0.5/0.7, 100 thresholds), strict pointing and best IoU as analyses/end_to_end_cub/scripts/k2_eval.py (imported, not re-implemented). Backbone in float32 (TF32 off); every map, including Grad-CAM (autograd through a float64 copy of the trained head), in float64. Differences are A minus B, seed-matched, mean over the 3 seeds; 95% percentile bootstrap over test images (1000 resamples, the same image resamples as k2_stats.py); P = 2 x smaller tail share, floored at 1/1000; Holm within each family at alpha = 0.05.

## Natural maps

| head | map | definition (class c = ground truth, x_t = cell t of the 14 x 14 layer4 map) |
|---|---|---|
| H0 | CAM | <w_c, x_t> (recomputed; must equal the stored H0 CAM) |
| H0 | Grad-CAM | ReLU(sum_k alpha_k x_tk), alpha_k = mean_t dz_c/dx_tk; for GAP + linear = ReLU(<w_c, x_t>)/196 |
| H1 | CAM | s_t = <w_c, x_t> + b_c, the per-cell class score that log-sum-exp pools |
| H1 | Grad-CAM | standard; alpha = mean_t softmax_t(s) w_c = w_c/196, so = ReLU(<w_c, x_t>)/196 |
| H2 | attention | a_t = softmax_t(e_t), the gated-attention weights (class independent) |
| H2 | attention x evidence | a_t <w_c, x_t>: the head pools then applies the linear layer, z_c = <w_c, sum_t a_t x_t> + b_c = sum_t a_t <w_c, x_t> + b_c; bias excluded |
| H2 | Grad-CAM | standard (gradient through softmax attention and pooling) |
| H3 | Grad-CAM | standard; = ReLU(<J^T w_c, x_t>)/196, J = LayerNorm Jacobian at the pooled vector |
| H3 | CAM | <w_c, x_t>, LayerNorm ignored (reference) |

Grad-CAM closed forms (H0, H1, H3) were checked against autograd on every image: largest relative difference 3.6e-14. Largest |float64 head logit - float32 network logit| 1.0e-05.

## Sanity check: H0 CAM recomputed with this pipeline vs the stored K2 evaluation

| seed | MaxBoxAccV2 recomputed | stored | abs. difference | largest per-image curve difference | largest map difference | strict pointing identical |
|---|---|---|---|---|---|---|
| 0 | 0.6106892187 | 0.6106892187 | 0.0e+00 | 0.0e+00 | 0.0e+00 (max |map| 217) | True |
| 1 | 0.6089632954 | 0.6089632954 | 0.0e+00 | 0.0e+00 | 0.0e+00 (max |map| 207) | True |
| 2 | 0.6085605799 | 0.6085605799 | 0.0e+00 | 0.0e+00 | 0.0e+00 (max |map| 217) | True |

Bootstrap check: re-running the stored primary contrast H3_S - H0_collapsed, MaxBoxAccV2 with this code gives an absolute error of 0.0e+00 in the difference and 0.0e+00 in the interval; P identical: True.

## Localisation of every natural map (mean of 3 seeds)

| map | MaxBoxAccV2 | per seed | strict pointing | best IoU | pointing, 15 px | all-zero maps (per seed) |
|---|---|---|---|---|---|---|
| H0 collapsed (stored, K2) | 0.6049 | | 0.9962 | 0.6557 | 0.9990 | |
| H0 CAM (stored, K2) | 0.6094 | | 0.9965 | 0.6575 | 0.9991 | |
| H0 CAM (recomputed) | 0.6094 | 0.6107, 0.6090, 0.6086 | 0.9965 | 0.6575 | 0.9991 | 0, 0, 0 |
| H0 Grad-CAM | 0.6111 | 0.6120, 0.6101, 0.6112 | 0.9965 | 0.6574 | 0.9991 | 0, 0, 0 |
| H1 CAM (per-cell class score) | 0.5106 | 0.5051, 0.5210, 0.5058 | 0.9916 | 0.6499 | 0.9971 | 0, 0, 0 |
| H1 Grad-CAM | 0.5527 | 0.5446, 0.5637, 0.5497 | 0.9914 | 0.6296 | 0.9969 | 3, 3, 1 |
| H2 attention weights | 0.6323 | 0.6185, 0.6496, 0.6287 | 0.9967 | 0.6324 | 0.9991 | 0, 0, 0 |
| H2 attention x class evidence | 0.5788 | 0.5766, 0.5935, 0.5664 | 0.9980 | 0.5926 | 0.9998 | 0, 0, 0 |
| H2 Grad-CAM | 0.4303 | 0.4242, 0.4211, 0.4455 | 0.8941 | 0.6166 | 0.9398 | 1, 0, 0 |
| H3 Grad-CAM | 0.5105 | 0.5035, 0.5203, 0.5077 | 0.9819 | 0.5978 | 0.9948 | 0, 0, 0 |
| H3 CAM (LayerNorm ignored) | 0.6203 | 0.6202, 0.6209, 0.6197 | 0.9976 | 0.6790 | 0.9994 | 0, 0, 0 |

All-zero maps (Grad-CAM with no positive cell) become a zero map after min-max; as in wlib.upsample they then cover the whole image at every threshold.

## Post hoc family: MaxBoxAccV2 against H0 collapsed (7 natural maps of H1-H3)

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 CAM (per-cell class score) | H0 collapsed (stored) | -0.0943 | [-0.0994, -0.0882] | -0.0998, -0.0835, -0.0996 | 0.007 | worse |
| H1 Grad-CAM | H0 collapsed (stored) | -0.0523 | [-0.0573, -0.0469] | -0.0603, -0.0408, -0.0557 | 0.007 | worse |
| H2 attention weights | H0 collapsed (stored) | +0.0273 | [+0.0215, +0.0324] | +0.0136, +0.0452, +0.0232 | 0.007 | better |
| H2 attention x class evidence | H0 collapsed (stored) | -0.0261 | [-0.0326, -0.0208] | -0.0282, -0.0110, -0.0391 | 0.007 | worse |
| H2 Grad-CAM | H0 collapsed (stored) | -0.1746 | [-0.1808, -0.1674] | -0.1806, -0.1834, -0.1599 | 0.007 | worse |
| H3 Grad-CAM | H0 collapsed (stored) | -0.0944 | [-0.1003, -0.0888] | -0.1014, -0.0842, -0.0977 | 0.007 | worse |
| H3 CAM (LayerNorm ignored) | H0 collapsed (stored) | +0.0154 | [+0.0111, +0.0197] | +0.0154, +0.0165, +0.0143 | 0.007 | better |

## Strict pointing against H0 collapsed

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 CAM (per-cell class score) | H0 collapsed (stored) | -0.0046 | [-0.0064, -0.0028] | -0.0040, -0.0041, -0.0057 | 0.007 | worse |
| H1 Grad-CAM | H0 collapsed (stored) | -0.0048 | [-0.0067, -0.0029] | -0.0043, -0.0043, -0.0059 | 0.007 | worse |
| H2 attention weights | H0 collapsed (stored) | +0.0005 | [-0.0011, +0.0021] | -0.0003, +0.0014, +0.0005 | 0.582 | no difference |
| H2 attention x class evidence | H0 collapsed (stored) | +0.0018 | [+0.0003, +0.0033] | +0.0007, +0.0022, +0.0024 | 0.036 | better |
| H2 Grad-CAM | H0 collapsed (stored) | -0.1021 | [-0.1086, -0.0953] | -0.1004, -0.1093, -0.0967 | 0.007 | worse |
| H3 Grad-CAM | H0 collapsed (stored) | -0.0143 | [-0.0169, -0.0118] | -0.0154, -0.0126, -0.0150 | 0.007 | worse |
| H3 CAM (LayerNorm ignored) | H0 collapsed (stored) | +0.0014 | [+0.0001, +0.0029] | +0.0012, +0.0005, +0.0026 | 0.08 | no difference |

## Best IoU against H0 collapsed

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 CAM (per-cell class score) | H0 collapsed (stored) | -0.0058 | [-0.0081, -0.0035] | -0.0078, +0.0002, -0.0096 | 0.007 | worse |
| H1 Grad-CAM | H0 collapsed (stored) | -0.0260 | [-0.0285, -0.0237] | -0.0269, -0.0203, -0.0309 | 0.007 | worse |
| H2 attention weights | H0 collapsed (stored) | -0.0233 | [-0.0260, -0.0207] | -0.0273, -0.0137, -0.0288 | 0.007 | worse |
| H2 attention x class evidence | H0 collapsed (stored) | -0.0630 | [-0.0656, -0.0604] | -0.0638, -0.0561, -0.0692 | 0.007 | worse |
| H2 Grad-CAM | H0 collapsed (stored) | -0.0391 | [-0.0419, -0.0362] | -0.0450, -0.0433, -0.0291 | 0.007 | worse |
| H3 Grad-CAM | H0 collapsed (stored) | -0.0579 | [-0.0606, -0.0551] | -0.0608, -0.0551, -0.0577 | 0.007 | worse |
| H3 CAM (LayerNorm ignored) | H0 collapsed (stored) | +0.0234 | [+0.0218, +0.0248] | +0.0223, +0.0259, +0.0219 | 0.007 | better |

## MaxBoxAccV2 against H0 CAM

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 CAM (per-cell class score) | H0 CAM (stored) | -0.0988 | [-0.1043, -0.0926] | -0.1056, -0.0880, -0.1027 | 0.007 | worse |
| H1 Grad-CAM | H0 CAM (stored) | -0.0567 | [-0.0619, -0.0514] | -0.0661, -0.0453, -0.0589 | 0.007 | worse |
| H2 attention weights | H0 CAM (stored) | +0.0229 | [+0.0172, +0.0278] | +0.0078, +0.0407, +0.0201 | 0.007 | better |
| H2 attention x class evidence | H0 CAM (stored) | -0.0306 | [-0.0368, -0.0251] | -0.0341, -0.0155, -0.0422 | 0.007 | worse |
| H2 Grad-CAM | H0 CAM (stored) | -0.1791 | [-0.1853, -0.1721] | -0.1865, -0.1878, -0.1630 | 0.007 | worse |
| H3 Grad-CAM | H0 CAM (stored) | -0.0989 | [-0.1043, -0.0934] | -0.1072, -0.0887, -0.1009 | 0.007 | worse |
| H3 CAM (LayerNorm ignored) | H0 CAM (stored) | +0.0109 | [+0.0066, +0.0152] | +0.0096, +0.0120, +0.0112 | 0.007 | better |

## Strict pointing against H0 CAM

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 CAM (per-cell class score) | H0 CAM (stored) | -0.0049 | [-0.0067, -0.0030] | -0.0040, -0.0043, -0.0064 | 0.007 | worse |
| H1 Grad-CAM | H0 CAM (stored) | -0.0051 | [-0.0070, -0.0032] | -0.0043, -0.0045, -0.0066 | 0.007 | worse |
| H2 attention weights | H0 CAM (stored) | +0.0002 | [-0.0014, +0.0018] | -0.0003, +0.0012, -0.0002 | 0.79 | no difference |
| H2 attention x class evidence | H0 CAM (stored) | +0.0015 | [+0.0001, +0.0029] | +0.0007, +0.0021, +0.0017 | 0.114 | no difference |
| H2 Grad-CAM | H0 CAM (stored) | -0.1024 | [-0.1087, -0.0956] | -0.1004, -0.1094, -0.0973 | 0.007 | worse |
| H3 Grad-CAM | H0 CAM (stored) | -0.0146 | [-0.0171, -0.0120] | -0.0154, -0.0128, -0.0157 | 0.007 | worse |
| H3 CAM (LayerNorm ignored) | H0 CAM (stored) | +0.0012 | [-0.0002, +0.0025] | +0.0012, +0.0003, +0.0019 | 0.16 | no difference |

## Best IoU against H0 CAM

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 CAM (per-cell class score) | H0 CAM (stored) | -0.0075 | [-0.0098, -0.0053] | -0.0098, -0.0017, -0.0112 | 0.007 | worse |
| H1 Grad-CAM | H0 CAM (stored) | -0.0278 | [-0.0302, -0.0256] | -0.0289, -0.0222, -0.0325 | 0.007 | worse |
| H2 attention weights | H0 CAM (stored) | -0.0251 | [-0.0279, -0.0225] | -0.0293, -0.0155, -0.0304 | 0.007 | worse |
| H2 attention x class evidence | H0 CAM (stored) | -0.0648 | [-0.0674, -0.0623] | -0.0658, -0.0579, -0.0707 | 0.007 | worse |
| H2 Grad-CAM | H0 CAM (stored) | -0.0409 | [-0.0438, -0.0381] | -0.0470, -0.0451, -0.0307 | 0.007 | worse |
| H3 Grad-CAM | H0 CAM (stored) | -0.0597 | [-0.0624, -0.0569] | -0.0628, -0.0570, -0.0593 | 0.007 | worse |
| H3 CAM (LayerNorm ignored) | H0 CAM (stored) | +0.0216 | [+0.0201, +0.0229] | +0.0203, +0.0240, +0.0204 | 0.007 | better |

## H0 references against H0 collapsed, MaxBoxAccV2

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H0 CAM (recomputed) | H0 collapsed (stored) | +0.0045 | [+0.0021, +0.0066] | +0.0058, +0.0045, +0.0031 | 0.002 | better |
| H0 Grad-CAM | H0 collapsed (stored) | +0.0062 | [+0.0038, +0.0087] | +0.0071, +0.0056, +0.0058 | 0.002 | better |

## H0 references against H0 collapsed, strict pointing

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H0 CAM (recomputed) | H0 collapsed (stored) | +0.0003 | [-0.0003, +0.0009] | +0.0000, +0.0002, +0.0007 | 0.792 | no difference |
| H0 Grad-CAM | H0 collapsed (stored) | +0.0003 | [-0.0003, +0.0009] | +0.0000, +0.0002, +0.0007 | 0.792 | no difference |

## H0 references against H0 collapsed, best IoU

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H0 CAM (recomputed) | H0 collapsed (stored) | +0.0018 | [+0.0011, +0.0025] | +0.0020, +0.0018, +0.0015 | 0.002 | better |
| H0 Grad-CAM | H0 collapsed (stored) | +0.0017 | [+0.0010, +0.0024] | +0.0019, +0.0017, +0.0015 | 0.002 | better |

Families: the 7 natural maps of H1-H3 form one Holm family per metric and per reference (H0 collapsed, H0 CAM); the two H0 references form their own family of 2 per metric. Every contrast is reported whatever its sign.

Scripts: analyses/end_to_end_cub/post/scripts/k2_natural.py (maps and metrics, per model), k2_natural_driver.py, k2_natural_stats.py (this file and analyses/end_to_end_cub/post/out/k2_natural.json). Per-image arrays: analyses/end_to_end_cub/post/out/per_image/<H>_s<seed>_natural.npz.
