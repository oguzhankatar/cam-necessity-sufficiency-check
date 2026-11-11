# K2 post hoc: rank-transform sensitivity of MaxBoxAccV2

**Post hoc sensitivity analysis, run after K2 and the natural-map analysis were complete; not part of the K2 protocol. analyses/end_to_end_cub/RESULTS.md and analyses/end_to_end_cub/post/RESULTS_natural.md are unchanged.**

Each stored 14 x 14 map (ground-truth class, 5794 CUB-200-2011 test images, 3 seeds) is scored twice with the unchanged K2 pipeline (wlib.upsample to 224 bilinear + min-max, wlib.box_curve at 100 thresholds, wlib.maxboxacc_v2 at 0.3/0.5/0.7, boxes rescaled as in k2_eval.py): **raw** = the stored map; **rank** = the map replaced by its within-image ranks (average ranks for ties) divided by 196. Differences are A minus H0 collapsed, seed-matched, mean of 3 seeds, raw against raw and rank against rank; 95% percentile bootstrap over test images with the 1000 image resamples of k2_stats.py; P = 2 x smaller tail share, floored at 1/1000; Holm within the families of the raw analyses: primary (6 N/S maps of H1-H3, analyses/end_to_end_cub/RESULTS.md), post hoc natural family (7 natural maps of H1-H3, RESULTS_natural.md), H0 references (H0 CAM, H0 Grad-CAM; RESULTS_natural.md). Grad-CAM maps with no positive cell are constant after ranking and, as in the raw analysis, become a zero map after min-max.

## Verification

Raw maps through this script reproduce the stored MaxBoxAccV2 of all 48 map x seed cells with largest absolute difference 0.0e+00 and the stored per-image curves with largest difference 0.0e+00. Weighted-indicator bootstrap vs wlib.maxboxacc_v2 on resampled indices: 0.0e+00. The 15 stored raw contrasts against H0 collapsed (analyses/end_to_end_cub/out/k2_contrasts.json primary; analyses/end_to_end_cub/post/out/k2_natural.json natural and H0-reference families; difference, interval and Holm P) are reproduced to 0.0e+00 (Holm P identical: True).

## MaxBoxAccV2, raw and rank

| map | raw, mean | raw, per seed | rank, mean | rank, per seed | rank minus raw |
|---|---|---|---|---|---|
| H0 collapsed | 0.6049 | 0.6049, 0.6045, 0.6055 | 0.5984 | 0.6004, 0.5968, 0.5980 | -0.0065 |
| H0 CAM | 0.6094 | 0.6107, 0.6090, 0.6086 | 0.6104 | 0.6107, 0.6114, 0.6091 | +0.0010 |
| H0 Grad-CAM | 0.6111 | 0.6120, 0.6101, 0.6112 | 0.6094 | 0.6103, 0.6107, 0.6072 | -0.0017 |
| H1 necessity | 0.2509 | 0.2509, 0.2509, 0.2509 | 0.5232 | 0.5215, 0.5320, 0.5162 | +0.2723 |
| H1 sufficiency | 0.5383 | 0.5333, 0.5471, 0.5345 | 0.5576 | 0.5553, 0.5651, 0.5524 | +0.0193 |
| H2 necessity | 0.4243 | 0.4189, 0.4262, 0.4278 | 0.4986 | 0.4817, 0.4975, 0.5165 | +0.0742 |
| H2 sufficiency | 0.4436 | 0.4479, 0.4448, 0.4383 | 0.5439 | 0.5353, 0.5563, 0.5403 | +0.1003 |
| H3 necessity | 0.4232 | 0.4212, 0.4264, 0.4219 | 0.4412 | 0.4367, 0.4491, 0.4378 | +0.0180 |
| H3 sufficiency | 0.6149 | 0.6134, 0.6186, 0.6128 | 0.6155 | 0.6118, 0.6221, 0.6127 | +0.0006 |
| H1 CAM (per-cell class score) | 0.5106 | 0.5051, 0.5210, 0.5058 | 0.5575 | 0.5547, 0.5655, 0.5523 | +0.0468 |
| H1 Grad-CAM | 0.5527 | 0.5446, 0.5637, 0.5497 | 0.5505 | 0.5501, 0.5596, 0.5417 | -0.0022 |
| H2 attention weights | 0.6323 | 0.6185, 0.6496, 0.6287 | 0.5976 | 0.5866, 0.6143, 0.5920 | -0.0347 |
| H2 attention x class evidence | 0.5788 | 0.5766, 0.5935, 0.5664 | 0.6188 | 0.6113, 0.6279, 0.6171 | +0.0399 |
| H2 Grad-CAM | 0.4303 | 0.4242, 0.4211, 0.4455 | 0.4836 | 0.4724, 0.4718, 0.5066 | +0.0533 |
| H3 Grad-CAM | 0.5105 | 0.5035, 0.5203, 0.5077 | 0.4863 | 0.4787, 0.4956, 0.4846 | -0.0242 |
| H3 CAM (LayerNorm ignored) | 0.6203 | 0.6202, 0.6209, 0.6197 | 0.6394 | 0.6349, 0.6459, 0.6374 | +0.0191 |

## Difference from H0 collapsed: raw vs rank

Raw and rank columns are each against H0 collapsed scored the same way (raw 0.6049, rank 0.5984).

| A | raw difference [95% interval] | raw P (Holm) | rank difference [95% interval] | rank, per seed | rank P (Holm) | sign change | verdict change |
|---|---|---|---|---|---|---|---|
| *primary (K2 protocol): 6 N/S maps vs H0 collapsed* | | | | | | | |
| H1 necessity | -0.3540 [-0.3616, -0.3471] | 0.006 (worse) | -0.0752 [-0.0805, -0.0694] | -0.0789, -0.0648, -0.0819 | 0.006 (worse) | no | no |
| H1 sufficiency | -0.0666 [-0.0724, -0.0617] | 0.006 (worse) | -0.0408 [-0.0459, -0.0351] | -0.0451, -0.0317, -0.0457 | 0.006 (worse) | no | no |
| H2 necessity | -0.1806 [-0.1876, -0.1733] | 0.006 (worse) | -0.0999 [-0.1061, -0.0935] | -0.1187, -0.0993, -0.0816 | 0.006 (worse) | no | no |
| H2 sufficiency | -0.1613 [-0.1681, -0.1547] | 0.006 (worse) | -0.0545 [-0.0594, -0.0499] | -0.0651, -0.0406, -0.0578 | 0.006 (worse) | no | no |
| H3 necessity | -0.1818 [-0.1876, -0.1753] | 0.006 (worse) | -0.1572 [-0.1626, -0.1516] | -0.1637, -0.1477, -0.1603 | 0.006 (worse) | no | no |
| H3 sufficiency | +0.0100 [+0.0059, +0.0145] | 0.006 (better) | +0.0171 [+0.0134, +0.0213] | +0.0114, +0.0253, +0.0147 | 0.006 (better) | no | no |
| *post hoc natural family: 7 natural maps of H1-H3 vs H0 collapsed* | | | | | | | |
| H1 CAM (per-cell class score) | -0.0943 [-0.0994, -0.0882] | 0.007 (worse) | -0.0409 [-0.0459, -0.0351] | -0.0457, -0.0314, -0.0457 | 0.007 (worse) | no | no |
| H1 Grad-CAM | -0.0523 [-0.0573, -0.0469] | 0.007 (worse) | -0.0480 [-0.0539, -0.0414] | -0.0503, -0.0372, -0.0564 | 0.007 (worse) | no | no |
| H2 attention weights | +0.0273 [+0.0215, +0.0324] | 0.007 (better) | -0.0008 [-0.0059, +0.0043] | -0.0138, +0.0175, -0.0060 | 0.776 (no difference) | yes | yes |
| H2 attention x class evidence | -0.0261 [-0.0326, -0.0208] | 0.007 (worse) | +0.0203 [+0.0153, +0.0252] | +0.0109, +0.0311, +0.0191 | 0.007 (better) | yes | yes |
| H2 Grad-CAM | -0.1746 [-0.1808, -0.1674] | 0.007 (worse) | -0.1148 [-0.1206, -0.1086] | -0.1280, -0.1250, -0.0915 | 0.007 (worse) | no | no |
| H3 Grad-CAM | -0.0944 [-0.1003, -0.0888] | 0.007 (worse) | -0.1121 [-0.1182, -0.1057] | -0.1217, -0.1013, -0.1135 | 0.007 (worse) | no | no |
| H3 CAM (LayerNorm ignored) | +0.0154 [+0.0111, +0.0197] | 0.007 (better) | +0.0410 [+0.0363, +0.0453] | +0.0345, +0.0491, +0.0394 | 0.007 (better) | no | no |
| *post hoc H0 references vs H0 collapsed* | | | | | | | |
| H0 CAM | +0.0045 [+0.0021, +0.0066] | 0.002 (better) | +0.0120 [+0.0088, +0.0145] | +0.0103, +0.0146, +0.0111 | 0.002 (better) | no | no |
| H0 Grad-CAM | +0.0062 [+0.0038, +0.0087] | 0.002 (better) | +0.0110 [+0.0077, +0.0140] | +0.0099, +0.0139, +0.0091 | 0.002 (better) | no | no |

## Difference from H0 collapsed, raw maps (full rows)

| family | A | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| primary | H1 necessity | -0.3540 | [-0.3616, -0.3471] | -0.3539, -0.3535, -0.3545 | 0.006 | worse |
| primary | H1 sufficiency | -0.0666 | [-0.0724, -0.0617] | -0.0716, -0.0574, -0.0710 | 0.006 | worse |
| primary | H2 necessity | -0.1806 | [-0.1876, -0.1733] | -0.1859, -0.1782, -0.1777 | 0.006 | worse |
| primary | H2 sufficiency | -0.1613 | [-0.1681, -0.1547] | -0.1570, -0.1597, -0.1672 | 0.006 | worse |
| primary | H3 necessity | -0.1818 | [-0.1876, -0.1753] | -0.1836, -0.1781, -0.1836 | 0.006 | worse |
| primary | H3 sufficiency | +0.0100 | [+0.0059, +0.0145] | +0.0085, +0.0142, +0.0074 | 0.006 | better |
| natural | H1 CAM (per-cell class score) | -0.0943 | [-0.0994, -0.0882] | -0.0998, -0.0835, -0.0996 | 0.007 | worse |
| natural | H1 Grad-CAM | -0.0523 | [-0.0573, -0.0469] | -0.0603, -0.0408, -0.0557 | 0.007 | worse |
| natural | H2 attention weights | +0.0273 | [+0.0215, +0.0324] | +0.0136, +0.0452, +0.0232 | 0.007 | better |
| natural | H2 attention x class evidence | -0.0261 | [-0.0326, -0.0208] | -0.0282, -0.0110, -0.0391 | 0.007 | worse |
| natural | H2 Grad-CAM | -0.1746 | [-0.1808, -0.1674] | -0.1806, -0.1834, -0.1599 | 0.007 | worse |
| natural | H3 Grad-CAM | -0.0944 | [-0.1003, -0.0888] | -0.1014, -0.0842, -0.0977 | 0.007 | worse |
| natural | H3 CAM (LayerNorm ignored) | +0.0154 | [+0.0111, +0.0197] | +0.0154, +0.0165, +0.0143 | 0.007 | better |
| h0_refs | H0 CAM | +0.0045 | [+0.0021, +0.0066] | +0.0058, +0.0045, +0.0031 | 0.002 | better |
| h0_refs | H0 Grad-CAM | +0.0062 | [+0.0038, +0.0087] | +0.0071, +0.0056, +0.0058 | 0.002 | better |

## Difference from H0 collapsed, rank-transformed maps (full rows)

| family | A | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| primary | H1 necessity | -0.0752 | [-0.0805, -0.0694] | -0.0789, -0.0648, -0.0819 | 0.006 | worse |
| primary | H1 sufficiency | -0.0408 | [-0.0459, -0.0351] | -0.0451, -0.0317, -0.0457 | 0.006 | worse |
| primary | H2 necessity | -0.0999 | [-0.1061, -0.0935] | -0.1187, -0.0993, -0.0816 | 0.006 | worse |
| primary | H2 sufficiency | -0.0545 | [-0.0594, -0.0499] | -0.0651, -0.0406, -0.0578 | 0.006 | worse |
| primary | H3 necessity | -0.1572 | [-0.1626, -0.1516] | -0.1637, -0.1477, -0.1603 | 0.006 | worse |
| primary | H3 sufficiency | +0.0171 | [+0.0134, +0.0213] | +0.0114, +0.0253, +0.0147 | 0.006 | better |
| natural | H1 CAM (per-cell class score) | -0.0409 | [-0.0459, -0.0351] | -0.0457, -0.0314, -0.0457 | 0.007 | worse |
| natural | H1 Grad-CAM | -0.0480 | [-0.0539, -0.0414] | -0.0503, -0.0372, -0.0564 | 0.007 | worse |
| natural | H2 attention weights | -0.0008 | [-0.0059, +0.0043] | -0.0138, +0.0175, -0.0060 | 0.776 | no difference |
| natural | H2 attention x class evidence | +0.0203 | [+0.0153, +0.0252] | +0.0109, +0.0311, +0.0191 | 0.007 | better |
| natural | H2 Grad-CAM | -0.1148 | [-0.1206, -0.1086] | -0.1280, -0.1250, -0.0915 | 0.007 | worse |
| natural | H3 Grad-CAM | -0.1121 | [-0.1182, -0.1057] | -0.1217, -0.1013, -0.1135 | 0.007 | worse |
| natural | H3 CAM (LayerNorm ignored) | +0.0410 | [+0.0363, +0.0453] | +0.0345, +0.0491, +0.0394 | 0.007 | better |
| h0_refs | H0 CAM | +0.0120 | [+0.0088, +0.0145] | +0.0103, +0.0146, +0.0111 | 0.002 | better |
| h0_refs | H0 Grad-CAM | +0.0110 | [+0.0077, +0.0140] | +0.0099, +0.0139, +0.0091 | 0.002 | better |
