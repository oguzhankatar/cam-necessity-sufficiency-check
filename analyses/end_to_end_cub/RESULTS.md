# K2: localisation of separable heads trained end to end (CUB-200-2011)

ResNet-50, 448 x 448, 3 seeds per head; 5794 test images; maps for the ground-truth class from 1 x 1 cell substitution in float64, averaged over 8 training-split references. Differences are A minus B, mean over seeds; intervals are 95% percentile bootstrap over test images (1000 resamples); P is Holm-adjusted within each family at alpha = 0.05.

## Accuracy and separation

| head | top-1 (mean of 3) | top-1 per seed | separation, 1 reference | rank separation, 1 reference | separation, 8 references | rank separation, 8 references |
|---|---|---|---|---|---|---|
| H0 (average pooling, degenerate) | 0.8238 | 0.8267, 0.8227, 0.8219 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| H1 (log-sum-exp, T = +1) | 0.7654 | 0.7661, 0.7644, 0.7656 | 0.4322 | 0.0100 | 0.4663 | 0.1793 |
| H2 (gated attention) | 0.8175 | 0.8160, 0.8193, 0.8172 | 0.5219 | 0.9943 | 0.4387 | 0.8053 |
| H3 (average pooling + LayerNorm) | 0.8187 | 0.8217, 0.8164, 0.8179 | 0.2381 | 0.3454 | 0.1937 | 0.3660 |

Separation = 1 - Pearson correlation of the necessity and sufficiency maps; rank separation = 1 - Spearman correlation; per image, mean over images and seeds.

## Localisation of every map (mean of 3 seeds)

| map | MaxBoxAccV2 | per seed | strict pointing | best IoU | pointing, 15 px |
|---|---|---|---|---|---|
| H0 collapsed | 0.6049 | 0.6049, 0.6045, 0.6055 | 0.9962 | 0.6557 | 0.9990 |
| H0 CAM | 0.6094 | 0.6107, 0.6090, 0.6086 | 0.9965 | 0.6575 | 0.9991 |
| H1 necessity | 0.2509 | 0.2509, 0.2509, 0.2509 | 0.9900 | 0.3802 | 0.9963 |
| H1 sufficiency | 0.5383 | 0.5333, 0.5471, 0.5345 | 0.9915 | 0.6082 | 0.9971 |
| H2 necessity | 0.4243 | 0.4189, 0.4262, 0.4278 | 0.8660 | 0.5627 | 0.9402 |
| H2 sufficiency | 0.4436 | 0.4479, 0.4448, 0.4383 | 0.9970 | 0.6135 | 0.9993 |
| H3 necessity | 0.4232 | 0.4212, 0.4264, 0.4219 | 0.9833 | 0.5940 | 0.9965 |
| H3 sufficiency | 0.6149 | 0.6134, 0.6186, 0.6128 | 0.9975 | 0.6751 | 0.9994 |

## Primary: MaxBoxAccV2 against the collapsed map of H0 (6 contrasts)

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H0 collapsed | -0.3540 | [-0.3616, -0.3471] | -0.3539, -0.3535, -0.3545 | 0.006 | worse |
| H1 sufficiency | H0 collapsed | -0.0666 | [-0.0724, -0.0617] | -0.0716, -0.0574, -0.0710 | 0.006 | worse |
| H2 necessity | H0 collapsed | -0.1806 | [-0.1876, -0.1733] | -0.1859, -0.1782, -0.1777 | 0.006 | worse |
| H2 sufficiency | H0 collapsed | -0.1613 | [-0.1681, -0.1547] | -0.1570, -0.1597, -0.1672 | 0.006 | worse |
| H3 necessity | H0 collapsed | -0.1818 | [-0.1876, -0.1753] | -0.1836, -0.1781, -0.1836 | 0.006 | worse |
| H3 sufficiency | H0 collapsed | +0.0100 | [+0.0059, +0.0145] | +0.0085, +0.0142, +0.0074 | 0.006 | better |

## Secondary: Strict pointing against H0 collapsed

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H0 collapsed | -0.0062 | [-0.0082, -0.0042] | -0.0057, -0.0059, -0.0071 | 0.006 | worse |
| H1 sufficiency | H0 collapsed | -0.0047 | [-0.0065, -0.0028] | -0.0043, -0.0040, -0.0059 | 0.006 | worse |
| H2 necessity | H0 collapsed | -0.1302 | [-0.1385, -0.1217] | -0.1239, -0.1344, -0.1322 | 0.006 | worse |
| H2 sufficiency | H0 collapsed | +0.0008 | [-0.0007, +0.0024] | +0.0003, +0.0009, +0.0012 | 0.32 | no difference |
| H3 necessity | H0 collapsed | -0.0129 | [-0.0154, -0.0105] | -0.0124, -0.0126, -0.0138 | 0.006 | worse |
| H3 sufficiency | H0 collapsed | +0.0013 | [-0.0001, +0.0027] | +0.0016, +0.0005, +0.0017 | 0.116 | no difference |

## Secondary: Best IoU against H0 collapsed

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H0 collapsed | -0.2755 | [-0.2790, -0.2722] | -0.2776, -0.2738, -0.2750 | 0.006 | worse |
| H1 sufficiency | H0 collapsed | -0.0475 | [-0.0500, -0.0452] | -0.0506, -0.0403, -0.0514 | 0.006 | worse |
| H2 necessity | H0 collapsed | -0.0930 | [-0.0964, -0.0889] | -0.0988, -0.0888, -0.0914 | 0.006 | worse |
| H2 sufficiency | H0 collapsed | -0.0422 | [-0.0446, -0.0398] | -0.0450, -0.0359, -0.0456 | 0.006 | worse |
| H3 necessity | H0 collapsed | -0.0617 | [-0.0641, -0.0593] | -0.0646, -0.0581, -0.0624 | 0.006 | worse |
| H3 sufficiency | H0 collapsed | +0.0194 | [+0.0179, +0.0208] | +0.0178, +0.0224, +0.0179 | 0.006 | better |

## Secondary: MaxBoxAccV2 against H0 CAM

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H0 CAM | -0.3585 | [-0.3661, -0.3513] | -0.3597, -0.3580, -0.3576 | 0.006 | worse |
| H1 sufficiency | H0 CAM | -0.0711 | [-0.0770, -0.0661] | -0.0774, -0.0618, -0.0741 | 0.006 | worse |
| H2 necessity | H0 CAM | -0.1851 | [-0.1920, -0.1778] | -0.1918, -0.1827, -0.1808 | 0.006 | worse |
| H2 sufficiency | H0 CAM | -0.1658 | [-0.1725, -0.1591] | -0.1628, -0.1642, -0.1703 | 0.006 | worse |
| H3 necessity | H0 CAM | -0.1862 | [-0.1922, -0.1798] | -0.1894, -0.1825, -0.1867 | 0.006 | worse |
| H3 sufficiency | H0 CAM | +0.0055 | [+0.0012, +0.0102] | +0.0027, +0.0097, +0.0043 | 0.018 | better |

## Secondary: Strict pointing against H0 CAM

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H0 CAM | -0.0065 | [-0.0085, -0.0045] | -0.0057, -0.0060, -0.0078 | 0.006 | worse |
| H1 sufficiency | H0 CAM | -0.0050 | [-0.0068, -0.0032] | -0.0043, -0.0041, -0.0066 | 0.006 | worse |
| H2 necessity | H0 CAM | -0.1305 | [-0.1387, -0.1220] | -0.1239, -0.1346, -0.1329 | 0.006 | worse |
| H2 sufficiency | H0 CAM | +0.0005 | [-0.0011, +0.0020] | +0.0003, +0.0007, +0.0005 | 0.516 | no difference |
| H3 necessity | H0 CAM | -0.0132 | [-0.0155, -0.0108] | -0.0124, -0.0128, -0.0145 | 0.006 | worse |
| H3 sufficiency | H0 CAM | +0.0010 | [-0.0003, +0.0024] | +0.0016, +0.0003, +0.0010 | 0.252 | no difference |

## Secondary: Best IoU against H0 CAM

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H0 CAM | -0.2773 | [-0.2807, -0.2738] | -0.2796, -0.2757, -0.2766 | 0.006 | worse |
| H1 sufficiency | H0 CAM | -0.0493 | [-0.0516, -0.0470] | -0.0526, -0.0422, -0.0530 | 0.006 | worse |
| H2 necessity | H0 CAM | -0.0948 | [-0.0983, -0.0907] | -0.1009, -0.0906, -0.0930 | 0.006 | worse |
| H2 sufficiency | H0 CAM | -0.0440 | [-0.0464, -0.0415] | -0.0471, -0.0377, -0.0471 | 0.006 | worse |
| H3 necessity | H0 CAM | -0.0635 | [-0.0661, -0.0609] | -0.0666, -0.0600, -0.0639 | 0.006 | worse |
| H3 sufficiency | H0 CAM | +0.0176 | [+0.0159, +0.0190] | +0.0158, +0.0206, +0.0164 | 0.006 | better |

## Secondary: Necessity vs sufficiency, MaxBoxAccV2

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H1 sufficiency | -0.2873 | [-0.2957, -0.2792] | -0.2824, -0.2962, -0.2835 | 0.003 | worse |
| H2 necessity | H2 sufficiency | -0.0193 | [-0.0270, -0.0111] | -0.0289, -0.0185, -0.0105 | 0.003 | worse |
| H3 necessity | H3 sufficiency | -0.1918 | [-0.1975, -0.1859] | -0.1922, -0.1922, -0.1909 | 0.003 | worse |

## Secondary: Necessity vs sufficiency, strict pointing

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H1 sufficiency | -0.0015 | [-0.0022, -0.0009] | -0.0014, -0.0019, -0.0012 | 0.003 | worse |
| H2 necessity | H2 sufficiency | -0.1310 | [-0.1391, -0.1230] | -0.1243, -0.1353, -0.1334 | 0.003 | worse |
| H3 necessity | H3 sufficiency | -0.0142 | [-0.0163, -0.0120] | -0.0140, -0.0131, -0.0155 | 0.003 | worse |

## Secondary: Necessity vs sufficiency, best IoU

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 necessity | H1 sufficiency | -0.2280 | [-0.2318, -0.2243] | -0.2269, -0.2335, -0.2236 | 0.003 | worse |
| H2 necessity | H2 sufficiency | -0.0509 | [-0.0544, -0.0464] | -0.0538, -0.0529, -0.0459 | 0.003 | worse |
| H3 necessity | H3 sufficiency | -0.0811 | [-0.0833, -0.0787] | -0.0824, -0.0806, -0.0803 | 0.003 | worse |

## Secondary: H0 collapsed vs CAM, MaxBoxAccV2

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H0 collapsed | H0 CAM | -0.0045 | [-0.0066, -0.0021] | -0.0058, -0.0045, -0.0031 | 0.001 | worse |

## Secondary: H0 collapsed vs CAM, strict pointing

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H0 collapsed | H0 CAM | -0.0003 | [-0.0009, +0.0003] | +0.0000, -0.0002, -0.0007 | 0.396 | no difference |

## Secondary: H0 collapsed vs CAM, best IoU

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H0 collapsed | H0 CAM | -0.0018 | [-0.0025, -0.0011] | -0.0020, -0.0018, -0.0015 | 0.001 | worse |

## Secondary: Top-1 against H0 (t interval over 3 seed-matched differences)

| A | B | difference | 95% interval | per seed | P (Holm) | verdict |
|---|---|---|---|---|---|---|
| H1 | H0 | -0.0584 | [-0.0638, -0.0530] | -0.0606, -0.0583, -0.0563 | 0.00136 | worse |
| H2 | H0 | -0.0063 | [-0.0159, +0.0034] | -0.0107, -0.0035, -0.0047 | 0.108 | no difference |
| H3 | H0 | -0.0051 | [-0.0081, -0.0021] | -0.0050, -0.0064, -0.0040 | 0.0364 | worse |

## Map checks (largest value over 3 images x 2 references x 3 seeds)

| head | necessity, map vs literal substitution | sufficiency, map vs literal substitution | literal necessity minus sufficiency (relative) | same, absolute |
|---|---|---|---|---|
| H0 | 1.8e-14 | 7.3e-15 | 2.0e-14 | 5.3e-15 |
| H1 | 8.7e-16 | 2.2e-15 | 9.8e-01 | 2.7e+01 |
| H2 | 1.2e-14 | 4.7e-15 | 1.2e+00 | 4.5e+00 |
| H3 | 1.2e-13 | 2.1e-14 | 8.8e-01 | 5.2e-01 |

Relative values are divided by the largest absolute map value of that image and reference.

## Departures from the protocol and readings of unspecified details

Departures: none in the recipe. Batch 32 without gradient accumulation (peak GPU memory 6.8 GiB). The protocol says the trained checkpoints will be archived; because of their size they are kept by the author and are available on request, and are not part of the archive.

Readings of details the protocol leaves open:
1. Evaluation resize: the whole image is resized to 448 x 448 (bilinear), no crop, as in the earlier localisation arms; boxes are rescaled to the 224 x 224 evaluation frame.
2. Cosine schedule stepped per iteration to zero, no warm-up; 187 iterations per epoch (last incomplete batch dropped), 5,610 in all.
3. Mixed precision covers the backbone; every head runs in float32. Weight decay applies to all parameters.
4. Gated attention: hidden size 256 with biases (scripts/heads.py `Attn`); H3 = LayerNorm (eps 1e-5, affine) after the mean, then linear.
5. Maps of H2 and H3 have no closed form: H2 by substitution of the cell-local attention logit and class projection (the head's only dependence on a cell), H3 by substitution in the pooled vector; both checked against the literal substitution loop through the head module. Checks: first 3 test images x first 2 references, every model.
6. Plain CAM = <w_c, x_t> without bias (earlier convention); it is min-max scaled, so the bias has no effect.
7. Bootstrap P value = 2 x the smaller tail share of the resampled differences at 0, floored at 1/1000 (the earlier MaxBoxAccV2 convention); the same image resamples serve every contrast. Secondary contrasts use the same bootstrap; top-1 uses a t interval over seeds.
8. Top-1 is that of the float32 network; the float64 head gives the same predictions up to the reported agreement.
9. cuDNN autotuning was on during training, so runs are not bitwise repeatable.
10. Interrupted runs: H0_s1, started at 21:45 on 10 November, was stopped by the operator after epoch 17 of 30 at about 22:43 (out/logs/train_H0_s1_interrupted_epoch17.log); a retry started at 22:43 was also stopped (its log was overwritten by the final run); no checkpoint or evaluation existed, and it was restarted from scratch with the same seed and recipe at 23:59 (times from out/logs/driver_part1.log and driver.log). The four seed-2 trainings were terminated by the operating system at 06:25 on 11 November (H0_s2 after 22 of 30 epochs, the others before their first epoch) together with the controlling process; no checkpoint or evaluation of them existed. They were restarted from scratch with the same seeds and recipe at 11:10; no seed-2 output was inspected before the restart.
