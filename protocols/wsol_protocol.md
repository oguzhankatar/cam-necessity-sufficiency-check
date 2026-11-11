# Localisation protocol

Extract of the working protocol document of the localisation experiment: the section that fixed the readouts, maps, reference, metrics, statistics and the pre-specified question before any localisation outcome was computed. Reproduced unchanged; section references point to the original document and are not part of this repository. The primary contrast set is defined in scripts/wsol/primary_set.py.

## 2. Protocol (fixed before any localization number was computed)

**Models.** Per backbone, three readouts over the same frozen cell matrix
`X ∈ R^{T×C}` (`T` spatial cells, ViT prefix tokens dropped):

```
collapsed   z_c = <w_c, mean_t x_t> + b_c                        (GAP, degenerate)
edited-LSE  z_c = (1/tau) log (1/T) sum_t exp(tau (<w_c,x_t>+b_c)), tau = +1.0
edited-LN   z_c = <w_c, LN(mean_t x_t)> + b_c                    (ViT only)
```

At `tau -> 0` the LSE head is algebraically GAP + FC, so the two rows differ by
exactly one design choice.  The LSE row **reuses the collapsed row's weights
unchanged** (Condition A): the only difference between the two models is the
pooling operator.  The LN-relocation row refits the linear head, because
`LN(mean_t z_t)` is a different pooled vector from `mean_t LN(z_t)`; the
LayerNorm itself is the shipped one, untouched.

**What is refit.** The backbone is frozen in every row.  A linear head is fitted
on GAP-pooled cells, once per (dataset, backbone), because CUB / VOC / COCO are
not ImageNet label spaces.  Nothing else is trained.

**Maps.** `collapsed` (the GAP model's N-map, which equals its S-map by the
identity), `N_lse` / `S_lse`, `N_ln` / `S_ln`, each also at a 3×3-block
perturbation scale (`_b3`); the published references `cam` and `gradcam`; and the
trivial controls `center` (fixed Gaussian, never sees the image), `actnorm`
(‖x_t‖₂, class-independent) and `random`.

**Baseline.** `N` and `S` are counterfactual against a *reference distribution*:
a fixed bank of 8 real images drawn from the FIT split, averaged.  A baseline
whose pooled vector equals the image's (e.g. a spatial shuffle) is forbidden and
is used nowhere.  A second basis — the dataset-mean cell broadcast to every cell —
is run as a robustness row (§7).

**Perturbation scales.** Both are reported everywhere: 1×1 cells and 3×3 cell
blocks.  Both closed forms are cross-checked against `rtlib`'s literal loop (§3).

**Metrics.** MaxBoxAccV2 (Choe et al. CVPR 2020: max over 100 CAM thresholds of the
box-accuracy, averaged over IoU thresholds 0.3, 0.5, 0.7) for CUB; the pointing game
(argmax inside the ground-truth region, strict and with the standard 15-px
tolerance) for VOC and COCO; plus PxAP and peak mask IoU where masks exist.
Maps are bilinearly upsampled to 224×224, min-max normalised per image, and ground
truth is rescaled to the same 224×224 frame (the `wsolevaluation` resize-to-square
convention).  CNN input 448 px and ViT input 224 px both give a 14×14 cell grid.

**Statistics.** Bootstrap CIs (2000 resamples; 1000 for MaxBoxAccV2, which is
recomputed inside every resample because it is a max over thresholds).  Paired
tests: Wilcoxon signed-rank with rank-biserial effect size for continuous
per-image scores, exact McNemar for the pointing game.  Holm-Bonferroni within
each (dataset, backbone, metric) family of the 8 primary contrasts.

**The pre-specified question.** Does `N_lse` or `S_lse` (or `N_ln` / `S_ln`) match
ground truth better than the collapsed map, and does either beat the trivial
controls?  A null result is a valid outcome and is reported as such.
