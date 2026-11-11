# K2 protocol: localisation of separable heads trained end to end (fixed before any K2 model is trained)

Date written: 2026-11-10 (before any K2 training run). Author-designated; not externally registered.

## Question
When the backbone is trained together with a head that can separate necessity from sufficiency, do the
separated maps localise annotated objects better than the collapsed map of an identically trained degenerate
(GAP-linear) model? This removes the two confounds of the earlier localisation arms: the head is not reused from
average pooling, and the separable heads are chosen so that their two maps can order the cells differently.

## Design
- Data: CUB-200-2011, official train (5,994) / test (5,794) split; boxes for evaluation only.
- Backbone: torchvision ResNet-50, ImageNet-1k initialisation (IMAGENET1K_V1), fine-tuned end to end at
  448 x 448 pixels (14 x 14 grid), identical recipe for every head: SGD, momentum 0.9, weight decay 1e-4,
  learning rate 0.01 for the head and 0.001 for the backbone, cosine schedule, 30 epochs, batch 32, mixed
  precision, random resized crop (scale 0.5-1) and horizontal flip; centre resize for evaluation.
- Heads (one class score per cell unless stated):
  - H0 degenerate: global average pooling + linear (the collapsed map is its necessity map).
  - H1 log-sum-exp pooling of the per-cell class scores, T = +1 (separable in magnitude; maps averaged over
    references can order cells differently).
  - H2 gated attention pooling (Ilse et al.) over the cells, then linear (separable, ordering can differ against
    one reference).
  - H3 global average pooling + layer normalisation + linear (normalisation-blocked; separable, ordering can
    differ).
- Seeds: 3 per head (0, 1, 2): 12 runs. Checkpoints are archived.
- Maps on the test split for the ground-truth class: necessity and sufficiency of each 1 x 1 cell, computed by
  direct substitution in float64 (closed form where available and checked against substitution), averaged over 8
  reference activations from the training split, as in the earlier localisation arms; plain CAM of H0.
- Reported for every model: top-1 accuracy; separation and rank separation (single reference and 8-reference
  average) of its two maps.

## Outcomes and analysis
- Metrics: MaxBoxAccV2 (thresholds 0.3/0.5/0.7), strict pointing game, best box IoU.
- **Primary family (6 contrasts):** for each separable head H1-H3 and each of its two maps (necessity,
  sufficiency), the paired difference in MaxBoxAccV2 against the collapsed map of H0, with seed-matched pairing,
  averaged over the 3 seeds; 95% percentile bootstrap over test images (1,000 resamples, seeds pooled by image);
  Holm correction over the 6 contrasts at alpha = 0.05.
- Secondary: the same contrasts on pointing and best IoU; against plain CAM of H0; within-model necessity vs
  sufficiency; top-1 accuracy differences (t interval over seeds).
- Every contrast is reported whatever its sign.
