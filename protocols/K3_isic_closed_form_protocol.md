# K3 protocol: reproducible ISIC arm in closed form (fixed before any K3 model is trained)

Date written: 2026-11-10. Author-designated; not externally registered.

## Question
Does the pre-specified ISIC primary contrast (H1: necessity map at T = +8 minus collapsed map, 1 x 1 cells, Dice)
hold when the arm is re-run end to end with archived models and closed-form maps?

## Design
- Same data, split, preprocessing, model (ResNet-50, ImageNet initialisation, class-weighted cross-entropy,
  12 epochs) and grid as the original ISIC arm (`scripts/medical/m5_isic.py`), with three seeds (0, 1, 2);
  every checkpoint and every per-item map is archived.
- Maps: necessity and sufficiency of the log-sum-exp head at T = +-8 (head reused, as pre-specified) computed in
  closed form (Eq. 2 of the manuscript) in float64, averaged over the same eight references; collapsed map;
  forward-difference maps computed as well for comparison with the original estimator.
- Primary: H1 at T = +8, 1 x 1, Dice (two-fold cross-tuned top-q threshold), per seed, with 95% bootstrap
  intervals (10,000 resamples) and the Wilcoxon signed-rank test; the full pre-specified grid (H1, H2; T = +-8;
  1 x 1 and 3 x 3; hit, AP, Dice) as secondary.
- Reported whatever the result, beside the original first-run value (-0.136).
