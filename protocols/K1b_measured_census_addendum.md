# K1b addendum: measured census and an out-of-library test (written before any K1/K1b result was seen)

Date written: 2026-11-10. Author-designated; not externally registered.

1. **Measured census.** Every one of the 895 classifiable timm 1.0.27 ImageNet-1k entries is certified directly with
   random weights: the float64 residuals eps1-eps3 of the manuscript (literal per-cell substitution at the head,
   three random input pairs, the second input of each pair supplying the reference), giving a measured label
   (additive, vacuous, normalisation-only, hard-blocked, class-token by inspection). Entries that cannot be run are
   listed with the reason. Outcomes: the measured census distribution; agreement of the original check, the audited
   check and the conservative check (K1) with the measured labels; false-degenerate counts; wall-clock cost per
   entry of measurement versus the structural checks.
2. **Out-of-library transfer.** The conservative check is run, unchanged, on every image classifier of torchvision
   0.21 with ImageNet weights (random weights suffice) and on the image-classification heads of a fixed list of
   HuggingFace transformers models available offline (if none are available offline, this part is reported as not
   run). Each is certified as in 1. Outcomes: false-degenerate count and coverage.
Reported whatever the result.
