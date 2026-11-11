# K1 and K1b results

Verdicts: original (frozen rule), audited (frozen rule plus three pattern rules), conservative (K1, protocol as written), post-hoc variant (K1 plus two operator-level conditions; not in the protocol). False degenerate: degenerate where the float64 certificate is not additive. Coverage: share not unknown. Binary accuracy: "can the two maps differ?" over decided entries / over all entries.

## False degenerate verdicts, coverage, binary accuracy

| Set | Check | False degenerate | Coverage | Binary acc. (decided) | Binary acc. (all) |
|---|---|---|---|---|---|
| 32 constructed heads | original | 8 | 100.0% | 65.6% | 65.6% |
| 32 constructed heads | audited | 5 | 100.0% | 75.0% | 75.0% |
| 32 constructed heads | conservative | 1 | 78.1% | 84.0% | 65.6% |
| 32 constructed heads | post-hoc variant | 0 | 75.0% | 87.5% | 65.6% |
| 148 architectures (147 labelled) | original | 0 | 100.0% | 99.3% | 99.3% |
| 148 architectures (147 labelled) | audited | 0 | 100.0% | 100.0% | 100.0% |
| 148 architectures (147 labelled) | conservative | 0 | 100.0% | 100.0% | 100.0% |
| 148 architectures (147 labelled) | post-hoc variant | 0 | 99.3% | 100.0% | 99.3% |
| census, measured (877) | original | 0 | 100.0% | 98.7% | 98.7% |
| census, measured (877) | audited | 0 | 100.0% | 100.0% | 100.0% |
| census, measured (877) | conservative | 0 | 100.0% | 100.0% | 100.0% |
| census, measured (877) | post-hoc variant | 0 | 98.6% | 100.0% | 98.6% |
| torchvision 0.21 (75) | original | 0 | 100.0% | 100.0% | 100.0% |
| torchvision 0.21 (75) | audited | 0 | 100.0% | 100.0% | 100.0% |
| torchvision 0.21 (75) | conservative | 0 | 92.0% | 100.0% | 92.0% |
| torchvision 0.21 (75) | post-hoc variant | 0 | 86.7% | 100.0% | 86.7% |

- Constructed heads, false degenerate: original P05, P07, P09, P23, P25, P26, P28, P29; audited P07, P09, P23, P28, P29; conservative P23; post-hoc none.
- Constructed heads left unknown by the conservative check: P07, P09, P11, P12, P27, P28, P29.
- 148 architectures, false degenerate: original none; audited none; conservative none; post-hoc none.
- 148 architectures, conservative unknown: 0; errors: none.
- Census, conservative coverage over the 895 checked entries: 100.0% (0 unknown). Failed conditions among unknown: {}.
- Census, false degenerate: original 0; audited 0; conservative 0; post-hoc 0.
- Census, degenerate on vacuous heads: original 3; audited 0; conservative 0; post-hoc 0.
- torchvision, false degenerate: original 0; audited 0; conservative 0; post-hoc 0; conservative unknown: swin_b, swin_s, swin_t, swin_v2_b, swin_v2_s, swin_v2_t, vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14.
- torchvision with the zero-initialised ViT heads re-drawn (80 labelled): degenerate on a class-token (vacuous) readout original 5; audited 5; conservative 0; post-hoc 0; coverage original 100.0%; audited 100.0%; conservative 86.2%; post-hoc 81.2%; five-class agreement original 75; audited 75; conservative 69; post-hoc 65.
- Census entries left unknown by the post-hoc variant: mvitv2_base, mvitv2_large, mvitv2_small, mvitv2_tiny, vit_medium_patch16_gap_256, vit_medium_patch16_gap_384, vit_relpos_base_patch32_plus_rpn_256, vit_relpos_medium_patch16_rpn_224, vit_so150m_patch16_reg4_gap_256, vit_so150m_patch16_reg4_gap_384, vit_srelpos_medium_patch16_224, vit_srelpos_small_patch16_224.
- torchvision split versus torchvision forward: largest relative output difference 0.0 over 80 models.

## Measured census (random weights, float64)

- Certified with a structural label: 877 of 895 (18 zero readout); not certified: none.
- Labels: {'ADDITIVE': 518, 'NORM-ONLY': 98, 'HARD-BLOCKED': 127, 'VACUOUS': 134}.
- Agreement with the measured label (five classes; unknown counted as disagreement): original 861 (98.2%), audited 877 (100.0%), conservative 877 (100.0%), post-hoc 865 (98.6%).
- Zero readout at initialisation: 18 census entries (gmixer_24_224, gmlp_s16_224, hiera_base_224, hiera_base_plus_224, hiera_huge_224, hiera_large_224, hiera_small_224, hiera_small_abswin_256, hiera_tiny_224, mixer_b16_224, mixer_l16_224, resmlp_12_224, resmlp_24_224, resmlp_36_224, resmlp_big_24_224, sequencer2d_l, sequencer2d_m, sequencer2d_s); torchvision: vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14.
- With those entries re-certified after re-drawing the zero weights (labels {'gmixer_24_224': 'ADDITIVE', 'gmlp_s16_224': 'ADDITIVE', 'hiera_base_224': 'NORM-ONLY', 'hiera_base_plus_224': 'NORM-ONLY', 'hiera_huge_224': 'NORM-ONLY', 'hiera_large_224': 'NORM-ONLY', 'hiera_small_224': 'NORM-ONLY', 'hiera_small_abswin_256': 'NORM-ONLY', 'hiera_tiny_224': 'NORM-ONLY', 'mixer_b16_224': 'ADDITIVE', 'mixer_l16_224': 'ADDITIVE', 'resmlp_12_224': 'ADDITIVE', 'resmlp_24_224': 'ADDITIVE', 'resmlp_36_224': 'ADDITIVE', 'resmlp_big_24_224': 'ADDITIVE', 'sequencer2d_l': 'ADDITIVE', 'sequencer2d_m': 'ADDITIVE', 'sequencer2d_s': 'ADDITIVE'}): false degenerate original 0; audited 0; conservative 0; post-hoc 0; agreement original 879/895; audited 895/895; conservative 895/895; post-hoc 883/895.
- Own-feature and unit-scale labels differ on 22 entries: [('convmixer_1024_20_ks9_p14', 'HARD-BLOCKED', 'ADDITIVE'), ('convmixer_1536_20', 'HARD-BLOCKED', 'ADDITIVE'), ('convmixer_768_32', 'HARD-BLOCKED', 'ADDITIVE'), ('efficientnet_b1_pruned', 'HARD-BLOCKED', 'ADDITIVE'), ('efficientnet_b2_pruned', 'HARD-BLOCKED', 'ADDITIVE'), ('efficientnet_b3_pruned', 'HARD-BLOCKED', 'ADDITIVE'), ('ghostnetv2_130', 'ADDITIVE', 'HARD-BLOCKED'), ('inception_v4', 'HARD-BLOCKED', 'ADDITIVE'), ('mobilenetv4_hybrid_large', 'ADDITIVE', 'HARD-BLOCKED'), ('nasnetalarge', 'HARD-BLOCKED', 'ADDITIVE')].
- Original verdict re-run vs stored census record: 895 of 895 equal; differ: [].
- Entries of the 30 audited traces: 30; measured label equal to the audited label: 30; conservative false degenerate: none; unknown: 0.
- 148 architectures: random-weight label equals stored label on 139 of 147; differ: [('hiera_tiny_224', 'NORM-ONLY', 'ZERO-READOUT'), ('mixer_b16_224', 'ADDITIVE', 'ZERO-READOUT'), ('gmixer_24_224', 'ADDITIVE', 'ZERO-READOUT'), ('gmlp_s16_224', 'ADDITIVE', 'ZERO-READOUT'), ('resmlp_12_224', 'ADDITIVE', 'ZERO-READOUT'), ('hiera_small_224', 'NORM-ONLY', 'ZERO-READOUT'), ('hiera_base_224', 'NORM-ONLY', 'ZERO-READOUT'), ('sequencer2d_s', 'ADDITIVE', 'ZERO-READOUT')].

## Cost per model (CPU wall-clock)

- 148 architectures: original check 0.162 s median, 0.202 s mean; conservative check 0.165 s median, 0.205 s mean; certificate 2.684 s median, 4.398 s mean.
- census: original check 0.237 s median, 0.560 s mean; conservative check 0.253 s median, 0.569 s mean; certificate 3.412 s median, 7.301 s mean.
- torchvision: original check 0.271 s median, 0.315 s mean; conservative check 0.269 s median, 0.332 s mean; certificate 3.809 s median, 10.766 s mean.

## Channel-wise pooling followed by a linear layer (ResNet-50 layer4, 64 images, 49 cells)

| Head | T or p | Spearman(N, S) mean | min | literal loop mean | literal loop min |
|---|---|---|---|---|---|
| ch-LSE | 1 | 0.6489 | 0.0355 | 0.6489 | 0.0355 |
| ch-LSE | 4 | 0.3421 | -0.3621 | 0.3421 | -0.3621 |
| ch-LSE | 8 | 0.3262 | -0.3960 | 0.3262 | -0.3960 |
| ch-GeM | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ch-GeM | 3 | 0.6540 | -0.0855 | 0.6540 | -0.0855 |
| ch-GeM | 4 | 0.5516 | -0.2334 | 0.5516 | -0.2334 |
| ch-GeM | 8 | 0.4004 | -0.3177 | 0.4004 | -0.3177 |
| stp-LSE | 1 | 1.0000 | 1.0000 | 0.9365 | 0.5254 |
| stp-LSE | 4 | 1.0000 | 1.0000 | 0.5141 | 0.2449 |
| stp-LSE | 8 | 1.0000 | 1.0000 | 0.3664 | 0.2449 |

ch-LSE: channel-wise log-sum-exp then the fc layer; ch-GeM: channel-wise generalised mean then the fc layer; stp-LSE: per-cell class score then log-sum-exp (closed form: Spearman = 1).

## Deviations

- Condition 1 is read as "the spatial reduction is the only one": in both branches the operator trace must hold exactly one operator that removes the spatial axes, and in the module branch it must run inside the aggregator module.
- Condition 3 covers non-leaf modules too: one that completes before the aggregator is accepted only when it runs no operator of its own (plain nn.Sequential, the only container class in the development set); its leaf children are checked against the per-cell whitelist.
- Whitelists were taken from the module classes of the whole 66 development models (not only their heads), filtered to the protocol categories (k1_whitelist.json).
- The protocol-as-written check returned degenerate on constructed head P23 (generalised mean written as tensor operations, measured hard-blocked): the target of zero false degenerate verdicts is not met on the constructed heads. A post-hoc variant with two operator-level conditions (only per-cell operators before the reduction, only affine operators after it) is reported separately; it is not part of the protocol.
- False degenerate = degenerate returned where the certificate says not additive (normalisation-only or hard-blocked); degenerate on a vacuous (additive, zero spatial mass) head is listed separately.
- Binary accuracy of the conservative check is given over decided entries and over all entries (unknown counted wrong).
- The 148 architectures are scored against their stored labels (pretrained weights, real images); the checks and the certificate were run here on random weights.
- All checks and certificates ran on the CPU (the shared GPU slowed the per-cell loop); costs are CPU wall-clock, with 4 to 6 threads per process and several processes in parallel.
- Measured census label: the label on the model's own random-weight features, replaced by the label on a unit-scale N(0,1) feature map where the two differ (rule of analyses/check_audit/scripts/a1_audit.py).
- Entries whose readout is identically zero at initialisation (classifier weights initialised to zero; |N| = 0 on every cell) carry no structural information at random weights; they are labelled zero-readout and left out of the measured comparisons. As a supplement they were re-certified after drawing the all-zero weight tensors from N(0, 0.02^2), and the comparisons are repeated with those labels.
- Class-token by inspection: the four PiT-distilled models return only their class and distillation tokens from forward_features (no spatial cell reaches the head); the certificate on those two tokens reads additive, and they are labelled vacuous (class-token).
- ConViT (3 models): the certificate uses a model converted to float64 before its first forward (ConViT caches float32 buffers on its first forward); the two CrossViT token sequences of unequal width were zero-padded to a common width in the cell matrix; for the six mixed-RoPE ViTs, which force float32 inside the trunk, the trunk of the certificate ran in float32 and the head and all residuals in float64.
- torchvision models were wrapped with a forward_features / forward_head split at the last trunk feature map (k1_tv.py); GoogLeNet and Inception-v3 were built without auxiliary heads, which are unused in evaluation.
- HuggingFace part not run: the transformers library is not installed in the environment and nothing was downloaded.
- Channel-wise pooling check: Spearman correlations are computed from the exact one-cell change evaluated without subtracting two logits; the literal per-cell loop agrees to within 1e-5 of the map scale but rounds changes below float64 resolution to zero (ties), which lowers its score-then-pool value below 1; both are reported.

frozen_rule.py SHA-256 (LF-normalised) b4caf08e85ee7049ab39acd5dbf40dfa85b8d15964dd5442cbcecf05188893e0, equal to the recorded hash: True.
