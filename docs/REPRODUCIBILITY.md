# Reproducibility map

Every table and figure of the article, and the numbers of its text, are listed below with the script that computes them and the result file that holds them. Paths are relative to the repository root, and every script is run from there. Numbering follows the article (Tables 1-4, Figures 1-4, Sections 1-6) and the extended document [`docs/EXTENDED_METHODS_RESULTS_TABLES.pdf`](EXTENDED_METHODS_RESULTS_TABLES.pdf) (Notes S1 extended methods and S2 extended results, Tables S1-S19, Figures S1-S6). Tables S1-S19 are also the article's Supplementary Material, with the same numbering. The extended document refers to a code-and-data archive: that archive is this repository.

`python scripts/verify_numbers.py` re-derives the reported numbers from the stored results in a few seconds, without running a model. The remaining numbers are read from `analyses/*/out/*.json`, each written by the script of the same stem in the `scripts/` folder next to it, except where a different script is named below.

## Verdict codes in the code and in the result files

The structural check and the result files use upper-case codes. Two of them read against the article: the code `DEGENERATE` marks a **class-token** readout, and the code `IN-SCOPE` marks what the article calls **degenerate** (necessity equals sufficiency as distributed). The codes are kept because they are emitted by `scripts/heldout/frozen_rule.py`, whose SHA-256 is recorded in `outputs/heldout/h0_freeze.json`, and by the hash-recorded blocks of `scripts/scope/s4_decide.py`. For example, `"DEGENERATE": 142` in `analyses/check_census/out/c5_census.json` is the article's 142 class-token readouts.

```
verdict of the check (keys verdict, v1; operator-level variant v2a, v2b)
  IN-SCOPE         degenerate as distributed
  NORM-BLOCKED     normalisation-blocked (blocked only by a normalisation after the pool)
  NONLIN-BLOCKED   nonlinearity-blocked
  AGG-BLOCKED      aggregator-blocked (the aggregator is not a mean, dense map or token)
  DEGENERATE       class-token readout, not the article's "degenerate"
measured label (key gt), from double-precision certification
  ADDITIVE         certified additive (eps1, eps2, eps3 < 1e-9); expected for degenerate
  VACUOUS          vacuous readout: no other cell moves the logit; expected for class-token
  NORM-ONLY        additive once the normalisations after the pool are deleted
  HARD-BLOCKED     not additive even then; expected for nonlinearity- and aggregator-blocked
  AFFINE-SHIPPED   label of the development stage: the head is linear along the segment between
                   two inputs as distributed (NORM-ONLY and HARD-BLOCKED are its other values)
aggregator category: MEAN, DENSE, TOKEN (class token), OTHER
operation after the aggregator: AFFINE, NORM (normalisation), NONLINEAR
```

`scripts/check_model.py` and `scripts/check_conservative.py` return the article's terms directly. `verify_numbers.py` prints the count of `DEGENERATE` under "class-token". The certification residuals eps1, eps2 and eps3 are stored in `outputs/heldout/h1_*.json` under `shipped` as `maxNS`, `Rhalf` and `I2`.

## The census as returned, as audited and as measured

The census records in `analyses/check_census/out/c5_census.json` hold the check's verdicts as returned. The audit (`analyses/check_audit/`) found three kinds of wrong verdict there:

* the three CoaT checkpoints are returned as degenerate, but their head reduces a stack of three class tokens with `Conv1d(3, 1, 1)`, so they are class-token readouts (measured in `analyses/check_audit/out/a1_audit.json`),
* the two ConvNeXt-Zepto-RMS checkpoints are returned as nonlinearity-blocked, but their `SimpleNorm2d` is an RMS normalisation and they measure as normalisation-blocked (`a1_audit.json`),
* the eleven VOLO checkpoints, returned as class-token, share one head trace with VOLO-D1, which was measured not additive (`analyses/check_census/out/c3_misc.json`) because `forward_head` adds a maximum over the patch tokens, so they are aggregator-blocked, as the operator-level variant (key `v2b`) also returns.

The audited counts are in `analyses/check_audit/out/a5_census_full.json` and `.log` (`analyses/check_audit/scripts/a5_census_full.py`). The census was then certified entry by entry in float64 with random weights (the "Measured" column of Table 2, Notes S1.2 and S2.6) in `analyses/conservative_check_census/`: per-model records in `k1_run_timm_*.jsonl` and `k1_reinit_timm.jsonl`, tally in `k1_verdicts.json` (keys `census` and `census.with_reinit`). The audited verdicts agree with these measured labels on all 895 entries.

Label rule of the measured census (`analyses/conservative_check_census/scripts/k1_run.py`, function `certificate()`): each entry was certified twice in float64, on its own random-weight features and on a unit-scale N(0,1) feature map of the same shape. The stored label is the unit-scale label. It equals the own-feature label except on 22 entries, where the readout nearly vanishes on its own features (20 entries) or the features are near zero and the tail nonlinearity never leaves its linear regime (2 entries). Both labels are stored per entry (keys `label_own`, `label_unit` and `label`) and the 22 are listed under `census.own_vs_unit_label_differ` of `k1_verdicts.json`. `k1_report.py` labels zero-readout the 18 entries whose readout is identically zero in both modes (re-certified after re-drawing the zero-initialised weights: `k1_reinit_timm.jsonl`, `census.zero_readout`), and vacuous the PiT-distilled models, whose `forward_features` returns only their class and distillation tokens.

## The checks in detail

`scripts/check_conservative.py` prints the original check (`verdict_check` of `check_model.py`), the audited check (`verdict` of `check_model.py`) and the conservative check (`check_model_sound` of `analyses/conservative_check_census/scripts/k1lib.py`, imported unchanged), with the operation the verdict names. The conservative check wraps the audited module-level verdict and returns degenerate only when the five conditions of `protocols/K1_sound_check_protocol.md` all hold, and unknown otherwise. A blocked or class-token module-level verdict is returned as it is. The article lists these as four conditions (i)-(iv): condition (i) combines protocol conditions 1 (exactly one spatial reduction, a mean) and 4 (the reduced axes are the spatial axes, from tensor shapes), and (ii)-(iv) are protocol conditions 2, 3 and 5 (5: agreement of the module-level and operator-level verdicts). The names `check_model_sound` and `K1_sound_check_protocol.md` are kept because the protocol's SHA-256 is recorded. Both mean the conservative check. `certify()` runs `measure()` of `analyses/check_audit/scripts/check_audit_lib.py`, whose per-cell loop is `NS_literal` of `scripts/crosscheck/rtlib.py`, on the model's own features and on a unit-scale feature map, as in the measured census.

`scripts/check_model.py` alone gives the original and audited verdicts:

```bash
python scripts/check_model.py resnet50
```

```python
import sys, timm
sys.path.insert(0, "scripts")
from check_model import check_model
print(check_model(timm.create_model("convnext_tiny")))
# {'verdict': 'normalisation-blocked', 'verdict_check': 'normalisation-blocked',
#  'rules_fired': [], 'blocking_op': 'head.norm(LayerNorm2d)', 'raw_code': 'NORM-BLOCKED', ...}
```

`verdict_check` is the check's own verdict, computed by `scripts/heldout/frozen_rule.py` imported unchanged. `verdict` is the audited verdict, which applies three pattern rules derived from the audit (a non-linear reduction over the tokens in `forward_head`, as in VOLO, gives aggregator-blocked; a reduction over a stack of class tokens, as in CoaT, gives class-token; a normalisation module missing from the check's tables, as `SimpleNorm2d`, gives normalisation-blocked) and names every rule that changed the verdict in `rules_fired` and `warnings`. The output also holds `blocking_op`, `raw_code` (the frozen code), the aggregator, the labelled tail and notes that flag verdicts resting on the check's remaining blind spots. Its output on seven timm models is `analyses/check_audit/out/check_model_demo.log` (`analyses/check_audit/scripts/a6_check_model_demo.py`). The error envelope of the check on 32 constructed heads is `analyses/check_audit/out/a3_probes.md`, and the audited verdict on the same heads is `analyses/check_audit/out/a9_probe_audited.log`.

## Computing both maps for a head

`scripts/crosscheck/rtlib.py` holds an independently written literal estimator: `NS_literal(head, XA, XB, c)` substitutes one cell at a time and returns the necessity and the sufficiency map of class `c` (float64 numpy arrays), with z(a) and z(b). `head` maps a batch of cell matrices (B, T, C) to logits, and `XA` and `XB` are the (T, C) cell matrices of the image and of the reference. For a timm model:

```python
import sys, timm, torch
sys.path.insert(0, "scripts/crosscheck")
from rtlib import NS_literal

m = timm.create_model("convnext_tiny").eval().double()
with torch.no_grad():
    fa = m.forward_features(torch.randn(1, 3, 224, 224, dtype=torch.float64))   # image a
    fb = m.forward_features(torch.randn(1, 3, 224, 224, dtype=torch.float64))   # reference b
    c = int(m.forward_head(fa).argmax())
_, C, H, W = fa.shape                                      # NCHW (ResNet, ConvNeXt)
cells = lambda f: f[0].reshape(C, H * W).T                 # (T, C), T = H*W cells
head = lambda x: m.forward_head(x.transpose(1, 2).reshape(-1, C, H, W))
N, S, za, zb = NS_literal(head, cells(fa), cells(fb), c)
print(abs(N - S).max() / abs(N).max())   # 0 up to rounding when degenerate
```

For NHWC features (Swin: `fa.shape == (1, H, W, C)`) use `cells = lambda f: f[0].reshape(H * W, C)` and `head = lambda x: m.forward_head(x.reshape(-1, H, W, C))`. With random weights the ratio is of order 1e-13 or below for `resnet50` and `swin_tiny_patch4_window7_224` (degenerate) and of order 0.1 for `convnext_tiny` (normalisation-blocked).

## Main text

| Item | Script | Result file |
|---|---|---|
| Table 1, the structural check | `scripts/heldout/frozen_rule.py` (the check as fixed), `scripts/check_model.py`, `scripts/check_conservative.py` | no data |
| Table 2, development and held-out columns | `scripts/heldout/h1_run.py` | `outputs/heldout/h1_D.json`, `h1_A.json`, `h1_B.json` |
| Table 2, census, check column | `analyses/check_census/scripts/c5_census.py` | `analyses/check_census/out/c5_census.json` (key `v1`) |
| Table 2, census, measured column, agreement of the original and conservative checks | `analyses/conservative_check_census/scripts/k1_run.py`, tallied by `k1_report.py` | `analyses/conservative_check_census/out/k1_verdicts.json` (keys `arch148`, `census.with_reinit`; cost per model `census.secs_*`) |
| Table 3, NIH, distributed head | `scripts/medical/m4_stats.py` | `outputs/medical/m4_stats.json` |
| Table 3, NIH, head refitted | `scripts/medical/m10_condB.py` | `outputs/medical/m10_condB.json` |
| Table 3, NIH, trained here | `scripts/medical/m6_own.py` | `outputs/medical/m6_own.json` |
| Table 3, RSNA, distributed model | `scripts/medical/m11_rsna.py` | `outputs/medical/m11_rsna.json` |
| Table 3, RSNA, outside its training split and five cross-fitted models | `analyses/medical_baselines/scripts/t6_score.py` (models from `t2_rsna.py`) | `analyses/medical_baselines/out/t6_score.json` (`rsna_xrv.xrv_heldout_subset`, `rsna_disjoint`) |
| Table 3, ISIC, first model | `scripts/medical/m5_isic.py` | `outputs/medical/m5_isic.json` (key `prespecified.H1_lse8.0_1x1_hit`, `_dice` for ISIC) |
| Table 3, primaries recomputed in closed form | `analyses/medical_extra/scripts/t1_nih_cf.py`, `t1_rsna_cf.py`, `t1_isic_cf.py` | `analyses/medical_extra/out/t1_nih_cf.json`, `t1_rsna_cf.json`, `t1_isic_cf.json` |
| Table 3, ISIC, seeds 0-2 | `analyses/isic_three_seeds/scripts/k3_seed.py` | `analyses/isic_three_seeds/out/k3_seed<S>.json` (`primary.closed_form`) |
| Table 4, comparison with earlier studies | | no data |
| Figure 1, design of the study | the experiments named in its panels | as in Table 2 and Figures 2-4 |
| Figure 2(a), correlation of the two maps against the temperature, closed-form maps of one image | `scripts/probe7_andor.py`, `scripts/probe10_fan.py`, `scripts/probe13_e2e_fan.py`, `analyses/closed_form_maps/scripts/t3_andor_exact.py` | `outputs/probe10_fan_*.json`, `outputs/probe13_e2e_fan.json`, `analyses/closed_form_maps/out/t3_andor_exact.json` |
| Figure 2(b), separation and residual of seven readouts | `scripts/probe1_aggregator.py`, `analyses/theory_sweeps/scripts/t2_locus.py`, `t2b_interaction64.py` | `outputs/probe1_aggregator.json`, `analyses/theory_sweeps/out/t2_locus.json`, `t2b_interaction64.json` |
| Figure 2(c), distributed ResNet-50 weights behind log-sum-exp pooling | `scripts/probe29_fan200.py`, `analyses/closed_form_maps/scripts/t2_in1k_fan.py` | `outputs/probe29_fan200.json`, `analyses/closed_form_maps/out/t2_in1k_fan.json` |
| Figure 2(d), eight CAM-based methods over nine temperatures | `scripts/probe30_camfamily.py` | `outputs/probe30_camfamily.json` |
| Figure 3(a), held-out agreement, nearest-trace and majority baselines | `analyses/check_census/scripts/c1_trace.py`, `scripts/heldout/h1_run.py` | `analyses/check_census/out/c1_trace.json`, `outputs/heldout/h1_A.json`, `h1_B.json` |
| Figure 3(b), (d), the normalisation moved across the mean | `scripts/probe25_lnplace.py` (`scripts/run_p1_sweep.sh`) | `outputs/probe25_lnplace_*.json` |
| Figure 3(c), top-1 cost of the cheapest edit | `scripts/probe25_lnplace.py`, `analyses/check_census/scripts/c6_mcnemar.py`, `c7_summary.py`, `scripts/scope/s3_edits.py`, `scripts/heldout/h4_price.py` | `outputs/probe25_lnplace_*.json`, `analyses/check_census/out/c6_mcnemar.json`, `c7_summary.json`, `outputs/scope/s3_edits.json`, `outputs/heldout/h4_price.json` |
| Figure 4(a), primary localisation contrasts and plain CAM | `scripts/wsol/w3_stats.py` (contrast set in `scripts/wsol/primary_set.py`), `analyses/localisation_reanalysis/scripts/c3_t12_recount_cam.py` | `outputs/wsol/w3_stats.json`, `analyses/localisation_reanalysis/out/t1_t2_contrasts.json` |
| Figure 4(b), relocation-arm control | `scripts/wsol/w8_control_test.py` | `outputs/wsol/w8_control_test.json` |
| Figure 4(c), medical contrasts | as Table 3 | as Table 3 |
| Figure 4(d), NIH hit rate against the average annotation box | `analyses/medical_baselines/scripts/t6_score.py`, `t10_avgbox_mcnemar.py` | `analyses/medical_baselines/out/t6_score.json`, `t10_avgbox_mcnemar.json` |

Sections 4.1, 4.2 and 4.4-4.6 report numbers detailed in Note S2 and mapped below. Section 4.3 rests on Table 2, Tables S11 and S16 and Note S2.6.

## Extended document

| Item | Script | Result file |
|---|---|---|
| Table S1, verdict and measured label of the 148 architectures | `scripts/heldout/h1_run.py` | `outputs/heldout/h1_D.json` (development), `h1_A.json`, `h1_B.json` (held out) |
| Table S2, the 192 primary localisation contrasts | `scripts/wsol/w3_stats.py` (per-record arrays from `scripts/wsol/w2_run.py`) | `outputs/wsol/w3_stats.json` |
| Table S2, contrasts against plain CAM, collapsed map against plain CAM | `analyses/localisation_reanalysis/scripts/c3_t12_recount_cam.py` | `analyses/localisation_reanalysis/out/t1_t2_contrasts.json` (`task2.rows_vs_cam`, `task2.collapsed_minus_cam`) |
| Table S2, whole-image box of the necessity map on CUB (0.2509) | `scripts/wsol/w2_run.py` | `outputs/wsol/curve_cub_resnet50_N_lse.npy`, `curve_cub_vgg16gap_N_lse.npy` (release asset `localisation_iou_curves.zip`) |
| Tables S2, S3, S8, table rows | `analyses/extended_tables/scripts/s3_localisation.py` | `analyses/extended_tables/out/s3_localisation.json` |
| Table S4, NIH hit rates by finding | `scripts/medical/m4_stats.py`, `analyses/extended_tables/scripts/s1_nih_perfinding.py` | `outputs/medical/m4_stats.json` (`per_finding_hit`), `analyses/extended_tables/out/s1_nih_perfinding.json`; the 7x7 pooled box (0.371) in `analyses/medical_baselines/out/t6_score.json` (`nih_m4.comparators_hit.avgbox7`) |
| Table S5, reference sensitivity | `scripts/crosscheck/rt2_real.py` | `scripts/crosscheck/rt2_*.json` (the degenerate ordering under the `pre_*` keys) |
| Table S6, status of the analyses | | no data; the RSNA and ISIC primaries are fixed in the headers of `scripts/medical/m11_rsna.py` and `scripts/medical/m5_isic.py`, the protocols are in `protocols/` |
| Table S7, every cell of the medical grids | `analyses/extended_tables/scripts/s2_medical_grid.py` | `analyses/extended_tables/out/s2_medical_grid.json` |
| Table S8, PxAP on VOC masks and COCO | `scripts/wsol/w2_run.py` | `outputs/wsol/w2_voc12seg_<backbone>.json`, `w2_coco17_<backbone>.json` (`per_map.*.PxAP`) |
| Table S9, within-model necessity against sufficiency | `analyses/localisation_extra/scripts/l1_ns_contrasts.py` | `analyses/localisation_extra/out/l1_ns_contrasts.json`, `.log` |
| Table S10, control, context, mean-reference and 3x3-block rows of the localisation protocol | `analyses/localisation_extra/scripts/l2_protocol_rows.py` | `analyses/localisation_extra/out/l2_protocol_rows.json`, `.log` |
| Table S11, the check on 32 constructed heads | `analyses/check_audit/scripts/a3_probes.py` | `analyses/check_audit/out/a3_probes.json`, `.log`, `.md` |
| Table S12, audit of the census architectures whose head trace was never measured | `analyses/check_audit/scripts/a1_audit.py`, `a5_census_full.py` | `analyses/check_audit/out/a1_audit.json`, `a5_census_full.json` |
| Table S13, conjunctive sufficiency against the collapsed map in every medical arm | `analyses/medical_extra/scripts/t2_sminus8.py`, `t1_isic_cf.py` | `analyses/medical_extra/out/t2_sminus8.json`, `log_t2.txt`, `t1_isic_cf.json` |
| Table S14, experimental design | | no data |
| Table S15, the normalisation next to the pool in ten distributed models | `scripts/probe25_lnplace.py` (`scripts/run_p1_sweep.sh`), `analyses/check_census/scripts/c6_mcnemar.py` | `outputs/probe25_lnplace_*.json`, `analyses/check_census/out/c6_mcnemar.json` |
| Table S16, false degenerate verdicts, coverage and accuracy of the three checks | `analyses/conservative_check_census/scripts/k1_report.py`, from `k1_probes.py` (32 constructed heads), `k1_run.py` (148 architectures and census), `k1_tv.py`, `k1_tv_verify.py` (torchvision), `k1_whitelist.py` | `analyses/conservative_check_census/out/k1_verdicts.json`, `k1.log`, `analyses/conservative_check_census/RESULTS.md` (keys `probes`, `arch148`, `census`, `torchvision`) |
| Note S2.6, the measured census | `analyses/conservative_check_census/scripts/k1_report.py` | `analyses/conservative_check_census/out/k1_verdicts.json` (`census.measured_distribution`, `census.zero_readout`, `census.with_reinit`, `census.own_vs_unit_label_differ`, `arch148.random_weight_label_vs_stored`) |
| Table S17, channel-wise log-sum-exp and generalised-mean pooling against score-then-pool log-sum-exp | `analyses/conservative_check_census/scripts/k1_lse.py` | `analyses/conservative_check_census/out/k1_lse.json`, `k1_lse.log` |
| Table S18, heads trained end to end on CUB-200-2011 | `analyses/end_to_end_cub/scripts/k2_train.py` (`k2_driver.py`), `k2_eval.py` (`k2_eval_driver.py`), `k2_h1_exact.py`, `k2_stats.py` | `analyses/end_to_end_cub/out/logs/`, `eval_<head>_s<seed>.json`, `h1_exact_sep_s<seed>.json`, `analyses/end_to_end_cub/out/k2_contrasts.json`, `analyses/end_to_end_cub/RESULTS.md`, departures from the protocol in `analyses/end_to_end_cub/out/DEVIATIONS.md` |
| Table S18, post hoc natural maps of each head | `analyses/end_to_end_cub/post/scripts/k2_natural.py` (`k2_natural_driver.py`), `k2_natural_stats.py`, `k2_h3_pair.py` | `analyses/end_to_end_cub/post/out/k2_natural.json`, `k2_h3_pair.json`, `analyses/end_to_end_cub/post/RESULTS_natural.md` |
| Note S2.8, MaxBoxAccV2 on within-image ranks of every 14x14 map | `analyses/end_to_end_cub/post/scripts/k2_rank.py` | `analyses/end_to_end_cub/post/out/k2_rank.json`, `analyses/end_to_end_cub/post/RESULTS_rank.md` |
| Table S19, ISIC with three retrained classifiers | `analyses/isic_three_seeds/scripts/k3_seed.py`, `k3_results.py` | `analyses/isic_three_seeds/out/k3_seed<S>.json`, `k3_seed<S>_maps.npz` (every map and the per-cell class scores), `analyses/isic_three_seeds/out/k3_grid_all_seeds.json`, `analyses/isic_three_seeds/RESULTS.md` |
| Figures S1-S6 | | drawn from the result files of the items above (Figure S1: Imagenette heads of `scripts/pilot.py`) |

## Further numbers of the text

| Number | Script | Result file |
|---|---|---|
| rank correlations of reference-averaged maps and of Grad-CAM with them (Notes S1.7.2, S2.7, S2.8, Section 4.2) | `analyses/closed_form_maps/scripts/t2_in1k_fan.py`, `t4_medical_nih.py` | `analyses/closed_form_maps/out/t2_in1k_fan.json` (`spNS_bank8`, `spGN_bank8`, `spGS_bank8`, `_bank32`), `t4_medical_nih.json` |
| Grad-CAM under shift-equivariant aggregators (Section 4.2) | `analyses/theory_sweeps/scripts/t3_gradcam.py`, `t3b_gradcam_predclass.py`, `t8b_gradcam_rho_decomp.py` | `analyses/theory_sweeps/out/` |
| rank correlations of the normalisation lever (Note S2.5, Section 4.4) | `scripts/probe25_lnplace.py` | `outputs/probe25_lnplace_*.json` (key `spNS`) |
| the algebra and the estimator checks | `analyses/closed_form_maps/scripts/t1_algebra.py`, `t5_wsol_estimator.py` | `analyses/closed_form_maps/out/t1_algebra.json`, `t5_wsol_estimator.json` |
| faithfulness, deletion and insertion curves (Note S2.3, Section 4.1) | `analyses/closed_form_maps/scripts/t6_faithfulness_exact.py`; random-ordering baseline `analyses/theory_sweeps/scripts/t4_faithfulness.py`, `t4b_faithfulness_posT.py`, `t4c_faithfulness_posT2.py`, `t4_merge.py` | `analyses/closed_form_maps/out/t6_faithfulness_exact.json`, `analyses/theory_sweeps/out/t4_faithfulness_merged.json` |
| census as returned, random-weight verdicts, operator-level variant (Note S2.6, Table 2) | `analyses/check_census/scripts/c5_census.py`, `c5_summary.py` | `analyses/check_census/out/c5_census.json`, `c5_summary.log` (keys `v1`, `v2a`, `v2b`) |
| out-of-trace architectures (35 of 41) | `analyses/conservative_check_census/scripts/k1_out_of_trace.py` | `analyses/conservative_check_census/out/k1_out_of_trace.json` |
| repair price of the nonlinearity-blocked architectures (Note S2.6, Figure 3(c)) | `scripts/scope/s3_edits.py`, `scripts/heldout/h4_price.py` | `outputs/scope/s3_edits.json`, `outputs/heldout/h4_price.json` |
| separation of the aggregator arm against its net wins (Note S2.8) | `analyses/localisation_reanalysis/scripts/c3_t56_extract.py` | `analyses/localisation_reanalysis/out/t5_t6_archive.json` |
| participation ratio of the two maps (Notes S2.8, S2.9) | `analyses/check_audit/scripts/a10_participation.py` | `analyses/check_audit/out/a10_participation.json` |
| token control at the 3x3 unit (Note S2.8) | `analyses/localisation_extra/scripts/l10_control_b3.py` | `analyses/localisation_extra/out/l10_control_b3.json` |
| rank separation of eight aggregators, tie shares of max pooling (Note S2.2) | `analyses/localisation_extra/scripts/l7_rank_sep.py` | `analyses/localisation_extra/out/l7_rank_sep.json` |
| medical primaries in closed form, summary of the second hypothesis (Note S2.9, Section 4.6) | `analyses/medical_extra/scripts/t1_nih_cf.py`, `t1_rsna_cf.py`, `t1_isic_cf.py`, `t3_h2.py` | `analyses/medical_extra/out/t3_h2.json`, `log_t3.txt` |

For the eight hit cells of the first ISIC run, the "original" P stored in `t1_isic_cf.json` is the Wilcoxon P that `m5_isic.py` stored. The exact McNemar P of Table S7 is computed from it by `analyses/extended_tables/scripts/s2_medical_grid.py`, and `t3_h2.py` uses the same computation.

## Folders

* `scripts/scope/`: development of the check on 66 architectures, the edits that reach the degenerate point and their float64 re-certification.
* `scripts/heldout/`: the check as fixed (`frozen_rule.py`, its source record `h0_freeze.py`), the held-out evaluation (`h1_run.py`), diagnosis, class-token readouts and repair costs. `rule_v2.py` is the later operator-level variant whose verdict the evaluation run also records.
* `scripts/wsol/`: localisation on CUB-200-2011, PASCAL VOC 2012 and MS-COCO (`run_all.sh` runs the arms in order).
* `scripts/medical/`: the NIH ChestX-ray14, RSNA Pneumonia and ISIC arms.
* `scripts/crosscheck/`: an independently written float64 estimator (`rtlib.py`) and the reference-sensitivity sweep (`rt2_real.py`).
* `analyses/check_census/`: trace novelty, nearest-trace and majority baselines, independent normalisation labels, the census of 895 architectures, exact McNemar tests of the edit costs, the repair-cost summary.
* `analyses/check_audit/`: float64 measurement of every census head trace not validated before, 32 constructed heads that measure the error envelope of the check, census counts after the audit corrections, `check_model.py` on seven timm models.
* `analyses/conservative_check_census/`: the conservative check (`k1lib.py`, protocols `K1_sound_check_protocol.md` and `K1b_measured_census_addendum.md`), run with the original and audited checks on the constructed heads, the 148 architectures, the census and 75 torchvision classifiers; float64 certification of every census entry; channel-wise pooling before the linear layer (`k1_lse.py`).
* `analyses/theory_sweeps/`: softmax contrasts, locus of separation by region size and reference, Grad-CAM under shift-equivariant pooling, faithfulness, reference distances, a check of training from scratch. `out/heads/` holds the three small heads fitted by `t0_fit_heads.py`.
* `analyses/closed_form_maps/`: exact, cancellation-free necessity and sufficiency of log-sum-exp heads in float64 and high-precision arithmetic, and the sweeps recomputed with them.
* `analyses/localisation_reanalysis/`: plain-CAM comparator, image-level re-analysis, temperatures, a refitted log-sum-exp head (`run_gpu_all.sh` runs the GPU pass).
* `analyses/localisation_extra/`: within-model necessity against sufficiency, the protocol's control and mean-reference rows, clustered bootstrap, learnable aggregator runs, rank separation.
* `analyses/end_to_end_cub/`: ResNet-50 fine-tuned on CUB-200-2011 with four heads (average pooling, log-sum-exp, gated attention, average pooling with layer normalisation), three seeds each (protocol `K2_end_to_end_protocol.md`); `post/` holds the post hoc natural-map analysis.
* `analyses/medical_baselines/`: training overlap of the distributed models, cross-fitted RSNA models, matched halves, average-box baselines, resolution ceiling, and the exact McNemar tests of Figure 4(d).
* `analyses/medical_extra/`: closed-form recomputation of the medical primaries, conjunctive sufficiency in every arm, cluster bootstrap, the all-zero rule of Grad-CAM.
* `analyses/isic_three_seeds/`: the ISIC arm with three retrained ResNet-50 classifiers, maps in closed form and by forward difference (protocol `K3_isic_closed_form_protocol.md`).
* `analyses/extended_tables/`: per-finding NIH hit rates, the medical grid and the localisation rows of the extended tables, recomputed from the stored results.
* `protocols/`: the pre-specification of the localisation and medical primaries (`wsol_protocol.md`, `medical_protocol.md`) and the protocols of the conservative check, the end-to-end heads and the ISIC seeds with their SHA-256 digests and time stamps (`PROTOCOL_SHA256.txt`). These five files are byte-identical to the recorded ones, so the digests can be checked directly.

## Source record of the check

`outputs/heldout/h0_freeze.json` stores the SHA-256 of `scripts/heldout/frozen_rule.py` and of every block of `scripts/scope/slib.py` and `scripts/scope/s4_decide.py` that the frozen file copies. `frozen_rule.py` is byte-identical to the recorded file (Windows line endings; the recorded hash `sha256_frozen_file` is taken after converting them to LF). Its docstring names the development iteration of the check that it froze (the fifth).

```bash
python -c "import hashlib,json; d=open('scripts/heldout/frozen_rule.py','rb').read().replace(b'\r\n',b'\n'); print(hashlib.sha256(d).hexdigest()==json.load(open('outputs/heldout/h0_freeze.json'))['sha256_frozen_file'])"
```

## Environment

All experiments ran with Python 3.11.15, PyTorch 2.6.0 (CUDA 12.4) and timm 1.0.27 on Windows 11 with one NVIDIA RTX 4090. The medical scripts also use TorchXRayVision. `requirements.txt` pins the versions of every package the scripts import, and `scripts/requirements-check.txt` lists a minimal CPU-only environment for `check_model.py`, `check_conservative.py`, `heldout/frozen_rule.py` and `verify_numbers.py`.
