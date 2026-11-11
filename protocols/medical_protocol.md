# Medical protocol

Extract of the working protocol document of the medical experiment: the hypothesis and the pre-specification, fixed before any medical outcome was computed. Reproduced unchanged; the outcome is reported in the article and in outputs/medical/.

**Hypothesis (H).** Saporta et al., *Benchmarking saliency methods for chest X-ray
interpretation*, Nature Machine Intelligence **4**, 867–878 (2022) and Arun et al.,
Radiology: AI **3**(6), e200267 (2021) report that saliency maps from DenseNet-121-class
CXR classifiers underperform a human radiologist benchmark at localisation. Those models
are GAP + linear heads — this project's degenerate point, where the necessity map and
the sufficiency map are *the same vector*. Stages 3–4 measured that the CAM family
tracks the **sufficiency** side once the model is moved off that point. H says the
radiologist annotation marks **necessary** evidence, so the published negative result
would have a structural explanation: that model class cannot express the quantity the
annotation encodes.

Operationally: `score(N-map, GT) > score(collapsed map, GT)` **and**
`score(N-map, GT) > score(S-map, GT)`.

### Pre-specified before any medical number existed

| item | choice |
|---|---|
| primary dataset | NIH ChestX-ray14, the images with radiologist-drawn boxes for 8 findings |
| primary model | DenseNet-121 (the Saporta / Arun model class), **obtained** — torchxrayvision `densenet121-res224-nih` |
| degenerate config | `fc(GAP(relu(features)))`, as shipped |
| off-degenerate edit | aggregator only: `mean_t -> LSE_T`, weights untouched (condition A) |
| primary temperature | **T = +8** (stage 3's OR cell, the largest measured N/S dissociation); T = −8 secondary |
| baseline | bank of 8 "No Finding" reference images without boxes; N and S share the baseline |
| perturbation units | **1×1 cells and 3×3 blocks**, both reported |
| primary metric | **pointing-game hit rate** (argmax of the bilinearly upsampled map inside the GT) |
| secondary metrics | IoU at a 2-fold cross-tuned top-q threshold; threshold-free pixel average precision |
| decisive test | exact McNemar (binary hit) + Wilcoxon signed-rank + 10 000-sample bootstrap CI of the paired difference |
| comparators | collapsed map, S-map, Grad-CAM on both models, centre prior, activation-norm map, random |
| certification | `scripts/crosscheck/rtlib.py` literal float64 loop: ADDITIVE ⇔ maxNS, Rhalf, I2 < 1e−9. `corrNS` recorded, never a criterion. Scope verdict from `scripts/heldout/frozen_rule.py`, on trained weights and real images only |
