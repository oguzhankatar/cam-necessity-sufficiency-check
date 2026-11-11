| ID | Constructed head | Frozen verdict (code) | eps1 | eps2 | eps3 | Measured | Agree | Operator-level variant (v2b) |
|---|---|---|---|---|---|---|---|---|
| P01 | mean as a module (AdaptiveAvgPool2d) -> Flatten -> Linear | degenerate (`IN-SCOPE`) | 1.9e-15 | 4.9e-15 | 1.4e-15 | degenerate | yes | degenerate (agrees) |
| P02 | mean as a tensor op x.mean((2,3)) -> Linear, no declared pool | degenerate (`IN-SCOPE`) | 1.9e-15 | 4.9e-15 | 1.4e-15 | degenerate | yes | degenerate (agrees) |
| P03 | max pool as a module (AdaptiveMaxPool2d) -> Linear | aggregator-blocked (`AGG-BLOCKED`) | 1.0e+00 | 1.1e+00 | 4.0e-02 | blocked (hard) | yes | aggregator-blocked (agrees) |
| P04 | max pool as timm SelectAdaptivePool2d(pool_type='max') -> Linear | aggregator-blocked (`AGG-BLOCKED`) | 1.0e+00 | 1.1e+00 | 4.0e-02 | blocked (hard) | yes | aggregator-blocked (agrees) |
| P05 | max as a tensor op x.amax((2,3)) -> Linear, no declared pool | degenerate (`IN-SCOPE`) | 1.0e+00 | 1.1e+00 | 4.0e-02 | blocked (hard) | **no** | aggregator-blocked (agrees) |
| P06 | max as a tensor op, model declares global_pool='max' | aggregator-blocked (`AGG-BLOCKED`) | 1.0e+00 | 1.1e+00 | 4.0e-02 | blocked (hard) | yes | aggregator-blocked (agrees) |
| P07 | attention pool module without MLP (timm AttentionPoolLatent, mlp=None) -> Linear; global_pool='map' | degenerate (`IN-SCOPE`) | 3.4e-02 | 1.0e-02 | 2.8e-03 | blocked (hard) | **no** | class-token (disagrees) |
| P08 | attention pool + LayerNorm + MLP (timm AttentionPoolLatent as in SigLIP 'map') -> Linear | nonlinearity-blocked (`NONLIN-BLOCKED`) | 3.1e-01 | 8.9e-02 | 1.7e-02 | blocked (hard) | yes (wrong cause) | class-token (disagrees) |
| P09 | attention pool written as tensor ops (Linear k, v; softmax over cells) -> Linear, no declared pool | degenerate (`IN-SCOPE`) | 2.5e-01 | 7.5e-02 | 1.6e-02 | blocked (hard) | **no** | degenerate (disagrees) |
| P10 | attention pool (tensor ops) followed by an MLP (Linear-GELU-Linear) | nonlinearity-blocked (`NONLIN-BLOCKED`) | 2.2e-01 | 1.0e-01 | 1.9e-02 | blocked (hard) | yes (wrong cause) | nonlinearity-blocked (agrees) |
| P11 | stack of 3 class tokens reduced by Conv1d(3,1,1) -> Linear (CoaT head) | degenerate (`IN-SCOPE`) | 3.7e-16 | 2.7e-16 | 9.1e-18 | class-token | **no** | degenerate (disagrees) |
| P12 | stack of 3 class tokens reduced by Linear(3,1) over the token axis -> Linear | degenerate (`IN-SCOPE`) | 7.4e-16 | 2.4e-16 | 1.8e-17 | class-token | **no** | degenerate (disagrees) |
| P13 | single class token x[:, 0] -> Linear, global_pool='token' | class-token (`DEGENERATE`) | 0.0e+00 | 0.0e+00 | 0.0e+00 | class-token | yes | class-token (agrees) |
| P14 | LayerNorm after the mean (AdaptiveAvgPool2d -> LayerNorm -> Linear) | normalisation-blocked (`NORM-BLOCKED`) | 6.1e-01 | 3.8e-01 | 1.1e-01 | normalisation-blocked | yes | normalisation-blocked (agrees) |
| P15 | LayerNorm before the mean, per cell (LayerNorm2d -> AdaptiveAvgPool2d -> Linear) | degenerate (`IN-SCOPE`) | 2.0e-15 | 2.0e-15 | 1.5e-15 | degenerate | yes | degenerate (agrees) |
| P16 | LayerNorm before the mean, per token, mean as a tensor op (LayerNorm -> x.mean(1) -> Linear) | normalisation-blocked (`NORM-BLOCKED`) | 1.8e-15 | 7.8e-15 | 1.3e-15 | degenerate | **no** | normalisation-blocked (disagrees) |
| P17 | RMSNorm after the mean (timm RmsNorm) | normalisation-blocked (`NORM-BLOCKED`) | 6.5e-01 | 3.7e-01 | 9.5e-02 | normalisation-blocked | yes | normalisation-blocked (agrees) |
| P18 | RMSNorm after the mean (torch nn.RMSNorm) | normalisation-blocked (`NORM-BLOCKED`) | 6.5e-01 | 3.7e-01 | 9.5e-02 | normalisation-blocked | yes | normalisation-blocked (agrees) |
| P19 | SimpleNorm2d after the mean (ConvNeXt-Zepto-RMS head: pool -> SimpleNorm2d -> Linear) | nonlinearity-blocked (`NONLIN-BLOCKED`) | 5.2e-01 | 3.7e-01 | 1.0e-01 | normalisation-blocked | **no** | nonlinearity-blocked (disagrees) |
| P20 | per-cell MLP before the mean (Conv1x1-GELU-Conv1x1 -> AdaptiveAvgPool2d -> Linear) | degenerate (`IN-SCOPE`) | 1.2e-14 | 6.0e-14 | 9.6e-15 | degenerate | yes | degenerate (agrees) |
| P21 | per-token MLP before a tensor-op mean (Linear-GELU-Linear -> x.mean(1) -> Linear) | nonlinearity-blocked (`NONLIN-BLOCKED`) | 1.2e-14 | 5.5e-14 | 1.0e-14 | degenerate | **no** | nonlinearity-blocked (disagrees) |
| P22 | GeM pooling as a module (p = 3) -> Linear | aggregator-blocked (`AGG-BLOCKED`) | 5.3e-01 | 2.5e-01 | 3.4e-02 | blocked (hard) | yes | aggregator-blocked (agrees) |
| P23 | GeM pooling as tensor ops -> Linear, no declared pool | degenerate (`IN-SCOPE`) | 5.3e-01 | 2.5e-01 | 3.4e-02 | blocked (hard) | **no** | degenerate (disagrees) |
| P24 | mixed avg+max concat as timm SelectAdaptivePool2d('catavgmax') -> Linear | aggregator-blocked (`AGG-BLOCKED`) | 6.8e-01 | 6.2e-01 | 5.1e-02 | blocked (hard) | yes | aggregator-blocked (agrees) |
| P25 | mixed avg+max concat as tensor ops -> Linear, no declared pool | degenerate (`IN-SCOPE`) | 6.8e-01 | 6.2e-01 | 5.1e-02 | blocked (hard) | **no** | aggregator-blocked (agrees) |
| P26 | per-cell class scores (Conv1x1) pooled by log-sum-exp over cells (tensor op), T = 1 | degenerate (`IN-SCOPE`) | 1.3e-01 | 6.8e-02 | 9.9e-03 | blocked (hard) | **no** | aggregator-blocked (agrees) |
| P27 | linear cell mixing before the mean (Conv3x3, no activation -> AdaptiveAvgPool2d -> Linear) | degenerate (`IN-SCOPE`) | 5.1e-15 | 1.3e-14 | 3.3e-15 | degenerate | yes | degenerate (agrees) |
| P28 | nonlinear cell mixing before the mean (Conv3x3 -> ReLU -> AdaptiveAvgPool2d -> Linear) | degenerate (`IN-SCOPE`) | 1.1e+00 | 8.8e-01 | 5.8e-02 | blocked (hard) | **no** | degenerate (disagrees) |
| P29 | self-attention over cells before the mean (nn.MultiheadAttention -> AdaptiveAvgPool2d -> Linear) | degenerate (`IN-SCOPE`) | 2.0e-01 | 9.0e-02 | 1.3e-02 | blocked (hard) | **no** | degenerate (disagrees) |
| P30 | MLP tail (AdaptiveAvgPool2d -> Linear -> GELU -> Linear) | nonlinearity-blocked (`NONLIN-BLOCKED`) | 2.0e-01 | 6.6e-02 | 1.7e-02 | blocked (hard) | yes | nonlinearity-blocked (agrees) |
| P31 | BatchNorm1d (inference) after the mean -> Linear | degenerate (`IN-SCOPE`) | 1.5e-14 | 5.6e-14 | 7.3e-15 | degenerate | yes | degenerate (agrees) |
| P32 | ReLU in the tail that is never crossed on the reachable set (Linear bias +50 -> ReLU -> Linear) | nonlinearity-blocked (`NONLIN-BLOCKED`) | 1.1e-12 | 2.0e-12 | 7.0e-13 | degenerate | **no** | nonlinearity-blocked (disagrees) |

Agreement (paper scoring): 18/32; with the blocking operation also correct: 16/32. False "degenerate": P05, P07, P09, P11, P12, P23, P25, P26, P28, P29. Missed degenerate: P16, P21, P32. Operator-level variant (v2b, not the frozen check): 20/32.
