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
