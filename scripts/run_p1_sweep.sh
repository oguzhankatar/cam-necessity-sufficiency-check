#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
# The interpreter this ran under; override with PY=... to use your own.
PY="${PY:-python}"
LOG=outputs/probe25_sweep.log
: > $LOG
for M in \
  vit_betwixt_patch16_reg4_gap_256.sbb_in1k \
  vit_medium_patch16_reg4_gap_256.sbb_in1k \
  vit_little_patch16_reg4_gap_256.sbb_in1k \
  vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  beitv2_base_patch16_224.in1k_ft_in1k \
  swin_tiny_patch4_window7_224.ms_in1k \
  convnext_tiny.fb_in1k \
  maxvit_tiny_tf_224.in1k \
  eva02_tiny_patch14_336.mim_in22k_ft_in1k \
  naflexvit_base_patch16_gap.e300_s576_in1k \
  ; do
  echo "########## $M" >> $LOG
  MODEL=$M NIMG=256 NEVAL=50000 "$PY" scripts/probe25_lnplace.py >> $LOG 2>&1
  echo "exit=$?" >> $LOG
done
echo "SWEEP DONE" >> $LOG
