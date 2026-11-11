#!/bin/sh
# runs c3_gpu_refit.py over the 12 (dataset, backbone) cells, sequentially
P=python
cd "$(dirname "$0")"
for BK in resnet50 vgg16gap vitb16; do
  for DS in cub voc12seg coco17 voc12; do
    if [ -f ../out/gpu/res_${DS}_${BK}.json ]; then echo "done $DS $BK"; continue; fi
    echo "##### $DS $BK"
    $P c3_gpu_refit.py $DS $BK > ../out/gpu/log_${DS}_${BK}.txt 2>&1
    tail -6 ../out/gpu/log_${DS}_${BK}.txt
  done
done
echo ALLDONE
