#!/bin/sh
# Stage 7-A: one clean pass.  A run is skipped only if its JSON already carries
# the CURRENT code_sha, so no stale result can survive a code change.
# The interpreter this ran under; override with P=... to use your own.
P="${P:-python}"
cd "$(dirname "$0")/../.." || exit 1
SHA=$("$P" -c "import sys;sys.path.insert(0,'scripts/wsol');import wlib;print(wlib.code_sha())")
echo "code_sha = $SHA"

run() {   # run <name.json> <args...>
  f="outputs/wsol/$1"; shift
  if [ -f "$f" ] && grep -q "\"$SHA\"" "$f"; then echo "   up to date: $f"; return; fi
  "$P" scripts/wsol/w2_run.py "$@" 2>&1 | tail -24
}

for DSX in cub voc12seg voc12 coco17; do
  for BK in resnet50 vgg16gap vitb16; do
    echo "##### PRIMARY $DSX $BK"
    run "w2_${DSX}_${BK}.json" "$DSX" "$BK"
  done
done

for DSX in cub voc12seg; do
  echo "##### BASIS $DSX resnet50 meancell"
  run "w2_${DSX}_resnet50_meancell.json" "$DSX" resnet50 0 meancell
done

for T in -8 -2 -0.5 0.5 2 8; do
  echo "##### TAU voc12seg resnet50 tau=$T"
  run "w2_voc12seg_resnet50_tau${T}.json" voc12seg resnet50 0 bank "$T"
done
for T in -8 -2 2 8; do
  echo "##### TAU cub resnet50 tau=$T"
  run "w2_cub_resnet50_tau${T}.json" cub resnet50 0 bank "$T"
done

echo "##### AGGREGATE"
"$P" scripts/wsol/w0_data.py   2>&1 | tail -6
"$P" scripts/wsol/w3_stats.py  2>&1 | tail -3
"$P" scripts/wsol/w9_report.py 2>&1 | tail -2
echo "##### DONE"
