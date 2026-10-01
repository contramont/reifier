#!/bin/bash
# usage: audit.sh <variant-name>   (runs both reference sets in parallel, waits)
S=/tmp/claude-1000/-home-ubuntu-eualethic-reifier/c682cc31-719f-4c8f-ba8f-3622acb8f7c2/scratchpad
A=$S/xof2/av/sparse-focus; R=$A/repo; v=$1; mod=${2:-sp}
cd $A
for ref in $S/xof/final/ref777_w6.pt $S/xof/av/combined-1/adv/ref_w6.pt; do
  tag=$(basename $ref .pt)
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/src:$R/experiments/xof_shrink:$A timeout 2400 $S/venv/bin/python $R/experiments/xof_shrink/audit/adv_check.py $ref $mod:$v > runs/a_${v}_$tag.json 2> runs/a_${v}_$tag.err &
done
wait
cat runs/a_${v}_ref777_w6.json runs/a_${v}_ref_w6.json
