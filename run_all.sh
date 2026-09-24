#!/bin/bash
# Phase 1: compare architectures with plain CE.  Phase 2: imbalance techniques on the best one.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
E=${EPOCHS:-20}
run() { local tag=$1; shift; [ -f results/$tag.json ] && { echo "skip $tag"; return; }
        $PY train.py --epochs $E --tag $tag "$@" > logs/$tag.log 2>&1 && tail -2 logs/$tag.log | head -1 || echo "FAILED $tag"; }

for m in early_fusion sscd bisrnet; do run A_${m}_ce --model $m --loss ce; done

BEST=$($PY -c "
import json
best = max(['early_fusion','sscd','bisrnet'],
           key=lambda m: max(h['SeK'] for h in json.load(open(f'results/A_{m}_ce.json'))['history']))
print(best)")
echo "best architecture by val SeK: $BEST" | tee results/best_arch.txt

run B_${BEST}_wce      --model $BEST --loss wce
run B_${BEST}_median   --model $BEST --loss median
run B_${BEST}_cb       --model $BEST --loss cb
run B_${BEST}_focal    --model $BEST --loss focal
run B_${BEST}_dice     --model $BEST --loss dice
run B_${BEST}_ohem     --model $BEST --loss ohem
run B_${BEST}_ce_rare  --model $BEST --loss ce --sampler rare
run B_${BEST}_combo    --model $BEST --loss combo
run B_${BEST}_combo_rare --model $BEST --loss combo --sampler rare
echo ALL_DONE
