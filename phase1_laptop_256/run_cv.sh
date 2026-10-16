#!/bin/bash
# 3-fold cross-validation of every configuration on train+val (the test split stays fixed).
# Each epoch also scores a fixed train subset, for over/underfitting curves.
# Finished folds are skipped, so re-running resumes. Most important configs run first.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
E=${EPOCHS:-20}
K=3
run() { local tag=$1; shift
        for f in $(seq 0 $((K - 1))); do
          [ -f results/cv/${tag}_f$f.json ] && { echo "skip ${tag}_f$f"; continue; }
          $PY train.py --epochs $E --folds $K --fold $f --track-train 1 --tag ${tag}_f$f "$@" \
              > logs/cv_${tag}_f$f.log 2>&1 && tail -2 logs/cv_${tag}_f$f.log | head -1 \
              || echo "FAILED ${tag}_f$f"
        done; }

run A_early_fusion_ce --model early_fusion --loss ce
run A_sscd_ce         --model sscd --loss ce
run A_bisrnet_ce      --model bisrnet --loss ce
run B_bisrnet_combo   --model bisrnet --loss combo
run B_bisrnet_dice    --model bisrnet --loss dice
run B_bisrnet_wce     --model bisrnet --loss wce
run B_bisrnet_median  --model bisrnet --loss median
run B_bisrnet_cb      --model bisrnet --loss cb
run B_bisrnet_focal   --model bisrnet --loss focal
run B_bisrnet_ce_rare --model bisrnet --loss ce --sampler rare
run B_bisrnet_combo_rare --model bisrnet --loss combo --sampler rare
run B_bisrnet_ohem    --model bisrnet --loss ohem
echo CV_DONE
