#!/bin/bash
# GPU2：E1 前半结束后（监视 python PID 2425805，带 cmdline 防误杀）
#       → 等 resnet50 标记 → r50 评估
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
MK=/mnt/mydisk/xinguan-brainai/runs/v2/marks

while [ -e /proc/2425805 ] && grep -q intervene_v2 /proc/2425805/cmdline 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%F %T')] E1(g2) finished" >> runs/v2/e2_r50_g2.log

while [ ! -f "$MK/train_resnet50.done" ]; do sleep 120; done

CUDA_VISIBLE_DEVICES=2 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2_r50 \
  --models resnet50_lr0.5_wd0.0002_s0,resnet50_lr0.5_wd5e-05_s0,resnet50_lr0.125_wd0.0002_s0,resnet50_lr0.125_wd5e-05_s0 \
  >> runs/v2/e2_r50_g2.log 2>&1
echo "[$(date '+%F %T')] E2 r50 eval done" >> runs/v2/e2_r50_g2.log
