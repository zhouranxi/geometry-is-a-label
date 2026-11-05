#!/bin/bash
# GPU3：E1 后半结束后（监视 python PID 2426211）
#       → 等 shufflenet 标记 → sf 评估
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
MK=/mnt/mydisk/xinguan-brainai/runs/v2/marks

while [ -e /proc/2426211 ] && grep -q intervene_v2 /proc/2426211/cmdline 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%F %T')] E1(g3) finished" >> runs/v2/e2_sf_g3.log

while [ ! -f "$MK/train_shufflenet_v2_x1_0.done" ]; do sleep 120; done

CUDA_VISIBLE_DEVICES=3 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2_sf \
  --models shufflenet_v2_x1_0_lr0.5_wd0.0002_s0,shufflenet_v2_x1_0_lr0.5_wd5e-05_s0,shufflenet_v2_x1_0_lr0.125_wd0.0002_s0,shufflenet_v2_x1_0_lr0.125_wd5e-05_s0 \
  >> runs/v2/e2_sf_g3.log 2>&1
echo "[$(date '+%F %T')] E2 sf eval done" >> runs/v2/e2_sf_g3.log
