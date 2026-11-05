#!/bin/bash
# GPU0：E3（显式 8 个 resnet18）→ 等 resnet34 标记 → r34 评估
#       → 等 regnet 标记 → regnet 评估
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
MK=/mnt/mydisk/xinguan-brainai/runs/v2/marks
M8="resnet18_lr0.125_wd2.5e-05_s1,resnet18_lr1.0_wd2.5e-05_s0,resnet18_lr1.0_wd5e-05_s0,resnet18_lr0.125_wd0.0001_s0,resnet18_lr0.5_wd5e-05_s2,resnet18_lr0.5_wd0.0002_s2,resnet18_lr0.5_wd0.0001_s1,resnet18_lr0.125_wd0.0002_s2"

CUDA_VISIBLE_DEVICES=0 $PY code/intervene_v2_paper.py --arm e3 \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e3 \
  --models "$M8" \
  >> runs/v2/e3_g0.log 2>&1
echo "[$(date '+%F %T')] E3 done on g0" >> runs/v2/e3_g0.log

wait_mark () {
  while [ ! -f "$MK/train_$1.done" ]; do sleep 120; done
}

wait_mark resnet34
CUDA_VISIBLE_DEVICES=0 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2_r34 \
  --models resnet34_lr0.5_wd0.0002_s0,resnet34_lr0.5_wd5e-05_s0,resnet34_lr0.125_wd0.0002_s0,resnet34_lr0.125_wd5e-05_s0 \
  >> runs/v2/e2_r34_g0.log 2>&1
echo "[$(date '+%F %T')] E2 r34 eval done" >> runs/v2/e2_r34_g0.log

wait_mark regnet_y_400mf
CUDA_VISIBLE_DEVICES=0 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2_reg \
  --models regnet_y_400mf_lr0.5_wd0.0002_s0,regnet_y_400mf_lr0.5_wd5e-05_s0,regnet_y_400mf_lr0.125_wd0.0002_s0,regnet_y_400mf_lr0.125_wd5e-05_s0 \
  >> runs/v2/e2_reg_g0.log 2>&1
echo "[$(date '+%F %T')] E2 regnet eval done" >> runs/v2/e2_reg_g0.log
