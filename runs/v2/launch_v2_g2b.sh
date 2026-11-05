#!/bin/bash
# GPU2 替换脚本：等原 E1 python(2425805) 结束 → resnet50 评估
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
CK=/mnt/mydisk/xinguan-brainai/runs/ckpt_grid

while kill -0 2425805 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] E1(g2 python) finished, start resnet50 eval wait" >> runs/v2/e2_r50_g2.log

for i in $(seq 1 288); do
  n=$(ls ${CK}/resnet50_lr*.pt 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && break
  sleep 300
done

CUDA_VISIBLE_DEVICES=2 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2_r50 \
  --models resnet50_lr0.5_wd0.0002_s0,resnet50_lr0.5_wd5e-05_s0,resnet50_lr0.125_wd0.0002_s0,resnet50_lr0.125_wd5e-05_s0 \
  >> runs/v2/e2_r50_g2.log 2>&1
echo "[$(date '+%F %T')] E2 resnet50 eval done" >> runs/v2/e2_r50_g2.log
