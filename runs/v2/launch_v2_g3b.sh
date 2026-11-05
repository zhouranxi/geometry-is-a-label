#!/bin/bash
# GPU3 替换脚本：等原 E1 python(2426211) 结束 → shufflenet 评估
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
CK=/mnt/mydisk/xinguan-brainai/runs/ckpt_grid

while kill -0 2426211 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] E1(g3 python) finished, start shufflenet eval wait" >> runs/v2/e2_sf_g3.log

for i in $(seq 1 288); do
  n=$(ls ${CK}/shufflenet_v2_x1_0_lr*.pt 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && break
  sleep 300
done

CUDA_VISIBLE_DEVICES=3 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2_sf \
  --models shufflenet_v2_x1_0_lr0.5_wd0.0002_s0,shufflenet_v2_x1_0_lr0.5_wd5e-05_s0,shufflenet_v2_x1_0_lr0.125_wd0.0002_s0,shufflenet_v2_x1_0_lr0.125_wd5e-05_s0 \
  >> runs/v2/e2_sf_g3.log 2>&1
echo "[$(date '+%F %T')] E2 shufflenet eval done" >> runs/v2/e2_sf_g3.log
