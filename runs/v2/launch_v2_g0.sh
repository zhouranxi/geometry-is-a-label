#!/bin/bash
# GPU0：E3（训练期白化，8 个 resnet18）→ resnet34 评估 → mobilenet 评估
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
CK=/mnt/mydisk/xinguan-brainai/runs/ckpt_grid

CUDA_VISIBLE_DEVICES=0 $PY code/intervene_v2_paper.py --arm e3 \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e3 \
  >> runs/v2/e3_g0.log 2>&1
echo "[$(date '+%F %T')] E3 done on g0" >> runs/v2/e3_g0.log

wait_arch () {  # $1=glob 前缀  $2=数量
  for i in $(seq 1 288); do
    n=$(ls ${CK}/$1*.pt 2>/dev/null | wc -l)
    [ "$n" -ge "$2" ] && return 0
    sleep 300
  done
}

wait_arch resnet34_lr 4
CUDA_VISIBLE_DEVICES=0 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2_r34 \
  --models resnet34_lr0.5_wd0.0002_s0,resnet34_lr0.5_wd5e-05_s0,resnet34_lr0.125_wd0.0002_s0,resnet34_lr0.125_wd5e-05_s0 \
  >> runs/v2/e2_r34_g0.log 2>&1
echo "[$(date '+%F %T')] E2 resnet34 eval done" >> runs/v2/e2_r34_g0.log

wait_arch mobilenet_v3_small_lr 4
CUDA_VISIBLE_DEVICES=0 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2_mnv3 \
  --models mobilenet_v3_small_lr0.5_wd0.0002_s0,mobilenet_v3_small_lr0.5_wd5e-05_s0,mobilenet_v3_small_lr0.125_wd0.0002_s0,mobilenet_v3_small_lr0.125_wd5e-05_s0 \
  >> runs/v2/e2_mnv3_g0.log 2>&1
echo "[$(date '+%F %T')] E2 mobilenet eval done" >> runs/v2/e2_mnv3_g0.log
