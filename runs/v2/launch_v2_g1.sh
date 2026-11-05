#!/bin/bash
# GPU1：训练 E2 跨架构 checkpoint（16 个，200 epoch，与 resnet18 网格同协议）
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
mkdir -p runs/v2
for ARCH in resnet34 resnet50 shufflenet_v2_x1_0 mobilenet_v3_small; do
  echo "[$(date '+%F %T')] train $ARCH start" >> runs/v2/train_g1.log
  CUDA_VISIBLE_DEVICES=1 $PY code/train_grid.py \
    --arch $ARCH --epochs 200 --seeds 0 \
    --lr-grid 0.5 0.125 --wd-grid 0.0002 0.00005 \
    --outdir /mnt/mydisk/xinguan-brainai/runs/ckpt_grid \
    >> runs/v2/train_g1.log 2>&1
  echo "[$(date '+%F %T')] train $ARCH done" >> runs/v2/train_g1.log
done
echo "[$(date '+%F %T')] E2 training ALL done" >> runs/v2/train_g1.log

# 训练完成后立即在 GPU1 启动 E3（训练期白化对照，8 个 resnet18）
CUDA_VISIBLE_DEVICES=1 $PY code/intervene_v2_paper.py --arm e3 \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e3 \
  >> runs/v2/e3_g1.log 2>&1
echo "[$(date '+%F %T')] E3 done" >> runs/v2/train_g1.log
