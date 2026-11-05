#!/bin/bash
# GPU1 主训练链：4 架构顺序训练，每批完成 touch 标记文件（评估链依赖标记）
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
MK=/mnt/mydisk/xinguan-brainai/runs/v2/marks
mkdir -p $MK

for ARCH in resnet34 resnet50 shufflenet_v2_x1_0 regnet_y_400mf; do
  echo "[$(date '+%F %T')] train $ARCH start" >> runs/v2/train_g1.log
  CUDA_VISIBLE_DEVICES=1 $PY code/train_grid.py \
    --arch $ARCH --epochs 200 --seeds 0 \
    --lr-grid 0.5 0.125 --wd-grid 0.0002 0.00005 \
    --outdir /mnt/mydisk/xinguan-brainai/runs/ckpt_grid \
    >> runs/v2/train_g1.log 2>&1
  touch $MK/train_${ARCH}.done
  echo "[$(date '+%F %T')] train $ARCH done -> mark" >> runs/v2/train_g1.log
done
echo "[$(date '+%F %T')] ALL training done" >> runs/v2/train_g1.log
