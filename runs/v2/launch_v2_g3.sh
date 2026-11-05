#!/bin/bash
# GPU3：E1 eval 后 4 个 resnet18 → 完成后接 E2 评估 shard 1
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
M5678="resnet18_lr0.5_wd5e-05_s2,resnet18_lr0.5_wd0.0002_s2,resnet18_lr0.5_wd0.0001_s1,resnet18_lr0.125_wd0.0002_s2"

CUDA_VISIBLE_DEVICES=3 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e1 \
  --models "$M5678" \
  >> runs/v2/e1_g3.log 2>&1
echo "[$(date '+%F %T')] E1 shard(g3) done" >> runs/v2/e1_g3.log

for i in $(seq 1 288); do   # 最多等 24h
  grep -q "E2 training ALL done" runs/v2/train_g1.log 2>/dev/null && break
  sleep 300
done

CUDA_VISIBLE_DEVICES=3 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2 \
  --shard 1 --num-shards 2 \
  >> runs/v2/e2_g3.log 2>&1
echo "[$(date '+%F %T')] E2 shard1 done" >> runs/v2/e1_g3.log
