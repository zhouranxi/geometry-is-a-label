#!/bin/bash
# GPU2：E1 eval 前 4 个 resnet18 → 完成后接 E2 评估 shard 0
cd /mnt/mydisk/xinguan-brainai
PY=~/venvs/dl/bin/python
M1234="resnet18_lr0.125_wd2.5e-05_s1,resnet18_lr1.0_wd2.5e-05_s0,resnet18_lr1.0_wd5e-05_s0,resnet18_lr0.125_wd0.0001_s0"
M5678="resnet18_lr0.5_wd5e-05_s2,resnet18_lr0.5_wd0.0002_s2,resnet18_lr0.5_wd0.0001_s1,resnet18_lr0.125_wd0.0002_s2"

CUDA_VISIBLE_DEVICES=2 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e1 \
  --models "$M1234" \
  >> runs/v2/e1_g2.log 2>&1
echo "[$(date '+%F %T')] E1 shard(g2) done" >> runs/v2/e1_g2.log

# 等 GPU1 的 E2 训练 checkpoint 全部落盘（train_g1.log 出现 ALL done 或 e3 已启动）
for i in $(seq 1 288); do   # 最多等 24h
  grep -q "E2 training ALL done" runs/v2/train_g1.log 2>/dev/null && break
  sleep 300
done

CUDA_VISIBLE_DEVICES=2 $PY code/intervene_v2_paper.py --arm eval \
  --out /mnt/mydisk/xinguan-brainai/runs/v2/e2 \
  --shard 0 --num-shards 2 \
  >> runs/v2/e2_g2.log 2>&1
echo "[$(date '+%F %T')] E2 shard0 done" >> runs/v2/e1_g2.log
