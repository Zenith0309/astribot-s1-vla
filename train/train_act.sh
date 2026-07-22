#!/usr/bin/env bash
# train_act.sh — 用 LeRobot 自带 ACT 实现训练（训练环境运行，非 ROS 容器）
# 用法: bash train/train_act.sh [数据集root] [输出目录]
set -e

DATA_ROOT=${1:-./datasets/astribot_pick_ball}
OUT_DIR=${2:-./outputs/train/act_pick_ball}
REPO_ID="local/astribot_pick_ball"

# 境内网络：模型 backbone(ResNet18) 权重从 HF 镜像下载
export HF_ENDPOINT=https://hf-mirror.com

# 单卡 >=12G 显存可跑；batch_size 按显存调（8 约需 10G）
# steps: 单任务 100-300 条数据，5w-10w 步通常足够；先 5w 看收敛曲线
lerobot-train \
  --policy.type=act \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATA_ROOT" \
  --output_dir="$OUT_DIR" \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=10000 \
  --wandb.enable=false

# 提示：
# - 若你的 lerobot 版本没有 lerobot-train 命令，改用:
#     python -m lerobot.scripts.train <同样的参数>
# - checkpoint 在 $OUT_DIR/checkpoints/<step>/pretrained_model/
#   评测时把该目录传给 eval/serve_policy.py --ckpt
