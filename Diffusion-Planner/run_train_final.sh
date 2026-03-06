#!/bin/bash
# 1. 显卡设置
export CUDA_VISIBLE_DEVICES=0

# 2. 设置 wandb 为离线模式 (防止报错)
export WANDB_MODE=offline

# 3. 路径设置
# 训练数据文件夹
TRAIN_SET_PATH="/home/xzl/diffusion_planner_test/Diffusion-Planner/data/mini_train_set"
# 训练数据索引列表
TRAIN_SET_LIST_PATH="./mini_train.json"
# 模型保存路径 (注意这里是文件夹)
SAVE_DIR="checkpoints/mini_retrain_final"
# Anchor 文件路径
ANCHOR_NPZ="/home/xzl/diffusion_planner_test/vis_output/anchors.npz"

echo "========================================"
echo "开始在 Mini 数据集上训练 (Anchor Truncated Diffusion)..."
echo "数据目录: $TRAIN_SET_PATH"
echo "保存目录: $SAVE_DIR"
echo "Anchor: $ANCHOR_NPZ  t_start=0.5"
echo "========================================"

# 4. 启动指令 (参数名已根据报错日志修正)
# 注意：
# - 删除了不支持的 val_set 相关参数
# - 修正了 save_dir, train_epochs, learning_rate

python -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone train_predictor.py \
    --train_set $TRAIN_SET_PATH \
    --train_set_list $TRAIN_SET_LIST_PATH \
    --save_dir $SAVE_DIR \
    --batch_size 4 \
    --train_epochs 50 \
    --learning_rate 1e-4 \
    --device cuda \
    --use_wandb False \
    --t_start 0.5 \
    --anchor_npz_path $ANCHOR_NPZ