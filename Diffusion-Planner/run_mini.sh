#!/bin/bash
export CUDA_VISIBLE_DEVICES=0  # 只使用第一张显卡，防止报错
export HYDRA_FULL_ERROR=1

# ================= 配置路径 =================
export NUPLAN_DEVKIT_ROOT="/home/xzl/diffusion_planner_test/nuplan-devkit"
export NUPLAN_DATA_ROOT="/home/xzl/diffusion_planner_test/nuplan-devkit/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/home/xzl/diffusion_planner_test/nuplan-devkit/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/home/xzl/diffusion_planner_test/nuplan-devkit/nuplan/exp"

# 指向你的 Mini .db 文件夹
export NUPLAN_DB_FILES="/home/xzl/diffusion_planner_test/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/mini"

# ================= 关键修改 =================
# 1. 改为 mini 分割，专门用于测试
SPLIT="mini"

# 2. 挑战类型
CHALLENGE="closed_loop_nonreactive_agents"

# 3. 模型文件路径（已更新为 2026-03-01 最新训练结果，epoch40 loss=0.0728）
ARGS_FILE="/home/xzl/diffusion_planner_test/Diffusion-Planner/checkpoints/mini_retrain_final/training_log/diffusion-planner-training/2026-03-01-19:24:10/args.json"
CKPT_FILE="/home/xzl/diffusion_planner_test/Diffusion-Planner/checkpoints/mini_retrain_final/training_log/diffusion-planner-training/2026-03-01-19:24:10/model_epoch_40_trainloss_0.0728.pth"

PLANNER=diffusion_planner
BRANCH_NAME=mini_test_run

# 只跑 10 个场景（覆盖 mini.yaml 中的 limit_total_scenarios）
NUM_SCENARIOS=10

# ================= 启动命令 =================
echo "正在启动仿真..."
echo "使用模型: $CKPT_FILE"
echo "使用数据: $NUPLAN_DB_FILES"
echo "场景数量: $NUM_SCENARIOS"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=nuplan_challenge \
    scenario_builder.db_files=$NUPLAN_DB_FILES \
    scenario_filter=$SPLIT \
    scenario_filter.limit_total_scenarios=$NUM_SCENARIOS \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=sequential \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.5 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"