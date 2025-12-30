#!/bin/bash

# DP3-GNN-EndPose 训练脚本 (stack_bowls_two数据集)
# 用途: 使用已处理的数据训练DP3-GNN-EndPose模型，训练300轮
# 用法: bash train_gnn_endpose_bowls.sh [gpu_id]

set -e  # 遇到错误立即退出

cd /data/zzb/RoboTwin/policy/DP3

# 任务配置
TASK_NAME="stack_bowls_two"
TASK_CONFIG="demo_clean"
NUM_EPISODES=100
SEED=42
RESUME_EPOCH=0

# GPU设置（可通过参数传入，默认使用GPU 0）
GPU_ID=${1:-0}

echo "========================================="
echo "DP3-GNN-EndPose 训练启动"
echo "========================================="
echo "任务名称: $TASK_NAME"
echo "数据配置: $TASK_CONFIG"
echo "Episodes数量: $NUM_EPISODES"
echo "随机种子: $SEED"
echo "GPU ID: $GPU_ID"
echo "训练轮数: 300 epochs"
echo "========================================="
echo ""

# 检查数据是否存在
DATA_PATH="./scripts/data_gnn/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}-gnn-endpose.zarr"
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 错误: 数据文件不存在: $DATA_PATH"
    echo "请先运行数据处理脚本:"
    echo "  bash process_data_gnn_endpose.sh ${TASK_NAME} ${TASK_CONFIG} ${NUM_EPISODES}"
    exit 1
fi

echo "✅ 数据文件存在: $DATA_PATH"
echo ""

# 启动训练
echo "🚀 开始训练..."
bash scripts/train_policy_gnn_endpose.sh ${TASK_NAME} ${TASK_CONFIG} ${NUM_EPISODES} ${GPU_ID} ${RESUME_EPOCH}

echo ""
echo "========================================="
echo "训练完成"
echo "========================================="

