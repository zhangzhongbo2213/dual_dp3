#!/bin/bash

# DP3-EndPose 训练脚本 (stack_bowls_two demo_randomized)
# 用法: bash train_endpose_bowls_randomized.sh [seed] [gpu_id]
# 例如: bash train_endpose_bowls_randomized.sh 0 0

set -e  # 遇到错误立即退出

cd /data/zzb/RoboTwin/policy/DP3

# 设置 Hugging Face 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 任务配置
TASK_NAME="stack_bowls_two"
TASK_CONFIG="demo_randomized"
NUM_EPISODES=100
SEED=${1:-0}  # 默认seed=0
GPU_ID=${2:-0}  # 默认GPU=0

echo "========================================="
echo "DP3-EndPose 训练启动"
echo "========================================="
echo "任务名称: $TASK_NAME"
echo "数据配置: $TASK_CONFIG"
echo "Episodes数量: $NUM_EPISODES"
echo "随机种子: $SEED"
echo "GPU ID: $GPU_ID"
echo "========================================="
echo ""

# 检查数据是否存在
DATA_PATH="./data/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}-endpose.zarr"
if [ ! -d "$DATA_PATH" ]; then
    echo "⚠️  数据文件不存在，开始处理数据..."
    echo "数据路径: $DATA_PATH"
    echo ""
    bash process_data_endpose.sh ${TASK_NAME} ${TASK_CONFIG} ${NUM_EPISODES}
    if [ $? -ne 0 ]; then
        echo "❌ 数据处理失败!"
        exit 1
    fi
    echo "✅ 数据处理完成"
    echo "========================================="
    echo ""
fi

echo "🚀 开始训练..."
echo "使用数据路径: $DATA_PATH"
echo ""

# 开始训练
bash train_endpose.sh ${TASK_NAME} ${TASK_CONFIG} ${NUM_EPISODES} ${SEED} ${GPU_ID}

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 训练完成!"
    echo "========================================="
    echo "Checkpoint位置: ./checkpoints/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}-endpose_${SEED}/"
else
    echo ""
    echo "========================================="
    echo "❌ 训练失败!"
    echo "========================================="
    exit 1
fi
