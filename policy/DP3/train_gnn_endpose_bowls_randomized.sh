#!/bin/bash

# DP3-GNN-EndPose 训练脚本 (stack_bowls_two demo_randomized)
# 用法: bash train_gnn_endpose_bowls_randomized.sh [gpu_id] [resume_epoch]
# 例如: bash train_gnn_endpose_bowls_randomized.sh 0 0

set -e  # 遇到错误立即退出

cd /data/zzb/RoboTwin/policy/DP3

# 设置 Hugging Face 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 任务配置
TASK_NAME="stack_bowls_two"
TASK_CONFIG="demo_randomized"
NUM_EPISODES=100
SEED=42
RESUME_EPOCH=${2:-0}  # 默认从epoch 0开始

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
echo "恢复训练: Epoch $RESUME_EPOCH"
echo "========================================="
echo ""

# 检查数据是否存在（可能有多个可能的路径）
DATA_PATH1="./scripts/data_gnn/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}-gnn-endpose.zarr"
DATA_PATH2="./scripts/data_processed/${TASK_NAME}/${TASK_CONFIG}/gnn_endpose/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}-gnn-endpose.zarr"

if [ -d "$DATA_PATH2" ]; then
    DATA_PATH="$DATA_PATH2"
    echo "✅ 数据文件存在: $DATA_PATH"
elif [ -d "$DATA_PATH1" ]; then
    DATA_PATH="$DATA_PATH1"
    echo "✅ 数据文件存在: $DATA_PATH"
else
    echo "⚠️  数据文件不存在，开始处理数据..."
    echo "路径1: $DATA_PATH1"
    echo "路径2: $DATA_PATH2"
    echo ""
    bash process_data_gnn_endpose.sh ${TASK_NAME} ${TASK_CONFIG} ${NUM_EPISODES}
    if [ $? -ne 0 ]; then
        echo "❌ 数据处理失败!"
        exit 1
    fi
    echo "✅ 数据处理完成"
    # 再次检查数据路径
    if [ -d "$DATA_PATH2" ]; then
        DATA_PATH="$DATA_PATH2"
    elif [ -d "$DATA_PATH1" ]; then
        DATA_PATH="$DATA_PATH1"
    else
        echo "❌ 数据处理后仍找不到数据文件!"
        exit 1
    fi
    echo "========================================="
fi

echo ""
echo "🚀 开始训练..."
echo "使用数据路径: $DATA_PATH"
echo ""

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU_ID
export HYDRA_FULL_ERROR=1

# 进入3D-Diffusion-Policy目录
cd 3D-Diffusion-Policy

# 获取DP3根目录的绝对路径
DP3_DIR="$(cd .. && pwd)"

# 构建绝对路径
if [[ "$DATA_PATH" == ./* ]]; then
    # 相对路径，去掉开头的 ./
    REL_PATH="${DATA_PATH#./}"
    ZARR_PATH="$DP3_DIR/$REL_PATH"
else
    # 已经是绝对路径或相对路径
    if [[ "$DATA_PATH" == /* ]]; then
        ZARR_PATH="$DATA_PATH"
    else
        ZARR_PATH="$DP3_DIR/$DATA_PATH"
    fi
fi

echo "训练数据路径: $ZARR_PATH"
echo ""

# 运行训练
python train_dp3.py \
    --config-name=robot_dp3_gnn_endpose.yaml \
    task_name=$TASK_NAME \
    setting=$TASK_CONFIG \
    expert_data_num=$NUM_EPISODES \
    task.dataset.zarr_path="$ZARR_PATH" \
    training.device="cuda:0" \
    training.seed=$SEED \
    training.resume=$RESUME_EPOCH

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 训练完成!"
    echo "========================================="
    echo "Checkpoint位置: ../checkpoints/${TASK_NAME}-${TASK_CONFIG}-${NUM_EPISODES}-gnn-endpose_${SEED}/"
else
    echo ""
    echo "========================================="
    echo "❌ 训练失败!"
    echo "========================================="
    exit 1
fi

