#!/bin/bash

# DP3-GNN-EndPose 模型训练脚本
# 用法: bash train_policy_gnn_endpose.sh <task_name> <task_config> <num_episodes> <gpu_id> <resume_epoch>

TASK_NAME=$1
TASK_CONFIG=$2
NUM=$3
GPU=$4
RESUME_EPOCH=$5

# 默认值
if [ -z "$TASK_NAME" ]; then
    TASK_NAME="stack_blocks_two"
fi

if [ -z "$TASK_CONFIG" ]; then
    TASK_CONFIG="demo_clean"
fi

if [ -z "$NUM" ]; then
    NUM=50
fi

if [ -z "$GPU" ]; then
    GPU=0
fi

if [ -z "$RESUME_EPOCH" ]; then
    RESUME_EPOCH=0
fi

echo "=========================================="
echo "Training DP3-GNN-EndPose Policy"
echo "=========================================="
echo "Task: $TASK_NAME"
echo "Config: $TASK_CONFIG"
echo "Episodes: $NUM"
echo "GPU: $GPU"
echo "Resume Epoch: $RESUME_EPOCH"
echo "=========================================="

# 设置 Hugging Face 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU
export HYDRA_FULL_ERROR=1

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# DP3目录在scripts的上一级
DP3_DIR="$(dirname "$SCRIPT_DIR")"

# 进入3D-Diffusion-Policy目录
cd "$DP3_DIR/3D-Diffusion-Policy"

# 设置数据路径（使用绝对路径）
ZARR_PATH="$DP3_DIR/scripts/data_gnn/${TASK_NAME}-${TASK_CONFIG}-${NUM}-gnn-endpose.zarr"

echo "Data path: $ZARR_PATH"
echo ""

# 运行训练
python train_dp3.py \
    --config-name=robot_dp3_gnn_endpose.yaml \
    task_name=$TASK_NAME \
    setting=$TASK_CONFIG \
    expert_data_num=$NUM \
    task.dataset.zarr_path="$ZARR_PATH" \
    training.device="cuda:0" \
    training.seed=42 \
    training.resume=$RESUME_EPOCH

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
