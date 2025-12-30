#!/bin/bash

# DP3-GNN-EndPose 数据处理和训练脚本
# 用法: bash train_gnn_endpose.sh <task_name> <task_config> <num_episodes> <gpu_id> <resume_epoch>
# 示例: bash train_gnn_endpose.sh stack_blocks_two demo_clean 50 0 0

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
echo "DP3-GNN-EndPose Training Pipeline"
echo "=========================================="
echo "Task Name: $TASK_NAME"
echo "Task Config: $TASK_CONFIG"
echo "Num Episodes: $NUM"
echo "GPU: $GPU"
echo "Resume Epoch: $RESUME_EPOCH"
echo "=========================================="

# 设置 Hugging Face 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# Step 1: 数据处理
echo ""
echo "Step 1: Processing data with GNN-EndPose format..."
cd scripts
python process_data_gnn_endpose.py $TASK_NAME $TASK_CONFIG $NUM --future_frames 6

if [ $? -ne 0 ]; then
    echo "❌ Data processing failed!"
    exit 1
fi

echo "✅ Data processing completed!"
cd ..

# Step 2: 训练模型
echo ""
echo "Step 2: Training DP3-GNN-EndPose model..."
bash scripts/train_policy_gnn_endpose.sh $TASK_NAME $TASK_CONFIG $NUM $GPU $RESUME_EPOCH

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ Training completed!"
echo "=========================================="
echo "Training pipeline finished successfully!"
echo "=========================================="
