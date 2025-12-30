#!/bin/bash

# DP3-GNN-EndPose 数据处理脚本
# 用法: bash process_data_gnn_endpose.sh <task_name> <task_config> <num_episodes>

TASK_NAME=$1
TASK_CONFIG=$2
NUM=$3

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

echo "=========================================="
echo "Processing Data for DP3-GNN-EndPose"
echo "=========================================="
echo "Task Name: $TASK_NAME"
echo "Task Config: $TASK_CONFIG"
echo "Num Episodes: $NUM"
echo "=========================================="

cd scripts
python process_data_gnn_endpose.py $TASK_NAME $TASK_CONFIG $NUM --future_frames 6

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Data processing completed successfully!"
    echo "Output: ./scripts/data/${TASK_NAME}-${TASK_CONFIG}-${NUM}-gnn-endpose.zarr"
else
    echo ""
    echo "❌ Data processing failed!"
    exit 1
fi

cd ..
