#!/bin/bash

# DP3 EndPose 预测 - 数据处理脚本
# 用法: bash process_data_endpose.sh <task_name> <task_config> <expert_data_num>
# 例如: bash process_data_endpose.sh beat_block_hammer demo_clean 50

task_name=${1}
task_config=${2}
expert_data_num=${3}

echo "========================================="
echo "DP3 EndPose Data Processing"
echo "========================================="
echo "Task: $task_name"
echo "Config: $task_config"
echo "Episodes: $expert_data_num"
echo "========================================="

python scripts/process_data_endpose.py $task_name $task_config $expert_data_num
