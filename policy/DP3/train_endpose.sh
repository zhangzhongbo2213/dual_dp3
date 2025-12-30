#!/bin/bash

# DP3 EndPose 预测 - 训练脚本
# 用法: bash train_endpose.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id>
# 例如: bash train_endpose.sh beat_block_hammer demo_clean 50 0 0

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

echo "========================================="
echo "DP3 EndPose Training"
echo "========================================="
echo "Task: $task_name"
echo "Config: $task_config"
echo "Episodes: $expert_data_num"
echo "Seed: $seed"
echo "GPU: $gpu_id"
echo "========================================="

# 设置 Hugging Face 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 检查数据是否存在，不存在则先处理
if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}-endpose.zarr" ]; then
    echo "Data not found. Processing data first..."
    bash process_data_endpose.sh ${task_name} ${task_config} ${expert_data_num}
    echo "Data processing completed."
    echo "========================================="
fi

# 开始训练
bash scripts/train_policy_endpose.sh robot_dp3_endpose ${task_name} ${task_config} ${expert_data_num} train ${seed} ${gpu_id}
