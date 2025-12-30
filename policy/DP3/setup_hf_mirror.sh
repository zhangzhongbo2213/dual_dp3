#!/bin/bash

# Hugging Face 国内镜像配置脚本
# 使用方法: source setup_hf_mirror.sh
# 或者在训练脚本开头添加: source setup_hf_mirror.sh

# 设置 Hugging Face 国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 可选：设置缓存目录（如果需要）
# export HF_HOME=~/.cache/huggingface

echo "✅ Hugging Face 镜像已设置为: $HF_ENDPOINT"
echo "✅ HF Transfer 已启用"

