#!/bin/bash
# 清理DP3-EndPose的旧数据和训练结果

echo "=========================================="
echo "清理DP3-EndPose旧数据和训练结果"
echo "=========================================="

cd "$(dirname "$0")/.."

# 1. 删除旧的endpose数据文件
echo ""
echo "1. 删除旧的endpose数据文件..."
DATA_DIRS=(
    "./data/*-endpose.zarr"
    "./scripts/data/*-endpose.zarr"
)

for pattern in "${DATA_DIRS[@]}"; do
    for dir in $pattern; do
        if [ -d "$dir" ]; then
            echo "  删除: $dir"
            rm -rf "$dir"
        fi
    done
done

# 2. 删除旧的endpose训练结果
echo ""
echo "2. 删除旧的endpose训练结果..."
CHECKPOINT_DIRS=(
    "./checkpoints/*endpose*"
    "./data/outputs/*/*endpose*"
)

for pattern in "${CHECKPOINT_DIRS[@]}"; do
    for dir in $pattern; do
        if [ -d "$dir" ]; then
            echo "  删除: $dir"
            rm -rf "$dir"
        fi
    done
done

# 3. 删除wandb日志（可选）
echo ""
read -p "是否删除wandb日志? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  删除wandb日志..."
    find . -name "wandb" -type d -exec rm -rf {} + 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "✅ 清理完成！"
echo "=========================================="

