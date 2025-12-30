#!/bin/bash

# 处理 randomized 数据的统一脚本
# 将 EndPose 和 GNN_EndPose 的输出保存在统一文件夹的不同子文件夹中

TASK_NAME="stack_bowls_two"
TASK_CONFIG="demo_randomized"
EXPERT_DATA_NUM=100  # 根据实际数据量调整

# 统一输出目录
OUTPUT_BASE_DIR="./data_processed/${TASK_NAME}/${TASK_CONFIG}"

# 创建输出目录结构
mkdir -p "${OUTPUT_BASE_DIR}/endpose"
mkdir -p "${OUTPUT_BASE_DIR}/gnn_endpose"

echo "=========================================="
echo "处理 Randomized 数据"
echo "=========================================="
echo "任务: ${TASK_NAME}"
echo "配置: ${TASK_CONFIG}"
echo "数据量: ${EXPERT_DATA_NUM} episodes"
echo "输出目录: ${OUTPUT_BASE_DIR}"
echo "=========================================="

# 切换到脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. 处理 EndPose 数据
echo ""
echo "【步骤 1/2】处理 EndPose 数据..."
echo "----------------------------------------"

# 临时修改输出路径（通过符号链接或直接修改脚本）
# 这里我们创建一个临时目录，然后移动结果
TEMP_ENDPOSE_OUTPUT="./data/${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-endpose.zarr"

python3 process_data_endpose.py \
    "${TASK_NAME}" \
    "${TASK_CONFIG}" \
    "${EXPERT_DATA_NUM}"

# 移动结果到统一目录
if [ -d "${TEMP_ENDPOSE_OUTPUT}" ]; then
    echo "移动 EndPose 数据到统一目录..."
    rm -rf "${OUTPUT_BASE_DIR}/endpose/${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-endpose.zarr"
    mv "${TEMP_ENDPOSE_OUTPUT}" "${OUTPUT_BASE_DIR}/endpose/"
    echo "✅ EndPose 数据已保存到: ${OUTPUT_BASE_DIR}/endpose/"
else
    echo "❌ EndPose 数据处理失败或输出不存在"
    exit 1
fi

# 2. 处理 GNN_EndPose 数据
echo ""
echo "【步骤 2/2】处理 GNN_EndPose 数据..."
echo "----------------------------------------"

TEMP_GNN_OUTPUT="./data_gnn/${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-gnn-endpose.zarr"

python3 process_data_gnn_endpose.py \
    "${TASK_NAME}" \
    "${TASK_CONFIG}" \
    "${EXPERT_DATA_NUM}"

# 移动结果到统一目录
if [ -d "${TEMP_GNN_OUTPUT}" ]; then
    echo "移动 GNN_EndPose 数据到统一目录..."
    rm -rf "${OUTPUT_BASE_DIR}/gnn_endpose/${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-gnn-endpose.zarr"
    mv "${TEMP_GNN_OUTPUT}" "${OUTPUT_BASE_DIR}/gnn_endpose/"
    echo "✅ GNN_EndPose 数据已保存到: ${OUTPUT_BASE_DIR}/gnn_endpose/"
else
    echo "❌ GNN_EndPose 数据处理失败或输出不存在"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 数据处理完成！"
echo "=========================================="
echo "输出目录结构:"
echo "  ${OUTPUT_BASE_DIR}/"
echo "    ├── endpose/"
echo "    │   └── ${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-endpose.zarr"
echo "    └── gnn_endpose/"
echo "        └── ${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}-gnn-endpose.zarr"
echo "=========================================="

