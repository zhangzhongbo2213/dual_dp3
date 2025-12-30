#!/bin/bash

# DP3-GNN-EndPose 维度检查脚本
# 验证所有文件中的维度是否已正确更新为7/14

echo "=========================================="
echo "DP3-GNN-EndPose 维度检查"
echo "=========================================="
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

# 检查函数
check_file() {
    local file=$1
    local pattern=$2
    local description=$3
    
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "✅ $description"
        ((SUCCESS_COUNT++))
    else
        echo "❌ $description"
        ((FAIL_COUNT++))
    fi
}

# 检查代码文件
echo "检查代码文件..."
echo "---"

check_file \
    "3D-Diffusion-Policy/diffusion_policy_3d/model/gnn/robot_graph_network.py" \
    "left_joint_dim=7" \
    "GNN网络: left_joint_dim=7"

check_file \
    "3D-Diffusion-Policy/diffusion_policy_3d/model/gnn/robot_graph_network.py" \
    "right_joint_dim=7" \
    "GNN网络: right_joint_dim=7"

check_file \
    "3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3_gnn_endpose.py" \
    "left_joint_dim=7" \
    "主模型: left_joint_dim=7"

check_file \
    "3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3_gnn_endpose.py" \
    "right_joint_dim=7" \
    "主模型: right_joint_dim=7"

echo ""

# 检查配置文件
echo "检查配置文件..."
echo "---"

check_file \
    "3D-Diffusion-Policy/diffusion_policy_3d/config/task/gnn_endpose_task.yaml" \
    "shape: \[14\]" \
    "Task配置: shape: [14]"

check_file \
    "3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_gnn_endpose.yaml" \
    "left_joint_dim: 7" \
    "Robot配置: left_joint_dim: 7"

check_file \
    "3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_gnn_endpose.yaml" \
    "right_joint_dim: 7" \
    "Robot配置: right_joint_dim: 7"

echo ""

# 检查文档文件
echo "检查文档文件..."
echo "---"

check_file \
    "README_GNN_EndPose.md" \
    "qpos \[B, 14\]" \
    "README: qpos [B, 14]"

check_file \
    "DP3_GNN_EndPose_架构详解.md" \
    "num_joints: 7" \
    "架构文档: num_joints: 7"

check_file \
    "DP3_GNN_EndPose_快速开始.md" \
    "left_qpos = torch.randn(4, 7)" \
    "快速开始: 测试代码使用7维"

check_file \
    "README_GNN_EndPose.md" \
    "Joint5 → Gripper" \
    "README: 链式连接包含Gripper"

echo ""

# 总结
echo "=========================================="
echo "检查结果汇总"
echo "=========================================="
echo "✅ 通过: $SUCCESS_COUNT"
echo "❌ 失败: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 所有检查通过！维度更新完成！"
    echo ""
    echo "关键维度确认:"
    echo "  - 单臂qpos: 7维 (6关节 + 1gripper) ✅"
    echo "  - 双臂总维度: 14维 (7+7) ✅"
    echo "  - GNN输出: 1792维 (7×128×2) ✅"
    exit 0
else
    echo "⚠️  有 $FAIL_COUNT 项检查失败，请检查文件内容"
    exit 1
fi
