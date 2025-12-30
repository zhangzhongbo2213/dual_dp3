#!/bin/bash

# 上传脚本：将本地 DP3 代码上传到 GitHub 仓库
# GitHub 仓库: https://github.com/zhangzhongbo2213/dual_dp3

set -e

echo "=========================================="
echo "DP3 代码上传到 GitHub"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "❌ 错误: 请在 /data/zzb/RoboTwin/policy/DP3 目录下运行此脚本"
    exit 1
fi

# 检查 git 是否初始化
if [ ! -d ".git" ]; then
    echo "⚠️  警告: 当前目录不是 git 仓库"
    echo "正在初始化 git 仓库..."
    git init
fi

# 检查远程仓库
echo "📡 检查远程仓库..."
if git remote | grep -q "dual_dp3"; then
    echo "✅ 远程仓库已配置: dual_dp3"
else
    echo "❌ 错误: 未找到 dual_dp3 远程仓库"
    echo "请先运行: git remote add dual_dp3 https://github.com/zhangzhongbo2213/dual_dp3.git"
    exit 1
fi

# 显示当前状态
echo ""
echo "📊 当前 git 状态:"
git status --short | head -20

# 询问用户确认
echo ""
read -p "是否继续上传? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消"
    exit 1
fi

# 添加文件
echo ""
echo "📦 添加文件到 git..."
git add .

# 检查是否有更改
if git diff --cached --quiet; then
    echo "⚠️  没有需要提交的更改"
    exit 0
fi

# 提交
echo ""
echo "💾 提交更改..."
COMMIT_MSG="Update DP3 implementation with EndPose and GNN-EndPose variants

- Add DP3-EndPose model implementation
- Add DP3-GNN-EndPose model implementation  
- Add inference scripts (inference_endpose.py, inference_gnn_endpose.py)
- Add deployment scripts (deploy_policy.py, combined_policy.py)
- Add comprehensive documentation in docs/
- Add training scripts for all model variants
- Add data processing tools and utilities
- Update README with detailed usage guide"

git commit -m "$COMMIT_MSG"

# 推送到 GitHub
echo ""
echo "🚀 推送到 GitHub..."
BRANCH=${1:-main}
echo "使用分支: $BRANCH"

# 检查分支是否存在
if git branch -a | grep -q "remotes/dual_dp3/$BRANCH"; then
    echo "分支 $BRANCH 已存在于远程仓库"
    git push dual_dp3 $BRANCH
else
    echo "创建新分支: $BRANCH"
    git push -u dual_dp3 $BRANCH
fi

echo ""
echo "✅ 上传完成!"
echo "📝 查看仓库: https://github.com/zhangzhongbo2213/dual_dp3"
echo ""

