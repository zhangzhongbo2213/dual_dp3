#!/bin/bash

# 传输训练好的模型权重到远程服务器
# 只传输200和300轮的checkpoint

REMOTE_HOST="ubuntun@10.7.44.73"
REMOTE_PASSWORD="123"
REMOTE_PATH="/mnt/4T/RoboTwin"
FOLDER_NAME="DP3_model_checkpoints_200_300_epochs"
LOCAL_DIR="/data/zzb/RoboTwin/policy/DP3/checkpoints_to_transfer"

echo "========================================="
echo "开始传输模型权重到远程服务器"
echo "========================================="
echo "远程服务器: $REMOTE_HOST"
echo "远程路径: $REMOTE_PATH"
echo "文件夹名称: $FOLDER_NAME"
echo "本地目录: $LOCAL_DIR"
echo "总大小: $(du -sh $LOCAL_DIR | cut -f1)"
echo "========================================="
echo ""

# 检查本地目录是否存在
if [ ! -d "$LOCAL_DIR" ]; then
    echo "❌ 错误: 本地目录不存在: $LOCAL_DIR"
    exit 1
fi

# 检查是否安装了sshpass
if ! command -v sshpass &> /dev/null; then
    echo "⚠️  sshpass未安装"
    echo "请运行以下命令安装: sudo apt-get install sshpass"
    echo "或者使用expect方式传输（如果已安装expect）"
    echo ""
    # 尝试使用expect
    if command -v expect &> /dev/null; then
        echo "✅ 检测到expect，将使用expect方式传输"
        USE_EXPECT=true
    else
        echo "❌ 请先安装sshpass或expect"
        exit 1
    fi
else
    USE_EXPECT=false
fi

# 使用rsync传输（支持断点续传和进度显示）
echo "🚀 开始传输..."

if [ "$USE_EXPECT" = true ]; then
    # 使用expect方式
    expect << EOF
set timeout 3600
spawn rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_PATH/$FOLDER_NAME/"
expect {
    "password:" {
        send "$REMOTE_PASSWORD\r"
        exp_continue
    }
    eof
}
EOF
    TRANSFER_STATUS=$?
else
    # 使用sshpass方式
    sshpass -p "$REMOTE_PASSWORD" rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
        "$LOCAL_DIR/" \
        "$REMOTE_HOST:$REMOTE_PATH/$FOLDER_NAME/"
    TRANSFER_STATUS=$?
fi

if [ $TRANSFER_STATUS -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 传输完成!"
    echo "========================================="
    echo "远程路径: $REMOTE_PATH/$FOLDER_NAME/"
    echo ""
    echo "传输的文件列表:"
    if [ "$USE_EXPECT" = true ]; then
        expect << EOF
set timeout 30
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $REMOTE_HOST "ls -lh $REMOTE_PATH/$FOLDER_NAME/*/*.ckpt 2>/dev/null | awk '{print \\\$9, \\\$5}'"
expect {
    "password:" {
        send "$REMOTE_PASSWORD\r"
        exp_continue
    }
    eof
}
EOF
    else
        sshpass -p "$REMOTE_PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $REMOTE_HOST "ls -lh $REMOTE_PATH/$FOLDER_NAME/*/*.ckpt 2>/dev/null | awk '{print \$9, \$5}'"
    fi
else
    echo ""
    echo "========================================="
    echo "❌ 传输失败!"
    echo "========================================="
    exit 1
fi

