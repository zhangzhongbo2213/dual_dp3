# DP3-GNN-EndPose 快速开始指南

## 5分钟快速部署

### 前置条件
- [x] 已安装PyTorch (>=1.13)
- [x] 已安装DP3依赖
- [x] 有GPU (推荐8GB+ VRAM)

### Step 1: 安装PyTorch Geometric (1分钟)

```bash
# 根据你的CUDA版本选择
# CUDA 11.8
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# CUDA 11.7
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

### Step 2: 准备数据 (2分钟)

确保你的数据在 `../../data/` 目录下，格式如下:
```
data/
└── stack_blocks_two/
    └── demo_clean/
        └── data/
            ├── episode0.hdf5
            ├── episode1.hdf5
            └── ...
```

每个HDF5文件需包含:
- `/pointcloud` - 点云数据
- `/joint_action/vector` - qpos数据
- `/endpose/left_endpose` - 左臂EndPose
- `/endpose/right_endpose` - 右臂EndPose
- `/endpose/left_gripper` - 左夹爪
- `/endpose/right_gripper` - 右夹爪

### Step 3: 一键训练 (2分钟启动)

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 完整流程: 数据处理 + 训练
bash train_gnn_endpose.sh stack_blocks_two demo_clean 50 0 0
```

就这么简单！脚本会自动:
1. ✅ 处理数据为GNN-EndPose格式
2. ✅ 启动训练
3. ✅ 保存checkpoints

## 分步操作 (可选)

如果你想分步执行，可以:

### 1. 仅数据处理
```bash
bash process_data_gnn_endpose.sh stack_blocks_two demo_clean 50
```

输出: `scripts/data/stack_blocks_two-demo_clean-50-gnn-endpose.zarr`

### 2. 仅训练
```bash
bash scripts/train_policy_gnn_endpose.sh stack_blocks_two demo_clean 50 0 0
```

### 3. 推理测试
```bash
python inference_gnn_endpose.py \
    --checkpoint_path <your_checkpoint.ckpt> \
    --task_name stack_blocks_two \
    --config demo_clean \
    --num_episodes 10
```

## 监控训练

训练会自动记录到W&B (如果已配置):
```
Project: dp3_gnn_endpose
```

或查看本地日志:
```bash
cd 3D-Diffusion-Policy/data/outputs/
# 找到最新的训练目录
tail -f <latest_run>/train.log
```

## 调整超参数

编辑配置文件进行调整:
```bash
vim 3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_gnn_endpose.yaml
```

常用调整:
```yaml
# 减小内存消耗
dataloader:
  batch_size: 32  # 默认64

# 调整GNN
policy:
  gnn_hidden_dim: 64  # 默认128
  num_graph_layers: 1  # 默认2

# 加快训练
training:
  num_epochs: 1000  # 默认3000
```

## Checkpoint位置

训练完成后，checkpoints保存在:
```
3D-Diffusion-Policy/data/outputs/YYYY.MM.DD/HH.MM.SS_dp3_gnn_endpose_<task>/
├── checkpoints/
│   ├── latest.ckpt
│   └── epoch=XXXX-test_mean_score=X.XXX.ckpt
└── train.log
```

## 验证模型

快速验证模型是否正常工作:

```bash
cd /mnt/4T/RoboTwin/policy/DP3
python -c "
from diffusion_policy_3d.model.gnn.robot_graph_network import RobotGraphNetwork
import torch

# 创建测试数据
left_qpos = torch.randn(4, 7)  # 7维: 6关节+1gripper
right_qpos = torch.randn(4, 7)
left_ep = torch.randn(4, 6, 4)
right_ep = torch.randn(4, 6, 4)

# 创建GNN
gnn = RobotGraphNetwork(
    left_joint_dim=7,
    right_joint_dim=7,
    endpose_dim=4,
    num_future_frames=6
)

# 前向传播
out = gnn(left_qpos, right_qpos, left_ep, right_ep)
print(f'✅ GNN test passed! Output shape: {out.shape}')
"
```

如果看到 `✅ GNN test passed!`，说明安装正确！

## 常见问题快速解决

### Q: ImportError: No module named 'torch_geometric'
```bash
pip install torch-geometric torch-scatter torch-sparse
```

### Q: CUDA out of memory
```bash
# 减小batch size
# 在 robot_dp3_gnn_endpose.yaml 中:
dataloader:
  batch_size: 16  # 或更小
```

### Q: 数据处理失败
```bash
# 检查数据路径
ls ../../data/stack_blocks_two/demo_clean/data/

# 检查HDF5文件内容
python -c "
import h5py
with h5py.File('../../data/stack_blocks_two/demo_clean/data/episode0.hdf5', 'r') as f:
    print(list(f.keys()))
"
```

### Q: 训练不收敛
- 检查数据质量
- 降低学习率 (lr: 5.0e-5)
- 增加warmup (lr_warmup_steps: 1000)

## 下一步

🎉 恭喜！你已经成功运行DP3-GNN-EndPose！

接下来可以:
1. 📖 阅读详细文档: `README_GNN_EndPose.md`
2. 🔧 调整模型架构: `DP3_GNN_EndPose_架构详解.md`
3. 🧪 在你自己的任务上训练
4. 📊 分析训练曲线和性能

## 需要帮助？

- 查看详细README: `README_GNN_EndPose.md`
- 查看架构文档: `DP3_GNN_EndPose_架构详解.md`
- 检查原始DP3文档
- 提交Issue到GitHub

祝你训练愉快！ 🚀
