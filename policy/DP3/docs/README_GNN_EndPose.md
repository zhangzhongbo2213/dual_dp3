# DP3-GNN-EndPose: 图神经网络增强的Diffusion Policy

## 概述

DP3-GNN-EndPose 是一个结合图神经网络(GNN)和EndPose指导的Diffusion Policy模型，专门设计用于双臂机器人操作任务。该模型通过图结构显式建模关节间、关节-EndPose间以及双臂间的关系，提升了动作预测的准确性和协调性。

## 核心特性

### 1. 多模态输入
- **点云观测**: 3帧的3D点云数据
- **机械臂qpos**: 当前关节位置(state)
- **未来EndPose**: 未来6帧的EndPose预测(xyz + gripper)

### 2. 图神经网络架构
包含三个关键的图结构：

#### a) 单臂内部关节图 (ArmInternalGraphNet)
- 建模单个机械臂内部关节之间的运动学关系
- 使用GCN (Graph Convolutional Network) 传播关节信息
- 链式连接结构: Joint0 → Joint1 → ... → Joint5 → Gripper (7个节点)

#### b) 关节-EndPose关联图 (JointEndPoseGraphNet)
- 建模当前关节状态与未来EndPose目标的关系
- 使用GAT (Graph Attention Network) 学习关联权重
- Gripper节点(第7个节点)连接到所有未来6帧EndPose

#### c) 双臂交互网络 (BiArmInteractionNet)
- 建模左右臂之间的协同关系
- 通过MLP传递跨臂信息
- 融合双臂特征以实现协调控制

### 3. Diffusion Policy框架
- 保持DP3原有的扩散模型架构
- 将GNN提取的图特征作为额外的条件输入
- 支持多步预测(horizon=8, n_action_steps=6)

## 模型架构

```
输入:
├── 点云 [B, 3, 1024, 3]          # 3帧点云观测
├── qpos [B, 14]                   # 当前关节位置 (7+7: 每臂6关节+1gripper)
├── left_endpose_future [B, 6, 4]  # 左臂未来6帧EndPose
└── right_endpose_future [B, 6, 4] # 右臂未来6帧EndPose

处理流程:
├── PointNet编码器 → 点云特征 [B, 128*3]
├── Robot Graph Network:
│   ├── 左臂内部图 → 左臂关节特征 (7个节点)
│   ├── 右臂内部图 → 右臂关节特征 (7个节点)
│   ├── 左臂关节-EndPose图 → 增强左臂特征
│   ├── 右臂关节-EndPose图 → 增强右臂特征
│   └── 双臂交互 → 融合特征 [B, 1792]
├── 特征融合: [点云特征 + GNN特征]
└── Conditional UNet1D → 预测动作 [B, 8, 14]

输出:
└── 未来6帧的qpos动作 [B, 6, 14]
```

## 安装依赖

除了原始DP3的依赖外，还需要安装PyTorch Geometric：

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

将 `${TORCH}` 和 `${CUDA}` 替换为你的PyTorch和CUDA版本，例如:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 使用方法

### 1. 数据处理

处理原始HDF5数据为GNN-EndPose格式的Zarr文件：

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 方法1: 使用脚本
bash process_data_gnn_endpose.sh stack_blocks_two demo_clean 50

# 方法2: 直接调用Python
cd scripts
python process_data_gnn_endpose.py stack_blocks_two demo_clean 50 --future_frames 6
```

**输入数据要求** (HDF5格式):
- `/pointcloud`: [T, 1024, 6]
- `/joint_action/vector`: [T, qpos_dim]
- `/endpose/left_endpose`: [T, 7] (xyz + quaternion)
- `/endpose/right_endpose`: [T, 7]
- `/endpose/left_gripper`: [T]
- `/endpose/right_gripper`: [T]

**输出数据格式** (Zarr):
- `point_cloud`: [N, 1024, 3]
- `state`: [N, 14] - 当前qpos (7+7)
- `action`: [N, 14] - 下一帧qpos (7+7)
- `left_endpose_future`: [N, 6, 4] - 未来6帧EndPose (xyz+gripper)
- `right_endpose_future`: [N, 6, 4]

### 2. 模型训练

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 完整训练流程 (数据处理 + 训练)
bash train_gnn_endpose.sh stack_blocks_two demo_clean 50 0 0
#                        任务名称         配置     episodes GPU 恢复epoch

# 或单独训练 (假设数据已处理)
bash scripts/train_policy_gnn_endpose.sh stack_blocks_two demo_clean 50 0 0
```

**训练参数说明**:
- `task_name`: 任务名称 (例如: stack_blocks_two)
- `task_config`: 任务配置 (例如: demo_clean)
- `expert_data_num`: 训练episode数量
- `gpu_id`: GPU设备ID (默认: 0)
- `resume_epoch`: 从哪个epoch恢复训练 (默认: 0, 从头开始)

**配置文件位置**:
- 主配置: `3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_gnn_endpose.yaml`
- 任务配置: `3D-Diffusion-Policy/diffusion_policy_3d/config/task/gnn_endpose_task.yaml`

**可调整的关键参数**:
```yaml
# 在 robot_dp3_gnn_endpose.yaml 中:
policy:
  use_gnn: true              # 是否使用GNN
  left_joint_dim: 7          # 左臂关节数 (6 joints + 1 gripper)
  right_joint_dim: 7         # 右臂关节数 (6 joints + 1 gripper)
  endpose_dim: 4             # EndPose维度 (xyz+gripper)
  gnn_hidden_dim: 128        # GNN隐藏层维度
  num_graph_layers: 2        # GNN层数
  
dataloader:
  batch_size: 64             # batch大小 (GNN计算量较大，可能需要调小)
  
training:
  num_epochs: 3000           # 训练轮数
  lr: 1.0e-4                 # 学习率
```

### 3. 模型推理

```bash
cd /mnt/4T/RoboTwin/policy/DP3

python inference_gnn_endpose.py \
    --checkpoint_path <path_to_checkpoint.ckpt> \
    --task_name stack_blocks_two \
    --config demo_clean \
    --num_episodes 10 \
    --device cuda:0
```

## 文件结构

```
policy/DP3/
├── train_gnn_endpose.sh                    # 完整训练流程
├── process_data_gnn_endpose.sh             # 数据处理脚本
├── inference_gnn_endpose.py                # 推理脚本
├── scripts/
│   ├── process_data_gnn_endpose.py         # 数据处理Python脚本
│   └── train_policy_gnn_endpose.sh         # 训练脚本
└── 3D-Diffusion-Policy/
    └── diffusion_policy_3d/
        ├── model/
        │   └── gnn/
        │       └── robot_graph_network.py  # GNN模块
        ├── policy/
        │   └── dp3_gnn_endpose.py          # 主模型
        └── config/
            ├── robot_dp3_gnn_endpose.yaml  # 主配置
            └── task/
                └── gnn_endpose_task.yaml   # 任务配置
```

## 与原始EndPose模型的区别

| 特性 | DP3-EndPose | DP3-GNN-EndPose |
|-----|-------------|-----------------|
| **输入** | 点云 | 点云 + qpos + 未来EndPose |
| **状态表示** | TCP位置(xyz+gripper) | qpos (关节角度) |
| **动作输出** | 未来TCP位置 | 未来qpos |
| **结构建模** | 无显式结构 | GNN建模关节+双臂关系 |
| **未来信息** | 无 | 未来6帧EndPose指导 |
| **双臂协调** | 隐式学习 | 显式双臂交互图 |
| **训练数据** | 需要EndPose预测器 | 直接使用演示数据 |

## 数据流分析

### 训练阶段
1. **数据加载**: 
   - 点云 [B, 3, 1024, 3]
   - qpos [B, 3, 12]
   - EndPose futures [B, 6, 4] × 2

2. **特征提取**:
   - PointNet: 点云 → 点云特征 [B, 384]
   - GNN: (qpos, EndPose) → 图特征 [B, 1536]

3. **条件输入**:
   - Global condition: [点云特征, 图特征] → [B, 1920]

4. **Diffusion训练**:
   - 输入噪声: [B, 8, 12]
   - 条件: Global condition
   - 输出: 去噪后的动作序列

5. **损失计算**:
   - MSE between 预测动作 和 真实动作

### 推理阶段
1. 输入当前观测 (点云 + qpos + EndPose预测)
2. 提取点云和图特征
3. Diffusion采样生成动作序列
4. 提取前6步动作执行

## 调试和优化建议

### 1. 内存优化
GNN计算会增加内存消耗，如果遇到OOM:
- 减小`batch_size` (推荐: 32-64)
- 减小`gnn_hidden_dim` (推荐: 64-128)
- 减小`num_graph_layers` (推荐: 1-2)

### 2. 训练速度
- 使用较小的batch size会降低训练速度
- 可以增加`gradient_accumulate_every`来弥补
- 考虑使用混合精度训练 (AMP)

### 3. 模型性能
- 如果双臂任务，确保`use_gnn=True`
- 如果单臂任务，可以设置`use_gnn=False`简化模型
- 调整`gnn_hidden_dim`和`num_graph_layers`平衡性能和效率

### 4. 数据问题
- 确保EndPose数据质量良好
- 检查qpos数据范围是否合理
- 验证点云数据是否正确对齐

## 实验结果预期

相比原始DP3-EndPose模型，DP3-GNN-EndPose预期在以下方面有提升:

1. **双臂协调性**: 通过双臂交互图显式建模协同关系
2. **任务完成率**: 图结构帮助理解运动学约束
3. **泛化能力**: 结构化表示提升对新场景的适应能力
4. **训练效率**: 未来EndPose指导加速收敛

## 常见问题

### Q1: 数据处理失败，提示找不到endpose数据
**A**: 确保HDF5文件包含 `/endpose/left_endpose` 等字段。如果没有，需要先运行endpose预测生成这些数据。

### Q2: 训练时GPU内存不足
**A**: 
- 减小batch_size (例如从64到32)
- 减小gnn_hidden_dim (例如从128到64)
- 使用梯度累积

### Q3: 模型不收敛
**A**:
- 检查数据质量和范围
- 尝试降低学习率
- 增加warmup steps
- 检查GNN是否输出nan (可能是图结构问题)

### Q4: 如何可视化GNN学习到的特征?
**A**: 可以在`robot_graph_network.py`中添加可视化代码，保存注意力权重或图特征。

## 引用

如果使用了这个模型，请引用原始DP3论文以及PyTorch Geometric:

```bibtex
@inproceedings{dp3,
  title={3D Diffusion Policy},
  author={...},
  booktitle={...},
  year={2024}
}

@inproceedings{pyg,
  title={Fast Graph Representation Learning with PyTorch Geometric},
  author={Fey, Matthias and Lenssen, Jan Eric},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019}
}
```

## 联系和支持

如有问题或建议，请通过以下方式联系:
- 提交GitHub Issue
- 查看原始DP3文档获取更多信息

## 许可证

遵循原始RoboTwin项目的许可证。
