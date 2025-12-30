# DP3 EndPose - stack_blocks_two 任务训练指南

> **任务**: stack_blocks_two  
> **数据路径**: `/mnt/4T/RoboTwin/data/stack_blocks_two/demo_clean`  
> **数据量**: 50 episodes  
> **日期**: 2025年12月19日

---

## 📋 前置准备

### 1. 确认工作目录

```bash
cd /mnt/4T/RoboTwin/policy/DP3
```

### 2. 确认数据路径

你的数据在：
```
/mnt/4T/RoboTwin/data/stack_blocks_two/demo_clean/data/
├── episode0.hdf5
├── episode1.hdf5
├── ...
└── episode49.hdf5
```

总共 **50个episodes** ✓

---

## 🚀 快速开始

### 方式1: 一键训练（推荐）

训练脚本会自动检查数据，如果没有处理过会先处理数据，然后开始训练。

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 使用所有50个episodes训练
bash train_endpose.sh stack_blocks_two demo_clean 50 0 0
```

**参数说明**:
- `stack_blocks_two`: 任务名称
- `demo_clean`: 数据类型
- `50`: 使用50个episodes
- `0`: 随机种子（可选：0, 1, 2等）
- `0`: GPU ID (cuda:0)

---

### 方式2: 分步执行

#### 步骤1: 数据处理

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 处理数据（HDF5 → Zarr）
bash process_data_endpose.sh stack_blocks_two demo_clean 50
```

**预期输出**:
```
=========================================
DP3 EndPose Data Processing
=========================================
Task: stack_blocks_two
Config: demo_clean
Episodes: 50
=========================================
Processing 50 episodes from ../../data/stack_blocks_two/demo_clean
================================================================================
Processing episode: 50 / 50
================================================================================
Total frames processed: 5700 (假设平均每个episode 117帧)
Total episodes: 50
Average frames per episode: 114.0
================================================================================

Shapes:
  point_cloud: (5700, 1024, 3)
  state: (5700, 8)
  action: (5700, 8)
  episode_ends: (50,)

Saving to Zarr format...

✅ Successfully saved to: ./data/stack_blocks_two-demo_clean-50-endpose.zarr

Data summary:
  - 50 episodes
  - 5700 frames
  - Point cloud: (5700, 1024, 3)
  - State/Action: (5700, 8)
```

#### 步骤2: 训练模型

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 开始训练
bash train_endpose.sh stack_blocks_two demo_clean 50 0 0
```

**预期输出**:
```
=========================================
DP3 EndPose Training
=========================================
Task: stack_blocks_two
Config: demo_clean
Episodes: 50
Seed: 0
GPU: 0
=========================================
Data already exists. Starting training...
=========================================

Loading config: robot_dp3_endpose
Task: stack_blocks_two
...
Epoch 1/3000 | Loss: 0.1234
Epoch 100/3000 | Loss: 0.0456 | Val Loss: 0.0487
✓ Checkpoint saved: outputs/stack_blocks_two_endpose/checkpoints/epoch_100.ckpt
...
```

---

## 📊 数据处理详细说明

### 处理过程

```
输入: /mnt/4T/RoboTwin/data/stack_blocks_two/demo_clean/data/*.hdf5
  ↓
数据处理 (process_data_endpose.py):
  1. 读取点云 [T, 1024, 6] → 只取xyz [T, 1024, 3]
  2. 读取endpose [T, 7] → TCP转换 → xyz [T, 3]
  3. 读取gripper [T] → 保持 [T]
  4. 组合: state = [left_xyz(3), left_grip(1), right_xyz(3), right_grip(1)] = 8维
  5. 未来帧对齐: obs[j] → action[j+3]
  ↓
输出: ./data/stack_blocks_two-demo_clean-50-endpose.zarr
```

### 输出文件结构

```
./data/stack_blocks_two-demo_clean-50-endpose.zarr
├── data/
│   ├── point_cloud: (N, 1024, 3) - 观测点云
│   ├── state: (N, 8) - 当前状态
│   └── action: (N, 8) - 目标动作（未来3帧）
└── meta/
    └── episode_ends: (50,) - 每个episode结束位置
```

---

## 🎓 训练配置

### 默认配置 (`robot_dp3_endpose.yaml`)

```yaml
# 模型参数
horizon: 8              # 预测8帧
n_obs_steps: 3          # 输入3帧观测
n_action_steps: 6       # 执行6帧动作

# 训练参数
num_epochs: 3000        # 训练轮数
batch_size: 256         # 批次大小
lr: 1.0e-4             # 学习率

# 扩散模型
num_train_timesteps: 100   # 训练扩散步数
num_inference_steps: 10    # 推理采样步数
```

### 输入输出

| 项目 | 维度 | 说明 |
|------|------|------|
| **输入** | `[3, 1024, 3]` | 3帧点云，每帧1024个点 |
| **输出** | `[6, 8]` | 6帧动作，每帧8维 |
| **8维向量** | `[left_xyz(3), left_grip(1), right_xyz(3), right_grip(1)]` | endpose+gripper |

---

## 🔧 常用命令

### 1. 只处理部分数据（测试）

```bash
# 只处理前10个episodes
bash process_data_endpose.sh stack_blocks_two demo_clean 10
```

### 2. 使用不同GPU

```bash
# 使用GPU 1
bash train_endpose.sh stack_blocks_two demo_clean 50 0 1
```

### 3. 使用不同随机种子

```bash
# 种子=42
bash train_endpose.sh stack_blocks_two demo_clean 50 42 0
```

### 4. 检查Zarr数据

```bash
python -c "
import zarr
z = zarr.open('./data/stack_blocks_two-demo_clean-50-endpose.zarr', 'r')
print('Point cloud shape:', z['data/point_cloud'].shape)
print('State shape:', z['data/state'].shape)
print('Action shape:', z['data/action'].shape)
print('Episode ends:', z['meta/episode_ends'][:])
"
```

---

## 📁 输出文件位置

### 数据处理输出

```
./data/stack_blocks_two-demo_clean-50-endpose.zarr/
```

### 训练输出

```
./outputs/stack_blocks_two_endpose/
├── checkpoints/
│   ├── epoch_100.ckpt
│   ├── epoch_200.ckpt
│   └── latest.ckpt
├── logs/
└── config.yaml
```

---

## ⚠️ 注意事项

### 1. 数据路径

数据处理脚本会自动从 `../../data/{task_name}/{task_config}/` 读取，确保：
```
/mnt/4T/RoboTwin/data/stack_blocks_two/demo_clean/data/episode*.hdf5
```
存在。

### 2. 磁盘空间

- 原始HDF5: ~1.3GB (50个episodes)
- Zarr处理后: ~500MB (压缩后)
- 训练checkpoints: ~100MB/checkpoint

### 3. 内存需求

- 数据处理: ~4GB RAM
- 训练: ~8GB GPU显存 (batch_size=256)

### 4. 训练时间

- 数据处理: ~2-5分钟
- 训练3000 epochs: ~6-12小时（取决于GPU）

---

## 🎯 完整命令示例

### 示例1: 快速开始（使用所有数据）

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 一键训练（自动处理数据）
bash train_endpose.sh stack_blocks_two demo_clean 50 0 0
```

### 示例2: 测试流程（只用10个episodes）

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 1. 处理测试数据
bash process_data_endpose.sh stack_blocks_two demo_clean 10

# 2. 训练测试
bash train_endpose.sh stack_blocks_two demo_clean 10 0 0
```

### 示例3: 完整训练（分步执行）

```bash
cd /mnt/4T/RoboTwin/policy/DP3

# 1. 处理所有数据
bash process_data_endpose.sh stack_blocks_two demo_clean 50

# 2. 检查数据
python -c "
import zarr
z = zarr.open('./data/stack_blocks_two-demo_clean-50-endpose.zarr', 'r')
print('✓ Data loaded successfully')
print('  Episodes:', len(z['meta/episode_ends'][:]))
print('  Total frames:', z['data/action'].shape[0])
"

# 3. 开始训练
bash train_endpose.sh stack_blocks_two demo_clean 50 0 0
```

---

## 📊 训练监控

### 查看训练日志

```bash
# 实时查看
tail -f outputs/stack_blocks_two_endpose/logs/train.log

# 查看loss曲线
tensorboard --logdir outputs/stack_blocks_two_endpose/
```

### 验证checkpoint

```bash
python inference_endpose.py \
    --checkpoint outputs/stack_blocks_two_endpose/checkpoints/epoch_3000.ckpt \
    --zarr_path data/stack_blocks_two-demo_clean-50-endpose.zarr \
    --episode_idx 0
```

---

## ✅ 检查清单

在开始训练前，确认：

- [ ] 当前目录: `/mnt/4T/RoboTwin/policy/DP3`
- [ ] 数据存在: `ls ../../data/stack_blocks_two/demo_clean/data/episode*.hdf5`
- [ ] GPU可用: `nvidia-smi`
- [ ] 磁盘空间充足: `df -h .`

---

**准备好了吗？运行命令开始训练吧！** 🚀

```bash
cd /mnt/4T/RoboTwin/policy/DP3
bash train_endpose.sh stack_blocks_two demo_clean 50 0 0
```
