# stack_bowls_two demo_randomized 训练脚本说明

## 概述

本文档说明如何使用为 `stack_bowls_two` 任务的 `demo_randomized` 配置（100 episodes）创建的训练脚本。

## 训练脚本

### 1. EndPose 模型训练脚本

**文件位置**: `/data/zzb/RoboTwin/policy/DP3/train_endpose_bowls_randomized.sh`

**用法**:
```bash
cd /data/zzb/RoboTwin/policy/DP3
bash train_endpose_bowls_randomized.sh [seed] [gpu_id]
```

**参数说明**:
- `seed`: 随机种子，默认为 0
- `gpu_id`: GPU设备ID，默认为 0

**示例**:
```bash
# 使用默认参数 (seed=0, gpu=0)
bash train_endpose_bowls_randomized.sh

# 指定seed和GPU
bash train_endpose_bowls_randomized.sh 42 1
```

**功能**:
- 自动检查数据是否存在，不存在则先处理数据
- 使用 `robot_dp3_endpose` 配置进行训练
- 训练输出保存在 `./checkpoints/stack_bowls_two-demo_randomized-100-endpose_<seed>/`

**数据路径**: `./data/stack_bowls_two-demo_randomized-100-endpose.zarr`

---

### 2. GNN-EndPose 模型训练脚本

**文件位置**: `/data/zzb/RoboTwin/policy/DP3/train_gnn_endpose_bowls_randomized.sh`

**用法**:
```bash
cd /data/zzb/RoboTwin/policy/DP3
bash train_gnn_endpose_bowls_randomized.sh [gpu_id] [resume_epoch]
```

**参数说明**:
- `gpu_id`: GPU设备ID，默认为 0
- `resume_epoch`: 恢复训练的epoch，默认为 0（从头开始）

**示例**:
```bash
# 使用默认参数 (gpu=0, resume_epoch=0)
bash train_gnn_endpose_bowls_randomized.sh

# 指定GPU和恢复训练
bash train_gnn_endpose_bowls_randomized.sh 1 100  # 从epoch 100继续训练
```

**功能**:
- 自动检查数据是否存在（支持多个可能的路径）
- 如果数据不存在，自动处理数据
- 使用 `robot_dp3_gnn_endpose` 配置进行训练
- 训练输出保存在 `./checkpoints/stack_bowls_two-demo_randomized-100-gnn-endpose_42/`

**数据路径**（按优先级）:
1. `./scripts/data_processed/stack_bowls_two/demo_randomized/gnn_endpose/stack_bowls_two-demo_randomized-100-gnn-endpose.zarr`
2. `./scripts/data_gnn/stack_bowls_two-demo_randomized-100-gnn-endpose.zarr`

---

## 数据检查

### EndPose 数据
```bash
ls -la /data/zzb/RoboTwin/policy/DP3/data/stack_bowls_two-demo_randomized-100-endpose.zarr
```

### GNN-EndPose 数据
```bash
ls -la /data/zzb/RoboTwin/policy/DP3/scripts/data_processed/stack_bowls_two/demo_randomized/gnn_endpose/stack_bowls_two-demo_randomized-100-gnn-endpose.zarr
```

---

## 训练配置

### EndPose 模型
- **配置文件**: `3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_endpose.yaml`
- **任务配置**: `3D-Diffusion-Policy/diffusion_policy_3d/config/task/endpose_task.yaml`
- **默认seed**: 0（可通过参数修改）

### GNN-EndPose 模型
- **配置文件**: `3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_gnn_endpose.yaml`
- **任务配置**: `3D-Diffusion-Policy/diffusion_policy_3d/config/task/gnn_endpose_task.yaml`
- **默认seed**: 42（固定）

---

## 训练输出

### Checkpoint 位置

**EndPose**:
```
./checkpoints/stack_bowls_two-demo_randomized-100-endpose_<seed>/
```

**GNN-EndPose**:
```
./checkpoints/stack_bowls_two-demo_randomized-100-gnn-endpose_42/
```

### 训练日志

训练日志保存在:
```
./data/outputs/<exp_name>_seed<seed>/train_dp3.log
```

---

## 注意事项

1. **数据预处理**: 如果数据不存在，脚本会自动处理数据，但这可能需要一些时间
2. **GPU内存**: GNN-EndPose 模型需要更多GPU内存（推荐8GB+）
3. **训练时间**: 
   - EndPose: 约1.5-2小时（300 epochs）
   - GNN-EndPose: 约2.5-3小时（300 epochs）
4. **恢复训练**: GNN-EndPose脚本支持从指定epoch恢复训练

---

## 快速开始

### 训练 EndPose 模型
```bash
cd /data/zzb/RoboTwin/policy/DP3
bash train_endpose_bowls_randomized.sh 0 0
```

### 训练 GNN-EndPose 模型
```bash
cd /data/zzb/RoboTwin/policy/DP3
bash train_gnn_endpose_bowls_randomized.sh 0 0
```

---

## 故障排除

### 问题1: 找不到数据文件
**解决方案**: 脚本会自动处理数据，如果失败，可以手动运行：
```bash
# EndPose
bash process_data_endpose.sh stack_bowls_two demo_randomized 100

# GNN-EndPose
bash process_data_gnn_endpose.sh stack_bowls_two demo_randomized 100
```

### 问题2: GPU内存不足
**解决方案**: 
- 减小batch_size（在配置文件中）
- 使用更小的GPU或减少并行训练

### 问题3: 路径错误
**解决方案**: 确保在 `/data/zzb/RoboTwin/policy/DP3` 目录下运行脚本

---

## 相关文档

- [DP3_EndPose_数据处理详解.md](./DP3_EndPose_数据处理详解.md)
- [DP3_GNN_EndPose_项目总览.md](./DP3_GNN_EndPose_项目总览.md)
- [DP3_模型对比总结.md](./DP3_模型对比总结.md)

