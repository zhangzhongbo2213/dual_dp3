# DP3 EndPose 预测系统

## 📝 项目概述

基于 DP3 (3D Diffusion Policy) 的双臂机器人末端位姿预测系统。

**功能**: 
- 输入 3 帧点云 → 预测未来 6 帧的双臂末端位置和夹爪状态

**应用场景**:
- 双臂协作任务
- 轨迹预测  
- 实时控制

---

## 🎯 输入输出格式

### 输入 (Input)
```python
{
    'point_cloud': [3, 1024, 3]  # 3帧点云，每帧1024个点，xyz坐标
}
```

### 输出 (Output)
```python
{
    'action': [6, 8]  # 6帧预测，每帧8维向量
}
```

**8维向量组成**:
```
索引 [0:3]  - 左臂末端位置 (x, y, z)
索引 [3]    - 左臂夹爪状态 (0或1)
索引 [4:7]  - 右臂末端位置 (x, y, z)
索引 [7]    - 右臂夹爪状态 (0或1)
```

---

## 🚀 快速开始

### 1. 数据处理
```bash
cd /mnt/4T/RoboTwin/policy/DP3
bash process_data_endpose.sh beat_block_hammer demo_clean 50
```

### 2. 训练模型
```bash
bash train_endpose.sh beat_block_hammer demo_clean 50 0 0
```

### 3. 推理预测
```bash
python inference_endpose.py \
    --checkpoint ./checkpoints/beat_block_hammer-demo_clean-50-endpose_0/3000.ckpt \
    --data /mnt/4T/RoboTwin/data/beat_block_hammer/demo_clean/data/episode0.hdf5 \
    --start_frame 10 \
    --output prediction.png
```

---

## 📂 文件说明

| 文件 | 说明 |
|------|------|
| `DP3_EndPose_需求说明.md` | 详细需求和数据格式说明 |
| `DP3_EndPose_使用指南.md` | 完整使用教程和配置说明 |
| `process_data_endpose.sh` | 数据处理启动脚本 |
| `train_endpose.sh` | 训练启动脚本 |
| `inference_endpose.py` | 推理和可视化脚本 |
| `scripts/process_data_endpose.py` | 数据处理核心代码 |
| `scripts/train_policy_endpose.sh` | 训练执行脚本 |
| `3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_endpose.yaml` | 主配置文件 |
| `3D-Diffusion-Policy/diffusion_policy_3d/config/task/endpose_task.yaml` | 任务配置 |

---

## ⚙️ 核心配置

```yaml
# robot_dp3_endpose.yaml
horizon: 8           # 预测8帧序列
n_obs_steps: 3       # 输入3帧观测
n_action_steps: 6    # 实际执行6帧

# 点云配置
shape_meta:
  obs:
    point_cloud:
      shape: [1024, 3]  # 每帧1024点，xyz坐标
  action:
    shape: [8]          # 8维动作向量

# 训练配置
training:
  num_epochs: 3000
  batch_size: 256
  lr: 1.0e-4
```

---

## 📊 数据格式

### 原始数据 (HDF5)
```
episode0.hdf5
├── pointcloud: [T, 1024, 6]          # T帧点云
├── endpose/left_endpose: [T, 7]      # 左臂位姿 (xyz + 四元数)
├── endpose/left_gripper: [T]         # 左夹爪状态
├── endpose/right_endpose: [T, 7]     # 右臂位姿
└── endpose/right_gripper: [T]        # 右夹爪状态
```

### 处理后数据 (Zarr)
```
beat_block_hammer-demo_clean-50-endpose.zarr
├── data/
│   ├── point_cloud: [总帧数, 1024, 3]  # 只取xyz
│   ├── state: [总帧数, 8]               # 8维状态向量
│   └── action: [总帧数, 8]              # 8维动作向量
└── meta/
    └── episode_ends: [50]               # Episode边界
```

---

## 🎓 技术特点

### DP3 架构
- **点云编码**: PointNet (输入3D点云 → 128维特征)
- **Diffusion策略**: DDIM调度器 (训练100步，推理10步)
- **条件融合**: FiLM (Feature-wise Linear Modulation)
- **时序建模**: UNet 1D (处理动作序列)

### 训练策略
- **滑动窗口采样**: 每帧都参与训练
- **数据增强**: 50个episodes → 数千个训练样本
- **EMA模型**: 指数移动平均提高稳定性
- **Cosine学习率**: 500步warmup

### 推理优化
- **DDIM加速**: 10步采样 vs 100步训练
- **稀疏执行**: 每次预测后执行6帧才重新预测
- **GPU加速**: CUDA推理，实时性能

---

## 📈 预期性能

| 指标 | 值 |
|------|---|
| 训练时间 | 4-8小时 (50 episodes, GPU) |
| 推理速度 | 0.1-0.2秒/次 (GPU) |
| 位置误差 | <1cm (良好训练) |
| 夹爪准确率 | >95% |
| 模型大小 | ~200MB |

---

## 🔍 与原始DP3的区别

| 项目 | 原始DP3 | EndPose版本 |
|------|---------|------------|
| 输入 | 点云 + 机器人状态 | 点云 (xyz only) |
| 输出 | 关节角度 (14维) | 末端位姿 (8维) |
| 点云通道 | 6 (xyz+rgb) | 3 (xyz only) |
| 预测目标 | 关节空间 | 笛卡尔空间 |
| 旋转预测 | ❌ | ❌ (只预测位置) |

---

## ⚠️ 注意事项

### ✅ 确认事项

1. **只预测位置，不预测旋转**
   - 如需旋转，需要增加四元数维度 (8维 → 14维)

2. **点云只用xyz坐标**
   - 如需颜色，修改为 `shape: [1024, 6]`

3. **夹爪状态是二值的**
   - 当前假设 0/1，如果是连续值需要调整

4. **TCP坐标**
   - 当前直接使用 endpose 坐标
   - 如需偏移 (+0.12m)，需要修改数据处理代码

### 🔧 常见问题

- **GPU内存不足**: 减少 `batch_size`
- **训练太慢**: 减少 `num_epochs` 或启用 `debug` 模式
- **数据路径错误**: 检查 HDF5 文件路径
- **依赖缺失**: 安装 torch, h5py, zarr, hydra

---

## 📚 文档索引

1. **需求说明**: `DP3_EndPose_需求说明.md`
   - 详细的输入输出格式
   - 数据转换流程
   - 确认检查清单

2. **使用指南**: `DP3_EndPose_使用指南.md`
   - 完整的使用教程
   - 配置说明
   - 常见问题解答

3. **代码文件**:
   - 数据处理: `scripts/process_data_endpose.py`
   - 推理脚本: `inference_endpose.py`
   - 配置文件: `3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_endpose.yaml`

---

## 🎯 示例结果

### 预测输出
```
Frame    Left X      Left Y      Left Z      L Grip   Right X     Right Y     Right Z     R Grip  
0        -0.2980     -0.3140     0.9420      1.000    0.3060      -0.3130     0.9410      1.000
1        -0.2970     -0.3130     0.9410      1.000    0.3050      -0.3120     0.9400      1.000
2        -0.2960     -0.3120     0.9400      1.000    0.3040      -0.3110     0.9390      1.000
3        -0.2950     -0.3110     0.9390      1.000    0.3030      -0.3100     0.9380      1.000
4        -0.2940     -0.3100     0.9380      0.000    0.3020      -0.3090     0.9370      0.000
5        -0.2930     -0.3090     0.9370      0.000    0.3010      -0.3080     0.9360      0.000
```

### 可视化
推理脚本会生成包含4个子图的可视化:
1. 观测帧0 (ground truth)
2. 观测帧1 (ground truth)
3. 观测帧2 (ground truth)
4. 预测结果 (预测 vs ground truth对比)

---

## 🌟 核心优势

1. **端到端学习**: 从点云直接预测末端位姿
2. **双臂协调**: 同时预测两个机械臂
3. **实时性**: 0.1秒预测，支持在线控制
4. **鲁棒性**: Diffusion模型处理不确定性
5. **可扩展**: 易于添加旋转、力控等新功能

---

## 📞 联系方式

如有问题或建议，请参考:
- 详细文档: `DP3_EndPose_使用指南.md`
- 需求说明: `DP3_EndPose_需求说明.md`
- 原始DP3: `policy/DP3/README.md` (如有)

---

**祝使用愉快！** 🎉
