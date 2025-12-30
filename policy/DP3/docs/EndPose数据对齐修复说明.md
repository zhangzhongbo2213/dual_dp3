# DP3-EndPose 数据对齐修复说明

> **修复日期**: 2025年12月22日  
> **修复内容**: 修正数据处理脚本中的时间对齐错误

---

## 🔴 修复的问题

### 原问题
数据处理脚本 `process_data_endpose.py` 中存在时间对齐错误：
- **注释说明**: 观测帧[j-2, j-1, j] → 预测动作帧[j+3, j+4, ..., j+8]
- **实际代码**: action使用的是帧j的数据，而不是帧j+3的数据
- **影响**: 模型学习的是"当前帧预测当前帧"，而不是"当前帧预测未来帧"

### 修复后的逻辑
- **目标**: 观测帧[j-2, j-1, j] → 预测动作帧[j+1, j+2, j+3, j+4, j+5, j+6]
- **实现**: action[i]存储帧i+1的动作（8维）
- **特殊处理**: 对第0, 1帧做特殊处理（用帧0填充观测）

---

## 📊 数据对齐策略

### 数据存储格式

**State数组** (索引i对应观测帧i):
- `state[i]`: 帧i的状态（8维：左xyz+左爪+右xyz+右爪）

**Action数组** (索引i对应观测帧i):
- `action[i]`: 帧i+1的动作（8维）
- `action[i+1]`: 帧i+2的动作（如果i+1是有效观测帧）
- ...
- `action[i+5]`: 帧i+6的动作（如果i+5是有效观测帧）
- `action[i+6]`: 帧i+6的动作（填充，重复第6帧）
- `action[i+7]`: 帧i+6的动作（填充，重复第6帧）

### Dataset采样

Dataset采样时，对于索引i（对应观测帧i）：
- **观测**: `state[i-2:i+1]` → 原始帧[i-2, i-1, i]的状态（3帧）
- **动作**: `action[i:i+8]` → 原始帧[i+1, i+2, ..., i+6, i+6, i+6]的动作（8帧）
- **实际使用**: 只取前6帧动作执行

### 特殊处理（第0, 1帧）

- **j=0**: 观测[0, 0, 0]（用帧0填充），动作[1, 2, 3, 4, 5, 6]
- **j=1**: 观测[0, 0, 1]（用帧0填充），动作[2, 3, 4, 5, 6, 7]
- **j>=2**: 观测[j-2, j-1, j]，动作[j+1, j+2, j+3, j+4, j+5, j+6]

---

## 🔧 修复的代码

### 主要修改

**文件**: `scripts/process_data_endpose.py`

**修改前** (第167-174行):
```python
# action: 使用第j+3帧的状态作为action target
if j >= 3:
    future_state_8d = create_state_vector(
        left_xyz[j], left_grip[j],      # ❌ 使用的是j，不是j+3！
        right_xyz[j], right_grip[j]
    )
    action_arrays.append(future_state_8d)
```

**修改后** (第174-181行):
```python
# 存储帧j+1的动作（8维）
# Dataset采样时，对于索引i（对应观测帧i），会采样action[i:i+8]
# 所以action[i]应该是帧i+1的动作
action_j1 = create_state_vector(
    left_xyz[j + 1], left_grip[j + 1],  # ✅ 使用j+1帧
    right_xyz[j + 1], right_grip[j + 1]
)
action_arrays.append(action_j1)  # 帧j+1的动作（8维）
```

### 添加的填充逻辑

为了满足horizon=8的要求，添加了action数组填充逻辑：
- 扩展action数组，确保有足够的长度
- 对于每个有效的索引i，填充action[i+6]和action[i+7]为帧i+6的动作（重复最后一帧）

---

## ✅ 验证方法

### 1. 数据对齐测试

运行测试脚本：
```bash
cd /mnt/4T/RoboTwin/policy/DP3
conda activate RoboTwin
python scripts/test_endpose_alignment.py
```

### 2. 重新处理数据

```bash
cd /mnt/4T/RoboTwin/policy/DP3
conda activate RoboTwin
bash process_data_endpose.sh stack_blocks_two demo_clean 50
```

### 3. 验证数据格式

```python
import zarr
import numpy as np

z = zarr.open('./data/stack_blocks_two-demo_clean-50-endpose.zarr', 'r')
print("Point cloud shape:", z['data/point_cloud'].shape)
print("State shape:", z['data/state'].shape)
print("Action shape:", z['data/action'].shape)

# 验证action数组长度
assert z['data/action'].shape[0] >= z['data/state'].shape[0], \
    "Action array should be at least as long as state array"
```

---

## 📋 清理旧数据

已创建清理脚本 `scripts/clean_endpose_data.sh`，用于删除：
- 旧的endpose数据文件（.zarr）
- 旧的endpose训练结果（checkpoints）
- 旧的训练输出（outputs）

运行清理：
```bash
cd /mnt/4T/RoboTwin/policy/DP3
conda activate RoboTwin
bash scripts/clean_endpose_data.sh
```

---

## 🚀 下一步

1. ✅ **修复完成** - 数据处理脚本已修复
2. ✅ **清理完成** - 旧数据已删除
3. ⏭️ **重新处理数据** - 使用修复后的脚本重新处理数据
4. ⏭️ **重新训练** - 使用新数据重新训练模型
5. ⏭️ **验证效果** - 验证修复后的模型性能

---

## 📝 注意事项

1. **数据对齐**: 确保理解观测帧和动作帧的对应关系
2. **特殊处理**: 第0, 1帧的观测需要特殊处理（用帧0填充）
3. **Horizon填充**: action数组需要填充后2帧以满足horizon=8的要求
4. **数据验证**: 重新处理数据后，务必验证数据格式是否正确

---

**修复完成时间**: 2025年12月22日  
**状态**: ✅ 已修复并清理

