# DP3 EndPose 数据处理详解

## 核心问题回答

**当原始数据帧为 0,1,2 时，训练数据使用的是哪些帧？**

答案：**使用的是第 3,4,5,6,7,8,9,10 帧的数据**

## 详细分析

### 1. 数据处理逻辑 (`process_data_endpose.py`)

代码中的关键逻辑（第155-174行）：

```python
for j in range(T):
    # 创建当前帧的8维状态向量
    state_8d = create_state_vector(
        left_xyz[j], left_grip[j],
        right_xyz[j], right_grip[j]
    )
    
    # observation: 使用当前帧j的点云和状态
    # 需要保证后面至少有3帧可以作为action (j+3存在)
    if j + 3 < T:
        point_cloud_arrays.append(pointcloud[j])
        state_arrays.append(state_8d)
    
    # action: 使用第j+3帧的状态作为action target
    # 这样训练时: 观测帧[j-2,j-1,j] → 预测帧[j+3,j+4,...,j+8]
    if j >= 3:  # 确保前面有至少3帧用于观测
        future_state_8d = create_state_vector(
            left_xyz[j], left_grip[j],
            right_xyz[j], right_grip[j]
        )
        action_arrays.append(future_state_8d)
```

### 2. State 数组构建

**条件**: `if j + 3 < T`

这个条件确保当前帧后面至少还有3帧数据。

| j   | 条件 j+3 < T | State 数组索引 | State 对应帧 |
|-----|--------------|----------------|--------------|
| 0   | ✅           | 0              | 帧0的状态    |
| 1   | ✅           | 1              | 帧1的状态    |
| 2   | ✅           | 2              | 帧2的状态    |
| ... | ...          | ...            | ...          |

### 3. Action 数组构建

**条件**: `if j >= 3`

这个条件确保当前帧前面至少有3帧用于观测。

**关键点**: Action 数组使用的是 `left_xyz[j]` 而不是 `left_xyz[j+3]`

| j   | 条件 j>=3 | Action 数组索引 | Action 对应帧 |
|-----|-----------|-----------------|---------------|
| 0   | ❌        | -               | -             |
| 1   | ❌        | -               | -             |
| 2   | ❌        | -               | -             |
| 3   | ✅        | 0               | **帧3的状态** |
| 4   | ✅        | 1               | **帧4的状态** |
| 5   | ✅        | 2               | **帧5的状态** |
| 6   | ✅        | 3               | **帧6的状态** |
| 7   | ✅        | 4               | **帧7的状态** |
| 8   | ✅        | 5               | **帧8的状态** |
| 9   | ✅        | 6               | **帧9的状态** |
| 10  | ✅        | 7               | **帧10的状态**|

### 4. 训练配置 (`robot_dp3_endpose.yaml`)

```yaml
horizon: 8           # 预测8帧序列
n_obs_steps: 3       # 输入3帧观测
n_action_steps: 6    # 执行6帧动作
```

### 5. 训练时的数据对应关系

当模型训练时，从数据集中读取：
- **输入 (observation)**: State 数组的索引 [0, 1, 2]
- **输出 (action)**: Action 数组的索引 [0, 1, 2, 3, 4, 5, 6, 7]

具体对应到原始帧：

| 数据集索引 | State 数组 (observation) | Action 数组 (target) |
|-----------|--------------------------|----------------------|
| 0         | 帧0的状态                | 帧3的状态            |
| 1         | 帧1的状态                | 帧4的状态            |
| 2         | 帧2的状态                | 帧5的状态            |
| -         | -                        | 帧6的状态            |
| -         | -                        | 帧7的状态            |
| -         | -                        | 帧8的状态            |
| -         | -                        | 帧9的状态            |
| -         | -                        | 帧10的状态           |

### 6. 完整示例

假设一个 episode 有 15 帧数据（T=15）：

**State 数组** (条件: j+3 < 15，即 j < 12)：
- 索引 0-11：对应原始帧 0-11 的状态
- 共 12 个样本

**Action 数组** (条件: j >= 3)：
- 索引 0-11：对应原始帧 3-14 的状态
- 共 12 个样本

**训练样本**：
当读取第 i 个样本时（i=0,1,2,...,11）：
- **Observation**:
  - Point cloud: 帧 i 的点云
  - State: 帧 i 的状态 (8维: left_xyz + left_grip + right_xyz + right_grip)
  
- **Action Target**:
  - 帧 i+3 的状态 (8维)

**具体示例 - 样本0**：
- Observation: 帧0的点云 + 帧0的状态
- Action: 帧3的状态

**具体示例 - 样本1**：
- Observation: 帧1的点云 + 帧1的状态
- Action: 帧4的状态

**具体示例 - 样本2**：
- Observation: 帧2的点云 + 帧2的状态
- Action: 帧5的状态

### 7. 训练时的滑动窗口

训练器会使用 `n_obs_steps=3` 的滑动窗口：

**第一个 batch (样本索引 0)**:
- Observations: State[0], State[1], State[2] → 帧0, 帧1, 帧2 的状态
- Actions: Action[0:8] → 帧3, 帧4, 帧5, 帧6, 帧7, 帧8, 帧9, 帧10 的状态

**第二个 batch (样本索引 1)**:
- Observations: State[1], State[2], State[3] → 帧1, 帧2, 帧3 的状态  
- Actions: Action[1:9] → 帧4, 帧5, 帧6, 帧7, 帧8, 帧9, 帧10, 帧11 的状态

## 总结

**当数据为 0,1,2 时：**

1. **Observation (输入)**:
   - 使用帧 0, 1, 2 的点云和状态

2. **Action (训练目标)**:
   - 使用帧 3, 4, 5, 6, 7, 8, 9, 10 的状态（共8帧）

3. **预测horizon**:
   - 从当前帧向前预测8帧
   - 但这8帧是从观测最后一帧（帧2）之后的第3帧开始
   - 即预测帧 2+3=5 往后的状态序列

4. **关键理解**:
   - State 和 Action 数组的索引是对齐的
   - 但 Action 使用的原始帧号 = State 使用的原始帧号 + 3
   - 这实现了"观测当前 → 预测未来第3帧开始的轨迹"

## 代码bug说明

注意：代码第167-169行的注释**有误**：
```python
# action: 使用第j+3帧的状态作为action target
# 这样训练时: 观测帧[j-2,j-1,j] → 预测帧[j+3,j+4,...,j+8]
```

**实际代码执行的是**：
```python
future_state_8d = create_state_vector(
    left_xyz[j], left_grip[j],  # 使用的是j，不是j+3！
    right_xyz[j], right_grip[j]
)
```

所以实际上：
- State[i] 使用的是帧 i 的状态
- Action[i] 使用的是帧 i+3 的状态（因为 Action 从 j=3 开始收集）

这样的数据对齐方式实现了：**观测帧 [0,1,2] → 预测帧 [3,4,5,6,7,8,9,10]**
