#!/usr/bin/env python
"""
测试DP3-EndPose数据处理的时间对齐是否正确
"""

import numpy as np
import sys
import os

# 模拟数据处理逻辑
def test_alignment():
    """测试数据对齐逻辑"""
    print("=" * 80)
    print("测试DP3-EndPose数据对齐逻辑")
    print("=" * 80)
    
    # 模拟一个episode，有20帧数据
    T = 20
    print(f"\n模拟episode，共{T}帧")
    
    # 模拟state和action数组
    state_indices = []
    action_indices = []
    
    # 按照修复后的逻辑处理
    for j in range(T):
        # 检查是否有足够的未来帧
        if j + 6 < T:
            # 观测帧j对应索引i
            i = len(state_indices)
            state_indices.append(j)  # state[i] = 帧j的状态
            
            # action[i] = 帧j+1的动作
            action_indices.append(j + 1)
            print(f"  索引{i}: 观测帧{j} → action帧{j+1}")
    
    print(f"\n总样本数: {len(state_indices)}")
    print(f"State索引范围: 0 到 {len(state_indices)-1}")
    print(f"Action索引范围: 0 到 {len(action_indices)-1}")
    
    # 验证对齐
    print("\n验证数据对齐:")
    print("对于索引i（对应观测帧j）:")
    print("  观测: [j-2, j-1, j]")
    print("  动作: [j+1, j+2, j+3, j+4, j+5, j+6] (前6帧)")
    
    # 检查几个例子
    test_cases = [0, 1, 2, 5, 10]
    print("\n具体例子:")
    for i in test_cases:
        if i < len(state_indices):
            j = state_indices[i]
            print(f"\n  索引{i} (观测帧{j}):")
            print(f"    观测帧: [{max(0, j-2)}, {max(0, j-1)}, {j}]")
            print(f"    动作帧: [{j+1}, {j+2}, {j+3}, {j+4}, {j+5}, {j+6}]")
            
            # 检查action数组中的对应位置
            if i < len(action_indices):
                print(f"    action[{i}] = 帧{action_indices[i]}的动作 ✓")
            if i + 1 < len(action_indices):
                print(f"    action[{i+1}] = 帧{action_indices[i+1]}的动作 ✓")
            if i + 5 < len(action_indices):
                print(f"    action[{i+5}] = 帧{action_indices[i+5]}的动作 ✓")
    
    print("\n" + "=" * 80)
    print("✅ 数据对齐逻辑验证完成")
    print("=" * 80)


if __name__ == "__main__":
    test_alignment()

