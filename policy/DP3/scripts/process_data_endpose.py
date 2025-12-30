"""
DP3 EndPose 预测 - 数据处理脚本
将 HDF5 格式的 episode 数据转换为 Zarr 格式用于训练

输入: 
  - pointcloud: [T, 1024, 6] (只用前3个通道xyz)
  - endpose: left/right [T, 7] (只用前3个xyz)
  - gripper: left/right [T] (开合状态)

输出:
  - Zarr 格式:
    - point_cloud: [总帧数, 1024, 3]
    - state: [总帧数, 8] (左xyz+左爪+右xyz+右爪)
    - action: [总帧数, 8] (同state，但是t+1时刻的)
"""

import pickle
import os
import numpy as np
from copy import deepcopy
import zarr
import shutil
import argparse
import h5py
import transforms3d as t3d


def shift_to_tcp(pose_7d):
    """
    将endpose坐标转换为TCP坐标
    沿局部x轴偏移0.12m
    
    参数:
        pose_7d: [7] numpy array [x, y, z, qw, qx, qy, qz]
    
    返回:
        tcp_xyz: [3] numpy array [x, y, z] TCP坐标
    """
    pos = pose_7d[:3].copy()
    quat = pose_7d[3:7]  # [qw, qx, qy, qz]
    
    # 四元数转旋转矩阵
    R = t3d.quaternions.quat2mat(quat)
    
    # 沿局部x轴偏移0.12m
    tcp_pos = pos + R[:, 0] * 0.12
    
    return tcp_pos


def load_hdf5(dataset_path):
    """
    加载单个 episode 的 HDF5 数据
    
    注意: 这里会将endpose坐标转换为TCP坐标 (加0.12m偏移)
    """
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # 点云数据 [T, 1024, 6] -> 只取前3个通道
        pointcloud = root["/pointcloud"][:, :, :3]  # [T, 1024, 3]
        
        # 读取完整的7维位姿 (需要四元数来计算偏移)
        left_endpose_7d = root["/endpose/left_endpose"][()]   # [T, 7]
        right_endpose_7d = root["/endpose/right_endpose"][()]  # [T, 7]
        
        left_gripper = root["/endpose/left_gripper"][()]      # [T]
        right_gripper = root["/endpose/right_gripper"][()]    # [T]
        
        T = pointcloud.shape[0]
        
        # 转换为TCP坐标
        left_xyz = np.zeros((T, 3))
        right_xyz = np.zeros((T, 3))
        
        for i in range(T):
            left_xyz[i] = shift_to_tcp(left_endpose_7d[i])
            right_xyz[i] = shift_to_tcp(right_endpose_7d[i])
        
    return pointcloud, left_xyz, left_gripper, right_xyz, right_gripper


def create_state_vector(left_xyz, left_grip, right_xyz, right_grip):
    """
    创建8维状态向量: [left_x, left_y, left_z, left_gripper,
                      right_x, right_y, right_z, right_gripper]
    """
    return np.concatenate([
        left_xyz,          # [3]
        [left_grip],       # [1]
        right_xyz,         # [3]
        [right_grip]       # [1]
    ])  # [8]


def main():
    parser = argparse.ArgumentParser(description="Process episodes for DP3 EndPose prediction.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str, help="Task configuration (e.g., demo_clean)")
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    # 获取脚本所在目录，然后构建正确的数据路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # scripts目录 -> DP3目录 -> RoboTwin目录 -> data目录
    dp3_dir = os.path.dirname(script_dir)
    robotwin_dir = os.path.dirname(os.path.dirname(dp3_dir))
    load_dir = os.path.join(robotwin_dir, "data", str(task_name), str(task_config))
    
    # 保存路径相对于DP3目录
    save_dir = os.path.join(dp3_dir, "data", f"{task_name}-{task_config}-{num}-endpose.zarr")

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0
    total_count = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    episode_ends_arrays = []
    state_arrays = []
    action_arrays = []

    print(f"Processing {num} episodes from {load_dir}")
    print("=" * 80)

    while current_ep < num:
        print(f"Processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        
        # 加载数据
        pointcloud, left_xyz, left_grip, right_xyz, right_grip = load_hdf5(load_path)
        
        T = pointcloud.shape[0]  # 时间步数
        
        # 处理每个时间步
        # 目标: 观测帧[j-2, j-1, j] → 预测动作帧[j+1, j+2, j+3, j+4, j+5, j+6]
        # 
        # 数据对齐策略:
        # - state数组索引i: 存储帧i的状态（用于构建观测序列）
        # - action数组索引i: 存储帧i+1到i+6的动作序列（horizon=8，但只用前6帧）
        # 
        # 特殊处理:
        # - j=0: 观测[0, 0, 0]（用帧0填充），动作[1, 2, 3, 4, 5, 6]
        # - j=1: 观测[0, 0, 1]（用帧0填充），动作[2, 3, 4, 5, 6, 7]
        # - j>=2: 观测[j-2, j-1, j]，动作[j+1, j+2, j+3, j+4, j+5, j+6]
        
        for j in range(T):
            # 创建当前帧j的8维状态向量（用于state数组）
            state_8d = create_state_vector(
                left_xyz[j], left_grip[j],
                right_xyz[j], right_grip[j]
            )
            
            # 检查是否有足够的未来帧用于动作预测
            # 需要j+6帧存在（预测6帧动作）
            if j + 6 < T:
                # 添加观测数据（点云和状态）
                point_cloud_arrays.append(pointcloud[j])
                state_arrays.append(state_8d)
                
                # 存储帧j+1的动作（8维）
                # Dataset采样时，对于索引i（对应观测帧i），会采样action[i:i+8]
                # 所以action[i]应该是帧i+1的动作
                action_j1 = create_state_vector(
                    left_xyz[j + 1], left_grip[j + 1],
                    right_xyz[j + 1], right_grip[j + 1]
                )
                action_arrays.append(action_j1)  # 帧j+1的动作（8维）
                
                total_count += 1

        current_ep += 1
        episode_ends_arrays.append(total_count)

    print()
    print("=" * 80)
    print(f"Total frames processed: {total_count}")
    print(f"Total episodes: {current_ep}")
    print(f"Average frames per episode: {total_count / current_ep:.1f}")
    print("=" * 80)

    # 转换为numpy数组（保持各键时间长度一致，以满足 ReplayBuffer 断言）
    print("Converting to numpy arrays...")
    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays, dtype=np.float32)
    point_cloud_arrays = np.array(point_cloud_arrays, dtype=np.float32)
    action_arrays = np.array(action_arrays, dtype=np.float32)  # [N, 8]

    print(f"Shapes after conversion:")
    print(f"  point_cloud: {point_cloud_arrays.shape}")
    print(f"  state:       {state_arrays.shape}")
    print(f"  action:      {action_arrays.shape}")
    print(f"  episode_ends:{episode_ends_arrays.shape}")

    # 确保时间长度一致
    assert len(action_arrays) == len(state_arrays) == len(point_cloud_arrays), \
        f"Length mismatch: action {len(action_arrays)}, state {len(state_arrays)}, pc {len(point_cloud_arrays)}"

    # 保存为Zarr格式
    print("\nSaving to Zarr format...")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])  # [100, 8] - 2D array
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])

    zarr_data.create_dataset(
        "point_cloud",
        data=point_cloud_arrays,
        chunks=point_cloud_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    print(f"\n✅ Successfully saved to: {save_dir}")
    print("\nData summary:")
    print(f"  - {len(episode_ends_arrays)} episodes")
    print(f"  - {total_count} frames")
    print(f"  - Point cloud: {point_cloud_arrays.shape}")
    print(f"  - State/Action: {state_arrays.shape}")


if __name__ == "__main__":
    main()
