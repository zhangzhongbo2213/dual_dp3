"""
DP3-GNN-EndPose 数据处理脚本
处理包含qpos、endpose的训练数据

输入HDF5格式:
  - pointcloud: [T, 1024, 6]
  - joint_action/vector: [T, qpos_dim] (关节位置)
  - endpose/left_endpose: [T, 7] (xyz + quat)
  - endpose/right_endpose: [T, 7]
  - endpose/left_gripper: [T]
  - endpose/right_gripper: [T]

输出Zarr格式:
  - point_cloud: [总帧数, 1024, 3]
  - state: [总帧数, qpos_dim] (当前qpos)
  - action: [总帧数, qpos_dim] (下一帧qpos)
  - left_endpose_future: [总帧数, 6, 4] (未来6帧的xyz+gripper)
  - right_endpose_future: [总帧数, 6, 4]
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
    """
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # 点云数据
        pointcloud = root["/pointcloud"][:, :, :3]  # [T, 1024, 3]
        
        # 关节位置 (qpos)
        qpos = root["/joint_action/vector"][()]  # [T, qpos_dim]
        
        # EndPose数据
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
        
    return pointcloud, qpos, left_xyz, left_gripper, right_xyz, right_gripper


def create_endpose_vector(xyz, gripper):
    """
    创建4维endpose向量: [x, y, z, gripper]
    """
    return np.concatenate([xyz, [gripper]])  # [4]


def extract_future_endpose(left_xyz, left_gripper, right_xyz, right_gripper, 
                          current_idx, future_frames=6):
    """
    提取未来n帧的endpose
    
    Args:
        left_xyz: [T, 3]
        left_gripper: [T]
        right_xyz: [T, 3]
        right_gripper: [T]
        current_idx: 当前帧索引
        future_frames: 未来帧数
    
    Returns:
        left_future: [future_frames, 4] - 左臂未来endpose
        right_future: [future_frames, 4] - 右臂未来endpose
        valid: bool - 是否有足够的未来帧
    """
    T = left_xyz.shape[0]
    
    # 检查是否有足够的未来帧
    if current_idx + future_frames >= T:
        return None, None, False
    
    left_future = []
    right_future = []
    
    for i in range(1, future_frames + 1):
        future_idx = current_idx + i
        left_ep = create_endpose_vector(left_xyz[future_idx], left_gripper[future_idx])
        right_ep = create_endpose_vector(right_xyz[future_idx], right_gripper[future_idx])
        left_future.append(left_ep)
        right_future.append(right_ep)
    
    left_future = np.array(left_future)  # [future_frames, 4]
    right_future = np.array(right_future)  # [future_frames, 4]
    
    return left_future, right_future, True


def main():
    parser = argparse.ArgumentParser(description="Process episodes for DP3-GNN-EndPose.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., stack_blocks_two)",
    )
    parser.add_argument("task_config", type=str, help="Task configuration (e.g., demo_clean)")
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--future_frames",
        type=int,
        default=6,
        help="Number of future endpose frames to predict (default: 6)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config
    future_frames = args.future_frames

    # 使用绝对路径指向 /data/zzb/RoboTwin/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))  # /data/zzb/RoboTwin/policy/DP3/scripts
    dp3_dir = os.path.dirname(script_dir)  # /data/zzb/RoboTwin/policy/DP3
    policy_dir = os.path.dirname(dp3_dir)  # /data/zzb/RoboTwin/policy
    robotwin_root = os.path.dirname(policy_dir)  # /data/zzb/RoboTwin
    load_dir = os.path.join(robotwin_root, "data", task_name, task_config)
    save_dir = os.path.join(script_dir, f"data_gnn/{task_name}-{task_config}-{num}-gnn-endpose.zarr")

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
    left_endpose_future_arrays = []
    right_endpose_future_arrays = []

    print(f"Processing {num} episodes from {load_dir}")
    print(f"Future frames: {future_frames}")
    print("=" * 80)

    while current_ep < num:
        print(f"Processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        
        # 加载数据
        pointcloud, qpos, left_xyz, left_grip, right_xyz, right_grip = load_hdf5(load_path)
        
        T = pointcloud.shape[0]  # 时间步数
        
        # 处理每个时间步
        # 保证：当前有观测，未来有action和endpose
        for j in range(T):
            # State: 当前帧的qpos
            current_state = qpos[j]
            
            # 检查是否有下一帧作为action
            if j + 1 >= T:
                break
            
            # Action: 下一帧的qpos
            next_action = qpos[j + 1]
            
            # 提取未来endpose (从j+1开始的future_frames帧)
            left_ep_future, right_ep_future, valid = extract_future_endpose(
                left_xyz, left_grip, right_xyz, right_grip,
                j, future_frames
            )
            
            if not valid:
                break  # 没有足够的未来帧
            
            # 添加到数组
            point_cloud_arrays.append(pointcloud[j])
            state_arrays.append(current_state)
            action_arrays.append(next_action)
            left_endpose_future_arrays.append(left_ep_future)
            right_endpose_future_arrays.append(right_ep_future)
            
            total_count += 1

        current_ep += 1
        episode_ends_arrays.append(total_count)

    print()
    print("=" * 80)
    print(f"Total frames processed: {total_count}")
    print(f"Total episodes: {current_ep}")
    print(f"Average frames per episode: {total_count / current_ep:.1f}")
    print("=" * 80)

    # 转换为numpy数组
    print("Converting to numpy arrays...")
    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays)
    action_arrays = np.array(action_arrays)
    point_cloud_arrays = np.array(point_cloud_arrays)
    left_endpose_future_arrays = np.array(left_endpose_future_arrays)
    right_endpose_future_arrays = np.array(right_endpose_future_arrays)

    print(f"Shapes:")
    print(f"  point_cloud: {point_cloud_arrays.shape}")
    print(f"  state: {state_arrays.shape}")
    print(f"  action: {action_arrays.shape}")
    print(f"  left_endpose_future: {left_endpose_future_arrays.shape}")
    print(f"  right_endpose_future: {right_endpose_future_arrays.shape}")
    print(f"  episode_ends: {episode_ends_arrays.shape}")

    # 保存为Zarr格式
    print("\nSaving to Zarr format...")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    endpose_chunk_size = (100, left_endpose_future_arrays.shape[1], left_endpose_future_arrays.shape[2])

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
    zarr_data.create_dataset(
        "left_endpose_future",
        data=left_endpose_future_arrays,
        chunks=endpose_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "right_endpose_future",
        data=right_endpose_future_arrays,
        chunks=endpose_chunk_size,
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
    print(f"  - State (qpos): {state_arrays.shape}")
    print(f"  - Action (qpos): {action_arrays.shape}")
    print(f"  - Left EndPose Future: {left_endpose_future_arrays.shape}")
    print(f"  - Right EndPose Future: {right_endpose_future_arrays.shape}")
    print("\nNote: State contains current qpos, EndPose futures contain [xyz, gripper] for next 6 frames")


if __name__ == "__main__":
    main()
