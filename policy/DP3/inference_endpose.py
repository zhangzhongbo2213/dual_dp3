"""
DP3 EndPose 预测 - 推理和可视化脚本

功能:
1. 加载训练好的模型
2. 读取测试数据的3帧点云
3. 预测未来6帧的末端位姿和夹爪状态
4. 可视化预测结果
"""

import sys
import os
import pathlib
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# 添加路径
DP3_ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy'))
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy', 'diffusion_policy_3d'))

import hydra
from omegaconf import OmegaConf
from diffusion_policy_3d.policy.dp3 import DP3


def load_model(checkpoint_path, config_path=None):
    """加载训练好的模型"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['cfg']
    
    # 创建模型
    model = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(checkpoint['state_dicts']['model'])
    model.eval()
    model.cuda()
    
    print(f"Model loaded successfully!")
    return model, cfg


def load_test_data(hdf5_path, start_frame=0):
    """
    加载测试数据
    返回3帧点云作为观测
    """
    print(f"Loading test data from: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 加载3帧点云 (只取xyz)
        point_clouds = f['pointcloud'][start_frame:start_frame+3, :, :3]  # [3, 1024, 3]
        
        # 加载ground truth (用于对比)
        left_xyz_gt = f['endpose/left_endpose'][start_frame:start_frame+9, :3]  # [9, 3]
        left_grip_gt = f['endpose/left_gripper'][start_frame:start_frame+9]     # [9]
        right_xyz_gt = f['endpose/right_endpose'][start_frame:start_frame+9, :3] # [9, 3]
        right_grip_gt = f['endpose/right_gripper'][start_frame:start_frame+9]    # [9]
        
    print(f"Loaded:")
    print(f"  Point clouds: {point_clouds.shape}")
    print(f"  Ground truth: {left_xyz_gt.shape}")
    
    return point_clouds, left_xyz_gt, left_grip_gt, right_xyz_gt, right_grip_gt


def predict(model, point_clouds):
    """
    使用模型预测未来6帧的末端位姿
    
    输入:
      point_clouds: [3, 1024, 3] numpy array
    
    输出:
      predictions: [6, 8] numpy array
                   [left_x, left_y, left_z, left_grip, 
                    right_x, right_y, right_z, right_grip]
    """
    # 准备输入
    obs = {
        'point_cloud': torch.from_numpy(point_clouds).float().unsqueeze(0).cuda()  # [1, 3, 1024, 3]
    }
    
    # 预测
    with torch.no_grad():
        result = model.predict_action(obs)
        action_pred = result['action']  # [1, 6, 8]
    
    predictions = action_pred.cpu().numpy()[0]  # [6, 8]
    
    return predictions


def visualize_predictions(point_clouds, predictions, gt_left, gt_right, 
                         gt_left_grip, gt_right_grip, save_path=None):
    """
    可视化预测结果
    
    参数:
      point_clouds: [3, 1024, 3] - 输入的3帧点云
      predictions: [6, 8] - 预测的6帧末端位姿
      gt_left/right: [9, 3] - ground truth位置
      gt_left/right_grip: [9] - ground truth夹爪状态
    """
    fig = plt.figure(figsize=(20, 5))
    
    # 只显示最后一帧点云作为背景
    pc_last = point_clouds[-1]  # [1024, 3]
    
    # 绘制4个子图：观测3帧 + 预测6帧
    n_cols = 4
    
    for i in range(n_cols):
        ax = fig.add_subplot(1, n_cols, i+1, projection='3d')
        
        # 绘制点云
        ax.scatter(pc_last[:, 0], pc_last[:, 1], pc_last[:, 2],
                  s=0.5, c='gray', alpha=0.2)
        
        if i < 3:
            # 观测帧 (0, 1, 2)
            frame_idx = i
            ax.set_title(f'Observation Frame {frame_idx}')
            
            # Ground truth位置
            ax.scatter([gt_left[frame_idx, 0]], [gt_left[frame_idx, 1]], [gt_left[frame_idx, 2]],
                      c='red', s=100, marker='o', label='Left (GT)')
            ax.scatter([gt_right[frame_idx, 0]], [gt_right[frame_idx, 1]], [gt_right[frame_idx, 2]],
                      c='blue', s=100, marker='o', label='Right (GT)')
            
        else:
            # 预测帧 (显示前6帧中的2个示例)
            pred_indices = [0, 2, 5]  # 显示第0,2,5帧预测
            pred_idx = pred_indices[i-3] if i-3 < len(pred_indices) else 0
            
            ax.set_title(f'Prediction Frame {pred_idx}')
            
            # 预测位置
            pred = predictions[pred_idx]
            left_xyz_pred = pred[:3]
            left_grip_pred = pred[3]
            right_xyz_pred = pred[4:7]
            right_grip_pred = pred[7]
            
            # 预测 (用星号)
            ax.scatter([left_xyz_pred[0]], [left_xyz_pred[1]], [left_xyz_pred[2]],
                      c='red', s=150, marker='*', label=f'Left Pred (grip={left_grip_pred:.2f})')
            ax.scatter([right_xyz_pred[0]], [right_xyz_pred[1]], [right_xyz_pred[2]],
                      c='blue', s=150, marker='*', label=f'Right Pred (grip={right_grip_pred:.2f})')
            
            # Ground truth (用圆圈，半透明)
            gt_frame = 3 + pred_idx  # 预测对应的真实帧
            if gt_frame < len(gt_left):
                ax.scatter([gt_left[gt_frame, 0]], [gt_left[gt_frame, 1]], [gt_left[gt_frame, 2]],
                          c='red', s=100, marker='o', alpha=0.3, label='Left GT')
                ax.scatter([gt_right[gt_frame, 0]], [gt_right[gt_frame, 1]], [gt_right[gt_frame, 2]],
                          c='blue', s=100, marker='o', alpha=0.3, label='Right GT')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(fontsize=8)
        ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_predictions(predictions):
    """打印预测结果的详细信息"""
    print("\n" + "="*80)
    print("Prediction Results:")
    print("="*80)
    print(f"{'Frame':<8} {'Left X':<10} {'Left Y':<10} {'Left Z':<10} {'L Grip':<8} "
          f"{'Right X':<10} {'Right Y':<10} {'Right Z':<10} {'R Grip':<8}")
    print("-"*80)
    
    for i in range(len(predictions)):
        pred = predictions[i]
        print(f"{i:<8} {pred[0]:<10.4f} {pred[1]:<10.4f} {pred[2]:<10.4f} {pred[3]:<8.3f} "
              f"{pred[4]:<10.4f} {pred[5]:<10.4f} {pred[6]:<10.4f} {pred[7]:<8.3f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="DP3 EndPose Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (.ckpt)")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to test HDF5 file")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Start frame for prediction")
    parser.add_argument("--output", type=str, default="prediction_vis.png",
                       help="Output visualization path")
    
    args = parser.parse_args()
    
    print("="*80)
    print("DP3 EndPose Prediction")
    print("="*80)
    
    # 1. 加载模型
    model, cfg = load_model(args.checkpoint)
    
    # 2. 加载测试数据
    point_clouds, left_gt, left_grip_gt, right_gt, right_grip_gt = load_test_data(
        args.data, args.start_frame
    )
    
    # 3. 预测
    print("\nPredicting...")
    predictions = predict(model, point_clouds)
    print(f"Predictions shape: {predictions.shape}")
    
    # 4. 打印结果
    print_predictions(predictions)
    
    # 5. 可视化
    print("\nGenerating visualization...")
    visualize_predictions(
        point_clouds, predictions, 
        left_gt, right_gt,
        left_grip_gt, right_grip_gt,
        save_path=args.output
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
