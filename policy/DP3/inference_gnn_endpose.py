"""
DP3-GNN-EndPose 推理脚本
用于部署和测试训练好的模型

Usage:
    python inference_gnn_endpose.py \
        --checkpoint_path <path_to_checkpoint> \
        --task_name stack_blocks_two \
        --config demo_clean \
        --num_episodes 10
"""

import sys
import os
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import shutil
from termcolor import cprint
import time
import h5py

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "3D-Diffusion-Policy"))

from diffusion_policy_3d.policy.dp3_gnn_endpose import DP3_GNN_EndPose
from diffusion_policy_3d.dataset.robot_dataset import RobotDataset
from diffusion_policy_3d.common.pytorch_util import dict_apply


class DP3_GNN_EndPose_Inference:
    """
    DP3-GNN-EndPose 推理器
    """
    def __init__(self, checkpoint_path, device='cuda:0'):
        """
        初始化推理器
        
        Args:
            checkpoint_path: 检查点文件路径
            device: 设备
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        # 加载检查点
        print(f"Loading checkpoint from: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device)
        
        # 加载配置
        self.cfg = payload['cfg']
        
        # 创建模型
        print("Creating model...")
        self.policy = hydra.utils.instantiate(self.cfg.policy)
        self.policy.load_state_dict(payload['state_dicts']['model'])
        self.policy.to(self.device)
        self.policy.eval()
        
        # 加载normalizer
        if 'normalizer' in payload:
            self.policy.set_normalizer(payload['normalizer'])
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Horizon: {self.policy.horizon}")
        print(f"   - n_obs_steps: {self.policy.n_obs_steps}")
        print(f"   - n_action_steps: {self.policy.n_action_steps}")
        print(f"   - use_gnn: {self.policy.use_gnn}")
        
    def predict(self, obs_dict):
        """
        预测动作
        
        Args:
            obs_dict: 观测字典，包含:
                - point_cloud: [B, T, N, C]
                - agent_pos: [B, T, qpos_dim]
                - left_endpose_future: [B, n_action_steps, 4]
                - right_endpose_future: [B, n_action_steps, 4]
        
        Returns:
            action: [B, n_action_steps, action_dim]
        """
        with torch.no_grad():
            # 移动到设备
            nobs = dict_apply(obs_dict, 
                            lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)
            
            # 预测
            result = self.policy.predict_action(nobs)
            action = result['action']
            
        return action.cpu().numpy()
    
    def evaluate_dataset(self, zarr_path, num_episodes=10):
        """
        在数据集上评估模型
        
        Args:
            zarr_path: Zarr数据集路径
            num_episodes: 评估的episode数量
        """
        print(f"\n{'='*80}")
        print(f"Evaluating on dataset: {zarr_path}")
        print(f"{'='*80}\n")
        
        # 创建数据集
        dataset = RobotDataset(
            zarr_path=zarr_path,
            horizon=self.policy.horizon,
            pad_before=self.policy.n_obs_steps - 1,
            pad_after=self.policy.n_action_steps - 1,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
        )
        
        # 创建dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # 评估
        total_mse = 0.0
        total_samples = 0
        
        print("Running evaluation...")
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            if i >= num_episodes:
                break
            
            obs = batch['obs']
            true_action = batch['action']
            
            # 预测
            pred_action = self.predict(obs)
            pred_action = torch.from_numpy(pred_action)
            
            # 计算MSE
            mse = torch.mean((pred_action - true_action) ** 2).item()
            total_mse += mse
            total_samples += 1
            
            if i < 5:  # 打印前5个样本的详细信息
                print(f"\nSample {i+1}:")
                print(f"  True action shape: {true_action.shape}")
                print(f"  Pred action shape: {pred_action.shape}")
                print(f"  MSE: {mse:.6f}")
        
        avg_mse = total_mse / total_samples
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Average MSE: {avg_mse:.6f}")
        print(f"{'='*80}\n")
        
        return avg_mse


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="DP3-GNN-EndPose Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--task_name", type=str, default="stack_blocks_two",
                       help="Task name")
    parser.add_argument("--config", type=str, default="demo_clean",
                       help="Task config")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = DP3_GNN_EndPose_Inference(
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )
    
    # 构建数据集路径
    zarr_path = f"./scripts/data/{args.task_name}-{args.config}-{args.num_episodes}-gnn-endpose.zarr"
    
    if not os.path.exists(zarr_path):
        print(f"❌ Dataset not found: {zarr_path}")
        print("Please run data processing first:")
        print(f"  bash process_data_gnn_endpose.sh {args.task_name} {args.config} {args.num_episodes}")
        return
    
    # 评估
    avg_mse = inference.evaluate_dataset(zarr_path, num_episodes=args.num_episodes)
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()
