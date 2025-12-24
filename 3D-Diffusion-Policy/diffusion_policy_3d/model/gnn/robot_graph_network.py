"""
Robot Graph Network for DP3-GNN-EndPose
图神经网络模块，用于建模机械臂关节间、关节-EndPose间以及双臂间的关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np


class ArmInternalGraphNet(nn.Module):
    """
    单臂内部关节图网络
    建模单个机械臂内部关节之间的关系
    """
    def __init__(self, node_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GCN layers for joint relationships
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
    def forward(self, x, edge_index):
        """
        Args:
            x: [num_joints, node_dim] - 关节特征
            edge_index: [2, num_edges] - 边索引
        Returns:
            x: [num_joints, hidden_dim] - 更新后的关节特征
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class JointEndPoseGraphNet(nn.Module):
    """
    关节-EndPose关联图网络
    建模关节状态与未来EndPose预测之间的关系
    """
    def __init__(self, joint_dim, endpose_dim, hidden_dim, num_future_frames=6):
        super().__init__()
        self.joint_dim = joint_dim
        self.endpose_dim = endpose_dim
        self.hidden_dim = hidden_dim
        self.num_future_frames = num_future_frames
        
        # Embed endpose predictions
        self.endpose_embed = nn.Linear(endpose_dim, hidden_dim)
        
        # Graph attention for joint-endpose relationships
        self.gat = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, joint_features, endpose_sequence, joint_to_endpose_edges):
        """
        Args:
            joint_features: [num_joints, hidden_dim] - 关节特征
            endpose_sequence: [num_future_frames, endpose_dim] - 未来EndPose序列
            joint_to_endpose_edges: [2, num_edges] - 关节到EndPose的边
        Returns:
            joint_features: [num_joints, hidden_dim] - 融合EndPose信息后的关节特征
        """
        # Embed endpose predictions
        endpose_features = self.endpose_embed(endpose_sequence)  # [6, hidden_dim]
        
        # Concatenate joint and endpose features
        all_features = torch.cat([joint_features, endpose_features], dim=0)  # [num_joints+6, hidden_dim]
        
        # Apply graph attention
        all_features = self.gat(all_features, joint_to_endpose_edges)
        all_features = self.norm(all_features)
        
        # Extract updated joint features
        num_joints = joint_features.shape[0]
        joint_features = all_features[:num_joints]
        
        return joint_features


class BiArmInteractionNet(nn.Module):
    """
    双臂交互网络
    建模左右臂之间的协同关系
    """
    def __init__(self, arm_feature_dim, hidden_dim):
        super().__init__()
        self.arm_feature_dim = arm_feature_dim
        self.hidden_dim = hidden_dim
        
        # MLPs for cross-arm interaction
        self.left_to_right_mlp = nn.Sequential(
            nn.Linear(arm_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.right_to_left_mlp = nn.Sequential(
            nn.Linear(arm_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Fusion layers
        self.left_fusion = nn.Linear(arm_feature_dim + hidden_dim, arm_feature_dim)
        self.right_fusion = nn.Linear(arm_feature_dim + hidden_dim, arm_feature_dim)
        
    def forward(self, left_arm_features, right_arm_features):
        """
        Args:
            left_arm_features: [left_feature_dim] - 左臂聚合特征
            right_arm_features: [right_feature_dim] - 右臂聚合特征
        Returns:
            left_enhanced: [left_feature_dim] - 增强后的左臂特征
            right_enhanced: [right_feature_dim] - 增强后的右臂特征
        """
        # Cross-arm information
        left_context = self.left_to_right_mlp(left_arm_features)  # Info from left to enhance right
        right_context = self.right_to_left_mlp(right_arm_features)  # Info from right to enhance left
        
        # Fuse with original features
        left_enhanced = self.left_fusion(torch.cat([left_arm_features, right_context], dim=-1))
        right_enhanced = self.right_fusion(torch.cat([right_arm_features, left_context], dim=-1))
        
        return left_enhanced, right_enhanced


class RobotGraphNetwork(nn.Module):
    """
    完整的机器人图网络
    整合单臂内部图、关节-EndPose图、双臂交互图
    """
    def __init__(
        self,
        left_joint_dim=7,  # 左臂关节数 (6个关节 + 1个gripper)
        right_joint_dim=7,  # 右臂关节数 (6个关节 + 1个gripper)
        qpos_dim_per_joint=1,  # 每个关节的qpos维度
        endpose_dim=4,  # 每帧endpose维度 (xyz + gripper)
        num_future_frames=6,  # 未来预测帧数
        hidden_dim=128,
        num_graph_layers=2
    ):
        super().__init__()
        
        self.left_joint_dim = left_joint_dim
        self.right_joint_dim = right_joint_dim
        self.qpos_dim_per_joint = qpos_dim_per_joint
        self.endpose_dim = endpose_dim
        self.num_future_frames = num_future_frames
        self.hidden_dim = hidden_dim
        
        # Embed qpos to higher dimension
        self.left_qpos_embed = nn.Linear(qpos_dim_per_joint, hidden_dim)
        self.right_qpos_embed = nn.Linear(qpos_dim_per_joint, hidden_dim)
        
        # Internal joint graphs
        self.left_arm_graph = ArmInternalGraphNet(hidden_dim, hidden_dim, num_graph_layers)
        self.right_arm_graph = ArmInternalGraphNet(hidden_dim, hidden_dim, num_graph_layers)
        
        # Joint-EndPose graphs
        self.left_joint_endpose_graph = JointEndPoseGraphNet(
            hidden_dim, endpose_dim, hidden_dim, num_future_frames
        )
        self.right_joint_endpose_graph = JointEndPoseGraphNet(
            hidden_dim, endpose_dim, hidden_dim, num_future_frames
        )
        
        # Bi-arm interaction
        arm_feature_dim = left_joint_dim * hidden_dim  # Flattened joint features
        self.biarm_interaction = BiArmInteractionNet(arm_feature_dim, hidden_dim)
        
        # Final output dimension
        self.output_dim = arm_feature_dim * 2  # Left + Right arm features
        
    def build_arm_edges(self, num_joints):
        """
        构建单臂内部的边连接（链式结构）
        Args:
            num_joints: 关节数量
        Returns:
            edge_index: [2, num_edges]
        """
        edges = []
        # Sequential connections: 0-1, 1-2, 2-3, ...
        for i in range(num_joints - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])  # Bidirectional
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def build_joint_endpose_edges(self, num_joints, num_future_frames):
        """
        构建关节到EndPose的边连接
        最后一个关节连接到所有未来EndPose
        Args:
            num_joints: 关节数量
            num_future_frames: 未来帧数
        Returns:
            edge_index: [2, num_edges]
        """
        edges = []
        # Last joint connects to all future endposes
        last_joint_idx = num_joints - 1
        for i in range(num_future_frames):
            endpose_idx = num_joints + i
            edges.append([last_joint_idx, endpose_idx])
            edges.append([endpose_idx, last_joint_idx])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def forward(self, left_qpos, right_qpos, left_endpose_future, right_endpose_future):
        """
        Args:
            left_qpos: [batch_size, left_joint_dim] - 左臂当前关节位置
            right_qpos: [batch_size, right_joint_dim] - 右臂当前关节位置
            left_endpose_future: [batch_size, num_future_frames, endpose_dim] - 左臂未来EndPose
            right_endpose_future: [batch_size, num_future_frames, endpose_dim] - 右臂未来EndPose
        Returns:
            graph_features: [batch_size, output_dim] - 图网络提取的特征
        """
        batch_size = left_qpos.shape[0]
        device = left_qpos.device
        
        # Build edge structures (same for all samples in batch)
        left_arm_edges = self.build_arm_edges(self.left_joint_dim).to(device)
        right_arm_edges = self.build_arm_edges(self.right_joint_dim).to(device)
        left_je_edges = self.build_joint_endpose_edges(self.left_joint_dim, self.num_future_frames).to(device)
        right_je_edges = self.build_joint_endpose_edges(self.right_joint_dim, self.num_future_frames).to(device)
        
        batch_left_features = []
        batch_right_features = []
        
        for b in range(batch_size):
            # Embed qpos
            left_joint_features = self.left_qpos_embed(left_qpos[b].unsqueeze(-1))  # [left_joint_dim, hidden_dim]
            right_joint_features = self.right_qpos_embed(right_qpos[b].unsqueeze(-1))  # [right_joint_dim, hidden_dim]
            
            # Apply internal arm graphs
            left_joint_features = self.left_arm_graph(left_joint_features, left_arm_edges)
            right_joint_features = self.right_arm_graph(right_joint_features, right_arm_edges)
            
            # Apply joint-endpose graphs
            left_joint_features = self.left_joint_endpose_graph(
                left_joint_features, left_endpose_future[b], left_je_edges
            )
            right_joint_features = self.right_joint_endpose_graph(
                right_joint_features, right_endpose_future[b], right_je_edges
            )
            
            # Aggregate arm features (flatten)
            left_arm_feature = left_joint_features.flatten()  # [left_joint_dim * hidden_dim]
            right_arm_feature = right_joint_features.flatten()  # [right_joint_dim * hidden_dim]
            
            batch_left_features.append(left_arm_feature)
            batch_right_features.append(right_arm_feature)
        
        # Stack batch
        left_arm_features = torch.stack(batch_left_features, dim=0)  # [batch_size, left_feature_dim]
        right_arm_features = torch.stack(batch_right_features, dim=0)  # [batch_size, right_feature_dim]
        
        # Bi-arm interaction
        left_enhanced, right_enhanced = self.biarm_interaction(left_arm_features, right_arm_features)
        
        # Concatenate left and right
        graph_features = torch.cat([left_enhanced, right_enhanced], dim=-1)  # [batch_size, output_dim]
        
        return graph_features


if __name__ == "__main__":
    # Test the graph network
    print("Testing RobotGraphNetwork...")
    
    batch_size = 4
    left_joint_dim = 7  # 6 joints + 1 gripper
    right_joint_dim = 7  # 6 joints + 1 gripper
    num_future_frames = 6
    endpose_dim = 4
    
    # Create dummy data
    left_qpos = torch.randn(batch_size, left_joint_dim)
    right_qpos = torch.randn(batch_size, right_joint_dim)
    left_endpose = torch.randn(batch_size, num_future_frames, endpose_dim)
    right_endpose = torch.randn(batch_size, num_future_frames, endpose_dim)
    
    # Create network
    net = RobotGraphNetwork(
        left_joint_dim=left_joint_dim,
        right_joint_dim=right_joint_dim,
        endpose_dim=endpose_dim,
        num_future_frames=num_future_frames,
        hidden_dim=128
    )
    
    # Forward pass
    output = net(left_qpos, right_qpos, left_endpose, right_endpose)
    
    print(f"Input shapes:")
    print(f"  left_qpos: {left_qpos.shape}")
    print(f"  right_qpos: {right_qpos.shape}")
    print(f"  left_endpose: {left_endpose.shape}")
    print(f"  right_endpose: {right_endpose.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim: {net.output_dim}")
    print("✓ Graph network test passed!")
