"""
DP3-GNN-EndPose Policy
结合图神经网络和EndPose指导的Diffusion Policy

输入:
- 点云: 3帧点云观测
- qpos: 当前机械臂关节位置 (state)
- endpose_future: 未来6帧的endpose预测

图结构:
- 单臂内部关节图
- 关节-EndPose关联图
- 双臂交互图

输出:
- 未来6帧的机械臂action (关节空间)
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy_3d.model.gnn.robot_graph_network import RobotGraphNetwork


class DP3_GNN_EndPose(BasePolicy):
    """
    DP3 + GNN + EndPose 指导的策略模型
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        # GNN parameters
        left_joint_dim=7,      # 左臂关节数 (6 joints + 1 gripper)
        right_joint_dim=7,     # 右臂关节数 (6 joints + 1 gripper)
        endpose_dim=4,         # xyz + gripper
        gnn_hidden_dim=128,    # GNN隐藏层维度
        num_graph_layers=2,    # GNN层数
        use_gnn=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type
        self.use_gnn = use_gnn
        self.left_joint_dim = left_joint_dim
        self.right_joint_dim = right_joint_dim
        self.endpose_dim = endpose_dim

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])

        # PointCloud Encoder
        obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # Robot Graph Network (optional)
        if self.use_gnn:
            self.robot_gnn = RobotGraphNetwork(
                left_joint_dim=left_joint_dim,
                right_joint_dim=right_joint_dim,
                qpos_dim_per_joint=1,
                endpose_dim=endpose_dim,
                num_future_frames=n_action_steps,  # Match action horizon
                hidden_dim=gnn_hidden_dim,
                num_graph_layers=num_graph_layers
            )
            gnn_feature_dim = self.robot_gnn.output_dim
        else:
            gnn_feature_dim = 0

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        
        # Total input dimension includes:
        # - action_dim: predicted actions
        # - obs_feature_dim: point cloud features (optional)
        # - gnn_feature_dim: graph features from GNN
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        
        if obs_as_global_cond:
            # Point cloud + GNN features as global condition
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                # Point cloud features as sequence, GNN as additional global cond
                global_cond_dim = obs_feature_dim + gnn_feature_dim // n_obs_steps
            else:
                # Flatten all observations + GNN features
                global_cond_dim = obs_feature_dim * n_obs_steps + gnn_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        cprint(f"[DP3_GNN_EndPose] use_pc_color: {self.use_pc_color}", "cyan")
        cprint(f"[DP3_GNN_EndPose] pointnet_type: {self.pointnet_type}", "cyan")
        cprint(f"[DP3_GNN_EndPose] use_gnn: {self.use_gnn}", "cyan")
        cprint(f"[DP3_GNN_EndPose] gnn_feature_dim: {gnn_feature_dim}", "cyan")
        cprint(f"[DP3_GNN_EndPose] global_cond_dim: {global_cond_dim}", "cyan")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.parameters()))

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict should contain:
            - point_cloud: [B, T, N, C]
            - agent_pos: [B, T, qpos_dim]  (current qpos)
            - left_endpose_future: [B, n_action_steps, endpose_dim]
            - right_endpose_future: [B, n_action_steps, endpose_dim]
        """
        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)
        
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        # Extract qpos and endpose from obs_dict
        if self.use_gnn:
            # Assume agent_pos contains qpos for both arms
            # Format: [left_qpos (6), right_qpos (6), ...]
            agent_pos = nobs["agent_pos"][:, 0, :]  # Use first timestep, [B, qpos_dim]
            left_qpos = agent_pos[:, :self.left_joint_dim]  # [B, left_joint_dim]
            right_qpos = agent_pos[:, self.left_joint_dim:self.left_joint_dim+self.right_joint_dim]  # [B, right_joint_dim]
            
            # EndPose futures
            left_endpose_future = nobs.get("left_endpose_future", torch.zeros(B, self.n_action_steps, self.endpose_dim, device=device))
            right_endpose_future = nobs.get("right_endpose_future", torch.zeros(B, self.n_action_steps, self.endpose_dim, device=device))
            
            # Extract GNN features
            gnn_features = self.robot_gnn(left_qpos, right_qpos, left_endpose_future, right_endpose_future)  # [B, gnn_dim]
        else:
            gnn_features = None

        # Build input
        local_cond = None
        global_cond = None
        
        if self.obs_as_global_cond:
            # Encode point cloud
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            
            if "cross_attention" in self.condition_type:
                # Point cloud as sequence
                pc_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
                if self.use_gnn:
                    # Append GNN features to each timestep
                    gnn_cond = gnn_features.unsqueeze(1).expand(-1, self.n_obs_steps, -1)  # [B, n_obs_steps, gnn_dim]
                    global_cond = torch.cat([pc_cond, gnn_cond], dim=-1)
                else:
                    global_cond = pc_cond
            else:
                # Flatten point cloud features
                pc_cond = nobs_features.reshape(B, -1)
                if self.use_gnn:
                    global_cond = torch.cat([pc_cond, gnn_features], dim=-1)
                else:
                    global_cond = pc_cond
                    
            # Empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Condition through impainting (not typically used with global cond)
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            
            if self.use_gnn:
                # Append GNN features to observations
                gnn_cond = gnn_features.unsqueeze(1).expand(-1, To, -1)
                nobs_features = torch.cat([nobs_features, gnn_cond], dim=-1)
                
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # Run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # Unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        batch should contain:
            - obs: dict with point_cloud, agent_pos, left_endpose_future, right_endpose_future
            - action: [B, T, action_dim]
        """
        # Normalize input
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Extract GNN features
        if self.use_gnn:
            agent_pos = nobs["agent_pos"][:, 0, :]  # [B, qpos_dim]
            left_qpos = agent_pos[:, :self.left_joint_dim]
            right_qpos = agent_pos[:, self.left_joint_dim:self.left_joint_dim+self.right_joint_dim]
            
            left_endpose_future = nobs.get("left_endpose_future", torch.zeros(batch_size, self.n_action_steps, self.endpose_dim, device=nactions.device))
            right_endpose_future = nobs.get("right_endpose_future", torch.zeros(batch_size, self.n_action_steps, self.endpose_dim, device=nactions.device))
            
            gnn_features = self.robot_gnn(left_qpos, right_qpos, left_endpose_future, right_endpose_future)
        else:
            gnn_features = None

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # Encode point cloud
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # Point cloud as sequence
                pc_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
                if self.use_gnn:
                    gnn_cond = gnn_features.unsqueeze(1).expand(-1, self.n_obs_steps, -1)
                    global_cond = torch.cat([pc_cond, gnn_cond], dim=-1)
                else:
                    global_cond = pc_cond
            else:
                # Flatten point cloud features
                pc_cond = nobs_features.reshape(batch_size, -1)
                if self.use_gnn:
                    global_cond = torch.cat([pc_cond, gnn_features], dim=-1)
                else:
                    global_cond = pc_cond
        else:
            # Impainting
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            
            if self.use_gnn:
                gnn_cond = gnn_features.unsqueeze(1).expand(-1, horizon, -1)
                nobs_features = torch.cat([nobs_features, gnn_cond], dim=-1)
                
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        # Add noise (forward diffusion)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Compute loss mask
        loss_mask = ~condition_mask

        # Apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        loss_dict = {
            "bc_loss": loss.item(),
        }

        return loss, loss_dict
