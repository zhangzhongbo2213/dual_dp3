import os
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import hydra
from termcolor import cprint


def _stack_last_n_obs(obs_list, n_steps: int):
    """
    与 RobotRunner.stack_last_n_obs 一致的逻辑：
    - obs_list: List[np.ndarray]，每个元素是同形状的单帧观测
    - n_steps:  需要堆叠的时间步数
    返回: np.ndarray，shape = [n_steps, *obs_shape]
    """
    assert len(obs_list) > 0
    obs_list = list(obs_list)
    last = obs_list[-1]
    result = np.zeros((n_steps,) + last.shape, dtype=last.dtype)
    start_idx = -min(n_steps, len(obs_list))
    result[start_idx:] = np.array(obs_list[start_idx:])
    if n_steps > len(obs_list):
        # 前面不足的部分用最早一帧填充
        result[:start_idx] = result[start_idx]
    return result


def load_dp3_endpose_policy(ckpt_path: str, device: torch.device):
    """
    加载 EndPose-DP3 模型
    ckpt 结构与 inference_endpose.py 中使用的一致：
      - payload['cfg']
      - payload['state_dicts']['model']
      - payload.get('normalizer', None)
    """
    assert os.path.isfile(ckpt_path), f"EndPose checkpoint not found: {ckpt_path}"
    cprint(f"[CombinedPolicy] Loading EndPose DP3 from: {ckpt_path}", "cyan")

    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = payload["cfg"]

    # hydra instantiate policy
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["model"])
    policy.to(device)
    policy.eval()

    if "normalizer" in payload:
        try:
            policy.set_normalizer(payload["normalizer"])
        except Exception as e:
            cprint(f"[CombinedPolicy] Warning: failed to load EndPose normalizer: {e}", "yellow")

    return policy, cfg


def load_dp3_gnn_endpose_policy(ckpt_path: str, device: torch.device):
    """
    加载 DP3_GNN_EndPose 模型
    ckpt 结构与 inference_gnn_endpose.py 中使用的一致：
      - payload['cfg']
      - payload['state_dicts']['model']
      - payload.get('normalizer', None)
    """
    assert os.path.isfile(ckpt_path), f"GNN-EndPose checkpoint not found: {ckpt_path}"
    cprint(f"[CombinedPolicy] Loading GNN-EndPose DP3 from: {ckpt_path}", "cyan")

    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = payload["cfg"]

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["model"])
    policy.to(device)
    policy.eval()

    if "normalizer" in payload:
        try:
            policy.set_normalizer(payload["normalizer"])
        except Exception as e:
            cprint(f"[CombinedPolicy] Warning: failed to load GNN-EndPose normalizer: {e}", "yellow")

    return policy, cfg


class CombinedEndPoseGNNPolicy:
    """
    将 EndPose-DP3 与 DP3_GNN_EndPose 串联的组合策略：
      1) 使用点云窗口 → EndPose-DP3 → 预测未来 EndPose 轨迹 [6, 8]
      2) 拆分为左右臂 endpose_future [6, 4] × 2
      3) 与点云窗口 + 当前 qpos 一起输入 DP3_GNN_EndPose → 输出未来 action
    """

    def __init__(
        self,
        endpose_ckpt_path: str,
        gnn_ckpt_path: str,
        device: str = "cuda:0",
        n_obs_steps: int = 3,
        n_action_steps: int = 6,
    ):
        self.device = torch.device(device)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # 加载两个子策略
        self.endpose_policy, self.endpose_cfg = load_dp3_endpose_policy(
            endpose_ckpt_path, self.device
        )
        self.gnn_policy, self.gnn_cfg = load_dp3_gnn_endpose_policy(
            gnn_ckpt_path, self.device
        )

        # 观测缓存：每个元素是一个 dict，包含
        #   - "point_cloud": [N, 3]  (单帧点云)
        #   - "agent_pos":  [14]     (当前 qpos)
        self.obs_deque: deque = deque(maxlen=self.n_obs_steps)

        cprint(
            f"[CombinedPolicy] Initialized. n_obs_steps={self.n_obs_steps}, "
            f"n_action_steps={self.n_action_steps}",
            "green",
        )

    # ========= 观测管理 =========
    def reset(self):
        """在每个 episode 开始时调用，清空观测缓存"""
        self.obs_deque.clear()

    def update_obs(self, obs: Dict):
        """
        obs 格式建议为（与 deploy_policy.encode_obs 对齐）:
            obs['agent_pos'] : np.ndarray, shape [14]
            obs['point_cloud']: np.ndarray, shape [1024, 3]
        """
        assert "point_cloud" in obs, "obs must contain 'point_cloud'"
        assert "agent_pos" in obs, "obs must contain 'agent_pos'"
        self.obs_deque.append(
            {
                "point_cloud": np.asarray(obs["point_cloud"]),
                "agent_pos": np.asarray(obs["agent_pos"]),
            }
        )

    # ========= 内部辅助函数 =========
    def _build_pointcloud_window(self) -> torch.Tensor:
        """
        使用 obs_deque 构造点云时间窗口:
            返回: [1, T, N, 3]，T = self.n_obs_steps
        """
        assert len(self.obs_deque) > 0, "No observation, please call update_obs first."
        pcs = [o["point_cloud"] for o in self.obs_deque]
        stacked = _stack_last_n_obs(pcs, self.n_obs_steps)  # [T, N, 3]
        tensor = torch.from_numpy(stacked).float().unsqueeze(0)  # [1, T, N, 3]
        return tensor.to(self.device)

    def _build_agentpos_window(self) -> torch.Tensor:
        """
        使用 obs_deque 构造 agent_pos 时间窗口:
            简单做法：与点云相同的时间窗口策略
            返回: [1, T, 14]
        """
        assert len(self.obs_deque) > 0, "No observation, please call update_obs first."
        qs = [o["agent_pos"] for o in self.obs_deque]
        stacked = _stack_last_n_obs(qs, self.n_obs_steps)  # [T, 14]
        tensor = torch.from_numpy(stacked).float().unsqueeze(0)  # [1, T, 14]
        return tensor.to(self.device)

    def _predict_endpose_future(
        self, pc_window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 EndPose-DP3 预测未来 EndPose 轨迹。
        输入:
            pc_window: [1, T, N, 3]
        输出:
            left_future:  [1, 6, 4]  (xyz + gripper)
            right_future: [1, 6, 4]
        """
        with torch.no_grad():
            obs_endpose = {"point_cloud": pc_window}
            res = self.endpose_policy.predict_action(obs_endpose)
            # res["action"]: [1, n_action_steps, 8]
            action = res["action"]  # [1, 6, 8]

            # 按 EndPose 数据处理约定拆分：
            #   [left_xyz(3), left_grip(1), right_xyz(3), right_grip(1)]
            left = action[..., :4]   # [1, 6, 4]
            right = action[..., 4:]  # [1, 6, 4]

        return left, right

    # ========= 对外主接口 =========
    def get_action(self, observation: Optional[Dict] = None) -> np.ndarray:
        """
        组合推理主入口。

        Args:
            observation: 可选。如果传入，则先 update_obs(observation)
        Returns:
            actions: np.ndarray, shape [n_action_steps, action_dim]
        """
        if observation is not None:
            self.update_obs(observation)

        assert len(self.obs_deque) > 0, "No observation to infer from."

        # 1) 构造点云 + agent_pos 时间窗口
        pc_window = self._build_pointcloud_window()    # [1, T, N, 3]
        agent_window = self._build_agentpos_window()   # [1, T, 14]

        # 2) 调 EndPose-DP3 预测未来 EndPose 轨迹
        left_future, right_future = self._predict_endpose_future(pc_window)

        # 3) 构造 GNN_EndPose 的观测字典
        obs_gnn = {
            "point_cloud": pc_window,           # [1, T, N, 3]
            "agent_pos": agent_window,          # [1, T, 14]
            "left_endpose_future": left_future,   # [1, 6, 4]
            "right_endpose_future": right_future, # [1, 6, 4]
        }

        # 4) 调用 GNN_EndPose 策略获取动作
        with torch.no_grad():
            res = self.gnn_policy.predict_action(obs_gnn)
            # res["action"]: [1, n_action_steps, action_dim]
            action = res["action"]

        actions_np = action.squeeze(0).detach().cpu().numpy()  # [n_action_steps, action_dim]
        return actions_np


class CombinedDP3:
    """
    面向 RoboTwin 的简单封装类，接口与 dp3_policy.DP3 一致：
        - update_obs(observation)
        - get_action(observation=None)

    方便在 deploy_policy.py 或 eval 脚本中直接替换使用。
    """

    class _CombinedRunner:
        """
        轻量封装，使 CombinedDP3 具备与原始 DP3 相同的 env_runner 接口：
          - .obs: 直接引用 CombinedEndPoseGNNPolicy 内部的 obs_deque
          - reset_obs(): 调用 CombinedEndPoseGNNPolicy.reset()
        """

        def __init__(self, combined_dp3: "CombinedDP3"):
            self._combined = combined_dp3

        @property
        def obs(self):
            # 直接返回内部策略的观测 deque
            return self._combined.policy.obs_deque

        def reset_obs(self):
            self._combined.policy.reset()

    def __init__(
        self,
        endpose_ckpt_path: str,
        gnn_ckpt_path: str,
        device: str = "cuda:0",
        n_obs_steps: int = 3,
        n_action_steps: int = 6,
    ) -> None:
        self.policy = CombinedEndPoseGNNPolicy(
            endpose_ckpt_path=endpose_ckpt_path,
            gnn_ckpt_path=gnn_ckpt_path,
            device=device,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )
        # 提供与原始 DP3 相同的 env_runner 接口，便于复用 eval/reset_model 逻辑
        self.env_runner = CombinedDP3._CombinedRunner(self)

    def update_obs(self, observation: Dict):
        """
        observation: 来自 RoboTwin 环境的原始观测
            - observation['joint_action']['vector'] : qpos (14)
            - observation['pointcloud']            : [1024, 3]
        """
        obs = {
            "agent_pos": observation["joint_action"]["vector"],
            "point_cloud": observation["pointcloud"],
        }
        self.policy.update_obs(obs)

    def get_action(self, observation: Optional[Dict] = None) -> np.ndarray:
        """
        与 dp3_policy.DP3.get_action 相同接口：
            - 如果 observation 不为 None，会先 update_obs 再推理
            - 返回: [n_action_steps, action_dim] 的 numpy 数组
        """
        if observation is not None:
            self.update_obs(observation)
        return self.policy.get_action()


__all__ = [
    "CombinedEndPoseGNNPolicy",
    "CombinedDP3",
    "load_dp3_endpose_policy",
    "load_dp3_gnn_endpose_policy",
]


