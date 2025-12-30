import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import zarr
import pdb


class RobotDataset(BaseDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
    ):
        super().__init__()
        self.task_name = task_name
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        
        # 兼容无 state 的 EndPose 数据：动态检查可用键
        zarr_root = zarr.open(zarr_path, mode="r")
        data_group = zarr_root["data"] if "data" in zarr_root else {}

        has_state = "state" in data_group
        has_endpose_future = (
            "left_endpose_future" in data_group and "right_endpose_future" in data_group
        )

        keys = ["action", "point_cloud"]
        if has_state:
            keys = ["state"] + keys
        if has_endpose_future:
            keys = keys + ["left_endpose_future", "right_endpose_future"]

        self.has_state = has_state
        self.has_endpose_future = has_endpose_future
        
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "point_cloud": self.replay_buffer["point_cloud"],
        }

        if self.has_state:
            data["agent_pos"] = self.replay_buffer["state"][..., :]
        
        # 如果存在endpose future数据，添加到normalizer
        if self.has_endpose_future:
            data["left_endpose_future"] = self.replay_buffer["left_endpose_future"]
            data["right_endpose_future"] = self.replay_buffer["right_endpose_future"]
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        point_cloud = sample["point_cloud"][
            :,
        ].astype(np.float32)  # (T, 1024, 6)

        data = {
            "obs": {
                "point_cloud": point_cloud,  # T, 1024, 6
            },
            "action": sample["action"].astype(np.float32),  # T, D_action
        }

        if self.has_state:
            agent_pos = sample["state"][
                :,
            ].astype(np.float32)  # (agent_posx2, block_posex3)
            data["obs"]["agent_pos"] = agent_pos  # T, D_pos
        
        # 如果存在endpose future数据，添加到obs中
        if self.has_endpose_future:
            if "left_endpose_future" in sample:
                # endpose future数据格式说明:
                # - 在zarr中存储为: [N, 6, 4] (N个样本，每个6帧，每帧4维)
                # - SequenceSampler采样后: [T, 6, 4] (T是horizon=8)
                # - GNN模型需要: [B, 6, 4] (batch维度在DataLoader中添加)
                #
                # 因为endpose future是每个观测帧对应的未来6帧EndPose，不随时间变化
                # 所以我们需要取第一个时间步的endpose future
                left_ep_future = sample["left_endpose_future"]
                right_ep_future = sample["right_endpose_future"]
                
                # 处理维度：
                # - 如果采样后是[T, 6, 4]，取第一个时间步得到[6, 4]
                # - 如果已经是[6, 4]，直接使用
                if left_ep_future.ndim == 3:
                    # 序列采样后是[T, 6, 4]，取第一个时间步
                    # 因为endpose future是每个观测帧固定的，不随时间变化
                    left_ep_future = left_ep_future[0]  # [6, 4]
                    right_ep_future = right_ep_future[0]  # [6, 4]
                elif left_ep_future.ndim == 2:
                    # 如果已经是[6, 4]，直接使用
                    pass
                else:
                    raise ValueError(f"Unexpected endpose future shape: {left_ep_future.shape}, expected [T, 6, 4] or [6, 4]")
                
                # 添加到obs中，DataLoader会添加batch维度，变成[B, 6, 4]
                data["obs"]["left_endpose_future"] = left_ep_future.astype(np.float32)
                data["obs"]["right_endpose_future"] = right_ep_future.astype(np.float32)
        
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
