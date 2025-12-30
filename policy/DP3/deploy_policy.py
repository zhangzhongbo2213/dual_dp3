# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

from dp3_policy import *
from combined_policy import CombinedDP3


def encode_obs(observation):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']
    return obs


def get_model(usr_args):
    """
    加载策略模型：
      - 默认行为：与原版一致，加载 DP3 模型（通过 TrainDP3Workspace）
      - 当 usr_args['combined_mode'] 为 True 时：加载 CombinedDP3（EndPose + GNN-EndPose 串联）
    """

    # ====== Combined 模式：EndPose + GNN-EndPose 串联策略 ======
    if usr_args.get("combined_mode", False):
        task_name = usr_args["task_name"]
        expert_data_num = usr_args["expert_data_num"]
        seed = usr_args["seed"]
        # 串联策略使用单独的 checkpoint 配置，默认为 300 轮
        checkpoint_num = usr_args.get("combined_checkpoint_num", 300)
        ckpt_setting = usr_args["ckpt_setting"]

        # 如果显式给了 ckpt 路径就直接用，否则按约定自动拼接
        endpose_ckpt_path = usr_args.get("endpose_ckpt_path")
        gnn_ckpt_path = usr_args.get("gnn_endpose_ckpt_path")

        if endpose_ckpt_path is None:
            # 目录命名与训练脚本保持一致: {task_name}-{ckpt_setting}-{expert_data_num}-endpose_{seed}
            endpose_dir = f"{task_name}-{ckpt_setting}-{expert_data_num}-endpose_{seed}"
            endpose_ckpt_path = os.path.join(
                parent_directory, "checkpoints", endpose_dir, f"{checkpoint_num}.ckpt"
            )
        if gnn_ckpt_path is None:
            gnn_dir = f"{task_name}-{ckpt_setting}-{expert_data_num}-gnn-endpose_{seed}"
            gnn_ckpt_path = os.path.join(
                parent_directory, "checkpoints", gnn_dir, f"{checkpoint_num}.ckpt"
            )

        # 设备：eval.sh 里已经通过 CUDA_VISIBLE_DEVICES 选择了 GPU，这里使用 cuda:0 即可
        device = "cuda:0"

        model = CombinedDP3(
            endpose_ckpt_path=endpose_ckpt_path,
            gnn_ckpt_path=gnn_ckpt_path,
            device=device,
            n_obs_steps=3,
            n_action_steps=6,
        )
        return model

    # ====== 原始 DP3 模式（保持不变） ======
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args['use_rgb']
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    return DP3_Model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()
