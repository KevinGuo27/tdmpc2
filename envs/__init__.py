# This method will be used to create the environemt for the agent to interact with
import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def get_env(env_id):
    def thunk():
        seed = random.randint(0, 2 ** 32 - 1)
        run_name = f"{env_id}-{int(time.time())}"
        video_path = f"videos/{run_name}"
        os.makedirs(video_path, exist_ok=True)  # Ensure the directory exists
        # Create environment with video recording
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_path)
        # record episode statistics
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # auto reset
        env = gym.wrappers.AutoResetWrapper(env)
        # clip actions
        env = gym.wrappers.ClipAction(env)
        # resize
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        # normalize
        env = gym.wrappers.NormalizeObservation(env)
        # grayscale
        env = gym.wrappers.GrayScaleObservation(env)
        # framestack
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk
