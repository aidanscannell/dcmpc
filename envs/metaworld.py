#!/usr/bin/env python3
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from torchrl.envs import default_info_dict_reader, GymWrapper


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat: int = 2):
        super().__init__(env)
        self.env = env
        self.action_repeat = action_repeat
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = obs.astype(np.float32)
        self.env.step(np.zeros(self.env.action_space.shape))
        return obs, info

    def step(self, action):
        reward = 0
        success = False
        for _ in range(self.action_repeat):
            obs, r, terminated, truncated, info = self.env.step(action.copy())
            success = success or info["success"]
            reward += r
            if terminated or truncated:
                break
        obs = obs.astype(np.float32)
        info.update({"success": success})
        return (obs, reward, terminated, truncated, info)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render(
            offscreen=True, resolution=(384, 384), camera_name=self.camera_name
        ).copy()


def make_env(
    env_name: str,
    from_pixels: bool = True,
    seed: int = 42,
    frame_skip: int = 2,
    pixels_only: bool = False,
    record_video: bool = False,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,  # if None defaults to 500
):
    assert record_video == False
    assert from_pixels == False
    """Make Meta-World environment."""
    if max_episode_steps is None:
        max_episode_steps = 200
    # assert cfg.obs == "state", "This task only supports state observations."
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed)
    env = MetaWorldWrapper(env, action_repeat=frame_skip)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    # env.max_episode_steps = env._max_episode_steps

    reader = default_info_dict_reader(["success"])
    env = GymWrapper(
        env=env,
        # TODO metaworld doesn't work with from_pixels=True
        from_pixels=from_pixels or record_video,
        # frame_skip=frame_skip, # frame_skip is handled by MetaWorldWrapper
        # pixels_only=pixels_only,
        device=device,
    ).set_info_dict_reader(info_dict_reader=reader)
    return env
