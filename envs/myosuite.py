#!/usr/bin/env python3
from typing import Optional
import numpy as np
from gymnasium.wrappers import TimeLimit
from gymnasium import Wrapper
from torchrl.envs import default_info_dict_reader, GymWrapper
from myosuite.utils import gym as gym_utils


MYOSUITE_TASKS = {
    "myo-reach": "myoHandReachFixed-v0",
    "myo-reach-hard": "myoHandReachRandom-v0",
    "myo-pose": "myoHandPoseFixed-v0",
    "myo-pose-hard": "myoHandPoseRandom-v0",
    "myo-obj-hold": "myoHandObjHoldFixed-v0",
    "myo-obj-hold-hard": "myoHandObjHoldRandom-v0",
    "myo-key-turn": "myoHandKeyTurnFixed-v0",
    "myo-key-turn-hard": "myoHandKeyTurnRandom-v0",
    "myo-pen-twirl": "myoHandPenTwirlFixed-v0",
    "myo-pen-twirl-hard": "myoHandPenTwirlRandom-v0",
}


class MyoSuiteWrapper(Wrapper):
    def __init__(self, env, action_repeat: int = 2, **kwargs):
        super().__init__(env)
        self.env = env
        self.camera_id = "hand_side_inter"
        self.action_repeat = action_repeat

    def reset(self):
        obs, info = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        reward = 0
        success = False
        for _ in range(self.action_repeat):
            obs, r, terminated, truncated, info = self.env.step(action.copy())
            info["success"] = info["solved"]
            success = success or info["success"]
            reward += r
            if terminated or truncated:
                break
        obs = obs.astype(np.float32)
        info.update({"success": success})
        return (obs, reward, False, False, info)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.unwrapped.sim.renderer.render_offscreen(
            width=384, height=384, camera_id=self.camera_id
        ).copy()


def make_env(
    env_name: str,
    from_pixels: bool = False,
    seed: int = 42,
    frame_skip: int = 1,
    record_video: bool = False,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,  # if None defaults to 100
):
    assert record_video == False
    assert from_pixels == False
    """Make MyoSuite environment."""
    if max_episode_steps is None:
        max_episode_steps = 100
    # assert cfg.obs == "state", "This task only supports state observations."
    env = gym_utils.make(MYOSUITE_TASKS[env_name], seed=seed)

    env = MyoSuiteWrapper(
        env,
        action_repeat=frame_skip,
    )
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
