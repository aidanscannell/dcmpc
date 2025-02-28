#!/usr/bin/env python3
from typing import Optional

from dm_control import suite
from envs.tasks import ball_in_cup, pendulum
from torchrl.envs import DMControlEnv, DMControlWrapper, TransformedEnv
from torchrl.envs.transforms import CatTensors, TransformedEnv


suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks("custom")
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)


def make_env(
    env_name: str,
    task_name: Optional[str] = None,
    from_pixels: bool = True,
    frame_skip: int = 2,
    pixels_only: bool = False,
    record_video: bool = False,
    device: str = "cpu",
):
    if env_name == "cup":
        env_name = "ball_in_cup"
    camera_id = dict(quadruped=2).get(env_name, 0)
    try:
        env = DMControlEnv(
            env_name=env_name,
            task_name=task_name,
            from_pixels=from_pixels or record_video,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
            camera_id=camera_id,
        )
    except RuntimeError:
        env = suite.load(env_name, task_name)
        env = DMControlWrapper(
            env,
            from_pixels=from_pixels or record_video,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
            camera_id=camera_id,
        )
    if not pixels_only:
        # Put "position"/"velocity"/"orientation" into "observation"
        obs_keys = [key for key in env.observation_spec.keys()]
        if from_pixels or record_video:
            obs_keys.remove("pixels")
        env = TransformedEnv(env, CatTensors(in_keys=obs_keys, out_key="observation"))

    return env
