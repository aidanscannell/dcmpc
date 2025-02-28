#!/usr/bin/env python3
import gymnasium as gym
from dm_control import suite
from envs.tasks import ball_in_cup, pendulum
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from omegaconf import OmegaConf
from torchrl.envs import GymEnv, ParallelEnv, SerialEnv, StepCounter, TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    PermuteTransform,
    RenameTransform,
    Resize,
    RewardSum,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.record import VideoRecorder

from .dmcontrol import make_env as dmcontrol_make_env
from .metaworld import make_env as metaworld_make_env
from .myosuite import make_env as myosuite_make_env
from .myosuite import MYOSUITE_TASKS


suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks("custom")
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)


def make_env(
    cfg, num_envs: int = 1, record_video: bool = False, tag: str = "eval", logger=None
):
    if record_video or not cfg.vec_env:
        # assert num_envs == 1
        env = SerialEnv(
            num_envs,
            make_env_fn,
            create_env_kwargs={"cfg": cfg, "record_video": record_video},
            serial_for_single=True,
            device=cfg.device,  # puts policy on device
        )
    else:
        env = ParallelEnv(
            num_envs,
            make_env_fn,
            create_env_kwargs={"cfg": cfg},
            serial_for_single=True,
            device=cfg.device,  # puts policy on device
        )
        try:
            env.set_seed(cfg.seed)
        except AttributeError:
            env = SerialEnv(
                num_envs,
                make_env_fn,
                create_env_kwargs={"cfg": cfg},
                serial_for_single=True,
                device=cfg.device,  # puts policy on device
            )

    env.set_seed(cfg.seed)

    if record_video:
        video_rec_in_keys = "pixels"
        render_size = cfg.get("render_size", 64)
        env = env.append_transform(
            Compose(
                ToTensorImage(from_int=False, in_keys="pixels"),
                Resize(render_size, render_size),
                PermuteTransform((-2, -1, -3), in_keys=["pixels"]),
                VideoRecorder(
                    logger=logger, tag=tag, skip=4, in_keys=video_rec_in_keys
                ),
            ),
        )

    # In DMControl set cfg.agent.r_min=0 and cfg.agent.r_max=1*action_repeat
    if (cfg.env_name, cfg.task_name) in suite.ALL_TASKS or cfg.env_name == "cup":
        OmegaConf.update(cfg.agent, "r_min", 0.0)
        OmegaConf.update(cfg.agent, "r_max", 1.0 * cfg.action_repeat)
    else:
        OmegaConf.update(cfg.agent, "r_min", None)
        OmegaConf.update(cfg.agent, "r_max", None)

    return env


def make_env_fn(cfg, record_video: bool = False):
    if cfg.env_name in gym.envs.registry.keys():
        env = GymEnv(
            env_name=cfg.env_name, frame_skip=cfg.action_repeat, device=cfg.env_device
        )
    elif (cfg.env_name, cfg.task_name) in suite.ALL_TASKS or cfg.env_name == "cup":
        env = dmcontrol_make_env(
            env_name=cfg.env_name,
            task_name=cfg.task_name,
            from_pixels=record_video,
            frame_skip=cfg.action_repeat,
            device=cfg.env_device,
        )
    elif (
        cfg.env_name.split("mw-", 1)[-1] + "-v2-goal-observable"
        in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    ):
        env = metaworld_make_env(
            env_name=cfg.env_name.split("-", 1)[-1] + "-v2-goal-observable",
            from_pixels=record_video,
            seed=cfg.seed,
            frame_skip=cfg.action_repeat,
            device=cfg.env_device,
            max_episode_steps=cfg.max_episode_steps,
        )
    elif cfg.env_name in MYOSUITE_TASKS:
        env = myosuite_make_env(
            env_name=cfg.env_name,
            from_pixels=record_video,
            seed=cfg.seed,
            frame_skip=cfg.action_repeat,
            device=cfg.env_device,
            max_episode_steps=cfg.max_episode_steps,
        )

    env = TransformedEnv(
        env,
        Compose(
            RenameTransform(in_keys=["observation"], out_keys=["state"]),
            RenameTransform(in_keys=["state"], out_keys=[("observation", "state")]),
            DoubleToFloat(),
            StepCounter(),
            RewardSum(),
        ),
    )
    return env
