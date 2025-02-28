#!/usr/bin/env python3
import time

import torch
from tensordict.nn import TensorDictModule


def evaluate(
    env,
    eval_policy_module: TensorDictModule,
    max_episode_steps: int,
    action_repeat: int = 2,
    video_env=None,
):
    """Calculate avg. episodic return (optionally avg. success)"""
    eval_metrics = {}
    with torch.no_grad():
        eval_start_time = time.time()
        eval_data = env.rollout(
            max_steps=max_episode_steps // action_repeat,
            policy=eval_policy_module,
        )
        eval_episode_time = time.time() - eval_start_time
        eval_episodic_return = torch.mean(eval_data["next"]["episode_reward"][:, -1, 0])
        eval_episodic_return_std = torch.std(
            eval_data["next"]["episode_reward"][:, -1, 0]
        )
        success = eval_data["next"].get("success", None)
        episode_len = eval_data["next"]["step_count"][0, -1, -1]
        if success is not None:
            episodic_success = torch.mean(success.any(-1).to(torch.float))
            eval_metrics.update({"episodic_success": episodic_success})

    ##### Eval metrics #####
    eval_metrics.update(
        {
            "episodic_return": eval_episodic_return,
            "episodic_return_std": eval_episodic_return_std,
            "episode_time": eval_episode_time,
            "episode_len": episode_len,
            "action_repeat": action_repeat,
            "max_episode_steps": max_episode_steps,
        }
    )

    if video_env is not None:
        with torch.no_grad():
            video_env.rollout(
                max_steps=max_episode_steps // action_repeat,
                policy=eval_policy_module,
            )
        video_env.transform.dump()

    return eval_metrics
