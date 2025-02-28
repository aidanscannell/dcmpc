#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from torch.linalg import cond, matrix_rank


def print_run(cfg, env):
    """Print information about run"""
    # Create a border
    border = "=" * 50

    def print_aligned(key, value, color="green"):
        key = key + ":"
        print(colored(f"  {key:<20}", color, attrs=["bold"]), f"{value:<30}")

    task = cfg.env_name if cfg.task_name == "" else cfg.env_name + "-" + cfg.task_name
    obs_spec = env.observation_spec["observation"][0]
    act_spec = env.action_spec[0]

    data = [
        ("Task", task),
        ("Steps", f"{(cfg.num_episodes * cfg.max_episode_steps) / 1e6}M"),
        ("Episodes", cfg.num_episodes),
        ("Observations", np.array(obs_spec["state"].shape).prod().item()),
        ("Actions", np.array(act_spec.shape).prod().item()),
        ("Action repeat", cfg.action_repeat),
        ("Device", cfg.device),
    ]

    print(f"\n{border}")

    for row in data:
        print_aligned(row[0], row[1], color="green")

    print(f"{border}")


def print_metrics(
    cfg, episode_idx: int, step: int, metrics: dict, eval_mode: bool = False
):
    prefix = "Eval" if eval_mode else "Train"
    color = "green" if eval_mode else "blue"
    if "success" in metrics.keys():
        success = metrics["success"]
    else:
        success = 0
    episode_return = f"{metrics['episodic_return']:.2f}"
    m = f"{prefix:<7} E: {episode_idx:<10}  I: {step:<12} R: {episode_return:<14} S: {success:<16}"
    print(colored(m, color))


@torch.no_grad()
def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LinearSchedule:
    def __init__(self, start: float, end: float, num_steps: int):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.step_idx = 0
        self.values = torch.linspace(start, end, num_steps)

    def __call__(self):
        return self.values[self.step_idx]

    def step(self):
        if self.step_idx < self.num_steps - 1:
            self.step_idx += 1


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))

@torch.no_grad()
def calc_rank(name, z):
    """Log rank of latent"""
    rank3 = matrix_rank(z, atol=1e-3, rtol=1e-3)
    rank2 = matrix_rank(z, atol=1e-2, rtol=1e-2)
    rank1 = matrix_rank(z, atol=1e-1, rtol=1e-1)
    condition = cond(z)
    info = {}
    full_rank = z.shape[-1]
    for j, rank in enumerate([rank1, rank2, rank3]):
        rank_percent = rank.item() / full_rank * 100
        info.update({f"{name}-rank-{j}": rank.item()})
        info.update({f"{name}-rank-percent-{j}": rank_percent})
    info.update({f"{name}-cond-num": condition.item()})
    return info


def calc_mean_opt_moments(opt):
    first_moment, second_moment = 0, 0
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state[p]
            try:
                first_moment += torch.sum(state["exp_avg"]) / len(state["exp_avg"])
                second_moment += torch.sum(state["exp_avg_sq"]) / len(state["exp_avg"])
            except KeyError:
                pass
    return {"first_moment_mean": first_moment, "second_moment_mean": second_moment}


def soft_update_params(model, model_target, tau: float):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for params, params_target in zip(model.parameters(), model_target.parameters()):
            params_target.data.lerp_(params.data, tau)
