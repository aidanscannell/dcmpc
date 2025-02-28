#!/usr/bin/env python3
import os

import hydra
from config import TrainConfig


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train")
def cluster_safe_train(cfg: TrainConfig):
    """Wrapper to ensure errors are logged properly when using hydra's submitit launcher

    This wrapper function is used to circumvent this bug in Hydra
    See https://github.com/facebookresearch/hydra/issues/2664
    """
    import sys
    import traceback

    try:
        train(cfg)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


def train(cfg: TrainConfig):
    import logging
    import random
    import time

    import numpy as np
    import torch
    import utils.helper as h
    from dcmpc import DCMPC
    from envs import make_env
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd
    from omegaconf import OmegaConf
    from tensordict.nn import TensorDictModule
    from termcolor import colored
    from torchrl.data.tensor_specs import BoundedTensorSpec
    from torchrl.record.loggers.wandb import WandbLogger
    from utils import evaluate, ReplayBuffer

    logger = logging.getLogger(__name__)

    ###### Fix seed for reproducibility ######
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available() and (cfg.device == "cuda"):
        cfg.device = "cuda"
    else:
        cfg.device = "cpu"

    ###### Initialise W&B ######
    os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
    writer = WandbLogger(
        exp_name=cfg.run_name,
        offline=not cfg.use_wandb,
        project=cfg.wandb_project_name,
        group=f"{cfg.env_name}-{cfg.task_name}",
        tags=[f"{cfg.env_name}-{cfg.task_name}", f"seed={str(cfg.seed)}"],
        save_code=True,
    )

    ###### Setup vectorized environment for training/evaluation/video recording ######
    env = make_env(cfg, num_envs=1)
    eval_env = make_env(cfg, num_envs=cfg.num_eval_episodes)
    if cfg.capture_eval_video:
        video_env = make_env(
            cfg,
            num_envs=1,
            record_video=cfg.capture_eval_video,
            tag="eval",
            logger=writer,
        )
    assert isinstance(
        env.action_spec, BoundedTensorSpec
    ), "only continuous act space supported"

    writer.log_hparams(cfg)
    writer.log_hparams(
        {"hydra": OmegaConf.to_container(HydraConfig.get(), throw_on_missing=False)}
    )

    ###### Prepare replay buffer ######
    rb = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        batch_size=cfg.agent.get("batch_size", 512),
        nstep=max(cfg.agent.get("nstep", 1), cfg.agent.get("horizon", 1)),
        gamma=cfg.agent.get("gamma", 0.99),
        prefetch=cfg.prefetch,
        pin_memory=True,  # will be set to False if device=="cpu"
        device=cfg.device,
    )

    ###### Init agent ######
    agent = DCMPC(
        cfg.agent,
        obs_spec=env.observation_spec["observation"][0],
        act_spec=env.action_spec[0],
    ).to(cfg.device)
    if cfg.checkpoint is not None:
        # Load state dict into this agent from filepath (or dictionary)
        state_dict = torch.load(
            os.path.join(get_original_cwd(), cfg.checkpoint), weights_only=True
        )
        agent.load_state_dict(state_dict["model"])
        logger.info(f"Loaded checkpoint from {cfg.checkpoint}")

    ##### Print information about run #####
    h.print_run(cfg, env)
    total_params = int(agent.total_params / 1e6)
    writer.log_hparams({"total_params": agent.total_params})
    writer.log_hparams({"world_model_params": agent.model.total_params})
    print(
        colored("Learnable parameters:", "yellow", attrs=["bold"]), f"{total_params}M"
    )
    print(colored("Architecture:", "yellow", attrs=["bold"]), agent)

    def build_policy_module(eval_mode):
        return TensorDictModule(
            lambda obs, step_count: agent.select_action(
                obs, t0=step_count[0] == 0, eval_mode=eval_mode
            ),
            in_keys=["observation", "step_count"],
            out_keys=["action"],
        )

    policy_module = build_policy_module(eval_mode=False)
    eval_policy_module = build_policy_module(eval_mode=True)

    def evaluate_and_log(best_episode_reward: float = 0.0):
        eval_metrics = evaluate(
            env=eval_env,
            eval_policy_module=eval_policy_module,
            max_episode_steps=cfg.max_episode_steps,
            action_repeat=cfg.action_repeat,
            video_env=video_env if cfg.capture_eval_video else None,
        )

        ##### Eval metrics #####
        eval_metrics.update(
            {
                "elapsed_time": time.time() - start_time,
                "SPS": int(step / (time.time() - start_time)),
                "env_step": step * cfg.action_repeat,
                "step": step,
                "episode": episode_idx,
                "train_time": train_time,
                "rollout_time": rollout_time,
            }
        )

        if cfg.verbose:
            h.print_metrics(cfg, episode_idx, step, eval_metrics, eval_mode=True)

        ##### Log rank of latent and active codebook percent #####
        batch = rb.sample(batch_size=agent.model.cfg.latent_dim)
        eval_metrics.update(agent.metrics(batch))

        ##### Log metrics to W&B or csv #####
        writer.log_scalar(name="eval/", value=eval_metrics)

        ##### Save model checkpoint #####
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            ckpt_metrics = {
                "episodic_return": eval_metrics["episodic_return"],
                "env_step": step * cfg.action_repeat,
                "step": step,
                "episode": episode_idx,
            }
            ckpt_path = "./checkpoint.pt"
            agent.save(path=ckpt_path, metrics=ckpt_metrics)
            writer.experiment.save(ckpt_path, policy="now")

        return eval_metrics, best_episode_reward

    step, start_time, train_time = 0, time.time(), 0
    for episode_idx in range(1, cfg.num_episodes + 1):
        episode_start_time = time.time()
        ##### Rollout the policy in the environment #####
        with torch.no_grad():
            data = env.rollout(
                max_steps=cfg.max_episode_steps // cfg.action_repeat,
                policy=None if episode_idx <= cfg.random_episodes else policy_module,
            )[0]
        rollout_time = time.time() - episode_start_time

        if cfg.scale_reward:
            data["next"]["reward"] = h.symlog(data["next"]["reward"])

        ##### Add data to the replay buffer #####
        rb.extend(data)
        episode_reward = data["next"]["episode_reward"][-1].cpu().item()

        if episode_idx == 1:
            print(colored("First episodes data:", "green", attrs=["bold"]), data)

            # Evaluate the initial agent
            _, best_episode_reward = evaluate_and_log(best_episode_reward=0)

        ##### Log episode metrics #####
        episode_len = data["next"]["step_count"][-1].cpu().item()
        step += episode_len
        rollout_metrics = {
            "episodic_return": episode_reward,
            "episodic_length": episode_len,
            "env_step": step * cfg.action_repeat,
        }
        success = data["next"].get("success", None)
        if success is not None:
            episode_success = success.any()
            if isinstance(episode_success, torch.Tensor):
                episode_success = episode_success.item()
            rollout_metrics.update({"episodic_success": int(episode_success)})

        if cfg.verbose:
            h.print_metrics(cfg, episode_idx, step, rollout_metrics, eval_mode=False)

        writer.log_scalar(name="rollout/", value=rollout_metrics)

        ##### Train agent (after collecting some random episodes) #####
        if episode_idx >= cfg.random_episodes:
            if episode_idx == cfg.random_episodes:
                episode_len = (
                    cfg.random_episodes * cfg.max_episode_steps // cfg.action_repeat
                )
                print("Pretraining...")
            train_start_time = time.time()
            train_metrics = agent.update(
                replay_buffer=rb, num_new_transitions=episode_len
            )
            train_time = time.time() - train_start_time

            ##### Log training metrics #####
            writer.log_scalar(name="train/", value=train_metrics)

        ###### Evaluate ######
        if episode_idx % cfg.eval_every_episodes == 0:
            _, best_episode_reward = evaluate_and_log(best_episode_reward)

        # Release some GPU memory (if possible)
        torch.cuda.empty_cache()

    ##### Evaluate the final agent #####
    _ = evaluate_and_log(best_episode_reward)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    cluster_safe_train()  # pyright: ignore
