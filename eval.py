#!/usr/bin/env python3
import hydra
from config import EvalConfig


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="eval")
def eval_checkpoint(cfg: EvalConfig):
    import logging
    import random

    import numpy as np
    import torch
    import wandb
    from dcmpc import DCMPC
    from envs import make_env
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd
    from omegaconf import OmegaConf
    from tensordict.nn import TensorDictModule
    from torchrl.record.loggers.wandb import WandbLogger
    from utils import evaluate

    logger = logging.getLogger(__name__)

    ##### Load the TrainConfig used to train the checkpoint #####
    if cfg.wandb_id:
        """Load from W&B ID"""
        api = wandb.Api(timeout=19)
        run = api.run(cfg.wandb_id)
        filename = "checkpoint.pt"
        cfg.checkpoint = filename
        checkpoint = filename
        train_cfg = OmegaConf.create(run.config)
        run.file(filename).download(replace=True)
    else:
        """Load local path"""
        train_cfg_path = f"{get_original_cwd()}/{cfg.checkpoint.split('checkpoint')[0]}"
        checkpoint = train_cfg_path
        train_cfg = OmegaConf.load(f"{train_cfg_path}/.hydra/config.yaml")
    logger.info(f"Env: {train_cfg.env_name} {train_cfg.task_name}")

    ###### Fix seed for reproducibility ######
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available() and (cfg.device == "cuda"):
        cfg.device = "cuda"
        train_cfg.device = "cuda"
    else:
        cfg.device = "cpu"
        train_cfg.device = "cpu"
        train_cfg.agent.device = "cpu"

    ###### Initialise W&B ######
    writer = WandbLogger(
        exp_name=cfg.run_name,
        offline=not cfg.use_wandb,
        project=cfg.wandb_project_name,
        group=f"{train_cfg.env_name}-{train_cfg.task_name}",
        tags=[
            f"{train_cfg.env_name}-{train_cfg.task_name}",
            f"seed={str(train_cfg.seed)}",
        ],
        save_code=True,
    )
    writer.log_hparams(cfg)
    writer.log_hparams(
        {
            "train": OmegaConf.to_container(train_cfg, throw_on_missing=False),
            "hydra": OmegaConf.to_container(HydraConfig.get(), throw_on_missing=False),
        }
    )

    ###### Setup vectorized environment for training/evaluation/video recording ######
    eval_env = make_env(train_cfg, num_envs=cfg.num_eval_episodes)
    train_cfg.update(render_size=cfg.render_size)

    # hack to work with new model
    train_cfg.agent.use_top_k = True
    train_cfg.agent.use_mppi_mean = False

    # train_cfg.update(max_episode_steps=250)
    # train_cfg.agent.update(mpc=False)
    video_env = make_env(
        train_cfg,
        num_envs=1,
        record_video=cfg.capture_eval_video,
        tag="eval",
        logger=writer,
    )

    ###### Init agent ######
    agent = DCMPC(
        train_cfg.agent,
        obs_spec=eval_env.observation_spec["observation"][0],
        act_spec=eval_env.action_spec[0],
    ).to(cfg.device)

    # Load state dict into this agent from filepath (or dictionary)
    if cfg.checkpoint is not None:
        state_dict = torch.load(
            checkpoint,
            weights_only=True,
            map_location=torch.device(cfg.device),
        )
        agent.load_state_dict(state_dict["model"])
        logger.info(f"Loaded checkpoint from {cfg.checkpoint}")

    ##### Evaluate the agent #####
    eval_metrics = evaluate(
        env=eval_env,
        eval_policy_module=TensorDictModule(
            lambda obs, step_count: agent.select_action(
                obs, t0=step_count[0] == 0, eval_mode=True
            ),
            in_keys=["observation", "step_count"],
            out_keys=["action"],
        ),
        max_episode_steps=train_cfg.max_episode_steps,
        action_repeat=train_cfg.action_repeat,
        video_env=video_env,
    )

    ##### Log metrics to W&B or csv #####
    writer.log_scalar(name="eval/", value=eval_metrics)
    logger.info(f"Eval return {eval_metrics['episodic_return']:.2f}")


if __name__ == "__main__":
    eval_checkpoint()  # pyright: ignore
