#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Any, List, Optional

from dcmpc import DCMPCConfig
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from utils import LUMIConfig, PUHTIConfig, SlurmConfig, TritonConfig


@dataclass
class TrainConfig:
    """Training config used in train.py"""

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"agent": "dcmpc"},
            {"env": "dog-run"},  # envs are specified in cfgs/env/
            # Use submitit to launch slurm jobs on cluster w/ multirun
            {"override hydra/launcher": "slurm"},
            {"override hydra/job_logging": "colorlog"},  # Make logging colourful
            {"override hydra/hydra_logging": "colorlog"},  # Make logging colourful
        ]
    )

    # Configure environment (overridden by defaults list)
    env_name: str = MISSING
    task_name: str = MISSING
    vec_env: bool = True

    # Agent (overridden by DCMPC in defaults list)
    agent: DCMPCConfig = field(default_factory=DCMPCConfig)

    # Experiment
    max_episode_steps: int = 1000  # Max episode length
    num_episodes: int = 4000  # Number of training episodes 4000*500*2=4M steps
    random_episodes: int = 10  # Number of random episodes at start
    action_repeat: int = 2
    buffer_size: int = 10_000_000
    prefetch: int = 5
    seed: int = 42
    checkpoint: Optional[str] = None  # /file/path/to/checkpoint
    device: str = "cuda"  # "cpu" or "cuda" etc
    env_device: str = "cpu"  # DMControl/MetaWorld on cpu but maniskill/isaac on cuda
    verbose: bool = False  # if true print training progress

    scale_reward: bool = False  # it true scale rewards using symlog

    # Evaluation
    eval_every_episodes: int = 25  # every 25k env steps
    num_eval_episodes: int = 10
    capture_eval_video: bool = False  # Fails on AMD GPU so set to False

    # W&B config
    use_wandb: bool = False
    wandb_silent: bool = True
    wandb_project_name: str = "DCWM"
    run_name: str = "dcmpc-${now:%Y-%m-%d_%H-%M-%S}"

    # Override the Hydra config to get better dir structure with W&B
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "output/hydra/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"},
            "verbose": False,
            "job": {"chdir": True},
            "sweep": {"dir": "${hydra.run.dir}", "subdir": "${hydra.job.num}"},
        }
    )


@dataclass
class EvalConfig:
    """Config for eval.py - loads ckpt, creates high-res video, and logs eval metrics"""

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            # Use submitit to launch slurm jobs on cluster w/ multirun
            {"override hydra/launcher": "slurm"},
            {"override hydra/job_logging": "colorlog"},  # Make logging colourful
            {"override hydra/hydra_logging": "colorlog"},  # Make logging colourful
        ]
    )

    checkpoint: str = "output/hydra/train/2024-11-24_13-41-38/checkpoint"
    num_eval_episodes: int = 10
    capture_eval_video: bool = True
    render_size: int = 1080  # height/width of pixel observations
    device: str = "cuda"

    wandb_id: Optional[str] = None

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = "DCWM-eval"
    run_name: str = "dcmpc-eval-${now:%Y-%m-%d_%H-%M-%S}"

    # Override the Hydra config to get better dir structure with W&B
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "output/hydra/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"},
            "verbose": False,
            "job": {"chdir": True},
            "sweep": {"dir": "${hydra.run.dir}", "subdir": "${hydra.job.num}"},
        }
    )


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="dcmpc", group="agent", node=DCMPCConfig)
cs.store(name="slurm", group="hydra/launcher", node=SlurmConfig)
cs.store(name="lumi", group="hydra/launcher", node=LUMIConfig)
cs.store(name="puhti", group="hydra/launcher", node=PUHTIConfig)
cs.store(name="triton", group="hydra/launcher", node=TritonConfig)
cs.store(name="eval", node=EvalConfig)


#####################
# Experiment Configs
#####################


@dataclass
class ContinuousMSEConfig(DCMPCConfig):
    """Config for Continuous+MSE experiment"""

    """Flag to turn FSQ off"""
    use_fsq: bool = False
    """Which loss function to use for consistency loss?"""
    consistency_loss: str = "mse"  # "cross-entropy", "mse", "cosine"


@dataclass
class DiscreteMSEConfig(DCMPCConfig):
    """Config for Discrete+MSE experiment"""

    """Which loss function to use for consistency loss?"""
    consistency_loss: str = "mse"  # "cross-entropy", "mse", "cosine"


@dataclass
class DiscreteCEDetConfig(DCMPCConfig):
    """Config for Discrete+CE+Det experiment"""

    """Predict logits with mse between pred and codebook"""
    ce_logits_mode: str = "mse"  # "standard", cosine", "mse"


cs.store(name="continuous_mse", group="agent", node=ContinuousMSEConfig)
cs.store(name="discrete_mse", group="agent", node=DiscreteMSEConfig)
cs.store(name="discrete_ce_det", group="agent", node=DiscreteCEDetConfig)
