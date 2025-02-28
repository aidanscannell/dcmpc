#!/usr/bin/env python3
from dataclasses import dataclass

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf


@dataclass
class SlurmConfig(SlurmQueueConf):
    """
    See here for config options
    https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py
    """

    timeout_min: int = 1440  # 24 hours
    mem_gb: int = 32
    cpus_per_task: int = 5
    name: str = "${env_name}-${task_name}"
    gres: str = "gpu:1"
    stderr_to_stdout: bool = True


@dataclass
class LUMIConfig(SlurmConfig):
    """
    See here for config options
    https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py
    """

    account: str = "project_462000623"
    partition: str = "small-g"  # Partition (queue) name


@dataclass
class PUHTIConfig(SlurmConfig):
    """
    See here for config options
    https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py
    """

    account: str = "project_2007966"
    partition: str = "gpu"  # Partition (queue) name
    gres: str = "gpu:v100:1"


@dataclass
class TritonConfig(SlurmQueueConf):
    """
    See here for config options
    https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_submitit_launcher/hydra_plugins/hydra_submitit_launcher/config.py
    """

    timeout_min: int = 1440  # 24 hours
    mem_gb: int = 32
    cpus_per_task: int = 5
    gres: str = "gpu:1"
    stderr_to_stdout: bool = True
