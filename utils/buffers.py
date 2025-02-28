import logging
from typing import NamedTuple, Optional

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer as TorchRLReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


logger = logging.getLogger(__name__)


class ReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    terminateds: torch.Tensor
    rewards: torch.Tensor
    next_state_gammas: torch.Tensor
    zs: Optional[TensorDict]
    next_zs: Optional[TensorDict]


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        nstep: int = 1,
        gamma: float = 0.99,
        prefetch: int = 10,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        if device == "cpu":
            pin_memory = False
            logger.info(f"On CPU so setting pin_memory=False")

        self.nstep = nstep
        self.gamma = gamma
        self.sampler = SliceSampler(
            slice_len=nstep,
            end_key=None,
            traj_key=("collector", "traj_ids"),
        )
        self.rb = TorchRLReplayBuffer(
            storage=LazyMemmapStorage(buffer_size, device="cpu"),
            pin_memory=pin_memory,
            sampler=self.sampler,
            prefetch=prefetch,
            batch_size=batch_size * nstep,
            transform=lambda x: x.to(device),
        )
        self.batch_size = batch_size

    def extend(self, data):
        self.rb.extend(data.cpu())

    def sample(
        self, return_nstep: bool = False, batch_size: Optional[int] = None
    ) -> ReplayBufferSamples:
        batch = self._sample()
        if batch_size is not None and batch_size > self.batch_size:
            # TODO Fix this hack to get larger batch size
            # If requesting large batch size sample multiple times and concat
            for _ in range(batch_size // self.batch_size):
                batch = torch.cat([batch, self._sample()], 1)
            batch = batch[:, :batch_size]
        batch = ReplayBufferSamples(
            observations=batch["observation"],
            actions=batch["action"],
            next_observations=batch["next"]["observation"],
            dones=batch["next"]["done"][..., 0],
            terminateds=batch["next"]["terminated"][..., 0].to(torch.int),
            rewards=batch["next"]["reward"][..., 0],
            next_state_gammas=batch["next_state_gammas"],
            zs=None,
            next_zs=None,
        )
        if not return_nstep:
            return batch
        else:
            return to_nstep(batch, nstep=self.nstep, gamma=self.gamma)

    def _sample(self) -> TensorDict:
        batch = self.rb.sample().view(-1, self.nstep).transpose(0, 1)
        next_state_gammas = torch.ones_like(
            batch["next"]["done"][..., 0], dtype=torch.float32
        )
        batch.update({"next_state_gammas": next_state_gammas}, inplace=True)
        return batch


@torch.no_grad()
def to_nstep(
    batch: ReplayBufferSamples, nstep: int, gamma: float = 0.99
) -> ReplayBufferSamples:
    """Form n-step samples (truncate if timeout)"""
    if nstep > 1:
        dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
        terminateds = torch.zeros_like(batch.terminateds[0], dtype=torch.bool)
        rewards = torch.zeros_like(batch.rewards[0])
        next_state_gammas = torch.ones_like(batch.dones[0], dtype=torch.float32)
        next_obs = torch.zeros_like(batch.observations[0])
        next_zs = (
            torch.zeros_like(batch.next_zs[0]) if batch.next_zs is not None else None
        )
        for t in range(nstep):
            next_obs = torch.where(
                dones[..., None], next_obs, batch.next_observations[t]
            )
            if next_zs is not None:
                next_zs = torch.where(dones[..., None], next_zs, batch.next_zs[t])
            dones = torch.logical_or(dones, batch.dones[t])
            next_state_gammas *= torch.where(dones, 1, gamma)
            terminateds *= torch.where(
                dones, terminateds, torch.logical_or(terminateds, batch.terminateds[t])
            )
            rewards += torch.where(dones, 0, gamma**t * batch.rewards[t])
        nstep_batch = ReplayBufferSamples(
            observations=batch.observations[0],
            actions=batch.actions[0],
            next_observations=next_obs,
            dones=dones.to(torch.int),
            terminateds=terminateds.to(torch.int),
            rewards=rewards,
            next_state_gammas=next_state_gammas,
            zs=batch.zs[0] if batch.zs is not None else None,
            next_zs=next_zs,
        )
    else:
        # TODO Can remove this else
        nstep_batch = ReplayBufferSamples(
            observations=batch.observations[0],
            actions=batch.actions[0],
            next_observations=batch.next_observations[0],
            dones=batch.dones[0],
            terminateds=batch.terminateds[0],
            rewards=batch.rewards[0],
            next_state_gammas=batch.next_state_gammas[0],
            zs=batch.zs[0] if batch.zs is not None else None,
            next_zs=batch.next_zs[0] if batch.next_zs is not None else None,
        )
    return nstep_batch


def flatten_batch(batch: ReplayBufferSamples) -> ReplayBufferSamples:
    return batch._replace(
        observations=batch.observations.flatten(),
        actions=batch.actions.flatten(0, 1),
        next_observations=batch.next_observations.flatten(),
        dones=batch.dones.flatten(0, 1),
        terminateds=batch.terminateds.flatten(0, 1),
        rewards=batch.rewards.flatten(),
        next_state_gammas=batch.next_state_gammas.flatten(),
        zs=batch.zs.flatten() if batch.zs is not None else batch.zs,
        next_zs=batch.next_zs.flatten() if batch.next_zs is not None else batch.next_zs,
    )
