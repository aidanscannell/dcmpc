#!/usr/bin/env python3
import copy
import logging
from functools import cached_property
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import utils.helper as h
import wandb
from einops import einsum, rearrange
from tensordict import TensorDict
from torch import autocast, GradScaler
from torchrl.data import BoundedTensorSpec, CompositeSpec
from utils import ReplayBuffer, ReplayBufferSamples
from utils.layers import FSQ, mlp, mlp_ensemble


logger = logging.getLogger(__name__)


@dataclass
class DCMPCConfig:
    """Discrete Codebook Model Predictive Control Config"""

    """What observation types to use? ["state"] or ["pixels"] or ["state", "pixels"]"""
    obs_types: List[str] = field(default_factory=lambda: ["state"])

    """WORLD MODEL CONFIG"""
    """Size of latent space"""
    latent_dim: int = 512
    """Horizon used for representation learning"""
    horizon: int = 5
    """Discount factor for representation learning"""
    rho: float = 0.9
    """MLP dims for encoder/decoder"""
    enc_mlp_dims: List[int] = field(default_factory=lambda: [256])
    """Learning rate for encoder/dynamics/reward"""
    enc_lr: float = 1e-4
    """Clips the gradient norm of the encoder"""
    grad_clip_norm: Optional[float] = 20
    """Predict change in latent or next latent? i.e. next_z = z + f(z, a) else next_z = f(z, a)"""
    use_delta: bool = False
    """Option to turn off consistency loss"""
    use_tc_loss: bool = True
    """Option to turn off reward prediction"""
    use_rew_loss: bool = True
    """Reward coefficient"""
    reward_coef: float = 1.0
    """Consistency coefficient"""
    consistency_coef: float = 1.0
    """If not None then bound the reward output"""
    r_min: Optional[float] = None
    """If not None then bound the reward output"""
    r_max: Optional[float] = None
    """Which loss function to use for consistency loss?"""
    consistency_loss: str = "cross-entropy"  # "cross-entropy", "mse", "cosine"
    """Predict logits with dynamics NN or use cosine/mse between pred and codebook?"""
    ce_logits_mode: str = "standard"  # "standard", cosine", "mse"
    """How to get propagate the state dist. during training"""
    unc_prop_mode: str = "sample"  # Literal["sample", "sample-no-grad", "weighted-avg"]
    """Flag to turn FSQ off"""
    use_fsq: bool = True
    """FSQ levels hyperparameter - [5, 3] corresponds to 15 codes"""
    fsq_levels: List[int] = field(default_factory=lambda: [5, 3])
    """(Optionally) use automatic mixed precision"""
    use_amp: bool = False
    """Use straight through Gumbel softmax (hard) or just Gumbel softmax (soft)"""
    straight_through_gumbel: bool = True

    """PLANNING (MPPI) CONFIG"""
    """Optionall turn MPC off and use policy"""
    mpc: bool = True
    """Number of MPPI iterations"""
    iterations: int = 6
    """Number action samples"""
    num_samples: int = 512
    """Number elites to use for re-sampling"""
    num_elites: int = 64
    """Number """
    num_pi_trajs: int = 24
    """Planning horizon"""
    plan_horizon: int = 3
    """Minimum action std during MPPI loop"""
    min_std: float = 0.05
    """Maximum action std during MPPI loop"""
    max_std: float = 2
    """MPPI temperature"""
    temperature: float = 0.5
    """How to get propagate the state dist. during planning"""
    plan_unc_prop_mode: str = "weighted-avg"  # "sample"/"sample-no-grad"/"weighted-avg"
    """Should MPPI only use top K samples or all"""
    use_top_k: bool = True
    """If not True then sample from Categorical over actions with weights from scores"""
    use_mppi_mean: bool = False

    """TD3 CONFIG"""
    """MLP dims for actor/critic"""
    mlp_dims: List[int] = field(default_factory=lambda: [512, 512])
    """Learning rate for actor/critic"""
    lr: float = 3e-4
    """Batch size - same for for representation learning and actor/critic"""
    batch_size: int = 512
    """Number of parameter updates per new data, i.e .UTD ratio """
    utd_ratio: int = 1
    """Update actor less frequently than critic"""
    actor_update_freq: int = 2
    """Discount factor"""
    gamma: float = 0.99
    """Target network update rate"""
    tau: float = 0.005
    """Number of critics"""
    num_critics: int = 5
    """Number of critics to sample"""
    q_sample_size: int = 2
    """Use N-step returns for Q-learning?"""
    nstep: int = 1
    """Clips the gradient norm of the encoder"""
    grad_clip_norm: Optional[float] = 20.0

    """EXPLORATION NOISE SCHEDULE"""
    """Initial variance"""
    exploration_noise_start: float = 1.0
    """Final variance"""
    exploration_noise_end: float = 0.1
    """Number of episodes do decay noise"""
    exploration_noise_num_steps: int = 50

    """POLICY SMOOTHING"""
    """Variance"""
    policy_noise: float = 0.2
    """Clip the noise"""
    noise_clip: float = 0.3

    """OTHER"""
    """All NNs will be put on this device"""
    device: str = "${device}"  # set from TrainConfig
    """Logging frequency"""
    logging_freq: int = 100
    """If True try to compile all NNs"""
    compile: bool = False
    """Print training losses?"""
    verbose: bool = "${verbose}"  # set from TrainConfig


class WorldModel(nn.Module):
    """Discrete Codebook World Model"""

    def __init__(self, cfg, obs_spec: CompositeSpec, act_spec: BoundedTensorSpec):
        super().__init__()
        self.cfg = cfg
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        act_dim = np.array(act_spec.shape).prod().item()

        ##### Configure FSQ stuff #####
        self.org_latent_dim = copy.copy(cfg.latent_dim)
        self.enc_latent_dim = copy.copy(cfg.latent_dim)
        if cfg.use_fsq:
            self.num_channels = len(cfg.fsq_levels)
            if not cfg.latent_dim % self.num_channels == 0:
                raise NotImplementedError(
                    "latent_dim must be divisible by number of FSQ channels"
                )
            self._fsq = FSQ(levels=cfg.fsq_levels)

            self.enc_latent_dim = self.org_latent_dim * self.num_channels
            self.cfg.latent_dim *= self.num_channels

        ##### Init encoder #####
        self._encoder = nn.ModuleDict()
        if "state" in cfg.obs_types:  # Encoder for state-based observations
            obs_dim = np.array(obs_spec["state"].shape).prod().item()
            self._encoder.update(
                {"state": mlp(obs_dim, cfg.enc_mlp_dims, self.enc_latent_dim)}
            )
            if cfg.compile:
                self._encoder["state"] = torch.compile(
                    self._encoder["state"], mode="default"
                )
        if "pixels" in cfg.obs_types:  # Encoder for pixel-based observations
            raise NotImplementedError

        ##### Init transition dynamics #####
        trans_out_dim = self.cfg.latent_dim
        if self.cfg.consistency_loss == "cross-entropy":
            if self.cfg.ce_logits_mode == "standard":
                """If training dynamics w/ cross entropy change output dim"""
                assert cfg.use_fsq
                trans_out_dim = int(self.org_latent_dim * self._fsq.codebook_size)
        self._trans = mlp(self.enc_latent_dim + act_dim, cfg.mlp_dims, trans_out_dim)
        if cfg.compile:
            self._trans = torch.compile(self._trans, mode="default")

        ##### Init reward #####
        if cfg.use_rew_loss:
            self._reward = mlp(self.cfg.latent_dim + act_dim, cfg.mlp_dims, 1)
            if cfg.compile:
                self._reward = torch.compile(self._reward, mode="default")

            if cfg.r_max is not None and cfg.r_min is not None:
                r_scale = (cfg.r_max - cfg.r_min) / 2.0
                r_bias = (cfg.r_max + cfg.r_min) / 2.0
                self.r_scale_fn = lambda r: torch.tanh(r) * r_scale + r_bias
            else:
                self.r_scale_fn = lambda r: r

    def encode(self, obs):
        zs = {}
        for key in obs.keys():
            zs.update({key: self._encoder[key](obs[key])})
        if "state" in self.cfg.obs_types and "pixels" not in self.cfg.obs_types:
            z = zs["state"]
        elif "state" not in self.cfg.obs_types and "pixels" in self.cfg.obs_types:
            z = zs["pixels"]
        else:
            raise NotImplementedError("Need to make encoder take both state and pixels")

        td = TensorDict({"state": z}, batch_size=obs.batch_size)
        if self.cfg.use_fsq:
            td.update(self.quantize(z))
        else:
            td.update({"codes": z})
        return td

    def trans(self, z, a, unc_prop_mode: Optional[str] = None):
        za = torch.concat([z, a], -1)

        if (
            self.cfg.consistency_loss == "cross-entropy"
            and self.cfg.ce_logits_mode == "standard"
        ):
            """Make predictions with dynamics as NN classifier"""
            # Returns logits for each class
            logits = self._trans(za)
            logits = logits.reshape(-1, self.org_latent_dim, self._fsq.codebook_size)

            if unc_prop_mode is None:
                unc_prop_mode = self.cfg.unc_prop_mode

            # Convert latent state logits to an actual latent state
            if "sample-no-grad" in unc_prop_mode:

                def gumbel_sample(logits):
                    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
                    adjusted_logits = logits + gumbel_noise
                    return torch.argmax(adjusted_logits, dim=-1)

                indices = gumbel_sample(logits)
                next_z = self._fsq.implicit_codebook[indices].flatten(-2)
                next_z_dict = {
                    "codes": next_z,
                    "logits": logits,
                    "indices": indices.to(torch.float),
                }
            elif "sample" in unc_prop_mode:
                z_one_hot = torch.nn.functional.gumbel_softmax(
                    logits, tau=1, hard=self.cfg.straight_through_gumbel, dim=-1
                )
                codebook = self._fsq.implicit_codebook
                next_z = einsum(z_one_hot, codebook, "b d c, c l -> b d l")
                next_z = rearrange(next_z, "b d l -> b (d l)")
                next_z_dict = {
                    "codes": next_z,
                    "logits": logits,
                    "one-hot": z_one_hot.flatten(-2),
                }
            elif "weighted-avg" in unc_prop_mode:
                probs = F.softmax(logits, dim=-1)
                codebook = self._fsq.implicit_codebook
                next_z = einsum(probs, codebook, "b d c, c l -> b d l")
                next_z = rearrange(next_z, "b d l -> b (d l)")
                next_z_dict = {"codes": next_z, "logits": logits}
            elif unc_prop_mode in ["mode", "max"]:
                # Note this has no gradients so should only be used for MPC
                indices = torch.max(logits, -1)[1]
                next_z = self._fsq.implicit_codebook[indices.to(torch.long)].flatten(-2)
                next_z_dict = {"codes": next_z, "logits": logits, "indices": indices}
            else:
                raise NotImplementedError
        else:
            """Make predictions with dynamics regression model"""
            delta_z = self._trans(za)
            next_z = z + delta_z if self.cfg.use_delta else delta_z
            if self.cfg.use_fsq:
                next_z = self.quantize(next_z)["codes"]

            next_z_dict = {"codes": next_z}

        if self.cfg.use_fsq:
            shape = *next_z.shape[0:-1], self.org_latent_dim, self.num_channels
        else:
            shape = *next_z.shape[0:-1], self.org_latent_dim
        next_z_dict.update({"z": next_z.reshape(shape)})
        return TensorDict(
            next_z_dict,
            batch_size=torch.Size([z.shape[0]]),
            device=self.cfg.device,
        )

    def reward(self, z, a):
        za = torch.concat([z, a], -1)
        r = self._reward(za)
        r = self.r_scale_fn(r)
        return r

    def quantize(self, z):
        """Quantize the latent state"""
        td = self._fsq(z)
        td["state"] = td["codes"]
        return td

    def loss(self, batch: ReplayBufferSamples) -> Tuple[torch.Tensor, dict]:
        tc_loss = torch.zeros(1).to(self.cfg.device)
        reward_loss = torch.zeros(1).to(self.cfg.device)

        ##### Create targets #####
        with torch.no_grad():
            next_obs = batch.next_observations
            zs_tar = self.encode(next_obs)

        ##### Create TensorDicts to fill #####
        zs = {
            "codes": torch.empty(
                self.cfg.horizon + 1,
                self.cfg.batch_size,
                self.enc_latent_dim,
                device=self.cfg.device,
            )
        }
        if self.cfg.consistency_loss == "cross-entropy":
            zs.update(
                {
                    "logits": torch.empty(
                        self.cfg.horizon + 1,
                        self.cfg.batch_size,
                        self.org_latent_dim,
                        self._fsq.codebook_size,
                        device=self.cfg.device,
                    )
                }
            )
        zs = TensorDict(
            zs,
            batch_size=torch.Size([self.cfg.horizon + 1, self.cfg.batch_size]),
            device=self.cfg.device,
        )

        ##### Latent rollout #####
        z = self.encode(batch.observations[0])["codes"]
        zs["codes"][0] = z
        dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
        terminateds_or_dones = torch.zeros_like(batch.dones, dtype=torch.bool)
        a = batch.actions
        for t in range(self.cfg.horizon):
            dones = torch.where(terminateds_or_dones[t], dones, batch.dones[t])
            terminateds_or_dones[t] = torch.logical_or(
                terminateds_or_dones[t], torch.logical_or(dones, batch.terminateds[t])
            )

            # Predict next latent
            next_z = self.trans(z=z, a=a[t])
            zs[t + 1] = next_z

            # Don't forget this
            z = next_z["codes"]

        rho = torch.tensor([self.cfg.rho**t for t in range(self.cfg.horizon)]).to(
            self.cfg.device
        )
        dones = batch.dones.to(torch.int)

        ##### (Optional) Reward prediction loss #####
        if self.cfg.use_rew_loss:
            r_tar = batch.rewards  # Reward target
            r_pred = self.reward(z=zs["codes"][:-1], a=a)[..., 0]
            assert r_pred.ndim == 2 and r_tar.ndim == 2
            _reward_loss = (r_pred - r_tar) ** 2
            _rho_reward_loss = rho * torch.mean((1 - dones) * _reward_loss, -1)
            reward_loss = torch.mean(_rho_reward_loss)

        ##### Temporal consistency loss #####
        if self.cfg.use_tc_loss:
            if self.cfg.consistency_loss == "cross-entropy":
                """Cross entropy"""
                if self.cfg.ce_logits_mode in ["cosine", "mse"]:
                    """If not predicting logits with dynamics NN use alternative method"""
                    zs_ = zs["codes"][1:].view(
                        self.cfg.horizon,
                        self.cfg.batch_size,
                        int(self.cfg.latent_dim / self.num_channels),
                        self.num_channels,
                    )[..., None, :]
                    codebook = self._fsq.implicit_codebook[None, None, None, ...]
                    if self.cfg.ce_logits_mode == "cosine":
                        """Cosine similarity with codebook"""
                        # TODO use compute_logits like CLIP
                        zs["logits"][1:] = nn.CosineSimilarity(dim=-1, eps=1e-6)(
                            zs_, codebook
                        )
                    elif self.cfg.ce_logits_mode == "mse":
                        """Inner product with codebook"""
                        zs["logits"][1:] = torch.einsum(
                            "hbdic,hbdCc->hbdC", zs_, codebook
                        )
                _tc_loss = torch.vmap(torch.vmap(F.cross_entropy))(
                    zs["logits"][1:],
                    zs_tar["indices"].to(torch.long),
                )
            elif self.cfg.consistency_loss == "cosine":
                """Cosine similarity"""
                _tc_loss = -nn.CosineSimilarity(dim=-1, eps=1e-6)(
                    zs["codes"][1:], zs_tar["codes"]
                )
            elif self.cfg.consistency_loss == "mse":
                """Mean squared error"""
                _tc_loss = torch.mean((zs["codes"][1:] - zs_tar["codes"]) ** 2, dim=-1)
            else:
                raise NotImplementedError(
                    f"cfg.consistency_loss should be 'cross-entropy', 'mse', 'cosine', not {self.cfg.consistency_loss}"
                )

            _rho_tc_loss = rho * torch.mean((1 - dones) * _tc_loss, -1)
            tc_loss = torch.mean(_rho_tc_loss)

        loss = self.cfg.consistency_coef * tc_loss + self.cfg.reward_coef * reward_loss
        info = {
            "tc_loss": tc_loss.item(),
            "reward_loss": reward_loss.item(),
            "enc_loss": loss.item(),
            "z_min": torch.min(zs["codes"]).item(),
            "z_max": torch.max(zs["codes"]).item(),
            "z_mean": torch.mean(zs["codes"].to(torch.float)).item(),
            "z_median": torch.median(zs["codes"]).item(),
        }
        if self.cfg.use_rew_loss:
            info.update(
                {
                    "r_min": r_pred.min().item(),
                    "r_max": r_pred.max().item(),
                    "r_mean": r_pred.mean().item(),
                }
            )
        return loss, info

    def metrics(self, batch):
        z = self.encode(batch.observations[0])

        # Calculate rank of latent
        metrics = h.calc_rank(name="z", z=z["state"])

        # Calculate percentage of codebook that's active
        if self.cfg.use_fsq:
            num_codes = torch.tensor(math.prod(self.cfg.fsq_levels), device=z.device)

            def act_percent_fn(z):
                # TODO can't vmap this because Tensor.unique() not allowed in vmap
                return z.unique().numel() / num_codes * 100

            active_percents = torch.empty(z["indices"].shape[1])
            for i in range(z["indices"].shape[1]):
                active_percents[i] = act_percent_fn(z["indices"][i])
            metrics.update(
                {
                    "active_percent_avg": active_percents.mean(),
                    "active_percent_min": active_percents.min(),
                    "active_percent_max": active_percents.max(),
                }
            )

        return metrics

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DCMPC(nn.Module):
    """Discrete Codebook Model Predictive Control"""

    def __init__(self, cfg, obs_spec: CompositeSpec, act_spec: BoundedTensorSpec):
        super().__init__()
        self.cfg = cfg
        self.obs_spec = obs_spec
        self.act_spec = act_spec
        self.act_dim = np.array(act_spec.shape).prod().item()

        if "state" not in cfg.obs_types:
            raise NotImplementedError("Only state observations supported")

        ##### Init World Model and Actor/Critic #####
        self.model = WorldModel(cfg, obs_spec, act_spec)
        self._pi = mlp(cfg.latent_dim, cfg.mlp_dims, self.act_dim)
        self._Qs = mlp_ensemble(
            cfg.latent_dim + self.act_dim, cfg.mlp_dims, 1, cfg.num_critics
        )
        if cfg.compile:
            self.model = torch.compile(self.model, mode="default")
            self._pi = torch.compile(self._pi, mode="default")
            self._Qs = torch.compile(self._Qs, mode="default")

        ##### Init target actor/critic (TD3) #####
        self._pi_tar = copy.deepcopy(self._pi).requires_grad_(False)
        self.Qs_tar = copy.deepcopy(self._Qs).requires_grad_(False)
        if cfg.compile:
            self._pi_tar = torch.compile(self._pi_tar, mode="default")
            self.Qs_tar = torch.compile(self.Qs_tar, mode="default")

        ##### Optimizers #####
        self.model_opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.enc_lr)
        self.pi_opt = torch.optim.Adam(self._pi.parameters(), lr=cfg.lr)
        self.q_opt = torch.optim.Adam(self._Qs.parameters(), lr=cfg.lr)

        ##### Exploration noise schedule #####
        self._exploration_noise_schedule = h.LinearSchedule(
            start=cfg.exploration_noise_start,
            end=cfg.exploration_noise_end,
            num_steps=cfg.exploration_noise_num_steps,
        )

        # Counters for number of param updates
        self.critic_update_counter = 0
        self.pi_update_counter = 0

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update world model and actor/critic (TD3) at same time"""
        n = int(num_new_transitions * self.cfg.utd_ratio)
        info = {}
        self.scaler = GradScaler()

        for i in range(n):
            batch = replay_buffer.sample()

            #### Update world model #####
            info.update(self.model_update_step(batch=batch))

            # Map observations to latent
            with torch.no_grad():
                zs = self.model.encode(batch.observations)
                next_zs = self.model.encode(batch.next_observations)
            batch = batch._replace(zs=zs, next_zs=next_zs)

            ##### Make nstep returns (or flatten) #####
            batch = utils.to_nstep(batch, nstep=self.cfg.nstep, gamma=self.cfg.gamma)

            ##### Update critic #####
            info.update(self.critic_update_step(batch=batch))

            ##### Update actor less frequently than critic #####
            if self.critic_update_counter % self.cfg.actor_update_freq == 0:
                info.update(self.pi_update_step(batch=batch))

            if i % self.cfg.logging_freq == 0:
                if wandb.run is not None:
                    wandb.log(info)

        self._exploration_noise_schedule.step()
        info.update({"exploration_noise": self.exploration_noise})
        return info

    def model_update_step(self, batch: ReplayBufferSamples):
        self.model.train()
        with autocast(
            device_type=self.cfg.device, dtype=torch.float16, enabled=self.cfg.use_amp
        ):
            loss, info = self.model.loss(batch=batch)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.model_opt)

        if self.cfg.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.grad_clip_norm
            )
            info.update({"grad_norm": float(grad_norm)})

        self.scaler.step(self.model_opt)
        self.scaler.update()
        self.model_opt.zero_grad(set_to_none=True)

        if hasattr(self, "mppi_std"):
            info.update({f"mppi_std": self.mppi_std[0].mean().item()})
        self.model.eval()
        return info

    def critic_update_step(self, batch: ReplayBufferSamples):
        self.critic_update_counter += 1
        self._Qs.train()

        # Check batch shapes
        assert batch.rewards.ndim == 1
        assert batch.rewards.shape[0] == batch.zs.shape[0]

        # Make Q target
        with torch.no_grad():
            next_zs = batch.next_zs["codes"]
            a_next = self.pi(next_zs, tar=True, eval_mode=True, smooth=True)
            min_q_next_tar = self.Q(next_zs, a=a_next, return_type="min", tar=True)[
                ..., 0
            ]

            assert min_q_next_tar.shape == batch.rewards.shape
            next_q_value = (
                batch.rewards
                + (1 - batch.terminateds) * batch.next_state_gammas * min_q_next_tar
            )

        q_values = self.Q(batch.zs["codes"], a=batch.actions, return_type="all")[..., 0]
        next_q_value = next_q_value.broadcast_to(q_values.shape)
        q_loss = F.mse_loss(q_values, next_q_value)

        info = {
            "q_loss": q_loss.item(),
            "q_mean": q_values.mean().item(),
            "q_min": q_values.min().item(),
            "q_max": q_values.max().item(),
            "q_std": q_values.std().item(),
            "q_targ_mean": next_q_value.mean().item(),
            "q_targ_min": next_q_value.min().item(),
            "q_targ_max": next_q_value.max().item(),
            "q_targ_std": next_q_value.std().item(),
            "critic_update_counter": self.critic_update_counter,
        }

        ##### Optimize critic #####
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()

        if self.cfg.grad_clip_norm is not None:
            q_params = list(self._Qs.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                q_params, self.cfg.grad_clip_norm, error_if_nonfinite=False
            )
            info.update({"q_grad_norm": float(grad_norm)})

        self.q_opt.step()

        ##### Update the target network #####
        h.soft_update_params(self._Qs, self.Qs_tar, tau=self.cfg.tau)

        for i in range(self.cfg.num_critics):
            info.update({f"q{i+1}_values": q_values[i].mean().item()})

        self._Qs.eval()
        return info

    def pi_update_step(self, batch: ReplayBufferSamples):
        self.pi_update_counter += 1
        self._pi.train()

        z = batch.zs["codes"]
        pi_loss = -self.Q(z, a=self.pi(z, eval_mode=True), return_type="avg").mean()
        info = {
            "actor_loss": pi_loss.item(),
            "actor_update_counter": self.pi_update_counter,
        }

        ##### Optimize actor #####
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()

        if self.cfg.grad_clip_norm is not None:
            pi_params = list(self._pi.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                pi_params, self.cfg.grad_clip_norm, error_if_nonfinite=False
            )
            info.update({"pi_grad_norm": float(grad_norm)})

        self.pi_opt.step()

        ##### Update the target network #####
        h.soft_update_params(self._pi, self._pi_tar, tau=self.cfg.tau)

        self._pi.eval()
        return info

    @torch.no_grad()
    def select_action(self, obs, t0: bool = False, eval_mode: bool = False):
        is_flat_obs = False
        if obs.batch_size == torch.Size([]):
            obs = obs.view(1)
            is_flat_obs = True

        z = self.model.encode(obs).to(torch.float)
        if self.cfg.mpc:
            a, self.mppi_std = self.plan(z, t0=t0, eval_mode=eval_mode)
        else:
            a = self.pi(z["codes"], tar=False, eval_mode=eval_mode)
            a = a[0] if is_flat_obs else a

        return a

    @torch.no_grad()
    def plan(self, z, t0: bool = False, eval_mode=False):
        """
        Plan a sequence of actions using MPPI within the learned world model.
        """
        z_td = z
        z = z["state"]
        batch_size = z.shape[0]
        pi_actions = torch.empty(
            batch_size,
            self.cfg.plan_horizon,
            self.cfg.num_pi_trajs,
            self.act_dim,
            device=self.device,
        )
        actions = torch.empty(
            batch_size,
            self.cfg.plan_horizon,
            self.cfg.num_samples,
            self.act_dim,
            device=self.device,
        )
        mean = torch.zeros(
            batch_size, self.cfg.plan_horizon, self.act_dim, device=self.device
        )
        self.std = self.cfg.max_std * torch.ones(
            self.cfg.plan_horizon, self.act_dim, device=self.device
        )

        def single_mppi(z, actions, pi_actions, mean, prev_mean):
            # Sample policy trajectories
            if self.cfg.num_pi_trajs > 0:
                _z = z.expand(self.cfg.num_pi_trajs)
                for t in range(self.cfg.plan_horizon - 1):
                    pi_actions[t] = self.pi(_z["codes"], eval_mode=False)
                    _z = self.model.trans(
                        _z["codes"],
                        pi_actions[t],
                        unc_prop_mode=self.cfg.plan_unc_prop_mode,
                    )
                pi_actions[-1] = self.pi(_z["codes"], eval_mode=False)

            # Initialize state and parameters
            z = z.expand(self.cfg.num_samples)
            std = self.std
            if not t0:
                mean[:-1] = prev_mean[1:]
            if self.cfg.num_pi_trajs > 0:
                actions[:, : self.cfg.num_pi_trajs] = pi_actions

            # Iterate MPPI
            for _ in range(self.cfg.iterations):
                # Sample actions
                actions[:, self.cfg.num_pi_trajs :] = (
                    mean.unsqueeze(1)
                    + std.unsqueeze(1)
                    * torch.randn(
                        self.cfg.plan_horizon,
                        self.cfg.num_samples - self.cfg.num_pi_trajs,
                        self.act_dim,
                        device=std.device,
                    )
                ).clamp(-1, 1)

                # Compute elite actions
                value = self._single_estimate_value(z, actions).nan_to_num_(0)
                if self.cfg.use_top_k:
                    elite_idxs = torch.topk(
                        value.squeeze(1), self.cfg.num_elites, dim=0
                    ).indices
                    elite_value, elite_actions = (
                        value[elite_idxs],
                        actions[:, elite_idxs],
                    )
                else:
                    elite_value, elite_actions = (value, actions)

                # Update parameters
                max_value = elite_value.max(0)[0]
                score = torch.exp(self.cfg.temperature * (elite_value - max_value))
                score /= score.sum(0)
                mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                    score.sum(0) + 1e-9
                )
                std = torch.sqrt(
                    torch.sum(
                        score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2,
                        dim=1,
                    )
                    / (score.sum(0) + 1e-9)
                )
                std = std.clamp(self.cfg.min_std, self.cfg.max_std)

            if self.cfg.use_mppi_mean:
                actions = mean
            else:
                act_dist = torch.distributions.Categorical(score[:, 0])
                act_idx = act_dist.sample()
                actions = torch.index_select(elite_actions, 1, act_idx)[:, 0, :]
            a, std = actions[0], std[0]
            if not eval_mode:
                std = self.action_scale * self.exploration_noise
                a += std * torch.randn(self.act_dim, device=std.device)
            return a, mean, std

        if hasattr(self, "_prev_mean") and not t0:
            prev_mean = self._prev_mean
        else:
            prev_mean = torch.empty(
                batch_size,
                self.cfg.plan_horizon,
                self.act_dim,
                device=self.device,
            )
        a, new_prev_mean, std = torch.vmap(
            single_mppi, in_dims=(0, 0, 0, 0, 0), randomness="different"
        )(z_td, actions, pi_actions, mean, prev_mean)

        self._prev_mean = new_prev_mean
        a.clamp_(self.act_spec_low, self.act_spec_high)
        return a, std

    @torch.no_grad()
    def _single_estimate_value(self, z, actions):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.plan_horizon):
            reward = self.model.reward(z["codes"], actions[t])
            z = self.model.trans(
                z["codes"], actions[t], unc_prop_mode=self.cfg.plan_unc_prop_mode
            )
            G += discount * reward
            discount *= self.cfg.rho
        z_pi = z["codes"]
        return G + discount * self.Q(z_pi, self.pi(z_pi), return_type="avg")

    def pi(self, z, tar: bool = False, eval_mode: bool = False, smooth: bool = False):
        a = self._pi_tar(z) if tar else self._pi(z)
        a = torch.tanh(a)
        a = a * self.action_scale + self.action_bias
        if not eval_mode:
            a += torch.normal(0, self.action_scale * self.exploration_noise)
        if smooth:
            clipped_noise = (
                torch.randn_like(a, device=self.cfg.device) * self.cfg.policy_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip) * self.action_scale
            a += clipped_noise
        a = a.clamp(self.act_spec_low, self.act_spec_high)
        return a

    def Q(self, z, a, return_type: str = "all", tar: bool = False):
        za = torch.cat([z, a], -1)
        qs = self.Qs_tar(za) if tar else self._Qs(za)
        if return_type == "all":
            return qs

        # Sample two Q values
        if self.cfg.q_sample_size is not None:
            idxs = torch.randperm(qs.shape[0])[: self.cfg.q_sample_size]
            qs = qs[idxs]
        if return_type == "min":
            return torch.min(qs, 0)[0]
        elif return_type == "avg":
            return torch.mean(qs, 0)
        else:
            raise NotImplementedError(
                f"return_type should be 'all' or 'min' or 'avg' not {return_type}"
            )

    def metrics(self, batch):
        metrics = self.model.metrics(batch)
        metrics.update({"model": h.calc_mean_opt_moments(self.model_opt)})
        metrics.update({"Q": h.calc_mean_opt_moments(self.q_opt)})
        metrics.update({"pi": h.calc_mean_opt_moments(self.pi_opt)})
        return metrics

    def save(self, path: str = "./checkpoint.pt", metrics: dict = {}):
        ckpt = metrics.copy()
        ckpt.update(
            {
                "model": self.state_dict(),
                "model_opt": self.model_opt.state_dict(),
                "pi_opt": self.model_opt.state_dict(),
                "q_opt": self.model_opt.state_dict(),
            }
        )
        torch.save(ckpt, path)

    @property
    def exploration_noise(self):
        return self._exploration_noise_schedule()

    @property
    def act_spec_low(self):
        return self.act_spec.low

    @property
    def act_spec_high(self):
        return self.act_spec.high

    @cached_property
    def action_scale(self):
        return (self.act_spec.high - self.act_spec.low) / 2.0

    @cached_property
    def action_bias(self):
        return (self.act_spec.high + self.act_spec.low) / 2.0

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return self.cfg.device
