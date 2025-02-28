#!/usr/bin/env python3
import copy
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state
from vector_quantize_pytorch import FSQ as _FSQ

from .helper import orthogonal_init


def mlp(
    in_dim: int,
    mlp_dims: List[int],
    out_dim: int,
    act_fn: Optional[Callable] = None,
    dropout: float = 0.0,
):
    """
    MLP with LayerNorm, Mish activations, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]

    dims = [int(in_dim)] + mlp_dims + [int(out_dim)]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=act_fn)
        if act_fn
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


def mlp_ensemble(in_dim: int, mlp_dims: List[int], out_dim: int, size: int):
    """Ensemble of MLPs with orthogonal initialization"""
    mlp_list = []
    for _ in range(size):
        mlp_list.append(mlp(in_dim, mlp_dims, out_dim))
        orthogonal_init(mlp_list[-1].parameters())
    return Ensemble(mlp_list)


class FSQ(_FSQ):
    """
    Finite Scalar Quantization
    """

    def __init__(self, levels: List[int]):
        super().__init__(levels=levels)
        self.levels = levels
        self.num_channels = len(levels)

    def forward(self, z):
        shp = z.shape
        z = z.view(*shp[:-1], -1, self.num_channels)
        if z.ndim > 3:  # TODO this might not work for CNN
            codes, indices = torch.func.vmap(super().forward)(z)
        else:
            codes, indices = super().forward(z)
        codes = codes.flatten(-2)
        return {"codes": codes, "indices": indices, "z": z, "state": codes}

    def __repr__(self):
        return f"FSQ(levels={self.levels})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, Mish activation, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None
        self.norm = nn.LayerNorm(self.out_features)

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)

        x = self.norm(x)
        x = self.act(x)
        return x

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, \
        out_features={self.out_features}, \
        bias={self.bias is not None}{repr_dropout}, \
        act={self.act.__class__.__name__})"


class Ensemble(nn.Module):
    """Vectorized ensemble of modules"""

    def __init__(self, modules, **kwargs):
        super().__init__()
        self.params_dict, self._buffers = stack_module_state(modules)
        self.params = nn.ParameterList([p for p in self.params_dict.values()])
        self.size = len(modules)

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        base_model = copy.deepcopy(modules[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))

        self.vmap = torch.vmap(
            fmodel, in_dims=(0, 0, None), randomness="different", **kwargs
        )
        self._repr = str(nn.ModuleList(modules))

    def forward(self, *args, **kwargs):
        return self.vmap(self._get_params_dict(), self._buffers, *args, **kwargs)

    def _get_params_dict(self):
        params_dict = {}
        for key, value in zip(self.params_dict.keys(), self.params):
            params_dict.update({key: value})
        return params_dict

    def __repr__(self):
        return f"Vectorized " + self._repr
