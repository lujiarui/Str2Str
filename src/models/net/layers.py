import math
from typing import Optional, Callable, List

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out

def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))

def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)

def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)

def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)

def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)

def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)

def normal_init_(weights):
    nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """
    A Linear layer with in-house nonstandard initializations.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)
        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


# Simply 3-layer MLP
class NodeTransition(nn.Module):
    def __init__(self, dim):
        super(NodeTransition, self).__init__()
        self.dim = dim
        self.linear_1 = Linear(dim, dim, init="relu")
        self.linear_2 = Linear(dim, dim, init="relu")
        self.linear_3 = Linear(dim, dim, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(dim)

    def forward(self, s):
        s_initial = s
        s = self.relu(self.linear_1(s))
        s = self.relu(self.linear_2(s))
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)
        return s


class EdgeTransition(nn.Module):
    def __init__(self,
                node_embed_size,
                edge_embed_in,
                edge_embed_out,
                num_layers=2,
                node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(
            node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ], axis=-1)
        edge_embed = torch.cat(
            [edge_embed, edge_bias], axis=-1).reshape(
                batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(
            batch_size, num_res, num_res, -1
        )
        return edge_embed
    
    
class TorsionAngleHead(nn.Module):
    def __init__(self, in_dim, n_torsion_angles, eps=1e-8):
        super(TorsionAngleHead, self).__init__()
        
        self.linear_1 = Linear(in_dim, in_dim, init="relu")
        self.linear_2 = Linear(in_dim, in_dim, init="relu")
        self.linear_3 = Linear(in_dim, in_dim, init="final")
        self.linear_final = Linear(in_dim, n_torsion_angles * 2, init="final")
        self.relu = nn.ReLU()
        self.eps = eps

    def forward(self, s):
        s_initial = s
        s = self.relu(self.linear_1(s))
        s = self.linear_2(s)
        s = s + s_initial
        
        unnormalized_s = self.linear_final(s)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom
        return normalized_s
    
    
class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s
        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)
        return update