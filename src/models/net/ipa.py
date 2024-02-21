"""
Adapted from [Openfold](https://github.com/aqlaboratory/openfold) IPA implementation.
"""

import math
from typing import Optional, List, Sequence

import torch
import torch.nn as nn

from src.common.rigid_utils import Rigid
from src.models.net.layers import Linear, NodeTransition, EdgeTransition, TorsionAngleHead, BackboneUpdate


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        
        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        
        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0]).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            ).to(dtype=z[0].dtype)
        )
        
        return s


class TranslationIPA(nn.Module):
    def __init__(self, 
                 c_s: int,
                 c_z: int,
                 coordinate_scaling: float,
                 no_ipa_blocks: int,
                 skip_embed_size: int,
                 transformer_num_heads: int = 4,
                 transformer_num_layers: int = 2,
                 c_hidden: int = 256,
                 no_heads: int = 8,
                 no_qk_points: int = 8,
                 no_v_points: int = 12,
                 dropout: float = 0.0,
    ):
        super(TranslationIPA, self).__init__()

        self.scale_pos = lambda x: x * coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)
        self.trunk = nn.ModuleDict()
        self.num_blocks = no_ipa_blocks
        
        for b in range(no_ipa_blocks):
            self.trunk[f'ipa_{b}'] = InvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                no_qk_points=no_qk_points,
                no_v_points=no_v_points,
            )
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(c_s)
            self.trunk[f'skip_embed_{b}'] = Linear(
                c_s,
                skip_embed_size,
                init="final"
            )
            _in_dim = c_s + skip_embed_size
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=_in_dim,
                nhead=transformer_num_heads,
                dim_feedforward=_in_dim,
            )
            self.trunk[f'transformer_{b}'] = nn.TransformerEncoder(transformer_layer, transformer_num_layers)
            self.trunk[f'linear_{b}'] = Linear(_in_dim, c_s, init="final")
            self.trunk[f'node_transition_{b}'] = NodeTransition(c_s)
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(c_s)

            if b < self.num_blocks - 1: # No need edge update on the last block
                self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                    node_embed_size=c_s,
                    edge_embed_in=c_z,
                    edge_embed_out=c_z,
                )
        self.torsion_pred = TorsionAngleHead(c_s, 1)
        # self.chi_pred = TorsionAngleHead(c_s, 4)

    def forward(self, node_embed, edge_embed, batch):
        node_mask = batch['residue_mask'].type(torch.float)
        diffuse_mask = (1 - batch['fixed_mask'].type(torch.float)) * node_mask
        edge_mask = node_mask[..., None] * node_mask[..., None, :]
        
        init_frames = batch['rigids_t'].type(torch.float)
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)
        curr_rigids = self.scale_rigids(curr_rigids)
        
        # Main trunk
        init_node_embed = node_embed
        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            # MHA + FF
            concat_node_embed = torch.cat([
                node_embed, self.trunk[f'skip_embed_{b}'](init_node_embed)
            ], dim=-1)
            concat_node_embed = torch.transpose(concat_node_embed, 0, 1)
            transformed_embed = self.trunk[f'transformer_{b}'](concat_node_embed, src_key_padding_mask=1.0 - node_mask)
            transformed_embed = torch.transpose(transformed_embed, 0, 1)
            # residual
            node_embed = node_embed + self.trunk[f'linear_{b}'](transformed_embed)
            
            # MLP
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            # apply mask
            node_embed = node_embed * node_mask[..., None]
            # backbone update
            rigid_update = self.trunk[f'bb_update_{b}'](node_embed * diffuse_mask[..., None])
            # apply update
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, diffuse_mask[..., None])

            if b < self.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed) * edge_mask[..., None]
        
        # torsion prediction
        psi_pred = self.torsion_pred(node_embed)
        # chi_pred = self.chi_pred(node_embed)
        
        # scale back
        curr_rigids = self.unscale_rigids(curr_rigids)
        
        model_out = {
            'in_rigids': init_rigids,
            'out_rigids': curr_rigids,
            'psi': psi_pred,
            # 'chi': chi_pred,
        }
        return model_out
