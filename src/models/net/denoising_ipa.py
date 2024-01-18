import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from src.common.all_atom import compute_backbone
from src.common.geo_utils import calc_distogram
from src.models.net.ipa import TranslationIPA


def get_positional_embedding(indices, embedding_dim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embedding_dim: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embedding_dim]
    """
    K = torch.arange(embedding_dim//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embedding_dim))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embedding_dim))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_len=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_len
    half_dim = embedding_dim // 2
    emb = math.log(max_len) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class EmbeddingModule(nn.Module):
    def __init__(self, 
                init_embed_size: int,
                node_embed_size: int,
                edge_embed_size: int,
                num_bins: int = 22,
                min_bin: float = 1e-5,
                max_bin: float = 20.0,
                self_conditioning: bool = True,
    ):
        super(EmbeddingModule, self).__init__()
        pos_embed_size = init_embed_size
        t_embed_size = init_embed_size
        
        # time embedding
        node_in_dim = t_embed_size + 1
        edge_in_dim = (t_embed_size + 1) * 2

        # positional embedding
        node_in_dim += pos_embed_size
        edge_in_dim += pos_embed_size

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )
        
        # self-conditioning trick used in RFDiffusion
        self.self_conditioning = self_conditioning
        if self_conditioning:
            edge_in_dim += num_bins
            
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in_dim, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.time_embed = partial(
            get_timestep_embedding, embedding_dim=t_embed_size
        )
        self.position_embed = partial(
            get_positional_embedding, embedding_dim=pos_embed_size
        )
        self.distogram_embed = partial(
            calc_distogram,
            min_bin=min_bin,
            max_bin=max_bin,
            num_bins=num_bins,
        )

    def forward(
        self, 
        residue_idx, 
        t, 
        fixed_mask, 
        self_conditioning_ca,
    ):
        """
        Args:
            residue_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        B, L = residue_idx.shape
        fixed_mask = fixed_mask[..., None].float()
        node_feats = []
        pair_feats = []
        
        # configure time embedding
        t_embed = torch.tile(self.time_embed(t)[:, None, :], (1, L, 1))
        t_embed = torch.cat([t_embed, fixed_mask], dim=-1)
        node_feats.append(t_embed)
        
        # make pair embedding from 1d time feats
        concat_1d = torch.cat(
                        [torch.tile(t_embed[:, :, None, :], (1, 1, L, 1)),
                        torch.tile(t_embed[:, None, :, :], (1, L, 1, 1))], 
                    dim=-1).float().reshape([B, L**2, -1])
        pair_feats.append(concat_1d)

        # positional embedding
        node_feats.append(self.position_embed(residue_idx))
        
        # relative 2d positional embedding
        rel_seq_offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([B, L**2])
        pair_feats.append(self.position_embed(rel_seq_offset))

        # self-conditioning distogram of C-alpha atoms
        if self.self_conditioning:
            ca_dist = self.distogram_embed(self_conditioning_ca)
            pair_feats.append(ca_dist.reshape([B, L**2, -1]))

        node_embed = self.node_embed(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embed(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([B, L, L, -1])
        return node_embed, edge_embed


class DenoisingNet(nn.Module):
    def __init__(self, 
                embedder: nn.Module, 
                translator: nn.Module,
    ):
        super(DenoisingNet, self).__init__()
        self.embedder = embedder    # embedding module
        self.translator = translator    # translationIPA

    def forward(self, batch, as_tensor_7=False):
        """Forward computes the denoised frames p(X^t|X^{t+1})
        """
        # Frames as [batch, res, 7] tensors.
        node_mask = batch['residue_mask'].type(torch.float)  # [B, N]
        fixed_mask = batch['fixed_mask'].type(torch.float)
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        # Get embeddings.
        node_embed, edge_embed = self.embedder(
            residue_idx=batch['residue_idx'],
            t=batch['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=batch['sc_ca_t'],
        )
        node_embed = node_embed * node_mask[..., None] # (L, D)
        edge_embed = edge_embed * edge_mask[..., None] # (L, L, D)
        
        # Translation for frames.
        model_out = self.translator(node_embed, edge_embed, batch)
       
        # Psi angle prediction
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = gt_psi * fixed_mask[..., None] + model_out['psi'] * (1 - fixed_mask[..., None])
        rigids_pred = model_out['out_rigids']
        
        bb_representations = compute_backbone(
            rigids_pred, psi_pred, aatype=batch['aatype'] if 'aatype' in batch else None
        )
        atom37_pos = bb_representations[0].to(rigids_pred.device)
        atom14_pos = bb_representations[-1].to(rigids_pred.device)
        
        if as_tensor_7:
            rigids_pred = rigids_pred.to_tensor_7()
            
        return {
            'rigids': rigids_pred,
            'psi': psi_pred,
            'atom37': atom37_pos,
            'atom14': atom14_pos,
        }