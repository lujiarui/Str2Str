from typing import Optional
import torch

from src.models.score import so3, r3
from src.common.rigid_utils import Rigid, Rotation, quat_multiply
from src.common import rotation3d


def assemble_rigid(rotvec: torch.Tensor, trans: torch.Tensor):
    rotvec_shape = rotvec.shape
    rotmat = rotation3d.axis_angle_to_matrix(rotvec).view(rotvec_shape[:-1] + (3, 3))
    return Rigid(
        rots=Rotation(rot_mats=rotmat),
        trans=trans,
    )

def apply_mask(x_tgt, x_src, tgt_mask):
    return tgt_mask * x_tgt + (1 - tgt_mask) * x_src


class FrameDiffuser:
    """
    Wrapper class for diffusion of rigid body transformations,
        including rotations and translations.
    """
    def __init__(self, 
                 trans_diffuser: Optional[r3.R3Diffuser] = None,
                 rot_diffuser: Optional[so3.SO3Diffuser] = None,
                 min_t: float = 0.001,
    ):
        # if None, then no diffusion for this component
        self.trans_diffuser = trans_diffuser
        self.rot_diffuser = rot_diffuser
        self.min_t = min_t

    def forward_marginal(
        self,
        rigids_0: Rigid,
        t: torch.Tensor,
        diffuse_mask: torch.Tensor = None,
        as_tensor_7: bool = True,
    ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
        Dict contains:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true. 
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        output = {}
        rot_0 = rotation3d.matrix_to_axis_angle(rigids_0.get_rots().get_rot_mats())
        trans_0 = rigids_0.get_trans()
        
        if self.rot_diffuser is None:
            rot_t = rot_0
            rot_score, rot_score_scaling = torch.zeros_like(rot_0), t
        else:
            rot_t, rot_score = self.rot_diffuser.forward_marginal(rot_0, t)
            rot_score_scaling = self.rot_diffuser.score_scaling(t)

        if self.trans_diffuser is None:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                torch.zeros_like(trans_0),
                torch.ones_like(t)
            )
        else:
            trans_t, trans_score = self.trans_diffuser.forward_marginal(trans_0, t)
            trans_score_scaling = self.trans_diffuser.score_scaling(t)

        # Perturb only a subset of residues
        if diffuse_mask is not None:
            diffuse_mask = torch.as_tensor(diffuse_mask, device=trans_t.device, dtype=trans_t.dtype)[..., None]
            
            rot_t = apply_mask(rot_t, rot_0, diffuse_mask)
            trans_t = apply_mask(trans_t, trans_0, diffuse_mask)

            trans_score = apply_mask(
                trans_score,
                torch.zeros_like(trans_score),
                diffuse_mask
            )
            rot_score = apply_mask(
                rot_score,
                torch.zeros_like(rot_score),
                diffuse_mask
            )
        
        rigids_t = assemble_rigid(rot_t, trans_t)
        
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
            
        output = {
            'rigids_t': rigids_t,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }
        return output

    def score(
        self,
        rigids_0: Rigid,
        rigids_t: Rigid,
        t: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        rot_0, trans_0  = rigids_0.get_rots(), rigids_0.get_trans()
        rot_t, trans_t = rigids_t.get_rots(), rigids_t.get_trans()

        if self.rot_diffuser is None:
            rot_score = torch.zeros_like(rot_0)
        else:
            rot_0_inv = rot_0.invert()
            quat_0_inv = rotation3d.matrix_to_quaternion(rot_0_inv.get_rot_mats())
            quat_t = rotation3d.matrix_to_quaternion(rot_t.get_rot_mats())
            # get relative rotation
            quat_0t = quat_multiply(quat_0_inv, quat_t)
            rotvec_0t = rotation3d.quaternion_to_axis_angle(quat_0t)
            # calculate score
            rot_score = self.rot_diffuser.score(rotvec_0t, t)

        if self.trans_diffuser is None:
            trans_score = torch.zeros_like(trans_0)
        else:
            trans_score = self.trans_diffuser.score(trans_t, trans_0, t, scale=True)
        
        if mask is not None:
            trans_score = trans_score * mask[..., None]
            rot_score = rot_score * mask[..., None]
        
        return {
            'trans_score': trans_score,
            'rot_score': rot_score
        }

    def score_scaling(self, t):
        rot_score_scaling = self.rot_diffuser.score_scaling(t)
        trans_score_scaling = self.trans_diffuser.score_scaling(t)
        return {
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }
    
    def reverse(
        self,
        rigids_t: Rigid,
        rot_score: torch.Tensor,
        trans_score: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        diffuse_mask: torch.Tensor = None,
        center_trans: bool = True,
        noise_scale: float = 1.0,
        probability_flow: bool = True,
    ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigids_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center_trans: true to set center of mass to zero after step
            probability_flow: whether to use probability flow ODE.

        Returns:
            rigids_t_1: [..., N] protein rigid objects at time t-1.
        """
        # extract rot and trans as tensors
        rot_t = rotation3d.matrix_to_axis_angle(rigids_t.get_rots().get_rot_mats())
        trans_t = rigids_t.get_trans()
    
        # reverse rot
        rot_t_1 = self.rot_diffuser.reverse(
            rot_t=rot_t,
            score_t=rot_score,
            t=t,
            dt=dt,
            noise_scale=noise_scale,
            probability_flow=probability_flow,
        ) if self.rot_diffuser is not None else rot_t   # if no diffusion module, return as-is

        # reverse trans
        trans_t_1 = self.trans_diffuser.reverse(
            x_t=trans_t,
            score_t=trans_score,
            t=t,
            dt=dt,
            center=center_trans,
            noise_scale=noise_scale,
            probability_flow=probability_flow,
        ) if self.trans_diffuser is not None else trans_t

        # apply mask
        if diffuse_mask is not None:
            trans_t_1 = apply_mask(trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = apply_mask(rot_t_1, rot_t, diffuse_mask[..., None])

        return assemble_rigid(rot_t_1, trans_t_1)

    def sample_prior(
        self,
        shape: torch.Size,
        device: torch.device,
        reference_rigids: Rigid = None,
        diffuse_mask: torch.Tensor = None,
        as_tensor_7: bool = False
    ):
        """Samples rigids from reference distribution.

        """
        if reference_rigids is not None:
            assert reference_rigids.shape[:-1] == shape, f"reference_rigids.shape[:-1] = {reference_rigids.shape[:-1]}, shape = {shape}"
            assert diffuse_mask is not None, "diffuse_mask must be provided if reference_rigids is given"
            rot_ref = rotation3d.matrix_to_axis_angle(reference_rigids.get_rots().get_rot_mats())
            trans_ref = reference_rigids.get_trans()
            
            trans_ref = self.trans_diffuser.scale(trans_ref)
        else:
            # sanity check
            assert diffuse_mask is None, "diffuse_mask must be None if reference_rigids is None"
            assert self.rot_diffuser is not None and self.trans_diffuser is not None
        
        # sample from prior
        trans_shape, rot_shape = shape + (3, ), shape + (3, )
        rot_sample = self.rot_diffuser.sample_prior(shape=rot_shape, device=device) \
            if self.rot_diffuser is not None else rot_ref
        trans_sample = self.trans_diffuser.sample_prior(shape=trans_shape, device=device) \
            if self.trans_diffuser is not None else trans_ref
        
        # apply mask
        if diffuse_mask is not None:
            rot_sample = apply_mask(rot_sample, rot_ref, diffuse_mask[..., None])
            trans_sample = apply_mask(trans_sample, trans_ref, diffuse_mask[..., None])
             
        trans_sample = self.trans_diffuser.unscale(trans_sample)
        
        # assemble sampled rot and trans -> rigid
        rigids_t = assemble_rigid(rot_sample, trans_sample)
        
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        
        return {'rigids_t': rigids_t}