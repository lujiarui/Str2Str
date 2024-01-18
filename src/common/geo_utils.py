"""
Utility functions for geometric operations (torch only).
"""
import torch


def rots_mul_vecs(m, v):
  """(Batch) Apply rotations 'm' to vectors 'v'."""
  return torch.stack([
        m[..., 0, 0] * v[..., 0] + m[..., 0, 1] * v[..., 1] + m[..., 0, 2] * v[..., 2],
        m[..., 1, 0] * v[..., 0] + m[..., 1, 1] * v[..., 1] + m[..., 1, 2] * v[..., 2],
        m[..., 2, 0] * v[..., 0] + m[..., 2, 1] * v[..., 1] + m[..., 2, 2] * v[..., 2],
  ], dim=-1)
  
def distance(p, eps=1e-10):
    """Calculate distance between a pair of points (dim=-2)."""
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5

def dihedral(p, eps=1e-10):
    """Calculate dihedral angle between a quadruple of points (dim=-2)."""
    # p: [*, 4, 3]

    # [*, 3]
    u1 = p[..., 1, :] - p[..., 0, :]
    u2 = p[..., 2, :] - p[..., 1, :]
    u3 = p[..., 3, :] - p[..., 2, :]

    # [*, 3]
    u1xu2 = torch.cross(u1, u2, dim=-1)
    u2xu3 = torch.cross(u2, u3, dim=-1)

    # [*]
    u2_norm = (eps + torch.sum(u2 ** 2, dim=-1)) ** 0.5
    u1xu2_norm = (eps + torch.sum(u1xu2 ** 2, dim=-1)) ** 0.5
    u2xu3_norm = (eps + torch.sum(u2xu3 ** 2, dim=-1)) ** 0.5

    # [*]
    cos_enc = torch.einsum('...d,...d->...', u1xu2, u2xu3)/ (u1xu2_norm * u2xu3_norm)
    sin_enc = torch.einsum('...d,...d->...', u2, torch.cross(u1xu2, u2xu3, dim=-1)) /  (u2_norm * u1xu2_norm * u2xu3_norm)

    return torch.stack([cos_enc, sin_enc], dim=-1)

def calc_distogram(pos: torch.Tensor, min_bin: float, max_bin: float, num_bins: int):
    # pos: [*, L, 3]
    dists_2d = torch.linalg.norm(
        pos[..., :, None, :] - pos[..., None, :, :], axis=-1
    )[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    distogram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return distogram

def rmsd(xyz1, xyz2):
    """ Abbreviation for squared_deviation(xyz1, xyz2, 'rmsd') """
    return squared_deviation(xyz1, xyz2, 'rmsd')

def squared_deviation(xyz1, xyz2, reduction='none'):
    """Squared point-wise deviation between two point clouds after alignment.
    
    Args:
        xyz1: (*, L, 3), to be transformed
        xyz2: (*, L, 3), the reference 
    
    Returns:
        rmsd: (*, ) or none: (*, L)
    """
    map_to_np = False
    if not torch.is_tensor(xyz1):
        map_to_np = True
        xyz1 = torch.as_tensor(xyz1)
        xyz2 = torch.as_tensor(xyz2)
    
    R, t = _find_rigid_alignment(xyz1, xyz2)

    # print(R.shape, t.shape) # B, 3, 3 & B, 3
    xyz1_aligned = (R.bmm(xyz1.transpose(-2,-1))).transpose(-2,-1) + t.unsqueeze(1)
    sd = ((xyz1_aligned - xyz2)**2).sum(dim=-1)    # (*, L)
    
    assert sd.shape == xyz1.shape[:-1]  
    if reduction == 'none':
        pass
    elif reduction == 'rmsd':
        sd = torch.sqrt(sd.mean(dim=-1))
    else:
        raise NotImplementedError()
    
    sd = sd.numpy() if map_to_np else sd
    return sd

def _find_rigid_alignment(src, tgt):
    """Inspired by https://research.pasteur.fr/en/member/guillaume-bouvier/;
        https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
    
    Args:
        src: Torch tensor of shape (*, L, 3) -- Point Cloud to Align (source)
        tgt: Torch tensor of shape (*, L, 3) -- Reference Point Cloud (target)
    Returns:
        R: optimal rotation (*, 3, 3)
        t: optimal translation (*, 3)
        
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    assert src.shape[-2] > 1
    src_com = src.mean(dim=-2, keepdim=True)
    tgt_com = tgt.mean(dim=-2, keepdim=True)
    src_centered = src - src_com
    tgt_centered = tgt - tgt_com

    # Covariance matrix
    H = src_centered.transpose(-2,-1).bmm(tgt_centered)    # *, 3, 3
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.bmm(U.transpose(-2,-1))
    # Translation vector
    t = tgt_com - R.bmm(src_com.transpose(-2,-1)).transpose(-2,-1)
    return R, t.squeeze(-2) # (B, 3, 3), (B, 3)
