"""Inspired by https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/so3_diffuser.py"""

import os
import math

import numpy as np
import torch

from src.utils.tensor_utils import inflate_array_like
from src.common import rotation3d


def compose_rotvec(rotvec1, rotvec2):
    """Compose two rotation euler vectors."""
    dtype = rotvec1.dtype
    R1 = rotation3d.axis_angle_to_matrix(rotvec1)
    R2 = rotation3d.axis_angle_to_matrix(rotvec2)
    cR = torch.einsum('...ij,...jk->...ik', R1.double(), R2.double())
    return rotation3d.matrix_to_axis_angle(cR).type(dtype)

def igso3_expansion(omega, eps, L=1000, use_torch=False):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, eps =
    sqrt(2) * eps_leach, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=eps^2.

    Args:
        omega: rotation of Euler vector (i.e. the angle of rotation)
        eps: std of IGSO(3).
        L: Truncation level
        use_torch: set true to use torch tensors, otherwise use numpy arrays.
    """

    lib = torch if use_torch else np
    ls = lib.arange(L)
    if use_torch:
        ls = ls.to(omega.device)
    
    if len(omega.shape) == 2:
        # Used during predicted score calculation.
        ls = ls[None, None]  # [1, 1, L]
        omega = omega[..., None]  # [num_batch, num_res, 1]
        eps = eps[..., None]
    elif len(omega.shape) == 1:
        # Used during cache computation.
        ls = ls[None]  # [1, L]
        omega = omega[..., None]  # [num_batch, 1]
    else:
        raise ValueError("Omega must be 1D or 2D.")
    p = (2*ls + 1) * lib.exp(-ls*(ls+1)*eps**2/2) * lib.sin(omega*(ls+1/2)) / lib.sin(omega/2)
    if use_torch:
        return p.sum(dim=-1)
    else:
        return p.sum(axis=-1)


def density(expansion, omega, marginal=True, use_torch=False):
    """IGSO(3) density.

    Args:
        expansion: truncated approximation of the power series in the IGSO(3)
        density.
        omega: length of an Euler vector (i.e. angle of rotation)
        marginal: set true to give marginal density over the angle of rotation,
            otherwise include normalization to give density on SO(3) or a
            rotation with angle omega.
    """
    lib = torch if use_torch else np
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1.0 - lib.cos(omega)) / np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / np.pi**2


def score(exp, omega, eps, L=1000, use_torch=False):  # score of density over SO(3)
    """score uses the quotient rule to compute the scaling factor for the score
    of the IGSO(3) density.

    This function is used within the Diffuser class to when computing the score
    as an element of the tangent space of SO(3).

    This uses the quotient rule of calculus, and take the derivative of the
    log:
        d hi(x)/lo(x) = (lo(x) d hi(x)/dx - hi(x) d lo(x)/dx) / lo(x)^2
    and
        d log expansion(x) / dx = (d expansion(x)/ dx) / expansion(x)

    Args:
        exp: truncated expansion of the power series in the IGSO(3) density
        omega: length of an Euler vector (i.e. angle of rotation)
        eps: scale parameter for IGSO(3) -- as in expansion() this scaling
            differ from that in Leach by a factor of sqrt(2).
        L: truncation level
        use_torch: set true to use torch tensors, otherwise use numpy arrays.

    Returns:
        The d/d omega log IGSO3(omega; eps)/(1-cos(omega))

    """
    lib = torch if use_torch else np
    ls = lib.arange(L)
    if use_torch:
        ls = ls.to(omega.device)
    ls = ls[None]
    if len(omega.shape) == 2:
        ls = ls[None]
    elif len(omega.shape) > 2:
        raise ValueError("Omega must be 1D or 2D.")
    omega = omega[..., None]
    eps = eps[..., None]
    hi = lib.sin(omega * (ls + 1 / 2))
    dhi = (ls + 1 / 2) * lib.cos(omega * (ls + 1 / 2))
    lo = lib.sin(omega / 2)
    dlo = 1 / 2 * lib.cos(omega / 2)
    dSigma = (2 * ls + 1) * lib.exp(-ls * (ls + 1) * eps**2/2) * (lo * dhi - hi * dlo) / lo ** 2
    if use_torch:
        dSigma = dSigma.sum(dim=-1)
    else:
        dSigma = dSigma.sum(axis=-1)
    return dSigma / (exp + 1e-4)


class SO3Diffuser:
    def __init__(
        self,
        cache_dir: str = './cache', 
        schedule: str = 'logarithmic',
        min_sigma: float = 0.1,
        max_sigma: float = 1.5,
        num_sigma: int = 1000,
        num_omega: int = 1000,
        use_cached_score: bool = False,
        eps: float = 1e-6,
    ):
        self.schedule = schedule
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.use_cached_score = use_cached_score
        self.eps = eps

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.discrete_omega = torch.linspace(0, np.pi, steps=num_omega+1)[1:]

        # Configure cache directory.
        replace_period = lambda x: str(x).replace('.', '_')
        cache_dir = os.path.join(
            cache_dir,
            f'eps_{num_sigma}_omega_{num_omega}_min_sigma_{replace_period(min_sigma)}_max_sigma_{replace_period(max_sigma)}_schedule_{schedule}'
        )
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        
        pdf_cache = os.path.join(cache_dir, 'pdf_vals.pt')
        cdf_cache = os.path.join(cache_dir, 'cdf_vals.pt')
        score_norms_cache = os.path.join(cache_dir, 'score_norms.pt')

        if os.path.exists(pdf_cache) \
            and os.path.exists(cdf_cache) \
                and os.path.exists(score_norms_cache):
            self._pdf = torch.load(pdf_cache, map_location='cpu')
            self._cdf = torch.load(cdf_cache, map_location='cpu')
            self._score_norms = torch.load(score_norms_cache, map_location='cpu')
        else:
            disc_omega = self.discrete_omega.numpy()
            disc_sigma = self.discrete_sigma.numpy()
            # Compute the expansion of the power series.
            exp_vals = np.asarray(
                [igso3_expansion(disc_omega, sigma) for sigma in disc_sigma]
            )
            # Compute the pdf and cdf values for the marginal distribution of the axis-angles (readily needed for sampling)
            self._pdf  = np.asarray(
                [density(x, disc_omega, marginal=True) for x in exp_vals]
            )
            self._cdf = np.asarray(
                [pdf.cumsum() / num_omega * np.pi for pdf in self._pdf]
            )
            # Compute the norms of the scores.  This are used to scale the rotation axis when
            # computing the score as a vector.
            self._score_norms = np.asarray(
                [score(exp_vals[i], disc_omega, x) for i, x in enumerate(disc_sigma)]
            )
            self._pdf, self._cdf, self._score_norms = map(
                torch.as_tensor, (self._pdf, self._cdf, self._score_norms)
            )
            # Cache the precomputed values
            torch.save(obj=self._pdf, f=pdf_cache)
            torch.save(obj=self._cdf, f=cdf_cache)
            torch.save(obj=self._score_norms, f=score_norms_cache)

        self._score_scaling = torch.sqrt(torch.abs(
            torch.sum(self._score_norms**2 * self._pdf, dim=-1) / torch.sum(self._pdf, dim=-1)
        )) / np.sqrt(3)
        
    @property
    def discrete_sigma(self):
        return self.sigma(
            torch.linspace(0.0, 1.0, self.num_sigma)
        )

    def sigma_idx(self, sigma: torch.Tensor):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return torch.as_tensor(np.digitize(sigma.cpu().numpy(), self.discrete_sigma) - 1, 
                    dtype=torch.long)

    def sigma(self, t: torch.Tensor):
        """Extract \sigma(t) corresponding to chosen sigma schedule."""
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'logarithmic':
            return torch.log(t * math.exp(self.max_sigma) + (1 - t) * math.exp(self.min_sigma))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def diffusion_coef(self, t: torch.Tensor):
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == 'logarithmic':
            g_t = torch.sqrt(
                2 * (math.exp(self.max_sigma) - math.exp(self.min_sigma)) \
                    * self.sigma(t) / torch.exp(self.sigma(t))
            )
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')
        return g_t

    def t_to_idx(self, t: torch.Tensor):
        """Helper function to go from time t to corresponding sigma_idx."""
        return self.sigma_idx(self.sigma(t))

    def sample_prior(self, shape: torch.Size, device=None):
        t = torch.ones(shape[0], dtype=torch.float, device=device)
        return self.sample(t, shape)
    
    def sample(self, t: torch.Tensor, shape: torch.Size):
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            shape: shape of the output tensor.

        Returns:
            (shape, ) axis-angle rotation vectors sampled from IGSO(3).
        """
        assert t.ndim == 1 and t.shape[0] == shape[0], \
                f"t.shape={t.shape}, shape={shape}"
        assert shape[-1] == 3, f"The last dim should be 3, but got shape={shape}"
        
        z = torch.randn(shape, device=t.device)   # axis-angle
        x = z / torch.linalg.norm(z, dim=-1, keepdims=True)
        
        # sample igso3 
        z_igso3 = torch.rand(shape[:-1])   # (num_batch, num_res)
        igso3_scaling = []
        for i, _t in enumerate(t):
            t_idx = self.t_to_idx(_t).item()    # [1]
            # np.interp can accept Tensor as input
            igso3_scaling.append(
                np.interp(z_igso3[i], self._cdf[t_idx], self.discrete_omega), 
            )   # (num_res, )
        igso3_scaling = torch.as_tensor(np.asarray(igso3_scaling), dtype=x.dtype, device=t.device)
        
        return x * igso3_scaling[..., None]

    def score(
        self,
        vec: torch.tensor,
        t: torch.tensor,
    ):
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        assert t.ndim == 1 and t.shape[0] == vec.shape[0], \
                f"t.shape={t.shape}, vec.shape={vec.shape}"
        device = vec.device
        
        omega = torch.linalg.norm(vec, dim=-1) + self.eps   # [num_batch, num_res]
        
        if self.use_cached_score:
            score_norms_t = self._score_norms[self.t_to_idx(t)]
            score_norms_t = torch.as_tensor(score_norms_t).to(device)
            omega_idx = torch.bucketize(
                omega, torch.as_tensor(self.discrete_omega[:-1]).to(device))
            omega_scores_t = torch.gather(score_norms_t, 1, omega_idx)
        else:   # compute on the fly
            sigma = self.discrete_sigma[self.t_to_idx(t)]
            sigma = torch.as_tensor(sigma).to(device)
            omega_vals = igso3_expansion(omega, sigma[:, None], use_torch=True)
            omega_scores_t = score(omega_vals, omega, sigma[:, None], use_torch=True)
            
        return omega_scores_t[..., None] * vec / (omega[..., None] + self.eps)

    def score_scaling(self, t: torch.Tensor):
        """Calculates scaling used for scores during trianing."""
        return torch.as_tensor(self._score_scaling[self.t_to_idx(t)], device=t.device)

    def forward_marginal(self, rot_0: torch.Tensor, t: torch.Tensor):
        """Samples from the forward diffusion process at time index t.

        Args:
            rot_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_score: [..., 3] score of rot_t as a rotation vector.
        """
        rotvec_0t = self.sample(t, shape=rot_0.shape)   # "delta_rot"
        rot_score = self.score(rotvec_0t, t)
        
        # Right multiply => vector_add in Euclidean space
        rot_t = compose_rotvec(rot_0, rotvec_0t)
        return rot_t, rot_score

    def reverse(
        self,
        rot_t: torch.Tensor,
        score_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        mask: torch.Tensor = None,
        noise_scale: float = 1.0,
        probability_flow: bool = True,
    ):
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.

        Args:
            rot_t: [..., 3] current rotations at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficent to 0.
            mask: True indicates which residues to diffuse.
            probability_flow: set True to use probability flow ODE.

        Returns:
            [..., 3] rotation vector at next step.
        """
        t = inflate_array_like(t, rot_t)
        
        g_t = self.diffusion_coef(t)
        z = noise_scale * torch.randn_like(score_t)
        
        rev_drift = -1.0 * (g_t ** 2) * score_t * dt * (0.5 if probability_flow else 1.)
        rev_diffusion = 0. if probability_flow else (g_t * np.sqrt(dt) * z)
        perturb = rev_drift + rev_diffusion

        if mask is not None: 
            perturb *= mask[..., None]

        # Right multiply.
        rot_t_1 = compose_rotvec(rot_t, -1.0 * perturb)    # reverse in time        
        return rot_t_1