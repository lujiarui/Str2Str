"""Inspired by https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/r3_diffuser.py"""

from math import sqrt
import torch

from src.utils.tensor_utils import inflate_array_like

class R3Diffuser:
    """VPSDE diffusion module."""
    def __init__(
        self, 
        min_b: float = 0.1,
        max_b: float = 20.0,
        coordinate_scaling: float = 1.0,
    ):
        self.min_b = min_b
        self.max_b = max_b
        self.coordinate_scaling = coordinate_scaling

    def scale(self, x):
        return x * self.coordinate_scaling

    def unscale(self, x):
        return x / self.coordinate_scaling

    def b_t(self, t: torch.Tensor):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t):
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        return -0.5 * self.b_t(t) * x

    def sample_prior(self, shape, device=None):
        return torch.randn(size=shape, device=device)

    def marginal_b_t(self, t):
        return t*self.min_b + 0.5*(t**2)*(self.max_b-self.min_b)

    def calc_trans_0(self, score_t, x_t, t):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (score_t * cond_var + x_t) / torch.exp(-0.5*beta_t)

    def forward_marginal(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor
    ):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        t = inflate_array_like(t, x_0)
        x_0 = self.scale(x_0)
        
        loc = torch.exp(-0.5 * self.marginal_b_t(t)) * x_0
        scale = torch.sqrt(1 - torch.exp(-self.marginal_b_t(t)))
        z = torch.randn_like(x_0)
        x_t = z * scale + loc
        score_t = self.score(x_t, x_0, t)
        
        x_t = self.unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: torch.Tensor):
        return 1.0 / torch.sqrt(self.conditional_var(t))

    def reverse(
        self,
        x_t: torch.Tensor,
        score_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        mask: torch.Tensor = None,
        center: bool = True,
        noise_scale: float = 1.0,
        probability_flow: bool = True,
    ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.
            probability_flow: whether to use probability flow ODE.

        Returns:
            [..., 3] positions at next step t-1.
        """
        t = inflate_array_like(t, x_t)
        x_t = self.scale(x_t)
        
        f_t = self.drift_coef(x_t, t)
        g_t = self.diffusion_coef(t)
        
        z = noise_scale * torch.randn_like(score_t)
        
        rev_drift = (f_t - g_t ** 2 * score_t) * dt * (0.5 if probability_flow else 1.)
        rev_diffusion = 0. if probability_flow else (g_t * sqrt(dt) * z)
        perturb = rev_drift + rev_diffusion

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = torch.ones_like(x_t[..., 0])
        x_t_1 = x_t - perturb   # reverse in time
        if center:
            com = torch.sum(x_t_1, dim=-2) / torch.sum(mask, dim=-1)[..., None] # reduce length dim
            x_t_1 -= com[..., None, :]
        
        x_t_1 = self.unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(xt|x0).
        Var[x_t|x_0] = conditional_var(t) * I
        """
        return 1.0 - torch.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, scale=False):
        t = inflate_array_like(t, x_t)
        if scale: 
            x_t, x_0 = self.scale(x_t), self.scale(x_0)
        return -(x_t - torch.exp(-0.5 * self.marginal_b_t(t)) * x_0) / self.conditional_var(t)

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self.scale(x_t)
        f_t = self.drift_coef(x_t, t)
        g_t = self.diffusion_coef(t)
        std = g_t * sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std