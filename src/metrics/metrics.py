import os
from typing import *

import numpy as np
from scipy.spatial import distance
from deeptime.decomposition import TICA

EPS = 1e-12
PSEUDO_C = 1e-6


def adjacent_ca_distance(coords):
    """Calculate distance array for a single chain of CA atoms. Only k=1 neighbors.
    Args:
        coords: (..., L, 3)
    return 
        dist: (..., L-1)
    """    
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dX = coords[..., :-1, :] - coords[..., 1:, :] # (..., L-1, 3)
    dist = np.sqrt(np.sum(dX**2, axis=-1))
    return dist # (..., L-1)


def distance_matrix_ca(coords):
    """Calculate distance matrix for a single chain of CA atoms. W/o exclude neighbors.
    Args:
        coords: (..., L, 3)
    Return:
        dist: (..., L, L)
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dX = coords[..., None, :, :] - coords[..., None, :] # (..., L, L, 3)
    dist = np.sqrt(np.sum(dX**2, axis=-1))
    return dist # (..., L, L)


def pairwise_distance_ca(coords, k=1):
    """Calculate pairwise distance vector for a single chain of CA atoms. W/o exclude neighbors.
    Args:
        coords: (..., L, 3)
    Return:
        dist: (..., D) (D=L * (L - 1) // 2) when k=1)
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dist = distance_matrix_ca(coords)
    L = dist.shape[-1]
    row, col = np.triu_indices(L, k=k)
    triu = dist[..., row, col]  # unified (but unclear) order
    return triu # (..., D)


def radius_of_gyration(coords, masses=None):
    """Compute the radius of gyration for every frame.
    
    Args:
        coords: (..., num_atoms, 3)
        masses: (num_atoms,)
        
    Returns:
        Rg: (..., )
        
    If masses are none, assumes equal masses.
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    
    if masses is None:
        masses = np.ones(coords.shape[-2])
    else:
        assert len(masses.shape) == 1, f"masses should be 1D, got {masses.shape}"
        assert masses.shape[0] == coords.shape[-2], f"masses {masses.shape} != number of particles {coords.shape[-2]}"

    weights = masses / masses.sum()
    centered = coords - coords.mean(-2, keepdims=True) 
    squared_dists = (centered ** 2).sum(-1)
    Rg = (squared_dists * weights).sum(-1) ** 0.5
    return Rg


def _steric_clash(coords, ca_vdw_radius=1.7, allowable_overlap=0.4, k_exclusion=0):
    """ https://www.schrodinger.com/sites/default/files/s3/public/python_api/2022-3/_modules/schrodinger/structutils/interactions/steric_clash.html#clash_iterator
    Calculate the number of clashes in a single chain of CA atoms.
    
    Usage: 
        n_clash = calc_clash(coords)
    
    Args:
        coords: (n_atoms, 3), CA coordinates, coords should from one protein chain.
        ca_vdw_radius: float, default 1.7.
        allowable_overlap: float, default 0.4.
        k_exclusion: int, default 0. Exclude neighbors within [i-k-1, i+k+1].
        
    """
    assert np.isnan(coords).sum() == 0, "coords should not contain nan"
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    assert k_exclusion >= 0, "k_exclusion should be non-negative"
    bar = 2 * ca_vdw_radius - allowable_overlap
    # L = len(coords)
    # dist = np.sqrt(np.sum((coords[:L-k_exclusion, None, :] - coords[None, k_exclusion:, :])**2, axis=-1))   
    pwd = pairwise_distance_ca(coords, k=k_exclusion+1) # by default, only excluding self (k=1)
    assert len(pwd.shape) == 2, f"pwd should be 2D, got {pwd.shape}"
    n_clash = np.sum(pwd < bar, axis=-1)
    return n_clash.astype(int) #(..., )  #np.prod(dist.shape)


def validity(ca_coords_dict, **clash_kwargs):
    """Calculate clash validity of ensembles. 
    Args:
        ca_coords_dict: {k: (B, L, 3)}
    Return:
        valid: {k: validity in [0,1]}
    """
    n_clash = {
        k: _steric_clash(v, **clash_kwargs)
            for k, v in ca_coords_dict.items()
    }
    results = {
        k: 1.0 - (v>0).mean() for k, v in n_clash.items()
    }
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results


def bonding_validity(ca_coords_dict, ref_key='target', eps=1e-6):
    """Calculate bonding dissociation validity of ensembles."""
    adj_dist = {k: adjacent_ca_distance(v)
            for k, v in ca_coords_dict.items()
    }
    thres = adj_dist[ref_key].max()+ 1e-6
    
    results = {
        k: (v < thres).all(-1).sum().item() / len(v) 
            for k, v in adj_dist.items()
    }
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results


def js_pwd(ca_coords_dict, ref_key='target', n_bins=50, pwd_offset=3, weights=None):
    # n_bins = 50 follows idpGAN
    # k=3 follows 2for1
    
    ca_pwd = {
        k: pairwise_distance_ca(v, k=pwd_offset) for k, v in ca_coords_dict.items()
    }   # (B, D)
    
    if weights is None:
        weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
        
    d_min = ca_pwd[ref_key].min(axis=0) # (D, )
    d_max = ca_pwd[ref_key].max(axis=0)
    ca_pwd_binned = {
        k: np.apply_along_axis(lambda a: np.histogram(a[:-2], bins=n_bins, weights=weights[k], range=(a[-2], a[-1]))[0]+PSEUDO_C, 0, 
                            np.concatenate([v, d_min[None], d_max[None]], axis=0))
        for k, v in ca_pwd.items()
    }   # (N_bins, D)-> (N_bins * D, )
    # js divergence per channel and average
    results = {k: distance.jensenshannon(v, ca_pwd_binned[ref_key], axis=0).mean() 
                    for k, v in ca_pwd_binned.items() if k != ref_key}
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results


def js_tica(ca_coords_dict, ref_key='target', n_bins=50, lagtime=20, return_tic=True, weights=None):
    # n_bins = 50 follows idpGAN
    
    ca_pwd = {
        k: pairwise_distance_ca(v) for k, v in ca_coords_dict.items()
    }   # (B, D)
    estimator = TICA(dim=2, lagtime=lagtime).fit(ca_pwd[ref_key])
    tica = estimator.fetch_model()
    # dimension reduction into 2D
    ca_dr2d = {  
        k: tica.transform(v) for k, v in ca_pwd.items()
    }
    if weights is None: weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
    
    d_min = ca_dr2d[ref_key].min(axis=0) # (D, )
    d_max = ca_dr2d[ref_key].max(axis=0)
    ca_dr2d_binned = {
        k: np.apply_along_axis(lambda a: np.histogram(a[:-2], bins=n_bins, weights=weights[k], range=(a[-2], a[-1]))[0]+PSEUDO_C, 0, 
                            np.concatenate([v, d_min[None], d_max[None]], axis=0))
                for k, v in ca_dr2d.items()
    }   # (N_bins, 2) 
    results = {k: distance.jensenshannon(v, ca_dr2d_binned[ref_key], axis=0).mean() 
                for k, v in ca_dr2d_binned.items() if k != ref_key}
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    if return_tic:
        return results, ca_dr2d
    return results


def js_rg(ca_coords_dict, ref_key='target', n_bins=50, weights=None):
    ca_rg = {
        k: radius_of_gyration(v) for k, v in ca_coords_dict.items()
    }   # (B, )
    if weights is None:
        weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
        
    d_min = ca_rg[ref_key].min() # (1, )
    d_max = ca_rg[ref_key].max()
    ca_rg_binned = {
        k: np.histogram(v, bins=n_bins, weights=weights[k], range=(d_min, d_max))[0]+PSEUDO_C 
            for k, v in ca_rg.items()
    }   # (N_bins, )
    # print("ca_rg_binned shape", {k: v.shape for k, v in ca_rg_binned.items()})
    results = {k: distance.jensenshannon(v, ca_rg_binned[ref_key], axis=0).mean() 
                for k, v in ca_rg_binned.items() if k != ref_key}
    
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results