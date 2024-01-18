from typing import Dict, Optional
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy import interpolate


##################################################
FONTSIZE = 18
##################################################
    
    
def scatterplot_2d(
    data_dict: Dict, 
    save_to: str,  
    ref_key: str = 'target',
    xlabel: str = 'tIC1',
    ylabel: str = 'tIC2',
    n_max_point: int = 1000,
    pop_ref: bool = False,
    xylim_key: bool = 'PDB_clusters', 
    plot_kde: bool = False,
    density_mapping: Optional[Dict] = None
):
    # configure min max
    if xylim_key and xylim_key in data_dict:
        xylim = data_dict.pop(xylim_key)
        # plot
        x_max = max(xylim[:,0]) 
        x_min = min(xylim[:,0]) 
        y_max = max(xylim[:,1]) 
        y_min = min(xylim[:,1])
    else:
        xylim = None
        x_max = max(data_dict[ref_key][:,0]) 
        x_min = min(data_dict[ref_key][:,0]) 
        y_max = max(data_dict[ref_key][:,1]) 
        y_min = min(data_dict[ref_key][:,1])
        
    # Add margin.
    x_min -= (x_max - x_min)/5.0
    x_max += (x_max - x_min)/5.0
    y_min -= (y_max - y_min)/5.0
    y_max += (y_max - y_min)/5.0
    
    # Remove reference data to save time.
    if pop_ref:
        data_dict.pop(ref_key)
    
    # plot tica
    print(f">>> Plotting scatter in 2D space. Image save to {save_to}")
    
    # Configure subplots.
    plot_n_row = len(data_dict) // 5 if len(data_dict) > 5 else 1  # at most 6 columns
    plot_n_columns = len(data_dict) // plot_n_row if len(data_dict) > 5 else len(data_dict)
    plt.figure(figsize=(6 * plot_n_columns , plot_n_row * 6))

    i = 0
    for k, v in data_dict.items():
        i += 1 
        plt.subplot(plot_n_row, plot_n_columns, i)
        
        if k != ref_key and v.shape[0] > n_max_point:    # subsample for visualize
            idx = np.random.choice(v.shape[0], n_max_point, replace=False)
            v = v[idx]
        
        if v.shape[0] < v.shape[1]:
            print(f"Warning: {k} has more dimensions than samples, using uniform density.")
            density = np.ones_like(v[:,0])
            density /= density.sum()
        else:
            cov = np.transpose(v)
            density = gaussian_kde(cov)(cov)
        
        # Optional precomputed density mapping.
        if density_mapping and k in density_mapping:
            density = density_mapping[k]

        plt.scatter(v[:, 0], v[:,1], s=10, alpha=0.7, c=density, cmap="mako_r", vmin=-0.05, vmax=0.40)
        # sns.scatterplot(x=v[:, 0], y=v[:,1], s=10, alpha=0.7, c=density, cmap="mako_r", vmin=-0.05, vmax=0.40)
        
        if plot_kde:
            sns.kdeplot(x=data_dict[ref_key][:, 0], y=data_dict[ref_key][:,1])    # landscape
        
        if xylim is not None:
            plt.scatter(xylim[:,0], xylim[:,1], s=40, marker="o", c="none", edgecolors="tab:red")   # cluster centers
        
        plt.xlabel(xlabel, fontsize=FONTSIZE, fontfamily="sans-serif")
        if (i-1) % plot_n_columns == 0:
            plt.ylabel(ylabel, fontsize=FONTSIZE, fontfamily="sans-serif")
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(k, fontsize=FONTSIZE, fontfamily="sans-serif")
                
    plt.tight_layout()
    plt.savefig(save_to, dpi=500)