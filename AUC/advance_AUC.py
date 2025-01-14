from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def plot_multi_roc(results_dict, 
                  figsize=(6, 5),
                  colors=None,
                  zoom_factor=5,
                  zoom_location=2,
                  zoom_bbox=(0.5, 0.9),
                  zoom_xlim=(0, 0.1),
                  zoom_ylim=(0.9, 1.0),
                  show_std=True):
    """
    Plot multiple ROC curves with optional standard deviation bands and zoom inset.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing ROC curve data for each model with structure:
        {model_name: {
            'mean_fpr': array,
            'mean_tpr': array,
            'mean_auc': float,
            'std_auc': float,  # optional
            'std_tpr': array   # optional
        }}
    figsize : tuple, optional (default=(6, 5))
        Figure size in inches
    colors : list, optional
        List of colors for each model. If None, uses default colors
    zoom_factor : int, optional (default=5)
        Magnification factor for the zoom inset
    zoom_location : int, optional (default=2)
        Location of zoom inset (1-4 for corners, following matplotlib convention)
    zoom_bbox : tuple, optional (default=(0.5, 0.9))
        Position of the zoom box anchor
    zoom_xlim : tuple, optional (default=(0, 0.1))
        X-axis limits for zoom inset
    zoom_ylim : tuple, optional (default=(0.9, 1.0))
        Y-axis limits for zoom inset
    show_std : bool, optional (default=True)
        Whether to show standard deviation bands
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The main axes
    axins : matplotlib.axes.Axes
        The inset axes
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    
    # Set default colors if none provided
    if colors is None:
        colors = ['#424874', '#8b91c1', '#d9a9cd', '#e6d3ae']
    
    # Create figure and main axes
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Plot each model's ROC curve
    for idx, (model_name, results) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]  # Cycle through colors if more models than colors
        
        # Plot mean ROC curve
        mean_fpr = results['mean_fpr']
        mean_tpr = results['mean_tpr']
        mean_auc = results['mean_auc']
        
        ax.plot(mean_fpr, mean_tpr, color=color,
                label=f'{model_name} (AUC = {mean_auc:.4f})',
                lw=2, alpha=0.8)
        
        # Plot standard deviation if available and requested
        if show_std and 'std_tpr' in results and 'std_auc' in results:
            std_tpr = results['std_tpr']
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, 
                          color=color, alpha=0.2)
    
    # Customize main plot
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('1-Specificity')
    ax.legend(loc="lower right")
    
    # Create zoomed inset axes
    axins = zoomed_inset_axes(ax, zoom_factor, loc=zoom_location, 
                             bbox_to_anchor=zoom_bbox, 
                             bbox_transform=ax.transAxes)
    
    # Plot curves in zoom inset
    for idx, (model_name, results) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        mean_fpr = results['mean_fpr']
        mean_tpr = results['mean_tpr']
        
        axins.plot(mean_fpr, mean_tpr, color=color, lw=2)
        
        # Add standard deviation bands to zoom if available and requested
        if show_std and 'std_tpr' in results:
            std_tpr = results['std_tpr']
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            axins.fill_between(mean_fpr, tprs_lower, tprs_upper, 
                             color=color, alpha=0.2)
    
    # Set the zoom area limits
    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)
    
    # Add ticks for the zoomed plot
    xticks = np.linspace(zoom_xlim[0], zoom_xlim[1], 3)
    yticks = np.linspace(zoom_ylim[0], zoom_ylim[1], 3)
    axins.set_xticks(xticks)
    axins.set_yticks(yticks)
    
    plt.tight_layout()
    
    return fig, ax, axins
def simulate_roc_curve(desired_auc=0.8, std_scale=0.02, n_points=100, n_samples=10000):
    """
    Simulate a ROC curve by generating proper score distributions for positive and negative cases.
    
    Parameters:
    -----------
    desired_auc : float
        Target AUC value for the curve (0.5 to 1.0)
    std_scale : float
        Scale of the standard deviation bands
    n_points : int
        Number of points to generate on the ROC curve
    n_samples : int
        Number of samples to generate for each class
        
    Returns:
    --------
    dict : Dictionary containing ROC curve data
    """
    import numpy as np
    from scipy.stats import norm
    
    # The separation between means that gives us our desired AUC
    # For normal distributions, AUC = Φ(d'/√2) where d' is the separation in standard deviations
    # So d' = √2 * Φ^(-1)(AUC)
    separation = np.sqrt(2) * norm.ppf(desired_auc)
    
    # Generate samples from two normal distributions
    negative_scores = np.random.normal(0, 1, n_samples)
    positive_scores = np.random.normal(separation, 1, n_samples)
    
    # Calculate ROC curve points
    thresholds = np.linspace(min(np.min(negative_scores), np.min(positive_scores)),
                            max(np.max(negative_scores), np.max(positive_scores)),
                            n_points)
    
    mean_fpr = []
    mean_tpr = []
    
    for threshold in thresholds:
        fp = np.sum(negative_scores >= threshold)
        tp = np.sum(positive_scores >= threshold)
        
        fpr = fp / len(negative_scores)
        tpr = tp / len(positive_scores)
        
        mean_fpr.append(fpr)
        mean_tpr.append(tpr)
    
    mean_fpr = np.array(mean_fpr)
    mean_tpr = np.array(mean_tpr)
    
    # Sort points by FPR
    sort_idx = np.argsort(mean_fpr)
    mean_fpr = mean_fpr[sort_idx]
    mean_tpr = mean_tpr[sort_idx]
    
    # Add endpoints if not present
    if mean_fpr[0] != 0 or mean_tpr[0] != 0:
        mean_fpr = np.concatenate([[0], mean_fpr])
        mean_tpr = np.concatenate([[0], mean_tpr])
    if mean_fpr[-1] != 1 or mean_tpr[-1] != 1:
        mean_fpr = np.concatenate([mean_fpr, [1]])
        mean_tpr = np.concatenate([mean_tpr, [1]])
    
    # Generate standard deviation bands - larger in middle, smaller at endpoints
    std_tpr = std_scale * np.sqrt(mean_tpr * (1 - mean_tpr)) * (1 - np.power(2*mean_fpr - 1, 2))
    std_tpr[0] = 0
    std_tpr[-1] = 0
    
    # Calculate actual AUC using trapezoidal rule
    mean_auc = np.trapz(mean_tpr, mean_fpr)
    std_auc = std_scale
    
    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'std_tpr': std_tpr
    }

def generate_sample_data(std_existance=True):
    """
    Generate sample ROC curves for five different models.
    
    Returns:
    --------
    dict : Dictionary containing ROC curve data for each model
    """
    # Define base AUC values for different models
    if std_existance:
        models = {
            'Model A': {'auc': 0.95, 'std': 0.1},
            'Model B': {'auc': 0.92, 'std': 0.15},
            'Model C': {'auc': 0.88, 'std': 0.2},
            'Model D': {'auc': 0.85, 'std': 0.25},
            'Model E': {'auc': 0.82, 'std': 0.3}
        }
    else:
        models = {
            'Model A': {'auc': 0.95,},
            'Model B': {'auc': 0.92, },
            'Model C': {'auc': 0.88, },
            'Model D': {'auc': 0.85, },
            'Model E': {'auc': 0.82,}
        }
    
    results_dict = {}
    for model_name, params in models.items():
        if "std" in params:
            results_dict[model_name] = simulate_roc_curve(
                desired_auc=params['auc'],
                std_scale=params['std']
            )
        else:
            results_dict[model_name] = simulate_roc_curve(
                desired_auc=params['auc'],
                std_scale=0
            )
    
    return results_dict

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate sample data
    results_dict = generate_sample_data(std_existance=True)
    
    # Custom colors for better visualization
    colors =  ['#424874', '#8b91c1', '#d9a9cd', '#e6d3ae', '#f9e1e1']
    
    # Create the plot
    fig, ax, axins = plot_multi_roc(
        results_dict,
        figsize=(6, 5),
        colors=colors,
        zoom_factor=1.3,
        zoom_location=2,
        zoom_bbox=(0.6, 0.9),
        zoom_xlim=(0, 0.3),
        zoom_ylim=(0.7, 1.0),
        show_std=True
    )

    
    # Show the plot
    plt.show()