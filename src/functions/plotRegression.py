import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import *

class PlotRegression:
    """
    Class to plot regression models used for threshold personalization.
    It can plot linear and polynomial regressions with confidence intervals.
    """
    def __init__(self, x, y, dominant_impared=None, modeltype='linear'):
        self.x = x.reshape(-1, 1)  # Reshape for scikit-learn compatibility
        self.y = y
        self.threshold_std = np.full_like(self.x, None, dtype=np.float64)
        self.dominant_impared = dominant_impared

        self.linear_model = None
        self.poly_models = {}
        if modeltype == 'linear':
            self.linear_model = True
        elif modeltype == 'polynomial':
            self.poly_models = True
            self.polydegree = 2
        else:
            raise ValueError(f"Invalid model type: {modeltype}. Use 'linear' or 'polynomial'.")

    def plot_regressions(self, title='Regression Plots', xlabel='x', ylabel='y'):
        colors = thesis_style.get_thesis_colours()
            
        fig, ax = plt.subplots(figsize=(10, 5))
            
        if self.dominant_impared is not None:
            dominant_legend_plotted = False
            non_dominant_legend_plotted = False
            dominant_color, non_dominant_color = colors['orange'], colors['light_orange']
            if 'unaffected' in title.lower():
                dominant_color, non_dominant_color = colors['turquoise'], colors['light_turquoise']

            for i, (x, y, err, is_dom_impaired) in enumerate(zip(self.x, self.y, self.threshold_std, self.dominant_impared)):
                if is_dom_impaired:
                    plt.errorbar(x, y, yerr=err, fmt="+", markersize=14, markeredgewidth=3, color=dominant_color, label='Thresholds of dominant side affected subjects' if not dominant_legend_plotted else "")
                    dominant_legend_plotted = True
                else:
                    plt.errorbar(x, y, yerr=err, fmt="x", markersize=13, markeredgewidth=3, color=non_dominant_color, label='Thresholds of non-dominant side affected subjects' if not non_dominant_legend_plotted else "")
                    non_dominant_legend_plotted = True
        else:
            plt.errorbar(self.x, self.y, yerr=self.threshold_std, fmt="x", color=colors['dark_blue'], label='Optimized thresholds ($\sigma$ over k-folds)')
            
        # Plot linear regression with confidence interval
        if self.linear_model is not None:
            sns.regplot(x=self.x, y=self.y, label=f'Linear regression with 95% confidence interval (CI)', color=colors['dark_blue'], ci=95, scatter=False, line_kws={'linewidth': 3})

        # Plot polynomial regressions with confidence interval
        if self.poly_models is not None:
            sns.regplot(x=self.x, y=self.y, order=self.polydegree, label=f'{self.polydegree}nd degree polynomial regression \nwith 95% confidence interval (CI)', color=colors['dark_blue'], ci=95, scatter=False, line_kws={'linewidth': 3})

        # Plot mean line    
        plt.axhline(y=np.mean(self.y), color=colors['dark_blue'], linestyle='dotted', label='Optimal threshold (mean)', linewidth=3)

        # Plot threshold line if relevant
        if ylabel == 'Counts per second':
            plt.axhline(y=0.0, color=colors['black_grey'], linestyle='--', label='Conventional threshold')
            ax.set_ylim([-0.15, max(self.y) + 1.1*max(self.threshold_std)])
        elif ylabel == 'Forearm elevation ±':
            plt.axhline(y=30.0, color=colors['black_grey'], linestyle='--', label='Conventional threshold')
            ax.set_yticks([30, 35, 40, 45, 50, 55, 60, 65])
            ax.set_yticklabels(['30°', '35°', '40°', '45°', '50°', '55°', '60°', '65°'])
        else:
            raise ValueError(f"Invalid ylabel: {ylabel}")
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, reverse=True)
        plt.tight_layout()
        plt.show()