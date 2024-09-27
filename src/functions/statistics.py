import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel, probplot, zscore

from utilities import thesis_style

class RegressionModel:
    def __init__(self, x, y, std=None, dominant_impared=None):
        self.x = x.reshape(-1, 1)  # Reshape for scikit-learn compatibility
        self.y = y
        self.threshold_std = std
        self.linear_model = None
        self.poly_models = {}
        self.dominant_impared = dominant_impared

    def fit_linear_regression(self):
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.x, self.y)

    def fit_polynomial_regression(self, degree):
        poly_features = PolynomialFeatures(degree)
        poly_model = make_pipeline(poly_features, LinearRegression())
        poly_model.fit(self.x, self.y)
        self.poly_models[degree] = poly_model  

    def predict_linear(self, x):
        if self.linear_model is None:
            raise ValueError("Linear model is not fitted yet.")
        x = np.array(x).reshape(-1, 1)
        return self.linear_model.predict(x)

    def predict_polynomial(self, x, degree):
        if degree not in self.poly_models:
            raise ValueError(f"Polynomial model of degree {degree} is not fitted yet.")
        x = np.array(x).reshape(-1, 1)
        return self.poly_models[degree].predict(x)

    def pearson_correlation(self):
        corr, pvalue = pearsonr(self.x.flatten(), self.y)
        return corr, pvalue

    def spearman_correlation(self):
        corr, pvalue = spearmanr(self.x.flatten(), self.y)
        return corr, pvalue

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
                    plt.errorbar(x, y, yerr=err, fmt="+", markersize=14, markeredgewidth=3, color=dominant_color, label='dominant arm affected ($\sigma$ over 5-folds)' if not dominant_legend_plotted else "")
                    dominant_legend_plotted = True
                else:
                    plt.errorbar(x, y, yerr=err, fmt="x", markersize=13, markeredgewidth=3, color=non_dominant_color, label='non-dominant arm affected ($\sigma$ over 5-folds)' if not non_dominant_legend_plotted else "")
                    non_dominant_legend_plotted = True
        else:
            plt.errorbar(self.x, self.y, yerr=self.threshold_std, fmt="x", color=colors['dark_blue'], label='Optimized thresholds ($\sigma$ over k-folds)')
        
        # Plot linear regression with confidence interval
        if self.linear_model is not None:
            r_squared = self.linear_model.score(self.x, self.y)
            sns.regplot(x=self.x, y=self.y, label=f'Linear regression with 95% CI\n(R-squared: {r_squared:.2f})', color=colors['dark_blue'], ci=95, scatter=False, line_kws={'linewidth': 3})

        # Plot polynomial regressions with confidence interval
        if self.poly_models is not None:
            for degree, poly_model in self.poly_models.items():
                r_squared = poly_model.score(self.x, self.y)
                sns.regplot(x=self.x, y=self.y, order=degree, label=f'{degree}nd degree polynomial regression with 95% CI\n(R-squared: {r_squared:.2f})', color=colors['dark_blue'], ci=95, scatter=False, line_kws={'linewidth': 3})
        
        # Plot threshold line if relevant
        if ylabel == 'Counts per second':
            plt.axhline(y=0.0, color=colors['black_grey'], linestyle='--', label='Conventional threshold')
            ax.set_ylim([-0.15, max(self.y) + 1.1*max(self.threshold_std)])
        elif ylabel == 'Elevation':
            plt.axhline(y=30.0, color=colors['black_grey'], linestyle='--', label='Conventional threshold')
            ax.set_yticks([30, 35, 40, 45, 50, 55, 60, 65])
            ax.set_yticklabels(['30°', '35°', '40°', '45°', '50°', '55°', '60°', '65°'])
        else:
            raise ValueError(f"Invalid ylabel: {ylabel}")
        plt.axhline(y=np.mean(self.y), color=dominant_color, linestyle='dotted', label='Optimal threshold (mean)', linewidth=2)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, reverse=True)
        plt.tight_layout()
        plt.show()


def extract_std_from_min_max_std(min_max_std):
    COUNT_min_max_std = min_max_std[:,0]
    count_std = [item['std'] for item in COUNT_min_max_std]
    PITCH_min_max_std = min_max_std[:,1]
    pitch_std = [item['std'] for item in PITCH_min_max_std]

    return count_std, pitch_std

def check_outliers(data, z_score_threshold=3):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    IQR_outliers = np.where((data < lower_bound) | (data > upper_bound))
    z_scores = zscore(data)
    z_score_outliers = np.where(np.abs(z_scores) > z_score_threshold)
    return z_score_outliers, IQR_outliers

def check_regression(x, y, std, x_label='x', y_label='y', title='Regression Analysis', dominant_impared=None, model_to_fit='linear', remove_outliers=False):
    if remove_outliers:
        z_score_outliers, IQR_outliers = check_outliers(y)
        x = np.delete(x, IQR_outliers)
        y = np.delete(y, IQR_outliers)
        std = np.delete(std, IQR_outliers)
        if dominant_impared is not None:
            dominant_impared = np.delete(dominant_impared, IQR_outliers)

    model = RegressionModel(x, y, std, dominant_impared)

    # Check distribution of the data
    #model.check_distribution()

    if model_to_fit == 'linear':
        # Fit linear regression
        model.fit_linear_regression()
        print("Linear model coefficients:", model.linear_model.coef_)
        print("Linear model intercept:", model.linear_model.intercept_)

    elif model_to_fit == 'polynomial':
        # Fit polynomial regression of degree 2
        model.fit_polynomial_regression(2)
        print("Polynomial model coefficients (degree 2):", model.poly_models[2].steps[1][1].coef_)
        print("Polynomial model intercept (degree 2):", model.poly_models[2].steps[1][1].intercept_)
        
    # Calculate Pearson correlation
    pearson_corr, pearson_pvalue = model.pearson_correlation()
    print("Pearson Correlation:", pearson_corr, "p-value:", pearson_pvalue)

    # Calculate Spearman correlation
    spearman_corr, spearman_pvalue = model.spearman_correlation()
    print("Spearman Correlation:", spearman_corr, "p-value:", spearman_pvalue)

    # Plot regressions
    model.plot_regressions(xlabel=x_label, ylabel=y_label, title=title)

def QQplot(count_threshols_1, count_threshols_2, eleation_threshols_1, elevation_threshols_2):
    '''
    Plot two QQ-plot to check if the difference between the paired samples is normally distributed
    '''
    # Create a figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Create a QQ-plot for count thresholds
    count_diff = np.array(count_threshols_1) - np.array(count_threshols_2)
    probplot(count_diff, dist='norm', plot=ax[0])
    #qqplot(count_threshols_1, count_threshols_2, line='45', ax=ax[0])
    ax[0].set_title('Q-Q Count Thresholds')
    # Create a QQ-plot for elevation thresholds
    elevation_diff = np.array(eleation_threshols_1) - np.array(elevation_threshols_2)
    probplot(elevation_diff, dist='norm', plot=ax[1])
    #qqplot(eleation_threshols_1, elevation_threshols_2, line='45', ax=ax[1])
    ax[1].set_title('Q-Q Elevation Thresholds')
    plt.tight_layout()
    plt.show()

def degree_formatter(x, pos):
    return f'{int(x)}°'
    
def check_distribution(count_threshold_ndh, count_threshold_dh, elevation_threshold_ndh, elevation_threshold_dh, x_data_label=['Affected side', 'Unaffected side']):
    # calculate mean optimal thresholds
    mean_count_threshold_ndh = np.mean(count_threshold_ndh)
    mean_count_threshold_dh = np.mean(count_threshold_dh)
    mean_elevation_threshold_ndh = np.mean(elevation_threshold_ndh)
    mean_elevation_threshold_dh = np.mean(elevation_threshold_dh)
    print(f"Mean count threshold {x_data_label[0]}: {mean_count_threshold_ndh}, Mean count threshold {x_data_label[1]}: {mean_count_threshold_dh}")
    #check count significance of difference between sides
    _, wilcoxon_pvalue_count = wilcoxon(count_threshold_ndh, count_threshold_dh) # Wilcoxon Signed-Rank Test
    print("Wilcoxon Signed-Rank Test Count:")
    print("p-value:", wilcoxon_pvalue_count)
    _, ttest_pvalue_count = ttest_rel(count_threshold_ndh, count_threshold_dh) # Paired Samples T-Test
    print("Paired Samples T-Test Count:")
    print("p-value:", ttest_pvalue_count)

    print(f"Mean elevation threshold {x_data_label[0]}: {mean_elevation_threshold_ndh}, Mean elevation threshold {x_data_label[1]}: {mean_elevation_threshold_dh}")
    #check elevation significance of difference between sides
    _, wilcoxon_pvalue_elevation = wilcoxon(elevation_threshold_ndh, elevation_threshold_dh) # Wilcoxon Signed-Rank Test
    print("Wilcoxon Signed-Rank Test Elevation:")
    print("p-value:", wilcoxon_pvalue_elevation)
    _, ttest_pvalue_elevation = ttest_rel(elevation_threshold_ndh, elevation_threshold_dh) # Paired Samples T-Test for elevation threshold
    print("Paired Samples T-Test Elevation:")
    print("p-value:", ttest_pvalue_elevation)

    plt.figure(figsize=(10, 5))
    #plt.suptitle('Optimzed individual thresholds', fontsize=16)

    # Color configuration
    affected_color = thesis_style.get_thesis_colours()['affected']
    healthy_color = thesis_style.get_thesis_colours()['healthy']
    line_color_affected = thesis_style.get_thesis_colours()['black']
    line_color_healthy = thesis_style.get_thesis_colours()['black']

    # Subplot for count threshold
    plt.subplot(1, 2, 1)
    box1 = plt.boxplot(count_threshold_ndh, positions=[1], widths=0.6, patch_artist=True, 
                       boxprops=dict(facecolor=affected_color, color=line_color_affected, alpha=0.5),
                       capprops=dict(color=line_color_affected),
                       whiskerprops=dict(color=line_color_affected),
                       flierprops=dict(markerfacecolor=affected_color, marker='o', markersize=5, linestyle='none', alpha=0.0), # don't show outliers since all data points are already scattered
                       medianprops=dict(color=line_color_affected))
    
    box2 = plt.boxplot(count_threshold_dh, positions=[2], widths=0.6, patch_artist=True, 
                       boxprops=dict(facecolor=healthy_color, color=line_color_healthy, alpha=0.5),
                       capprops=dict(color=line_color_healthy),
                       whiskerprops=dict(color=line_color_healthy),
                       flierprops=dict(markerfacecolor=healthy_color, marker='o', markersize=5, linestyle='none', alpha=0.0), # don't show outliers since all data points are already scattered
                       medianprops=dict(color=line_color_healthy))

    plt.title('Count')
    plt.xlabel('')
    plt.ylabel('Counts per second')
    plt.xticks([1, 2], x_data_label)
    # Show all data points
    plt.scatter(np.random.normal(1.0, 0.06, size=len(count_threshold_ndh)), count_threshold_ndh, color=affected_color)
    plt.scatter(np.random.normal(2.0, 0.06, size=len(count_threshold_dh)), count_threshold_dh, color=healthy_color)
    # Plot horizontal line at conventional threshold
    plt.axhline(y=0.0, color=thesis_style.get_thesis_colours()['black_grey'], linestyle='--', label='Conventional GMAC threshold')
    # Add p-values between healthy and affected
    #plt.annotate(f"Significance levels between sides \npaired t-test p-value: {ttest_pvalue_count:.4f}", xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=9)
    plt.annotate(f"Wilcoxon signed-rank $\mathbf{{p-value: {wilcoxon_pvalue_count:.4f}}}$", xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=8)

    plt.legend()

    # Subplot for elevation threshold
    plt.subplot(1, 2, 2)
    box3 = plt.boxplot(elevation_threshold_ndh, positions=[1], widths=0.6, patch_artist=True, 
                       boxprops=dict(facecolor=affected_color, color=line_color_affected, alpha=0.5),
                       capprops=dict(color=line_color_affected),
                       whiskerprops=dict(color=line_color_affected),
                       flierprops=dict(markerfacecolor=line_color_affected, marker='o', markersize=5, linestyle='none', alpha=0.0),
                       medianprops=dict(color=line_color_affected))
    
    box4 = plt.boxplot(elevation_threshold_dh, positions=[2], widths=0.6, patch_artist=True, 
                       boxprops=dict(facecolor=healthy_color, color=line_color_healthy, alpha=0.5),
                       capprops=dict(color=line_color_healthy),
                       whiskerprops=dict(color=line_color_healthy),
                       flierprops=dict(markerfacecolor=line_color_healthy, marker='o', markersize=5, linestyle='none', alpha=0.0),
                       medianprops=dict(color=line_color_healthy))

    plt.title('Functional space')
    plt.xlabel('')
    plt.ylabel('Elevation')
    plt.xticks([1, 2], x_data_label)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(degree_formatter))
    # Show all data points
    plt.scatter(np.random.normal(1.0, 0.0125, size=len(elevation_threshold_ndh)), elevation_threshold_ndh, color=affected_color)
    plt.scatter(np.random.normal(2.0, 0.0125, size=len(elevation_threshold_dh)), elevation_threshold_dh, color=healthy_color)
    # Plot horizontal line at conventional threshold
    plt.axhline(y=30.0, color=thesis_style.get_thesis_colours()['black_grey'], linestyle='--', label='Conventional GMAC threshold')
    # Add p-values between healthy and affected
    #plt.annotate(f"Significance levels between sides \npaired t-test p-value: {ttest_pvalue_elevation:.4f}", xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=9)
    plt.annotate(f"Wilcoxon signed-rank $\mathbf{{p-value: {wilcoxon_pvalue_elevation:.4f}}}$", xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()