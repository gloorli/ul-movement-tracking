import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr

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

        fig, ax = plt.subplots(1,1)
        
        if self.dominant_impared is not None:
            dominant_legend_plotted = False
            non_dominant_legend_plotted = False
            for i, (x, y, err, is_dom_impaired) in enumerate(zip(self.x, self.y, self.threshold_std, self.dominant_impared)):
                if is_dom_impaired:
                    plt.errorbar(x, y, yerr=err, fmt="+", color=colors['dark_blue'], label='Individual optimized thresholds \n(std over k-folds, dominant arm)' if not dominant_legend_plotted else "")
                    dominant_legend_plotted = True
                else:
                    plt.errorbar(x, y, yerr=err, fmt="x", color=colors['light_blue'], label='Individual optimized thresholds \n(std over k-folds, non-dominant arm)' if not non_dominant_legend_plotted else "")
                    non_dominant_legend_plotted = True
        else:
            plt.errorbar(self.x, self.y, yerr=self.threshold_std, fmt="x", color=colors['dark_blue'], label='Optimized thresholds (std over k-folds)')

        # Plot linear regression
        if self.linear_model is not None:
            x_range = np.linspace(self.x.min(), self.x.max(), 500).reshape(-1, 1)
            y_linear_pred = self.linear_model.predict(x_range)
            r_squared = self.linear_model.score(self.x, self.y)
            plt.plot(x_range, y_linear_pred, color='red', label=f'Linear Regression (R-squared: {r_squared:.2f})')

        # Plot polynomial regressions
        for degree, poly_model in self.poly_models.items():
            y_poly_pred = poly_model.predict(x_range)
            r_squared = poly_model.score(self.x, self.y)
            plt.plot(x_range, y_poly_pred, label=f'Polynomial Regression (degree {degree}) (R-squared: {r_squared:.2f})', color=colors['light_blue'])

        if ylabel == 'Count Threshold':
            plt.axhline(y=0.0, color=colors['black_grey'], linestyle='--', label='Conventional threshold')
        elif ylabel == 'Pitch Threshold':
            plt.axhline(y=30.0, color=colors['black_grey'], linestyle='--', label='Conventional threshold')
            ax.set_yticks([30, 35, 40, 45, 50, 55, 60, 65])
            ax.set_yticklabels(['30°', '35°', '40°', '45°', '50°', '55°', '60°', '65°'])
        else:
            raise ValueError(f"Invalid ylabel: {ylabel}")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        plt.tight_layout()
        plt.show()


def extract_std_from_min_max_std(min_max_std):
    COUNT_min_max_std = min_max_std[:,0]
    count_std = [item['std'] for item in COUNT_min_max_std]
    PITCH_min_max_std = min_max_std[:,1]
    pitch_std = [item['std'] for item in PITCH_min_max_std]

    return count_std, pitch_std

def check_regression(x, y, std, x_label='x', y_label='y', title='Regression Analysis', dominant_impared=None):
    # Create a regression model
    model = RegressionModel(x, y, std, dominant_impared)

    # Check distribution of the data
    #model.check_distribution()

    # Fit linear regression
    model.fit_linear_regression()
    print("Linear model coefficients:", model.linear_model.coef_)
    print("Linear model intercept:", model.linear_model.intercept_)

    # Fit polynomial regression of degree 2
    model.fit_polynomial_regression(2)
        
    # Calculate Pearson correlation
    pearson_corr, pearson_pvalue = model.pearson_correlation()
    print("Pearson Correlation:", pearson_corr, "p-value:", pearson_pvalue)

    # Calculate Spearman correlation
    spearman_corr, spearman_pvalue = model.spearman_correlation()
    print("Spearman Correlation:", spearman_corr, "p-value:", spearman_pvalue)

    # Plot regressions
    model.plot_regressions(xlabel=x_label, ylabel=y_label, title=title)
    
def check_distribution(count_threshold_ndh, count_threshold_dh, elevation_threshold_ndh, elevation_threshold_dh):
        mean_count_threshold_ndh = np.mean(count_threshold_ndh)
        mean_count_threshold_dh = np.mean(count_threshold_dh)
        mean_elevation_threshold_ndh = np.mean(elevation_threshold_ndh)
        mean_elevation_threshold_dh = np.mean(elevation_threshold_dh)
        print(f"Mean count threshold (affected): {mean_count_threshold_ndh}, Mean count threshold (healthy): {mean_count_threshold_dh}")
        print(f"Mean elevation threshold (affected): {mean_elevation_threshold_ndh}, Mean elevation threshold (healthy): {mean_elevation_threshold_dh}")

        plt.figure(figsize=(12, 5))

        # Subplot for count threshold
        plt.subplot(1, 2, 1)
        plt.boxplot([count_threshold_ndh, count_threshold_dh], showmeans=True)
        plt.title('Count Threshold Distribution')
        plt.xlabel('')
        plt.ylabel('Counts')
        plt.xticks([1, 2], ['Affected', 'Healthy'])
        # Show all data points
        plt.plot(np.ones_like(count_threshold_ndh), count_threshold_ndh, 'ro', label='Affected')
        plt.plot(2 * np.ones_like(count_threshold_dh), count_threshold_dh, 'go', label='Healthy')
        # Plot horizontal line at conventional threshold
        plt.axhline(y=0.0, color=thesis_style.get_thesis_colours()['black_grey'], linestyle='--', label='Conventional threshold')
        plt.legend()

        # Subplot for elevation threshold
        plt.subplot(1, 2, 2)
        plt.boxplot([elevation_threshold_ndh, elevation_threshold_dh], showmeans=True)
        plt.title('Elevation Threshold Distribution')
        plt.xlabel('')
        plt.ylabel('Elevation (degrees)')
        plt.xticks([1, 2], ['Affected', 'Healthy'])
        # Show all data points
        plt.plot(np.ones_like(elevation_threshold_ndh), elevation_threshold_ndh, 'ro')
        plt.plot(2 * np.ones_like(elevation_threshold_dh), elevation_threshold_dh, 'go')
        # Plot horizontal line at conventional threshold
        plt.axhline(y=30.0, color=thesis_style.get_thesis_colours()['black_grey'], linestyle='--', label='Conventional threshold')

        plt.tight_layout()
        plt.show()