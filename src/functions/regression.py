import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr, zscore

from utilities import thesis_style

class RegressionModel:
    def __init__(self, x, y, dominant_impared=None):
        self.x = x.reshape(-1, 1)  # Reshape for scikit-learn compatibility
        self.y = y
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

            for i, (x, y, is_dom_impaired) in enumerate(zip(self.x, self.y, self.dominant_impared)):
                if is_dom_impaired:
                    plt.scatter(x, y, marker="+", linewidths=14, color=dominant_color, label='dominant arm affected' if not dominant_legend_plotted else "")
                    dominant_legend_plotted = True
                else:
                    plt.scatter(x, y, marker="x", linewidths=13, color=non_dominant_color, label='non-dominant arm affected' if not non_dominant_legend_plotted else "")
                    non_dominant_legend_plotted = True
        else:
            plt.scatter(self.x, self.y, marker="x", color=colors['dark_blue'], label='Optimized thresholds')
        
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
            ax.set_ylim([-0.15, max(self.y)])
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

def check_regression_paper(x, y, x_label='x', y_label='y', title='Regression Analysis', dominant_impared=None, remove_outliers=False):
    if remove_outliers:
        z_score_outliers, IQR_outliers = check_outliers(y)
        x = np.delete(x, IQR_outliers)
        y = np.delete(y, IQR_outliers)
        std = np.delete(std, IQR_outliers)
        if dominant_impared is not None:
            dominant_impared = np.delete(dominant_impared, IQR_outliers)

    model = RegressionModel(x, y, dominant_impared)

    # Fit linear regression
    model.fit_linear_regression()
    r_squared_linear = model.linear_model.score(model.x, model.y)
    print("Linear model coefficients:", model.linear_model.coef_)
    print("Linear model intercept:", model.linear_model.intercept_)
    print("Linear model R-squared:", r_squared_linear)
    print("Linear model adjusted R-squared:", 1 - (1 - r_squared_linear) * (len(model.y) - 1) / (len(model.y) - model.x.shape[1] - 1))

    # Fit polynomial regression of degree 2
    model.fit_polynomial_regression(2)
    r_squared_poly = model.poly_models[2].score(model.x, model.y)
    print("Polynomial model coefficients (degree 2):", model.poly_models[2].steps[1][1].coef_)
    print("Polynomial model intercept (degree 2):", model.poly_models[2].steps[1][1].intercept_)
    print("Polynomial model R-squared (degree 2):", r_squared_poly)
    print("Polynomial model adjusted R-squared (degree 2):", 1 - (1 - r_squared_poly) * (len(model.y) - 1) / (len(model.y) - model.x.shape[1] - 1))
        
    # Calculate Pearson correlation
    pearson_corr, pearson_pvalue = model.pearson_correlation()
    print("Pearson Correlation:", pearson_corr, "p-value:", pearson_pvalue)

    # Calculate Spearman correlation
    spearman_corr, spearman_pvalue = model.spearman_correlation()
    print("Spearman Correlation:", spearman_corr, "p-value:", spearman_pvalue)

    # Plot regressions
    #model.plot_regressions(xlabel=x_label, ylabel=y_label, title=title)