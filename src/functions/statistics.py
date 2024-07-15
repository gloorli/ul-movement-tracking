import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr

class RegressionModel:
    def __init__(self, x, y):
        self.x = x.reshape(-1, 1)  # Reshape for scikit-learn compatibility
        self.y = y
        self.linear_model = None
        self.poly_models = {}

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
    
    def check_distribution(self):
        plt.figure(figsize=(12, 5))

        # Distribution of x
        plt.subplot(1, 2, 1)
        plt.hist(self.x, bins=self.x.shape[0], edgecolor='k', alpha=0.7)
        plt.title('Distribution of x')
        plt.xlabel('x')
        plt.ylabel('Frequency')

        # Distribution of y
        plt.subplot(1, 2, 2)
        plt.hist(self.y, bins=66, edgecolor='k', alpha=0.7)
        plt.title('Distribution of y')
        plt.xlabel('y')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def pearson_correlation(self):
        corr, _ = pearsonr(self.x.flatten(), self.y)
        return corr

    def spearman_correlation(self):
        corr, _ = spearmanr(self.x.flatten(), self.y)
        return corr

    def plot_regressions(self, title='Regression Plots', xlabel='x', ylabel='y'):
        plt.scatter(self.x, self.y, color='blue', label='Data')

        # Plot linear regression
        if self.linear_model is not None:
            x_range = np.linspace(self.x.min(), self.x.max(), 500).reshape(-1, 1)
            y_linear_pred = self.linear_model.predict(x_range)
            plt.plot(x_range, y_linear_pred, color='red', label='Linear Regression')

        # Plot polynomial regressions
        for degree, poly_model in self.poly_models.items():
            y_poly_pred = poly_model.predict(x_range)
            plt.plot(x_range, y_poly_pred, label=f'Polynomial Regression (degree {degree})')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

def check_regression(x, y, x_label='x', y_label='y', title='Regression Analysis'):
    # Create a regression model
    model = RegressionModel(x, y)

    # Check distribution of the data
    model.check_distribution()

    # Fit linear regression
    model.fit_linear_regression()
    print("Linear model coefficients:", model.linear_model.coef_)
    print("Linear model intercept:", model.linear_model.intercept_)

    # Predict using linear regression
    predicted_linear = model.predict_linear([11, 12, 13])
    print("Linear Predictions:", predicted_linear)

    # Fit polynomial regression of degree 2
    model.fit_polynomial_regression(2)

    # Predict using polynomial regression
    predicted_poly = model.predict_polynomial([11, 12, 13], 2)
    print("Polynomial Predictions:", predicted_poly)
        
    # Calculate Pearson correlation
    pearson_corr = model.pearson_correlation()
    print("Pearson Correlation:", pearson_corr)

    # Calculate Spearman correlation
    spearman_corr = model.spearman_correlation()
    print("Spearman Correlation:", spearman_corr)

    # Plot regressions
    model.plot_regressions(xlabel=x_label, ylabel=y_label, title=title)
    

# Example usage:
if __name__ == "__main__":
    # Example data
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

    check_regression(x, y)