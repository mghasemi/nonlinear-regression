from sklearn.base import BaseEstimator, RegressorMixin


class GenericRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, funcs, regressor=None, ci=.95, **kwargs):
        from sklearn.linear_model import LinearRegression
        self.funcs = funcs
        self.kw_ar = kwargs
        # to store the standard deviations of the predicted values
        self.std = False
        # convert the confidence interval value to find the ppf
        self.ci = (1 + ci) / 2.
        self.standard_deviation = []
        if regressor is None:
            self.regressor = LinearRegression
        else:
            self.regressor = regressor
        self.model = None
        # to store the size of training
        self.size = 0.
        # to store the confidence interval radius
        self.ci_band = []

    def fit(self, X, y):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.linear_model import BayesianRidge
        if self.regressor is BayesianRidge:
            self.std = True
            regressor = self.regressor(compute_score=True)
        else:
            regressor = self.regressor()
        self.model = make_pipeline(  # MinMaxScaler((-1, 1)),
            FunctionTransformer(self.funcs, kw_args=self.kw_ar, validate=False),
            regressor)
        self.size = y.shape[0]
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.std:
            from numpy import sqrt
            from scipy.stats import norm
            pred, std = self.model.predict(X, return_std=True)
            z_bar = norm.ppf(self.ci, 0, 1)
            self.ci_band = z_bar * std / sqrt(max(self.size - 1 , 1))
        else:
            pred = self.model.predict(X)
            std = None
        self.standard_deviation = std
        return pred


###########################################################
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import BayesianRidge


def f(x):
    return x * np.sin(x) - np.sqrt(x) * np.cos(2 * x) + x ** 2


# Fourier base generator
def fourier(X, n=1, l=1.):
    points = []
    for x in X:
        point = []
        for deg in range(n + 1):
            if deg == 0:
                point.append(1.)
            else:
                point.append(np.sin(deg * x[0] / l))
                point.append(np.cos(deg * x[0] / l))
        point.append(x[0])
        point.append(x[0] ** 2)
        points.append(np.array(point))
    return np.array(points)
"""
N = 10
x_range = np.linspace(-5, 20, 100)
x_training = 10 * np.random.sample(N)
y_training = f(x_training) + np.random.normal(scale=3., size=N)
y_actual = f(x_range)
colors = ['yellowgreen', 'gold', 'salmon', 'purple', 'teal', 'orange', 'green', 'red']

plt.plot(x_range, y_actual, color='cornflowerblue', linewidth=1, label="ground truth")
plt.scatter(x_training, y_training, color='navy', s=5, marker='o', label="training points")

for count, degree in enumerate([7, 10]):
    model = GenericRegressor(fourier, regressor=BayesianRidge, **dict(n=degree, l=3))
    model.fit(x_training.reshape((-1, 1)), y_training)
    y_plot = model.predict(x_range.reshape((-1, 1)))
    plt.plot(x_range, y_plot, color=colors[count], linewidth=1, label="degree %d" % degree)
    plt.fill_between(x_range,
                     y_plot - model.ci_band,
                     y_plot + model.ci_band,
                     color=colors[count],
                     alpha=0.1)
plt.legend(loc=2)

plt.show()
"""