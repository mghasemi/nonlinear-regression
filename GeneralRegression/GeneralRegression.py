from sklearn.base import BaseEstimator, RegressorMixin


class GenericRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, funcs, regressor=None, **kwargs):
        from sklearn.linear_model import LinearRegression
        self.funcs = funcs
        self.kw_ar = kwargs
        self.std = False
        self.standard_deviation = []
        if regressor is None:
            self.regressor = LinearRegression
        else:
            self.regressor = regressor
        self.model = None

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
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.std:
            pred, std = self.model.predict(X, return_std=True)
        else:
            pred = self.model.predict(X)
            std = None
        print(std)
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


# generate points used to plot
x_plot = np.linspace(0, 10, 100)
# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random  # .RandomState(0)
size = 10
rng.shuffle(x)
x = np.sort(x[:size])
y = f(x) + np.random.normal(scale=3., size=size)
# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]
colors = ['yellowgreen', 'gold', 'salmon', 'purple', 'teal', 'orange', 'green', 'red']
lw = 2
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw, label="ground truth")
plt.scatter(x, y, color='navy', s=10, marker='o', label="training points")

for count, degree in enumerate([7, 10]):
    model = GenericRegressor(fourier, regressor=BayesianRidge, **dict(n=degree, l=3))
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw, label="degree %d" % degree)
    plt.fill_between(x_plot,
                     y_plot - 1.96 * model.standard_deviation,# / np.sqrt(size),
                     y_plot + 1.96 * model.standard_deviation,# / np.sqrt(size),
                     color=colors[count],
                     alpha=0.1)
plt.legend(loc=2)

print(np.exp(x))
print(model.predict([[15]]), model.standard_deviation)
plt.show()
