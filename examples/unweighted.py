from random import randint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import BayesianRidge

from GeneralRegression.GeneralRegression import GenericRegressor

plt.figure(randint(1, 1000), figsize=(16, 12))

# Make up a 1-dim regression data
n_samples = 100
f = lambda x: x * np.exp(np.sin(x ** 2)) + np.cos(x) * x ** 2
X = np.linspace(0., 10, n_samples).reshape((-1, 1))
y = f(X).reshape((1, -1))[0]


# Function basis generator
def mixed(X, p_d=3, f_d=1, l=1., e_d=2):
    """
    A mixture of polynomial, Fourier and exponential functions

    :param X: the domain to be transformed
    :param p_d: the maximum degree of polynomials to be included
    :param f_d: the maximum degree of discrete Fourier transform
    :param e_d: the maximum degree of the `x` coefficient to be included as :math:`x^d\times e^{\pm x}`

    :return: the transformed data points
    """
    points = []
    for x in X:
        point = [1.]
        for deg in range(1, f_d + 1):
            point.append(np.sin(deg * x[0] / l))
            point.append(np.cos(deg * x[0] / l))
        for deg in range(1, p_d + 1):
            point.append(x[0] ** deg)
        for deg in range(e_d + 1):
            point.append((x[0] ** deg) * np.exp(-x[0] / l))
            point.append((x[0] ** deg) * np.exp(x[0] / (2.5 * l)))
        points.append(np.array(point))
    return np.array(points)


domain = np.linspace(min(X), max(X), 150)

regressor = GenericRegressor(mixed, regressor=BayesianRidge, **dict(p_d=3, f_d=50, l=1., e_d=0))
regressor.fit(X, y)
y_pred = regressor.predict(domain)

plt.scatter(X, y, color='red', s=10, marker='o', alpha=0.5, label="Data points")
plt.plot(domain, y_pred, color='blue', label='Fit')
plt.fill_between(domain.reshape((1, -1))[0],
                 y_pred - regressor.ci_band,
                 y_pred + regressor.ci_band,
                 color='purple',
                 alpha=0.1, label='CI: 95%')
plt.legend(loc=2)
plt.grid(True, alpha=.4)
plt.show()