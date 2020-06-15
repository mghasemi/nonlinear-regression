from random import randint

import matplotlib.pyplot as plt
import numpy as np

from GeneralRegression.NpyProximation import HilbertRegressor, Measure
from GeneralRegression.extras import FunctionBasis

plt.figure(randint(1, 1000), figsize=(16, 12))

# Make up a 1-dim regression data
n_samples = 100
f = lambda x: x * np.exp(np.sin(x ** 2)) + np.cos(x) * x ** 2
X = np.linspace(0., 10, n_samples).reshape((-1, 1))
y = f(X)


def pfe_1d(p_d=3, f_d=1, l=1.):
    basis = FunctionBasis()
    p_basis = basis.poly(1, p_d)
    f_basis = basis.fourier(1, f_d, l)[1:]
    e_basis = []
    return p_basis + f_basis + e_basis


domain = np.linspace(min(X), max(X), 150)

x_min, x_max = X.min(), X.max()
x_mid = (x_min + x_max) / 2.
w_min = .1
w_max = 5.
ws1 = {_[0]: np.exp(-1. / max(abs((_[0] - x_min) * (_[0] - x_max)) / 10, 1.e-5))
       for _ in X}
Xs1 = [_[0] for _ in X]
Ws1 = [ws1[_] for _ in Xs1]

ws2 = {_[0]: .1 if _[0] < x_mid else 1.
       for _ in X}
Xs2 = [_[0] for _ in X]
Ws2 = [ws2[_] for _ in Xs2]

meas1 = Measure(ws1)
ell = .7
B1 = pfe_1d(p_d=3, f_d=20, l=ell)

regressor1 = HilbertRegressor(base=B1, meas=meas1)
regressor1.fit(X, y)
y_pred1 = regressor1.predict(domain)

meas2 = Measure(ws2)

regressor2 = HilbertRegressor(base=B1, meas=meas2)
regressor2.fit(X, y)
y_pred2 = regressor2.predict(domain)

fig = plt.figure(randint(1, 10000), constrained_layout=True, figsize=(16, 10))
gs = fig.add_gridspec(6, 1)
f_ax1 = fig.add_subplot(gs[:4, :])
f_ax1.scatter(X, y, color='red', s=10, marker='o', alpha=0.5, label="Data points")
f_ax1.plot(domain, y_pred1, color='blue', label='Fit 1')
f_ax1.plot(domain, y_pred2, color='teal', label='Fit 2')
f_ax1.fill_between(domain.reshape((1, -1))[0],
                   y_pred1 - regressor1.ci_band,
                   y_pred1 + regressor1.ci_band,
                   color='purple',
                   alpha=0.1, label='CI: 95%')
f_ax1.fill_between(domain.reshape((1, -1))[0],
                   y_pred2 - regressor2.ci_band,
                   y_pred2 + regressor2.ci_band,
                   color='orange',
                   alpha=0.1, label='CI: 95%')
f_ax1.legend(loc=1)
f_ax1.grid(True, linestyle='-.', alpha=.4)

f_ax2 = fig.add_subplot(gs[4, :])
f_ax2.set_title('Weight 1')
f_ax2.fill_between(Xs1, [0. for _ in Ws1], Ws1, label='Distibution', color='purple', alpha=.3)
f_ax2.set_ylabel('Weight')

f_ax3 = fig.add_subplot(gs[5:, :])
f_ax3.set_title('Weight 2')
f_ax3.fill_between(Xs2, [0. for _ in Ws2], Ws2, label='Distibution', color='orange', alpha=.3)
f_ax3.set_ylabel('Weight')
plt.show()
