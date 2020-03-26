import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


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


def Poly(n, deg):
    """
    Returns a basis consisting of polynomials in `n` variables of degree at most `deg`.

    :param n: number of variables
    :param deg: highest degree of polynomials in the basis
    :return: the raw basis consists of polynomials of degrees up to `n`
    """
    from itertools import product
    from numpy import prod

    B = []
    for o in product(range(deg + 1), repeat=n):
        if sum(o) <= deg:
            B.append(lambda x, e=o: prod([x[i] ** e[i] for i in range(n)]))
    return B


def Fourier(n, deg, l=1.0):
    """
    Returns the Fourier basis of degree `deg` in `n` variables with period `l`

    :param n: number of variables
    :param deg: the maximum degree of trigonometric combinations in the basis
    :param l: the period
    :return: the raw basis consists of trigonometric functions of degrees up to `n`
    """

    from numpy import sin, cos, prod
    from itertools import product

    B = [lambda x: 1.0]
    E = list(product([0, 1], repeat=n))
    raw_coefs = list(product(range(deg + 1), repeat=n))
    coefs = set()
    for prt in raw_coefs:
        p_ = list(prt)
        p_.sort()
        coefs.add(tuple(p_))
    for o in coefs:
        if (sum(o) <= deg) and (sum(o) > 0):
            for ex in E:
                if sum(ex) >= 0:
                    f_ = lambda x, o_=o, ex_=ex: prod(
                        [
                            sin(o_[i] * x[i] / l) ** ex_[i]
                            * cos(o_[i] * x[i] / l) ** (1 - ex_[i])
                            if o_[i] > 0
                            else 1.0
                            for i in range(n)
                        ]
                    )
                    B.append(f_)
    return B


# generate points used to plot
x_plot = np.linspace(0, 10, 100)
# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:10])
y = f(x)
# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]
colors = ['teal', 'yellowgreen', 'gold', 'salmon', 'purple', 'orange', 'green', 'red']
lw = 2
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw, label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([7, 8, 9, 10]):
    model = make_pipeline(FunctionTransformer(fourier, kw_args=dict(n=degree, l=3), validate=False), LinearRegression())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw, label="degree %d" % degree)
plt.legend(loc=2)

print(fourier([[-1]], n=3))
B = Fourier(1, 3)
print([_([-1.]) for _ in B])

plt.show()
