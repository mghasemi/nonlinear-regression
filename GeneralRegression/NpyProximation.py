"""
Hilbert Space based regression
==================================
"""
from numpy import sqrt, array, ndarray, delete, concatenate, power
from numpy.linalg import det
from scipy import integrate
from scipy.stats import t

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ModuleNotFoundError:
    BaseEstimator = type("BaseEstimator", (object,), dict())
    RegressorMixin = type("RegressorMixin", (object,), dict())

Infinitesimal = 1e-7


class Error(Exception):
    r"""
    Generic errors that may occur in the course of a run.
    """

    def __init__(self, *args):
        super(Error, self).__init__(*args)


class Measure(object):
    r"""
    Constructs a measure :math:`\mu` based on `density` and `domain`.

    :param density: the density over the domain:
            + if none is given, it assumes uniform distribution

            + if a callable `h` is given, then :math:`d\mu=h(x)dx`

            + if a dictionary is given, then :math:`\mu=\sum w_x\delta_x` a discrete measure.
              The points :math:`x` are the keys of the dictionary (tuples) and the weights :math:`w_x` are the values.
    :param domain: if `density` is a dictionary, it will be set by its keys. If callable, then `domain` must be a list
                   of tuples defining the domain's box. If None is given, it will be set to :math:`[-1, 1]^n`
    """

    def __init__(self, density=None, domain=None):

        # set the density
        if density is None:
            self.density = lambda x: 1.0
        elif callable(density):
            self.density = density
        elif isinstance(density, dict):
            self.density = lambda x, density_=density: density_[x] if x in density else 0.
        else:
            raise Error(
                "The `density` must be either a callable or a dictionary of real numbers."
            )
        # check and set the domain
        self.continuous = True
        self.dim = 0
        if isinstance(domain, list):
            self.dim = len(domain)
            for intrvl in domain:
                if (not isinstance(intrvl, (list, tuple))) or (len(intrvl) != 2):
                    raise Error("`domain` should be a list of 2-tuples.")
            self.supp = domain
        elif isinstance(density, dict):
            self.supp = density.keys()
            self.continuous = False
        else:
            raise Error("No domain is specified.")

    def integral(self, f):
        r"""
        Calculates :math:`\int_{domain} fd\mu`.

        :param f: the integrand
        :return: the value of the integral
        """
        from types import FunctionType

        m = 0.0
        if not isinstance(
                f, (dict, FunctionType)
        ):
            raise Error("The integrand must be a `function` or a `dict`")
        if isinstance(f, dict):
            fn = lambda x: f[x] if x in f else 0.0
        else:
            fn = f
        if self.continuous:
            fw = lambda *x: fn(*x) * self.density(*x)
            m = integrate.nquad(fw, self.supp)[0]
        else:
            for p in self.supp:
                m += self.density(p) * fn(p)
        return m

    def norm(self, p, f):
        r"""
        Computes the norm-`p` of the `f` with respect to the current measure,
        i.e., :math:`(\int_{domain}|f|^p d\mu)^{1/p}`.

        :param p: a positive real number
        :param f: the function whose norm is desired.
        :return: :math:`\|f\|_{p, \mu}`
        """
        absfp = lambda *x: pow(abs(f(*x)), p)
        return pow(self.integral(absfp), 1.0 / p)


class FunctionSpace(object):
    r"""
    A class tha facilitates a few types of computations over function spaces of type :math:`L_2(X, \mu)`

    :param dim: the dimension of 'X' (default: 1)
    :param measure: an object of type `Measure` representing :math:`\mu`
    :param basis: a finite basis of functions to construct a subset of :math:`L_2(X, \mu)`
    """
    dim = 1  # type: int

    def __init__(self, dim=1, measure=None, basis=None):
        self.dim = int(dim)
        if (measure is not None) and (isinstance(measure, Measure)):
            self.measure = measure
        else:
            # default measure is set to be the Lebesgue measure on [0, 1]^dim
            D = [(0.0, 1.0) for _ in range(self.dim)]
            self.measure = Measure(domain=D)
        if basis is None:
            # default basis is linear
            base = [lambda x: 1.0]
            for i in range(self.dim):
                base.append(lambda x, i_=i: x[i_] if isinstance(x, array) else x)
            self.base = base
        else:
            self.base = basis
        self.orth_base = []
        self.Gram = None

    def inner(self, f, g):
        r"""
        Computes the inner product of the two parameters with respect to
        the measure `measure`, i.e., :math:`\int_Xf\cdot g d\mu`.

        :param f: callable
        :param g: callable
        :return: the quantity of :math:`\int_Xf\cdot g d\mu`
        """
        fn = lambda x, f_=f, g_=g: f_(x) * g_(x)
        return self.measure.integral(fn)

    def project(self, f, g):
        r"""
        Finds the projection of `f` on `g` with respect to the inner
        product induced by the measure `measure`.

        :param f: callable
        :param g: callable
        :return: the quantity of :math:`\frac{\langle f, g\rangle}{\|g\|_2}g`
        """
        a = self.inner(f, g)
        b = self.inner(g, g)
        return lambda x, a_=a, b_=b: a_ * g(x) / b_

    def gram_mat(self):
        num = len(self.base)
        cfs = array([[0.0] * num] * num)
        for i in range(num):
            for j in range(i, num):
                cf = self.inner(self.base[i], self.base[j])
                cfs[i][j] = cf
                cfs[j][i] = cf
        self.Gram = cfs

    def minor_gram(self, i):
        if self.Gram is None:
            self.gram_mat()
        return array(
            [[self.Gram[idx][jdx] for idx in range(i + 1)] for jdx in range(i + 1)]
        )

    def minor(self, i, j):
        if j == 1:
            return 1.0
        cfs = array([[0.0] * j] * (j - 1))
        for jdx in range(j):
            for idx in range(j - 1):
                cfs[idx][jdx] = self.Gram[idx][jdx]
        return det(delete(cfs, i, 1))

    def form_basis(self):
        """
        Call this method to generate the orthogonal basis corresponding
        to the given basis.
        The result will be stored in a property called ``orth_base`` which
        is a list of function that are orthogonal to each other with
        respect to the measure ``measure`` over the given range ``domain``.
        """
        num = len(self.base)
        gram_dets = [1.0] + [det(self.minor_gram(i)) for i in range(num)]
        base = []
        cfs = []
        for j in range(1, num + 1):
            j_ = j
            cf = [
                (-1) ** (i + j - 1)
                * self.minor(i, j_)
                / sqrt(gram_dets[j_ - 1] * gram_dets[j_])
                for i in range(j_)
            ]
            base.append(lambda x: sum([cf[i] * self.base[i](x) for i in range(j_)]))
            cfs.append(cf)
        self.orth_base = []
        for i in range(len(cfs)):
            fn = lambda x, i_=i: sum(
                [cfs[i_][_j] * self.base[_j](x) for _j in range(len(cfs[i_]))]
            )
            self.orth_base.append(fn)

    def series(self, f):
        r"""
        Given a function `f`, this method finds and returns the
        coefficients of the	series that approximates `f` as a
        linear combination of the elements of the orthogonal basis :math:`B`.
        In symbols :math:`\sum_{b\in B}\langle f, b\rangle b`.

        :return: the list of coefficients :math:`\langle f, b\rangle` for :math:`b\in B`
        """
        cfs = []
        for b in self.orth_base:
            cfs.append(self.inner(f, b))
        return cfs


class Regression(object):
    """
    Given a set of points, i.e., a list of tuples of the equal lengths `P`, this class computes the best approximation
    of a function that fits the data, in the following sense:

        + if no extra parameters is provided, meaning that an object is initiated like ``R = Regression(P)`` then
          calling ``R.fit()`` returns the linear regression that fits the data.
        + if at initiation the parameter `deg=n` is set, then ``R.fit()`` returns the polynomial regression of
          degree `n`.
        + if a basis of functions provided by means of an `OrthSystem` object (``R.SetOrthSys(orth)``) then
          calling ``R.fit()`` returns the best approximation that can be found using the basic functions of
          the `orth` object.

    :param points: a list of points to be fitted or a callable to be approximated
    :param dim: dimension of the domain
    """

    def __init__(self, points, dim=None):
        self.Points = None
        if isinstance(points, (ndarray, list, array)):
            self.Points = list(points)
            self.dim = len(points[0]) - 1
            supp = {}
            for p in points:
                supp[tuple(p[:-1])] = 1.0
            self.meas = Measure(supp)
            self.f = lambda x: sum(
                [
                    p_[-1] * (1 * (abs(x - array(p_[:-1])) < 1.0e-4)).min()
                    for p_ in points
                ]
            )
        elif callable(points):
            if dim is None:
                raise Error("The dimension can not be determined")
            else:
                self.dim = dim
            self.f = points
            self.meas = Measure(domain=[(-1.0, 1.0) for _ in range(self.dim)])
        self.orth = FunctionSpace(dim=self.dim, measure=self.meas)

    def set_measure(self, meas):
        """
        Sets the default measure for approximation.

        :param meas: a measure.Measure object
        :return: None
        """
        if not isinstance(meas, Measure):
            raise AssertionError("`set_measure` accepts a `NpyProximation.Measure` object.")
        self.meas = meas

    def set_func_spc(self, sys):
        """
        Sets the bases of the orthogonal basis

        :param sys: `orthsys.OrthSystem` object.
        :return: None

        .. Note::
            For technical reasons, the measure needs to be given via `set_measure` method. Otherwise, the Lebesque
            measure on :math:`[-1, 1]^n` is assumed.
        """
        if self.dim != sys.dim:
            raise AssertionError(
                "Dimensions of points and the orthogonal system do not match."
            )
        sys.measure = self.meas
        self.orth = sys
        self.orth.form_basis()

    def fit(self):
        """
        Fits the best curve based on the optional provided orthogonal basis.
        If no basis is provided, it fits a polynomial of a given degree (at initiation)
        :return: The fit.
        """
        coefs = self.orth.series(self.f)
        aprx = lambda x: sum(
            [
                coefs[i] * self.orth.orth_base[i](x)
                for i in range(len(self.orth.orth_base))
            ]
        )
        return aprx


class HilbertRegressor(BaseEstimator, RegressorMixin):
    r"""
    Regression using Hilbert Space techniques Scikit-Learn style.

    :param deg: int, default=3
        The degree of polynomial regression. Only used if `base` is `None`
    :param base: list, default = None
        a list of function to form an orthogonal function basis
    :param meas: `NpyProximation.Measure`, default = None
        the measure to form the :math:`L_2(\mu)` space. If `None` a discrete measure will be constructed based
        on `fit` inputs
    :param f_space: `NpyProximation.FunctionBasis`, default = None
        the function subspace of :math:`L_2(\mu)`, if `None` it will be initiated according to `self.meas`
    :param c_limit: for confidence interval
    :param apprx: It is a callable, this will be constructed on `fit` method. It use for approximate after
        fitting/learning.
    """

    def __init__(self, deg=3, base=None, meas=None, f_space=None, c_limit=.95):
        self.deg = deg
        self.meas = meas
        self.base = base
        self.f_space = f_space
        self.regressor = None
        self.dim = 0
        self.apprx = None
        self.c_limit = (c_limit + 1.0) / 2.0
        self.t_stat = 0.
        self.training_var = 0.
        self.training_size = 0
        self.ci_band = None
        self.x_mean = 0.
        self.sum_sqrd_x = 0

    def fit(self, X, y):
        """
        Calculates an orthonormal basis according to the given function space basis and the discrete measure from the
        training points.

        :param X: Training data
        :param y: Target values
        :return: `self`
        """
        if len(X.shape) != 2:
            X = X.reshape(X.shape[0], 1)
        self.training_size = X.shape[0]
        points = concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
        self.regressor = Regression(points)
        self.dim = X[0].shape[0]
        if self.meas is not None:
            self.regressor.set_measure(self.meas)
        if self.f_space is not None:
            self.regressor.set_func_spc(self.f_space)
        else:
            from .extras import FunctionBasis
            bs = FunctionBasis()
            B = bs.poly(n=self.dim, deg=self.deg) if self.base is None else self.base
            self.f_space = FunctionSpace(dim=self.dim, basis=B, measure=self.meas)
            self.regressor.set_func_spc(self.f_space)
        self.apprx = self.regressor.fit()
        res = array([self.apprx(x) for x in X])
        self.x_mean = X.mean()
        self.sum_sqrd_x = sum(power(X, 2))
        self.training_var = sqrt(sum(power(res - y.reshape((1, -1))[0], 2)) / max(self.training_size - 2, 1))
        self.t_stat = t.ppf(self.c_limit, self.training_size - 1)
        return self

    def predict(self, X):
        """
        Predict using the Hilbert regression method

        :param X: data points for prediction
        :return: returns predicted values
        """
        if len(X.shape) != 2:
            X = X.reshape(X.shape[0], 1)
        # Adjust the `self.x_mean` according to the weight
        if X.shape[1] == 1:
            self.ci_band = (
                    self.t_stat
                    * self.training_var
                    * (
                            1. / self.training_size
                            + power(X - self.x_mean, 2)
                            / sqrt(self.sum_sqrd_x - self.training_size * power(self.x_mean, 2))
                    )
            ).reshape(1, -1)[0]
        return array([self.apprx(x) for x in X])

    def score(self, X, y, sample_weight=None):
        """
        The default scoring method is the weighted mean square error

        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        num_points = len(X)
        weights = [self.regressor.meas.density(tuple(x)) for x in X]
        values = [self.apprx(x) for x in X]
        sum_w = sum(weights)
        errors = [weights[_] * (values[_] - y[_]) ** 2 for _ in range(num_points)]
        try:
            return sum(errors)[0] / sum_w
        except IndexError:
            return sum(errors) / sum_w
