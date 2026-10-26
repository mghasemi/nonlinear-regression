"""
Hilbert Space based regression
==================================
"""
from numpy import sqrt, array, ndarray, delete, concatenate, power
import numpy as np
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


# Condition-number threshold below which the Gram-determinant orthogonalization
# is used; above it the QR fallback on basis evaluations takes over (see
# ``FunctionSpace.form_basis``). Empirical crossover for monomial bases on
# [0, 1]: cond ~ 2e12 already degrades fits by O(1e-3), so we stay well below.
GRAM_COND_LIMIT = 1.0e10


def _spike_value(x, abscissa, value):
    r"""
    Exact-match spike kernel used by :class:`Regression` to build the data
    function :math:`f`.

    Returns ``value`` iff ``x`` coincides with ``abscissa`` within a relative
    tolerance (float round-off only); otherwise returns 0. The old absolute
    window of :math:`10^{-4}` silently merged distinct abscissas that are
    closer than that in *normalized* units — e.g., hourly timestamps mapped by
    ``Time2Interval`` onto a multi-year span (where one unit spans thousands of
    days), so adjacent observations collapsed into the same spike.

    :param x: evaluation point (tuple or array)
    :param abscissa: abscissa array to match against
    :param value: contribution at that abscissa
    """
    from numpy import abs as _abs, maximum, ravel

    d = _abs(ravel(array(x)) - abscissa)
    tol = 1.0e-9 * maximum(1.0, _abs(abscissa))
    return value if bool((d <= tol).all()) else 0.0


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
        self.idx = 0
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
                dp = self.density(p)
                if type(p) is not tuple:
                    p = (p,)
                m += dp * fn(p)
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
                # NumPy>=2 no longer implicitly casts (even single-element) arrays to scalars
                cf = self.inner(self.base[i], self.base[j])
                cf = cf.item() if isinstance(cf, ndarray) else cf
                cfs[i][j] = cf
                cfs[j][i] = cf
        self.Gram = cfs

    def minor_gram(self, i):
        if self.Gram is None:
            self.gram_mat()
        # Gram is symmetric, so the leading (i+1)x(i+1) slice equals the
        # element-wise construction below, but avoids an O(i^2) Python loop.
        return self.Gram[: i + 1, : i + 1]

    def minor(self, i, j):
        if j == 1:
            return 1.0
        # equivalent to building cfs[idx][jdx] = self.Gram[idx][jdx] via a
        # double loop, but avoids the O(j^2) Python-level overhead.
        cfs = self.Gram[: j - 1, :j]
        return det(delete(cfs, i, 1))

    def _eval_matrix(self, nodes=None, weights=None):
        r"""
        Evaluates the raw basis at support/quadrature nodes: returns ``(E, w)``
        with :math:`E_{ki} = \varphi_i(s_k)` and the positive node weights.

        For discrete measures the support points *are* the nodes (exact); for
        continuous one-dimensional measures a Gauss–Legendre grid on the domain
        box is used, so orthogonality holds w.r.t. the quadrature rather than the
        full integral. Not available for continuous measures in dim >= 2.

        The ``nodes``/``weights`` arguments are reserved (currently unused).
        """
        if not self.measure.continuous:
            keys = list(self.measure.supp)
            E = np.array([
                [float(np.ravel(phi(array(k))) [0]) for phi in self.base]
                for k in keys
            ])
            w = np.array([self.measure.density(k) for k in keys], dtype=float)
            return E, w
        if len(self.measure.supp) == 1:
            lo, hi = float(self.measure.supp[0][0]), float(self.measure.supp[0][1])
            from numpy.polynomial.legendre import leggauss
            n_nodes = max(2 * len(self.base) + 8, 32)
            z, wz = leggauss(n_nodes)
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            E = np.array([
                [float(np.ravel(phi(array([mid + half * zz]))) [0]) for phi in self.base]
                for zz in z
            ])
            w = np.asarray(wz, dtype=float) * half
            return E, w
        raise Error("QR fallback is not available for continuous measures in dimension >= 2; "
                    "use a discrete measure or reduce the basis size.")

    def _form_basis_qr(self):
        r"""
        Weighted QR orthogonalization on basis evaluations (fallback when the
        Gram matrix is ill-conditioned). Produces callables
        :math:`\tilde{\varphi}_i = \sum_j c_{ij}\,\varphi_j` whose evaluation vectors at the
        nodes are orthonormal, i.e. mutually orthogonal w.r.t. the (quadrature of
        the) measure — exact for discrete measures.

        Raises :class:`Error` if the basis is linearly dependent on the support.
        """
        from numpy import sqrt as _sqrt

        E, weights = self._eval_matrix(None, None)
        A = E * _sqrt(weights)[:, None]
        rank = np.linalg.matrix_rank(A)
        if rank < len(self.base):
            raise Error(
                f"the basis (size {len(self.base)}) is linearly dependent on the support "
                f"(rank {rank}); reduce deg or the basis."
            )
        Q, _ = np.linalg.qr(A)
        # coefficients c_i with A c_i = q_i; lstsq returns them COLUMN-wise:
        # column i of C is the raw-basis coefficient vector of tilde-phi_i.
        C = np.linalg.lstsq(A, Q, rcond=None)[0]
        self.orth_base = []
        for i_ in range(C.shape[1]):
            fn = lambda x, i_=i_: sum(
                [C[j, i_] * self.base[j](x) for j in range(len(self.base))]
            )
            self.orth_base.append(fn)

    def form_basis(self):
        """
        Call this method to generate the orthogonal basis corresponding
        to the given basis.
        The result will be stored in a property called ``orth_base`` which
        is a list of function that are orthogonal to each other with
        respect to the measure ``measure`` over the given range ``domain``.

        Uses Gram-determinant coefficients while :math:`\\mathrm{cond}(G) \\leq
        \\mathtt{GRAM\\_COND\\_LIMIT}`; above it, falls back to a weighted QR
        orthogonalization on basis evaluations (exact for discrete measures),
        because the determinant ratios lose all digits once the Gram matrix is
        ill-conditioned — previously this produced silently wrong fits
        (e.g., a degree-10 monomial fit over [0, 1] returned O(2e2) where the
        answer was O(3)).
        """
        num = len(self.base)
        if self.Gram is None:
            self.gram_mat()
        cond_G = np.linalg.cond(self.Gram)
        if not np.isfinite(cond_G) or cond_G > GRAM_COND_LIMIT:
            self._form_basis_qr()
            return
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
            # Exact-match spikes (relative tolerance only): under a discrete measure
            # supported exactly on the abscissas, <f, b>_mu reduces to sum_j y_j b(x_j),
            # i.e. ordinary (weighted) least squares. The previous absolute 1e-4 window
            # merged distinct but close abscissas in normalized units (see _spike_value).
            self.f = lambda x: sum(
                [
                    _spike_value(x, array(p_[:-1]), p_[-1])
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
        # --- exact mean-response confidence-band machinery -------------------------
        # The model is linear in its parameters (L2(mu) projection onto a basis), so
        # the variance of the fitted value at x* is  s^2 * b(x*)' M^-1 b(x*), where
        # M = B' W B is the weighted Gram matrix of the raw basis and s^2 is the
        # weighted residual mean square with df = n - k. The old closed form
        # t*s*(1/n + (x-xbar)^2/sqrt(Sxx)) was only valid for one-predictor linear
        # regression and, as written (missing outer sqrt, sqrt(Sxx) instead of Sxx),
        # produced bands ~8x too narrow; it is replaced by the exact expression.
        n = self.training_size
        k = len(self.f_space.base)
        self.df = n - k
        meas_ = self.regressor.meas
        res_vec = np.array([float(np.ravel(self.apprx(x))[0]) for x in X]).reshape(-1) \
            - np.asarray(y).reshape(-1)
        w_all = (np.ones(n, dtype=float) if meas_.continuous
                 else np.array([meas_.density(tuple(row)) for row in X], dtype=float))
        self.training_var = float(np.sqrt(
            float(w_all @ res_vec ** 2) / max(self.df, 1))) \
            if self.df > 0 else float("nan")
        self.t_stat = t.ppf(self.c_limit, max(self.df, 1)) \
            if self.df > 0 and not np.isnan(self.training_var) else 0.0
        # Backward-compatible attributes (no longer used by predict).
        if X.shape[1] == 1:
            self.x_mean = float(X.mean())
            self.sum_sqrd_x = float(np.sum(power(X, 2)))
        self.ci_available = bool((not meas_.continuous) and self.df > 0)
        if self.ci_available:
            E = np.array([[float(np.ravel(phi(x))[0]) for phi in self.f_space.base] for x in X])
            self._gram_w = (E.T * w_all) @ E
            try:
                np.linalg.solve(self._gram_w, np.eye(k))  # early singularity check
            except np.linalg.LinAlgError:
                import warnings
                warnings.warn("weighted Gram matrix is singular; confidence bands use pinv",
                              stacklevel=2)
        return self

    def predict(self, X):
        """
        Predict using the Hilbert regression method

        :param X: data points for prediction
        :return: returns predicted values

        Side effect: when a valid band is available (discrete measure and
        ``n >`` number of basis functions), ``self.ci_band`` holds the two-sided
        mean-response confidence band at level ``c_limit`` for the queried points.
        """
        if len(X.shape) != 2:
            X = X.reshape(X.shape[0], 1)
        if self.ci_available and X.shape[1] == 1:
            Bq = np.array([[float(np.ravel(phi(x))[0]) for phi in self.f_space.base] for x in X])
            try:
                Mi_bT = np.linalg.solve(self._gram_w, Bq.T)      # M^-1 b_q'   (k x m)
            except np.linalg.LinAlgError:
                Mi_bT = np.linalg.pinv(self._gram_w) @ Bq.T
            var_mean = np.einsum("ij,ji->i", Bq, Mi_bT)          # diag(b M^-1 b')
            self.ci_band = (self.t_stat * self.training_var
                            * sqrt(np.clip(var_mean, 0.0, None))).reshape(-1)
        return array([self.apprx(x) for x in X]).reshape(-1)

    def score(self, X, y, sample_weight=None):
        r"""
        Coefficient of determination :math:`R^2 = 1 - \mathrm{SS}_{res}/\mathrm{SS}_{tot}`.

        Follows the scikit-learn convention (higher is better; 1.0 for a perfect fit).
        The previous version returned the weighted mean square error — use :meth:`mse`
        for that statistic.
        """
        if len(X.shape) != 2:
            X = X.reshape(X.shape[0], 1)
        vals = np.array([self.apprx(x) for x in X]).reshape(-1, 1).ravel()
        yv = np.asarray(y).reshape(-1)
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - float(np.sum((yv - vals) ** 2)) / ss_tot

    def mse(self, X, y):
        r"""
        Weighted mean square error :math:`\sum_i w_i (y_i - \hat{y}_i)^2 / \sum_i w_i`
        with the measure weights `w_i` (the pre-0.3 ``score`` semantics).

        Lower is better — do not use it where a scikit-learn-style ``score`` is expected.
        """
        if len(X.shape) != 2:
            X = X.reshape(X.shape[0], 1)
        weights = np.array([self.regressor.meas.density(tuple(x)) for x in X], dtype=float)
        vals = np.array([self.apprx(x) for x in X]).reshape(-1, 1).ravel()
        errs = weights * (vals - np.asarray(y).reshape(-1)) ** 2
        wsum = float(weights.sum())
        return float(errs.sum() / wsum) if wsum > 0 else float("nan")
