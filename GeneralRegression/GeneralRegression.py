from sklearn.base import BaseEstimator, RegressorMixin


class GenericRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, funcs, regressor=None, ci=.95, **kwargs):
        from sklearn.linear_model import LinearRegression
        self.funcs = funcs
        self.kw_ar = kwargs
        # to store the standard deviations of the predicted values
        self.std = False
        # convert the confidence interval value to find the ppf
        self.ci = ci  # (1 + ci) / 2.
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
            from scipy.stats import norm
            pred, std = self.model.predict(X, return_std=True)
            z_bar = norm.ppf(self.ci, 0, 1)
            self.ci_band = z_bar * std  # / sqrt(max(self.size - 1 , 1))
        else:
            pred = self.model.predict(X)
            std = None
        self.standard_deviation = std
        return pred
