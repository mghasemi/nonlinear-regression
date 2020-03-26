from sklearn.base import BaseEstimator, RegressorMixin


class GenericRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, funcs, **kwargs):
        self.funcs = funcs
        self.kw_ar = kwargs
