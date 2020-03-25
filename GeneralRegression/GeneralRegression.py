from sklearn.base import BaseEstimator, RegressorMixin


class GenericRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, funcs):
        self.funcs = funcs
