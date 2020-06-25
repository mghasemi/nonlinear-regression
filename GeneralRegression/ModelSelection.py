"""
Time Series Tools
========================
"""
from abc import ABCMeta

from sklearn.model_selection import BaseCrossValidator


class TimeSeriesCV(BaseCrossValidator, metaclass=ABCMeta):
    """
    This is a very naive cross validator for time series. It simply sorts the given index (default 0)
    and splits the sorted index into a train and a test index set according to the given ratios.

    :param test_ratio: (default .2) float betweem 0. and 1., the portion of test data
    :param train_ratio: (default `None`-> .8) float betweem 0. and 1., the portion of train data
    :param index: (default 0) the index of the column that corresponds to a time parameter in the data
    """

    def __init__(self, test_ratio=.2, train_ratio=None, index=0):
        self.index = index
        if train_ratio is not None:
            if train_ratio > 1.:
                raise ValueError("the value of `rain_ratio` should be smaller than 1.")
            self.train_ratio = train_ratio
            self.test_ratio = 1. - train_ratio
        elif test_ratio is not None:
            if test_ratio > 1.:
                raise ValueError("the value of `test_ratio` should be smaller than 1.")
            self.train_ratio = 1. - test_ratio
            self.test_ratio = test_ratio
        else:
            self.train_ratio = .8
            self.test_ratio = .2

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator

        :param X: Always ignored, exists for compatibility.
        :param y: Always ignored, exists for compatibility.
        :param groups: Always ignored, exists for compatibility.
        :return: Returns the number of splitting iterations in the cross-validator which is 1 for time series.
        """
        return 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        :param X: array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        :param y: array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        :param groups: array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: `train` The training set indices for that split. `test` The testing set indices for that split.
        """
        from copy import copy
        X_c = copy(X)
        sorted_index = X_c[:, self.index].argsort()
        cut = int(self.train_ratio * sorted_index.shape[0])
        train_index = sorted_index[: cut]
        test_index = sorted_index[cut:]
        yield train_index, test_index
