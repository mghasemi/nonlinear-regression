"""
Time Series Tools
========================
"""
from abc import ABCMeta

import numpy as np

from sklearn.model_selection import BaseCrossValidator


class TimeSeriesCV(BaseCrossValidator, metaclass=ABCMeta):
    """
    A walk-forward (expanding-window) cross validator for time series. The given index column
    (default 0) is sorted and the sorted positions are split into training and test sets: no
    observation may ever train on data that comes after it, so folds never shuffle or overlap
    in time.

    With ``n_splits=1`` (the default) this reproduces the original single-split behavior
    exactly: the first ``train_ratio`` of the observations form the training set and the rest
    the test set. For ``n_splits >= 2`` the test region is partitioned into that many
    contiguous chunks and fold ``i`` trains on everything strictly before its chunk, so the
    training window expands from fold to fold (the standard walk-forward scheme).

    :param test_ratio: (default .2) float between 0. and 1., the portion of data used for testing overall
    :param train_ratio: (default `None` -> .8) float between 0. and 1.; if given, overrides ``test_ratio``
    :param index: (default 0) the column index that corresponds to a time parameter in the data;
        must satisfy ``0 <= index < n_features``
    :param n_splits: (default 1) int >= 1, number of walk-forward folds. With more than one fold
        the test region is split into contiguous chunks and each fold trains on all earlier data.
    """

    def __init__(self, test_ratio=.2, train_ratio=None, index=0, n_splits=1):
        self.index = index
        if not isinstance(n_splits, (int, np.integer)) or n_splits < 1:
            raise ValueError("`n_splits` must be an integer >= 1")
        self.n_splits = int(n_splits)
        if train_ratio is not None:
            if not 0. <= train_ratio < 1.:
                raise ValueError("the value of `train_ratio` should be in [0., 1.)")
            self.train_ratio = train_ratio
            self.test_ratio = 1. - train_ratio
        elif test_ratio is not None:
            if not 0. <= test_ratio < 1.:
                raise ValueError("the value of `test_ratio` should be in [0., 1.)")
            self.train_ratio = 1. - test_ratio
            self.test_ratio = test_ratio
        else:
            self.train_ratio = .8
            self.test_ratio = .2

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        :param X: Always ignored, exists for compatibility.
        :param y: Always ignored, exists for compatibility.
        :param groups: Always ignored, exists for compatibility.
        :return: The number of walk-forward folds (``self.n_splits``).
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set(s), in time order.

        :param X: array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features. Must be two-dimensional;
            ``X[:, self.index]`` is used as the time coordinate.
        :param y: array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        :param groups: array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set(s).
        :return: yields one (train_index, test_index) pair per fold; within a fold every
            training position precedes every test position in time order.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional so that `index` selects a column")
        if not 0 <= self.index < X.shape[1]:
            raise ValueError(
                f"`index={self.index}` is out of range for {X.shape[1]} feature columns"
            )

        n = X.shape[0]
        # argsort only reads the column, so no copy of X is needed
        sorted_index = np.argsort(X[:, self.index], kind="stable")

        n_test_total = int(self.test_ratio * n)
        if not 1 <= n_test_total < n:
            raise ValueError(
                f"cannot build a non-degenerate split from {n} observations with "
                f"test_ratio={self.test_ratio}: the test region must contain at least one "
                "observation and leave at least one for training"
            )

        # walk-forward layout: the last n_test_total observations form the "future" test
        # region; partition it into n_splits contiguous chunks and let fold i train on
        # everything strictly before its chunk (expanding window, no time leakage).
        # With n_splits=1 this reproduces the original single 80/20-style split exactly.
        base = n_test_total // self.n_splits
        extra = n_test_total % self.n_splits
        start = n - n_test_total
        for fold in range(self.n_splits):
            size = base + (1 if fold < extra else 0)
            end = start + size
            train_index = sorted_index[:start]
            test_index = sorted_index[start:end]
            yield train_index, test_index
            start = end
