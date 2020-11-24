"""
Function Basis and Time to Interval transformer
================================================
"""
from datetime import datetime, date, timedelta

from pandas import Timestamp


class FunctionBasis(object):
    """
    This class generates two typical basis of functions: Polynomials and Trigonometric
    """

    def __init__(self):
        pass

    @staticmethod
    def poly(n, deg):
        """
        Returns a basis consisting of polynomials in `n` variables of degree at most `deg`.

        :param n: number of variables
        :param deg: highest degree of polynomials in the basis
        :return: the raw basis consists of polynomials of degrees up to `n`
        """
        from itertools import product
        from numpy import prod, power

        base = []
        for o in product(range(deg + 1), repeat=n):
            if sum(o) <= deg:
                if n > 1:
                    base.append(lambda x, e=o: prod([x[i] ** e[i] for i in range(n)]))
                else:  # One-dimensional case
                    base.append(lambda x, e=o: power(x, e[0]))  # if x.shape[0] > 0 else x ** e[0])
        return base

    @staticmethod
    def fourier(n, deg, l=1.0):
        """
        Returns the Fourier basis of degree `deg` in `n` variables with period `l`

        :param n: number of variables
        :param deg: the maximum degree of trigonometric combinations in the basis
        :param l: the period
        :return: the raw basis consists of trigonometric functions of degrees up to `n`
        """

        from numpy import sin, cos, prod
        from itertools import product

        base = [lambda x: 1.0]
        exponents = list(product([0, 1], repeat=n))
        raw_coefs = list(product(range(deg + 1), repeat=n))
        coefs = set()
        for prt in raw_coefs:
            p_ = list(prt)
            p_.sort()
            coefs.add(tuple(p_))
        for o in coefs:
            if (sum(o) <= deg) and (sum(o) > 0):
                for ex in exponents:
                    if sum(ex) >= 0:
                        if n > 1:
                            f_ = lambda x, o_=o, ex_=ex: prod(
                                [
                                    sin(o_[i] * x[i] / l) ** ex_[i]
                                    * cos(o_[i] * x[i] / l) ** (1 - ex_[i])
                                    if o_[i] > 0
                                    else 1.0
                                    for i in range(n)
                                ]
                            )
                        else:
                            f_ = lambda x, o_=o, ex_=ex: (sin(o_[0] * x[0] / l) ** ex_[0]) * cos(o_[0] * x[0] / l) ** (
                                        1 - ex_[0]) \
                                if tuple(o_)[0] > 0 \
                                else 1.0

                        base.append(f_)
        return base


class Time2Interval(object):
    """
    Transforms a given date-time period into a real interval.

    :param min_date: the start date of the time period (`datetime` object)
    :param max_date: the end date of the time period (`datetime` object)
    :param lower: (default = 0.) the real number that  corresponds to `min_date`
    :param upper: (default = 1.) the real number that  corresponds to `max_date`
    """

    def __init__(self, min_date, max_date, lower=0, upper=1.):
        if type(min_date) not in [datetime, date, Timestamp]:
            raise TypeError("`min_date` must be `datetime` object")
        if type(max_date) not in [datetime, date, Timestamp]:
            raise TypeError("`max_date` must be `datetime` object")
        self.min_date = min_date
        self.max_date = max_date
        self.lower = lower
        self.upper = upper
        t_length = (max_date - min_date).days
        n_length = upper - lower
        self.trans_coef = n_length / t_length
        self.intercept = lower

    def date2num(self, calendar_date):
        """
        Transforms a datetime object into a real number according to the initial parameters
        :param calendar_date: a datetime object to be converted into real number
        :return: the transformed number
        """
        if type(calendar_date) not in [datetime, date, Timestamp]:
            raise TypeError("`calendar_date` must be `datetime` object")
        corresponding_x_number = (calendar_date - self.min_date).days
        corresponding_y_number = self.trans_coef * corresponding_x_number + self.intercept
        return corresponding_y_number

    def num2date(self, number):
        """
        Transforms back a given real number to a datetime relative to the initiation data
        :param number: a real number to be converted to a date
        :return: the transformed back datetime object associated to the input
        """
        return self.min_date + timedelta(days=(number - self.intercept) / self.trans_coef)
