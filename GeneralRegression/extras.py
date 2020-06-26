"""
Function Basis for `HilbertRegressor`
===========================================
"""


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
        from numpy import prod

        base = []
        for o in product(range(deg + 1), repeat=n):
            if sum(o) <= deg:
                if n > 1:
                    base.append(lambda x, e=o: prod([x[i] ** e[i] for i in range(n)]))
                else:
                    base.append(lambda x, e=o: prod([x ** e[i] for i in range(n)]))
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
                            f_ = lambda x, o_=o, ex_=ex: prod(
                                [
                                    sin(o_[i] * x / l) ** ex_[i]
                                    * cos(o_[i] * x / l) ** (1 - ex_[i])
                                    if o_[i] > 0
                                    else 1.0
                                    for i in range(n)
                                ]
                            )

                        base.append(f_)
        return base
