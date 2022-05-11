from itertools import product
import math
import numpy as np

# Singular basis function


def constant(c):
    return lambda x: c


def power(i, k):
    return lambda x: x[i] ** k


def sin(i, c):
    return lambda x: math.sin(c * x[i])


def cos(i, c):
    return lambda x: math.cos(c * x[i])


# Basis functions set


def linear(n):
    return [constant(1.0)] + [power(i, 1) for i in range(n)]


def polynomial_1d(i, k):
    def power(i, k):
        return lambda x: x[i] ** k
    return [power(i, _k) for _k in range(k + 1)]


def polynomial(n, k):
    terms = []
    bases = [polynomial_1d(i, k) for i in range(n)]

    for ks in product(*[[i for i in range(k + 1)] for _ in range(n)]):
        if sum(ks) <= k:
            def func(ks):
                return lambda x: np.prod(np.array([bases[i][ks[i]](x) for i in range(n)]))
            terms.append(func(ks))

    return terms


def sinusoidal_1d(i, k, a, b):
    terms = [constant(0.5)]

    T = b[i] - a[i]
    for j in range(1, k + 1):
        terms.append(sin(i, 2.0 * math.pi * j / T))
        terms.append(cos(i, 2.0 * math.pi * j / T))

    return terms


def sinusoidal(n, k, a, b):
    terms = []
    bases = [sinusoidal_1d(i, k, a, b) for i in range(n)]

    for ks in product(*[[i for i in range(k + 1)] for _ in range(n)]):
        if sum([_k + 1 // 2 for _k in ks]) <= k:
            def func(ks):
                return lambda x: np.prod(np.array([bases[i][ks[i]](x) for i in range(n)]))
            terms.append(func(ks))

    return terms


def radial(f, C, p=2):
    def func(c):
        return lambda x: f(np.linalg.norm(x - c, p))

    return [func(c) for c in C]
