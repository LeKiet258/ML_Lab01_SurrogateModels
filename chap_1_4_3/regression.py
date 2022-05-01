import numpy as np
import itertools

def linear_bases(i):
    f = lambda x: x[i]
    return f

def regression(X, y, bases):
    B = [b(x) for x, b in itertools.product(X, bases)]
    theta = np.linalg.pinv(B)*y
    f = lambda x: np.sum([theta[i]*bases[i](x) for i in range(len(theta))])
    return f