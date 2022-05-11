import numpy as np
import itertools

def polynomial_bases_1d(i,k):
    def func(x):
        f = []
        for p in np.arange(0, k+1, 1):
            f.append(np.power(x[i], p))
        return np.array(f)

    f = func
    return f

def polynomial_bases(n,k):
    bases = []
    for i in range(n):
        bases.append(polynomial_bases_1d(i, k))
    
    terms = []
    for ks in itertools.product(*[np.arange(0, k+1) for i in range(n)]):
        if np.sum(ks) <= k:
            def func(x, ks=ks):
                term = 1
                for j, b in zip(ks, bases):
                    term *= b(x)[j]
                return term

            terms.append(func)

    return terms