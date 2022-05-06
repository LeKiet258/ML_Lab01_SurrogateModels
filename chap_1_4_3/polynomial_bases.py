import numpy as np
import itertools

# create Basic Function for each component with level k
def polynomial_bases_1d(i,k):
    f = lambda x: np.power(x[i], np.arange(0, k+1, 1))
    return f

# create Basic Function for n element in array with level k
def polynomial_bases(n,k):
    # list of basic functions
    bases = [polynomial_bases_1d(i, k) for i in range(n)]
    # list for 
    terms = []
    for ks in itertools.product(*[np.arange(0,k+1) for i in range(n)]):
        if np.sum(ks) <= k:
            def func(x, ks=ks):
                return np.prod([b(x)[j] for j, b in zip(ks,bases)])
            terms.append(func)

    return terms