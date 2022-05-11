import numpy as np
import itertools

def sinusoidal_bases_1d(j, k, a, b):
    T= b[j] - a[j]
    
    funcs = []

    def func(x, j, T, i):
        return np.sin(2 * np.pi * i * x[j]/T)
    funcv = np.vectorize(func, excluded=[0, 1, 2])
    
    def func2(x, j, T, i):
        return np.cos(2 * np.pi * i * x[j]/T)
    funcv2 = np.vectorize(func2, excluded=[0, 1, 2])

    def func_cat(x, j=j, T=T, i=np.arange(1, k+1)):
        return np.concatenate((funcv(x, j, T, i), funcv2(x, j, T, i)))

    return func_cat

def sinusoidal_bases(k, a, b):
    n = len(a)
    bases = []
    for i in range(n):
        bases.append(sinusoidal_bases_1d(i, k, a, b))
    
    terms = []
    for ks in itertools.product(*[list(np.arange(0,(2*k))) for i in range(n)]):
        powers = [(k + 1)//2 for k in ks]
        if np.sum(powers) <= k:
            def func(x, ks=ks):
                term = 1
                for j, b in zip(ks,bases):
                    term *= b(x)[j]
                return term

            terms.append(func)

    return terms