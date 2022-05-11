import numpy as np

def radial_bases(psi, C, p=2):
    bases = []
    for c in C:
        def func(x, c=c):
            return psi(np.linalg.norm(x-c, p))
        bases.append(func)
    return bases