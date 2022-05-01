import numpy as np

def regression(X, y, bases, lamb):
    B = np.array([[b(x) for b in bases] for x in X])
    theta = np.linalg.solve(np.matmul(B.T,B) + np.diag(lamb*np.ones((len(bases),1))),np.matmul(B.T,y))
    return lambda x: np.sum([theta[i]*bases[i](x) for i in range(len(theta))])