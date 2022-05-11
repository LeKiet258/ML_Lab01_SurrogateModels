import numpy as np

from metrics import MSE


class SurrogateModel:
    def __init__(self, bases, smoothing=0.2, metric=MSE):
        self.__bases = bases
        self.__smoothing = smoothing
        self.__metric = metric
        return

    def transform(self, X, y=None):
        return X

    def fit(self, X, y):
        transformedX = np.array(
            [np.array([base(x) for base in self.__bases]) for x in X])

        m = len(transformedX[0])
        lambda_I = np.matrix([[self.__smoothing * (row == col)
                             for col in range(m)] for row in range(m)])
        b = np.matrix(transformedX, copy=True)
        bT = b.transpose()
        bTb = np.matmul(bT, b)
        bTy = np.matmul(bT, np.matrix([[y] for y in y]))
        bp = np.add(bTb, lambda_I)
        bpi = np.linalg.inv(bp)
        self.theta = np.matmul(bpi, bTy)
        return self

    def predict(self, X):
        return np.array(np.matmul(np.matrix(np.array(
            [np.array([b(x) for b in self.__bases]) for x in X])), self.theta))

    def validate(self, X, y):
        return self.__metric(self.predict(X), y)
