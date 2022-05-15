import numpy as np

from train_test import train_test
from surrogate_model import SurrogateModel


# Holdout


# k-fold cross validation estimate


# Bootstrap estimate

def bootstrap_sets(m, b, random_state=None):
    if random_state:
        np.random.seed(random_state)
    return np.array([train_test(np.random.choice(m, m, replace=True), np.array(range(m))) for i in range(b)])


def bootstrap_estimate(X, y, bases, smoothing, metric, sets):
    b = len(sets)
    res = 0.0
    model = SurrogateModel(bases, smoothing, metric)
    for tt in sets:
        model.fit(X[tt['train']], y[tt['train']])
        res += model.validate(X[tt['test']], y[tt['test']])
    return res / b


def leave_one_out_bootstrap_estimate(X, y, bases, smoothing, metric, sets):
    m = len(X)
    b = len(sets)
    res = 0.0
    models = [SurrogateModel(bases, smoothing, metric) for i in range(b)]
    for i in range(b):
        models[i].fit(X[sets[i]['train']], y[sets[i]['train']])
    for j in range(m):
        c = 0
        delta = 0.0
        for i in range(b):
            if j not in sets[i]['train']:
                c += 1
                delta += models[i].validate([X[j]], [y[j]])
        if c != 0:
            res += delta / c
    return res / m


def bootstrap_632_estimate(X, y, bases, smoothing, metric, sets):
    eboot = bootstrap_estimate(X, y, bases, smoothing, metric, sets)
    eloob = leave_one_out_bootstrap_estimate(
        X, y, bases, smoothing, metric, sets)
    return 0.632 * eloob + 0.368 * eboot
