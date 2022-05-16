from types import NoneType
import numpy as np

from train_test import train_test
from surrogate_model import SurrogateModel

# Holdout

def holdout_sets(m, k, h=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    if h is None:
        h = m // 2

    res = []

    for i in range(k):
        p = np.random.permutation(m)
        res.append(train_test(p[:h], p[h:]))

    return res

def holdout_estimate(X, y, bases, smoothing, metric, sets=None):
    if sets is None:
        sets = holdout_sets(len(y), int(len(y) * 0.4))

    b = len(sets)
    res = 0.0
    model = SurrogateModel(bases, smoothing, metric)
    for tt in sets:
        model.fit(X[tt['train']], y[tt['train']])
        res += model.validate(X[tt['test']], y[tt['test']])
    return res / b

# k-fold cross validation estimate

def k_fold_cross_validation_sets(m, k, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    p = np.random.permutation(m)
    res = []

    for i in range(k):
        validate = p[i:m:k]
        train    = list(set(p).symmetric_difference(set(validate)))
        res.append(train_test(train, validate))

    return res

def k_fold_cross_validation_estimate(X, y, bases, smoothing, metric, sets=None):
    if sets is None:
        sets = k_fold_cross_validation_sets(len(y), 10)

    b = len(sets)
    res = 0.0
    model = SurrogateModel(bases, smoothing, metric)
    for tt in sets:
        model.fit(X[tt['train']], y[tt['train']])
        res += model.validate(X[tt['test']], y[tt['test']])
    return res / b

# Bootstrap estimate

def bootstrap_sets(m, b, random_state=None):
    if random_state:
        np.random.seed(random_state)
    return np.array([train_test(np.random.choice(m, m, replace=True), np.array(range(m))) for i in range(b)])


def bootstrap_estimate(X, y, bases, smoothing, metric, sets=None):
    if sets is None:
        sets = bootstrap_sets(len(y), int(len(y) * 0.4))

    b = len(sets)
    res = 0.0
    model = SurrogateModel(bases, smoothing, metric)
    for tt in sets:
        model.fit(X[tt['train']], y[tt['train']])
        res += model.validate(X[tt['test']], y[tt['test']])
    return res / b


def leave_one_out_bootstrap_estimate(X, y, bases, smoothing, metric, sets=None):
    if sets is None:
        sets = bootstrap_sets(len(y), int(len(y) * 0.4))

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


def bootstrap_632_estimate(X, y, bases, smoothing, metric, sets=None):
    if sets is None:
        sets = bootstrap_sets(len(y), int(len(y) * 0.4))

    eboot = bootstrap_estimate(X, y, bases, smoothing, metric, sets)
    eloob = leave_one_out_bootstrap_estimate(
        X, y, bases, smoothing, metric, sets)
    return 0.632 * eloob + 0.368 * eboot
