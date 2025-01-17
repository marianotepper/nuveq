import numpy as np

from nuveq import forward_kumaraswamy, normalized_logistic, normalize, logistic, \
    logistic_nqt

def apply_kumaraswamy(sol, data):
    data_max = data.max()
    data_min = data.min()

    return forward_kumaraswamy(normalize(data, data_min, data_max),
                               sol[0], sol[1])

def loss_kumaraswamy(histogram, x):
    diff = apply_kumaraswamy(x, histogram[0]) - histogram[1]
    return np.sum(diff ** 2)

def apply_triple_kumaraswamy(sol, data):
    data_max = data.max()
    data_min = data.min()

    z = normalize(data, data_min, data_max)
    return (1 - (1 - z ** sol[0]) ** sol[1]) ** sol[2]

def loss_triple_kumaraswamy(histogram, x):
    diff = apply_triple_kumaraswamy(x, histogram[0]) - histogram[1]
    return np.sum(diff ** 2)


def apply_logistic(logistic_fun, sol, data):
    data_max = data.max()
    data_min = data.min()



    delta = data_max - data_min
    alpha = sol[0] / delta
    x0 = sol[1] * delta + data_min

    bias = logistic_fun(data_min, alpha, x0)
    scale = logistic_fun(data_max, alpha, x0) - bias

    return normalized_logistic(logistic_fun, data, alpha, x0, scale, bias)

def loss_logistic(logistic_fun, histogram, x):
    diff = apply_logistic(logistic_fun, x, histogram[0]) - histogram[1]
    return np.sum(diff ** 2)