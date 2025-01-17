from dataclasses import dataclass
import numpy as np

from .nes import ExponentialNES

def quantize(x, n_bits):
    return np.round((2 ** n_bits - 1) * x) / (2 ** n_bits - 1)


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def denormalize(y, x_min, x_max):
    return y * (x_max - x_min) + x_min


def forward_kumaraswamy(x, a, b):
    return 1 - (1 - x ** a) ** b


def inverse_kumaraswamy(y, a, b):
    z = (1 - (1 - y) ** (1 / b)) ** (1 / a)
    return z


def loss_kumaraswamy(sol, data, n_bits, x_min, x_max):
    a, b = sol[0], sol[1]
    z = normalize(data, x_min, x_max)
    z = forward_kumaraswamy(z, a, b)
    z = quantize(z, n_bits)
    z = inverse_kumaraswamy(z, a, b)
    z = denormalize(z, x_min, x_max)
    diff = z - data
    return np.sum(diff ** 2)


def logistic(x, alpha, x0):
    return 1 / (1 + np.exp(-alpha * (x - x0)))


def logit(y, alpha, x0):
    z = y / (1 - y)
    return np.log(z) / alpha + x0


def logistic_nqt(x, alpha, x0):
    z = alpha * (x - x0)
    p = np.round(z + 0.5)

    m = 0.5 * (z - p) + 1
    y = m * (2 ** p)
    return y / (y + 1)


def logit_nqt(y, alpha, x0):
    z = y / (1 - y)
    m, p = np.frexp(z)
    return (2 * m - 2 + p) / alpha + x0


def normalized_logistic(logistic_fun, x, alpha, x0, scale, bias):
    y = logistic_fun(x, alpha, x0)
    return (y - bias) / scale


def normalized_logit(logit_fun, y, alpha, x0, scale, bias):
    y_scaled = scale * y + bias
    return logit_fun(y_scaled, alpha, x0)


def loss_logistic(sol, data, n_bits, x_min, x_max, logistic_fun, logit_fun):
    delta = x_max - x_min
    alpha = sol[0] / delta
    x0 = sol[1] * delta

    bias = logistic_fun(x_min, alpha, x0)
    scale = logistic_fun(x_max, alpha, x0) - bias

    z = normalized_logistic(logistic_fun, data, alpha, x0, scale, bias)
    z = quantize(z, n_bits)
    z = normalized_logit(logit_fun, z, alpha, x0, scale, bias)
    diff = z - data
    return np.sum(diff ** 2)


def loss_uniform(data, n_bits):
    x_min = data.min()
    x_max = data.max()

    z = normalize(data, x_min, x_max)
    z = quantize(z, n_bits)
    z = denormalize(z, x_min, x_max)
    return np.sum((data - z) ** 2)


@dataclass
class NVQParams:
    x_min: float
    x_max: float
    distribution_params: np.ndarray


class NonuniformVectorQuantization:
    def __init__(self, n_bits, n_subvectors=1,
                 nonlinearity='kumaraswamy',
                 optimization_tol=1e-4,
                 optimization_max_iter=None,
                 optimization_seed=0,
                 store_optimization_loss_history=False):
        self.n_bits = n_bits
        self.n_subvectors = n_subvectors
        self.nonlinearity = nonlinearity
        self.optimization_tol = optimization_tol
        self.optimization_max_iter = optimization_max_iter
        self.optimization_seed = optimization_seed
        self.store_opt_loss_history = store_optimization_loss_history

        self.optimization_loss_history = None

    def _get_init_state(self, x_min, x_max):
        if self.nonlinearity == 'kumaraswamy':
            init_sol = np.ones((2,))
            bounds = [(1e-6, np.inf)] * 2
            sigma = 1
        elif self.nonlinearity == 'logistic' or self.nonlinearity == 'NQT':
            delta = x_max - x_min
            init_sol = np.array([10., 0.])
            bounds = [(1e-6, np.inf), (x_min / delta, x_max / delta)]
            sigma = np.array([2, 0.5])
        else:
            raise ValueError('Unknown nonlinearity')

        return init_sol, bounds, sigma

    def _get_loss_fun(self, data, x_min, x_max, relative=True):
        baseline = loss_uniform(data, self.n_bits)

        if self.nonlinearity == 'kumaraswamy':
            loss_fun = lambda sol: loss_kumaraswamy(
                sol, data, self.n_bits, x_min, x_max
            )
        elif self.nonlinearity == 'logistic':
            loss_fun = lambda sol: loss_logistic(
                sol, data, self.n_bits, x_min, x_max, logistic, logit
            )
        elif self.nonlinearity == 'NQT':
            loss_fun = lambda sol: loss_logistic(
                sol, data, self.n_bits, x_min, x_max, logistic_nqt, logit_nqt
            )
        else:
            raise ValueError('Unknown nonlinearity')

        if relative:
            return lambda sol: baseline / loss_fun(sol)
        else:
            return loss_fun

    def split_in_subvectors(self, data):
        step = len(data) // self.n_subvectors
        return [data[j:j + step] for j in np.arange(0, len(data), step)]

    def optimize(self, data):
        nvq_params = []

        for subvector in self.split_in_subvectors(data):
            x_min = subvector.min()
            x_max = subvector.max()

            loss_fun = self._get_loss_fun(subvector, x_min, x_max)
            init_sol, bounds, sigma = self._get_init_state(x_min, x_max)

            xnes = ExponentialNES(tol=self.optimization_tol,
                                  max_iter=self.optimization_max_iter,
                                  seed=self.optimization_seed,
                                  distribution='separable')
            result = xnes.optimize(loss_fun, init_sol, bounds=bounds,
                                   sigma=sigma,
                                   return_history=self.store_opt_loss_history)
            if self.store_opt_loss_history:
                xnes_sol, self.optimization_loss_history = result
            else:
                xnes_sol = result

            nvq_params.append(NVQParams(x_min, x_max, xnes_sol))

        if self.n_subvectors == 1:
            return nvq_params[0], loss_fun(xnes_sol)
        else:
            return nvq_params, self.relative_loss(data, nvq_params)


    def loss(self, data, params):
        loss = 0
        for subvector in self.split_in_subvectors(data):
            loss_fun = self._get_loss_fun(subvector,
                                          params.x_min, params.x_max,
                                          relative=False)
            loss += loss_fun(params.distribution_params)

        return loss

    def relative_loss(self, data, params):
        baseline = loss_uniform(data, self.n_bits)

        loss = 0
        for subvector, par in zip(self.split_in_subvectors(data), params):
            loss_fun = self._get_loss_fun(subvector,
                                          par.x_min, par.x_max,
                                          relative=False)
            loss += loss_fun(par.distribution_params)

        return baseline / loss

