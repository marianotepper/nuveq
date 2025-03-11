import itertools
import numpy as np
from typing import Optional, Union


class ExponentialNES:
    """
    Implements Exponential Natural Evolution Strategies (xNES)
    for the multinormal and separable cases.
    These correspond to algorithms 5 and 6 in:

    Wierstra, Schaul, Glasmachers, Sun, Peters, and Schmidhuber
    Natural Evolution Strategies
    https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
    """

    def __init__(self, n_samples: Optional[int] = None,
                 lr_mu: Optional[int] = None, lr_sigma: Optional[int] = None,
                 tol: float = 1e-6, max_iter: Optional[int] = None,
                 distribution: str = 'separable',
                 seed: Union[None, int, np.random.Generator] = None):
        if distribution not in ['multinormal', 'separable']:
            raise ValueError(f"Distribution {distribution} is not supported: "
                             f"choose 'multinormal' or 'separable'")

        self.n_samples = n_samples
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.tol = tol
        self.max_iter = max_iter
        self.distribution = distribution
        self.seed = seed

    def _get_hyperparameters(self, n_dims: int):
        if n_dims <= 0:
            raise ValueError('n_dims must be a positive integer')

        # See Table 1 in
        # https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        if self.n_samples is None:
            n_samples = 2 * (4 + int(np.floor(3 * np.log(n_dims))))
        else:
            n_samples = self.n_samples

        if self.lr_mu is None:
            lr_mu = 1
        else:
            lr_mu = self.lr_mu

        if self.lr_sigma is None:
            lr_sigma = (9 + 3 * np.log(n_dims)) / (
                        5 * n_dims * np.sqrt(n_dims))
        else:
            lr_sigma = self.lr_sigma

        return n_samples, lr_mu, lr_sigma

    def compute_utilities(self, fun, samples):
        idx = np.arange(1, len(samples) + 1)
        utilities = np.maximum(0, np.log(1 + len(idx) / 2) - np.log(idx))
        utilities /= utilities.sum()
        utilities -= 1 / len(idx)

        fun_values = [fun(elem) for elem in samples]

        idx_sorted = np.argsort(fun_values)[::-1]
        utilities[idx_sorted] = utilities
        return utilities

    def optimize(self, fun, initial_solution: np.ndarray,
                 sigma: Union[float, np.ndarray] = 0.5,
                 bounds: Optional[list[tuple]] = None,
                 return_history: bool = False):
        if np.any(sigma <= 0):
            raise ValueError('sigma must be positive')

        rng = np.random.default_rng(self.seed)
        n_dims = len(initial_solution)

        if bounds is not None:
            min_bounds = np.array([bounds[d][0] for d in range(n_dims)])
            max_bounds = np.array([bounds[d][1] for d in range(n_dims)])
        else:
            min_bounds = -np.inf
            max_bounds = np.inf

        if self.distribution == 'multinormal':
            return self._optimize_multinormal(fun, initial_solution, sigma,
                                              rng, min_bounds, max_bounds,
                                              return_history)
        elif self.distribution == 'separable':
            return self._optimize_separable(fun, initial_solution, sigma,
                                            rng, min_bounds, max_bounds,
                                            return_history)
        else:
            raise ValueError(f"Distribution {self.distribution} is not "
                             f"supported: choose 'multinormal' or 'separable'")

    def _optimize_multinormal(self, fun, initial_solution: np.ndarray,
                              sigma: Union[float, np.ndarray],
                              rng: np.random.Generator,
                              min_bounds: Union[np.ndarray, float],
                              max_bounds: Union[np.ndarray, float],
                              return_history: bool):

        n_dims = len(initial_solution)

        # Compute number of random raw_samples and learning rates:
        n_samples, lr_mu, lr_sigma = self._get_hyperparameters(n_dims)

        mu = initial_solution.copy()
        B = np.eye(n_dims)
        Id = np.eye(n_dims)

        if self.max_iter is None:
            iter_range = itertools.count()
        else:
            iter_range = range(self.max_iter)

        for _ in iter_range:
            # draw raw_samples
            raw_samples = rng.normal(size=(n_samples, n_dims))
            samples = mu + sigma * raw_samples @ B.T
            samples = np.maximum(min_bounds, samples)
            samples = np.minimum(max_bounds, samples)

            utilities = self.compute_utilities(fun, samples)

            # Compute gradients:
            delta_mu = utilities @ raw_samples
            corr = raw_samples[:, :, np.newaxis] @ raw_samples[:, np.newaxis, :]
            delta_M = (corr - Id[np.newaxis, :, :]).T @ utilities
            delta_sigma = np.trace(delta_M) / n_dims
            delta_B = delta_M - delta_sigma * Id

            old_fun_val = fun(mu)

            # Update mean
            mu += lr_mu * sigma * B @ delta_mu
            mu = np.maximum(min_bounds, mu)
            mu = np.minimum(max_bounds, mu)
            # Update sqrt of covariance
            sigma *= np.exp(delta_sigma * lr_sigma / 2)
            B *= np.exp(delta_B * lr_sigma / 2)


            new_fun_val = fun(mu)
            if np.abs(new_fun_val - old_fun_val) < self.tol:
                break

        return mu

    def _optimize_separable(self, fun, initial_solution: np.ndarray,
                            sigma: Union[float, np.ndarray],
                            rng: np.random.Generator,
                            min_bounds: Union[np.ndarray, float],
                            max_bounds: Union[np.ndarray, float],
                            return_history: bool):

        n_dims = len(initial_solution)

        # Compute number of random raw_samples and learning rates:
        n_samples, lr_mu, lr_sigma = self._get_hyperparameters(n_dims)

        mu = initial_solution.copy()
        loss_history = [fun(mu)]

        if self.max_iter is None:
            iter_range = itertools.count()
        else:
            iter_range = range(self.max_iter)

        for iter in iter_range:
            # draw raw_samples
            raw_samples = rng.normal(size=(n_samples, n_dims))
            samples = mu + sigma * raw_samples
            samples = np.maximum(min_bounds, samples)
            samples = np.minimum(max_bounds, samples)

            utilities = self.compute_utilities(fun, samples)

            # Compute gradients:
            delta_mu = utilities @ raw_samples
            delta_sigma = utilities @ (raw_samples ** 2 - 1)

            # Update mean
            mu += lr_mu * sigma * delta_mu
            mu = np.maximum(min_bounds, mu)
            mu = np.minimum(max_bounds, mu)
            # Update standard deviation
            sigma *= np.exp(delta_sigma * lr_sigma / 2)

            loss_history.append(fun(mu))
            loss_diff = np.abs(loss_history[-1] - loss_history[-2])
            if iter > 10 and loss_diff < self.tol:
                break

        if return_history:
            return mu, loss_history
        else:
            return mu


def initial_test():
    fun = lambda x: -np.sum((x) ** 2)

    xnes = ExponentialNES(seed=0)
    sol = xnes.optimize(fun, np.ones((2,)), sigma=0.5,
                        bounds=[(-1000, 1000)] * 2)
    print(sol)


if __name__ == '__main__':
    initial_test()
