from functools import partial
import numpy as np
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy.optimize
import scipy.stats
import sys

from datasets import select_dataset
from nuveq import (NonuniformVectorQuantization, forward_kumaraswamy,
                   normalized_logistic, normalize, logistic, logit,
                   logistic_nqt, logit_nqt,
                   loss_uniform, loss_logistic, loss_kumaraswamy)


def apply_kumaraswamy(sol, data):
    data_max = data.max()
    data_min = data.min()

    return forward_kumaraswamy(normalize(data, data_min, data_max),
                               sol[0], sol[1])


def loss_histogram_fitting_kumaraswamy(histogram, x):
    diff = apply_kumaraswamy(x, histogram[0]) - histogram[1]
    return np.sum(diff ** 2)


def apply_logistic(logistic_fun, sol, data):
    data_max = data.max()
    data_min = data.min()

    delta = data_max - data_min
    alpha = sol[0] / delta
    x0 = sol[1] * delta

    bias = logistic_fun(data_min, alpha, x0)
    scale = logistic_fun(data_max, alpha, x0) - bias

    return normalized_logistic(logistic_fun, data, alpha, x0, scale, bias)


def loss_histogram_fitting_logistic(logistic_fun, histogram, sol):
    diff = apply_logistic(logistic_fun, sol, histogram[0]) - histogram[1]
    return np.sum(diff ** 2)


def main():
    dirname = sys.argv[1]
    dataset = select_dataset(dirname, 'ada002-100k')
    # dataset = select_dataset(dirname, 'openai-v3-small-100k')
    # dataset = select_dataset(dirname, 'gecko-100k')
    # dataset = select_dataset(dirname, 'nv-qa-v4-100k')
    # dataset = select_dataset(dirname, 'colbert-1M')
    print(dataset.name)

    all_data = dataset.X_db
    all_data -= np.mean(all_data, axis=0)
    vector = all_data[1]

    histogram = np.sort(vector), np.arange(0, len(vector)) / (len(vector) - 1)

    result_kuma = scipy.optimize.minimize(
        partial(loss_histogram_fitting_kumaraswamy, histogram),
        np.ones((2,)),
        bounds=[(1e-6, np.inf)] * 2
    )
    loss_hist_fitting_kuma = loss_kumaraswamy(result_kuma.x, vector, 8,
                                              vector.min(), vector.max())
    print('Reconstruction error histogram fitting - Kumaraswamy:',
          loss_uniform(vector, 8) / loss_hist_fitting_kuma)
    print(result_kuma.x)

    result_logistic = scipy.optimize.minimize(
        partial(loss_histogram_fitting_logistic, logistic, histogram),
        np.array([30, 0.5]),
        bounds=[(1e-6, np.inf), (-np.inf, np.inf)]
    )
    loss_hist_fitting_logistic = loss_logistic(result_logistic.x, vector, 8,
                                               vector.min(), vector.max(),
                                               logistic, logit)
    print('Reconstruction error histogram fitting - Log-Log:',
          loss_uniform(vector, 8) / loss_hist_fitting_logistic)

    result_nqt = scipy.optimize.minimize(
        partial(loss_histogram_fitting_logistic, logistic_nqt, histogram),
        np.array([30, 0.5]),
        bounds=[(1e-6, np.inf),
                (-np.inf, np.inf)]
    )
    loss_hist_fitting_nqt = loss_logistic(result_nqt.x, vector, 8,
                                               vector.min(), vector.max(),
                                               logistic_nqt, logit_nqt)
    print('Reconstruction error histogram fitting - NQT:',
          loss_uniform(vector, 8) / loss_hist_fitting_nqt)

    model = NonuniformVectorQuantization(n_bits=8,
                                         n_subvectors=1,
                                         optimization_seed=0,
                                         nonlinearity='kumaraswamy')
    sol, loss_value = model.optimize(vector)
    print('Reconstruction error NVQ-NQT', loss_value)
    print(sol.distribution_params)

    palette = plotly.colors.qualitative.Plotly

    lspace = np.linspace(vector.min(), vector.max(), num=10_000, endpoint=True)

    fig = go.Figure(data=[
        go.Scatter(name='Empirical CDF',
                   x=histogram[0], y=histogram[1],
                   line=dict(color=palette[0])),
        go.Scatter(name='Kumaraswamy CDF',
                   x=lspace, y=apply_kumaraswamy(result_kuma.x, lspace),
                   line=dict(color=palette[1])),
        go.Scatter(name='Logistic CDF',
                   x=lspace,
                   y=apply_logistic(logistic, result_logistic.x, lspace),
                   line=dict(color=palette[3])),
        go.Scatter(name='NQT Logistic CDF',
                   x=lspace,
                   y=apply_logistic(logistic_nqt, result_nqt.x, lspace),
                   line=dict(color=palette[4])),
    ])
    fig.add_vline(x=np.median(histogram[0]))
    fig.add_hline(y=0.5)
    fig.update_traces(mode='lines',
                      line=dict(width=3), marker=dict(size=5, ))
    fig.update_layout(template='plotly_white', legend_title=None,
                      yaxis_title=None)
    fig.show()

    # fig = go.Figure(data=[
    #     go.Box(name='Kumaraswamy CDF',
    #            y=np.abs(
    #                apply_kumaraswamy(result_kuma.x, histogram[0]) - histogram[
    #                    1]),
    #            boxpoints='outliers', marker_color=palette[1]),
    #     go.Box(name='Logistic CDF',
    #            y=np.abs(
    #                apply_logistic(logistic, result_logistic.x, histogram[0]) -
    #                histogram[1]),
    #            boxpoints='outliers', marker_color=palette[3]),
    #     go.Box(name='NQT Logistic CDF',
    #            y=np.abs(
    #                apply_logistic(logistic_nqt, result_nqt.x, histogram[0]) -
    #                histogram[1]),
    #            boxpoints='outliers', marker_color=palette[4]),
    # ])
    # fig.update_layout(legend_title=None, yaxis_title=None)
    # fig.show()

    fig = go.Figure(data=[
        go.Scatter(name='Kumaraswamy CDF',
                   x=apply_kumaraswamy(result_kuma.x, histogram[0]),
                   y=histogram[1],
                   line=dict(color=palette[1])),
        go.Scatter(name='Logistic CDF',
                   x=apply_logistic(logistic, result_logistic.x, histogram[0]),
                   y=histogram[1],
                   line=dict(color=palette[3])),
        go.Scatter(name='NQT Logistic CDF',
                   x=apply_logistic(logistic_nqt, result_nqt.x, histogram[0]),
                   y=histogram[1],
                   line=dict(color=palette[4])),
    ])
    fig.update_traces(mode='lines',
                      line=dict(width=3), marker=dict(size=5, ))
    fig.update_layout(legend_title=None, yaxis_title=None)
    fig.show()


if __name__ == '__main__':
    main()
