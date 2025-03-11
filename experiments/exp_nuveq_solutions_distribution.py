from multiprocess import Pool
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys

from nuveq import NonuniformVectorQuantization
from datasets import select_dataset
from plot_utils import write_image

pio.templates.default = "plotly_white"


def plot_nuveq_solutions(dirname, dataset_name, nonlinearity, plot_title,
                         xaxes_title, yaxes_title, n_samples=10_000):
    dataset = select_dataset(dirname, dataset_name)
    data = dataset.X_db
    data -= np.mean(data, axis=0, keepdims=True)
    data = data[:n_samples]

    n_bits = 8

    print(f'{nonlinearity} @ {n_bits} bits')

    model = NonuniformVectorQuantization(n_bits, nonlinearity=nonlinearity)

    def run_single_vector(vector):
        sol, _ = model.optimize(vector)
        return {xaxes_title: sol.distribution_params[0],
                yaxes_title: sol.distribution_params[1]}

    with Pool() as p:
        records_solutions = p.map(run_single_vector, data)

    df = pd.DataFrame(records_solutions)

    fig = px.scatter(df, x=xaxes_title, y=yaxes_title,
                     marginal_x='histogram',
                     marginal_y='histogram')
    fig.add_vline(x=df[xaxes_title].mean(),
                  row=2, col=1)
    fig.add_hline(y=df[yaxes_title].mean(),
                  row=1, col=2)

    fig.update_layout(
        title=dict(text=plot_title, x=0.5, xanchor='center'),
        font=dict(size=20),
        margin={'l': 0, 'r': 0, 't': 50, 'b': 60},
        height=600,
        width=600,
    )
    fig.show()

    write_image(fig, f'solutions_distribution_{dataset_name}_{nonlinearity}.png',
                scale=3)


def main():
    dirname = sys.argv[1]

    for dataset_name in ['gecko-100k', 'ada002-100k']:
        for nonlinearity, plot_title, xaxes_title, yaxes_title in [
            ('logistic', 'Log-Log', r'$\LARGE{\alpha}$', r'$\LARGE{x_0}$'),
            ('kumaraswamy', 'Kumaraswamy', r'$\LARGE{a}$', r'$\LARGE{b}$'),
            ('NQT', 'NQT', r'$\LARGE{\alpha}$', r'$\LARGE{x_0}$')
        ]:
            plot_nuveq_solutions(dirname, dataset_name, nonlinearity,
                                 plot_title, xaxes_title, yaxes_title,
                                 n_samples=10_000)

if __name__ == '__main__':
    main()
