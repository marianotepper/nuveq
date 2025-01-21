from multiprocess import Pool
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys

from nuveq import NonuniformVectorQuantization
from datasets import select_dataset

pio.templates.default = "plotly_white"


def plot_nuveq_solutions(data, nonlinearity, plot_title, xaxes_title, yaxes_title):
    n_bits = 8

    print(f'{nonlinearity} @ {n_bits} bits')

    model = NonuniformVectorQuantization(n_bits,
                                         nonlinearity=nonlinearity)

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
        font=dict(size=18),
    )
    fig.show()


def main():
    dirname = sys.argv[1]
    # dataset = select_dataset(dirname, 'ada002-100k')
    # dataset = select_dataset(dirname, 'openai-v3-small-100k')
    dataset = select_dataset(dirname, 'gecko-100k')
    # dataset = select_dataset(dirname, 'nv-qa-v4-100k')
    # dataset = select_dataset(dirname, 'colbert-1M')

    data = dataset.X_db

    data -= np.mean(data, axis=0, keepdims=True)

    for nonlinearity, plot_title, xaxes_title, yaxes_title in [
        ('logistic', 'Log-Log', r'$\alpha$', r'$x_0$'),
        ('kumaraswamy', 'Kumaraswamy', r'$a$', r'$b$'),
        ('NQT', 'NQT', r'$\alpha$', r'$x_0$')
    ]:
        plot_nuveq_solutions(data[:10_000], nonlinearity, plot_title,
                             xaxes_title, yaxes_title)

if __name__ == '__main__':
    main()
