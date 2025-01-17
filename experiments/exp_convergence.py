import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sys

from datasets import select_dataset
from nuveq import NonuniformVectorQuantization

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None


def plot_convergence_single_vector(vector, n_bits):
    n_seeds = 100

    nonlinearities = ['logistic', 'NQT', 'kumaraswamy']

    records = []

    for nonlinearity in nonlinearities:
        for seed in range(n_seeds):
            model = NonuniformVectorQuantization(
                n_bits,
                optimization_tol=1e-9,
                optimization_max_iter=100,
                optimization_seed=seed,
                nonlinearity=nonlinearity,
                store_optimization_loss_history=True
            )
            params, _ = model.optimize(vector)
            loss_history = model.optimization_loss_history

            records.extend([
                dict(Nonlinearity=nonlinearity,
                     Seed=seed,
                     Iteration=it,
                     Loss=loss_val)
                for it, loss_val in enumerate(loss_history)
            ])

    df = pd.DataFrame.from_records(records)

    palette, _ = plotly.colors.convert_colors_to_same_type(
        px.colors.qualitative.Set1,
        colortype='rgb'
    )

    data_plot = []
    for j, nonlinearity in enumerate(nonlinearities):
        mask = df['Nonlinearity'] == nonlinearity
        df_nl = df.loc[mask].filter(['Iteration', 'Loss'])
        df_mean = df_nl.groupby(['Iteration'], as_index=False).mean()
        df_std = df_nl.groupby(['Iteration'], as_index=False).std()

        name = nonlinearity
        if name != 'NQT':
            name = name.title()

        data_plot.extend([
            go.Scatter(
                name='Upper Bound',
                x=df_mean['Iteration'],
                y=df_mean['Loss'] + df_std['Loss'],
                line=dict(width=0),
                fillcolor=f"rgba{palette[j][3:-1]}, {0.4})",
                showlegend=False,
                mode = 'lines',
            ),
            go.Scatter(
                name='Lower Bound',
                x=df_mean['Iteration'],
                y=df_mean['Loss'] - df_std['Loss'],
                line=dict(width=0),
                fillcolor=f"rgba{palette[j][3:-1]}, {0.4})",
                fill='tonexty',
                showlegend=False,
                mode = 'lines',
            ),
            go.Scatter(
                name=name,
                x=df_mean['Iteration'],
                y=df_mean['Loss'],
                line=dict(color=palette[j]),
                mode='lines',
            )
        ])
    fig = go.Figure(data=data_plot)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        font=dict(size=20),
    )

    # fig.update_layout(title=dict(text=f'{n_bits} bits - {n_dims} dimensions',
    #                              x=0.5, xanchor='center'))
    fig.show()
    # fig.write_image(f'exp_truncnorm_improvement_{n_dims}-dims.pdf')



def plot_convergence_iterations(X, n_bits, generate_from_scratch=True):
    # rng = np.random.default_rng(1)
    #
    # generate_from_scratch = False
    # n_trials = 10_000

    nonlinearities = ['logistic', 'NQT', 'kumaraswamy']

    records = []

    for i, vector in enumerate(X):
        if i % 100 == 0:
            print(i, len(X))

        for nonlinearity in nonlinearities:
            model = NonuniformVectorQuantization(
                n_bits,
                optimization_tol=1e-4,
                nonlinearity=nonlinearity,
                optimization_seed=0,
                store_optimization_loss_history=True
            )
            params, _ = model.optimize(vector)
            loss_history = model.optimization_loss_history

            name = nonlinearity
            if name != 'NQT':
                name = name.title()

            records.append(
                dict(Nonlinearity=name,
                     Idx=i,
                     Iterations=len(loss_history),
                     Loss=loss_history[-1])
            )

    df = pd.DataFrame.from_records(records)

    fig = px.box(df, x='Nonlinearity', y='Iterations', color='Nonlinearity')

    fig.update_layout(
        showlegend=False,
        font=dict(size=20),
        xaxis_title=None,
    )
    fig.show()
    # fig.write_image(f'exp_truncnorm_improvement_{n_dims}-dims.pdf')


def main():
    dirname = sys.argv[1]
    dataset = select_dataset(dirname, 'ada002-100k')
    # dataset = select_dataset(dirname, 'openai-v3-small-100k')
    # dataset = select_dataset(dirname, 'gecko-100k')
    # dataset = select_dataset(dirname, 'nv-qa-v4-100k')
    # dataset = select_dataset(dirname, 'colbert-1M')

    data = dataset.X_db

    data -= np.mean(data, axis=0, keepdims=True)

    generate_from_scratch = True
    # for nonlinearity in ['logistic', 'NQT', 'kumaraswamy']:
    # for nonlinearity in ['NQT']:
    #     plot_convergence(data[:100], nonlinearity, 4,
    #                      generate_from_scratch)

    # plot_convergence_single_vector(data[2], 4)
    # plot_convergence_single_vector(data[2], 8)
    # plot_convergence_single_vector(data[3], 4)
    # plot_convergence_single_vector(data[3], 8)

    plot_convergence_iterations(data[:1_000], 8)


if __name__ == '__main__':
    main()