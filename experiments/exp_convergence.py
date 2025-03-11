import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sys

from datasets import select_dataset
from nuveq import NonuniformVectorQuantization
from plot_utils import write_image

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None


def plot_convergence_single_vector(dirname, dataset_name, n_bits, id):
    dataset = select_dataset(dirname, dataset_name)
    data = dataset.X_db
    data -= np.mean(data, axis=0, keepdims=True)
    vector = data[id]

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
            x=0.5),
        xaxis_title='Iterations',
        font=dict(size=20),
    )

    fig.show()
    write_image(fig, f'convergence_improvement_{n_bits}-bits.pdf')



def plot_convergence_iterations(dirname, dataset_name, n_bits, n_samples=1000):
    dataset = select_dataset(dirname, dataset_name)
    X = dataset.X_db
    X -= np.mean(X, axis=0, keepdims=True)
    X = X[:n_samples]

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

    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        xaxis_title=None,
        showlegend=False,
        font=dict(size=20),
    )
    fig.show()
    write_image(fig, f'convergence_iterations_{dataset_name}.pdf')


def main():
    dirname = sys.argv[1]

    plot_convergence_single_vector(dirname, 'ada002-100k', 4, id=2)
    plot_convergence_single_vector(dirname, 'ada002-100k', 8, id=2)
    plot_convergence_iterations(dirname, 'ada002-100k', 8, n_samples=1000)


if __name__ == '__main__':
    main()