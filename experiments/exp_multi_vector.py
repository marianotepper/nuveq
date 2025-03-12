from multiprocess import Pool
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import sys

from nuveq import NonuniformVectorQuantization
from datasets import select_dataset
from plot_utils import write_image

pio.templates.default = "plotly_white"


def plot_nuveq_multi_vector(dirname, dataset_name, n_samples=10_000):
    dataset = select_dataset(dirname, dataset_name)

    data = dataset.X_db
    data -= np.mean(data, axis=0, keepdims=True)
    data = data[:n_samples]

    n_bits = 8

    print(f'{n_bits} bits')

    ls_loss_kumaraswamy = []
    ls_loss_logistic = []
    ls_loss_nqt = []

    for nonlinearity, ls in [('logistic', ls_loss_logistic),
                             ('kumaraswamy', ls_loss_kumaraswamy),
                             ('NQT', ls_loss_nqt)]:
        model = NonuniformVectorQuantization(n_bits,
                                             nonlinearity=nonlinearity)

        def run_single_vector(vector):
            _, loss_value = model.optimize(vector)
            return loss_value

        with Pool() as p:
            ls_local = p.map(run_single_vector, data)

        ls.extend(ls_local)
        print(f'Mean loss with {nonlinearity} nonlinearity',
              np.mean(ls))

    upper_diagonal = 1.1 * max([np.max(ls_loss_kumaraswamy),
                                np.max(ls_loss_logistic),
                                np.max(ls_loss_nqt)])

    fig = go.Figure(data=[
        go.Scatter(name='vectors',
                   x=ls_loss_kumaraswamy, y=ls_loss_logistic,
                   mode='markers'),
        go.Scatter(name='iso-line',
                   x=[1, upper_diagonal], y=[1, upper_diagonal], mode='lines',
                   line=dict(dash='dash', color='black')),
        go.Scatter(name='Uniform',
                   x=[1], y=[1],
                   marker=dict(size=[15], symbol='circle-dot',
                               line_width=3, line_color='#e41a1c',
                               color='white'),
                   mode='markers'),
    ])
    fig.update_xaxes(title_text='Loss with Kumaraswamy')
    fig.update_yaxes(title_text='Loss with Log-Log')
    fig.update_layout(
        legend=dict(x=0.5, y=1.02, orientation="h",
                    yanchor="bottom", xanchor="center"),
        font=dict(size=25),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, f'multi_vector_kuma_vs_logistic_{dataset.name}.pdf')

    fig = go.Figure(data=[
        go.Scatter(name='vectors',
                   x=ls_loss_logistic, y=ls_loss_nqt,
                   mode='markers'),
        go.Scatter(name='iso-line',
                   x=[1, upper_diagonal], y=[1, upper_diagonal], mode='lines',
                   line=dict(dash='dash', width=3, color='black')),
        go.Scatter(name='Uniform',
                   x=[1], y=[1],
                   marker=dict(size=[15], symbol='circle-dot',
                               line_width=3, line_color='#e41a1c',
                               color='white'),
                   mode='markers'),
    ])
    fig.update_xaxes(title_text='Loss with Log-Log')
    fig.update_yaxes(title_text='Loss with NQT')
    fig.update_layout(
        legend=dict(x=0.5, y=1.02, orientation="h",
                    yanchor="bottom", xanchor="center"),
        font=dict(size=25),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, f'multi_vector_logistic_vs_nqt_{dataset.name}.pdf')

    fig = go.Figure(data=[
        go.Box(name='Kumaraswamy', y=ls_loss_kumaraswamy,
               boxpoints='outliers', showlegend=False),
        go.Box(name='Log-Log', y=ls_loss_logistic,
               boxpoints='outliers', showlegend=False),
        go.Box(name='NQT', y=ls_loss_nqt,
               boxpoints='outliers', showlegend=False),
    ])
    fig.add_hline(y=1, line=dict(dash='dash', color='black'))
    fig.add_annotation(text='Uniform',
                       x=0.5, y=1, xref='paper',
                       bgcolor='white',
                       showarrow=False)
    fig.update_layout(
        font=dict(size=25),
        xaxis=dict(side='top'),
        yaxis_title='Loss',
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, f'multi_vector_improvement_{dataset.name}.pdf')


def main():
    dirname = sys.argv[1]

    plot_nuveq_multi_vector(dirname, 'ada002-100k',
                            n_samples=10_000)


if __name__ == '__main__':
    main()
