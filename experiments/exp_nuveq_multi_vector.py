import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import sys

from nuveq import NonuniformVectorQuantization
from datasets import select_dataset

pio.templates.default = "plotly_white"


def plot_nuveq_multi_vector(data):
    n_bits = 8

    print(f'{n_bits} bits')

    ls_loss_kumaraswamy = []
    ls_loss_logistic = []
    ls_loss_nqt = []

    for i, vector in enumerate(data):
        for nonlinearity, ls in [('logistic', ls_loss_logistic),
                                 ('kumaraswamy', ls_loss_kumaraswamy),
                                 ('NQT', ls_loss_nqt)]:
            model = NonuniformVectorQuantization(n_bits,
                                                 nonlinearity=nonlinearity)
            _, loss_value = model.optimize(vector)

            ls.append(loss_value)

    print('Mean loss with Kumaraswamy nonlinearity',
          np.mean(ls_loss_kumaraswamy))
    print('Mean loss with scaled logistic nonlinearity',
          np.mean(ls_loss_logistic))
    print('Mean loss with NQT nonlinearity',
          np.mean(ls_loss_nqt))

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
    ])
    fig.add_annotation(text='Uniform', x=1, y=1,
                       arrowwidth=2,
                       showarrow=True, arrowhead=3)
    fig.update_xaxes(title_text='Loss with Kumaraswamy nonlinearity')
    fig.update_yaxes(title_text='Loss with Log-Log nonlinearity')
    fig.update_layout(
        font=dict(size=18),
    )
    fig.show()

    fig = go.Figure(data=[
        go.Scatter(name='vectors',
                   x=ls_loss_logistic, y=ls_loss_nqt,
                   mode='markers'),
        go.Scatter(name='iso-line',
                   x=[1, upper_diagonal], y=[1, upper_diagonal], mode='lines',
                   line=dict(dash='dash', color='black')),
    ])
    fig.add_annotation(text='Uniform', x=1, y=1,
                       arrowwidth=2,
                       showarrow=True, arrowhead=3)
    fig.update_xaxes(title_text='Loss with Log-Log nonlinearity')
    fig.update_yaxes(title_text='Loss with NQT nonlinearity')
    fig.update_layout(
        font=dict(size=18),
    )
    fig.show()

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
        font=dict(size=18),
        xaxis=dict(side='top'),
    )
    fig.show()


def main():
    dirname = sys.argv[1]
    dataset = select_dataset(dirname, 'ada002-100k')
    # dataset = select_dataset(dirname, 'openai-v3-small-100k')
    # dataset = select_dataset(dirname, 'gecko-100k')
    # dataset = select_dataset(dirname, 'nv-qa-v4-100k')
    # dataset = select_dataset(dirname, 'colbert-1M')

    data = dataset.X_db

    data -= np.mean(data, axis=0, keepdims=True)

    plot_nuveq_multi_vector(data[:1_000])

if __name__ == '__main__':
    main()
