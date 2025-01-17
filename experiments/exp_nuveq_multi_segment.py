import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import sys

from nuveq import NonuniformVectorQuantization
from datasets import select_dataset

pio.templates.default = "plotly_white"


def plot_nuveq_multi_vector(data):
    segments = [1, 2, 4, 8]
    n_bits = 8

    print(f'{n_bits} bits')

    ls_loss_kumaraswamy = dict([(s, []) for s in segments])
    ls_loss_logistic = dict([(s, []) for s in segments])
    ls_loss_nqt = dict([(s, []) for s in segments])

    for i, vector in enumerate(data):
        for nonlinearity, ls in [('logistic', ls_loss_logistic),
                                 ('kumaraswamy', ls_loss_kumaraswamy),
                                 ('NQT', ls_loss_nqt)]:
            model = NonuniformVectorQuantization(n_bits,
                                                 nonlinearity=nonlinearity)
            for n_segments in segments:
                step = len(vector) // n_segments
                loss_value_avg = 1
                for j in np.arange(0, len(vector), len(vector) // n_segments):
                    _, loss_value = model.optimize(vector[j:j+step])
                    loss_value_avg *= loss_value
                print(n_segments, np.pow(loss_value_avg, 1. / n_segments))
            # ls.append(loss_value)

    # print('Mean loss with Kumaraswamy nonlinearity',
    #       np.mean(ls_loss_kumaraswamy))
    # print('Mean loss with scaled Logistic nonlinearity',
    #       np.mean(ls_loss_logistic))
    # print('Mean loss with NQT nonlinearity',
    #       np.mean(ls_loss_nqt))

    # fig = go.Figure(data=[
    #     go.Box(name='Kumaraswamy', y=ls_loss_kumaraswamy,
    #            boxpoints='outliers', showlegend=False),
    #     go.Box(name='Log-Log', y=ls_loss_logistic,
    #            boxpoints='outliers', showlegend=False),
    #     go.Box(name='NQT', y=ls_loss_nqt,
    #            boxpoints='outliers', showlegend=False),
    # ])
    # fig.add_hline(y=1, line=dict(dash='dash', color='black'))
    # fig.add_annotation(text='Uniform',
    #                    x=0.5, y=1, xref='paper',
    #                    bgcolor='white',
    #                    showarrow=False)
    # fig.update_layout(
    #     font=dict(size=18),
    #     xaxis=dict(side='top'),
    # )
    # fig.show()


def main():
    dirname = sys.argv[1]
    dataset = select_dataset(dirname, 'ada002-100k')
    # dataset = select_dataset(dirname, 'openai-v3-small-100k')
    # dataset = select_dataset(dirname, 'gecko-100k')
    # dataset = select_dataset(dirname, 'nv-qa-v4-100k')
    # dataset = select_dataset(dirname, 'colbert-1M')

    data = dataset.X_db

    data -= np.mean(data, axis=0, keepdims=True)

    plot_nuveq_multi_vector(data[:1])

if __name__ == '__main__':
    main()
