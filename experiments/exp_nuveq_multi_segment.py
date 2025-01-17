import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import sys

from nuveq import NonuniformVectorQuantization
from datasets import select_dataset

pio.templates.default = "plotly_white"


def plot_nuveq_multi_vector(data, n_bits):
    ls_n_subvectors = [1, 2, 4, 8]

    print(f'{n_bits} bits')

    records = []

    for nonlinearity, box_name in [('logistic', 'Log-Log'),
                                   ('kumaraswamy', 'Kumaraswamy'),
                                   ('NQT', 'NQT')]:
        for ns in ls_n_subvectors:
            model = NonuniformVectorQuantization(n_bits,
                                                 n_subvectors=ns,
                                                 nonlinearity=nonlinearity)

            for i, vector in enumerate(data):
                _, loss_value = model.optimize(vector)

                records.append(
                    dict(Nonlinearity=box_name,
                         Subvectors=ns,
                         Loss=loss_value)
                )

    df = pd.DataFrame.from_records(records)

    fig = px.box(df, x='Nonlinearity', y='Loss', color='Subvectors')
    fig.add_hline(y=1, line=dict(dash='dash', color='black'))
    fig.add_annotation(text='Uniform',
                       x=0.5, y=1, xref='paper',
                       bgcolor='white',
                       showarrow=False)
    fig.update_layout(
        font=dict(size=20),
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

    plot_nuveq_multi_vector(data[:100], 4)
    plot_nuveq_multi_vector(data[:100], 8)

if __name__ == '__main__':
    main()
