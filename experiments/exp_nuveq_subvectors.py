from multiprocess import Pool
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys

from nuveq import NonuniformVectorQuantization
from datasets import select_dataset

pio.templates.default = "plotly_white"


def plot_nuveq_subvectors(data):
    ls_n_subvectors = [1, 2, 4, 8]

    filename = f'./exp_nuveq_subvectors.pickle'
    try:
        df = pd.read_pickle(filename)
    except FileNotFoundError:
        records = []

        for nonlinearity, box_name in [('logistic', 'Log-Log'),
                                       ('kumaraswamy', 'Kumaraswamy'),
                                       ('NQT', 'NQT')]:
            for ns in ls_n_subvectors:
                model = NonuniformVectorQuantization(n_bits=8,
                                                     n_subvectors=ns,
                                                     optimization_seed=0,
                                                     nonlinearity=nonlinearity)

                def run_single_vector(vector):
                    _, loss_value = model.optimize(vector)
                    return dict(Nonlinearity=box_name,
                                Subvectors=ns,
                                Loss=loss_value)

                with Pool() as p:
                    records_local = p.map(run_single_vector, data)

                records.extend(records_local)

        df = pd.DataFrame.from_records(records)
        df.to_pickle(filename)

    print(df.groupby(['Nonlinearity', 'Subvectors'], as_index=False).mean())

    fig = px.box(df, x='Nonlinearity', y='Loss', color='Subvectors',
                 points=False)
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
    print(dataset.name)

    data = dataset.X_db

    data -= np.mean(data, axis=0, keepdims=True)

    plot_nuveq_subvectors(data)

if __name__ == '__main__':
    main()
