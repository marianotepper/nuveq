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


def plot_nuveq_subvectors(dirname, dataset_name, n_bits, n_samples=10_000):
    dataset = select_dataset(dataset_name, dirname=dirname)
    print(dataset.name)

    data = dataset.X_db
    data -= np.mean(data, axis=0, keepdims=True)
    data = data[:n_samples]

    ls_n_subvectors = [1, 2, 4, 8]

    filename = f'./exp_nuveq_subvectors_{n_bits}bits.pickle'
    try:
        df = pd.read_pickle(filename)
    except FileNotFoundError:
        records = []

        for nonlinearity, box_name in [('logistic', 'Log-Log'),
                                       ('kumaraswamy', 'Kumaraswamy'),
                                       ('NQT', 'NQT')]:
            for ns in ls_n_subvectors:
                model = NonuniformVectorQuantization(n_bits=n_bits,
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

    fig = px.box(df, x='Nonlinearity', y='Loss', color='Subvectors')
    fig.add_hline(y=1, line=dict(dash='dash', color='black'))
    fig.add_annotation(text='Uniform',
                       x=0.5, y=1, xref='paper',
                       bgcolor='white',
                       showarrow=False)
    fig.update_layout(
        xaxis_title=None,
        font=dict(size=20),
        xaxis=dict(side='top'),
        width=1300,
        height=300,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    fig.show()
    write_image(fig, f'subvectors_loss_{dataset.name}_{n_bits}bits.pdf')


def main():
    if len(sys.argv) == 2:
        dirname = sys.argv[1]
    else:
        dirname = './wikipedia_squad'

    dataset_name = 'ada002-100k'
    plot_nuveq_subvectors(dirname, dataset_name, 4, n_samples=10_000)
    plot_nuveq_subvectors(dirname, dataset_name, 8, n_samples=10_000)

if __name__ == '__main__':
    main()
