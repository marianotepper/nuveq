from multiprocess import Pool
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

from accuracy_metrics import compute_recall
from datasets import select_dataset
from experiments.accuracy_metrics import compute_map
from nuveq import NonuniformVectorQuantization


def plot_nuveq_multi_vector(dataset, n_bits):
    X_query, X_db, idx_gt = dataset.X_query, dataset.X_db, dataset.gt

    nonlinearities = [('logistic', 'Log-Log'),
                      ('kumaraswamy', 'Kumaraswamy'),
                      ('NQT', 'NQT')]

    mean = np.mean(X_db, axis=0, keepdims=True)

    ls_n_subvectors = [1, 2, 4, 8]

    records = []

    for nonlinearity, _ in nonlinearities:
        for ns in ls_n_subvectors:
            model = NonuniformVectorQuantization(n_bits=n_bits,
                                                 n_subvectors=ns,
                                                 nonlinearity=nonlinearity,
                                                 optimization_seed=0)

            def run_single_vector(vector):
                sol, _ = model.optimize(vector)
                return model.quantize(vector, sol)


            filename = (f'./exp_nuveq_recall_{dataset.name}'
                        f'_{nonlinearity}_bits-{n_bits}_subvectors-{ns}.npz')
            try:
                npzfile = np.load(filename)
                rec_error = npzfile['rec_error']
                dp_error = npzfile['dp_error']
                idx = npzfile['idx']
            except OSError:
                with Pool() as p:
                    X_quantized = p.map(run_single_vector, X_db - mean)

                X_quantized = np.vstack(X_quantized)
                X_quantized += mean

                dot_products_gt = X_query @ X_db.T
                dot_products_quant = X_query @ X_quantized.T

                rec_error = np.sum((X_db - X_quantized) ** 2)
                rec_error /= np.prod(X_db.shape)
                dp_error = np.sum((dot_products_gt - dot_products_quant) ** 2)
                dp_error /= np.prod(dot_products_gt.shape)

                idx = np.argsort(dot_products_quant, axis=1)[:, ::-1]
                idx = idx[:, :200]

                np.savez(
                    filename,
                    rec_error=rec_error,
                    dp_error=dp_error,
                    idx=idx
                )


            print(nonlinearity, ns, rec_error, dp_error)

            for k, at in [(1, 1), (2, 2), (10, 10)]:
                print(f'{k}-recall@{at} = '
                      f'{compute_recall(idx_gt, idx, k=k, at=at): .4f}')

            print(f'MAP@{10} = {compute_map(idx_gt, idx, k=10): .4f}')

            records.append(
                dict(nonlinearity=nonlinearity,
                     n_segments=ns,
                     reconstruction_error=rec_error,
                     dot_product_error=dp_error,
                     map=compute_map(idx_gt, idx, k=10))
            )

    df = pd.DataFrame.from_records(records)

    palette = px.colors.qualitative.Set1

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Reconstruction MSE',  'Query dot product MSE', 'MAP'),
    )

    for i, (nonlinearity, plot_name) in enumerate(nonlinearities):
        mask = df['nonlinearity'] == nonlinearity

        fig.add_trace(
            go.Scatter(name=plot_name,
                       x=df.loc[mask]['n_segments'],
                       y=df.loc[mask]['reconstruction_error'],
                       line=dict(dash='solid', color=palette[i]),
                       mode='markers+lines'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(name=plot_name,
                       x=df.loc[mask]['n_segments'],
                       y=df.loc[mask]['dot_product_error'],
                       line=dict(dash='solid', color=palette[i]),
                       mode='markers+lines',
                       showlegend=False),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(name=plot_name,
                       x=df.loc[mask]['n_segments'],
                       y=df.loc[mask]['map'],
                       line=dict(dash='solid', color=palette[i]),
                       mode='markers+lines',
                       showlegend=False),
            row=1, col=3
        )

    x_range = [min(ls_n_subvectors) - 0.1, max(ls_n_subvectors) + 0.1]
    for i in range(1, 4):
        fig['layout']['xaxis{}'.format(i)]['title'] = 'Number of subvectors'
        fig['layout']['xaxis{}'.format(i)]['range'] = x_range
        fig['layout']['xaxis{}'.format(i)]['dtick'] = 1
        fig['layout']['yaxis{}'.format(i)]['showexponent'] = 'all'
        fig['layout']['yaxis{}'.format(i)]['exponentformat'] = 'e'

    fig.update_annotations(font_size=20)
    fig.update_layout(
        template='plotly_white',
        font=dict(size=20),
        legend=dict(
            yanchor="bottom",
            y=0.05,
            xanchor="right",
            x=1),
        autosize=False,
        width=1300,
        height=300,
        margin={'l': 0, 'r': 0, 't': 60, 'b': 0},
    )
    fig.show()
    fig.write_image(f'recall_{dataset.name}.pdf')


def main():
    dirname = sys.argv[1]
    dataset = select_dataset(dirname, 'ada002-100k')
    # dataset = select_dataset(dirname, 'openai-v3-small-100k')
    # dataset = select_dataset(dirname, 'gecko-100k')
    # dataset = select_dataset(dirname, 'nv-qa-v4-100k')
    # dataset = select_dataset(dirname, 'colbert-1M')
    print(dataset.name)

    plot_nuveq_multi_vector(dataset, 4)
    # plot_nuveq_multi_vector(dataset, 8)

if __name__ == '__main__':
    main()
