import numpy as np
import plotly.figure_factory as ff
import plotly.io as pio
import sys

from datasets import select_dataset
from plot_utils import write_image

pio.templates.default = "plotly_white"


def plot_histograms(dirname, dataset_name, ids):
    dataset = select_dataset(dataset_name, dirname=dirname)

    X = dataset.X_db
    X -= np.mean(X, axis=0, keepdims=True)
    X = X[ids]

    # -----------------------------------
    # fig = px.histogram(vector, nbins=100)

    hist_data = [vector for vector in X]
    group_labels = [f'Vector {i+1}' for i in range(len(X))]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels,
                             show_hist=False, show_rug=False)
    # for vector in X:
    #     fig.add_trace(go.Histogram(x=vector))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75, line=dict(width=5))

    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=-0.05,
            dtick=0.05
        ),
        yaxis = dict(
            tickmode='linear',
            tick0=0,
            dtick=5
        ),
        showlegend=False,
        font=dict(size=25),
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )

    fig.show()

    write_image(fig, f'histograms_{dataset_name}.pdf')


def main():
    if len(sys.argv) == 2:
        dirname = sys.argv[1]
    else:
        dirname = './wikipedia_squad'

    plot_histograms(dirname, 'gecko-100k', [2, 4, 8, 10])
    plot_histograms(dirname, 'openai-v3-small-100k', [2, 4, 6, 8])


if __name__ == '__main__':
    main()
