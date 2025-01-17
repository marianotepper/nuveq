import numpy as np
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.graph_objects as go
import sys

from datasets import select_dataset

pio.templates.default = "plotly_white"


def plot_histograms(X):

    # -----------------------------------
    # fig = px.histogram(vector, nbins=100)

    fig = go.Figure()
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
    fig.update_traces(opacity=0.75)

    fig.update_layout(
        showlegend=False,
        font=dict(size=20),
    )

    fig.show()

    # fig.write_image(f'exp_nuveq_cross_cuts_{nonlinearity}.pdf')


def main():
    dirname = sys.argv[1]

    dataset = select_dataset(dirname, 'gecko-100k')
    ids = [2, 4, 8, 10]

    data = dataset.X_db
    data -= np.mean(data, axis=0, keepdims=True)
    plot_histograms(data[ids])

    dataset = select_dataset(dirname, 'openai-v3-small-100k')
    ids = [2, 4, 6, 8]

    data = dataset.X_db
    data -= np.mean(data, axis=0, keepdims=True)
    plot_histograms(data[ids])


if __name__ == '__main__':
    main()
