from functools import partial
import numpy as np
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy.optimize
import scipy.stats

import vector_equalization as veq
import vecs_io

pio.templates.default = "plotly_white"

n_dims = 1024
rng = np.random.default_rng(0)
# all_data = rng.normal(size=(100, n_dims))
# all_data = rng.beta(a=0.5, b=0.5, size=(10, n_dims))
# all_data = rng.beta(a=2, b=2, size=(100, n_dims))

dirname = '/Users/mariano.tepper/IdeaProjects/jvector/fvec/wikipedia_squad/100k/'
# filename = 'text-embedding-3-large_1536_100000_base_vectors.fvec'
filename = 'ada_002_100000_base_vectors.fvec'
all_data = reader.fvecs_read(dirname + filename)

all_data -= np.mean(all_data, axis=0)
data = all_data[1]

histogram = np.sort(data), np.arange(0, len(data)) / (len(data) - 1)

result_kuma = scipy.optimize.minimize(
    partial(veq.loss_kumaraswamy, histogram),
    np.ones((3,)),
    bounds=[(1e-6, np.inf)] * 3
)
print('loss Kumaraswamy CDF fitting', result_kuma.fun)

result_triple_kuma = scipy.optimize.minimize(
    partial(veq.loss_triple_kumaraswamy, histogram),
    np.ones((3,)),
    bounds=[(1e-6, np.inf)] * 3
)
print('loss Triple Kumaraswamy CDF fitting', result_triple_kuma.fun)

result_logistic = scipy.optimize.minimize(
    partial(veq.loss_logistic, veq.logistic, histogram),
    np.array([30, 0.5]),
    bounds=[(1e-6, np.inf), (-np.inf, np.inf)]
)
print('loss Logistic CDF fitting', result_logistic.fun)
print(result_logistic.x)

result_nqt = scipy.optimize.minimize(
    partial(veq.loss_logistic, veq.logistic_nqt, histogram),
    np.array([30, 0.5]),
    bounds=[(1e-6, np.inf),
            (-np.inf, np.inf)]
)
print('loss NQT Logistic CDF fitting', result_nqt.fun)
print(result_nqt.x)

palette = plotly.colors.qualitative.Plotly

lspace = np.linspace(data.min(), data.max(), num=10_000, endpoint=True)

fig = go.Figure(data=[
    go.Scatter(name='Empirical CDF',
               x=histogram[0], y=histogram[1],
               line=dict(color=palette[0])),
    go.Scatter(name='Kumaraswamy CDF',
               x=lspace, y=veq.apply_kumaraswamy(result_kuma.x, lspace),
               line=dict(color=palette[1])),
    go.Scatter(name='Triple Kumaraswamy CDF',
               x=lspace, y=veq.apply_triple_kumaraswamy(result_triple_kuma.x, lspace),
               line=dict(color=palette[2])),
    go.Scatter(name='Logistic CDF',
               x=lspace, y=veq.apply_logistic(veq.logistic, result_logistic.x, lspace),
               line=dict(color=palette[3])),
    go.Scatter(name='NQT Logistic CDF',
               x=lspace, y=veq.apply_logistic(veq.logistic_nqt, result_nqt.x, lspace),
               line=dict(color=palette[4])),
])
fig.add_vline(x=np.median(histogram[0]))
fig.add_hline(y=0.5)
fig.update_traces(mode='lines',
                  line=dict(width=3), marker=dict(size=5, ))
fig.update_layout(legend_title=None, yaxis_title=None)
fig.show()

fig = go.Figure(data=[
    go.Box(name='Kumaraswamy CDF',
           y=np.abs(veq.apply_kumaraswamy(result_kuma.x, histogram[0]) - histogram[1]),
           boxpoints='outliers', marker_color=palette[1]),
    go.Box(name='Triple Kumaraswamy CDF',
           y=np.abs(
               veq.apply_triple_kumaraswamy(result_triple_kuma.x, histogram[0]) - histogram[1]),
           boxpoints='outliers', marker_color=palette[2]),
    go.Box(name='Logistic CDF',
           y=np.abs(veq.apply_logistic(veq.logistic, result_logistic.x, histogram[0]) - histogram[1]),
           boxpoints='outliers', marker_color=palette[3]),
    go.Box(name='NQT Logistic CDF',
           y=np.abs(veq.apply_logistic(veq.logistic_nqt, result_nqt.x, histogram[0]) - histogram[1]),
           boxpoints='outliers', marker_color=palette[4]),
])
fig.update_layout(legend_title=None, yaxis_title=None)
fig.show()

fig = go.Figure(data=[
    go.Scatter(name='Kumaraswamy CDF',
               x=veq.apply_kumaraswamy(result_kuma.x, histogram[0]), y=histogram[1],
               line=dict(color=palette[1])),
    go.Scatter(name='Triple Kumaraswamy CDF',
               x=veq.apply_triple_kumaraswamy(result_triple_kuma.x, histogram[0]), y=histogram[1],
               line=dict(color=palette[2])),
    go.Scatter(name='Logistic CDF',
               x=veq.apply_logistic(veq.logistic, result_logistic.x, histogram[0]), y=histogram[1],
               line=dict(color=palette[3])),
    go.Scatter(name='NQT Logistic CDF',
               x=veq.apply_logistic(veq.logistic_nqt, result_nqt.x, histogram[0]), y=histogram[1],
               line=dict(color=palette[4])),
])
fig.update_traces(mode='lines',
                  line=dict(width=3), marker=dict(size=5, ))
fig.update_layout(legend_title=None, yaxis_title=None)
fig.show()

data_subset = all_data[:10]
data_uniformized = []
loss_fun_mean = 0

for datum in data_subset:
    histogram = np.sort(datum), np.arange(len(datum)) / (len(datum) - 1)

    # result = scipy.optimize.minimize(
    #     partial(veq.loss_triple_kumaraswamy, histogram),
    #     np.ones((3,)),
    #     bounds=[(1e-6, np.inf)] * 3
    # )
    # datum_uniformized = veq.apply_triple_kumaraswamy(result.x, datum)

    result = scipy.optimize.minimize(
        partial(veq.loss_logistic, veq.logistic_nqt, histogram),
        np.array([20, 0]),
        bounds=[(1e-6, np.inf), (-np.inf, np.inf)]
    )
    datum_uniformized = veq.apply_logistic(veq.logistic_nqt, result.x, datum)

    # result = scipy.optimize.minimize(
    #     partial(veq.loss_logistic, veq.logistic, histogram),
    #     np.array([20, 0]),
    #     bounds=[(1e-6, np.inf), (-np.inf, np.inf)]
    # )
    # datum_uniformized = veq.apply_logistic(veq.logistic, result.x, datum)

    data_uniformized.append(datum_uniformized)

    print(result.x, result.fun)
    loss_fun_mean += result.fun

loss_fun_mean /= len(data_subset)
print(loss_fun_mean)


data_uniformized = np.vstack(data_uniformized)
# data_uniformized = data_subset
rng = np.random.default_rng()
rng.shuffle(data_uniformized, axis=1)
data_uniformized = data_uniformized.reshape((-1, 2))

bins = dict(start=0, end=1, size=0.05)

fig = make_subplots(rows=2, cols=2,
                    row_heights=[0.15, 0.85],
                    column_widths=[0.85, 0.15],
                    shared_xaxes=True, shared_yaxes=True,
                    horizontal_spacing=0.02, vertical_spacing=0.02)
fig.add_trace(
    go.Histogram2d(x=data_uniformized[:, 0], y=data_uniformized[:, 1],
                   xbins=bins,
                   ybins=bins,
                   showlegend=False),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(name='', mode='markers',
               x=data_uniformized[:, 0], y=data_uniformized[:, 1],
               marker=dict(color='silver', size=3),
               showlegend=False),
    row=2, col=1,
)
fig.add_trace(
    go.Histogram(x=data_uniformized[:, 0],
                 xbins=bins,
                 marker=dict(color='#386cb0'),
                 showlegend=False),
    row=1, col=1
)
fig.add_trace(
    go.Histogram(y=data_uniformized[:, 1], orientation='h',
                 ybins=bins,
                 marker=dict(color='#386cb0'),
                 showlegend=False),
    row=2, col=2
)
fig.update_layout(
    width=800,
    height=800
)
# fig.update_traces(mode='lines',
#                   line=dict(width=3), marker=dict(size=5, ))
# fig.update_layout(legend_title=None, yaxis_title=None)
fig.show()

plotbins = np.arange(start=bins['start'], stop=bins['end']+bins['size'], step=bins['size'])
hist2d, x_eges, y_edges = np.histogram2d(data_uniformized[:, 0], data_uniformized[:, 1], bins=[plotbins, plotbins])
hist2d_mean = float(np.mean(hist2d))
hist2d_median = float(np.median(hist2d))
if hist2d_median > hist2d_mean:
    median_plot_position = 'top right'
    median_plot_shift = 5
    mean_plot_position = 'top left'
    mean_plot_shift = -5
else:
    median_plot_position = 'top left'
    median_plot_shift = -5
    mean_plot_position = 'top right'
    mean_plot_shift = 5

fig = px.histogram(x=hist2d.flatten(), marginal='box')
fig.add_vline(x=hist2d_median,
              line=dict(width=2, dash='dot'),
              annotation_text='Median count',
              annotation_position=median_plot_position,
              annotation_xshift=median_plot_shift,
              row=1, col=1)
fig.add_vline(x=hist2d_median,
              line=dict(width=2, dash='dot'),
              row=2, col=1)
fig.add_vline(x=hist2d_mean,
              line=dict(width=2, dash='dash'),
              annotation_text='Mean count',
              annotation_position=mean_plot_position,
              annotation_xshift=mean_plot_shift,
              row=1, col=1)
fig.add_vline(x=hist2d_mean,
              line=dict(width=2, dash='dash'),
              row=2, col=1)

fig.update_layout(
    xaxis=dict(
        title=dict(
            text="Number of points per cluster"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Number of clusters"
        )
    ),
    font=dict(
        size=18,
    )
)
fig.show()