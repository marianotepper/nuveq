import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy.optimize
import scipy.stats

from nuveq import NonuniformVectorQuantization, NVQParams, forward_kumaraswamy, inverse_kumaraswamy, quantize, normalize, denormalize
import vecs_io

pio.templates.default = "plotly_white"

n_bits = 4

# rng = np.random.default_rng(0)
# loc = 0
# scale = 0.3
# min_trunc, max_trunc = (data_min - loc) / scale, (data_max - loc) / scale
# truncnorm = scipy.stats.truncnorm(min_trunc, max_trunc, loc=loc, scale=scale)
# data = truncnorm.rvs(n_dims, random_state=rng)
# data = np.sort(data)

# n_dims = 1024
# rng = np.random.default_rng(0)
# data = rng.normal(size=n_dims)

dirname = '/Users/mariano.tepper/IdeaProjects/jvector/fvec/wikipedia_squad/100k/'
filename = 'text-embedding-3-large_1536_100000_base_vectors.fvec'
# filename = 'ada_002_100000_base_vectors.fvec'
all_data = reader.fvecs_read(dirname + filename)
data = all_data[0] - np.mean(all_data, axis=0)

data_min = data.min()
data_max = data.max()
data = np.sort(data)
histogram = data, np.arange(0, len(data)) / (len(data) - 1)


def loss_cdf(x):
    diff = forward_kumaraswamy(normalize(histogram[0], data_min, data_max), x[0], x[1]) - histogram[1]
    return (diff ** 2).sum()


model = NonuniformVectorQuantization(n_bits, nonlinearity='kumaraswamy')
params, loss_value = model.optimize(data)
print('Optimization loss', loss_value)
a_ae, b_ae = params.distribution_params

loss_fun = lambda sol: model.loss(
    data[:, np.newaxis, np.newaxis],
    NVQParams(params.x_min, params.x_max, sol)
)


result = scipy.optimize.minimize(loss_cdf, np.ones((2,)), bounds=[(1e-6, np.inf)] * 2)
a_cdf, b_cdf = result.x
print('loss CDF fitting', loss_fun([a_cdf, b_cdf]))


lspace = np.linspace(data_min, data_max, num=1000, endpoint=True)
lspace_norm = normalize(lspace, data_min, data_max)

palette = plotly.colors.qualitative.D3

data_norm = normalize(data, data_min, data_max)
error_kuma_cdf = np.abs(data - denormalize(inverse_kumaraswamy(quantize(forward_kumaraswamy(data_norm, a_cdf, b_cdf), n_bits), a_cdf, b_cdf), data_min, data_max))
error_kuma_ae = np.abs(data - denormalize(inverse_kumaraswamy(quantize(forward_kumaraswamy(data_norm, a_ae, b_ae), n_bits), a_ae, b_ae), data_min, data_max))
error_uniform = np.abs(data - denormalize(quantize(data_norm, n_bits), data_min, data_max))
print(np.sum(error_kuma_cdf ** 2), np.sum(error_kuma_ae ** 2), np.sum(error_uniform ** 2))

def hex_to_rgba(h, alpha):
    """
    converts color value in hex format to rgba format with alpha transparency
    """
    return 'rgba' + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(
        name='Empirical CDF',
        x=histogram[0],
        y=histogram[1],
        line=dict(color=palette[0]),
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        name='Kumaraswamy - fitted CDF',
        x=lspace,
        y=quantize(forward_kumaraswamy(lspace_norm, a_cdf, b_cdf), n_bits),
        line=dict(color=palette[1]),
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        name='Kumaraswamy AE',
        x=lspace,
        y=quantize(forward_kumaraswamy(lspace_norm, a_ae, b_ae), n_bits),
        line=dict(color=palette[2]),
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        name='Uniform',
        x=lspace,
        y=quantize(lspace_norm, n_bits),
        line=dict(color=palette[3]),
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(name='Kumaraswamy - fitted CDF',
               x=data, y=error_kuma_cdf,
               mode="markers+lines", opacity=0.4,
               line=dict(color=palette[1]),
               showlegend=False,),
    row=1, col=2
)
fig.add_hline(y=error_kuma_cdf.mean(),
              line=dict(color=palette[1], width=4),
              row=1, col=2)
fig.add_trace(
    go.Scatter(name='Kumaraswamy AE',
               x=data, y=error_kuma_ae,
               mode="markers+lines", opacity=0.4,
               line=dict(color=palette[2]),
               showlegend=False,),
    row=1, col=2
)
fig.add_hline(y=error_kuma_ae.mean(),
              line=dict(color=palette[2], width=4),
              row=1, col=2)
fig.add_trace(
    go.Scatter(name='Quantized Uniform CDF',
               x=data, y=error_uniform,
               mode="markers+lines", opacity=0.4,
               line=dict(color=palette[3]),
               showlegend=False,),
    row=1, col=2
)
fig.add_hline(y=error_uniform.mean(),
              line=dict(color=palette[3], width=4),
              row=1, col=2)


# fig.add_trace(
#     go.Box(
#         name='Kumaraswamy - fitted CDF',
#         y=error_kuma_cdf,
#         marker_color=palette[1],
#         showlegend=False,
#     ),
#     row=1, col=3
# )
# fig.add_trace(
#     go.Box(
#         name='Kumaraswamy AE',
#         y=error_kuma_ae,
#         marker_color=palette[2],
#         showlegend=False,
#     ),
#     row=1, col=3
# )
# fig.add_trace(
#     go.Box(
#         name='Quantized Uniform CDF',
#         y=error_uniform,
#         marker_color=palette[3],
#         showlegend=False,
#     ),
#     row=1, col=3
# )

fig.update_traces(mode='lines',
                  line=dict(width=3), marker=dict(size=5, ),
                  row=1, col=1)
fig.update_traces(mode='lines',
                  line=dict(width=3), marker=dict(size=6, ),
                  row=1, col=2)
fig.update_xaxes(showticklabels=False,
                 row=1, col=3)
# fig.update_traces(boxmean='sd',
#                   row=1, col=3)

fig.update_layout(
    legend=dict(orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5)
)

fig.show()
fig.write_image('exp_truncnorm_error.pdf')

