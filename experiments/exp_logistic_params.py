import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from nuveq import normalized_logistic, normalized_logit, logistic, logit
from plot_utils import write_image

pio.templates.default = "plotly_white"

x = np.linspace(-1, 1, 1_000, endpoint=True)
y = np.linspace(0, 1, 1_000, endpoint=True)

palette = px.colors.qualitative.Plotly

fig = make_subplots(rows=1, cols=2, subplot_titles=('Logistic',  'Logit'))

for i, (alpha, x0, dash) in enumerate([(1, 0, 'solid'),
                                       (7.5, 0, 'dot'), (15, 0, 'dashdot'),
                                       (10, -0.3, 'dot'), (10, 0.3, 'dashdot')]):
    bias = logistic(x.min(), alpha, x0)
    scale = logistic(x.max(), alpha, x0) - bias

    str_alpha = f'{alpha}'
    str_x0 = f'{x0}'
    name = r'$\huge{\alpha=' + str_alpha + ', x_0=' + str_x0 + '}$'

    fig.add_trace(
        go.Scatter(name=name,
                   x=x, y=normalized_logistic(logistic, x, alpha, x0, scale, bias),
                   line=dict(dash=dash, color=palette[i], width=5),
                   showlegend=True,
                   mode='lines'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(name=name,
                   x=y, y=normalized_logit(logit, y, alpha, x0, scale, bias),
                   line=dict(dash=dash, color=palette[i], width=5),
                   showlegend=False,
                   mode='lines'),
        row=1, col=2
    )

fig.update_xaxes(range=[-1.05, 1.05], row=1, col=1)
fig.update_xaxes(range=[-0.05, 1.05], row=1, col=2)

fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
fig.update_yaxes(range=[-1.05, 1.05], row=1, col=2)

fig.update_annotations(font_size=25)

fig.update_layout(
    height=300,
    width=1200,
    xaxis1=dict(
        tickfont=dict(size=20)),
    yaxis1= dict(
        tickfont=dict(size=20)),
    xaxis2=dict(
        tickfont=dict(size=20)),
    yaxis2=dict(
        tickfont=dict(size=20)),
    autosize=True,
    margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05),
)
fig.show()
write_image(fig, 'logistic_parameter_examples.png', scale=3)