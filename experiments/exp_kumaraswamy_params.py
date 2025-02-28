import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from nuveq import forward_kumaraswamy
from plot_utils import write_image

pio.templates.default = "plotly_white"


def kumaraswamy_pdf(x, a, b):
    return a * b * (x ** (a - 1)) * ((1 - x ** a) ** (b - 1))

x = np.linspace(0, 1, 1_000, endpoint=True)

palette = px.colors.qualitative.Plotly

fig = make_subplots(rows=1, cols=2, shared_xaxes=True)

for i, (a, b, dash) in enumerate([(1, 1, 'solid'), (0.5, 0.5, 'dash'),
                                  (2, 2, 'dot'), (5, 1, 'dashdot'),
                                  (1, 5, 'longdashdot')]):
    fig.add_trace(
        go.Scatter(name=f'a={a}, b={b}', x=x, y=kumaraswamy_pdf(x, a, b),
                   line=dict(dash=dash, color=palette[i], width=5),
                   showlegend=False,
                   mode='lines'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(name=f'a={a}, b={b}', x=x, y=forward_kumaraswamy(x, a, b),
                   line=dict(dash=dash, color=palette[i], width=5),
                   mode='lines'),
        row=1, col=2
    )

fig.update_xaxes(range=[0, 1], row=1, col=1)
fig.update_xaxes(range=[0, 1], row=1, col=2)

fig.update_yaxes(range=[0, 2.5], row=1, col=1)
fig.update_yaxes(range=[0, 1], row=1, col=2)

fig.update_layout(
    height=600,
    width=1200,
    font=dict(size=25),
    autosize=True,
    margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
)
fig.show()
write_image(fig, 'kumaraswamy_parameter_examples.pdf')