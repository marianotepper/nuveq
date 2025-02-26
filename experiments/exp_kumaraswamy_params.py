import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from nuveq import forward_kumaraswamy
from plot_utils import write_image

pio.templates.default = "plotly_white"

x = np.linspace(0, 1, 10_000, endpoint=True)

fig = go.Figure(data=[
    go.Scatter(name=f'a={a}, b={b}', x=x, y=forward_kumaraswamy(x, a, b),
               line=dict(dash=dash, width=3),
               mode='lines')
    for a, b, dash in [(1, 1, 'solid'), (0.5, 0.5, 'dash'), (2, 2, 'dot'),
                       (5, 1, 'dashdot'), (1, 5, 'longdashdot')]
])
fig.update_layout(
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1.01], scaleanchor="x"),
    font=dict(size=18),
    autosize=True,
    margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
)
fig.show()
write_image(fig, 'kumaraswamy_parameter_examples.pdf')