import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
pio.kaleido.scope.mathjax = None


def logistic(x, alpha, x0):
    return 1 / (1 + 2 ** (-alpha * (x - x0)))


def logistic_nqt(x, alpha, x0):
    z = alpha * (x - x0)
    p = np.round(z + 0.5)

    m = 0.5 * (z - p) + 1
    y = m * (2 ** p)
    return y / (y + 1)

# logistic_nqt(np.array([ -0.016253993, 0.057707842, 0.021979338, -3.190555E-4 ]), 55.66698, 0)

def logit(y, alpha, x0):
    z = y / (1 - y)
    return np.log2(z) / alpha + x0

def logit_nqt(y, alpha, x0):
    z = y / (1 - y)
    m, p = np.frexp(z)
    return (2 * m - 2 + p) / alpha + x0


logit_nqt(np.array([ 0.4440233, 0.44765344, 0.94861066, 0.25525683 ]), 0.014675592, 0)

x = np.linspace(-10, 10, num=1_000, endpoint=False)

fig = go.Figure(data=[
    go.Scatter(name='NQT',
               x=x, y=logistic_nqt(x, 1, 0),
               line=dict(width=3, color='#1b9e77')),
    go.Scatter(name='Base-2',
               x=x, y=logistic(x, 1, 0),
               line=dict(dash='dash', width=3, color='#d95f02')),
])
fig.update_xaxes(range=[x.min() - 0.02, x.max() + 0.02])
fig.update_layout(
    title=dict(text='Logistic', x=0.5, xanchor='center'),
    font=dict(size=30),
    showlegend=False,
    autosize=True,
    margin={'l': 0, 'r': 0, 't': 60, 'b': 0},
)
fig.show()
fig.write_image('nqt_logistic.pdf')

x = np.linspace(1e-6, 1 - 1e-6, num=1_000, endpoint=True)

fig = go.Figure(data=[
    go.Scatter(name='NQT',
               x=x, y=logit_nqt(x, 1, 0),
               line=dict(width=3, color='#1b9e77')),
    go.Scatter(name='Base-2',
               x=x, y=logit(x, 1, 0),
               line=dict(dash='dash', width=3, color='#d95f02')),
])
fig.update_xaxes(range=[-0.02, 1.02])
fig.update_layout(
    title=dict(text='Logit', x=0.5, xanchor='center'),
    font=dict(size=30),
    showlegend=False,
    autosize=True,
    margin={'l': 0, 'r': 0, 't': 60, 'b': 0},
)
fig.show()
fig.write_image('nqt_logit.pdf')

x = np.linspace(-10, 20, num=1_000, endpoint=False)

def relative_error_inversion(fun, invfun, x):
    return np.abs(invfun(fun(x, 1, 0), 1, 0) - x) / np.abs(x)

fig = go.Figure(data=[
    go.Scatter(name='Base-2',
               x=x, y=relative_error_inversion(logistic, logit, x),
               opacity=0.5, line=dict(color='#d95f02')),
    go.Scatter(name='NQT',
               x=x, y=relative_error_inversion(logistic_nqt, logit_nqt, x),
               opacity=0.5, line=dict(color='#1b9e77')),
])
fig.update_xaxes(range=[x.min() - 0.02, x.max() + 0.02])
fig.update_yaxes(type='log', dtick=1)
fig.update_layout(
    title=dict(text='Relative inversion error', x=0.5, xanchor='center'),
    font=dict(size=30),
    yaxis=dict(
        title=dict(text='Error'),
        showexponent='all',
        exponentformat='e'
    ),
    autosize=True,
    margin={'l': 0, 'r': 0, 't': 60, 'b': 0},
)
fig.show()
fig.write_image('nqt_inversion_error.pdf')
