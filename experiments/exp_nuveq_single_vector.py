import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import timeit

from nuveq import NonuniformVectorQuantization, NVQParams
from datasets import select_dataset

pio.templates.default = "plotly_white"


def plot_nuveq_single_vector(vector, nonlinearity):
    n_bits = 8

    print(f'{nonlinearity} @ {n_bits} bits')

    model = NonuniformVectorQuantization(n_bits, nonlinearity=nonlinearity)
    tic = timeit.default_timer()
    params, loss_value = model.optimize(vector)
    toc = timeit.default_timer()

    print('\toptimization solution', params.distribution_params)
    print('\toptimization loss', loss_value)
    print('\toptimization time', toc - tic)

    # return

    if nonlinearity == 'logistic':
        param0_name = r'$\alpha$'
        param1_name = r'$x_0$'
        p0_limits = (1e-6, 10)

        x_min = vector.min()
        x_max = vector.max()
        p1_limits = (x_min / (x_max - x_min), x_max / (x_max - x_min))
    if nonlinearity == 'NQT':
        param0_name = r'$\alpha$'
        param1_name = r'$x_0$'
        p0_limits = (1e-6, 20)

        x_min = vector.min()
        x_max = vector.max()
        p1_limits = (x_min / (x_max - x_min), x_max / (x_max - x_min))
    elif nonlinearity == 'kumaraswamy':
        param0_name = 'a'
        param1_name = 'b'
        p0_limits = (1e-6, params.distribution_params[0] * 3)
        p1_limits = (1e-6, params.distribution_params[0] * 3)


    loss_fun = lambda sol: model.loss(
            vector[:, np.newaxis, np.newaxis],
            NVQParams(params.x_min, params.x_max, sol)
    )

    # -----------------------------------
    # fig = px.histogram(vector, nbins=100)
    # fig.show()

    #-----------------------------------
    fig = make_subplots(cols=2, shared_yaxes=True)
    palette = px.colors.qualitative.Dark2

    p0_ls = np.linspace(*p0_limits,
                        num=10_000, endpoint=True)
    fig.add_trace(go.Scatter(name='NVQ',
                             x=p0_ls,
                             y=[loss_fun((a, params.distribution_params[1]))
                                for a in p0_ls],
                             line=dict(color=palette[0])),
                  row=1, col=1)
    fig.add_trace(go.Scatter(name='Uniform quantization',
                             x=[p0_ls[0], p0_ls[-1]], y=[1] * 2,
                             mode='lines',
                             line=dict(color=palette[1], dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(name='Optimization solution',
                             x=[params.distribution_params[0]], y=[loss_value],
                             mode='markers',
                             marker=dict(symbol='x', size=10, color='black')),
                  row=1, col=1)
    fig.update_xaxes(title_text=param0_name, row=1, col=1)
    fig.update_yaxes(title_text='Quantization error reduction', row=1, col=1)


    p1_ls = np.linspace(*p1_limits,
                        num=10_000, endpoint=True)
    fig.add_trace(go.Scatter(name='NVQ',
                             x=p1_ls,
                             y=[loss_fun((params.distribution_params[0], b))
                                for b in p1_ls],
                             line=dict(color=palette[0]),
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(name='Uniform quantization',
                             x=[p1_ls[0], p1_ls[-1]], y=[1] * 2,
                             mode='lines',
                             line=dict(color=palette[1], dash='dash'),
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(name='Optimization solution',
                             x=[params.distribution_params[1]], y=[loss_value],
                             mode='markers',
                             marker=dict(symbol='x', size=10, color='black'),
                             showlegend=False),
                  row=1, col=2)
    fig.update_xaxes(title_text=param1_name, row=1, col=2)

    fig.update_layout(
        legend=dict(x=0.5, y=1.02, orientation="h",
                    yanchor="bottom", xanchor="center")
    )

    fig.show()

    fig.write_image(f'exp_nuveq_cross_cuts_{nonlinearity}.pdf')

    return

    # -----------------------------------
    p0_ls = np.linspace(*p0_limits,
                        num=300, endpoint=True)
    p1_ls = np.linspace(*p1_limits,
                        num=300, endpoint=True)
    loss_kuma = np.array([loss_fun((a, b)) for a in p0_ls for b in p1_ls])
    loss_kuma = loss_kuma.reshape(len(p0_ls), len(p1_ls))

    idx = np.unravel_index(loss_kuma.argmax(), loss_kuma.shape)
    print(p0_ls[idx[0]], p1_ls[idx[1]], loss_kuma.max())

    # return

    palette = px.colors.diverging.RdBu_r
    colorscale_lower = list(np.linspace(0, 1 / loss_kuma.max(),
                                        num=len(palette) // 2, endpoint=False))
    colorscale_upper = list(np.linspace(1 / loss_kuma.max(), 1,
                                        num=1 + len(palette) // 2))
    colorscale = list(zip(colorscale_lower + colorscale_upper, palette))

    fig = go.Figure(
        data=[
            go.Contour(x=p0_ls, y=p1_ls, z=loss_kuma.T, ncontours=1000,
                       colorscale=colorscale,
                       contours=dict(showlines=False)),
            go.Scatter(name='NES solution',
                       x=[params.distribution_params[0]],
                       y=[params.distribution_params[1]],
                       mode='markers',
                       marker=dict(symbol='circle-open', size=20,
                                   color='black',
                                   line=dict(width=1.5, color='black')
                                   ),
                       showlegend=False)
            ],
    )
    fig.add_vline(params.distribution_params[0],
                  line=dict(color='black', dash='dot'))
    fig.add_hline(params.distribution_params[1],
                  line=dict(color='black', dash='dot'))
    fig.update_traces(zmid=1, selector=dict(type='contour'))
    fig.update_xaxes(title_text=param0_name)
    fig.update_yaxes(title_text=param1_name)

    fig.show()

    fig.write_image(f'exp_nuveq_ratio_{nonlinearity}.png', scale=3)


def main():
    dirname = sys.argv[1]
    dataset = select_dataset(dirname, 'ada002-100k')
    # dataset = select_dataset(dirname, 'openai-v3-small-100k')
    # dataset = select_dataset(dirname, 'gecko-100k')
    # dataset = select_dataset(dirname, 'nv-qa-v4-100k')
    # dataset = select_dataset(dirname, 'colbert-1M')

    data = dataset.X_db

    data -= np.mean(data, axis=0, keepdims=True)

    idx = 1
    plot_nuveq_single_vector(data[idx], 'logistic')
    plot_nuveq_single_vector(data[idx], 'NQT')
    plot_nuveq_single_vector(data[idx], 'kumaraswamy')


if __name__ == '__main__':
    main()
