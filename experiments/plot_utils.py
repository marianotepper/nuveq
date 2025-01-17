import plotly.graph_objects as go


def build_histogram2d(fig, x, colorscale=None):
    fig.add_trace(go.Histogram2dContour(
        x=x[:, 0],
        y=x[:, 1],
        histnorm='probability',
        colorscale=colorscale,
        ncontours=1000,
        nbinsx=100,
        nbinsy=100,
        contours=dict(showlines=False),
        reversescale=True,
        xaxis='x',
        yaxis='y'
    ))
    fig.add_trace(go.Histogram(
        y=x[:, 1],
        histnorm='probability',
        nbinsy=50,
        xaxis='x2',
        marker=dict(
            color=f'rgba({55 / 255},{126 / 255},{184 / 255},0.5)'
        )
    ))
    fig.add_trace(go.Histogram(
        x=x[:, 0],
        histnorm='probability',
        nbinsx=50,
        yaxis='y2',
        marker=dict(
            color=f'rgba({55 / 255},{126 / 255},{184 / 255},0.5)'
        )
    ))

    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.83],
            showgrid=False
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.83],
            showgrid=False
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            # showgrid=False
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            # showgrid=False
        ),
        height=600,
        width=600,
        bargap=0,
        hovermode='closest',
        showlegend=False
    )
