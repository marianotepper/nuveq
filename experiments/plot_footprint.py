import plotly
import plotly.graph_objects as go

from plot_utils import write_image

small_font = 35
medium_font = 45
large_font = 45

print("Plotly version:", plotly.__version__)


# Input data
datasets = ["dpr-768-10M", "cohere-1024-10M", "cap-1536-6M"]
fp_bytes = [3.34e10, 4.36e10, 3.89e10]
nvq_bytes = [1.09e10, 1.35e10, 1.13e10]

# Bytes to gigabytes (GB)
fp_gb = [val / 1e9 for val in fp_bytes]
nvq_gb = [val / 1e9 for val in nvq_bytes]

# Create a grouped bar chart with updated layout syntax
fig = go.Figure()

fig.add_trace(go.Bar(
    x=datasets,
    y=fp_gb,
    name="FP",
    marker_color="#e41a1c",
    text=[round(v, 1) for v in fp_gb],
    textposition="auto",
    textfont=dict(size=small_font),
))

fig.add_trace(go.Bar(
    x=datasets,
    y=nvq_gb,
    name="NVQ",
    marker_color="#377eb8",
    text=[round(v, 1) for v in nvq_gb],
    textposition="auto",
    textfont=dict(size=small_font)
))

# Use `xaxis_title`, `xaxis_tickfont`, `yaxis_title`, and `yaxis_title_font` for correct font settings
fig.update_layout(
    template="plotly_white",
    title_text="On-Disk Index Reduction",
    title_font=dict(size=large_font),
    xaxis_title="Dataset",
    xaxis_title_font=dict(size=medium_font),
    xaxis_tickfont=dict(size=small_font),
    yaxis_title="Index size (GB)",
    yaxis_title_font=dict(size=medium_font),
    yaxis_tickfont=dict(size=small_font),
    barmode="group",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=small_font)
    ),
    width=1024,  # larger pixel width
    height=768,  # larger pixel height (maintains ~4:3 aspect ratio)
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.show()
write_image(
    fig,
    "index-footprint-hist.pdf",
    scale=3
)
