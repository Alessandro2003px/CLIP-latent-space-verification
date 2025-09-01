import numpy as np
import plotly.graph_objects as go

# --- Generate synthetic "embedding-like" data ---
rng = np.random.default_rng(42)
N = 800
dims = 3
points = rng.normal(size=(N, dims)) * 5  # Gaussian cloud

# Assign random colors (like word clusters)
colors = rng.choice(
    ["#ff6b6b", "#ffd93d", "#6bcBef", "#b692ff", "#7bd389"],
    size=N
)

# --- Scatter trace ---
scatter = go.Scatter3d(
    x=points[:,0], y=points[:,1], z=points[:,2],
    mode="markers",
    marker=dict(size=4, color=colors, opacity=0.9),
    hoverinfo="skip"
)

# --- Axis style for "embedding space" ---
axis_style = dict(
    showbackground=False,       # no gray planes
    showgrid=True,              # light grid lines
    gridcolor="rgba(200,200,200,0.2)", # faint grid
    zeroline=True,
    zerolinecolor="white",      # bright central axes
    showticklabels=False,
    ticks=""
)

# --- Build layout ---
fig = go.Figure(data=[scatter])
fig.update_layout(
    scene=dict(
        xaxis=axis_style,
        yaxis=axis_style,
        zaxis=axis_style,
        bgcolor="black"          # dark "space" background
    ),
    paper_bgcolor="black",
    margin=dict(l=0, r=0, t=0, b=0)
)

# Keep cubic aspect ratio
fig.update_scenes(aspectmode="cube")

fig.show()
