# pip install torch-directml git+https://github.com/openai/CLIP.git pillow requests plotly

import io, os, math, requests
import numpy as np
from PIL import Image

import torch
import clip

# Try DirectML first, then CUDA, then CPU
try:
    import torch_directml
    device = torch_directml.device()
except Exception:
    device = "cuda" if torch.cuda.is_available() else "cpu"

import plotly.graph_objects as go

# ----------------------------- CONFIG ---------------------------------
# You can use any number of images ≥ 2 (first shown as center initially).
# Can be URLs or local paths.
images = [
    "dataset_immagini/gattorazzo.png",
    "dataset_immagini/gattoGhibli.png",
    "dataset_immagini/gattorazzoBIG.png",
    "dataset_immagini/razzo.png",
    "dataset_immagini/barba.png",
    "dataset_immagini/donna.png",
    "dataset_immagini/cane.png",
    "dataset_immagini/gatto.png"
]

# Show only the basename after the folder
labels = [os.path.basename(p) for p in images]

# Distance metric: "cosine", "angular", or "euclidean"
metric = "cosine"

# Normalize radii so the largest = 1 (keeps plot tidy). Set False to keep absolute radii.
normalize_radii = True
# ----------------------------------------------------------------------

def load_image(path_or_url) -> Image.Image:
    if isinstance(path_or_url, str) and (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
        r = requests.get(path_or_url, stream=True, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        return Image.open(path_or_url).convert("RGB")

# ---- Load CLIP on CPU, then move to DirectML/CUDA if available
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
if device != "cpu":
    model = model.eval().to(device)

# ---- Encode all images once
embeds = []
for p in images:
    im = load_image(p)
    inp = preprocess(im).unsqueeze(0)
    if device != "cpu":
        inp = inp.to(device)
    with torch.no_grad():
        f = model.encode_image(inp)
        f = f / f.norm(dim=-1, keepdim=True)   # L2-normalize → dot == cosine
    embeds.append(f.squeeze(0).cpu().numpy())

X = np.vstack(embeds)             # shape (N, D)
C = np.clip(X @ X.T, -1.0, 1.0)   # cosine similarities (N x N), clipped for stability
N = len(images)

def distances_from_center(center_idx: int):
    """Compute distances from chosen center to all others (order: center first, then the rest)."""
    # Build the order: [center, all others...]
    others = [i for i in range(N) if i != center_idx]
    order = [center_idx] + others  # kept if you want to use it later

    # Cosine similarities from center to others
    cos = C[center_idx, others]

    if metric == "cosine":
        dists = 1.0 - cos
    elif metric == "angular":
        dists = np.arccos(cos)          # radians
    elif metric == "euclidean":
        dists = np.sqrt(2 - 2*cos)      # on unit sphere
    else:
        raise ValueError("metric must be 'cosine', 'angular', or 'euclidean'")

    radii_orig = dists.copy()
    if normalize_radii and np.max(dists) > 0:
        radii = dists / np.max(dists)
    else:
        radii = dists

    # Place M = (N-1) points around a circle on XY plane
    M = len(radii)
    angles = np.linspace(0, 2*np.pi, M, endpoint=False)

    x = [0.0]; y = [0.0]; z = [0.0]
    labs = [labels[center_idx]] + [labels[i] for i in others]
    hover = [f"{labels[center_idx]}<br>dist=0.000 (center)"]

    for r, ang, lab, d0 in zip(radii, angles, labs[1:], radii_orig):
        x.append(float(r * math.cos(ang)))
        y.append(float(r * math.sin(ang)))
        z.append(0.0)
        hover.append(f"{lab}<br>dist={d0:.4f} ({metric})")

    return x, y, z, labs, hover

def build_trace_set_for_center(center_idx: int):
    """Return [points_trace, *line_traces] for a given center choice."""
    x, y, z, labs, hover = distances_from_center(center_idx)

    pts = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        text=labs,
        textposition="top center",
        marker=dict(size=[10] + [7]*(len(labs)-1)),
        hovertext=hover, hoverinfo="text",
        name="points",
        textfont=dict(color="white", size=14, family="Arial")
    )

    lines = []
    for i in range(1, len(x)):
        lines.append(go.Scatter3d(
            x=[0, x[i]], y=[0, y[i]], z=[0, z[i]],
            mode="lines",
            line=dict(width=2),
            hoverinfo="skip",
            showlegend=False
        ))
    return [pts] + lines

def compute_extent_from_traces(traces):
    """Flatten x/y/z from all Scatter3d traces and return a symmetric half-range m."""
    xs, ys, zs = [], [], []
    for t in traces:
        if isinstance(t, go.Scatter3d):
            xs.extend(np.ravel(t.x))
            ys.extend(np.ravel(t.y))
            zs.extend(np.ravel(t.z))
    # pad a bit; keep at least ±1 so axes don't collapse
    m = 1.2 * max(
        1.0,
        float(np.max(np.abs(xs))) if xs else 1.0,
        float(np.max(np.abs(ys))) if ys else 1.0,
        float(np.max(np.abs(zs))) if zs else 1.0,
    )
    return m

# --- Build the base figure using images[0] as the initial center
initial_center = 0
base_traces = build_trace_set_for_center(initial_center)
m = compute_extent_from_traces(base_traces)

fig = go.Figure(data=base_traces)

fig.update_layout(
    title=f"CLIP distances from {labels[initial_center]} (center) — metric: {metric}",
    scene=dict(
        aspectmode="data",
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        xaxis=dict(title='x', range=[-m, m], backgroundcolor="black", gridcolor="#333"),
        yaxis=dict(title='y', range=[-m, m], backgroundcolor="black", gridcolor="#333"),
        zaxis=dict(title='z', range=[-m, m], backgroundcolor="black", gridcolor="#333"),
        bgcolor='black'
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    paper_bgcolor="black",
    font=dict(color="white")
)

# --- Create a frame for each possible center (keeps same axis ranges so the box doesn't jump)
frames = []
for ci in range(N):
    traces = build_trace_set_for_center(ci)
    frames.append(go.Frame(
        name=f"center-{ci}",
        data=traces,
        layout=go.Layout(
            title=f"CLIP distances from {labels[ci]} (center) — metric: {metric}",
            scene=go.layout.Scene(
                xaxis=dict(range=[-m, m]),
                yaxis=dict(range=[-m, m]),
                zaxis=dict(range=[-m, m]),
            ),
        ),
    ))
fig.frames = frames

# --- Left-side vertical buttons (one per image) to switch center instantly
buttons = []
for ci, lab in enumerate(labels):
    buttons.append(dict(
        label=lab,
        method="animate",
        args=[
            [f"center-{ci}"],
            {
                "mode": "immediate",
                "transition": {"duration": 0},
                "frame": {"duration": 0, "redraw": True}
            }
        ]
    ))

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="down",       # stack vertically
            x=0.02, y=0.9,          # left side
            xanchor="left", yanchor="top",
            showactive=True,
            bgcolor="rgba(50,50,50,0.6)",
            bordercolor="#666",
            pad={"r": 8, "t": 8, "b": 8, "l": 8},
            buttons=buttons
        )
    ]
)

fig.show()
