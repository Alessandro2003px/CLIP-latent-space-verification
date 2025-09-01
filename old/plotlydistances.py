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
# Put *exactly 7* images (first is the center). Can be URLs or local paths.
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

# Extract only the part after "dataset_immagini/"
labels = [os.path.basename(path) for path in images]

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


# ---- Load CLIP on CPU, then move to DirectML/CUDA if needed
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
if device != "cpu":
    model = model.eval().to(device)

# ---- Encode 7 images
embeds = []
for p in images:
    im = load_image(p)
    inp = preprocess(im).unsqueeze(0)
    if device != "cpu":
        inp = inp.to(device)
    with torch.no_grad():
        f = model.encode_image(inp)
        # L2-normalize → dot == cosine
        f = f / f.norm(dim=-1, keepdim=True)
    embeds.append(f.squeeze(0).cpu().numpy())

X = np.vstack(embeds)             # shape (7, D)
C = X @ X.T                       # cosine similarities (7x7)
C = np.clip(C, -1.0, 1.0)         # numerical stability

# Distances from center (row 0) to others
cos = C[0, 1:]                    # cos with #1
if metric == "cosine":
    dists = 1.0 - cos
elif metric == "angular":
    dists = np.arccos(cos)        # radians
elif metric == "euclidean":
    dists = np.sqrt(2 - 2*cos)    # on unit sphere
else:
    raise ValueError("metric must be 'cosine', 'angular', or 'euclidean'")

# Optional normalization of radii
radii_orig = dists.copy()
if normalize_radii and np.max(dists) > 0:
    radii = dists / np.max(dists)
else:
    radii = dists

# Place 6 points at equal angles (hexagon) on XY plane, z=0
N = len(radii)  # should be 6
angles = np.linspace(0, 2*np.pi, N, endpoint=False)

x = [0.0]; y = [0.0]; z = [0.0]
hover = [f"{labels[0]}<br>dist=0.000 (center)"]

for r, ang, lab, d0 in zip(radii, angles, labels[1:], radii_orig):
    x.append(r * math.cos(ang))
    y.append(r * math.sin(ang))
    z.append(0.0)
    hover.append(f"{lab}<br>dist={d0:.4f} ({metric})")

# Build single 3D figure
pts = go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers+text",
    text=labels,
    textposition="top center",
    marker=dict(size=[10] + [7]*(len(labels)-1)),
    hovertext=hover, hoverinfo="text",
    name="points",
     textfont=dict(
        color="white",       # any CSS color name, hex, or rgb string
        size=14,           # font size
        family="Arial"     # optional font family
    )
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

fig = go.Figure(data=[pts] + lines)
pad = 1.2 * max(1.0, np.max(np.abs(x) + np.abs(y)))
fig.update_layout(
    title=f"CLIP distances from {labels[0]} (center) — metric: {metric}",
    scene=dict(
        aspectmode="data",
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        xaxis=dict(title='x', 
                    backgroundcolor="black", gridcolor="#333"),
        yaxis=dict(title='y', 
                    backgroundcolor="black", gridcolor="#333"),
        zaxis=dict(title='z',
                    backgroundcolor="black", gridcolor="#333"),
        bgcolor='black'
    ),
   
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.show()
