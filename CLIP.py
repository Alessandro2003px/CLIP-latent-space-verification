import torch, open_clip
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import torch_directml

device=""
device = torch_directml.device()
zlim= 1.0
x = torch.ones(3,3).to(device)
print("OK:", x.device)

import numpy as np
import plotly.graph_objects as go

# pip install plotly
import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

def show_vector_23x23_heights(vec, title="23×23 grid: height=value, diverging colors",
                              scale=1.0, normalize=False):
    """
    vec: 1D array/tensor (expected length 512).
    - Arranged row-major into a 23×23 grid (529 cells).
    - First 512 values fill the grid; the remaining 17 are filled with a fixed "sanified" bottom value.
    - Z height = value * scale (optionally normalized first).
    - Color: diverging (negative vs positive).
    """
    # accept torch tensors
    try:
        import torch
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
    except ImportError:
        pass

    v = np.asarray(vec).reshape(-1)
    if v.size < 512:
        raise ValueError(f"Need at least 512 values, got {v.size}")

    v = v[:512]  # use first 512 dims

    if normalize:
        m = np.max(np.abs(v)) or 1.0
        v = v / m

    grid_w = grid_h = 23
    total_cells = grid_w * grid_h  # 529

    # padding with "sanified" negative bottom value
    min_val = np.min(v)
    pad_val = min_val - (0.1 * abs(min_val) if min_val != 0 else 1.0)
    padded = np.full(total_cells, pad_val, dtype=float)
    padded[:v.size] = v

    # coordinates
    y_idx, x_idx = np.divmod(np.arange(total_cells), grid_w)
    X = x_idx
    Y = y_idx
    vals = padded
    Z = vals * float(scale)

    # symmetric range for coloring
    M = float(np.max(np.abs(vals))) or 1.0

    if zlim is None:
        Zlim = float(np.max(np.abs(Z))) or 1.0
    else:
        Zlim = float(zlim)


    # Better diverging colorscale (blue for neg, red for pos, gray ~0)
    colorscale = [
        [0.0, "rgb(0, 0, 180)"],     # deep blue = most negative
        [0.5, "rgb(111, 111, 111)"], # light gray around zero
        [1.0, "rgb(255, 255, 255)"]      # red = most positive
    ]

    fig = go.Figure(data=go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=4,
            color=vals,
            colorscale=colorscale,
            cmin=-M, cmax=M,
            opacity=0.95,
            colorbar=dict(title="value", tickformat=".3f")
        ),
        hovertemplate=(
            "dim=%{customdata}<br>x=%{x}, y=%{y}<br>"
            "z=%{z:.3f} (val=%{marker.color:.3f})<extra></extra>"
        ),
        customdata=np.arange(total_cells)  # 0..528, shows which dim
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='x', range=[-0.5, 22.5], autorange=False,
                       backgroundcolor="black", gridcolor="#333"),
            yaxis=dict(title='y', range=[-0.5, 22.5], autorange=False,
                       backgroundcolor="black", gridcolor="#333"),
            zaxis=dict(title='z', range=[-Zlim, Zlim], autorange=False,
                       backgroundcolor="black", gridcolor="#333"),
            aspectmode='cube',  # keep proportions fixed
            bgcolor='black'
        ),
        paper_bgcolor='black',
        font_color='white',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()



model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device)
while(True):

    print("avvio")
    immagine=input("inserisci nome immagine\n")
    if not immagine.endswith(".png") and not immagine.endswith(".png"):
       print("inserisci formato immagine corretto")
       continue
    immagine="dataset_immagini//"+immagine
   
   
    # Load and preprocess image
    try:
        image = preprocess(Image.open(immagine)).unsqueeze(0).to(device)
    except Exception as e:
        print("ERRORE:",e)
        continue
    # Get embedding vector
    with torch.no_grad():
        image_features = model.encode_image(image)   # [1, 512] for ViT-B-32
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    vector = image_features.cpu().numpy()[0]

    print("Vector dimension:", vector.shape)   # should be (512,)
    print("First 10 values:", vector[:10])
    show_vector_23x23_heights(vector)

