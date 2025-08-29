import torch, open_clip
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import torch_directml

device=""
device = torch_directml.device()
zlim= 10.0
x = torch.ones(3,3).to(device)
print("OK:", x.device)

import numpy as np
import plotly.graph_objects as go

# pip install plotly
import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

def show_vector_23x23_heights(vec, title="23×23 grid | percentile + line toggle + scale buttons",
                              scale=60.0, normalize=False, zlim=None,
                              percentiles=(50,60,70,80,85,90,92,94,95,96,97,98,99),
                              initial_percentile=95,
                              scale_buttons=(10,20,30,40,50,60,70,80)):

    v = np.asarray(vec).reshape(-1)[:512]
    if normalize:
        m = np.max(np.abs(v)) or 1.0
        v = v / m

    grid_w = grid_h = 23
    total_cells = grid_w * grid_h

    padded = np.full(total_cells, np.nan, dtype=float)
    padded[:v.size] = v

    # grid coords
    y_idx, x_idx = np.divmod(np.arange(total_cells), grid_w)
    X, Y = x_idx, y_idx

    # initial Z with provided `scale`
    Z0 = padded * float(scale)
    Z_clean0 = Z0[~np.isnan(Z0)]
    if Z_clean0.size == 0:
        raise ValueError("No valid Z values")

    # axis Z range (independent of color scaling)
    Zlim = float(np.nanmax(np.abs(Z0))) if zlim is None else float(zlim)

    colorscale = [
        [0.0, "rgb(0, 0, 180)"],
        [0.5, "rgb(111, 111, 111)"],
        [1.0, "rgb(255, 255, 255)"]
    ]

    # initial color scaling by percentile of |Z0|
    M0 = np.percentile(np.abs(Z_clean0), initial_percentile)
    cmin0, cmax0 = -M0, M0

    fig = go.Figure()

    # --- markers (trace 0) ---
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z0,
        mode='markers',
        marker=dict(
            size=4,
            color=Z0,
            colorscale=colorscale,
            cmin=cmin0, cmax=cmax0,
            opacity=0.95,
            colorbar=dict(title="value", tickformat=".3f")
        ),
        showlegend=False
    ))

    # --- row/col line traces (1..46) ---
    Xg = X.reshape(grid_h, grid_w)
    Yg = Y.reshape(grid_h, grid_w)
    Zg0 = Z0.reshape(grid_h, grid_w)

    # rows
    for r in range(grid_h):
        fig.add_trace(go.Scatter3d(
            x=Xg[r, :], y=Yg[r, :], z=Zg0[r, :],
            mode='lines',
            line=dict(color='rgb(80,80,80)', width=2),
            hoverinfo='skip',
            showlegend=False,
            visible=True
        ))
    # cols
    for c in range(grid_w):
        fig.add_trace(go.Scatter3d(
            x=Xg[:, c], y=Yg[:, c], z=Zg0[:, c],
            mode='lines',
            line=dict(color='rgb(80,80,80)', width=2),
            hoverinfo='skip',
            showlegend=False,
            visible=True
        ))

    n_rows = grid_h
    n_cols = grid_w
    total_traces = 1 + n_rows + n_cols  # 47 traces total

    # --- Percentile menu (left, top) ---
    buttons_percentile = []
    for p in percentiles:
        M = np.percentile(np.abs(Z_clean0), p)
        buttons_percentile.append(dict(
            method="restyle",
            args=[{"marker.cmin":[-M], "marker.cmax":[M]}, [0]],  # only markers trace
            label=f"{p}%"
        ))

    # --- Line visibility menu (left, middle) ---
    buttons_lines = [
        dict(label="Hide lines",
             method="restyle",
             args=[{"visible": [True] + [False]*(total_traces-1)}, list(range(total_traces))]),
        dict(label="Rows only",
             method="restyle",
             args=[{"visible": [True] + [True]*n_rows + [False]*n_cols}, list(range(total_traces))]),
        dict(label="Cols only",
             method="restyle",
             args=[{"visible": [True] + [False]*n_rows + [True]*n_cols}, list(range(total_traces))]),
        dict(label="All lines",
             method="restyle",
             args=[{"visible": [True]*total_traces}, list(range(total_traces))]),
    ]

    # --- Scale buttons menu (left, lower) ---
    # We update: marker z, marker color, AND every line trace z.
    buttons_scale = []
    for s in scale_buttons:
        Zs = padded * s
        Zgs = Zs.reshape(grid_h, grid_w)
        # scale color limits proportionally (percentile scales linearly with s)
        factor = s / float(scale) if scale != 0 else 1.0
        cmin_s, cmax_s = -M0*factor, M0*factor

        # Build args for all traces
        args_updates = {
            "z": [Zs],                 # markers z
            "marker.color": [Zs],      # markers color
            "marker.cmin": [cmin_s],
            "marker.cmax": [cmax_s]
        }
        trace_idxs = [0]  # markers trace index

        # rows
        for r in range(grid_h):
            args_updates.setdefault("z", []).append(Zgs[r, :])
            trace_idxs.append(1 + r)
        # cols
        for c in range(grid_w):
            args_updates.setdefault("z", []).append(Zgs[:, c])
            trace_idxs.append(1 + n_rows + c)

        buttons_scale.append(dict(
            method="restyle",
            args=[args_updates, trace_idxs],
            label=f"scale {s}"
        ))

    fig.update_layout(
        title=title,
        updatemenus=[
            dict(  # percentile
                type="buttons",
                direction="down",
                buttons=buttons_percentile,
                x=-0.15, xanchor="left",
                y=1.0,  yanchor="top",
                showactive=True,
                bgcolor="rgba(50,50,50,0.7)",
                font=dict(color="white")
            ),
            dict(  # line toggles
                type="buttons",
                direction="down",
                buttons=buttons_lines,
                x=-0.15, xanchor="left",
                y=0.62,  yanchor="top",
                showactive=True,
                bgcolor="rgba(50,50,50,0.7)",
                font=dict(color="white")
            ),
            dict(  # scale buttons
                type="buttons",
                direction="down",
                buttons=buttons_scale,
                x=-0.15, xanchor="left",
                y=0.34,  yanchor="top",
                showactive=True,
                bgcolor="rgba(50,50,50,0.7)",
                font=dict(color="white")
            ),
        ],
        scene=dict(
            xaxis=dict(title='x', range=[-0.5, 22.5], autorange=False,
                       backgroundcolor="black", gridcolor="#333"),
            yaxis=dict(title='y', range=[-0.5, 22.5], autorange=False,
                       backgroundcolor="black", gridcolor="#333"),
            zaxis=dict(title='z', range=[-Zlim, Zlim], autorange=False,
                       backgroundcolor="black", gridcolor="#333"),
            aspectmode='data',
            bgcolor='black'
        ),
        paper_bgcolor='black',
        font_color='white',
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision=True
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

