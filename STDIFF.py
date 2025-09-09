from diffusers import AutoencoderKL
from PIL import Image
#import torch_directml as torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
latents="a"
device=""
immagine=""
while(True):

    print("avvio")
    immagine=input("inserisci nome immagine\n")
    if not immagine.endswith(".png") and not immagine.endswith(".png"):
       print("inserisci formato immagine corretto")
       continue
    immagine="dataset_immagini//"+immagine
    
    import torch_directml
    device = torch_directml.device()
    import torch
    x = torch.ones(3,3).to(device)
    print("OK:", x.device)
    import plotly.graph_objects as go

    # VAE di Stable Diffusion 1.5
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device).eval()
    print("eval")
    # carica e normalizza l'immagine a 512x512 in [-1, 1]
    try:
        img = Image.open(immagine).convert("RGB").resize((512,512))
    except Exception as e:
        print("ERRORE:",e)
        continue
    tfm = T.Compose([T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    x = tfm(img).unsqueeze(0).to(device)  # [1,3,512,512]
    print("compose e normalize")
    with torch.no_grad():
        print("starting latent encoding")
        Autoencoder = vae.encode(x)
        print("a. ottenuto:")
        print(type(Autoencoder))
        latents=Autoencoder.latent_dist.mean #per approccio deterministico 
        #latents = vae.encode(x).latent_dist.sample() * 0.18215 # per approccio probabilistico, utilizzato per confrontare immagini che dovrebbero essere simili fra loro
        print("oggetto ottenuto:")
        print(type(latents))
    print(latents.shape)
    #print(latents)
    lat = latents[0].cpu().numpy()  # shape [4,64,64]
    print("lat:")
    print(lat.shape)
    print("inizio draw")
    # normalizzazione per valori in [0,1]
    def normalize(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr

    # prendiamo i 4 canali
    r = normalize(lat[0])
    g = normalize(lat[1])
    b = normalize(lat[2])
    h = normalize(lat[3])  # usato per l'altezza

    # griglia
    X, Y = np.meshgrid(np.arange(64), np.arange(64))

    # altezza (z)
    Z = h * 5.0  # stesso scaling che usavi per le barre

    # colori RGBA dai tre canali (0..1 -> 0..255)
    rgba = np.stack([r, g, b, np.full_like(r, 0.67)], axis=-1).reshape(-1,4)
    colors = ['rgba({},{},{},{})'.format(int(R*255), int(G*255), int(B*255), A)
            for R,G,B,A in rgba]

    fig = go.Figure(data=go.Scatter3d(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        mode='markers',
        marker=dict(size=3, color=colors)  # size=2..4 ok
    ))
    color1="white"
    color2="black"
    '''fig.update_layout(
        title="Latent RGB (colore) + Height (Z) - WebGL",
        scene=dict(
            xaxis=dict(showbackground=True, backgroundcolor=color1, showgrid=True, color=color1),
            yaxis=dict(showbackground=True, backgroundcolor=color1, showgrid=True, color=color1),
            zaxis=dict(showbackground=True, backgroundcolor=color1, showgrid=True, color=color1),
            xaxis_title='X', yaxis_title='Y', zaxis_title='H',
            bgcolor='black'
        ),
        paper_bgcolor=color1, font_color=color2
    )'''
    fig.update_layout(
    title="Latent RGB (colore) + Height (Z) - WebGL",
    scene=dict(
        xaxis=dict(
            showbackground=True, backgroundcolor="white",
            showgrid=True, gridcolor="black", color="black"
        ),
        yaxis=dict(
            showbackground=True, backgroundcolor="white",
            showgrid=True, gridcolor="black", color="black"
        ),
        zaxis=dict(
            showbackground=True, backgroundcolor="white",
            showgrid=True, gridcolor="black", color="black"
        ),
        xaxis_title='X', yaxis_title='Y', zaxis_title='H',
        bgcolor="white"
    ),
    paper_bgcolor="white",
    font_color="black"
)

    fig.show()