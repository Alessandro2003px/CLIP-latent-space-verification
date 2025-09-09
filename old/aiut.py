import torch, clip
import torch_directml
from PIL import Image
import requests

# ---- Device DirectML
dml = torch_directml.device()

# ---- Carica modello CLIP su CPU, poi spostalo su DML
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
model = model.eval().to(dml)

# ---- Immagine di prova
#url = "https://huggingface.co/datasets/nateraw/blip-images/resolve/main/apple.jpg"
image = Image.open("../dataset_immagini/gatto+aereo.png").convert("RGB")
image_in = preprocess(image).unsqueeze(0).to(dml)

# ---- Testo (anche uno solo va bene)
prompts = [
    "a photo of a cat and a plane",
    "a photo of a cat, plane background",
    "a photo of a cat starship blurred background",
    "a photo of a cat blurred background",
"a close-up photo of a striped cat outdoors",
"a close-up photo of a striped cat outdoors plane",

"a cat with golden eyes staring at the camera",

"a brown tabby cat with blurred plane behind",

"a cat portrait with background plane slightly out of focus",

"a photo of a curious cat in natural light",

"a striped cat with yellow eyes and plane background",

"a close-up of cat face sharp focus blurred plane",

"a tabby cat sitting outside with plane behind",

"a cat with alert ears and plane in background",

"a photo of a tabby cat sharp and plane blurry",

"a portrait of a brown cat golden eyes outdoors",

"a plane in background with cat foreground photo",

"a cat in foreground, plane visible in distance",

"a brown cat portrait with outdoor blurred background",

"a photo of a tabby cat and tiny plane",

"a close-up cat photo with plane behind trees",

"a striped cat with clear eyes and plane background",

"a tabby cat sharp focus, outdoor plane blurred",

"a portrait of a cat with whiskers and plane",

"a cat in natural setting, plane in background",

"a close-up of a cat with plane far away",

"a cat sitting upright with plane in blurred background",

"a photo of a cat outdoors with red plane",
"a cat staring directly ahead, plane behind",

"a cat with stripes and plane in background",

"a sharp cat portrait with plane slightly visible",

"a brown striped cat golden eyes and plane",

"a cat with blurred plane background in green field",
"a cat in green field",

"a cat face close-up with plane behind",

"a detailed cat portrait outdoors plane background",

"a photo of a plane and cat foreground",

"a cat focused forward with plane blurred",

"a curious tabby cat, plane in background",

"a photo of a cat with toy plane",

"a cat portrait nature background with plane",

"a striped tabby cat golden eyes plane behind",

"a close-up cat face with plane blurred",

"a cat outdoors in focus, plane behind",

"a detailed photo of a cat and plane",

"a portrait of a tabby cat with background plane",
    "a photo of a plane",
    "a photo of a ghibli cat",
    "a photo of a black ghibli cat",
    "a photo of a cat"
]
text_tokens = clip.tokenize(prompts, truncate=True).to(dml)

with torch.no_grad():
    img_f  = model.encode_image(image_in)
    txt_f  = model.encode_text(text_tokens)

    # L2-normalize → il dot product diventa cosine similarity
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

    sims = (img_f @ txt_f.T).squeeze(0)    # (N,) cosine similarity per prompt

# Stampa risultati
for p, s in zip(prompts, sims.tolist()):
    print(f"{p:28s} -> {s:.3f}")
print("BEST:", prompts[int(torch.argmax(sims))])
