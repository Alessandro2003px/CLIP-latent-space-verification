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
image = Image.open("dataset_immagini/gattorazzo.png").convert("RGB")
image_in = preprocess(image).unsqueeze(0).to(dml)

# ---- Testo (anche uno solo va bene)
prompts = [
    "a photo of a cat and a rocket",
    "a photo of a cat, rocket background",
    "a photo of a cat starship blurred background",
    "a photo of a cat blurred background",
"a close-up photo of a striped cat outdoors",
"a close-up photo of a striped cat outdoors rocket",

"a cat with golden eyes staring at the camera",

"a brown tabby cat with blurred rocket behind",

"a cat portrait with background rocket slightly out of focus",

"a photo of a curious cat in natural light",

"a striped cat with yellow eyes and rocket background",

"a close-up of cat face sharp focus blurred rocket",

"a tabby cat sitting outside with rocket behind",

"a cat with alert ears and rocket in background",

"a photo of a tabby cat sharp and rocket blurry",

"a portrait of a brown cat golden eyes outdoors",

"a rocket in background with cat foreground photo",

"a cat in foreground, rocket visible in distance",

"a brown cat portrait with outdoor blurred background",

"a photo of a tabby cat and tiny rocket",

"a close-up cat photo with rocket behind trees",

"a striped cat with clear eyes and rocket background",

"a tabby cat sharp focus, outdoor rocket blurred",

"a portrait of a cat with whiskers and rocket",

"a cat in natural setting, rocket in background",

"a close-up of a cat with rocket far away",

"a cat sitting upright with rocket in blurred background",

"a photo of a cat outdoors with red rocket",
"a cat staring directly ahead, rocket behind",

"a cat with stripes and rocket in background",

"a sharp cat portrait with rocket slightly visible",

"a brown striped cat golden eyes and rocket",

"a cat with blurred rocket background in green field",

"a cat face close-up with rocket behind",

"a detailed cat portrait outdoors rocket background",

"a photo of a rocket and cat foreground",

"a cat focused forward with rocket blurred",

"a curious tabby cat, rocket in background",

"a photo of a cat with toy rocket",

"a cat portrait nature background with rocket",

"a striped tabby cat golden eyes rocket behind",

"a close-up cat face with rocket blurred",

"a cat outdoors in focus, rocket behind",

"a detailed photo of a cat and rocket",

"a portrait of a tabby cat with background rocket",
    "a photo of a rocket",
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
