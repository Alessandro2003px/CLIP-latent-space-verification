#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from typing import List
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import MinLengthLogitsProcessor

from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)

# ---------- Device (DirectML -> CPU fallback) ----------
def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")

# ---------- Utils ----------
def load_image(path):
    return Image.open(path)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a @ b.T).squeeze(-1)

# ---------- CLIP-guided logits processor ----------
class ClipGuidanceProcessor(LogitsProcessor):
    """
    Reranking dei logits BLIP in base a CLIP cosine(img, testo_parziale+token).
    Per efficienza, valuta solo i top_k token proposti da BLIP per ogni beam.
    I punteggi CLIP sono normalizzati (min-max) nel gruppo dei top_k prima di essere combinati.
    """
    def __init__(
        self,
        blip_tokenizer,
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        clip_image_features: torch.Tensor,
        weight: float = 2.0,
        top_k: int = 30,
        device=None,
    ):
        super().__init__()
        self.tok = blip_tokenizer
        self.clip_model = clip_model
        self.clip_proc = clip_processor
        self.img_feat = F.normalize(clip_image_features, dim=-1)  # [1, d]
        self.weight = float(weight)
        self.top_k = int(top_k)
        self.device = device or get_device()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bs, _ = scores.shape
        adjusted = scores.clone()

        for i in range(bs):
            row_scores = scores[i]
            k = min(self.top_k, row_scores.shape[-1])
            top_vals, top_idx = torch.topk(row_scores, k, dim=-1)

            seq_ids = input_ids[i].tolist()
            cand_texts: List[str] = []
            for tid in top_idx.tolist():
                new_ids = seq_ids + [tid]
                text = self.tok.decode(new_ids, skip_special_tokens=True)
                cand_texts.append(text if text.strip() else " ")

            with torch.no_grad():
                clip_inputs = self.clip_proc(text=cand_texts, return_tensors="pt", padding=True).to(self.device)
                tfeat = self.clip_model.get_text_features(
                    input_ids=clip_inputs["input_ids"],
                    attention_mask=clip_inputs["attention_mask"]
                )
                tfeat = F.normalize(tfeat, dim=-1)  # [k, d]
                cos = (tfeat @ self.img_feat.T).squeeze(-1)

            cos_min = cos.min()
            cos_max = cos.max()
            if (cos_max - cos_min) > 1e-6:
                cos_norm = (cos - cos_min) / (cos_max - cos_min)
            else:
                cos_norm = torch.zeros_like(cos)

            adjusted[i, top_idx] = row_scores[top_idx] + self.weight * cos_norm

        return adjusted

# ---------- Cosine similarity (immagine vs caption) ----------
def clip_cosine_similarity_transformers(clip_model, clip_processor, image: Image.Image, text: str, device):
    """Usa CLIP di Hugging Face per ottenere cos(img, text)."""
    with torch.no_grad():
        batch = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        img_feat = clip_model.get_image_features(batch["pixel_values"])
        txt_feat = clip_model.get_text_features(batch["input_ids"], batch["attention_mask"])
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        cos = (img_feat @ txt_feat.T)[0, 0].item()
    return cos

def clip_cosine_similarity_openclip(image: Image.Image, text: str, device):
    """
    Fallback con open_clip. Se DirectML non supporta qualche op, fa fallback su CPU.
    """
    try:
        import open_clip
    except Exception as e:
        raise RuntimeError("open_clip_torch non installato. Installa con: pip install open_clip_torch") from e

    # prova su device attuale (es. DirectML). Se fallisce, passa a CPU.
    try_devices = [device]
    if str(device).lower() != "cpu":
        try_devices.append(torch.device("cpu"))

    last_err = None
    for dev in try_devices:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=dev)
            tokenizer = open_clip.get_tokenizer("ViT-B-32")

            # preprocess dell’immagine (ritorna Tensor [3,H,W], normalizzato)
            img_t = preprocess(image).unsqueeze(0).to(dev)
            txt_t = tokenizer([text]).to(dev)

            with torch.no_grad():
                img_feat = model.encode_image(img_t)
                txt_feat = model.encode_text(txt_t)
                img_feat = F.normalize(img_feat, dim=-1)
                txt_feat = F.normalize(txt_feat, dim=-1)
                cos = (img_feat @ txt_feat.T)[0, 0].item()
            return cos
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"open_clip non è riuscito né su {device} né su CPU. Ultimo errore: {last_err}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="BLIP caption con decoding CLIP-guided + cosine similarity (DirectML compat).")
    ap.add_argument("image", help="Percorso immagine")
    ap.add_argument("--blip", default="Salesforce/blip-image-captioning-base", help="Modello BLIP (captioning)")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32", help="Modello CLIP (HF)")
    ap.add_argument("--num-beams", type=int, default=3, help="Beam size (BLIP).")
    ap.add_argument("--max-new-tokens", type=int, default=30, help="Lunghezza max caption.")
    ap.add_argument("--top-k", type=int, default=30, help="Token rivisti da CLIP ad ogni step.")
    ap.add_argument("--lambda-w", type=float, default=2.0, help="Peso λ della guida CLIP.")
    args = ap.parse_args()

    device = get_device()
    print(f"[INFO] device: {device}")

    # Carica BLIP
    print("[INFO] Carico BLIP…")
    blip_proc = BlipProcessor.from_pretrained(args.blip, use_safetensors=True)
    blip = BlipForConditionalGeneration.from_pretrained(args.blip, use_safetensors=True).eval().to(device)

    # Carica CLIP (HF)
    print("[INFO] Carico CLIP (Hugging Face)…")
    clip_proc = CLIPProcessor.from_pretrained(args.clip, use_safetensors=True)
    clip = CLIPModel.from_pretrained(args.clip, use_safetensors=True).eval().to(device)

    # Immagine
    img = load_image(args.image)

    # Precalcola feature CLIP dell’immagine (per guida durante decoding)
    with torch.no_grad():
        clip_img_inputs = clip_proc(images=img, return_tensors="pt").to(device)
        img_feat = clip.get_image_features(clip_img_inputs["pixel_values"])

    # Prepara inputs BLIP
    #prefix="an accurate, detailed caption:"
    prefix=""
    blip_inputs = blip_proc(images=img, text=prefix, return_tensors="pt").to(device)

    # LogitsProcessor con guida CLIP
    #min_len = 12
    #eos_token_id = blip_proc.tokenizer.eos_token_id

    lp = LogitsProcessorList([
        #MinLengthLogitsProcessor(min_len, eos_token_id=eos_token_id),
        ClipGuidanceProcessor(
            blip_tokenizer=blip_proc.tokenizer,
            clip_model=clip,
            clip_processor=clip_proc,
            clip_image_features=img_feat,
            weight=args.lambda_w,
            top_k=args.top_k,
            device=device
        )
    ])

    # Generazione guidata
    print("[INFO] Genero con guida CLIP (potrebbe essere lento: top-k x beams x passi)…")
    print("num beans "+str(args.num_beams))
    print("lambda "+str(args.lambda_w))
    print("top-k "+str(args.top_k))

    with torch.no_grad():
        out_ids = blip.generate(
            **blip_inputs,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            logits_processor=lp,
            min_new_tokens=12,
            length_penalty=1.2,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    caption = blip_proc.decode(out_ids[0], skip_special_tokens=True)
    print("\n=== CAPTION (BLIP + CLIP-guided) ===")
    print(caption)

    # ---------- Analisi vettoriale + cosine similarity nello spazio CLIP ----------
    print("\n[INFO] Calcolo cosine similarity (immagine ↔ caption) nello spazio CLIP…")
    try:
        
       # cos_val = clip_cosine_similarity_transformers(clip, clip_proc, img, caption[len(prefix):].strip(), device)
        
        #text = "a dog"
        model, _, preprocess = clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
        tokenizer = clip.get_tokenizer("ViT-B-32")

            # preprocess dell’immagine (ritorna Tensor [3,H,W], normalizzato)
        img_t = preprocess(img).unsqueeze(0).to(device)
        text_features = clip.encode_text(clip.tokenize([caption]).to(device))

        image_features = clip.encode_image(img_t)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
# shape: (1, 512)

        sim = torch.cosine_similarity(image_features, text_features)
        print(sim.item())
        
        backend = "transformers"
    except Exception as e_hf:
        print(f"[WARN] CLIP (transformers) ha dato errore: {e_hf}\n[INFO] Provo fallback open_clip…")
        cos_val = clip_cosine_similarity_openclip(img, caption, device)
        backend = "open_clip"

    print("\n=== CLIP Cosine Similarity ===")
    print(f"backend: {backend}  |  cos ≈ {cos_val:.4f}")
    print("(valori più alti ⇒ caption più coerente con l'immagine nello spazio CLIP)")


    '''usage:
 #CLIP-BLIP-first.py path/foto.png --num-beams 5 (piu alto-> piu esplorazione) --top-k 20..50 (velocità/qualità) --lambda-w 1.0..3.0 (influenza di clip su blip)
 .\CLIP-BLIP-first.py dataset_immagini/foto.png --num-beams 5  --top-k 20..50  --lambda-w 1.0..3.0
 .\CLIP-BLIP-first.py dataset_immagini/foto.png --num-beams 5  --top-k 20..50
 .\CLIP-BLIP-first.py dataset_immagini/foto.png --num-beams 5  
    '''
if __name__ == "__main__":
    main()
