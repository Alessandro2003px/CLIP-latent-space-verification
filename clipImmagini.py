#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from typing import Tuple
from PIL import Image
import torch
import torch.nn.functional as F

# HF Transformers
from transformers import CLIPProcessor, CLIPModel


# ------------------------- Device selection -------------------------
def get_device() -> torch.device:
    """
    Prefer DirectML (Windows), then CUDA, then CPU.
    """
    try:
        import torch_directml  # noqa: F401
        return torch_directml.device()
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ------------------------- Image utils -------------------------
def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ------------------------- HF/Transformers backend -------------------------
def clip_image_similarity_hf(
    img_path1: str,
    img_path2: str,
    model_name: str = "openai/clip-vit-base-patch32",
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, torch.Tensor]:
    """
    Compute cosine similarity between two images using Hugging Face CLIP.
    Returns (cosine, features_tensor[2, D]).
    """
    clip = CLIPModel.from_pretrained(model_name, use_safetensors=True).eval().to(device)
    proc = CLIPProcessor.from_pretrained(model_name, use_safetensors=True)

    img1 = load_rgb(img_path1)
    img2 = load_rgb(img_path2)

    with torch.no_grad():
        batch = proc(images=[img1, img2], return_tensors="pt").to(device)
        feats = clip.get_image_features(batch["pixel_values"])  # [2, D]
        feats = F.normalize(feats, dim=-1)
        cos = (feats[0] @ feats[1].T).item()
    return cos, feats


# ------------------------- open_clip fallback -------------------------
def clip_image_similarity_openclip(
    img_path1: str,
    img_path2: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, torch.Tensor]:
    """
    Compute cosine similarity between two images using open_clip.
    Returns (cosine, features_tensor[2, D]).
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )

    img1 = preprocess(load_rgb(img_path1)).unsqueeze(0).to(device)  # [1,3,H,W]
    img2 = preprocess(load_rgb(img_path2)).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        f1 = model.encode_image(img1)  # [1, D]
        f2 = model.encode_image(img2)  # [1, D]
        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)
        cos = (f1 @ f2.T)[0, 0].item()
        feats = torch.cat([f1, f2], dim=0)  # [2, D]
    return cos, feats


# ------------------------- Optional: angle from cosine -------------------------
def cosine_to_angle_deg(cosine: float) -> float:
    """
    Convert cosine similarity to angle (degrees) in [0, 180].
    Guards against tiny numeric issues.
    """
    c = max(-1.0, min(1.0, float(cosine)))
    return float(torch.rad2deg(torch.arccos(torch.tensor(c))).item())


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute CLIP cosine similarity between two images."
    )
    ap.add_argument("image1", help="Path to first image")
    ap.add_argument("image2", help="Path to second image")
    ap.add_argument(
        "--backend",
        choices=["hf", "openclip", "auto"],
        default="auto",
        help="Which backend to use (default: auto = HF, with open_clip fallback).",
    )
    ap.add_argument(
        "--hf-model",
        default="openai/clip-vit-base-patch32",
        help="HF model name (when backend is hf/auto).",
    )
    ap.add_argument(
        "--oc-model",
        default="ViT-B-32",
        help="open_clip model name (when backend is openclip/auto).",
    )
    ap.add_argument(
        "--oc-pretrained",
        default="openai",
        help="open_clip pretrained tag (when backend is openclip/auto).",
    )
    ap.add_argument(
        "--show-angle",
        action="store_true",
        help="Also print the angular distance (degrees).",
    )
    args = ap.parse_args()

    device = get_device()
    print(f"[INFO] device: {device}")

    cosine = None
    backend_used = None

    if args.backend in ("hf", "auto"):
        try:
            cosine, _ = clip_image_similarity_hf(
                args.image1, args.image2, model_name=args.hf_model, device=device
            )
            backend_used = "hf/transformers"
        except Exception as e:
            if args.backend == "hf":
                raise
            print(f"[WARN] HF backend failed: {e}")

    if cosine is None and args.backend in ("openclip", "auto"):
        try:
            cosine, _ = clip_image_similarity_openclip(
                args.image1,
                args.image2,
                model_name=args.oc_model,
                pretrained=args.oc_pretrained,
                device=device,
            )
            backend_used = "open_clip"
        except Exception as e:
            if args.backend == "openclip":
                raise
            print(f"[WARN] open_clip backend failed: {e}")

    if cosine is None:
        raise RuntimeError("Both backends failed. Check models and dependencies.")

    print("\n=== CLIP Image↔Image Cosine Similarity ===")
    print(f"backend: {backend_used}")
    print(f"cosine ≈ {cosine:.6f}  (range [-1, 1], higher = more similar)")
    if args.show_angle:
        ang = cosine_to_angle_deg(cosine)
        print(f"angle  ≈ {ang:.3f}°  (0° = identical direction)")

    # Small tip on interpretation
    print("\nTip: values around 0.25–0.35 often indicate weak semantic relation;")
    print("0.35–0.55 moderate; 0.55–0.75 strong; >0.75 very strong (rough intuition).")


if __name__ == "__main__":
    main()
