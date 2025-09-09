# pip install pillow
from PIL import Image
import os, glob, math

def make_collage(
    image_paths=None,
    folder=None,
    exts=(".png", ".jpg", ".jpeg", ".webp"),
    out_path="collage.png",
    cols=4,                 # quante colonne
    thumb_size=(256, 256),  # dimensione riquadro per immagine
    pad=1,                 # spazio tra riquadri
    margin=1,              # margine esterno
    bg_color=(255, 255, 255),
    keep_aspect=False,       # mantieni aspect ratio dentro thumb_size
    border=0,               # bordo attorno alla miniatura
    border_color=(0,0,0),
    sort_names=True         # ordina alfabeticamente
):
    # Raccogli i path

    if image_paths is None:
        paths = []
        if folder:
            for e in exts:
                paths.extend(glob.glob(os.path.join(folder, f"*{e}")))
        else:
            raise ValueError("Fornisci image_paths o folder.")
    else:
        paths = [p for p in image_paths if os.path.isfile(p)]

    if not paths:
        raise ValueError("Nessuna immagine trovata.")
    if sort_names:
        paths.sort()

    N = len(paths)
    rows = math.ceil(N / cols)

    cell_w, cell_h = thumb_size
    out_w = margin*2 + cols*cell_w + (cols-1)*pad
    out_h = margin*2 + rows*cell_h + (rows-1)*pad

    canvas = Image.new("RGB", (out_w, out_h), bg_color)

    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")

        # ridimensiona
        if keep_aspect:
            im.thumbnail(thumb_size)  # in-place
            thumb = Image.new("RGB", thumb_size, bg_color)
            ox = (thumb_size[0] - im.width)//2
            oy = (thumb_size[1] - im.height)//2
            thumb.paste(im, (ox, oy))
        else:
            thumb = im.resize(thumb_size)

        # bordo opzionale
        if border > 0:
            bordered = Image.new("RGB", (thumb_size[0]+2*border, thumb_size[1]+2*border), border_color)
            bordered.paste(thumb, (border, border))
            thumb = bordered

        r, c = divmod(i, cols)
        x = margin + c*(cell_w + pad)
        y = margin + r*(cell_h + pad)
        # centra se c'è il bordo
        x += (cell_w - thumb.size[0])//2
        y += (cell_h - thumb.size[1])//2
        canvas.paste(thumb, (x, y))

    canvas.save(out_path)
    print("Salvato:", out_path)
    return out_path

if __name__ == "__main__":
    # esempio: crea collage da cartella
    make_collage(
        folder="dataset_immagini/im_usate",     # <-- qui deve esserci la cartella giusta
        out_path="collage.png",
        cols=9,
        thumb_size=(220,220),
        pad=1,
        margin=1
    )

# ---- Esempi d'uso ----
# 1) Da cartella:
# make_collage(folder="dataset_immagini", out_path="collage.png", cols=5, thumb_size=(220,220), pad=10, margin=30)

# 2) Da lista specifica (ordine fissato):
# imgs = ["dataset_immagini/gatto.png", "dataset_immagini/razzo.png", ...]
# make_collage(image_paths=imgs, out_path="collage.png", cols=4)
