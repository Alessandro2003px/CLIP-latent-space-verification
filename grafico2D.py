import matplotlib.pyplot as plt
import re
import numpy as np

# --- INCOLLA QUI I TUOI DATI ---
data_text = """
Ref. uomo.png, cos=1.0000
gatto+aereo(small).png, cos=0.438706
gatto+aereo.png, cos=0.436192
 aereo.png, cos=0.336573
gattoGhibli.png, cos=0.487291
 barba.png, cos=0.684460
 donna.png, cos=0.554561
  cane.png, cos=0.494252
 gatto.png, cos=0.494063
"""

# --- PARSING ---
labels = []
cos_values = []

for line in data_text.strip().splitlines():
    match = re.search(r"(.+?),\s*cos=([\d\.]+)", line.strip())
    if match:
        labels.append(match.group(1).strip())
        cos_values.append(float(match.group(2)))

# --- ORDINAMENTO DESC ---
sorted_pairs = sorted(zip(cos_values, labels), reverse=True)
sorted_values, sorted_labels = zip(*sorted_pairs)

# --- COLORI (colormap viridis, ma puoi cambiare) ---
cmap = plt.cm.cividis
colors = cmap(np.linspace(0, 1, len(sorted_values)))

# --- GRAFICO ---
plt.figure(figsize=(8, 4.5))
plt.barh(sorted_labels, sorted_values, color=colors)
plt.xlabel("Cosine similarity")
plt.title("cosine similarity immagini diverse")
plt.gca().invert_yaxis()

# Aggiungi valori numerici alla fine delle barre
for i, v in enumerate(sorted_values):
    if(i==0):
            plt.text(v + 0.01, i, f"{v:.2f}", va="center")
    #elif(i==1):
            #plt.text(v + 0.01, i, f"{v:.4f}", va="center")
    else:
            plt.text(v + 0.01, i, f"{v:.6f}", va="center")

plt.tight_layout()
plt.show()
