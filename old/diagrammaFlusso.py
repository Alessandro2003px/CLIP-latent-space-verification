import matplotlib
matplotlib.use("Qt5Agg")   # forza Qt backend

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import math
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch

# ---- Look & Feel ----
BG = "black"
NEON = "#00ffff"   # prova "#ff00ff" per look più synthwave
TXT = "#e6ffff"

# ---- Geometria (centri) ----
#w, h = 4, 2.0

from matplotlib.transforms import Affine2D

from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

def add_multiline_text(ax, x, y, text, fontsize=12, color="white", line_spacing=1.2):
    lines = text.splitlines()
    paths = []
    widths, heights = [], []

    # crea un path per ogni riga (in (0,0))
    for line in lines:
        tp = TextPath((0, 0), line, size=fontsize)
        bb = tp.get_extents()
        widths.append(bb.width)
        heights.append(bb.height)
        paths.append(tp)

    max_width = max(widths)
    total_height = sum(h*line_spacing for h in heights)

    # y di partenza (in alto)
    y_start = y + total_height/1.1
    patches = []

    for tp, w, h in zip(paths, widths, heights):
        bb = tp.get_extents()
        cx = (bb.x0 + bb.x1)/2.0
        cy = (bb.y0 + bb.y1)/2.0

        # posizione corrente della riga
        y_start -= h*line_spacing
        # centro della riga
        tx = x - cx
        ty = y_start - cy

        trans = Affine2D().translate(tx, ty) + ax.transData
        patch = PathPatch(tp, transform=trans, color=color)
        ax.add_patch(patch)
        patches.append(patch)

    return patches



def rect_with_glow(fig,ax, color=NEON, label=None,master=False,linker=None,angle=0,distance=None,arrowAngle=True):
    ox,oy=0,0
    Fsize = 12 
    offset=0
    w,h=1,1 #default size of rectangle
    if label:
        longest = max(label.splitlines(), key=len)

        w=(1*Fsize*len(longest))/15
        print(w)
        print(Fsize)
        h=1*Fsize*len(label.splitlines())/4

    if not master:
        ox,oy,linkerW,linkerH=linker #unpacking coordinates (only first two are needed, aside for arrows)

        if arrowAngle: #in futuro da implementare direzionalità (poter andare verso il basso in modo smooth)
            offset=linkerW/2 +w/2
        if distance:
            ox=ox+offset+distance*math.cos(math.radians(angle))
            oy=oy+distance*math.sin(math.radians(angle))
        else:
            ox=ox+offset+5*math.cos(math.radians(angle))
            oy=oy+5*math.sin(math.radians(angle))
    
    x0, y0 = ox - w/2, oy - h/2

    r = patches.Rectangle((x0, y0), w, h, fill=False, linewidth=2, edgecolor=color)
    r.set_path_effects([
        pe.Stroke(linewidth=10, foreground=color, alpha=0.12),  # alone largo
        pe.Stroke(linewidth=6,  foreground=color, alpha=0.25),  # alone medio
        pe.Normal(),                                            # bordo netto
    ])
    ax.add_patch(r)
    if label:
        add_multiline_text(ax, ox, oy, label, fontsize=1.5, color="white")
    
    if not master: #una volta terminato il rettangolo, creo la freccia automaticamente
        arrow_with_glow(ax, right_edge(*linker), left_edge(ox,oy,w,h))

    return ox,oy,w,h

def arrow_with_glow(ax, start, end, color=NEON):
   
    for lw, alpha in [(8, 0.18), (5, 0.35)]:
        arr = patches.FancyArrowPatch(
            posA=start, posB=end, arrowstyle='-', mutation_scale=18,
            linewidth=lw, color=color, alpha=alpha, shrinkA=4, shrinkB=4,
            joinstyle="round", capstyle="round"
        )
        ax.add_patch(arr)

    arr = patches.FancyArrowPatch(
        posA=start, posB=end, arrowstyle='->', mutation_scale=18,
        linewidth=2.2, color=color, shrinkA=1, shrinkB=0,
        joinstyle="round", capstyle="round"
    )
    ax.add_patch(arr)

def right_edge(cx, cy,w,h):  return (cx + w/2, cy)
def left_edge(cx, cy,w,h):   return (cx - w/2, cy)
plt.rcParams['toolbar'] = 'toolmanager'   # disabilita toolbar per tutte le figure

fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
ax.set_facecolor(BG)

tm = fig.canvas.manager.toolmanager
tb = fig.canvas.manager.toolbar               # Toolbar (Qt)
tb.setStyleSheet("""
    QToolBar { background: black; }
    QToolButton { background: cyan;color: gray; }
""")
print("Tools disponibili:", tm.tools.keys())
keep = {'save','pan','home','zoom','viewpos','rubberband'}  # tieni solo il pulsante "Save"

for name in list(tm.tools.keys()):
    if name not in keep:
        try:
            tm.remove_tool(name)
        except Exception:
            pass
toolitems = getattr(tb, "_toolitems", {})
for name in list(toolitems.keys()):
    if name not in keep:
        try:
            tb.remove_toolitem(name)
        except KeyError:
            pass
# Rettangoli

C1=rect_with_glow(fig,ax,label="Immagine generata",master=True)
C2=rect_with_glow(fig,ax,label="Vettore vicino a molte immagini,\n concetto che il modello ha imparato [forse a memoria]",linker=C1,angle=60)
C3=rect_with_glow(fig,ax, label="maggior parte del contributo da queste immagini",linker=C2,angle=0)

C4=rect_with_glow(fig,ax,label="Vettore lontano, concetto nuovo",linker=C1,angle=-60)
C5=rect_with_glow(fig,ax,label="Se il risultato non è gibberish (possibile nei modelli piu avanzati),\nscomposizione in componenti",linker=C4,angle=0)

C6=rect_with_glow(fig,ax,label="utilizzare clip per estrarre vett. immagine finale",linker=C5,angle=30,distance=15)
C7=rect_with_glow(fig,ax,label="[???cpu] utilizzare blip per generare un caption,\npotendolo verificare con clip",linker=C5,angle=0,distance=13)
C8=rect_with_glow(fig,ax,label="analizzare tramite analisi semantica [!Ai]+clip i\nrisultati dei componenti singoli ",linker=C5,angle=-30,distance=15)

C9=rect_with_glow(fig,ax,label="risultato: scomposizione dei singoli concetti usati dall'ia\n per generare l'immagine",linker=C7,angle=0,distance=15)

arrow_with_glow(ax, right_edge(*C6), left_edge(*C9))
arrow_with_glow(ax, right_edge(*C8), left_edge(*C9))

# Frecce (uscita dal bordo destro, ingresso sul bordo sinistro)
#arrow_with_glow(ax, right_edge(*C1), left_edge(*C2))

#arrow_with_glow(ax, right_edge(*C4), left_edge(*C1))  

# Layout
ax.set_aspect("equal")

#ax.set_xlim(-25, 25)
#ax.set_ylim(-25, 25)
#ax.autoscale_view()
ax.autoscale(True)         # blocca autoscaling

ax.axis("on")

plt.tight_layout()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.show()
