# q5.py - diagramma con glow in PyQt5 + pyqtgraph (niente TextPath)

import sys, math, pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPen, QBrush, QColor, QPolygonF, QFont, QFontMetrics
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QApplication, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem
from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem
# ---- Look & Feel ----
BG   = QColor(0, 0, 0)
#NEON = QColor(0, 255, 255)     # ciano neon
NEON = QColor(0, int(255/1.1), int(255/1.1))     # bianco 

TXT  = QColor(230, 255, 255)   # testo

# ---------- utility ----------
def mk_pen(color, width, alpha=255):
    c = QColor(color); c.setAlpha(alpha)
    pen = QPen(c)
    pen.setWidthF(width)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    return pen

# testo centrato con pg.TextItem (supporta \n)
def add_centered_text(view, cx, cy, text, font):
    item = pg.TextItem(text, color=(TXT.red(), TXT.green(), TXT.blue()), anchor=(0.5, 0.5))
    item.setFont(font)
    item.setPos(cx, cy)
    view.addItem(item)
    # misura “grezza” col font per dimensionare il box
    fm = QFontMetrics(font)
    lines = text.splitlines() or [text]
    maxw  = max(fm.horizontalAdvance(ln) for ln in lines)
    htot  = sum(fm.height() for _ in lines)
    return item, maxw, htot

def right_edge(cx, cy, w, h): return (cx + w/2.0, cy)
def left_edge(cx, cy, w, h):  return (cx - w/2.0, cy)

def top_edge(cx, cy, w, h): return (cx , cy- h/2.0)
def bottom_edge(cx, cy, w, h):  return (cx, cy + h/2.0)

def vertical_rect(view, *, color=NEON, label=None,
                   master=False, linker=None, angle=0.0, distance=None, arrowAngle=True,
                   base_font=QFont("DejaVu Sans", 11), pad_x=16, pad_y=10,direction="DOWN"):
    
    if label:
        fm = QFontMetrics(base_font)
        lines = label.splitlines() or [label]
        maxw  = max(fm.horizontalAdvance(ln) for ln in lines)
        htot  = sum(fm.height() for _ in lines)
        w = max(80, maxw + 2*pad_x)
        h = max(40, htot + 2*pad_y)
    else:
        w, h = 120, 60

    # centro (ox, oy)
    if master or linker is None:
        ox, oy = 0.0, 0.0
    else:
        dir = 1 if direction == "DOWN" else (-1 if direction == "UP" else 0)
        lx, ly, lw, lh = linker
        offs = lh/2.0 + h/2.0 if arrowAngle else 0.0
        dist = distance if distance is not None else 90.0
        rad  = math.radians(angle-(90*dir))
        ox   = lx + dist * math.cos(rad)
        oy   = ly -(offs*dir)+ dist * math.sin(rad)

    # shape glow (due layer) + bordo netto
    x = ox - w/2.0; y = oy - h/2.0

    #glow1 = QGraphicsRectItem(x, y, w, h); glow1.setPen(mk_pen(color, 18, alpha=45)); view.addItem(glow1)
    for i in range(1, 9):  #12*2=24 6*3=18 6*2=12
        glow = QGraphicsRectItem(x, y, w, h); glow.setPen(mk_pen(color, 2*i,alpha=(50-5*i))); view.addItem(glow) #gli oggetti glow si sovrappongono
    
    core  = QGraphicsRectItem(x, y, w, h); core.setPen(mk_pen(color, 2.6,  alpha=255)); view.addItem(core)

    if label:
        add_centered_text(view, ox, oy, label, base_font)
    if not master:
        add_arrow_with_glow(view, *top_edge(*linker), *bottom_edge(ox,oy,w,h))
    return (ox, oy, w, h)
# ---------- rettangolo con glow ----------
def rect_with_glow(view, *, color=NEON, label=None,
                   master=False, linker=None, angle=0.0, distance=None, arrowAngle=True,
                   base_font=QFont("DejaVu Sans", 11), pad_x=16, pad_y=10,direction="RIGHT"):
    """
    Crea un rettangolo glow con etichetta centrata.
    Ritorna (cx, cy, w, h).
    - master=True -> parte da (0,0)
    - altrimenti usa linker=(cx,cy,w,h) e posiziona con angle/distance (+ offset bordi se arrowAngle)
    """
    # dimensioni dal testo (se presente)
    if label:
        fm = QFontMetrics(base_font)
        lines = label.splitlines() or [label]
        maxw  = max(fm.horizontalAdvance(ln) for ln in lines)
        htot  = sum(fm.height() for _ in lines)
        w = max(80, maxw + 2*pad_x)
        h = max(40, htot + 2*pad_y)
    else:
        w, h = 120, 60

    # centro (ox, oy)
    if master or linker is None:
        ox, oy = 0.0, 0.0
    else:
        dir = 1 if direction == "RIGHT" else (-1 if direction == "LEFT" else 0)
        lx, ly, lw, lh = linker
        offs = lw/2.0 + w/2.0 if arrowAngle else 0.0
        dist = distance if distance is not None else 90.0
        rad  = math.radians(angle)
        ox   = lx + offs*dir + dist *dir* math.cos(rad)
        oy   = ly + dist * math.sin(rad)

    # shape glow (due layer) + bordo netto
    x = ox - w/2.0; y = oy - h/2.0

    #glow1 = QGraphicsRectItem(x, y, w, h); glow1.setPen(mk_pen(color, 18, alpha=45)); view.addItem(glow1)
    for i in range(1, 9):  #12*2=24 6*3=18 6*2=12
        glow = QGraphicsRectItem(x, y, w, h); glow.setPen(mk_pen(color, 2*i,alpha=(50-5*i))); view.addItem(glow) #gli oggetti glow si sovrappongono
    
    core  = QGraphicsRectItem(x, y, w, h); core.setPen(mk_pen(color, 2.6,  alpha=255)); view.addItem(core)

    if label:
        add_centered_text(view, ox, oy, label, base_font)
    if not master:
        if direction=="RIGHT":
            add_arrow_with_glow(view, *right_edge(*linker), *left_edge(ox,oy,w,h))
        elif direction=="LEFT":
            add_arrow_with_glow(view, *left_edge(*linker), *right_edge(ox,oy,w,h))
    return (ox, oy, w, h)

# ---------- freccia con glow ----------
def add_arrow_with_glow(view, x0, y0, x1, y1, color=NEON):
    # shaft glow
    for i in range(1, 9):  #12*2=24 6*3=18 6*2=12
        l2 = QGraphicsLineItem(x0, y0, x1, y1); l2.setPen(mk_pen(color, 2*i,alpha=(50-5*i))); view.addItem(l2)
        
    

    # shaft netto
    lc = QGraphicsLineItem(x0, y0, x1, y1); lc.setPen(mk_pen(color, 2.8, alpha=255)); view.addItem(lc)

    # punta triangolare (con alone)
    ang = math.atan2(y1 - y0, x1 - x0)
    head_len, head_w = 18.0, 12.0

    bx = x1 - math.cos(ang)*head_len
    by = y1 - math.sin(ang)*head_len
    cost=8
    cost2=1.1
    left  = QPointF(bx + math.cos(ang + math.pi/cost)*head_w/cost2,
                    by + math.sin(ang + math.pi/cost)*head_w/cost2)
    right = QPointF(bx + math.cos(ang - math.pi/cost)*head_w/cost2,
                    by + math.sin(ang - math.pi/cost)*head_w/cost2)
    tip   = QPointF(x1, y1)

    # One path, two segments: tip→left and tip→right
    path = QPainterPath(tip)
    path.lineTo(left)
    path.moveTo(tip)
    path.lineTo(right)

    for i in range(1, 9):  #12*2=24 6*3=18 6*2=12

       head_glow = QGraphicsPathItem(path); head_glow.setPen(mk_pen(color, 2*i,alpha=(50-5*i))); view.addItem(head_glow) #gli oggetti glow si sovrappongono
    
    #head_glow = QGraphicsPolygonItem(poly); head_glow.setPen(mk_pen(color, 10, alpha=60)); view.addItem(head_glow)
    head = QGraphicsPathItem(path); head.setPen(mk_pen(color,  2.6, alpha=255));   view.addItem(head)

# ---------- main ----------
def main():
    app  = QApplication(sys.argv)
    win  = pg.GraphicsLayoutWidget(show=True, title="Diagramma Glow (Qt5)")
    view = win.addViewBox()
    view.setAspectLocked(True)               # unità uguali su x/y (45° corretti)
    view.setBackgroundColor(QBrush(BG))
    view.invertY(False)                      # Y verso l’alto (testi non rovesciati)

    base_font = QFont("Arial", 12)

    # --- nodi come nel tuo script ---
    C1 = rect_with_glow(view, label="Immagine generata", master=True, color=NEON, base_font=base_font)

    C2 = rect_with_glow(view, label="Vettore vicino a molte immagini,\nconcetto che il modello ha imparato [forse a memoria]",
                        linker=C1, angle=60,direction="RIGHT", color=NEON, base_font=base_font)
    C3 = rect_with_glow(view, label="maggior parte del contributo da queste immagini",
                        linker=C2, angle=0, color=NEON, base_font=base_font)

    C4 = rect_with_glow(view, label="Vettore lontano, concetto nuovo",
                        linker=C1, angle=-60, color=NEON, base_font=base_font)
    C5 = rect_with_glow(view, label="Se il risultato non è gibberish\n (possibile nei modelli piu avanzati),\nscomposizione in componenti",
                        linker=C4, angle=0, color=NEON, base_font=base_font)

    C6 = rect_with_glow(view, label="utilizzare CLIP per estrarre vett. immagine finale",
                        linker=C5, angle=45, color=NEON, base_font=base_font)
    C7 = vertical_rect(view, label="[???cpu] utilizzare blip per generare un caption,\npotendolo verificare con clip",
                        linker=C6, angle=0,distance=30, color=NEON, base_font=base_font)
    C8 = vertical_rect(view, label="analizzare tramite analisi semantica [!Ai]\n per separare il token testuale equivalente\n in concetti singoli ",
                        linker=C7, angle=0,distance=30, color=NEON, base_font=base_font)
    C9 = vertical_rect(view, label="reimmettere i token singoli in CLIP",
                        linker=C8, angle=0,distance=30, color=NEON, base_font=base_font)
    C9b = rect_with_glow(view, label="risultato: scomposizione dei singoli concetti usati dall'ia\nper generare l'immagine",linker=C9, angle=0, color=NEON, base_font=base_font)

    # Frecce aggiuntive
    #add_arrow_with_glow(view, *right_edge(*C6), *left_edge(*C9))
    #add_arrow_with_glow(view, *right_edge(*C8), *left_edge(*C9))

    C10=rect_with_glow(view, label="considerare le distanze in gioco\n per decidere se approccio 1 o approccio 2",
                        linker=C3,  distance=500,angle=0, color=NEON, base_font=base_font)
    
    C11=vertical_rect(view, label="opzionale: \nstudio 0-shot accuracy",
                        linker=C5,angle=0,direction="DOWN", color=NEON, base_font=base_font)
    add_arrow_with_glow(view, *bottom_edge(*C9b), *top_edge(*C10))

    
    # massimizza finestra
    try:
        win.showMaximized()
    except Exception:
        pass

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
