r"""
FIGURA 1 DEL PAPER: PANORAMA DEL AGENTE, POR ETAPAS. (2026-09-15, v2)

QUE ES
------
La figura que la catedra exige desde la Semana 3 --el agente como cuatro piezas y
un lazo, con la persona y los puntos de medicion-- dibujada con el lenguaje de la
**figura de ejemplo que dio la catedra** (el panorama del marco PINN-GRN):

    - dos columnas de ancho, flujo horizontal por ETAPAS
    - cada etapa es un panel con tinte suave, titulo y **un dibujo dentro**,
      no una lista de texto
    - flechas rotuladas entre etapas
    - una banda inferior que cuenta la fase de aprendizaje, con datos reales
    - pie largo, que se sostiene solo

LOS NUMEROS SON REALES
----------------------
La curva de la banda inferior se lee de `out/PAPER_CURVA/curva.json` (la misma que
alimenta la Fig. de curva de aprendizaje). Las barras de la etapa 4 son las cifras
de la politica de planta. Nada esta dibujado a ojo.

AVISO SOBRE LAS CIFRAS
----------------------
Las cifras de jornada nueva salen de las 112 hojas del 11/09, que son **jornada de
desarrollo**, no conjunto de prueba: desde el 12/09 se usaron para aceptar o
rechazar mejoras (Tabaco.md 79.6). Cuando llegue la tanda nueva --la del acuerdo
entre tecnicos, que si es test virgen y trae BANDA-- se sustituyen aqui arriba.

Uso:  python scripts/paper_figura1.py
"""
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
from matplotlib.patches import (Circle, FancyArrowPatch,            # noqa: E402
                                FancyBboxPatch, PathPatch, Polygon)
from matplotlib.path import Path as MplPath                         # noqa: E402
from rutas import REPO_ROOT  # repository root

OUT = REPO_ROOT / "out"
DEST = OUT / "PAPER_FIGURAS"
COPIA = REPO_ROOT / "results" / "figuras"

# --- IDIOMA (2026-09-20) -----------------------------------------------------
# El paper se envia en ingles, asi que la figura tiene que existir en los dos
# idiomas. En vez de duplicar el guion --que es como las dos versiones acaban
# diciendo cosas distintas-- se traduce en el sitio con esta tabla.
IDIOMA = os.environ.get("FIG_IDIOMA", "es")

EN = {
    "Captura": "Capture",
    "1 foto por hoja": "one photograph per leaf",
    "luz constante · cinta como escala": "constant lighting · tape sets the scale",
    "{} hojas · {} clases": "{} leaves · {} grades",
    "Percepción": "Perception",
    "rasgos físicos medidos": "measured physical features",
    "máscara + defectos": "mask + defects",
    "{} rasgos de calidad · ampliable": "{} grade features · extensible",
    "Clasificación": "Classification",
    "tres preguntas encadenadas": "three chained questions",
    "1 · ¿sudada o verdosa?": "1 · sweated or greenish?",
    "sí → banda": "yes → binder",
    "sudado y color": "sweating and colour",
    "2 · ¿un solo lado dañado?": "2 · damage on one side only?",
    "no → capa": "no → wrapper",
    "forma del daño": "shape of the damage",
    "3 · ¿qué lado sirve?": "3 · which half is usable?",
    "XL izq  /  XR der": "XL left  /  XR right",
    "solo asimetrías": "asymmetries only",
    "cada pregunta mira\nsolo sus rasgos": "each question sees\nonly its own features",
    "Decisión": "Decision",
    "con regla de abstención": "with an abstention rule",
    "umbral τ": "threshold τ",
    "deriva": "defers",
    "decide": "decides",
    "confianza calibrada →": "calibrated confidence →",
    "decide\ntodas": "decides\nall",
    "con\nabstención": "with\nabstention",
    "decide el {:.1f} % de las hojas": "decides {:.1f} % of the leaves",
    "Actuación": "Actuation",
    "brazo simulado · RoboDK": "simulated arm · RoboDK",
    "Ca": "Wr", "Ba": "Bi", "XL": "XL", "XR": "XR", "Rev": "Rev",
    "Rev = revisión de un inspector": "Rev = sent to a human inspector",
    "4 bandejas + revisión": "4 bins + review",
    "8 bandejas + revisión": "8 bins + review",
    "H = Habano · C = Connecticut": "H = Habano · C = Connecticut",
    "ETAPA {}": "STAGE {}",
    "foto": "photo", "rasgos": "features", "clase": "grade", "acción": "action",
    "AGENTE": "AGENT",
    "reentrena": "retrains",
    "el inspector revisa\nlas hojas derivadas":
        "the inspector reviews\nthe deferred leaves",
    "REALIMENTACIÓN · cada hoja que el inspector corrige vuelve al modelo":
        "FEEDBACK · every leaf the inspector corrects goes back into the model",
    "+{:.1f} puntos": "+{:.1f} points",
    "intervalo de confianza 95 % [+1,8, +4,0]":
        "95 % confidence interval [+1.8, +4.0]",
    "eje Y: acierto sobre una captura nueva\n"
    "eje X: cuántas hojas de esa misma captura\n"
    "se le corrigieron antes de reentrenar\n"
    "la banda gris es el intervalo del 95 %\n"
    "sobre 40 repeticiones":
        "y axis: accuracy on a new capture\n"
        "x axis: how many leaves from that capture\n"
        "were corrected before retraining\n"
        "the grey band is the 95 % interval\n"
        "over 40 repetitions",
}


def T(t):
    """Traduce un rotulo si la figura se esta dibujando en ingles."""
    if IDIOMA != "en":
        return t
    if t in EN:
        return EN[t]
    return t


def num(t):
    """El ingles usa punto decimal; el espanol, coma."""
    return t.replace(",", ".") if IDIOMA == "en" else t


SUFIJO = "_en" if IDIOMA == "en" else ""

TINTA = "#1a1a1a"
GRIS = "#6f6f6f"
LINEA = "#b9b3a6"
TINTES = ["#f1efe8", "#f1efe8", "#e7eef7", "#fbeeda", "#eaf1ea"]

CIFRAS = dict(
    n_rasgos=69, n_rasgos_var=17, n_hojas=527,
    sin_rechazo=84.8,        # decide las 112 enteras (95/112)
    con_rechazo=91.1,        # acierto sobre las que decide, umbral 0,60
    cobertura=70.5,          # que porcentaje decide con el umbral 0,60
)

# --- lienzo, en milimetros ---------------------------------------------------
ANCHO = 182.0            # dos columnas IEEE
ALTO = 92.0
MARGEN = 3.0
Y_ETIQUETA = 5.0
Y_PANEL, H_PANEL = 7.5, 50.0
Y_BANDA, H_BANDA = 66.0, 20.0


def estilo():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 6.4,
        "text.color": TINTA,
        "figure.dpi": 400,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
    })


def mm_a_fig(x, y, w, h):
    """Rectangulo en mm (y hacia abajo) -> [left, bottom, w, h] de figura."""
    return [x / ANCHO, (ALTO - y - h) / ALTO, w / ANCHO, h / ALTO]


def panel(ax, x, y, w, h, tinte):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=0.5, edgecolor=LINEA, facecolor=tinte, zorder=1,
        clip_on=False))


def titulo_panel(ax, x, y, w, texto, sub=None):
    ax.text(x + w / 2, y + 3.0, texto, fontsize=7, fontweight="bold",
            ha="center", va="top", zorder=4)
    if sub:
        ax.text(x + w / 2, y + 6.6, sub, fontsize=5.8, color=GRIS,
                ha="center", va="top", zorder=4)


def etiqueta_etapa(ax, x, w, n):
    ax.text(x + w / 2, Y_ETIQUETA, T("ETAPA {}").format(n), fontsize=6,
            color=GRIS, ha="center", va="bottom", zorder=4,
            fontweight="bold")


def dibuja_hoja(ax, cx, cy, alto, relleno="#dfd8c6", borde="#8a8272",
                venas=True, zorder=3, lw=0.5, guion=None):
    """Silueta de hoja de capa: base ancha y punta, con nervadura."""
    a = alto / 2.0
    an = alto * 0.34
    verts = [(cx, cy + a),                                   # punta
             (cx + an, cy + a * 0.35), (cx + an * 0.95, cy - a * 0.45),
             (cx, cy - a),                                   # base
             (cx - an * 0.95, cy - a * 0.45), (cx - an, cy + a * 0.35),
             (cx, cy + a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=relleno,
                           edgecolor=borde, linewidth=lw, zorder=zorder,
                           linestyle=guion if guion else "solid",
                           clip_on=False))
    if venas:
        ax.plot([cx, cx], [cy - a * 0.92, cy + a * 0.9], color=borde,
                linewidth=0.4, zorder=zorder + 1, clip_on=False)
        for t in (-0.45, -0.1, 0.28):
            y0 = cy + a * t
            for s in (-1, 1):
                ax.plot([cx, cx + s * an * 0.72], [y0, y0 + a * 0.22],
                        color=borde, linewidth=0.3, zorder=zorder + 1,
                        clip_on=False)


def flecha(ax, x0, y0, x1, y1, etiqueta=None, discontinua=False, dy=-1.6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.5,
        linewidth=0.65, color=TINTA if not discontinua else GRIS,
        linestyle="solid" if not discontinua else (0, (2.2, 1.6)),
        shrinkA=0, shrinkB=0, zorder=5, clip_on=False))
    if etiqueta:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, etiqueta, fontsize=5.6,
                color=GRIS, ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                          edgecolor="none"))


# =============================================================================
#  las cinco etapas
# =============================================================================
def etapa1(fig, ax, x, w):
    """Captura: la hoja sobre el meson, una foto, con su escala."""
    titulo_panel(ax, x, Y_PANEL, w, T("Captura"), T("1 foto por hoja"))
    cx = x + w / 2
    # meson
    ax.add_patch(FancyBboxPatch(
        (x + 3.0, Y_PANEL + 15.0), w - 6.0, 24.0,
        boxstyle="round,pad=0,rounding_size=1.0", linewidth=0.45,
        edgecolor=LINEA, facecolor="white", zorder=2, clip_on=False))
    dibuja_hoja(ax, cx, Y_PANEL + 25.5, 17.5)
    # CINTA METRICA, VERTICAL (2026-09-17, a peticion del experto: en la planta la
    # cinta se coloca a lo largo de la hoja, no cruzada, porque lo que se mide
    # es el LARGO de la lamina).
    cin_x = x + w - 8.4          # a la derecha de la hoja, dentro del meson
    cin_y0, cin_alto = Y_PANEL + 17.0, 20.0
    ax.add_patch(FancyBboxPatch(
        (cin_x, cin_y0), 2.6, cin_alto,
        boxstyle="square,pad=0", linewidth=0.4, edgecolor="#8a8272",
        facecolor="#f6f1e2", zorder=4, clip_on=False))
    for i in range(1, 10):                      # graduaciones horizontales
        yt = cin_y0 + cin_alto * i / 10.0
        largo = 1.6 if i % 5 == 0 else 1.0      # la marca de cada 5 es mas larga
        ax.plot([cin_x, cin_x + largo], [yt, yt], color="#8a8272",
                linewidth=0.3, zorder=5, clip_on=False)
    ax.text(x + w / 2, Y_PANEL + 40.2, T("luz constante · cinta como escala"),
            fontsize=5.4, color=GRIS, ha="center", va="top", zorder=4)
    ax.text(x + w / 2, Y_PANEL + 44.0,
            T("{} hojas · {} clases").format(CIFRAS["n_hojas"], 4),
            fontsize=5.6, ha="center", va="top", zorder=4)


def etapa2(fig, ax, x, w):
    """Percepcion: mascara y rasgos medidos."""
    titulo_panel(ax, x, Y_PANEL, w, T("Percepción"), T("rasgos físicos medidos"))
    cx = x + w / 2
    dibuja_hoja(ax, cx, Y_PANEL + 21.0, 17.0, relleno="white")
    # contorno de la mascara, con la forma de la hoja
    dibuja_hoja(ax, cx, Y_PANEL + 21.0, 19.6, relleno="none",
                borde="#4a6fa5", venas=False, lw=0.5, zorder=5,
                guion=(0, (1.8, 1.4)))
    # defectos localizados
    for dx, dy, r in ((-1.9, 2.4, 0.85), (1.6, -1.2, 1.15), (0.4, 4.6, 0.6)):
        ax.add_patch(Circle((cx + dx, Y_PANEL + 21 + dy), r, facecolor="#b9553f",
                            edgecolor="none", alpha=0.75, zorder=6,
                            clip_on=False))
    ax.text(cx, Y_PANEL + 32.0, T("máscara + defectos"), fontsize=5.4, color=GRIS,
            ha="center", va="top", zorder=4)
    # barras de rasgos
    rng = np.random.default_rng(4)
    vals = rng.uniform(0.25, 1.0, 9)
    x0, wbar = x + 4.0, (w - 8.0) / 9.0
    for i, v in enumerate(vals):
        ax.add_patch(FancyBboxPatch(
            (x0 + i * wbar + 0.25, Y_PANEL + 43.0 - 6.0 * v),
            wbar - 0.5, 6.0 * v, boxstyle="square,pad=0", linewidth=0,
            facecolor="#7d8a99", zorder=4, clip_on=False))
    ax.plot([x + 4.0, x + w - 4.0], [Y_PANEL + 43.0] * 2, color=GRIS,
            linewidth=0.4, zorder=4, clip_on=False)
    ax.text(cx, Y_PANEL + 44.0,
            T("{} rasgos de calidad · ampliable").format(CIFRAS["n_rasgos"]),
            fontsize=5.6, ha="center", va="top", zorder=4)


def etapa3(fig, ax, x, w):
    """Las tres preguntas encadenadas, cada una con sus propios rasgos."""
    titulo_panel(ax, x, Y_PANEL, w, T("Clasificación"), T("tres preguntas encadenadas"))
    cx = x + w / 2
    PREG = [[T(z) for z in fila] for fila in (
        ("1 · ¿sudada o verdosa?", "sí → banda", "sudado y color"),
        ("2 · ¿un solo lado dañado?", "no → capa", "forma del daño"),
        ("3 · ¿qué lado sirve?", "XL izq  /  XR der", "solo asimetrías"))]
    ys = (Y_PANEL + 14.0, Y_PANEL + 25.0, Y_PANEL + 36.0)
    for (preg, salida, rasgos), yy in zip(PREG, ys):
        ax.add_patch(FancyBboxPatch(
            (cx - 13.6, yy - 4.1), 27.2, 8.2,
            boxstyle="round,pad=0,rounding_size=0.8", linewidth=0.45,
            edgecolor="#7f8a99", facecolor="white", zorder=4, clip_on=False))
        ax.text(cx, yy - 2.2, preg, fontsize=5.0, ha="center", va="center",
                zorder=5)
        ax.text(cx, yy + 0.7, salida, fontsize=4.9, ha="center", va="center",
                color="#3a3a3a", zorder=5)
        ax.text(cx, yy + 2.9, rasgos, fontsize=4.5, ha="center", va="center",
                color=GRIS, style="italic", zorder=5)
    for a, b in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((cx, a + 4.1), (cx, b - 4.1),
                                     arrowstyle="-|>", mutation_scale=5,
                                     linewidth=0.45, color="#7f8a99", zorder=3,
                                     clip_on=False))
    ax.text(cx, Y_PANEL + 43.2, T("cada pregunta mira\nsolo sus rasgos"),
            fontsize=5.2, color=GRIS, ha="center", va="top", zorder=4,
            linespacing=1.35)


def etapa4(fig, ax, x, w):
    """Decision con opcion de rechazo."""
    titulo_panel(ax, x, Y_PANEL, w, T("Decisión"), T("con regla de abstención"))
    cx = x + w / 2
    # eje de confianza con umbral
    xa, xb, yb = x + 4.5, x + w - 4.5, Y_PANEL + 19.0
    # el corte va donde de verdad esta: lo que NO firma es 100 - cobertura
    xt = xa + (xb - xa) * (1.0 - CIFRAS["cobertura"] / 100.0)
    ax.add_patch(FancyBboxPatch((xa, yb - 2.2), xt - xa, 4.4,
                                boxstyle="square,pad=0", linewidth=0,
                                facecolor="#e2c9a0", zorder=3, clip_on=False))
    ax.add_patch(FancyBboxPatch((xt, yb - 2.2), xb - xt, 4.4,
                                boxstyle="square,pad=0", linewidth=0,
                                facecolor="#cfd9c8", zorder=3, clip_on=False))
    ax.plot([xt, xt], [yb - 3.4, yb + 3.4], color=TINTA, linewidth=0.7,
            zorder=5, clip_on=False)
    ax.text(xt, yb - 3.8, T("umbral τ"), fontsize=5.4, ha="center", va="bottom",
            zorder=5)
    ax.text((xa + xt) / 2, yb + 0.1, T("deriva"), fontsize=5.2, ha="center",
            va="center", zorder=5)
    ax.text((xt + xb) / 2, yb + 0.1, T("decide"), fontsize=5.2, ha="center",
            va="center", zorder=5)
    ax.text(cx, yb + 5.0, T("confianza calibrada →"), fontsize=5.4, color=GRIS,
            ha="center", va="top", zorder=5)
    # barras: sin rechazo contra con rechazo
    eje = fig.add_axes(mm_a_fig(x + 6.0, Y_PANEL + 26.0, w - 12.0, 11.0))
    eje.set_facecolor("none")
    vals = [CIFRAS["sin_rechazo"], CIFRAS["con_rechazo"]]
    eje.bar([0, 1], vals, width=0.58, color=["#a9b3bd", "#c58f3f"],
            edgecolor=TINTA, linewidth=0.4)
    for i, v in enumerate(vals):
        eje.text(i, v + 0.6, num("{:.1f}".format(v).replace(".", ",")),
                 fontsize=5.4, ha="center", va="bottom")
    # el techo se calcula: con 94.5 el limite fijo de 88 dejaba la barra fuera
    # del eje y su rotulo se montaba sobre el texto de arriba.
    eje.set_ylim(75, max(vals) + 4.0)
    eje.set_xlim(-0.6, 1.6)
    eje.set_xticks([0, 1])
    eje.set_xticklabels([T("decide\ntodas"), T("con\nabstención")],
                        fontsize=5.2)
    eje.set_yticks([])
    eje.tick_params(axis="x", length=0, pad=1)
    for lado in ("top", "right", "left"):
        eje.spines[lado].set_visible(False)
    eje.spines["bottom"].set_linewidth(0.4)
    eje.spines["bottom"].set_color(GRIS)
    ax.text(cx, Y_PANEL + 44.0,
            num(T("decide el {:.1f} % de las hojas").format(
                CIFRAS["cobertura"]).replace(".", ",")),
            fontsize=5.6, ha="center", va="top", zorder=4)


def etapa5(fig, ax, x, w):
    """Actuacion: brazo simulado, ocho bandejas y revision humana."""
    titulo_panel(ax, x, Y_PANEL, w, T("Actuación"), T("brazo simulado · RoboDK"))
    cx = x + w / 2
    # BRAZO ARTICULADO (2026-09-17, a peticion del experto: el trazo anterior eran
    # dos rayas y no se reconocia como robot). Silueta de manipulador industrial
    # de seis ejes, que es lo que hay en la celda: KUKA IONTEC KR 120 R2700.
    bx, by = x + 5.5, Y_PANEL + 27.6          # (y crece hacia ABAJO)
    hombro = (bx, by - 4.6)
    codo = (bx + 5.4, by - 10.6)
    muneca = (bx + 12.6, by - 7.4)
    brida = (bx + 14.6, by - 6.2)

    def _eslabon(p0, p1, grueso, relleno="#d8d2c4"):
        """Un tramo del brazo con cuerpo, no una raya."""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        px, py = -dy / n * grueso / 2.0, dx / n * grueso / 2.0
        ax.add_patch(Polygon(
            [(p0[0] + px, p0[1] + py), (p1[0] + px, p1[1] + py),
             (p1[0] - px, p1[1] - py), (p0[0] - px, p0[1] - py)],
            closed=True, facecolor=relleno, edgecolor=TINTA, linewidth=0.5,
            zorder=5, clip_on=False))

    # zocalo y base que gira
    ax.add_patch(FancyBboxPatch(
        (bx - 4.0, by - 1.4), 8.0, 1.8,
        boxstyle="round,pad=0,rounding_size=0.3", linewidth=0.5,
        edgecolor=TINTA, facecolor="#b9b3a6", zorder=4, clip_on=False))
    ax.add_patch(Polygon([(bx - 3.0, by - 1.4), (bx + 3.0, by - 1.4),
                          (bx + 2.1, by - 4.6), (bx - 2.1, by - 4.6)],
                         closed=True, facecolor="#d8d2c4", edgecolor=TINTA,
                         linewidth=0.5, zorder=4, clip_on=False))
    # eslabones
    _eslabon(hombro, codo, 2.6)
    _eslabon(codo, muneca, 2.2)
    _eslabon(muneca, brida, 1.4, "#b9b3a6")
    # ejes
    for p, r in ((hombro, 1.5), (codo, 1.2), (muneca, 0.95)):
        ax.add_patch(Circle(p, r, facecolor="white", edgecolor=TINTA,
                            linewidth=0.5, zorder=6, clip_on=False))
        ax.add_patch(Circle(p, r * 0.32, facecolor=TINTA, edgecolor="none",
                            zorder=7, clip_on=False))
    # ventosa y la hoja que lleva
    ax.add_patch(Polygon([(brida[0] - 0.5, brida[1] + 0.9),
                          (brida[0] + 1.7, brida[1] + 0.9),
                          (brida[0] + 1.1, brida[1] + 2.4),
                          (brida[0] + 0.1, brida[1] + 2.4)],
                         closed=True, facecolor="#7f8a99", edgecolor=TINTA,
                         linewidth=0.4, zorder=6, clip_on=False))
    dibuja_hoja(ax, brida[0] + 0.6, brida[1] + 5.4, 5.0, lw=0.4, venas=False)
    # OCHO bandejas de grado --una por grado y variedad-- mas la de revision.
    # 2026-09-22: vuelve a 8 + 1 porque es la celda que existe de verdad en
    # RoboDK (`celda_robodk.py`: 1-4 Connecticut, 5-8 Habano, 9 revision). El
    # 20/09 se habia dibujado 4 + 1 al pasar el paper a Connecticut, y el dibujo
    # dejo de coincidir con la estacion.
    # 2026-09-22, tarde: las cajas se salian del panel por la derecha --el experto lo
    # marco en la figura--. El panel mide 28,64 de ancho y el bloque media 29,0
    # sin contar los rotulos H/C. Ahora mide 24,3 con ellos, centrado de verdad.
    y0 = Y_PANEL + 32.0
    anb, altb, sep, sepf = 3.8, 3.3, 0.6, 1.0
    ancho_rej = 4 * anb + 3 * sep
    hueco, rotulo = 1.3, 2.2
    x0 = cx - (rotulo + ancho_rej + hueco + anb) / 2 + rotulo
    for fila, var in enumerate(("H", "C")):
        for col, nombre in enumerate(T(z) for z in ("Ca", "Ba", "XL", "XR")):
            ax.add_patch(FancyBboxPatch(
                (x0 + col * (anb + sep), y0 + fila * (altb + sepf)), anb, altb,
                boxstyle="round,pad=0,rounding_size=0.5", linewidth=0.45,
                edgecolor="#7f8a99", facecolor="white", zorder=4,
                clip_on=False))
            ax.text(x0 + col * (anb + sep) + anb / 2,
                    y0 + fila * (altb + sepf) + altb / 2, nombre,
                    fontsize=4.2, ha="center", va="center", zorder=5)
        ax.text(x0 - 0.9, y0 + fila * (altb + sepf) + altb / 2, var,
                fontsize=4.6, color=GRIS, ha="right", va="center", zorder=5)
    # la bandeja de revision, alta como las dos filas
    xrev = x0 + ancho_rej + hueco
    ax.add_patch(FancyBboxPatch(
        (xrev, y0), anb, 2 * altb + sepf,
        boxstyle="round,pad=0,rounding_size=0.5", linewidth=0.45,
        edgecolor="#7f8a99", facecolor="#f2ede2", zorder=4, clip_on=False))
    ax.text(xrev + anb / 2, y0 + (2 * altb + sepf) / 2, T("Rev"),
            fontsize=4.2, ha="center", va="center", zorder=5)
    ax.text(cx, Y_PANEL + 41.4, T("H = Habano · C = Connecticut"),
            fontsize=5.0, color=GRIS, ha="center", va="top", zorder=4)
    ax.text(cx, Y_PANEL + 44.0, T("8 bandejas + revisión"), fontsize=5.6,
            ha="center", va="top", zorder=4)


# =============================================================================
def banda_aprendizaje(fig, ax):
    """Banda inferior: la fase de aprendizaje, con la curva real."""
    ax.add_patch(FancyBboxPatch(
        (MARGEN, Y_BANDA), ANCHO - 2 * MARGEN, H_BANDA,
        boxstyle="round,pad=0,rounding_size=1.6", linewidth=0.5,
        edgecolor=LINEA, facecolor="#f7f6f2", zorder=1, clip_on=False))
    ax.text(MARGEN + 3.0, Y_BANDA + 3.2,
            T("REALIMENTACIÓN · cada hoja que el inspector corrige vuelve al modelo"),
            fontsize=6.2, fontweight="bold", ha="left", va="top", zorder=4)

    d = json.load(open(OUT / "PAPER_CURVA" / "curva.json", encoding="utf-8"))
    ks = d["ks"]
    med = np.array([d["media"][str(k)] for k in ks]) * 100
    lo = np.array([d["ic"][str(k)][0] for k in ks]) * 100
    hi = np.array([d["ic"][str(k)][1] for k in ks]) * 100

    # bajado 0.9 mm y un poco mas plano para que el rotulo del eje Y no roce el
    # titulo de la banda y el del eje X caiga DENTRO del recuadro (2026-09-17)
    eje = fig.add_axes(mm_a_fig(ANCHO / 2 - 34.0, Y_BANDA + 6.2, 68.0, 10.0))
    eje.set_facecolor("none")
    eje.fill_between(ks, lo, hi, color="#d9d9d9", linewidth=0)
    eje.plot(ks, med, color=TINTA, linewidth=0.9, marker="o", markersize=1.9)
    eje.set_xlim(-2, 58)
    # margen abajo y arriba para que las dos etiquetas de la curva quepan
    # SIN montarse sobre la linea ni sobre el eje (2026-09-20)
    eje.set_ylim(min(lo) - 2.0, max(hi) + 1.4)
    eje.set_xticks([0, 16, 32, 48, 56])
    eje.tick_params(labelsize=5.0, length=1.6, pad=2.2, colors=GRIS)
    # EJES CON SU SIGNIFICADO (2026-09-17, a peticion del experto: antes el eje Y no
    # tenia ni marcas ni nombre, asi que la curva no se podia leer sola).
    y0, y1 = eje.get_ylim()
    eje.set_yticks([round(v, 1) for v in np.linspace(y0 + 1.6, y1 - 0.6, 3)])
    # SIN rotulos de eje. Los dos se montaban sobre lineas --el de Y sobre el
    # titulo de la banda, el de X sobre el borde del recuadro-- y el bloque de
    # texto de la izquierda ya dice que es cada eje.
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("bottom", "left"):
        eje.spines[lado].set_linewidth(0.4)
        eje.spines[lado].set_color(GRIS)
    CAJA = dict(boxstyle="square,pad=0.14", facecolor="white",
                edgecolor="none", alpha=0.92)
    eje.annotate(num("{:.1f} %".format(med[0]).replace(".", ",")), (ks[0], med[0]),
                 fontsize=5.4, textcoords="offset points", xytext=(3, -10),
                 ha="left", color=TINTA, bbox=CAJA, zorder=6)
    eje.annotate(num("{:.1f} %".format(med[-1]).replace(".", ",")),
                 (ks[-1], med[-1]),
                 fontsize=5.4, textcoords="offset points", xytext=(-2, 7),
                 ha="right", color=TINTA, bbox=CAJA, zorder=6)
    ax.text(ANCHO / 2 + 36.0, Y_BANDA + 8.0,
            num(T("+{:.1f} puntos").format(med[-1] - med[0]).replace(".", ",")),
            fontsize=6.6,
            fontweight="bold", ha="left", va="center", zorder=4)
    ax.text(ANCHO / 2 + 36.0, Y_BANDA + 11.6, T("intervalo de confianza 95 % [+1,8, +4,0]"),
            fontsize=5.4, color=GRIS, ha="left", va="center", zorder=4)
    ax.text(MARGEN + 3.0, Y_BANDA + 11.6,
            T("eje Y: acierto sobre una captura nueva\n"
              "eje X: cuántas hojas de esa misma captura\n"
              "se le corrigieron antes de reentrenar\n"
              "la banda gris es el intervalo del 95 %\n"
              "sobre 40 repeticiones"),
            fontsize=5.2, color=GRIS, ha="left", va="center", zorder=4,
            linespacing=1.45)


def dibuja():
    estilo()
    fig = plt.figure(figsize=(ANCHO / 25.4, ALTO / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ANCHO)
    ax.set_ylim(ALTO, 0)
    ax.axis("off")

    n = 5
    hueco = 8.2
    w = (ANCHO - 2 * MARGEN - (n - 1) * hueco) / n
    xs = [MARGEN + i * (w + hueco) for i in range(n)]

    for i, xx in enumerate(xs):
        panel(ax, xx, Y_PANEL, w, H_PANEL, TINTES[i])
        etiqueta_etapa(ax, xx, w, i + 1)

    etapa1(fig, ax, xs[0], w)
    etapa2(fig, ax, xs[1], w)
    etapa3(fig, ax, xs[2], w)
    etapa4(fig, ax, xs[3], w)
    etapa5(fig, ax, xs[4], w)

    ym = Y_PANEL + H_PANEL / 2
    rotulos = [T(z) for z in ("foto", "rasgos", "clase", "acción")]
    for i in range(4):
        x0 = xs[i] + w + 0.6
        flecha(ax, x0, ym, xs[i + 1] - 0.6, ym, rotulos[i], dy=0.8)

    banda_aprendizaje(fig, ax)

    # la persona y el lazo: de la etapa 5 a la banda, y de la banda al modelo
    x_op = xs[4] + w * 0.5
    flecha(ax, x_op, Y_PANEL + H_PANEL + 0.8, x_op, Y_BANDA - 0.8,
           None, discontinua=True)
    ax.text(x_op - 1.6, (Y_PANEL + H_PANEL + Y_BANDA) / 2,
            T("el inspector revisa\nlas hojas derivadas"), fontsize=5.4,
            color=GRIS, ha="right", va="center", zorder=6, linespacing=1.25,
            bbox=dict(boxstyle="square,pad=0.22", facecolor="white",
                      edgecolor="none"))
    x_mod = xs[2] + w * 0.5
    flecha(ax, x_mod, Y_BANDA - 0.8, x_mod, Y_PANEL + H_PANEL + 0.8,
           None, discontinua=True)
    ax.text(x_mod - 1.4, (Y_PANEL + H_PANEL + Y_BANDA) / 2,
            T("reentrena"), fontsize=5.4, color=GRIS, ha="right", va="center",
            zorder=5)

    # marco del agente: de la etapa 2 a la 5
    ax.add_patch(FancyBboxPatch(
        (xs[1] - 2.6, Y_PANEL - 1.8), xs[4] + w + 2.6 - (xs[1] - 2.6),
        H_PANEL + 4.4, boxstyle="round,pad=0,rounding_size=2.0",
        linewidth=0.55, edgecolor=GRIS, facecolor="none",
        linestyle=(0, (1.6, 1.8)), zorder=0, clip_on=False))
    ax.text(xs[1] + 1.0, Y_PANEL + H_PANEL + 2.6, T("AGENTE"), fontsize=6,
            color=GRIS, fontweight="bold", ha="left", va="center", zorder=2,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                      edgecolor="none"))
    return fig


PIE = (
    "Fig. 1. Panorama del sistema. La hoja se fotografía una sola vez sobre el "
    "mesón (Etapa 1); la percepción separa la hoja del fondo y mide {n} rasgos "
    "físicos --agujeros, roturas, mordiscos, manchas, sudado, reparto del daño, "
    "tamaño, color y forma de la punta-- en lugar de operar sobre los píxeles "
    "(Etapa 2); el grado se resuelve en tres preguntas encadenadas, y cada una "
    "mira solo los rasgos que su regla de oficio nombra (Etapa 3); la regla de "
    "abstención decide la hoja cuando la confianza supera el umbral y deriva el "
    "resto a un inspector (Etapa 4); y el brazo simulado la lleva a su bandeja "
    "de destino --una por grado y variedad-- o a la de revisión (Etapa 5). Las correcciones del inspector "
    "realimentan el modelo: sobre una captura nueva el acierto sube de {a} % "
    "a {b} % tras 56 hojas corregidas (banda inferior, media e intervalo del "
    "95 % sobre 40 repeticiones)."
)


PIE_EN = (
    "Fig. 1. System overview. The leaf is photographed once on the sorting table "
    "(Stage 1); perception separates the leaf from the background and measures {n} "
    "physical features --holes, tears, edge bites, spots, sweating, how the damage "
    "is split between the two halves, size, colour and tip shape-- instead of "
    "operating on pixels (Stage 2); the grade is resolved through three chained "
    "questions, each of which sees only the features its craft rule names "
    "(Stage 3); the abstention rule decides the leaf when confidence exceeds the "
    "threshold and defers the rest to a human inspector (Stage 4); and the "
    "simulated arm carries it to its destination bin --one per grade and "
    "variety-- or to the review bin "
    "(Stage 5). The inspector's corrections feed back into the model: on a new "
    "capture, accuracy rises from {a} % to {b} % after 56 corrected leaves (lower "
    "band, mean and 95 % interval over 40 repetitions)."
)

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    fig = dibuja()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(DEST / "fig1_agente{}.{}".format(SUFIJO, ext))
    plt.close(fig)
    print("   escrita", DEST / "fig1_agente{}.pdf".format(SUFIJO))
    d = json.load(open(OUT / "PAPER_CURVA" / "curva.json", encoding="utf-8"))
    ks = d["ks"]
    a = d["media"][str(ks[0])] * 100
    b = d["media"][str(ks[-1])] * 100
    (DEST / "fig1_pie{}.txt".format(SUFIJO)).write_text(
        (PIE_EN if IDIOMA == "en" else PIE).format(n=CIFRAS["n_rasgos"],
                   a=num(("%.1f" % a).replace(".", ",")),
                   b=num(("%.1f" % b).replace(".", ","))),
        encoding="utf-8")
    print("   pie en", DEST / "fig1_pie{}.txt".format(SUFIJO))
    if COPIA.exists():
        for f in ("fig1_agente{0}.pdf", "fig1_agente{0}.svg",
                  "fig1_agente{0}.png", "fig1_pie{0}.txt"):
            f = f.format(SUFIJO)
            shutil.copy2(DEST / f, COPIA)
        print("   copiada a", COPIA)


if __name__ == "__main__":
    main()
