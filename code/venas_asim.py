r"""
VENA RESALTADA, medida como ASIMETRIA ENTRE MITADES DE LA MISMA HOJA (2026-09-03)

POR QUE UN SEGUNDO GUION Y NO UN PARCHE DE `venas.py`
------------------------------------------------------
`venas.py` mide crestas con un Hessiano multiescala ISOTROPO y quedo aparcado en
§25.5: marca el 86 % de la lamina, porque una hoja de tabaco curado esta llena de
arrugas y una arruga tambien es una cresta. El filtro no distingue vena de arruga
y no hay umbral que lo arregle -- es el mismo error de forma que la mancha blanca
(§42). Se deja como esta, para no perder lo medido, y aqui se cambia la medida.

LOS DOS CAMBIOS, Y DE DONDE SALEN
---------------------------------
1. **GEOMETRIA EN VEZ DE ASPECTO LOCAL.** Lo dijo el experto describiendo el defecto:
   *"a medida se recorre, la vena secundaria es gruesa y se hace mas delgada
   conforme se acerca al borde"*. O sea que una vena es una linea LARGA y de
   orientacion CONSTANTE que sale de la vena central; una arruga es corta, curva y
   sin orientacion preferente. Asi que el filtro no es isotropo: es una cresta
   ALARGADA, suavizada mucho a lo largo y derivada solo a lo ancho. Un nucleo
   largo mata las arrugas por construccion.

2. **ASIMETRIA, NO NIVEL ABSOLUTO.** el experto (2026-09-03): *"es lo mismo, las venas
   resaltadas del lado izquierdo, en el derecho pasa igual... al final es el mismo
   defecto, solo que esta en un lado o otro"*. Tiene razon, y de ahi sale la forma
   correcta de la medida:

       vena = f(mitad izquierda) - f(mitad DERECHA EN ESPEJO)

   La mitad derecha se voltea y se le aplica **la misma funcion f**. Con eso la
   medida es **simetrica por construccion**: no puede favorecer un lado, porque no
   sabe en que lado esta. Eso responde solo la objecion que bloqueaba el asunto
   -- que las cuatro hojas marcadas son todas "derecho mas resaltado" y una medida
   con sesgo de lado las acertaria sin medir ninguna vena -- y **quita la necesidad
   de fotografiar hojas con la vena resaltada a la izquierda para PODER empezar**
   (siguen haciendo falta para comprobar que generaliza, pero de cualquier lado).

   De paso, al restar un lado del otro se cancelan la luz, la variedad y el nivel
   de arruga de esa hoja, que es lo que ensucia cualquier umbral absoluto.

EL CONTROL NEGATIVO ES LA 120, Y ES LA MAS VALIOSA DE LAS CINCO
---------------------------------------------------------------
el experto no pudo decidir la clase de la 120 *precisamente porque* los dos lados estan
igual. O sea que en ella la medida tiene que dar **cerca de cero**. Una medida que
de un numero grande en la 120 esta midiendo otra cosa, por muy bien que acierte en
las otras cuatro.

QUE SE ESPERA (verdad-terreno de `REVISION_EXPERTO.txt`)
    115, 116, 117, 119   vena mas resaltada a la DERECHA  -> asimetria negativa
    120                  los dos lados iguales            -> cerca de cero

Uso:
    python scripts/venas_asim.py --casos     # los 5 del par + los ejemplos del experto
    python scripts/venas_asim.py --par       # las hojas del par XL/XR, con QC
    python scripts/venas_asim.py             # todas las del manifiesto
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentacion import mascara_hoja, mascara_marca
from zonas import perfil, extremo_base, lamina
from agujeros import rota
from rutas import REPO_RAIZ  # raiz del repositorio

RAIZ = REPO_RAIZ / "dataset"
OUT = REPO_RAIZ / "out"
QC = OUT / "qc_venas"
LADO = 1400                 # las venas son finas: mas fino que la segmentacion

# Banco de orientaciones, en grados respecto de la vena central (que tras
# rectificar queda vertical). Las venas secundarias salen en abanico; se excluye
# la banda casi vertical porque ahi esta la vena central, que es gruesa en TODAS
# las hojas y no es el defecto del que hablal experto.
# Se EXCLUYE la banda casi vertical (70-110): tras rectificar, lo vertical es la
# VENA CENTRAL, que es gruesa en todas las hojas y no es el defecto. Se vio en el
# primer QC: casi toda la deteccion era la franja de la vena central. Las venas
# secundarias salen del nervio a unos 30-60 grados, o sea 20-65 y 115-160 aqui.
ANGULOS = [20, 35, 50, 65, 115, 130, 145, 160]
SIG_LARGO = 14.0            # px de suavizado A LO LARGO de la cresta
SIG_ANCHO = 2.2             # px de la derivada A LO ANCHO (grosor de vena)
LARGO_MIN = 0.28            # una cresta cuenta si mide >= 28 % del ancho de la mitad
BANDA_VENA = 0.22           # fraccion del semiancho que se descarta junto al nervio
# La orilla de la hoja va enrollada y es la cresta mas fuerte de la imagen: en el
# segundo QC toda la margen izquierda salia marcada y se llevaba ella sola la
# asimetria. Se descarta una franja proporcional al tamano de la hoja, no un
# numero fijo de pixeles, y ademas se RELLENA el fondo con el nivel medio de la
# lamina antes de filtrar, para que el escalon mascara/fondo no genere cresta.
EROSION_FRAC = 0.045        # fraccion del semiancho que se come del contorno
# El reborde de un AGUJERO es una cresta tan fuerte como una vena, y la mascara
# viene rellena (segmentacion.py paso 4), asi que el hueco esta DENTRO de ella y
# la erosion del contorno no lo toca. Visto en el QC de 20260824_160531288: la
# mitad izquierda daba 0.149 y casi todo era el aro alrededor de un agujero.
# Se descarta lo que esta muy por debajo del nivel de la lamina, con su entorno.
D_OSCURO = 28.0             # unidades L por debajo de la mediana de la hoja


def carga(p, lado=LADO):
    with Image.open(p) as im:
        try:
            im.draft("RGB", (im.width // 4, im.height // 4))
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((lado, lado), Image.LANCZOS)
        return np.asarray(im)


def cresta_alargada(L, angulos=ANGULOS):
    """Maxima respuesta de cresta CLARA y LARGA sobre un banco de orientaciones.

    Para cada angulo se gira la imagen hasta poner esa direccion en horizontal, se
    suaviza mucho a lo largo (SIG_LARGO) y se toma la segunda derivada a lo ancho
    (SIG_ANCHO). Una cresta clara da segunda derivada negativa cruzandola, asi que
    la respuesta es su negativo. El suavizado largo es lo que separa la vena de la
    arruga: una arruga de 20 px no sobrevive a una gaussiana de sigma 14 a lo largo.
    """
    salida = np.zeros_like(L, np.float32)
    h, w = L.shape
    for th in angulos:
        g = rota(L, -th)
        g = cv2.GaussianBlur(g, (0, 0), SIG_LARGO, SIG_ANCHO)   # sx largo, sy corto
        d2 = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=5)
        r = np.maximum(0.0, -d2)
        r = rota(r, th)
        # `rota` amplia el lienzo; se recorta al centro para volver al original
        y0 = (r.shape[0] - h) // 2
        x0 = (r.shape[1] - w) // 2
        r = r[y0:y0 + h, x0:x0 + w]
        if r.shape != L.shape:                       # por redondeos de 1 px
            r = cv2.resize(r, (w, h))
        salida = np.maximum(salida, r)
    return salida


def estadisticos(rn, msk, umb, ancho_ref):
    """Cuanta cresta LARGA hay dentro de `msk`, con un umbral YA fijado.

    El umbral entra como argumento a proposito. En el primer intento se calculaba
    dentro de cada mitad como su propio percentil 90, y entonces la fraccion salia
    ~0.10 en las dos por construccion y la asimetria era ruido -- exactamente el
    error que la cabecera de `venas.py` ya dejaba escrito ("con un percentil, la
    fraccion de cresta salia constante por construccion"). El umbral tiene que ser
    UNO para las dos mitades, medido sobre la hoja entera.
    """
    if msk.sum() < 2000:
        return dict(frac=np.nan, resp=np.nan, largo=np.nan)
    fuerte = (rn >= umb) & msk
    n, etq, st, _ = cv2.connectedComponentsWithStats(fuerte.astype(np.uint8), 8)
    largo_min = LARGO_MIN * ancho_ref
    total, largos = 0, []
    for i in range(1, n):
        d = float(np.hypot(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]))
        if d >= largo_min:
            total += int(st[i, cv2.CC_STAT_AREA])
            largos.append(d / ancho_ref)
    return dict(frac=total / float(msk.sum()),
                resp=float(np.mean(rn[fuerte])) if fuerte.any() else 0.0,
                largo=float(np.mean(largos)) if largos else 0.0)


def rectifica(rgb, m):
    """Gira la hoja hasta dejar el eje mayor vertical y la base abajo."""
    ev, perp, mu, t, s, anch = perfil(m)
    base_al_final, _ = extremo_base(anch)
    if not base_al_final:
        ev = -ev
    ang = np.degrees(np.arctan2(ev[1], ev[0])) - 90.0
    rec = rota(rgb, ang)
    rem = rota(m.astype(np.uint8), ang, True) > 0
    if not rem.any():
        return None, None
    ys, xs = np.nonzero(rem)
    return (rec[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
            rem[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def mide_hoja(rgb, m, guardar=None, titulo=""):
    """asimetria = f(mitad izquierda) - f(mitad derecha), con umbral comun.

    No hace falta voltear una mitad para que la medida sea simetrica: el banco de
    orientaciones ya es simetrico respecto de la vertical (20/35/50/65 y sus
    reflejos 160/145/130/115), asi que la respuesta de un pixel no depende de en
    que lado de la hoja este. La simetria queda garantizada por el filtro, que es
    mas limpio que garantizarla volteando recortes (voltear obliga a rectificar
    dos veces y el remuestreo mete diferencias de milesimas).
    """
    rec, rem = rectifica(rgb, m)
    if rec is None:
        return None
    L = cv2.cvtColor(rec, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    W = rem.shape[1]
    cx = int(np.average(np.arange(W), weights=rem.sum(0)))
    semi = max(cx, W - cx)
    er = max(5, int(EROSION_FRAC * semi))
    dentro = cv2.erode(rem.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (2 * er + 1,) * 2)) > 0
    if dentro.sum() < 5000:
        return None
    # fuera los agujeros y su reborde. El ensanche es SIG_LARGO y no un numero
    # elegido a ojo: es el alcance del propio filtro, o sea hasta donde puede
    # llegar la respuesta que provoca un escalon oscuro. Con 4*er (18 % del
    # semiancho) se comia media hoja y la propia amputacion generaba asimetria.
    ref_L = float(np.median(L[rem]))
    oscuro = rem & (cv2.medianBlur(L.astype(np.uint8), 9) < ref_L - D_OSCURO)
    if oscuro.any():
        oscuro = cv2.dilate(oscuro.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (2 * int(SIG_LARGO) + 1,) * 2)) > 0
        dentro &= ~oscuro
    # sin escalon en el contorno: fuera de la hoja, el nivel medio de la lamina
    L = np.where(rem, L, ref_L).astype(np.float32)
    banda = int(BANDA_VENA * semi)
    izq_m = dentro.copy(); izq_m[:, cx - banda:] = False
    der_m = dentro.copy(); der_m[:, :cx + banda] = False
    if izq_m.sum() < 2000 or der_m.sum() < 2000:
        return None

    rn = cresta_alargada(L) / float(L[dentro].std() + 1e-6)
    util = izq_m | der_m
    umb = float(np.percentile(rn[util], 90))     # UNO solo, sobre las dos mitades
    fi = estadisticos(rn, izq_m, umb, semi)
    fd = estadisticos(rn, der_m, umb, semi)
    out = {}
    for k in ("frac", "resp", "largo"):
        out["vena_izq_" + k] = round(fi[k], 5) if np.isfinite(fi[k]) else ""
        out["vena_der_" + k] = round(fd[k], 5) if np.isfinite(fd[k]) else ""
        out["asim_vena_" + k] = (round(fi[k] - fd[k], 5)
                                 if np.isfinite(fi[k]) and np.isfinite(fd[k]) else "")
    if guardar:
        QC.mkdir(exist_ok=True)
        vis = cv2.cvtColor(np.clip(L, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vis[~util] = (30, 30, 30)
        vis[(rn >= umb) & util] = (40, 60, 255)
        cv2.line(vis, (cx, 0), (cx, vis.shape[0]), (0, 220, 220), 1)
        cv2.putText(vis, "{}  izq={:.4f} der={:.4f} asim={:+.4f}".format(
            titulo, fi["frac"], fd["frac"], fi["frac"] - fd["frac"]),
            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        k = max(1, vis.shape[0] // 900)
        cv2.imwrite(str(QC / (guardar + ".jpg")), vis[::k, ::k],
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


CASOS = [
    ("117 vena DER clarisima", r"Habano\XL Izq\20260828_170906209_iOS.heic", "der"),
    ("115 vena DER",           r"Habano\XL Izq\20260828_170356063_iOS.heic", "der"),
    ("116 vena DER",           r"Habano\XL Izq\20260828_170601248_iOS.heic", "der"),
    ("119 vena DER poco",      r"Habano\XL Izq\20260828_171906258_iOS.heic", "der"),
    ("120 CONTROL iguales",    r"Habano\XL Izq\20260828_172333023_iOS.heic", "cero"),
]


def casos():
    print("hoja                      izq      der      asimetria   esperado  sale")
    for nom, rel, esp in CASOS:
        rgb = carga(RAIZ / rel)
        m = mascara_hoja(rgb, mascara_marca(rgb))[0]
        o = mide_hoja(rgb, m, guardar=nom.split()[0] + "_asim", titulo=nom)
        if not o:
            print("  {:24s} sin medida".format(nom)); continue
        a = o["asim_vena_frac"]
        sale = "cero" if abs(a) < 0.010 else ("izq" if a > 0 else "der")
        print("  {:24s} {:8} {:8} {:+10} {:>9s}  {}{}".format(
            nom, o["vena_izq_frac"], o["vena_der_frac"], a, esp, sale,
            "" if sale == esp else "   <-- NO"))
    # comprobacion de simetria: la hoja volteada debe dar el numero opuesto
    print("\n  espejo (debe salir el mismo numero con el signo cambiado):")
    for nom, rel, _ in CASOS[:2]:
        rgb = carga(RAIZ / rel)
        m = mascara_hoja(rgb, mascara_marca(rgb))[0]
        o1 = mide_hoja(rgb, m)
        o2 = mide_hoja(rgb[:, ::-1].copy(), m[:, ::-1].copy())
        print("    {:24s} {:+.5f}  ->  {:+.5f}".format(
            nom, o1["asim_vena_frac"], o2["asim_vena_frac"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", action="store_true")
    ap.add_argument("--par", action="store_true")
    ap.add_argument("--clases", default="")
    ap.add_argument("max_nuevas", nargs="?", type=int, default=0)
    a = ap.parse_args()
    if a.casos:
        return casos()

    man = list(csv.DictReader(open(OUT / "manifiesto_limpio.csv", encoding="utf-8")))
    # `--par` = solo XL/XR. `--clases capa,xl_izq,xr_der` amplia sin rehacer nada:
    # el fichero se abre en modo append y se saltan las que ya estan, asi que
    # anadir las capas cuesta solo las capas.
    #
    # LA IDEA ES DEL EXPERTO (2026-09-04) y es la que desatasca la vena:
    #   "cuando son XR o XL y no tiene ni roturas ni manchas y pareciera que es
    #    una capa, no lo es simplemente porque las venas estan resaltadas...
    #    puedes compararlas con capas, y ahi la comparacion es facil"
    # O sea que la verdad-terreno que §44.6 pedia a mano SE PUEDE DERIVAR: una
    # XL/XR sin roturas ni manchas lo es POR LA VENA, y una capa tiene la vena
    # sin resaltar. No hacen falta 10-15 etiquetas suyas: hay cientos de hojas.
    if a.clases:
        quiere = [c.strip() for c in a.clases.split(",")]
        man = [r for r in man if r["clase"] in quiere]
    elif a.par:
        man = [r for r in man if r["clase"] in ("xl_izq", "xr_der")]
    salida = OUT / ("venas_asim_par.csv" if (a.par or a.clases)
                    else "venas_asim.csv")
    hechas = set()
    nuevo = not salida.exists()
    if not nuevo:
        hechas = {r["ruta"] for r in csv.DictReader(open(salida, encoding="utf-8"))}
    pend = [r for r in man if r["ruta"] not in hechas]
    if a.max_nuevas:
        pend = pend[:a.max_nuevas]
    print("fotos: {}   pendientes: {}".format(len(man), len(pend)))
    campos = (["ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id"] +
              ["vena_{}_{}".format(l, k) for k in ("frac", "resp", "largo")
               for l in ("izq", "der")] +
              ["asim_vena_" + k for k in ("frac", "resp", "largo")])
    fh = open(salida, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
    if nuevo:
        w.writeheader()
    for i, r in enumerate(pend, 1):
        try:
            rgb = carga(RAIZ / r["ruta"].replace("/", "\\"))
            m = mascara_hoja(rgb, mascara_marca(rgb))[0]
            o = mide_hoja(rgb, m)
        except Exception as exc:
            print("  {} FALLO {}".format(r["archivo"], exc))
            continue
        if not o:
            continue
        fila = {k: r.get(k, "") for k in
                ("ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id")}
        fila.update(o)
        w.writerow(fila)
        fh.flush()
        if i % 25 == 0:
            print("  {}/{}".format(i, len(pend)), flush=True)
    fh.close()
    print("->", salida)


if __name__ == "__main__":
    main()
