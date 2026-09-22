r"""
Detecta MANCHAS en la hoja: verde ("sudada"), blanca y negra.

POR QUE HACE FALTA, Y POR QUE NO BASTA CON LOS AGUJEROS
-------------------------------------------------------
el experto, 2026-09-01: *"defecto si puede ser mancha verde, negra, blancas"*. Y sus
propios ficheros del dataset ya lo decian, en `XL_IZQ/defectos_que_se_permiten`:

    Defecto1: "la hoja no tiene roturas, pero el lado derecho gran parte de la
               parte superior tiene manchas verdes 'sudada' mientras que el otro
               lado no, es por ello que forma parte de un XL izq"
    Defecto4: "las manchas blancas que ves ahi, esas no se pueden quitar porque
               son manchas que vienen por la aplicacion de agroquimicos"

O sea que **un XL puede no tener ni un agujero**. Un sistema que solo vea huecos
no puede separar esas clases ni en principio.

COMO SE MIDE, Y CONTRA QUE
--------------------------
Igual que los agujeros: **contra la propia hoja de esa misma foto**. No hay un
"verde" absoluto -- el Habano es pardo y el Connecticut amarillo, y la misma hoja
cambia de tono entre el centro y el borde. Lo que se mide es la DESVIACION
respecto de la mediana de esa hoja:

    verde   a* por debajo de la mediana de la hoja   (menos rojo = mas verde)
    blanca  L muy por encima y croma por debajo      (parche palido y sin color)
    negra   L muy por debajo y croma por debajo      (parche oscuro y sin color)

CALIBRADO CONTRA UN PAR DE VERDAD-TERRENO
-----------------------------------------
`Defecto2.heic` + `Defecto2_marcado.jpg` de XL_IZQ son la misma foto, una con el
contorno de la mancha dibujado encima. Medido dentro y fuera del contorno:

    dentro del contorno   a* p25=0  p50=2  p75=3
    resto de la hoja      a* p25=2  p50=4  p75=5

    fraccion con a* < mediana-2:   dentro 0.28   fuera 0.08   (x3.5)
    fraccion con a* < mediana-3:   dentro 0.14   fuera 0.03   (x4.7)

La diferencia es de solo 2-4 unidades de a*, asi que **el umbral tiene que ser
relativo a la hoja y el resultado hay que agrupar en parches**: pixel a pixel el
ruido de textura se come la senal, pero una mancha es una REGION.

DOS CAUTELAS MEDIDAS
--------------------
1. El brillo especular es palido y sin color, igual que una mancha blanca. Se
   separa por tamano y por forma: el brillo sigue las arrugas (fino y alargado),
   la mancha es un parche compacto. Se exige area minima y compacidad.
2. Una mancha negra y un agujero se parecen cuando la mesa es negra (sesion del
   28/08). Por eso se descuenta lo que `agujeros.py` ya marco como hueco.

Uso:
    python scripts/manchas.py --ejemplos          # solo los casos documentados, con QC
    python scripts/manchas.py 200                 # 200 fotos nuevas y para
    python scripts/manchas.py --qc 60             # todas
    python scripts/manchas.py --reagrupa          # rehace el CSV desde el detalle
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zonas import (perfil, extremo_base, lamina, bandas,
                   ancho_cinta_por_carpeta)
from segmentacion import mascara_marca, mascara_hoja
from agujeros import carga, rota, disco, componentes
from rutas import REPO_RAIZ  # raiz del repositorio

RAIZ = REPO_RAIZ / "dataset"
OUT = REPO_RAIZ / "out"
LADO = 1024
PULG_POR_ANCHO_CINTA = 1.11

# umbrales, en unidades Lab RELATIVAS a la mediana de cada hoja (ver cabecera)
D_VERDE = 3.0          # a* por debajo de la mediana
# Umbrales de la mancha BLANCA, calibrados el 2026-09-03 contra los 14 circulos
# de `dataset/Manchas blancas o de Agroquimicos` (§45). Son desviaciones respecto
# de la referencia LOCAL, no de la mediana de la hoja. Barrido medido:
#     L>+25 & croma<-3  ->  3.7x     L>+35 & croma<-3  ->  6.0x
#     L>+45 & croma<-3  -> 10.4x  (el elegido)
D_BLANCA_L = 45.0      # L por encima de la referencia local
D_BLANCA_C = 3.0       # croma por debajo de la referencia local
RADIO_LOCAL = 0.030    # nucleo de la mediana local, como fraccion del lado
# ---------------------------------------------------------------------------
# CAMBIOS DEL 2026-09-05, todos con motivo y ninguno por tocar hasta que salga
# bonito. Vienen de dos sitios: la verdad-terreno de PROGRESO 51 y una frase de
# El experto.
#
# 1. D_NEGRA_L: 40 -> 10.  MEDIDO contra las dos manchas negras que marco el experto
#    (PROGRESO 51.2): en las manchas de verdad la L cae 10.6 y 24.3 respecto de la
#    lamina. Pedir 40 era pedir un parche casi negro, que no es lo que son.
#
# 2. La condicion de CROMA se quita para la negra.  Tambien medido: en una de las
#    dos el croma SUBE (-0.8) y en la otra baja 3.5, contra los 8 que se pedian.
#    La regla suponia que una mancha negra es GRIS -- oscura y sin color-- y no lo
#    es: son oscuras y **conservan el color de la hoja**. Es un error de concepto,
#    no de numero. Para la BLANCA se mantiene: ahi si tiene sentido.
#
# 3. MIN_PULG2: 0.05 -> 0.004, y SUAVIZADO: 9 -> 5.  Lo dijo el experto el 2026-09-05:
#       "una mancha es una mancha, hay desde PEQUENITAS hasta grandes, el chiste
#        es identificarlas. Aparte se nota, ya que la calidad de la camara es muy
#        buena."
#    La definicion de Banda es literalmente "manchITAS" (PROGRESO 22.1), en
#    diminutivo, y el guion las tiraba a la basura por pequenas: 0.05 pulg2 es una
#    mancha de 5.7 mm. Y el suavizado de 9 px borraba lo que quedara. El nuevo
#    minimo, 0.004 pulg2, son ~1.6 mm: 3-4 px a LADO=1024, que es el suelo de ruido
#    de la imagen. Por debajo de eso ya no es que se descarte: es que no esta.
# ---------------------------------------------------------------------------
# 4. Y UNA CORRECCION DEL MISMO DIA, tras probar el punto 1. Bajar el umbral a 10
#    contra la mediana de la hoja marcaba el **25 % de cada hoja** como mancha
#    negra, en las diez carpetas. No son manchas: son sombras de arruga. Dentro de
#    una hoja la L va de 110 a 216, asi que ningun umbral GLOBAL sirve -- que es
#    exactamente lo que ya se habia aprendido con la blanca en PROGRESO 42.
#
#    Con referencia LOCAL tampoco basta: 8-32 % de la hoja y solo 1.4-2.8 veces mas
#    dentro de los circulos que marco el experto. Lo que si funciona es anadir la FORMA,
#    el mismo mecanismo que la mancha blanca en PROGRESO 53:
#
#        sombra de arruga -> es un valle ALARGADO a lo largo del pliegue
#        mancha negra     -> es un valle COMPACTO, redondo
#
#    Con el Hessiano: un valle tiene los dos autovalores positivos, y uno compacto
#    los tiene ademas parecidos (anisotropia baja). Medido en las dos fotos
#    marcadas, dentro del circulo contra el resto de la hoja:
#
#        L_loc-12 solo                 1.9x    y marca el 16-21 % de la hoja
#        + valle                       1.8x            9-11 %
#        + valle y redondo (<2.5)      4.0x            1.8-2.1 %
#        L_loc-18 + valle + redondo    6.0x            1.1-1.4 %   <- el elegido
#
#    AVISO: esto se fija con DOS fotos marcadas. Es un punto de partida con
#    mecanismo detras, no una calibracion. Hacen falta mas hojas con la mancha
#    negra circulada para poder afirmar nada -- la misma peticion que la blanca.
D_NEGRA_L = 18.0       # L por debajo de la referencia LOCAL (antes 40, global)
NEGRA_LOCAL = True     # referencia local, como la blanca
NEGRA_FORMA = True     # exigir valle compacto, no sombra de pliegue
NEGRA_ANISO = 2.5      # |lam1|/|lam2| maximo; una sombra de arruga da mucho mas
NEGRA_SIGMA = 2.0      # escala de la mancha, en px a LADO=1024
NEGRA_USA_CROMA = False   # ver el punto 2 de arriba
D_CROMA = 8.0          # croma por debajo. Solo para la BLANCA
SUAVIZADO = 5          # px de mediana antes de umbralizar (antes 9)
MIN_PULG2 = 0.004      # una manchita de ~1.6 mm. Lo pidio el experto (antes 0.05)


def _impar(n):
    return int(n) | 1


def detecta(rgb, m, hueco=None):
    """(verde, blanca, negra) dentro de la hoja, como mascaras booleanas.

    El suavizado va sobre el Lab en 8 bits porque `medianBlur` solo admite
    nucleos grandes en 8U; la mediana (y no la media) es a proposito: una media
    arrastra el borde de la mancha hacia fuera.
    """
    lab8 = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB), SUAVIZADO)
    lab = lab8.astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128
    b = lab[:, :, 2] - 128
    croma = np.hypot(a, b)

    ref_L = float(np.median(L[m]))
    ref_a = float(np.median(a[m]))
    ref_c = float(np.median(croma[m]))

    verde = m & (a < ref_a - D_VERDE) & (b > 5)
    negra = m & (L < ref_L - D_NEGRA_L)
    if NEGRA_USA_CROMA:
        negra &= (croma < ref_c - D_CROMA)

    # BLANCA: referencia LOCAL, no la mediana de la hoja (2026-09-03, §45).
    # La version anterior comparaba contra la mediana de toda la lamina y daba
    # `frac_blanca` = 0.000 en las diez carpetas (§37.7, §40.4). Medido contra los
    # 14 circulos que el experto marco en `dataset/Manchas blancas o de Agroquimicos`:
    # dentro del circulo la mediana es la misma que fuera, y lo que cambia es la
    # desviacion respecto de lo que la mancha tiene ALREDEDOR. Dentro de una hoja
    # L va de 110 a 216 por las arrugas y las sombras, asi que una mota clara en
    # zona de sombra nunca alcanza un umbral global.
    # La referencia local es una mediana de nucleo grande -- el fondo sin la
    # mancha -- y la regla queda: clara Y descolorida respecto de su entorno.
    # Enriquecimiento medido dentro del circulo contra su corona: 10x a 15x.
    kloc = _impar(max(31, int(RADIO_LOCAL * max(m.shape))))
    L_loc = cv2.medianBlur(lab8[:, :, 0], kloc).astype(np.float32)
    c_loc = cv2.medianBlur(
        np.clip(croma, 0, 255).astype(np.uint8), kloc).astype(np.float32)
    blanca = m & (L > L_loc + D_BLANCA_L) & (croma < c_loc - D_BLANCA_C)

    # NEGRA: referencia local + forma, ver el punto 4 de la cabecera.
    if NEGRA_LOCAL:
        negra = m & (L < L_loc - D_NEGRA_L)
        if NEGRA_FORMA:
            g = cv2.GaussianBlur(L, (0, 0), NEGRA_SIGMA)
            Lxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=5)
            Lyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=5)
            Lxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=5)
            tr_ = Lxx + Lyy
            det = Lxx * Lyy - Lxy * Lxy
            dsc = np.sqrt(np.maximum(0.0, tr_ * tr_ / 4.0 - det))
            l1, l2 = tr_ / 2.0 + dsc, tr_ / 2.0 - dsc
            a1, a2 = np.abs(l1), np.abs(l2)
            aniso = np.maximum(a1, a2) / (np.minimum(a1, a2) + 1e-6)
            negra &= (l1 > 0) & (l2 > 0) & (aniso < NEGRA_ANISO)
    if hueco is not None:
        negra &= ~hueco
        verde &= ~hueco
        blanca &= ~hueco

    k = disco(3)
    salida = []
    for x in (verde, blanca, negra):
        x = cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, k)
        x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, k) > 0
        salida.append(x)
    return tuple(salida) + (dict(L=ref_L, a=ref_a, croma=ref_c),)


CAMPOS_FOTO = ["ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id",
               "pulg_por_px", "n_verde", "n_blanca", "n_negra",
               "area_verde_pulg2", "area_blanca_pulg2", "area_negra_pulg2",
               "frac_verde", "frac_blanca", "frac_negra",
               "verde_izq_pulg2", "verde_der_pulg2",
               "blanca_izq_pulg2", "blanca_der_pulg2",
               "negra_izq_pulg2", "negra_der_pulg2",
               "mancha_izq_pulg2", "mancha_der_pulg2", "asimetria_mancha",
               "mayor_mancha_pulg2", "mayor_mancha_tipo", "mayor_mancha_zona"]

CAMPOS_DET = ["ruta", "carpeta", "clase", "hoja_id", "tipo", "area_px",
              "area_pulg2", "zona", "dist_base_pulg", "s_pulg"]


def agrega(dets, area_hoja, pulg_px, base):
    k2 = (pulg_px ** 2) if pulg_px else None

    def suma(cond):
        return sum(d["area_px"] for d in dets if cond(d))

    def p2(x):
        return round(x * k2, 4) if k2 else ""

    izq = suma(lambda d: d["zona"] == "util_izq")
    der = suma(lambda d: d["zona"] == "util_der")
    mayor = max(dets, key=lambda d: d["area_px"]) if dets else None
    f = dict(base)
    for t in ("verde", "blanca", "negra"):
        f["n_" + t] = sum(1 for d in dets if d["tipo"] == t)
        f["area_{}_pulg2".format(t)] = p2(suma(lambda d, t=t: d["tipo"] == t))
        f["frac_" + t] = (round(suma(lambda d, t=t: d["tipo"] == t) / area_hoja, 5)
                          if area_hoja else "")
        f["{}_izq_pulg2".format(t)] = p2(suma(
            lambda d, t=t: d["tipo"] == t and d["zona"] == "util_izq"))
        f["{}_der_pulg2".format(t)] = p2(suma(
            lambda d, t=t: d["tipo"] == t and d["zona"] == "util_der"))
    f["mancha_izq_pulg2"] = p2(izq)
    f["mancha_der_pulg2"] = p2(der)
    # la asimetria es LO QUE DECIDE la clase: XL izq / XR der es el lado donde
    # sobrevive la mitad limpia (PROGRESO 16.2, 36.1)
    f["asimetria_mancha"] = round((izq - der) / (izq + der + 1e-6), 4)
    f["mayor_mancha_pulg2"] = p2(mayor["area_px"]) if mayor else ""
    f["mayor_mancha_tipo"] = mayor["tipo"] if mayor else ""
    f["mayor_mancha_zona"] = mayor["zona"] if mayor else ""
    return f


def marco(m):
    """Eje con la base al final, y extremos de la LAMINA (sin peciolo)."""
    e, perp, mu, t, s, anch = perfil(m)
    base_al_final, _ = extremo_base(anch)
    if not base_al_final:
        e, perp, t, s = -e, -perp, -t, -s
    lam0, lam1 = lamina(t, anch)
    return e, perp, mu, lam0, lam1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_nuevas", nargs="?", type=int, default=0)
    ap.add_argument("--qc", type=int, default=0)
    ap.add_argument("--solo", default="")
    ap.add_argument("--muestra", type=int, default=0)
    ap.add_argument("--ejemplos", action="store_true",
                    help="corre sobre los casos documentados del experto, con QC")
    a = ap.parse_args()

    if a.ejemplos:
        return ejemplos()

    seg = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "segmentacion.csv", encoding="utf-8"))}
    man = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "manifiesto_limpio.csv", encoding="utf-8"))}
    end = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "enderezado.csv", encoding="utf-8"))}
    ancho_cinta = ancho_cinta_por_carpeta(OUT)

    rutas = sorted(r for r in seg if r in man)
    if a.solo:
        rutas = [r for r in rutas if a.solo in r]
    parcial = bool(a.solo or a.muestra)
    if a.muestra:
        porc = defaultdict(list)
        for r in rutas:
            porc["/".join(r.split("/")[:2])].append(r)
        rutas = []
        for c in sorted(porc):
            v = porc[c]
            idx = np.linspace(0, len(v) - 1, min(a.muestra, len(v))).astype(int)
            rutas += [v[i] for i in sorted(set(idx))]

    f_foto = OUT / ("manchas_parcial.csv" if parcial else "manchas.csv")
    f_det = OUT / ("manchas_detalle_parcial.csv" if parcial else "manchas_detalle.csv")
    hechas = set()
    if not parcial and f_foto.exists():
        hechas = {r["ruta"] for r in
                  csv.DictReader(open(f_foto, encoding="utf-8")) if r.get("ruta")}
        print("ya calculadas: {}".format(len(hechas)))
    pend = [r for r in rutas if r not in hechas]
    if a.max_nuevas:
        pend = pend[:a.max_nuevas]
    print("fotos: {}   pendientes: {}".format(len(rutas), len(pend)))

    qc_dir = OUT / "qc_manchas"
    qc = set()
    if a.qc and pend:
        qc_dir.mkdir(parents=True, exist_ok=True)
        idx = np.linspace(0, len(pend) - 1, min(a.qc, len(pend))).astype(int)
        qc = {pend[i] for i in idx}

    nuevo = parcial or not f_foto.exists()
    fh1 = open(f_foto, "w" if nuevo else "a", newline="", encoding="utf-8")
    w1 = csv.DictWriter(fh1, fieldnames=CAMPOS_FOTO, extrasaction="ignore")
    if nuevo:
        w1.writeheader()
    nuevo2 = parcial or not f_det.exists()
    fh2 = open(f_det, "w" if nuevo2 else "a", newline="", encoding="utf-8")
    w2 = csv.DictWriter(fh2, fieldnames=CAMPOS_DET, extrasaction="ignore")
    if nuevo2:
        w2.writeheader()

    for i, ruta in enumerate(pend, 1):
        r, mr = seg[ruta], man[ruta]
        fmask = OUT / "mascaras" / (Path(r["archivo"]).stem + ".npy")
        if not fmask.exists():
            continue
        h, w = int(r["alto"]), int(r["ancho"])
        m = np.unpackbits(np.load(fmask))[:h * w].reshape(h, w).astype(bool)
        if m.sum() < 500:
            continue
        rgb = carga(RAIZ / ruta)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h))
        e_row = end.get(ruta, {})
        rot = float(e_row["rotacion"]) if e_row.get("rotacion") else 0.0
        vol = int(e_row.get("voltea_180") or 0)
        if rot:
            m = rota(m.astype(np.uint8), rot, True) > 0
            rgb = rota(rgb, rot)
        if vol:
            m = cv2.rotate(m.astype(np.uint8), cv2.ROTATE_180) > 0
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        if m.sum() < 500:
            continue

        carp = mr["carpeta"]
        ac = ancho_cinta.get(carp)
        pulg_px = PULG_POR_ANCHO_CINTA / ac if ac else None
        min_px = max(60, int(MIN_PULG2 / (pulg_px ** 2))) if pulg_px else 60

        verde, blanca, negra, ref = detecta(rgb, m)
        e, perp, mu, t0, t1 = marco(m)      # t0/t1 = extremos de la lamina
        punta_px, base_px, vena_px, _borde = bandas(t1 - t0, pulg_px, ac)

        det = []
        for tipo, msk in (("verde", verde), ("blanca", blanca), ("negra", negra)):
            for c in componentes(msk, min_px):
                pxy = np.array([c["cx"], c["cy"]], np.float32) - mu
                tc, sc = float(pxy @ e), float(pxy @ perp)
                if tc > t1 - base_px:
                    zona = "base"
                elif tc < t0 + punta_px:
                    zona = "punta"
                elif abs(sc) < vena_px / 2:
                    zona = "vena"
                else:
                    zona = "util_izq" if sc > 0 else "util_der"
                det.append(dict(ruta=ruta, carpeta=carp, clase=mr["clase"],
                                hoja_id=mr["hoja_id"], tipo=tipo,
                                area_px=c["area"],
                                area_pulg2=(round(c["area"] * pulg_px ** 2, 4)
                                            if pulg_px else ""),
                                zona=zona,
                                dist_base_pulg=(round((t1 - tc) * pulg_px, 3)
                                                if pulg_px else ""),
                                s_pulg=(round(sc * pulg_px, 3) if pulg_px else ""),
                                _mask=c["mask"]))
        for d in det:
            w2.writerow(d)
        base = dict(ruta=ruta, carpeta=carp, archivo=r["archivo"],
                    variedad=mr["variedad"], clase=mr["clase"],
                    hoja_id=mr["hoja_id"],
                    pulg_por_px=round(pulg_px, 6) if pulg_px else "")
        w1.writerow(agrega(det, float(m.sum()), pulg_px, base))
        fh1.flush()
        fh2.flush()

        if ruta in qc:
            qc_dir.mkdir(parents=True, exist_ok=True)
            pinta(rgb, m, det, qc_dir / (Path(r["archivo"]).stem.replace(" ", "_")
                                         + ".jpg"), mr["clase"])
        if i % 25 == 0:
            print("  {}/{}".format(i, len(pend)))

    fh1.close()
    fh2.close()
    print("-> {}".format(f_foto))
    informe(f_foto)


def pinta(rgb, m, det, destino, titulo=""):
    vis = rgb.copy()
    fuera = ~m
    vis[fuera] = (0.4 * vis[fuera] + 0.6 * np.array([25, 25, 25])).astype(np.uint8)
    col = {"verde": (40, 220, 40), "blanca": (255, 255, 255), "negra": (255, 0, 255)}
    for d in det:
        v = vis[d["_mask"]]
        vis[d["_mask"]] = (0.45 * v + 0.55 * np.array(col[d["tipo"]])).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 0, 255), 2)
    n = {t: sum(1 for d in det if d["tipo"] == t) for t in ("verde", "blanca", "negra")}
    cv2.putText(vis, "{}  verde={} blanca={} negra={}".format(titulo, n["verde"],
                n["blanca"], n["negra"]), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2, cv2.LINE_AA)
    k = max(1, vis.shape[0] // 800)
    cv2.imwrite(str(destino), cv2.cvtColor(vis[::k, ::k], cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def ejemplos():
    """Corre sobre los casos que el experto documento y escribe QC para mirarlos."""
    qc = OUT / "qc_manchas_ejemplos"
    qc.mkdir(parents=True, exist_ok=True)
    casos = []
    for d in sorted(RAIZ.rglob("defectos_que_se_permiten")):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".heic", ".jpg") and "marcad" not in f.stem.lower():
                casos.append(f)
    print("casos documentados: {}".format(len(casos)))
    for f in casos:
        rgb = carga(f)
        marca = mascara_marca(rgb)
        m, _, _ = mascara_hoja(rgb, marca)
        if m.sum() < 500:
            print("  {} : sin hoja".format(f.name))
            continue
        verde, blanca, negra, ref = detecta(rgb, m)
        det = []
        for tipo, msk in (("verde", verde), ("blanca", blanca), ("negra", negra)):
            for c in componentes(msk, 400):
                det.append(dict(tipo=tipo, area_px=c["area"], _mask=c["mask"]))
        area = float(m.sum())
        print("  {:34s} verde {:5.1f}%  blanca {:4.1f}%  negra {:4.1f}%".format(
            f.parent.parent.name[:20] + "/" + f.stem[:13],
            100 * sum(d["area_px"] for d in det if d["tipo"] == "verde") / area,
            100 * sum(d["area_px"] for d in det if d["tipo"] == "blanca") / area,
            100 * sum(d["area_px"] for d in det if d["tipo"] == "negra") / area))
        pinta(rgb, m, det, qc / (f.parent.parent.name[:14] + "_" + f.stem.replace(" ", "_")
                                 + ".jpg"), f.stem)
    print("-> {}".format(qc))


def informe(f_foto):
    filas = list(csv.DictReader(open(f_foto, encoding="utf-8")))
    if not filas:
        return
    por = defaultdict(list)
    for r in filas:
        por[r["carpeta"]].append(r)
    L = ["== manchas por carpeta (mediana del % de hoja afectado) ==",
         "  {:44s} {:>5s} {:>8s} {:>8s} {:>8s} {:>10s}".format(
             "carpeta", "n", "verde", "blanca", "negra", "asimetria")]
    for c in sorted(por):
        v = por[c]
        def col(k):
            return np.array([float(x[k] or 0) for x in v])
        L.append("  {:44s} {:5d} {:7.2f}% {:7.2f}% {:7.2f}% {:10.2f}".format(
            c[:44], len(v), 100 * np.median(col("frac_verde")),
            100 * np.median(col("frac_blanca")), 100 * np.median(col("frac_negra")),
            float(np.median(np.abs(col("asimetria_mancha"))))))
    txt = "\n".join(L)
    print("\n" + txt)
    (OUT / "MANCHAS_INFORME.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
