r"""
Segmenta la hoja y separa las marcas azules de anotacion.

No usa recortes manuales: la segmentacion es un paso DENTRO del sistema, porque
en la celda real la camara vera una hoja sobre una mesa con lo que haya alrededor.

Regla calibrada contra valores medidos en Lab sobre zonas de verdad-terreno
(no umbrales adivinados):

    hoja iluminada   L=183..198   a*=-2..+4    b*=+17..+47
    hoja EN SOMBRA   L=116..135   a*= 0..+4    b*= +7..+22
    mesa azul marino L= 66..127   a*=+1..+5    b*=-29..-2
    piso de concreto L= 98..173   a*=+0.5..+3  b*= +4..+27
    bota             L= 77        a*=+1        b*= -2
    cinta metrica    L=204..220   a*=-2        b*= -9..-0.5
    piel             L=135..167   a*=+11..+12  b*= +7..+11

CORRECCION DEL 2026-08-30, MEDIDA SOBRE HABANO
----------------------------------------------
La tabla de arriba se midio sobre Connecticut del 17/08. El 28/08 se fotografio
Habano en OTRO montaje y el guion fallaba: cobertura de 0.005 a 0.08 y la mascara
reducida a la vena central. Medido sobre las fotos que fallaban:

    hoja de HABANO   L= 81..206   a*=+5..+13   b*=+20..+48   <- a* como la piel
    piel             L=135..167   a*=+11..+12  b*= +7..+11

El culpable era el umbral `a* < 9`, puesto para dejar fuera las manos. La hoja de
Habano es PARDA ROJIZA y su a* vale 9..12, el mismo de la piel: ese umbral borraba
la hoja entera y solo sobrevivia la vena, que es palida. Con `b*>6` a secas el
candidato pasa de 8 % a 32 % del cuadro, que es la cobertura real de la hoja.

Piel y Habano no se separan por a*, pero SI por b*: la piel es apenas amarilla
(b*=+7..+11) y el tabaco es muy amarillo (b*>=+20). La frontera correcta no es
vertical sino inclinada en el plano a*-b*:   b* > 1.5 * a*
Comprobada contra las siete poblaciones medidas del proyecto:
    hoja iluminada  a=+4  b=+17  ->  17 > 6    dentro
    hoja en sombra  a=+4  b= +7  ->   7 > 6    dentro (justa, la histeresis ayuda)
    hoja de Habano  a=+12 b=+28  ->  28 > 18   dentro
    piel            a=+11 b=+11  ->  11 > 16   FUERA
    mesa / cinta / bota                        FUERA por b* < 0
Ademas la mesa del 28/08 es NEGRA, no azul marino: su b* ronda 0 en vez de -20.
Por eso el margen de B_MIN importa mas en Habano que en Connecticut.

La luminancia NO sirve para separar hoja de mesa: la hoja en sombra (L=116) es
tan oscura como la mesa (L=127). Un umbral sobre L parte la hoja por la mitad —
se comprobo a ojo. Quien separa es b* (amarillo-azul): el tabaco curado es calido
pase lo que pase y la mesa es azul. La piel entra por b* pero sale por a* (+12
contra +4 de la hoja). Cinta y bota caen por b*<0.

El competidor real es el PISO, no la mesa: es calido como la hoja y, en franja
larga, la supera en area. No se puede separar por pixel — una hoja en sombra
(L=126, b*=+8) es identica al piso (L=105, b*=+13) — asi que se separa por el
color MEDIO de cada componente conexa, donde la zona en sombra va unida al resto
de su hoja y la media sube:  hoja L=193..195 b*=+37..+47  vs  piso L=102..109
b*=+12..+18.

Se probaron y se descartaron, midiendo, tres alternativas que parecian mejores:
  - Otsu sobre la distancia al color de la mesa de cada foto: el reflejo especular
    de la mesa queda lejos de ese color y entraba en la mascara.
  - "la hoja esta rodeada de mesa": la corona del piso resulto tener MAS mesa
    (0.77) que la de la hoja (0.45), porque la franja del piso es angosta y la
    mesa esta justo al otro lado de la cinta.
  - recortar por la envolvente convexa de la mesa: cuando la mesa queda a un solo
    lado, esa envolvente rebana la hoja con un corte recto.

Uso:
    python scripts/segmentacion.py            # todas, sin QC
    python scripts/segmentacion.py --qc 24    # ademas escribe 24 superposiciones
    python scripts/segmentacion.py --solo 1845 --qc 5   # una sola, para revisar
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pillow_heif
from rutas import REPO_ROOT  # repository root
pillow_heif.register_heif_opener()

RAIZ = REPO_ROOT / "dataset"
OUT = REPO_ROOT / "out"
LADO = 1024  # lado largo de trabajo; la mascara se guarda a esta escala

# --- umbrales derivados de la tabla de arriba, puestos en medio de cada brecha
B_MIN = 6       # brecha: mesa/cinta/bota <= -2 ... hoja en sombra >= +7
L_MIN = 45      # solo descarta sombra dura donde el color ya no es fiable
B_FLOJO = 0     # umbral flojo de la histeresis: apenas del lado calido
K_AB = 1.5      # frontera b* > K_AB * a* ; sustituye al viejo A_MAX (ver arriba)
A_MAX = 20      # tope duro para lo francamente rojo; el que decide es K_AB

# marcador azul: firma validada en la sesion anterior contra el par _def
MARCA = dict(br_min=120, s_min=0.75, v_min=120)


def carga(p, lado=LADO):
    """Miniatura RGB con decodificacion rapida (draft usa el DCT del JPEG)."""
    with Image.open(p) as im:
        try:
            im.draft("RGB", (im.width // 4, im.height // 4))   # HEIC no tiene draft
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((lado, lado), Image.LANCZOS)
        return np.asarray(im)


def disco(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def mascara_marca(rgb):
    """Pixeles de marcador azul. El azul saturado no existe en tabaco curado."""
    b = rgb[:, :, 2].astype(np.int16)
    r = rgb[:, :, 0].astype(np.int16)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.int16)
    return ((b - r) > MARCA["br_min"]) & (s > MARCA["s_min"]) & (v > MARCA["v_min"])


def mascara_hoja(rgb, marca, crudo=False):
    """Devuelve (mascara_de_hoja, n_componentes, fraccion_descartada).

    Con `crudo=True` anade un cuarto elemento: la mascara ANTES de rellenar
    los huecos interiores. La diferencia entre las dos es lo que el paso 4
    tapa -- agujeros de la hoja, pero tambien nervaduras palidas y brillos --
    y es la materia prima del detector de agujeros (`agujeros.py`).
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128
    b = lab[:, :, 2].astype(np.float32) - 128

    # 0. la CINTA METRICA, fuera. Misma regla medida que usa `escala_cinta.py`:
    #    clara y no calida. Hasta el 2026-09-01 la cinta entraba en la mascara por
    #    la histeresis (es clara y su b* ronda 0), la hoja y la cinta quedaban
    #    unidas, y el paso 4 RELLENABA el hueco entre las dos: aparecia como
    #    "agujero" una franja de mesa de mas de una pulgada cuadrada. Se vio en
    #    20260828_200025952 y lo destapo `agujeros.py`.
    #    Solo se quita de la semilla y de la histeresis, no de la mascara final:
    #    asi un brillo especular SOBRE la hoja lo sigue rellenando el paso 4.
    cinta = (L > 170) & (a > -8) & (a < 6) & (b < 8)

    # 1. semilla firme por color: calido, y mas amarillo que rojo (ver cabecera)
    cruda = (L > L_MIN) & (b > B_MIN) & (b > K_AB * a) & (a < A_MAX) & ~cinta
    cruda |= marca      # el marcador esta ENCIMA de la hoja: es hoja, no fondo

    m = cruda.astype(np.uint8)
    k = max(3, (min(rgb.shape[:2]) // 200) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)      # quita motas del piso
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)     # cierra nervaduras

    # 2. elegir cual componente es la hoja. "La mas grande" no basta: el piso y su
    #    linea amarilla son tan calidos como la hoja y, siendo una franja larga, la
    #    superan en area cuando la hoja sale pequena (se vio en 183558155 y 192411218)
    n, etq, est, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        vacia = np.zeros(m.shape, bool)
        return (vacia, 0, 0.0, vacia) if crudo else (vacia, 0, 0.0)
    areas = est[1:, cv2.CC_STAT_AREA]
    anchos = est[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    altos = est[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    llenado = areas / np.maximum(1, anchos * altos)
    alarga = np.maximum(anchos, altos) / np.maximum(1.0, np.minimum(anchos, altos))
    Lm = np.array([L[etq == j + 1].mean() for j in range(len(areas))])
    bm = np.array([b[etq == j + 1].mean() for j in range(len(areas))])
    # El veto del piso lleva TRES condiciones, no dos. Con solo color (L bajo y b*
    # moderado) borraba hojas de verdad: una hoja parda en primer plano mide
    # Lm=143..149 bm=12..23, dentro del rango del piso (Lm=110 bm=18). Lo que de
    # verdad distingue al piso es la FORMA: es una franja que cruza el cuadro.
    # Medido en las dos fotos que motivaron la regla (183558155, 192411218):
    #     piso   bbox 872x133 y 127x854  ->  alargamiento 6.6 y 6.7
    #     hoja   bbox 660x312 y 294x731  ->  alargamiento 2.1 y 2.5
    #     hoja que llena el cuadro                     ->  alargamiento ~1.3
    # El corte en 4 cae en medio de esa brecha.
    es_piso = (Lm < 150) & (bm < 25) & (alarga > 4)
    puntaje = areas * llenado
    # salvaguarda: vetar solo si queda una alternativa comparable. Si lo unico que
    # sobrevive al veto son motas, el veto solo puede destruir -- que es justo lo
    # que pasaba en 173202752 y 164715384, donde la hoja perdia contra un pixel.
    if es_piso.any() and not es_piso.all():
        mejor_no_piso = puntaje[~es_piso].max()
        if mejor_no_piso >= 0.2 * puntaje[es_piso].max():
            puntaje = np.where(es_piso, 0.0, puntaje)
    i = int(np.argmax(puntaje)) + 1
    mayor = etq == i
    # cuanto del area candidata quedo fuera de la elegida (hoja ajena, piso, ruido)
    resto = float(areas.sum() - est[i, cv2.CC_STAT_AREA]) / max(1.0, float(areas.sum()))

    # 3. histeresis, con el umbral flojo MEDIDO EN CADA FOTO.
    #    El umbral b*>6 deja fuera el borde oscuro de las hojas verde oliva, y eso
    #    encoge el area justo en las hojas mas oscuras. Se crece desde la componente
    #    elegida hacia pixeles apenas calidos conectados a ella.
    #
    #    "Apenas calido" NO puede ser un numero fijo. Con b*>0 la mesa de Connecticut
    #    (azul marino, b* = -29..-2) queda fuera, pero la mesa del 28/08 es NEGRA y su
    #    b* ronda 0: medido sobre 30 fotos repartidas por las diez carpetas,
    #        fondo de Connecticut   b* p99 = -14 .. +3
    #        fondo de Habano        b* p99 =  +2 .. +4      <- el umbral 0 la deja pasar
    #    y por ahi se colaba. En 20260828_182624874 el 39 % de la mascara eran pixeles
    #    de b* = +1..+2, o sea mesa: la hoja ocupaba 0.59 del cuadro en vez de ~0.35.
    #    Lo destapo el detector de agujeros, que marcaba media foto como "hueco"
    #    porque, dentro de la mascara, esos pixeles se parecen al fondo -- y tenia
    #    razon: no eran hoja.
    #
    #    Asi que el umbral se pone por encima de la cola del fondo de ESA foto, y se
    #    limita por arriba a B_MIN para que la histeresis nunca sea mas dura que la
    #    semilla.
    lejos = ~(cv2.dilate(cruda.astype(np.uint8), disco(15)) > 0)
    if lejos.sum() > 5000:
        b_flojo = float(np.clip(np.percentile(b[lejos], 99) + 1, B_FLOJO, B_MIN))
    else:
        b_flojo = float(B_FLOJO)
    floja = (((b > b_flojo) & (b > K_AB * a) & (a < A_MAX) & (L > L_MIN))
             & ~cinta) | mayor
    floja = cv2.morphologyEx(floja.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    _, etqf = cv2.connectedComponents(floja, 8)
    tocadas = np.unique(etqf[mayor])
    mayor = np.isin(etqf, tocadas[tocadas > 0])

    # 4. rellenar huecos interiores (nervaduras, brillos, agujeros del defecto).
    #    Se rodea la mascara con un marco de fondo antes de inundar: si se siembra
    #    en (0,0) y la hoja toca esa esquina, se inunda la hoja y se rellena TODO.
    lleno = mayor.astype(np.uint8)
    h, w = lleno.shape
    marco = np.zeros((h + 2, w + 2), np.uint8)
    marco[1:-1, 1:-1] = lleno
    inv = (1 - marco).astype(np.uint8)
    cv2.floodFill(inv, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    fondo = (inv[1:-1, 1:-1] == 2)          # fondo alcanzable desde fuera del cuadro
    if crudo:
        return (~fondo), int(n - 1), resto, mayor
    return (~fondo), int(n - 1), resto


def metricas(hoja, marca, forma):
    h, w = forma
    area = int(hoja.sum())
    if area == 0:
        return dict(area_px=0, frac_cuadro=0.0, bbox="", solidez=0.0,
                    toca_borde=1, p_marca_en_hoja=0.0)
    ys, xs = np.nonzero(hoja)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cnts, _ = cv2.findContours(hoja.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    casco = cv2.contourArea(cv2.convexHull(c))
    borde = int(y0 == 0 or x0 == 0 or y1 == h - 1 or x1 == w - 1)
    return dict(
        area_px=area,
        frac_cuadro=round(area / (h * w), 4),
        bbox=f"{x0}:{y0}:{x1 - x0 + 1}:{y1 - y0 + 1}",
        solidez=round(cv2.contourArea(c) / casco, 4) if casco > 0 else 0.0,
        toca_borde=borde,
        p_marca_en_hoja=round(float((marca & hoja).sum()) / area, 6),
    )


def mascara_hoja_alta(ruta, forma_alta, margen=0.010):
    """La mascara VALIDADA, llevada a una imagen de alta resolucion.

    POR QUE EXISTE ESTA FUNCION (2026-09-05)
    ----------------------------------------
    Los detectores que trabajan a resolucion alta --`blanca_blob.py` a 4032 px--
    se hicieron su PROPIA mascara de hoja, con dos umbrales y un cierre de 25x25:

        m = (croma > 14) & (L > 25)

    Y se rompio como tenia que romperse. Lo vio el experto el 2026-09-05 en la prueba
    de la mancha blanca: **cinco de las seis hojas del tramo alto tenian el punto
    senalado SOBRE LA MESA**, no sobre la hoja. El cierre de 25x25 une la lamina
    con la mesa contigua, la componente mayor se traga las dos, y una mota clara
    sobre mesa oscura da el contraste mas alto de la foto -- asi que esos puntos
    falsos se colocaban los primeros de la lista.

    `mascara_hoja` ya sabia todo eso: lleva la regla del piso por alargamiento, la
    de la cinta, el veto del reflejo especular y la histeresis medida en cada foto
    (PROGRESO 14, 27, 32). Duplicarla con dos umbrales fue el error.

    Se calcula a LADO=1024, que es donde esta validada, y se escala. Y se ERODE un
    margen, porque el borde de la hoja tiene brillos de canto que a 4032 px son
    justo lo que un detector de motas claras confunde con una mancha.
    """
    rgb = carga(ruta, LADO)
    m = mascara_hoja(rgb, mascara_marca(rgb))[0]
    H, W = forma_alta[:2]
    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    r = max(9, int(margen * np.sqrt(max(1.0, float(m.sum())))))
    m = cv2.erode(m, disco(r))
    return m > 0


def superpone(rgb, hoja, marca):
    v = rgb.copy()
    fuera = ~hoja
    v[fuera] = (0.45 * v[fuera] + 0.55 * np.array([20, 20, 20])).astype(np.uint8)
    cnts, _ = cv2.findContours(hoja.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cnts, -1, (255, 0, 255), 3)      # contorno de hoja: magenta
    v[marca] = (0, 255, 0)                                # marcador detectado: verde
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", type=int, default=0, help="cuantas superposiciones escribir")
    ap.add_argument("--solo", default="", help="subcadena para filtrar archivos o carpeta")
    ap.add_argument("--rehacer", action="store_true",
                    help="ignora el CSV existente y vuelve a segmentar todo")
    ap.add_argument("--muestra", type=int, default=0,
                    help="N fotos por carpeta, repartidas a lo largo de la sesion")
    a = ap.parse_args()

    man = list(csv.DictReader(open(OUT / "manifiesto_limpio.csv", encoding="utf-8")))
    if a.solo:
        man = [r for r in man if a.solo in r["archivo"] or a.solo in r["ruta"]]
    if a.muestra:
        # reparte la muestra a lo largo de cada carpeta: la luz cambia dentro del
        # bloque horario, asi que las primeras N fotos no representan la sesion
        porc = defaultdict(list)
        for r in man:
            porc[r["carpeta"]].append(r)
        man = []
        for c in sorted(porc):
            v = porc[c]
            idx = np.linspace(0, len(v) - 1, min(a.muestra, len(v))).astype(int)
            man += [v[i] for i in sorted(set(idx))]
    print(f"imagenes a segmentar: {len(man)}")

    (OUT / "mascaras").mkdir(parents=True, exist_ok=True)
    qc_dir = OUT / "qc_segmentacion"
    if a.qc:
        qc_dir.mkdir(parents=True, exist_ok=True)
        idx = np.linspace(0, len(man) - 1, min(a.qc, len(man))).astype(int)
        qc = {man[i]["ruta"] for i in idx}
    else:
        qc = set()

    # Reanudable: en esta maquina los procesos largos se matan solos (PROGRESO 30.4).
    # Cada fila se escribe en cuanto la mascara esta guardada; al relanzar se saltan
    # las fotos que ya figuran en el CSV.
    parcial = bool(a.solo or a.muestra)
    fsal = OUT / ("segmentacion_parcial.csv" if parcial else "segmentacion.csv")
    campos = ["ruta", "archivo", "hoja_id", "area_px", "frac_cuadro", "bbox", "solidez",
              "toca_borde", "p_marca_en_hoja", "n_componentes", "frac_otras_componentes",
              "ancho", "alto"]
    filas = []
    hechas = set()
    if not parcial and fsal.exists() and not a.rehacer:
        filas = [r0 for r0 in csv.DictReader(open(fsal, encoding="utf-8")) if r0.get("ruta")]
        hechas = {r0["ruta"] for r0 in filas}
        print("ya segmentadas: {}".format(len(hechas)))
    nuevo_csv = parcial or not hechas
    fh = open(fsal, "w" if nuevo_csv else "a", newline="", encoding="utf-8")
    wcsv = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
    if nuevo_csv:
        wcsv.writeheader()
    man = [r for r in man if r["ruta"] not in hechas]
    print("pendientes: {}".format(len(man)))
    for i, r in enumerate(man, 1):
        p = RAIZ / r["ruta"]
        rgb = carga(p)
        marca = mascara_marca(rgb)
        hoja, n_comp, resto = mascara_hoja(rgb, marca)
        m = metricas(hoja, marca, rgb.shape[:2])
        m.update(ruta=r["ruta"], archivo=r["archivo"], hoja_id=r["hoja_id"],
                 n_componentes=n_comp, frac_otras_componentes=round(resto, 4),
                 alto=rgb.shape[0], ancho=rgb.shape[1])
        filas.append(m)
        wcsv.writerow(m)
        fh.flush()
        np.save(OUT / "mascaras" / (Path(r["archivo"]).stem + ".npy"),
                np.packbits(hoja, axis=None))
        if r["ruta"] in qc:
            cv2.imwrite(str(qc_dir / (Path(r["archivo"]).stem.replace(" ", "_") + ".jpg")),
                        cv2.cvtColor(superpone(rgb, hoja, marca), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        if i % 25 == 0:
            print(f"  {i}/{len(man)}")

    fh.close()
    f = fsal

    # `filas` mezcla dicts recien calculados (numeros) y filas releidas del CSV
    # (cadenas), asi que se convierte antes de resumir
    fc = np.array([float(x["frac_cuadro"]) for x in filas])
    so = np.array([float(x["solidez"]) for x in filas])
    print(f"\ncobertura de hoja: min={fc.min():.3f} p25={np.percentile(fc,25):.3f} "
          f"mediana={np.median(fc):.3f} p75={np.percentile(fc,75):.3f} max={fc.max():.3f}")
    print(f"solidez:           min={so.min():.3f} mediana={np.median(so):.3f}")
    print(f"tocan el borde del cuadro: {sum(int(x['toca_borde']) for x in filas)}/{len(filas)}")
    print(f"cobertura < 0.05 (posible fallo): {int((fc < 0.05).sum())}")
    print(f"\n-> {f}")
    if a.qc:
        print(f"-> {qc_dir}  ({len(qc)} superposiciones)")


if __name__ == "__main__":
    main()
