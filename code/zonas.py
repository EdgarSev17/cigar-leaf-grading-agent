r"""
Divide cada hoja en las zonas que la regla de decision necesita (PROGRESO 16).

LA REGLA, RECORDADA
-------------------
Una hoja es Capa si en la parte que el cliente usa queda una region continua sin
defectos lo bastante grande para envolver un puro. Las zonas que NO se usan, segun
las siete explicaciones del experto (PROGRESO 16.2):

    base    "normalmente en la parte de abajo no se utiliza"
    punta   "tiene danada la punta y la parte de la base tambien... pasa como capa"
    vena    "los agujeros cerquita de la vena no importan, esta parte se desecha"

Lo que se usa es el CONTORNO, o sea las dos mitades laterales. Y la clase la da el
lado donde sobrevive una region limpia suficiente:

    las dos mitades sirven  -> Capa
    solo la izquierda       -> XL Izquierdo
    solo la derecha         -> XR Derecho

Este guion construye esa geometria. NO detecta defectos todavia: prepara el terreno
para que, cuando el detector exista, "la mayor region limpia de cada mitad" sea una
cuenta de dos lineas.

DE DONDE SALE CADA COSA
-----------------------
  mascara de hoja      out/mascaras/  (segmentacion.py)
  orientacion          out/enderezado.csv  (rotacion + volteo por la cinta)
  escala               out/escala_cinta.csv  (ancho de cinta en px)
  extremo de la base   se deduce aqui, del pecioolo (ver abajo)

COMO SE ENCUENTRA LA BASE
-------------------------
Recorriendo el eje mayor y midiendo la anchura perpendicular en 60 tramos, el
extremo de la base arrastra un RABO FINO -- el pecioolo -- mucho mas largo que el
de la punta. Medido sobre las 505 hojas enteras del conjunto:

    rabo con anchura < 18 % del maximo, en sesentavos del largo
    punta: mediana 1-2        base: mediana 6-8

Se probo antes "el punto mas ancho de la hoja" y no sirve: cae en 0.44 del eje con
p25=0.38 y p75=0.51, demasiado cerca del centro y demasiado disperso.

La diferencia entre los dos rabos es la CONFIANZA, y no es un umbral inventado
sino la resta de dos longitudes medidas. Cuando los dos extremos se parecen, el
sistema no sabe cual es la base: eso se marca `base_dudosa` y es un motivo legitimo
para pedir otra vista (PROGRESO 18.3).

LOS TAMANOS DE LAS ZONAS ESTAN PENDIENTES, Y SE DICE
-----------------------------------------------------
`BASE_PULG`, `PUNTA_PULG` y `VENA_PULG` son las anchuras de cada zona. Hoy son
valores de arranque, no medidos: la unica cifra que dio el experto es "las ~2 pulgadas
de abajo" (PROGRESO 11), y el cuaderno del 24/08 avisa de que la zona cambia con el
cliente y con el tamano de la hoja. Se expresan en ANCHOS DE CINTA para que, en
cuanto se sepa cuanto mide la cinta, queden en milimetros sin tocar el codigo.

Uso:
    python scripts/zonas.py --muestra 8 --qc 40
    python scripts/zonas.py --qc 60
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from rutas import REPO_ROOT  # repository root

OUT = REPO_ROOT / "out"

N_TRAMOS = 60
FRAC_RABO = 0.18          # anchura por debajo de la cual el tramo es "rabo"
MARGEN_BASE = 3           # sesentavos de diferencia para fiarse del extremo

PULG_POR_ANCHO_CINTA = 1.11

# ----------------------------------------------------------------------------
# LAS BANDAS PERMISIBLES, MEDIDAS (2026-09-01). Ya no son provisionales.
#
# El experto dibujo los limites sobre dos hojas:
#     dataset/Limites para agujeros en capa (Hoja Grande).jpg
#     dataset/Limites para agujeros en capa (Hoja Pequena).jpg
# y `scripts/limites.py` las midio con esta misma tuberia. Aplica a las dos
# variedades (dicho por el).
#
#                        lamina      punta permisible   base permisible
#     hoja GRANDE       16.9 pulg         12.0 %            17.6 %
#     hoja PEQUENA      14.7 pulg          3.5 %             8.5 %
#
# Las fracciones son de la LAMINA (hoja sin peciolo) y se miden desde los
# extremos de la lamina, no desde la punta del peciolo: el peciolo mide entre 140
# y 341 px segun la foto, y meterlo en el denominador hacia que la hoja "pequena"
# saliera MAS larga que la "grande".
#
# Entre las dos hojas se interpola linealmente y fuera del rango se recorta. Son
# dos puntos, no una curva: si aparecen mas hojas marcadas, esta tabla crece.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# EL ESTANDAR DE ZONA USABLE (2026-09-15)
#
# Medido sobre las hojas que el experto marco con leyenda en
# `dataset/LIMITES Y ZONAS USABLES_ CONNT Y HABANO/`:
#     morado = zona usable   rojo = lo que corta la chaveta (bordes + vena)
#     cian   = limite superior y limite inferior
#
#                    lamina    punta    base    borde
#     CNT grande      18.1"    1.68"   3.76"   0.71"
#     CNT pequena     13.4"    1.41"   3.20"   0.30"
#     HAB grande      17.3"    0.88"   5.28"   0.45"
#
# Lo que dicen las tres: la PUNTA es casi la misma en todas (~1.3"), mientras que
# la BASE y el BORDE crecen con la hoja. Lo del borde lo dijo el experto antes de que
# lo midieramos: "claramente una hoja mas grande, mas se dana su borde".
#
# El limite lo miden a ojo en la planta; el estandar de aqui lo fijamos nosotros,
# por encargo suyo, en PULGADAS -- no en fraccion de la lamina, que es lo que
# habia antes y dejaba la hoja pequena casi sin descartar (0.47" de punta y 1.14"
# de base, cuando el marca 1.41" y 3.20").
#
# OJO: esto CONTRADICE la tabla anterior, sacada de dos hojas marcadas sin
# leyenda (base de 1.25" y 2.97"). Se queda la nueva porque se sabe que
# significa cada color; la vieja queda abajo comentada por si hay que volver.
#     LIM_PEQUENA = (14.7, 0.035, 0.085)
#     LIM_GRANDE  = (16.9, 0.120, 0.176)
# ----------------------------------------------------------------------------
# EL INTERRUPTOR (2026-09-15, al cerrar el dia). El estandar nuevo esta medido y
# programado, pero NO adoptado: la prueba de aceptacion fallo. Al descontar la
# franja del filo, la banda baja de 5 a 3 defectos en zona util -- pero la XL
# izquierda baja de 4 a 1, porque **buena parte del dano que define a XL y XR
# esta justo en el filo**. Se le quita ruido a la banda y señal a la XL.
#
# Y sobre todo: el modelo instalado se entreno con la zona VIEJA. Dejar la nueva
# encendida haria que `clasifica.py` midiera distinto de como se entreno, que es
# el fallo de la escala otra vez. Para adoptarlo hay que, en este orden:
#   1) ESTANDAR = "nuevo"
#   2) recalcular zonas.csv, agujeros.csv y manchas.csv del dataset
#   3) reentrenar
#   4) medir sobre las 112 y comparar contra el 82.1 % de hoy
ESTANDAR = "viejo"        # "viejo" | "nuevo"

if ESTANDAR == "nuevo":
    LIM_PEQUENA = (13.5, 1.35, 3.20, 0.30)   # (lamina", punta", base", borde")
    LIM_GRANDE = (18.0, 1.35, 4.50, 0.60)
else:
    # la tabla de antes, en FRACCION de lamina y sin borde (borde = 0)
    # tal y como estaba: FRACCIONES de la lamina de cada hoja, sin borde
    LIM_PEQUENA = (14.7, 0.035, 0.085, 0.0)
    LIM_GRANDE = (16.9, 0.120, 0.176, 0.0)

VENA_ANCHOS = 0.6         # ~0.65" medido en la CNT grande: coincide con este valor

# compatibilidad hacia atras; ya no se usan para decidir
BASE_ANCHOS = 2.0
PUNTA_ANCHOS = 1.0


def lamina(t, anch, n_tramos=N_TRAMOS):
    """(inicio, fin) de la LAMINA sobre el eje: la hoja sin el rabo del peciolo."""
    t0, t1 = float(t.min()), float(t.max())
    largo = t1 - t0
    aa = anch / max(1e-6, anch.max())
    idx = np.nonzero(aa >= FRAC_RABO)[0]
    if len(idx) == 0:
        return t0, t1
    return (t0 + (idx[0] / n_tramos) * largo,
            t0 + ((idx[-1] + 1) / n_tramos) * largo)


def bandas(lam_px, pulg_px, ancho_cinta_px):
    """(punta_px, base_px, vena_px, borde_px) para una lamina de `lam_px` px.

    `pulg_px` puede ser None: entonces se usa el punto medio de la tabla, porque
    sin escala no se sabe si la hoja es grande o pequena.
    """
    (l0, p0, b0, e0), (l1, p1, b1, e1) = LIM_PEQUENA, LIM_GRANDE
    u = (float(np.clip((lam_px * pulg_px - l0) / (l1 - l0), 0.0, 1.0))
         if pulg_px else 0.5)
    pu, ba, bo = (p0 + u * (p1 - p0)), (b0 + u * (b1 - b0)), (e0 + u * (e1 - e0))
    if ESTANDAR == "nuevo" and pulg_px:
        # el estandar nuevo esta en PULGADAS: a pixeles con la escala de la foto
        punta, base, borde = pu / pulg_px, ba / pulg_px, bo / pulg_px
    elif ESTANDAR == "nuevo":
        # sin escala no se sabe el tamano: se usa la fraccion tipica equivalente
        punta, base, borde = (pu / 15.75 * lam_px, ba / 15.75 * lam_px,
                              bo / 15.75 * lam_px)
    else:
        # el de siempre: FRACCION de la lamina de esta hoja
        punta, base, borde = pu * lam_px, ba * lam_px, bo * lam_px
    vena = VENA_ANCHOS * (ancho_cinta_px if ancho_cinta_px else 0.12 * lam_px)
    return punta, base, vena, borde


def ancho_cinta_por_carpeta(OUT_DIR):
    """Ancho de cinta en px DE LA MASCARA, por carpeta.

    `escala_cinta.py` trabaja a lado 1800 y `segmentacion.py` a 1024: durante
    varias sesiones el ancho de cinta se aplico sin convertir, o sea 1.76 veces
    mas grande de lo que toca. Todas las zonas salian 1.76x mas anchas y todas
    las "pulgadas" del proyecto eran el 57 % de su valor. Corregido el 2026-09-01.
    """
    import csv as _csv
    seg = {r["ruta"]: r for r in
           _csv.DictReader(open(OUT_DIR / "segmentacion.csv", encoding="utf-8"))}
    porc = defaultdict(list)
    for r in _csv.DictReader(open(OUT_DIR / "escala_cinta.csv", encoding="utf-8")):
        if not r.get("ancho_cinta_px") or not r.get("alto_trabajo"):
            continue
        s = seg.get(r["ruta"])
        if not s:
            continue
        f = float(s["alto"]) / float(r["alto_trabajo"])
        porc[r["carpeta"]].append(float(r["ancho_cinta_px"]) * f)
    return {c: float(np.median(v)) for c, v in porc.items()}


def perfil(m):
    """(eje, centro, t, s, anchura por tramo) de una mascara."""
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mu = pts.mean(0)
    val, vec = np.linalg.eigh(np.cov((pts - mu).T))
    e = vec[:, 1]
    if e[1] < 0:
        e = -e
    perp = np.array([-e[1], e[0]], dtype=np.float32)
    t = (pts - mu) @ e
    s = (pts - mu) @ perp
    idx = np.clip(((t - t.min()) / max(1e-6, t.max() - t.min()) * N_TRAMOS)
                  .astype(int), 0, N_TRAMOS - 1)
    anch = np.zeros(N_TRAMOS)
    for i in range(N_TRAMOS):
        sel = idx == i
        if sel.sum() > 3:
            anch[i] = s[sel].max() - s[sel].min()
    return e, perp, mu, t, s, anch


VENA_SUAV_FRAC = 0.08     # ventana de la mediana, en fraccion del largo del eje
VENA_TRAMOS = 4 * N_TRAMOS


def eje_vena(t, s, n_tramos=VENA_TRAMOS, suav_frac=VENA_SUAV_FRAC):
    """Eje de la VENA CENTRAL en el marco (t, s): (tt, cc) para np.interp.

    el experto, 2026-09-13 (Tabaco 84.3), literal: «DEBES USAR LA VENA CENTRAL PARA
    DIVIDIR LA HOJA ENTRE IZQUIERDO Y DERECHO».

    Hasta hoy el lado se decidia con s = 0, que es una recta por el CENTROIDE de
    la silueta. Se rompe justo cuando importa: si a la hoja le falta un pedazo
    de un lado, el centroide se corre hacia el lado sano y la raya se va con el,
    o sea que el propio dano mueve la linea que decide de que lado esta. Seis de
    las catorce hojas de 84 tienen la marca del experto pegada a la vena por la
    izquierda, que es la franja donde ese error cambia la respuesta.

    Es la misma idea que agarre.zona_descartada (centro fila a fila + mediana de
    ~41 mm para alisar el borde), traida al marco del eje mayor, que es donde
    trabajan agujeros.py y manchas.py.
    """
    t = np.asarray(t, np.float64).ravel()
    s = np.asarray(s, np.float64).ravel()
    t0, t1 = float(t.min()), float(t.max())
    largo = max(1e-6, t1 - t0)
    idx = np.clip(((t - t0) / largo * n_tramos).astype(np.int64), 0, n_tramos - 1)
    tt = t0 + (np.arange(n_tramos) + 0.5) / n_tramos * largo

    orden = np.argsort(idx, kind="stable")
    idx_o, s_o = idx[orden], s[orden]
    ini = np.searchsorted(idx_o, np.arange(n_tramos), side="left")
    fin = np.searchsorted(idx_o, np.arange(n_tramos), side="right")
    cc = np.full(n_tramos, np.nan)
    for i in range(n_tramos):
        if fin[i] - ini[i] > 3:
            tr = s_o[ini[i]:fin[i]]
            cc[i] = 0.5 * (float(tr.min()) + float(tr.max()))

    val = np.nonzero(~np.isnan(cc))[0]
    if len(val) == 0:
        return tt, np.zeros(n_tramos)
    cc = np.interp(np.arange(n_tramos), val, cc[val])   # rellena tramos vacios

    # scipy se importa aqui y no arriba: zonas.py lo carga media tuberia y
    # no tiene por que arrastrar la dependencia si nadie llama a esta funcion.
    from scipy.ndimage import median_filter
    k = max(3, int(round(suav_frac * n_tramos)) | 1)
    cc = median_filter(cc, size=k, mode="nearest")
    return tt, cc


def lado_vena(tc, sc, tt, cc, vena_px):
    """Zona lateral de un punto (tc, sc) respecto del eje de la vena.

    Devuelve ("vena" | "util_izq" | "util_der", distancia con signo a la vena).
    El signo es el mismo que tenia `sc`: positivo = izquierda (agujeros.py:507).
    """
    d = float(sc) - float(np.interp(tc, tt, cc))
    if abs(d) < vena_px / 2.0:
        return "vena", d
    return ("util_izq" if d > 0 else "util_der"), d


def extremo_base(anch):
    """(base_al_final, margen). base_al_final=True si la base esta en t.max()."""
    if anch.max() <= 0:
        return True, 0
    a = anch / anch.max()
    ini = 0
    while ini < N_TRAMOS and a[ini] < FRAC_RABO:
        ini += 1
    fin = 0
    while fin < N_TRAMOS and a[N_TRAMOS - 1 - fin] < FRAC_RABO:
        fin += 1
    return (fin >= ini), int(abs(fin - ini))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", type=int, default=0)
    ap.add_argument("--qc", type=int, default=0)
    a = ap.parse_args()

    seg = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "segmentacion.csv", encoding="utf-8"))}
    end = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "enderezado.csv", encoding="utf-8"))} \
        if (OUT / "enderezado.csv").exists() else {}
    ancho_cinta = (ancho_cinta_por_carpeta(OUT)
                   if (OUT / "escala_cinta.csv").exists() else {})
    print("segmentacion: {}   enderezado: {}   carpetas con escala: {}"
          .format(len(seg), len(end), len(ancho_cinta)))

    rutas = sorted(seg)
    if a.muestra:
        porc = defaultdict(list)
        for r in rutas:
            porc["/".join(r.split("/")[:2])].append(r)
        rutas = []
        for c in sorted(porc):
            v = porc[c]
            idx = np.linspace(0, len(v) - 1, min(a.muestra, len(v))).astype(int)
            rutas += [v[i] for i in sorted(set(idx))]

    qc_dir = OUT / "qc_zonas"
    if a.qc:
        qc_dir.mkdir(parents=True, exist_ok=True)
        sel = {rutas[i] for i in
               np.linspace(0, len(rutas) - 1, min(a.qc, len(rutas))).astype(int)}
    else:
        sel = set()

    filas = []
    for n_i, ruta in enumerate(rutas, 1):
        r = seg[ruta]
        f = OUT / "mascaras" / (Path(r["archivo"]).stem + ".npy")
        if not f.exists():
            continue
        h, w = int(r["alto"]), int(r["ancho"])
        m = np.unpackbits(np.load(f))[:h * w].reshape(h, w).astype(bool)
        if m.sum() < 500:
            continue

        # 1. orientar: rotacion de la cinta y volteo, si se conocen
        e_row = end.get(ruta)
        rot = 0.0
        voltea = 0
        if e_row and e_row.get("rotacion") not in ("", None):
            rot = float(e_row["rotacion"])
            voltea = int(e_row["voltea_180"] or 0)
        if rot:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            m = cv2.warpAffine(m.astype(np.uint8), M, (nw, nh),
                               flags=cv2.INTER_NEAREST) > 0
        if voltea:
            m = cv2.rotate(m.astype(np.uint8), cv2.ROTATE_180) > 0
        if m.sum() < 500:
            continue

        # 2. eje, perfil de anchura y extremo de la base
        e, perp, mu, t, s, anch = perfil(m)
        base_al_final, margen = extremo_base(anch)
        base_al_final_orig = base_al_final
        # Marco canonico: la base SIEMPRE al final del eje. Si el pecioolo salio
        # arriba, se da la vuelta al eje. Hay que hacerlo antes de hablar de
        # izquierda y derecha: con la hoja invertida, los dos lados se
        # intercambian, y ese es exactamente el error que PROGRESO 13.1 avisa que
        # no se nota hasta el final.
        if not base_al_final:
            e, perp, t, s = -e, -perp, -t, -s
            anch = anch[::-1]
            base_al_final = True
        largo = float(t.max() - t.min())
        ancho_max = float(anch.max())

        # 3. zonas, con las bandas MEDIDAS sobre las dos hojas que marco el experto
        carp = "/".join(ruta.split("/")[:2])
        ac = ancho_cinta.get(carp)
        pulg_px = (PULG_POR_ANCHO_CINTA / ac) if ac else None
        lam0, lam1 = lamina(t, anch)
        lam = lam1 - lam0
        punta_px, base_px, vena_px, borde_px = bandas(lam, pulg_px, ac)

        t0, t1 = t.min(), t.max()
        # el peciolo (mas alla de lam1) no es hoja util: entra en la banda de base
        en_base = t > (lam1 - base_px)
        en_punta = t < (lam0 + punta_px)
        en_vena = np.abs(s) < (vena_px / 2.0)
        util = ~(en_base | en_punta | en_vena)
        # El lado. Con el eje `e` apuntando hacia la base y
        # `perp = (-e[1], e[0])`, un punto a la IZQUIERDA de la hoja da s > 0.
        # Se comprobo a ojo en las superposiciones de QC: con el signo al reves,
        # `izq` se pintaba sobre la mitad derecha.
        izq = util & (s > 0)
        der = util & (s < 0)

        area = float(m.sum())
        fila = dict(
            ruta=ruta, archivo=r["archivo"], carpeta=carp,
            rot=round(rot, 2), voltea=voltea,
            base_invertida=int(not base_al_final_orig), margen_base=margen,
            base_dudosa=int(margen < MARGEN_BASE),
            largo_px=round(largo), lamina_px=round(lam),
            lamina_pulg=(round(lam * pulg_px, 2) if pulg_px else ""),
            punta_px=round(punta_px), base_px=round(base_px),
            ancho_max_px=round(ancho_max),
            ancho_cinta_px=round(ac, 1) if ac else "",
            largo_anchos=round(largo / ac, 2) if ac else "",
            frac_base=round(float(en_base.sum()) / area, 4),
            frac_punta=round(float(en_punta.sum()) / area, 4),
            frac_vena=round(float(en_vena.sum()) / area, 4),
            frac_util=round(float(util.sum()) / area, 4),
            frac_util_izq=round(float(izq.sum()) / area, 4),
            frac_util_der=round(float(der.sum()) / area, 4),
            escala="cinta" if ac else "relativa",
        )
        filas.append(fila)

        if ruta in sel:
            vis = np.zeros(m.shape + (3,), np.uint8)
            ys, xs = np.nonzero(m)
            col = np.zeros((len(ys), 3), np.uint8)
            col[:] = (60, 60, 60)
            col[en_base] = (0, 0, 200)          # base   rojo
            col[en_punta] = (0, 140, 220)       # punta  naranja
            col[en_vena] = (0, 200, 200)        # vena   amarillo
            col[izq] = (80, 200, 80)            # util izquierda  verde
            col[der] = (200, 120, 60)           # util derecha    azul
            vis[ys, xs] = col
            k = max(1, vis.shape[0] // 700)
            vis = vis[::k, ::k]
            cv2.imwrite(str(qc_dir / (Path(r["archivo"]).stem.replace(" ", "_") + ".jpg")),
                        vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if n_i % 100 == 0:
            print("  {}/{}".format(n_i, len(rutas)))

    campos = ["ruta", "carpeta", "archivo", "rot", "voltea", "base_invertida",
              "margen_base", "base_dudosa", "largo_px", "lamina_px", "lamina_pulg",
              "punta_px", "base_px", "ancho_max_px",
              "ancho_cinta_px", "largo_anchos", "frac_base", "frac_punta",
              "frac_vena", "frac_util", "frac_util_izq", "frac_util_der", "escala"]
    f = OUT / ("zonas_parcial.csv" if a.muestra else "zonas.csv")
    tmp = f.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(filas)
    tmp.replace(f)

    print("\n== extremo de la base, DESPUES de enderezar ==")
    print("  {:46s} {:>5s} {:>12s} {:>10s}".format(
        "carpeta", "n", "base abajo", "dudosas"))
    print("  " + "-" * 78)
    por = defaultdict(list)
    for r in filas:
        por[r["carpeta"]].append(r)
    for c in sorted(por):
        v = por[c]
        ba = 1.0 - np.mean([x["base_invertida"] for x in v])
        du = np.mean([x["base_dudosa"] for x in v])
        print("  {:46s} {:5d} {:11.0%} {:10.0%}".format(c, len(v), ba, du))

    print("\n== reparto de area por zona (mediana) ==")
    print("  {:46s} {:>7s} {:>7s} {:>7s} {:>8s} {:>7s} {:>7s}".format(
        "carpeta", "base", "punta", "vena", "util", "izq", "der"))
    print("  " + "-" * 96)
    for c in sorted(por):
        v = por[c]
        g = lambda k: np.median([x[k] for x in v])
        print("  {:46s} {:7.1%} {:7.1%} {:7.1%} {:8.1%} {:7.1%} {:7.1%}".format(
            c, g("frac_base"), g("frac_punta"), g("frac_vena"),
            g("frac_util"), g("frac_util_izq"), g("frac_util_der")))

    n_esc = sum(1 for r in filas if r["escala"] == "cinta")
    print("\n{} hojas; {} con escala de cinta, {} con zonas relativas al largo"
          .format(len(filas), n_esc, len(filas) - n_esc))
    print("Las anchuras de zona son PROVISIONALES (ver cabecera): faltan las")
    print("medidas del experto. Cambiarlas no toca el codigo, solo las constantes.")
    print("\n-> {}".format(f))
    if a.qc:
        print("-> {}".format(qc_dir))


if __name__ == "__main__":
    main()
