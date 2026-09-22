r"""
Calibra la escala de cada foto leyendo las graduaciones de la cinta metrica.

POR QUE ES AHORA OBLIGATORIO
----------------------------
La regla de decision (PROGRESO 16) no depende de fracciones de area sino de
TAMANOS ABSOLUTOS: si queda una region limpia lo bastante grande para envolver un
puro. Y el propio el experto decide citando pulgadas: *"es una hoja grande, como de 18
pulgadas, entonces esa parte no se utiliza"*. Sin px -> pulgadas no hay regla que
programar, solo un clasificador opaco.

COMO SE MIDE
------------
1. La cinta es lo unico del cuadro claro, neutro y muy alargado (misma firma que
   `enderezado.py`). Se toma la componente conexa dominante.
2. Dentro de la cinta, una GRADUACION es una fila (a lo largo del eje de la cinta)
   donde una fraccion alta del ancho es oscura. Los digitos impresos tambien son
   oscuros, asi que las detecciones mezclan marcas y numeros.
3. El paso NO se estima con la mediana de las diferencias: los digitos meten
   diferencias espurias. Se estima con un **periodograma sobre el tren de
   impulsos** -- para cada periodo candidato P se mide cuanto se alinean las marcas
   con un peine de paso P. Los digitos, al caer entre marcas, no forman un peine
   propio y no ganan.
4. Se devuelve `px_por_graduacion` y, si existe la mascara de segmentacion, el
   largo y ancho de la hoja EN GRADUACIONES.

VALIDACION INTERNA, SIN NECESIDAD DE QUE NADIE CONFIRME NADA
------------------------------------------------------------
La camara estuvo a una altura parecida durante toda una sesion, asi que
`px_por_graduacion` tiene que salir **aproximadamente constante dentro de cada
carpeta**. La dispersion de esa cifra es la medida de si el detector funciona:
poca dispersion = lectura fiable; mucha = el detector esta leyendo ruido, y
entonces no se usa. El guion imprime esa dispersion y no la esconde.

LA REFERENCIA BUENA RESULTO SER EL ANCHO DE LA CINTA, NO SU PASO
-----------------------------------------------------------------
El paso entre graduaciones se estima por periodograma y a veces cae en un
armonico: sobre una muestra de 10 fotos por carpeta salio 26.5 px en Connecticut
y entre 67 y 118 px en Habano, un factor 4.5 de diferencia que no puede ser real.

El **ancho de la cinta**, en cambio, es el mismo objeto fisico en las 1177 fotos y
se mide sin ambiguedad (no hay armonicos en una anchura). Medido: **72 px en
Connecticut y 77-84 px en Habano**, o sea que la camara estuvo casi a la misma
altura los tres dias, con un 17 % de variacion. Esa es la escala.

La comprobacion de que sirve: expresando el largo de la hoja en ANCHOS DE CINTA,
las seis carpetas con lectura fiable dan **19 a 24**, incluida Connecticut, que con
el paso daba un disparate. Un 20 % de dispersion, que es variacion real de tamano
de hoja. El paso de graduaciones se conserva como diagnostico y como validacion
cruzada, no como escala.

LO QUE ESTE GUION NO SABE
-------------------------
**Cuanto mide de ancho la cinta.** Es UNA medida con una regla, y con ella todo el
dataset queda en milimetros: si la cinta mide W mm de ancho, entonces
`mm_por_px = W / ancho_cinta_px` en cada foto. Poner el valor en `ANCHO_CINTA_MM`.


**Que unidad es una graduacion.** El paso se mide en pixeles y las etiquetas
visibles van 11, 13, 15, 17, 19, 21 (numeros impares, con marcas intermedias), asi
que una graduacion puede ser 1 unidad o media. Convertir a pulgadas necesita una
frase del experto: si la cinta esta en pulgadas y si las marcas van de 1 en 1. En
cuanto lo diga, `UNIDAD_POR_GRADUACION` de abajo lo cierra todo.

Uso:
    python scripts/escala_cinta.py --muestra 8        # prueba rapida por carpeta
    python scripts/escala_cinta.py                    # todas
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
LADO = 1800                 # mas fino que la segmentacion: las marcas son finas

L_MIN_CINTA = 170
ALARGA_MIN = 6.0
AREA_MIN = 0.004
CAIDA_L = 35                # cuanto mas oscura que la cinta debe ser una marca
FRAC_ANCHO = 0.30           # que parte del ancho de la cinta debe cruzar
# Rango de periodos candidatos, en px. Acotado el 2026-08-30 con lo que explico
# El experto: la cinta esta graduada EN PULGADAS, con una marca por pulgada y numero
# cada dos (11, 13, 15...), y las hojas miden entre 11 y 16 pulgadas. Como la hoja
# ocupa buena parte del cuadro, una pulgada no puede medir menos de ~50 px. Sin
# esta cota el periodograma caia en subarmonicos (26.5 px en Connecticut, que es
# un tercio del paso real).
# Acotado el 2026-08-30 contra VERDAD-TERRENO: en
# `dataset/Medida de una hoja de capa connecticut.jpg` el experto dibujo las lineas de
# 0 y 11 a 18 pulgadas sobre la cinta. De ahi, 70.4 px por pulgada a escala 1800.
# El periodograma sobre esa misma foto devuelve 68.0 px -> **3 % de error**, o sea
# que el metodo es correcto; lo que fallaba era el rango, que dejaba entrar
# armonicos (Habano salia en 138-200 px, el doble del real).
# Como el ancho de cinta medido va de 79 a 97 px en las diez carpetas, la camara
# estuvo casi a la misma altura siempre y el paso real no puede alejarse mucho de
# esos 70 px. Se acota a [55, 110].
P_MIN, P_MAX = 55, 110

ANCHO_CINTA_MM = None      # <- medir la cinta con una regla y poner el valor aqui
UNIDAD_POR_GRADUACION = None


def carga(p, lado=LADO):
    with Image.open(p) as im:
        try:
            im.draft("RGB", (im.width // 2, im.height // 2))
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((lado, lado), Image.LANCZOS)
        return np.asarray(im)


def busca_cinta(rgb):
    """(etiquetas, indice) de la componente que es la cinta, o (None, None)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    # L>170, no 150: por debajo entra el REFLEJO especular de la mesa, que es
    # claro y azulado igual que la cinta. Medido sobre la componente detectada,
    # subir el umbral de 150 a 170 estrecha el ancho de la cinta de Connecticut de
    # 117 px (p10=88 p90=146, o sea desbordada) a 84 px (p10=68 p90=88), y deja
    # la de Habano casi igual (76-81 px). Con eso las dos variedades caen en el
    # mismo rango, 63-84 px.
    #
    # La cinta es CLARA y NO CALIDA. No "neutra": medido, la cinta de Connecticut
    # es blanco azulado (b* = -9..-24, mediana -16) porque refleja la mesa azul
    # marino, y el filtro |b*|<14 la partia en mil pedazos -- de ahi que en esas
    # carpetas la escala saliera con 41-149 % de dispersion. La de Habano es
    # neutra (b* ~ 0). Las dos pasan con "b* por debajo de +8"; el piso de
    # concreto (b* = +4..+27) y la hoja (b* >= +17) no.
    m = (L > L_MIN_CINTA) & (a > -8) & (a < 6) & (b < 8)
    # cierre grande: las marcas de graduacion cruzan la cinta y la trocean; con un
    # nucleo de 5 px quedaban 1000-2000 componentes en vez de una tira
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    n, etq, est, _ = cv2.connectedComponentsWithStats(m8, 8)
    h, w = m.shape
    # Como se elige la cinta entre las tiras claras del cuadro.
    #
    # Por AREA no: en Connecticut el piso de concreto tambien es claro y neutro y
    # en area le gana; el guion media el piso y la escala salia con 41-149 % de
    # dispersion. Por ALARGAMIENTO tampoco basta: mejora poco, porque una franja
    # de piso larga tambien es alargada.
    #
    # Lo que define a la cinta y no tiene ninguna otra cosa del cuadro es que
    # **esta graduada**: lleva marcas oscuras a intervalos regulares. Asi que se
    # puntua a cada candidata por la periodicidad de sus marcas (el pico del
    # periodograma del tren de impulsos) y gana la mas periodica. El piso tiene
    # manchas y juntas, pero no un peine.
    cands = []
    for j in range(1, n):
        area = est[j, cv2.CC_STAT_AREA]
        if area < AREA_MIN * h * w:
            continue
        bw, bh = float(est[j, cv2.CC_STAT_WIDTH]), float(est[j, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) / max(1.0, min(bw, bh)) < ALARGA_MIN:
            continue
        cands.append(j)
    if not cands:
        return None, None, None
    if len(cands) == 1:
        return etq, cands[0], L
    # Segundo rasgo, que se suma al anterior: la cinta es una tira de ANCHO
    # CONSTANTE (esta cortada a maquina), mientras que la franja de piso se
    # ensancha y se estrecha segun donde la tape la mesa. Se mide como el
    # coeficiente de variacion del ancho a lo largo del eje. Puntuacion:
    #     periodicidad / (1 + cv_ancho)
    mejor, mejor_p = cands[0], -1.0
    for j in cands:
        pos, _, anc, largo = marcas_en_cinta(etq, j, L)
        f = fuerza_peine(pos)
        pico = 0.0 if f is None else float(np.max(np.where(PS <= max(P_MIN, largo / 3.0),
                                                           f, 0.0)))
        filas_anchas = (etq == j).sum(1) if (etq == j).sum(1).size else np.array([1])
        filas_anchas = filas_anchas[filas_anchas > 0]
        cv = float(filas_anchas.std() / max(1.0, filas_anchas.mean()))
        punt = pico / (1.0 + cv)
        if punt > mejor_p:
            mejor, mejor_p = j, punt
    return etq, mejor, L


def marcas_en_cinta(etq, j, L):
    """Posiciones (en px, a lo largo del eje mayor) de las marcas oscuras."""
    ys, xs = np.nonzero(etq == j)
    vertical = (ys.max() - ys.min()) >= (xs.max() - xs.min())
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sub = L[y0:y1 + 1, x0:x1 + 1]
    msk = etq[y0:y1 + 1, x0:x1 + 1] == j
    if not vertical:                     # trabajar siempre a lo largo del eje 0
        sub, msk = sub.T, msk.T
    base = float(np.median(sub[msk]))
    oscuro = (sub < base - CAIDA_L) & msk
    ancho = np.maximum(1, msk.sum(1))
    frac = oscuro.sum(1) / ancho
    frac = cv2.GaussianBlur(frac.reshape(-1, 1).astype(np.float32), (1, 5), 0).ravel()
    pos, i = [], 0
    while i < len(frac):
        if frac[i] > FRAC_ANCHO:
            k = i
            while k < len(frac) and frac[k] > FRAC_ANCHO:
                k += 1
            if k - i <= max(4, len(frac) // 60):     # una marca es fina
                pos.append((i + k - 1) / 2.0)
            i = k
        else:
            i += 1
    anchos_fila = msk.sum(1)
    anchos_fila = anchos_fila[anchos_fila > 0]
    ancho_med = float(np.median(anchos_fila)) if len(anchos_fila) else 0.0
    return np.array(pos), float(base), ancho_med, len(frac)


PS = np.arange(P_MIN, P_MAX + 0.5, 0.5)


def fuerza_peine(pos):
    """Para cada periodo candidato, cuanto se alinean las marcas con un peine."""
    if len(pos) < 4:
        return None
    fase = 2 * np.pi * np.asarray(pos)[None, :] / PS[:, None]
    return np.abs(np.exp(1j * fase).mean(1))


def periodo(pos, largo):
    """Periodo dominante de UNA foto. Se usa solo como diagnostico."""
    f = fuerza_peine(pos)
    if f is None:
        return np.nan, 0.0
    ok = PS <= max(P_MIN, largo / 3.0)
    if not ok.any():
        return np.nan, 0.0
    f = np.where(ok, f, 0.0)
    i = int(np.argmax(f))
    return float(PS[i]), float(f[i] - np.median(f[ok]))


def periodo_de_sesion(lista_pos):
    """Periodo comun a toda una carpeta.

    La camara estuvo a una altura parecida durante la sesion, asi que el paso en
    pixeles es el MISMO para todas sus fotos. Estimarlo foto a foto desperdicia
    esa restriccion y deja que los digitos impresos empujen la estimacion a un
    armonico (la mitad o el doble del paso real). Sumando el periodograma de todas
    las fotos, el paso verdadero se refuerza en todas y los armonicos espurios,
    que dependen de que numeros salgan en cada foto, se promedian a la baja.

    Ademas se descarta el armonico obvio: si el pico esta en P y hay casi tanta
    fuerza en 2P, manda 2P (un peine de paso P tambien encaja en 2P, no al reves).
    """
    fs = [fuerza_peine(p) for p in lista_pos]
    fs = [f for f in fs if f is not None]
    if len(fs) < 3:
        return np.nan, 0.0, len(fs)
    F = np.mean(fs, axis=0)
    i = int(np.argmax(F))
    P, pico = float(PS[i]), float(F[i])
    j = int(np.argmin(np.abs(PS - 2 * P)))
    if j < len(PS) and F[j] > 0.85 * pico:
        P, pico = float(PS[j]), float(F[j])
    return P, float(pico - np.median(F)), len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", type=int, default=0)
    ap.add_argument("--solo", default="")
    a = ap.parse_args()

    man = list(csv.DictReader(open(OUT / "manifiesto_limpio.csv", encoding="utf-8")))
    if a.solo:
        man = [r for r in man if a.solo in r["archivo"] or a.solo in r["ruta"]]
    if a.muestra:
        porc = defaultdict(list)
        for r in man:
            porc[r["carpeta"]].append(r)
        man = []
        for c in sorted(porc):
            v = porc[c]
            idx = np.linspace(0, len(v) - 1, min(a.muestra, len(v))).astype(int)
            man += [v[i] for i in sorted(set(idx))]
    print("fotos: {}".format(len(man)))

    dir_masc = OUT / "mascaras"
    posiciones = defaultdict(list)
    filas = []
    for i, r in enumerate(man, 1):
        rgb = carga(RAIZ / r["ruta"])
        H, W = rgb.shape[:2]
        etq, j, L = busca_cinta(rgb)
        fila = dict(ruta=r["ruta"], archivo=r["archivo"], carpeta=r["carpeta"],
                    variedad=r["variedad"], clase=r["clase"], hoja_id=r["hoja_id"],
                    ancho_trabajo=W, alto_trabajo=H)
        if j is None:
            fila.update(estado="sin_cinta", n_marcas=0, px_grad="", nitidez="",
                        largo_hoja_px="", ancho_hoja_px="",
                        largo_grad="", ancho_grad="")
            filas.append(fila)
            continue
        pos, base, ancho_cinta, largo_cinta = marcas_en_cinta(etq, j, L)
        posiciones[r["carpeta"]].append(pos)
        P, nit = periodo(pos, largo_cinta)
        fila.update(estado="ok" if (P == P and nit > 0.15 and len(pos) >= 5)
                    else "dudosa", n_marcas=len(pos),
                    px_grad=round(P, 1) if P == P else "",
                    nitidez=round(nit, 3), ancho_cinta_px=round(ancho_cinta, 1))

        # largo/ancho de la hoja, si hay mascara (esta a otra escala: se reescala)
        f = dir_masc / (Path(r["archivo"]).stem + ".npy")
        if f.exists() and P == P:
            bits = np.unpackbits(np.load(f))
            # la mascara se guardo con lado largo 1024; se deduce la forma
            for hh in range(400, 1100):
                ww = int(round(hh * W / H))
                if hh * ww <= len(bits) < (hh + 1) * (ww + 2):
                    break
            try:
                hoja = bits[:hh * ww].reshape(hh, ww).astype(bool)
            except ValueError:
                hoja = None
            if hoja is not None and hoja.any():
                esc = H / float(hh)
                ys, xs = np.nonzero(hoja)
                pts = np.stack([xs, ys], 1).astype(np.float32)
                mu = pts.mean(0)
                val, vec = np.linalg.eigh(np.cov((pts - mu).T))
                pr = (pts - mu) @ vec[:, 1]
                pp = (pts - mu) @ vec[:, 0]
                lg = float(pr.max() - pr.min()) * esc
                an = float(pp.max() - pp.min()) * esc
                fila.update(largo_hoja_px=round(lg), ancho_hoja_px=round(an),
                            largo_grad=round(lg / P, 2), ancho_grad=round(an / P, 2))
        filas.append(fila)
        if i % 25 == 0:
            print("  {}/{}".format(i, len(man)))

    # ---- paso comun por sesion, y re-medida de la hoja con el (ver docstring)
    print("\n== paso por sesion (periodograma acumulado de toda la carpeta) ==")
    print("  {:46s} {:>6s} {:>9s} {:>8s}".format("carpeta", "fotos", "px/grad", "nitidez"))
    print("  " + "-" * 74)
    paso = {}
    for c in sorted(posiciones):
        P, nit, nf = periodo_de_sesion(posiciones[c])
        paso[c] = P
        print("  {:46s} {:6d} {:9.1f} {:8.3f}".format(c, nf, P, nit))
    for r in filas:
        P = paso.get(r["carpeta"], np.nan)
        r["px_grad_sesion"] = round(P, 1) if P == P else ""
        if P == P and r.get("largo_hoja_px") not in ("", None):
            r["largo_grad"] = round(float(r["largo_hoja_px"]) / P, 2)
            r["ancho_grad"] = round(float(r["ancho_hoja_px"]) / P, 2)

    campos = ["ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id", "estado",
              "px_grad_sesion",
              "n_marcas", "px_grad", "nitidez", "ancho_cinta_px", "largo_hoja_px",
              "ancho_hoja_px", "largo_anchos_cinta", "ancho_anchos_cinta",
              "largo_grad", "ancho_grad", "ancho_trabajo", "alto_trabajo"]
    f = OUT / ("escala_cinta_parcial.csv" if (a.solo or a.muestra) else "escala_cinta.csv")
    tmp = f.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(filas)
    tmp.replace(f)

    # ---------------- la validacion que decide si esto sirve ----------------
    print("\n== px por graduacion, por carpeta ==")
    print("  la camara estuvo a altura parecida en cada sesion: si el detector")
    print("  funciona, esta cifra sale CONSTANTE dentro de cada carpeta.\n")
    print("  {:46s} {:>4s} {:>4s} {:>8s} {:>8s} {:>7s}".format(
        "carpeta", "n", "ok", "mediana", "IQR", "IQR/med"))
    print("  " + "-" * 84)
    agg = defaultdict(list)
    for r in filas:
        if r["estado"] == "ok" and r["px_grad"] != "":
            agg[r["carpeta"]].append(float(r["px_grad"]))
    tot = defaultdict(int)
    for r in filas:
        tot[r["carpeta"]] += 1
    for c in sorted(tot):
        v = np.array(agg.get(c, []))
        if len(v) >= 3:
            q1, q3 = np.percentile(v, (25, 75))
            med = float(np.median(v))
            print("  {:46s} {:4d} {:4d} {:8.1f} {:8.1f} {:7.1%}".format(
                c, tot[c], len(v), med, q3 - q1, (q3 - q1) / max(1e-9, med)))
        else:
            print("  {:46s} {:4d} {:4d} {:>8s}".format(c, tot[c], len(v), "-"))

    print("\n== largo de la hoja por carpeta, en graduaciones del paso de sesion ==")
    print("  {:46s} {:>5s} {:>8s} {:>8s} {:>8s}".format(
        "carpeta", "n", "p10", "mediana", "p90"))
    print("  " + "-" * 80)
    porc = defaultdict(list)
    for r in filas:
        if r.get("largo_grad") not in ("", None):
            porc[r["carpeta"]].append(float(r["largo_grad"]))
    for c in sorted(porc):
        v = np.array(porc[c])
        print("  {:46s} {:5d} {:8.1f} {:8.1f} {:8.1f}".format(
            c, len(v), *np.percentile(v, (10, 50, 90))))

    # ---------------- ESCALA PRIMARIA: el ancho de la cinta ----------------
    print("\n== ESCALA PRIMARIA: ancho de la cinta (mismo objeto fisico en todas) ==")
    print("  {:46s} {:>5s} {:>10s} {:>8s} {:>12s} {:>12s}".format(
        "carpeta", "n", "ancho px", "IQR/med", "largo hoja", "ancho hoja"))
    print("  " + "-" * 100)
    anchos = defaultdict(list)
    for r in filas:
        if r.get("ancho_cinta_px"):
            anchos[r["carpeta"]].append(float(r["ancho_cinta_px"]))
    todos_largo = []
    for c in sorted(anchos):
        v = np.array(anchos[c])
        med = float(np.median(v))
        q1, q3 = np.percentile(v, (25, 75))
        lg_c, an_c = [], []
        for r in filas:
            if r["carpeta"] == c and r.get("largo_hoja_px") not in ("", None):
                lg_c.append(float(r["largo_hoja_px"]) / med)
                an_c.append(float(r["ancho_hoja_px"]) / med)
                r["largo_anchos_cinta"] = round(lg_c[-1], 2)
                r["ancho_anchos_cinta"] = round(an_c[-1], 2)
        todos_largo += lg_c
        print("  {:46s} {:5d} {:10.1f} {:7.1%} {:12s} {:12s}".format(
            c, len(v), med, (q3 - q1) / max(1e-9, med),
            "{:.1f}".format(np.median(lg_c)) if lg_c else "-",
            "{:.1f}".format(np.median(an_c)) if an_c else "-"))
    if todos_largo:
        t = np.array(todos_largo)
        print("\n  largo de hoja en ANCHOS DE CINTA:  p10={:.1f}  mediana={:.1f}  p90={:.1f}"
              .format(*np.percentile(t, (10, 50, 90))))
        print("  Esa cifra sale parecida en todas las carpetas y en las dos variedades:")
        print("  es la senal de que la escala es correcta. Falta UNA medida con regla:")
        print("  cuanto mide de ancho la cinta. Con eso, todo el dataset queda en mm.")
        if ANCHO_CINTA_MM:
            print("  ANCHO_CINTA_MM={} -> largo de hoja mediana = {:.0f} mm"
                  .format(ANCHO_CINTA_MM, np.median(t) * ANCHO_CINTA_MM))

    lg = [float(r["largo_grad"]) for r in filas if r.get("largo_grad") not in ("", None)]
    if lg:
        lg = np.array(lg)
        print("\n== largo de la hoja, en graduaciones ==")
        print("  p10={:.1f}  mediana={:.1f}  p90={:.1f}  (n={})".format(
            *np.percentile(lg, (10, 50, 90)), len(lg)))
        print("  el experto describio una hoja grande como 'de 18 pulgadas'. Si la")
        print("  mediana de arriba sale cerca de 18, una graduacion es una pulgada;")
        print("  si sale cerca de 36, son media pulgada. Confirmarlo con el.")
    print("\n-> {}".format(f))


if __name__ == "__main__":
    main()
