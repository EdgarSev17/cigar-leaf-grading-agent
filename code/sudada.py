r"""
DETECTOR DE HOJA SUDADA — las pizquitas verdes. (2026-09-13)

QUE ES, EN PALABRAS DEL EXPERTO
-----------------------------
    «son esas pisquitas como verde que estan de manera repetida en la hoja.
     Todas esas pisquitas verdes es a lo que se le llama hoja sudada. Eso al
     final es un hongo producido por excesiva humedad.»

Y de ahi salen DOS de las cuatro fronteras del problema, que es lo que hace que
este detector importe mas que ningun otro:

                      pizquitas verdes     agujeros
        CAPA                no                no
        BANDA               SI                no
        1/2 BANDA           SI                SI
        XL / XR             no                SI, de un lado

    «la diferencia entre capa y banda es que la capa no presenta estas pisquitas
     verdes. Y la diferencia entre 1/2 banda y la XL y XR es que la 1/2 banda
     presenta el defecto de hoja sudada y tiene agujero o agujeros, mientras que
     el XL y XR presenta agujeros en una parte de la hoja pero no presenta
     manchitas verdes.»

O sea: **ni capa/banda ni 1/2banda/XL se separan por el agujero. Las dos se
separan por el sudado.** El sistema medía el agujero bien y el sudado no, y ahi
esta el 89 % de §58.4 --«¿esta manchada?», la pregunta que peor va y la que el experto
contesta de un vistazo.

POR QUE LOS DOS INTENTOS ANTERIORES FALLARON, Y EN QUE SE DIFERENCIA ESTE
--------------------------------------------------------------------------
1. **`frac_verde` de §37.2** mide `a*` por debajo de **la mediana de la propia
   hoja**. Eso vale para una mancha dentro de una hoja sana y es lo que hace que
   el eje USO funcione (§37.3: 3.5x mas verde dentro del circulo del experto). Pero
   es **ciego a una hoja sudada ENTERA**: si todo el limbo se pone verde, la
   mediana sube con el y no sobresale nada (§74.4).
2. **El verde absoluto** --la mediana de `a*` de la hoja-- se probo en §74.4 y
   **no separa**: Connecticut da 4.0 en banda, capa, 1/2 banda y xl_izq.
3. **El jaspeado** --la dispersion de `a*`-- tambien se probo y tampoco (§74.4).

Lo que hace este, y es lo que ninguno hacia: **buscar las pizquitas como OBJETOS
pequeños**, no como un desplazamiento del color medio. Una pizquita es una mancha
de pocos milimetros **mas verde que la lamina a su alrededor a esa escala**, y eso
se mide con un sombrero de copa (top-hat) sobre `a*`: la diferencia entre el `a*`
local y el `a*` de un entorno mayor. Un sudado general **no borra la señal**,
porque cada pizquita sigue siendo mas verde que los dos milimetros que la rodean.

Es el mismo mecanismo que desatasco la mancha blanca en §53 --separar por FORMA y
ESCALA en vez de por color absoluto-- aplicado al otro defecto.

LA CALIBRACION, Y DE DONDE SALE CADA LADO
------------------------------------------
    POSITIVOS  las 10 hojas de `dataset/Hojas sudadas/`, que el experto marco con
               rectangulos sobre una COPIA alineada pixel a pixel. Se busca el
               rectangulo en la copia y se mide en la LIMPIA (§53.2).
    NEGATIVOS  las hojas de **CAPA**, que por definicion no tienen pizquitas
               --lo dice la propia regla de arriba--. Ya estaban en el dataset:
               no hizo falta pedir nada (Regla A).

Medido sobre las marcas, dentro contra el resto de la misma hoja:

        a*      -2.11   mas verde     en 8 de 9 hojas
        desv b* -1.32   mas parejo    en 8 de 9
        b*      -1.22                 en 6 de 9
        L       +2.2    cambia de signo: no dice nada

**El eje es `a*`.** Y un hallazgo que corrige una intuicion vieja: lo marcado es
mas **PAREJO**, no mas jaspeado -- donde hay hongo el tono se uniformiza. Por eso
medir «dispersion» no funcionaba.

RESOLUCION
----------
Se trabaja a **2048 px de lado**, no a los 1024 de la tuberia. A 1024 una hoja de
16 pulgadas da ~0.5 mm por pixel y una pizquita de 2-3 mm son 4-6 pixeles, que es
el tamaño del grano de la propia textura. Es la leccion de §42.2 con la mancha
blanca, aplicada antes de tropezar.

Uso:
    python scripts/sudada.py --calibra      mide las 10 marcadas contra las capa
    python scripts/sudada.py --clases       el detector sobre las cinco clases
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentacion import mascara_hoja, mascara_marca                  # noqa: E402
from rutas import REPO_RAIZ  # raiz del repositorio

RAIZ = REPO_RAIZ / "dataset"
OUT = REPO_RAIZ / "out"
SUD = RAIZ / "Hojas sudadas"
LADO = 2048

# escalas en pixeles a LADO=2048 (~0.25 mm/px en una hoja de 16 pulgadas)
R_PIZCA = 6          # radio de la pizquita: ~3 mm
R_ENTORNO = 26       # radio del entorno contra el que se compara: ~13 mm
MIN_PX = 12          # por debajo es ruido de textura
UMBRAL_A = 2.0       # cuanto mas verde que su entorno, en unidades de a*


def carga(p, lado=LADO):
    with Image.open(p) as im:
        try:
            im.draft("RGB", (im.width // 2, im.height // 2))
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((lado, lado), Image.LANCZOS)
        return np.asarray(im)


def disco(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


MM_BORDE = 20        # cuanto se encoge la mascara, en px de la malla


def mapa_pizcas(rgb, hoja, umbral=UMBRAL_A):
    """Mascara de pizquitas verdes y el mapa continuo del que sale.

    Una pizquita es un trozo de lamina **mas verde que su entorno inmediato**.
    Se mide como `a*` suavizado a la escala de la pizca menos `a*` suavizado a la
    escala del entorno: donde eso es negativo, hay mas verde del que le toca a esa
    zona de la hoja. Como las dos escalas salen de la MISMA hoja, un sudado
    general no la borra -- que es justo lo que le pasaba a `frac_verde`.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    # fuera de la hoja no se mide: se rellena con la mediana para no arrastrar
    # la mesa al suavizado
    med = float(np.median(a[hoja])) if hoja.any() else 0.0
    aa = np.where(hoja, a, med)
    fino = cv2.GaussianBlur(aa, (0, 0), R_PIZCA / 2.0)
    grueso = cv2.GaussianBlur(aa, (0, 0), R_ENTORNO / 2.0)
    verdor = grueso - fino              # positivo = mas verde que su entorno
    m = (verdor > umbral) & hoja
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disco(1)) > 0
    return m, verdor


def componentes(m, min_px=MIN_PX):
    n, etq, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return [st[i] for i in range(1, n) if st[i, 4] >= min_px]


def mide(rgb, hoja, umbral=UMBRAL_A):
    """Los rasgos de sudado de una hoja."""
    m, verdor = mapa_pizcas(rgb, hoja, umbral)
    comps = componentes(m)
    area = float(hoja.sum()) or 1.0
    # densidad de pizquitas por millon de pixeles de hoja, que es invariante
    # al tamaño de la foto
    return dict(
        sud_frac=float(m.sum()) / area,
        sud_n=len(comps),
        sud_dens=1e6 * len(comps) / area,
        sud_verdor_p95=float(np.percentile(verdor[hoja], 95)) if hoja.any() else 0.0,
        sud_verdor_med=float(np.median(verdor[hoja])) if hoja.any() else 0.0,
    )


def mide_ero(rgb, hoja, mm=MM_BORDE, umbral=UMBRAL_A):
    """Los rasgos de sudado CON EL BORDE DE LA HOJA FUERA. (2026-09-15)

    El detector se encendia en el borde, donde la lamina se dobla y se ve mas
    verde. Encogiendo la mascara `mm` milimetros el AUC de banda contra capa
    sube de 0.724 a 0.845 y el de 1/2 banda contra XL/XR de 0.695 a 0.747
    (`_drift_erosion.py`, 14/09).

    El mapa de pizquitas se calcula sobre la hoja ENTERA --la referencia de
    color tiene que salir de toda la lamina-- y solo se CUENTA dentro de la
    mascara encogida. Esta funcion es la unica que mide sudado para el modelo:
    la usan `entrena.py` (via el CSV) y `rasgos_foto.py` (al clasificar), para
    que la cifra signifique lo mismo en los dos lados.
    """
    m, verdor = mapa_pizcas(rgb, hoja, umbral)
    k = np.ones((2 * mm + 1, 2 * mm + 1), np.uint8)
    h2 = cv2.erode(hoja.astype(np.uint8), k).astype(bool)
    if h2.sum() < 500:
        return mide(rgb, hoja, umbral)      # hoja diminuta: mejor la de siempre
    m2 = m & h2
    comps = componentes(m2)
    area = float(h2.sum()) or 1.0
    return dict(sud_frac=float(m2.sum()) / area,
                sud_n=len(comps),
                sud_dens=1e6 * len(comps) / area,
                sud_verdor_p95=float(np.percentile(verdor[h2], 95)),
                sud_verdor_med=float(np.median(verdor[h2])))


# ------------------------------------------------------------ el sudado POR LADO
def _lados(hoja):
    """(izq, der): la hoja partida por la VENA CENTRAL, en la mascara de 2048.

    Lo pide la regla del experto, escrita por el en dos sitios y que el detector de
    hoja entera no puede expresar (Tabaco 22.2 y 87.4):

        sudada en UN lado    -> XL / XR del lado limpio
        sudada en AMBOS      -> Banda (sin agujero) o 1/2 Banda (con agujero)

    El divisor es `zonas.eje_vena` -- el centro de la lamina tramo a tramo -- y no
    la recta por el centroide, que es lo que el mismo pidio el 13/09 («DEBES USAR
    LA VENA CENTRAL PARA DIVIDIR LA HOJA»). Aqui no cuesta nada: es un detector
    nuevo, no toca la tuberia.
    """
    from zonas import perfil, extremo_base, eje_vena
    e, perp, mu, t, s, anch = perfil(hoja)
    if not extremo_base(anch)[0]:
        e, perp, t, s = -e, -perp, -t, -s
    tt, cc = eje_vena(t, s)
    ys, xs = np.nonzero(hoja)
    p = np.stack([xs, ys], 1).astype(np.float32) - mu
    d = (p @ perp) - np.interp(p @ e, tt, cc)
    izq = np.zeros(hoja.shape, bool)
    der = np.zeros(hoja.shape, bool)
    izq[ys[d > 0], xs[d > 0]] = True
    der[ys[d <= 0], xs[d <= 0]] = True
    return izq, der


def mide_lados(rgb, hoja, umbral=UMBRAL_A):
    """El sudado de cada mitad, y las medidas RELATIVAS entre las dos.

    POR QUE RELATIVAS, y es la mitad del hallazgo (Tabaco 87.3): el nivel
    absoluto de `sud_frac` NO cruza de sesion -- mediana 0.0214 al entrenar,
    0.0160 en la tanda del 24/08 por la tarde y 0.0104 en la del 11/09-- porque
    el umbral esta en unidades fijas de a* y la escena mueve toda la
    distribucion. Un COCIENTE entre las dos mitades se mide en la misma foto,
    con la misma luz y la misma camara, asi que la deriva se cancela sola.

    `sud_razon` es la que lleva la regla:  1 = igual de sudada por los dos lados
    (banda / 1/2 banda);  cerca de 0 = sudada de un solo lado (XL / XR).

    El mapa de pizquitas se calcula UNA vez sobre la hoja entera y despues se
    cuenta dentro de cada mitad: partir la foto antes de suavizar meteria un
    borde falso justo en la vena, que es donde mas pizquitas hay.
    """
    m, verdor = mapa_pizcas(rgb, hoja, umbral)
    izq, der = _lados(hoja)
    out = {}
    for nom, mitad in (("izq", izq), ("der", der)):
        area = float(mitad.sum()) or 1.0
        mm = m & mitad
        out["sud_frac_" + nom] = float(mm.sum()) / area
        out["sud_dens_" + nom] = 1e6 * len(componentes(mm)) / area
    i, d = out["sud_frac_izq"], out["sud_frac_der"]
    hi, lo = max(i, d), min(i, d)
    out["sud_asim"] = (i - d) / (i + d + 1e-9)      # signo: + = mas sudada la izq
    out["sud_razon"] = lo / (hi + 1e-9)             # 1 = ambos lados, 0 = uno solo
    out["sud_dif_abs"] = abs(i - d)
    ii, dd = out["sud_dens_izq"], out["sud_dens_der"]
    out["sud_dens_razon"] = min(ii, dd) / (max(ii, dd) + 1e-9)
    return out


def mide_robusto(rgb, hoja, umbral=UMBRAL_A):
    """El mismo sudado, pero con el umbral en MADs de la PROPIA hoja.

    La otra mitad del arreglo de 87.3. `mide()` corta el verdor en 2.0 unidades
    fijas de a*; si la escena cambia la saturacion, toda la distribucion se
    escala y la fraccion que pasa el corte cambia sin que la hoja haya cambiado.
    Aqui el corte se pone en k desviaciones robustas de la propia hoja, que
    absorbe ese escalado.

    No sustituye a `mide()` mientras no se compruebe que separa igual: se mide
    aparte y se compara la deriva entre jornadas.
    """
    _, verdor = mapa_pizcas(rgb, hoja, umbral)
    v = verdor[hoja]
    if v.size == 0:
        return {"sudr_frac_3": 0.0, "sudr_frac_4": 0.0, "sudr_frac_5": 0.0}
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    z = (verdor - med) / (mad + 1e-6)
    area = float(hoja.sum()) or 1.0
    return {"sudr_frac_%i" % k: float(((z > k) & hoja).sum()) / area
            for k in (3, 4, 5)}


# ------------------------------------------------------------------ calibracion
def marcas_de(n):
    """(rgb limpio, mascara de hoja, interior de los rectangulos del experto)."""
    lim = SUD / ("Hoja_Sudada%d.heic" % n)
    cop = None
    for suf in (" - copiaaaa", " - copia"):
        p = SUD / ("Hoja_Sudada%d%s.heic" % (n, suf))
        if p.exists():
            with Image.open(lim) as a, Image.open(p) as b:
                if a.size != b.size:
                    continue
            A = carga(lim)
            B = carga(p)
            if np.abs(A.astype(int) - B.astype(int)).sum(2).max() > 60:
                cop = p
                break
    if cop is None:
        return None
    a, b = carga(lim), carga(cop)
    dif = np.abs(a.astype(int) - b.astype(int)).sum(2) > 40
    nn, etq, st, _ = cv2.connectedComponentsWithStats(dif.astype(np.uint8))
    dentro = np.zeros(dif.shape, bool)
    for i in range(1, nn):
        x, y, w, h, ar = st[i]
        if ar < 20 or w < 12 or h < 12:
            continue
        dentro[y + 4:y + h - 4, x + 4:x + w - 4] = True
    hoja = mascara_hoja(a, mascara_marca(a))[0]
    return a, hoja & ~cv2.dilate(dif.astype(np.uint8), disco(4)).astype(bool), dentro & hoja


def calibra():
    print("=" * 74)
    print("CALIBRACION DEL DETECTOR DE SUDADA")
    print("=" * 74)
    print("   positivos: dentro de los rectangulos del experto, 10 hojas")
    print("   negativos: hojas de CAPA, que por definicion no tienen pizquitas")
    print()
    dentro_v, fuera_v = [], []
    for n in range(1, 11):
        r = marcas_de(n)
        if r is None:
            print("   %2d  sin marcas" % n)
            continue
        rgb, hoja_limpia, dentro = r
        _, verdor = mapa_pizcas(rgb, hoja_limpia | dentro)
        if dentro.sum() < 300:
            print("   %2d  marcas muy pequenas" % n)
            continue
        dv = verdor[dentro]
        fv = verdor[hoja_limpia & ~dentro]
        dentro_v.append(dv)
        fuera_v.append(fv)
        print("   %2d  verdor p90 dentro %+5.2f   fuera %+5.2f   (marcado %.1f %%)"
              % (n, np.percentile(dv, 90), np.percentile(fv, 90),
                 100 * dentro.sum() / max(1, (hoja_limpia | dentro).sum())))
    if not dentro_v:
        return
    D = np.concatenate(dentro_v)
    F = np.concatenate(fuera_v)
    print()
    print("   TODO JUNTO   dentro de las marcas: p90 %+5.2f   resto de hoja: p90 %+5.2f"
          % (np.percentile(D, 90), np.percentile(F, 90)))
    print()
    print("   %-10s %12s %12s %10s" % ("umbral a*", "% dentro", "% fuera", "razon"))
    for u in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        pd, pf = float((D > u).mean()), float((F > u).mean())
        print("   %-10.1f %11.2f %% %11.2f %% %10s"
              % (u, 100 * pd, 100 * pf, "%.1fx" % (pd / pf) if pf > 0 else "-"))
    print()
    print("   (el `resto de hoja` NO es limpio: la hoja entera esta sudada y el experto")
    print("    marco ejemplos, no todo. El control limpio son las capa: --clases)")


def por_clases(n_por_clase=25):
    """El detector sobre las cinco clases, que es donde se ve si separa."""
    man = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "manifiesto_limpio.csv", encoding="utf-8"))}
    seg = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "segmentacion.csv", encoding="utf-8"))}
    porc = defaultdict(list)
    for r, d in man.items():
        if (seg.get(r, {}).get("toca_borde") == "0"
                and d["variedad"] == "connecticut"):
            porc[d["clase"]].append(r)
    rng = np.random.default_rng(0)
    filas = []
    print("=" * 74)
    print("EL DETECTOR SOBRE LAS CINCO CLASES  (Connecticut, %d por clase)"
          % n_por_clase)
    print("=" * 74)
    for cl in ("capa", "banda", "media_banda", "xl_izq", "xr_der"):
        v = porc.get(cl, [])
        if not v:
            continue
        sel = rng.choice(len(v), size=min(n_por_clase, len(v)), replace=False)
        for i in sel:
            r = v[i]
            try:
                rgb = carga(RAIZ / r)
                hoja = mascara_hoja(rgb, mascara_marca(rgb))[0]
                f = mide(rgb, hoja)
                f.update(clase=cl, ruta=r)
                filas.append(f)
            except Exception:
                pass
        print("   %-13s medidas %i" % (cl, sum(1 for f in filas if f["clase"] == cl)))
    if not filas:
        return
    print()
    print("   %-13s %10s %12s %14s" % ("clase", "n", "sud_frac", "densidad"))
    for cl in ("capa", "banda", "media_banda", "xl_izq", "xr_der"):
        v = [f for f in filas if f["clase"] == cl]
        if not v:
            continue
        print("   %-13s %10i %11.4f %14.1f"
              % (cl, len(v), np.median([f["sud_frac"] for f in v]),
                 np.median([f["sud_dens"] for f in v])))
    # lo que decide: ¿separa capa (limpia) de banda + 1/2 banda (sudadas)?
    lim = [f["sud_frac"] for f in filas if f["clase"] == "capa"]
    sud = [f["sud_frac"] for f in filas if f["clase"] in ("banda", "media_banda")]
    xl = [f["sud_frac"] for f in filas if f["clase"] in ("xl_izq", "xr_der")]
    if lim and sud:
        auc = np.mean([[1.0 if s > l else (0.5 if s == l else 0.0) for l in lim]
                       for s in sud])
        print()
        print("   AUC  capa (limpia) contra banda+1/2banda (sudadas):  **%.2f**" % auc)
        print("   (0.50 = no sabe nada;  §58.4 midio que esta pregunta va al 89 %)")
    if lim and xl:
        auc2 = np.mean([[1.0 if s > l else (0.5 if s == l else 0.0) for l in lim]
                        for s in xl])
        print("   AUC  capa contra XL/XR (que NO deben estar sudadas):  %.2f" % auc2)
        print("   (esta deberia salir cerca de 0.50: si sale alta, el detector")
        print("    esta midiendo otra cosa y no el sudado)")
    with open(OUT / "sudada_clases.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(filas)
    print()
    print("   -> out/sudada_clases.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibra", action="store_true")
    ap.add_argument("--clases", action="store_true")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    if a.calibra or not (a.calibra or a.clases):
        calibra()
    if a.clases:
        por_clases(a.n)


# ------------------------------------------------------------------ el lote
def lote(destino, rutas_y_raiz, max_nuevas=0):
    """Mide el sudado de una lista de fotos. REANUDABLE: escribe cada fila.

    En esta maquina los procesos largos se mueren solos --paso con
    `clasificador.py`, `segmentacion.py` y `entrena.py` cuatro veces-- y la regla
    escrita entonces es que los guiones largos guarden lo que van calculando.
    """
    import time
    hechas = set()
    if destino.exists():
        hechas = {r["ruta"] for r in csv.DictReader(open(destino, encoding="utf-8"))}
    pend = [(r, base) for r, base in rutas_y_raiz if r not in hechas]
    if max_nuevas:
        pend = pend[:max_nuevas]
    print("   %s: por medir %i (ya hechas %i)" % (destino.name, len(pend), len(hechas)))
    nuevo = not destino.exists()
    cab = ["ruta", "sud_frac", "sud_n", "sud_dens", "sud_verdor_p95", "sud_verdor_med"]
    fh = open(destino, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=cab, extrasaction="ignore")
    if nuevo:
        w.writeheader()
        fh.flush()
    t0 = time.time()
    for k, (r, base) in enumerate(pend, 1):
        try:
            p = Path(r) if Path(r).is_absolute() else Path(base) / r
            rgb = carga(p)
            hoja = mascara_hoja(rgb, mascara_marca(rgb))[0]
            f = mide(rgb, hoja)
            f["ruta"] = r
            w.writerow(f)
            fh.flush()
        except Exception as ex:
            print("      fallo %s: %s" % (Path(r).name, str(ex)[:50]), flush=True)
        if k % 25 == 0:
            print("      %i/%i  (%.1f s/foto)" % (k, len(pend),
                                                  (time.time() - t0) / k), flush=True)
    fh.close()
