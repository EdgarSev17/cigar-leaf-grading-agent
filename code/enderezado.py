r"""
Endereza cada foto usando la cinta metrica como referencia fisica.

POR QUE HACE FALTA
------------------
Las 1179 fotos traen orientacion EXIF = 1: la rotacion esta quemada en los pixeles
y no hay etiqueta que la deshaga. Y la proporcion de fotos acostadas cambia
muchisimo entre clases (Habano/XL Derecho 1 de 183; Connecticut/XL Izq 0 de 88;
Habano/Banda 58 de 90). Un modelo puede sacar accuracy alto leyendo SI LA FOTO
ESTA ACOSTADA, que no tiene nada que ver con la hoja.

Peor: las clases `XL Izquierdo` y `XR Derecho` se definen por el LADO donde esta
el dano. "Izquierda" y "derecha" no existen hasta que la foto tenga una
orientacion canonica. Sin enderezar, esas dos clases no son separables ni en
principio.

LA REFERENCIA LA DIO EL EXPERTO, Y ES FISICA
---------------------------------------
Todas las fotos son del haz, y **el lado izquierdo de la hoja siempre va junto a
la cinta metrica**. Cualquier otra orientacion es un error de captura. Entonces:
localizar la cinta define la orientacion sin ambiguedad, y de paso da la escala
px -> mm, que la regla de decision necesita (PROGRESO 11.1).

COMO SE LOCALIZA LA CINTA
-------------------------
La cinta es la unica cosa del cuadro que es a la vez CLARA, NO CALIDA y MUY LARGA:

    cinta Connecticut  L=173..238  a*=-4..+2   b*= -9..-24  <- blanco AZULADO:
                                                              refleja la mesa
    cinta Habano       L=207..239  a*= 0..+3   b*=  -2..+6  <- neutra
    hoja               b* >= +17                            <- se cae por calida
    mesa azul          L < 130                              <- se cae por oscura
    piso de concreto   b* = +4..+27                         <- se cae por calido
    piel               a* = +11..+12                        <- se cae por roja
    reflejo de la mesa L < 170                              <- se cae por oscuro

OJO: "neutra" era la palabra equivocada y costo caro. Con |b*|<12 la cinta de
Connecticut, que es azulada, quedaba medio descartada y troceada, y el guion la
daba por ausente en 440 de 555 fotos. Con "no calida" (b* < +8) aparece en el
80-90 %.

El alargamiento es lo que la separa de un reflejo o de un papel suelto: se mide
como (eje mayor / eje menor) de la elipse ajustada a la componente.

QUE DEVUELVE, Y QUE NO
----------------------
No reescribe el dataset. Escribe `out/enderezado.csv` con el angulo y el volteo
que hay que aplicar, mas la escala si se pudo leer. La rotacion se aplica al
vuelo dentro del cargador, por dos razones: (1) el sistema desplegado va a recibir
fotos crudas y tiene que enderezarlas el mismo, y (2) no se duplican 3.8 GB.

`estado` dice que paso:
    ok            cinta encontrada, angulo fiable
    sin_cinta     no hay componente que cumpla; queda a mano o se descarta
    ambigua       hay mas de una candidata larga y no dominan una sobre otra

Uso:
    python scripts/enderezado.py --muestra 6 --qc 60
    python scripts/enderezado.py --qc 80
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
LADO = 1024

# --- firma de la cinta. Corregida el 2026-08-30 contra medidas (ver escala_cinta.py):
#     la cinta de Connecticut es blanco AZULADO (b* = -9..-24) porque refleja la
#     mesa azul marino; la de Habano es neutra (b* ~ 0). Pedir "neutra" (|b*|<12)
#     descartaba media cinta de Connecticut y la troceaba. La regla buena es
#     CLARA y NO CALIDA, que ademas deja fuera el piso (b* = +4..+27) y la hoja
#     (b* >= +17). L>170 excluye el reflejo especular de la mesa.
L_MIN_CINTA = 170
A_MIN, A_MAX_C = -8, 6
B_MAX_C = 8
ALARGA_MIN = 6.0     # eje mayor / eje menor
AREA_MIN = 0.004


def carga(p, lado=LADO):
    with Image.open(p) as im:
        try:
            im.draft("RGB", (im.width // 4, im.height // 4))
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((lado, lado), Image.LANCZOS)
        return np.asarray(im)


def busca_cinta(rgb):
    """(mascara, angulo_grados, alargamiento, estado). Angulo del eje mayor."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    m = (L > L_MIN_CINTA) & (a > A_MIN) & (a < A_MAX_C) & (b < B_MAX_C)
    h, w = m.shape
    # cierre grande: las marcas de graduacion cruzan la cinta y la trocean
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    k = max(3, (min(h, w) // 250) | 1)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    n, etq, est, cen = cv2.connectedComponentsWithStats(m8, 8)
    cands = []
    for j in range(1, n):
        area = est[j, cv2.CC_STAT_AREA]
        if area < AREA_MIN * h * w:
            continue
        ys, xs = np.nonzero(etq == j)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        mu = pts.mean(0)
        cov = np.cov((pts - mu).T)
        val, vec = np.linalg.eigh(cov)
        if val[0] <= 1e-6:
            continue
        alarg = float(np.sqrt(val[1] / val[0]))
        if alarg < ALARGA_MIN:
            continue
        ev = vec[:, 1]                       # eje mayor
        ang = float(np.degrees(np.arctan2(ev[1], ev[0])))
        cands.append((area, alarg, ang, j, float(mu[0]), float(mu[1])))
    if not cands:
        return None, np.nan, np.nan, "sin_cinta", np.nan, np.nan
    cands.sort(reverse=True)
    area, alarg, ang, j, cx, cy = cands[0]
    estado = "ok"
    if len(cands) > 1 and cands[1][0] > 0.6 * area:
        # dos tiras largas y parecidas: puede ser la cinta partida por una sombra,
        # o la cinta mas el borde claro de la mesa. Si son casi paralelas se acepta.
        if abs(((cands[1][2] - ang + 90) % 180) - 90) > 12:
            estado = "ambigua"
    return (etq == j), ang, alarg, estado, cx, cy


def canon(ang):
    """Angulo del eje mayor en [-90, 90) y la rotacion MINIMA que lo pone vertical.

    BUG CORREGIDO EL 2026-08-30, y lo detecto el experto mirando `out/qc_enderezado`:
    las miniaturas salian giradas cuando el no habia dejado ninguna foto torcida.

    Un eje no tiene sentido, asi que "vertical" son DOS angulos: +90 y -90. La
    version anterior llevaba siempre a +90 con `rot = 90 - a`. Para una cinta que
    ya estaba vertical con a = -88, eso daba **178 grados** de rotacion en vez de
    -2. Medido sobre las 979 fotos con cinta: mediana de rotacion aplicada 178.0,
    y el 87 % giraba mas de 90 grados.

    El destrozo lo tapaba a medias el paso siguiente: como tras girar 178 grados la
    cinta quedaba a la derecha, el volteo automatico de 180 lo deshacia, y el
    resultado neto era el correcto (-2). Pero eso ocurria en 847 de 979 fotos;
    **en las otras 132 la hoja quedaba boca abajo y con izquierda y derecha
    intercambiadas**, que es justo el tipo de error que puede haber inflado la
    lista de `lado_al_reves` de PROGRESO 25.

    Ahora se elige el giro mas corto de los dos posibles.
    """
    a = (ang + 90) % 180 - 90        # el eje no distingue sentido
    rot = (90 - a) if a >= 0 else (-90 - a)
    return a, rot


def rota(img, grados):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), grados, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0)), M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", type=int, default=0)
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

    qc_dir = OUT / "qc_enderezado"
    if a.qc:
        qc_dir.mkdir(parents=True, exist_ok=True)
        sel = {man[i]["ruta"] for i in
               np.linspace(0, len(man) - 1, min(a.qc, len(man))).astype(int)}
    else:
        sel = set()

    # mascaras de hoja, si ya existen, para decidir de que lado queda la cinta
    dir_masc = OUT / "mascaras"

    filas = []
    for i, r in enumerate(man, 1):
        rgb = carga(RAIZ / r["ruta"])
        h, w = rgb.shape[:2]
        mc, ang, alarg, estado, cx, cy = busca_cinta(rgb)
        fila = dict(ruta=r["ruta"], archivo=r["archivo"], carpeta=r["carpeta"],
                    clase=r["clase"], variedad=r["variedad"], hoja_id=r["hoja_id"],
                    estado=estado, alargamiento=round(alarg, 2) if alarg == alarg else "",
                    ancho=w, alto=h)
        if mc is None:
            fila.update(angulo_cinta="", rotacion="", voltea_180="", cinta_a_la="")
            filas.append(fila)
            continue

        aj, rot = canon(ang)
        # donde queda la cinta respecto de la hoja, ya con la rotacion aplicada
        f = dir_masc / (Path(r["archivo"]).stem + ".npy")
        lado = ""
        voltea = ""
        if f.exists():
            hoja = np.unpackbits(np.load(f))[:h * w].reshape(h, w).astype(bool)
            mrot, M = rota(mc.astype(np.uint8) * 255, rot)
            hrot, _ = rota(hoja.astype(np.uint8) * 255, rot)
            if mrot.any() and hrot.any():
                xc = np.nonzero(mrot)[1].mean()
                xh = np.nonzero(hrot)[1].mean()
                lado = "izq" if xc < xh else "der"
                # la cinta debe quedar a la IZQUIERDA de la hoja (dato del experto)
                voltea = int(lado == "der")
        fila.update(angulo_cinta=round(aj, 2), rotacion=round(rot, 2),
                    voltea_180=voltea, cinta_a_la=lado)
        filas.append(fila)

        if r["ruta"] in sel:
            vis = rgb.copy()
            vis[mc] = (0, 255, 255)
            vis, _ = rota(vis, rot)
            if voltea == 1:
                vis = cv2.rotate(vis, cv2.ROTATE_180)
            vis = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
            cv2.imwrite(str(qc_dir / (Path(r["archivo"]).stem.replace(" ", "_") + ".jpg")),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
        if i % 50 == 0:
            print("  {}/{}".format(i, len(man)))

    campos = ["ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id", "estado",
              "angulo_cinta", "rotacion", "voltea_180", "cinta_a_la", "alargamiento",
              "ancho", "alto"]
    f = OUT / ("enderezado_parcial.csv" if (a.solo or a.muestra) else "enderezado.csv")
    tmp = f.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(filas)
    tmp.replace(f)

    from collections import Counter
    print("\nestado por carpeta:")
    print("  {:46s} {:>5s} {:>5s} {:>10s} {:>9s}".format(
        "carpeta", "n", "ok", "sin_cinta", "ambigua"))
    agg = defaultdict(Counter)
    for r in filas:
        agg[r["carpeta"]][r["estado"]] += 1
    for c in sorted(agg):
        q = agg[c]
        print("  {:46s} {:5d} {:5d} {:10d} {:9d}".format(
            c, sum(q.values()), q["ok"], q["sin_cinta"], q["ambigua"]))
    ang = np.array([r["angulo_cinta"] for r in filas
                    if isinstance(r["angulo_cinta"], float)])
    if len(ang):
        print("\nangulo de la cinta (grados, 0 = horizontal, +-90 = vertical):")
        print("  p5={:.1f} p25={:.1f} mediana={:.1f} p75={:.1f} p95={:.1f}".format(
            *[np.percentile(ang, p) for p in (5, 25, 50, 75, 95)]))
        print("  |angulo| > 45 (cinta ya casi vertical): {}/{}".format(
            int((np.abs(ang) > 45).sum()), len(ang)))
    lados = Counter(r["cinta_a_la"] for r in filas if r.get("cinta_a_la"))
    if lados:
        print("\nlado en que queda la cinta tras rotar: {}".format(dict(lados)))
        print("  (debe quedar a la IZQUIERDA; las 'der' se voltean 180)")
    print("\n-> {}".format(f))
    if a.qc:
        print("-> {}".format(qc_dir))


if __name__ == "__main__":
    main()
