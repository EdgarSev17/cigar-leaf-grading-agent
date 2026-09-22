r"""
EL PUNTO DE AGARRE SALE DE LA REGLA, NO DEL CENTRO DE MASA (2026-09-08)

Punto E.4 / 67.7.2 de «LO SIGUIENTE». Hoy `mallas_3d.csv` trae como agarre el
**centro de masa de la silueta**, y esta escrito ahi mismo que es provisional. La
regla de los tecnicos dice cual es el sitio correcto y no hay que inventarlo
(§16, §19, §65.4):

    se agarra por LA ZONA QUE LA REGLA YA DESCARTA -- la base y la franja de la
    vena --, que ademas es la parte RIGIDA de la hoja. Si la ventosa marca la
    hoja, marca la parte que se corta igual.

Las dos mitades de esa frase son comprobables con lo que ya esta medido, y este
guion las comprueba en vez de darlas por buenas:

  1. **Que las copas se apoyen en hoja**, no en un agujero ni en el aire. La
     malla trae los agujeros como agujeros de verdad (§65.1), asi que esto se
     mide, no se supone.
  2. **Que las copas caigan dentro de la zona descartada.** `zonas.py` ya define
     esa zona: base (8.5-17.6 % de la lamina, interpolado por el largo) y franja
     de la vena (0.6 anchos de cinta = 16.9 mm). Aqui se reusan esos MISMOS
     numeros; no se define una zona nueva para la ocasion.

LO QUE MIDE, Y POR QUE ESTE ORDEN
---------------------------------
Para cada hoja se busca la barra de cuatro ventosas (§65.4: r = 22 mm, paso
80 mm, en linea a lo largo de la vena) que:

    a) apoye las CUATRO copas enteras sobre hoja  -- condicion dura;
    b) maximice el area de copa dentro de la zona descartada;
    c) y, a igualdad, agarre lo MAS BAJO posible.

(a) es dura porque una copa sobre un agujero no sujeta. (b) antes que (c) porque
la promesa de la tesis es «marca la parte que se corta igual», no «agarra
abajo»: si hay que elegir, manda no estropear hoja util.

Y se mide una cosa mas, que es la que decide si la herramienta dibujada sirve:
**el radio de copa mas grande que la regla permite** -- el mayor r tal que las
cuatro copas quepan ENTERAS dentro de la zona descartada. Es un numero de
geometria, no una opinion sobre vacio.

CONTRA QUE SE COMPARA
---------------------
Contra el centro de masa que hay hoy en `mallas_3d.csv`, con la misma barra y la
misma medida. Sin esa columna el cambio seria una preferencia; con ella es una
comparacion. Es la regla de §45.2 aplicada a algo que no es exactitud.

Uso:
    python scripts/agarre.py --max 20 --qc 8
    python scripts/agarre.py
Salidas: out/agarre.csv, out/AGARRE.txt, out/qc_agarre/*.png
"""
import argparse
import csv
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))
from zonas import LIM_PEQUENA, LIM_GRANDE, VENA_ANCHOS, FRAC_RABO     # noqa: E402
from clasificador import PULG_POR_ANCHO_CINTA                          # noqa: E402
from rutas import REPO_RAIZ  # raiz del repositorio

OUT = REPO_RAIZ / "out"
MM_POR_PULG = 25.4
VENA_MM = VENA_ANCHOS * PULG_POR_ANCHO_CINTA * MM_POR_PULG   # 16.9 mm de ancho

# La herramienta, tal cual esta en celda_geom.barra_ventosas(largo=320, n=4).
R_CUP = 22.0
N_CUP = 4
LARGO_BARRA = 320.0
OFF_CUP = [-LARGO_BARRA / 2.0 + LARGO_BARRA * (k + 0.5) / N_CUP for k in range(N_CUP)]

PASO = 1.0          # mm por pixel del raster. La copa mide 44 mm: sobra resolucion
PASO_BUSCA = 2      # px de paso en la rejilla de posiciones


# ------------------------------------------------------------------ la malla
def lee_stl(ruta):
    """Triangulos (n,3,3) de un STL binario, en milimetros."""
    b = Path(ruta).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
    return a[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)


def rasteriza(T, paso=PASO, margen=30.0):
    """Mascara de HOJA vista desde arriba, con los agujeros como huecos.

    Se pintan los triangulos de la lamina; las paredes de los agujeros son
    verticales y se proyectan en una linea, asi que no rellenan el hueco.

    OJO, y costo una sonda (2026-09-08): `cv2.fillPoly(m, lista_de_triangulos)`
    **no rellena cada triangulo**. Trata la lista como UN poligono de varios
    contornos con regla par-impar, asi que los triangulos vecinos se anulan y lo
    que queda es el contorno de la hoja, de 4 mm de ancho. Con eso, la holgura
    maxima de cualquier hoja salia 2.2 mm y el guion contestaba "0 de 12 hojas
    se pueden agarrar", que es falso. Rellenados de uno en uno: 62 727 px contra
    los 62 189 mm2 que dice `mallas_3d.csv`. Y no es lento: 0.01 s por hoja.
    """
    P = T.reshape(-1, 3)
    x0, y0 = P[:, 0].min() - margen, P[:, 1].min() - margen
    x1, y1 = P[:, 0].max() + margen, P[:, 1].max() + margen
    w = int(np.ceil((x1 - x0) / paso)) + 1
    h = int(np.ceil((y1 - y0) / paso)) + 1
    m = np.zeros((h, w), np.uint8)
    q = np.empty((len(T), 3, 2), np.int32)
    q[:, :, 0] = np.round((T[:, :, 0] - x0) / paso)
    q[:, :, 1] = np.round((T[:, :, 1] - y0) / paso)
    for t in q:
        cv2.fillPoly(m, [t], 1)
    return m, (x0, y0)


def desde_contornos(fuera, dentro, paso=PASO, margen=30.0):
    """La misma mascara, pero de los contornos en mm que devuelve `canoniza`.

    Es el camino de UNA FOTO NUEVA: la celda no tiene el STL de la hoja que
    viene por la banda, tiene la foto. `malla_3d.canoniza` da los mismos
    contornos con los que se tejio la malla, asi que el agarre de una hoja nueva
    se calcula con el mismo codigo que el de las 714 y no con una copia.
    """
    x0, y0 = fuera[:, 0].min() - margen, fuera[:, 1].min() - margen
    w = int(np.ceil((fuera[:, 0].max() + margen - x0) / paso)) + 1
    h = int(np.ceil((fuera[:, 1].max() + margen - y0) / paso)) + 1
    m = np.zeros((h, w), np.uint8)
    def _px(c):
        q = np.empty((len(c), 2), np.int32)
        q[:, 0] = np.round((c[:, 0] - x0) / paso)
        q[:, 1] = np.round((c[:, 1] - y0) / paso)
        return q
    cv2.fillPoly(m, [_px(fuera)], 1)
    for c in (dentro or []):
        if len(c) >= 3:
            cv2.fillPoly(m, [_px(c)], 0)        # los agujeros, agujeros
    return m, (float(x0), float(y0))


# ------------------------------------------------------------------ la regla
def frac_base_punta(lamina_pulg):
    """(frac_punta, frac_base) permisibles, interpoladas como en zonas.py."""
    l0, p0, b0 = LIM_PEQUENA
    l1, p1, b1 = LIM_GRANDE
    u = 0.0 if l1 == l0 else (lamina_pulg - l0) / (l1 - l0)
    u = float(np.clip(u, 0.0, 1.0))
    return p0 + u * (p1 - p0), b0 + u * (b1 - b0)


def zona_descartada(m):
    """(descartada, diag): base + punta + franja de la vena, en pixeles.

    El eje de la hoja es y (la base en y=0, marco canonico de malla_3d) y la
    vena se sigue fila a fila por el centro de la silueta, no se supone en x=0:
    una hoja algo torcida movria la franja.
    """
    h, w = m.shape
    anch = m.sum(1).astype(float)                 # anchura aparente por fila
    if anch.max() <= 0:
        return None, {"error": "mascara vacia"}
    # lamina = hoja sin el rabo del peciolo, mismo criterio que zonas.lamina
    dentro = np.nonzero(anch >= FRAC_RABO * anch.max())[0]
    y0, y1 = int(dentro[0]), int(dentro[-1])
    lam_mm = (y1 - y0) * PASO
    f_punta, f_base = frac_base_punta(lam_mm / MM_POR_PULG)

    # centro de la silueta por fila -> el eje de la vena
    xs = np.arange(w)[None, :]
    con = m > 0
    izq = np.where(con.any(1), np.argmax(con, 1), 0)
    der = np.where(con.any(1), w - 1 - np.argmax(con[:, ::-1], 1), 0)
    cen = (izq + der) / 2.0
    k = max(3, int(round(41 / PASO)) | 1)          # mediana de ~41 mm, alisa el borde
    cen = median_filter(cen, size=k, mode="nearest")

    en_vena = np.abs(xs - cen[:, None]) <= (VENA_MM / 2.0) / PASO
    ys = np.arange(h)[:, None]
    en_base = ys <= (y0 + f_base * (y1 - y0))      # incluye el peciolo (y < y0)
    en_punta = ys >= (y1 - f_punta * (y1 - y0))
    desc = ((en_vena | en_base | en_punta) & con).astype(np.uint8)
    return desc, {"y0": y0, "y1": y1, "lam_mm": lam_mm, "eje": cen,
                  "frac_base": f_base, "frac_punta": f_punta}


# ------------------------------------------------------------------ el agarre
def _dist(m):
    """Distancia en mm de cada pixel al borde de la mascara (0 fuera)."""
    return cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32) * PASO


def _peor(D, desplaz):
    """Para cada centro de barra, el valor de D en la PEOR de las cuatro copas.

    `desplaz` son los (dy, dx) de las copas; el resultado en (y, x) es el minimo
    de D(y+dy, x+dx). Fuera del cuadro vale -1, que nunca gana.
    """
    h, w = D.shape
    out = np.full((h, w), np.inf, np.float32)
    for dy, dx in desplaz:
        s = np.full((h, w), -1.0, np.float32)
        ys0, ys1 = max(0, -dy), min(h, h - dy)
        xs0, xs1 = max(0, -dx), min(w, w - dx)
        if ys0 < ys1 and xs0 < xs1:
            s[ys0:ys1, xs0:xs1] = D[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        out = np.minimum(out, s)
    return out


def _desplaz(off_px, ang_grados):
    """(dy, dx) de las cuatro copas para una barra girada `ang`."""
    a = np.radians(ang_grados)
    return [(int(round(o * np.cos(a))), int(round(-o * np.sin(a)))) for o in off_px]


def nucleo(ang_grados, r_px, off_px):
    """Las cuatro copas como un solo nucleo, girado `ang` sobre el centro."""
    a = np.radians(ang_grados)
    ext = int(np.ceil(abs(max(off_px)) * max(abs(np.cos(a)), abs(np.sin(a))))) + r_px + 1
    k = np.zeros((2 * ext + 1, 2 * ext + 1), np.float32)
    yy, xx = np.ogrid[-ext:ext + 1, -ext:ext + 1]
    for o in off_px:
        cy, cx = o * np.cos(a), -o * np.sin(a)
        k[((xx - cx) ** 2 + (yy - cy) ** 2) <= r_px * r_px] = 1.0
    return k


def busca(m, desc, angulos):
    """Mejor barra de 4 copas: mas area de copa en la zona que se descarta.

    POR QUE EL AREA Y NO LA PROFUNDIDAD (corregido el 2026-09-08, mirando el QC)
    ---------------------------------------------------------------------------
    El primer criterio fue «la copa mas metida en la zona descartada». Con la
    copa de 22 mm y la franja de vena de 16.9 mm, **ninguna copa cabe entera**,
    asi que esa medida vale 0 en TODAS las posiciones: empataban todas y
    desempataba la altura, que dejaba la barra fuera de la vena, sobre hoja
    util. Se vio en la imagen de control, no en los numeros.

    El area de copa dentro de la zona no empata nunca y es exactamente lo que
    luego se reporta, asi que se optimiza lo que se mide.

    Y la barra GIRA. El robot puede orientarla, y la vena de una hoja real no
    cae siempre paralela al eje canonico de la malla.
    """
    off = [int(round(o / PASO)) for o in OFF_CUP]
    r_px = int(round(R_CUP / PASO))
    Dh = _dist(m)                       # holgura a hoja (agujero o borde)
    Dd = _dist(desc)                    # holgura a zona descartada
    fdesc = desc.astype(np.float32)
    fuera = (m == 0).astype(np.float32)  # agujero o aire

    res = {"r_max_regla": 0.0, "apoyo": False}
    mejor = None
    for ang in angulos:
        d = _desplaz(off, ang)
        k = nucleo(ang, r_px, off)
        area_k = float(k.sum())
        a_desc = cv2.filter2D(fdesc, -1, k, borderType=cv2.BORDER_CONSTANT)
        a_fuera = cv2.filter2D(fuera, -1, k, borderType=cv2.BORDER_CONSTANT)
        ok = a_fuera < 0.5              # ni un pixel de copa fuera de la hoja
        # el radio que la regla permite no depende del apoyo: es geometria
        res["r_max_regla"] = max(res["r_max_regla"], float(_peor(Dd, d).max()))
        if not ok.any():
            continue
        ys, xs = np.nonzero(ok)
        val = a_desc[ys, xs] / area_k
        i = np.lexsort((ys, -val))[0]
        cand = (float(val[i]), int(ys[i]), int(xs[i]), ang, d)
        if mejor is None or cand[0] > mejor[0]:
            mejor = cand
    if mejor is None:
        return res
    val, y, x, ang, d = mejor
    res.update(apoyo=True, y_px=y, x_px=x, angulo=ang, area_en_zona=val,
               holgura_mm=float(_peor(Dh, d)[y, x]))
    return res


def area_en_zona(desc, cx_px, cy_px, desplaz, r_px):
    """Fraccion del area de las cuatro copas que cae en la zona descartada."""
    h, w = desc.shape
    yy, xx = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    disco = (xx * xx + yy * yy) <= r_px * r_px
    tot = dentro = 0
    for dy, dx in desplaz:
        y0, x0 = cy_px + dy - r_px, cx_px + dx - r_px
        y1, x1 = y0 + disco.shape[0], x0 + disco.shape[1]
        tot += int(disco.sum())
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue
        dentro += int(((desc[y0:y1, x0:x1] > 0) & disco).sum())
    return dentro / tot if tot else 0.0


def qc(m, desc, dibuja, destino):
    """Imagen de control: hoja, zona descartada y las copas de las dos reglas."""
    img = np.zeros(m.shape + (3,), np.uint8)
    img[m > 0] = (120, 150, 170)                # BGR: gris tabaco
    img[desc > 0] = (40, 120, 235)              # BGR: naranja
    for (cx, cy), desplaz, col in dibuja:
        pts = [(int(round(cx + dx)), int(round(cy + dy))) for dy, dx in desplaz]
        for c in pts:
            cv2.circle(img, c, int(round(R_CUP / PASO)), col, 2)
        cv2.line(img, pts[0], pts[-1], col, 1)
    img = cv2.flip(img, 0)              # y hacia arriba, la base abajo
    cv2.putText(img, "naranja=zona que la regla descarta   verde=regla   "
                "rojo=centro de masa", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (240, 240, 240), 1, cv2.LINE_AA)
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="solo las primeras N hojas")
    ap.add_argument("--qc", type=int, default=10, help="cuantas imagenes de control")
    ap.add_argument("--giro", type=float, default=30.0,
                    help="cuanto puede girar la barra, en grados a cada lado")
    ap.add_argument("--paso-giro", type=float, default=5.0)
    a = ap.parse_args()
    angulos = list(np.arange(-a.giro, a.giro + 1e-9, a.paso_giro)) if a.giro else [0.0]

    idx = list(csv.DictReader(open(OUT / "mallas_3d.csv", encoding="utf-8")))
    if a.max:
        idx = idx[:a.max]
    print("hojas: {}".format(len(idx)), flush=True)

    off_px = [int(round(o / PASO)) for o in OFF_CUP]
    r_px = int(round(R_CUP / PASO))
    filas, n_qc = [], 0
    for i, row in enumerate(idx, 1):
        try:
            T = lee_stl(row["ruta_stl"])
            m, (ox, oy) = rasteriza(T)
            desc, dg = zona_descartada(m)
            if desc is None:
                raise RuntimeError(dg["error"])
            r = busca(m, desc, angulos)

            # el agarre de hoy, medido con la misma vara: misma barra, sin giro,
            # que es como esta puesto en mallas_3d.csv
            d0 = _desplaz(off_px, 0.0)
            cx0 = int(round((float(row["agarre_x_mm"]) - ox) / PASO))
            cy0 = int(round((float(row["agarre_y_mm"]) - oy) / PASO))
            Dh = _dist(m)
            cm_h = float(_peor(Dh, d0)[cy0, cx0]) if (
                0 <= cy0 < Dh.shape[0] and 0 <= cx0 < Dh.shape[1]) else -1.0
            cm_area = area_en_zona(desc, cx0, cy0, d0, r_px)

            f = dict(archivo=row["archivo"], variedad=row["variedad"],
                     clase=row["clase"], apoyo=int(r["apoyo"]),
                     r_max_regla_mm=round(r["r_max_regla"], 2))
            if r["apoyo"]:
                f.update(agarre_x_mm=round(r["x_px"] * PASO + ox, 1),
                         agarre_y_mm=round(r["y_px"] * PASO + oy, 1),
                         giro_grados=round(r["angulo"], 1),
                         holgura_mm=round(r["holgura_mm"], 1),
                         area_en_zona=round(r["area_en_zona"], 3),
                         y_rel_lamina=round((r["y_px"] - dg["y0"]) /
                                            max(1, dg["y1"] - dg["y0"]), 3))
            f.update(cm_holgura_mm=round(cm_h, 1), cm_apoyo=int(cm_h >= R_CUP),
                     cm_area_en_zona=round(cm_area, 3),
                     lamina_mm=round(dg["lam_mm"], 1))
            filas.append(f)

            if n_qc < a.qc and r["apoyo"]:
                qc(m, desc,
                   [((r["x_px"], r["y_px"]), _desplaz(off_px, r["angulo"]),
                     (90, 220, 90)),
                    ((cx0, cy0), d0, (80, 80, 235))],
                   OUT / "qc_agarre" / (Path(row["archivo"]).stem + ".png"))
                n_qc += 1
        except Exception as ex:
            print("   [!] {}: {}".format(row["archivo"], ex), flush=True)
        if i % 50 == 0:
            print("   {} de {}".format(i, len(idx)), flush=True)

    if not filas:
        sys.exit("ninguna hoja medida")
    cols = ["archivo", "variedad", "clase", "apoyo", "agarre_x_mm", "agarre_y_mm",
            "giro_grados", "holgura_mm", "area_en_zona", "y_rel_lamina", "r_max_regla_mm",
            "cm_apoyo", "cm_holgura_mm", "cm_area_en_zona", "lamina_mm"]
    with open(OUT / "agarre.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    ap_ = [f for f in filas if f["apoyo"]]
    L = []
    P = L.append
    P("EL PUNTO DE AGARRE POR LA REGLA  ({} hojas)".format(len(filas)))
    P("=" * 70)
    P("")
    P("herramienta: {} copas de r = {:.0f} mm, paso {:.0f} mm, en linea sobre la vena"
      .format(N_CUP, R_CUP, OFF_CUP[1] - OFF_CUP[0]))
    P("zona que la regla descarta: vena de {:.1f} mm de ancho + base + punta"
      .format(VENA_MM))
    P("")
    P("1. APOYO DE LAS CUATRO COPAS SOBRE HOJA")
    P("   por la regla        {:3d} de {:3d} hojas ({:.1f} %)"
      .format(len(ap_), len(filas), 100.0 * len(ap_) / len(filas)))
    ncm = sum(f["cm_apoyo"] for f in filas)
    P("   centro de masa      {:3d} de {:3d} hojas ({:.1f} %)"
      .format(ncm, len(filas), 100.0 * ncm / len(filas)))
    P("")
    if ap_:
        ar = np.array([f["area_en_zona"] for f in ap_])
        arc = np.array([f["cm_area_en_zona"] for f in filas])
        yr = np.array([f["y_rel_lamina"] for f in ap_])
        P("2. AREA DE COPA DENTRO DE LA ZONA QUE SE DESCARTA")
        P("   por la regla        {:.1f} %  (mediana; p10 {:.1f}, p90 {:.1f})"
          .format(100 * np.median(ar), 100 * np.percentile(ar, 10),
                  100 * np.percentile(ar, 90)))
        P("   centro de masa      {:.1f} %  (mediana)".format(100 * np.median(arc)))
        P("")
        P("3. DONDE AGARRA, EN FRACCION DE LAMINA DESDE LA BASE")
        P("   por la regla        {:.2f}  (mediana)".format(np.median(yr)))
        P("")
    rm = np.array([f["r_max_regla_mm"] for f in filas])
    P("4. EL RADIO DE COPA QUE LA REGLA PERMITE")
    P("   mayor r con las 4 copas ENTERAS dentro de la zona descartada:")
    P("   mediana {:.1f} mm   p10 {:.1f}   p90 {:.1f}   (la barra lleva {:.0f} mm)"
      .format(np.median(rm), np.percentile(rm, 10), np.percentile(rm, 90), R_CUP))
    P("   hojas que admiten r = {:.0f} mm entero en zona descartada: {} de {}"
      .format(R_CUP, int((rm >= R_CUP).sum()), len(rm)))
    P("")
    (OUT / "AGARRE.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("-> {}".format(OUT / "agarre.csv"))
    print("-> {}".format(OUT / "AGARRE.txt"))
    if n_qc:
        print("-> {}  ({} imagenes)".format(OUT / "qc_agarre", n_qc))


if __name__ == "__main__":
    main()
