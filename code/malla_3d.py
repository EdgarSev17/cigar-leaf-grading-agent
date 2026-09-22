r"""
DE LA FOTO AL OBJETO 3D PARA ROBODK (2026-09-06)

POR QUE NO HACE FALTA RECONSTRUIR NADA, Y POR QUE NO SE PODRIA
--------------------------------------------------------------
Una hoja de capa es una **lamina**: mide 17 x 9 pulgadas de ancho y menos de dos
decimas de milimetro de grueso. Todo lo que un robot necesita saber de ella --por
donde la agarra, cuanto ocupa, donde tiene los agujeros-- **esta en su silueta y en
su escala**, y las dos ya estan medidas y comprobadas en este proyecto: la mascara
de `segmentacion.py` y las pulgadas por pixel de la cinta metrica (§17, §24, §26).

Reconstruir profundidad de verdad (fotogrametria) **no se puede con estas fotos** y
conviene decirlo antes de intentarlo: hay una toma amplia y un acercamiento por
hoja, tomadas con el telefono en la mano, sin poses de camara conocidas y sin
solape suficiente. Una red de profundidad monocular daria una superficie *plausible*
pero **no metrica**, o sea justo lo que no sirve ni para simular una pinza ni para
defender en un tribunal. Aqui no se inventa relieve: se extruye lo medido.

QUE SACA ESTE GUION
-------------------
Por cada foto, un **STL binario en milimetros**, con:

  - la silueta real de esa hoja, simplificada a decimas de milimetro;
  - **los agujeros son agujeros de verdad en la malla** -- los mismos que detecta
    `agujeros.py`, no un dibujo;
  - un grosor uniforme (0.3 mm por defecto: el real es 0.1-0.2, pero tan fino da
    problemas de colision y casi no se ve);
  - y **una pose canonica**: la base de la hoja en el origen, la hoja creciendo
    hacia +Y, el ancho en X y la cara de abajo apoyada en Z = 0. Asi RoboDK la
    coloca sobre la banda sin tener que orientarla a mano, y el punto de agarre
    siempre cae en el mismo sitio relativo.

Y un indice, `out/mallas_3d.csv`, con la clase, la variedad, el largo y el ancho en
milimetros, el area, los agujeros y el punto de agarre de cada hoja. Es lo que lee
el agente para elegir a que brazo la manda.

LO QUE ESTE OBJETO **NO** ES
----------------------------
No tiene relieve ni curvatura: es una lamina plana. Si en la simulacion hace falta
que la hoja se vea arqueada, se le puede dar una curvatura sintetica, pero seria
**inventada**, no medida, y por eso no esta puesta.

Uso:
    python scripts/malla_3d.py --max 5           # cinco hojas, para mirarlas
    python scripts/malla_3d.py                   # todas las del manifiesto
    python scripts/malla_3d.py --foto una.heic   # una foto suelta
    python scripts/malla_3d.py --espesor 0.5 --paso 0
Se puede parar y relanzar: salta las que ya tienen STL (los procesos largos se
mueren solos en esta maquina).
"""
import argparse
import csv
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).parent))

from segmentacion import mascara_hoja, mascara_marca                    # noqa: E402
from enderezado import busca_cinta as cinta_angulo, canon               # noqa: E402
from zonas import perfil, extremo_base, ancho_cinta_por_carpeta         # noqa: E402
from rasgos_foto import carga, rota_exp, ancho_cinta_de_la_foto, LADO   # noqa: E402
from clasificador import PULG_POR_ANCHO_CINTA                           # noqa: E402
import agujeros as AG                                                    # noqa: E402
from rutas import REPO_RAIZ  # raiz del repositorio

RAIZ = REPO_RAIZ / "dataset"
OUT = REPO_RAIZ / "out"
DIR_MALLA = OUT / "mallas"
MM_POR_PULG = 25.4
AREA_MIN_AGUJERO_MM2 = 4.0      # menos de 2x2 mm no es un agujero, es ruido de mascara


# ---------------------------------------------------------------- geometria
def canoniza(ruta, escala="foto", ac_carpeta=None):
    """(contornos en mm, mm_por_px, diagnostico) de una foto.

    Repite la canonizacion de la tuberia --misma mascara, mismo giro por la
    cinta, mismo criterio de base-- y devuelve la silueta ya en milimetros y en
    la pose que quiere RoboDK. No reimplementa ninguna deteccion: importa las
    mismas funciones que produjeron las cifras del proyecto (la regla de §60.1).
    """
    diag = {"archivo": Path(ruta).name, "avisos": []}
    rgb0 = carga(ruta, LADO)
    hoja, _, _ = mascara_hoja(rgb0, mascara_marca(rgb0))
    if hoja.sum() < 500:
        return None, None, dict(diag, error="no se encontro hoja")

    # --- giro por la cinta, igual que la tuberia (y con el mismo redondeo, §60.3)
    mc, ang, _, estado_c, _, _ = cinta_angulo(rgb0)
    rot, voltea = 0.0, 0
    if mc is not None:
        _, rot = canon(ang)
        rot = round(float(rot), 2)
        mrot = rota_exp(mc.astype(np.uint8) * 255, rot)
        hrot = rota_exp(hoja.astype(np.uint8) * 255, rot)
        if mrot.any() and hrot.any():
            voltea = int(np.nonzero(mrot)[1].mean() > np.nonzero(hrot)[1].mean())
    else:
        diag["avisos"].append("no se ve la cinta metrica")
    m, rgb = hoja, rgb0
    if rot:
        m = rota_exp(m.astype(np.uint8), rot, True) > 0
        rgb = rota_exp(rgb, rot)
    if voltea:
        m = cv2.rotate(m.astype(np.uint8), cv2.ROTATE_180) > 0
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    if m.sum() < 500:
        return None, None, dict(diag, error="la hoja se perdio al rotar")

    # --- escala
    if escala == "carpeta" and ac_carpeta:
        ac_px, origen = ac_carpeta, "mediana de la carpeta (como la tuberia)"
    else:
        ac_px, origen = ancho_cinta_de_la_foto(ruta, rgb0.shape[0])
        if not ac_px and ac_carpeta:
            ac_px, origen = ac_carpeta, "mediana de la carpeta (no se midio la cinta)"
    if not ac_px:
        return None, None, dict(diag, error="sin escala: la cinta no se ve")
    mm_px = PULG_POR_ANCHO_CINTA / ac_px * MM_POR_PULG
    diag.update(origen_escala=origen, mm_por_px=round(mm_px, 5),
                estado_cinta=estado_c)

    # --- el eje de la hoja y donde esta la base (mismo criterio que zonas.py)
    e, perp, mu, t, s, anch = perfil(m)
    base_al_final, margen = extremo_base(anch)
    # OJO CON EL SIGNO, que se vio mirando la miniatura y no la metrica (§29):
    # `zonas.py` deja la base en t.MAX porque su marco de referencia lo pide asi.
    # Aqui la queremos en el ORIGEN, o sea en t.MIN, asi que la condicion es la
    # contraria. Y se le da la vuelta a `e` Y a `perp` a la vez --un giro de 180
    # grados-- porque cambiar solo uno seria un ESPEJO, y un espejo intercambia
    # izquierda y derecha: convertiria una XL en una XR dentro de RoboDK.
    if base_al_final:
        e, perp = -e, -perp
    diag["margen_base"] = int(margen)
    if margen < 3:
        diag["avisos"].append("la base no se distingue bien de la punta")

    # --- LOS AGUJEROS SE PIDEN AL DETECTOR, no a la mascara.
    # `mascara_hoja` RELLENA los huecos interiores (paso 4), asi que la silueta
    # sola nunca trae agujeros. Y la mascara CRUDA tampoco vale: ahi son agujeros
    # las nervaduras palidas y los brillos. Los de verdad son los de `agujeros.py`
    # (§31): pixeles que se parecen mas al fondo de ESTA foto que a la hoja.
    pulg_px = PULG_POR_ANCHO_CINTA / ac_px
    try:
        cand, _, _, _ = AG.detecta(rgb, m, ac_px)
        min_px = max(AG.MIN_PX, int(AG.MIN_PULG2 / (pulg_px ** 2)))
        n, et, est, _ = cv2.connectedComponentsWithStats(
            (cand & m).astype(np.uint8), 8)
        # FILTRO DE FORMA, Y ES SOLO PARA LA MALLA (2026-09-06).
        #
        # Mirando el QC de Habano aparecieron 14 "agujeros" que en la foto no
        # existen: son **astillas finas** de una o dos decimas de milimetro de
        # ancho, a lo largo de las venas y de los pliegues del borde. El detector
        # las da por agujeros --y este guion no cambia el detector, que es lo que
        # produjo todas las cifras del proyecto-- pero un hueco de dos pixeles de
        # ancho no es un agujero fisico y no tiene por que salir en un objeto 3D.
        #
        # El criterio es el mismo que usa el propio detector para su tamaño
        # minimo (MIN_PULG2, ~3 mm de diametro): se exige que dentro del hueco
        # QUEPA UN CIRCULO de 1.5 mm de radio. Una astilla no lo cumple por muy
        # larga que sea; un agujero redondo si.
        #
        # OJO: esto NO arregla el detector. Que reporte astillas es un hallazgo
        # que hay que mirar aparte, porque `n_agujeros` es rasgo del modelo.
        r_min_px = max(1.0, 1.5 / mm_px)
        quita = np.zeros_like(m)
        n_astillas = 0
        for k in range(1, n):
            if est[k, cv2.CC_STAT_AREA] < min_px:
                continue
            comp = (et == k).astype(np.uint8)
            if cv2.distanceTransform(comp, cv2.DIST_L2, 3).max() < r_min_px:
                n_astillas += 1
                continue
            quita |= comp > 0
        diag["astillas_descartadas"] = n_astillas
        m = m & ~quita
        diag["px_agujero"] = int(quita.sum())
    except Exception as ex:                     # el detector no debe tumbar la malla
        diag["avisos"].append("no se pudieron marcar los agujeros: {}".format(ex))

    # --- contornos: el exterior y los agujeros de dentro
    cnts, jer = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP,
                                 cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, dict(diag, error="sin contorno")
    i_ext = int(np.argmax([cv2.contourArea(c) for c in cnts]))
    eps = max(1.0, 0.4 / mm_px)          # simplificar a ~0.4 mm
    fuera = cv2.approxPolyDP(cnts[i_ext], eps, True).reshape(-1, 2)
    dentro = []
    for i, c in enumerate(cnts):
        if jer[0][i][3] == i_ext and cv2.contourArea(c) * mm_px ** 2 >= AREA_MIN_AGUJERO_MM2:
            p = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(p) >= 3:
                dentro.append(p)

    def a_mm(pts):
        """px de la imagen -> mm en la pose canonica (base en 0, hoja hacia +Y)."""
        p = pts.astype(np.float64) - mu
        x = p @ perp
        y = p @ e
        return np.stack([x, y], 1) * mm_px

    fuera_mm = a_mm(fuera)
    dentro_mm = [a_mm(p) for p in dentro]
    # base en Y = 0 y ancho centrado en X = 0
    y0 = fuera_mm[:, 1].min()
    x0 = 0.5 * (fuera_mm[:, 0].min() + fuera_mm[:, 0].max())
    fuera_mm = fuera_mm - [x0, y0]
    dentro_mm = [p - [x0, y0] for p in dentro_mm]
    diag.update(largo_mm=round(float(fuera_mm[:, 1].max()), 1),
                ancho_mm=round(float(fuera_mm[:, 0].max()
                                     - fuera_mm[:, 0].min()), 1),
                n_agujeros_malla=len(dentro_mm))
    # LA TRANSFORMACION, PARA PODER DESHACERLA. `imprime_hojas.py` necesita ir de
    # un punto de la malla (mm) al PIXEL del que salio, para pintar la malla con
    # la foto. Se guarda aqui en vez de recalcularse alli, que es la regla de
    # §60.1: una sola definicion de cada cosa. La inversa es
    #     p_px = mu + x*perp + y*e,   con [x, y] = (mm + [x0, y0]) / mm_px
    # y la imagen que hay que muestrear es `rgb`, la YA GIRADA, no la original.
    # `rot` y `voltea` van tambien porque permiten repetir ESTE MISMO giro sobre
    # la foto cargada a mas resolucion, y muestrear el color con el detalle que
    # tiene la foto de verdad en vez del de la copia de trabajo (§72).
    diag["_tf"] = dict(mu=mu, e=e, perp=perp, x0=float(x0), y0=float(y0),
                       mm_px=mm_px, rgb=rgb, mascara=m,
                       rot=float(rot), voltea=int(voltea))
    return (fuera_mm, dentro_mm), mm_px, diag


def _cruz2(u, v):
    """Producto vectorial en 2D. numpy 2.x ya no acepta `cross` de vectores de 2."""
    return float(u[0] * v[1] - u[1] * v[0])


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def teje(fuera, dentro, espesor, paso_mm=8.0):
    """Malla cerrada de una lamina plana con agujeros.

    Se triangula el plano una sola vez (Delaunay sobre el contorno mas una
    rejilla interior) y se descartan los triangulos cuyo centro cae fuera de la
    hoja o dentro de un agujero -- que es lo que hace que un agujero de la foto
    sea un agujero de la malla. Luego esa misma tapa se copia arriba y abajo y
    se cosen las paredes.

    La rejilla interior no hace falta para una lamina plana; se deja porque es
    lo que permitiria darle curvatura mas adelante sin rehacer nada.
    """
    pts = [fuera] + list(dentro)
    P = np.vstack(pts)
    if paso_mm and paso_mm > 0:
        x0, y0 = P[:, 0].min(), P[:, 1].min()
        x1, y1 = P[:, 0].max(), P[:, 1].max()
        gx = np.arange(x0 + paso_mm, x1, paso_mm)
        gy = np.arange(y0 + paso_mm, y1, paso_mm)
        if len(gx) and len(gy):
            G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
            cf = fuera.astype(np.float32)
            dent = [np.array([cv2.pointPolygonTest(cf, (float(a), float(b)), True)
                              for a, b in G])]
            for h in dentro:
                dent.append(-np.array([cv2.pointPolygonTest(h.astype(np.float32),
                                                            (float(a), float(b)), True)
                                       for a, b in G]))
            ok = np.min(np.stack(dent), 0) > paso_mm * 0.35
            if ok.any():
                P = np.vstack([P, G[ok]])

    tri = Delaunay(P)
    cen = P[tri.simplices].mean(1)
    cf = fuera.astype(np.float32)
    dentro_f = [h.astype(np.float32) for h in dentro]
    vale = np.array([cv2.pointPolygonTest(cf, (float(a), float(b)), False) > 0
                     for a, b in cen])
    for h in dentro_f:
        vale &= np.array([cv2.pointPolygonTest(h, (float(a), float(b)), False) < 0
                          for a, b in cen])
    caras = tri.simplices[vale]
    if not len(caras):
        return None

    arriba = np.column_stack([P, np.full(len(P), espesor)])
    abajo = np.column_stack([P, np.zeros(len(P))])
    T = []
    for a, b, c in caras:
        pa, pb, pc = arriba[a], arriba[b], arriba[c]
        if _cruz2(pb[:2] - pa[:2], pc[:2] - pa[:2]) < 0:        # normal hacia +Z
            pb, pc = pc, pb
        T.append((pa, pb, pc))
        qa, qb, qc = abajo[a], abajo[b], abajo[c]
        if _cruz2(qb[:2] - qa[:2], qc[:2] - qa[:2]) > 0:        # normal hacia -Z
            qb, qc = qc, qb
        T.append((qa, qb, qc))

    # paredes: el contorno exterior antihorario y los agujeros al reves, para que
    # todas las normales miren hacia afuera de la hoja
    for k, poly in enumerate([fuera] + list(dentro)):
        q = poly if (_area(poly) > 0) == (k == 0) else poly[::-1]
        for i in range(len(q)):
            a, b = q[i], q[(i + 1) % len(q)]
            a0 = np.array([a[0], a[1], 0.0]); a1 = np.array([a[0], a[1], espesor])
            b0 = np.array([b[0], b[1], 0.0]); b1 = np.array([b[0], b[1], espesor])
            T.append((a0, b0, b1))
            T.append((a0, b1, a1))
    return T


def escribe_stl(triangulos, destino, nombre="hoja"):
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as f:
        f.write(("malla de hoja de tabaco -- " + nombre).ljust(80)[:80].encode())
        f.write(struct.pack("<I", len(triangulos)))
        for a, b, c in triangulos:
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 1e-12 else np.zeros(3)
            f.write(struct.pack("<3f", *n))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))


def qc(ruta_foto, fuera, dentro, mm_px, destino):
    """Miniatura para MIRAR: la foto al lado de la silueta que se ha extruido.

    La regla mas cara de este proyecto: **toda transformacion geometrica deja
    miniaturas que alguien mire**. Los dos errores mas caros --el signo
    izquierda/derecha y el giro de 178 grados-- se vieron mirando imagenes, no
    leyendo metricas (§29, §30).
    """
    px_mm = 0.5                                   # 2 px por milimetro
    x0, x1 = fuera[:, 0].min(), fuera[:, 0].max()
    y1 = fuera[:, 1].max()
    W = int((x1 - x0) / px_mm) + 20
    H = int(y1 / px_mm) + 20

    def a_px(p):
        return np.stack([(p[:, 0] - x0) / px_mm + 10,
                         H - 10 - p[:, 1] / px_mm], 1).astype(np.int32)

    lienzo = np.zeros((H, W, 3), np.uint8)
    cv2.fillPoly(lienzo, [a_px(fuera)], (60, 150, 60))
    for h in dentro:
        cv2.fillPoly(lienzo, [a_px(h)], (0, 0, 0))
    cv2.polylines(lienzo, [a_px(fuera)], True, (255, 255, 255), 1)
    cv2.circle(lienzo, tuple(a_px(np.array([[0.0, 0.0]]))[0]), 6, (0, 128, 255), -1)
    cv2.putText(lienzo, "base (0,0)", (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 128, 255), 1)

    foto = carga(ruta_foto, LADO)[:, :, ::-1]
    esc = H / foto.shape[0]
    foto = cv2.resize(foto, (int(foto.shape[1] * esc), H))
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), np.hstack([foto, lienzo]))


# ---------------------------------------------------------------- el recorrido
def lee(n):
    f = OUT / n
    return ({} if not f.exists()
            else {r["ruta"]: r for r in csv.DictReader(open(f, encoding="utf-8"))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foto", default=None, help="una foto suelta")
    ap.add_argument("--max", type=int, default=0, help="cuantas hojas nuevas")
    ap.add_argument("--espesor", type=float, default=0.3, help="mm")
    ap.add_argument("--paso", type=float, default=8.0, help="rejilla interior, mm")
    ap.add_argument("--escala", choices=("foto", "carpeta"), default="foto")
    ap.add_argument("--qc", type=int, default=0,
                    help="guarda miniatura de las N primeras, para mirarlas")
    ap.add_argument("--todas", action="store_true",
                    help="todas las fotos, no una por hoja (incluye acercamientos)")
    a = ap.parse_args()

    man = lee("manifiesto_limpio.csv")
    ac_carp = ancho_cinta_por_carpeta(OUT)

    if a.foto:
        tareas = [(Path(a.foto), {"carpeta": "", "clase": "", "variedad": ""})]
    elif a.todas:
        tareas = [(RAIZ / r, man[r]) for r in sorted(man)]
    else:
        # UNA MALLA POR HOJA, Y LA DEL ENCUADRE MAS AMPLIO. Dos motivos, los dos
        # medidos: un ACERCAMIENTO recorta la hoja, asi que su silueta no es la de
        # la hoja (§27); y una hoja que TOCA EL BORDE del cuadro esta cortada por
        # el encuadre, asi que su contorno es en parte el borde de la foto (§3).
        # Es el mismo filtro con el que se entrena el clasificador.
        seg, zon = lee("segmentacion.csv"), lee("zonas.csv")
        mejor = {}
        for r in man:
            if seg.get(r, {}).get("toca_borde") != "0":
                continue
            h = man[r]["hoja_id"]
            try:
                f = float(seg[r]["frac_cuadro"])
            except (KeyError, ValueError):
                continue
            if h not in mejor or f < mejor[h][1]:
                mejor[h] = (r, f)
        tareas = [(RAIZ / v[0], man[v[0]]) for v in
                  sorted(mejor.values(), key=lambda v: v[0])]
    print("hojas en cola: {}".format(len(tareas)), flush=True)

    f_idx = OUT / "mallas_3d.csv"
    hechas = set()
    if f_idx.exists():
        hechas = {r["archivo"] for r in csv.DictReader(open(f_idx, encoding="utf-8"))}
    fh = open(f_idx, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not hechas:
        w.writerow(["archivo", "ruta_stl", "variedad", "clase", "largo_mm",
                    "ancho_mm", "area_mm2", "n_agujeros_malla", "triangulos",
                    "espesor_mm", "agarre_x_mm", "agarre_y_mm", "escala",
                    "mm_por_px", "avisos"])

    n_new = n_err = 0
    for i, (p, meta) in enumerate(tareas, 1):
        if p.name in hechas:
            continue
        carpeta = meta.get("carpeta", "")
        destino = (DIR_MALLA / carpeta / (p.stem + ".stl")) if carpeta \
            else DIR_MALLA / (p.stem + ".stl")
        try:
            cont, mm_px, diag = canoniza(
                p, a.escala, ac_carp.get(carpeta) if carpeta else None)
            if cont is None:
                raise RuntimeError(diag.get("error", "?"))
            fuera, dentro = cont
            T = teje(fuera, dentro, a.espesor, a.paso)
            if not T:
                raise RuntimeError("no se pudo triangular")
            escribe_stl(T, destino, p.stem)
            if a.qc and n_new < a.qc:
                qc(p, fuera, dentro, mm_px, OUT / "qc_mallas" / (p.stem + ".png"))
            area = abs(_area(fuera)) - sum(abs(_area(h)) for h in dentro)
            # punto de agarre provisional: el centro de masa de la silueta. Es
            # donde una ventosa tiene mas hoja alrededor; afinarlo con las zonas
            # sanas es lo siguiente, y ya esta medido de que lado estan.
            cx = float(np.mean(fuera[:, 0]))
            cy = float(np.mean(fuera[:, 1]))
            w.writerow([p.name, str(destino), meta.get("variedad", ""),
                        meta.get("clase", ""), diag["largo_mm"], diag["ancho_mm"],
                        round(area, 1), diag["n_agujeros_malla"], len(T),
                        a.espesor, round(cx, 1), round(cy, 1),
                        diag.get("origen_escala", ""), diag.get("mm_por_px", ""),
                        "; ".join(diag.get("avisos", []))[:120]])
            fh.flush()
            n_new += 1
        except Exception as ex:
            n_err += 1
            print("   [!] {}: {}".format(p.name, ex), flush=True)
        if n_new and n_new % 10 == 0:
            print("   {} mallas ({} de {} fotos vistas)".format(n_new, i, len(tareas)),
                  flush=True)
        if a.max and n_new >= a.max:
            print("   [corte voluntario tras {} nuevas; relanzar para seguir]"
                  .format(n_new))
            break
    fh.close()
    print("\nmallas nuevas: {}   fallos: {}".format(n_new, n_err))
    print("-> {}".format(DIR_MALLA))
    print("-> {}".format(f_idx))


if __name__ == "__main__":
    main()
