r"""
Detecta AGUJEROS y ROTURAS DE BORDE en la hoja.   (PROGRESO: "MANANA: DETECTAR AGUJEROS")

POR QUE ESTE ES EL DEFECTO FACIL
--------------------------------
Un agujero no es un color, es la AUSENCIA de hoja: por el hueco se ve la mesa.
Asi que no hace falta aprender nada -- basta medir, dentro de la mascara de hoja,
que pixeles se parecen mas al FONDO DE ESA MISMA FOTO que a la hoja de esa misma
foto. Las dos referencias se miden en cada imagen, no se fijan por adelantado:
es la leccion que este proyecto ya pago tres veces (el umbral `a*<9` que borraba
el Habano, la mesa que el 28/08 es negra y no azul marino, el detector de azul
que en realidad media la mesa).

    referencia de fondo   anillo alrededor de la hoja  (mesa contigua, misma luz)
    referencia de hoja    interior de la mascara erosionada
    candidato a agujero   dentro de la hoja, mas cerca del fondo que de la hoja,
                          y claramente menos calido que la hoja (b* por debajo)

DOS TIPOS, Y LA DIFERENCIA IMPORTA
----------------------------------
    AGUJERO   hueco rodeado de hoja por todos lados. La mascara guardada por
              `segmentacion.py` ya viene RELLENA (su paso 4 inunda desde fuera),
              asi que el agujero esta DENTRO de la mascara y se encuentra por color.
    ROTURA    mella abierta al contorno, ANGOSTA: no esta dentro de la mascara,
              esta arrancada de ella. Se encuentra por FORMA: se cierra la mascara
              con un disco de radio R y lo que el cierre anade son las entradas cuya
              BOCA mide menos de 2R.
    MORDIDA   el trozo que falta cuando la boca es ANCHA. Se vio mirando
              20260828_200025952 (una 1/2 Banda): le falta una bahia de mas de una
              pulgada de boca en el lado izquierdo -- que es justo lo que define la
              clase -- y el cierre no la sellaba, asi que el defecto que MAS importa
              se quedaba sin medir. Se mide contra la envolvente convexa de la propia
              hoja: hoja sana convexa - hoja real = lo que falta. Solo se cuenta
              dentro de la zona util, porque hacia la base la hoja converge al
              peciolo y ahi la envolvente anade area que no es ningun defecto.

DE LA MORDIDA IMPORTA LA PENETRACION, NO EL AREA  (2026-09-03, §40.6)
---------------------------------------------------------------------
El area de mordida se excluyo del clasificador en §35 porque empeoraba: la
mordida mediana mas grande era la de Habano/Capa, la clase con MENOS dano, o sea
que estaba midiendo curvatura de la hoja y no dano. La exclusion era correcta,
pero se paso de frenada -- porque la regla que dio el experto dice otra cosa:

    "El borde si va cortado con la chaveta... se corta hasta que se termina lo
     arrugado, eso va a depender de que tan danado este el borde. Le esta
     cortando lo feo de las orillas."

O sea que el borde no es una banda de anchura fija: el corte se ADAPTA al dano, y
lo que descalifica es que el dano llegue **mas adentro de lo que la chaveta se
puede llevar**. La magnitud que decide es entonces **cuanto penetra**, no cuantas
pulgadas cuadradas ocupa. Se mide (`pen_pulg`) como la distancia del pixel mas
hondo de la mordida al exterior de la envolvente convexa, y separa justo lo que el
area confundia: una hoja arqueada deja un deficit grande pero PEGADO al borde
(penetracion pequena), y un pedazo que falta entra hacia el centro.

Como la curvatura afecta a los dos lados por igual y un pedazo que falta es de UN
lado, el rasgo util es la ASIMETRIA: `asim_pen_mordida` = izquierda - derecha.

POR QUE HACIA FALTA AHORA
-------------------------
La matriz de PROGRESO 30.3: 1/2 Banda acierta 22 de 52 y se confunde con Banda.
1/2 Banda es "una Banda pero con un agujero notable en un lado" (PROGRESO 22.1) y
el clasificador todavia no ve agujeros. Este guion produce los rasgos que faltan.

TRAMPAS CONOCIDAS, YA TAPADAS
-----------------------------
  - Los circulos de marcador azul son azules, o sea que se parecen a la mesa: sin
    excluirlos, CADA circulo se contaria como un agujero. Se excluyen -- y de paso
    sirven de comprobacion, porque el defecto de verdad cae DENTRO del circulo.
  - Los brillos especulares tambien rompen el color de la hoja, pero son CLAROS,
    no oscuros; se descartan por luminancia.
  - Las nervaduras palidas siguen siendo calidas (b* alto): no pasan el filtro.
  - Los circulos NO son anotacion exhaustiva de defectos (PROGRESO 22.4). Sirven
    para verificar casos, no como verdad-terreno completa.

SALIDAS
-------
  out/agujeros.csv          una fila por foto (rasgos para el clasificador)
  out/agujeros_detalle.csv  una fila por defecto (tipo, tamano, zona, distancia)
  out/qc_agujeros/          superposiciones para MIRAR -- obligatorio (PROGRESO 29.3)
  out/AGUJEROS_INFORME.txt

Uso:
    python scripts/agujeros.py --muestra 8 --qc 40    # prueba por carpeta
    python scripts/agujeros.py 200                    # 200 nuevas y para (reanudable)
    python scripts/agujeros.py --qc 60                # todas
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
from segmentacion import mascara_marca
from rutas import REPO_ROOT  # repository root

RAIZ = REPO_ROOT / "dataset"
OUT = REPO_ROOT / "out"
LADO = 1024
PULG_POR_ANCHO_CINTA = 1.11        # PROGRESO 26, +-15 %

MIN_PX = 25            # suelo de ruido, en pixeles
MIN_PULG2 = 0.010      # ~3 mm de diametro; por debajo no se afirma nada
MARGEN_B = 6.0         # b* minimo por debajo de la hoja para creerse un hueco
# 2026-09-19. K_DISP > 0 anade la condicion que hace al detector INDEPENDIENTE DEL
# COLOR DE LA MESA. Medido ese dia: la mesa del 17 y 24/08 (Connecticut) tiene
# b* = -15 y la del 28/08 (Habano) b* = -2, mientras que la hoja vale 36 y 38. La
# regla `dist(fondo) < dist(hoja)` es el punto MEDIO entre las dos, asi que la
# frontera se para en b* ~ 10 con la mesa azul y en b* ~ 18 con la negra -- y las
# venas y las sombras de la propia hoja viven en b* 10..20. Resultado: 11 agujeros
# por hoja en Habano/Capa contra 1 en Connecticut/Capa, cuando una capa no tiene
# agujeros. Con K_DISP el hueco tiene que estar a K desviaciones robustas del color
# de LA PROPIA HOJA, que no depende de la mesa. Solo puede QUITAR candidatos, nunca
# anadir, asi que Connecticut no puede empeorar.
K_DISP = 0.0
# 2026-09-19, el segundo criterio y el bueno. K_DISP exige que el hueco se salga
# del color de la hoja, y eso no basta: el Habano tiene manchas pardas que tambien
# se salen. Lo que de verdad define un agujero es que POR EL HUECO SE VE LA MESA,
# asi que el color mediano del candidato tiene que caer CERCA del color de la mesa
# de esa misma foto -- a menos de FRAC_FONDO veces la separacion hoja-mesa. Al ser
# una fraccion de esa separacion, el criterio significa lo mismo con mesa azul
# (b*=-15) que con mesa negra (b*=-2), que es justo lo que se rompio. 0 = apagado.
FRAC_FONDO = 0.0
# 2026-09-19, tercer criterio y el que respeta las dos variedades. FRAC_FONDO mide
# la tolerancia en fraccion de la separacion hoja-mesa, y esa separacion es MAS
# CORTA en Habano (40 contra 52,5 de Connecticut, porque la mesa del 28/08 es negra
# y no azul), asi que al Habano le tocaba una tolerancia mas estrecha justo donde
# hacia falta mas: se llevaba tambien agujeros de verdad y el XL izquierdo se
# quedaba sin senal. TOL_FONDO normaliza por lo que NO se encoge: el ruido propio
# de la mesa en esa foto (mediana + MAD de la distancia de los pixeles del anillo
# al color mediano del anillo). Un agujero ensena la mesa, asi que su color tiene
# que caer dentro del ruido de la mesa, y eso significa lo mismo con mesa azul que
# con mesa negra. 0 = apagado.
TOL_FONDO = 0.0
# 2026-09-19. AGUJERO_REAL = 1 aplica DENTRO del detector la verificacion que
# `agujero_de_verdad.py` (15/09) ya midio y dejo escrita, con su metodo, que es
# mejor que el de FRAC_FONDO: se EROSIONA cada componente para quedarse con su
# NUCLEO --en un componente pequenio el contorno es casi todo, y el contorno de un
# agujero real esta mezclado con hoja-- y el nucleo se compara con las dos
# referencias de la misma foto. Un agujero de verdad se parece al meson; un pliegue,
# a la lamina. Medido el 15/09 sobre las 112: el area de danio en capa baja de 0,413
# a 0,000 y en xr_der de 1,131 a 0,504, o sea que la mitad de lo que el detector
# marcaba en una capa eran pliegues. Aqui se usa para que `area_util_izq/der` --y por
# tanto `asim_area_util`, el rasgo que dice de que lado esta el danio-- se calculen
# solo con agujeros verificados. 0 = apagado.
AGUJERO_REAL = 0
EROSION_NUCLEO = 3   # px, el mismo de agujero_de_verdad.py
# 2026-09-19, MIRANDO LAS FOTOS. En una capa de Habano del experto el detector marcaba
# 35 cosas, y al dibujarlas se ve lo que son: **las arrugas y los dobleces de la
# hoja**. Lineas finas, largas y curvas que siguen los pliegues, el borde y las
# venas. No son manchas ni pizquitas -- excluir el sudado no cambio nada (28 -> 22).
# Lo que las separa de un agujero no es el color: es la FORMA. Un agujero es
# compacto; una sombra de doblez es un hilo. Un hilo desaparece al adelgazar la
# marca unos pixeles y un agujero no. GROSOR_MIN es ese adelgazamiento, en pixeles
# de radio; 0 lo apaga.
GROSOR_MIN = 3
R_ROTURA = 0.25        # radio de cierre, en anchos de cinta (~0.28 pulgadas)
GROSOR_MORDIDA = 0.12  # pulgadas de penetracion minima para llamarlo mordida


def carga(p, lado=LADO):
    with Image.open(p) as im:
        try:
            im.draft("RGB", (im.width // 4, im.height // 4))
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((lado, lado), Image.LANCZOS)
        return np.asarray(im)


def rota(img, grados, nearest=False):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), grados, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def disco(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def rellena(m):
    """Rellena los huecos interiores de una mascara binaria."""
    h, w = m.shape
    marco = np.zeros((h + 2, w + 2), np.uint8)
    marco[1:-1, 1:-1] = m.astype(np.uint8)
    inv = (1 - marco).astype(np.uint8)
    cv2.floodFill(inv, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    return ~(inv[1:-1, 1:-1] == 2)


def referencias(lab, m, r):
    """Color mediano del fondo contiguo y de la hoja. Se miden en CADA foto."""
    mu8 = m.astype(np.uint8)
    fuera = (cv2.dilate(mu8, disco(3 * r)) > 0) & ~(cv2.dilate(mu8, disco(r)) > 0)
    if fuera.sum() < 200:
        fuera = ~m
    dentro = cv2.erode(mu8, disco(r)) > 0
    if dentro.sum() < 200:
        dentro = m
    f = np.median(lab[fuera], axis=0)
    h = np.median(lab[dentro], axis=0)
    desv = float(lab[dentro][:, 2].std())
    return f, h, desv


def es_agujero_real(lab, mask_comp, f, h, k=None):
    """True si por el hueco SE VE EL MESON (metodo de `agujero_de_verdad.py`).

    Se erosiona el componente para quedarse con su NUCLEO --en uno pequenio el
    contorno es casi todo, y el contorno de un agujero real esta mezclado con
    hoja-- y el nucleo se compara con las dos referencias de la MISMA foto. Un
    agujero de verdad se parece al meson; un pliegue, a la lamina.

    Devuelve None si el componente no sobrevive a la erosion: no se puede juzgar.
    """
    er = cv2.erode(mask_comp.astype(np.uint8),
                   disco(EROSION_NUCLEO if k is None else k))
    if er.sum() < 12:
        return None
    c = lab[er > 0].reshape(-1, 3).mean(0)
    d_mesa = np.sqrt(((c[0] - f[0]) / 2.0) ** 2 + (c[1] - f[1]) ** 2
                     + (c[2] - f[2]) ** 2)
    d_hoja = np.sqrt(((c[0] - h[0]) / 2.0) ** 2 + (c[1] - h[1]) ** 2
                     + (c[2] - h[2]) ** 2)
    return bool(d_mesa < d_hoja)


def detecta(rgb, m, ac_px, k_disp=None, frac_fondo=None, tol_fondo=None,
            agujero_real=None, excluir=None, grosor_min=None):
    """(agujeros, roturas, marca, diagnostico). Mascaras booleanas."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 1] -= 128
    lab[:, :, 2] -= 128
    lado_eq = float(np.sqrt(max(1.0, float(m.sum()))))
    r = max(2, int(0.02 * lado_eq))

    f, h, desv_b = referencias(lab, m, r)

    # distancia en Lab con la luminancia a mitad de peso: dentro de una misma hoja
    # la sombra mueve L varios puntos sin que deje de ser hoja
    def dist(ref):
        d = lab - ref
        return np.sqrt((d[:, :, 0] / 2.0) ** 2 + d[:, :, 1] ** 2 + d[:, :, 2] ** 2)

    margen = float(np.clip(desv_b, MARGEN_B, 15.0))

    marca = mascara_marca(rgb)
    marca_ancha = cv2.dilate(marca.astype(np.uint8), disco(max(2, r // 2))) > 0

    d_h = dist(h)
    cand = (m & (dist(f) < d_h)
            & (lab[:, :, 2] < h[2] - margen)      # menos calido que la hoja
            & (lab[:, :, 0] < h[0] + 15)          # no es un brillo especular
            & ~marca_ancha)                       # no es el circulo del marcador

    # LO QUE OTRO DETECTOR YA EXPLICO NO ES UN AGUJERO. (2026-09-19)
    # Era el pendiente "EL HALLAZGO GORDO: EL DETECTOR DE AGUJEROS CUENTA MANCHAS"
    # (PENDIENTES, 15/09), que quedo medido y sin hacer. Confirmado el 19/09 sobre
    # las 217 hojas del experto: correlacion 0,45 entre pizquitas de sudado y agujeros
    # marcados, y la BANDA --que por su regla no tiene agujeros-- es la que mas
    # marca (36 por hoja, contra 16 de la XR derecha, que si los tiene). Una
    # pizquita verde es una manchita oscura y el detector la contaba como hueco,
    # asi que una banda le salia mas danada que una XL. `excluir` lleva la union de
    # las pizquitas del sudado y de las manchas verdes y blancas.
    # NO se excluyen las manchas NEGRAS a proposito: un agujero sobre mesa oscura
    # se ve negro, y excluirlas se llevaria agujeros de verdad.
    if excluir is not None:
        cand &= ~excluir

    # ...y, si K_DISP esta puesto, que el hueco se salga del color de la PROPIA
    # hoja. La escala es la mediana y la MAD de la distancia al color de hoja
    # medidas dentro de la hoja: los agujeros son una fraccion pequena del area,
    # asi que ninguna de las dos se les va detras.
    k = K_DISP if k_disp is None else k_disp
    if k > 0:
        dentro = cv2.erode(m.astype(np.uint8), disco(r)) > 0
        if dentro.sum() < 200:
            dentro = m
        dd = d_h[dentro]
        med = float(np.median(dd))
        mad = float(np.median(np.abs(dd - med)))
        cand &= d_h > med + k * 1.4826 * mad
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, disco(1))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, disco(max(1, r // 3))) > 0

    # ...y el criterio de la mesa, componente a componente: la mediana de color
    # del candidato tiene que parecerse a la mesa, no solo diferenciarse de la
    # hoja. Se mide con la MISMA distancia de arriba y en fraccion de la
    # separacion hoja-mesa de esta foto, para que no dependa de la mesa.
    # UN HILO NO ES UN AGUJERO (2026-09-19). Puramente geometrico: no mira color,
    # asi que vale igual con mesa azul o negra y con Habano o Connecticut.
    gr = GROSOR_MIN if grosor_min is None else grosor_min
    if gr > 0 and cand.any():
        n_et, et = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        vale = np.zeros(n_et, bool)
        for e in range(1, n_et):
            sel = (et == e).astype(np.uint8)
            vale[e] = cv2.erode(sel, disco(gr)).sum() >= 4
        cand = vale[et]

    real = AGUJERO_REAL if agujero_real is None else agujero_real
    if real and cand.any():
        n_et, et = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        vale = np.zeros(n_et, bool)
        for e in range(1, n_et):
            vale[e] = bool(es_agujero_real(lab, et == e, f, h))
        cand = vale[et]

    tol = TOL_FONDO if tol_fondo is None else tol_fondo
    if tol > 0 and cand.any():
        fuera_an = ((cv2.dilate(m.astype(np.uint8), disco(3 * r)) > 0)
                    & ~(cv2.dilate(m.astype(np.uint8), disco(r)) > 0))
        if fuera_an.sum() < 200:
            fuera_an = ~m
        d_fondo = dist(f)[fuera_an]
        med_f = float(np.median(d_fondo))
        mad_f = float(np.median(np.abs(d_fondo - med_f)))
        ruido = med_f + 1.4826 * mad_f
        n_et, et = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        vale = np.zeros(n_et, bool)
        for e in range(1, n_et):
            sel = et == e
            if sel.sum() < MIN_PX:
                continue
            c = np.median(lab[sel], axis=0)
            d_f = float(np.sqrt(((c[0] - f[0]) / 2.0) ** 2 + (c[1] - f[1]) ** 2
                                + (c[2] - f[2]) ** 2))
            vale[e] = d_f < tol * ruido
        cand = vale[et]

    fr = FRAC_FONDO if frac_fondo is None else frac_fondo
    if fr > 0 and cand.any():
        sep = float(np.sqrt(((f[0] - h[0]) / 2.0) ** 2 + (f[1] - h[1]) ** 2
                            + (f[2] - h[2]) ** 2))
        n_et, et = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        vale = np.zeros(n_et, bool)
        for e in range(1, n_et):
            sel = et == e
            if sel.sum() < MIN_PX:
                continue
            c = np.median(lab[sel], axis=0)
            d_f = float(np.sqrt(((c[0] - f[0]) / 2.0) ** 2 + (c[1] - f[1]) ** 2
                                + (c[2] - f[2]) ** 2))
            vale[e] = d_f < fr * sep
        cand = vale[et]

    R = max(3, int(R_ROTURA * ac_px)) if ac_px else max(3, int(0.05 * lado_eq))
    cerr = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disco(R)) > 0
    rotura = cerr & ~m & ~marca_ancha

    return cand, rotura, marca, dict(fondo=f, hoja=h, margen=margen, r=r, R=R,
                                     k_disp=k, frac_fondo=fr,
                                     tol_fondo=tol, agujero_real=real)


def gruesas(comps, mask, grosor_px):
    """Se queda con las componentes que PENETRAN, no con las lascas del contorno.

    La envolvente convexa de una hoja no coincide con su contorno: la hoja es
    ondulada y el casco corta las curvas, dejando una lasca fina a lo largo de
    todo el borde. Medido en 20260828_200025952: la bahia que falta tiene ~0.5
    pulgadas de espesor y las lascas 0.02-0.06. Se separan por el maximo de la
    transformada de distancia, que es el espesor real de cada trozo.
    """
    if not comps:
        return comps
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return [c for c in comps if d[c["mask"]].max() >= grosor_px]


def componentes(mask, min_px):
    n, etq, est, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for j in range(1, n):
        area = int(est[j, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        out.append(dict(area=area, cx=float(cen[j][0]), cy=float(cen[j][1]),
                        mask=(etq == j)))
    return out


CAMPOS_FOTO = ["ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id",
               "pulg_por_px", "escala", "base_dudosa", "toca_borde",
               "n_agujeros", "n_roturas", "n_defectos",
               "area_def_pulg2", "frac_area_def", "mayor_pulg2", "mayor_tipo",
               "mayor_zona", "mayor_dist_base_pulg", "mayor_diam_pulg",
               "n_util", "area_util_pulg2", "area_util_izq_pulg2",
               "area_util_der_pulg2", "n_util_izq", "n_util_der",
               "area_base_pulg2", "area_punta_pulg2", "area_vena_pulg2",
               "n_mordidas", "area_mordida_pulg2", "area_mordida_izq_pulg2",
               "area_mordida_der_pulg2", "mayor_mordida_pulg2",
               "pen_mordida_izq_pulg", "pen_mordida_der_pulg",
               "pen_mordida_pulg", "asim_pen_mordida",
               "n_en_marca", "n_marcas"]

CAMPOS_DET = ["ruta", "carpeta", "clase", "hoja_id", "tipo", "area_px",
              "area_pulg2", "diam_pulg", "zona", "dist_base_pulg", "s_pulg",
              "pen_pulg", "en_marca"]


def agrega(dets, area_hoja, pulg_px, base):
    """Fila por foto a partir de sus defectos.

    `mayor_*` describe SOLO agujeros y roturas, nunca una mordida: son cosas
    distintas y mezclarlas contamino los rasgos del clasificador (5 clases
    81.0 -> 79.6 al colarse mordidas en `mayor_pulg2`). La mordida tiene su
    propia columna.
    """
    k2 = (pulg_px ** 2) if pulg_px else None
    hueco = [d for d in dets if d["tipo"] in ("agujero", "rotura")]
    mord = [d for d in dets if d["tipo"] == "mordida"]
    util = [d for d in hueco if d["zona"].startswith("util")]

    def suma(lista, cond=lambda d: True):
        return sum(d["area_px"] for d in lista if cond(d))

    a_tot = suma(hueco)
    mayor = max(hueco, key=lambda d: d["area_px"]) if hueco else None
    may_mo = max(mord, key=lambda d: d["area_px"]) if mord else None
    f = dict(base)
    f.update(
        n_agujeros=sum(1 for d in dets if d["tipo"] == "agujero"),
        n_roturas=sum(1 for d in dets if d["tipo"] == "rotura"),
        n_defectos=len(hueco),
        area_def_pulg2=round(a_tot * k2, 4) if k2 else "",
        frac_area_def=round(a_tot / area_hoja, 5) if area_hoja else "",
        mayor_pulg2=(round(mayor["area_px"] * k2, 4) if (mayor and k2) else ""),
        mayor_tipo=mayor["tipo"] if mayor else "",
        mayor_zona=mayor["zona"] if mayor else "",
        mayor_dist_base_pulg=mayor["dist_base_pulg"] if mayor else "",
        mayor_diam_pulg=mayor["diam_pulg"] if mayor else "",
        n_util=len(util),
        area_util_pulg2=(round(suma(util) * k2, 4) if k2 else ""),
        area_util_izq_pulg2=(round(suma(util, lambda d: d["zona"] == "util_izq") * k2, 4)
                             if k2 else ""),
        area_util_der_pulg2=(round(suma(util, lambda d: d["zona"] == "util_der") * k2, 4)
                             if k2 else ""),
        n_util_izq=sum(1 for d in util if d["zona"] == "util_izq"),
        n_util_der=sum(1 for d in util if d["zona"] == "util_der"),
        area_base_pulg2=(round(suma(hueco, lambda d: d["zona"] == "base") * k2, 4)
                         if k2 else ""),
        area_punta_pulg2=(round(suma(hueco, lambda d: d["zona"] == "punta") * k2, 4)
                          if k2 else ""),
        area_vena_pulg2=(round(suma(hueco, lambda d: d["zona"] == "vena") * k2, 4)
                         if k2 else ""),
        n_mordidas=len(mord),
        area_mordida_pulg2=(round(suma(mord) * k2, 4) if k2 else ""),
        area_mordida_izq_pulg2=(round(suma(mord, lambda d: d["zona"] == "util_izq") * k2, 4)
                                if k2 else ""),
        area_mordida_der_pulg2=(round(suma(mord, lambda d: d["zona"] == "util_der") * k2, 4)
                                if k2 else ""),
        mayor_mordida_pulg2=(round(may_mo["area_px"] * k2, 4) if (may_mo and k2) else ""),
        n_en_marca=sum(int(d["en_marca"]) for d in dets),
    )
    # penetracion por lado (§40.6). `or 0` porque el detalle escrito antes del
    # 2026-09-03 no trae la columna, y reagrupa() lee ese CSV.
    def pen(cond):
        v = [float(d.get("pen_pulg") or 0) for d in mord if cond(d)]
        return round(max(v), 3) if v else 0.0

    pi, pd = pen(lambda d: d["zona"] == "util_izq"), pen(lambda d: d["zona"] == "util_der")
    f.update(pen_mordida_izq_pulg=pi, pen_mordida_der_pulg=pd,
             pen_mordida_pulg=max(pi, pd), asim_pen_mordida=round(pi - pd, 3))
    return f


def reagrupa():
    """Rehace out/agujeros.csv desde el detalle, sin volver a mirar las fotos.

    Las reglas de agregacion cambian mas veces que la deteccion, y volver a
    decodificar 1177 fotos por cambiar una suma cuesta 20 minutos.
    """
    fotos = list(csv.DictReader(open(OUT / "agujeros.csv", encoding="utf-8")))
    det = defaultdict(list)
    for d in csv.DictReader(open(OUT / "agujeros_detalle.csv", encoding="utf-8")):
        d["area_px"] = int(d["area_px"])
        det[d["ruta"]].append(d)
    salida = []
    for r in fotos:
        ds = det.get(r["ruta"], [])
        pulg_px = float(r["pulg_por_px"]) if r["pulg_por_px"] else None
        # el area de hoja se recupera exacta de la fila vieja: a_tot / frac
        a_viejo = sum(d["area_px"] for d in ds)
        frac = float(r["frac_area_def"]) if r["frac_area_def"] else 0.0
        area_hoja = (a_viejo / frac) if frac > 0 else 0.0
        base = {k: r[k] for k in ("ruta", "carpeta", "archivo", "variedad", "clase",
                                  "hoja_id", "pulg_por_px", "escala", "base_dudosa",
                                  "toca_borde", "n_marcas")}
        salida.append(agrega(ds, area_hoja, pulg_px, base))
    with open(OUT / "agujeros.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_FOTO, extrasaction="ignore")
        w.writeheader(); w.writerows(salida)
    print("reagrupadas {} fotos".format(len(salida)))
    informe(OUT / "agujeros.csv", OUT / "agujeros_detalle.csv", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_nuevas", nargs="?", type=int, default=0,
                    help="procesa N fotos nuevas y para (reanudable)")
    ap.add_argument("--muestra", type=int, default=0, help="N fotos por carpeta")
    ap.add_argument("--qc", type=int, default=0)
    ap.add_argument("--solo", default="")
    ap.add_argument("--frac-fondo", type=float, default=None,
                    help="prueba el criterio de la mesa (ver FRAC_FONDO). Con "
                         "--muestra escribe en agujeros_parcial.csv y no toca "
                         "produccion.")
    ap.add_argument("--grosor-min", type=int, default=None,
                    help="px de radio: una marca que no sobrevive a ese "
                         "adelgazamiento no es un agujero, es un doblez")
    ap.add_argument("--agujero-real", type=int, default=None,
                    help="1 = solo cuenta agujeros donde se ve el meson por el "
                         "hueco (nucleo erosionado, metodo de agujero_de_verdad.py)")
    ap.add_argument("--tol-fondo", type=float, default=None,
                    help="prueba el criterio del ruido de la mesa (ver TOL_FONDO)")
    ap.add_argument("--k-disp", type=float, default=None,
                    help="prueba el criterio de dispersion (ver K_DISP)")
    ap.add_argument("--reagrupa", action="store_true",
                    help="rehace agujeros.csv desde el detalle, sin decodificar fotos")
    a = ap.parse_args()
    if a.reagrupa:
        return reagrupa()

    seg = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "segmentacion.csv", encoding="utf-8"))}
    man = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "manifiesto_limpio.csv", encoding="utf-8"))}
    end = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "enderezado.csv", encoding="utf-8"))}
    zon = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "zonas.csv", encoding="utf-8"))}
    # ancho de cinta EN PIXELES DE LA MASCARA (ver `ancho_cinta_por_carpeta`)
    ancho_cinta = ancho_cinta_por_carpeta(OUT)

    rutas = sorted(r for r in seg if r in man)
    if a.solo:
        rutas = [r for r in rutas if a.solo in r]
    global FRAC_FONDO, K_DISP, TOL_FONDO, AGUJERO_REAL, GROSOR_MIN
    if a.frac_fondo is not None:
        FRAC_FONDO = a.frac_fondo
        print("*** FRAC_FONDO = %.2f (criterio de la mesa ENCENDIDO)" % FRAC_FONDO)
    if a.grosor_min is not None:
        GROSOR_MIN = a.grosor_min
        print("*** GROSOR_MIN = %i (un hilo no es un agujero)" % GROSOR_MIN)
    if a.agujero_real is not None:
        AGUJERO_REAL = a.agujero_real
        print("*** AGUJERO_REAL = %i (solo huecos donde se ve el meson)"
              % AGUJERO_REAL)
    if a.tol_fondo is not None:
        TOL_FONDO = a.tol_fondo
        print("*** TOL_FONDO = %.2f (criterio del ruido de la mesa)" % TOL_FONDO)
    if a.k_disp is not None:
        K_DISP = a.k_disp
        print("*** K_DISP = %.2f" % K_DISP)

    parcial = bool(a.muestra or a.solo)
    if a.muestra:
        porc = defaultdict(list)
        for r in rutas:
            porc["/".join(r.split("/")[:2])].append(r)
        rutas = []
        for c in sorted(porc):
            v = porc[c]
            idx = np.linspace(0, len(v) - 1, min(a.muestra, len(v))).astype(int)
            rutas += [v[i] for i in sorted(set(idx))]

    f_foto = OUT / ("agujeros_parcial.csv" if parcial else "agujeros.csv")
    f_det = OUT / ("agujeros_detalle_parcial.csv" if parcial else "agujeros_detalle.csv")
    hechas = set()
    if not parcial and f_foto.exists():
        hechas = {r["ruta"] for r in
                  csv.DictReader(open(f_foto, encoding="utf-8")) if r.get("ruta")}
        print("ya calculadas: {}".format(len(hechas)))
    pend = [r for r in rutas if r not in hechas]
    if a.max_nuevas:
        pend = pend[:a.max_nuevas]
    print("fotos: {}   pendientes en esta corrida: {}".format(len(rutas), len(pend)))

    qc_dir = OUT / "qc_agujeros"
    if a.qc and pend:
        qc_dir.mkdir(parents=True, exist_ok=True)
        idx = np.linspace(0, len(pend) - 1, min(a.qc, len(pend))).astype(int)
        qc = {pend[i] for i in idx}
    else:
        qc = set()

    nuevo_foto = parcial or not f_foto.exists()
    fh1 = open(f_foto, "w" if nuevo_foto else "a", newline="", encoding="utf-8")
    w1 = csv.DictWriter(fh1, fieldnames=CAMPOS_FOTO, extrasaction="ignore")
    if nuevo_foto:
        w1.writeheader()
    nuevo_det = parcial or not f_det.exists()
    fh2 = open(f_det, "w" if nuevo_det else "a", newline="", encoding="utf-8")
    w2 = csv.DictWriter(fh2, fieldnames=CAMPOS_DET, extrasaction="ignore")
    if nuevo_det:
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

        # 1. mismo marco que zonas.py: enderezar por la cinta y voltear
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

        # 2. detectar
        ag, ro, marca, dg = detecta(rgb, m, ac)
        min_px = MIN_PX
        if pulg_px:
            min_px = max(MIN_PX, int(MIN_PULG2 / (pulg_px ** 2)))
        c_ag = componentes(ag, min_px)
        c_ro = componentes(ro, min_px)

        # 3. marco de la hoja: eje mayor con la base SIEMPRE al final (zonas.py)
        e, perp, mu, t, s, anch = perfil(m)
        base_al_final, _ = extremo_base(anch)
        if not base_al_final:
            e, perp, t, s = -e, -perp, -t, -s
        t0, t1 = float(t.min()), float(t.max())
        largo = t1 - t0
        # bandas permisibles medidas sobre las dos hojas que marco el experto (zonas.py)
        lam0, lam1 = lamina(t, anch)
        punta_px, base_px, vena_px, _borde = bandas(lam1 - lam0, pulg_px, ac)

        # 3b. MORDIDAS: lo que le falta a la hoja respecto de su propia envolvente
        #     convexa, contado solo en la zona util (fuera de base y punta).
        cnts_h, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        casco = np.zeros(m.shape, np.uint8)
        if cnts_h:
            cv2.fillPoly(casco, [cv2.convexHull(max(cnts_h, key=cv2.contourArea))], 1)
        falta = (casco > 0) & ~m
        if falta.any():
            yy2, xx2 = np.nonzero(falta)
            pp = np.stack([xx2, yy2], 1).astype(np.float32) - mu
            tt = pp @ e
            fuera_util = (tt > lam1 - base_px) | (tt < lam0 + punta_px)
            falta[yy2[fuera_util], xx2[fuera_util]] = False
        g_px = (GROSOR_MORDIDA / pulg_px) if pulg_px else 0.02 * float(max(m.shape))
        c_mo = gruesas(componentes(falta, 4 * min_px), falta, g_px)

        # 3c. PENETRACION de cada mordida: cuanto se mete el dano hacia dentro.
        #     Es la medida que pide la regla de la chaveta (§40.6): de un
        #     mordisco de borde no importa el AREA sino cuanto entra, porque el
        #     corte se adapta al dano ("se corta hasta que se termina lo
        #     arrugado") y lo que descalifica es que el dano llegue mas adentro
        #     de lo que la chaveta se puede llevar.
        #     Se mide como la distancia del pixel mas hondo de la mordida al
        #     exterior de la envolvente convexa, o sea al contorno que tendria
        #     la hoja entera. Distingue lo que el AREA confundia (§35): una hoja
        #     arqueada deja un deficit enorme pero pegado al borde -- penetracion
        #     pequena -- mientras que un pedazo que falta entra hacia el centro.
        d_casco = cv2.distanceTransform(casco, cv2.DIST_L2, 3)

        # interior de los circulos de marcador, para la comprobacion
        marca_llena = rellena(marca) if marca.any() else marca
        n_marcas = len(componentes(marca, 30))

        det = []
        for tipo, lista in (("agujero", c_ag), ("rotura", c_ro), ("mordida", c_mo)):
            for c in lista:
                pxy = np.array([c["cx"], c["cy"]], np.float32) - mu
                tc, sc = float(pxy @ e), float(pxy @ perp)
                if tc > lam1 - base_px:
                    zona = "base"
                elif tc < lam0 + punta_px:
                    zona = "punta"
                elif abs(sc) < vena_px / 2.0:
                    zona = "vena"
                else:
                    zona = "util_izq" if sc > 0 else "util_der"
                ap2 = c["area"] * (pulg_px ** 2) if pulg_px else None
                pen = (float(d_casco[c["mask"]].max()) if tipo == "mordida" else 0.0)
                det.append(dict(
                    ruta=ruta, carpeta=carp, clase=mr["clase"], hoja_id=mr["hoja_id"],
                    tipo=tipo, area_px=c["area"],
                    area_pulg2=round(ap2, 4) if ap2 is not None else "",
                    diam_pulg=(round(float(2 * np.sqrt(ap2 / np.pi)), 3)
                               if ap2 is not None else ""),
                    zona=zona,
                    dist_base_pulg=(round((t1 - tc) * pulg_px, 3) if pulg_px else ""),
                    s_pulg=(round(sc * pulg_px, 3) if pulg_px else ""),
                    pen_pulg=(round(pen * pulg_px, 3) if pulg_px else ""),
                    en_marca=int(bool((c["mask"] & marca_llena).any())),
                    _area=c["area"], _zona=zona, _tipo=tipo, _mask=c["mask"]))
        for d in det:
            w2.writerow(d)

        base = dict(ruta=ruta, carpeta=carp, archivo=r["archivo"],
                    variedad=mr["variedad"], clase=mr["clase"], hoja_id=mr["hoja_id"],
                    pulg_por_px=round(pulg_px, 6) if pulg_px else "",
                    escala="cinta" if pulg_px else "sin",
                    base_dudosa=zon.get(ruta, {}).get("base_dudosa", ""),
                    toca_borde=r["toca_borde"], n_marcas=n_marcas)
        fila = agrega(det, float(m.sum()), pulg_px, base)
        w1.writerow(fila)
        fh1.flush()
        fh2.flush()

        if ruta in qc:
            vis = rgb.copy()
            fuera = ~m
            vis[fuera] = (0.40 * vis[fuera] +
                          0.60 * np.array([25, 25, 25])).astype(np.uint8)
            cap = np.zeros(m.shape, bool)
            cro = np.zeros(m.shape, bool)
            cmo = np.zeros(m.shape, bool)
            for d in det:
                {"agujero": cap, "rotura": cro, "mordida": cmo}[d["_tipo"]][d["_mask"]] = True
            vis[cmo] = (120, 80, 255)         # mordida  violeta
            vis[cro] = (255, 170, 30)         # rotura   naranja
            vis[cap] = (255, 40, 40)          # agujero  rojo
            vis[marca] = (60, 255, 60)        # marcador verde
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 0, 255), 2)
            # linea de la zona de base, para ver contra que se compara
            pb = mu + e * (lam1 - base_px)
            d2 = perp * float(max(m.shape))
            cv2.line(vis, tuple((pb - d2).astype(int)), tuple((pb + d2).astype(int)),
                     (0, 220, 220), 2)
            txt = "{}  ag={} rot={} mor={}".format(mr["clase"], len(c_ag),
                                                    len(c_ro), len(c_mo))
            if fila["mayor_pulg2"] != "":
                txt += "  mayor={}p2 {} a {}pulg de la base".format(
                    fila["mayor_pulg2"], fila["mayor_zona"],
                    fila["mayor_dist_base_pulg"])
            cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            k = max(1, vis.shape[0] // 800)
            cv2.imwrite(str(qc_dir / (Path(r["archivo"]).stem.replace(" ", "_") + ".jpg")),
                        cv2.cvtColor(vis[::k, ::k], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        if i % 25 == 0:
            print("  {}/{}".format(i, len(pend)))

    fh1.close()
    fh2.close()
    print("-> {}".format(f_foto))
    print("-> {}".format(f_det))
    informe(f_foto, f_det, parcial)


def informe(f_foto, f_det, parcial):
    filas = list(csv.DictReader(open(f_foto, encoding="utf-8")))
    det = list(csv.DictReader(open(f_det, encoding="utf-8")))
    if not filas:
        return
    L = []

    def p(s=""):
        L.append(s)
        print(s)

    p("")
    p("== defectos por carpeta ==")
    p("  {:44s} {:>5s} {:>7s} {:>7s} {:>10s} {:>10s} {:>10s} {:>7s}".format(
        "carpeta", "n", "ag/med", "rot/med", "mayor med", "mayor p90",
        "mordida", "% con"))
    p("  " + "-" * 96)
    por = defaultdict(list)
    for r in filas:
        por[r["carpeta"]].append(r)
    for c in sorted(por):
        v = por[c]
        ag = np.array([int(x["n_agujeros"]) for x in v])
        ro = np.array([int(x["n_roturas"]) for x in v])
        may = np.array([float(x["mayor_pulg2"]) if x["mayor_pulg2"] else 0.0
                        for x in v])
        mor = np.array([float(x.get("area_mordida_pulg2") or 0.0) for x in v])
        p("  {:44s} {:5d} {:7.1f} {:7.1f} {:10.3f} {:10.3f} {:10.3f} {:6.0f}%".format(
            c[:44], len(v), float(np.median(ag)), float(np.median(ro)),
            float(np.median(may)), float(np.percentile(may, 90)),
            float(np.median(mor)), 100.0 * float((may > 0).mean())))

    p("")
    p("== zona del defecto mayor ==")
    zs = ["util_izq", "util_der", "base", "punta", "vena"]
    p("  {:44s} {:>9s} {:>9s} {:>7s} {:>7s} {:>7s}".format("carpeta", *zs))
    for c in sorted(por):
        cnt = defaultdict(int)
        for x in por[c]:
            if x["mayor_zona"]:
                cnt[x["mayor_zona"]] += 1
        p("  {:44s} {:9d} {:9d} {:7d} {:7d} {:7d}".format(
            c[:44], *[cnt[z] for z in zs]))

    marc = [r for r in filas if int(r["n_marcas"] or 0) > 0]
    if marc:
        p("")
        p("== comprobacion contra los circulos de marcador ==")
        p("  fotos con circulo: {}   circulos: {}   defectos dentro de un circulo: {}"
          .format(len(marc), sum(int(r["n_marcas"]) for r in marc),
                  sum(int(r["n_en_marca"]) for r in marc)))
        for r in marc:
            p("    {:50s} circulos={} defectos={} dentro={}".format(
                r["archivo"][:50], r["n_marcas"], r["n_defectos"], r["n_en_marca"]))

    p("")
    p("== tamano de los defectos (pulg2) ==")
    ar = [float(d["area_pulg2"]) for d in det if d["area_pulg2"]]
    if ar:
        ar = np.array(ar)
        p("  n={}  min={:.3f}  mediana={:.3f}  p90={:.3f}  max={:.3f}".format(
            len(ar), float(ar.min()), float(np.median(ar)),
            float(np.percentile(ar, 90)), float(ar.max())))
    f = OUT / ("AGUJEROS_INFORME_parcial.txt" if parcial else "AGUJEROS_INFORME.txt")
    f.write_text("\n".join(L), encoding="utf-8")
    print("")
    print("-> {}".format(f))


if __name__ == "__main__":
    main()
