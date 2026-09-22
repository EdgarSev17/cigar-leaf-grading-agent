r"""
LOS RASGOS DE UNA FOTO SUELTA, SIN PASAR POR NINGUN CSV. (2026-09-05)

EL HUECO QUE TAPA ESTE MODULO
-----------------------------
PROGRESO 54.4: "no existe ningun camino de una foto nueva a una clase". Y no es
por falta de medidas -- estan todas-- sino porque **cada medida vive en un guion
que recorre el conjunto entero y escribe un CSV**. `clasificador.py` lee
`segmentacion.csv`, `enderezado.csv`, `zonas.csv`, `agujeros.csv`,
`manchas.csv`... y ninguno de esos existe para una hoja que se acaba de
fotografiar.

Este modulo hace la misma cadena **en memoria, para una imagen**:

    foto -> mascara -> enderezado por la cinta -> escala -> zonas
         -> color y contrastes -> agujeros y mordidas -> manchas -> punta

LA REGLA QUE SE HA SEGUIDO AL ESCRIBIRLO
----------------------------------------
**No se reimplementa ni una sola deteccion.** Todas las funciones se IMPORTAN de
los guiones que produjeron las cifras del proyecto (`segmentacion.mascara_hoja`,
`agujeros.detecta`, `manchas.detecta`, `zonas.perfil`...). Lo unico que se
escribe aqui es el ORDEN en que se llaman, que hasta ahora vivia repartido entre
cuatro `main()`.

Y eso tiene una consecuencia que hay que comprobar, no suponer: si el orden
copiado se desvia del original, los numeros cambian y nadie se entera.
`verifica_foto.py` compara, hoja por hoja, lo que sale de aqui contra lo que hay
en los CSV. **Esa comprobacion es parte del modulo, no un extra.**

LA DIFERENCIA HONESTA CON LA TUBERIA: LA ESCALA
-----------------------------------------------
La tuberia usa el ancho de cinta **mediano de la carpeta** (`zonas.
ancho_cinta_por_carpeta`), o sea de las ~60 fotos de esa sesion. Una foto suelta
no tiene carpeta: hay que medir la cinta **en ella misma**, que es mas ruidoso.

Se soportan las dos cosas a proposito:
  - `ac_px=<numero>`  reproduce exactamente la tuberia (para verificar);
  - `ac_px=None`      mide la cinta en la propia foto (lo que pasaria en planta).

La diferencia entre las dos NO es un detalle: la escala entra en las pulgadas, en
el tamano minimo de defecto y en el ancho de las zonas. Cuanto cuesta se mide en
`verifica_foto.py`, y sale en el informe.

Uso como modulo:
    from rasgos_foto import mide
    r = mide(Path("foto.jpg"))           # r["rasgos"], r["diag"], r["defectos"]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()
sys.path.insert(0, str(Path(__file__).parent))

from segmentacion import mascara_hoja, mascara_marca, metricas          # noqa: E402
from enderezado import busca_cinta as cinta_angulo, canon               # noqa: E402
from escala_cinta import (busca_cinta as cinta_escala,                  # noqa: E402
                          marcas_en_cinta)
from zonas import perfil, lamina, bandas, extremo_base                  # noqa: E402
from clasificador import medidas, MED, PULG_POR_ANCHO_CINTA             # noqa: E402
import agujeros as AG                                                    # noqa: E402
import manchas as MA                                                     # noqa: E402

LADO = 1024          # el de segmentacion.py y el de todas las mascaras
LADO_CINTA = 1800    # el de escala_cinta.py: las graduaciones son finas
N_TRAMOS = 60
DIST_PUNTA = [0.25, 0.5, 1.0, 2.0]     # pulgadas; las que usa el modelo de variedad


def carga(p, lado=LADO):
    """Igual que `segmentacion.carga`: el `draft` decodifica desde el DCT."""
    with Image.open(p) as im:
        try:
            im.draft("RGB", (im.width // 4, im.height // 4))
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((lado, lado), Image.LANCZOS)
        return np.asarray(im)


def rota_exp(img, grados, nearest=False):
    """Rotacion que AGRANDA el lienzo. La de `clasificador.py` y `agujeros.py`."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), grados, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def rota_fija(img, grados, nearest=False):
    """Rotacion que NO agranda. Es la que uso `forma_punta.py`, y se conserva
    aqui por eso: los rasgos de punta del modelo salieron con esta."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), grados, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def ancho_cinta_de_la_foto(ruta, alto_masc):
    """Ancho de la cinta EN PIXELES DE LA MASCARA, medido en esta misma foto.

    `escala_cinta.py` trabaja a 1800 px y las mascaras a 1024: la conversion es
    la misma que hace `zonas.ancho_cinta_por_carpeta`, y olvidarla fue el error
    que durante semanas dejo todas las pulgadas del proyecto al 57 % de su valor
    (PROGRESO, nota de esa funcion).
    """
    rgb = carga(ruta, LADO_CINTA)
    etq, j, L = cinta_escala(rgb)
    if j is None:
        return None, "sin_cinta"
    pos, base, ancho, largo = marcas_en_cinta(etq, j, L)
    if not ancho or ancho <= 0:
        return None, "cinta_sin_ancho"
    return float(ancho) * (alto_masc / float(rgb.shape[0])), "medida_en_la_foto"


def anchuras_punta(m, pulg_px):
    """Anchura de la lamina a varias distancias de la PUNTA, en pulgadas.

    Copiada de `forma_punta.anchuras` --no importada-- porque ese guion ejecuta
    su recorrido entero al importarlo. Es la unica funcion duplicada del modulo,
    y `verifica_foto.py` comprueba que da lo mismo.
    """
    if not pulg_px:
        return {}
    e, perp, mu, t, s, anch = perfil(m)
    l0, l1 = lamina(t, anch)
    if not (l1 > l0):
        return {}
    dentro = (t >= l0) & (t <= l1)
    tt, ss = t[dentro], s[dentro]
    if len(tt) < 500:
        return {}
    out = {}
    for d in DIST_PUNTA:
        c = l0 + d / pulg_px
        h = max(2.0, 0.06 / pulg_px)
        sel = np.abs(tt - c) <= h
        out["punta_{:g}".format(d)] = (float(ss[sel].max() - ss[sel].min()) * pulg_px
                                       if sel.sum() >= 20 else np.nan)
    return out


def mide(ruta, ac_px=None, sin_cinta=False):
    """Todos los rasgos de una foto. Devuelve dict con:

        rasgos    nombre -> numero, con los nombres de los CSV del proyecto
        diag      como se llego a ellos (escala, giro, avisos)
        defectos  la lista de agujeros / roturas / mordidas / manchas, con su
                  zona y su tamano. Es la materia prima del MOTIVO.
    """
    ruta = Path(ruta)
    diag = dict(archivo=ruta.name, avisos=[])

    # ---- 1. la hoja -------------------------------------------------------
    rgb0 = carga(ruta, LADO)
    marca = mascara_marca(rgb0)
    hoja, ncomp, resto = mascara_hoja(rgb0, marca)
    if hoja.sum() < 500:
        return dict(rasgos={}, diag=dict(diag, error="no se encontro hoja"),
                    defectos=[])
    met = metricas(hoja, marca, rgb0.shape[:2])
    diag.update(frac_cuadro=met["frac_cuadro"], solidez=met["solidez"],
                toca_borde=met["toca_borde"], area_px=met["area_px"])
    if met["toca_borde"]:
        diag["avisos"].append(
            "la hoja TOCA EL BORDE del cuadro: el largo, el ancho y los "
            "contrastes entre mitades no son fiables (PROGRESO 3)")
    if met["p_marca_en_hoja"] > 0.001:
        diag["avisos"].append(
            "hay marcador azul sobre la hoja ({:.2%} del area): esta foto esta "
            "ANOTADA y no debe usarse como foto de hoja (PROGRESO 2)"
            .format(met["p_marca_en_hoja"]))

    # ---- 2. enderezar por la cinta ---------------------------------------
    #
    # MODO BANDA (2026-09-06). En la celda que quiere el experto la hoja llega sola
    # sobre una banda: NO hay cinta metrica en el cuadro. Con `sin_cinta=True` el
    # enderezado se hace con la propia hoja --su eje mayor-- y la escala tiene que
    # venir dada, que es justo lo que da una camara fija calibrada una vez (§60.4).
    # La izquierda y la derecha NO dependian de la cinta: salen de donde esta la
    # BASE, que se detecta en la mascara sola con un 97-98 % de acierto (§33).
    if sin_cinta:
        if ac_px is None:
            return dict(rasgos={}, defectos=[], diag=dict(
                diag, error="modo banda sin escala: hay que dar la calibracion"))
        # SI SE GIRA LA IMAGEN, Y ESTA ES LA SEGUNDA VERSION (medido el 06/09).
        #
        # Primero se enderezaba llevando el eje de la hoja a la vertical. Al ver
        # que girar la foto costaba clases, lo quite pensando que el marco de la
        # hoja hacia el giro innecesario. **Medido, y era peor**: sin enderezar,
        # la MISMA foto ya cambia el 10 % de las clases (90 % de coincidencia
        # contra el 100 % con enderezado). Asi que algo de la cadena si depende de
        # que la imagen venga alineada, y el enderezado se queda.
        #
        # Lo que si quedo medido, y es lo que hay que arreglar de verdad: la
        # perdida sigue al NUMERO DE INTERPOLACIONES, no al angulo (§65.6). Cada
        # giro emborrona y `textura` es la energia del gradiente. En la banda eso
        # se ataca subiendo resolucion, interpolando mejor que bilineal, o --lo
        # correcto-- midiendo la textura a una escala FISICA fija (mm por pixel)
        # en vez de en pixeles.
        e0 = perfil(hoja)[0]
        rot = round(90.0 - np.degrees(np.arctan2(float(e0[1]), float(e0[0]))), 2)
        mc, estado_c, voltea = None, "sin cinta (modo banda)", 0
        diag["modo"] = "banda"
    else:
        mc, ang, alarg, estado_c, cx, cy = cinta_angulo(rgb0)
        rot, voltea = 0.0, 0
    if mc is not None:
        _, rot = canon(ang)
        # REDONDEO A 2 DECIMALES, Y NO ES UN CAPRICHO (2026-09-05).
        # `enderezado.py` escribe `rotacion=round(rot, 2)` en el CSV, y TODOS los
        # guiones de la tuberia leen de ahi: las cifras del proyecto salieron con
        # el angulo redondeado. Si aqui se usa el angulo completo, la foto queda
        # girada 0.001 grados distinto, la interpolacion mueve unos pocos pixeles
        # y un defecto que estaba justo en el limite de 25 px cruza el umbral:
        # medido en 20260828_175653139, 7 agujeros con 0.70 y 6 con 0.7013.
        #
        # Lo que hay que llevarse de esto NO es el redondeo, es lo otro: **el
        # recuento de defectos pequenos es fragil**, porque hay componentes justo
        # en el minimo. Vale para el conteo; el area total apenas se mueve.
        rot = round(float(rot), 2)
        mrot = rota_exp(mc.astype(np.uint8) * 255, rot)
        hrot = rota_exp(hoja.astype(np.uint8) * 255, rot)
        if mrot.any() and hrot.any():
            voltea = int(np.nonzero(mrot)[1].mean() > np.nonzero(hrot)[1].mean())
    elif not sin_cinta:
        diag["avisos"].append(
            "no se ve la cinta metrica: sin escala no hay pulgadas, y el "
            "enderezado se queda como esta la foto (PROGRESO 17)")
    diag.update(estado_cinta=estado_c, rotacion=round(float(rot), 2),
                voltea_180=voltea)

    m = hoja
    rgb = rgb0
    if rot:
        m = rota_exp(m.astype(np.uint8), rot, True) > 0
        rgb = rota_exp(rgb, rot)
    if voltea:
        m = cv2.rotate(m.astype(np.uint8), cv2.ROTATE_180) > 0
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    if m.sum() < 500:
        return dict(rasgos={}, diag=dict(diag, error="la hoja se perdio al rotar"),
                    defectos=[])

    # ---- 3. la escala -----------------------------------------------------
    if sin_cinta:
        origen = "calibracion de la camara (modo banda)"
    elif ac_px is None:
        ac_px, origen = ancho_cinta_de_la_foto(ruta, rgb0.shape[0])
    else:
        origen = "dada (mediana de la carpeta, como la tuberia)"
    pulg_px = (PULG_POR_ANCHO_CINTA / ac_px) if ac_px else None
    diag.update(ancho_cinta_px=(round(ac_px, 1) if ac_px else None),
                origen_escala=origen,
                pulg_por_px=(round(pulg_px, 6) if pulg_px else None))

    # ---- 4. el marco de la hoja: eje, lamina y zonas ----------------------
    e, perp, mu, t, s, anch = perfil(m)
    base_al_final, margen = extremo_base(anch)
    if not base_al_final:
        e, perp, t, s = -e, -perp, -t, -s
    diag["margen_base"] = int(margen)
    if margen < 3:
        diag["avisos"].append(
            "no se distingue bien cual extremo es la base ({} de 60 tramos de "
            "margen): izquierda y derecha podrian estar cambiadas".format(margen))
    lam0, lam1 = lamina(t, anch)
    ac_para_bandas = ac_px if ac_px else 0.08 * float(t.max() - t.min())
    punta_px, base_px, vena_px, borde_px = bandas(lam1 - lam0, pulg_px,
                                                  ac_para_bandas)

    ys, xs = np.nonzero(m)
    util = ~((t > lam1 - base_px) | (t < lam0 + punta_px) |
             (np.abs(s) < vena_px / 2.0))
    M_util = np.zeros(m.shape, bool)
    M_izq = np.zeros(m.shape, bool)
    M_der = np.zeros(m.shape, bool)
    M_util[ys[util], xs[util]] = True
    sel = util & (s > 0)
    M_izq[ys[sel], xs[sel]] = True
    sel = util & (s < 0)
    M_der[ys[sel], xs[sel]] = True
    if M_izq.sum() < 200 or M_der.sum() < 200:
        return dict(rasgos={}, diag=dict(diag, error="zona util vacia"), defectos=[])

    # ---- 5. color, contrastes entre mitades y forma ----------------------
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ener = np.sqrt(gx * gx + gy * gy)

    mu_h = medidas(rgb, M_util, ener)
    mi = medidas(rgb, M_izq, ener)
    md = medidas(rgb, M_der, ener)
    escL = mu_h[1] if mu_h[1] == mu_h[1] and mu_h[1] > 1e-6 else 1.0
    escb = mu_h[4] if mu_h[4] == mu_h[4] and mu_h[4] > 1e-6 else 1.0
    lados = []
    for k, (xi, xd) in enumerate(zip(mi, md)):
        if k == 0:
            lados.append((xi - xd) / escL)
        elif k in (2, 3):
            lados.append((xi - xd) / escb)
        else:
            lados.append((xi - xd) / (abs(xi) + abs(xd) + 1e-6))
    lados += [abs(x) for x in lados]

    largo = float(t.max() - t.min())
    anchomax = float(anch.max())
    ac_f = ac_para_bandas
    forma = [(lam1 - lam0) / ac_f * PULG_POR_ANCHO_CINTA,
             anchomax / ac_f * PULG_POR_ANCHO_CINTA,
             anchomax / max(1.0, largo),
             met["solidez"], met["frac_cuadro"],
             float(m.sum()) / max(1.0, float(M_util.sum()))]
    fondo = medidas(rgb, ~m, ener)

    R = {}
    for k, v in zip(["hoja_" + x for x in MED], mu_h):
        R[k] = v
    for k, v in zip(["lado_" + x for x in MED] + ["lado_abs_" + x for x in MED],
                    lados):
        R[k] = v
    for k, v in zip(["largo_pulg", "ancho_pulg", "aspecto", "solidez", "cobertura",
                     "hoja_entre_util"], forma):
        R[k] = v
    for k, v in zip(["fondo_" + x for x in MED], fondo):
        R[k] = v

    # ---- 5-bis. LOS OTROS DETECTORES, ANTES DE LOS AGUJEROS (2026-09-19) --
    #     Era el pendiente "EL DETECTOR DE AGUJEROS CUENTA MANCHAS" del 15/09,
    #     confirmado el 19/09 sobre las 217 hojas del experto: la BANDA --que por su
    #     regla NO tiene agujeros, tiene pizquitas-- era la clase con mas agujeros
    #     marcados (36 por hoja contra 16 de la XR derecha, que si los tiene), y la
    #     correlacion entre pizquitas y agujeros marcados es 0,45. El detector de
    #     agujeros se estaba comiendo el sudado, y por eso banda y XL/XR se
    #     confundian desde el principio.
    #     Los tres detectores ya existian y corrian en cada hoja, pero NO se
    #     hablaban. Ahora lo que uno explica, el otro no lo cuenta.
    verde, blanca, negra, ref = MA.detecta(rgb, m)
    pizcas = np.zeros(m.shape, bool)
    sud_hecho = None
    try:
        import sudada as SU
        rgb_alto = SU.carga(ruta)
        hoja_alto = mascara_hoja(rgb_alto, mascara_marca(rgb_alto))[0]
        pz, _vd = SU.mapa_pizcas(rgb_alto, hoja_alto, SU.UMBRAL_A)
        #     las pizquitas se detectan a 2048, donde se ven; para restarlas del
        #     mapa de agujeros hay que traerlas a la resolucion de este paso
        pizcas = cv2.resize(pz.astype(np.uint8), (m.shape[1], m.shape[0]),
                            interpolation=cv2.INTER_AREA) > 0
        sud_hecho = SU.mide_ero(rgb_alto, hoja_alto)
    except Exception as ex:                                          # noqa: BLE001
        diag["avisos"].append("sudado no medido: {}".format(str(ex)[:60]))
    #     verde y blanca SI se excluyen; NEGRA no, porque un agujero sobre mesa
    #     oscura se ve negro y excluirla se llevaria agujeros de verdad.
    ya_explicado = verde | blanca | pizcas
    if ya_explicado.any():
        ya_explicado = cv2.dilate(ya_explicado.astype(np.uint8), AG.disco(2)) > 0

    # ---- 6. agujeros, roturas y mordidas  (agujeros.py) ------------------
    #     `excluir=ya_explicado` se probo el 19/09 y NO ayuda: quitar las
    #     pizquitas y las manchas deja la cuenta igual (capa 28 -> 22) porque
    #     lo que el detector marca no son manchas, son ARRUGAS. Se deja el
    #     parametro en `agujeros.py` documentado y sin usar, para que el
    #     camino de entrenar y el de clasificar hagan lo MISMO: la perilla
    #     que si sirve es `AG.GROSOR_MIN`, que vale para los dos.
    ag, ro, marca_r, dg = AG.detecta(rgb, m, ac_px)
    min_px = AG.MIN_PX
    if pulg_px:
        min_px = max(AG.MIN_PX, int(AG.MIN_PULG2 / (pulg_px ** 2)))
    c_ag = AG.componentes(ag, min_px)
    c_ro = AG.componentes(ro, min_px)

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
    g_px = (AG.GROSOR_MORDIDA / pulg_px) if pulg_px else 0.02 * float(max(m.shape))
    c_mo = AG.gruesas(AG.componentes(falta, 4 * min_px), falta, g_px)
    d_casco = cv2.distanceTransform(casco, cv2.DIST_L2, 3)
    marca_llena = AG.rellena(marca_r) if marca_r.any() else marca_r
    t1 = float(t.max())

    # EL BORDE (2026-09-15). La chaveta corta una franja del filo, y hasta hoy
    # el codigo no la descontaba: un desgarro en el borde contaba como dano en
    # la zona usable. Medido en las hojas que marco el experto: de 0.30" en la hoja
    # pequena a 0.60" en la grande. "Claramente una hoja mas grande, mas se dana
    # su borde" -- por eso la franja crece con la hoja.
    d_hoja = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)

    def zona_de(tc, sc, mascara=None):
        if tc > lam1 - base_px:
            return "base"
        if tc < lam0 + punta_px:
            return "punta"
        if abs(sc) < vena_px / 2.0:
            return "vena"
        if mascara is not None and borde_px > 0:
            # si el defecto entero cabe dentro de la franja del filo, lo corta
            # la chaveta y no cuenta como dano de la zona usable
            if float(d_hoja[mascara].max()) < borde_px:
                return "borde"
        return "util_izq" if sc > 0 else "util_der"

    lab_ag = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_ag[:, :, 1] -= 128
    lab_ag[:, :, 2] -= 128

    det = []
    for tipo, lista in (("agujero", c_ag), ("rotura", c_ro), ("mordida", c_mo)):
        for c in lista:
            pxy = np.array([c["cx"], c["cy"]], np.float32) - mu
            tc, sc = float(pxy @ e), float(pxy @ perp)
            ap2 = c["area"] * (pulg_px ** 2) if pulg_px else None
            pen = float(d_casco[c["mask"]].max()) if tipo == "mordida" else 0.0
            real = (AG.es_agujero_real(lab_ag, c["mask"], dg["fondo"], dg["hoja"])
                    if tipo == "agujero" else None)
            det.append(dict(
                real=(1 if real else 0),
                tipo=tipo, area_px=c["area"], zona=zona_de(tc, sc, c["mask"]),
                area_pulg2=round(ap2, 4) if ap2 is not None else "",
                diam_pulg=(round(float(2 * np.sqrt(ap2 / np.pi)), 3)
                           if ap2 is not None else ""),
                dist_base_pulg=round((t1 - tc) * pulg_px, 3) if pulg_px else "",
                s_pulg=round(sc * pulg_px, 3) if pulg_px else "",
                pen_pulg=round(pen * pulg_px, 3) if pulg_px else "",
                en_marca=int(bool((c["mask"] & marca_llena).any()))))
    fila_ag = AG.agrega(det, float(m.sum()), pulg_px, {})

    # LA SEGUNDA MEDIDA, EN LA MISMA PASADA (2026-09-19). Ademas de la cuenta
    # cruda se guarda la VERIFICADA --solo los agujeros por los que se ve el
    # meson-- y de las dos sale el cociente, que dice cuanto de lo que parece
    # agujero es en realidad pliegue. Eso es informacion de FORMA y no de color,
    # y es lo que usa `arbol_habano.py` en sus nodos 2 y 3.
    # Se ANADE, no sustituye: sustituir la medida cruda por la verificada costo
    # 9,2 puntos en Connecticut (PENDIENTES A-terdecies).
    det_real = [d for d in det
                if d["tipo"] != "agujero" or d.get("real") == 1]
    fila_real = AG.agrega(det_real, float(m.sum()), pulg_px, {})
    for k_ in ("area_util_pulg2", "area_util_izq_pulg2", "area_util_der_pulg2",
               "n_agujeros", "area_def_pulg2"):
        R["v_" + k_] = fila_real.get(k_)
    iz_, de_ = (fila_real.get("area_util_izq_pulg2") or 0.0,
                fila_real.get("area_util_der_pulg2") or 0.0)
    R["v_asim"] = round((iz_ - de_) / (iz_ + de_ + 1e-6), 4)
    cru_a = fila_ag.get("area_util_pulg2") or 0.0
    cru_n = fila_ag.get("n_agujeros") or 0.0
    R["coc_area"] = round((fila_real.get("area_util_pulg2") or 0.0)
                          / (cru_a + 1e-6), 4)
    R["coc_n"] = round((fila_real.get("n_agujeros") or 0.0) / (cru_n + 1e-6), 4)

    # ---- 7. manchas  (manchas.py) ----------------------------------------
    #     ya estan calculadas en el paso 5-bis, se reutilizan
    min_px_m = max(60, int(MA.MIN_PULG2 / (pulg_px ** 2))) if pulg_px else 60
    det_m = []
    for tipo, msk in (("verde", verde), ("blanca", blanca), ("negra", negra)):
        for c in MA.componentes(msk, min_px_m):
            pxy = np.array([c["cx"], c["cy"]], np.float32) - mu
            tc, sc = float(pxy @ e), float(pxy @ perp)
            det_m.append(dict(
                tipo=tipo, area_px=c["area"], zona=zona_de(tc, sc, c["mask"]),
                area_pulg2=(round(c["area"] * pulg_px ** 2, 4) if pulg_px else ""),
                dist_base_pulg=round((t1 - tc) * pulg_px, 3) if pulg_px else "",
                s_pulg=round(sc * pulg_px, 3) if pulg_px else ""))
    fila_ma = MA.agrega(det_m, float(m.sum()), pulg_px, {})

    for d in (fila_ag, fila_ma):
        for k, v in d.items():
            if k in ("ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id"):
                continue
            try:
                R[k] = float(v)
            except (TypeError, ValueError):
                R[k] = np.nan
    iz = R.get("area_util_izq_pulg2", np.nan)
    de = R.get("area_util_der_pulg2", np.nan)
    R["asim_area_util"] = (iz - de) / (iz + de + 1e-6)
    R["dif_area_util"] = abs(iz - de)

    # ---- 8. la punta  (forma_punta.py) -----------------------------------
    m_fija = hoja
    if rot:
        m_fija = rota_fija(m_fija.astype(np.uint8), rot, True) > 0
    if voltea:
        m_fija = cv2.rotate(m_fija.astype(np.uint8), cv2.ROTATE_180) > 0
    ap = anchuras_punta(m_fija, pulg_px) if m_fija.sum() >= 500 else {}
    anc_pulg = R.get("ancho_pulg", np.nan)
    for k, v in ap.items():
        R[k] = v
        R[k + "_norm"] = (v / anc_pulg if (anc_pulg == anc_pulg and anc_pulg > 1)
                          else np.nan)
    for d in DIST_PUNTA:
        k = "punta_{:g}".format(d)
        R.setdefault(k, np.nan)
        R.setdefault(k + "_norm", np.nan)

    # ---- 9. el sudado  (sudada.py) ---------------------------------------
    #     Se RECARGA la foto a 2048 en vez de reaprovechar la mascara de 1024:
    #     el detector busca objetos de ~6 px de radio y a media resolucion el
    #     suavizado se los come. Es el unico rasgo del proyecto que ataca dos
    #     fronteras de la rejilla a la vez -- capa contra banda, y XL/XR contra
    #     1/2 banda (EXPLICACION.txt de `dataset/Hojas sudadas`).
    #     CON EL BORDE FUERA: la misma `mide_ero` que alimenta el CSV del
    #     entrenamiento, para que la cifra signifique lo mismo en los dos lados.
    #     medido ya en el paso 5-bis, que necesitaba el mapa de pizquitas para
    #     restarlo de los agujeros. Aqui solo se recogen los rasgos.
    if sud_hecho is not None:
        for _k, _v in sud_hecho.items():
            R[_k] = float(_v)
    else:
        for _k in ("sud_frac", "sud_n", "sud_dens", "sud_verdor_p95",
                   "sud_verdor_med"):
            R[_k] = np.nan

    if not pulg_px:
        diag["avisos"].append(
            "SIN ESCALA: todo lo que se mide en pulgadas queda vacio, y con ello "
            "la mitad de los rasgos de defecto")

    return dict(rasgos=R, diag=diag, defectos=det + det_m,
                mascara=m, rgb=rgb, marco=dict(e=e, perp=perp, mu=mu, t=t, s=s,
                                               lam0=lam0, lam1=lam1,
                                               punta_px=punta_px, base_px=base_px,
                                               vena_px=vena_px))


if __name__ == "__main__":
    import json
    r = mide(sys.argv[1])
    print(json.dumps({k: (None if v != v else round(float(v), 4))
                      for k, v in sorted(r["rasgos"].items())},
                     indent=2, ensure_ascii=False))
    print(json.dumps(r["diag"], indent=2, ensure_ascii=False, default=str))
    print("defectos: {}".format(len(r["defectos"])))
