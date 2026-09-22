r"""
DATOS COMPARTIDOS PARA LOS EXPERIMENTOS DEL PAPER. (2026-09-14)

POR QUE EXISTE
--------------
Los dos experimentos que pide el paper --la curva de aprendizaje y la linea base
CNN-- tienen que correr sobre **exactamente las mismas hojas y la misma particion**
que el modelo final, o la comparacion no vale nada. Este modulo es el unico sitio
donde se decide eso, para que los dos guiones no se separen.

LA SELECCION ES LA DE `entrena.py`, COPIADA A PROPOSITO
--------------------------------------------------------
    - fuera las omitidas de `dataset/CASOS_DUDOSOS.csv`  (hecho cerrado 5)
    - fuera `toca_borde == 1`                            (hecho cerrado 26)
    - fuera `base_dudosa == 1`
    - UNA foto por hoja: la de encuadre mas amplio        (hecho cerrado 4)
    - fuera `media_banda`                                 (hecho cerrado 29)

LOS RASGOS SE PIDEN POR NOMBRE AL MODELO GUARDADO
--------------------------------------------------
`out/modelo/modelo.json` guarda los 64 nombres de calidad y los 17 de variedad, en
el orden exacto en que se apilaron al entrenar. Se leen de ahi en vez de volver a
construir la lista, que es como se cuelan los desajustes de orden.

EL SUDADO, Y POR QUE FALTABA (2026-09-17)
------------------------------------------
Este modulo se escribio el 14/09, cuando el modelo tenia **64** rasgos. El 15/09
entro el sudado y paso a **69**. Los nombres se piden a `modelo.json`, asi que
desde ese dia se pedian 69 y solo se sabian rellenar 64: `sud_frac`, `sud_n`,
`sud_dens`, `sud_verdor_p95` y `sud_verdor_med` salian **NaN en el 100 % de las
hojas**, sin aviso ninguno. El imputador las descartaba en silencio y todo lo que
colgaba de aqui media un modelo de 64 rasgos que ya no existe.

Se vio porque una prueba de importancia por permutacion daba exactamente 0.0
para el grupo del sudado: barajar columnas vacias no cambia nada.

Arreglado leyendo `sudada_ero.csv` (entrenamiento, cruza 527 de 527 por `ruta`) y
`sudada_11sep_ero.csv` (las 112, cruza 112 de 112 **por nombre de archivo**: ese
CSV escribe la ruta con barras normales y `rasgos_FINAL_112.csv` con barras
invertidas, asi que por `ruta` no cruza ninguna).

**Y para que no se repita**, `_comprueba()` revienta si algun rasgo pedido sale
entero vacio. El fallo no fue el NaN: fue que nadie se entero.

Seis rasgos son DERIVADOS y no estan en ningun CSV; sus formulas salen de
`entrena.py` y se reproducen aqui igual:

    asim_area_util = (izq - der) / (izq + der + 1e-6)
    dif_area_util  = |izq - der|
    punta_X_norm   = punta_X / ancho_pulg
"""
import csv
import json
from pathlib import Path

import numpy as np
from rutas import REPO_RAIZ  # raiz del repositorio

RAIZ = REPO_RAIZ
OUT = RAIZ / "out"
DATASET = RAIZ / "dataset"
DIR_MODELO = OUT / "modelo"

COLS_PUNTA = ["punta_0.25", "punta_0.5", "punta_1", "punta_2"]


def num(v):
    try:
        if v is None or v == "":
            return np.nan
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _lee(nombre, clave="ruta"):
    f = OUT / nombre
    with open(f, encoding="utf-8") as fh:
        return {r[clave]: r for r in csv.DictReader(fh)}


def _lee_por_archivo(nombre):
    """Igual, pero indexado por el NOMBRE del fichero.

    `sudada_11sep_ero.csv` guarda la ruta con barras normales y
    `rasgos_FINAL_112.csv` con barras invertidas: por `ruta` no cruza ni una.
    """
    with open(OUT / nombre, encoding="utf-8") as fh:
        return {Path(r["ruta"]).name: r for r in csv.DictReader(fh)}


def _comprueba(X, nombres, donde):
    """Ningun rasgo puede salir entero vacio. Falla ruidosamente, a proposito.

    El 15/09 el sudado entro al modelo y este modulo no leia su CSV: cinco de los
    69 rasgos salieron NaN al 100 % durante dos dias sin que nada se quejara.
    """
    if len(X) == 0:
        return X
    vacios = [n for i, n in enumerate(nombres) if np.isnan(X[:, i]).all()]
    if vacios:
        raise RuntimeError(
            "{}: {} rasgo(s) salen VACIOS en las {} hojas -> {}.\n"
            "Falta leer el CSV que los trae, o el cruce de claves no pega."
            .format(donde, len(vacios), len(X), ", ".join(vacios)))
    return X


def nombres_rasgos():
    meta = json.loads((DIR_MODELO / "modelo.json").read_text(encoding="utf-8"))
    return meta["rasgos_calidad"], meta["rasgos_variedad"]


def _fila_derivados(fila, agu, fpu, ruta):
    """Los seis rasgos que no estan en ningun CSV."""
    d = {}
    a0 = agu.get(ruta)
    if a0:
        iz, de = num(a0.get("area_util_izq_pulg2")), num(a0.get("area_util_der_pulg2"))
        d["asim_area_util"] = (iz - de) / (iz + de + 1e-6)
        d["dif_area_util"] = abs(iz - de)
    else:
        d["asim_area_util"] = d["dif_area_util"] = np.nan
    anc = num(fila.get("ancho_pulg"))
    p0 = fpu.get(ruta, {})
    for c in COLS_PUNTA:
        d[c + "_norm"] = (num(p0.get(c)) / anc
                          if (p0 and anc == anc and anc > 1) else np.nan)
    return d


def entrenamiento(sin_clase=("media_banda",)):
    """Las hojas de agosto, tal y como las ve el modelo final.

    Devuelve (X_cal, X_var, y, variedad, grupo, rutas).
    """
    cal, var = nombres_rasgos()
    man = _lee("manifiesto_limpio.csv")
    seg, zon = _lee("segmentacion.csv"), _lee("zonas.csv")
    ras, agu = _lee("clasificador_rasgos.csv"), _lee("agujeros.csv")
    mch, fpu = _lee("manchas.csv"), _lee("forma_punta.csv")
    sud = _lee("sudada_ero.csv")      # los 5 rasgos de sudado (2026-09-17)

    omitidas = set()
    f_dud = DATASET / "CASOS_DUDOSOS.csv"
    if f_dud.exists():
        with open(f_dud, encoding="utf-8") as fh:
            omitidas = {r["ruta"] for r in csv.DictReader(fh)
                        if r.get("estado", "").startswith("omitida")}

    limpias = [r for r in man
               if r not in omitidas
               and seg.get(r, {}).get("toca_borde") == "0"
               and zon.get(r, {}).get("base_dudosa") == "0"]
    mejor = {}
    for r in limpias:
        h = man[r]["hoja_id"]
        f = num(seg[r]["frac_cuadro"])
        if h not in mejor or f < mejor[h][1]:
            mejor[h] = (r, f)
    rutas = sorted(v[0] for v in mejor.values() if v[0] in ras)
    fuera = set(sin_clase)
    rutas = [r for r in rutas if man[r]["clase"] not in fuera]

    Xc, Xv = [], []
    for r in rutas:
        fila = dict(ras[r])
        for d in (agu.get(r), mch.get(r), zon.get(r), seg.get(r), fpu.get(r),
                  sud.get(r)):
            if d:
                for k, v in d.items():
                    fila.setdefault(k, v)
        fila.update(_fila_derivados(ras[r], agu, fpu, r))
        Xc.append([num(fila.get(c)) for c in cal])
        Xv.append([num(fila.get(c)) for c in var])

    Xc, Xv = np.array(Xc, float), np.array(Xv, float)
    _comprueba(Xc, cal, "entrenamiento / calidad")
    _comprueba(Xv, var, "entrenamiento / variedad")
    return (Xc, Xv,
            np.array([man[r]["clase"] for r in rutas]),
            np.array([man[r]["variedad"] for r in rutas]),
            np.array([man[r]["hoja_id"] for r in rutas]),
            list(rutas))


def jornada_112():
    """Las 112 del 11/09: rasgos ya medidos, etiqueta de la carpeta.

    Son TODAS Connecticut (el modelo dice 112 de 112, conf minima 0.851) y no
    traen ni una BANDA: 71 capa, 22 xl_izq, 19 xr_der. Eso ultimo es el agujero
    de 92.5 y hay que decirlo siempre que se cite una cifra de este lote.
    """
    cal, var = nombres_rasgos()
    d = OUT / "SESION_FINAL_112"
    with open(d / "rasgos_FINAL_112.csv", encoding="utf-8") as fh:
        ras = list(csv.DictReader(fh))
    with open(d / "DECISION_FINAL_112.csv", encoding="utf-8") as fh:
        dec = {r["archivo"]: r for r in csv.DictReader(fh, delimiter=";")}

    # `rasgos_FINAL_112.csv` es del 14/09 y NO trae el sudado: se pega aparte.
    sud = _lee_por_archivo("sudada_11sep_ero.csv")

    kar = "archivo" if "archivo" in ras[0] else "ruta"
    Xc, Xv, y, arch = [], [], [], []
    for f in ras:
        a = Path(f[kar]).name
        if a not in dec:
            continue
        fila = dict(f)
        s0 = sud.get(a)
        if s0:
            for k, v in s0.items():
                fila.setdefault(k, v)
        Xc.append([num(fila.get(c)) for c in cal])
        Xv.append([num(fila.get(c)) for c in var])
        y.append(dec[a]["clase_real"])
        arch.append(a)
    Xc, Xv = np.array(Xc, float), np.array(Xv, float)
    _comprueba(Xc, cal, "jornada 112 / calidad")
    _comprueba(Xv, var, "jornada 112 / variedad")
    return (Xc, Xv,
            np.array(y), np.array(["connecticut"] * len(y)),
            np.array(arch), arch)


if __name__ == "__main__":
    from collections import Counter
    Xc, Xv, y, v, g, rutas = entrenamiento()
    print("AGOSTO   hojas {}  rasgos cal {}  var {}".format(
        len(y), Xc.shape[1], Xv.shape[1]))
    print("         clases  ", dict(sorted(Counter(y).items())))
    print("         variedad", dict(sorted(Counter(v).items())))
    print("         nan en calidad: {:.2f} %".format(
        100 * np.isnan(Xc).mean()))
    Xc2, Xv2, y2, v2, g2, a2 = jornada_112()
    print("11/09    hojas {}  rasgos cal {}".format(len(y2), Xc2.shape[1]))
    print("         clases  ", dict(sorted(Counter(y2).items())))
    print("         nan en calidad: {:.2f} %".format(100 * np.isnan(Xc2).mean()))
