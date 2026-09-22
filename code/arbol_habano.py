# -*- coding: utf-8 -*-
r"""
EL ARBOL DE LA REGLA DE OFICIO, PARA LA CALIDAD DEL HABANO. (2026-09-19)

POR QUE EXISTE
--------------
El 19/09 el experto paso 217 hojas de Habano NUEVAS, a ciegas, de una jornada distinta.
El modelo plano de 69 rasgos saco **26,7 %** -- peor que decir siempre "capa"
(38,7 %) -- y de lo que firmaba solo acertaba el 32,1 %, cuando dentro de su sesion
esa cifra era 94,5 %. El diagnostico, medido: de las hojas cuya asimetria MEDIDA es
negativa (la firma de XL izquierdo), el modelo decia XR en 17 y XL en 1. **Nunca
aprendio el lado del danio**: aprendio el tamanio, que en su sesion iba pegado a la
clase, y eso no viaja.

LA IDEA, QUE ES DEL EXPERTO Y NO MIA
---------------------------------
Su regla de oficio dice QUE mirar para cada frontera:

                  pizquitas verdes (sudado)     agujeros
    CAPA                  no                       no
    BANDA                 SI                       no
    XL / XR               no                       SI, de un lado

O sea tres decisiones, y **cada una tiene que decidirse con los rasgos que su regla
nombra**, no con los 69 a la vez. Medido: con los 69, el nodo del lado saca 49,6 %
fuera de sesion; con solo los de asimetria, 73,9 %. **Los otros 64 no ayudan,
envenenan**, porque cada uno trae la firma de su jornada.

    nodo 1  banda contra el resto   SUDADO + COLOR                      84,1 %
    nodo 2  capa contra lateral     FORMA_DANO + SUDADO + COCIENTES     67,2 %
    nodo 3  XL contra XR            ASIMETRIA + asimetria VERIFICADA    73,9 %

    cuatro clases, examen en la OTRA sesion:
        agosto -> 19/09    plano 26,7 %   ARBOL 55,3 %   (mayoritaria 38,7 %)
        19/09 -> agosto    plano 29,3 %   ARBOL 50,2 %   (mayoritaria 36,2 %)

LOS COCIENTES, Y LA LECCION QUE COSTO UNA CASCADA ENTERA
--------------------------------------------------------
Hay dos medidas del agujero: la CRUDA y la VERIFICADA (`AGUJERO_REAL=1` en
`agujeros.py`: se erosiona cada componente y se mira si su nucleo se parece al meson,
o sea si por el hueco se ve la mesa). El 19/09 se probo **sustituir** la cruda por la
verificada: re-extraidas las 1.177 fotos y reentrenado, Connecticut cayo de 93,5 a
84,3 % y hubo que revertirlo (PENDIENTES A-terdecies). La verificacion deja la
asimetria impecable pero se lleva la CANTIDAD, y el modelo usaba la cantidad.

Aqui entra **ANADIDA**: el cociente verificada/cruda dice *cuanto de lo que parece
agujero es en realidad pliegue*, que es informacion de forma y no de color. Anadirla
gana 10 puntos en el nodo 2 y 4 en el nodo 3.

LA CONFIANZA SE CALIBRA ENTRE SESIONES, NO DENTRO
--------------------------------------------------
La confianza del modelo viejo esta calibrada dentro de su jornada y fuera no
significa nada (firma el 38,7 % y acierta el 32,1 %). Aqui cada nodo guarda su tabla
de acierto por tramo de confianza **medida entrenando en una sesion y examinando en
la otra**, que es la unica forma honesta con dos sesiones.

EL TECHO, DICHO CLARO
---------------------
55 % no es un sistema que no se equivoque. El techo lo pone la captura: sobre mesa
oscura (b*=-2 contra -15 de las jornadas de Connecticut) un pliegue en sombra y un
agujero son el mismo color, y eso NO se arregla por software -- tres criterios
medidos y rechazados el 19/09. El nodo 2 es el que mas tiene que ganar con una
captura mejor: hoy manda 33 de 59 XL izquierdo a "capa".

Uso:
    python scripts/arbol_habano.py --entrena
    python scripts/arbol_habano.py --prueba out/SESION_HABANO_19SEP
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.impute import SimpleImputer                              # noqa: E402
from sklearn.linear_model import LogisticRegression                   # noqa: E402
from sklearn.pipeline import make_pipeline                            # noqa: E402
from sklearn.preprocessing import StandardScaler                      # noqa: E402
from rutas import REPO_RAIZ  # raiz del repositorio

OUT = REPO_RAIZ / "out"
DEST = OUT / "modelo_arbol"
ETIQUETAS = REPO_RAIZ / "fotos" / "habano_prueba"
CARPETA_CLASE = {"banda": "banda", "capa": "capa", "xl izquierdo": "xl_izq",
                 "xr derecho": "xr_der"}
CLASES = ("capa", "banda", "xl_izq", "xr_der")
COLS_V = ["area_util_pulg2", "area_util_izq_pulg2", "area_util_der_pulg2",
          "n_agujeros", "area_def_pulg2"]
NOM_EXTRA = ["v_area_util", "v_izq", "v_der", "v_n_agujeros", "v_area_def",
             "v_asim", "coc_area", "coc_n"]
SEMILLAS = 6
TRAMOS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def mk(sem):
    """La misma receta del modelo guardado: logistica con mediana y escalado."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=4000,
                                            class_weight="balanced"))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def familias(nom):
    """Los rasgos que nombra la regla del experto, por nodo."""
    hoja = [c for c in nom if c.startswith("hoja_") and c != "hoja_entre_util"]
    sud = [c for c in nom if c.startswith("sud_")]
    forma_dano = [c for c in ("solidez", "n_roturas", "pen_mordida_izq_pulg",
                              "pen_mordida_der_pulg", "pen_mordida_pulg",
                              "asim_pen_mordida") if c in nom]
    asim = [c for c in nom if c.startswith("asim") or c.startswith("dif_")
            or "pen_mordida" in c]
    return {
        # nodo 1: la banda se delata por las pizquitas verdes
        "n1": sud + hoja,
        # nodo 2: falta un trozo? forma del danio, sudado, y cuanto de lo que
        #         parece agujero es pliegue
        "n2": forma_dano + sud + ["coc_area", "coc_n", "v_asim"],
        # nodo 3: de que lado. Solo asimetrias, que son adimensionales
        "n3": asim + ["v_asim"],
    }


def extras(fila, X_cruda, idx):
    """Las columnas verificadas y los cocientes, para una fila."""
    v = [num(fila.get(c)) for c in COLS_V]
    iz, de = v[1], v[2]
    asim = (iz - de) / (iz + de + 1e-6)
    coc = v[0] / (X_cruda[idx["area_util_pulg2"]] + 1e-6)
    coc_n = v[3] / (X_cruda[idx["n_agujeros"]] + 1e-6)
    return v + [asim, coc, coc_n]


def lee_sesion_agosto(nom, idx):
    """Las hojas de Habano de agosto, con su medida cruda y su verificada."""
    import paper_datos as D
    X, Xv, y, var, grp, rutas = D.entrenamiento()
    s = np.asarray(var) == "habano"
    X, y = X[s], np.asarray(y)[s]
    rutas = [str(r) for r in np.asarray(rutas)[s]]
    ver = {}
    with open(OUT / "agujeros_real.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ver[r["ruta"]] = r
    E = np.array([extras(ver.get(rutas[i], {}), X[i], idx)
                  for i in range(len(rutas))], float)
    return np.column_stack([X, E]), y, np.array(rutas)


def lee_carpeta(carpeta, nom, idx, con_etiqueta=True):
    """Una sesion medida por `clasifica_carpeta.py`, cruda + verificada."""
    carpeta = Path(carpeta)
    f_cru = next(carpeta.glob("rasgos_*.csv"))
    nombre = f_cru.stem.replace("rasgos_", "")
    cru = list(csv.DictReader(open(f_cru, encoding="utf-8")))
    k = "archivo" if "archivo" in cru[0] else "ruta"
    # Camino nuevo (2026-09-19): `rasgos_foto.mide()` saca las dos medidas en la
    # MISMA pasada y deja las columnas `v_*` y los cocientes en este mismo CSV.
    # Camino viejo: una segunda corrida con AGUJERO_REAL=1 en la carpeta _REAL.
    if "v_area_util_pulg2" in cru[0]:
        ver = None
        print("   (las dos medidas vienen en el mismo CSV, una sola pasada)")
    else:
        f_ver = (carpeta.parent / (carpeta.name + "_REAL")
                 / ("rasgos_%s_REAL.csv" % nombre))
        if not f_ver.exists():
            raise SystemExit("falta la medida verificada: %s -- vuelve a medir la "
                             "carpeta con la tuberia de hoy, que la saca en la "
                             "misma pasada" % f_ver)
        ver = {Path(r[k]).name: r
               for r in csv.DictReader(open(f_ver, encoding="utf-8"))}
    etq = {}
    if con_etiqueta and ETIQUETAS.exists():
        for sub in ETIQUETAS.iterdir():
            if sub.is_dir() and CARPETA_CLASE.get(sub.name.lower()):
                for p in sub.iterdir():
                    if p.suffix.lower() in (".heic", ".jpg", ".jpeg", ".png"):
                        etq[p.name] = CARPETA_CLASE[sub.name.lower()]
    elif con_etiqueta:
        # SIN FOTOS (2026-09-22). Las fotografias de la planta no se publican,
        # asi que las mismas etiquetas que estaban en los nombres de las
        # carpetas de imagenes viajan en un CSV. El resultado es identico.
        f_etq = REPO_RAIZ / "dataset" / "etiquetas_habano.csv"
        if not f_etq.exists():
            raise SystemExit("faltan las etiquetas: %s" % f_etq)
        for r in csv.DictReader(open(f_etq, encoding="utf-8")):
            etq[r["archivo"]] = r["clase"]
    X, y, arch = [], [], []
    for r in cru:
        a = Path(r[k]).name
        if con_etiqueta and etq.get(a) is None:
            continue
        xc = np.array([num(r.get(c)) for c in nom], float)
        if ver is None:
            e = [num(r.get("v_" + c)) for c in COLS_V]
            e += [num(r.get("v_asim")), num(r.get("coc_area")), num(r.get("coc_n"))]
        else:
            e = extras(ver.get(a, {}), xc, idx)
        X.append(np.concatenate([xc, e]))
        y.append(etq.get(a, ""))
        arch.append(a)
    return np.array(X, float), np.array(y), np.array(arch)


# ---------------------------------------------------------------- el arbol ---
def entrena_nodos(X, y, cols, sem):
    """Los tres modelos del arbol. Devuelve dict de modelos."""
    m = {}
    m["n1"] = mk(sem).fit(X[:, cols["n1"]], np.where(y == "banda", "banda", "otro"))
    t2 = y != "banda"
    m["n2"] = mk(sem).fit(X[t2][:, cols["n2"]],
                          np.where(y[t2] == "capa", "capa", "lateral"))
    t3 = np.isin(y, ("xl_izq", "xr_der"))
    m["n3"] = mk(sem).fit(X[t3][:, cols["n3"]], y[t3])
    return m


def decide(m, X, cols):
    """Clase y confianza (el producto de las confianzas de los nodos usados)."""
    n = len(X)
    pr = np.array(["?"] * n, dtype=object)
    conf = np.ones(n)
    p1 = m["n1"].predict_proba(X[:, cols["n1"]])
    c1 = m["n1"].classes_
    i_banda = list(c1).index("banda")
    es_banda = p1[:, i_banda] > 0.5
    pr[es_banda] = "banda"
    conf[es_banda] = p1[es_banda, i_banda]
    resto = ~es_banda
    conf[resto] = 1 - p1[resto, i_banda]
    if resto.any():
        ir = np.flatnonzero(resto)
        p2 = m["n2"].predict_proba(X[ir][:, cols["n2"]])
        c2 = list(m["n2"].classes_)
        i_capa = c2.index("capa")
        es_capa = p2[:, i_capa] > 0.5
        pr[ir[es_capa]] = "capa"
        conf[ir[es_capa]] *= p2[es_capa, i_capa]
        lat = ir[~es_capa]
        conf[lat] *= (1 - p2[~es_capa, i_capa])
        if len(lat):
            p3 = m["n3"].predict_proba(X[lat][:, cols["n3"]])
            c3 = np.array(m["n3"].classes_)
            j = p3.argmax(1)
            pr[lat] = c3[j]
            conf[lat] *= p3[np.arange(len(lat)), j]
    return pr, conf


def tabla_tramos(conf, ok):
    """Acierto MEDIDO por tramo de confianza (la honesta: entre sesiones)."""
    t = []
    for a, b in zip(TRAMOS, TRAMOS[1:]):
        s = (conf >= a) & (conf < b)
        t.append(dict(desde=a, hasta=b, n=int(s.sum()),
                      acierto=(float(ok[s].mean()) if s.any() else None)))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrena", action="store_true")
    ap.add_argument("--prueba", default="")
    a = ap.parse_args()

    import paper_datos as D
    nom, _ = D.nombres_rasgos()
    idx = {c: i for i, c in enumerate(nom)}
    nom_todo = list(nom) + NOM_EXTRA
    idx_todo = {c: i for i, c in enumerate(nom_todo)}
    fam = familias(nom_todo)
    cols = {k: [idx_todo[c] for c in v] for k, v in fam.items()}

    if a.prueba:
        f = DEST / "arbol_habano.joblib"
        if not f.exists():
            raise SystemExit("no hay modelo; corre antes --entrena")
        import joblib
        guardado = joblib.load(f)
        X, y, arch = lee_carpeta(a.prueba, nom, idx, con_etiqueta=True)
        pr, conf = decide(guardado["modelos"][0], X, cols)
        print("%s: %i hojas" % (Path(a.prueba).name, len(X)))
        print("   dice:", dict(Counter(pr)))
        if (y != "").all():
            print("   ACIERTO: %.1f %%  (mayoritaria %.1f %%)"
                  % (100 * (pr == y).mean(),
                     100 * max(Counter(y).values()) / len(y)))
        return

    # ---- las dos sesiones -------------------------------------------------
    XA, yA, _ = lee_sesion_agosto(nom, idx)
    XB, yB, _ = lee_carpeta(OUT / "SESION_HABANO_19SEP", nom, idx)
    print("agosto   %i hojas  %s" % (len(yA), dict(Counter(yA))))
    print("19/09    %i hojas  %s" % (len(yB), dict(Counter(yB))))
    print()

    # ---- calibracion ENTRE sesiones, que es la honesta --------------------
    print("=" * 74)
    print("EXAMEN CRUZADO ENTRE SESIONES (lo que se puede prometer)")
    print("=" * 74)
    conf_all, ok_all, res = [], [], {}
    for etq, Xtr, ytr, Xte, yte in (("agosto -> 19/09", XA, yA, XB, yB),
                                    ("19/09 -> agosto", XB, yB, XA, yA)):
        acc, pr_u = [], None
        for sem in range(SEMILLAS):
            m = entrena_nodos(Xtr, ytr, cols, sem)
            pr, cf = decide(m, Xte, cols)
            acc.append(float((pr == yte).mean()))
            conf_all.append(cf); ok_all.append(pr == yte)
            pr_u = pr
        may = max(Counter(yte).values()) / len(yte)
        res[etq] = float(np.mean(acc))
        print("   %-18s %5.1f %%   (mayoritaria %.1f %%)"
              % (etq, 100 * np.mean(acc), 100 * may))
        print("      {:<9}".format("") + "".join("{:>9}".format(c) for c in CLASES)
              + "{:>9}".format("acierto"))
        for r in CLASES:
            fila = [int(((yte == r) & (pr_u == c)).sum()) for c in CLASES]
            tot = int((yte == r).sum())
            print("      {:<9}".format(r) + "".join("{:>9}".format(x) for x in fila)
                  + "{:>8.0f} %".format(100 * fila[CLASES.index(r)] / tot if tot else 0))
        print()
    conf_all = np.concatenate(conf_all); ok_all = np.concatenate(ok_all)
    tramos = tabla_tramos(conf_all, ok_all)
    print("CONFIANZA CONTRA ACIERTO, medido ENTRE sesiones:")
    for t in tramos:
        if t["n"]:
            print("   conf %.2f - %.2f   n=%5i   acierto %5.1f %%"
                  % (t["desde"], t["hasta"], t["n"], 100 * t["acierto"]))
    print()

    # ---- el modelo final: las DOS sesiones juntas -------------------------
    X = np.vstack([XA, XB]); y = np.concatenate([yA, yB])
    modelos = [entrena_nodos(X, y, cols, sem) for sem in range(SEMILLAS)]
    DEST.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(dict(modelos=modelos, cols=cols, nom=nom_todo, familias=fam),
                DEST / "arbol_habano.joblib")
    meta = dict(
        fecha=__import__("datetime").datetime.now().isoformat(timespec="seconds"),
        n_hojas=int(len(y)), reparto={k: int(v) for k, v in Counter(y).items()},
        sesiones={"agosto": int(len(yA)), "19sep": int(len(yB))},
        familias=fam, cruzado=res, tramos=tramos,
        aviso=("Entrenado con DOS sesiones de Habano. Las cifras de `cruzado` son "
               "entrenando en una y examinando en la otra, que es lo unico honesto "
               "con dos sesiones: no hay una tercera con la que examinar este "
               "modelo. La confianza esta calibrada ENTRE sesiones."))
    (DEST / "arbol_habano.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-> %s" % (DEST / "arbol_habano.joblib"))
    print("-> %s" % (DEST / "arbol_habano.json"))
    print()
    print("AVISO: el modelo guardado usa las DOS sesiones, asi que no hay conjunto")
    print("con el que examinarlo. Lo que se puede prometer es lo del examen cruzado")
    print("de arriba, y es lo que va en el paper.")


if __name__ == "__main__":
    main()
