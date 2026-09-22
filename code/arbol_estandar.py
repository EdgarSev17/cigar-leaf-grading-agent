# -*- coding: utf-8 -*-
r"""
LA MISMA ARQUITECTURA PARA LAS DOS VARIEDADES. (2026-09-19)

POR QUE EXISTE
--------------
el experto, y es la pregunta correcta: *"que todo sea estandarizado, para evitar
identificar mejores rasgos en una variedad de tabaco y en la otra no"*.

El arbol de `arbol_habano.py` salio midiendo sobre Habano. Si los rasgos que elige
cada nodo fueran buenos solo ahi, seria un ajuste a una variedad y no un metodo. La
prueba es aplicar **el mismo arbol, con los mismos rasgos en los mismos nodos**, al
Connecticut, y examinarlo fuera de sesion:

    Habano        entrena agosto (307)        examina 19/09 (217)
    Connecticut   entrena agosto (220)        examina las 112 del 11/09

Y comparar cada uno contra el modelo plano de 69 rasgos, que es lo que hay hoy.

Las 112 no tienen ninguna BANDA, asi que el nodo 1 no se puede puntuar ahi; se deja
en la cadena igual que en produccion --si dice banda, se equivoca y cuenta como
error-- porque lo que se mide es el sistema entero, no un nodo suelto.

Uso:  python scripts/arbol_estandar.py
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))

import arbol_habano as AH                                             # noqa: E402
import paper_datos as D                                               # noqa: E402
from rutas import REPO_ROOT  # repository root

OUT = REPO_ROOT / "out"
CLASES = ("capa", "banda", "xl_izq", "xr_der")
SEM = 6


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def agosto(variedad, nom, idx):
    """Las hojas de agosto de esa variedad, cruda + verificada."""
    X, Xv, y, var, grp, rutas = D.entrenamiento()
    s = np.asarray(var) == variedad
    X, y = X[s], np.asarray(y)[s]
    rutas = [str(r) for r in np.asarray(rutas)[s]]
    ver = {}
    with open(OUT / "agujeros_real.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ver[r["ruta"]] = r
    E = np.array([AH.extras(ver.get(rutas[i], {}), X[i], idx)
                  for i in range(len(rutas))], float)
    return np.column_stack([X, E]), y


def carpeta(dir_, nom, idx):
    """Una sesion medida con la tuberia de hoy: etiqueta de la subcarpeta."""
    f = next(Path(dir_).glob("rasgos_*.csv"))
    filas = list(csv.DictReader(open(f, encoding="utf-8")))
    k = "archivo" if "archivo" in filas[0] else "ruta"
    X, y = [], []
    for r in filas:
        cl = (r.get("clase") or "").strip()
        if cl not in CLASES:
            continue
        xc = np.array([num(r.get(c)) for c in nom], float)
        e = [num(r.get("v_" + c)) for c in AH.COLS_V]
        e += [num(r.get("v_asim")), num(r.get("coc_area")), num(r.get("coc_n"))]
        X.append(np.concatenate([xc, e]))
        y.append(cl)
    return np.array(X, float), np.array(y)


def main():
    nom, _ = D.nombres_rasgos()
    idx = {c: i for i, c in enumerate(nom)}
    nom_todo = list(nom) + AH.NOM_EXTRA
    idx_todo = {c: i for i, c in enumerate(nom_todo)}
    fam = AH.familias(nom_todo)
    cols = {k: [idx_todo[c] for c in v] for k, v in fam.items()}
    print("LOS RASGOS DE CADA NODO, IDENTICOS PARA LAS DOS VARIEDADES")
    for k, v in fam.items():
        print("   %s (%i): %s" % (k, len(v), ", ".join(v)))
    print()

    casos = []
    XA, yA = agosto("habano", nom, idx)
    XB, yB = carpeta(OUT / "SESION_HABANO_UNA_PASADA", nom, idx)
    if not len(yB):
        XB, yB = AH.lee_carpeta(OUT / "SESION_HABANO_19SEP", nom, idx)[:2]
    casos.append(("HABANO", "agosto", "19/09", XA, yA, XB, yB))
    XC, yC = agosto("connecticut", nom, idx)
    XD, yD = carpeta(OUT / "SESION_CNT_112_HOY", nom, idx)
    casos.append(("CONNECTICUT", "agosto", "112 del 11/09", XC, yC, XD, yD))

    print("=" * 78)
    print("MISMA ARQUITECTURA, LAS DOS VARIEDADES, FUERA DE SESION")
    print("=" * 78)
    print("   {:<13} {:<22} {:>9} {:>9} {:>9} {:>11}".format(
        "variedad", "entrena -> examina", "plano", "ARBOL", "mayorit.", "n examen"))
    for etq, e1, e2, Xtr, ytr, Xte, yte in casos:
        if not len(yte):
            print("   %-13s SIN DATOS (falta medir la carpeta)" % etq)
            continue
        acc_a, acc_p, pr_u = [], [], None
        for sem in range(SEM):
            m = AH.entrena_nodos(Xtr, ytr, cols, sem)
            pr, cf = AH.decide(m, Xte, cols)
            acc_a.append(float((pr == yte).mean()))
            pl = AH.mk(sem).fit(Xtr, ytr).predict(Xte)
            acc_p.append(float((pl == yte).mean()))
            pr_u = pr
        may = max(Counter(yte).values()) / len(yte)
        print("   {:<13} {:<22} {:>8.1f} % {:>8.1f} % {:>8.1f} % {:>11}".format(
            etq, "%s -> %s" % (e1, e2), 100 * np.mean(acc_p), 100 * np.mean(acc_a),
            100 * may, len(yte)))
        print("      {:<9}".format("") + "".join("{:>9}".format(c) for c in CLASES)
              + "{:>9}".format("acierto"))
        for r in CLASES:
            tot = int((yte == r).sum())
            if not tot:
                continue
            fila = [int(((yte == r) & (pr_u == c)).sum()) for c in CLASES]
            print("      {:<9}".format(r) + "".join("{:>9}".format(x) for x in fila)
                  + "{:>8.0f} %".format(100 * fila[CLASES.index(r)] / tot))
        print()


if __name__ == "__main__":
    main()
