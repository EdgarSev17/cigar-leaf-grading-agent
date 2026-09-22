# -*- coding: utf-8 -*-
"""REPRODUCE LAS CIFRAS DEL ARTICULO. SIN FOTOS.

Las fotografias de la planta no se publican. Lo que si se publica son los
rasgos ya medidos, y con ellos este guion reproduce **el mismo protocolo que
reporta el articulo**: seis reentrenamientos del arbol de decisiones encadenadas
sobre el conjunto de entrenamiento, cada uno evaluado sobre las 112 hojas del
lote independiente, que el modelo nunca vio. Son 6 x 112 = 672 decisiones, y es
el pie de la Tabla IV del articulo.

Reproduce, en una sola pasada:

    Tabla IV   cuantas hojas decide el sistema y cuanto acierta en ellas
    Tabla III  el reparto de aciertos y errores por calidad
    Figura 2   el error segun que fraccion se aparta
    y el 84,8 % con kappa 0,705 del resumen

Uso:
    python code/reproduce.py
    python code/reproduce.py --semillas 12    mas repeticiones, mas estable

POR QUE SEIS REENTRENAMIENTOS Y NO UN MODELO GUARDADO. Un solo modelo da un
numero que depende de la particion con la que se entreno; sobre este lote, el
modelo de `out/modelo/` da 86,6 %. La media de seis reentrenamientos --84,8 %--
es la cifra defendible, y es la que reporta el articulo. Los dos numeros son
correctos y miden cosas distintas.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                   # noqa: E402
from rutas import REPO_RAIZ                                          # noqa: E402
import arbol_habano as AH                                            # noqa: E402
import arbol_estandar as AE                                          # noqa: E402
import paper_datos as D                                              # noqa: E402

OUT = REPO_RAIZ / "out"
BONITO = {"capa": "capa", "xl_izq": "XL izq.", "xr_der": "XR der.", "banda": "banda"}


def kappa(real, dicho, n):
    clases = sorted(set(real) | set(dicho))
    ix = {c: i for i, c in enumerate(clases)}
    M = np.zeros((len(clases), len(clases)))
    for a, b in zip(real, dicho):
        M[ix[a], ix[b]] += 1
    po = np.trace(M) / n
    pe = float((M.sum(0) * M.sum(1)).sum()) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Reproduce las cifras del articulo.")
    ap.add_argument("--semillas", type=int, default=6,
                    help="reentrenamientos (el articulo usa 6)")
    args = ap.parse_args()

    nom, _ = D.nombres_rasgos()
    idx = {c: i for i, c in enumerate(nom)}
    nom_todo = list(nom) + AH.NOM_EXTRA
    idx_todo = {c: i for i, c in enumerate(nom_todo)}
    cols = {k: [idx_todo[c] for c in v] for k, v in AH.familias(nom_todo).items()}

    XH, yH = AE.agosto("habano", nom, idx)
    XC, yC = AE.agosto("connecticut", nom, idx)
    XB, yB, _ = AH.lee_carpeta(OUT / "SESION_HABANO_GROSOR3", nom, idx)
    X112, y112 = AE.carpeta(OUT / "SESION_CNT_112_GROSOR3", nom, idx)
    Xtr = np.vstack([XC, XH, XB])
    ytr = np.concatenate([yC, yH, yB])

    C, K, P, Y = [], [], [], []
    for sem in range(args.semillas):
        m = AH.entrena_nodos(Xtr, ytr, cols, sem)
        pr, cf = AH.decide(m, X112, cols)
        C.append(cf); K.append(pr == y112); P.append(pr); Y.append(y112)

    conf = np.concatenate(C)
    ok = np.concatenate(K).astype(bool)
    pred = np.concatenate(P)
    real = np.concatenate(Y)
    n = len(conf)

    print()
    print("=" * 70)
    print("LOTE INDEPENDIENTE: %d decisiones  (%d reentrenamientos x %d hojas)"
          % (n, args.semillas, len(y112)))
    print("=" * 70)
    print()
    print("   acierto decidiendo todas    %5.1f %%      kappa %.3f"
          % (100.0 * ok.mean(), kappa(real, pred, n)))
    print()

    print("   TABLA IV -- cuantas decide y cuanto acierta")
    print("      %-30s %8s %10s %9s" % ("regla", "decide", "acierta", "deriva"))
    print("      %-30s %7.1f %% %9.1f %% %8.1f %%"
          % ("decide todas", 100.0, 100.0 * ok.mean(), 0.0))
    for u in (0.60, 0.70, 0.80):
        m = conf >= u
        if m.sum():
            print("      %-30s %7.1f %% %9.1f %% %8.1f %%"
                  % ("umbral de confianza %.2f" % u, 100.0 * m.mean(),
                     100.0 * ok[m].mean(), 100.0 * (1 - m.mean())))
    orden = np.argsort(-conf, kind="stable")
    for frac in (0.10, 0.20):
        k = max(1, int(round(n * (1 - frac))))
        print("      %-30s %7.1f %% %9.1f %% %8.1f %%"
              % ("aparta el %.0f %% menos seguro" % (100 * frac),
                 100.0 * k / n, 100.0 * ok[orden[:k]].mean(),
                 100.0 * (1 - k / float(n))))
    print()

    print("   TABLA III -- por calidad")
    for c in ("capa", "xl_izq", "xr_der", "banda"):
        m = real == c
        if m.sum():
            print("      %-10s %4d de %-4d   %5.1f %%"
                  % (BONITO.get(c, c), ok[m].sum(), m.sum(), 100.0 * ok[m].mean()))
    print()

    print("   FIGURA 2 -- error segun la fraccion apartada")
    for frac in (0.0, 0.10, 0.20, 0.30):
        k = max(1, int(round(n * (1 - frac))))
        print("      aparta el %2.0f %%   error %5.1f %%"
              % (100 * frac, 100.0 * (1 - ok[orden[:k]].mean())))
    print()
    print("   El punto de trabajo del articulo es el umbral 0,60.")
    print()


if __name__ == "__main__":
    main()
