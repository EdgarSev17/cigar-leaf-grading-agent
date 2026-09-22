# -*- coding: utf-8 -*-
"""REPRODUCE THE ARTICLE'S FIGURES. NO PHOTOGRAPHS NEEDED.

The plant's photographs are not published. The measured features are, and with
them this script reproduces **the same protocol the article reports**: six
retrainings of the chained-decision tree on the training set, each one evaluated
on the 112 leaves of the independent batch, which the model never saw. That is
6 x 112 = 672 decisions, which is the footnote of Table IV.

It reproduces, in one pass:

    Table IV    how many leaves the system decides and how well it does on them
    Table III   the breakdown of hits and errors by grade
    Figure 2    the error as a function of the fraction set aside
    and the 84.8 % with kappa 0.705 of the abstract

Usage:
    python code/reproduce.py
    python code/reproduce.py --seeds 12      more repetitions, steadier

WHY SIX RETRAININGS AND NOT ONE STORED MODEL. A single model gives a number that
depends on the partition it was fitted with; on this batch the model in
`out/modelo/` gives 86.6 %. The mean of six retrainings, 84.8 %, is the
defensible figure and the one the article reports. Both numbers are correct and
they measure different things.

A note on names: the grades keep the Spanish names the model was fitted with.
capa is wrapper, banda is binder, xl_izq is XL left, xr_der is XR right.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                   # noqa: E402
from rutas import REPO_ROOT                                          # noqa: E402
import arbol_habano as AH                                            # noqa: E402
import arbol_estandar as AE                                          # noqa: E402
import paper_datos as D                                              # noqa: E402

OUT = REPO_ROOT / "out"
ENGLISH = {"capa": "wrapper", "xl_izq": "XL left",
           "xr_der": "XR right", "banda": "binder"}


def kappa(truth, said, n):
    """Cohen's kappa, without depending on sklearn.metrics."""
    classes = sorted(set(truth) | set(said))
    ix = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)))
    for a, b in zip(truth, said):
        M[ix[a], ix[b]] += 1
    po = np.trace(M) / n
    pe = float((M.sum(0) * M.sum(1)).sum()) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Reproduce the article's figures.")
    ap.add_argument("--seeds", type=int, default=6,
                    help="retrainings (the article uses 6)")
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
    for seed in range(args.seeds):
        m = AH.entrena_nodos(Xtr, ytr, cols, seed)
        pr, cf = AH.decide(m, X112, cols)
        C.append(cf); K.append(pr == y112); P.append(pr); Y.append(y112)

    conf = np.concatenate(C)
    ok = np.concatenate(K).astype(bool)
    pred = np.concatenate(P)
    truth = np.concatenate(Y)
    n = len(conf)

    print()
    print("=" * 70)
    print("INDEPENDENT BATCH: %d decisions  (%d retrainings x %d leaves)"
          % (n, args.seeds, len(y112)))
    print("=" * 70)
    print()
    print("   accuracy, deciding every leaf   %5.1f %%      kappa %.3f"
          % (100.0 * ok.mean(), kappa(truth, pred, n)))
    print()

    print("   TABLE IV -- how many it decides and how well it does on them")
    print("      %-30s %8s %10s %9s" % ("rule", "decides", "accuracy", "defers"))
    print("      %-30s %7.1f %% %9.1f %% %8.1f %%"
          % ("decides every leaf", 100.0, 100.0 * ok.mean(), 0.0))
    for u in (0.60, 0.70, 0.80):
        m = conf >= u
        if m.sum():
            print("      %-30s %7.1f %% %9.1f %% %8.1f %%"
                  % ("confidence threshold %.2f" % u, 100.0 * m.mean(),
                     100.0 * ok[m].mean(), 100.0 * (1 - m.mean())))
    order = np.argsort(-conf, kind="stable")
    for frac in (0.10, 0.20):
        k = max(1, int(round(n * (1 - frac))))
        print("      %-30s %7.1f %% %9.1f %% %8.1f %%"
              % ("sets aside the least sure %.0f %%" % (100 * frac),
                 100.0 * k / n, 100.0 * ok[order[:k]].mean(),
                 100.0 * (1 - k / float(n))))
    print()

    print("   TABLE III -- by grade")
    for c in ("capa", "xl_izq", "xr_der", "banda"):
        m = truth == c
        if m.sum():
            print("      %-10s %4d of %-4d   %5.1f %%"
                  % (ENGLISH.get(c, c), ok[m].sum(), m.sum(), 100.0 * ok[m].mean()))
    print()

    print("   FIGURE 2 -- error by the fraction set aside")
    for frac in (0.0, 0.10, 0.20, 0.30):
        k = max(1, int(round(n * (1 - frac))))
        print("      sets aside %2.0f %%   error %5.1f %%"
              % (100 * frac, 100.0 * (1 - ok[order[:k]].mean())))
    print()
    print("   The article's operating point is the 0.60 threshold.")
    print()


if __name__ == "__main__":
    main()
