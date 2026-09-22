# -*- coding: utf-8 -*-
"""PARTITIONS SAVED TO DISK, NOT REGENERATED EACH TIME.

The course asks for this in as many words: *"la particion guardada en disco. No
se regenera cada vez, se guarda la lista de indices o de archivos. Asi el split
de la Semana 1 es el mismo de la Semana N."*

Fixing the seed already makes the split deterministic, but it makes it depend on
the version of scikit-learn, on the order the rows happen to be in, and on
nobody ever touching the call. Writing it down removes all three.

What is written. One JSON file per partition under `results/particiones/`,
holding **the leaf identifiers** of each fold, not row numbers. Leaf ids survive
a reordering of the tables and can be read by a person; row numbers cannot.

    {
      "nombre": "cifras_5calidades_sem0",
      "n_splits": 5,
      "random_state": 0,
      "n_filas": 627,
      "n_grupos": 527,
      "huella_grupos": "3f2a…",          the group list this was built for
      "folds": [{"test": ["hoja_0001", …]}, …]
    }

How it is used. `split()` is a drop-in for `StratifiedGroupKFold(...).split(...)`.
The first run computes the folds and saves them; every run after that reads them
back. If the data changed underneath, the fingerprint no longer matches and it
says so instead of quietly giving different numbers.
"""
import hashlib
import io
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from rutas import REPO_ROOT

DIR = REPO_ROOT / "results" / "particiones"


def _huella(grupos):
    """A fingerprint of the group list, to notice if the data changed."""
    h = hashlib.sha256()
    for g in grupos:
        h.update(str(g).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


def split(X, y, groups, n_splits, random_state, prefijo, guardar=True):
    """Like StratifiedGroupKFold(...).split(...), but from disk.

    Yields (train_idx, test_idx) as numpy arrays, in the saved order.

    The file is named after the caller, the number of folds, the seed and a
    fingerprint of the group list, so a caller that splits several different
    subsets gets one file each without having to name them.
    """
    groups = list(groups)
    huella = _huella(groups)
    nombre = "%s_k%d_sem%d_%s" % (prefijo, n_splits, random_state, huella)
    fichero = DIR / ("%s.json" % nombre)

    if fichero.exists():
        d = json.loads(io.open(fichero, encoding="utf-8").read())
        if d.get("huella_grupos") != huella:
            raise SystemExit(
                "The saved partition %s was built for different data.\n"
                "   saved:  %s\n   now:    %s\n"
                "Delete %s to rebuild it, and expect the figures to move."
                % (nombre, d.get("huella_grupos"), huella, fichero))
        for f in d["folds"]:
            prueba = set(f["test"])
            te = np.array([i for i, g in enumerate(groups) if str(g) in prueba], int)
            tr = np.array([i for i, g in enumerate(groups) if str(g) not in prueba], int)
            yield tr, te
        return

    # First time: compute, hand back, and write down.
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
    folds, salida = [], []
    for tr, te in cv.split(X, y, groups=groups):
        folds.append({"test": sorted({str(groups[i]) for i in te})})
        salida.append((tr, te))

    if guardar:
        os.makedirs(str(DIR), exist_ok=True)
        d = {
            "nombre": nombre,
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "n_filas": len(groups),
            "n_grupos": len(set(map(str, groups))),
            "huella_grupos": huella,
            "folds": folds,
        }
        io.open(fichero, "w", encoding="utf-8").write(
            json.dumps(d, indent=1, ensure_ascii=False) + "\n")

    for tr, te in salida:
        yield tr, te


def existe(nombre):
    return (DIR / ("%s.json" % nombre)).exists()


class Particion(object):
    """A drop-in replacement for StratifiedGroupKFold, reading from disk.

    Works both ways the original is used: handed to `cross_val_predict` as
    `cv=`, and called directly as `.split(X, y, groups)`.
    """

    def __init__(self, n_splits, random_state, prefijo):
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.prefijo = prefijo

    def split(self, X, y=None, groups=None):
        return split(X, y, groups, self.n_splits, self.random_state, self.prefijo)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
