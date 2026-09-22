r"""
LAS CIFRAS DEL SISTEMA, CON RANGO. El unico sitio del que se citan. (2026-09-04)

POR QUE EXISTE ESTE GUION
-------------------------
`clasificador.py` imprime UNA particion. PROGRESO 45.2 midio lo que eso vale: con
los mismos datos y la misma configuracion, cambiar la particion mueve **3.5 puntos
en las cinco calidades y 5.3 en el par**. O sea que **ninguna cifra suelta de ese
informe es citable**, ni siquiera la del bloque final.

Este guion repite la validacion cruzada con N semillas y da **media y rango**. Es lo
que va al paper y lo que va en la cabecera de PROGRESO. Cuando cambie algo del
sistema, se vuelve a correr esto -- no se lee del log del clasificador.

Se reporta cada bloque junto a su **control de fondo** (el modelo que no mira la
hoja) porque es lo que hace defendible el numero, y junto a su **linea base** (la
clase mas comun), porque sin ella un porcentaje no significa nada.

Uso:  python scripts/cifras.py [n_semillas]
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from rutas import REPO_RAIZ  # raiz del repositorio

OUT = REPO_RAIZ / "out"
RAIZ = REPO_RAIZ / "dataset"
N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 15

DEF = ["n_agujeros", "n_roturas", "n_defectos", "area_def_pulg2", "frac_area_def",
       "mayor_pulg2", "mayor_dist_base_pulg", "area_util_pulg2",
       "area_util_izq_pulg2", "area_util_der_pulg2", "n_util_izq", "n_util_der"]
MAN = ["frac_verde", "frac_blanca", "frac_negra",
       "verde_izq_pulg2", "verde_der_pulg2", "blanca_izq_pulg2", "blanca_der_pulg2",
       "negra_izq_pulg2", "negra_der_pulg2", "mancha_izq_pulg2", "mancha_der_pulg2",
       "asimetria_mancha", "mayor_mancha_pulg2"]
PEN = ["pen_mordida_izq_pulg", "pen_mordida_der_pulg", "pen_mordida_pulg",
       "asim_pen_mordida"]


def lee(n):
    f = OUT / n
    return ({} if not f.exists()
            else {r["ruta"]: r for r in csv.DictReader(open(f, encoding="utf-8"))})


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


man, agu = lee("manifiesto_limpio.csv"), lee("agujeros.csv")
mch, ras = lee("manchas.csv"), lee("clasificador_rasgos.csv")
seg, zon = lee("segmentacion.csv"), lee("zonas.csv")

omitidas = set()
f_dud = RAIZ / "CASOS_DUDOSOS.csv"
if f_dud.exists():
    omitidas = {r["ruta"] for r in csv.DictReader(open(f_dud, encoding="utf-8"))
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

cols_h = [c for c in next(iter(ras.values())).keys()
          if c not in ("ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id")
          and not c.startswith("fondo_")]
cols_f = [c for c in next(iter(ras.values())).keys() if c.startswith("fondo_")]

X, XF = [], []
for r in rutas:
    v = [num(ras[r][c]) for c in cols_h]
    a0, m0 = agu.get(r), mch.get(r)
    v += ([num(a0.get(k)) for k in DEF] if a0 else [np.nan] * len(DEF))
    if a0:
        iz, de = num(a0.get("area_util_izq_pulg2")), num(a0.get("area_util_der_pulg2"))
        v += [(iz - de) / (iz + de + 1e-6), abs(iz - de)]
    else:
        v += [np.nan, np.nan]
    v += ([num(m0.get(k)) for k in MAN] if m0 else [np.nan] * len(MAN))
    v += ([num(a0.get(k)) for k in PEN] if a0 else [np.nan] * len(PEN))
    X.append(v)
    XF.append([num(ras[r][c]) for c in cols_f])
X, XF = np.array(X, float), np.array(XF, float)
y = np.array([man[r]["clase"] for r in rutas])
var_ = np.array([man[r]["variedad"] for r in rutas])
grp = np.array([man[r]["hoja_id"] for r in rutas])

print("hojas: {}   rasgos de hoja: {}   de fondo: {}"
      .format(len(y), X.shape[1], XF.shape[1]))
print("reparto: {}".format(dict(sorted(Counter(y).items()))))
if omitidas:
    print("omitidas por decision del experto (PROGRESO 50): {}".format(len(omitidas)))
print("semillas de particion: {}".format(N_SEM))


def mk(nombre, sem):
    if nombre == "logistica":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=4000,
                                                class_weight="balanced"))
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(max_iter=300,
                                                        random_state=sem))


def corre(XX, mask, sem):
    yy, gg = y[mask], grp[mask]
    k = min(5, min(len(set(gg[yy == c])) for c in set(yy)))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=sem)
    mejor_ac, mejor_pr = -1.0, None
    for nm in ("logistica", "arboles"):
        pr = cross_val_predict(mk(nm, sem), XX[mask], yy, cv=cv, groups=gg)
        ac = float((pr == yy).mean())
        if ac > mejor_ac:
            mejor_ac, mejor_pr = ac, pr
    return mejor_ac, mejor_pr


def rng(v):
    a = np.array(v, float)
    return "{:5.1f} %  [{:4.1f} - {:4.1f}]".format(100 * a.mean(), 100 * a.min(),
                                                   100 * a.max())


BLOQUES = [("las cinco calidades", np.ones(len(y), bool)),
           ("cinco calidades, solo Connecticut", var_ == "connecticut"),
           ("cinco calidades, solo Habano", var_ == "habano"),
           ("el par XL izq / XR der", np.isin(y, ["xl_izq", "xr_der"]))]

print()
print("=" * 78)
print("LAS CIFRAS, CON {} SEMILLAS DE PARTICION".format(N_SEM))
print("=" * 78)
print()
print("   {:36s} {:>5s} {:>7s} {:>20s}"
      .format("", "n", "base", "exactitud"))
print("   " + "-" * 72)
for etq, mask in BLOQUES:
    yy = y[mask]
    base = max(Counter(yy).values()) / len(yy)
    ah, ab, af = [], [], []
    for sem in range(N_SEM):
        a, pr = corre(X, mask, sem)
        ah.append(a)
        ab.append(balanced_accuracy_score(yy, pr))
        af.append(corre(XF, mask, sem)[0])
    print("   {:36s} {:5d} {:6.1f} % {:>20s}"
          .format(etq, int(mask.sum()), 100 * base, rng(ah)))
    print("   {:36s} {:5s} {:8s} {:>20s}".format("   balanceada", "", "", rng(ab)))
    print("   {:36s} {:5s} {:8s} {:>20s}"
          .format("   CONTROL: solo el fondo", "", "", rng(af)))
    d = 100 * (np.mean(ah) - np.mean(af))
    ok = "supera al fondo por {:.1f} puntos".format(d) if d > 0 else "[!] NO supera al fondo"
    print("   {:36s} {}".format("", ok))
    print()
print("   Recordatorio de PROGRESO 45.2: el rango de arriba NO es un intervalo de")
print("   confianza estadistico, es lo que se mueve el numero al cambiar la")
print("   particion con los MISMOS datos. Cualquier comparacion antes/despues")
print("   menor que esa anchura no significa nada.")
