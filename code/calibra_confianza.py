r"""
LA CONFIANZA, CALIBRADA DE VERDAD (2026-09-06)  -- punto B.5 de "LO SIGUIENTE"

QUE PROBLEMA RESUELVE
---------------------
§60.5: la logistica de Connecticut **dice 0.91 y acierta el 74.7 %** -- le sobran
7.6 puntos de confianza. Y §60.7 dice por que importa: el agente no tiene que
acertar siempre, tiene que **saber cuando no sabe**. Hoy resuelve solo el 65 % de
las hojas del conjunto (y solo el 22 % de las hojas nuevas de §62), y **cada punto
de confianza que se pueda creer es una hoja menos que va a revision manual**.

LO QUE HAY HOY, Y POR QUE NO ES SUFICIENTE
------------------------------------------
`entrena.py` ya no se cree la probabilidad a pelo: la mete en una tabla de **cinco
tramos** (0-0.5, 0.5-0.7, 0.7-0.85, 0.85-0.95, 0.95-1) y usa el **acierto medido**
de ese tramo. Eso ya es una calibracion, pero de grano muy grueso: un tramo entero
pasa o no pasa. Una hoja con 0.86 y otra con 0.94 reciben la misma estimacion.

QUE SE PRUEBA AQUI
------------------
Cuatro formas de convertir la probabilidad declarada en un **acierto estimado**:

    declarada   se cree el numero tal cual (la trampa de §60.5, como control)
    tramos      los cinco tramos de hoy
    isotonica   regresion monotona conf -> acierto  (grano fino, sin forma fija)
    platt       logistica sobre logit(conf)         (grano fino, forma de S)

Y se comparan por lo unico que decide en planta:
    - **error de calibracion**: distancia entre lo estimado y lo que se acierta;
    - **cuantas hojas pasa solas** con el umbral de acierto estimado >= 90 %;
    - **y cuanto acierta DE VERDAD en esas**, que es la comprobacion que impide
      hacer trampa: un metodo que suba el % de hojas automaticas bajando el
      acierto real por debajo del 90 % no ha mejorado nada, ha mentido mejor.

LA REGLA DE HONESTIDAD (§ B.5: "dentro de la validacion cruzada, nunca sobre todas
las hojas") -- POR QUE ESTE GUION ES ANIDADO
--------------------------------------------------------------------------------
Ajustar la calibracion con las mismas predicciones con las que luego se mide da un
numero inflado por construccion: el calibrador ya ha visto que hojas se fallan. Asi
que la particion es **anidada**:

    externa (5 grupos por hoja)  -> da las hojas sobre las que se MIDE
      interna (4 grupos, solo con el trozo de entrenamiento) -> da las
      predicciones fuera de muestra con las que se AJUSTA el calibrador

El calibrador nunca ve la hoja que va a puntuar, ni directamente ni por su gemela:
los grupos son `hoja_id` en las dos particiones, asi que las dos fotos de una hoja
caen siempre del mismo lado (§ la regla de siempre).

Se repite con N semillas y se reporta media y rango (§45.2).

Uso:  python scripts/calibra_confianza.py [n_semillas]     (por defecto 10)
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from rutas import REPO_ROOT  # repository root

OUT = REPO_ROOT / "out"
RAIZ = REPO_ROOT / "dataset"
N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 10
UMBRAL = 0.90          # el mismo de §60.7: acierto estimado para pasar sola
MARGEN = 0.02          # coeficiente de seguridad del corte por conjunto (§63)

DEF = ["n_agujeros", "n_roturas", "n_defectos", "area_def_pulg2", "frac_area_def",
       "mayor_pulg2", "mayor_dist_base_pulg", "area_util_pulg2",
       "area_util_izq_pulg2", "area_util_der_pulg2", "n_util_izq", "n_util_der"]
MAN = ["frac_verde", "frac_blanca", "frac_negra",
       "verde_izq_pulg2", "verde_der_pulg2", "blanca_izq_pulg2", "blanca_der_pulg2",
       "negra_izq_pulg2", "negra_der_pulg2", "mancha_izq_pulg2", "mancha_der_pulg2",
       "asimetria_mancha", "mayor_mancha_pulg2"]
PEN = ["pen_mordida_izq_pulg", "pen_mordida_der_pulg", "pen_mordida_pulg",
       "asim_pen_mordida"]
TRAMOS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01))


def lee(n):
    f = OUT / n
    return ({} if not f.exists()
            else {r["ruta"]: r for r in csv.DictReader(open(f, encoding="utf-8"))})


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def mk(nombre, sem):
    if nombre == "logistica":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=4000,
                                                class_weight="balanced"))
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(max_iter=300,
                                                        random_state=sem))


# ---------------------------------------------------------------- los cuatro
def cal_declarada(conf_in, ok_in):
    return lambda c: np.clip(c, 0, 1)


def cal_tramos(conf_in, ok_in):
    tabla = []
    for lo, hi in TRAMOS:
        s = (conf_in >= lo) & (conf_in < hi)
        if s.sum() >= 10:
            tabla.append((lo, hi, float(ok_in[s].mean())))

    def f(c):
        out = np.full(len(c), np.nan)
        for lo, hi, a in tabla:
            out[(c >= lo) & (c < hi)] = a
        # un tramo con menos de 10 casos no tiene estimacion: no pasa sola
        out[~np.isfinite(out)] = 0.0
        return out
    return f


def cal_isotonica(conf_in, ok_in):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(conf_in, ok_in.astype(float))
    return lambda c: np.clip(iso.predict(c), 0, 1)


def cal_platt(conf_in, ok_in):
    z = np.log(np.clip(conf_in, 1e-6, 1 - 1e-6)
               / (1 - np.clip(conf_in, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    if len(set(ok_in.tolist())) < 2:            # todo aciertos o todo fallos
        p = float(ok_in.mean())
        return lambda c: np.full(len(c), p)
    lr.fit(z, ok_in.astype(int))

    def f(c):
        zz = np.log(np.clip(c, 1e-6, 1 - 1e-6)
                    / (1 - np.clip(c, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return np.clip(lr.predict_proba(zz)[:, 1], 0, 1)
    return f


def cal_conjunto(conf_in, ok_in):
    """Acierto del CONJUNTO que pasaria con este corte, no de la hoja.

    POR QUE HACE FALTA UN QUINTO METODO, Y ES EL QUE CAMBIA EL RESULTADO
    -------------------------------------------------------------------
    Los cuatro de arriba estiman "que acierto tiene UNA hoja con esta
    confianza", y la regla de §60.7 los usa para exigir que **cada tramo**
    llegue al 90 %. Pero lo que promete el sistema en planta no es eso: es que
    **el lote que pasa solo** acierte el 90 %. Exigirselo a cada tramo por
    separado es mas duro que la promesa, y por eso los otros metodos entregan
    95-97 % cuando solo hacia falta 90 -- o sea que estan mandando a revision
    hojas que podian pasar.

    Aqui la estimacion es la **acumulada**: para un corte c, el acierto de
    todas las hojas con confianza >= c, medido en la parte interna. Es
    exactamente la cantidad que aparece en la promesa.
    """
    o = np.argsort(-conf_in)
    c_ord, ok_ord = conf_in[o], ok_in[o].astype(float)
    cum = np.cumsum(ok_ord) / np.arange(1, len(ok_ord) + 1)
    n_min = min(20, len(cum))            # con menos de 20 hojas la acumulada es ruido

    def f(c):
        k = np.searchsorted(-c_ord, -np.asarray(c), side="right")
        k = np.clip(k - 1, n_min - 1, len(cum) - 1)
        return cum[k]
    return f


def cal_conjunto_margen(conf_in, ok_in):
    """El corte por conjunto, con MARGEN.

    Medido el 06/09: el corte sin margen entrega 89.5 % cuando promete 90 --
    se pasa medio punto, porque la acumulada de la parte interna es ella misma
    una estimacion con ruido y el corte cae justo en el filo. Se le pide al
    trozo interno un {:.0f} % para prometer el 90 %. Es la misma idea que un
    coeficiente de seguridad: el precio se paga en cobertura, no en promesas
    incumplidas.
    """
    base = cal_conjunto(conf_in, ok_in)
    return lambda c: np.clip(base(c) - MARGEN, 0, 1)


METODOS = [("declarada (control)", cal_declarada),
           ("tramos (lo de hoy)", cal_tramos),
           ("isotonica", cal_isotonica),
           ("platt", cal_platt),
           ("corte por conjunto", cal_conjunto),
           ("corte por conjunto +margen", cal_conjunto_margen)]


def error_calibracion(est, ok, n_bins=10):
    """Distancia media entre lo estimado y lo acertado, pesada por hojas.

    Se agrupa por DECILES de lo estimado, no por tramos fijos: si se usaran los
    tramos de `cal_tramos` el metodo de tramos saldria ganador por construccion.
    """
    if len(est) == 0:
        return float("nan")
    bordes = np.quantile(est, np.linspace(0, 1, n_bins + 1))
    bordes[0], bordes[-1] = -np.inf, np.inf
    err, n = 0.0, len(est)
    for i in range(n_bins):
        s = (est >= bordes[i]) & (est < bordes[i + 1])
        if s.sum() == 0:
            continue
        err += (s.sum() / n) * abs(est[s].mean() - ok[s].mean())
    return err


def main():
    man, agu, mch = lee("manifiesto_limpio.csv"), lee("agujeros.csv"), lee("manchas.csv")
    ras, seg, zon = (lee("clasificador_rasgos.csv"), lee("segmentacion.csv"),
                     lee("zonas.csv"))

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

    X = []
    for r in rutas:
        v = [num(ras[r][c]) for c in cols_h]
        a0, m0 = agu.get(r), mch.get(r)
        v += ([num(a0.get(k)) for k in DEF] if a0 else [np.nan] * len(DEF))
        if a0:
            iz = num(a0.get("area_util_izq_pulg2"))
            de = num(a0.get("area_util_der_pulg2"))
            v += [(iz - de) / (iz + de + 1e-6), abs(iz - de)]
        else:
            v += [np.nan, np.nan]
        v += ([num(m0.get(k)) for k in MAN] if m0 else [np.nan] * len(MAN))
        v += ([num(a0.get(k)) for k in PEN] if a0 else [np.nan] * len(PEN))
        X.append(v)
    X = np.array(X, float)
    y = np.array([man[r]["clase"] for r in rutas])
    var_ = np.array([man[r]["variedad"] for r in rutas])
    grp = np.array([man[r]["hoja_id"] for r in rutas])

    lineas = []

    def di(s=""):
        print(s, flush=True)
        lineas.append(s)

    di("hojas: {}   rasgos: {}   semillas: {}".format(len(y), X.shape[1], N_SEM))
    di("calidad: {}".format(dict(sorted(Counter(y).items()))))
    di()
    di("=" * 78)
    di("LA CONFIANZA, CALIBRADA DENTRO DE LA VALIDACION CRUZADA (anidada)")
    di("=" * 78)

    resumen = {}
    for v in ("connecticut", "habano"):
        sel = var_ == v
        Xv, yv, gv = X[sel], y[sel], grp[sel]
        di()
        di("-" * 78)
        di("{}  ({} hojas)".format(v.upper(), int(sel.sum())))
        di("-" * 78)

        # acumuladores por metodo: (estimado, acierto) de todas las semillas
        acu = {nm: ([], []) for nm, _ in METODOS}
        for sem in range(N_SEM):
            k = min(5, min(len(set(gv[yv == c])) for c in set(yv)))
            ext = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=sem)
            for tr, te in ext.split(Xv, yv, groups=gv):
                mod = mk("logistica", sem).fit(Xv[tr], yv[tr])
                pp = mod.predict_proba(Xv[te])
                conf_te = pp.max(1)
                pred_te = mod.classes_[pp.argmax(1)]
                ok_te = (pred_te == yv[te])

                # --- interna: predicciones fuera de muestra SOLO del trozo de
                #     entrenamiento, para ajustar el calibrador sin ver el test
                ki = min(4, min(len(set(gv[tr][yv[tr] == c])) for c in set(yv[tr])))
                itr = StratifiedGroupKFold(n_splits=max(2, ki), shuffle=True,
                                           random_state=sem)
                pp_in = cross_val_predict(mk("logistica", sem), Xv[tr], yv[tr],
                                          cv=itr, groups=gv[tr],
                                          method="predict_proba")
                pr_in = cross_val_predict(mk("logistica", sem), Xv[tr], yv[tr],
                                          cv=itr, groups=gv[tr])
                conf_in, ok_in = pp_in.max(1), (pr_in == yv[tr])

                for nm, fabrica in METODOS:
                    est = fabrica(conf_in, ok_in)(conf_te)
                    acu[nm][0].append(est)
                    acu[nm][1].append(ok_te)
            print("   [{}] semilla {}/{} lista".format(v[:4], sem + 1, N_SEM),
                  flush=True)

        di("   {:27s} {:>9s} {:>10s} {:>12s} {:>12s}"
           .format("", "error", "pasa sola", "acierta ahi", "y en el resto"))
        for nm, _ in METODOS:
            est = np.concatenate(acu[nm][0])
            ok = np.concatenate(acu[nm][1])
            # el corte por conjunto no estima la hoja sino el lote: el error por
            # hoja no le corresponde y se deja en blanco a proposito
            err = (float("nan") if nm.startswith("corte por conjunto")
                   else error_calibracion(est, ok))
            auto = est >= UMBRAL
            f_auto = float(auto.mean())
            ac_auto = float(ok[auto].mean()) if auto.any() else float("nan")
            ac_rev = float(ok[~auto].mean()) if (~auto).any() else float("nan")
            aviso = ""
            if auto.any() and ac_auto < UMBRAL - 0.005:
                aviso = "  [!] promete {:.0f} % y da {:.1f} %".format(
                    100 * UMBRAL, 100 * ac_auto)
            di("   {:27s} {:>7s} {:9.1f} % {:11.1f} % {:11.1f} %{}"
               .format(nm, "-" if err != err else "{:5.1f} p".format(100 * err),
                       100 * f_auto, 100 * ac_auto, 100 * ac_rev, aviso))
            resumen[(v, nm)] = (err, f_auto, ac_auto, ac_rev, len(est))

        # --- LA CURVA, que es lo que ningun calibrador puede mover -------------
        # Toda recalibracion monotona (isotonica, platt, tramos) deja el ORDEN de
        # las hojas igual: solo mueve el corte. Asi que esta curva es la misma
        # para los cuatro, y es el techo de lo que se puede prometer.
        conf = np.concatenate(acu["declarada (control)"][0])
        ok_c = np.concatenate(acu["declarada (control)"][1])
        o = np.argsort(-conf)
        ok_o = ok_c[o]
        di()
        di("   la curva de cobertura -- IGUAL para todo metodo monotono,"
           " solo cambia el corte:")
        for cob in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            k = max(1, int(round(cob * len(ok_o))))
            di("      pasando las {:3.0f} % mas seguras -> acierto real {:5.1f} %"
               .format(100 * cob, 100 * ok_o[:k].mean()))

    # ------------------------------------------------------------------ conjunto
    di()
    di("=" * 78)
    di("EN CONJUNTO (las dos variedades, pesadas por hojas)")
    di("=" * 78)
    di("   {:27s} {:>9s} {:>10s} {:>12s}".format("", "error", "pasa sola",
                                                 "acierta ahi"))
    for nm, _ in METODOS:
        n_t = sum(resumen[(v, nm)][4] for v in ("connecticut", "habano"))
        err = sum(resumen[(v, nm)][0] * resumen[(v, nm)][4]
                  for v in ("connecticut", "habano")) / n_t
        f_a = sum(resumen[(v, nm)][1] * resumen[(v, nm)][4]
                  for v in ("connecticut", "habano")) / n_t
        ac_a = sum(resumen[(v, nm)][1] * resumen[(v, nm)][2] * resumen[(v, nm)][4]
                   for v in ("connecticut", "habano")) / (n_t * f_a) if f_a else float("nan")
        di("   {:27s} {:>7s} {:9.1f} % {:11.1f} %"
           .format(nm, "-" if err != err else "{:5.1f} p".format(100 * err),
                   100 * f_a, 100 * ac_a))
    di()
    di("COMO SE LEE, Y QUE NO VALE")
    di("-" * 78)
    di("  'pasa sola' solo cuenta si 'acierta ahi' llega al {:.0f} %. Un metodo que"
       .format(100 * UMBRAL))
    di("  suba el primero hundiendo el segundo no ha calibrado nada: ha aprendido a")
    di("  mentir con mas resolucion, que es exactamente lo de §60.5 otra vez.")
    di("  El error esta medido por DECILES de lo estimado, no por los tramos de hoy,")
    di("  para no darle ventaja al metodo de tramos por construccion.")
    di("  Y todo es anidado: el calibrador nunca vio la hoja que puntua.")
    di()
    di("  LO QUE NINGUN CALIBRADOR PUEDE HACER, y conviene tenerlo claro antes de")
    di("  pedirle mas: isotonica, platt y tramos son transformaciones MONOTONAS de")
    di("  la confianza, asi que dejan el ORDEN de las hojas exactamente igual. No")
    di("  pueden acertar en una hoja mas; solo mueven DONDE se corta. Lo unico que")
    di("  sube el % de hojas automaticas es cortar en el sitio correcto -- que es lo")
    di("  que hace el corte por conjunto -- o un modelo mejor.")

    f = OUT / "CALIBRA_CONFIANZA.txt"
    f.write_text("\n".join(lineas), encoding="utf-8")
    print("\n-> {}".format(f))


if __name__ == "__main__":
    main()
