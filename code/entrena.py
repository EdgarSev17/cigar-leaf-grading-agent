r"""
ENTRENA Y GUARDA EL MODELO DE DOS ETAPAS. (2026-09-05)

POR QUE HACE FALTA ESTE GUION
-----------------------------
Hasta hoy **no existia ningun modelo guardado** (PROGRESO 54.4). `clasificador.py`
y `dos_etapas.py` entrenan y evaluan en la misma corrida y tiran el modelo al
terminar: sirven para MEDIR, no para RESPONDER. Sin un modelo en disco no hay
forma de contestar a una foto nueva, que es lo que pidel experto:

    "la idea es probarlo con hojas nuevas, poner una camara y que el detector
     sea capaz de detectar que clase de hoja es solo con verla"

Este guion es la mitad que faltaba: deja en `out/modelo/` los tres modelos ya
entrenados con TODAS las hojas, y la lista ordenada de rasgos que espera cada uno.
`clasifica.py` los carga y no vuelve a entrenar nada.

LA ARQUITECTURA ES LA DEL EXPERTO (PROGRESO 55), NO SE ELIGE AQUI
--------------------------------------------------------------
    etapa 1   variedad   Habano / Connecticut
    etapa 2   calidad    con el modelo de ESA variedad

Medido en `dos_etapas.py`: 84.7 % de punta a punta contra 80.7 % del modelo unico.

QUE SE DECIDE AQUI, Y COMO
--------------------------
1. **Que familia usa cada modelo.** `dos_etapas.py` corre las dos (logistica y
   arboles) y se queda con la mejor de esa particion -- eso vale para medir el
   techo, pero un modelo guardado tiene que ser UNO. Se elige por exactitud media
   en validacion cruzada agrupada por hoja sobre N semillas, y se escribe cual
   gano y por cuanto. Si la diferencia es menor que el ruido de particion
   (PROGRESO 45.2: 3.5 puntos en cinco clases) se dice explicitamente, porque
   entonces la eleccion es de conveniencia, no de merito.

2. **Que significa la confianza.** `predict_proba` da un numero entre 0 y 1 que no
   es una probabilidad hasta que se comprueba. Se mide con las predicciones fuera
   de muestra: para cada tramo de confianza, cuantas veces acierta de verdad. Esa
   tabla se guarda con el modelo y `clasifica.py` la imprime. Un agente que dice
   "XL izquierdo, confianza 0.9" cuando en ese tramo acierta el 60 % esta
   mintiendo, y la tesis es un agente que decide bajo confianza.

LO QUE ESTE GUION NO ARREGLA, Y HAY QUE SEGUIR DICIENDO
-------------------------------------------------------
El modelo se entrena con **una sola sesion de fotos** en la que la clase y la hora
del dia van juntas (PROGRESO 54.2). Guardarlo no lo hace mas transferible: sigue
sin poderse prometer que estas cifras se repitan en una captura nueva. Lo que
decide eso es la sesion intercalada de PROGRESO 54.3, y no depende de este codigo.

Uso:  python scripts/entrena.py [n_semillas]      (por defecto 10)
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import rejilla as RJ
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
import joblib
from rutas import REPO_RAIZ  # raiz del repositorio

OUT = REPO_RAIZ / "out"
RAIZ = REPO_RAIZ / "dataset"
DIR_MODELO = OUT / "modelo"
# Por debajo de esto la acumulada de un cajon es ruido y el cajon se queda sin
# curva propia: entonces no se le veta nada y decide la regla de siempre.
MIN_CAJON = 25
N_SEM = int(sys.argv[1]) if (len(sys.argv) > 1
                             and sys.argv[1].isdigit()) else 10

# Las mismas familias de rasgos que `dos_etapas.py`. Se repiten aqui a proposito:
# el modelo guardado tiene que llevar SU lista de rasgos dentro, para que
# `clasifica.py` no pueda construir el vector en otro orden sin que se note.
DEF = ["n_agujeros", "n_roturas", "n_defectos", "area_def_pulg2", "frac_area_def",
       "mayor_pulg2", "mayor_dist_base_pulg", "area_util_pulg2",
       "area_util_izq_pulg2", "area_util_der_pulg2", "n_util_izq", "n_util_der"]
MAN = ["frac_verde", "frac_blanca", "frac_negra",
       "verde_izq_pulg2", "verde_der_pulg2", "blanca_izq_pulg2", "blanca_der_pulg2",
       "negra_izq_pulg2", "negra_der_pulg2", "mancha_izq_pulg2", "mancha_der_pulg2",
       "asimetria_mancha", "mayor_mancha_pulg2"]
PEN = ["pen_mordida_izq_pulg", "pen_mordida_der_pulg", "pen_mordida_pulg",
       "asim_pen_mordida"]
COLS_PUNTA = ["punta_0.25", "punta_0.5", "punta_1", "punta_2"]
# EL SUDADO (2026-09-15). Es uno de los DOS ejes de la taxonomia, no un rasgo
# mas: capa y banda se diferencian SOLO en esto, igual que XL/XR y 1/2 banda
# (`dataset/Hojas sudadas/EXPLICACION.txt`). Se lee del CSV medido con el borde
# de la hoja FUERA (`sudada_ero.csv`), que es lo mismo que mide `rasgos_foto.py`
# al clasificar.
SUD = ["sud_frac", "sud_n", "sud_dens", "sud_verdor_p95", "sud_verdor_med"]


def lee(n):
    f = OUT / n
    return ({} if not f.exists()
            else {r["ruta"]: r for r in csv.DictReader(open(f, encoding="utf-8"))})


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def acumulada(conf, ok, n_puntos=60):
    """La curva que decide DONDE se corta (2026-09-06, seccion 63).

    Para cada corte c: que acierto tiene EL LOTE de hojas con confianza >= c.
    No es lo mismo que el acierto de una hoja con confianza c, y la diferencia
    es justo la que sobraba: la tabla de tramos exige que **cada tramo** llegue
    al 90 %, cuando lo que promete el sistema es que **el lote que pasa solo**
    llegue al 90 %. Pedirselo tramo a tramo es mas duro que la promesa, y por
    eso mandaba a revision hojas que podian pasar (medido: 57 % automatico
    contra 79 %, con el mismo modelo y sin bajar del 90 %).

    AVISO DE HONESTIDAD. Esta curva se ajusta con TODAS las predicciones fuera
    de muestra, que es lo correcto para el modelo que se guarda. Lo que ese
    procedimiento entrega de verdad se mide aparte y ANIDADO, en
    `calibra_confianza.py` -- ahi el calibrador nunca ve la hoja que puntua.
    """
    o = np.argsort(-conf)
    c_ord, ok_ord = conf[o], ok[o].astype(float)
    cum = np.cumsum(ok_ord) / np.arange(1, len(ok_ord) + 1)
    n_min = min(20, len(cum))       # con menos de 20 hojas la acumulada es ruido
    idx = np.unique(np.linspace(n_min - 1, len(cum) - 1, n_puntos).astype(int))
    return [dict(conf=float(c_ord[i]), acierto=float(cum[i]),
                 cobertura=float((i + 1) / len(cum))) for i in idx]


def mk(nombre, sem):
    if nombre == "logistica":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=4000,
                                                class_weight="balanced"))
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(max_iter=300,
                                                        random_state=sem))


FORZAR_FAMILIA = None      # lo pone main() desde --familia


def main():
    global FORZAR_FAMILIA
    import argparse as _ap
    _a = _ap.ArgumentParser()
    _a.add_argument("--sin-clase", default="", dest="sin_clase",
                    help="clases a dejar FUERA del modelo, separadas por coma. "
                         "Decision del experto el 2026-09-14: el modelo final va "
                         "SIN media_banda, y esa clase la entrena el aparte. "
                         "Ver Tabaco 92. Las hojas no se borran de ningun sitio: "
                         "solo no entran a este entrenamiento.")
    _a.add_argument("--rejilla", action="store_true",
                    help="etapa 2 como REJILLA de dos preguntas en vez de una "
                         "eleccion entre cuatro. Medido el 15/09: fuera de "
                         "sesion BAJA de 82.1 a 75.0 y NO impide la confusion "
                         "que buscaba evitar, porque la pregunta de los "
                         "agujeros tambien es un modelo y falla. Queda apagada.")
    _a.add_argument("--familia", default="auto",
                    choices=["auto", "logistica", "arboles"],
                    help="fija la familia de las etapas de CALIDAD en vez de "
                         "elegirla por exactitud. Ver Tabaco 89: un margen de "
                         "0.2 puntos sobre el umbral de 3.5 cambio el modelo "
                         "entero (97 -> 58 sobre las 100).")
    _args, _ = _a.parse_known_args()
    FORZAR_FAMILIA = None if _args.familia == "auto" else _args.familia
    if FORZAR_FAMILIA:
        print("FAMILIA FIJADA A MANO: {}".format(FORZAR_FAMILIA))
    man, agu, mch = lee("manifiesto_limpio.csv"), lee("agujeros.csv"), lee("manchas.csv")
    ras, seg, zon = (lee("clasificador_rasgos.csv"), lee("segmentacion.csv"),
                     lee("zonas.csv"))
    fpu = lee("forma_punta.csv")
    sud = lee("sudada_ero.csv")

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
    fuera = {c.strip() for c in _args.sin_clase.split(",") if c.strip()}
    if fuera:
        antes = len(rutas)
        rutas = [r for r in rutas if man[r]["clase"] not in fuera]
        print("CLASES FUERA DE ESTE MODELO: {}   ({} hojas de {} no entran; "
              "no se borra nada del dataset)".format(
                  ", ".join(sorted(fuera)), antes - len(rutas), antes))

    cols_h = [c for c in next(iter(ras.values())).keys()
              if c not in ("ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id")
              and not c.startswith("fondo_")]
    COLS_VAR = [c for c in cols_h if c.startswith("hoja_")] + \
               ["largo_pulg", "ancho_pulg", "aspecto"]

    # --- los NOMBRES, en el orden exacto en que se apilan los numeros.
    #     Es lo que se guarda y lo que `clasifica.py` tiene que rellenar.
    nom_cal = (list(cols_h) + list(DEF) + ["asim_area_util", "dif_area_util"]
               + list(MAN) + list(PEN) + list(SUD))
    nom_var = list(COLS_VAR) + [c + "_norm" for c in COLS_PUNTA]

    X, XV = [], []
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
        s0 = sud.get(r)
        v += ([num(s0.get(k)) for k in SUD] if s0 else [np.nan] * len(SUD))
        X.append(v)
        anc = num(ras[r].get("ancho_pulg"))
        vv = [num(ras[r][c]) for c in COLS_VAR]
        p0 = fpu.get(r, {})
        vv += [(num(p0.get(c)) / anc if (p0 and anc == anc and anc > 1) else np.nan)
               for c in COLS_PUNTA]
        XV.append(vv)
    X, XV = np.array(X, float), np.array(XV, float)
    y = np.array([man[r]["clase"] for r in rutas])
    var_ = np.array([man[r]["variedad"] for r in rutas])
    grp = np.array([man[r]["hoja_id"] for r in rutas])

    assert X.shape[1] == len(nom_cal), (X.shape[1], len(nom_cal))
    assert XV.shape[1] == len(nom_var), (XV.shape[1], len(nom_var))

    print("hojas: {}".format(len(y)))
    print("variedad: {}".format(dict(sorted(Counter(var_).items()))))
    print("calidad:  {}".format(dict(sorted(Counter(y).items()))))
    print("rasgos: calidad {}  variedad {}".format(len(nom_cal), len(nom_var)))
    print("semillas para elegir familia y medir la confianza: {}".format(N_SEM))

    # El mismo (rasgos, modelo, semilla) se pedia DOS veces --una para la exactitud
    # y otra dentro de `calibracion`--, y cada peticion son dos validaciones
    # cruzadas completas. Con arboles eso es la mitad del tiempo del guion, y en
    # esta maquina los procesos largos se mueren solos (paso el 06/09). Se guarda.
    _memo = {}
    CACHE = OUT / "_cache_oof"
    CACHE.mkdir(exist_ok=True)

    def oof(XX, yy, gg, nombre, sem, rej=False):
        """Predicciones fuera de muestra, guardadas EN DISCO.

        POR QUE EN DISCO Y NO SOLO EN MEMORIA (2026-09-06)
        --------------------------------------------------
        El 06/09 el sistema mato este guion DOS veces por falta de memoria, a
        mitad de la etapa 2, y las dos veces se perdio todo. Es exactamente lo
        que ya paso con `clasificador.py` y `segmentacion.py`, y la regla escrita
        entonces era: **los guiones largos guardan lo que van calculando y se
        reanudan**. Aqui la unidad es una validacion cruzada completa.

        La clave incluye el contenido de los rasgos y de las etiquetas, no su
        posicion: si cambia un rasgo, la cache no acierta y se recalcula sola.
        """
        clave_mem = (id(XX), id(yy), nombre, sem, rej)
        if clave_mem in _memo:
            return _memo[clave_mem]
        h = hashlib.md5()
        h.update(np.ascontiguousarray(np.nan_to_num(XX, nan=-9.87654321)).tobytes())
        h.update("|".join(map(str, yy)).encode())
        h.update("{}|{}|{}".format(nombre, sem, rej).encode())
        f = CACHE / "{}.npz".format(h.hexdigest())
        if f.exists():
            try:
                d = np.load(f, allow_pickle=False)
                pr, pp = d["pr"], d["pp"]
                _memo[clave_mem] = (pr, pp)
                return pr, pp
            except Exception:
                pass                      # cache corrupta: se recalcula
        k = min(5, min(len(set(gg[yy == c])) for c in set(yy)))
        part = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=sem)
        if rej:
            # LA REJILLA, FUERA DE MUESTRA. La calibracion tiene que medir el
            # modelo que se GUARDA; si aqui se midiera el plano, la tabla de
            # confianza describiria otro sistema y la politica de rechazo
            # estaria leyendo un numero que no es el suyo.
            clases = np.array(sorted(set(yy)))
            pp = np.zeros((len(yy), len(clases)), float)
            pr = np.empty(len(yy), dtype=object)
            for tr, te in part.split(XX, yy, gg):
                yag = RJ.reparte(yy[tr])
                m_ag = mk(nombre, sem).fit(XX[tr], yag)
                i1, i2 = yag == RJ.SIN, yag == RJ.CON
                m_sin = mk(nombre, sem).fit(XX[tr][i1], yy[tr][i1])
                m_con = mk(nombre, sem).fit(XX[tr][i2], yy[tr][i2])
                rj = RJ.Rejilla(m_ag, m_sin, m_con)
                P = rj.predict_proba(XX[te])
                col = [list(rj.classes_).index(c) for c in clases]
                pp[te] = P[:, col]
                pr[te] = clases[np.argmax(pp[te], axis=1)]
            pr = pr.astype("U20")
        else:
            pr = cross_val_predict(mk(nombre, sem), XX, yy, cv=part, groups=gg)
            pp = cross_val_predict(mk(nombre, sem), XX, yy, cv=part, groups=gg,
                                   method="predict_proba")
        np.savez_compressed(f, pr=pr.astype("U20"), pp=pp)
        _memo[clave_mem] = (pr, pp)
        return pr, pp

    def calibracion(XX, yy, gg, nombre, rej=False):
        """(tabla por tramos, error de calibracion) fuera de muestra.

        El error es la distancia media entre lo que el modelo DICE y lo que
        acierta de verdad, pesada por cuantas hojas caen en cada tramo.
        """
        conf, ok, dicho = [], [], []
        for s in range(N_SEM):
            pr, pp = oof(XX, yy, gg, nombre, s, rej)
            conf.append(pp.max(1))
            ok.append(pr == yy)
            dicho.append(pr)
        conf, ok = np.concatenate(conf), np.concatenate(ok)
        dicho = np.concatenate(dicho)
        tabla, err = [], 0.0
        for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)):
            sel = (conf >= lo) & (conf < hi)
            if sel.sum() < 10:
                continue
            tabla.append(dict(desde=lo, hasta=min(hi, 1.0), n=int(sel.sum()),
                              frac=float(sel.mean()),
                              dice=float(conf[sel].mean()),
                              acierto=float(ok[sel].mean())))
            err += float(sel.mean()) * abs(float(conf[sel].mean())
                                           - float(ok[sel].mean()))
        # La curva de CADA CAJON, o sea de cada clase que el modelo DICE
        # (2026-09-09, seccion 72.5). Hace falta para el veto: un cajon que no
        # llega al umbral ni en su punto mas seguro no puede automatizarse, por
        # confiada que venga la hoja. Se agrupa por lo que el modelo dice y no
        # por lo que la hoja es, porque en planta la verdadera no se conoce.
        por_cajon = {}
        for c in sorted(set(dicho.tolist())):
            m = (dicho == c)
            if m.sum() >= MIN_CAJON:
                por_cajon[str(c)] = acumulada(conf[m], ok[m])
        return tabla, err, acumulada(conf, ok), por_cajon

    def elige(XX, yy, gg, titulo, rej=False):
        """Devuelve (familia, acc_media, tabla_confianza).

        EL CRITERIO, ESCRITO ANTES DE MIRAR EL RESULTADO (2026-09-05)
        ------------------------------------------------------------
        Manda la exactitud **solo cuando la diferencia supera el ruido de
        particion** (3.5 puntos, PROGRESO 45.2). Por debajo de eso las dos
        familias son indistinguibles en exactitud, y entonces decide la
        **calibracion**: cuanto se parece la confianza que declara el modelo a
        las veces que acierta de verdad.

        No es un detalle de presentacion. La tesis es un **agente que decide bajo
        confianza** y enruta a revision manual lo que no tiene claro; con un
        modelo que dice 0.90 y acierta el 61 %, esa regla de enrutamiento no
        funciona -- la maquina manda a la cinta hojas que deberia apartar. Un
        punto de exactitud que esta dentro del ruido no paga eso.
        """
        res, cal = {}, {}
        # OJO: la cache va por identidad de array, y los `X[sel]` de cada llamada
        # son temporales -- al liberarse, otro array puede caer en la misma
        # direccion y dar un acierto de cache FALSO. Se vacia en cada etapa.
        _memo.clear()
        print("   [{}] midiendo...".format(titulo.split("--")[0].strip()), flush=True)
        for nombre in ("logistica", "arboles"):
            acc = []
            for s in range(N_SEM):
                acc.append(float((oof(XX, yy, gg, nombre, s, rej)[0] == yy).mean()))
                print("      {} semilla {}/{}".format(nombre, s + 1, N_SEM),
                      flush=True)
            res[nombre] = np.array(acc)
            cal[nombre] = calibracion(XX, yy, gg, nombre, rej)
        print()
        print("   {}".format(titulo))
        for n in ("logistica", "arboles"):
            print("      {:12s} exactitud {:5.1f} %  [{:4.1f} - {:4.1f}]"
                  "   error de confianza {:4.1f} puntos"
                  .format(n, 100 * res[n].mean(), 100 * res[n].min(),
                          100 * res[n].max(), 100 * cal[n][1]))
        a, b = "logistica", "arboles"
        d = 100 * (res[a].mean() - res[b].mean())
        if FORZAR_FAMILIA and "variedad" not in titulo.lower():
            gana = FORZAR_FAMILIA
            print("      -> FIJADA a {} por --familia (la automatica habria"
                  " elegido {})".format(
                      gana, a if abs(d) >= 3.5 and d > 0 else
                      (b if abs(d) >= 3.5 else min((a, b), key=lambda n: cal[n][1]))))
        elif abs(d) >= 3.5:
            gana = a if d > 0 else b
            print("      -> se guarda {}: gana {:.1f} puntos de exactitud, mas que"
                  " el ruido".format(gana, abs(d)))
        else:
            gana = min((a, b), key=lambda n: cal[n][1])
            print("      -> las dos empatan en exactitud ({:.1f} puntos, por debajo"
                  " del ruido de 3.5)".format(abs(d)))
            print("         decide la CALIBRACION: se guarda {} ({:.1f} contra {:.1f}"
                  " puntos de error)".format(gana, 100 * cal[gana][1],
                                             100 * cal[a if gana == b else b][1]))
        tabla, acum, cajones = cal[gana][0], cal[gana][2], cal[gana][3]
        print("      confianza declarada  ->  acierto real (fuera de muestra)")
        for t in tabla:
            print("         {:.2f} - {:.2f}   dice {:4.2f}   acierta {:5.1f} %"
                  "   ({:4.0f} casos, {:4.1f} % de las hojas)"
                  .format(t["desde"], t["hasta"], t["dice"], 100 * t["acierto"],
                          t["n"], 100 * t["frac"]))
        return gana, float(res[gana].mean()), tabla, acum, cajones

    print()
    print("=" * 78)
    print("QUE FAMILIA SE GUARDA, Y QUE VALE SU CONFIANZA")
    print("=" * 78)
    fam_v, acc_v, tab_v, acum_v, _caj_v = elige(XV, var_, grp,
                                                "ETAPA 1 -- la variedad")
    modelos = {}
    info_cal = {}
    for v in ("connecticut", "habano"):
        sel = var_ == v
        fam, acc, tab, acum, cajones = elige(X[sel], y[sel], grp[sel],
                              "ETAPA 2 -- la calidad, {} ({} hojas)"
                              .format(v, int(sel.sum())), rej=_args.rejilla)
        # LA REJILLA DE DOS PREGUNTAS (2026-09-15). Antes aqui habia UN modelo
        # que elegia entre las cuatro clases a la vez, y nada le impedia decir
        # "banda" de una hoja con seis agujeros -- medido: 4 de 112 predicciones
        # eran imposibles por definicion. Ahora la clase sale de dos respuestas
        # encadenadas y esa confusion no puede ocurrir por construccion.
        if not _args.rejilla:
            modelos[v] = mk(fam, 0).fit(X[sel], y[sel])
            info_cal[v] = dict(familia=fam, acc_cv=acc, confianza=tab,
                               acumulada=acum, acumulada_cajon=cajones,
                               n=int(sel.sum()),
                               clases=sorted(set(y[sel].tolist())))
            continue
        yag = RJ.reparte(y[sel])
        m_ag = mk(fam, 0).fit(X[sel], yag)
        s1 = sel.copy(); s1[sel] = (yag == RJ.SIN)
        s2 = sel.copy(); s2[sel] = (yag == RJ.CON)
        m_sin = mk(fam, 0).fit(X[s1], y[s1])
        m_con = mk(fam, 0).fit(X[s2], y[s2])
        modelos[v] = RJ.Rejilla(m_ag, m_sin, m_con)
        print("      rejilla: agujeros {} hojas ({} sin / {} con) -> "
              "capa|banda {} y xl|xr {}"
              .format(int(sel.sum()), int(s1.sum()), int(s2.sum()),
                      list(m_sin.classes_), list(m_con.classes_)))
        info_cal[v] = dict(familia=fam, acc_cv=acc, confianza=tab,
                           acumulada=acum, acumulada_cajon=cajones,
                           n=int(sel.sum()),
                           clases=sorted(set(y[sel].tolist())))

    m_var = mk(fam_v, 0).fit(XV, var_)

    # ------------------------------------------------------------------
    # LO QUE ESTO SIGNIFICA EN PLANTA, que es la cifra que falta en la tesis.
    #
    # El agente no tiene que acertar siempre: tiene que saber CUANDO no sabe y
    # apartar esa hoja. Con la tabla de calibracion medida se puede decir, antes
    # de instalar nada, las dos cifras que decide un jefe de planta:
    #     cuantas hojas pasa solo, y con que acierto;
    #     cuantas manda a una persona.
    # Se cuenta un tramo como "lo pasa solo" si su acierto MEDIDO llega al
    # umbral -- no si su confianza declarada llega, que es la trampa que este
    # guion evita eligiendo por calibracion.
    # ------------------------------------------------------------------
    UMBRAL = 0.90
    # 4 puntos desde el 2026-09-09, decidido por el experto (§72.6): en su planta una
    # hoja mal enrutada --que nadie mira-- cuesta MUCHO mas que una hoja que va
    # al recipiente 11 y cuesta el rato de una persona. Con 4 puntos y el veto,
    # el sistema automatiza el 42.9 % acertando el 95.4 %; con 2, el 51.3 % al
    # 93.5 %. Medido en 30 semillas nuevas (`out/MARGEN_Y_CLASE_sem100.txt`).
    MARGEN = 0.04
    print()
    print("=" * 78)
    print("LO QUE SIGNIFICA EN PLANTA, con un umbral de {:.0f} % de acierto medido"
          .format(100 * UMBRAL))
    print("=" * 78)
    # EL CORTE SE ELIGE CON LA CURVA ACUMULADA, no tramo a tramo (2026-09-06, §63)
    #
    # Lo que promete el sistema es que **el lote que pasa solo** acierte el 90 %.
    # Exigirselo a cada tramo por separado es mas duro que la promesa y manda a
    # revision hojas que podian pasar: medido anidado, 57 % automatico contra
    # 79 %, con el MISMO modelo y sin bajar del 90 %.
    #
    # El margen no es decoracion: sin el, el corte entrega 89.5 % cuando promete
    # 90 -- la acumulada es ella misma una estimacion con ruido y el corte cae
    # justo en el filo. Con 2 puntos de margen las dos variedades cumplen.
    # Las dos cifras estan medidas en `out/CALIBRA_CONFIANZA.txt`.
    planta = {}
    for v in ("connecticut", "habano"):
        acum = info_cal[v]["acumulada"]
        ac_tot = acum[-1]["acierto"]            # el punto de cobertura 100 %
        # QUE CAJONES QUEDAN VETADOS (§72.5), para poder decirlo aqui: un cajon
        # cuyo techo no llega al umbral no automatiza ninguna hoja.
        vetados = sorted(c for c, cu in (info_cal[v].get("acumulada_cajon") or {}).items()
                         if max(t["acierto"] for t in cu) < UMBRAL + MARGEN)
        info_cal[v]["cajones_vetados"] = vetados
        valen = [t for t in acum if t["acierto"] >= UMBRAL + MARGEN]
        elegido = max(valen, key=lambda t: t["cobertura"]) if valen else None
        if elegido is None:
            f_solo = ac_solo = 0.0
            corte = 1.01
        else:
            f_solo, ac_solo, corte = (elegido["cobertura"], elegido["acierto"],
                                      elegido["conf"])
        f_res = 1.0 - f_solo
        ac_res = ((ac_tot - f_solo * ac_solo) / f_res) if f_res > 1e-9 else float("nan")
        # lo de antes, para poder comparar en el informe
        tab = info_cal[v]["confianza"]
        f_viejo = sum(t["frac"] for t in tab if t["acierto"] >= UMBRAL)
        planta[v] = dict(umbral=UMBRAL, margen=MARGEN, corte_conf=corte,
                         frac_sola=f_solo, acierto_sola=ac_solo,
                         frac_revision=f_res, acierto_revision=ac_res,
                         frac_sola_por_tramos=f_viejo, n=info_cal[v]["n"],
                         cajones_vetados=vetados)
        info_cal[v]["corte_conf"] = corte
        print("   {:12s} pasa solo el {:4.1f} % de las hojas (confianza >= {:.3f}),"
              " y ahi acierta el {:4.1f} %"
              .format(v, 100 * f_solo, corte, 100 * ac_solo))
        print("   {:12s} manda a una persona el {:4.1f} %  (donde hoy acertaria el"
              " {:4.1f} %)".format("", 100 * f_res, 100 * ac_res))
        print("   {:12s} [con la regla vieja, tramo a tramo, pasaba el {:4.1f} %]"
              .format("", 100 * f_viejo))
        if vetados:
            print("   {:12s} CAJONES VETADOS (§72.5): {} -- no automatizan ninguna"
                  " hoja".format("", ", ".join(vetados)))
            print("   {:12s} porque su lote no llega al {:.0f} % ni en su mejor"
                  " corte. La cobertura de arriba"
                  .format("", 100 * (UMBRAL + MARGEN)))
            print("   {:12s} NO lo descuenta: la cifra con veto sale de"
                  " `prueba_agente.py`, medida fuera de la sesion."
                  .format(""))
    n_c, n_h = info_cal["connecticut"]["n"], info_cal["habano"]["n"]
    w = n_c + n_h
    f_g = (n_c * planta["connecticut"]["frac_sola"]
           + n_h * planta["habano"]["frac_sola"]) / w
    a_g = (n_c * planta["connecticut"]["frac_sola"] * planta["connecticut"]["acierto_sola"]
           + n_h * planta["habano"]["frac_sola"] * planta["habano"]["acierto_sola"]) \
        / (w * f_g)
    print()
    print("   EN CONJUNTO: el sistema resuelve solo {:.0f} de cada 100 hojas, con"
          " {:.0f} % de acierto".format(100 * f_g, 100 * a_g))
    print("   y aparta las otras {:.0f} para que las vea una persona."
          .format(100 * (1 - f_g)))
    print()
    print("   Es una cifra de DENTRO de la sesion, y hay que decirlo siempre; pero")
    print("   es la primera vez que el proyecto puede contestar 'cuanta gente hace")
    print("   falta', que es lo que pregunta una planta antes que la exactitud.")

    DIR_MODELO.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(variedad=m_var, calidad=modelos), DIR_MODELO / "modelo.joblib")
    meta = dict(
        fecha=datetime.now().isoformat(timespec="seconds"),
        n_hojas=len(y), n_semillas=N_SEM,
        rasgos_variedad=nom_var, rasgos_calidad=nom_cal,
        variedad=dict(familia=fam_v, acc_cv=acc_v, confianza=tab_v,
                      acumulada=acum_v, clases=sorted(set(var_.tolist()))),
        calidad=info_cal,
        planta=planta,
        reparto=dict(sorted(Counter(y).items())),
        aviso=("Entrenado con una sola sesion de fotos, en la que la clase y la "
               "hora del dia van juntas (PROGRESO 54.2). Las cifras de aqui son "
               "de validacion cruzada agrupada por hoja DENTRO de esa sesion; no "
               "predicen lo que hara en una captura nueva."),
    )
    (DIR_MODELO / "modelo.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("-> {}".format(DIR_MODELO / "modelo.joblib"))
    print("-> {}".format(DIR_MODELO / "modelo.json"))
    print()
    print("AVISO QUE VIAJA CON EL MODELO:")
    print("   " + meta["aviso"])


if __name__ == "__main__":
    main()
