r"""
Clasificador de las cinco calidades, Connecticut y Habano juntos.

POR QUE LAS DOS VARIEDADES JUNTAS
---------------------------------
Lo dijo el experto en `LEEME IMPORTANTE.txt`: *"Los requisitos para ambas clases de
tabaco son los mismos"*. La regla de los tecnicos no cambia entre Habano y
Connecticut, asi que el modelo debe aprender la regla, no la variedad. Se entrena
con las dos mezcladas y se reporta ademas el resultado por separado.

QUE MIRA, Y QUE NO
------------------
Solo la HOJA, y dentro de ella solo la ZONA UTIL (hoja menos base, punta y franja
de vena), que es la que decide segun la regla de PROGRESO 16. Nunca la mesa.

Tres familias de rasgos:
  1. `hoja`     color y textura de la zona util entera
  2. `lados`    contrastes izquierda-derecha, adimensionales (PROGRESO 25).
                Son los que pueden separar XL izq de XR der, porque esas dos
                clases se diferencian por DONDE esta el dano.
  3. `forma`    geometria: largo y ancho en pulgadas, cobertura, solidez.
                Es senal legitima: el tamano de la hoja entra en la decision.

LA REGLA DE HONESTIDAD DE ESTE GUION
------------------------------------
Cada resultado se reporta **al lado del de un modelo que no mira la hoja**, hecho
con los mismos rasgos calculados sobre el FONDO. No es desconfianza del montaje de
el experto --que esta bien iluminado y es constante-- sino lo que hace defendible el
numero: demuestra que el modelo aprendio tabaco y no el entorno. Si el fondo
alcanza al modelo de hoja, el resultado no vale, por bonito que sea.

Y siempre:
  - **validacion cruzada agrupada por hoja** (`StratifiedGroupKFold`): dos fotos de
    la misma hoja jamas caen en lados distintos de la particion;
  - **una sola foto por hoja**, la de encuadre mas amplio, y solo hojas enteras.
    Un acercamiento recorta la hoja y falsea tanto la forma como los contrastes
    entre mitades (medido en PROGRESO 27).

Uso:
    python scripts/clasificador.py
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
import particiones as PART                                           # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from rutas import REPO_ROOT  # repository root

RAIZ = REPO_ROOT / "dataset"
OUT = REPO_ROOT / "out"
LADO = 1024
N_TRAMOS = 60
FRAC_RABO = 0.18
BASE_ANCHOS, PUNTA_ANCHOS, VENA_ANCHOS = 2.0, 1.0, 0.6
PULG_POR_ANCHO_CINTA = 1.11        # PROGRESO 26, +-15 %

MED = ["L_med", "L_desv", "a_med", "b_med", "b_desv",
       "f_oscuro", "f_verdoso", "f_claro", "textura"]

# Rasgos de defecto, calculados por `agujeros.py` y leidos de out/agujeros.csv.
# Son la familia que faltaba: 1/2 Banda es "una Banda con un agujero notable en un
# lado" (PROGRESO 22.1) y hasta ahora el modelo no veia agujeros.
DEF = ["n_agujeros", "n_roturas", "n_defectos", "area_def_pulg2", "frac_area_def",
       "mayor_pulg2", "mayor_dist_base_pulg", "area_util_pulg2",
       "area_util_izq_pulg2", "area_util_der_pulg2", "n_util_izq", "n_util_der"]

# Manchas (verde "sudada", blanca de agroquimico, negra), de `manchas.py`. Familia
# aparte de los agujeros a proposito: los ficheros del experto dicen que un XL puede no
# tener NINGUNA rotura y ser XL solo por la sudada de un lado (PROGRESO 37.1), asi
# que interesa ver cuanto aporta cada cosa por separado.
MAN = ["frac_verde", "frac_blanca", "frac_negra",
       "verde_izq_pulg2", "verde_der_pulg2", "blanca_izq_pulg2", "blanca_der_pulg2",
       "negra_izq_pulg2", "negra_der_pulg2", "mancha_izq_pulg2", "mancha_der_pulg2",
       "asimetria_mancha", "mayor_mancha_pulg2"]
# `area_mordida_*` NO entra, y es un resultado medido, no un olvido: metiendo las
# mordidas (deficit contra la envolvente convexa) el modelo BAJA -- 5 clases 81.0
# -> 80.7, Habano 86.1 -> 82.5, el par XL/XR 90.3 -> 88.4. La causa se ve en el
# informe de agujeros: la mordida mediana mas grande es la de Habano/Capa (1.9
# pulg2), o sea que la medida no captura dano sino CURVATURA de la hoja -- una hoja
# arqueada tiene un deficit enorme contra su casco convexo sin que le falte nada.
# Se deja calculada en agujeros.csv y pintada en el QC, pero fuera del modelo.

# PENETRACION de la mordida (2026-09-03). Familia aparte de `DEF` a proposito, para
# poder medir lo que aporta ELLA SOLA -- que es toda la razon de anadirla.
# El area de mordida mide curvatura; la penetracion mide dano. La regla de la
# chaveta (§40.6) dice que del mordisco de borde no importa cuanto ocupa sino
# **cuanto se mete**, porque el corte se adapta: "se corta hasta que se termina lo
# arrugado". Y como la curvatura afecta a los dos lados por igual mientras que un
# pedazo que falta es de UN lado, el rasgo con sentido es la ASIMETRIA.
# Caso demostrativo, la 102 (§40.2): 0 agujeros detectados, y la penetracion da
# 1.72 pulg a la derecha contra 0.50 a la izquierda -- el lado correcto segun el experto.
PEN = ["pen_mordida_izq_pulg", "pen_mordida_der_pulg", "pen_mordida_pulg",
       "asim_pen_mordida"]


def carga(p, lado=LADO):
    with Image.open(p) as im:
        try:
            im.draft("RGB", (im.width // 4, im.height // 4))
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((lado, lado), Image.LANCZOS)
        return np.asarray(im)


def rota(img, grados, nearest=False):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), grados, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def medidas(rgb, sel, ener):
    if sel.sum() < 200:
        return [np.nan] * len(MED)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0][sel]
    a = lab[..., 1][sel] - 128
    b = lab[..., 2][sel] - 128
    Lm, Ls = float(L.mean()), float(L.std() + 1e-6)
    return [Lm, Ls, float(a.mean()), float(b.mean()), float(b.std()),
            float((L < Lm - 1.5 * Ls).mean()),
            float((a < a.mean() - 1.5 * a.std()).mean()),
            float((L > Lm + 1.5 * Ls).mean()),
            float(ener[sel].mean())]


def main(max_nuevas=0):
    seg = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "segmentacion.csv", encoding="utf-8"))}
    man = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "manifiesto_limpio.csv", encoding="utf-8"))}
    end = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "enderezado.csv", encoding="utf-8"))}
    zon = {r["ruta"]: r for r in
           csv.DictReader(open(OUT / "zonas.csv", encoding="utf-8"))}
    agu = {}
    if (OUT / "agujeros.csv").exists():
        agu = {r["ruta"]: r for r in
               csv.DictReader(open(OUT / "agujeros.csv", encoding="utf-8"))}
    print("defectos medidos: {} fotos".format(len(agu)))
    mch = {}
    if (OUT / "manchas.csv").exists():
        mch = {r["ruta"]: r for r in
               csv.DictReader(open(OUT / "manchas.csv", encoding="utf-8"))}
    print("manchas medidas: {} fotos".format(len(mch)))
    # en pixeles DE LA MASCARA: escala_cinta trabaja a 1800 y las mascaras a 1024
    from zonas import ancho_cinta_por_carpeta, lamina, bandas
    ancho_cinta = ancho_cinta_por_carpeta(OUT)

    # ------------------------------------------------------------------
    # HOJAS OMITIDAS POR DECISION DEL EXPERTO (2026-09-04)
    #
    #   "Las hojas que no te pude decir que eran, porque no estaba seguro,
    #    omitelas, no retrasemos el proyecto por esas fotos. no lo vale"
    #
    # Son las nueve de PROGRESO 46. Salen del conjunto ENTERO: ni entrenan ni
    # puntuan. La foto no se borra ni se mueve; la decision vive en
    # dataset/CASOS_DUDOSOS.csv, con quien la tomo y cuando.
    #
    # AVISO QUE HAY QUE SEGUIR IMPRIMIENDO, y no es una objecion a la decision:
    # quitar hojas dificiles sube la exactitud POR CONSTRUCCION, porque son justo
    # las que se fallan. La decision es correcta --esas etiquetas no existen, no
    # son "dificiles", son desconocidas-- pero el numero que sale de aqui NO es
    # comparable con el de antes del 04/09 sin decir esto. Por eso se imprime
    # cuantas salen y de que clase.
    # ------------------------------------------------------------------
    omitidas = set()
    f_dud = RAIZ / "CASOS_DUDOSOS.csv"
    if f_dud.exists():
        omitidas = {r["ruta"] for r in csv.DictReader(open(f_dud, encoding="utf-8"))
                    if r.get("estado", "").startswith("omitida")}
    if omitidas:
        print("omitidas por decision del experto: {} hojas ({})"
              .format(len(omitidas),
                      dict(sorted(Counter(man[r]["clase"] for r in omitidas
                                          if r in man).items()))))

    limpias = [r for r in man
               if r not in omitidas
               and seg.get(r, {}).get("toca_borde") == "0"
               and zon.get(r, {}).get("base_dudosa") == "0"]
    mejor = {}
    for r in limpias:
        h = man[r]["hoja_id"]
        f = float(seg[r]["frac_cuadro"])
        if h not in mejor or f < mejor[h][1]:
            mejor[h] = (r, f)
    rutas = sorted(v[0] for v in mejor.values())
    print("fotos {}  ->  hoja entera y base fiable {}  ->  una por hoja {}"
          .format(len(man), len(limpias), len(rutas)))

    # ------------------------------------------------------------------
    # Los rasgos se guardan SEGUN SE CALCULAN y se pueden reanudar.
    # Extraerlos cuesta decodificar 420 fotos; si el proceso se corta a la mitad
    # (paso varias veces en esta maquina) se perdia todo. Ahora cada hoja se
    # escribe en cuanto esta, y al arrancar se saltan las que ya estan.
    # ------------------------------------------------------------------
    COLS = (["ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id"]
            + ["hoja_" + k for k in MED]
            + ["lado_" + k for k in MED] + ["lado_abs_" + k for k in MED]
            + ["largo_pulg", "ancho_pulg", "aspecto", "solidez", "cobertura",
               "hoja_entre_util"]
            + ["fondo_" + k for k in MED])
    cache = OUT / "clasificador_rasgos.csv"
    hechas = {}
    if cache.exists():
        for r0 in csv.DictReader(open(cache, encoding="utf-8")):
            if r0.get("ruta"):
                hechas[r0["ruta"]] = r0
        print("rasgos ya calculados en cache: {}".format(len(hechas)))
    fh_cache = open(cache, "a" if hechas else "w", newline="", encoding="utf-8")
    wr_cache = csv.writer(fh_cache)
    if not hechas:
        wr_cache.writerow(COLS)
        fh_cache.flush()

    filas, Xh, Xl, Xf, Xg = [], [], [], [], []
    nuevas = 0
    for i, ruta in enumerate(rutas, 1):
        if ruta in hechas:
            r0 = hechas[ruta]
            v = [float(r0[c]) for c in COLS[6:]]
            n = len(MED)
            Xh.append(v[:n]); Xl.append(v[n:3 * n])
            Xg.append(v[3 * n:3 * n + 6]); Xf.append(v[3 * n + 6:])
            # LA ETIQUETA SE LEE DEL MANIFIESTO, NUNCA DE LA CACHE.
            # La cache guarda RASGOS, que son caros de calcular y no cambian. La
            # clase si cambia: `dataset/CORRECCIONES_ETIQUETA.csv` corrige carpetas
            # equivocadas (PROGRESO 40.1) y el manifiesto la aplica. Si se leyera de
            # aqui, una correccion de etiqueta no tendria ningun efecto hasta borrar
            # la cache entera -- y nadie se enteraria, porque el guion no fallaria.
            mr0 = man.get(ruta, r0)
            filas.append(dict(ruta=ruta, archivo=r0["archivo"],
                              carpeta=r0["carpeta"], variedad=mr0["variedad"],
                              clase=mr0["clase"], hoja_id=mr0["hoja_id"]))
            continue
        r, mr = seg[ruta], man[ruta]
        f = OUT / "mascaras" / (Path(r["archivo"]).stem + ".npy")
        if not f.exists():
            continue
        h, w = int(r["alto"]), int(r["ancho"])
        m = np.unpackbits(np.load(f))[:h * w].reshape(h, w).astype(bool)
        rgb = carga(RAIZ / ruta)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h))
        e_row = end.get(ruta, {})
        rot = float(e_row["rotacion"]) if e_row.get("rotacion") else 0.0
        vol = int(e_row.get("voltea_180") or 0)
        if rot:
            m = rota(m.astype(np.uint8), rot, True) > 0
            rgb = rota(rgb, rot)
        if vol:
            m = cv2.rotate(m.astype(np.uint8), cv2.ROTATE_180) > 0
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        if m.sum() < 500:
            continue

        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        ener = np.sqrt(gx * gx + gy * gy)

        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        mu = pts.mean(0)
        val, vec = np.linalg.eigh(np.cov((pts - mu).T))
        e = vec[:, 1]
        if e[1] < 0:
            e = -e
        perp = np.array([-e[1], e[0]], dtype=np.float32)
        t = (pts - mu) @ e
        s = (pts - mu) @ perp
        idx = np.clip(((t - t.min()) / max(1e-6, t.max() - t.min()) * N_TRAMOS)
                      .astype(int), 0, N_TRAMOS - 1)
        anch = np.zeros(N_TRAMOS)
        for k in range(N_TRAMOS):
            sl = idx == k
            if sl.sum() > 3:
                anch[k] = s[sl].max() - s[sl].min()
        if anch.max() <= 0:
            continue
        aa = anch / anch.max()
        ini = 0
        while ini < N_TRAMOS and aa[ini] < FRAC_RABO:
            ini += 1
        fin = 0
        while fin < N_TRAMOS and aa[N_TRAMOS - 1 - fin] < FRAC_RABO:
            fin += 1
        if fin < ini:
            e, perp, t, s = -e, -perp, -t, -s

        carp = mr["carpeta"]
        ac = ancho_cinta.get(carp, 0.08 * (t.max() - t.min()))
        lam0, lam1 = lamina(t, anch)
        punta_px, base_px, vena_px, _b = bandas(lam1 - lam0,
                                            PULG_POR_ANCHO_CINTA / ac if ac else None,
                                            ac)
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
            continue

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
        forma = [(lam1 - lam0) / ac * PULG_POR_ANCHO_CINTA,
                 anchomax / ac * PULG_POR_ANCHO_CINTA,
                 anchomax / max(1.0, largo),
                 float(r["solidez"]), float(r["frac_cuadro"]),
                 float(m.sum()) / max(1.0, float(M_util.sum()))]
        fondo = medidas(rgb, ~m, ener)          # el control: NO mira la hoja

        Xh.append(mu_h); Xl.append(lados); Xg.append(forma); Xf.append(fondo)
        filas.append(dict(ruta=ruta, archivo=r["archivo"], carpeta=carp,
                          variedad=mr["variedad"], clase=mr["clase"],
                          hoja_id=mr["hoja_id"]))
        wr_cache.writerow([ruta, carp, r["archivo"], mr["variedad"], mr["clase"],
                           mr["hoja_id"]]
                          + [round(float(x), 4) for x in
                             list(mu_h) + list(lados) + list(forma) + list(fondo)])
        fh_cache.flush()
        nuevas += 1
        if i % 25 == 0:
            print("  {}/{}".format(i, len(rutas)), flush=True)
        if max_nuevas and nuevas >= max_nuevas:
            print("  [corte voluntario tras {} nuevas; relanzar para seguir]".format(nuevas))
            fh_cache.close()
            return
    fh_cache.close()

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    Xd = []
    for r0 in filas:
        a0 = agu.get(r0["ruta"])
        if not a0:
            Xd.append([np.nan] * (len(DEF) + 2))
            continue
        v = [_num(a0.get(k)) for k in DEF]
        izq, der = _num(a0.get("area_util_izq_pulg2")), _num(a0.get("area_util_der_pulg2"))
        # asimetria: es LO QUE DECIDE la clase segun los tecnicos -- no cuanto dano
        # hay, sino de que lado sobrevive la mitad limpia (PROGRESO 16.2)
        v.append((izq - der) / (izq + der + 1e-6))
        v.append(abs(izq - der))
        Xd.append(v)
    Xd = np.array(Xd, float)

    Xp = []
    for r0 in filas:
        a0 = agu.get(r0["ruta"])
        Xp.append([np.nan] * len(PEN) if not a0
                  else [_num(a0.get(k)) for k in PEN])
    Xp = np.array(Xp, float)

    Xm = []
    for r0 in filas:
        m0 = mch.get(r0["ruta"])
        if not m0:
            Xm.append([np.nan] * (len(MAN) + 1))
            continue
        v = [_num(m0.get(k)) for k in MAN]
        vi, vd = _num(m0.get("verde_izq_pulg2")), _num(m0.get("verde_der_pulg2"))
        v.append(vi - vd)          # el signo dice de que lado esta la sudada
        Xm.append(v)
    Xm = np.array(Xm, float)
    n_con = int(np.isfinite(Xd[:, 0]).sum()) if len(Xd) else 0
    print("hojas con defectos medidos: {}/{}".format(n_con, len(filas)))

    Xh, Xl, Xg, Xf = (np.array(x, float) for x in (Xh, Xl, Xg, Xf))
    y = np.array([r["clase"] for r in filas])
    var = np.array([r["variedad"] for r in filas])
    grp = np.array([r["hoja_id"] for r in filas])
    print("hojas usables: {}".format(len(filas)))
    print("reparto: {}".format(dict(sorted(Counter(y).items()))))

    def evalua(nombre, X, yy, gg, detalle=False, dev=False):
        base = max(Counter(yy).values()) / len(yy)
        k = min(5, min(len(set(gg[yy == c])) for c in set(yy)))
        if k < 2:
            return None
        cv = PART.Particion(k, 0, "clasificador")
        mods = {
            "logistica": make_pipeline(SimpleImputer(strategy="median"),
                                       StandardScaler(),
                                       LogisticRegression(max_iter=4000,
                                                          class_weight="balanced")),
            "arboles": make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(max_iter=300, random_state=0)),
        }
        mejor_pr, mejor_ac = None, -1
        for nm, mod in mods.items():
            pr = cross_val_predict(mod, X, yy, cv=cv, groups=gg)
            ac = accuracy_score(yy, pr)
            if ac > mejor_ac:
                mejor_ac, mejor_pr, mejor_nm = ac, pr, nm
        print("  {:38s} {:6.1%}  (balanceada {:5.1%}, linea base {:.1%}, {})"
              .format(nombre, mejor_ac, balanced_accuracy_score(yy, mejor_pr),
                      base, mejor_nm))
        if dev:
            return mejor_ac, mejor_pr
        if detalle:
            et = sorted(set(yy))
            M = confusion_matrix(yy, mejor_pr, labels=et)
            print("\n    matriz de confusion (fila = lo que dice la carpeta):")
            print("    {:14s}".format("") + "".join("{:>13s}".format(x) for x in et))
            for i2, x in enumerate(et):
                print("    {:14s}".format(x) + "".join("{:13d}".format(v) for v in M[i2]))
            print()
            print(classification_report(yy, mejor_pr, digits=2, zero_division=0))
        return mejor_ac

    def bloque(titulo, mask):
        print("\n" + "=" * 78)
        print(titulo)
        print("=" * 78)
        yy, gg = y[mask], grp[mask]
        print("  n = {} hojas".format(int(mask.sum())))
        a_f = evalua("CONTROL: solo el FONDO, sin la hoja", Xf[mask], yy, gg)
        evalua("solo color y textura de la hoja", Xh[mask], yy, gg)
        evalua("solo contrastes izquierda-derecha", Xl[mask], yy, gg)
        evalua("solo la forma (tamano en pulgadas)", Xg[mask], yy, gg)
        hay_def = bool(len(Xd)) and bool(np.isfinite(Xd).any())
        hay_man = bool(len(Xm)) and bool(np.isfinite(Xm).any())
        hay_pen = bool(len(Xp)) and bool(np.isfinite(Xp).any())
        if hay_def:
            evalua("solo los agujeros y roturas", Xd[mask], yy, gg)
        if hay_man:
            evalua("solo las manchas (verde/blanca/negra)", Xm[mask], yy, gg)
        if hay_pen:
            evalua("solo la penetracion de la mordida", Xp[mask], yy, gg)
        a_t2 = a_t3 = None
        a_t = evalua("TODO lo de la hoja (color+lados+forma)",
                     np.hstack([Xh, Xl, Xg])[mask], yy, gg,
                     detalle=not (hay_def or hay_man or hay_pen))
        if hay_def:
            a_t2 = evalua("TODO + agujeros", np.hstack([Xh, Xl, Xg, Xd])[mask],
                          yy, gg, detalle=not hay_man)
            if a_t2 is not None and a_t is not None:
                print("  --> los agujeros suman {:+.1f} puntos"
                      .format(100 * (a_t2 - a_t)))
        if hay_man:
            a_t3 = evalua("TODO + agujeros + manchas",
                          np.hstack([Xh, Xl, Xg, Xd, Xm])[mask], yy, gg,
                          detalle=not hay_pen)
            if a_t3 is not None and a_t2 is not None:
                print("  --> las manchas suman {:+.1f} puntos mas"
                      .format(100 * (a_t3 - a_t2)))
            a_t = max(x for x in (a_t, a_t2, a_t3) if x is not None)
        if hay_pen:
            a_t4 = evalua("TODO + agujeros + manchas + penetracion",
                          np.hstack([Xh, Xl, Xg, Xd, Xm, Xp])[mask], yy, gg,
                          detalle=True)
            if a_t4 is not None and a_t3 is not None:
                print("  --> la penetracion suma {:+.1f} puntos mas"
                      .format(100 * (a_t4 - a_t3)))
            a_t = max(x for x in (a_t, a_t4) if x is not None)
        if a_f is not None and a_t is not None:
            if a_t > a_f:
                print("  --> el modelo de HOJA supera al de fondo por {:.1f} puntos"
                      .format(100 * (a_t - a_f)))
            else:
                print("  --> [!] el fondo iguala o supera a la hoja: el numero NO es"
                      " defendible todavia")

    bloque("LAS CINCO CALIDADES, Connecticut y Habano juntos", np.ones(len(y), bool))
    for v in ("connecticut", "habano"):
        bloque("LAS CINCO CALIDADES, solo {}".format(v.upper()), var == v)
    bloque("SOLO EL PAR XL IZQ vs XR DER", np.isin(y, ["xl_izq", "xr_der"]))
    bloque("CAPA contra el resto (la pregunta de negocio)",
           np.ones(len(y), bool)) if False else None

    # ------------------------------------------------------------------
    # EL PAR, DESCONTANDO LAS HOJAS CUYA ETIQUETA ESTA EN DISPUTA
    #
    # El experto duda de 9 de las 32 hojas que reviso (PROGRESO 40.5) y lo dice el, sin
    # que nadie le pregunte: en dos de ellas se inclina por la clase CONTRARIA a la
    # de su propia carpeta. Contra eso ningun clasificador puede: no es error del
    # modelo, es desacuerdo sobre la verdad-terreno.
    #
    # COMO SE MIDE, Y POR QUE ASI. El modelo es EL MISMO: se entrena y se predice
    # sobre el par entero, con las dudosas dentro. Lo unico que cambia es el
    # DENOMINADOR: no se le puntuan las hojas cuya etiqueta esta pendiente de los
    # expertos. Sacarlas tambien del entrenamiento seria hacer trampa dos veces.
    #
    # Y AUN ASI HAY QUE LEERLO CON CUIDADO. Quitar hojas dificiles sube la exactitud
    # por construccion: son justo las que se fallan. Por eso se imprime la
    # aritmetica entera --cuantas hojas salen, cuantas de ellas eran fallos-- para
    # que se vea que el numero no mide un modelo mejor, sino un conjunto distinto.
    # La cifra que se reporta como resultado del sistema sigue siendo la de arriba.
    # ------------------------------------------------------------------
    m_par = np.isin(y, ["xl_izq", "xr_der"])
    dudosas, rutas_f = None, np.array([r["ruta"] for r in filas])
    if f_dud.exists() and m_par.sum():
        # Solo las que sigan PENDIENTES. Desde el 04/09 las nueve estan omitidas
        # (arriba), asi que este bloque no encuentra ninguna y no imprime nada:
        # se deja para cuando vuelva a haber hojas en disputa dentro del conjunto.
        dudosas = {r["ruta"] for r in csv.DictReader(open(f_dud, encoding="utf-8"))
                   if r.get("estado", "").startswith("pendiente")}
        if not (dudosas & set(rutas_f)):
            dudosas = None          # ninguna en disputa DENTRO del conjunto
    if dudosas:
        m_dud = np.isin(rutas_f, sorted(dudosas))
        print("\n" + "=" * 78)
        print("EL PAR XL IZQ / XR DER, SIN LAS HOJAS CON LA ETIQUETA EN DISPUTA")
        print("=" * 78)
        print("  dudosas en la lista: {}   de ellas, en el par y usables: {}"
              .format(len(dudosas), int((m_dud & m_par).sum())))
        faltan = sorted(dudosas - set(rutas_f[m_par]))
        for r_ in faltan:
            print("  [!] no entra al par (recorte, base dudosa o no es la foto"
                  " elegida de su hoja): {}".format(r_))
        XX = np.hstack([Xh, Xl, Xg] + [x for x in (Xd, Xm, Xp)
                                       if len(x) and np.isfinite(x).any()])
        res = evalua("TODO, el par entero (referencia)", XX[m_par], y[m_par],
                     grp[m_par], dev=True)
        if res is not None:
            ac_par, pr_par = res
            sub = ~m_dud[m_par]
            ok_par = (pr_par == y[m_par])
            n_par, n_sub = int(m_par.sum()), int(sub.sum())
            fallos_par = int((~ok_par).sum())
            fallos_fuera = int((~ok_par & ~sub).sum())
            ac_sub = float(ok_par[sub].mean())
            print("  {:38s} {:6.1%}".format("mismo modelo, sin las dudosas", ac_sub))
            print("\n  la aritmetica, para que no se lea como una mejora:")
            print("    par entero        {:3d} hojas, {:3d} fallos  -> {:5.1%}"
                  .format(n_par, fallos_par, ac_par))
            print("    salen las dudosas {:3d} hojas, {:3d} de ellas eran fallos"
                  .format(n_par - n_sub, fallos_fuera))
            print("    par sin dudosas   {:3d} hojas, {:3d} fallos  -> {:5.1%}"
                  .format(n_sub, fallos_par - fallos_fuera, ac_sub))
            print("\n  Lectura: de los {} fallos del par, {} son hojas cuya etiqueta"
                  " el propio\n  el experto da por discutible ({:.0f} %). El modelo no"
                  " mejora; lo que se mide es\n  cuanto del error que queda NO es"
                  " del modelo."
                  .format(fallos_par, fallos_fuera,
                          100.0 * fallos_fuera / max(1, fallos_par)))
            print("  Aviso de PROGRESO 39.1.b: con {} hojas, un punto porcentual"
                  " son {:.1f} hojas.".format(n_par, n_par / 100.0))

    with open(OUT / "clasificador_rasgos.csv", "w", newline="",
              encoding="utf-8") as fh:
        cols = (["ruta", "carpeta", "archivo", "variedad", "clase", "hoja_id"]
                + ["hoja_" + k for k in MED]
                + ["lado_" + k for k in MED] + ["lado_abs_" + k for k in MED]
                + ["largo_pulg", "ancho_pulg", "aspecto", "solidez", "cobertura",
                   "hoja_entre_util"]
                + ["fondo_" + k for k in MED])
        wr = csv.writer(fh)
        wr.writerow(cols)
        for r, a1, a2, a3, a4 in zip(filas, Xh, Xl, Xg, Xf):
            wr.writerow([r["ruta"], r["carpeta"], r["archivo"], r["variedad"],
                         r["clase"], r["hoja_id"]]
                        + [round(float(x), 4) for x in list(a1) + list(a2)
                           + list(a3) + list(a4)])
    print("\n-> {}".format(OUT / "clasificador_rasgos.csv"))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
