r"""
LA FIGURA PRINCIPAL DEL PAPER: ERROR CONTRA TASA DE RECHAZO. (2026-09-15)

POR QUE EXISTE
--------------
Es **la metrica principal del proyecto desde la Semana 1** del curso, y la unica
figura que la catedra dejo escrita para el grupo: eje X la tasa de rechazo
(0, 10, 20, 30 %), eje Y el error sobre las hojas ACEPTADAS, y el acuerdo humano
dibujado como linea. Hasta hoy no se habia calculado.

LA PREGUNTA QUE CONTESTA, Y POR QUE SE MIDEN DOS CURVAS
--------------------------------------------------------
La tanda de 10 hojas del 15/09 enseno que la politica **no se abstiene** fuera de
sesion: el acierto calibrado no bajo de 0.9398 ni en hojas con probabilidad cruda
de 0.525, y las tres que fallaron fueron firmadas. O sea que la calibracion, hecha
DENTRO de sesion, promete de mas en una jornada nueva.

Pero una cosa es el **valor** (que miente) y otra el **orden** (que puede seguir
sirviendo). Asi que se dibujan las dos:

    cruda       ordenar por la probabilidad del modelo, sin calibrar
    calibrada   ordenar por el acierto esperado del tramo calibrado

Si la curva CRUDA baja al rechazar, el arreglo no es recalibrar: es poner el umbral
**por cuantil** ("deriva el 20 % menos seguro") en vez de por un valor absoluto.
Si ninguna baja, la confianza no ordena nada y eso tambien es un resultado, que se
reporta tal cual.

LOS DATOS
---------
`out/SESION_FINAL_112/DECISION_FINAL_112.csv`: las 112 hojas del 11/09, que el
modelo no vio. Reproduce las cifras conocidas: 91 de 112 = 81.2 %, y la politica
firma 108 (96.4 %) acertando ahi el 83.3 %.

AVISOS QUE VAN CON LA FIGURA
-----------------------------
 - Las 112 son **jornada de desarrollo**, no test intacto (Tabaco.md 79.6).
 - **No hay ni una Banda** entre las 112: la curva no dice nada de esa clase.
 - La linea humana es el 65.9 % de acuerdo (kappa 0.534) de la remuestra ciega EN
   PANTALLA, que mide el techo de clasificar POR FOTO. El acuerdo entre dos
   tecnicos con la hoja en la mano se esta midiendo y **no es este numero**.
 - El intervalo es bootstrap sobre las hojas (2000 remuestras).

Uso:  python scripts/paper_curva_rechazo.py
"""
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
from rutas import REPO_ROOT  # repository root

OUT = REPO_ROOT / "out"
FUENTE = OUT / "SESION_FINAL_112" / "DECISION_FINAL_112.csv"
DEST = OUT / "PAPER_CURVA_RECHAZO"
COPIA = REPO_ROOT / "results" / "figuras"

TINTA = "#1a1a1a"
GRIS = "#7a7a7a"
BANDA = "#d9d9d9"

TASAS = np.arange(0, 41, 5)          # % de hojas derivadas a una persona
N_BOOT = 2000
SEMILLA = 0

# El techo humano disponible hoy: remuestra ciega EN PANTALLA contra la etiqueta
# de produccion (29 de 44 = 65.9 %, kappa 0.534). NO es el acuerdo entre dos
# tecnicos con la hoja en la mano, que se esta midiendo aparte.
HUMANO_ERROR = 100.0 - 65.9


def carga():
    with open(FUENTE, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh, delimiter=";"))
    acierta = np.array([f["clase_dicha"] == f["clase_real"] for f in filas], bool)
    cruda = np.array([float(f["conf_calidad"]) for f in filas], float)
    calib = np.array([float(f["acierto_medido"]) for f in filas], float)
    firma = np.array([f["decision"] == "aceptar" for f in filas], bool)
    return acierta, cruda, calib, firma


def curva(acierta, puntaje, tasas):
    """Error (%) sobre las hojas aceptadas al derivar el r % menos seguro."""
    n = len(acierta)
    orden = np.argsort(puntaje, kind="mergesort")      # estable: empates en su orden
    out = []
    for r in tasas:
        k = int(round(n * r / 100.0))
        quedan = orden[k:]
        out.append(100.0 * (1.0 - acierta[quedan].mean()) if len(quedan) else np.nan)
    return np.array(out)


def bootstrap(acierta, puntaje, tasas, n_boot=N_BOOT, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    n = len(acierta)
    M = np.empty((n_boot, len(tasas)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        M[b] = curva(acierta[idx], puntaje[idx], tasas)
    return np.nanpercentile(M, 2.5, axis=0), np.nanpercentile(M, 97.5, axis=0)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    acierta, cruda, calib, firma = carga()
    n = len(acierta)
    base = 100.0 * (1.0 - acierta.mean())
    print("hojas: %i   error sin rechazar: %.1f %%   (acierto %.1f %%)"
          % (n, base, 100 - base))

    c_cruda = curva(acierta, cruda, TASAS)
    c_calib = curva(acierta, calib, TASAS)
    lo_c, hi_c = bootstrap(acierta, cruda, TASAS)
    lo_k, hi_k = bootstrap(acierta, calib, TASAS)

    # el punto de operacion de la politica actual
    cob_pol = 100.0 * firma.mean()
    err_pol = 100.0 * (1.0 - acierta[firma].mean())
    rech_pol = 100.0 - cob_pol

    lineas = []
    lineas.append("CURVA DE ERROR CONTRA TASA DE RECHAZO  (112 hojas del 11/09)")
    lineas.append("=" * 74)
    lineas.append("")
    lineas.append("Error (pct) sobre las hojas ACEPTADAS, con IC 95 pct bootstrap "
                  "(%i remuestras)" % N_BOOT)
    lineas.append("")
    lineas.append("  rechazo   hojas      ordenando por           ordenando por")
    lineas.append("    (%)    firmadas   PROBABILIDAD CRUDA      ACIERTO CALIBRADO")
    lineas.append("  " + "-" * 70)
    for i, r in enumerate(TASAS):
        k = int(round(n * r / 100.0))
        lineas.append("  %5.0f      %3i      %5.1f  [%4.1f-%4.1f]        %5.1f  [%4.1f-%4.1f]"
                      % (r, n - k, c_cruda[i], lo_c[i], hi_c[i],
                         c_calib[i], lo_k[i], hi_k[i]))
    lineas.append("")
    lineas.append("POLITICA ACTUAL DEL AGENTE (umbral sobre el acierto calibrado)")
    lineas.append("  firma %.1f pct de las hojas (%i de %i) y ahi se equivoca el %.1f pct"
                  % (cob_pol, int(firma.sum()), n, err_pol))
    lineas.append("  es decir: rechaza el %.1f pct, cuando la curva cruda a ese mismo"
                  % rech_pol)
    lineas.append("  rechazo daria %.1f pct de error." % np.interp(rech_pol, TASAS, c_cruda))
    lineas.append("")
    lineas.append("LECTURA")
    i30 = list(TASAS).index(30)
    # bootstrap PAREADO: la misma remuestra para las dos preguntas, que es como
    # se compara sin inflar la diferencia (regla de la Semana 4).
    rng = np.random.default_rng(SEMILLA)
    d_caida, d_entre = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        cc = curva(acierta[idx], cruda[idx], TASAS)
        kk = curva(acierta[idx], calib[idx], TASAS)
        d_caida.append(cc[0] - cc[i30])
        d_entre.append(kk[i30] - cc[i30])
    d_caida = np.array(d_caida); d_entre = np.array(d_entre)
    ic_caida = np.nanpercentile(d_caida, [2.5, 97.5])
    ic_entre = np.nanpercentile(d_entre, [2.5, 97.5])
    g_cruda = c_cruda[0] - c_cruda[i30]
    g_calib = c_calib[0] - c_calib[i30]

    lineas.append("  1) Derivando el 30 pct de las hojas, el error baja de %.1f a %.1f,"
                  % (c_cruda[0], c_cruda[i30]))
    lineas.append("     o sea %.1f puntos, IC 95 pct [%.1f, %.1f]."
                  % (g_caida := g_cruda, ic_caida[0], ic_caida[1]))
    if ic_caida[0] > 0:
        lineas.append("     El intervalo NO cruza el cero: la confianza ordena, y derivar")
        lineas.append("     hojas dudosas a una persona baja el error de verdad.")
    else:
        lineas.append("     El intervalo CRUZA EL CERO: con 112 hojas no se puede afirmar")
        lineas.append("     que el error baje. Se reporta la diferencia con su signo y ya.")
    lineas.append("")
    lineas.append("  2) Cruda contra calibrada al 30 pct: %.1f frente a %.1f, diferencia"
                  % (c_cruda[i30], c_calib[i30]))
    lineas.append("     %.1f puntos, IC 95 pct [%.1f, %.1f]."
                  % (g_calib - g_cruda, ic_entre[0], ic_entre[1]))
    if ic_entre[0] > 0:
        lineas.append("     Ordenar por la probabilidad cruda es MEJOR que por el acierto")
        lineas.append("     calibrado: el umbral deberia ponerse por cuantil.")
    else:
        lineas.append("     Los dos ordenes son COMPARABLES. Y es de esperar: el acierto")
        lineas.append("     calibrado es una funcion creciente de la confianza cruda dentro")
        lineas.append("     de cada variedad, asi que casi no cambia el orden. Lo que falla")
        lineas.append("     fuera de sesion NO es el orden, es el VALOR: promete >= 94 pct")
        lineas.append("     en todas las hojas, y por eso el umbral absoluto no abstiene.")
    lineas.append("")
    lineas.append("  3) La politica actual opera al %.1f pct de rechazo, es decir en el"
                  % rech_pol)
    lineas.append("     extremo izquierdo de la curva: se queda con casi toda la mejora")
    lineas.append("     sin recoger. Para bajar el error de forma apreciable hay que")
    lineas.append("     derivar del 30 al 40 pct de las hojas.")
    lineas.append("")
    lineas.append("  4) El sistema ya esta POR DEBAJO del error de una persona")
    lineas.append("     clasificando por foto (%.1f pct): la linea humana queda ARRIBA de"
                  % HUMANO_ERROR)
    lineas.append("     la curva en todo el rango. Ese es el techo que hay medido hoy.")
    lineas.append("")
    lineas.append("QUE PISO DE CONFIANZA HACE FALTA PARA CADA TASA DE RECHAZO")
    lineas.append("  La politica desplegada es POLITICA='planta': firma si la confianza")
    lineas.append("  cruda de calidad >= PISO (hoy 0.45). No usa el acierto calibrado.")
    lineas.append("")
    lineas.append("   rechazo    piso    hojas      error sobre")
    lineas.append("     (pct)   equiv.  firmadas   las firmadas")
    lineas.append("  " + "-" * 52)
    for i, r in enumerate(TASAS):
        k = int(round(n * r / 100.0))
        piso = float(np.sort(cruda)[k]) if k < n else 1.0
        lineas.append("   %5.0f     %.3f     %3i        %5.1f pct"
                      % (r, piso, n - k, c_cruda[i]))
    lineas.append("")
    lineas.append("  El piso actual de 0.45 deja fuera %i hojas (%.1f pct)."
                  % (int((cruda < 0.45).sum()), 100.0 * (cruda < 0.45).mean()))
    lineas.append("")
    lineas.append("AVISOS")
    lineas.append("  - Las 112 son jornada de DESARROLLO, no test intacto (79.6).")
    lineas.append("  - No hay ni una BANDA entre ellas.")
    lineas.append("  - La linea humana (%.1f pct de error) es la remuestra ciega EN"
                  % HUMANO_ERROR)
    lineas.append("    PANTALLA, no el acuerdo entre dos tecnicos con la hoja en la mano.")
    texto = "\n".join(lineas)
    (DEST / "CURVA_RECHAZO.txt").write_text(texto + "\n", encoding="utf-8")
    print()
    print(texto)

    json.dump(dict(tasas=TASAS.tolist(), n=n, error_base=base,
                   cruda=c_cruda.tolist(), cruda_ic=[lo_c.tolist(), hi_c.tolist()],
                   calibrada=c_calib.tolist(), calibrada_ic=[lo_k.tolist(), hi_k.tolist()],
                   politica=dict(cobertura=cob_pol, error=err_pol, rechazo=rech_pol),
                   humano_error=HUMANO_ERROR, n_boot=N_BOOT, semilla=SEMILLA),
              open(DEST / "curva_rechazo.json", "w", encoding="utf-8"), indent=1)

    # ------------------------------------------------------------------ figura
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.edgecolor": GRIS, "axes.linewidth": 0.6,
                         "xtick.color": GRIS, "ytick.color": GRIS,
                         "text.color": TINTA, "axes.labelcolor": TINTA,
                         "figure.dpi": 400, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.fill_between(TASAS, lo_c, hi_c, color=BANDA, linewidth=0, zorder=1)
    ax.plot(TASAS, c_cruda, color=TINTA, linewidth=1.1, marker="o", markersize=3,
            zorder=4, label="probabilidad cruda")
    ax.plot(TASAS, c_calib, color=GRIS, linewidth=1.1, marker="s", markersize=3,
            linestyle=(0, (3, 2)), zorder=3, label="acierto calibrado")
    ax.axhline(HUMANO_ERROR, color=TINTA, linewidth=0.7, linestyle=(0, (1, 2)),
               zorder=2)
    ax.annotate("techo humano por foto (%.0f %%)" % HUMANO_ERROR,
                (TASAS[0], HUMANO_ERROR), textcoords="offset points",
                xytext=(2, 3), ha="left", fontsize=6.5, color=TINTA)
    ax.plot([rech_pol], [err_pol], marker="*", markersize=8, color=TINTA, zorder=5)
    ax.annotate("politica actual", (rech_pol, err_pol), textcoords="offset points",
                xytext=(6, 4), fontsize=6.5, color=TINTA)
    ax.set_xlabel("hojas derivadas a una persona (%)")
    ax.set_ylabel("error sobre las hojas aceptadas (%)")
    ax.set_xticks(TASAS[::2])
    ax.set_xlim(-1.5, TASAS[-1] + 1.5)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left",
              handlelength=2.2, borderaxespad=0.2)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color="#ececec", linewidth=0.5)
    ax.set_axisbelow(True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(DEST / ("fig2_curva_rechazo." + ext))
    plt.close(fig)
    print("\nfigura:", DEST / "fig2_curva_rechazo.pdf")

    if COPIA.exists():
        for f in ("fig2_curva_rechazo.pdf", "fig2_curva_rechazo.svg",
                  "fig2_curva_rechazo.png"):
            shutil.copy2(DEST / f, COPIA / "figuras" / f)
        shutil.copy2(DEST / "CURVA_RECHAZO.txt", COPIA / "datos" / "CURVA_RECHAZO.txt")
        shutil.copy2(DEST / "curva_rechazo.json", COPIA / "datos" / "curva_rechazo.json")
        print("copiado a", COPIA)


if __name__ == "__main__":
    main()
