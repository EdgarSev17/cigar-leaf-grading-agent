r"""
EL SERVICIO DE DECISION DE LA CELDA: UNA FOTO ENTRA, SALE UN RECIPIENTE.
(2026-09-08)

Punto E.5 de «LO SIGUIENTE». Es lo UNICO que el robot necesita llamar, con las
palabras del experto del 2026-09-06 (§65, hecho cerrado 17):

    "la camara dispara al agente y el agente le dice a que recipiente va.
     Diez recipientes = 5 calidades x 2 variedades"

Y once, no diez: el once es la revision manual, que es lo que hace el agente
cuando el acierto medido no llega al umbral (§63). Sin ese recipiente, «el
agente decide bajo confianza» no tendria donde poner lo que no decide.

QUE DEVUELVE, Y POR QUE ESAS DOS COSAS JUNTAS
---------------------------------------------
    A DONDE VA    variedad, calidad, confianza, acierto medido y numero de
                  recipiente (1-11).
    POR DONDE SE
    AGARRA        el punto de agarre POR LA REGLA (§68): centro de la barra de
                  ventosas en el marco de la hoja, su giro y cuanta copa cae en
                  la zona que la regla descarta.

Van juntas porque son las dos cosas que el brazo necesita del agente en el mismo
instante, y las dos salen de la misma foto. Separarlas obligaria a medir la hoja
dos veces.

DE DONDE SALE CADA PARTE -- NINGUNA ES CODIGO NUEVO
---------------------------------------------------
    la clase      `clasifica.decide()`, la MISMA funcion que la linea de
                  comandos. No hay una segunda copia de la decision.
    el agarre     `agarre.busca()`, el mismo que se corrio sobre las 714 hojas.
    el recipiente `celda_robodk.posiciones()`, el mismo orden que la estacion.

Si manana se cambia el orden de los recipientes en la celda, este servicio
cambia solo. Es a proposito: el error de §29 (izquierda/derecha) fue de dos
sitios que decian lo mismo de dos maneras.

LA ESCALA, QUE ES LA TRAMPA DE ESTE GUION
-----------------------------------------
`--ancho-cinta` es la calibracion de la camara en pixeles de mascara, y se le
pasa **a la vez** al clasificador y al agarre. No es un adorno: §60.4 midio que
usar una escala distinta al clasificar que al entrenar cuesta 8 hojas de 60.
Con una camara fija se mide una vez el dia que se instala y no vuelve a tocarse
--y entonces la cinta metrica no tiene que salir en la foto.

Uso:
    python scripts/servicio.py foto.jpg
    python scripts/servicio.py foto.jpg --json
    python scripts/servicio.py *.jpg --ancho-cinta 118 --json   (lote, un modelo)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import clasifica as CL                                              # noqa: E402
import agarre as AG                                                 # noqa: E402
from malla_3d import canoniza                                       # noqa: E402
from celda_robodk import posiciones                                 # noqa: E402

REVISION = "revision"


def recipientes():
    """{(variedad, clase): (numero, nombre)} + el de revision. Del mismo sitio."""
    d = {}
    for k, (nombre, var, cls, x, y, ang) in enumerate(posiciones(), 1):
        d[(var, cls) if cls else REVISION] = (k, nombre)
    return d


def agarre_de_la_foto(foto, ancho_cinta=None, angulos=None):
    """Punto de agarre por la regla, calculado sobre la foto que llega.

    OJO con el modo de escala: `canoniza(..., "foto")` mide la cinta EN CADA
    FOTO y solo cae en el ancho de sesion si no la ve. Eso es justo lo que
    PROGRESO 60.4 midio que cuesta 8 hojas de 60. Cuando hay calibracion de
    camara se pide modo "carpeta", que es el que usa la tuberia.
    """
    cont, mm_px, diag = canoniza(foto, "carpeta" if ancho_cinta else "foto",
                                 ancho_cinta)
    if cont is None:
        return None, diag.get("error", "no se pudo canonizar")
    fuera, dentro = cont
    m, (ox, oy) = AG.desde_contornos(fuera, dentro)
    desc, dg = AG.zona_descartada(m)
    if desc is None:
        return None, dg.get("error", "sin zona")
    r = AG.busca(m, desc, angulos if angulos is not None else
                 list(np.arange(-30.0, 30.1, 5.0)))
    if not r["apoyo"]:
        return None, "ninguna posicion apoya las cuatro copas"
    return dict(x_mm=round(r["x_px"] * AG.PASO + ox, 1),
                y_mm=round(r["y_px"] * AG.PASO + oy, 1),
                giro_grados=round(r["angulo"], 1),
                holgura_mm=round(r["holgura_mm"], 1),
                area_en_zona=round(r["area_en_zona"], 3),
                r_max_regla_mm=round(r["r_max_regla"], 2)), None


def atiende(foto, modelo, rec, umbral=CL.UMBRAL, ancho_cinta=None, con_agarre=True):
    """La respuesta entera para una foto. No imprime: devuelve el diccionario."""
    t0 = time.time()
    D = CL.decide(foto, umbral, ancho_cinta, False, modelo)
    j = CL.a_json(D)

    if D["acepta"]:
        n, nombre = rec[(D["var"], D["cal"])]
    else:
        n, nombre = rec[REVISION]
    j["recipiente"] = n
    j["recipiente_nombre"] = nombre

    j["agarre"] = None
    j["agarre_aviso"] = None
    if con_agarre:
        try:
            g, err = agarre_de_la_foto(foto, ancho_cinta)
            j["agarre"], j["agarre_aviso"] = g, err
        except Exception as ex:                  # el brazo no se queda sin clase
            j["agarre_aviso"] = "{}: {}".format(type(ex).__name__, ex)
    j["segundos"] = round(time.time() - t0, 2)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fotos", nargs="+")
    ap.add_argument("--umbral", type=float, default=CL.UMBRAL)
    ap.add_argument("--ancho-cinta", type=float, default=None,
                    help="calibracion de la camara, en px de mascara (PROGRESO 60.4)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sin-agarre", action="store_true",
                    help="solo la clase y el recipiente, sin medir el agarre")
    a = ap.parse_args()

    t0 = time.time()
    modelo = CL.carga_modelo()
    rec = recipientes()
    carga = time.time() - t0

    if a.ancho_cinta is None and not a.json:
        print()
        print("  [!] SIN CALIBRACION DE CAMARA (--ancho-cinta).")
        print("      Se mide la cinta en cada foto, que NO es como se entreno:")
        print("      PROGRESO 60.4 midio que ese modo cambia la clase en 8 hojas")
        print("      de 60. En la celda la camara es fija y se calibra una vez.")

    salida = []
    for f in a.fotos:
        try:
            salida.append(atiende(f, modelo, rec, a.umbral, a.ancho_cinta,
                                  not a.sin_agarre))
        except Exception as ex:
            salida.append({"archivo": Path(f).name, "error": str(ex),
                           "recipiente": rec[REVISION][0],
                           "recipiente_nombre": rec[REVISION][1]})

    if a.json:
        print(json.dumps(salida if len(salida) > 1 else salida[0],
                         indent=2, ensure_ascii=False, default=str))
        return

    print()
    print("  modelo cargado en {:.2f} s. El arranque de verdad se ve en la PRIMERA"
          " hoja".format(carga))
    print("  (importar y calentar la tuberia); de la segunda en adelante es el"
          " ritmo real.")
    print()
    for j in salida:
        if "error" in j:
            print("  {:<26s} ERROR: {}  -> recipiente {} ({})"
                  .format(j["archivo"][:26], j["error"], j["recipiente"],
                          j["recipiente_nombre"]))
            continue
        ac = j["acierto_medido_total"]
        print("  {:<26s} {:<12s} {:<12s} acierto {:>5s}  ->  RECIPIENTE {:2d}  {}"
              .format(j["archivo"][:26], j["variedad"], j["calidad"],
                      "{:.0f} %".format(100 * ac) if ac is not None else "?",
                      j["recipiente"], j["recipiente_nombre"]))
        g = j.get("agarre")
        if g:
            print("     agarre  x {:7.1f}  y {:7.1f} mm   giro {:+.0f} grados   "
                  "holgura {:.1f} mm   {:.0f} % de copa en zona que se descarta"
                  .format(g["x_mm"], g["y_mm"], g["giro_grados"],
                          g["holgura_mm"], 100 * g["area_en_zona"]))
        elif j.get("agarre_aviso"):
            print("     agarre  NO CALCULADO: {}".format(j["agarre_aviso"]))
        print("     {:.2f} s".format(j["segundos"]))
    print()
    n = len(salida)
    tot = sum(j.get("segundos", 0) for j in salida)
    if n:
        print("  {} hojas en {:.1f} s   ->  {:.2f} s por hoja, {:.0f} por minuto"
              .format(n, tot, tot / n, 60.0 * n / tot if tot else 0))
    print("  Lo que este servicio NO puede prometer: que acierte con una camara,")
    print("  una mesa o un dia distintos de los del entrenamiento. No esta medido")
    print("  (PROGRESO 54.2), y lo decide la sesion intercalada de PROGRESO 54.3.")


if __name__ == "__main__":
    main()
