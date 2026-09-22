r"""
EL AGENTE, ENGANCHADO A LA CELDA DEL EXPERTO (Entorno1) POR SEÑALES DE 4 BITS.
(2026-09-11)

`Entorno1.rdk` para la banda, la hoja bajo la camara y levanta
`HOJA_EN_CAMARA`. Este proceso la clasifica y le devuelve el destino como un
**codigo de 4 bits**, con el handshake de cuatro fases que describe
`LEEME_Entorno1.md`. El robot no decide nada; el agente no mueve nada.

POR QUE ESTE PUENTE Y NO EL SERVIDOR DE §73
--------------------------------------------
La celda de §73 preguntaba por un socket JSON (`agente_servidor.py`, puerto
5555). La del experto habla por **parametros de estacion de RoboDK** (puerto
20500), que es lo que en el KUKA real seran cuatro entradas digitales. Como el
agente ya vive en un Python con el modelo, hablar directo con RoboDK deja una
pieza menos: no hay servidor en medio.

La decision la toma `clasifica.decide()`, la MISMA funcion que se mide en el
paper, a traves de `servicio.atiende`. No hay una segunda copia.

EL NUMERO DE CAJA SALE DE UN SOLO SITIO
----------------------------------------
`servicio.recipientes()` -> `celda_robodk.posiciones()`, y ese orden se puso el
11/09 igual al de la celda (1 cnt banda ... 10 hab xr_der, 11 revision). Antes
iba por calidad y **la hoja acertada habria ido a la caja equivocada sin que
nada fallara en voz alta**. Si mananal experto renumera las cajas, se cambia alli y
esto no se toca.

LA CLASE VERDADERA NO ENTRA EN LA DECISION
-------------------------------------------
La celda publica `HOJA_ID` -- el nombre del fichero de la foto, que no dice la
clase -- y el agente busca en `cola_hojas.csv` **la fotografia real** de esa
hoja. El nombre del objeto en RoboDK ('Hoja 7 - habano banda') lleva la etiqueta
y **no se lee**: es la verdad de campo, y leerla es lo que hacia que la celda
original "clasificara" sin clasificar (§73.1).

Y la foto real, no el render: Cam2D con licencia Free da 3.05 x 6.02 mm por
pixel (§70.1), de 5 a 10 veces mas basto que la resolucion con la que el modelo
mide. La camara dispara el evento; la foto decide (camino A, §65.3).

LA ESCALA, QUE ES LA TRAMPA DE SIEMPRE
---------------------------------------
Sin `--ancho-cinta` se usa **la mediana de la carpeta de cada foto**, que es
exactamente como se calculo al entrenar (§60.4, hecho cerrado 12). Medir la
cinta en cada foto cambia la clase en 8 hojas de 60. En la celda real, con
camara fija, esto es UNA calibracion el dia que se instala y entonces se pasa
con `--ancho-cinta`.

Uso:
    python scripts/agente_entorno1.py                  atiende las 50 de la cola
    python scripts/agente_entorno1.py --n 10           para despues de 10
    python scripts/agente_entorno1.py --ancho-cinta 88 calibracion unica
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rutas import REPO_RAIZ  # raiz del repositorio

RAIZ = REPO_RAIZ / "dataset"
OUT = REPO_RAIZ / "out"
COLA = OUT / "celda" / "entorno1" / "cola_hojas.csv"
REGISTRO = OUT / "celda" / "entorno1" / "registro_agente.csv"
# EL REGISTRO QUE PIDE LA CATEDRA (2026-09-15). Una linea JSON por decision con
# las CUATRO piezas del lazo -- percibe, puntua, razona, decide -- y su coste.
# "Sin ese registro no hay ablacion ni hay figura" (Semana 3, diapositiva 10).
# El CSV de al lado se queda: sirve para puntuar, este sirve para auditar.
DECISIONES = OUT / "celda" / "entorno1" / "logs" / "decisiones.jsonl"
CAJA_REVISION = 9        # 2026-09-15: era 11, con las dos cajas de media_banda

ENTRADAS = ["TIPO_B0", "TIPO_B1", "TIPO_B2", "TIPO_B3", "TIPO_VALIDO"]


# --------------------------------------------------------------- señales
def leer(RDK, nombre):
    try:
        return int(float(str(RDK.getParam(nombre)).strip()))
    except (TypeError, ValueError):
        return 0


def texto(RDK, nombre):
    v = RDK.getParam(nombre)
    v = "" if v is None else str(v).strip()
    return "" if v.lower() in ("none", "null") else v


def esperar(RDK, nombre, valor, limite, parar=None):
    t0 = time.time()
    while leer(RDK, nombre) != valor:
        if parar is not None and parar():
            return False
        if time.time() - t0 > limite:
            return False
        time.sleep(0.05)
    return True


def envia_codigo(RDK, codigo, espera_ack=60.0):
    """Fases 2 a 4 del handshake. Devuelve si el robot acuso recibo."""
    for k in range(4):
        RDK.setParam("TIPO_B%i" % k, str((codigo >> k) & 1))
    RDK.setParam("TIPO_VALIDO", "1")
    ok = esperar(RDK, "ORDEN_RECIBIDA", 1, espera_ack)
    RDK.setParam("TIPO_VALIDO", "0")
    if ok:
        esperar(RDK, "ORDEN_RECIBIDA", 0, espera_ack)
    return ok


# --------------------------------------------------------------- la cola
def lee_cola(ruta):
    """id -> lo que el agente necesita (la foto) y lo que solo sirve para puntuar."""
    d = {}
    with open(ruta, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            try:
                anc = float(r.get("ancho_cinta") or "")
            except ValueError:
                anc = None
            d[r["id"]] = {"orden": int(r["orden"]), "foto": r["foto"],
                          "variedad": r["variedad"], "clase": r["clase"],
                          "caja_real": int(r["caja_real"]),
                          "ancho_cinta": anc}
    return d


def anchos_por_carpeta():
    import zonas
    return zonas.ancho_cinta_por_carpeta(OUT)


def ancho_de(foto, tabla):
    """La mediana de SU carpeta: la misma escala con la que se entreno."""
    try:
        rel = Path(foto).resolve().relative_to(RAIZ)
    except ValueError:
        return None
    return tabla.get("/".join(rel.parts[:-1]))


# --------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cola", default=str(COLA))
    ap.add_argument("--registro", default=str(REGISTRO))
    ap.add_argument("--n", type=int, default=0, help="0 = todas las de la cola")
    ap.add_argument("--ancho-cinta", type=float, default=None,
                    help="calibracion unica en PX DE MASCARA; sin ella se usa la "
                         "mediana de la carpeta de cada foto (§60.4)")
    ap.add_argument("--espera", type=float, default=900.0,
                    help="s maximos esperando una hoja bajo la camara")
    a = ap.parse_args()

    from robodk.robolink import Robolink
    import clasifica as CL
    import servicio as SV

    cola = lee_cola(a.cola)
    total = a.n if a.n else len(cola)
    tabla = anchos_por_carpeta() if a.ancho_cinta is None else {}

    print("cargando el modelo...")
    t0 = time.time()
    modelo = CL.carga_modelo()
    rec = SV.recipientes()
    print("modelo cargado en %.2f s" % (time.time() - t0))

    RDK = Robolink()
    RDK.setParam("CONTROL", "RUN")
    for n in ENTRADAS:
        RDK.setParam(n, "0")

    print("=" * 78)
    print("AGENTE BRYAN  ·  enganchado a Entorno1  ·  %i hojas en la cola" % len(cola))
    print("=" * 78)
    # El mensaje dice LA POLITICA QUE DE VERDAD CORRE. Hasta el 15/09 anunciaba
    # la politica "prudente" (umbral sobre el acierto calibrado + veto al cajon)
    # cuando la desplegada es "planta", que solo mira la confianza cruda. Un
    # mensaje que describe otra cosa es peor que no tener mensaje.
    if CL.POLITICA == "planta":
        print("Politica 'planta': firma si la confianza de calidad >= %.2f."
              % CL.PISO)
        if CL.PISO_CAJON:
            for cj, pz in sorted(CL.PISO_CAJON.items()):
                if pz > 1.0:
                    print("  cajon '%s': NUNCA se firma, siempre a revision." % cj)
                else:
                    print("  cajon '%s': piso propio de %.2f." % (cj, pz))
        print("El acierto calibrado se anota en el registro, pero NO decide.")
    else:
        print("Politica 'prudente': umbral %.0f %% + %.0f de margen sobre el "
              "acierto calibrado," % (100 * CL.UMBRAL, 100 * CL.MARGEN))
        print("con veto al cajon incapaz (§72.7).")
    print("Lo que no llegue va al codigo %i y lo mira una persona." % CAJA_REVISION)
    print("Escala: %s" % ("--ancho-cinta %.1f px" % a.ancho_cinta if a.ancho_cinta
                          else "mediana de la carpeta de cada foto (como al entrenar)"))
    print()

    nuevo = not Path(a.registro).exists()
    fh = open(a.registro, "a", encoding="utf-8", newline="")
    DECISIONES.parent.mkdir(parents=True, exist_ok=True)
    fj = open(DECISIONES, "w", encoding="utf-8")
    w = csv.writer(fh, delimiter=";")
    if nuevo:
        w.writerow(["fecha", "orden", "id", "variedad_real", "clase_real", "caja_real",
                    "variedad_dicha", "clase_dicha", "confianza", "acierto_medido",
                    "decision", "veto_cajon", "codigo", "clase_ok", "caja_ok",
                    "segundos"])

    hechas, clase_ok, caja_ok, aceptadas, errores = 0, 0, 0, 0, 0
    con_etiqueta = 0
    try:
        while hechas < total:
            print("esperando hoja bajo la camara...")
            if not esperar(RDK, "HOJA_EN_CAMARA", 1, a.espera):
                print("no llego ninguna hoja. Esta corriendo Control_Senales?")
                break

            ident = texto(RDK, "HOJA_ID")
            info = cola.get(ident)
            t_ini = time.time()

            if info is None:
                # Sin saber que hoja es no se puede clasificar: al 11, que es el
                # recipiente que existe justo para lo que el agente no resuelve.
                codigo, j = CAJA_REVISION, {}
                print("  [!] HOJA_ID='%s' no esta en la cola -> codigo 11" % ident)
                errores += 1
            else:
                ac = (a.ancho_cinta if a.ancho_cinta is not None
                      else (info.get("ancho_cinta")
                            or ancho_de(info["foto"], tabla)))
                try:
                    j = SV.atiende(info["foto"], modelo, rec, CL.UMBRAL, ac, False)
                    codigo = int(j["recipiente"])
                except Exception as ex:
                    codigo, j = CAJA_REVISION, {"error": str(ex)}
                    print("  [!] %s: %s -> codigo 11" % (type(ex).__name__, ex))
                    errores += 1

            seg = time.time() - t_ini
            if not envia_codigo(RDK, codigo):
                print("  [!] el robot no acuso recibo del codigo %i" % codigo)

            hechas += 1
            if info is not None:
                dice_clase = (j.get("variedad"), j.get("calidad"))
                # caja_real = 0 es una hoja SIN etiqueta: no se puntua, se anota.
                sin_etiqueta = info["caja_real"] == 0
                ok_c = (not sin_etiqueta) and dice_clase == (info["variedad"], info["clase"])
                ok_b = (not sin_etiqueta) and codigo == info["caja_real"]
                clase_ok += ok_c
                caja_ok += ok_b
                con_etiqueta += (not sin_etiqueta)
                aceptadas += (j.get("decision") == "aceptar")
                print("  %2i/%i  %-28s %s %-11s -> dice %s %-11s  conf %.3f  "
                      "codigo %2i  %s"
                      % (info["orden"], total, ident[:28], info["variedad"][:3],
                         info["clase"], str(j.get("variedad"))[:3],
                         str(j.get("calidad")), j.get("confianza_calidad") or 0.0,
                         codigo,
                         "sin etiqueta" if sin_etiqueta else
                         ("CAJA OK" if ok_b else
                          ("revision" if codigo == CAJA_REVISION else "CAJA MAL"))))
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), info["orden"], ident,
                            info["variedad"], info["clase"], info["caja_real"],
                            j.get("variedad"), j.get("calidad"),
                            j.get("confianza_calidad"), j.get("acierto_medido_total"),
                            j.get("decision"), j.get("veto_cajon"), codigo,
                            int(ok_c), int(ok_b), round(seg, 2)])
                fj.write(json.dumps(dict(
                    id=ident, orden=info["orden"],
                    percibe=dict(foto=info["foto"],
                                 escala=j.get("escala"),
                                 avisos=j.get("avisos") or []),
                    puntua=dict(variedad=j.get("variedad"),
                                calidad=j.get("calidad"),
                                confianza_variedad=j.get("confianza_variedad"),
                                confianza_calidad=j.get("confianza_calidad"),
                                segunda=j.get("segunda_opcion"),
                                acierto_calibrado=j.get("acierto_medido_total")),
                    razona=dict(regla="politica '%s': firma si confianza >= %s"
                                      % (CL.POLITICA, CL.PISO_CAJON.get(
                                          str(j.get("calidad")), CL.PISO)),
                                piso=CL.PISO_CAJON.get(str(j.get("calidad")),
                                                       CL.PISO),
                                sin_llm=True,
                                defectos=[d.get("tipo") + "/" + str(d.get("zona"))
                                          for d in (j.get("defectos") or [])][:8]),
                    decide=dict(accion=("recipiente_%02d" % codigo),
                                codigo=codigo,
                                revision=bool(codigo == CAJA_REVISION),
                                guarda="ninguna hoja se descarta"),
                    verdad=dict(clase=info["clase"], variedad=info["variedad"],
                                caja=info["caja_real"], acerto_caja=bool(ok_b)),
                    costo=dict(segundos=round(seg, 3), llamadas_llm=0, tokens=0),
                    fecha=time.strftime("%Y-%m-%dT%H:%M:%S")),
                    ensure_ascii=False) + "\n")
                fj.flush()
            else:
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "", ident, "", "", "",
                            "", "", "", "", "sin_cola", "", codigo, "", "",
                            round(seg, 2)])
            fh.flush()
    except KeyboardInterrupt:
        print("\ncortado a mano")
    finally:
        fh.close()

    print()
    print("=" * 78)
    print("%i hojas atendidas" % hechas)
    if hechas:
        if con_etiqueta:
            print("  clase correcta         %i de %i" % (clase_ok, con_etiqueta))
            print("  caja correcta          %i de %i   (el resto, a revision)" % (caja_ok, con_etiqueta))
        if con_etiqueta < hechas:
            print("  sin etiqueta           %i  (no se puntuan: falta la verdad de campo)"
                  % (hechas - con_etiqueta))
        print("  resueltas sin persona  %i de %i" % (aceptadas, hechas))
        if errores:
            print("  hojas que fallaron     %i  (fueron al 11, no se perdio ninguna)" % errores)
    print("registro: %s" % a.registro)
    print()
    if con_etiqueta:
        print("AVISO: si las hojas de la cola son del conjunto de ENTRENAMIENTO, esto")
        print("mide que la cadena esta bien conectada, NO el acierto (§73.6).")
    else:
        print("Ninguna hoja de esta tanda traia etiqueta: lo que hay es lo que el")
        print("agente dijo. Para que mida algo hace falta que el experto diga su clase.")


if __name__ == "__main__":
    main()
