# -*- coding: utf-8 -*-
r"""
LA CELDA DE ESCOGIDA EN RoboDK -- se genera por guion (2026-09-06)

LA CELDA, DICHA POR EL EXPERTO
-------------------------
*"Va a haber una banda transportadora con el tabaco abierto uno por uno, entonces
el brazo va a estar listo y solo ve la hoja, la agarra y la pone donde va; la idea
es que la camara vea que hoja es y dispare el agente, y a su vez mueva el robot
indicandole donde es que va el tabaco. Va a tener 10 recipientes."*

**Diez recipientes = 5 calidades x 2 variedades**, que es exactamente la salida de
las dos etapas del modelo (§55). La arquitectura que se eligio midiendo resulta ser
la que pide la planta. Y hace falta **un recipiente once**: el agente resuelve solo
el 69.5 % de las hojas acertando el 93 % (§63); forzarlo a elegir uno de los diez
cuando no esta seguro tira por la borda esa garantia. El 11 es la salida de
revision, y es lo que hace que el 93 % sea un numero real (§65.5).

POR QUE LA ESTACION SE GENERA POR GUION Y NO SE ARMA A MANO
-----------------------------------------------------------
Es la misma disciplina que el resto del proyecto. Una estacion armada pinchando en
la interfaz no se puede reproducir, no dice **por que** una cota vale lo que vale,
y no se puede rehacer cuando cambia una medida. Aqui cada cota esta escrita con su
motivo, y volver a montar la celda entera cuesta segundos.

LAS COTAS, Y DE DONDE SALE CADA UNA
-----------------------------------
- **Hoja**: mediana 512 x 246 mm, maxima 620 x 351 (§65.1, las 714 mallas). De ahi
  salen el ancho de banda y el tamano de recipiente, no de un numero redondo.
- **Cuadro de la camara**: ~500 x 650 mm, el mismo de las fotos de entrenamiento.
  **Esto no es cosmetico**: si la camara se monta mas lejos "para que quepa
  cualquier cosa", la hoja sale mas pequena en el cuadro, y encogerla un 25 %
  cuesta el 20 % de las clases (§66.5).
- **Dos estaciones separadas, camara y recogida, a un paso de indexado**: la banda
  ya se para por el brazo, asi que la camara dispara en ese mismo paro. Coste
  anadido al ciclo: cero. Y el agente tiene todo el avance para contestar, cuando
  solo tarda **0.75 s** por hoja (medido el 06/09).
- **La camara NO ve al operario**: la estacion de vision va aguas abajo de donde
  el operario suelta la hoja. Las manos en el cuadro son un riesgo medido -- salen
  justo donde la hoja esta rota, asi que el modelo puede aprender "mano ->
  defecto" (§2).
- **Brazo**: Fanuc M-710iC/50, 2050 mm de alcance, el que hay en la biblioteca
  local. Manda el ALCANCE, no la carga: once recipientes que aceptan una hoja de
  medio metro ocupan mucho. La carga (50 kg para una hoja de gramos) sobra, y hay
  que decirlo en la tesis en vez de disimularlo.
- **Recipientes en dos arcos**, con el lado largo en direccion radial: asi cada uno
  ocupa 450 mm de arco en vez de 750, y los once caben en media vuelta.

LO QUE ESTE GUION COMPRUEBA, QUE ES LO QUE VALE
-----------------------------------------------
No basta con dibujar la celda bonita: hay que **demostrar que el brazo llega**. El
guion resuelve la cinematica inversa de los 24 puntos de trabajo (aproximacion y
destino de los 11 recipientes, mas la recogida) y **falla en voz alta** si alguno
no es alcanzable. Una celda que se ve bien y no llega a un recipiente es el tipo de
error que se descubre tarde.

    python scripts/celda_robodk.py                 monta la estacion y comprueba
    python scripts/celda_robodk.py --sin-abrir     solo genera la geometria STL
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rutas import REPO_RAIZ, ROBODK_RAIZ  # raiz del repositorio
sys.path.insert(0, str(ROBODK_RAIZ / "Python"))

from celda_geom import (escribe_stl, caja, cilindro, recipiente,       # noqa: E402
                        barra_ventosas)

OUT = REPO_RAIZ / "out"
GEO = OUT / "celda" / "geom"
LIB = ROBODK_RAIZ / "Library"
ROBOT = LIB / "Fanuc-M-710iC-50.robot"

# ---------------------------------------------------------------- cotas (mm)
BANDA_Y = -1400.0        # eje de la banda, delante del brazo
BANDA_Z = 900.0          # altura de trabajo de la banda
BANDA_ANCHO = 700.0      # > 620 mm, la hoja mas larga del conjunto (§65.1)
BANDA_LARGO = 4000.0
PASO = 800.0             # un indexado: hoja de 512 mm + holgura
X_CAMARA = -PASO         # estacion de vision, aguas arriba
X_RECOGE = 0.0           # estacion de recogida, delante del brazo
CAM_ALTURA = 1100.0      # sobre la banda: da un cuadro de ~500 x 650 mm

R_INT, R_EXT = 1100.0, 1900.0        # los dos arcos de recipientes
REC_RADIAL, REC_TANG, REC_ALTO = 750.0, 450.0, 250.0
REC_PEANA = 400.0                    # los cajones sobre peana, para vaciarlos
APROX = 250.0                        # altura de aproximacion sobre cada punto

# Los once recipientes. El orden NO es decorativo, y desde el 2026-09-11 NO LO
# DECIDE ESTE FICHERO: lo manda la celda del experto (`Entorno1.rdk`), donde las
# cajas ya estan construidas, numeradas y pintadas, y el robot recibe el destino
# como un codigo de 4 bits (`LEEME_Entorno1.md`). Alli el orden es alfabetico
# dentro de cada variedad:
#
#     1 cnt banda   2 cnt capa   3 cnt xl_izq   4 cnt xr_der
#     5 hab banda   6 hab capa   7 hab xl_izq   8 hab xr_der
#     9 revision / no reconocida
#
# EL 2026-09-15 SE QUITO LA MEDIA BANDA Y SE RENUMERO A 8 + 1. El modelo final
# es de CUATRO calidades por variedad, asi que las dos cajas de media_banda no
# se llenaban nunca. Se cambio a la vez aqui, en `Construir_Entorno.py`, en
# `Scripts/Control_Senales.py`, en `agente_entorno1.py` (CAJA_REVISION), en
# `cola_entorno1.py` y en la tabla del LEEME: los seis sitios o ninguno.
#
# Antes de hoy este fichero decia ["capa", "xl_izq", "xr_der", "banda",
# "media_banda"], o sea el orden de las dos etapas del modelo (§55). Con esa
# lista el agente acertaba la clase y **la hoja iba a la caja equivocada**, sin
# que nada fallara en voz alta: es el error de §29 -- dos sitios diciendo lo
# mismo de dos maneras. El numero de recipiente sale de UN solo sitio, y ese
# sitio es la celda.
CLASES = ["banda", "capa", "xl_izq", "xr_der"]
RECIPIENTES = (
    [("connecticut", c, R_INT) for c in CLASES] +
    [("habano", c, R_EXT) for c in CLASES] +
    [("revision", "", R_EXT)]
)
COLOR = {"capa": [0.20, 0.65, 0.25, 1], "xl_izq": [0.25, 0.45, 0.85, 1],
         "xr_der": [0.55, 0.35, 0.85, 1], "banda": [0.90, 0.65, 0.15, 1],
         "media_banda": [0.85, 0.35, 0.20, 1], "": [0.75, 0.15, 0.15, 1]}


def angulos(n, centro=90.0, paso_grados=None, radio=R_INT):
    """Reparte n recipientes sobre un arco, centrados detras del brazo."""
    if paso_grados is None:
        paso_grados = np.degrees((REC_TANG + 190.0) / radio)
    a0 = centro - paso_grados * (n - 1) / 2.0
    return [a0 + paso_grados * k for k in range(n)]


def posiciones():
    """(nombre, variedad, clase, x, y, angulo_grados) de los once recipientes."""
    int_ = [r for r in RECIPIENTES if r[2] == R_INT]
    ext = [r for r in RECIPIENTES if r[2] == R_EXT]
    out, k = [], 0
    for grupo, radio in ((int_, R_INT), (ext, R_EXT)):
        for (var, cls, _), ang in zip(grupo, angulos(len(grupo), radio=radio)):
            k += 1
            a = np.radians(ang)
            nombre = "{:02d}_{}".format(k, var if not cls else var + "_" + cls)
            out.append((nombre, var, cls, radio * np.cos(a), radio * np.sin(a), ang))
    return out


# ------------------------------------------------------------------ geometria
def construye_geometria():
    GEO.mkdir(parents=True, exist_ok=True)
    f = {}

    f["suelo"] = escribe_stl(GEO / "suelo.stl",
                             caja(9000, 7000, 20, (0, -300, -20)))

    # La banda. Va OSCURA a proposito: el modelo se entreno sobre un meson azul
    # marino / negro (§14.5), y los detectores miden su referencia de fondo en la
    # propia foto (§31, §37). Un fondo claro seria un cambio de dominio gratuito.
    t = caja(BANDA_LARGO, BANDA_ANCHO, 60, (0, BANDA_Y, BANDA_Z - 60))
    for x in (-BANDA_LARGO / 2 + 100, BANDA_LARGO / 2 - 100):
        for y in (BANDA_Y - BANDA_ANCHO / 2 + 60, BANDA_Y + BANDA_ANCHO / 2 - 60):
            t += caja(80, 80, BANDA_Z - 60, (x, y, 0))
    f["banda"] = escribe_stl(GEO / "banda.stl", t)

    # El portico de la camara. La camara mira hacia abajo desde 1100 mm, que es lo
    # que da el cuadro de ~500 x 650 mm de las fotos de entrenamiento (§66.5).
    zc = BANDA_Z + CAM_ALTURA
    t = caja(120, 120, zc, (X_CAMARA, BANDA_Y - BANDA_ANCHO / 2 - 200, 0))
    t += caja(120, BANDA_ANCHO + 400, 120, (X_CAMARA, BANDA_Y - 100, zc))
    t += caja(180, 180, 160, (X_CAMARA, BANDA_Y, zc - 160))
    f["portico"] = escribe_stl(GEO / "portico_camara.stl", t)

    # Las dos lamparas LED alargadas, como las de los mesones del experto (§28). No
    # son adorno: la luz constante es una condicion del montaje, y dibujarlas
    # obliga a colocarlas.
    t = []
    for dy in (-260, 260):
        t += caja(900, 90, 70, (X_CAMARA, BANDA_Y + dy, zc - 260))
    f["lamparas"] = escribe_stl(GEO / "lamparas.stl", t)

    # La mesa del operario, al principio de la banda y FUERA del cuadro de la
    # camara: ahi se abren las hojas, y las manos no pueden salir en la foto (§2).
    t = caja(1200, 800, 40, (-BANDA_LARGO / 2 - 500, BANDA_Y, 860))
    for dx in (-520, 520):
        for dy in (-330, 330):
            t += caja(60, 60, 860, (-BANDA_LARGO / 2 - 500 + dx, BANDA_Y + dy, 0))
    f["mesa_operario"] = escribe_stl(GEO / "mesa_operario.stl", t)

    for nombre, var, cls, x, y, ang in posiciones():
        t = recipiente(REC_RADIAL, REC_TANG, REC_ALTO, centro=(0, 0, REC_PEANA))
        t += caja(REC_RADIAL - 150, REC_TANG - 150, REC_PEANA, (0, 0, 0))
        f["rec_" + nombre] = escribe_stl(GEO / ("rec_" + nombre + ".stl"), t)

    geo_util, largo_util = barra_ventosas()
    f["util"] = escribe_stl(GEO / "util_ventosas.stl", geo_util)
    return f, largo_util


# -------------------------------------------------------------------- estacion
def monta(f, largo_util):
    from robodk import robolink as rl
    from robodk import robomath as rm

    RDK = rl.Robolink()
    RDK.Render(False)
    # UNA SOLA ESTACION, Y LIMPIA. Cada corrida abria una nueva y RoboDK se
    # llenaba de copias: la captura de control salio de una estacion vacia y
    # tarde un rato en entender por que. Se borran las anteriores por nombre --
    # nunca todas, para no tocar nada que el experto tenga abierto.
    # SE CONSTRUYE EN LA ESTACION ACTIVA, y no en una nueva. `AddStation` crea la
    # pestana pero `AddFile` siguio anadiendo a la que ya estaba activa: la celda
    # se montaba en una estacion y RoboDK ensenaba otra, asi que las capturas de
    # control salian VACIAS y los 25 puntos alcanzables eran de una escena que
    # nadie veia. Con `New()` no hay ambiguedad posible.
    NOMBRE = "Celda escogida tabaco"
    RDK.Command("Popups", "0")          # nada que bloquee al cerrar estaciones
    for _ in range(12):
        abiertas = RDK.getOpenStations()
        if not abiertas:
            break
        RDK.setActiveStation(abiertas[0])
        RDK.CloseStation()
    est = RDK.AddStation(NOMBRE)
    RDK.setActiveStation(est)

    robot = RDK.AddFile(str(ROBOT))
    if not robot.Valid():
        raise SystemExit("no se pudo cargar el brazo: {}".format(ROBOT))
    robot.setPose(rm.transl(0, 0, 0))

    def pon(clave, pose, color=None, nombre=None):
        it = RDK.AddFile(str(f[clave]))
        if not it.Valid():
            raise SystemExit("no se pudo cargar {}".format(f[clave]))
        it.setPose(pose)
        if color:
            it.setColor(color)
        if nombre:
            it.setName(nombre)
        return it

    pon("suelo", rm.transl(0, 0, 0), [0.72, 0.72, 0.70, 1], "Suelo")
    pon("banda", rm.transl(0, 0, 0), [0.13, 0.16, 0.28, 1], "Banda transportadora")
    pon("portico", rm.transl(0, 0, 0), [0.55, 0.57, 0.60, 1], "Portico de camara")
    pon("lamparas", rm.transl(0, 0, 0), [0.95, 0.95, 0.85, 1], "Lamparas LED")
    pon("mesa_operario", rm.transl(0, 0, 0), [0.60, 0.50, 0.40, 1],
        "Mesa del operario")

    util = RDK.AddFile(str(f["util"]), robot)
    util.setName("Barra de ventosas")
    util.setColor([0.30, 0.32, 0.35, 1])
    # El TCP en la cara de las copas, mirando hacia -Z de la brida.
    util.setPoseTool(rm.transl(0, 0, largo_util) * rm.rotx(np.pi))

    marco_banda = RDK.AddFrame("Banda")
    marco_banda.setPose(rm.transl(X_RECOGE, BANDA_Y, BANDA_Z))

    # LA HERRAMIENTA MIRA HACIA ABAJO, Y ESTO NO ES UN DETALLE DE POSTURA.
    # Los marcos tienen Z hacia arriba, asi que un objetivo sin girar le pide al
    # brazo que meta las ventosas **mirando al techo**. Con eso, 13 de los 24
    # puntos salian "no alcanzables" y el envolvente no tenia la culpa: a la altura
    # de los recipientes el M-710iC/50 llega de 600 a 2000 mm, medido.
    # `rotx(180)` deja el eje Z de la herramienta hacia -Z del mundo, y con eso el
    # eje X de la barra --las cuatro ventosas en linea-- queda a lo largo del eje
    # X del marco, que es el de la banda y el radial en cada recipiente. O sea, a
    # lo largo de la vena central de la hoja, que es como tiene que agarrar (§65.4).
    ABAJO = rm.rotx(np.pi)

    puntos = []          # (nombre, marco, pose) para comprobar alcance
    p_rec, p_reca = ABAJO, rm.transl(0, 0, APROX) * ABAJO
    RDK.AddTarget("recoger", marco_banda, robot).setPose(p_rec)
    RDK.AddTarget("recoger_aprox", marco_banda, robot).setPose(p_reca)
    puntos += [("recoger", marco_banda, p_rec),
               ("recoger_aprox", marco_banda, p_reca)]

    z_suelta = REC_PEANA + REC_ALTO + 120.0
    for nombre, var, cls, x, y, ang in posiciones():
        pon("rec_" + nombre,
            rm.transl(x, y, 0) * rm.rotz(np.radians(ang)),
            COLOR[cls], "Recipiente {}".format(nombre))
        mk = RDK.AddFrame("F_" + nombre)
        mk.setPose(rm.transl(x, y, z_suelta) * rm.rotz(np.radians(ang)))
        p0, p1 = ABAJO, rm.transl(0, 0, APROX) * ABAJO
        RDK.AddTarget("soltar_" + nombre, mk, robot).setPose(p0)
        RDK.AddTarget("aprox_" + nombre, mk, robot).setPose(p1)
        puntos += [("soltar_" + nombre, mk, p0), ("aprox_" + nombre, mk, p1)]

    marco_casa = RDK.AddFrame("Casa")
    marco_casa.setPose(rm.transl(0, -700, BANDA_Z + 500))
    RDK.AddTarget("casa", marco_casa, robot).setPose(ABAJO)
    puntos += [("casa", marco_casa, ABAJO)]

    RDK.Render(True)
    # Comprobacion barata que habria ahorrado una hora: ¿esta de verdad en la
    # estacion que se ve? Si esto sale a cero, todo lo que venga despues es de
    # una escena vacia.
    n_obj = len(RDK.ItemList(rl.ITEM_TYPE_OBJECT))
    n_rob = len(RDK.ItemList(rl.ITEM_TYPE_ROBOT))
    print("   estacion '{}': {} objetos, {} robot(s)"
          .format(RDK.ActiveStation().Name(), n_obj, n_rob))
    if n_obj < 15 or n_rob < 1:
        raise SystemExit("[!] la estacion activa no tiene la celda: no sigo")
    return RDK, robot, util, puntos


def comprueba_alcance(RDK, robot, util, puntos):
    """La comprobacion que hace que esto valga algo: ¿llega el brazo a todo?"""
    from robodk import robomath as rm
    robot.setPoseTool(util)
    malos = []
    for nombre, marco, pose in puntos:
        robot.setPoseFrame(marco)
        objetivo = marco.PoseAbs() * pose
        j = robot.SolveIK(objetivo, None, util.PoseTool(), rm.eye(4))
        ok = False
        try:
            ok = len(j.list()) >= 6 and all(v == v for v in j.list())
        except Exception:
            ok = False
        if not ok:
            malos.append(nombre)
    return malos


def mira_a(desde, hacia, arriba=(0.0, 0.0, 1.0)):
    """Pose de una camara colocada en `desde` y mirando a `hacia`.

    En RoboDK una camara mira a lo largo de su eje Z, asi que se construye la
    terna a partir de la direccion de vista en vez de a ojo con angulos de Euler.
    """
    from robodk import robomath as rm
    d = np.asarray(hacia, float) - np.asarray(desde, float)
    z = d / np.linalg.norm(d)
    a = np.asarray(arriba, float)
    if abs(float(z @ a)) > 0.999:
        a = np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return rm.Mat([[x[0], y[0], z[0], float(desde[0])],
                   [x[1], y[1], z[1], float(desde[1])],
                   [x[2], y[2], z[2], float(desde[2])],
                   [0.0, 0.0, 0.0, 1.0]])


def fotos_qc(RDK):
    """Imagenes de control de la celda, en la MISMA corrida que la construye.

    Es la regla que este proyecto ha pagado seis veces (§29.3, §32, §44.4, §53.3,
    §61.1): **toda geometria tiene que dejar una imagen que alguien mire**. Los
    25 puntos alcanzables no dicen nada de si la celda esta bien puesta -- eso se
    ve mirando. Y se hace aqui dentro porque entre procesos RoboDK puede cerrar
    la estacion y las capturas salen de una escena vacia (paso el 06/09).
    """
    from robodk import robomath as rm
    qc = OUT / "celda" / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    for f in qc.glob("*.png"):
        f.unlink()

    # LAS CAPTURAS DE CONTROL VAN POR `Snapshot`, Y HAY QUE SACAR LA VENTANA AL
    # FRENTE. `Snapshot` entrega lo que la ventana tenga RENDERIZADO: con RoboDK
    # detras devolvia siempre el mismo fotograma, y cuatro vistas distintas
    # salieron con el mismo md5. Y `setViewPose` acepta la pose y hasta la
    # devuelve al leerla, pero no mueve la vista; las vistas estandar hay que
    # pedirlas como acciones de menu.
    #
    # La camara 2D de la estacion (`Cam2D_Add`) si renderiza sola y seria mas
    # limpia, pero **la licencia gratuita la limita a 160x108**: al pedir mas
    # resolucion RoboDK responde "Invalid license". Se deja escrito porque afecta
    # al camino B de §65.3 -- clasificar desde el render-- que con esta licencia
    # no se puede hacer en serio. Al camino A, que es el recomendado y el que
    # esta medido, no le afecta: alli la percepcion corre sobre la foto REAL.
    import time
    RDK.ShowRoboDK()
    try:
        RDK.setWindowState(rl.WINDOWSTATE_MAXIMIZED)
    except Exception:
        pass
    for c, v in (("ShowText", "0"), ("ShowTextObject", "0"),
                 ("ShowCoords", "0"), ("ToolbarLayout", "Viewer")):
        RDK.Command(c, v)
    RDK.Command("TriggerAction", "actionClose_Windows")
    RDK.Render(True)
    time.sleep(1.2)

    for nombre, accion in (("01_isometrica", "actionIsometric"),
                           ("02_planta", "actionTop"),
                           ("03_frente", "actionFront"),
                           ("04_lateral", "actionRight")):
        RDK.Command("TriggerAction", accion)
        time.sleep(0.5)
        RDK.Command("TriggerAction", "actionFit_All")
        time.sleep(1.1)
        RDK.Command("Snapshot", str(qc / (nombre + ".png")))
    RDK.Command("TriggerAction", "actionIsometric")

    # La camara de la celda se queda en la estacion aunque su resolucion este
    # limitada: lo que importa de ella es el CUADRO, y ese es el mismo de las
    # fotos de entrenamiento (§66.5), no uno cualquiera.
    fov = 2 * np.degrees(np.arctan(325.0 / CAM_ALTURA))
    mk = RDK.AddFrame("Camara de la celda")
    mk.setPose(rm.transl(X_CAMARA, BANDA_Y, BANDA_Z + CAM_ALTURA) * rm.rotx(np.pi))
    try:
        cam = RDK.Cam2D_Add(mk, "FOV={:.1f} FAR_LENGTH=4000".format(fov))
        RDK.Cam2D_Snapshot(str(qc / "05_lo_que_ve_la_camara.png"), cam)
    except Exception as ex:
        print("   [!] camara 2D no disponible: {}".format(ex))
    alto = 2 * CAM_ALTURA * np.tan(np.radians(fov / 2))
    print("   camara de la celda: FOV {:.1f} grados a {:.0f} mm -> cuadro de "
          "~{:.0f} x {:.0f} mm".format(fov, CAM_ALTURA, alto * 0.75, alto))
    return qc


def main():
    print("1. geometria de la celda ...", flush=True)
    f, largo_util = construye_geometria()
    print("   {} piezas STL en {}".format(len(f), GEO))
    print("   herramienta: barra de ventosas, {:.0f} mm de largo util"
          .format(largo_util))
    print("\n   los once recipientes:")
    for nombre, var, cls, x, y, ang in posiciones():
        print("     {:<26} r={:6.0f}  ang={:5.1f}   ({:7.0f}, {:7.0f})"
              .format(nombre, np.hypot(x, y), ang, x, y))
    if "--sin-abrir" in sys.argv:
        print("\n(--sin-abrir: no se toca RoboDK)")
        return

    print("\n2. montando la estacion en RoboDK ...", flush=True)
    RDK, robot, util, puntos = monta(f, largo_util)
    print("   brazo: {}".format(robot.Name()))

    print("\n3. ¿LLEGA EL BRAZO? ({} puntos de trabajo)".format(len(puntos)),
          flush=True)
    malos = comprueba_alcance(RDK, robot, util, puntos)
    if malos:
        print("   [!] NO ALCANZABLES: {}".format(len(malos)))
        for m in malos:
            print("       - {}".format(m))
        print("   La celda NO es valida como esta: hay que acercar esos")
        print("   recipientes, subir el brazo sobre peana, o cambiar de brazo.")
    else:
        print("   los {} puntos son alcanzables.".format(len(puntos)))

    print("\n4. imagenes de control ...", flush=True)
    qc = fotos_qc(RDK)
    print("   -> {}".format(qc))

    destino = OUT / "celda" / "celda_tabaco.rdk"
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        RDK.Save(str(destino))
        print("\n-> {}".format(destino))
    except Exception as ex:
        print("\n[!] no se pudo guardar la estacion: {}".format(ex))


if __name__ == "__main__":
    main()
