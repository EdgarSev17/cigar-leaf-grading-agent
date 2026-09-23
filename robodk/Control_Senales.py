# Programa principal de la celda controlada por senales (modelo agentico).
#
# Ciclo por hoja:
#   1. Coloca al azar una hoja al inicio de la linea y la lleva hasta la camara.
#   2. Para la banda bajo la camara (minimo 3 s), guarda la imagen y pone
#      HOJA_EN_CAMARA = 1: el agente ya puede clasificarla.
#   3. Espera la orden del agente: codigo de 4 bits (TIPO_B0..TIPO_B3) + TIPO_VALIDO.
#   4. Acusa recibo (ORDEN_RECIBIDA = 1), lleva la hoja a la zona de recogida y
#      ejecuta los programas de robot Recoger_Hoja y Dejar_Caja_NN.
#   5. Anota el resultado en registro_clasificacion.csv.
#
# Codigos:  1..8 -> caja del tipo de hoja   |   9 -> caja 9 (no reconocida)
#           0 y 10..15 (no validos) -> caja 9, para no perder la hoja.
#
# Para detenerlo de forma ordenada: poner el parametro CONTROL = STOP. Solo se atiende en
# puntos seguros (esperando orden o entre ciclos): si llega con el robot en movimiento,
# se termina la hoja en curso, se anota y despues se para.
import os
import random
import time
from robodk.robolink import *
from robodk.robomath import *

RDK = Robolink()

robot = RDK.Item('KUKA IONTEC ultra KR 120 R2700', ITEM_TYPE_ROBOT)
tool = RDK.Item('Ventosa', ITEM_TYPE_TOOL)
marco = RDK.Item('Marco Celda', ITEM_TYPE_FRAME)
banda = RDK.Item('Banda', ITEM_TYPE_OBJECT)
camref = RDK.Item('Camara Ref', ITEM_TYPE_FRAME)
camara = RDK.Item('Camara Banda', ITEM_TYPE_CAMERA)

# --- geometria de la linea (igual que en Construir_Entorno.py)
BELT_CX = 1500.0
BELT_TOP = 766.5
BELT_Y0 = -3229.5            # inicio de la linea
CAM_Y = -700.0               # camara
Y_RECOGIDA = 0.0             # zona de recogida
VEL_BANDA = 300.0            # mm/s
REFRESCO = 0.02
PARADA_MIN_CAMARA = 3.0      # s minimos bajo la camara (tiempo de simulacion)
CAM_PARAMS = 'FOCAL_LENGTH=6 FOV=28.39 FAR_LENGTH=2500 SIZE=640x480'

# --- tipos de hoja: tipo N -> caja N. Mismo orden que CLASES en Construir_Entorno.py
CLASES = ['connecticut banda', 'connecticut capa',
          'connecticut xl_izq', 'connecticut xr_der',
          'habano banda', 'habano capa',
          'habano xl_izq', 'habano xr_der']
CAJA_ESPECIAL = 9
N_HOJAS = 0                  # 0 = sin fin (o hasta agotar la cola, si la hay)

# La carpeta se averigua, no se clava: la ruta escrita era la de otra maquina
# (C:\Users\LENOVO\Desktop\RoboDK) y aqui el usuario es otro, asi que la imagen de
# la camara y el registro se escribian en un sitio que no existe -- y
# `guardar_imagen` se lo tragaba en silencio por su try/except.
def _carpeta():
    try:
        est = str(RDK.getParam('PATH_OPENSTATION'))
        if os.path.isdir(est):
            return est
    except Exception:
        pass
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        return os.path.expanduser('~')


CARPETA = _carpeta()
IMAGEN = os.path.join(CARPETA, 'camara_hoja_actual.png')
REGISTRO = os.path.join(CARPETA, 'registro_clasificacion.csv')

# La cola de hojas que prepara el agente (`scripts/cola_entorno1.py`). Si existe,
# las hojas salen EN ESE ORDEN y con su malla propia; si no, se alimenta al azar
# con las diez hojas patron, que es como estaba.
# 2026-09-22: la cola se busca desde la raiz del repositorio (la carpeta que
# contiene `code/`, `out/` y `robodk/`), de modo que esto corre en cualquier
# maquina sin editar nada.
REPO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLA = os.path.join(REPO_RAIZ, 'out', 'celda', 'entorno1', 'cola_hojas.csv')

# Se puede pasar otra cola como argumento:
#     python robodk/Control_Senales.py out/celda/entorno1/cola_demo_20.csv
import sys as _sys
if len(_sys.argv) > 1:
    COLA = _sys.argv[1]
    if not os.path.isabs(COLA):
        COLA = os.path.join(REPO_RAIZ, COLA)
    print('cola: ' + COLA)

ENTRADAS = ['TIPO_B0', 'TIPO_B1', 'TIPO_B2', 'TIPO_B3', 'TIPO_VALIDO']
SALIDAS = ['HOJA_EN_CAMARA', 'ORDEN_RECIBIDA', 'ROBOT_OCUPADO', 'CAJA_DESTINO', 'ULTIMO_CODIGO']


class Detenido(Exception):
    pass


def leer_cola():
    '''Las hojas a pasar, en orden. Lista vacia = alimentar al azar, como antes.'''
    if not os.path.isfile(COLA):
        return []
    import csv
    with open(COLA, encoding='utf-8') as f:
        return [r for r in csv.DictReader(f, delimiter=';') if r.get('obj')]


COLA_FILAS = leer_cola()
COLA_I = [0]                 # en lista para poder tocarlo desde alimentar()


# ----------------------------------------------------------------- senales
def leer(nombre):
    try:
        return int(float(str(RDK.getParam(nombre)).strip()))
    except (TypeError, ValueError):
        return 0


def escribir(nombre, valor):
    RDK.setParam(nombre, str(int(valor)))


def parada_pedida():
    return str(RDK.getParam('CONTROL')).strip().upper() == 'STOP'


def comprobar_parada():
    """Solo se llama en puntos seguros (sin hoja en la ventosa ni robot en movimiento)."""
    if parada_pedida():
        raise Detenido()


def leer_codigo():
    """Compone el codigo de 4 bits: B0 es el bit menos significativo."""
    return sum((leer('TIPO_B%i' % k) & 1) << k for k in range(4))


def codigo_a_caja(codigo):
    if 1 <= codigo <= len(CLASES):
        return codigo
    return CAJA_ESPECIAL          # 9 = no reconocida; 0 y 10..15 = no validos


def cerrar_handshake():
    """Fase 4: cuando el agente baja TIPO_VALIDO, el robot baja ORDEN_RECIBIDA."""
    if leer('ORDEN_RECIBIDA') == 1 and leer('TIPO_VALIDO') == 0:
        escribir('ORDEN_RECIBIDA', 0)


# ----------------------------------------------------------------- tiempo
def esperar_sim(segundos):
    """Espera en tiempo de simulacion (escala con la velocidad de simulacion)."""
    tic()
    t_ant, t = toc(), 0.0
    while t < segundos:
        comprobar_parada()
        t_act = toc()
        t += (t_act - t_ant) * RDK.SimulationSpeed()
        t_ant = t_act
        pause(REFRESCO)


# ----------------------------------------------------------------- hojas y banda
def centro_y(obj):
    """El origen de la hoja esta en su lado corto: el centro sale del bounding box."""
    bb = obj.setParam('BoundingBox')
    return (bb['min'][1] + bb['max'][1]) / 2.0


def siguiente_numero():
    usados = []
    for o in RDK.ItemList(ITEM_TYPE_OBJECT):
        if o.Name().startswith('Hoja '):
            try:
                usados.append(int(o.Name().split()[1]))
            except ValueError:
                pass
    return max(usados) + 1 if usados else 1


def coloca(hoja, clase, ident):
    '''Nombra la hoja y publica su identificador para el agente.

    `HOJA_ID` es el nombre del fichero de su FOTO, que no dice la clase: es lo
    que el agente usa para encontrar la fotografia real. El nombre del objeto si
    lleva la etiqueta, y el agente NO lo lee -- leerla es lo que hacia que la
    celda "clasificara" sin clasificar.
    '''
    hoja.setName('Hoja %i - %s' % (siguiente_numero(), clase))
    RDK.setParam('HOJA_ID', ident)


def alimentar():
    '''Saca la siguiente hoja de la cola; si no hay cola, una patron al azar.'''
    if COLA_FILAS and COLA_I[0] < len(COLA_FILAS):
        r = COLA_FILAS[COLA_I[0]]
        COLA_I[0] += 1
        clase = '%s %s' % (r['variedad'], r['clase'])
        tipo_real = CLASES.index(clase) + 1 if clase in CLASES else 0
        # RoboDK resuelve las rutas relativas contra SU directorio, no contra
        # el de este guion, asi que la malla se le pasa siempre en absoluto.
        ruta_obj = r['obj']
        if not os.path.isabs(ruta_obj):
            ruta_obj = os.path.join(REPO_RAIZ, ruta_obj)
        hoja = RDK.AddFile(ruta_obj, marco)
        if not hoja.Valid():
            raise Exception('RoboDK no pudo cargar la malla: ' + ruta_obj)
        coloca(hoja, clase, r['id'])
        hoja.setPoseAbs(transl(BELT_CX, BELT_Y0, BELT_TOP))
        hoja.setVisible(True)
        RDK.setCollisionActivePair(COLLISION_OFF, tool, hoja)
        RDK.setCollisionActivePair(COLLISION_OFF, banda, hoja)
        for c in RDK.ItemList(ITEM_TYPE_OBJECT):
            if c.Name().startswith('Caja '):
                RDK.setCollisionActivePair(COLLISION_OFF, c, hoja)
        return hoja, clase, tipo_real

    tipo_real = random.randint(1, len(CLASES))
    clase = CLASES[tipo_real - 1]
    patron = RDK.Item('Patron ' + clase, ITEM_TYPE_OBJECT)
    if not patron.Valid():
        raise Exception('Falta la hoja patron: ' + clase)
    RDK.Copy(patron)
    hoja = RDK.Paste(marco)
    coloca(hoja, clase, '')
    hoja.setPoseAbs(transl(BELT_CX, BELT_Y0, BELT_TOP))
    hoja.setVisible(True)
    # Contactos previstos por diseno: no deben marcarse como colision.
    RDK.setCollisionActivePair(COLLISION_OFF, tool, hoja)
    RDK.setCollisionActivePair(COLLISION_OFF, banda, hoja)
    for c in RDK.ItemList(ITEM_TYPE_OBJECT):
        if c.Name().startswith('Caja '):
            RDK.setCollisionActivePair(COLLISION_OFF, c, hoja)
    return hoja, clase, tipo_real


def transportar(hoja, y_destino):
    """Banda en marcha hasta que el CENTRO de la hoja llega a y_destino.
    No se interrumpe: una parada pedida se atiende al llegar."""
    RDK.setParam('BandaEstado', 'ON')
    tic()
    t_ant = toc()
    while True:
        cerrar_handshake()
        t_act = toc()
        dt = (t_act - t_ant) * RDK.SimulationSpeed()
        t_ant = t_act
        x, y, z = hoja.PoseAbs().Pos()
        falta = y_destino - centro_y(hoja)
        if falta <= VEL_BANDA * dt:
            hoja.setPoseAbs(transl(x, y + falta, z))
            break
        hoja.setPoseAbs(transl(x, y + VEL_BANDA * dt, z))
        pause(REFRESCO)
    RDK.setParam('BandaEstado', 'OFF')


def guardar_imagen(cam):
    """Deja la imagen de la camara en disco para quien la necesite."""
    try:
        if RDK.Cam2D_Snapshot(IMAGEN, cam):
            RDK.setParam('IMAGEN_HOJA', IMAGEN)
    except Exception:
        pass


# ----------------------------------------------------------------- robot
def ejecutar(nombre):
    """Lanza un programa de robot y espera a que termine. No se interrumpe."""
    prog = RDK.Item(nombre, ITEM_TYPE_PROGRAM)
    if not prog.Valid():
        raise Exception('No existe el programa de robot: ' + nombre)
    prog.RunProgram()
    t0 = time.time()
    while not prog.Busy() and time.time() - t0 < 2.0:
        pause(0.02)
    while prog.Busy():
        cerrar_handshake()
        pause(0.05)


def esperar_orden():
    """Pide clasificacion y espera el codigo de 4 bits del agente."""
    # El handshake anterior tiene que estar cerrado antes de pedir otra orden.
    while leer('TIPO_VALIDO') == 1:
        comprobar_parada()
        pause(0.05)
    escribir('ORDEN_RECIBIDA', 0)

    escribir('HOJA_EN_CAMARA', 1)
    RDK.Command('SilentMessage', 'Hoja bajo la camara: esperando el tipo (4 bits)')
    esperar_sim(PARADA_MIN_CAMARA)          # inspeccion minima de 3 s
    while leer('TIPO_VALIDO') != 1:
        comprobar_parada()
        pause(0.05)
    codigo = leer_codigo()
    escribir('ORDEN_RECIBIDA', 1)
    escribir('HOJA_EN_CAMARA', 0)
    return codigo


def registrar(hoja, clase, tipo_real, codigo, caja):
    nuevo = not os.path.isfile(REGISTRO)
    if tipo_real == 0:
        # Hoja sin etiqueta: no hay verdad de campo contra la que puntuar, y
        # decir ACIERTO o ERROR seria inventarsela.
        resultado = 'SIN_ETIQUETA'
    elif codigo == tipo_real:
        resultado = 'ACIERTO'
    elif caja == CAJA_ESPECIAL:
        resultado = 'ESPECIAL'
    else:
        resultado = 'ERROR'
    with open(REGISTRO, 'a', encoding='utf-8') as f:
        if nuevo:
            f.write('fecha;hoja;clase_real;tipo_real;codigo_recibido;bits_B3_B0;caja;resultado\n')
        f.write('%s;%s;%s;%i;%i;%s;%i;%s\n' % (
            time.strftime('%Y-%m-%d %H:%M:%S'), hoja.Name(), clase, tipo_real,
            codigo, format(codigo, '04b'), caja, resultado))
    return resultado


# ----------------------------------------------------------------- programa principal
for n in ENTRADAS + SALIDAS:
    escribir(n, 0)
RDK.setParam('CONTROL', 'RUN')

robot.setPoseFrame(marco)
robot.setPoseTool(tool)
robot.MoveJ(RDK.Item('Inicio', ITEM_TYPE_TARGET))

# La imagen solo sale bien si la vista se abre desde este mismo proceso.
cam = camara
try:
    RDK.Cam2D_Close()
    cam = RDK.Cam2D_Add(camref, CAM_PARAMS, camara)
    cam.setName('Camara Banda')
except Exception:
    pass

n = 0
try:
    cola_agotada = lambda: bool(COLA_FILAS) and COLA_I[0] >= len(COLA_FILAS)
    while (N_HOJAS == 0 or n < N_HOJAS) and not cola_agotada():
        comprobar_parada()                      # punto seguro: entre hojas
        n += 1
        hoja, clase, tipo_real = alimentar()
        transportar(hoja, CAM_Y)
        guardar_imagen(cam)

        codigo = esperar_orden()                # punto seguro: esperando al agente
        caja = codigo_a_caja(codigo)
        escribir('ULTIMO_CODIGO', codigo)
        escribir('CAJA_DESTINO', caja)
        escribir('ROBOT_OCUPADO', 1)
        RDK.Command('SilentMessage', 'Orden %s (codigo %i) -> Caja %i' % (format(codigo, '04b'), codigo, caja))

        # Desde aqui la hoja se termina siempre, aunque se pida parar.
        transportar(hoja, Y_RECOGIDA)
        ejecutar('Recoger_Hoja')
        ejecutar('Dejar_Caja_%02d' % caja)

        escribir('ROBOT_OCUPADO', 0)
        resultado = registrar(hoja, clase, tipo_real, codigo, caja)
        RDK.Command('SilentMessage', '%s: %s -> Caja %i (%s)' % (hoja.Name(), clase, caja, resultado))
except Detenido:
    RDK.Command('SilentMessage', 'Control_Senales detenido')
finally:
    RDK.setParam('BandaEstado', 'OFF')
    escribir('HOJA_EN_CAMARA', 0)
    escribir('ROBOT_OCUPADO', 0)
