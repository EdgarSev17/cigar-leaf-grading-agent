# -*- coding: utf-8 -*-
"""
Construye la celda de trabajo completa alrededor del robot KUKA IONTEC KR 120 R2700.

Genera:
  1. Linea transportadora de 4306 mm delante del robot, con su bancada.
  2. Programas para encenderla y apagarla (Banda_ON / Banda_OFF).
  3. Camara 2D simulada, a la altura justa para ver la hoja entera cuando se para bajo
     ella, sobre un poste en voladizo montado en el lado opuesto al robot.
  4. Deteccion de colisiones configurada.
  5. Once cajas abiertas en arco detras del robot: una por tipo de hoja (1..10) y
     la 11, en rojo, para los casos especiales.

La banda se modela con geometria propia y el transporte lo mueve un script (que es
como lo resuelve el propio RoboDK en Example-06.e y en Library/Macros/RunConveyor.py).
Dos ventajas frente a usar un mecanismo de banda de la libreria:
  - No depende de la libreria de RoboDK, que cambia entre versiones.
  - La estacion queda con UN SOLO mecanismo (el robot), asi que se puede guardar
    incluso con la licencia Free, que no admite guardar mas de un robot.

Uso: con la estacion abierta en RoboDK, ejecutar este script.
Es idempotente: se puede volver a ejecutar sin duplicar elementos.
"""
import os
from robodk.robolink import *
from robodk.robomath import *

# ----------------------------------------------------------------- parametros
NOMBRE_ROBOT = 'KUKA IONTEC ultra KR 120 R2700'
# Al lado de este fichero, no clavada a una maquina: esta carpeta se sincroniza
# por OneDrive y en el otro equipo el usuario tiene otro nombre.
CARPETA_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scripts")

# --- linea transportadora (avance en +Y)
BELT_CX = 1500.0                                  # eje longitudinal de la linea
BELT_TOP = 766.5                                  # superficie de transporte
BELT_Y1 = 1076.5                                  # final de la linea
LARGO_LINEA = 4306.0
BELT_Y0 = BELT_Y1 - LARGO_LINEA                   # -3229.5 -> inicio de la linea
Y_CEN = (BELT_Y0 + BELT_Y1) / 2.0
VEL_BANDA = 300.0                                 # mm/s

ANCHO_NEGRO = 606.0                               # ancho de la cinta negra
NEGRO_X0 = BELT_CX - ANCHO_NEGRO / 2.0            # 1197
NEGRO_X1 = BELT_CX + ANCHO_NEGRO / 2.0            # 1803

ESP_CINTA = 30.0                                  # espesor de la cinta
Z_CINTA = BELT_TOP - ESP_CINTA / 2.0              # 751.5
Z_BAST_BOT, Z_BAST_TOP = 656.5, 776.5             # bastidor (guias 10 mm sobre la cinta)
ANCHO_GUIA = 90.0
GUIA_X0 = NEGRO_X0 - ANCHO_GUIA / 2.0             # 1152
GUIA_X1 = NEGRO_X1 + ANCHO_GUIA / 2.0             # 1848

# --- camara
POST_X = 2170.0                                   # columna (lado opuesto al robot)
CAM_Y = -700.0                                    # aguas arriba de la recogida
# La camara se subio para que la hoja entera quepa en la imagen. El FOV y el sensor
# (640x480 apaisado) no cambian: solo la altura. El parametro FOV de RoboDK es el
# vertical, que aqui corresponde al sentido de avance de la banda.
CAM_FOV = 28.39                                   # grados
LARGO_HOJA_MAX = 553.2                            # hoja mas larga (habano banda)
ALTO_VISTA = 620.0                                # cobertura a lo largo de la banda (~33 mm de margen por extremo)
_media = CAM_FOV / 2.0 * pi / 180.0
DIST_CAM = (ALTO_VISTA / 2.0) * cos(_media) / sin(_media)   # 1225.5 mm hasta la cinta
CAM_Z = BELT_TOP + DIST_CAM                       # 1992 mm
ANCHO_VISTA = ALTO_VISTA * 640.0 / 480.0          # 827 mm: ademas de la cinta negra entran las guias

# --- cajas en arco
# Arco continuo y simetrico respecto a 180 (justo detras del robot), con paso de 22 grados.
# Cajas 1..8: una por tipo de hoja. Caja 9: casos especiales (hoja no reconocida).
# El 2026-09-15 se quitaron las dos cajas de media_banda: el modelo final es de
# CUATRO calidades (capa, banda, XL izq, XR der) por variedad. Renumerado a 8 + 1.
# Con cajas de 660 x 430 el arco puede llegar a 70..290 sin tocar la linea transportadora;
# el hueco minimo entre cajas contiguas (radio interior) es de 55 mm.
R_ARC = 1600.0
N_CAJAS = 9
CAJA_ESPECIAL = 9
ANG_INI, ANG_FIN = 70.0, 290.0
ANGULOS = [ANG_INI + (ANG_FIN - ANG_INI) * i / (N_CAJAS - 1.0) for i in range(N_CAJAS)]
# Caja dimensionada para la hoja mas grande (553,2 x 290,2 mm): interior 636 x 406,
# es decir 41 mm de holgura a lo largo y 58 mm a lo ancho.
# CAJA_SX es la dimension radial (a lo largo del arco la caja "mira" al robot).
CAJA_SX, CAJA_SY, CAJA_SZ, ESPESOR = 660.0, 430.0, 320.0, 12.0

# --- hojas a transportar (sustituyen a la pieza cubica)
# La ruta NO se clava: el archivo esta al lado de este script (LEEME, tres rutas).
HOJAS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'hojas_foto_impresa.rdk')
# Orden = caja de destino. Cada clase de hoja tiene su caja en el arco.
CLASES = ['connecticut banda', 'connecticut capa',
          'connecticut xl_izq', 'connecticut xr_der',
          'habano banda', 'habano capa',
          'habano xl_izq', 'habano xr_der']
GROSOR_HOJA = 0.3                                 # espesor de la hoja

TCP_L = 250.0                                     # longitud de la ventosa
Z_APROX = 350.0                                   # altura de aproximacion

# Postura compacta de reposo y transferencia: girar el eje 1 en ella no toca nada.
J_REPOSO = [0.0, -117.0, 97.5, 0.0, 109.5, 0.0]

RDK = Robolink()
RDK.TIMEOUT = 300
RDK.COM.settimeout(300)
RDK.Render(False)
station = RDK.ActiveStation()
robot = RDK.Item(NOMBRE_ROBOT, ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No se encuentra el robot: ' + NOMBRE_ROBOT)


# ----------------------------------------------------------------- geometria
def box_tris(cx, cy, cz, sx, sy, sz):
    """Triangulos de una caja solida centrada en (cx,cy,cz)."""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    v = [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
         (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
         (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
         (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]
    caras = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return [v[i] for f in caras for i in f]


def solido(cajas, nombre, color):
    """Crea un unico objeto a partir de una lista de cajas."""
    pts = []
    for c in cajas:
        pts.extend(box_tris(*c))
    obj = RDK.AddShape(Mat([[p[0] for p in pts],
                            [p[1] for p in pts],
                            [p[2] for p in pts]]))
    obj.setName(nombre)
    obj.setColor(color)
    return obj


def caja_abierta(sx, sy, sz, t):
    """Contenedor abierto por arriba, centrado en XY, con la base en z=0."""
    return [(0, 0, t / 2.0, sx, sy, t),
            (0, -(sy - t) / 2.0, sz / 2.0, sx, t, sz),
            (0, (sy - t) / 2.0, sz / 2.0, sx, t, sz),
            (-(sx - t) / 2.0, 0, sz / 2.0, t, sy - 2 * t, sz),
            ((sx - t) / 2.0, 0, sz / 2.0, t, sy - 2 * t, sz)]


# ----------------------------------------------------------------- 0) limpieza
RDK.Cam2D_Close()
for nombre in ['Banda', 'Banda Bastidor', 'Soporte Banda', 'Estructura Camara',
               'Camara Ref', 'Camara Banda', 'Pieza', 'Ref Pieza', 'Ventosa',
               'Marco Celda', 'Hojas Patron']:
    it = RDK.Item(nombre)
    if it.Valid():
        it.Delete()
for i in range(1, 31):        # margen amplio por si cambia el numero de cajas
    for nombre in ['Caja %i' % i, 'Pieza %i' % i]:
        it = RDK.Item(nombre)
        if it.Valid():
            it.Delete()
for it in RDK.ItemList(ITEM_TYPE_OBJECT):
    if it.Name().startswith('Patron ') or it.Name().startswith('Hoja '):
        it.Delete()
for t in RDK.ItemList(ITEM_TYPE_TARGET):
    t.Delete()
for p in RDK.ItemList(ITEM_TYPE_PROGRAM):       # programas de robot (se regeneran)
    p.Delete()

# ----------------------------------------------------------------- 1) linea
# La cinta negra es un objeto aparte: es lo que ve la camara y donde apoyan las piezas.
banda = solido([(BELT_CX, Y_CEN, Z_CINTA, ANCHO_NEGRO, LARGO_LINEA, ESP_CINTA)],
               'Banda', [0.13, 0.13, 0.14, 1])

alto_bast = Z_BAST_TOP - Z_BAST_BOT
bastidor = solido([
    (GUIA_X0, Y_CEN, (Z_BAST_BOT + Z_BAST_TOP) / 2.0, ANCHO_GUIA, LARGO_LINEA, alto_bast),
    (GUIA_X1, Y_CEN, (Z_BAST_BOT + Z_BAST_TOP) / 2.0, ANCHO_GUIA, LARGO_LINEA, alto_bast),
    (BELT_CX, Y_CEN, (Z_BAST_BOT + BELT_TOP - ESP_CINTA) / 2.0,
     ANCHO_NEGRO, LARGO_LINEA, BELT_TOP - ESP_CINTA - Z_BAST_BOT),
], 'Banda Bastidor', [0.05, 0.60, 0.90, 1])

# ----------------------------------------------------------------- 2) bancada
Z_VIGA_TOP = Z_BAST_BOT
Z_VIGA_BOT = Z_VIGA_TOP - 80.0
N_PORTICOS = 9
paso = (LARGO_LINEA - 360.0) / (N_PORTICOS - 1)
ys = [BELT_Y0 + 180.0 + i * paso for i in range(N_PORTICOS)]
patas = [(x, y, Z_VIGA_BOT / 2.0, 100.0, 100.0, Z_VIGA_BOT)
         for y in ys for x in (GUIA_X0, GUIA_X1)]
vigas = [(x, Y_CEN, (Z_VIGA_BOT + Z_VIGA_TOP) / 2.0, 120.0, LARGO_LINEA - 100.0, 80.0)
         for x in (GUIA_X0, GUIA_X1)]
sop = solido(patas + vigas, 'Soporte Banda', [0.35, 0.37, 0.40, 1])

# ----------------------------------------------------------------- 3) camara
# Voladizo anclado en el lado opuesto al robot: nada se interpone entre robot y banda.
# Cotas referidas a la altura de la camara, para que la estructura suba con ella.
Z_SOPORTE = CAM_Z + 60.0                          # base del soporte de la camara
Z_BRAZO = Z_SOPORTE + 70.0                        # base del brazo en voladizo
Z_TOPE = Z_BRAZO + 120.0                          # parte alta del brazo y de la columna
est = solido([
    (POST_X, CAM_Y, 15.0, 380.0, 380.0, 30.0),                              # placa de anclaje
    (POST_X, CAM_Y, (30.0 + Z_TOPE) / 2.0, 140.0, 140.0, Z_TOPE - 30.0),    # columna
    (1795.0, CAM_Y, Z_BRAZO + 60.0, 750.0, 120.0, 120.0),                   # brazo en voladizo
    (BELT_CX, CAM_Y, Z_SOPORTE + 35.0, 160.0, 160.0, 70.0),                 # soporte de la camara
], 'Estructura Camara', [0.85, 0.55, 0.10, 1])

camref = RDK.AddFrame('Camara Ref', station)
camref.setParentStatic(est)
camref.setPoseAbs(transl(BELT_CX, CAM_Y, CAM_Z) * rotx(pi))   # +Z de la camara hacia abajo
camara = RDK.Cam2D_Add(camref, 'FOCAL_LENGTH=6 FOV=%.2f FAR_LENGTH=2500 SIZE=640x480' % CAM_FOV)
camara.setName('Camara Banda')

# ----------------------------------------------------------------- 4) cajas en arco
# Un color distinto por caja para identificarlas a simple vista. La 9 (especial) en rojo.
# Cada CLASE conserva el color que tenia con once cajas: solo cambia el numero.
COLORES_CAJAS = {
    1: ('amarillo', [0.95, 0.85, 0.10, 1]),       # connecticut banda
    2: ('naranja', [1.00, 0.50, 0.00, 1]),        # connecticut capa
    3: ('verde oscuro', [0.05, 0.50, 0.20, 1]),   # connecticut xl_izq
    4: ('turquesa', [0.05, 0.75, 0.75, 1]),       # connecticut xr_der
    5: ('azul', [0.15, 0.30, 0.85, 1]),           # habano banda
    6: ('morado', [0.50, 0.20, 0.75, 1]),         # habano capa
    7: ('marron', [0.50, 0.30, 0.10, 1]),         # habano xl_izq
    8: ('blanco', [0.92, 0.92, 0.92, 1]),         # habano xr_der
    9: ('rojo', [0.80, 0.10, 0.10, 1]),           # revision / no reconocida
}
for i, a in enumerate(ANGULOS, start=1):
    x, y = R_ARC * cos(a * pi / 180.0), R_ARC * sin(a * pi / 180.0)
    color = COLORES_CAJAS.get(i, ('tierra', [0.75, 0.60, 0.35, 1]))[1]
    obj = solido(caja_abierta(CAJA_SX, CAJA_SY, CAJA_SZ, ESPESOR), 'Caja %i' % i, color)
    obj.setPose(transl(x, y, 0) * rotz(a * pi / 180.0))

# ----------------------------------------------------------------- 5) hojas patron
# Se importan las hojas del archivo del usuario y se dejan ocultas fuera de la celda:
# son solo plantillas de las que el alimentador saca copias.
marco = RDK.AddFrame('Marco Celda', station)
marco.setPoseAbs(eye(4))
patrones = RDK.AddFrame('Hojas Patron', station)
patrones.setPoseAbs(transl(0, 0, -3000.0))        # aparcadas bajo el suelo, ocultas

st_hojas = RDK.AddFile(HOJAS_SRC)
if not st_hojas.Valid():
    raise Exception('No se pudo abrir: ' + HOJAS_SRC)
RDK.setActiveStation(st_hojas)
origen = {}
for it in RDK.ItemList(ITEM_TYPE_OBJECT):
    for clase in CLASES:
        if it.Name().startswith(clase + ' '):
            origen[clase] = it
if len(origen) != len(CLASES):
    raise Exception('Faltan hojas en el archivo: %s' % [c for c in CLASES if c not in origen])

for clase in CLASES:
    RDK.setActiveStation(st_hojas)
    RDK.Copy(origen[clase])
    RDK.setActiveStation(station)
    hoja = RDK.Paste(patrones)
    hoja.setName('Patron ' + clase)
    hoja.setPose(eye(4))
    hoja.setVisible(False)
RDK.setActiveStation(station)
st_hojas.Delete()                                 # se cierra sin modificar el archivo

# ----------------------------------------------------------------- 6) herramienta
tool = robot.AddTool(transl(0, 0, TCP_L), 'Ventosa')
pts = []
for c in [(0, 0, 20.0, 180.0, 180.0, 40.0),      # brida
          (0, 0, 120.0, 120.0, 120.0, 160.0),    # cuerpo
          (0, 0, 225.0, 200.0, 200.0, 50.0)]:    # copa de succion
    pts.extend(box_tris(*c))
RDK.AddShape(Mat([[p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]]), tool)
tool.setColor([0.15, 0.15, 0.18, 1])
robot.setTool(tool)
robot.setPoseFrame(marco)
robot.setPoseTool(tool)


# ----------------------------------------------------------------- 7) targets
def transf(j1):
    return [j1] + J_REPOSO[1:]


def envolver(a):
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def target_cart(nombre, pose, semilla):
    """IK con semilla explicita. Hay que fijar la postura ANTES de crear el
    target, porque AddTarget hereda la configuracion actual del robot."""
    j = robot.SolveIK(pose, semilla, tool.PoseTool(), eye(4))
    if len(j.list()) < 6:
        raise Exception('Punto fuera de alcance: ' + nombre)
    robot.setJoints(j)
    t = RDK.AddTarget(nombre, marco, robot)
    t.setPose(pose)
    return t


robot.setJoints(J_REPOSO)
t_ini = RDK.AddTarget('Inicio', marco, robot)
t_ini.setJoints(J_REPOSO)
t_ini.setAsJointTarget()

target_cart('Recogida Aprox',
            transl(BELT_CX, 0.0, BELT_TOP + GROSOR_HOJA + Z_APROX) * rotx(pi), transf(0.0))
target_cart('Recogida',
            transl(BELT_CX, 0.0, BELT_TOP + GROSOR_HOJA) * rotx(pi), transf(0.0))
Z_DEPOSITO = ESPESOR + GROSOR_HOJA + 3.0          # 15.3: la hoja queda 3 mm sobre el fondo
for i, a in enumerate(ANGULOS, start=1):
    x, y = R_ARC * cos(a * pi / 180.0), R_ARC * sin(a * pi / 180.0)
    semilla = transf(envolver(-a))        # el eje 1 del KUKA gira al reves del azimut
    # La hoja se recoge con su lado largo segun +Y. Al girar la herramienta (a - 90) la
    # hoja entra alineada con el lado largo de la caja, que apunta radialmente.
    ori = rotz((a - 90.0) * pi / 180.0) * rotx(pi)
    target_cart('Caja %i Aprox' % i, transl(x, y, 700.0) * ori, semilla)
    target_cart('Caja %i Deposito' % i, transl(x, y, Z_DEPOSITO) * ori, semilla)
robot.setJoints(J_REPOSO)

# ----------------------------------------------------------------- 8) colisiones
RDK.Command('CollisionMap', 'Default')
RDK.setCollisionActive(COLLISION_ON)
# Contactos previstos por diseno: no son colisiones reales.
RDK.setCollisionActivePair(COLLISION_OFF, robot, tool, 6, 0)   # la ventosa va en la brida
RDK.setCollisionActivePair(COLLISION_OFF, banda, bastidor)     # la cinta apoya en el bastidor
RDK.setCollisionActivePair(COLLISION_OFF, bastidor, sop)       # el bastidor apoya en la bancada

# ----------------------------------------------------------------- 9) programas Python
PROGRAMAS_PY = ['Control_Senales', 'Ventosa_Agarrar', 'Ventosa_Soltar',
                'Banda_ON', 'Banda_OFF', 'Alimentar_Pieza', 'Ciclo_Pick_Place', 'Reset_Celda']
for nombre in PROGRAMAS_PY:
    it = RDK.Item(nombre, ITEM_TYPE_PROGRAM_PYTHON)
    if it.Valid():
        it.Delete()
    ruta = os.path.join(CARPETA_SCRIPTS, nombre + '.py')
    if os.path.isfile(ruta):
        RDK.AddFile(ruta)

# ----------------------------------------------------------------- 10) programas de robot
# Rutinas nativas de RoboDK (se pueden generar como programa KUKA con el postprocesador).
# La ventosa se acciona llamando a las macros Ventosa_Agarrar / Ventosa_Soltar: en
# simulacion RoboDK ejecuta el script y espera a que termine antes del siguiente movimiento;
# en el robot real esas llamadas se sustituyen por la activacion de la salida de vacio.
T = lambda nombre: RDK.Item(nombre, ITEM_TYPE_TARGET)


def programa_robot(nombre):
    p = RDK.AddProgram(nombre, robot)
    p.setPoseFrame(marco)
    p.setPoseTool(tool)
    p.setSpeed(900, 250)                  # mm/s lineal, grados/s articular
    return p


p = programa_robot('Recoger_Hoja')
p.MoveJ(T('Inicio'))
p.MoveJ(T('Recogida Aprox'))
p.MoveL(T('Recogida'))
p.RunInstruction('Ventosa_Agarrar', INSTRUCTION_CALL_PROGRAM)
p.MoveL(T('Recogida Aprox'))
p.MoveJ(T('Inicio'))

for i in range(1, N_CAJAS + 1):
    p = programa_robot('Dejar_Caja_%02d' % i)
    p.MoveJ(T('Inicio'))
    p.MoveJ(T('Caja %i Aprox' % i))
    p.MoveL(T('Caja %i Deposito' % i))
    p.RunInstruction('Ventosa_Soltar', INSTRUCTION_CALL_PROGRAM)
    p.MoveL(T('Caja %i Aprox' % i))
    p.MoveJ(T('Inicio'))

# ----------------------------------------------------------------- 11) senales
# Entradas (las escribe el agente): TIPO_B0..TIPO_B3 (B0 = bit menos significativo), TIPO_VALIDO
# Salidas (las lee el agente):       HOJA_EN_CAMARA, ORDEN_RECIBIDA, ROBOT_OCUPADO,
#                                    CAJA_DESTINO, ULTIMO_CODIGO
for nombre in ['TIPO_B0', 'TIPO_B1', 'TIPO_B2', 'TIPO_B3', 'TIPO_VALIDO', 'HOJA_EN_CAMARA',
               'ORDEN_RECIBIDA', 'ROBOT_OCUPADO', 'CAJA_DESTINO', 'ULTIMO_CODIGO', 'VENTOSA']:
    RDK.setParam(nombre, '0')
RDK.setParam('CONTROL', 'RUN')
RDK.setParam('BandaEstado', 'OFF')
RDK.Render(True)
RDK.Command('FitIsometric')

mecanismos = RDK.ItemList(ITEM_TYPE_ROBOT)
print('Celda construida. Colisiones en reposo:', RDK.Collisions())
print('Mecanismos en la estacion: %i -> %s' % (len(mecanismos), [m.Name() for m in mecanismos]))
print('Linea de transporte: %.0f mm (Y %.1f -> %.1f), cinta negra de %.0f mm de ancho'
      % (LARGO_LINEA, BELT_Y0, BELT_Y1, ANCHO_NEGRO))
print('Recorrido antes de la camara: %.0f mm' % (CAM_Y - BELT_Y0))
print('Recorrido total hasta la recogida: %.0f mm' % (0.0 - BELT_Y0))
print('Camara: FOV %.2f deg -> cubre %.0f x %.0f mm sobre la cinta' % (CAM_FOV, ANCHO_VISTA, ALTO_VISTA))
print('Cajas: %i en arco de R=%.0f, de %.0f a %.0f deg (paso %.2f deg), interior %.0f x %.0f mm'
      % (N_CAJAS, R_ARC, ANG_INI, ANG_FIN, (ANG_FIN - ANG_INI) / (N_CAJAS - 1.0),
         CAJA_SX - 2 * ESPESOR, CAJA_SY - 2 * ESPESOR))
print('Hojas patron importadas: %i' % len([o for o in RDK.ItemList(ITEM_TYPE_OBJECT)
                                           if o.Name().startswith('Patron ')]))
for i, clase in enumerate(CLASES, start=1):
    print('   Caja %-2i <- %s' % (i, clase))
print('   Caja %-2i <- casos especiales (hoja no reconocida)' % CAJA_ESPECIAL)
print('Programas de robot: %s' % [p.Name() for p in RDK.ItemList(ITEM_TYPE_PROGRAM)])
print('Programas Python:   %s' % [p.Name() for p in RDK.ItemList(ITEM_TYPE_PROGRAM_PYTHON)])
