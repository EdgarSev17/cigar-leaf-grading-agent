r"""
UNA FOTO ENTRA, SALE LA CLASE CON SU MOTIVO. (2026-09-05)

Es lo que pedia PROGRESO 54.4, con las palabras del experto del 2026-09-04:

    "la idea es probarlo con hojas nuevas, poner una camara y que el detector
     sea capaz de detectar que clase de hoja es solo con verla"

    clasifica.py foto.jpg  ->  clase + defectos + confianza + motivo

POR QUE EL MOTIVO NO ES ADORNO
------------------------------
La tesis es un **agente clasificador**, no un porcentaje. Un agente que dice "XL
izquierdo" sin decir por que no sirve en planta: nadie puede comprobarlo, nadie
puede corregirlo y, cuando se equivoque, nadie sabra si fue la hoja o la camara.

Aqui el motivo se da en DOS partes, y estan separadas a proposito porque no son
lo mismo:

  LO QUE SE VE EN LA HOJA   las medidas: donde esta el dano, cuanto se mete,
                            que manchas hay y de que lado. Esto lo puede
                            comprobar un tecnico mirando la hoja.

  LO QUE PESO EN LA DECISION  la aritmetica del modelo: que rasgos empujaron
                            hacia la clase elegida y en contra de la segunda.
                            Es lo que el modelo hizo DE VERDAD, no una historia
                            escrita despues.

Mezclar las dos seria contar un cuento: el modelo no "ve" un roto, ve 64
numeros. Se enseñan las dos y se dice cual es cual.

LA CONFIANZA SE USA, NO SE ENSEÑA
---------------------------------
`entrena.py` midio, fuera de muestra, cuanto acierta el modelo en cada tramo de
confianza. `clasifica.py` **no imprime el numero del modelo a secas**: lo traduce
al acierto medido en ese tramo y decide con el. Si el tramo no llega al umbral,
la hoja va a REVISION MANUAL en vez de a la cinta.

Eso es exactamente lo que la tesis promete --un agente que decide bajo
confianza-- y es la razon por la que `entrena.py` elige la familia por
calibracion cuando la exactitud empata: con el modelo de arboles, un 0.90 valia
un 61 % de acierto real, y esta regla de enrutamiento no habria funcionado.

LO QUE ESTE GUION NO PUEDE PROMETER
-----------------------------------
Que funcione con una foto de otro dia, otra mesa u otra camara. **No esta
medido y con los datos de hoy no se puede medir** (PROGRESO 54.2: la clase y la
hora del dia van juntas). Lo decide la sesion de fotos intercalada de 54.3. El
guion lo dice en cada corrida, y no es un formalismo: es la unica linea del
informe que el tribunal va a preguntar.

Uso:
    python scripts/clasifica.py foto.jpg
    python scripts/clasifica.py foto.jpg --json         (para el agente/RoboDK)
    python scripts/clasifica.py foto.jpg --umbral 0.95  (mas exigente)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))
try:                                                               # noqa: E402
    from rasgos_foto import mide
except ImportError:          # reproducir desde rasgos medidos no usa OpenCV
    mide = None
from rutas import REPO_RAIZ  # raiz del repositorio

OUT = REPO_RAIZ / "out"
DIR_MODELO = OUT / "modelo"
DIR_ARBOL = OUT / "modelo_arbol"

# ===== LA CALIDAD LA DECIDE EL ARBOL (2026-09-19, noche. El experto: "si, metelo") =
# El modelo plano trata los 69 rasgos por igual y eso lo hunde FUERA de sesion:
# cada rasgo trae la firma de la jornada en que se tomo la foto. El arbol usa en
# cada nodo SOLO los rasgos que nombra la regla de oficio del experto --banda por el
# sudado y el color, capa por la forma del danio, lado por las asimetrias-- y es
# UN SOLO modelo para las dos variedades, entrenado con las cuatro jornadas.
#
# Medido con la jornada del examen dejada fuera del entrenamiento, con el detector
# de agujeros ya arreglado (GROSOR_MIN = 3, que no cuenta las arrugas):
#
#                        modelo plano      ARBOL
#     Habano 19/09           31.3 %       57.0 %     (mayoritaria 38.7)
#     Connecticut 112        82.1 %       84.8 %     (mayoritaria 63.4)
#
# Y por clase en el Habano, que es donde se veia el desastre: el XL izquierdo
# pasa de 1 de 59 (2 %) a 26 de 57 (46 %), sin que ninguna otra se hunda
# (capa 65 %, banda 49 %, XR der 66 %).
#
# LO QUE EL ARBOL NO CUBRE, y hay que decirlo: la clase "media_banda" no existe
# en el --no estaba en las carpetas que ordeno el experto ni en el entrenamiento del
# arbol--, asi que esas hojas caeran en una de las cuatro. El modelo plano la
# decia en 7 de las 112 y en 10 de la tanda de Habano, y en los dos examenes
# fallaba TODAS. Para recuperarla hace falta material etiquetado de media banda.
ARBOL = True

# El piso propio del arbol, medido sobre las 326 decisiones de los dos examenes
# fuera de sesion:  0.70 firma el 49.4 % y acierta el 74.5 %;  0.90 firma el
# 13.5 % y acierta el 79.5 %;  sin piso, 66.6 %.
#
# 2026-09-22: BAJA DE 0.70 A 0.60, decision del experto, para que el programa diga
# lo mismo que el paper. Sobre el lote independiente de Connecticut el 0.60
# decide el 70.5 % y acierta el 91.1 %, contra 46.4 % y 90.4 % del 0.70: mas
# cobertura y mas acierto. Sus palabras: "no podemos defender el elegir el 0.70,
# es imposible defender ese numero". Respaldo: clasifica_ANTES_piso070.py
PISO_ARBOL = 0.60
# ---------------------------------------------------------------- LA POLITICA
# Hay DOS, y la diferencia no es un numero: es a quien sirve la maquina.
#
#   "prudente"  la de §63 y §72.5. Solo firma cuando el LOTE que pasa promete el
#               90 % (+4 de margen) y ademas ninguna casilla incapaz puede colar
#               una hoja. Es honesta y es la que se cita en el paper... y en la
#               tanda del 11/09 firma 20 hojas de 112. El 82 % va a una persona.
#
#   "planta"    la que pidio el experto el 2026-09-14, y el argumento es de planta,
#               no de laboratorio: *«si duda en todas, de nada sirve ponerlo en
#               planta»*. La maquina clasifica SIEMPRE salvo caso extremo, y el
#               error se corrige con la retroalimentacion del operario.
#
# Medido antes de ponerlo, con piso 0.45 (Tabaco 91):
#
#                              clasifica sola   acierta   aparta   acierto de
#                                                                  las apartadas
#   5 clases, otra jornada          96 %         67.6 %     4 %        50 %
#   4 clases, otra jornada          96 %         83.3 %     4 %        25 %
#   5 clases, misma sesion          89 %         96.6 %    11 %       100 %
#
# **Bajar el corte NO rompe el cajon de rechazo**: lo poco que aparta sigue
# siendo lo peor del lote (25-50 % de acierto contra el 83 % del resto). Lo que
# se pierde es la promesa del 90 % sobre lo que pasa, y eso hay que decirlo
# junto a la cifra siempre.
POLITICA = "planta"  # "planta" | "prudente"
PISO = 0.90          # confianza minima de calidad para firmar en "planta"
# EL PISO SUBE DE 0.75 A 0.90 (2026-09-15, tarde. Decision del experto sobre la
# tabla del punto de trabajo, con la regla del cero ya puesta):
#
#     piso   firma   acierta ahi   ERRORES COLADOS   van a revisar
#     0.75    92       92.4 %            7                20
#     0.90    74       94.6 %            4                38
#
# Se cambian 18 revisiones de mas por CASI LA MITAD de las hojas que se van
# a la caja equivocada (7 -> 4). El razonamiento es de planta, no de metrica:
# la hoja colada NADIE la vuelve a mirar; la revision de mas la ve un tecnico.
# Respaldo del archivo anterior: clasifica_ANTES_piso090.py
#
# LO DE ANTES, que se conserva porque explica de donde viene el 0.75:
# EL PISO SUBIO DE 0.45 A 0.75 (2026-09-15, manana).
#
# Por que 0.75 y no otro: es el piso donde el sistema CUMPLE LA PROMESA que lleva
# haciendo desde agosto -- "lo que firmo solo acierta el 90 %". Medido sobre las
# 112 hojas del 11/09 con el modelo que lleva el sudado:
#
#     piso   firma sola   acierta ahi   a revision   mal enrutadas
#     0.45    109 (97%)      83.5 %        3            18      <- lo de antes
#     0.60     92 (82%)      87.0 %       20            12
#     0.70     85 (76%)      88.2 %       27            10
#     0.75     80 (71%)      90.0 %       32             8      <- aqui
#     0.80     71 (63%)      91.5 %       41             6
#
# Con 0.45 el sistema prometia 90 y entregaba 83.5. Con 0.75 entrega 90.0 y las
# hojas mal enrutadas bajan de 18 a 8. Lo que cuesta: una de cada 3.5 hojas va a
# revision, y eso lo decidio el experto sabiendo lo que cuesta en gente.
#
# LO QUE EL PISO NO ARREGLA: los errores capa <-> XL izquierdo, que son 13 de los
# 20 y vienen con confianza alta. Eso es el detector de agujeros, no la politica.

# PISO PROPIO POR CAJON (2026-09-15). Regla del experto, dicha con sus palabras:
# "en vez de que clasifiques mal una calidad de hoja, la envies al recipiente de
# no estoy seguro, necesito que alguien la revise".
#
# BANDA no se firma NUNCA. Por que:
#   - La banda se define por SUDADO y manchas, y **no** por agujeros en la zona
#     usable. De los 64 rasgos de calidad, NINGUNO mide sudado: hay 7 de mancha
#     y 9 de agujero. El modelo no tiene con que reconocerla.
#   - En las 112 hojas del 11/09 dijo "banda" **7 veces y las 7 eran falsas**
#     (en esa jornada no hay ni una banda real): un tercio de todos los errores.
#   - Mandarlas a revision quita 6 hojas mal enrutadas (18 -> 12) a cambio de
#     dejar de firmar 6. Uno a uno, que es mucho mas barato que subir el piso
#     global: con 0.71 se sacrifican 30 hojas de cobertura para evitar 8 errores.
#
# El veto se levanta cuando (a) el modelo mida el sudado y (b) haya bandas en un
# conjunto de prueba para comprobar que las reconoce. Hoy no se cumple ninguna.
# 1.01 es inalcanzable a proposito: ninguna confianza llega ahi.
PISO_CAJON = {}     # 2026-09-15, tarde: el experto revoca el veto a banda. "LA BANDA
                    # SI DEBE DE SER CLASIFICADA": es una capa sudada/manchada, y
                    # lo que hay que arreglar es que el modelo la reconozca, no
                    # esconderla en el recipiente de revision.

UMBRAL = 0.90        # acierto MEDIDO minimo para aceptar sin revision humana
MARGEN = 0.04        # margen del corte. Era 0.02 (§63); 4 puntos desde el
                     # 2026-09-09 por decision del experto (§72.6): en su planta una
                     # hoja mal enrutada cuesta MUCHO mas que una revisada.

CLASE_ES = {"capa": "CAPA", "banda": "BANDA", "media_banda": "1/2 BANDA",
            "xl_izq": "XL IZQUIERDO", "xr_der": "XR DERECHO"}
VAR_ES = {"habano": "Habano", "connecticut": "Connecticut"}

# Nombres de rasgo en castellano llano. Solo los que suelen salir arriba; los que
# no esten se imprimen con su nombre tecnico, que es mejor que inventarles uno.
NOMBRES = {
    "area_util_izq_pulg2": "superficie perdida por defectos en la mitad IZQUIERDA",
    "area_util_der_pulg2": "superficie perdida por defectos en la mitad DERECHA",
    "asim_area_util": "de que lado esta el dano (izquierda menos derecha)",
    "dif_area_util": "cuanto mas dano tiene un lado que el otro",
    "area_def_pulg2": "superficie total de agujeros y roturas",
    "frac_area_def": "parte de la hoja que se ha ido en defectos",
    "mayor_pulg2": "tamano del defecto mas grande",
    "mayor_dist_base_pulg": "a que altura de la hoja esta el defecto mas grande",
    "n_agujeros": "cuantos agujeros", "n_roturas": "cuantas roturas",
    "n_defectos": "cuantos defectos en total",
    "n_util_izq": "defectos en la mitad izquierda",
    "n_util_der": "defectos en la mitad derecha",
    "pen_mordida_izq_pulg": "cuanto se mete el mordisco por la izquierda",
    "pen_mordida_der_pulg": "cuanto se mete el mordisco por la derecha",
    "pen_mordida_pulg": "cuanto se mete el mordisco mas hondo",
    "asim_pen_mordida": "de que lado se mete mas el mordisco",
    "frac_verde": "parte de la hoja con mancha verde (sudada)",
    "verde_izq_pulg2": "mancha verde en la mitad izquierda",
    "verde_der_pulg2": "mancha verde en la mitad derecha",
    "asimetria_mancha": "de que lado estan las manchas",
    "mayor_mancha_pulg2": "tamano de la mancha mas grande",
    "mancha_izq_pulg2": "manchas en la mitad izquierda",
    "mancha_der_pulg2": "manchas en la mitad derecha",
    "largo_pulg": "largo de la hoja", "ancho_pulg": "ancho de la hoja",
    "aspecto": "lo ancha que es para lo larga que es",
    "solidez": "lo entera que esta la silueta",
    "cobertura": "cuanto del cuadro ocupa la hoja",
    "hoja_entre_util": "cuanto de la hoja queda fuera de la zona util",
    "hoja_L_med": "lo clara que es la hoja", "hoja_L_desv": "lo pareja que es su luz",
    "hoja_a_med": "lo verde o roja que es", "hoja_b_med": "lo amarilla que es",
    "hoja_b_desv": "lo variado de su amarillo",
    "hoja_f_oscuro": "cuanta hoja hay mas oscura que el resto",
    "hoja_f_verdoso": "cuanta hoja hay mas verde que el resto",
    "hoja_f_claro": "cuanta hoja hay mas clara que el resto",
    "hoja_textura": "lo rugosa que se ve",
    "lado_L_med": "contraste de claridad entre las dos mitades",
    "lado_f_oscuro": "que mitad tiene mas zonas oscuras",
    "lado_f_verdoso": "que mitad tiene mas zonas verdes",
    "lado_textura": "que mitad se ve mas rugosa",
    "punta_0.25_norm": "lo afilada que es la punta (a 1/4 de pulgada)",
    "punta_0.5_norm": "lo afilada que es la punta (a 1/2 pulgada)",
    "punta_1_norm": "lo afilada que es la punta (a 1 pulgada)",
    "punta_2_norm": "la anchura a 2 pulgadas de la punta",
}


# Las nueve medidas de color y textura, para poder nombrar cualquier rasgo
# `lado_*` sin tener que escribirlos uno a uno (son 18 con sus valores absolutos).
MED_ES = {"L_med": "la claridad", "L_desv": "lo pareja que es la luz",
          "a_med": "el tono verde-rojo", "b_med": "el tono amarillo",
          "b_desv": "lo variado del amarillo",
          "f_oscuro": "las zonas oscuras", "f_verdoso": "las zonas verdes",
          "f_claro": "las zonas claras", "textura": "la rugosidad"}


def bonito(c):
    n = NOMBRES.get(c)
    if n:
        return n
    if c.startswith("lado_abs_") and c[9:] in MED_ES:
        return "cuanto se diferencian las dos mitades en {} (sin mirar cual)".format(
            MED_ES[c[9:]])
    if c.startswith("lado_") and c[5:] in MED_ES:
        return "que mitad gana en {}".format(MED_ES[c[5:]])
    if c.startswith("hoja_") and c[5:] in MED_ES:
        return "{} de la hoja".format(MED_ES[c[5:]])
    return c


def acierto_del_tramo(tabla, conf):
    """Que acierto MEDIDO corresponde a esta confianza. None si no hay tramo."""
    for t in tabla:
        if t["desde"] <= conf < t["hasta"] or (conf >= 1.0 and t["hasta"] >= 1.0):
            return t
    return None


def acierto_del_conjunto(acum, conf):
    """Que acierto tiene EL LOTE de hojas con confianza >= esta (2026-09-06, §63).

    No es lo mismo que el acierto de UNA hoja con esta confianza, y la diferencia
    es la que decide cuanta gente hace falta: la promesa del sistema es sobre el
    lote que pasa solo, no sobre cada tramo por separado. Medido anidado, cortar
    asi pasa el 70 % de las hojas contra el 57 % de la regla vieja, cumpliendo el
    mismo 90 % (`out/CALIBRA_CONFIANZA.txt`).

    La curva viene ordenada de mas confianza a menos. Se busca el punto de MAS
    cobertura entre los que siguen por encima de esta confianza.
    """
    if not acum:
        return None
    arriba = [t for t in acum if t["conf"] >= conf]
    if not arriba:
        return float(acum[-1]["acierto"])       # confianza por debajo de todo
    return float(max(arriba, key=lambda t: t["cobertura"])["acierto"])


def contribuciones(modelo, x, clases, i_gana, i_segunda, nombres, n=6):
    """Que rasgos empujaron hacia la clase elegida y contra la segunda.

    Solo tiene sentido con la regresion logistica, donde la decision ES una suma
    de terminos. Con arboles se devuelve None y se dice, en vez de inventar una
    explicacion que el modelo no uso.
    """
    # Con la rejilla, quien decidio entre las dos finalistas es el submodelo de
    # su rama, no el conjunto: se explica con ese, que es el que hizo la suma.
    if hasattr(modelo, "rama"):
        sub, cls_r = modelo.rama(x)
        if clases[i_gana] not in cls_r or clases[i_segunda] not in cls_r:
            return None            # las finalistas estan en ramas distintas
        i_gana = cls_r.index(clases[i_gana])
        i_segunda = cls_r.index(clases[i_segunda])
        modelo, clases = sub, cls_r
    pasos = dict(modelo.named_steps)
    clf = pasos.get("logisticregression")
    if clf is None:
        return None
    z = x
    for k in ("simpleimputer", "standardscaler"):
        if k in pasos:
            z = pasos[k].transform(z)
    z = z[0]
    if len(clases) == 2:
        w = clf.coef_[0] * (1 if i_gana == 1 else -1)
    else:
        w = clf.coef_[i_gana] - clf.coef_[i_segunda]
    ap = w * z
    orden = np.argsort(-np.abs(ap))[:n]
    return [(nombres[j], float(ap[j]), float(z[j])) for j in orden]


def evidencia(r):
    """Las medidas que un tecnico puede comprobar mirando la hoja."""
    R, det = r["rasgos"], r["defectos"]
    L = []
    izq = R.get("area_util_izq_pulg2", np.nan)
    der = R.get("area_util_der_pulg2", np.nan)
    if izq == izq and der == der and (izq + der) > 0.01:
        lado, otro = ("IZQUIERDA", "derecha") if izq > der else ("DERECHA", "izquierda")
        L.append("el dano esta en la mitad {}: {:.2f} pulg2 perdidas contra {:.2f} "
                 "en la {}".format(lado, max(izq, der), min(izq, der), otro))
        if min(izq, der) < 0.15 * max(izq, der):
            L.append("   -> la mitad {} queda limpia, que es lo que define XL/XR "
                     "(PROGRESO 16.2)".format(otro.upper()))
    elif izq == izq and der == der:
        L.append("no hay defectos en la zona util: ni izquierda ni derecha")
    n_ag = R.get("n_agujeros", np.nan)
    n_ro = R.get("n_roturas", np.nan)
    if n_ag == n_ag:
        L.append("{:.0f} agujeros y {:.0f} roturas en toda la hoja".format(n_ag, n_ro))
    may = R.get("mayor_pulg2", np.nan)
    if may == may and may > 0:
        d = [x for x in det if x["tipo"] in ("agujero", "rotura")]
        d = max(d, key=lambda x: x["area_px"]) if d else None
        if d:
            L.append("el mayor mide {:.2f} pulg2 ({} pulg de diametro), esta en "
                     "{} y a {} pulg de la base"
                     .format(may, d["diam_pulg"], d["zona"].replace("util_", "mitad "),
                             d["dist_base_pulg"]))
    pi = R.get("pen_mordida_izq_pulg", np.nan)
    pd = R.get("pen_mordida_der_pulg", np.nan)
    if (pi == pi and pd == pd) and max(pi, pd) > 0:
        L.append("el mordisco de borde se mete {:.2f} pulg por la izquierda y "
                 "{:.2f} por la derecha".format(pi, pd))
        L.append("   -> es la medida de la regla de la chaveta: no importa cuanto "
                 "ocupa sino cuanto entra (PROGRESO 43)")
    ver = R.get("frac_verde", np.nan)
    if ver == ver and ver > 0.0005:
        vi, vd = R.get("verde_izq_pulg2", 0.0), R.get("verde_der_pulg2", 0.0)
        L.append("mancha verde (sudada) en el {:.1%} de la hoja: {:.2f} pulg2 a la "
                 "izquierda, {:.2f} a la derecha".format(ver, vi, vd))
    nb = R.get("n_blanca", np.nan)
    if nb == nb and nb > 0:
        L.append("{:.0f} manchas blancas detectadas -- pero OJO: el detector de "
                 "blanca no esta validado (PROGRESO 53), no se puede afirmar"
                 .format(nb))
    la, an = R.get("largo_pulg", np.nan), R.get("ancho_pulg", np.nan)
    if la == la:
        L.append("la hoja mide {:.1f} x {:.1f} pulgadas".format(la, an))
    p1 = R.get("punta_1_norm", np.nan)
    if p1 == p1:
        L.append("a una pulgada de la punta conserva el {:.1%} de su anchura "
                 "(Habano ~36 %, Connecticut ~43 %; PROGRESO 50.5)".format(p1))
    return L


def carga_modelo():
    """(modelo, meta) del disco. Se paga una vez: 2.2 s (PROGRESO 67.1)."""
    if not (DIR_MODELO / "modelo.joblib").exists():
        raise FileNotFoundError(
            "No hay modelo guardado. Corre antes:  python scripts/entrena.py")
    mod = joblib.load(DIR_MODELO / "modelo.joblib")
    meta = json.loads((DIR_MODELO / "modelo.json").read_text(encoding="utf-8"))
    return mod, meta


# ------------------------------------------------------------- el arbol ---
_ARBOL_CACHE = None
ARBOL_CLASES = ("banda", "capa", "xl_izq", "xr_der")
# los ocho rasgos extra del arbol se llaman distinto en la medida de la foto
ARBOL_EXTRA = {"v_area_util": "v_area_util_pulg2",
               "v_izq": "v_area_util_izq_pulg2",
               "v_der": "v_area_util_der_pulg2",
               "v_area_def": "v_area_def_pulg2"}


def _acumulada_de_tramos(tramos):
    """De tramo a LOTE: que acierta TODO lo que pasa cada piso (§63).

    El arbol guarda su acierto por tramo medido ENTRE sesiones; el que decide es
    el del lote que pasa, no el de la hoja suelta.
    """
    ts = [t for t in tramos if t.get("n") and t.get("acierto") is not None]
    tot = sum(t["n"] for t in ts) or 1
    n = ok = 0
    out = []
    for t in reversed(ts):
        n += t["n"]
        ok += t["n"] * t["acierto"]
        out.append(dict(conf=t["desde"], cobertura=n / tot, acierto=ok / n))
    return out


def carga_arbol():
    """El arbol junto, sus columnas y su acierto medido. Se paga una vez."""
    global _ARBOL_CACHE
    if _ARBOL_CACHE is None:
        import arbol_habano as AH
        d = joblib.load(DIR_ARBOL / "arbol_junto.joblib")
        f_meta = DIR_ARBOL / "arbol_junto.json"
        meta = json.loads(f_meta.read_text(encoding="utf-8")) if f_meta.exists() else {}
        tramos = meta.get("tramos") or []
        # Las seis semillas salen IDENTICAS --la logistica de los nodos es
        # determinista, comprobado hoja a hoja en los dos examenes-- asi que se
        # usa una y las cifras medidas valen tal cual, sin promediar nada.
        _ARBOL_CACHE = dict(m=d["modelos"][0], cols=d["cols"], nom=d["nom"],
                            AH=AH, tramos=tramos,
                            acumulada=_acumulada_de_tramos(tramos))
    return _ARBOL_CACHE


def vector_arbol(R, nom):
    """La hoja medida, en el orden de rasgos que espera el arbol."""
    return np.array([[R.get(ARBOL_EXTRA.get(c, c), np.nan) for c in nom]], float)


def probas_arbol(A, xq):
    """La cadena de probabilidad de los tres nodos, en el orden ARBOL_CLASES.

    Solo se usa para decir la SEGUNDA opcion y su probabilidad. Quien decide es
    `AH.decide`, que es lo que se midio: con la cadena, el Connecticut baja de
    84.8 a 82.1 % porque el umbral duro del nodo 2 hace de regla del cero.
    """
    m, cols = A["m"], A["cols"]
    p1 = m["n1"].predict_proba(xq[:, cols["n1"]])[0]
    pb = float(p1[list(m["n1"].classes_).index("banda")])
    p2 = m["n2"].predict_proba(xq[:, cols["n2"]])[0]
    pc = float(p2[list(m["n2"].classes_).index("capa")])
    p3 = m["n3"].predict_proba(xq[:, cols["n3"]])[0]
    c3 = list(m["n3"].classes_)
    return np.array([pb, (1 - pb) * pc,
                     (1 - pb) * (1 - pc) * float(p3[c3.index("xl_izq")]),
                     (1 - pb) * (1 - pc) * float(p3[c3.index("xr_der")])])


def nodo_del_arbol(A, cal):
    """(pipeline, columnas, clases) del nodo que hizo la suma para esta clase."""
    k = "n1" if cal == "banda" else ("n2" if cal == "capa" else "n3")
    m = A["m"][k]
    return m, A["cols"][k], list(m.classes_)


def decide(foto, umbral=UMBRAL, ancho_cinta=None, con_vena=False, modelo=None,
           politica=None, piso=None, rasgos=None):
    """UNA foto -> la decision entera, sin imprimir nada.

    Sacado de `main()` el 2026-09-08 para que el servicio de la celda (E.5)
    llame a la MISMA decision que la linea de comandos, en vez de tener una
    copia que se va separando. El modelo se pasa ya cargado (`modelo`) cuando
    hay que clasificar hoja tras hoja: cargarlo cuesta 2.2 s y decidir 0.75 s.

    No cambia ni un numero respecto a lo que hacia antes: se comprobo con la
    salida `--json` de cinco fotos, identica byte a byte.
    """
    mod, meta = modelo if modelo else carga_modelo()

    # REPRODUCCION SIN FOTOS (2026-09-22). Si el llamante ya trae los rasgos
    # medidos --por ejemplo del CSV del lote independiente que publica este
    # repositorio-- se salta la medicion y se decide exactamente igual.
    if rasgos is not None:
        R = rasgos
        r = {"rasgos": R, "diag": {"origen": "rasgos medidos"}}
    else:
        if mide is None:
            raise RuntimeError("Falta OpenCV/pillow-heif para medir una foto.")
        r = mide(foto, ac_px=ancho_cinta)
        if not r["rasgos"]:
            raise RuntimeError("No se pudo medir la foto: {}"
                               .format(r["diag"].get("error")))
        R = r["rasgos"]

    xv = np.array([[R.get(c, np.nan) for c in meta["rasgos_variedad"]]], float)
    mv = mod["variedad"]
    pv = mv.predict_proba(xv)[0]
    var = mv.classes_[int(pv.argmax())]
    conf_v = float(pv.max())

    xq = np.array([[R.get(c, np.nan) for c in meta["rasgos_calidad"]]], float)
    mq = mod["calidad"][var]
    pq = mq.predict_proba(xq)[0]

    # LA REGLA DEL CERO (2026-09-15). El experto: "el XL y XR es por defecto en uno de
    # los dos lados". Si la tuberia NO MIDE DANO en la zona usable --ni a un lado
    # ni al otro-- la hoja no puede ser XL ni XR, y decirlo no es predecir nada:
    # el cero esta medido.
    #
    # No es la "rejilla" de esta tarde, que fallo porque preguntaba a un MODELO si
    # habia agujeros. Aqui no hay modelo: o hay area danada o no la hay.
    #
    # Medido sobre las 112 del 11/09: 63 hojas tienen cero dano medido (60 son
    # capa, 3 son XL/XR a las que el detector no les vio el dano). El modelo decia
    # XL/XR en 5 de esas 63 y solo acertaba 1. Con la regla: +4 aciertos, -1.
    #     sin la regla  92 de 112 = 82.1 %
    #     con la regla  95 de 112 = 84.8 %
    #
    # LO QUE CUESTA, y hay que decirlo: las 3 hojas que SON XL/XR sin dano medido
    # quedan condenadas a fallar. Dos de las tres ya fallaban. El dia que el
    # detector vea ese dano, esta regla deja de quitar nada.
    _iz = R.get("area_util_izq_pulg2", np.nan)
    _de = R.get("area_util_der_pulg2", np.nan)
    regla_cero = (_iz == 0.0 and _de == 0.0)
    if regla_cero:
        _cls = list(mq.classes_)
        for _c in ("xl_izq", "xr_der"):
            if _c in _cls:
                pq[_cls.index(_c)] = 0.0
        _t = float(pq.sum())
        pq = pq / _t if _t > 0 else mq.predict_proba(xq)[0]

    orden = np.argsort(-pq)
    cal = mq.classes_[orden[0]]
    conf_q = float(pq[orden[0]])
    segunda = mq.classes_[orden[1]]
    prob_segunda = float(pq[orden[1]])

    # EL ARBOL DECIDE LA CALIDAD (2026-09-19, noche). El plano de arriba se
    # sigue calculando --cuesta un milisegundo y deja comparar en el registro--
    # pero quien manda es el arbol: fuera de sesion 57.0 % contra 31.3 % en el
    # Habano, y 84.8 contra 82.1 en el Connecticut.
    A = (carga_arbol() if (ARBOL and (DIR_ARBOL / "arbol_junto.joblib").exists())
         else None)
    cal_plano, conf_plano = str(cal), conf_q
    if A is not None:
        xa = vector_arbol(R, A["nom"])
        pr_a, cf_a = A["AH"].decide(A["m"], xa, A["cols"])
        cal, conf_q = str(pr_a[0]), float(cf_a[0])
        P = probas_arbol(A, xa)
        i2 = [int(i) for i in np.argsort(-P) if ARBOL_CLASES[int(i)] != cal][0]
        segunda, prob_segunda = ARBOL_CLASES[i2], float(P[i2])
        # la regla del cero no se aplica: el umbral duro del nodo 2 ya hace ese
        # trabajo (con ella el Habano bajaba de 57.0 a 56.5 y el Cnt no subia)
        regla_cero = False

    # EL ACIERTO QUE DECIDE ES EL DEL LOTE, no el del tramo (§63, 2026-09-06).
    # La regla vieja (tramo a tramo) sigue calculada y se imprime, porque es la
    # que describe UNA hoja; pero mandaba a revision hojas que podian pasar.
    t_v = acierto_del_tramo(meta["variedad"]["confianza"], conf_v)
    tabla_q = A["tramos"] if A is not None else meta["calidad"][var]["confianza"]
    t_q = acierto_del_tramo(tabla_q, conf_q)
    ac_v_tramo = t_v["acierto"] if t_v else None
    ac_q_tramo = t_q["acierto"] if t_q else None
    ac_v = acierto_del_conjunto(meta["variedad"].get("acumulada"), conf_v)
    acum_q = (A["acumulada"] if A is not None
              else meta["calidad"][var].get("acumulada"))
    ac_q = acierto_del_conjunto(acum_q, conf_q)
    if ac_v is None or ac_q is None:            # modelo guardado antes de §63
        ac_v, ac_q = ac_v_tramo, ac_q_tramo
    # los dos aciertos se multiplican: la calidad solo vale si la variedad acerto
    ac_total = (ac_v * ac_q) if (ac_v is not None and ac_q is not None) else None
    # el margen es el mismo de entrena.py: sin el, el corte entrega 89.5 cuando
    # promete 90, porque la curva es ella misma una estimacion con ruido
    acepta = ac_total is not None and ac_total >= umbral + MARGEN

    # EL VETO AL CAJON INCAPAZ (2026-09-09, §72.5). La pregunta no es sobre esta
    # hoja sino sobre el cajon entero: de todas las hojas a las que el modelo
    # llama como a esta, ¿el lote llega al umbral EN ALGUN corte? Con la regla
    # de antes, el cajon "xl_izq" enrutaba 39 hojas acertando el 74.4 % y era por
    # donde se escapaba el lote; su curva no llega al 90 % ni en su punto mas
    # seguro, asi que ninguna hoja de ese cajon puede pasar sola, por confiada
    # que venga. Los demas cajones NO se tocan: capa sigue igual que antes.
    # Un cajon sin curva propia (pocas hojas) no se veta: decide la regla vieja.
    caj = (None if A is not None else
           (meta["calidad"][var].get("acumulada_cajon") or {}).get(str(cal)))
    techo_cajon = max((t["acierto"] for t in caj), default=None) if caj else None
    veto = techo_cajon is not None and techo_cajon < umbral + MARGEN
    if veto:
        acepta = False

    # POLITICA "planta": clasificar salvo caso extremo. Se sustituyen las dos
    # reglas de arriba -- el umbral del lote y el veto al cajon -- por un piso
    # de confianza. El veto se quita a proposito: es lo que bloqueaba clases
    # enteras (xl_izq no podia firmar NINGUNA hoja, §72.5), y con el puesto no
    # hay piso que sirva.
    pol = POLITICA if politica is None else politica
    pis = (PISO_ARBOL if A is not None else PISO) if piso is None else piso
    if pol == "planta":
        veto = False
        piso_cajon = PISO_CAJON.get(str(cal), pis)
        acepta = conf_q >= piso_cajon
        # se marca como veto para que el registro diga POR QUE se fue a revision
        veto = (not acepta) and str(cal) in PISO_CAJON

    if A is not None:
        nodo, col_n, cls_n = nodo_del_arbol(A, cal)
        gana = cal if cal in cls_n else cls_n[0]
        otra = [c for c in cls_n if c != gana][0]
        contrib = contribuciones(nodo, xa[:, col_n], cls_n, cls_n.index(gana),
                                 cls_n.index(otra),
                                 [A["nom"][i] for i in col_n])
    else:
        contrib = contribuciones(mq, xq, list(mq.classes_), int(orden[0]),
                                 int(orden[1]), meta["rasgos_calidad"])

    vena = None
    if con_vena:
        import venas_asim as VA
        rgb14 = VA.carga(foto)
        from segmentacion import mascara_hoja, mascara_marca
        m14 = mascara_hoja(rgb14, mascara_marca(rgb14))[0]
        vena = VA.mide_hoja(rgb14, m14)

    return dict(
        r=r, R=R, var=var, cal=cal, segunda=segunda, conf_v=conf_v,
        conf_q=conf_q, prob_segunda=prob_segunda, ac_v=ac_v, ac_q=ac_q,
        arbol=(A is not None), cal_plano=cal_plano, conf_plano=conf_plano,
        ac_total=ac_total, ac_v_tramo=ac_v_tramo, ac_q_tramo=ac_q_tramo,
        acepta=acepta, contrib=contrib, vena=vena, umbral=umbral,
        veto_cajon=veto, techo_cajon=techo_cajon, politica=pol, piso=pis,
        regla_cero=bool(regla_cero))


def a_json(D):
    """El mismo diccionario que imprimia `--json` antes del refactor."""
    r = D["r"]
    return dict(
        archivo=r["diag"]["archivo"], variedad=D["var"], calidad=D["cal"],
        confianza_variedad=round(D["conf_v"], 4),
        confianza_calidad=round(D["conf_q"], 4),
        acierto_medido_variedad=D["ac_v"], acierto_medido_calidad=D["ac_q"],
        acierto_medido_total=D["ac_total"], umbral=D["umbral"], margen=MARGEN,
        acierto_del_tramo_variedad=D["ac_v_tramo"],
        acierto_del_tramo_calidad=D["ac_q_tramo"],
        decision="aceptar" if D["acepta"] else "revision_manual",
        veto_cajon=D.get("veto_cajon", False),
        regla_cero=D.get("regla_cero", False),
        techo_del_cajon=D.get("techo_cajon"),
        segunda_opcion=D["segunda"], prob_segunda=round(D["prob_segunda"], 4),
        defectos=[{k: v for k, v in d.items() if not k.startswith("_")}
                  for d in r["defectos"]],
        avisos=r["diag"]["avisos"], escala=r["diag"]["origen_escala"],
        vena=D["vena"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("foto")
    ap.add_argument("--politica", default=POLITICA,
                    choices=["planta", "prudente"],
                    help="planta: clasifica salvo caso extremo (piso de "
                         "confianza). prudente: la regla del lote de §63 mas el "
                         "veto por casilla de §72.5.")
    # default None para que mande el piso de quien decide: PISO_ARBOL (0.60)
    # con el arbol, PISO (0.90) con el modelo plano. Se sigue pudiendo dar.
    ap.add_argument("--piso", type=float, default=None,
                    help="confianza minima para firmar en politica planta")
    ap.add_argument("--umbral", type=float, default=UMBRAL,
                    help="acierto medido minimo para no mandar a revision")
    ap.add_argument("--ancho-cinta", type=float, default=None,
                    help="ancho de cinta en px de mascara, si se sabe de la sesion")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--vena", action="store_true",
                    help="mide ademas la asimetria de vena (PROGRESO 52). No entra"
                         " en la decision: el modelo no tiene ningun rasgo de vena")
    a = ap.parse_args()

    try:
        D = decide(a.foto, a.umbral, a.ancho_cinta, a.vena,
                   politica=a.politica, piso=a.piso)
    except (FileNotFoundError, RuntimeError) as ex:
        sys.exit(str(ex))

    r = D["r"]
    var, cal, segunda = D["var"], D["cal"], D["segunda"]
    conf_v, conf_q = D["conf_v"], D["conf_q"]
    ac_v, ac_q, ac_total = D["ac_v"], D["ac_q"], D["ac_total"]
    acepta, contrib, vena = D["acepta"], D["contrib"], D["vena"]
    con_arbol = bool(D.get("arbol"))

    if a.json:
        print(json.dumps(a_json(D), indent=2, ensure_ascii=False, default=str))
        return

    print()
    print("=" * 74)
    print("  {}".format(r["diag"]["archivo"]))
    print("=" * 74)
    print()
    print("  VARIEDAD   {:<16s} el modelo dice {:.2f}  ->  el lote que pasa con"
          " esa confianza acierta {}"
          .format(VAR_ES.get(var, var), conf_v,
                  "{:.0f} %".format(100 * ac_v) if ac_v is not None else "?"))
    print("  CALIDAD    {:<16s} el modelo dice {:.2f}  ->  el lote que pasa con"
          " esa confianza acierta {}"
          .format(CLASE_ES.get(cal, cal), conf_q,
                  "{:.0f} %".format(100 * ac_q) if ac_q is not None else "?"))
    print()
    # Con la politica "planta" quien decide es el PISO de confianza, no el
    # umbral de acierto del lote: decirlo al reves era contar otra regla.
    planta = D.get("politica") == "planta"
    if acepta and planta:
        print("  DECISION   ACEPTAR: confianza {:.2f}, por encima del piso de {:.2f}"
              .format(conf_q, D["piso"]))
        if ac_q is not None:
            print("             (de todo lo que pasa ese piso, acierta el {:.0f} %"
                  " medido)".format(100 * ac_q))
    elif not acepta and planta and not D.get("veto_cajon"):
        print("  DECISION   A REVISION MANUAL: confianza {:.2f}, por debajo del piso"
              " de {:.2f}".format(conf_q, D["piso"]))
    elif acepta:
        print("  DECISION   ACEPTAR: {:.0f} % de acierto medido, por encima del "
              "umbral de {:.0f} % (+{:.0f} de margen)"
              .format(100 * ac_total, 100 * a.umbral, 100 * MARGEN))
    elif D.get("veto_cajon"):
        print("  DECISION   A REVISION MANUAL: el cajon {} no llega al umbral NI EN"
              " SU MEJOR CORTE".format(CLASE_ES.get(cal, cal)))
        print("             (su techo es {:.0f} %, y hace falta {:.0f} %). No es"
              " esta hoja: es que"
              .format(100 * D["techo_cajon"], 100 * (a.umbral + MARGEN)))
        print("             de todas las hojas que el modelo llama asi, el lote no"
              " cumple la promesa. §72.5")
    else:
        print("  DECISION   A REVISION MANUAL: {} de acierto medido, por debajo del"
              " umbral de {:.0f} % (+{:.0f} de margen)"
              .format("{:.0f} %".format(100 * ac_total) if ac_total is not None
                      else "acierto desconocido", 100 * a.umbral, 100 * MARGEN))
    print("             segunda opcion: {} ({:.2f})"
          .format(CLASE_ES.get(segunda, segunda), D["prob_segunda"]))

    print()
    print("  LO QUE SE VE EN LA HOJA   (medidas, comprobables mirandola)")
    for s in evidencia(r):
        print("     - " + s if not s.startswith("   ") else "     " + s.strip())

    print()
    if contrib:
        # Con el arbol, la suma que se ensena es la del NODO que decidio, y su
        # contraste no es "esta clase contra la segunda" sino el del nodo: el 2
        # separa capa de lateral sin decir aun de que lado. Se dice tal cual.
        if con_arbol:
            contra = {"banda": "EL RESTO", "capa": "LATERAL (XL/XR)",
                      "xl_izq": "XR DERECHO", "xr_der": "XL IZQUIERDO"}.get(cal, "")
            print("  LO QUE PESO EN LA DECISION   (la aritmetica del nodo, {} contra {})"
                  .format(CLASE_ES.get(cal, cal), contra))
        else:
            print("  LO QUE PESO EN LA DECISION   (la aritmetica del modelo, {} contra {})"
                  .format(CLASE_ES.get(cal, cal), CLASE_ES.get(segunda, segunda)))
        for nombre, aporte, z in contrib:
            signo = "a favor" if aporte > 0 else "en contra"
            print("     {:9s} {:5.2f}   {}".format(signo, abs(aporte),
                                                   bonito(nombre)))
        print("     (el numero es cuanto mueve la balanza, no pulgadas ni por ciento)")
    else:
        print("  LO QUE PESO EN LA DECISION")
        print("     este modelo es de arboles: su decision no es una suma de")
        print("     terminos, asi que no se puede repartir en rasgos sin inventar.")
        print("     Se enseñan las medidas de arriba, que son lo que si es cierto.")

    if vena:
        print()
        print("  LA VENA   (medida aparte; NO entra en la decision, PROGRESO 52)")
        ai = vena.get("asim_vena_frac", "")
        if ai not in ("", None):
            lado = "IZQUIERDA" if float(ai) > 0 else "DERECHA"
            print("     vena mas resaltada en la mitad {} (asimetria {:+.4f})"
                  .format(lado, float(ai)))
            print("     acierta el lado en el 70 % de las hojas sin otro defecto;")
            print("     el modelo todavia no tiene ningun rasgo de vena.")

    if r["diag"]["avisos"]:
        print()
        print("  AVISOS DE ESTA FOTO")
        for s in r["diag"]["avisos"]:
            print("     - {}".format(s))

    print()
    print("  COMO SE MIDIO")
    print("     escala: {}   ({} px de cinta, {} pulg/px)"
          .format(r["diag"]["origen_escala"], r["diag"]["ancho_cinta_px"],
                  r["diag"]["pulg_por_px"]))
    print("     giro aplicado: {}{}"
          .format(r["diag"]["rotacion"],
                  " y volteo de 180" if r["diag"]["voltea_180"] else ""))
    print()
    if con_arbol:
        print("  LO QUE SE PUEDE PROMETER, Y LO QUE NO")
        print("     La calidad la decide el arbol junto, y sus cifras estan")
        print("     medidas con la jornada del examen FUERA del entrenamiento:")
        print("     57.0 % en las 214 hojas de Habano del 19/09 y 84.8 % en las")
        print("     112 de Connecticut del 11/09 (decir siempre la clase mas")
        print("     comun daria 38.7 y 63.4). Con el piso de 0.70 firma la mitad")
        print("     de las hojas y ahi acierta el 74.5 %.")
        print("     La VARIEDAD la sigue diciendo el modelo plano: 214 de 217 y")
        print("     111 de 112 en esos mismos examenes.")
        print("     NO esta medido: la clase media banda --el arbol no la tiene--")
        print("     ni que pasa con una camara o una mesa distintas de las cuatro")
        print("     jornadas con las que se entreno.")
    else:
        print("  LO QUE NO SE PUEDE PROMETER")
        print("     Este modelo se entreno con UNA sesion de fotos en la que la clase")
        print("     y la hora del dia van juntas (PROGRESO 54.2). Las cifras de")
        print("     acierto de arriba son de validacion cruzada DENTRO de esa sesion.")
        print("     Que se repitan con otra camara, otra mesa u otro dia NO esta")
        print("     medido, y con los datos de hoy no se puede medir.")
    print()


if __name__ == "__main__":
    main()
