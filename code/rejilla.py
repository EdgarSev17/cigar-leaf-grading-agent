r"""
LA REJILLA DE DOS PREGUNTAS, COMO CLASIFICADOR. (2026-09-15)

LA REGLA, DICHA POR EL EXPERTO
--------------------------
    "RECUERDA QUE EL XL Y XR ES POR DEFECTO EN UNO DE LOS DOS LADOS, LA BANDA ES
     UNA CAPA QUE NO LLEGA A SER CAPA PORQUE ESTA MANCHADA, SUDADA O CON MANCHAS
     BLANCAS."

O sea que la calidad no son cuatro opciones sueltas, son DOS preguntas:

                        sin agujeros        con agujeros en un lado
    no sudada              CAPA                  XL izq / XR der
    sudada                 BANDA                 (1/2 banda, fuera del modelo)

POR QUE HACIA FALTA
-------------------
El modelo plano elegia entre las cuatro a la vez, y nada le impedia saltarse la
rejilla. Medido sobre las 112 del 11/09 con el modelo del sudado ya dentro: de
las tres hojas que llamaba "banda", **las tres tenian agujeros** (4, 5 y 6), y
dos de ellas eran XR derecho. Cuatro de 112 predicciones eran imposibles por
definicion.

Con la rejilla eso **no puede pasar por construccion**: la clase sale de dos
respuestas encadenadas, no de una eleccion entre cuatro.

COMO SE COMPONE LA CONFIANZA
-----------------------------
    P(capa)   = P(sin agujeros) * P(capa | sin)
    P(banda)  = P(sin agujeros) * P(banda | sin)
    P(xl_izq) = P(con agujeros) * P(xl | con)
    P(xr_der) = P(con agujeros) * P(xr | con)

Es una probabilidad como la de antes --suma 1-- asi que la calibracion, el piso
de confianza y todo lo que hay aguas abajo siguen funcionando sin tocarlos.

LO QUE ESTA CLASE NO HACE
--------------------------
No decide el umbral ni la abstencion: eso sigue en `clasifica.py`. Aqui solo se
responde, con la estructura correcta.
"""
import numpy as np

SIN, CON = "sin_agujeros", "con_agujeros"


def reparte(y):
    """La etiqueta de la primera pregunta, a partir de la clase final."""
    return np.where(np.isin(y, ["xl_izq", "xr_der"]), CON, SIN)


class Rejilla:
    """Se comporta como un clasificador de sklearn: `classes_`, `predict_proba`.

    `m_ag`  : ¿hay agujeros en un lado?      sin_agujeros / con_agujeros
    `m_sin` : dentro de las limpias          capa / banda        (lo decide el sudado)
    `m_con` : dentro de las agujereadas      xl_izq / xr_der     (lo decide el lado)
    """

    def __init__(self, m_ag, m_sin, m_con):
        self.m_ag, self.m_sin, self.m_con = m_ag, m_sin, m_con
        cls = list(m_sin.classes_) + list(m_con.classes_)
        self.classes_ = np.array(sorted(cls))

    def _bloques(self, X):
        p_ag = self.m_ag.predict_proba(X)
        i_sin = list(self.m_ag.classes_).index(SIN)
        i_con = list(self.m_ag.classes_).index(CON)
        return (p_ag[:, i_sin], p_ag[:, i_con],
                self.m_sin.predict_proba(X), self.m_con.predict_proba(X))

    def predict_proba(self, X):
        X = np.asarray(X, float)
        p_sin, p_con, p1, p2 = self._bloques(X)
        out = np.zeros((X.shape[0], len(self.classes_)), float)
        for j, c in enumerate(self.m_sin.classes_):
            out[:, list(self.classes_).index(c)] = p_sin * p1[:, j]
        for j, c in enumerate(self.m_con.classes_):
            out[:, list(self.classes_).index(c)] = p_con * p2[:, j]
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # --- para que `clasifica.contribuciones` pueda seguir explicando -------
    def rama(self, x):
        """(submodelo, clases) de la rama por la que se fue esta hoja."""
        x = np.asarray(x, float)
        p_sin, p_con, _, _ = self._bloques(x)
        if p_sin[0] >= p_con[0]:
            return self.m_sin, list(self.m_sin.classes_)
        return self.m_con, list(self.m_con.classes_)

    @property
    def named_steps(self):
        return {}

    def __repr__(self):
        return "Rejilla(agujeros -> {capa|banda} / {xl_izq|xr_der})"
