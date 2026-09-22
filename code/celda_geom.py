# -*- coding: utf-8 -*-
"""Geometria de la celda: cajas y cilindros en STL binario, en MILIMETROS.

Existe para que la estacion de RoboDK **se genere por guion y no a mano**. Es la
misma disciplina que el resto del proyecto: si la celda se arma pinchando en la
interfaz, nadie puede reproducirla ni saber por que una cota es la que es. Asi
cada medida esta escrita, con su motivo al lado, y rehacer la estacion entera
cuesta segundos.

Las unidades son milimetros porque es lo que usan RoboDK y las 714 mallas de hoja
(§65.1), asi que no hay ninguna conversion en ningun sitio.
"""
import struct
from pathlib import Path

import numpy as np


def _tri(v0, v1, v2):
    n = np.cross(np.asarray(v1) - np.asarray(v0), np.asarray(v2) - np.asarray(v0))
    m = np.linalg.norm(n)
    n = n / m if m > 1e-12 else np.array([0.0, 0.0, 1.0])
    return n, v0, v1, v2


def escribe_stl(destino, triangulos, cabecera=b"celda tabaco"):
    """STL binario. `triangulos` es una lista de (v0, v1, v2)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as fh:
        fh.write(cabecera.ljust(80, b" ")[:80])
        fh.write(struct.pack("<I", len(triangulos)))
        for a, b, c in triangulos:
            n, a, b, c = _tri(a, b, c)
            fh.write(struct.pack("<12fH", *n, *a, *b, *c, 0))
    return destino


def caja(lx, ly, lz, centro=(0.0, 0.0, 0.0), apoyada=True):
    """Caja maciza. `apoyada=True` deja la cara de abajo en z = centro_z.

    Se apoya por abajo a proposito: en una celda casi todo descansa en el suelo o
    sobre la banda, y asi la cota que se escribe es la altura del objeto, no la de
    su centro -- que es como se mide en planta.
    """
    cx, cy, cz = centro
    x0, x1 = cx - lx / 2.0, cx + lx / 2.0
    y0, y1 = cy - ly / 2.0, cy + ly / 2.0
    z0, z1 = (cz, cz + lz) if apoyada else (cz - lz / 2.0, cz + lz / 2.0)
    p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    caras = [(0, 3, 2), (0, 2, 1),      # abajo
             (4, 5, 6), (4, 6, 7),      # arriba
             (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6),
             (3, 0, 4), (3, 4, 7)]
    return [(p[i], p[j], p[k]) for i, j, k in caras]


def cilindro(radio, alto, centro=(0.0, 0.0, 0.0), lados=24):
    """Prisma de `lados` caras, apoyado por abajo. Para ventosas y postes."""
    cx, cy, cz = centro
    ang = np.linspace(0, 2 * np.pi, lados, endpoint=False)
    aro0 = [(cx + radio * np.cos(a), cy + radio * np.sin(a), cz) for a in ang]
    aro1 = [(x, y, cz + alto) for x, y, _ in aro0]
    t = []
    for i in range(lados):
        j = (i + 1) % lados
        t.append((aro0[i], aro0[j], aro1[j]))
        t.append((aro0[i], aro1[j], aro1[i]))
        t.append(((cx, cy, cz), aro0[j], aro0[i]))
        t.append(((cx, cy, cz + alto), aro1[i], aro1[j]))
    return t


def recipiente(lx, ly, lz, espesor=15.0, centro=(0.0, 0.0, 0.0)):
    """Cajon abierto por arriba: suelo y cuatro paredes.

    Abierto porque el brazo deposita la hoja desde arriba, y con tapa la
    simulacion de colision no dejaria entrar la herramienta.
    """
    cx, cy, cz = centro
    t = caja(lx, ly, espesor, (cx, cy, cz))
    t += caja(lx, espesor, lz, (cx, cy - ly / 2.0 + espesor / 2.0, cz))
    t += caja(lx, espesor, lz, (cx, cy + ly / 2.0 - espesor / 2.0, cz))
    t += caja(espesor, ly - 2 * espesor, lz,
              (cx - lx / 2.0 + espesor / 2.0, cy, cz))
    t += caja(espesor, ly - 2 * espesor, lz,
              (cx + lx / 2.0 - espesor / 2.0, cy, cz))
    return t


def barra_ventosas(largo=320.0, n=4, r_cup=22.0, alto_cup=35.0,
                   ancho=70.0, alto_barra=45.0, brida_r=63.0, brida_h=25.0):
    """La herramienta: brida + barra + n ventosas de fuelle en linea.

    POR QUE ESTA FORMA, Y NO UNA PINZA (§65.4)
    ------------------------------------------
    - **Agujas descartadas**: perforan la capa, que es justo la parte que se vende.
    - **Ventosa de sellado sola, mal**: la hoja es porosa y viene arrugada, no
      sella. Lo que funciona es venturi de **alto caudal**, que tolera la fuga, con
      copas de fuelle blandas y poca depresion; o un agarre Bernoulli, que crea
      sustentacion sin sellar y casi sin tocar -- lo que se usa para tela y hoja
      fina.
    - **En LINEA y no en cuadro**: las copas van a lo largo de la **vena central**,
      que es la parte rigida de la hoja. Repartidas en cuadro, la hoja se dobla por
      la lamina al levantarla.
    - Y lo que lo cierra: se agarra por **la zona que la regla ya descarta** -- la
      base y la franja de la vena (§16, §19). Si la ventosa marca la hoja, marca la
      parte que se corta igual.

    El origen es la brida (la muneca del robot); las copas miran hacia -Z.
    """
    t = cilindro(brida_r, brida_h, (0, 0, -brida_h), lados=24)
    t += caja(largo, ancho, alto_barra, (0, 0, -brida_h - alto_barra))
    z = -brida_h - alto_barra - alto_cup
    for k in range(n):
        x = -largo / 2.0 + largo * (k + 0.5) / n
        t += cilindro(r_cup, alto_cup, (x, 0, z), lados=16)
    return t, brida_h + alto_barra + alto_cup      # (geometria, largo del util)
