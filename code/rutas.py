# -*- coding: utf-8 -*-
"""Raiz del repositorio y ubicacion de RoboDK.

Todos los guiones resuelven sus rutas desde aqui, de modo que el repositorio
corre en cualquier maquina sin editar una sola linea. REPO_RAIZ es la carpeta
que contiene `code/`, `out/` y `dataset/`.

RoboDK solo hace falta para la celda simulada. Si esta instalado en otro sitio,
define la variable de entorno ROBODK_DIR antes de correr:

    set ROBODK_DIR=D:/Programas/RoboDK        (Windows)
    export ROBODK_DIR=/opt/robodk             (Linux)
"""
import os
from pathlib import Path

REPO_RAIZ = Path(__file__).resolve().parent.parent
ROBODK_RAIZ = Path(os.environ.get("ROBODK_DIR", "C:/RoboDK"))
