# -*- coding: utf-8 -*-
"""Repository root, and where RoboDK is installed.

Every script resolves its paths from here, so the repository runs on any
machine without editing a line. REPO_ROOT is the folder that contains `code/`,
`out/` and `dataset/`.

RoboDK is only needed for the simulated cell. If it is installed somewhere
else, set the ROBODK_DIR environment variable before running:

    set ROBODK_DIR=D:/Programs/RoboDK        (Windows)
    export ROBODK_DIR=/opt/robodk            (Linux)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBODK_ROOT = Path(os.environ.get("ROBODK_DIR", "C:/RoboDK"))
