# The simulated cell — Entorno1 (RoboDK)

The station in `Entorno1_4clases.rdk`: a KUKA IONTEC ultra KR 120 R2700, a
conveyor line, an overhead camera and nine bins on an arc. The robot decides
nothing and the agent moves nothing; they talk over a four-bit handshake.

To run it, see **Running the simulated cell** in the repository README.

## Layout

| Element | Position and dimensions |
|---|---|
| Robot | base at (0, 0, 0); the robot faces +X |
| Conveyor line | axis at X = 1500; runs Y from −3229.5 to +1076.5 (**4306 mm**); surface at Z = 766.5 |
| — black belt (`Banda`) | 606 mm wide (X 1197…1803), 30 thick |
| — frame (`Banda Bastidor`) | blue side guides, 10 mm above the belt |
| — bed (`Soporte Banda`) | 9 gantries and 2 beams, from Z = 0 to Z = 656.5 |
| Camera structure | column at X = 2170, Y = −700, up to Z ≈ 2242; cantilever arm to X = 1420 |
| 2D camera | (1500, −700, ≈ 1992), looking along −Z; FOV 28.39°, 640×480 |
| 9 bins | on an arc of radius 1600, from 70° to 290° in steps of 27.5°; open at the top, 660×430×320 (636×406 inside). Each one a different colour; bin 9, red, takes the special cases |
| Leaves | up to 553.2 × 290.2 × 0.3 mm |
| Suction tool | TCP 250 mm from the flange |

Travel is along **+Y**. The camera sits **upstream** of the pick-up point
(Y = 0), so it inspects each leaf before the robot takes it.

| Segment | Distance |
|---|---|
| Line start to camera | 2529.5 mm |
| Camera to pick-up point | 700 mm |
| **Line start to pick-up (total)** | **3229.5 mm** |

## The cycle

1. A leaf is placed at the start of the line and the belt carries it to the
   camera.
2. The belt **stops under the camera** for at least 3 s, the image is saved to
   `camara_hoja_actual.png`, and the cell raises **`HOJA_EN_CAMARA = 1`**.
3. The agent grades the leaf and sends the four-bit code.
4. The cell acknowledges, the belt carries the leaf to the pick-up point, and
   the robot runs `Recoger_Hoja` and then `Dejar_Caja_NN` for that code.
5. The result is appended to the log, and the cycle starts again.

The cell grades the **photograph**, not the render. The camera only raises the
event: at the free licence it gives 3.05 × 6.02 mm per pixel, ten times cruder
than the model measures at.

## Codes and bins

Four bits, `B3` most significant.

| Code | Bits B3..B0 | Bin | Bin colour | Leaf |
|---|---|---|---|---|
| 1 | 0001 | 1 | yellow | Connecticut binder |
| 2 | 0010 | 2 | orange | Connecticut wrapper |
| 3 | 0011 | 3 | dark green | Connecticut XL left |
| 4 | 0100 | 4 | turquoise | Connecticut XR right |
| 5 | 0101 | 5 | blue | Habano binder |
| 6 | 0110 | 6 | purple | Habano wrapper |
| 7 | 0111 | 7 | brown | Habano XL left |
| 8 | 1000 | 8 | white | Habano XR right |
| **9** | **1001** | **9** | **red** | **not recognised, goes to a human inspector** |
| 0, 10–15 | — | 9 | red | invalid code: also bin 9, so that no leaf is lost |

The numbering went from eleven bins to nine on 2026-09-15, when the half-binder
grade was left out of the system: those two bins were never filled. Every other
class kept its colour and only changed its number.

**The bin numbering is defined here and nowhere else.** Six files have to agree
on it, and they are changed together or not at all: this table,
`Construir_Entorno.py`, `Control_Senales.py`, `code/celda_robodk.py` (`CLASES`),
`code/agente_entorno1.py` (`CAJA_REVISION`) and the queue builder.

## Signals

In simulation the signals are RoboDK **station parameters**, which act as
virtual I/O and are read and written through the API (`RDK.setParam` /
`RDK.getParam`, port 20500). All of them are `0` or `1` except the two
informational ones.

| Signal | Direction | Meaning |
|---|---|---|
| `TIPO_B0` … `TIPO_B3` | agent → cell | the four bits of the code; `B0` is least significant |
| `TIPO_VALIDO` | agent → cell | the bits are stable and can be read |
| `HOJA_EN_CAMARA` | cell → agent | a leaf is stopped under the camera, waiting to be graded |
| `ORDEN_RECIBIDA` | cell → agent | acknowledgement |
| `ROBOT_OCUPADO` | cell → agent | the robot is picking or placing |
| `CAJA_DESTINO`, `ULTIMO_CODIGO` | cell → agent | informational: last bin and last code |
| `HOJA_ID` | cell → agent | the file name of the photograph of the leaf under the camera |
| `CONTROL` | anyone → cell | setting it to `STOP` stops the cell cleanly |

## The four-phase handshake

1. The cell sets `HOJA_EN_CAMARA = 1`.
2. The agent writes `TIPO_B0..TIPO_B3` and **then** `TIPO_VALIDO = 1`.
3. The cell reads the bits and sets `ORDEN_RECIBIDA = 1` and
   `HOJA_EN_CAMARA = 0`.
4. The agent sets `TIPO_VALIDO = 0` and the cell answers `ORDEN_RECIBIDA = 0`.

The cell does not ask for another order until the previous handshake has closed,
so a `TIPO_VALIDO` left at 1 cannot be read twice by mistake.

`CONTROL = STOP` is only honoured at **safe points**, waiting for an order or
between leaves. If it arrives while the robot is moving, the leaf in progress is
finished and logged, and the cell stops after it.

## Which leaf is under the camera

`HOJA_ID` carries the file name of the photograph, which **does not say the
grade**. The agent looks that name up in `out/celda/entorno1/cola_hojas.csv` and
grades the real photograph.

The name of the object inside RoboDK — `Hoja 7 - connecticut xl_izq` — does carry
the label, and it is deliberately **not read**. Reading it is what made an
earlier version of the cell appear to classify without classifying anything.

## The queue

`out/celda/entorno1/cola_hojas.csv`, semicolon separated:

| column | meaning |
|---|---|
| `orden` | the order the leaves enter the line |
| `id` | the photograph's file name without extension; this is what `HOJA_ID` carries |
| `variedad`, `clase` | ground truth, used only to score the run, never to decide |
| `caja_real` | the bin that leaf should end up in |
| `foto` | the photograph, relative to the repository root |
| `obj` | the 3D mesh, relative to the repository root |
| `ancho_cinta` | the tape width the folder was measured with, in pixels |

**A row with an empty `obj` is dropped.** If every row is dropped the cell falls
back to feeding leaves at random, and those have no `HOJA_ID`, so every leaf ends
in bin 9. If that is what you see, look at the queue first.

Paths in the queue are relative to the repository root. `Control_Senales.py`
resolves them before handing them to RoboDK, because RoboDK resolves relative
paths against its own working directory, not against the script's.

## Robot programs

Ten, all in the station: `Recoger_Hoja` and `Dejar_Caja_01` through
`Dejar_Caja_09`. These are the ten programs the article reports as running
without collisions.

Collision detection is on between the tool and the structure, and between the
robot and the camera gantry. It is turned off between the tool and the leaf
being carried, and between the belt and that leaf, which would otherwise be
flagged as contact on every cycle.

## Going to a real robot

The signals are already the ones a PLC would use. What changes is where they
live: station parameters become digital I/O, and the camera becomes a real one.
The agent does not change at all — it reads a photograph and answers with four
bits.
