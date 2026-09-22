# Celda de trabajo — Entorno1 (RoboDK)

Celda construida alrededor del **KUKA IONTEC ultra KR 120 R2700** (base en el origen,
alcance 2701 mm). Todas las medidas están en milímetros y en coordenadas de la estación.

## Distribución

| Elemento | Posición / dimensiones |
|---|---|
| Robot | base en (0, 0, 0); el frente del robot es +X |
| Línea de transporte | eje en X = 1500; recorre Y de −3229,5 a +1076,5 (**4306 mm**); superficie a Z = 766,5 |
| — cinta negra (`Banda`) | 606 mm de ancho (X 1197…1803), espesor 30 |
| — bastidor (`Banda Bastidor`) | guías laterales azules, 10 mm por encima de la cinta |
| — bancada (`Soporte Banda`) | 9 pórticos + 2 vigas, de Z = 0 a Z = 656,5 |
| Estructura de la cámara | columna en X = 2170, Y = −700, hasta Z ≈ 2242; brazo en voladizo hasta X = 1420 |
| Cámara 2D | (1500, −700, **≈ 1992**), mirando hacia −Z; FOV 28,39°, 640×480 |
| 9 cajas | arco de radio 1600, de 70° a 290° (paso 27,5°); abiertas por arriba, 660×430×320 (interior 636×406). **Cada una de un color; la 9, roja, es la de casos especiales** |
| Hojas | 8 clases importadas de `hojas_foto_impresa.rdk`; hasta 553,2 × 290,2 × 0,3 mm |
| Ventosa (herramienta) | TCP a 250 mm de la brida |

El sentido de avance es **+Y**. La cámara está **aguas arriba** del punto de recogida (Y = 0),
de modo que inspecciona las hojas antes de que el robot las tome.

| Tramo | Distancia |
|---|---|
| Inicio de línea → cámara | 2529,5 mm |
| Cámara → punto de recogida | 700 mm |
| **Inicio de línea → recogida (total)** | **3229,5 mm** |

## Programa controlado por señales (modelo agéntico)

El programa principal es **`Control_Senales`**. El robot no decide nada: espera a que el
agente le diga, con una señal binaria de 4 bits, dónde va la hoja que tiene bajo la cámara.

### Ciclo

1. Sale una hoja al azar al inicio de la línea y la banda la lleva hasta la cámara.
2. La banda **para bajo la cámara** (mínimo 3 s), se guarda la imagen en
   `camara_hoja_actual.png` y el robot pone **`HOJA_EN_CAMARA = 1`**.
3. El agente identifica la hoja y envía el código de 4 bits.
4. El robot acusa recibo, la banda lleva la hoja al punto de recogida, y el robot ejecuta
   `Recoger_Hoja` y después `Dejar_Caja_NN` según el código.
5. El resultado se anota en `registro_clasificacion.csv`, y vuelta a empezar.

### Códigos (4 bits, B3 = más significativo) y cajas

| Código | Bits B3..B0 | Caja | Color de la caja | Tipo de hoja |
|---|---|---|---|---|
| 1 | 0001 | 1 | amarillo | connecticut banda |
| 2 | 0010 | 2 | naranja | connecticut capa |
| 3 | 0011 | 3 | verde oscuro | connecticut xl_izq |
| 4 | 0100 | 4 | turquesa | connecticut xr_der |
| 5 | 0101 | 5 | azul | habano banda |
| 6 | 0110 | 6 | morado | habano capa |
| 7 | 0111 | 7 | marrón | habano xl_izq |
| 8 | 1000 | 8 | blanco | habano xr_der |
| **9** | **1001** | **9** | **rojo** | **hoja no reconocida / revisión humana** |
| 0, 10–15 | — | 9 | rojo | código no válido: también a la caja 9, para no perder la hoja |

> **Cambio del 2026-09-15: se quitó la media banda y se renumeró a 8 + 1.** El
> modelo final es de **cuatro calidades por variedad** (capa, banda, XL izquierdo,
> XR derecho), así que las dos cajas de media banda no se llenaban nunca. Roberto
> las había borrado en `Entorno1_Roberto.rdk`, pero sin renumerar y sobre una copia
> anterior al enganche del agente. Aquí se rehizo desde `Entorno1.rdk`, que sí lo
> tiene.
>
> **Cada clase conserva el color que tenía**: sólo cambia su número. Desaparecen el
> verde lima y el rosa, que eran las dos medias bandas.
>
> **Se cambió en los seis sitios a la vez, o en ninguno:** esta tabla ·
> `Construir_Entorno.py` · `Scripts/Control_Senales.py` ·
> `D:	esis-tabaco\scripts\celda_robodk.py` (`CLASES`) ·
> `agente_entorno1.py` (`CAJA_REVISION`) · `cola_entorno1.py`.

### Señales

En simulación, las señales son **parámetros de estación** de RoboDK: hacen de E/S virtuales
y se leen y escriben por el API (`RDK.setParam` / `RDK.getParam`, puerto 20500). Todas valen
`0` o `1`, salvo las dos informativas.

| Señal | Sentido | Significado |
|---|---|---|
| `TIPO_B0` … `TIPO_B3` | agente → robot | Los 4 bits del código. `B0` es el menos significativo. |
| `TIPO_VALIDO` | agente → robot | Los bits ya son estables y se pueden leer. |
| `HOJA_EN_CAMARA` | robot → agente | Hay una hoja parada bajo la cámara esperando clasificación. |
| `ORDEN_RECIBIDA` | robot → agente | Acuse de recibo. |
| `ROBOT_OCUPADO` | robot → agente | El robot está recogiendo o depositando. |
| `CAJA_DESTINO`, `ULTIMO_CODIGO` | robot → agente | Informativas: última caja y último código. |
| `CONTROL` | cualquiera → robot | Poner `STOP` detiene `Control_Senales` de forma ordenada. |

### Handshake de 4 fases

1. El robot pone `HOJA_EN_CAMARA = 1`.
2. El agente escribe `TIPO_B0..TIPO_B3` y **después** `TIPO_VALIDO = 1`.
3. El robot lee los bits y pone `ORDEN_RECIBIDA = 1` y `HOJA_EN_CAMARA = 0`.
4. El agente pone `TIPO_VALIDO = 0` y el robot responde `ORDEN_RECIBIDA = 0`.

El robot no pide otra orden hasta que el handshake anterior se ha cerrado, así que un
`TIPO_VALIDO` que se quede en 1 no se puede leer dos veces por error.

`CONTROL = STOP` solo se atiende en **puntos seguros** (esperando orden o entre hojas). Si
llega con el robot en movimiento, se termina la hoja en curso, se anota y después se para.

### Enviar una orden a mano (y referencia para el agente)

`Enviar_Senal.py` hace exactamente lo que hará el agente:

```
C:\RoboDK\Python-Embedded\python.exe Enviar_Senal.py 2
```

Espera a que haya una hoja bajo la cámara, escribe los bits del 2 (`0010`), levanta
`TIPO_VALIDO`, espera el acuse y lo baja. La función `enviar(tipo)` se puede importar
directamente desde el código del agente.

### Rutinas del robot

Son **programas nativos de RoboDK**, que se pueden generar como programa KUKA con el
postprocesador:

| Programa | Recorrido |
|---|---|
| `Recoger_Hoja` | Inicio → Recogida Aprox → Recogida → *Ventosa_Agarrar* → Recogida Aprox → Inicio |
| `Dejar_Caja_01` … `Dejar_Caja_09` | Inicio → Caja N Aprox → Caja N Deposito → *Ventosa_Soltar* → Caja N Aprox → Inicio |

La ventosa se acciona con dos llamadas a programa: `Ventosa_Agarrar` y `Ventosa_Soltar`. En
simulación, RoboDK ejecuta esos scripts y **espera a que terminen** antes del siguiente
movimiento (comprobado). En el robot real esas llamadas se sustituyen por la activación de la
salida de vacío.

### Para pasar al robot real

En simulación la decodificación de los 4 bits la hace `Control_Senales` (Python), porque los
programas nativos de RoboDK no tienen condicionales. En el KUKA real ese papel lo haría un
programa KRL que lea las 4 entradas digitales y llame a la rutina `Dejar_Caja_NN` que
corresponda. Las rutinas de movimiento generadas desde RoboDK se reutilizan tal cual.

## El agente enganchado (2026-09-11)

El que contesta los 4 bits es el agente clasificador de la tesis
(`D:\tesis-tabaco\scripts\agente_entorno1.py`). Corre en su propio Python —el que
tiene el modelo, OpenCV y scikit-learn— y habla con RoboDK por el API (puerto
20500), igual que `Enviar_Senal.py`.

    D:\tesis-tabaco\.venv\Scripts\python.exe D:\tesis-tabaco\scripts\agente_entorno1.py

Se arranca **antes** que `Control_Senales` y espera. Por cada hoja: lee
`HOJA_EN_CAMARA`, clasifica, escribe los bits y cierra el handshake.

### La numeración de las cajas manda aquí, no en el agente

El agente saca el número de recipiente de `celda_robodk.posiciones()`, y ese orden
se puso igual al de esta celda el 11/09. Antes iba por calidad (1 = capa,
4 = banda) y con eso **la hoja bien clasificada habría ido a la caja equivocada sin
que nada fallara en voz alta**. Si algún día se renumeran las cajas, se cambia en
un solo sitio: la tabla de códigos de este LEEME y `CLASES` en `celda_robodk.py`.

### Qué hoja está bajo la cámara: `HOJA_ID`

`Control_Senales` publica el parámetro **`HOJA_ID`** con el nombre del fichero de
la **foto real** de esa hoja — que no dice la clase. El agente lo busca en
`cola_hojas.csv` y clasifica esa fotografía.

**El nombre del objeto (`Hoja 7 - habano banda`) lleva la etiqueta verdadera y el
agente no lo lee nunca.** Leerla es «clasificar» sin clasificar. La etiqueta sólo
la usa `registro_clasificacion.csv` para puntuar después.

Y se clasifica la foto, y no la imagen de la cámara simulada, porque `Cam2D` con
licencia Free da 3,05 × 6,02 mm por píxel: de 5 a 10 veces más basto que la
resolución con la que el modelo mide, y la malla no tiene fibra ni manchas. La
cámara dispara el evento; la foto decide.

### La cola de hojas

Si existe `D:\tesis-tabaco\out\celda\entorno1\cola_hojas.csv`, el alimentador saca
las hojas **en ese orden**, cada una con su propia malla, y el ciclo termina al
agotarla. Si no existe, alimenta al azar con las diez hojas patrón, como antes. La
cola se genera con `python scripts/cola_entorno1.py`.

### Tres rutas que estaban clavadas a la otra máquina

Esta carpeta se sincroniza por OneDrive y en el otro equipo el usuario se llama
distinto. Estas estaban escritas a mano y aquí no existen:

- `Control_Senales.py` escribía la imagen y el registro en
  `C:\Users\LENOVO\Desktop\RoboDK`. Ahora la carpeta sale de `PATH_OPENSTATION`.
- `Construir_Entorno.py` buscaba los scripts en esa misma ruta. Ahora los busca al
  lado de sí mismo.
- Y la que rompía la celda entera en este equipo: **RoboDK tenía el intérprete de
  Python en `C:/RoboDK/Python-Embedded`** y aquí RoboDK está en `D:`. Con esa ruta
  muerta RoboDK **no ejecuta ningún script suyo, y no avisa**: `Control_Senales`
  moría al arrancar y `Recoger_Hoja` se quedaba colgado para siempre en la llamada
  a `Ventosa_Agarrar`. Corregido en `%APPDATA%\RoboDK\settings.ini` (respaldo:
  `settings_ANTES_20260911.ini`).

> **Cómo comprobarlo en diez segundos, si vuelve a pasar:** poner `BandaEstado` a
> cualquier cosa, ejecutar `Banda_OFF` y mirar si vuelve a `OFF`. Si no vuelve,
> RoboDK no está ejecutando sus scripts y el problema no está en la celda.

## Cámara

La cámara está **subida a Z ≈ 1992** (1225,5 mm sobre la cinta) para que la hoja entera
quepa en la imagen cuando se para bajo ella.

| Parámetro | Valor |
|---|---|
| Altura sobre la cinta | 1225,5 mm |
| Sensor | 640 × 480, apaisado |
| FOV (vertical, el que usa RoboDK) | 28,39° |
| Cobertura sobre la cinta | 827 mm a lo ancho × 620 mm a lo largo |
| Hoja más larga (habano banda) | 553,2 mm → unos 33 mm de margen por cada extremo |

Al subirla se mantuvieron el FOV y el formato de la imagen, así que ahora **a los lados se ven
también las guías azules** (la cinta negra mide 606 mm y la imagen abarca 827). Fue una
decisión explícita: la alternativa era girar el sensor a vertical (480×640), que habría
dejado solo cinta negra pero cambiando el formato de la imagen que recibe el agente.

`Construir_Entorno.py` calcula la altura a partir de la cobertura deseada (`ALTO_VISTA`) y
del FOV, y la estructura de la cámara sube con ella: sus cotas están referidas a `CAM_Z`.

## Cómo está modelada la banda

La banda es **geometría propia** (objetos), y el transporte lo mueve un script. Es el mismo
enfoque que usa RoboDK en su ejemplo `Example-06.e-Conveyor with 2 UR robots` y en
`Library/Macros/RunConveyor.py`.

Se hizo así, en vez de cargar un mecanismo de banda de la librería, por dos razones:

1. **No depende de la librería de RoboDK**, que cambia entre versiones.
2. **La estación queda con un solo mecanismo** (el robot). La licencia Free no admite guardar
   estaciones con más de un robot, así que este diseño se puede guardar sin licencia de pago.

## Las hojas

Lo que circula por la banda son las 10 hojas de `hojas_foto_impresa.rdk`: dos variedades
(*connecticut* y *habano*) por cinco clases (*banda*, *capa*, *media_banda*, *xl_izq*,
*xr_der*). Se importan como plantillas ocultas llamadas `Patron <clase>`, aparcadas a
Z = −3000, y el alimentador saca copias de ellas **al azar**.

El orden de `CLASES` fija la correspondencia tipo → caja, y aparece igual en
`Construir_Entorno.py`, `Scripts\Control_Senales.py` y `Scripts\Ciclo_Pick_Place.py`.

Dos detalles de la geometría de las hojas:

- **El origen no está en el centro**, sino en el centro de su lado corto. Por eso los scripts
  calculan el centro con el *bounding box* en vez de usar la posición del objeto.
- **Se giran al depositarlas.** Se recogen con el lado largo según +Y; al depositar, la
  herramienta gira (ángulo − 90°) para que la hoja entre alineada con el lado largo de la
  caja, que apunta radialmente. Sin ese giro no cabrían.

## El arco de cajas

Las 11 cajas van en un único arco de radio 1600, de 70° a 290° con paso de 22°, simétrico
respecto a 180° (justo detrás del robot). **Cada caja tiene un color distinto** para
identificarla a simple vista (ver la tabla de códigos); la 11, la de casos especiales, está en
un extremo (290°) y es roja. Los colores están en `COLORES_CAJAS`, en `Construir_Entorno.py`.

Con las cajas de 660 × 430 el arco puede llegar a 70°…290° sin tocar la línea
transportadora. El hueco mínimo entre cajas contiguas, en el radio interior del arco, es de
55 mm.

Para cambiar el número de cajas basta tocar `N_CAJAS`, `ANG_INI` y `ANG_FIN` en
`Construir_Entorno.py`; los targets y las rutinas `Dejar_Caja_NN` se generan solos.

## Por qué el robot no puede chocar con la estructura de la cámara

La columna y el brazo que sostienen la cámara están anclados en el **lado opuesto al robot**
(X ≥ 1420, con la columna en X = 2170). No hay ningún elemento estructural entre el robot y
la banda, así que el robot nunca tiene que atravesar el soporte para trabajar.

Además, todos los traslados entre la banda y las cajas pasan por la postura compacta `Inicio`
(J = [·, −117, 97,5, 0, 109,5, 0]).

## Programas (carpeta `Scripts`, embebidos también en el .rdk)

| Programa | Qué hace |
|---|---|
| **`Control_Senales`** | **Programa principal por señales**: alimenta, para bajo la cámara, espera el código de 4 bits y deposita. |
| `Ventosa_Agarrar` / `Ventosa_Soltar` | Macros de la ventosa, llamadas desde las rutinas del robot. |
| `Banda_ON` | Enciende la banda: arrastra las hojas hacia +Y, **para 3 s bajo la cámara** y retira las que llegan al final. |
| `Banda_OFF` | Apaga la banda; las hojas se quedan donde estén. |
| `Alimentar_Pieza` | Coloca **al azar** una hoja nueva al inicio de la línea. |
| `Ciclo_Pick_Place` | Demo automática sin agente ×10: cada hoja va a la caja de su clase. |
| `Reset_Celda` | Devuelve todo al estado inicial. |

La parada de 3 s es **tiempo de simulación**: si subes la velocidad de simulación, la espera
se acorta en proporción, igual que el avance de la banda.

Scripts fuera de la estación (se ejecutan con **Shift+S** en RoboDK o con el Python de RoboDK):

| Script | Para qué |
|---|---|
| `Enviar_Senal.py` | Envía un código de 4 bits al robot. Referencia para el agente. |
| `Construir_Entorno.py` | Reconstruye la celda entera. Idempotente. |
| `Verificar_Celda.py` | Cotas, que cada hoja entre entera en la cámara, giro del eje 1 y la secuencia de cada caja. |
| `Verificar_Encaje_Hojas.py` | Que cada hoja depositada quepa dentro de su caja. Ejecutar **después** de un ciclo. |

## Verificaciones

`Verificar_Celda.py` comprueba sobre la celda construida:

- **Cámara**: lee la altura real de la cámara y, para cada una de las 10 hojas, el margen que
  queda en la imagen por cada extremo y por cada lado cuando la hoja se para con su centro
  bajo la cámara.
- **Giro del eje 1**: barrido de −185° a +185° en la postura `Inicio`, buscando colisiones.
- **Recorridos**: las secuencias Inicio → recogida → caja → Inicio de las 11 cajas,
  segmento a segmento con `MoveJ_Test` a pasos de 1°.

Resultado con la cámara subida (2026-09-11): **las 10 hojas entran enteras**. El caso más
justo es habano banda, con 33,4 mm de margen por extremo; el más holgado, connecticut capa,
con 100,1 mm. A lo ancho sobran más de 268 mm por lado en todas. El eje 1 no choca en ningún
ángulo, las 11 secuencias no tienen colisiones, y la estructura de la cámara, ahora hasta
Z = 2242, tampoco interfiere.

Prueba funcional de `Control_Senales` con `Enviar_Senal.py` (2026-09-11):

| Hoja real | Código enviado | Resultado |
|---|---|---|
| habano banda (tipo 6) | 6 → `0110` | Caja 6 — ACIERTO |
| habano capa (tipo 7) | 11 → `1011` | Caja 11 — ESPECIAL |
| habano media_banda (tipo 8) | 0 → `0000` (no válido) | Caja 11 — ESPECIAL |

Después se pidió `CONTROL = STOP` con el robot volviendo de la caja 11. El programa terminó
la hoja, la anotó en el registro, volvió a `Inicio` y se detuvo. Handshake cerrado (todas
las señales en 0) y 0 colisiones.

## Detección de colisiones

Mapa `Default` activado. Se han excluido sólo los contactos previstos por diseño:
ventosa↔brida, cinta↔bastidor, bastidor↔bancada, y (en tiempo de ejecución) ventosa↔hoja,
hoja↔cinta y hoja↔caja. **Todo lo demás sí se comprueba**, incluidas todas las combinaciones
entre el robot y la estructura de la cámara.

## Notas de la API que costaron tiempo

- `AddTarget` hereda la configuración articular **actual** del robot, así que hay que fijar
  `setJoints` con la solución de IK antes de crear el target.
- `setParentStatic` conserva la pose **relativa**, no la absoluta; hay que corregir después
  con `setPoseAbs`.
- `Cam2D_Snapshot` sólo devuelve imagen si la vista se abrió con `Cam2D_Add` **en el mismo
  proceso**. Por eso `Control_Senales` reabre la cámara al arrancar. Un agente que quiera su
  propia imagen tiene que hacer lo mismo, o leer `camara_hoja_actual.png`.
- Un programa nativo que llama a un programa Python con `INSTRUCTION_CALL_PROGRAM` espera a
  que el script termine antes de seguir.
- El comando de captura de pantalla es `Snapshot`, no `Screenshot`.
- Para pilotar RoboDK en este equipo: `C:\RoboDK\Python-Embedded\python.exe` (trae `robodk`
  instalado). No llamar a ningún script `inspect.py`: tapa el módulo estándar y revienta el
  arranque de PySide2.
