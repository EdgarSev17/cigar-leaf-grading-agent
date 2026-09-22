# HOJA DE CIFRAS AUTORIZADAS — proyecto de clasificacion de hoja de capa

Todas verificadas contra los registros del proyecto. Es la UNICA fuente de numeros
para redactar. Si una cifra no esta aqui, no se escribe.

## Que son los cuatro grados (definicion del oficio, dos preguntas)

El tecnico hace dos preguntas sobre cada hoja:
  A) la hoja entera, esta sana? (sin vena brotada, sin pelotas, sin sudado)
  B) hay un lado danado, y cual?

|                        | ambos lados limpios | un lado danado      |
|------------------------|---------------------|---------------------|
| hoja sana              | **capa**            | **XL izq / XR der** |
| vena, pelota o sudado  | **banda**           | (media banda)       |

CONVENCION CRITICA: el nombre de la clase es EL LADO BUENO, no el danado.
"XL izquierdo" = la mitad izquierda es la que sirve, o sea el dano esta a la derecha.

La casilla "media banda" (hoja con anomalia general Y un lado danado) se elimino del
sistema por decision del equipo: el sistema separa cuatro grados.

Zona util = la parte de la hoja que se aprovecha; excluye la base, la punta y la vena
central.

## Datos

- Conjunto de entrenamiento: 627 hojas, fotografiadas en tres capturas distintas
  (dos de Connecticut, una de Habano). Para el GRADO se usan 527 (Connecticut 220,
  Habano 307) tras eliminar media banda; para la VARIEDAD se usan las 627.
- Reparto de grados en esas 527: capa 238, xr_der 109, banda 91, xl_izq 89.
  Clase mayoritaria = capa, 45,2 %.
- Rasgos medidos: 69 para el grado, 17 para la variedad.
- Lote independiente A (Connecticut): 112 hojas. capa 71, XL izq 22, XR der 19.
  NO contiene ninguna banda. Clase mayoritaria 63,4 % (71 de 112).
- Lote independiente B (Habano): 218 imagenes; 1 no es hoja y se excluye; 217 con
  etiqueta, que corresponden a 211 hojas distintas. El arbol se evalua sobre 214
  hojas. Reparto por imagen: capa 84, XL izq 59, banda 42, XR der 32.
  Clase mayoritaria: 38,7 % por imagen (84 de 217); 39,3 % por hoja (84 de 214).
  Las respuestas del sistema quedaron guardadas ANTES de que existiera etiqueta.

## Lineas base, remedidas todas con la MISMA medida (la corregida)

| modelo unico de 4 grados | dentro de sesion | lote A independiente | conf. media | con umbral 0,90 firma | acierta ahi | mal clasificadas que se cuelan |
|---|---|---|---|---|---|---|
| rasgos + logistica       | 87,5 % | 77,9 % [74,1-80,4] | 0,914 | 75,0 % | 91,7 % | 7 |
| arboles por gradiente    | 89,2 % | 78,7 % [75,9-82,1] | 0,961 | 87,5 % | 82,7 % | 17 |
| ConvNeXt-Tiny (384 px)   | 94,9 % | 40,8 % [30,4-50,0] | --    | --     | --     | -- |

- Los corchetes son el RANGO minimo-maximo observado, NO intervalos de confianza:
  15 reentrenos con el 85 % de las hojas para los rasgos, 6 semillas completas para
  la red convolucional. (El paper anterior los llamaba IC 95 %: es un error.)
- Modelo pleno (entrenado con las 527 enteras): rasgos 76,8 %, arboles 79,5 %.
- La fila de la red convolucional no depende del detector de dano: mira pixeles.

## Sistema de dos etapas (el clasificador anterior, plano)

- Variedad, validacion cruzada agrupada por hoja: 99,2 % (logistica), 99,3 % (arboles).
  Fuera: 111 de 112 en el lote A; 217 de 217 en el lote B.
- Grado, validacion cruzada: Connecticut 93,1 % [91,4-94,5]; Habano 88,6 % [87,0-91,5].
- Grado en el lote A: 95 de 112 (84,8 %) con la medida vieja del dano;
  92 de 112 (82,1 %) con la medida corregida.
- Grado en el lote B: 58 de 217 (26,7 %) con la medida vieja;
  68 de 217 (31,3 %) con la medida corregida. XL izquierdo: 2 de 59, luego 1 de 59.
- Abstencion con umbral 0,90:
  - lote A: decide 73 de 112 (65,2 %) y acierta 69 (94,5 %).
  - lote B: decide el 38,7 % y acierta el 32,1 % de lo que decide.
- Dentro del propio lote B (entrenando y examinando dentro de el, validacion cruzada
  que mantiene juntas las fotos de una misma hoja): 74,3 %.
- Transferencia entre capturas: entrenamiento -> lote B 26,7 %; lote B -> entrenamiento
  29,3 %. Las dos por debajo de su clase mayoritaria.
- Normalizar cada rasgo contra la mediana y la desviacion absoluta mediana de su
  propia captura: 26,7 % -> 47,5 %, sin coste dentro.

## El clasificador en arbol (tres decisiones encadenadas)

Cada nodo usa SOLO los rasgos que nombra su regla del oficio, y es UN modelo para las
dos variedades.

| nodo | pregunta | rasgos | acierto del nodo |
|---|---|---|---|
| 1 | es banda? | sudado y color | 84,1 % |
| 2 | es capa o lateral? | forma del dano, sudado, cocientes | 67,2 % |
| 3 | el lado bueno es izquierdo o derecho? | solo asimetrias | 73,9 % |

- Con los 69 rasgos a la vez, el nodo 3 baja a 49,6 %.
- Acierto fuera, con la captura del examen dejada FUERA del entrenamiento:
  - lote A (Connecticut): 95 de 112 = 84,8 % (85,7 % si se entrena solo con las dos
    capturas de agosto; 78,6 % si se entrena solo con Connecticut).
  - lote B (Habano): 122 de 214 hojas = 57,0 %, frente al 39,3 % de la mayoritaria.
- Por grado en el lote B: capa 65 %, banda 49 %, XL izquierdo 46 %, XR derecho 66 %.
- Por grado en el lote A: capa 92 %, XL izquierdo 68 %, XR derecho 79 %.
- Curva de abstencion del arbol, 1.956 decisiones de los dos lotes:
  sin umbral 66,6 %; umbral 0,70 decide 49,4 % y acierta 74,5 %;
  umbral 0,80 decide 29,1 % y acierta 76,8 %; umbral 0,90 decide 13,5 % y acierta 79,5 %.
  Es monotona: a mas umbral, mas acierto.
- Tramos de confianza (mismas 1.956 decisiones): 0,00-0,50 acierta 44,4 %;
  0,50-0,60 54,9 %; 0,60-0,70 75,0 %; 0,70-0,80 71,2 %; 0,80-0,90 74,5 %;
  0,90-1,00 79,5 %.
- Ya cableado en la tuberia, con umbral 0,70: lote A decide 65 de 112 (58,0 %) y
  acierta 61 (93,8 %), 96 de 112 en total (85,7 %); lote B decide 100 de 217 (46,1 %)
  y acierta 81 (81,0 %), 137 de 217 en total (63,1 %). AVISO: el modelo cableado se
  entrena con todo, incluidos esos lotes, asi que esas dos cifras son optimistas; las
  honestas son 84,8 % y 57,0 %.

## El defecto de medida que se corrigio

El detector de dano contaba como agujeros las arrugas y los pliegues de la hoja. Se
corrige exigiendo que la marca sobreviva a un adelgazamiento de tres pixeles: un hilo
desaparece, un hueco no. No mira el color.

Agujeros por hoja (mediana), antes -> despues:

| variedad | capa | banda | XL izq | XR der |
|---|---|---|---|---|
| Connecticut | 1,0 -> 0,0 | 4,5 -> 0,0 | 7,0 -> 2,5 | 1,5 -> 1,0 |
| Habano      | 25,5 -> 1,0 | 34,5 -> 9,0 | 20,5 -> 2,0 | 13,5 -> 1,5 |

- Las roturas y los mordiscos de borde quedan identicos: el filtro solo toca agujeros.
- Asimetria del dano en XL izquierdo de Habano, entre las hojas que conservan dano en
  zona util: -0,26 -> -0,61.
- Decidir el lado solo por el signo de la asimetria: Connecticut 74,5 % -> 78,7 %;
  Habano 59,2 % -> 61,4 %.
- Precio: hojas de XL izquierdo de Habano con algun dano medible en zona util,
  97 % -> 50 %.
- Efecto en el acierto: al clasificador plano le da +4,6 puntos en el lote B
  (26,7 -> 31,3) y le quita 2,7 en el lote A (84,8 -> 82,1); al arbol le da menos de
  un punto (55,8 -> 57,0).
- En el lote B, antes del arreglo: mediana de 23 agujeros por imagen y 0,778 pulg2 de
  dano en zona util, contra 0,304 pulg2 en las hojas de Habano del entrenamiento.
- **CORREGIDO el 20/09 por la tarde, a peticion del experto. La escala NO es la causa y
  la frase anterior era falsa.** Decia que la camara estuvo mas lejos en el lote B
  comparando 46,6 px de cinta contra 79-92. Comprobado en los registros de la propia
  tuberia: `ancho_cinta` mediana **46,6 px en el lote B y 44,7 px en el lote A**,
  medidos igual y por el mismo codigo. Son la misma escala, y el lote A acierta 84,8 %.
  El 79-92 estaba medido a otra escala de trabajo. Ademas: las dos tandas son la misma
  camara y la misma resolucion (6048x8064, 48,8 Mpx); y las hojas miden lo mismo en
  pulgadas (largo mediano 16,94 en Habano de agosto contra 16,38 en el lote B, 3 % de
  diferencia), cosa imposible si la escala estuviera mal. Tabaco.md §80.8 ya lo decia:
  "LA ESCALA NO ESTABA MAL", la calibracion reproduce el ancla dentro del 5 %.
  ESTA FRASE NUNCA ENTRO AL ARTICULO: solo estaba en estas notas.

- **Lo que SI se desplaza entre capturas, medido con control (20/09).** Globalmente el
  lote A se desplaza tanto como el B (mediana 0,41 contra 0,50 MAD sobre los 69 rasgos),
  asi que "hubo desplazamiento" tampoco explica nada por si solo. Lo que cambia es QUE
  rasgos se mueven: en Habano son la **textura y el brillo de la hoja** (`hoja_textura`
  3,09 MAD, `hoja_L_med` 2,22, `hoja_L_desv` 2,20, `cobertura` 2,02); en Connecticut son
  el **tamano y las manchas** (`largo_pulg` 2,69, `mancha_izq` 2,00). Es decir, esas
  hojas de Habano SE VEN distintas, no estan fotografiadas a otra escala. Cuadra con las
  arrugas: 25 marcas por hoja contra 1 en Connecticut.

- **Normalizar SOLO la apariencia** (31 rasgos de color, brillo y textura) contra la
  mediana y la MAD de su propia tanda: lote B **57,0 -> 58,4 %** (kappa 0,402 -> 0,419)
  y lote A **sin cambio, 84,8 %**. Normalizar TODO en cambio hunde el lote A a 55,4 %.
  No necesita fotos nuevas, pero exige clasificar por tandas y no hoja a hoja.

- **Kappa de Cohen del sistema de hoy:** lote A **0,705**, lote B **0,402**. Acierto
  promediado por grado: 79,6 % y 56,4 %. Lote B sin banda (3 grados, como el A): 59,0 %.

## La mezcla de efectos en Connecticut

- Todas las hojas de grado capa de Connecticut vienen de una sola captura (127 hojas)
  y todas las demas calidades de Connecticut de otra (banda 22, XL izq 36, XR der 35).
- Por tanto, en Connecticut separar capa del resto se puede hacer sin mirar la hoja.
  La validacion cruzada agrupa por hoja, no por captura: el 93,1 % de Connecticut esta
  inflado y la cifra que debe citarse es la del lote independiente.
- El Habano no tiene ese problema: sus cuatro grados se fotografiaron en la misma
  captura, y su 88,6 % dentro de sesion es honesto.

## Tamaño del efecto (d de Cohen) — medido el 20/09 por la tarde

Definición usada, la del catedrático: d = diferencia entre desviación conjunta.
Guión `guiones/d_cohen.py`, salida `medidas/tamano_del_efecto.txt`.

- Árboles por gradiente contra rasgos+logística, lote A, los 15 reentrenos con el 85 %
  (emparejados, las mismas submuestras): diferencia **+0,83 puntos**, desviación conjunta
  1,97 puntos, **d = 0,42**; emparejada dz = 0,34. El gradiente gana en **9 de 15**.
  (Los 1,7 puntos del paper son DENTRO del conjunto de entrenamiento, donde no hay
  dispersión medida: ahí no se puede dar d.)
- Tres decisiones encadenadas contra el sistema plano de dos etapas, acierto por hoja 0/1
  y desviación de Bernoulli: lote A 84,8 contra 82,1 % -> **d = 0,07**;
  lote B 57,0 contra 31,3 % -> **d = 0,54**.
- Contra el grado más frecuente del lote: A **d = 0,50**; B **d = 0,36**.
- De la red convolucional NO se da d: sus corchetes son un rango entre 6 semillas, no una
  desviación; calcularla con rango/4 sale 10,1 y sería inventarse la dispersión.

## Rechazo por fracción fija — lo que pidió el catedrático, medido el 20/09 por la tarde

Mismas 1.956 decisiones del árbol (dos exámenes fuera de sesión x 6 semillas). Se aparta la
fracción MENOS segura, en vez de fijar un umbral. Guión `guiones/rechazo_10_20.py`,
salida `medidas/rechazo_10_20.txt`.

| regla | decide | acierta ahí | deriva | umbral de corte |
|---|---|---|---|---|
| clasificar todas       | 100 % | 66,6 % | 0 %  | —     |
| apartar el 10 % menos seguro | 90,0 % | **70,2 %** | 10 % | 0,442 |
| apartar el 20 % menos seguro | 80,0 % | **72,0 %** | 20 % | 0,516 |

- Por lote: A 84,8 -> 88,1 % (10 %) y 87,7 % (20 %); B 57,0 -> **60,7 %** y **63,7 %**.
- En los dos cortes hay 6 decisiones con el valor exacto del umbral (son las 6 semillas de
  una misma hoja), así que el corte cae limpio entre hojas.
- Equivale a la curva por umbral, pero leída al revés: para llegar al 74,5 % hay que
  apartar la mitad.

## Techo de la tarea

- Sobre 44 hojas, una persona experta clasificando POR FOTOGRAFIA coincide con la
  etiqueta de produccion en 29 casos (65,9 %; kappa 0,534).
- Sobre esas mismas hojas: el sistema coincide el 72,1 % y la persona el 67,4 %.
- Es el techo de clasificar por fotografia, no el techo del oficio: la vena brotada es
  relieve y no color, y el tecnico la nota con el tacto.
- El acuerdo entre dos tecnicos sobre las hojas fisicas no esta medido.

## Celda robotica y coste

- Robot KUKA IONTEC ultra KR 120 R2700, celda simulada en RoboDK.
- 8 bandejas de destino (4 grados x 2 variedades) + 1 de revision humana = 9.
  Antes eran 11, con dos de media banda; se renumeraron al eliminar ese grado.
- 0 colisiones en los 10 programas generados.
- Coste: cargar el modelo 2,2 s una sola vez. Medir y decidir una hoja completa,
  entre 3,1 y 4,1 s por fotografia en las corridas de los dos lotes, en CPU y sin
  tarjeta grafica.

## CORRECCION (20/09, noche): el aprendizaje por correccion es +2,9, no +3,8

El +3,8 [+2,7, +4,8] que circulaba sale de `out/PAPER_CURVA/curva_ANTES_69RASGOS.json`,
es decir de ANTES de que el modelo pasara a 69 rasgos. La medida vigente esta en
`out/PAPER_CURVA/curva.json` y en `CURVA.txt` (19/09):

- examen fijo de 55 hojas, 40 repeticiones, correcciones de la misma captura
- k=0 **82,3 %** [81,1-83,4] -> k=56 **85,1 %** [84,0-86,3]
- **GANANCIA +2,9 puntos, IC 95 % [+1,8, +4,0]**, mejora en **31 de 40** repeticiones
- por clase: capa +4,0 · XR derecho +2,5 · **XL izquierdo -0,5** (no mejora)

Corregido en el paper de seis paginas y en el de ocho.
