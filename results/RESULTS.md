# Results of a full run

Produced by `python run_all.py` on 2026-09-22 15:27.

Python 3.12.10, 441 s.

## The article's figures, checked

| figure | article | this run | |
|---|---|---|---|
| accuracy on the independent batch | 84.8 | 84.8 | matches |
| Cohen's kappa | 0.705 | 0.705 | matches |
| at threshold 0.60, leaves decided | 70.5 | 70.5 | matches |
| at threshold 0.60, accuracy on those | 91.1 | 91.1 | matches |
| wrapper | 91.5 | 91.5 | matches |
| XL left | 68.2 | 68.2 | matches |
| XR right | 78.9 | 78.9 | matches |
| error setting aside 10 % | 11.9 | 11.9 | matches |
| error setting aside 30 % | 8.9 | 8.9 | matches |

## Full output

### The article's protocol

```
   (las dos medidas vienen en el mismo CSV, una sola pasada)

======================================================================
INDEPENDENT BATCH: 672 decisions  (6 retrainings x 112 leaves)
======================================================================

   accuracy, deciding every leaf    84.8 %      kappa 0.705

   TABLE IV -- how many it decides and how well it does on them
      rule                            decides   accuracy    defers
      decides every leaf               100.0 %      84.8 %      0.0 %
      confidence threshold 0.60         70.5 %      91.1 %     29.5 %
      confidence threshold 0.70         46.4 %      90.4 %     53.6 %
      confidence threshold 0.80         30.4 %      91.2 %     69.6 %
      sets aside the least sure 10 %    90.0 %      88.1 %     10.0 %
      sets aside the least sure 20 %    80.1 %      87.7 %     19.9 %

   TABLE III -- by grade
      wrapper     390 of 426     91.5 %
      XL left      90 of 132     68.2 %
      XR right     90 of 114     78.9 %

   FIGURE 2 -- error by the fraction set aside
      sets aside  0 %   error  15.2 %
      sets aside 10 %   error  11.9 %
      sets aside 20 %   error  12.3 %
      sets aside 30 %   error   8.9 %

   The article's operating point is the 0.60 threshold.
```

### Cross-validated figures

```
hojas: 627   rasgos de hoja: 64   de fondo: 9
reparto: {np.str_('banda'): 91, np.str_('capa'): 238, np.str_('media_banda'): 100, np.str_('xl_izq'): 89, np.str_('xr_der'): 109}
omitidas por decision del experto (PROGRESO 50): 9
semillas de particion: 15

==============================================================================
LAS CIFRAS, CON 15 SEMILLAS DE PARTICION
==============================================================================

                                            n    base            exactitud
   ------------------------------------------------------------------------
   las cinco calidades                    627   38.0 %  80.6 %  [78.9 - 81.5]
      balanceada                                        75.7 %  [73.9 - 76.9]
      CONTROL: solo el fondo                            65.8 %  [64.3 - 67.1]
                                        supera al fondo por 14.8 puntos

   cinco calidades, solo Connecticut      242   52.5 %  84.0 %  [82.2 - 85.5]
      balanceada                                        71.1 %  [67.1 - 73.8]
      CONTROL: solo el fondo                            76.3 %  [75.2 - 78.5]
                                        supera al fondo por 7.7 puntos

   cinco calidades, solo Habano           385   28.8 %  84.5 %  [82.3 - 86.2]
      balanceada                                        83.2 %  [81.1 - 85.2]
      CONTROL: solo el fondo                            61.1 %  [59.0 - 63.4]
                                        supera al fondo por 23.4 puntos

   el par XL izq / XR der                 198   55.1 %  90.7 %  [88.4 - 92.4]
      balanceada                                        90.7 %  [88.3 - 92.3]
      CONTROL: solo el fondo                            66.0 %  [61.6 - 70.2]
                                        supera al fondo por 24.7 puntos

   Recordatorio de PROGRESO 45.2: el rango de arriba NO es un intervalo de
   confianza estadistico, es lo que se mueve el numero al cambiar la
   particion con los MISMOS datos. Cualquier comparacion antes/despues
   menor que esa anchura no significa nada.
```

### One photograph, end to end

```

==========================================================================
  20260911_152039111_iOS.heic
==========================================================================

  VARIEDAD   Connecticut      el modelo dice 1.00  ->  el lote que pasa con esa confianza acierta 100 %
  CALIDAD    CAPA             el modelo dice 1.00  ->  el lote que pasa con esa confianza acierta 100 %

  DECISION   ACEPTAR: confianza 1.00, por encima del piso de 0.90
             (de todo lo que pasa ese piso, acierta el 100 % medido)
             segunda opcion: BANDA (0.00)

  LO QUE SE VE EN LA HOJA   (medidas, comprobables mirandola)
     - no hay defectos en la zona util: ni izquierda ni derecha
     - 1 agujeros y 1 roturas en toda la hoja
     - el mayor mide 0.15 pulg2 (0.441 pulg de diametro), esta en base y a 0.991 pulg de la base
     - el mordisco de borde se mete 0.42 pulg por la izquierda y 0.30 por la derecha
     -> es la medida de la regla de la chaveta: no importa cuanto ocupa sino cuanto entra (PROGRESO 43)
     - mancha verde (sudada) en el 5.4% de la hoja: 0.42 pulg2 a la izquierda, 5.49 a la derecha
     - la hoja mide 16.1 x 9.3 pulgadas
     - a una pulgada de la punta conserva el 39.3% de su anchura (Habano ~36 %, Connecticut ~43 %; PROGRESO 50.5)

  LO QUE PESO EN LA DECISION   (la aritmetica del modelo, CAPA contra BANDA)
     a favor    3.58   lo clara que es la hoja
     a favor    1.73   lo rugosa que se ve
     en contra  1.23   que mitad tiene mas zonas oscuras
     a favor    1.19   lo pareja que es su luz
     a favor    1.00   cuanta hoja hay mas oscura que el resto
     en contra  0.70   que mitad gana en el tono verde-rojo
     (el numero es cuanto mueve la balanza, no pulgadas ni por ciento)

  COMO SE MIDIO
     escala: dada (mediana de la carpeta, como la tuberia)   (43.8 px de cinta, 0.025342 pulg/px)
     giro aplicado: -2.67

  LO QUE NO SE PUEDE PROMETER
     Este modelo se entreno con UNA sesion de fotos en la que la clase
     y la hora del dia van juntas (PROGRESO 54.2). Las cifras de
     acierto de arriba son de validacion cruzada DENTRO de esa sesion.
     Que se repitan con otra camara, otra mesa u otro dia NO esta
     medido, y con los datos de hoy no se puede medir.
```

