# Results of a full run

Produced by `python run_all.py` on 2026-09-22 15:49.

Python 3.12.10, 9 s.

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

