# Resultados por superficie y campañas

Cuatro condiciones con 3.072 parches cada una. Cada condición compara U-Net y Transformer puro. La cantidad de tiles indica la extensión geográfica de procedencia, no un aumento del área total muestreada.

## 3 tiles · 1 campaña

- Superficie de procedencia: 20HLG, 20HLH y 20HMG.
- Campañas de entrenamiento: 2020/2021, 3.072 parches.
- Total: 3.072 parches por modelo.

| Modelo | mIoU (%) | F1 maní (%) | Precisión maní (%) | Recall maní (%) |
|---|---:|---:|---:|---:|
| U-Net | 50,81 | 5,14 | 3,02 | 17,42 |
| Transformer puro | 51,80 | 10,74 | 5,86 | 64,92 |

[Abrir mapas, curvas y matrices de este sector](index.html#3-tiles-1-campana).

## 3 tiles · 3 campañas

- Superficie de procedencia: 20HLG, 20HLH y 20HMG.
- Campañas de entrenamiento: 2018/2019, 2019/2020 y 2020/2021, 1.024 parches por campaña.
- Total: 3.072 parches por modelo.

| Modelo | mIoU (%) | F1 maní (%) | Precisión maní (%) | Recall maní (%) |
|---|---:|---:|---:|---:|
| U-Net | 53,20 | 10,73 | 5,92 | 57,36 |
| Transformer puro | 52,05 | 8,97 | 4,73 | 86,94 |

[Abrir mapas, curvas y matrices de este sector](index.html#3-tiles-3-campanas).

## 14 tiles · 1 campaña

- Superficie de procedencia: 13 tiles del TRAIN original más 20HLJ, incorporado para reforzar el núcleo manisero de Córdoba.
- Campañas de entrenamiento: 2020/2021, 3.072 parches.
- Total: 3.072 parches por modelo.

| Modelo | mIoU (%) | F1 maní (%) | Precisión maní (%) | Recall maní (%) |
|---|---:|---:|---:|---:|
| U-Net | 49,75 | 9,95 | 5,48 | 53,65 |
| Transformer puro | 52,53 | 6,53 | 3,54 | 41,12 |

[Abrir mapas, curvas y matrices de este sector](index.html#14-tiles-1-campana).

## 14 tiles · 3 campañas

- Superficie de procedencia: 13 tiles del TRAIN original más 20HLJ, incorporado para reforzar el núcleo manisero de Córdoba.
- Campañas de entrenamiento: 2018/2019, 2019/2020 y 2020/2021, 1.024 parches por campaña.
- Total: 3.072 parches por modelo.

| Modelo | mIoU (%) | F1 maní (%) | Precisión maní (%) | Recall maní (%) |
|---|---:|---:|---:|---:|
| U-Net | 54,57 | 7,71 | 4,31 | 36,50 |
| Transformer puro | 51,18 | 1,36 | 0,74 | 8,12 |

[Abrir mapas, curvas y matrices de este sector](index.html#14-tiles-3-campanas).

<!-- INFORME_TECNICO_PREVIO -->

## Análisis técnico y controles originales

# Ocho modelos: U-Net y Segmenter, cobertura y campañas — 22/10/2026

Ocho entrenamientos de 100 épocas y ocho evaluaciones terminados con código 0. Duración total: 14 h 40 min 23 s. Fin: 21/10 a las 23:50 de Argentina. No se reentrenó para este reporte.

## Comparación principal: 2021/2022 no vista

TEST 20HNJ/20HNH, exclusión común TRAIN/VAL, 177.822.530 píxeles válidos. Todas las métricas están en porcentaje. Checkpoints seleccionados por VAL.

| Condición | Modelo | Accuracy | mIoU | F1 macro | F1 maní | Precisión maní | Recall maní |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reducida_2020_2021 | unet | 81.20 | 50.81 | 61.25 | 5.14 | 3.02 | 17.42 |
| reducida_2020_2021 | segmenter | 81.34 | 51.80 | 62.85 | 10.74 | 5.86 | 64.92 |
| reducida_tres | unet | 82.06 | 53.20 | 63.86 | 10.73 | 5.92 | 57.36 |
| reducida_tres | segmenter | 81.59 | 52.05 | 62.77 | 8.97 | 4.73 | 86.94 |
| amplia_2020_2021 | unet | 78.18 | 49.75 | 61.21 | 9.95 | 5.48 | 53.65 |
| amplia_2020_2021 | segmenter | 82.50 | 52.53 | 62.74 | 6.53 | 3.54 | 41.12 |
| amplia_tres | unet | 84.09 | 54.57 | 64.34 | 7.71 | 4.31 | 36.50 |
| amplia_tres | segmenter | 81.89 | 51.18 | 60.96 | 1.36 | 0.74 | 8.12 |

## Lectura de los resultados

- La mayor mIoU observada corresponde a U-Net amplia con tres campañas (54,57%). Segmenter supera a U-Net con una campaña en ambas coberturas, pero U-Net lo supera con tres. No hay una ventaja uniforme de arquitectura.
- Agregar campañas mejora mIoU de U-Net en 2,39 puntos con cobertura reducida y 4,81 con amplia. En Segmenter cambia +0,26 y −1,34 puntos respectivamente. Ampliar cobertura no ayuda de manera uniforme.
- Maní no está resuelto: el mayor F1 ronda 10,74% (Segmenter reducida/una y U-Net reducida/tres son casi iguales). Las precisiones de todos los modelos son menores al 6%. El mayor recall, 86,94%, corresponde a Segmenter reducida/tres, pero con precisión 4,73%. Detectar más maní no implica un mapa más confiable.
- Para desempeño general, conservaría U-Net amplia/tres como referencia observada; antes de otro entrenamiento revisaría etiquetas, falsos positivos y desempeño por lote de maní. Las diferencias pequeñas no justifican elegir un ganador definitivo.

## Contrastes en puntos porcentuales

| Arquitectura | Contraste | Δ mIoU | Δ F1 maní |
| --- | --- | --- | --- |
| unet | Campañas: reducida | 2.39 | 5.59 |
| unet | Campañas: amplia | 4.81 | -2.24 |
| unet | Cobertura: una campaña | -1.06 | 4.81 |
| unet | Cobertura: tres campañas | 1.37 | -3.03 |
| unet | Interacción: efecto campañas amplia menos reducida | 2.42 | -7.83 |
| segmenter | Campañas: reducida | 0.26 | -1.77 |
| segmenter | Campañas: amplia | -1.34 | -5.17 |
| segmenter | Cobertura: una campaña | 0.73 | -4.22 |
| segmenter | Cobertura: tres campañas | -0.87 | -7.61 |
| segmenter | Interacción: efecto campañas amplia menos reducida | -1.60 | -3.40 |
| Segmenter menos U-Net | reducida_2020_2021 | 0.99 | 5.60 |
| Segmenter menos U-Net | reducida_tres | -1.15 | -1.76 |
| Segmenter menos U-Net | amplia_2020_2021 | 2.78 | -3.42 |
| Segmenter menos U-Net | amplia_tres | -3.38 | -6.35 |

## Controles y límites

Se verificaron ocho historiales de 100 épocas, códigos 0 de entrenamiento y evaluación, coincidencia de configuraciones por pares salvo arquitectura, firmas de selección y el mismo soporte de referencia. Cada modelo usa 3072 parches; la mezcla asigna 1024 por campaña y conserva ubicaciones dentro de cada cobertura. Pesos comunes: [0.0, 0.793, 1.002, 0.861, 3.922].

Fondo incluye otros cultivos y coberturas; no significa exclusivamente no agrícola. NoData de referencia se ignora. Las matrices muestran cuatro clases válidas y conservan una columna NoData predicha cuando hay errores de ese tipo; no se eliminan para mejorar métricas.

Las cifras principales suman las matrices de ambos tiles (no promedian sus F1). Los CSV distinguen mapa completo e independiente. La referencia es MNC, no verificación de campo independiente. En TRAIN se ignoraron positivos IPAVE que discrepan con MNC; VAL/TEST permanecieron originales. Cambiaron también los pesos comunes frente al piloto anterior: sin un control adicional no se puede atribuir una diferencia entre ensayos solo a IPAVE.

Una semilla por celda: sin pruebas de significación ni intervalos entre semillas. La cobertura describe un dominio geográfico concreto, no área total observada adicional. Se igualó el número de parches, no píxeles de maní ni número de píxeles válidos después de ignorar conflictos. La misma semilla no garantiza la misma secuencia de aumentos entre arquitecturas. Igual número de épocas no asegura igual convergencia; no hubo búsqueda de hiperparámetros por arquitectura. TEST se consultó en trabajos previos, por lo que el estudio es exploratorio.

Las panorámicas están reducidas para visualizar; detalles de maní elegidos por referencia y comunes a todos los modelos, no muestras aleatorias. Gris: excluido/NoData; verde: acierto; rojo: error. RGB de febrero. No se transfieren pesos ni TIFF al mirror.

## Archivos

- `index.html`: galería local con filtros por condición, campaña y tile.
- `metricas/todas_las_metricas.csv`: 64 filas, los dos criterios y todas las campañas.
- `metricas/contrastes_2021_2022_pp.csv`: efectos descriptivos de cobertura, campañas, arquitectura e interacción.
- `metricas/validacion_y_epocas.csv`: épocas de menor pérdida y métricas VAL.
- `metricas/resultados_completos.json`: matrices completas y métricas originales.
- `metricas/procedencia_checkpoints.json`: configuraciones y firmas de pesos.
- `sha256.json`: comprobación de integridad de esta copia.

Fuente física: `/mnt/yacy_3/prod_bis/ferreyra/dataset/factorial_unet_segmenter_mani_20261021_v1`. Figuras y métricas generadas desde resultados congelados; no se eligieron otros pesos con TEST.

## Nota adicional de validación

Los mínimos de pérdida VAL de Segmenter aparecen en las épocas 90–99; los de U-Net en 16–57. No se demostró convergencia equivalente. El F1 de maní de VAL fue 44,45–57,18%, frente a 1,36–10,74% en este TEST: hay una brecha importante que requiere revisión por lote.
