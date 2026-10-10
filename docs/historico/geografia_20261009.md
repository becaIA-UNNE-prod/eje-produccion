> Registro histórico consolidado el 10/10/2026. Los estados, PID, comandos y splits describen su fecha original; no son instrucciones vigentes. Leer las guías en `../guias/`.

# Auditoría de coordenadas y área efectivamente usada

Verificación del 9 de octubre de 2026 en `floryacy:~/cordoba_ia_unne`.

## Resultado

No se detectaron discrepancias en las comprobaciones siguientes.

- 306 cabeceras de composiciones mensuales: CRS, transformada y dimensiones coinciden con la grilla de su tile.
- 34 imágenes originales B04 contrastadas con sus compuestos, una por tile y campaña disponible para 2018/2019 y 2020/2021.
- Las 51 máscaras se contrastaron con la grilla de sus seis composiciones.
- Las esquinas usadas por el mapa local coinciden con las esquinas transformadas a WGS84 de los compuestos de las tres campañas.
- 153 parches Y de muestra comparados píxel a píxel con las máscaras remapeadas.
- 51 parches X de muestra comparados con las seis composiciones y la normalización /10.000 float16.
- Para cada uno de los 51 datasets se reconstruyó la selección de ventanas y se comparó su cantidad con los arrays guardados.

El control de X/Y es por muestras, no un checksum de cada píxel de todos los datasets. La campaña 2019/2020 se verificó desde compuestos, máscaras y arrays; no se incluyeron sus escenas originales en el control de cabeceras de origen.

## Por qué aparecen provincias vecinas

El flujo actual no aplica un polígono provincial de Córdoba. `00_descargar_stac.py` utiliza un bounding box de búsqueda `[-66.93, -35.33, -60.69, -29.79]`; esa caja selecciona escenas que la intersectan, pero no recorta sus píxeles. Los archivos se descargan con la extensión completa del tile.

`03c_generar_dataset_cordoba_f16.py` genera ventanas de 256 × 256 sobre la grilla entera y solo descarta las que tienen todas las etiquetas remapeadas en 0. Las máscaras provienen del mapa nacional de INTA, no de un recorte al límite provincial. Por tanto, las zonas vecinas con etiquetas válidas también pueden aportar parches y pérdida durante el entrenamiento.

La inclusión de San Luis, Santa Fe y Buenos Aires no se explica por invertir latitud/longitud ni por un desplazamiento del mapa: las comprobaciones de grillas y muestras permiten contrastar esa hipótesis. Si la intención era entrenar exclusivamente dentro de Córdoba, la preparación actual no impone ese requisito. Corregir ese alcance exigiría una máscara provincial y regenerar los datasets antes de una nueva corrida; esta auditoría no modificó el entrenamiento ni sus datos.

## Ejemplos comprobados en provincias vecinas

Los puntos siguientes están dentro de las imágenes de diciembre de 2020 y dentro de una ventana conservada del dataset 2020/2021. La etiqueta es la clase remapeada en el punto de la localidad; un centro urbano puede ser Fondo. Valores espectrales distintos de cero acreditan imagen en ese punto, no ausencia de nubes.

| Tile | Localidad | Provincia | Coordenadas (lat, lon) | Clase en el punto | B02/B03/B04/B08 |
|---|---|---|---|---|---|
| 19HGC | Zanjitas | San Luis | -33.80512, -66.41623 | Fondo | [844, 1064, 1268, 2576] |
| 19HGC | Varela | San Luis | -34.12360, -66.46398 | Fondo | [1042, 1434, 1754, 2898] |
| 19HGC | Travesía | San Luis | -33.78366, -66.06908 | Fondo | [468, 722, 824, 2128] |
| 19HGC | Talleres | San Luis | -33.46667, -66.63333 | Fondo | [975, 1268, 1484, 2664] |
| 20HLG | Realicó | La Pampa | -35.03821, -64.24631 | NoData | [1011, 1488, 1826, 3540] |
| 20HLG | Rancul | La Pampa | -35.06777, -64.68422 | NoData | [2014, 2316, 2688, 3484] |
| 20HLG | Parera | La Pampa | -35.14676, -64.50231 | NoData | [1498, 1710, 1920, 2662] |
| 20HMG | General Villegas | Buenos Aires | -35.03264, -63.01484 | NoData | [2152, 2540, 2868, 3286] |
| 20HMG | Intendente Alvear | La Pampa | -35.23468, -63.59209 | NoData | [1226, 1672, 1960, 2720] |
| 20HMG | Banderaló | Buenos Aires | -35.01482, -63.37568 | NoData | [1366, 1704, 1980, 3054] |
| 20HNG | Florentino Ameghino | Buenos Aires | -34.84405, -62.46701 | NoData | [1542, 1868, 2176, 3012] |
| 20HNG | Tres Algarrobos | Buenos Aires | -35.19560, -62.77423 | NoData | [3412, 3708, 3972, 4768] |
| 20HPH | Firmat | Santa Fe | -33.45937, -61.48320 | NoData | [2820, 2832, 3072, 3796] |
| 20HPH | Villa Cañás | Santa Fe | -34.00406, -61.60685 | NoData | [1874, 2480, 2962, 3628] |
| 20HPH | María Teresa | Santa Fe | -34.01224, -61.91436 | NoData | [1854, 2176, 2648, 3472] |
| 20HPH | Wheelwright | Santa Fe | -33.79156, -61.21089 | NoData | [1438, 1710, 2182, 2712] |

## Huella de los parches

El mapa local (`artifacts/archive/legacy_root/figuras/mapa_tiles_multicampana.html`, copia del mirror local) conserva los contornos completos y permite activar la **huella de ventanas conservadas** para cada campaña. Esa huella muestra parches enteros, incluidos los píxeles NoData que puede haber dentro de un parche válido. No representa una máscara de cultivos ni el límite de Córdoba.

La extensión completa es de 109,8 km por lado. Las ventanas conservadas cubren una grilla interior de hasta 107,52 km por lado; el generador deja bordes sin parches y descarta ventanas totalmente NoData. La selección puede variar por campaña.

Evidencia completa: auditoria_geografia_tiles.json (`artifacts/archive/legacy_root/inventarios/auditoria_geografia_tiles.json`, copia del mirror local). Polígonos de ventanas: huellas_parches_multicampana.geojson (`artifacts/archive/legacy_root/inventarios/huellas_parches_multicampana.geojson`, copia del mirror local).


---

# Imágenes, cobertura geográfica y red del experimento multicampaña

Documento verificado el 9 de octubre de 2026 contra los rasters, la configuración guardada y el código de `floryacy:~/cordoba_ia_unne`.

## 1. Qué imágenes recibe el modelo

Se usaron imágenes ópticas Sentinel-2, organizadas en tiles de la grilla MGRS. Cada tile es una escena geográfica grande; no corresponde exclusivamente a una ciudad ni sigue límites municipales o provinciales. Los rasters inspeccionados tienen 10.980 × 10.980 píxeles a 10 m: aproximadamente 109,8 × 109,8 km por raster completo. La cobertura útil puede ser menor por zonas sin datos. Hay solapamiento entre tiles vecinos.

Para cada campaña y tile se reúnen seis composiciones mensuales, de diciembre a mayo. Cada mes aporta cuatro bandas, en este orden:

| Banda | Señal | Resolución utilizada |
|---|---|---|
| B02 | Azul | 10 m |
| B03 | Verde | 10 m |
| B04 | Rojo | 10 m |
| B08 | Infrarrojo cercano, útil para distinguir vegetación | 10 m |

La entrada es **24 canales = 6 meses × 4 bandas**. Se concatenan diciembre, enero, febrero, marzo, abril y mayo, manteniendo el orden de bandas dentro de cada mes. No se incorporan NDVI, radar ni bandas de onda corta a esta corrida. La reflectancia se divide por 10.000; el dataset guarda X como float16 y las etiquetas como uint8.

| Campaña | Ventana de imágenes | Uso |
|---|---|---|
| 2018/2019 | Diciembre de 2018 a mayo de 2019 | Entrenamiento, validación y prueba espacial |
| 2019/2020 | Diciembre de 2019 a mayo de 2020 | Entrenamiento, validación y prueba espacial |
| 2020/2021 | Diciembre de 2020 a mayo de 2021 | Entrenamiento, validación y prueba espacial |
| 2021/2022 | Diciembre de 2021 a mayo de 2022 | Evaluación temporal nueva, en preparación al escribir este documento |

2017/2018 quedó fuera por diferencias de leyenda en las etiquetas. La prueba 2021/2022 usa el modelo congelado, sin reentrenar ni elegir pesos en función de ese año.

### Cómo se construye cada imagen mensual

Los archivos terminan en `_median.tif`, pero el nombre se conserva por compatibilidad: **no garantiza una mediana matemática**. En 2018/2019, 2020/2021 y la preparación 2021/2022, `02d_composites_scl.py` ordena escenas por nubosidad, usa SCL para excluir nubes, sombras, cirros y datos defectuosos, expande bordes de nubes y aplica un filtro adicional de banda azul. Cada píxel toma la primera observación válida; si no encuentra una despejada, intenta el menor azul disponible. Las cuatro bandas se toman de la misma observación elegida para ese píxel.

2019/2020 reutiliza los compuestos existentes `composites_dicmay_2019_2020`, del flujo anterior de selección de escenas y relleno de huecos. No se debe afirmar que las tres campañas tienen exactamente el mismo enmascarado de nubes. Una diferencia de resultados entre años también puede reflejar calidad del compuesto, además de cambios en los cultivos.

Las fechas exactas pueden variar por tile y mes; una composición puede integrar varias fechas. El anexo conserva las fechas disponibles. El archivo `fechas_usadas.json` de 2019/2020 solo contiene mayo: no alcanza para reconstruir fechas exactas de diciembre–abril y no se inventan aquí.

> **Auditoría posterior:** se verificaron las 306 composiciones, las 51 máscaras y muestras de los 51 datasets. No se detectó un desplazamiento geográfico. Hay parches usados fuera de Córdoba porque no se aplicó un recorte provincial. Ver [evidencia y alcance](geografia_20261009.md) y las capas «Parches usados» del mapa (`artifacts/archive/legacy_root/figuras/mapa_tiles_multicampana.html`, copia del mirror local).

## 2. Qué lugares cubre cada tile

Las localidades de la tabla se comprobaron transformando sus coordenadas de GeoNames al CRS del raster y verificando que el punto caiga dentro de su extensión. Son referencias geográficas, no una lista exhaustiva. Se verifica el punto de la localidad, no la totalidad de su municipio ni la disponibilidad de imagen en cada píxel.

**La selección se llama “Córdoba”, pero la extensión completa de los rasters incluye provincias vecinas.** Una figura de tile completo puede mostrar sectores de esas provincias.

| Tile | División | Localidades dentro de la extensión del raster | Provincias de esas localidades |
|---|---|---|---|
| `20HLK` | TEST | Río Tercero, Alta Gracia, Embalse, Merlo, Almafuerte, Santa Rosa de Calamuchita | Córdoba, San Luis |
| `20HLJ` | TEST | Río Cuarto, Sampacho, Las Higueras, San Basilio, Elena, Achiras | Córdoba |
| `20HKJ` | VAL | La Toma, Tilisarao, Concarán, San Francisco del Monte de Oro, Naschel, Virorco | San Luis |
| `20HMH` | VAL | Laboulaye, Adelia María, General Levalle, Viamonte, La Cesira, Huanchillas | Córdoba |
| `19HGC` | TRAIN | Zanjitas, Varela, Travesía, Talleres, Santa Teresa, Santa Rosa | San Luis |
| `20HLG` | TRAIN | Huinca Renancó, Realicó, Villa Huidobro, Rancul, Parera, Del Campillo | Córdoba, La Pampa |
| `20HLH` | TRAIN | Vicuña Mackenna, Coronel Moldes, Adelia María, Del Campillo, San Basilio, Villa Valeria | Córdoba |
| `20HMJ` | TRAIN | La Carlota, General Cabrera, Ucacha, Alejandro Roca, Las Perdices, Etruria | Córdoba |
| `20HMK` | TRAIN | Villa María, Villa Nueva, Río Segundo, Pilar, Oncativo, Oliva | Córdoba |
| `20HNK` | TRAIN | Las Varillas, San Jorge, Pozo del Molle, Noetinger, Alicia, Morrison | Córdoba, Santa Fe |
| `20JML` | TRAIN | Río Segundo, Arroyito, Pilar, Villa del Rosario, Montecristo, Río Primero | Córdoba |
| `20HKH` | TRAIN | Villa Mercedes, Justo Daract, Villa Salles, Villa Reynolds, Travesía, Sovén | San Luis |
| `20HMG` | TRAIN | General Villegas, Intendente Alvear, Jovita, Serrano, Hipólito Bouchard, Bernardo Larroudé | Buenos Aires, La Pampa, Córdoba |
| `20HNG` | TRAIN | Florentino Ameghino, Juan Bautista Alberdi, Tres Algarrobos, Germania, Volta, Villa Saboya | Buenos Aires |
| `20HPG` | TRAIN | Junín, Lincoln, Los Toldos, Vedia, Juan Bautista Alberdi, Alfredo Demarchi | Buenos Aires |
| `20HPH` | TRAIN | Colón, Rojas, Firmat, Villa Cañás, María Teresa, Wheelwright | Buenos Aires, Santa Fe |
| `20JNL` | TRAIN | San Francisco, Brinkmann, Freyre, Balnearia, Devoto, Porteña | Córdoba |

### Las cuatro zonas del año nunca visto

- **20HLJ (TEST):** Río Cuarto, Las Higueras, Sampacho y zonas agrícolas próximas.
- **20HLK (TEST):** Río Tercero, Alta Gracia, Embalse, Santa Rosa de Calamuchita y sectores serranos; también incluye Merlo, San Luis.
- **20HMH (VAL espacial):** Laboulaye, General Levalle, Adelia María y alrededores.
- **20HKJ (VAL espacial):** La Toma, Tilisarao, Concarán y otras localidades de San Luis.

“VAL” y “TEST” conservan la designación geográfica del experimento original. En 2021/2022 los cuatro se usan para evaluar el modelo congelado; ese año no participa en la selección del checkpoint.

El split asigna 13 tiles a TRAIN, 2 a VAL y 2 a TEST, repetidos en las tres campañas. Los tiles de TEST no se entregan al optimizador. Sin embargo, **separar por identificador de tile no demuestra ausencia de fuga espacial**: las escenas Sentinel-2 se solapan en los bordes. No se aplicó aquí una auditoría de exclusión por coordenadas de esos solapamientos. Los resultados deben interpretarse con esa limitación; la prueba 2021/2022 sí agrega un año que el modelo nunca vio.

## 3. Etiquetas y tamaños del dataset

Las etiquetas provienen de los Mapas Nacionales de Cultivos de INTA, reproyectados por vecino más cercano a la grilla de las imágenes y remapeados a estas clases comunes:

| ID | Clase | Tratamiento |
|---|---|---|
| 0 | NoData | Se ignora en la pérdida y como referencia de evaluación |
| 1 | Fondo | Agrupa categorías documentadas que no son los tres cultivos objetivo |
| 2 | Maíz | Cultivo objetivo |
| 3 | Soja | Cultivo objetivo |
| 4 | Maní | Cultivo objetivo |

Fondo no significa únicamente suelo desnudo. Incluye otras categorías de la leyenda. El mapa de referencia es una clasificación de INTA; medir concordancia con él no equivale a una validación independiente de campo.

En el mapa oficial 2020/2021 se detectó el código 20 fuera de la leyenda: por decisión explícita se remapea a NoData. Esta excepción es exclusiva de esa campaña; 2021/2022 detiene la preparación si aparecen códigos desconocidos en el área evaluada.

Se entrenó con parches de 256 × 256 píxeles, aproximadamente 2,56 × 2,56 km. Los totales combinados son 66.731 parches TRAIN, 10.468 VAL y 10.074 TEST. La inferencia de los mapas completos usa bloques de 512 × 512, aproximadamente 5,12 × 5,12 km, con batch 4. La red es convolucional y admite ambos tamaños divisibles por 16, pero las métricas de mapa completo y de parches no son intercambiables.

## 4. Estructura de la U-Net entrenada

La clase efectiva es `SimpleUNet(24, 5)` de `utils/model.py`: **31.055.877 parámetros**. Algunos comentarios históricos del archivo mencionan 28 canales y siete meses; la corrida actual usa 24 canales y seis meses, según la configuración y los pesos.

La red clasifica cada píxel, no la imagen entera. Primero reduce la resolución para reunir contexto y después la recupera para producir un mapa de clases. Las conexiones laterales (“skip connections”) llevan detalle del encoder al decoder y ayudan a conservar bordes de lotes.

| Etapa | Canales | Tamaño espacial para un parche de entrenamiento |
|---|---:|---:|
| Entrada | 24 | 256 × 256 |
| Encoder 1 | 64 | 256 × 256 |
| Encoder 2 | 128 | 128 × 128 |
| Encoder 3 | 256 | 64 × 64 |
| Encoder 4 | 512 | 32 × 32 |
| Cuello de botella | 1024 | 16 × 16 |
| Decoder 1 | 512 | 32 × 32 |
| Decoder 2 | 256 | 64 × 64 |
| Decoder 3 | 128 | 128 × 128 |
| Decoder 4 | 64 | 256 × 256 |
| Convolución final 1 × 1 | 5 | 256 × 256 |

Cada bloque `DoubleConv` contiene dos convoluciones 3 × 3 con padding 1, cada una seguida de BatchNorm y ReLU. Cuatro MaxPool 2 × 2 reducen la resolución. El decoder usa convoluciones transpuestas 2 × 2 con stride 2; concatena el skip del mismo nivel y aplica otro bloque DoubleConv. Hay Dropout2d de 0,2 en el cuarto encoder y primer decoder, y 0,3 en el cuello de botella.

La salida son cinco valores (“logits”) por píxel. La clase se elige por argmax. Aunque NoData se ignora como referencia, existe un canal de salida para esa clase; predecir 0 donde la referencia es válida cuenta como error en los mapas evaluados.

Los meses entran apilados como canales: las convoluciones pueden aprender combinaciones entre bandas y meses, pero esta U-Net no tiene un módulo temporal explícito, recurrente ni Transformer. Las posiciones de canal conservan un calendario fijo, lo que permite aprender patrones estacionales de vegetación.

### Aprendizaje y checkpoint elegido

- Batch 32; semilla 42; Adam con learning rate inicial 0,0001 y weight decay 0,001.
- Pérdida combinada de entropía cruzada ponderada y Dice, ignorando NoData. Pesos de clase: `[0, 0.765, 1.057, 0.855, 6.139]`; el peso alto de maní compensa su menor frecuencia.
- `ReduceLROnPlateau`: reduce la tasa a la mitad cuando la pérdida de VAL deja de mejorar durante 10 épocas.
- Hasta 100 épocas; parada temprana con paciencia 30. Terminó en la época 90; se eligió el mejor modelo de la época 60 según pérdida de VAL.
- Se usó precisión mixta en GPU. La lectura por bloques, las cachés y la precarga paralela aceleran la entrega de datos sin cambiar la arquitectura.

### Modelos de comparación

La inferencia compara esta U-Net multicampaña con una U-Net y una FCRTransUNet previas, entrenadas en 2019/2020. FCRTransUNet incorpora Transformer en el cuello de botella, atención en los skips y módulos adicionales de realce de características. El código identifica esa implementación como una adaptación; no se debe presentarla como reproducción exacta de un artículo.

Comparar las tres redes sobre los mismos tiles y las mismas entradas permite contrastar sus predicciones. No aísla exclusivamente la arquitectura: la U-Net nueva también vio más campañas. Las curvas de validación originales tampoco constituyen una comparación controlada porque corresponden a distintos conjuntos de aprendizaje.

## 5. Cómo leer las imágenes de resultados

Las comparaciones por tile muestran una composición RGB de febrero (rojo B04, verde B03 y azul B02), la etiqueta INTA, las predicciones y el mapa de errores. Ese RGB es una vista ilustrativa: la predicción utilizó los seis meses y las cuatro bandas, incluido el infrarrojo cercano.

Las figuras son vistas reducidas para visualización; los GeoTIFF de predicción conservan la grilla a 10 m. Las matrices de confusión muestran qué clases se confunden; F1 por clase ayuda a detectar si el buen resultado global oculta problemas con maní. Las curvas de pérdida y accuracy describen el entrenamiento. El experimento 2021/2022 permitirá medir generalización temporal, pendiente de finalizar al escribir este documento.

Resultados locales:

- Figuras y métricas multicampaña: ubicación de copia histórica `resultados_multicampana_20261009/` (consultar el índice de artefactos para su ubicación actual)
- Evaluación temporal 2021/2022: ubicación de copia histórica `resultados_ano_no_visto_2021_2022/` (no se acredita aquí su ubicación actual)

## 6. Fuentes y trazabilidad

- Código remoto: `utils/model.py`, `utils/multicampana.py`, `src/1_train_cordoba.py`, `src/02d_composites_scl.py`, `src/postprocesar_multicampana.py` y `src/evaluar_ano_no_visto.py`.
- Configuración: `/mnt/yacy_3/prod_bis/ferreyra/dataset/exp_multicampana_1819_1920_2021_unet_bloques_20261007T094404/config_multicampana.json`.
- Compuestos: `/mnt/yacy_3/prod_bis/ferreyra/composites_2018_2019_scl`, `composites_dicmay_2019_2020`, `composites_2020_2021_scl`; para el año nuevo, `composites_2021_2022_scl`.
- Localidades: [GeoNames, descarga Argentina](https://download.geonames.org/export/dump/AR.zip), consultada el 09/10/2026. Sus puntos se cruzaron con las extensiones reales, en sus respectivos CRS; los nombres no se dedujeron del código del tile. Atribución GeoNames, licencia CC BY.
- Referencia 2021/2022: [INTA en Zenodo](https://zenodo.org/records/8284081), archivo `MNC_verano2022.tif` y leyenda de verano del informe (página física 7 del PDF).

## Anexo A. Extensiones geográficas de referencia

Esquinas de los rasters de diciembre de 2018, transformadas a longitud/latitud WGS84. Se muestran como rango envolvente aproximado: el raster proyectado no es un rectángulo perfecto en coordenadas geográficas. Esto describe la grilla, no una máscara de cobertura útil ni un límite provincial.

| Tile | Longitud oeste–este | Latitud sur–norte |
|---|---|---|
| `19HGC` | -66.8494 a -65.6312 | -34.4100 a -33.3943 |
| `20HLG` | -65.1999 a -63.9806 | -35.3272 a -34.3218 |
| `20HLH` | -65.1760 a -63.9703 | -34.4254 a -33.4204 |
| `20HMJ` | -64.0773 a -62.8949 | -33.5277 a -32.5328 |
| `20HMK` | -64.0663 a -62.8960 | -32.6255 a -31.6306 |
| `20HNK` | -63.0002 a -61.8298 | -32.6256 a -31.6298 |
| `20JML` | -64.0559 a -62.8970 | -31.7232 a -30.7283 |
| `20HKH` | -66.2630 a -65.0459 | -34.4118 a -33.3971 |
| `20HMG` | -64.1007 a -62.8926 | -35.3312 a -34.3363 |
| `20HNG` | -63.0002 a -61.7922 | -35.3313 a -34.3353 |
| `20HPG` | -61.9129 a -60.6924 | -35.3263 a -34.3199 |
| `20HPH` | -61.9242 a -60.7174 | -34.4245 a -33.4185 |
| `20JNL` | -63.0002 a -61.8413 | -31.7232 a -30.7275 |
| `20HKJ` | -66.2288 a -65.0252 | -33.5109 a -32.4968 |
| `20HMH` | -64.0887 a -62.8938 | -34.4293 a -33.4344 |
| `20HLK` | -65.1314 a -63.9511 | -32.6219 a -31.6176 |
| `20HLJ` | -65.1532 a -63.9605 | -33.5240 a -32.5193 |

## Anexo B. Fechas de escenas disponibles

Fechas en formato AAAA-MM-DD, tomadas de los manifiestos remotos. En 2021/2022 son escenas seleccionadas para la preparación, no una afirmación de que la evaluación ya terminó. En 2019/2020 solo consta mayo en el manifiesto disponible.

| Campaña | Tile | Mes | Fechas de escenas |
|---|---|---|---|
| 2018/2019 | `19HGC` | 2018-12 | 2018-12-14 |
| 2018/2019 | `19HGC` | 2019-01 | 2019-01-18 |
| 2018/2019 | `19HGC` | 2019-02 | 2019-02-27 |
| 2018/2019 | `19HGC` | 2019-03 | 2019-03-14 |
| 2018/2019 | `19HGC` | 2019-04 | 2019-04-28 |
| 2018/2019 | `19HGC` | 2019-05 | 2019-05-13 |
| 2018/2019 | `20HKH` | 2018-12 | 2018-12-14 |
| 2018/2019 | `20HKH` | 2019-01 | 2019-01-18, 2019-01-28 |
| 2018/2019 | `20HKH` | 2019-02 | 2019-02-14, 2019-02-27 |
| 2018/2019 | `20HKH` | 2019-03 | 2019-03-14 |
| 2018/2019 | `20HKH` | 2019-04 | 2019-04-28 |
| 2018/2019 | `20HKH` | 2019-05 | 2019-05-13 |
| 2018/2019 | `20HKJ` | 2018-12 | 2018-12-24 |
| 2018/2019 | `20HKJ` | 2019-01 | 2019-01-15, 2019-01-20, 2019-01-18, 2019-01-08 |
| 2018/2019 | `20HKJ` | 2019-02 | 2019-02-14, 2019-02-17 |
| 2018/2019 | `20HKJ` | 2019-03 | 2019-03-04, 2019-03-24 |
| 2018/2019 | `20HKJ` | 2019-04 | 2019-04-10, 2019-04-05, 2019-04-03, 2019-04-28 |
| 2018/2019 | `20HKJ` | 2019-05 | 2019-05-13 |
| 2018/2019 | `20HLG` | 2018-12 | 2018-12-21 |
| 2018/2019 | `20HLG` | 2019-01 | 2019-01-03, 2019-01-05 |
| 2018/2019 | `20HLG` | 2019-02 | 2019-02-17, 2019-02-27, 2019-02-14 |
| 2018/2019 | `20HLG` | 2019-03 | 2019-03-14, 2019-03-26, 2019-03-01 |
| 2018/2019 | `20HLG` | 2019-04 | 2019-04-28, 2019-04-05 |
| 2018/2019 | `20HLG` | 2019-05 | 2019-05-15 |
| 2018/2019 | `20HLH` | 2018-12 | 2018-12-14, 2018-12-21 |
| 2018/2019 | `20HLH` | 2019-01 | 2019-01-28, 2019-01-05 |
| 2018/2019 | `20HLH` | 2019-02 | 2019-02-07, 2019-02-17, 2019-02-27, 2019-02-14 |
| 2018/2019 | `20HLH` | 2019-03 | 2019-03-14, 2019-03-01, 2019-03-21, 2019-03-26 |
| 2018/2019 | `20HLH` | 2019-04 | 2019-04-28, 2019-04-05 |
| 2018/2019 | `20HLH` | 2019-05 | 2019-05-10 |
| 2018/2019 | `20HLJ` | 2018-12 | 2018-12-24, 2018-12-14, 2018-12-04, 2018-12-21 |
| 2018/2019 | `20HLJ` | 2019-01 | 2019-01-18, 2019-01-08, 2019-01-28, 2019-01-15, 2019-01-20, 2019-01-05, 2019-01-30 |
| 2018/2019 | `20HLJ` | 2019-02 | 2019-02-17, 2019-02-14 |
| 2018/2019 | `20HLJ` | 2019-03 | 2019-03-24, 2019-03-04, 2019-03-21, 2019-03-16, 2019-03-01 |
| 2018/2019 | `20HLJ` | 2019-04 | 2019-04-28, 2019-04-10 |
| 2018/2019 | `20HLJ` | 2019-05 | 2019-05-28, 2019-05-13, 2019-05-23, 2019-05-10 |
| 2018/2019 | `20HLK` | 2018-12 | 2018-12-14, 2018-12-21 |
| 2018/2019 | `20HLK` | 2019-01 | 2019-01-18, 2019-01-08, 2019-01-28, 2019-01-20, 2019-01-15, 2019-01-25 |
| 2018/2019 | `20HLK` | 2019-02 | 2019-02-17, 2019-02-19 |
| 2018/2019 | `20HLK` | 2019-03 | 2019-03-16, 2019-03-24, 2019-03-14, 2019-03-04 |
| 2018/2019 | `20HLK` | 2019-04 | 2019-04-03, 2019-04-13, 2019-04-08, 2019-04-10 |
| 2018/2019 | `20HLK` | 2019-05 | 2019-05-13, 2019-05-23, 2019-05-15 |
| 2018/2019 | `20HMG` | 2018-12 | 2018-12-23, 2018-12-16 |
| 2018/2019 | `20HMG` | 2019-01 | 2019-01-05 |
| 2018/2019 | `20HMG` | 2019-02 | 2019-02-16, 2019-02-06, 2019-02-01, 2019-02-04, 2019-02-19 |
| 2018/2019 | `20HMG` | 2019-03 | 2019-03-13, 2019-03-18, 2019-03-01, 2019-03-26 |
| 2018/2019 | `20HMG` | 2019-04 | 2019-04-05 |
| 2018/2019 | `20HMG` | 2019-05 | 2019-05-15 |
| 2018/2019 | `20HMH` | 2018-12 | 2018-12-23, 2018-12-21, 2018-12-01 |
| 2018/2019 | `20HMH` | 2019-01 | 2019-01-05, 2019-01-20, 2019-01-30 |
| 2018/2019 | `20HMH` | 2019-02 | 2019-02-26, 2019-02-06, 2019-02-04, 2019-02-19, 2019-02-09 |
| 2018/2019 | `20HMH` | 2019-03 | 2019-03-18, 2019-03-01, 2019-03-26, 2019-03-21 |
| 2018/2019 | `20HMH` | 2019-04 | 2019-04-02, 2019-04-05, 2019-04-10 |
| 2018/2019 | `20HMH` | 2019-05 | 2019-05-10 |
| 2018/2019 | `20HMJ` | 2018-12 | 2018-12-31, 2018-12-26 |
| 2018/2019 | `20HMJ` | 2019-01 | 2019-01-05, 2019-01-20, 2019-01-15, 2019-01-30 |
| 2018/2019 | `20HMJ` | 2019-02 | 2019-02-19, 2019-02-24 |
| 2018/2019 | `20HMJ` | 2019-03 | 2019-03-01, 2019-03-26, 2019-03-16, 2019-03-21 |
| 2018/2019 | `20HMJ` | 2019-04 | 2019-04-10, 2019-04-05, 2019-04-20 |
| 2018/2019 | `20HMJ` | 2019-05 | 2019-05-10 |
| 2018/2019 | `20HMK` | 2018-12 | 2018-12-09, 2018-12-31, 2018-12-21 |
| 2018/2019 | `20HMK` | 2019-01 | 2019-01-15, 2019-01-05, 2019-01-20, 2019-01-25 |
| 2018/2019 | `20HMK` | 2019-02 | 2019-02-07, 2019-02-19 |
| 2018/2019 | `20HMK` | 2019-03 | 2019-03-24, 2019-03-16, 2019-03-21, 2019-03-26, 2019-03-01 |
| 2018/2019 | `20HMK` | 2019-04 | 2019-04-10, 2019-04-05 |
| 2018/2019 | `20HMK` | 2019-05 | 2019-05-13, 2019-05-10, 2019-05-15 |
| 2018/2019 | `20HNG` | 2018-12 | 2018-12-23 |
| 2018/2019 | `20HNG` | 2019-01 | 2019-01-05, 2019-01-12, 2019-01-22, 2019-01-07 |
| 2018/2019 | `20HNG` | 2019-02 | 2019-02-16 |
| 2018/2019 | `20HNG` | 2019-03 | 2019-03-18 |
| 2018/2019 | `20HNG` | 2019-04 | 2019-04-07 |
| 2018/2019 | `20HNG` | 2019-05 | 2019-05-12 |
| 2018/2019 | `20HNK` | 2018-12 | 2018-12-03, 2018-12-23, 2018-12-26 |
| 2018/2019 | `20HNK` | 2019-01 | 2019-01-02, 2019-01-27, 2019-01-07, 2019-01-05, 2019-01-20, 2019-01-25, 2019-01-15, 2019-01-30 |
| 2018/2019 | `20HNK` | 2019-02 | 2019-02-21, 2019-02-06, 2019-02-19 |
| 2018/2019 | `20HNK` | 2019-03 | 2019-03-18, 2019-03-23, 2019-03-26, 2019-03-01, 2019-03-31, 2019-03-11, 2019-03-16 |
| 2018/2019 | `20HNK` | 2019-04 | 2019-04-07, 2019-04-12, 2019-04-05, 2019-04-25, 2019-04-10 |
| 2018/2019 | `20HNK` | 2019-05 | 2019-05-15 |
| 2018/2019 | `20HPG` | 2018-12 | 2018-12-10, 2018-12-30, 2018-12-08 |
| 2018/2019 | `20HPG` | 2019-01 | 2019-01-04, 2019-01-12, 2019-01-22, 2019-01-07 |
| 2018/2019 | `20HPG` | 2019-02 | 2019-02-18, 2019-02-16 |
| 2018/2019 | `20HPG` | 2019-03 | 2019-03-20, 2019-03-18 |
| 2018/2019 | `20HPG` | 2019-04 | 2019-04-07 |
| 2018/2019 | `20HPG` | 2019-05 | 2019-05-24, 2019-05-12 |
| 2018/2019 | `20HPH` | 2018-12 | 2018-12-23 |
| 2018/2019 | `20HPH` | 2019-01 | 2019-01-12, 2019-01-02, 2019-01-27, 2019-01-22 |
| 2018/2019 | `20HPH` | 2019-02 | 2019-02-16 |
| 2018/2019 | `20HPH` | 2019-03 | 2019-03-18 |
| 2018/2019 | `20HPH` | 2019-04 | 2019-04-07 |
| 2018/2019 | `20HPH` | 2019-05 | 2019-05-12 |
| 2018/2019 | `20JML` | 2018-12 | 2018-12-24, 2018-12-14, 2018-12-04, 2018-12-26, 2018-12-11, 2018-12-31 |
| 2018/2019 | `20JML` | 2019-01 | 2019-01-28, 2019-01-25, 2019-01-15, 2019-01-20, 2019-01-05 |
| 2018/2019 | `20JML` | 2019-02 | 2019-02-17, 2019-02-19 |
| 2018/2019 | `20JML` | 2019-03 | 2019-03-24, 2019-03-01, 2019-03-21, 2019-03-16, 2019-03-31 |
| 2018/2019 | `20JML` | 2019-04 | 2019-04-08, 2019-04-05, 2019-04-10 |
| 2018/2019 | `20JML` | 2019-05 | 2019-05-13, 2019-05-23, 2019-05-15, 2019-05-10, 2019-05-25 |
| 2018/2019 | `20JNL` | 2018-12 | 2018-12-13, 2018-12-26 |
| 2018/2019 | `20JNL` | 2019-01 | 2019-01-07, 2019-01-25, 2019-01-05 |
| 2018/2019 | `20JNL` | 2019-02 | 2019-02-21, 2019-02-06, 2019-02-19 |
| 2018/2019 | `20JNL` | 2019-03 | 2019-03-18, 2019-03-23, 2019-03-01, 2019-03-26, 2019-03-31 |
| 2018/2019 | `20JNL` | 2019-04 | 2019-04-12, 2019-04-10, 2019-04-05, 2019-04-25 |
| 2018/2019 | `20JNL` | 2019-05 | 2019-05-12, 2019-05-10, 2019-05-15 |
| 2019/2020 | `19HGC` | 2020-05 | 2020-05-07 |
| 2019/2020 | `20HKH` | 2020-05 | 2020-05-27 |
| 2019/2020 | `20HKJ` | 2020-05 | 2020-05-27 |
| 2019/2020 | `20HLG` | 2020-05 | 2020-05-09 |
| 2019/2020 | `20HLH` | 2020-05 | 2020-05-14 |
| 2019/2020 | `20HLJ` | 2020-05 | 2020-05-17, 2020-05-02, 2020-05-14 |
| 2019/2020 | `20HLK` | 2020-05 | 2020-05-27, 2020-05-07, 2020-05-17 |
| 2019/2020 | `20HMG` | 2020-05 | 2020-05-14 |
| 2019/2020 | `20HMH` | 2020-05 | 2020-05-14 |
| 2019/2020 | `20HMJ` | 2020-05 | 2020-05-14 |
| 2019/2020 | `20HMK` | 2020-05 | 2020-05-19 |
| 2019/2020 | `20HNG` | 2020-05 | 2020-05-06 |
| 2019/2020 | `20HNK` | 2020-05 | 2020-05-19 |
| 2019/2020 | `20HPG` | 2020-05 | 2020-05-26 |
| 2019/2020 | `20HPH` | 2020-05 | 2020-05-01 |
| 2019/2020 | `20JML` | 2020-05 | 2020-05-19 |
| 2019/2020 | `20JNL` | 2020-05 | 2020-05-19 |
| 2020/2021 | `19HGC` | 2020-12 | 2020-12-03 |
| 2020/2021 | `19HGC` | 2021-01 | 2021-01-17 |
| 2020/2021 | `19HGC` | 2021-02 | 2021-02-06 |
| 2020/2021 | `19HGC` | 2021-03 | 2021-03-23 |
| 2020/2021 | `19HGC` | 2021-04 | 2021-04-27 |
| 2020/2021 | `19HGC` | 2021-05 | 2021-05-27 |
| 2020/2021 | `20HKH` | 2020-12 | 2020-12-08 |
| 2020/2021 | `20HKH` | 2021-01 | 2021-01-22 |
| 2020/2021 | `20HKH` | 2021-02 | 2021-02-06 |
| 2020/2021 | `20HKH` | 2021-03 | 2021-03-23 |
| 2020/2021 | `20HKH` | 2021-04 | 2021-04-27 |
| 2020/2021 | `20HKH` | 2021-05 | 2021-05-27 |
| 2020/2021 | `20HKJ` | 2020-12 | 2020-12-08 |
| 2020/2021 | `20HKJ` | 2021-01 | 2021-01-22 |
| 2020/2021 | `20HKJ` | 2021-02 | 2021-02-06 |
| 2020/2021 | `20HKJ` | 2021-03 | 2021-03-10, 2021-03-23, 2021-03-28 |
| 2020/2021 | `20HKJ` | 2021-04 | 2021-04-27 |
| 2020/2021 | `20HKJ` | 2021-05 | 2021-05-27 |
| 2020/2021 | `20HLG` | 2020-12 | 2020-12-25 |
| 2020/2021 | `20HLG` | 2021-01 | 2021-01-24 |
| 2020/2021 | `20HLG` | 2021-02 | 2021-02-23 |
| 2020/2021 | `20HLG` | 2021-03 | 2021-03-30 |
| 2020/2021 | `20HLG` | 2021-04 | 2021-04-14 |
| 2020/2021 | `20HLG` | 2021-05 | 2021-05-04 |
| 2020/2021 | `20HLH` | 2020-12 | 2020-12-30 |
| 2020/2021 | `20HLH` | 2021-01 | 2021-01-17, 2021-01-09 |
| 2020/2021 | `20HLH` | 2021-02 | 2021-02-18 |
| 2020/2021 | `20HLH` | 2021-03 | 2021-03-30 |
| 2020/2021 | `20HLH` | 2021-04 | 2021-04-14 |
| 2020/2021 | `20HLH` | 2021-05 | 2021-05-29 |
| 2020/2021 | `20HLJ` | 2020-12 | 2020-12-23, 2020-12-10 |
| 2020/2021 | `20HLJ` | 2021-01 | 2021-01-22, 2021-01-17, 2021-01-09 |
| 2020/2021 | `20HLJ` | 2021-02 | 2021-02-06, 2021-02-18 |
| 2020/2021 | `20HLJ` | 2021-03 | 2021-03-23, 2021-03-30 |
| 2020/2021 | `20HLJ` | 2021-04 | 2021-04-27, 2021-04-14 |
| 2020/2021 | `20HLJ` | 2021-05 | 2021-05-12, 2021-05-24 |
| 2020/2021 | `20HLK` | 2020-12 | 2020-12-23, 2020-12-10 |
| 2020/2021 | `20HLK` | 2021-01 | 2021-01-17, 2021-01-07, 2021-01-29, 2021-01-19 |
| 2020/2021 | `20HLK` | 2021-02 | 2021-02-06, 2021-02-18 |
| 2020/2021 | `20HLK` | 2021-03 | 2021-03-28, 2021-03-30 |
| 2020/2021 | `20HLK` | 2021-04 | 2021-04-27, 2021-04-12, 2021-04-14 |
| 2020/2021 | `20HLK` | 2021-05 | 2021-05-27, 2021-05-14, 2021-05-24 |
| 2020/2021 | `20HMG` | 2020-12 | 2020-12-25 |
| 2020/2021 | `20HMG` | 2021-01 | 2021-01-19 |
| 2020/2021 | `20HMG` | 2021-02 | 2021-02-23 |
| 2020/2021 | `20HMG` | 2021-03 | 2021-03-30 |
| 2020/2021 | `20HMG` | 2021-04 | 2021-04-11, 2021-04-29 |
| 2020/2021 | `20HMG` | 2021-05 | 2021-05-24 |
| 2020/2021 | `20HMH` | 2020-12 | 2020-12-30 |
| 2020/2021 | `20HMH` | 2021-01 | 2021-01-09 |
| 2020/2021 | `20HMH` | 2021-02 | 2021-02-23 |
| 2020/2021 | `20HMH` | 2021-03 | 2021-03-30 |
| 2020/2021 | `20HMH` | 2021-04 | 2021-04-14 |
| 2020/2021 | `20HMH` | 2021-05 | 2021-05-01, 2021-05-11, 2021-05-29, 2021-05-24 |
| 2020/2021 | `20HMJ` | 2020-12 | 2020-12-05 |
| 2020/2021 | `20HMJ` | 2021-01 | 2021-01-09 |
| 2020/2021 | `20HMJ` | 2021-02 | 2021-02-18 |
| 2020/2021 | `20HMJ` | 2021-03 | 2021-03-30 |
| 2020/2021 | `20HMJ` | 2021-04 | 2021-04-29 |
| 2020/2021 | `20HMJ` | 2021-05 | 2021-05-24 |
| 2020/2021 | `20HMK` | 2020-12 | 2020-12-18, 2020-12-10 |
| 2020/2021 | `20HMK` | 2021-01 | 2021-01-09, 2021-01-04 |
| 2020/2021 | `20HMK` | 2021-02 | 2021-02-18 |
| 2020/2021 | `20HMK` | 2021-03 | 2021-03-28, 2021-03-30 |
| 2020/2021 | `20HMK` | 2021-04 | 2021-04-22, 2021-04-29 |
| 2020/2021 | `20HMK` | 2021-05 | 2021-05-02, 2021-05-24 |
| 2020/2021 | `20HNG` | 2020-12 | 2020-12-07 |
| 2020/2021 | `20HNG` | 2021-01 | 2021-01-19, 2021-01-21 |
| 2020/2021 | `20HNG` | 2021-02 | 2021-02-05 |
| 2020/2021 | `20HNG` | 2021-03 | 2021-03-30, 2021-03-07, 2021-03-12 |
| 2020/2021 | `20HNG` | 2021-04 | 2021-04-26 |
| 2020/2021 | `20HNG` | 2021-05 | 2021-05-24, 2021-05-04, 2021-05-06 |
| 2020/2021 | `20HNK` | 2020-12 | 2020-12-10 |
| 2020/2021 | `20HNK` | 2021-01 | 2021-01-21, 2021-01-09, 2021-01-04 |
| 2020/2021 | `20HNK` | 2021-02 | 2021-02-18 |
| 2020/2021 | `20HNK` | 2021-03 | 2021-03-30 |
| 2020/2021 | `20HNK` | 2021-04 | 2021-04-14 |
| 2020/2021 | `20HNK` | 2021-05 | 2021-05-24 |
| 2020/2021 | `20HPG` | 2020-12 | 2020-12-22 |
| 2020/2021 | `20HPG` | 2021-01 | 2021-01-21 |
| 2020/2021 | `20HPG` | 2021-02 | 2021-02-20 |
| 2020/2021 | `20HPG` | 2021-03 | 2021-03-07 |
| 2020/2021 | `20HPG` | 2021-04 | 2021-04-13, 2021-04-28, 2021-04-16 |
| 2020/2021 | `20HPG` | 2021-05 | 2021-05-06 |
| 2020/2021 | `20HPH` | 2020-12 | 2020-12-22 |
| 2020/2021 | `20HPH` | 2021-01 | 2021-01-21 |
| 2020/2021 | `20HPH` | 2021-02 | 2021-02-05 |
| 2020/2021 | `20HPH` | 2021-03 | 2021-03-07, 2021-03-22 |
| 2020/2021 | `20HPH` | 2021-04 | 2021-04-16 |
| 2020/2021 | `20HPH` | 2021-05 | 2021-05-01 |
| 2020/2021 | `20JML` | 2020-12 | 2020-12-10 |
| 2020/2021 | `20JML` | 2021-01 | 2021-01-07, 2021-01-04, 2021-01-19 |
| 2020/2021 | `20JML` | 2021-02 | 2021-02-26, 2021-02-06, 2021-02-18 |
| 2020/2021 | `20JML` | 2021-03 | 2021-03-30 |
| 2020/2021 | `20JML` | 2021-04 | 2021-04-14 |
| 2020/2021 | `20JML` | 2021-05 | 2021-05-14 |
| 2020/2021 | `20JNL` | 2020-12 | 2020-12-10 |
| 2020/2021 | `20JNL` | 2021-01 | 2021-01-19 |
| 2020/2021 | `20JNL` | 2021-02 | 2021-02-18 |
| 2020/2021 | `20JNL` | 2021-03 | 2021-03-10 |
| 2020/2021 | `20JNL` | 2021-04 | 2021-04-29 |
| 2020/2021 | `20JNL` | 2021-05 | 2021-05-24 |
| 2021/2022 | `20HKJ` | 2021-12 | 2021-12-18 |
| 2021/2022 | `20HKJ` | 2022-01 | 2022-01-29, 2022-01-04, 2022-01-02, 2022-01-17 |
| 2021/2022 | `20HKJ` | 2022-02 | 2022-02-03, 2022-02-13, 2022-02-06 |
| 2021/2022 | `20HKJ` | 2022-03 | 2022-03-25, 2022-03-15, 2022-03-08, 2022-03-13, 2022-03-28, 2022-03-03 |
| 2021/2022 | `20HKJ` | 2022-04 | 2022-04-04, 2022-04-02 |
| 2021/2022 | `20HKJ` | 2022-05 | 2022-05-04, 2022-05-07, 2022-05-17, 2022-05-02 |
| 2021/2022 | `20HLJ` | 2021-12 | 2021-12-18, 2021-12-15 |
| 2021/2022 | `20HLJ` | 2022-01 | 2022-01-29, 2022-01-02 |
| 2021/2022 | `20HLJ` | 2022-02 | 2022-02-01, 2022-02-08 |
| 2021/2022 | `20HLJ` | 2022-03 | 2022-03-30, 2022-03-03 |
| 2021/2022 | `20HLJ` | 2022-04 | 2022-04-17, 2022-04-02, 2022-04-04 |
| 2021/2022 | `20HLJ` | 2022-05 | 2022-05-17, 2022-05-04, 2022-05-19 |
| 2021/2022 | `20HLK` | 2021-12 | 2021-12-18, 2021-12-15, 2021-12-23 |
| 2021/2022 | `20HLK` | 2022-01 | 2022-01-14, 2022-01-29, 2022-01-09, 2022-01-02 |
| 2021/2022 | `20HLK` | 2022-02 | 2022-02-18, 2022-02-06 |
| 2021/2022 | `20HLK` | 2022-03 | 2022-03-28, 2022-03-25 |
| 2021/2022 | `20HLK` | 2022-04 | 2022-04-02, 2022-04-04 |
| 2021/2022 | `20HLK` | 2022-05 | 2022-05-04, 2022-05-17 |
| 2021/2022 | `20HMH` | 2021-12 | 2021-12-15 |
| 2021/2022 | `20HMH` | 2022-01 | 2022-01-31, 2022-01-29 |
| 2021/2022 | `20HMH` | 2022-02 | 2022-02-10, 2022-02-08 |
| 2021/2022 | `20HMH` | 2022-03 | 2022-03-12, 2022-03-30 |
| 2021/2022 | `20HMH` | 2022-04 | 2022-04-16, 2022-04-14 |
| 2021/2022 | `20HMH` | 2022-05 | 2022-05-06, 2022-05-01, 2022-05-04, 2022-05-14, 2022-05-24 |
