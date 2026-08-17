# Carpeta `src/` — Pipeline de clasificación de cultivos (Córdoba, Sentinel-2)

Este proyecto arma un pipeline de **clasificación de cultivos** (maíz, soja, maní, etc.) a partir de
imágenes **Sentinel-2** de la provincia de Córdoba, usando una **U-Net** entrenada con **PyTorch**.

El pipeline **vigente** (Córdoba completa, campaña 2019-2020, etiquetas oficiales MNC-INTA) es:

```
Sentinel-2 2019-2020 (por tile/fecha) ──► 02b: composites mensuales (mediana temporal, filtro de cobertura)
Etiquetas MNC-INTA + composites ──► dataset .npy por tile (5 clases ya remapeadas, ver nota abajo)
dataset .npy ──► 1_train_cordoba.py: entrenar U-Net ──► 2_inferencia.py: mapa de clasificación
```

> **Nota sobre el dataset de entrenamiento:** los `.npy` que consume `1_train_cordoba.py`
> (`train_cordoba_f16/dataset_<TILE>_X.npy` / `_Y.npy`) vienen de convertir con
> `03b_convertir_datasets_a_mmap.py` unos `.npz` que ya traen el remapeo a 5 clases aplicado
> (`NUM_CLASSES=5`: 0=NoData, 1=Fondo, 2=Maíz, 3=Soja, 4=Maní — sin Sorgo, ver `README_MNC.md`).
> Esos `.npz` de origen se generan con un **proceso externo a este repo** (no hay, hoy, un script
> acá que produzca `train_cordoba_f16` desde cero) — si se necesita reconstruirlos desde
> composites + máscara, hay que escribir el equivalente de `03_generar_datasets_npz.py` pero con
> la tabla de remapeo MNC de 5 clases (ver `REMAP_MNC` en `0_verificar_pipeline.py`) en vez de la
> `LABEL_REMAP` de 9 clases que usa ese script para el pipeline viejo (ver más abajo).

Los archivos están numerados según la etapa del pipeline a la que pertenecen (`01_`, `02_`, `03_`, `1_`, `2_`).

> **Nota de historial — pipeline 2017-2018 (legacy):** las secciones `01`, `02` y `03` de más abajo
> (`01_moldear_mascaras.py`, `02_composites_mensuales.py`, `03_generar_datasets_npz.py`) documentan
> un pipeline **anterior**, sobre la campaña 2017-2018 con una capa de etiquetas distinta
> (`Nivel3_28_dic_2018_30m_completo.tif`, no la leyenda oficial MNC-INTA) y un esquema de 9 clases.
> Ese pipeline ya **no tiene un script de entrenamiento vigente que lo consuma** (el que lo hacía,
> `1_train_multitile.py`, se eliminó por quedar redundante con `1_train_cordoba.py`) — se conserva
> por si hace falta reconstruir un dataset comparativo, y porque `5_generar_matriz_confusion.py` /
> `6_generar_comparacion_visual.py` todavía saben leer su formato de 9 clases vía `--prefijo dataset_T`.
> Originalmente también existían variantes redundantes de cada etapa (versiones "single tile" y
> "multi-tile", con/sin manejo de `.jp2` corruptos, un pipeline de entrenamiento aún más viejo basado
> en `.npy` sueltos por parche) que ya se habían unificado antes de esta nota.

---

## Directorio de imágenes Sentinel-2 filtradas (pipeline 2017-2018, legacy)

`/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba_2017_2018_filtrado`

Es la fuente de imágenes que usa `BASE_S2` en `02_composites_mensuales.py`, en reemplazo del directorio
original `sentinel2_cordoba_2017_2018`. Contiene las mismas escenas Sentinel-2 pero con las
fechas/imágenes con **cobertura nubosa** ya descartadas (el filtrado de nubes se hizo con un proceso
externo a `src/`, no hay un script de filtrado en este repo).

- **Estructura:** idéntica al directorio original — `<TILE>/<FECHA>/<TILE>_<FECHA>_<BANDA>.jp2`, por
  ejemplo `20JLL/20170719/20JLL_20170719_B04.jp2`. Solo cambia el contenido (menos fechas por tile, ya
  que las nubladas fueron eliminadas), no la nomenclatura ni el formato.
- **Por qué importa:** al tener menos imágenes "ruidosas" por mes, la mediana temporal que calcula
  `02_composites_mensuales.py` es más representativa (menos posibilidad de que una escena nublada
  contamine la mediana de ese mes).
- `01_moldear_mascaras.py` todavía usa el directorio **original** (`sentinel2_cordoba_2017_2018`) para
  tomar la banda de referencia (B04) al armar la máscara a 10 m — como el filtrado no cambia la grilla
  espacial ni la nomenclatura, es indistinto usar el original o el filtrado para ese paso puntual.

---

## 01 — Preparación de máscaras (pipeline 2017-2018, legacy)

### `01_moldear_mascaras.py`
Toma la capa de etiquetas original (`Nivel3_28_dic_2018_30m_completo.tif`, resolución 30 m) y la
**reproyecta/remuestrea a 10 m**, alineándola exactamente con la grilla de una imagen Sentinel-2 de
referencia (misma CRS, transform, ancho y alto), usando `rasterio.warp.reproject` con resampleo
`nearest` (apto para datos categóricos).

- **Función clave:** `generar_mascara_10m(ruta_etiqueta_30m, ruta_referencia_s2, ruta_salida)`
- **Entrada:** TIF de etiquetas a 30 m + un `.jp2` de banda (B04) de un tile/fecha de Sentinel-2 como referencia espacial.
- **Salida:** `mascaras_procesadas/etiqueta_<TILE>_10m_test.tif` — máscara categórica a 10 m, en píxeles idénticos a Sentinel-2.
- En el bloque `if __main__` está *hardcodeado* al tile de prueba `20JLL` (fecha `20170719`).

---

## 02 — Composites mensuales (mediana temporal)

### `02_composites_mensuales.py` (pipeline 2017-2018, legacy)
Agrupa las escenas Sentinel-2 de uno o varios tiles por **mes**, apila las bandas `B02, B03, B04, B08`
de todas las fechas de ese mes y calcula la **mediana por píxel** (reduce nubes/outliers). Escribe un
GeoTIFF por mes (`<TILE>_<AAAAMM>_median.tif`, 4 bandas). Usa `ProcessPoolExecutor` para paralelizar
por mes, y tolera `.jp2` corruptos (los saltea con un aviso en vez de abortar).

- **Funciones clave:** `procesar_un_mes(args)` (por worker) y
  `generar_composiciones_mensuales(tile_id, ruta_base_s2, dir_salida, max_workers)`.
- Si el archivo de salida del mes ya existe, lo omite (permite reanudar procesos largos).
- **Configuración (`__main__`):** `TILES = None` procesa **todos** los tiles encontrados en `BASE_S2`;
  `TILES = ["20JLL"]` (o cualquier lista) procesa solo esos tiles puntuales.
- Este script reemplaza a los antiguos `02_temporal_median_composites.py`,
  `02_temporal_median_composites_filtrado.py` y `02_composites_filtrado_multitile.py` (idénticos salvo
  por el manejo de errores y el alcance de 1 tile vs. todos).

### `02b_composites_2019_2020.py` (pipeline vigente)
Genera los composites mensuales para la campaña **2019-2020** (la que efectivamente entrena
`1_train_cordoba.py`). A diferencia de `02_composites_mensuales.py`: exige una lista fija de 7 meses
(`201910` a `202004`; si falta alguno, el tile no se genera parcial) y verifica **cobertura real de
píxeles válidos** (≥80%) antes de aceptar una fecha, para descartar escenas de borde de órbita.

- **Función clave:** `elegir_fecha(tile, mes)` (primera fecha del mes que supera el umbral) y
  `generar_composite(tile, mes)`.
- **Entrada:** `sentinel2_cordoba_2019_2020/<TILE>/<FECHA>/<TILE>_<FECHA>_<BANDA>.jp2`.
- **Salida:** `cordoba_dataset_filtrado/composites_2019_2020/<TILE>_<AAAAMM>_median.tif`.
- Cubre 21 tiles de Córdoba (lista `TILES` en el script).

---

## 03 — Generación de dataset (pipeline 2017-2018, legacy)

### `03_generar_datasets_npz.py`
Recorre la máscara de un tile en ventanas de `256x256` px (sin solapamiento), descarta parches
totalmente vacíos (`NoData`), y para cada parche útil concatena, mes a mes, las 4 bandas de todos los
composites disponibles → arma un tensor `X` de forma `(Meses*4, 256, 256)`, **normalizado** (`/10000.0`).
Las etiquetas se **remapean** vía la tabla `LABEL_REMAP` (colapsa ~28 clases originales en 9 clases
agrupadas: 0=NoData, 1=Natural/Fondo, 2=Urbano, 3=Trigo, 4=Maíz, 5=Soja, 6=Maní, 7=Sorgo, 8=Otros).
Genera **un par `.npy` (`_X`/`_Y`) sin comprimir por tile**, apto para mmap.

- **Función clave:** `generar_dataset_tile(tile_id, ruta_mascara, dir_composites, dir_salida, meses=None, size=256, step=256)`
- **Entrada:** composites `<TILE>_*_median.tif` + máscara `etiqueta_<TILE>_10m_test.tif`.
- **Salida:** `dataset/train_multitile7/dataset_T<TILE>_X.npy` / `_Y.npy` (uno por tile; si ya existe lo omite).
- En `__main__` procesa una lista fija de 6 tiles (`20HLJ, 20HLK, 20JLL, 20JML, 20JNL, 20HNK`) sobre los
  composites 2017-2018 filtrados.
- Sin consumidor directo entre los scripts de entrenamiento vigentes (ver nota de historial arriba);
  útil sobre todo junto con `5_generar_matriz_confusion.py` / `6_generar_comparacion_visual.py`
  (`--prefijo dataset_T --n_clases 9`) para comparar contra el esquema de 9 clases original.

### `03b_convertir_datasets_a_mmap.py`
Convierte `.npz` comprimidos (hay que cargarlos enteros en RAM para leer cualquier parche) a pares de
`.npy` sueltos sin comprimir, que se pueden abrir con `mmap_mode="r"` (`utils/tile_dataset.py`). Valida,
si el `.npz` trae el flag `remap_aplicado`, que sea `1` antes de convertir (evita entrenar por error con
un dataset sin remapear). Es el paso previo obligatorio para poder usar un dataset en formato `.npz` con
`1_train_cordoba.py`.

### `03c_generar_dataset_cordoba_f16.py` (pipeline vigente)
El generador original de `train_cordoba_f16` (11 tiles) se corrió como algo suelto en la máquina remota
y nunca quedó versionado (ver `bash_history`, solo sobrevivió el fragmento `remap_aplicado=np.array([1])`).
Este script es su reemplazo limpio, mismo formato de salida (`.npy` sin comprimir, `X` float16, `Y`
uint8, listo para `TileDatasetMmap` sin pasar por `03b`) — pensado para sumar tiles nuevos a
`train_cordoba_f16` a medida que se van verificando con `0_verificar_pipeline.py`.

- **Tabla de remapeo (5 clases):** Maíz=2, Soja=3, Maní=4, todo el resto de códigos IDVER documentados en
  la Tabla 2 del informe MNC-INTA (incluido Sorgo, excluido a propósito) colapsa a Fondo=1, y `255` (Sin
  datos) a NoData=0 — ver `README_MNC.md`. Distinta de la tabla `REMAP_MNC` de `0_verificar_pipeline.py`,
  que tiene un bug (Maíz/Soja invertidos, Maní mapeado a Fondo) y solo se usa ahí para un chequeo de
  cobertura, no para generar datos de entrenamiento.
- **Entrada:** composites `composites_2019_2020/<TILE>_*_median.tif` + máscara `etiqueta_mnc_<TILE>_10m.tif`.
- **Salida:** `train_cordoba_f16/dataset_<TILE>_X.npy` / `_Y.npy` (si ya existe lo omite).
- Por default corre sobre los 6 tiles que `0_verificar_pipeline.py` marca como aptos (composites + máscara)
  pero que todavía no están en el split de `1_train_cordoba.py`: `20HKH, 20HMG, 20HNG, 20HPG, 20HPH, 20JNL`.
  Imprime la distribución de clases de cada tile generado, para comparar a ojo contra los 11 tiles
  existentes antes de sumarlos al entrenamiento.

---

## 0 — Verificación y pesos por clase

### `0_verificar_pipeline.py`
Chequea integridad de descargas Sentinel-2 2019-2020, composites y datasets `.npz` antes de lanzar el
entrenamiento (`--etapa descarga|datasets|todas`). Incluye `REMAP_MNC`, la tabla que traduce los códigos
`IDVER` oficiales del MNC-INTA (ver `README_MNC.md`) al esquema de clases del pipeline.

### `0b_calcular_pesos.py`
Calcula la distribución de píxeles por clase y sugiere pesos para la loss (raíz del inverso de
frecuencia, y alternativa de inverso puro) sobre el mismo split de `TRAIN` que usa `1_train_cordoba.py`.
Correr de nuevo si cambia el split de tiles o el dataset, y pisar `PESOS` en `1_train_cordoba.py` con el
resultado.

---

## Entrenamiento

### `1_train_cordoba.py` (pipeline vigente)
Entrena la arquitectura `utils/model.py::SimpleUNet` (U-Net de 4 niveles de encoder/decoder, 64→128→256→512
+ cuello de botella 1024, con `Dropout2d` en los niveles profundos y skip connections) sobre **Córdoba
completa**: 11 tiles con **split explícito por tile** (no aleatorio) — `TILES_TRAIN` (7 tiles),
`TILE_VAL` (2 tiles), `TILES_TEST` (2 tiles) — elegido para mantener cobertura geográfica
oeste-centro-este y presencia de Maní (clase minoritaria) en los tres conjuntos.

- `NUM_CLASSES = 5`: 0=NoData, 1=Fondo, 2=Maíz, 3=Soja, 4=Maní (sin Sorgo — ver `README_MNC.md`).
- Pérdida `CEDiceLoss` (`utils/losses.py`): CrossEntropy con **pesos por clase** (`ignore_index=0`) +
  Dice, para reforzar específicamente la clase minoritaria (Maní).
- Augmentación en TRAIN (flips + rotaciones de 90°, vía `TileDatasetMmap(..., augment=True)`),
  mixed precision (AMP) + `cudnn.benchmark`, `EPOCHS = 100`, `PATIENCE = 30`, `BATCH_SIZE = 32`.
- Carga los `.npy` con `TileDatasetMmap` (mmap bajo demanda, no carga los tiles enteros en RAM) y
  paraleliza la lectura con `NUM_WORKERS` workers.
- Guarda `best_model.pth`, `modelo_final.pth` e `historial.npy` en `DIR_EXP`, y evalúa en test al final.
- Reemplaza a los antiguos `1_train_mnc2.py`, `1_train_multitile.py` (recortes chicos de tiles, ya
  redundantes con esta versión completa) y `1_train_cordoba_aug.py` (misma corrida pero sin AMP/mmap,
  separada solo para probar augmentación — ahora todo vive acá).

### `1b_finetune_manfredi.py`
Parte del checkpoint de `1_train_cordoba.py` (`exp_cordoba_f16/best_model.pth`) y lo especializa con
LR bajo (`1e-5`) sobre un único tile de interés (Manfredi). Mismo esquema de 5 clases y misma loss
(`CEDiceLoss`) que el entrenamiento base, para no cambiar el objetivo de optimización entre ambos.

---

## Inferencia

### `2_inferencia.py`
Aplica un modelo `SimpleUNet` ya entrenado sobre un tile completo, por bloques (`ventanas de 512x512`),
para generar un mapa de clasificación en formato GeoTIFF de 1 sola banda (`uint8`, `nodata=0`).

- **Función clave:** `generar_mapa_clasificacion()` (config al inicio de la función: tile `20HMK` por
  default, `NUM_CLASSES = 5`, pesos en `.../dataset/exp_cordoba_f16/best_model.pth` — alineado con
  `1_train_cordoba.py`).
- Autodetecta la cantidad de canales de entrada contando los composites mensuales disponibles (`meses * 4 bandas`).
- Recorre la imagen en ventanas de 512x512, arma el mismo tipo de tensor de entrada que en entrenamiento
  (concatenación de bandas por mes, normalizado `/10000.0`), corre el modelo y escribe el `argmax` de la
  predicción directamente en el TIF de salida (`./prediccion_<TILE>.tif`).

---

## Evaluación / visualización

### `4_generar_metricas.py`
Grafica curvas de Train/Val Loss y Accuracy a partir del `historial.npy` de un experimento
(`--exp <DIR_EXP>`).

### `5_generar_matriz_confusion.py` / `6_generar_comparacion_visual.py`
Matriz de confusión y comparación visual (target vs. predicción vs. mapa de errores) sobre un tile de
evaluación. Por default apuntan al esquema vigente de 5 clases (`--n_clases 5 --prefijo dataset_`,
`train_cordoba_f16`); pasando `--n_clases 6` o `--n_clases 9` (junto con `--prefijo dataset_T`) también
saben leer, respectivamente, el esquema con Sorgo o el esquema de 9 clases del pipeline 2017-2018 legacy.

---

## Dependencias externas usadas por `src/` (carpeta `utils/`)

- **`utils/model.py`** → `SimpleUNet`: U-Net de 4 niveles (encoder 64→128→256→512, cuello de botella 1024,
  con `Dropout2d` en los niveles más profundos), con skip connections y `ConvTranspose2d` para el
  upsampling. Usada por `1_train_cordoba.py`, `1b_finetune_manfredi.py`, `2_inferencia.py`,
  `5_generar_matriz_confusion.py` y `6_generar_comparacion_visual.py`.
- **`utils/losses.py`** → `CEDiceLoss` / `DiceLoss`: CrossEntropy ponderada + Dice, usada por
  `1_train_cordoba.py` y `1b_finetune_manfredi.py` para reforzar las clases minoritarias.
- **`utils/tile_dataset.py`** → `TileDatasetMmap`: lee parches `.npy` bajo demanda vía mmap (sin cargar
  el tile entero en RAM), con soporte opcional de remapeo de clases y augmentación (flips/rotaciones).
- **`utils/dataset.py`** → `CordobaDataset`: quedó **sin uso** tras eliminar `1_train.py` (era el único
  script que la consumía). Se dejó sin tocar porque no forma parte de `src/`; avisar si también se quiere eliminar.

## Resumen rápido de rutas de entrada/salida

| Etapa | Script | Entrada principal | Salida principal |
|---|---|---|---|
| 02b | `02b_composites_2019_2020.py` | `sentinel2_cordoba_2019_2020/<TILE>/<FECHA>/*.jp2` | `.../composites_2019_2020/<TILE>_<AAAAMM>_median.tif` |
| — | *(externo al repo)* | composites 2019-2020 + etiquetas MNC-INTA | `dataset/train_cordoba_f16/dataset_<TILE>.npz` (5 clases, `remap_aplicado=1`) |
| 03b | `03b_convertir_datasets_a_mmap.py` | `.npz` de origen | `dataset_<TILE>_X.npy` / `_Y.npy` |
| 1  | `1_train_cordoba.py` | `.npy` del paso anterior | pesos `.pth` + historial |
| 1b | `1b_finetune_manfredi.py` | checkpoint de `1_train_cordoba.py` | checkpoint especializado |
| 2  | `2_inferencia.py` | composites + pesos entrenados | `prediccion_<TILE>.tif` |

Rutas del pipeline 2017-2018 legacy (`01_moldear_mascaras.py` → `02_composites_mensuales.py` →
`03_generar_datasets_npz.py`), sin consumidor de entrenamiento vigente: ver las secciones
correspondientes más arriba.
