# Carpeta `src/` — Pipeline de clasificación de cultivos (Córdoba, Sentinel-2)

Este proyecto arma un pipeline de **clasificación de cultivos** (maíz, soja, maní, sorgo, etc.) a partir de
imágenes **Sentinel-2** de la provincia de Córdoba, usando una **U-Net** entrenada con **PyTorch**.

El flujo general es:

```
Etiquetas (30m) ──► 01: remuestrear a 10m ──► máscara alineada con Sentinel-2
Imágenes Sentinel-2 (por tile/fecha) ──► 02: composites mensuales (mediana temporal)
Composites + máscara ──► 03: dataset .npz por tile ──► 1_train_multitile: entrenar U-Net ──► 2_inferencia: mapa de clasificación
```

Los archivos están numerados según la etapa del pipeline a la que pertenecen (`01_`, `02_`, `03_`, `1_`, `2_`).

> Nota de historial: originalmente existían varias variantes redundantes de cada etapa (una versión
> "single tile" y otra "multi-tile", con/sin manejo de `.jp2` corruptos, un pipeline de entrenamiento
> viejo basado en `.npy` sueltos vs. el actual basado en `.npz`). Esas variantes se unificaron en un
> único script parametrizable por etapa; el pipeline viejo (`1_train.py` + `03_extraer_parches.py`,
> que dependían de `NUM_CLASSES=50` como placeholder y de `utils/dataset.py::CordobaDataset`) se eliminó
> por estar superado.

---

## Directorio de imágenes Sentinel-2 filtradas

`/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba_2017_2018_filtrado`

Es la fuente de imágenes que usa el pipeline vigente (`BASE_S2` en `02_composites_mensuales.py`), en
reemplazo del directorio original `sentinel2_cordoba_2017_2018`. Contiene las mismas escenas Sentinel-2
pero con las fechas/imágenes con **cobertura nubosa** ya descartadas (el filtrado de nubes se hizo con
un proceso externo a `src/`, no hay un script de filtrado en este repo).

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

## 01 — Preparación de máscaras

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

### `02_composites_mensuales.py`
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

---

## 03 — Generación de dataset

### `03_generar_datasets_npz.py`
Recorre la máscara de un tile en ventanas de `256x256` px (sin solapamiento), descarta parches
totalmente vacíos (`NoData`), y para cada parche útil concatena, mes a mes, las 4 bandas de todos los
composites disponibles → arma un tensor `X` de forma `(Meses*4, 256, 256)`, **normalizado** (`/10000.0`).
Las etiquetas se **remapean** vía la tabla `LABEL_REMAP` (colapsa ~28 clases originales en 9 clases
agrupadas: 0=NoData, 1=Natural/Fondo, 2=Urbano, 3=Trigo, 4=Maíz, 5=Soja, 6=Maní, 7=Sorgo, 8=Otros).
Genera **un `.npz` comprimido por tile** con todos sus parches apilados (`X`, `Y`).

- **Función clave:** `generar_dataset_tile(tile_id, ruta_mascara, dir_composites, dir_salida, size=256, step=256)`
- **Entrada:** composites `<TILE>_*_median.tif` + máscara `etiqueta_<TILE>_10m_test.tif`.
- **Salida:** `dataset/train/dataset_T<TILE>.npz` (uno por tile; si ya existe lo omite).
- En `__main__` procesa una lista fija de 6 tiles de interés: `20HLJ, 20HLK, 20HMJ, 20HMK, 20JLL, 20JML`.
- Este es el formato que consume `1_train_multitile.py`.

---

## Entrenamiento

### `1_train_multitile.py`
Entrena la arquitectura `utils/model.py::SimpleUNet` (U-Net chica de 2 niveles de encoder/decoder con
skip connections) sobre varios `.npz` (uno por tile), con **splits explícitos por tile** (no aleatorios):
`TILES_TRAIN = [20HMJ, 20HMK, 20JML]`, `TILE_VAL = 20JNL`, `TILES_TEST = [20HNK]` — así el modelo se
valida/testea en tiles geográficamente distintos a los de entrenamiento.

- `NUM_CLASSES = 6`: 0=NoData, 1=Fondo, 2=Maíz, 3=Soja, 4=Maní, 5=Sorgo.
- Pérdida `CrossEntropyLoss` con **pesos por clase** (para compensar desbalance) e `ignore_index=0`,
  `EPOCHS = 100`, `PATIENCE = 50` (early stopping), `BATCH_SIZE = 16`.
- Guarda `best_model.pth`, `modelo_final.pth` e `historial.npy` en `DIR_EXP`, y evalúa en test al final.
- **Configurable al inicio del script** (antes eran dos scripts casi duplicados, ahora es un único punto
  de configuración):
  - `DIR_TRAIN` / `DIR_EXP` / `DATASET_PREFIX`: por defecto apunta a lo que efectivamente genera
    `03_generar_datasets_npz.py` (`dataset/train/dataset_T<TILE>.npz` → `dataset/exp_verano/`). Es el
    único generador de `.npz` que existe en este repo, y es el método principal del pipeline.
  - `APLICAR_REMAP_6 = True` por default: esos `.npz` traen 9 clases sin remapear, así que `TileDataset`
    aplica el remapeo a 6 clases (cultivos de verano) al vuelo, con la tabla `REMAP_6` definida en el
    propio script.
  - Si en algún momento existe un dataset ya curado a 6 clases generado por un proceso externo a este
    repo (ej. `dataset/train_mnc/dataset_mnc_T<TILE>.npz`), se puede apuntar ahí cambiando esas 4
    variables (`APLICAR_REMAP_6 = False` en ese caso) — ver comentario en el propio archivo.

---

## Inferencia

### `2_inferencia.py`
Aplica un modelo `SimpleUNet` ya entrenado sobre un tile completo, por bloques (`ventanas de 512x512`),
para generar un mapa de clasificación en formato GeoTIFF de 1 sola banda (`uint8`, `nodata=0`).

- **Función clave:** `generar_mapa_clasificacion()` (config al inicio de la función: tile `20JLL`,
  `NUM_CLASSES = 6`, pesos en `.../dataset/exp_verano/best_model.pth` — alineado con `1_train_multitile.py`).
- Autodetecta la cantidad de canales de entrada contando los composites mensuales disponibles (`meses * 4 bandas`).
- Recorre la imagen en ventanas de 512x512, arma el mismo tipo de tensor de entrada que en entrenamiento
  (concatenación de bandas por mes, normalizado `/10000.0`), corre el modelo y escribe el `argmax` de la
  predicción directamente en el TIF de salida (`./prediccion_<TILE>.tif`).

---

## Dependencias externas usadas por `src/` (carpeta `utils/`)

- **`utils/model.py`** → `SimpleUNet`: U-Net pequeña (encoder de 2 niveles: 64→128→256 canales, con
  skip connections y `ConvTranspose2d` para el upsampling). Usada por `1_train_multitile.py` y `2_inferencia.py`.
- **`utils/dataset.py`** → `CordobaDataset`: quedó **sin uso** tras eliminar `1_train.py` (era el único
  script que la consumía). Se dejó sin tocar porque no forma parte de `src/`; avisar si también se quiere eliminar.

## Resumen rápido de rutas de entrada/salida

| Etapa | Script | Entrada principal | Salida principal |
|---|---|---|---|
| 01 | `01_moldear_mascaras.py` | `Nivel3_28_dic_2018_30m_completo.tif` + `.jp2` referencia | `mascaras_procesadas/etiqueta_<TILE>_10m_test.tif` |
| 02 | `02_composites_mensuales.py` | `sentinel2_cordoba_2017_2018_filtrado/<TILE>/<FECHA>/*.jp2` | `.../composites_filtrado/<TILE>_<AAAAMM>_median.tif` |
| 03 | `03_generar_datasets_npz.py` | composites + máscara | `dataset/train/dataset_T<TILE>.npz` |
| 1  | `1_train_multitile.py` | `.npz` del paso 03 | pesos `.pth` + historial |
| 2  | `2_inferencia.py` | composites + pesos entrenados | `prediccion_<TILE>.tif` |
