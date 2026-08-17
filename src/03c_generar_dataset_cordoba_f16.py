"""
Genera el dataset de entrenamiento (formato train_cordoba_f16, 5 clases sin
Sorgo) para tiles de Cordoba que todavia no lo tienen, a partir de los
composites 2019-2020 y las mascaras MNC ya verificados por
0_verificar_pipeline.py.

El generador original de train_cordoba_f16 se corrio como algo suelto en la
maquina remota y no quedo versionado (ver bash_history: aparecia
"remap_aplicado=np.array([1])" pero no un script completo) -- este es un
reemplazo limpio, mismo formato de salida (.npy sin comprimir, X float16,
Y uint8, listo para TileDatasetMmap sin pasar por 03b_convertir_datasets_a_mmap.py).

Tabla de remapeo (5 clases, ver README_MNC.md, Tabla 2 del informe MNC-INTA
verano 2020): Maiz=2, Soja=3, Mani=4, todo el resto de codigos IDVER
documentados (incluido Sorgo, que se excluye a proposito como clase propia)
colapsa a Fondo=1, y 255 (Sin datos) a NoData=0. Cualquier codigo IDVER no
documentado en la Tabla 2 tambien cae en NoData (se ignora en vez de
arriesgarse a mal-clasificarlo como Fondo).

Uso:
    python3 src/03c_generar_dataset_cordoba_f16.py
    python3 src/03c_generar_dataset_cordoba_f16.py --tiles 20HKH 20HMG
"""
import argparse
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

DIR_COMPOSITES = Path("/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_2019_2020")
DIR_MASCARAS   = Path("/home1/ferreyra/cordoba_ia_unne/mascaras_procesadas")
DIR_SALIDA     = Path("/mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16")

MESES_FIJOS = ["201910", "201911", "201912", "202001", "202002", "202003", "202004"]
SIZE = 256
STEP = 256

# Tiles con composites (7/7 meses) y mascara con distribucion aceptable segun
# 0_verificar_pipeline.py, que todavia no estan en train_cordoba_f16 (no
# forman parte de TILES_TRAIN/TILE_VAL/TILES_TEST en 1_train_cordoba.py).
TILES_DEFAULT = ["20HKH", "20HMG", "20HNG", "20HPG", "20HPH", "20JNL"]

# Leyenda oficial MNC-INTA verano 2020 (Tabla 2, ver README_MNC.md).
REMAP_5 = np.zeros(256, dtype=np.int64)
REMAP_5[10] = 2   # Maiz
REMAP_5[11] = 3   # Soja
REMAP_5[16] = 4   # Mani
for _c in (12, 13, 14, 15, 17, 18, 19, 21, 22):  # resto (incl. Sorgo) -> Fondo
    REMAP_5[_c] = 1
REMAP_5[255] = 0  # Sin datos -> NoData
# Cualquier otro codigo no documentado en la Tabla 2 queda en 0 (NoData/
# ignorado via ignore_index=0 en la loss) por el np.zeros inicial.


def _procesar_chunk_ventanas(args):
    tile_id, coords, composites, ruta_mascara = args
    X_local, Y_local = [], []
    with rasterio.open(ruta_mascara) as src_label:
        composites_abiertos = [rasterio.open(c) for c in composites]
        try:
            for x, y in coords:
                ventana = Window(x, y, SIZE, SIZE)
                parche_y = REMAP_5[src_label.read(1, window=ventana)]

                if np.all(parche_y == 0):
                    continue

                parche_x = np.concatenate(
                    [src_mes.read(window=ventana).astype(np.float32) / 10000.0
                     for src_mes in composites_abiertos],
                    axis=0,
                )
                X_local.append(parche_x.astype(np.float16))
                Y_local.append(parche_y.astype(np.uint8))
        finally:
            for src_mes in composites_abiertos:
                src_mes.close()

    print(f"[{tile_id}] chunk terminado: {len(X_local)} parches de {len(coords)} ventanas", flush=True)
    return X_local, Y_local


def generar_dataset_tile(tile_id, max_workers=None):
    ruta_x = DIR_SALIDA / f"dataset_{tile_id}_X.npy"
    ruta_y = DIR_SALIDA / f"dataset_{tile_id}_Y.npy"
    if ruta_x.exists() and ruta_y.exists():
        print(f"{tile_id}: ya existe, omitiendo", flush=True)
        return

    composites = [DIR_COMPOSITES / f"{tile_id}_{mes}_median.tif" for mes in MESES_FIJOS]
    faltantes = [c.name for c in composites if not c.exists()]
    if faltantes:
        print(f"{tile_id}: faltan composites {faltantes}, se omite", flush=True)
        return

    ruta_mascara = DIR_MASCARAS / f"etiqueta_mnc_{tile_id}_10m.tif"
    if not ruta_mascara.exists():
        print(f"{tile_id}: falta mascara {ruta_mascara.name}, se omite", flush=True)
        return

    with rasterio.open(ruta_mascara) as src_label:
        alto, ancho = src_label.height, src_label.width

    coords = [(x, y) for y in range(0, alto - SIZE, STEP) for x in range(0, ancho - SIZE, STEP)]
    max_workers = max_workers or os.cpu_count()
    chunks = [c for c in np.array_split(coords, max_workers) if len(c) > 0]
    tareas = [(tile_id, chunk.tolist(), composites, ruta_mascara) for chunk in chunks]

    print(f"{tile_id}: {len(coords)} ventanas repartidas en {len(tareas)} workers", flush=True)

    X_list, Y_list = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for X_local, Y_local in executor.map(_procesar_chunk_ventanas, tareas):
            X_list.extend(X_local)
            Y_list.extend(Y_local)

    if not X_list:
        print(f"{tile_id}: 0 parches validos, no se genera archivo", flush=True)
        return

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    tmp_x, tmp_y = ruta_x.with_suffix(".npy.tmp"), ruta_y.with_suffix(".npy.tmp")
    with open(tmp_x, "wb") as f:
        np.save(f, X)
    with open(tmp_y, "wb") as f:
        np.save(f, Y)
    os.replace(tmp_x, ruta_x)
    os.replace(tmp_y, ruta_y)

    # Distribucion de clases del tile, para comparar a ojo contra los tiles
    # ya existentes (ej. Mani deberia ser una fraccion chica, ausente en
    # tiles del este segun README_MNC.md).
    total = (Y > 0).sum()
    dist = ", ".join(f"{n}={100 * (Y == c).sum() / total:.1f}%"
                      for c, n in [(1, "Fondo"), (2, "Maiz"), (3, "Soja"), (4, "Mani")])
    print(f"{tile_id}: guardado {len(X_list)} parches -> {ruta_x.name} / {ruta_y.name} ({dist})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", nargs="*", default=TILES_DEFAULT)
    args = parser.parse_args()

    for tile in args.tiles:
        generar_dataset_tile(tile)

    print("\nListo.", flush=True)
