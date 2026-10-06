"""
Genera el dataset de entrenamiento (formato train_cordoba_f16, 5 clases sin
Sorgo) para la campana 2017-2018 (legacy), remapeando su capa de etiquetas de
9 clases (Nivel3_28_dic_2018_30m_completo.tif, ver 01_moldear_mascaras.py y
03_generar_datasets_npz.py) al esquema MNC de 5 clases vigente, para poder
sumarla al entrenamiento de 1_train_cordoba.py junto con los tiles 2019-2020.

Motivacion: sumar diversidad interanual (otra campana agricola, no solo otro
tile) al set de TRAIN. La campana 2017-2018 ya tiene medianas mensuales
precomputadas (composites_filtrado) y mascara reproyectada a 10 m por tile
(mascaras_procesadas/etiqueta_<TILE>_10m_test.tif) de un trabajo previo -- ver
03_generar_datasets_npz.py, que es el generador original (formato .npz
comprimido, 9 clases, dataset/train_multitile7) para estos mismos tiles.

Remapeo 9->5 clases: se preserva el criterio ya usado en el esquema MNC
vigente (Maiz/Soja/Mani como clases propias, todo el resto -- incluido
Sorgo -- colapsa a Fondo, ver docstring de 03c_generar_dataset_cordoba_f16.py)
aplicado sobre los codigos crudos del raster Nivel3 en vez de sobre IDVER:
  raw 0        -> 0 NoData
  raw 1-10     -> 1 Fondo   (agrupacion "Natural" de LABEL_REMAP)
  raw 11-14    -> 1 Fondo   (Urbano)
  raw 15       -> 1 Fondo   (Trigo)
  raw 16       -> 2 Maiz
  raw 17       -> 3 Soja
  raw 18       -> 4 Mani
  raw 19       -> 1 Fondo   (Sorgo)
  raw 20-27    -> 1 Fondo   (Otros)
(ver LABEL_REMAP en 03_generar_datasets_npz.py para el agrupamiento 9 clases
del que sale esta tabla).

Los archivos de salida usan un tile_id pseudo "<TILE>_2017" (no "<TILE>" a
secas) porque varios de estos tiles (20HLJ, 20HLK, 20JNL) ya tienen un dataset
2019-2020 con ese mismo TILE en train_cordoba_f16 -- son la misma celda de
grilla Sentinel-2 pero de otra campana, y un mismo nombre de archivo pisaria
al otro.

IMPORTANTE -- alineacion de meses entre campanas (bug encontrado y corregido):
la primera version de este script usaba MESES_2017 = jul-dic 2017 + abr 2018
(la interseccion de meses disponible en los 6 tiles segun
03_generar_datasets_npz.py), pero esa ventana NO corresponde al mismo tramo
del ciclo agricola que MESES_FIJOS de 03c_generar_dataset_cordoba_f16.py
(oct 2019 - abr 2020: siembra -> floracion -> cosecha). Como el modelo no
tiene ningun embedding de fecha, la posicion del canal es la unica senal de
"que mes es" -- entrenar con canales que en una campana significan invierno/
barbecho y en la otra significan floracion es entrenar con una senal
temporal inconsistente entre ejemplos. Se corrige usando la ventana
calendario equivalente (oct 2017 - abr 2018), y por eso TILES_DEFAULT bajo a
solo los tiles que la tienen completa (ver comentario debajo).

Uso:
    python3 src/03d_generar_dataset_cordoba_2017_f16.py
    python3 src/03d_generar_dataset_cordoba_2017_f16.py --tiles 20HLJ 20JLL
"""
import argparse
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

DIR_COMPOSITES = Path("/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_filtrado")
DIR_MASCARAS   = Path("/home1/ferreyra/cordoba_ia_unne/mascaras_procesadas")
DIR_SALIDA     = Path("/mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16")

# Ventana calendario equivalente a MESES_FIJOS de 03c (oct 2019 - abr 2020),
# desplazada un anio: oct 2017 - abr 2018. Mismo tramo del ciclo agricola
# (siembra -> floracion -> cosecha) en ambas campanas -- ver nota arriba.
MESES_2017 = ["201710", "201711", "201712", "201801", "201802", "201803", "201804"]
SIZE = 256
STEP = 256

# De los 6 tiles con mascara 10m generada (ver 03_generar_datasets_npz.py),
# solo estos 4 tienen los 7 meses de MESES_2017 completos: a 20JML le faltan
# 201802/201803 y a 20HNK le faltan 201801/201802 (verificado listando
# composites_filtrado). 19HGB tambien se descarta (0 parches validos, ver
# nota en 03_generar_datasets_npz.py).
TILES_DEFAULT = ["20HLJ", "20HLK", "20JLL", "20JNL"]

# Tabla de remapeo 9->5 clases, ver docstring arriba.
REMAP_2017_A_5 = np.zeros(256, dtype=np.int64)
REMAP_2017_A_5[1:16] = 1   # Natural/Fondo, Urbano, Trigo -> Fondo
REMAP_2017_A_5[16] = 2     # Maiz
REMAP_2017_A_5[17] = 3     # Soja
REMAP_2017_A_5[18] = 4     # Mani
REMAP_2017_A_5[19:28] = 1  # Sorgo, Otros -> Fondo


def _procesar_chunk_ventanas(args):
    tile_id, coords, composites, ruta_mascara = args
    X_local, Y_local = [], []
    with rasterio.open(ruta_mascara) as src_label:
        composites_abiertos = [rasterio.open(c) for c in composites]
        try:
            for x, y in coords:
                ventana = Window(x, y, SIZE, SIZE)
                parche_y = REMAP_2017_A_5[src_label.read(1, window=ventana)]

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
    tile_out = f"{tile_id}_2017"
    ruta_x = DIR_SALIDA / f"dataset_{tile_out}_X.npy"
    ruta_y = DIR_SALIDA / f"dataset_{tile_out}_Y.npy"
    if ruta_x.exists() and ruta_y.exists():
        print(f"{tile_out}: ya existe, omitiendo", flush=True)
        return

    composites = [DIR_COMPOSITES / f"{tile_id}_{mes}_median.tif" for mes in MESES_2017]
    faltantes = [c.name for c in composites if not c.exists()]
    if faltantes:
        print(f"{tile_out}: faltan composites {faltantes}, se omite", flush=True)
        return

    ruta_mascara = DIR_MASCARAS / f"etiqueta_{tile_id}_10m_test.tif"
    if not ruta_mascara.exists():
        print(f"{tile_out}: falta mascara {ruta_mascara.name}, se omite", flush=True)
        return

    with rasterio.open(ruta_mascara) as src_label:
        alto, ancho = src_label.height, src_label.width

    coords = [(x, y) for y in range(0, alto - SIZE, STEP) for x in range(0, ancho - SIZE, STEP)]
    max_workers = max_workers or os.cpu_count()
    chunks = [c for c in np.array_split(coords, max_workers) if len(c) > 0]
    tareas = [(tile_out, chunk.tolist(), composites, ruta_mascara) for chunk in chunks]

    print(f"{tile_out}: {len(coords)} ventanas repartidas en {len(tareas)} workers", flush=True)

    X_list, Y_list = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for X_local, Y_local in executor.map(_procesar_chunk_ventanas, tareas):
            X_list.extend(X_local)
            Y_list.extend(Y_local)

    if not X_list:
        print(f"{tile_out}: 0 parches validos, no se genera archivo", flush=True)
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

    total = (Y > 0).sum()
    dist = ", ".join(f"{n}={100 * (Y == c).sum() / total:.1f}%"
                      for c, n in [(1, "Fondo"), (2, "Maiz"), (3, "Soja"), (4, "Mani")])
    print(f"{tile_out}: guardado {len(X_list)} parches -> {ruta_x.name} / {ruta_y.name} ({dist})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", nargs="*", default=TILES_DEFAULT)
    args = parser.parse_args()

    for tile in args.tiles:
        generar_dataset_tile(tile)

    print("\nListo.", flush=True)
