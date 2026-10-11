"""
Genera el dataset de entrenamiento (formato train_cordoba_f16, 5 clases sin
Sorgo) para tiles de Cordoba que todavia no lo tienen, a partir de los
composites 2019-2020 y las mascaras MNC ya verificados por
src.historico.verificar_pipeline.

El generador original de train_cordoba_f16 se corrio como algo suelto en la
maquina remota y no quedo versionado (ver bash_history: aparecia
"remap_aplicado=np.array([1])" pero no un script completo) -- este es un
reemplazo limpio, mismo formato de salida (.npy sin comprimir, X float16,
Y uint8, listo para TileDatasetMmap sin pasar por src/datos/convertir_datasets_a_mmap.py).

Tabla de remapeo (5 clases, ver docs/referencias/leyenda_mnc.md, Tabla 2 del informe MNC-INTA
verano 2020): Maiz=2, Soja=3, Mani=4, todo el resto de codigos IDVER
documentados (incluido Sorgo, que se excluye a proposito como clase propia)
colapsa a Fondo=1, y 255 (Sin datos) a NoData=0. Cualquier codigo IDVER no
documentado en la Tabla 2 tambien cae en NoData (se ignora en vez de
arriesgarse a mal-clasificarlo como Fondo).

Uso:
    python3 -m src.datos.generar_dataset_cordoba_f16
    python3 -m src.datos.generar_dataset_cordoba_f16 --tiles 20HKH 20HMG

Otra ventana de meses / otro dataset (ej. dic-may, ver 02c_composites_cog.py):
    python3 -m src.datos.generar_dataset_cordoba_f16 --tiles <17 tiles> \
        --meses 201912 202001 202002 202003 202004 202005 \
        --dir-composites /mnt/yacy_3/prod_bis/ferreyra/composites_dicmay_2019_2020 \
        --dir-salida /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_f16

Campana 2017-2018 con la capa Nivel3 (--leyenda nivel3, mascaras de
01c_mascaras_nivel3_2017_2018.py, ver pipeline_dicmay_1718.sh):
    python3 -m src.datos.generar_dataset_cordoba_f16 --leyenda nivel3 --tiles <13 tiles train> \
        --meses 201712 201801 201802 201803 201804 201805 \
        --dir-composites /mnt/yacy_3/prod_bis/ferreyra/composites_dicmay_2017_2018 \
        --dir-salida /mnt/yacy_3/prod_bis/ferreyra/dataset/train_cordoba_dicmay_1718_f16
"""
import argparse
import concurrent.futures
import os
import sys
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from utils.leyendas_mnc import REMAP_2021, verificar_codigos

DIR_COMPOSITES = Path("/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_2019_2020")
DIR_MASCARAS   = Path("/home1/ferreyra/cordoba_ia_unne/mascaras_procesadas")
DIR_SALIDA     = Path("/mnt/yacy_1/prod/ferreyra/dataset/train_cordoba_f16")

MESES_FIJOS = ["201910", "201911", "201912", "202001", "202002", "202003", "202004"]
SIZE = 256
STEP = 256

# Tiles con composites (7/7 meses) y mascara con distribucion aceptable segun
# src.historico.verificar_pipeline, que todavia no estan en train_cordoba_f16 (no
# forman parte de TILES_TRAIN/TILE_VAL/TILES_TEST en 1_train_cordoba.py).
TILES_DEFAULT = ["20HKH", "20HMG", "20HNG", "20HPG", "20HPH", "20JNL"]

# Leyenda oficial MNC-INTA verano 2020 (Tabla 2, ver docs/referencias/leyenda_mnc.md).
REMAP_5 = np.zeros(256, dtype=np.int64)
REMAP_5[10] = 2   # Maiz
REMAP_5[11] = 3   # Soja
REMAP_5[16] = 4   # Mani
for _c in (12, 13, 14, 15, 17, 18, 19, 21, 22):  # resto (incl. Sorgo) -> Fondo
    REMAP_5[_c] = 1
REMAP_5[255] = 0  # Sin datos -> NoData
# Cualquier otro codigo no documentado en la Tabla 2 queda en 0 (NoData/
# ignorado via ignore_index=0 en la loss) por el np.zeros inicial.

# Capa Nivel3 2017-2018 (Nivel3_28_dic_2018_30m.qml), 0 = sin dato. A
# diferencia de 03d (oct-abr), los dobles cultivos van por su cultivo de
# verano (Trigo-Maiz 2da -> Maiz, Trigo-Soja 2da -> Soja): es lo que ve la
# ventana dic-may y el mismo criterio que REMAP_MNC_1819 (CI-Soja -> Soja).
REMAP_NIVEL3 = np.zeros(256, dtype=np.int64)
REMAP_NIVEL3[1:28] = 1       # natural, urbano, Trigo, Sorgo, pasturas, etc. -> Fondo
REMAP_NIVEL3[[16, 20]] = 2   # Maiz, Trigo-Maiz de segunda
REMAP_NIVEL3[[17, 21]] = 3   # Soja, Trigo-Soja de segunda
REMAP_NIVEL3[18] = 4         # Mani

# MNC 2018/2019 (IDN1, ver informe INTA): 0 Maiz, 1 Soja 1ra, 2 CI-Soja,
# 3 CI-Maiz, 4 Mani, 16 sin dato -> 0 (ignorado). Igual que 01b y 11.
REMAP_MNC_1819 = np.zeros(256, dtype=np.int64)
REMAP_MNC_1819[[0, 3]] = 2
REMAP_MNC_1819[[1, 2]] = 3
REMAP_MNC_1819[4] = 4
REMAP_MNC_1819[[5, 6, 7, 8, 9, 11, 12, 13, 14, 15]] = 1

LEYENDAS = {"1819": (REMAP_MNC_1819, "etiqueta_mnc1819_{}_10m.tif"), "2020":   (REMAP_5,      "etiqueta_mnc_{}_10m.tif"),
            "nivel3": (REMAP_NIVEL3, "etiqueta_nivel3_1718_{}_10m.tif")}
LEYENDAS["2021"] = (REMAP_2021, "etiqueta_mnc2021_{}_10m.tif")
LEYENDA = "2020"
ESTRICTO = False
MAX_WORKERS = None
IGNORAR_CODIGO20 = False


def _procesar_chunk_ventanas(args):
    tile_id, coords, composites, ruta_mascara, remap = args
    X_local, Y_local = [], []
    with rasterio.open(ruta_mascara) as src_label:
        composites_abiertos = [rasterio.open(c) for c in composites]
        try:
            for x, y in coords:
                ventana = Window(x, y, SIZE, SIZE)
                crudo = src_label.read(1, window=ventana)
                if LEYENDA == '2021':
                    verificar_codigos(np.unique(crudo), IGNORAR_CODIGO20)
                parche_y = remap[crudo]

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
        if ESTRICTO:
            raise FileNotFoundError(f"{tile_id}: {faltantes}")
        return

    remap, patron = LEYENDAS[LEYENDA]
    ruta_mascara = DIR_MASCARAS / patron.format(tile_id)
    if not ruta_mascara.exists():
        print(f"{tile_id}: falta mascara {ruta_mascara.name}, se omite", flush=True)
        if ESTRICTO:
            raise FileNotFoundError(ruta_mascara)
        return

    with rasterio.open(ruta_mascara) as src_label:
        alto, ancho = src_label.height, src_label.width
        grilla = (src_label.crs, src_label.transform, src_label.shape)
    for composite in composites:
        with rasterio.open(composite) as src:
            if (src.crs, src.transform, src.shape) != grilla or src.count != 4:
                raise ValueError(f'{tile_id}: grilla/bandas incompatible en {composite}')
            if (src.descriptions != (None, None, None, None)
                    and src.descriptions != ('B02', 'B03', 'B04', 'B08')):
                raise ValueError(f'{composite}: orden de bandas no acreditado {src.descriptions}')

    coords = [(x, y) for y in range(0, alto - SIZE, STEP) for x in range(0, ancho - SIZE, STEP)]
    max_workers = max_workers or MAX_WORKERS or os.cpu_count()
    chunks = [c for c in np.array_split(coords, max_workers) if len(c) > 0]
    tareas = [(tile_id, chunk.tolist(), composites, ruta_mascara, remap) for chunk in chunks]

    print(f"{tile_id}: {len(coords)} ventanas repartidas en {len(tareas)} workers", flush=True)

    X_list, Y_list = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for X_local, Y_local in executor.map(_procesar_chunk_ventanas, tareas):
            X_list.extend(X_local)
            Y_list.extend(Y_local)

    if not X_list:
        print(f"{tile_id}: 0 parches validos, no se genera archivo", flush=True)
        if ESTRICTO:
            raise ValueError(f"{tile_id}: sin parches validos")
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
    procedencia = {'tile': tile_id, 'leyenda': LEYENDA, 'meses': MESES_FIJOS,
                  'composites': [str(c.resolve()) for c in composites],
                  'composites_mtime_ns': [c.stat().st_mtime_ns for c in composites],
                  'mascara': str(ruta_mascara.resolve()),
                  'mascara_mtime_ns': ruta_mascara.stat().st_mtime_ns,
                  'bandas': ['B02', 'B03', 'B04', 'B08'], 'normalizacion': 10000,
                  'shape_x': list(X.shape), 'shape_y': list(Y.shape),
                  'leyenda_version': 1, 'codigos_ignorados': [20] if IGNORAR_CODIGO20 and LEYENDA == '2021' else []}
    meta = DIR_SALIDA / f'dataset_{tile_id}_procedencia.json'
    tmp_meta = meta.with_suffix('.json.tmp')
    tmp_meta.write_text(json.dumps(procedencia, indent=2))
    tmp_meta.replace(meta)

    # Distribucion de clases del tile, para comparar a ojo contra los tiles
    # ya existentes (ej. Mani deberia ser una fraccion chica, ausente en
    # tiles del este segun docs/referencias/leyenda_mnc.md).
    total = (Y > 0).sum()
    dist = ", ".join(f"{n}={100 * (Y == c).sum() / total:.1f}%"
                      for c, n in [(1, "Fondo"), (2, "Maiz"), (3, "Soja"), (4, "Mani")])
    print(f"{tile_id}: guardado {len(X_list)} parches -> {ruta_x.name} / {ruta_y.name} ({dist})", flush=True)


def crear_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", nargs="*", default=TILES_DEFAULT)
    parser.add_argument("--meses", nargs="*", default=MESES_FIJOS)
    parser.add_argument("--dir-composites", type=Path, default=DIR_COMPOSITES)
    parser.add_argument("--dir-salida", type=Path, default=DIR_SALIDA)
    parser.add_argument("--dir-mascaras", type=Path, default=DIR_MASCARAS)
    parser.add_argument("--leyenda", choices=LEYENDAS, default=LEYENDA,
                        help="2020 = MNC verano 2020 (default), nivel3 = capa Nivel3 2017-2018")
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--ignorar-codigo20', action='store_true')
    parser.add_argument('--estricto', action='store_true', help='abortar ante faltantes')
    return parser


def main(argv=None):
    global IGNORAR_CODIGO20, ESTRICTO, MAX_WORKERS, LEYENDA
    global MESES_FIJOS, DIR_COMPOSITES, DIR_SALIDA, DIR_MASCARAS
    parser = crear_parser()
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error('--workers debe ser positivo')
    if args.ignorar_codigo20 and args.leyenda != '2021':
        parser.error('--ignorar-codigo20 solo aplica a --leyenda 2021')
    IGNORAR_CODIGO20 = args.ignorar_codigo20
    ESTRICTO = args.estricto
    MAX_WORKERS = args.workers
    LEYENDA = args.leyenda
    MESES_FIJOS = args.meses
    DIR_COMPOSITES = args.dir_composites
    DIR_SALIDA = args.dir_salida
    DIR_MASCARAS = args.dir_mascaras

    for tile in args.tiles:
        generar_dataset_tile(tile)

    print("\nListo.", flush=True)


if __name__ == "__main__":
    main()
