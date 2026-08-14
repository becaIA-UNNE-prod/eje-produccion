import os
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import concurrent.futures

LABEL_REMAP = np.array(
    [0] + [1]*10 + [2]*4 + [3, 4, 5, 6, 7] + [8]*8,
    dtype=np.int64
)

def _procesar_chunk_ventanas(args):
    tile_id, coords, composites, ruta_mascara, size = args

    X_local, Y_local = [], []
    with rasterio.open(ruta_mascara) as src_label:
        composites_abiertos = [rasterio.open(c) for c in composites]
        try:
            for x, y in coords:
                ventana = Window(x, y, size, size)
                parche_y = src_label.read(1, window=ventana)

                if np.all(parche_y == 0):
                    continue

                parche_y_remapeado = LABEL_REMAP[parche_y]

                parche_x_temporal = [
                    src_mes.read(window=ventana).astype(np.float32) / 10000.0
                    for src_mes in composites_abiertos
                ]
                parche_x = np.concatenate(parche_x_temporal, axis=0)

                X_local.append(parche_x)
                Y_local.append(parche_y_remapeado)
        finally:
            for src_mes in composites_abiertos:
                src_mes.close()

    print(f"[{tile_id}] chunk terminado: {len(X_local)} parches de {len(coords)} ventanas")
    return X_local, Y_local

def generar_dataset_tile(tile_id, ruta_mascara, dir_composites, dir_salida, meses=None, size=256, step=256, max_workers=None):
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    # X/Y se guardan como .npy sueltos sin comprimir (no .npz comprimido) para
    # poder abrirlos con mmap_mode="r" al entrenar: un array comprimido siempre
    # se descomprime entero en RAM al leerlo, uno sin comprimir se puede leer
    # parche a parche bajo demanda (ver utils/tile_dataset.py).
    ruta_x = dir_salida / f"dataset_T{tile_id}_X.npy"
    ruta_y = dir_salida / f"dataset_T{tile_id}_Y.npy"

    if ruta_x.exists() and ruta_y.exists():
        try:
            np.load(ruta_x, mmap_mode="r").shape
            np.load(ruta_y, mmap_mode="r").shape
            print(f"{tile_id}: ya existe, omitiendo")
            return
        except Exception:
            print(f"{tile_id}: existente corrupto, se regenera")

    if meses:
        # Filtra a un calendario fijo de meses (p.ej. la intersección común entre
        # varios tiles) para que todos los tiles del dataset final tengan la misma
        # cantidad de canales, sin importar cuántos meses de composites tenga cada uno.
        composites = [Path(dir_composites) / f"{tile_id}_{mes}_median.tif" for mes in meses]
        faltantes = [c.name for c in composites if not c.exists()]
        if faltantes:
            print(f"{tile_id}: faltan composites para {faltantes}")
            return
    else:
        composites = sorted(Path(dir_composites).glob(f"{tile_id}_*_median.tif"))
        if not composites:
            print(f"{tile_id}: sin composites")
            return

    print(f"{tile_id}: {len(composites)} meses")

    with rasterio.open(ruta_mascara) as src_label:
        alto, ancho = src_label.height, src_label.width

    coords = [
        (x, y)
        for y in range(0, alto - size, step)
        for x in range(0, ancho - size, step)
    ]

    max_workers = max_workers or os.cpu_count()
    chunks = [c for c in np.array_split(coords, max_workers) if len(c) > 0]
    tareas = [(tile_id, chunk.tolist(), composites, ruta_mascara, size) for chunk in chunks]

    print(f"{tile_id}: {len(coords)} ventanas repartidas en {len(tareas)} workers")

    X_list, Y_list = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for X_local, Y_local in executor.map(_procesar_chunk_ventanas, tareas):
            X_list.extend(X_local)
            Y_list.extend(Y_local)

    # float16/uint8: mismo truco de reduccion de RAM ya validado en el
    # pipeline Cordoba 2019-2020 (entrenamiento float16), aplicado aca tambien.
    X = np.stack(X_list, axis=0).astype(np.float16)
    Y = np.stack(Y_list, axis=0).astype(np.uint8)

    tmp_x = ruta_x.with_suffix(ruta_x.suffix + ".tmp")
    tmp_y = ruta_y.with_suffix(ruta_y.suffix + ".tmp")
    with open(tmp_x, "wb") as f:
        np.save(f, X)
    with open(tmp_y, "wb") as f:
        np.save(f, Y)
    os.replace(tmp_x, ruta_x)
    os.replace(tmp_y, ruta_y)
    print(f"{tile_id}: guardado {len(X_list)} parches -> {ruta_x.name} / {ruta_y.name}")

if __name__ == "__main__":
    # 7 tiles con máscara ya generada y >= 7 meses de composites en común.
    # MESES_COMUNES = intersección de meses presentes en TODOS estos tiles
    # (evita mezclar tiles con distinta cantidad de canales en el mismo dataset).
    # 19HGB se descartó: sus 1764 ventanas dieron 0 parches válidos (máscara sin
    # datos de cultivo utilizables en esa zona).
    tiles = ["20HLJ", "20HLK", "20JLL", "20JML", "20JNL", "20HNK"]
    MESES_COMUNES = ["201707", "201708", "201709", "201710", "201711", "201712", "201804"]
    DIR_COMPOSITES = "/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_filtrado"
    DIR_SALIDA = "/mnt/yacy_1/prod/ferreyra/dataset/train_multitile7"

    for tile in tiles:
        ruta_mascara = f"./mascaras_procesadas/etiqueta_{tile}_10m_test.tif"
        if not os.path.exists(ruta_mascara):
            print(f"{tile}: máscara no encontrada")
            continue
        generar_dataset_tile(tile, ruta_mascara, DIR_COMPOSITES, DIR_SALIDA, meses=MESES_COMUNES)

    print("\nListo - todos los datasets generados")
