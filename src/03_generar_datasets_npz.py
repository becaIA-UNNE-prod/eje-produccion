import os
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path

LABEL_REMAP = np.array(
    [0] + [1]*10 + [2]*4 + [3, 4, 5, 6, 7] + [8]*8,
    dtype=np.int64
)

def generar_dataset_tile(tile_id, ruta_mascara, dir_composites, dir_salida, size=256, step=256):
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta_salida = dir_salida / f"dataset_T{tile_id}.npz"
    
    if ruta_salida.exists():
        print(f"{tile_id}: ya existe, omitiendo")
        return

    composites = sorted(Path(dir_composites).glob(f"{tile_id}_*_median.tif"))
    if not composites:
        print(f"{tile_id}: sin composites")
        return

    print(f"{tile_id}: {len(composites)} meses")

    with rasterio.open(ruta_mascara) as src_label:
        alto, ancho = src_label.height, src_label.width
        X_list, Y_list = [], []
        contador = 0

        for y in range(0, alto - size, step):
            for x in range(0, ancho - size, step):
                ventana = Window(x, y, size, size)
                parche_y = src_label.read(1, window=ventana)

                if np.all(parche_y == 0):
                    continue

                parche_y_remapeado = LABEL_REMAP[parche_y]

                parche_x_temporal = []
                for composite in composites:
                    with rasterio.open(composite) as src_mes:
                        parche_mes = src_mes.read(window=ventana).astype(np.float32) / 10000.0
                        parche_x_temporal.append(parche_mes)

                parche_x = np.concatenate(parche_x_temporal, axis=0)
                X_list.append(parche_x)
                Y_list.append(parche_y_remapeado)
                contador += 1

                if contador % 100 == 0:
                    print(f"  {contador} parches...")

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    np.savez_compressed(ruta_salida, X=X, Y=Y)
    print(f"{tile_id}: guardado {contador} parches -> {ruta_salida}")

if __name__ == "__main__":
    tiles = ["20HLJ", "20HLK", "20HMJ", "20HMK", "20JLL", "20JML"]
    DIR_COMPOSITES = "/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_filtrado"
    DIR_SALIDA = "/mnt/yacy_1/prod/ferreyra/dataset/train"

    for tile in tiles:
        ruta_mascara = f"./mascaras_procesadas/etiqueta_{tile}_10m_test.tif"
        if not os.path.exists(ruta_mascara):
            print(f"{tile}: máscara no encontrada")
            continue
        generar_dataset_tile(tile, ruta_mascara, DIR_COMPOSITES, DIR_SALIDA)

    print("\nListo - todos los datasets generados")
