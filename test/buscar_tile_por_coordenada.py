import os
import rasterio
from rasterio.warp import transform as warp_transform
from pathlib import Path

# Coordenadas de Manfredi (EEA INTA): 31°49'S 63°36'O
LAT = -(31 + 49 / 60)
LON = -(63 + 36 / 60)

DIR_COMPOSITES = "/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_filtrado"


def punto_en_raster(ruta_tif, lon, lat):
    with rasterio.open(ruta_tif) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        x, y = xs[0], ys[0]

        b = src.bounds
        if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
            return False, None

        fila, col = src.index(x, y)
        if not (0 <= fila < src.height and 0 <= col < src.width):
            return False, None

        valor = next(src.sample([(x, y)]))[0]
        return True, (fila, col, valor)


def buscar_tile(lat, lon, dir_composites):
    archivos = sorted(Path(dir_composites).glob("*.tif"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron .tif en {dir_composites}")

    # Un archivo de ejemplo alcanza por tile: todos los meses de un mismo
    # tile comparten la misma huella espacial.
    tiles = {}
    for f in archivos:
        tile_id = f.name.split("_")[0]
        tiles.setdefault(tile_id, f)

    print(f"Punto a ubicar: lat={lat}, lon={lon}")
    print(f"{len(tiles)} tiles encontrados en {dir_composites}\n")

    encontrados = []
    for tile_id, ruta in sorted(tiles.items()):
        dentro, info = punto_en_raster(ruta, lon, lat)
        if dentro:
            fila, col, valor = info
            print(f"  {tile_id}: SI (fila={fila}, col={col}, valor={valor})")
            encontrados.append(tile_id)
        else:
            print(f"  {tile_id}: no")

    return encontrados


if __name__ == "__main__":
    encontrados = buscar_tile(LAT, LON, DIR_COMPOSITES)
    print()
    if encontrados:
        print(f"El punto está dentro de: {', '.join(encontrados)}")
    else:
        print("El punto no está dentro de ningún tile revisado.")

