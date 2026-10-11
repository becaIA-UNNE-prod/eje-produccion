"""
Reproyecta el Mapa Nacional de Cultivos 2018/2019 (INTA, Zenodo
10.5281/zenodo.8286326, mnc_2018-2019_V1R1.tif, 30 m, EPSG:4326) a la grilla
10 m de cada tile Sentinel-2 (misma CRS/transform/tamano que los composites),
con resampleo nearest. Equivalente a las etiqueta_mnc_<TILE>_10m.tif de verano
2020, pero para la campana 2018-2019.

Se guardan los codigos crudos IDN1 (no remapeados), igual que las mascaras
2020 guardan IDVER crudo. OJO: la leyenda es OTRA que la de 2020 (Tabla 1 del
informe, informe_mnc_2018_2019.pdf) y describe la campana completa (simples y
dobles cultivos):
  0 Maiz, 1 Soja 1ra, 2 CI-Soja, 3 CI-Maiz, 4 Mani, 5 Girasol,
  6 CI sin cultivo de verano, 7 Ganaderia, 8 Lenosa natural,
  9 Lenosas cultivadas, 11 Algodon, 12 Poroto, 13 Cana, 14 Girasol-CV,
  15 Mascara (agua, humedales, urbano), 16 sin datos (nodata)
Remapeo al esquema de 5 clases: REMAP_MNC_1819 (abajo).

Uso:
    python3 -m src.datos.mascaras_mnc_2018_2019
    python3 -m src.datos.mascaras_mnc_2018_2019 --tiles 20HLJ 20HLK
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

RUTA_MNC = Path("/mnt/yacy_1/prod/ferreyra/MNC_INTA/2018_2019/mnc_2018-2019_V1R1.tif")
DIR_REF = Path("/mnt/yacy_3/prod_bis/ferreyra/sentinel2_cordoba_2018_2019")
DIR_SALIDA = Path(__file__).resolve().parents[2] / "mascaras_procesadas"
NODATA = 16

# IDN1 2018/2019 -> 0=NoData, 1=Fondo, 2=Maiz, 3=Soja, 4=Mani. Los dobles
# cultivos se asignan por su cultivo de verano (CI-Soja -> Soja, CI-Maiz ->
# Maiz), que es lo que ve la ventana dic-may y lo que etiqueta el MNC verano
# 2020. Codigos no documentados quedan en NoData.
REMAP_MNC_1819 = np.zeros(256, dtype=np.int64)
REMAP_MNC_1819[[0, 3]] = 2
REMAP_MNC_1819[[1, 2]] = 3
REMAP_MNC_1819[4] = 4
REMAP_MNC_1819[[5, 6, 7, 8, 9, 11, 12, 13, 14, 15]] = 1


def generar_mascara(tile):
    salida = DIR_SALIDA / f"etiqueta_mnc1819_{tile}_10m.tif"
    if salida.exists():
        print(f"{tile}: ya existe", flush=True)
        return
    ref = next(p for p in sorted((DIR_REF / tile).glob("*/B04.tif")))
    with rasterio.open(ref) as r:
        meta = r.meta.copy()
    meta.update(dtype="uint8", count=1, nodata=NODATA, driver="GTiff", compress="deflate")
    destino = np.full((meta["height"], meta["width"]), NODATA, dtype=np.uint8)
    with rasterio.open(RUTA_MNC) as src:
        reproject(source=rasterio.band(src, 1), destination=destino,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=NODATA,
                  dst_transform=meta["transform"], dst_crs=meta["crs"], dst_nodata=NODATA,
                  resampling=Resampling.nearest)
    tmp = salida.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(destino, 1)
    tmp.rename(salida)
    y = REMAP_MNC_1819[destino]
    total = (y > 0).sum()
    dist = ", ".join(f"{n}={100 * (y == c).sum() / max(total, 1):.1f}%"
                     for c, n in [(1, "Fondo"), (2, "Maiz"), (3, "Soja"), (4, "Mani")])
    print(f"{tile}: {100 * total / y.size:.1f}% con dato ({dist})", flush=True)


def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=None, help="default: todos los de DIR_REF")
    return ap


def main(argv=None):
    a = crear_parser().parse_args(argv)
    tiles = a.tiles or sorted(d.name for d in DIR_REF.iterdir() if d.is_dir())
    for t in tiles:
        generar_mascara(t)


if __name__ == "__main__":
    main()
