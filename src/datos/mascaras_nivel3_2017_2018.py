"""
Reproyecta la capa de etiquetas 2017-2018 (Nivel3_28_dic_2018_30m_completo.tif,
30 m, EPSG:4326, leyenda en Nivel3_28_dic_2018_30m.qml) a la grilla 10 m de
cada tile Sentinel-2, con resampleo nearest. Mismo esquema que
01b_mascaras_mnc_2018_2019.py, pero para la campana 2017-2018 y para los 17
tiles de Cordoba (01_moldear_mascaras.py solo genero las etiqueta_<TILE>_10m_test.tif
de unos pocos tiles, sobre la grilla de los .jp2 legacy).

Se guardan los codigos crudos Nivel3 (0 = sin dato); el remapeo a 5 clases es
REMAP_NIVEL3 de 03c_generar_dataset_cordoba_f16.py (--leyenda nivel3):
  16 Maiz, 20 Trigo-Maiz de segunda -> Maiz
  17 Soja, 21 Trigo-Soja de segunda -> Soja
  18 Mani -> Mani, resto -> Fondo

Uso:
    python3 -m src.datos.mascaras_nivel3_2017_2018 --ref /mnt/yacy_3/prod_bis/ferreyra/sentinel2_cordoba_2017_2018_dicmay
    python3 -m src.datos.mascaras_nivel3_2017_2018 --ref <DIR> --tiles 20HLJ 20HLK
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

RAIZ = Path(__file__).resolve().parents[2]
RUTA_NIVEL3 = RAIZ / "data/referencias/Nivel3_28_dic_2018_30m_completo.tif"
DIR_SALIDA = Path("/home1/ferreyra/cordoba_ia_unne/mascaras_procesadas")
NODATA = 0

# Mismo remapeo que el que usa 03c para entrenar (no duplicar la tabla).
_spec = importlib.util.spec_from_file_location(
    "dataset03c", RAIZ / "src" / "03c_generar_dataset_cordoba_f16.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
REMAP_NIVEL3 = _mod.REMAP_NIVEL3


def generar_mascara(tile, dir_ref):
    salida = DIR_SALIDA / f"etiqueta_nivel3_1718_{tile}_10m.tif"
    if salida.exists():
        print(f"{tile}: ya existe", flush=True)
        return
    ref = next(p for p in sorted((dir_ref / tile).glob("*/B04.tif")))
    with rasterio.open(ref) as r:
        meta = r.meta.copy()
    meta.update(dtype="uint8", count=1, nodata=NODATA, driver="GTiff", compress="deflate")
    destino = np.full((meta["height"], meta["width"]), NODATA, dtype=np.uint8)
    with rasterio.open(RUTA_NIVEL3) as src:
        reproject(source=rasterio.band(src, 1), destination=destino,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=NODATA,
                  dst_transform=meta["transform"], dst_crs=meta["crs"], dst_nodata=NODATA,
                  resampling=Resampling.nearest)
    tmp = salida.with_suffix(".tif.tmp")
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(destino, 1)
    tmp.rename(salida)
    y = REMAP_NIVEL3[destino]
    total = (y > 0).sum()
    dist = ", ".join(f"{n}={100 * (y == c).sum() / max(total, 1):.1f}%"
                     for c, n in [(1, "Fondo"), (2, "Maiz"), (3, "Soja"), (4, "Mani")])
    print(f"{tile}: {100 * total / y.size:.1f}% con dato ({dist})", flush=True)


def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True,
                    help="descarga COG 2017-2018 (<REF>/<TILE>/<FECHA>/B04.tif) para tomar la grilla")
    ap.add_argument("--tiles", nargs="*", default=None, help="default: todos los de --ref")
    return ap


def main(argv=None):
    a = crear_parser().parse_args(argv)
    tiles = a.tiles or sorted(d.name for d in a.ref.iterdir() if d.is_dir())
    for t in tiles:
        generar_mascara(t, a.ref)


if __name__ == "__main__":
    main()
