"""MNC verano 2021 -> mascaras crudas alineadas a composites SCL (nearest)."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from utils.leyendas_mnc import verificar_codigos
from utils.multicampana import SPLIT

Y3 = Path('/mnt/yacy_3/prod_bis/ferreyra')
MNC = Path('/mnt/yacy_1/prod/ferreyra/MNC_INTA/2020_2021/verano-2021.tif')


def verificar_fuente(ruta, ignorar_codigo20=False):
    conteos = np.zeros(256, dtype=np.int64)
    with rasterio.open(ruta) as src:
        if src.count != 1 or src.dtypes != ('uint8',) or src.nodata != 255:
            raise ValueError(f'MNC2021: metadatos inesperados {src.count}/{src.dtypes}/{src.nodata}')
        # Ventanas de franjas para no cargar el raster nacional en memoria.
        for y in range(0, src.height, 512):
            ventana = rasterio.windows.Window(0, y, src.width, min(512, src.height - y))
            conteos += np.bincount(src.read(1, window=ventana).ravel(), minlength=256)
    presentes = np.flatnonzero(conteos)
    if conteos[20]:
        print(f'ATENCION: codigo 20 no documentado, {conteos[20]} pixeles nacionales; '
              f'politica={"ignorar como NoData" if ignorar_codigo20 else "abortar"}', flush=True)
    verificar_codigos(presentes, ignorar_codigo20)
    if not {10, 11, 16}.issubset(set(presentes)):
        raise ValueError('Falta algun cultivo objetivo en MNC2021')
    resultado = {str(int(c)): int(conteos[c]) for c in presentes}
    print(f'Labels MNC2021 verificados (raster completo): {resultado}', flush=True)
    return resultado


def generar(tile, fuente, composites, salida, ignorar_codigo20=False):
    referencia = composites / f'{tile}_202012_median.tif'
    destino = salida / f'etiqueta_mnc2021_{tile}_10m.tif'
    with rasterio.open(referencia) as ref:
        meta = ref.meta.copy()
        grilla = (ref.crs, ref.transform, ref.shape)
    if destino.exists():
        with rasterio.open(destino) as m:
            if (m.crs, m.transform, m.shape) != grilla or m.nodata != 255 or m.dtypes != ('uint8',):
                raise ValueError(f'Mascara existente incompatible: {destino}')
            verificar_codigos(np.unique(m.read(1)), ignorar_codigo20)
        print(f'{tile}: mascara existente verificada', flush=True)
        return
    meta.update(dtype='uint8', count=1, nodata=255, compress='deflate', driver='GTiff')
    datos = np.full((meta['height'], meta['width']), 255, dtype=np.uint8)
    with rasterio.open(fuente) as src:
        reproject(source=rasterio.band(src, 1), destination=datos,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=255,
                  dst_transform=meta['transform'], dst_crs=meta['crs'], dst_nodata=255,
                  resampling=Resampling.nearest, num_threads=4)
    verificar_codigos(np.unique(datos), ignorar_codigo20)
    if not np.any(datos != 255):
        raise ValueError(f'{tile}: mascara sin datos')
    salida.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix('.tif.tmp')
    with rasterio.open(temporal, 'w', **meta) as dst:
        dst.write(datos, 1)
    temporal.replace(destino)
    print(f'{tile}: mascara escrita ({np.mean(datos != 255):.1%} con datos)', flush=True)


def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fuente', type=Path, default=MNC)
    ap.add_argument('--dir-composites', type=Path, default=Y3/'composites_2020_2021_scl')
    ap.add_argument('--dir-salida', type=Path, default=Path(__file__).resolve().parents[2]/'mascaras_procesadas')
    ap.add_argument('--tiles', nargs='+', default=sum(SPLIT.values(), []))
    ap.add_argument('--solo-verificar', action='store_true')
    ap.add_argument('--ignorar-codigo20', action='store_true', help='exclusion explicita de codigo desconocido 20')
    return ap


def main(argv=None):
    a = crear_parser().parse_args(argv)
    verificar_fuente(a.fuente, a.ignorar_codigo20)
    if not a.solo_verificar:
        for tile in a.tiles:
            generar(tile, a.fuente, a.dir_composites, a.dir_salida, a.ignorar_codigo20)


if __name__ == "__main__":
    main()
