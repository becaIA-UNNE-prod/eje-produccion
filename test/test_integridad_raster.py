"""Regresion: un header TIFF correcto no garantiza bloques legibles."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.integridad_raster import verificar_raster, auditar_rasters


class IntegridadRasterTest(unittest.TestCase):
    def test_bloque_corrupto_header_legible_y_cache_invalidada(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d)/'raster.tif'; cache = Path(d)/'cache.json'
            with rasterio.open(ruta, 'w', driver='GTiff', height=1024, width=512,
                               count=1, dtype='uint16', crs='EPSG:32720',
                               transform=from_origin(0,0,10,10), tiled=True,
                               blockxsize=512, blockysize=512) as dst:
                dst.write(np.ones((1024,512), dtype=np.uint16),1)
            self.assertEqual(verificar_raster(ruta)['bloques'],2)
            self.assertFalse(auditar_rasters([ruta],cache))
            with open(ruta,'r+b') as f:
                f.truncate(ruta.stat().st_size-100)
            with rasterio.open(ruta) as src:
                self.assertEqual(src.shape,(1024,512))
            with self.assertRaises(ValueError): verificar_raster(ruta)
            self.assertIn(str(ruta),auditar_rasters([ruta],cache))


if __name__ == '__main__':
    unittest.main()
