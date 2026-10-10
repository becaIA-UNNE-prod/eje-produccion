"""Fixtures de máscaras fuera del repositorio."""
from pathlib import Path
import importlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin

class MascarasTest(unittest.TestCase):
    def test_variantes_nearest_y_codigos_crudos(self):
        for name, datos, nodata, patron in [
            ('mascaras_mnc_2018_2019', [[0,3],[4,16]], 16, 'etiqueta_mnc1819_A_10m.tif'),
            ('mascaras_nivel3_2017_2018', [[16,17],[18,0]], 0, 'etiqueta_nivel3_1718_A_10m.tif'),
            ('moldear_mascaras', [[16,17],[18,0]], 0, 'moldeada.tif'),
        ]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                mod = importlib.import_module('src.datos.'+name)
                ref = base/'ref'/'A'/'fecha'/'B04.tif'
                ref.parent.mkdir(parents=True)
                meta = dict(driver='GTiff', height=2, width=2, count=1, dtype='uint8',
                            crs='EPSG:4326', transform=from_origin(0,2,1,1), nodata=nodata)
                fuente = base/'fuente.tif'
                with rasterio.open(fuente, 'w', **meta) as dst:
                    dst.write(np.array(datos, dtype=np.uint8), 1)
                with rasterio.open(ref, 'w', **dict(meta, height=4, width=4, transform=from_origin(0,2,.5,.5))) as dst:
                    dst.write(np.zeros((4,4), dtype=np.uint8), 1)
                salida = base/patron
                if name == 'moldear_mascaras':
                    mod.generar_mascara_10m(fuente, ref, salida)
                else:
                    atributo = 'RUTA_MNC' if nodata == 16 else 'RUTA_NIVEL3'
                    with patch.object(mod, atributo, fuente), patch.object(mod, 'DIR_SALIDA', base):
                        if nodata == 16:
                            with patch.object(mod, 'DIR_REF', base/'ref'):
                                mod.main(['--tiles', 'A'])
                        else:
                            mod.main(['--ref', str(base/'ref'), '--tiles', 'A'])
                        with patch('rasterio.open', side_effect=AssertionError('No releer existente')):
                            if nodata == 16:
                                mod.generar_mascara('A')
                            else:
                                mod.generar_mascara('A', base/'ref')
                with rasterio.open(salida) as dst:
                    np.testing.assert_array_equal(dst.read(1), np.repeat(np.repeat(datos,2,axis=0),2,axis=1))
                    self.assertEqual(dst.transform, from_origin(0,2,.5,.5))
                    self.assertEqual(dst.nodata, nodata)
                    self.assertEqual(dst.dtypes, ('uint8',))
                self.assertFalse(salida.with_suffix('.tif.tmp').exists())

    def test_imports_sin_operaciones(self):
        with patch('rasterio.open', side_effect=AssertionError('No abrir raster')), patch('argparse.ArgumentParser.parse_args', side_effect=AssertionError('No parsear')):
            for name in ('moldear_mascaras', 'mascaras_mnc_2018_2019', 'mascaras_nivel3_2017_2018', 'mascaras_mnc_2020_2021'):
                importlib.reload(importlib.import_module('src.datos.'+name))
