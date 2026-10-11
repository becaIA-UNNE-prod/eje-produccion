"""Contratos de evaluación MNC anterior y comparación de variantes."""
import contextlib
import importlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
evaluation = importlib.import_module('src.historico.evaluar_contra_mnc')
comparison = importlib.import_module('src.historico.comparar_con_1718')


def raster(path, values, x=100):
    with rasterio.open(path, 'w', driver='GTiff', width=values.shape[1], height=values.shape[0], count=1, dtype='uint8', crs='EPSG:32720', transform=from_origin(x, 200, 10, 10)) as dst:
        dst.write(values, 1)


class EvaluacionComparacionMncTest(unittest.TestCase):
    def test_imports_sin_operaciones_ni_cambios_de_estilo(self):
        code = """
import importlib, sys, numpy, rasterio, torch, matplotlib.pyplot as plt
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
sys.argv = ['prueba', '--invalido']
import matplotlib
matplotlib.use("Agg")
before = dict(plt.rcParams)
with patch('numpy.load', side_effect=AssertionError('datos')), patch('rasterio.open', side_effect=AssertionError('raster')), patch('torch.load', side_effect=AssertionError('pesos')), patch('matplotlib.pyplot.subplots', side_effect=AssertionError('figura')):
    for name in ['src.historico.evaluar_contra_mnc', 'src.historico.comparar_con_1718', 'src.11_evaluar_contra_mnc', 'src.12_comparar_con_1718']:
        importlib.import_module(name)
assert before == dict(plt.rcParams)
"""
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-B', '-c', code, str(ROOT)], cwd=directory, check=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_ayuda_exports_y_cli_sin_efectos(self):
        pairs = [('11_evaluar_contra_mnc', evaluation, ['matriz', 'metricas', 'graficar', 'LEYENDAS']), ('12_comparar_con_1718', comparison, ['series', 'leer_reducido', 'a_rgb', 'VARIANTES'])]
        with tempfile.TemporaryDirectory() as directory:
            for name, module, exports in pairs:
                wrapper = importlib.import_module('src.' + name)
                for export in exports + ['main', 'crear_parser']:
                    self.assertIs(getattr(wrapper, export), getattr(module, export))
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'src' / (name + '.py')), '--help'], cwd=directory, capture_output=True, text=True, check=True)
                self.assertIn('usage:', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])
        args = evaluation.crear_parser().parse_args(['--leyenda', '1819', '--dir-pred', '/tmp/pred', '--dir-salida', '/tmp/out'])
        self.assertEqual(args.modelos, ['unet', 'fcrtransunet'])
        self.assertEqual(args.tiles[-2:], ['20HLK', '20HLJ'])
        self.assertEqual(comparison.crear_parser().parse_args([]).comparacion, '1718')

    def test_matriz_excluye_pred0_y_exige_grilla(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = np.array([[10, 10, 11], [12, 16, 255]], dtype=np.uint8)
            pred = np.array([[2, 0, 3], [1, 4, 2]], dtype=np.uint8)
            raster(base / 'raw.tif', raw)
            raster(base / 'pred.tif', pred)
            cm = evaluation.matriz(base / 'pred.tif', base / 'raw.tif', evaluation.REMAP_2020)
            np.testing.assert_array_equal(cm, np.eye(4, dtype=np.int64))
            self.assertEqual(evaluation.metricas(cm)['OA'], 1.)
            self.assertEqual(evaluation.metricas(np.zeros((4, 4)))['pixeles'], 0)
            raster(base / 'otra.tif', raw, x=110)
            with self.assertRaisesRegex(ValueError, 'grilla distinta'):
                evaluation.matriz(base / 'pred.tif', base / 'otra.tif', evaluation.REMAP_2020)
        self.assertEqual(evaluation.REMAP_1819[0], 2)
        self.assertEqual(evaluation.REMAP_2020[0], 0)

    def test_variantes_repetidas_y_default_salida(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(comparison, 'Y3', Path(directory)), patch.object(comparison, 'VARIANTES', comparison.VARIANTES), patch.object(comparison, 'EVALS', comparison.EVALS), patch.object(comparison, 'REF', comparison.REF), patch.object(comparison, 'NUEVA', comparison.NUEVA), patch.object(comparison, 'graficar_metricas', return_value=[]) as metrics, patch.object(comparison, 'graficar_kappa_tiles'), patch.object(comparison, 'graficar_curvas'), patch.object(comparison, 'graficar_mapa'), contextlib.redirect_stdout(io.StringIO()):
            wrapper = importlib.import_module('src.12_comparar_con_1718')
            for variant in ['scl', '1718', 'scl']:
                comparison.main(['--comparacion', variant, '--tiles-mapa'])
                self.assertIs(wrapper.VARIANTES, comparison.COMPARACIONES[variant])
                self.assertTrue((Path(directory) / ('comparacion_' + variant)).is_dir())
            self.assertEqual(metrics.call_count, 4)
            self.assertEqual(comparison.REF, '02c')

    def test_estilo_se_restaura_incluso_ante_fallo(self):
        before = dict(plt.rcParams)
        def fail(*args, **kwargs):
            self.assertEqual(plt.rcParams['font.size'], 9)
            raise RuntimeError('render')
        with tempfile.TemporaryDirectory() as directory, patch.object(comparison, 'cargar_resumen', return_value={}), patch.object(comparison.plt, 'subplots', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'render'):
                comparison.graficar_metricas('2018_2019', Path(directory))
        self.assertEqual(before, dict(plt.rcParams))

    def test_lectura_nearest_rgb_y_matriz_png(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = np.tile(np.array([[0, 1, 4, 255]], dtype=np.uint8), (16, 4))
            raster(base / 'raw.tif', raw)
            reduced = comparison.leer_reducido(base / 'raw.tif', 8, comparison.REMAP_1819)
            self.assertEqual(reduced.shape, (2, 2))
            np.testing.assert_array_equal(comparison.a_rgb(np.array([[0, 2]], dtype=np.uint8)), [[[255, 255, 255], [255, 165, 0]]])
            evaluation.graficar(np.eye(4), 'MNC', base / 'matrix.png')
            self.assertTrue((base / 'matrix.png').read_bytes().startswith(b'\x89PNG'))


if __name__ == '__main__':
    unittest.main()
