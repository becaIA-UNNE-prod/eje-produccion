"""Contratos de las tres comparaciones anteriores trasladadas."""
import contextlib
import importlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
compare = importlib.import_module('src.historico.comparar_modelos')
visual = importlib.import_module('src.historico.comparar_inferencia_etiqueta')
metrics = importlib.import_module('src.historico.metricas_test')


class ModeloSintetico(torch.nn.Module):
    canales = []
    entradas = []

    def __init__(self, channels, classes):
        super().__init__()
        self.classes = classes
        self.canales.append((channels, classes))

    def forward(self, inputs):
        self.entradas.append(inputs.detach().cpu().clone())
        labels = inputs[:, 0].long()
        return torch.nn.functional.one_hot(labels, self.classes).permute(0, 3, 1, 2).float()


class ComparacionesTrasladadasTest(unittest.TestCase):
    def test_wrappers_exports_y_ayuda_desde_otro_cwd(self):
        pairs = [('8_comparar_modelos', compare, ['clase_a_rgb', 'graficar_curvas', 'graficar_mapa']),
                 ('9_comparar_inferencia_etiqueta', visual, ['clase_a_rgb', 'leer_downsampled', 'REMAP_5']),
                 ('10_metricas_test', metrics, ['confusion_tile', 'REMAP_5', 'NOMBRES'])]
        with tempfile.TemporaryDirectory() as directory:
            for name, module, names in pairs:
                wrapper = importlib.import_module('src.' + name)
                for export in names + ['main', 'crear_parser']:
                    self.assertIs(getattr(wrapper, export), getattr(module, export))
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'src' / (name + '.py')), '--help'], cwd=directory, text=True, capture_output=True, check=True)
                self.assertIn('usage:', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_precedencia_de_salidas_y_run_dir(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from utils.rutas import RUNS_ROOT
from src.historico import comparar_modelos as c, comparar_inferencia_etiqueta as v, metricas_test as m
base = Path(sys.argv[2])
expected = base / 'run' if sys.argv[3] == 'run' else RUNS_ROOT / 'legacy_octabr'
assert c.crear_parser().parse_args([]).salida_dir == str(expected / 'figuras')
assert v.crear_parser().parse_args([]).dir_pred == expected / 'predicciones'
assert v.crear_parser().parse_args([]).dir_salida == expected / 'figuras'
assert m.crear_parser().parse_args([]).dir_pred == expected / 'predicciones'
assert c.crear_parser().parse_args(['--salida_dir', str(base / 'explicit')]).salida_dir == str(base / 'explicit')
assert v.crear_parser().parse_args(['--dir-pred', str(base / 'pred'), '--dir-salida', str(base / 'out')]).dir_salida == base / 'out'
assert m.crear_parser().parse_args(['--dir-pred', str(base / 'pred')]).dir_pred == base / 'pred'
"""
        with tempfile.TemporaryDirectory() as directory:
            for mode in ['legacy', 'run']:
                env = dict(os.environ)
                env.pop('CORDOBA_RUN_DIR', None)
                if mode == 'run':
                    env['CORDOBA_RUN_DIR'] = str(Path(directory) / 'run')
                subprocess.run([sys.executable, '-B', '-c', code, str(ROOT), directory, mode], cwd=directory, env=env, check=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_del_8_conserva_defaults_y_argumentos(self):
        default = compare.crear_parser().parse_args([])
        self.assertEqual((default.tile, default.prefijo, default.n_clases), ('20HLJ', 'dataset_', 5))
        with tempfile.TemporaryDirectory() as directory, patch.object(compare, 'graficar_curvas') as curves, patch.object(compare, 'graficar_mapa') as maps:
            out = Path(directory) / 'figuras'
            compare.main(['--tile', 'A', '--dir_train', '/tmp/train', '--prefijo', 'dataset_T', '--n_clases', '6', '--salida_dir', str(out)])
            self.assertTrue(out.is_dir())
            curves.assert_called_once_with(str(out / 'comparacion_curvas.png'))
            maps.assert_called_once_with('A', '/tmp/train', 'dataset_T', 6, str(out / 'comparacion_mapa_A.png'))

    def test_mosaico_npz_modelos28_y_ventanas_descartadas(self):
        ModeloSintetico.canales = []
        ModeloSintetico.entradas = []
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            x = np.zeros((5, 28, 256, 256), dtype=np.float16)
            y = np.zeros((5, 256, 256), dtype=np.uint8)
            for i in range(5):
                x[i] = i % 4 + 1
                y[i] = i % 4 + 1
            np.savez(base / 'dataset_A.npz', X=x, Y=y)
            models = {}
            for name in compare.MODELOS:
                exp = base / name
                exp.mkdir()
                (exp / 'best_model.pth').touch()
                np.save(exp / 'historial.npy', [[1, .5, .6, .7, .8], [2, .4, .7, .6, .9]])
                models[name] = (ModeloSintetico, str(exp))
            with patch.object(compare, 'MODELOS', models), patch.object(compare.torch, 'load', return_value={}), patch.object(compare.torch.cuda, 'is_available', return_value=False), patch.object(compare.torch.cuda, 'empty_cache'), contextlib.redirect_stdout(io.StringIO()):
                compare.graficar_curvas(base / 'curvas.png')
                compare.graficar_mapa('A', directory, 'dataset_', 5, base / 'mapa.png')
            self.assertEqual(ModeloSintetico.canales, [(28, 5)] * 4)
            for inputs in ModeloSintetico.entradas:
                self.assertEqual(inputs.dtype, torch.float32)
                np.testing.assert_array_equal(inputs.numpy(), x[:4].astype(np.float32))
            for name in ['curvas.png', 'mapa.png']:
                self.assertTrue((base / name).read_bytes().startswith(b'\x89PNG'))

    def test_confusion_bloques1024_y_error_pred0(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = np.full((1025, 3), 10, dtype=np.uint8)
            pred = np.full_like(raw, 2)
            raw[0, 0] = 255
            pred[1024, 0] = 0
            pred[1024, 1] = 3
            for name, values in [('etiqueta_mnc_A_10m.tif', raw), ('prediccion_A.tif', pred)]:
                with rasterio.open(base / name, 'w', driver='GTiff', width=3, height=1025, count=1, dtype='uint8', crs='EPSG:32720', transform=from_origin(100, 200, 10, 10)) as dst:
                    dst.write(values, 1)
            args = metrics.crear_parser().parse_args(['--modelo', 'unet', '--tiles', 'A', '--dir-pred', directory, '--dir_mascaras', directory])
            confusion = metrics.confusion_tile('A', args)
            self.assertEqual(confusion.sum(), raw.size - 1)
            self.assertEqual((confusion[2, 0], confusion[2, 3], confusion[2, 2]), (1, 1, raw.size - 3))


if __name__ == '__main__':
    unittest.main()
