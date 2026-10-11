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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
verifier = importlib.import_module('src.historico.verificar_pipeline')
finetune = importlib.import_module('src.entrenamiento.finetune_manfredi')


class PasoUnoTest(unittest.TestCase):
    def test_imports_sin_efectos_y_exports_compatibles(self):
        code = """
import importlib, sys, torch, numpy as np
from unittest.mock import patch
sys.path.insert(0, ROOT)
sys.argv = ['prueba', '--invalido']
with patch('os.makedirs', side_effect=AssertionError('mkdir')), patch('torch.manual_seed', side_effect=AssertionError('seed')), patch('numpy.random.seed', side_effect=AssertionError('seed')), patch('rasterio.open', side_effect=AssertionError('raster')), patch('numpy.load', side_effect=AssertionError('dataset')), patch('torch.load', side_effect=AssertionError('pesos')):
    for old, new, names in [('src.0_verificar_pipeline', 'src.historico.verificar_pipeline', ['main', 'REMAP_MNC', 'verificar_pesos']), ('src.1b_finetune_manfredi', 'src.entrenamiento.finetune_manfredi', ['main', 'finetune', 'evaluar', 'calcular_correctos_total'])]:
        implementation = importlib.import_module(new)
        wrapper = importlib.import_module(old)
        for name in names:
            assert getattr(wrapper, name) is getattr(implementation, name)
""".replace('ROOT', repr(str(ROOT)))
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-B', '-c', code], cwd=directory, check=True)

    def test_ayuda_wrappers_desde_otro_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in ['0_verificar_pipeline.py', '1b_finetune_manfredi.py']:
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'src' / script), '--help'], cwd=directory, capture_output=True, text=True, check=True)
                self.assertIn('usage:', result.stdout)

    def test_etapas_y_tiles_delegados(self):
        with patch.object(verifier, 'verificar_pesos') as weights, patch.object(verifier, 'verificar_descarga') as download, contextlib.redirect_stdout(io.StringIO()):
            verifier.main(['--etapa', 'pesos', '--tiles_train', '20HMK', '20HLG'])
        weights.assert_called_once_with(['20HMK', '20HLG'])
        download.assert_not_called()
        with patch.object(finetune, 'finetune', return_value='ok') as run:
            self.assertEqual(finetune.main([]), 'ok')
            with self.assertRaises(SystemExit):
                finetune.main(['--invalido'])
        run.assert_called_once_with()

    def test_npz_pesos_y_contratos(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            labels = np.array([0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5], dtype=np.uint8)
            np.savez(base / 'dataset_mnc_20HMK.npz', X=np.zeros((1, 28, 2, 2), dtype=np.float16), Y=labels)
            (base / 'corrupto.npz').write_bytes(b'invalido')
            output = io.StringIO()
            with patch.object(verifier, 'DIR_DATASETS', base), contextlib.redirect_stdout(output):
                self.assertEqual(verifier.verificar_datasets(), ['dataset_mnc_20HMK.npz'])
                verifier.verificar_pesos(['20HMK'])
            expected = [0.0] + [round(float(np.sqrt(15 / (5 * count))), 3) for count in range(1, 6)]
            self.assertIn('pesos = ' + str(expected), output.getvalue())
            self.assertIn('CORRUPTO', output.getvalue())
        self.assertEqual(verifier.MESES_FIJOS, ['201910', '201911', '201912', '202001', '202002', '202003', '202004'])
        self.assertEqual(finetune.TILE_MANFREDI, '20HMK')
        self.assertEqual((finetune.NUM_CLASSES, finetune.FRACCION_VAL, finetune.SEED), (5, .15, 42))

    def test_metricas_y_inicializacion_solo_al_ejecutar(self):
        logits = torch.tensor([[[[9., 0., 0.]], [[0., 9., 9.]], [[0., 0., 0.]]]])
        labels = torch.tensor([[[0, 1, 2]]])
        correct, total = finetune.calcular_correctos_total(logits, labels)
        self.assertEqual((correct.item(), total.item()), (1, 2))
        model = torch.nn.Identity()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss, accuracy = finetune.evaluar(model, [(logits, labels)], criterion, torch.device('cpu'))
        self.assertAlmostEqual(loss, criterion(logits, labels).item())
        self.assertEqual(accuracy, .5)
        with patch.object(finetune.os, 'makedirs') as mkdir, patch.object(finetune.torch, 'manual_seed') as torch_seed, patch.object(finetune.np.random, 'seed') as numpy_seed, patch.object(finetune.torch, 'device', side_effect=RuntimeError('detener antes de datos')):
            with self.assertRaisesRegex(RuntimeError, 'detener antes de datos'):
                finetune.finetune()
        mkdir.assert_called_once_with(finetune.DIR_EXP, exist_ok=True)
        torch_seed.assert_called_once_with(42)
        numpy_seed.assert_called_once_with(42)


if __name__ == '__main__':
    unittest.main()
