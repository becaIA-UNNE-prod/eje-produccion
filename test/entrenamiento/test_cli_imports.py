import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
trainer = importlib.import_module('src.1_train_cordoba')
figures = importlib.import_module('src.regenerar_figuras_test_actual')


class CliImportsTest(unittest.TestCase):
    def test_imports_no_procesan_argumentos_ni_alteran_semillas(self):
        code = (
            "import sys, pathlib, importlib, torch, numpy as np; "
            "sys.path.insert(0, " + repr(str(ROOT)) + "); "
            "sys.argv = ['prueba', '--argumento-invalido']; "
            "torch_state = torch.get_rng_state().clone(); np_state = np.random.get_state(); "
            "benchmark = torch.backends.cudnn.benchmark; "
            "importlib.import_module('src.1_train_cordoba'); "
            "importlib.import_module('src.regenerar_figuras_test_actual'); "
            "assert torch.equal(torch_state, torch.get_rng_state()); "
            "actual = np.random.get_state(); "
            "assert np_state[0] == actual[0] and np.array_equal(np_state[1], actual[1]) and np_state[2:] == actual[2:]; "
            "assert benchmark == torch.backends.cudnn.benchmark; "
            "assert not list(pathlib.Path.cwd().iterdir())"
        )
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-B', '-c', code], cwd=directory, check=True)

    def test_configuracion_por_ejecucion_no_comparte_pesos(self):
        default = trainer.configurar(trainer.leer_argumentos([]))
        default['PESOS'].append(99)
        fresh = trainer.configurar(trainer.leer_argumentos([]))
        self.assertEqual(len(fresh['PESOS']), 5)
        utae = trainer.configurar(trainer.leer_argumentos(['--modelo', 'utae', '--sin-2017']))
        self.assertEqual(utae['BATCH_SIZE'], 16)
        self.assertTrue(all(not tile.endswith('_2017') for tile in utae['TILES_TRAIN']))
        self.assertEqual(utae['PESOS'], [0., .851, 1.019, .79, 4.245])

    def test_main_entrenador_delega_argumentos_y_configuracion(self):
        with patch.object(trainer, 'entrenar', return_value='ok') as run:
            self.assertEqual(trainer.main(['--modelo', 'utae', '--sin-2017']), 'ok')
        args, config = run.call_args.args
        self.assertEqual(args.modelo, 'utae')
        self.assertEqual(config['BATCH_SIZE'], 16)

    def test_regeneracion_conserva_protocolo_y_exitcode(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)
            protocol = {'preparacion_base': str(out / 'mascaras_base'),
                        'split': {'campanas': ['2019_2020'], 'test': ['20HNJ']},
                        'modelos': {'principal': {'nombre': 'QA'}}}
            (out / 'protocolo.json').write_text(json.dumps(protocol))
            with patch.object(figures, 'figura_extension') as render:
                figures.main(['--salida', directory])
            self.assertEqual(render.call_count, 1)
            self.assertEqual(render.call_args.kwargs['title'], 'QA')
            self.assertEqual(render.call_args.kwargs['maskbase'], out / 'mascaras_base')
            self.assertEqual((out / 'figuras.exitcode').read_text(), '0\n')
            with patch.object(figures, 'figura_extension', side_effect=RuntimeError('fallo')):
                with self.assertRaises(RuntimeError):
                    figures.main(['--salida', directory])
            self.assertEqual((out / 'figuras.exitcode').read_text(), '1\n')


if __name__ == '__main__':
    unittest.main()
