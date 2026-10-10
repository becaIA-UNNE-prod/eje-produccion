"""Coordinador de cachés con señales, esperas y lanzamientos simulados."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class TestOptimizarLectura(unittest.TestCase):
    def test_importacion_compatibilidad_y_ayuda(self):
        code = """
import sys, importlib
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import utils.cache_lectura, utils.multicampana
sys.argv = ['comando', '--inexistente']
with patch('os.kill', side_effect=AssertionError('senal')), patch('subprocess.run', side_effect=AssertionError('ejecucion')), patch('subprocess.check_output', side_effect=AssertionError('relanzamiento')), patch('time.sleep', side_effect=AssertionError('espera')):
    nuevo = importlib.import_module('src.mantenimiento.optimizar_lectura_multicampana')
    viejo = importlib.import_module('src.optimizar_lectura_multicampana')
    for nombre in ('main','preparar','copiar_verificado','detener','proceso_actual','ejecutar'):
        assert getattr(viejo,nombre) is getattr(nuevo,nombre)
"""
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT)], cwd=d, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for command, cwd in (([str(ROOT / 'src/optimizar_lectura_multicampana.py')], d), (['-m', 'src.mantenimiento.optimizar_lectura_multicampana'], ROOT)):
                help_result = subprocess.run([sys.executable, '-B', *command, '--help'], cwd=cwd, capture_output=True, text=True)
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertIn('--solo-copiar', help_result.stdout)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_solo_copiar_y_argumentos_incompletos(self):
        mod = importlib.import_module('src.mantenimiento.optimizar_lectura_multicampana')
        with tempfile.TemporaryDirectory() as d, patch.object(mod, 'preparar') as preparar, patch.object(mod, 'detener') as detener, patch.object(mod.subprocess, 'run') as ejecutar, contextlib.redirect_stderr(io.StringIO()):
            mod.main(['--dir-exp', d, '--solo-copiar'])
            preparar.assert_called_once_with(Path(d).resolve(), Path(d).resolve() / 'cache_lectura.json')
            with self.assertRaises(SystemExit) as error:
                mod.main(['--dir-exp', d])
            self.assertEqual(error.exception.code, 2)
            detener.assert_not_called(); ejecutar.assert_not_called()
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_checkpoint_finalizando_no_detiene_y_copia_usa_modulo(self):
        mod = importlib.import_module('src.mantenimiento.optimizar_lectura_multicampana')
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            exp = Path(d); (exp / 'ultimo_checkpoint.pth').write_bytes(b'checkpoint')
            with patch.object(mod.subprocess, 'run') as copiar, patch.object(mod, 'cargar_cache'), patch('torch.load', return_value={'epoch': 100, 'sin_mejora': 0}), patch.object(mod, 'detener') as detener, patch.object(mod.subprocess, 'check_output') as relanzar:
                mod.main(['--dir-exp', d, '--pid-train', '12345', '--sin-esperar-epoca'])
                self.assertEqual(copiar.call_args.args[0], ['ionice', '-c2', '-n7', sys.executable, '-u', '-m', 'src.mantenimiento.optimizar_lectura_multicampana', '--dir-exp', str(exp), '--solo-copiar'])
                self.assertEqual(copiar.call_args.kwargs, {'check': True, 'cwd': ROOT})
                detener.assert_not_called(); relanzar.assert_not_called()
            self.assertEqual(json.loads((exp / 'optimizacion_lectura.json').read_text())['estado'], 'cache_lista_corrida_finalizando')

    def test_corrida_terminada_y_timeout_no_detienen(self):
        mod = importlib.import_module('src.mantenimiento.optimizar_lectura_multicampana')
        for modo in ('terminada', 'timeout'):
            with self.subTest(modo=modo), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                exp = Path(d); (exp / 'ultimo_checkpoint.pth').write_bytes(b'checkpoint')
                with patch.object(mod.subprocess, 'run'), patch.object(mod, 'cargar_cache'), patch.object(mod, 'proceso_actual', return_value=modo != 'terminada'), patch.object(mod.time, 'monotonic', side_effect=[0, 10801]), patch.object(mod, 'detener') as detener, patch.object(mod.subprocess, 'check_output') as relanzar:
                    if modo == 'timeout':
                        with self.assertRaises(TimeoutError): mod.main(['--dir-exp', d, '--pid-train', '12345'])
                    else:
                        mod.main(['--dir-exp', d, '--pid-train', '12345'])
                        self.assertEqual(json.loads((exp / 'optimizacion_lectura.json').read_text())['estado'], 'cache_lista_corrida_terminada')
                    detener.assert_not_called(); relanzar.assert_not_called()

    def test_pid_ajeno_rechazado_sin_senales(self):
        mod = importlib.import_module('src.mantenimiento.optimizar_lectura_multicampana')
        with patch.object(Path, 'exists', return_value=True), patch.object(Path, 'read_bytes', return_value=b'otro_proceso'), patch.object(mod.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'proceso ajeno'):
                mod.detener(12345, Path('/tmp/exp'))
            kill.assert_not_called()

    def test_shell_sin_argumentos_desde_fuera_del_repo(self):
        with tempfile.TemporaryDirectory() as d:
            for f in ('src/optimizar_lectura_multicampana.sh', 'scripts/mantenimiento/optimizar_lectura_multicampana.sh'):
                r = subprocess.run(['bash', str(ROOT / f)], cwd=d, capture_output=True, text=True)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('Directorio de experimento requerido', r.stderr)
                self.assertEqual(list(Path(d).iterdir()), [])

    def test_relanzamiento_simulado_usa_ultimo_checkpoint(self):
        mod = importlib.import_module('src.mantenimiento.optimizar_lectura_multicampana')
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            exp = Path(d)
            for name in ('ultimo_checkpoint.pth', 'train.log', 'train.exitcode', 'fin.txt'):
                (exp / name).write_bytes(b'fixture')
            guardar_original = mod.guardar
            registros = []
            def guardar(ruta, datos):
                registros.append((ruta, datos))
                if str(ruta).startswith(d): guardar_original(ruta, datos)
            existe_original = Path.exists
            def existe(ruta):
                return True if ruta == Path('/proc/67890') else existe_original(ruta)
            with patch.object(mod.subprocess, 'run'), patch.object(mod, 'cargar_cache'), patch('torch.load', side_effect=[{'epoch': 7, 'sin_mejora': 0}, {'epoch': 8, 'sin_mejora': 0}]), patch.object(mod, 'detener') as detener, patch.object(mod.time, 'sleep'), patch.object(mod.subprocess, 'check_output', return_value='67890\n') as relanzar, patch.object(mod, 'guardar', side_effect=guardar), patch.object(Path, 'exists', existe), patch.object(mod.os, 'kill', side_effect=AssertionError('senal real')):
                mod.main(['--dir-exp', d, '--pid-train', '12345', '--sin-esperar-epoca'])
                detener.assert_called_once_with(12345, exp)
                command = relanzar.call_args.args[0]
                self.assertIn('src/entrenar_multicampana_unet.sh', command)
                self.assertIn('--reanudar', command)
                self.assertIn('checkpoint_antes_precarga_paralela.pth', command)
                self.assertIn('--bloques-pool 16 --workers 4 --prefetch 2', command)
                self.assertEqual(relanzar.call_args.kwargs, {'shell': True, 'cwd': ROOT, 'text': True})
            record = json.loads((exp / 'optimizacion_lectura.json').read_text())
            self.assertEqual((record['estado'], record['epoch_recuperada'], record['pid']), ('relanzado', 8, 67890))
            self.assertTrue((exp / 'train_antes_paralelo_epoch8.log').exists())
            self.assertTrue((exp / 'train.exitcode.antes_paralelo').exists())
            self.assertTrue((exp / 'fin.txt.antes_paralelo').exists())
            self.assertEqual(registros[-1][0], Path('/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/ultimo_entrenamiento_unet.json'))
