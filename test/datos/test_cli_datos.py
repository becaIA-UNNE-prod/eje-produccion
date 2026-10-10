"""Contratos de las CLI de pesos y conversión, con datos sintéticos."""
import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class TestCLIDatos(unittest.TestCase):
    def test_importacion_sin_operaciones(self):
        codigo = """
import sys, importlib
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import numpy
sys.argv = ['comando', '--inexistente']
with patch('numpy.load', side_effect=AssertionError('lectura')), patch('numpy.save', side_effect=AssertionError('escritura')), patch('pathlib.Path.glob', side_effect=AssertionError('inventario')), patch('pathlib.Path.unlink', side_effect=AssertionError('borrado')):
    importlib.import_module('src.entrenamiento.calcular_pesos')
    importlib.import_module('src.datos.convertir_datasets_a_mmap')
    importlib.import_module('src.0b_calcular_pesos')
    importlib.import_module('src.03b_convertir_datasets_a_mmap')
"""
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, '-B', '-c', codigo, str(ROOT)], cwd=d, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout, '')
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_pesos_repetidos_train_y_faltantes(self):
        mod = importlib.import_module('src.entrenamiento.calcular_pesos')
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for tile in mod.SPLIT['train']:
                np.save(base / f'dataset_{tile}_Y.npy', [0, 1, 2, 3, 4])
            for tile in mod.SPLIT['val'] + mod.SPLIT['test']:
                np.save(base / f'dataset_{tile}_Y.npy', [4] * 100)
            salidas = []
            for _ in range(2):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    mod.main(['--dirs', d])
                salidas.append(out.getvalue())
            self.assertEqual(*salidas)
            self.assertIn('PESOS_ARGS 0.0 1.0 1.0 1.0 1.0', salidas[0])
            (base / f"dataset_{mod.SPLIT['train'][0]}_Y.npy").unlink()
            with self.assertRaises(FileNotFoundError):
                mod.main(['--dirs', d])
            with patch.object(mod, 'DIR', base), contextlib.redirect_stdout(io.StringIO()) as out:
                rutas = mod.seleccionar_rutas()
                self.assertEqual(len(rutas), len(mod.TILES_TRAIN))
                mod.main([])
            self.assertIn('no existe, salteado', out.getvalue())

    def test_conversion_y_omision_no_borran_original_existente(self):
        mod = importlib.import_module('src.datos.convertir_datasets_a_mmap')
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base = Path(d); p = base / 'datos.npz'
            x = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
            y = np.arange(8, dtype=np.uint8).reshape(2, 4)
            np.savez_compressed(p, X=x, Y=y, remap_aplicado=True)
            mod.main([d])
            for name, expected in [('datos_X.npy', x), ('datos_Y.npy', y)]:
                actual = np.load(base / name, mmap_mode='r')
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, expected.dtype)
            mod.main([d, '--borrar-original'])
            self.assertTrue(p.exists())
            (base / 'datos_Y.npy').unlink()
            mod.main([d, '--borrar-original'])
            self.assertFalse(p.exists())
            self.assertFalse(list(base.glob('*.tmp')))

    def test_conversion_rechaza_incompatibles_y_remap(self):
        mod = importlib.import_module('src.datos.convertir_datasets_a_mmap')
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base = Path(d)
            np.savez(base / 'longitudes.npz', X=np.zeros((2, 1)), Y=np.zeros((1, 1)))
            np.savez(base / 'sin_remap.npz', X=np.zeros((1, 1)), Y=np.zeros((1, 1)), remap_aplicado=False)
            mod.main([d, '--borrar-original'])
            self.assertEqual({p.name for p in base.iterdir()}, {'longitudes.npz', 'sin_remap.npz'})

    def test_verificacion_fallida_preserva_origen_y_limpia_temporales(self):
        mod = importlib.import_module('src.datos.convertir_datasets_a_mmap')
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            p = Path(d) / 'datos.npz'
            np.savez(p, X=np.zeros((1, 1)), Y=np.zeros((1, 1)))
            with patch.object(mod.np, 'array_equal', return_value=False):
                mod.main([d, '--borrar-original'])
            self.assertEqual(list(Path(d).iterdir()), [p])

    def test_sin_npz_falla(self):
        mod = importlib.import_module('src.datos.convertir_datasets_a_mmap')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, 'No hay .npz'):
                mod.main([d])

    def test_wrappers_delegan_y_cli_desde_otro_directorio(self):
        pesos = importlib.import_module('src.entrenamiento.calcular_pesos')
        conversion = importlib.import_module('src.datos.convertir_datasets_a_mmap')
        for anterior, nuevo, nombres in (
            ('src.0b_calcular_pesos', pesos, ('main', 'crear_parser', 'contar_rutas', 'seleccionar_rutas', 'imprimir_pesos')),
            ('src.03b_convertir_datasets_a_mmap', conversion, ('main', 'crear_parser', 'convertir_npz')),
        ):
            wrapper = importlib.import_module(anterior)
            for nombre in nombres:
                self.assertIs(getattr(wrapper, nombre), getattr(nuevo, nombre))
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for tile in pesos.SPLIT['train']:
                np.save(base / f'dataset_{tile}_Y.npy', [0, 1, 2, 2, 3, 4])
            comandos = [
                ([str(ROOT / 'src/0b_calcular_pesos.py')], base),
                (['-m', 'src.entrenamiento.calcular_pesos'], ROOT),
            ]
            resultados = [subprocess.run([sys.executable, '-B', *cmd, '--dirs', d], cwd=cwd, capture_output=True, text=True) for cmd, cwd in comandos]
            self.assertEqual(resultados[0].returncode, 0, resultados[0].stderr)
            self.assertEqual([(r.returncode, r.stdout, r.stderr) for r in resultados], [(0, resultados[0].stdout, '')] * 2)
            archivos = []
            salidas = []
            for cmd, cwd in (([str(ROOT / 'src/03b_convertir_datasets_a_mmap.py')], base), (['-m', 'src.datos.convertir_datasets_a_mmap'], ROOT)):
                for p in base.glob('datos*'):
                    p.unlink()
                np.savez_compressed(base / 'datos.npz', X=np.arange(8).reshape(2, 4), Y=np.arange(4).reshape(2, 2), remap_aplicado=True)
                r = subprocess.run([sys.executable, '-B', *cmd, d, '--borrar-original'], cwd=cwd, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                salidas.append((r.stdout, r.stderr))
                archivos.append({p.name: p.read_bytes() for p in base.glob('datos*')})
            self.assertEqual(*salidas)
            self.assertEqual(*archivos)
            self.assertEqual(set(archivos[0]), {'datos_X.npy', 'datos_Y.npy'})
