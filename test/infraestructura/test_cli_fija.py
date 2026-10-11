"""CLI fija: validación anterior a dependencias y operaciones reales."""
import contextlib
import importlib
import io
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
ENTRADAS = (
    ('src.mantenimiento.recuperar_multicampana', 'recuperar_multicampana.py'),
    ('src.datos.moldear_mascaras', '01_moldear_mascaras.py'),
    ('src.datos.composites_mensuales', '02_composites_mensuales.py'),
    ('src.datos.composites_2019_2020', '02b_composites_2019_2020.py'),
    ('src.datos.generar_arrays_originales', '03_generar_datasets_npz.py'),
)


class TestCLIFija(unittest.TestCase):
    def test_ayuda_y_argumentos_invalidos_antes_de_dependencias(self):
        codigo = """
import importlib.abc, pathlib, runpy, sys
from unittest.mock import patch
root, modo, entrada, argumento = sys.argv[1:]
sys.path.insert(0, root)
class Bloquear(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('numpy', 'rasterio', 'torch') or fullname in (
            'utils.rutas', 'utils.integridad_raster',
            'src.datos.preparar_multicampana', 'src.datos.composites_scl'):
            raise AssertionError('dependencia operativa: ' + fullname)
sys.meta_path.insert(0, Bloquear())
sys.argv = [entrada, argumento]
with patch('pathlib.Path.mkdir', side_effect=AssertionError('mkdir')), patch('pathlib.Path.write_text', side_effect=AssertionError('escritura')):
    if modo == 'modulo':
        runpy.run_module(entrada, run_name='__main__')
    else:
        runpy.run_path(str(pathlib.Path(root) / 'src' / entrada), run_name='__main__')
"""
        with tempfile.TemporaryDirectory() as d:
            for modulo, wrapper in ENTRADAS:
                for modo, entrada in (('modulo', modulo), ('wrapper', wrapper)):
                    for argumento in ('--help', '-h', '--workers', 'inesperado'):
                        with self.subTest(entrada=entrada, argumento=argumento):
                            r = subprocess.run([sys.executable, '-B', '-S', '-c', codigo,
                                                str(ROOT), modo, entrada, argumento],
                                               cwd=d, capture_output=True, text=True, timeout=15)
                            self.assertEqual(r.returncode, 0 if argumento in ('--help', '-h') else 2, r.stderr)
                            self.assertNotIn('AssertionError', r.stderr)
                            if r.returncode == 0:
                                self.assertIn('Sin argumentos', r.stdout)
                            else:
                                self.assertIn('unrecognized arguments', r.stderr)
                            self.assertEqual(list(Path(d).iterdir()), [])

    def test_wrappers_sin_argumentos_delegan_una_vez(self):
        for modulo, wrapper in ENTRADAS:
            with self.subTest(wrapper=wrapper):
                fake = ModuleType(modulo)
                fake.main = Mock()
                fake.ejecutar = Mock()
                fake.ROOT = ROOT
                fake.LOGS = Path('/no-usar')
                fake.cargar_dependencias = Mock()
                fake.reparar = Mock()
                fake._procesar_chunk_ventanas = Mock()
                with patch.dict(sys.modules, {modulo: fake}), patch.object(sys, 'argv', [wrapper]):
                    runpy.run_path(str(ROOT / 'src' / wrapper), run_name='__main__')
                operativo = fake.ejecutar if 'recuperar' in modulo else fake.main
                operativo.assert_called_once_with()
                fake.cargar_dependencias.assert_not_called()
                fake.reparar.assert_not_called()

    def test_imports_con_argumentos_invalidos_no_ejecutan(self):
        with patch.object(sys, 'argv', ['importacion', '--inexistente']), patch('pathlib.Path.mkdir', side_effect=AssertionError('mkdir')), patch('subprocess.run', side_effect=AssertionError('proceso')):
            for modulo, wrapper in ENTRADAS:
                mod = importlib.import_module(modulo)
                anterior = importlib.import_module('src.' + wrapper.removesuffix('.py'))
                self.assertIs(anterior.main, mod.main)

    def test_parser_fijo_acepta_solo_lista_vacia(self):
        from src.soporte.cli_fija import DESCRIPCIONES, validar_entrada_fija
        for nombre in DESCRIPCIONES:
            self.assertEqual(vars(validar_entrada_fija(nombre, [])), {})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as salida:
                validar_entrada_fija(nombre, ['--inexistente'])
            self.assertEqual(salida.exception.code, 2)
