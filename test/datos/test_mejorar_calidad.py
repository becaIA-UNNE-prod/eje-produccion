"""Mantenimiento probado con archivos mínimos y operaciones externas simuladas."""
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


def fixture(base, out):
    out.mkdir(parents=True)
    for year, key in (('2018_2019', '20HNJ_201903'), ('2020_2021', '20HNH_202103')):
        source = base / ('sentinel2_' + year)
        comp = base / ('composites_' + year + '_scl')
        source.mkdir(parents=True); comp.mkdir(parents=True)
        tile, month = key.split('_')
        products = [{'tile': tile, 'date': month + day} for day in ('20', '01', '20', '15')]
        products += [{'tile': 'OTRO', 'date': month + '01'}, {'tile': tile, 'date': '19990101'}]
        (source / 'productos.json').write_text(json.dumps(products))
        for day in ('01', '20'):
            p = source / tile / (month + day) / 'SCL.tif'
            p.parent.mkdir(parents=True); p.write_bytes(b'SCL')
        (source / 'seleccion_scl.json').write_text(json.dumps({key: {'fechas': [month + '01'], 'extra': 7}, 'otro': {'preservar': True}}))
        (comp / 'fechas_usadas.json').write_text(json.dumps({key: {'despejado': .5, 'cobertura': .7}, 'otro': {'preservar': True}}))
        (comp / (key + '_median.tif')).write_bytes(b'original')


def composer(*args, **kwargs):
    command = args[0]
    if 'componer' not in command:
        return
    out = Path(command[command.index('--out') + 1])
    source = Path(command[command.index('--base') + 1])
    year = source.name.removeprefix('sentinel2_')
    key = '20HNJ_201903' if year == '2018_2019' else '20HNH_202103'
    out.mkdir(exist_ok=True)
    (out / (key + '_median.tif')).write_bytes(b'mejorado')
    (out / 'fechas_usadas.json').write_text(json.dumps({key: {'despejado': .6, 'cobertura': .8}}))


class TestMejorarCalidad(unittest.TestCase):
    def test_importacion_inocua(self):
        code = """
import sys, importlib
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import utils.integridad_raster
sys.argv = ['comando']
with patch('pathlib.Path.read_text', side_effect=AssertionError('lectura')), patch('pathlib.Path.write_text', side_effect=AssertionError('escritura')), patch('subprocess.run', side_effect=AssertionError('subprocess')), patch('utils.integridad_raster.auditar_rasters', side_effect=AssertionError('auditoria')):
    importlib.import_module('src.mantenimiento.mejorar_calidad_extension')
    importlib.import_module('src.mejorar_calidad_extension')
"""
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT)], cwd=d, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout, '')
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_promocion_respaldos_y_repeticion(self):
        mod = importlib.import_module('src.mantenimiento.mejorar_calidad_extension')
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base = Path(d) / 'base'; out = Path(d) / 'out'; fixture(base, out)
            with patch.object(mod, 'BASE', base), patch.object(mod.subprocess, 'run', side_effect=composer) as run, patch.object(mod, 'auditar_rasters', return_value=[]) as audit:
                mod.main([str(out)])
                self.assertEqual(run.call_count, 6)
                self.assertEqual(audit.call_count, 4)
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs, {'cwd': mod.ROOT, 'check': True})
                    self.assertEqual(call.args[0][:2], [sys.executable, '-u'])
                mod.main([str(out)])
            reports = json.loads((out / 'mejoras_calidad.json').read_text())
            self.assertEqual(len(reports), 2)
            for year, key in mod.COMPUESTOS:
                comp = base / ('composites_' + year + '_scl')
                source = base / ('sentinel2_' + year)
                self.assertEqual((comp / (key + '_median.tif')).read_bytes(), b'mejorado')
                self.assertEqual((out / (key + '_antes_mejora.tif')).read_bytes(), b'original')
                original = json.loads((out / ('seleccion_antes_mejora_' + year + '.json')).read_text())
                self.assertEqual(len(original[key]['fechas']), 1)
                selected = json.loads((source / 'seleccion_scl.json').read_text())
                self.assertEqual(selected['otro'], {'preservar': True})
                self.assertEqual(selected[key]['extra'], 7)
                self.assertEqual(selected[key]['fechas'], [key.split('_')[1] + '01', key.split('_')[1] + '20'])
                self.assertEqual(json.loads((comp / 'fechas_usadas.json').read_text())['otro'], {'preservar': True})
            self.assertFalse(list(base.rglob('*.tmp')))
            self.assertFalse(list(base.rglob('*.partial.tif')))

    def test_rechaza_empeoramiento_y_fallos_sin_promover(self):
        mod = importlib.import_module('src.mantenimiento.mejorar_calidad_extension')
        for mode in ('despejado', 'cobertura', 'fuentes', 'compuesto', 'subprocess'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                base = Path(d) / 'base'; out = Path(d) / 'out'; fixture(base, out)
                before = {p.relative_to(base): p.read_bytes() for p in base.rglob('*') if p.is_file()}
                def execute(*args, **kwargs):
                    if mode == 'subprocess':
                        raise subprocess.CalledProcessError(1, args[0])
                    composer(*args, **kwargs)
                    if 'componer' in args[0] and mode in ('despejado', 'cobertura'):
                        p = out / 'compuestos_mejorados_2018_2019' / 'fechas_usadas.json'
                        data = json.loads(p.read_text()); data['20HNJ_201903'][mode] = .1
                        p.write_text(json.dumps(data))
                audits = [['fallo']] if mode == 'fuentes' else [[], ['fallo']] if mode == 'compuesto' else [[], []]
                with patch.object(mod.subprocess, 'run', side_effect=execute), patch.object(mod, 'auditar_rasters', side_effect=audits):
                    with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                        mod.mejorar_calidad(base, out)
                self.assertEqual(before, {p.relative_to(base): p.read_bytes() for p in base.rglob('*') if p.is_file()})
                self.assertFalse((out / 'mejoras_calidad.json').exists())
                self.assertFalse(list(out.glob('*antes_mejora*')))

    def test_parser_y_ayuda_no_operan(self):
        mod = importlib.import_module('src.mantenimiento.mejorar_calidad_extension')
        self.assertEqual(mod.crear_parser().parse_args(['/tmp/salida']).salida, Path('/tmp/salida'))
        with patch.object(mod, 'mejorar_calidad', side_effect=AssertionError('mantenimiento')), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as help_exit:
                mod.main(['--help'])
            self.assertEqual(help_exit.exception.code, 0)
            with self.assertRaises(SystemExit) as missing_exit:
                mod.main([])
            self.assertEqual(missing_exit.exception.code, 2)

    def test_compatibilidad_y_shell_sin_argumento(self):
        nuevo = importlib.import_module('src.mantenimiento.mejorar_calidad_extension')
        anterior = importlib.import_module('src.mejorar_calidad_extension')
        self.assertEqual(nuevo.ROOT, ROOT)
        for nombre in ('main', 'crear_parser', 'ejecutar_comando', 'mejorar_compuesto', 'mejorar_calidad'):
            self.assertIs(getattr(anterior, nombre), getattr(nuevo, nombre))
        with tempfile.TemporaryDirectory() as d:
            for ruta in ('src/recuperar_extension_este.sh', 'scripts/mantenimiento/recuperar_extension_este.sh'):
                r = subprocess.run(['bash', str(ROOT / ruta)], cwd=d, capture_output=True, text=True)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('Directorio de resultados requerido', r.stderr)
                self.assertEqual(list(Path(d).iterdir()), [])
