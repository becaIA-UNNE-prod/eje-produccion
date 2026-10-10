"""Entradas TEST vigente: preflight sin inferencia y espera de figuras."""
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
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]


class TestCLITestActual(unittest.TestCase):
    def test_wrappers_ayuda_e_importacion_inocua(self):
        code = """
import sys, importlib
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import numpy, torch, rasterio, src.datos.campanas, matplotlib.pyplot
sys.argv = ['prueba', '--inexistente']
with patch('numpy.load', side_effect=AssertionError('arrays')), patch('torch.load', side_effect=AssertionError('pesos')), patch('rasterio.open', side_effect=AssertionError('raster')), patch('time.sleep', side_effect=AssertionError('espera')):
    for viejo, nuevo in [('src.evaluar_test_actual','src.evaluacion.test_actual'), ('src.regenerar_figuras_test_actual','src.figuras.regenerar_test_actual')]:
        a=importlib.import_module(viejo);b=importlib.import_module(nuevo)
        assert a.main is b.main and a.crear_parser is b.crear_parser
"""
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT)], cwd=d, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for viejo, nuevo in [('evaluar_test_actual.py','src.evaluacion.test_actual'), ('regenerar_figuras_test_actual.py','src.figuras.regenerar_test_actual')]:
                for argv, cwd in (([str(ROOT / 'src' / viejo)], d), (['-m', nuevo], ROOT)):
                    r = subprocess.run([sys.executable, '-B', *argv, '--help'], cwd=cwd, capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertIn('--salida', r.stdout)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_verificar_respeta_split_modelo24_y_no_crea_salida(self):
        mod = importlib.import_module('src.evaluacion.test_actual')
        class Modelo:
            def __init__(self, canales, clases):
                self.canales, self.clases = canales, clases
                if (canales, clases) != (24, 5): raise AssertionError('contrato modelo')
            def load_state_dict(self, pesos): pass
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as stdout:
            base = Path(d); exp = base / 'exp'; exp.mkdir(); salida = base / 'salida'
            path = base / 'composite.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=2, height=2, count=4, dtype='uint16', crs='EPSG:32720', transform=from_origin(500000,6200000,10,10)) as dst:
                dst.write(np.ones((4,2,2), dtype=np.uint16))
            cfg = mod.cargar_config()
            self.assertEqual(cfg['test'], ['20HNJ','20HNH'])
            with patch.object(mod, 'preflight', return_value=base / 'mascaras') as preflight, patch.object(mod, 'files', return_value=[path] * 6), patch.object(mod, 'constructor', return_value=Modelo), patch.object(mod.torch, 'load', return_value={}), patch.object(mod, 'sha256', return_value='fixture'), patch.object(mod, 'inferir_tile', side_effect=AssertionError('inferencia')), patch.object(mod, 'auditar_rasters', side_effect=AssertionError('auditoria operativa')):
                mod.main(['--exp',str(exp),'--salida',str(salida),'--sin-comparar','--verificar'])
            self.assertEqual(preflight.call_args.args[0]['test'], cfg['test'])
            self.assertFalse(salida.exists())
            self.assertIn("['20HNJ', '20HNH']", stdout.getvalue())
            self.assertIn('PREFLIGHT TEST ACTUAL OK', stdout.getvalue())

    def test_regenerador_espera_fallo_y_timeout(self):
        mod = importlib.import_module('src.figuras.regenerar_test_actual')
        for modo in ('correcto','fallo','timeout'):
            with self.subTest(modo=modo), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                out=Path(d)
                protocolo={'preparacion_base':str(out/'base'),'split':{'campanas':['2019_2020'],'test':['20HNJ','20HNH']},'modelos':{'principal':{'nombre':'fixture'}}}
                (out/'protocolo.json').write_text(json.dumps(protocolo))
                if modo != 'timeout':(out/'inferencia.exitcode').write_text('0' if modo=='correcto' else '1')
                with patch.object(mod.time,'sleep') as sleep, patch.object(mod,'files',return_value=[Path('/tmp/fixture')] * 6), patch.object(mod,'figura_extension') as figura:
                    if modo=='correcto':
                        mod.main(['--salida',d,'--esperar'])
                        self.assertEqual(figura.call_count,2);sleep.assert_not_called()
                    else:
                        with self.assertRaises(TimeoutError if modo=='timeout' else RuntimeError):mod.main(['--salida',d,'--esperar'])
                        figura.assert_not_called()
                        self.assertEqual(sleep.call_count,1080 if modo=='timeout' else 0)
                self.assertEqual((out/'figuras.exitcode').read_text(),'0\n' if modo=='correcto' else '1\n')
