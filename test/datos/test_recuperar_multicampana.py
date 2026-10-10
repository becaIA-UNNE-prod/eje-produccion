"""Recuperación con dependencias simuladas y archivos mínimos en /tmp."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def escribir_json(ruta, datos):
    ruta.write_text(json.dumps(datos))


class TestRecuperarMulticampana(unittest.TestCase):
    def test_importacion_sin_cargar_comandos(self):
        codigo = """
import sys, importlib
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import utils.integridad_raster, utils.rutas
original = importlib.import_module
def cargar(nombre, *args):
    if nombre in ('src.preparar_multicampana', 'src.datos.composites_scl'):
        raise AssertionError('carga de comando')
    return original(nombre, *args)
with patch('importlib.import_module', side_effect=cargar), patch('subprocess.run', side_effect=AssertionError('ejecucion')), patch('utils.integridad_raster.auditar_rasters', side_effect=AssertionError('auditoria')):
    nuevo = importlib.import_module('src.mantenimiento.recuperar_multicampana')
    viejo = importlib.import_module('src.recuperar_multicampana')
    assert viejo.main is nuevo.main and viejo.reparar is nuevo.reparar
    assert nuevo.ROOT == __import__('pathlib').Path(sys.argv[1])
"""
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, '-B', '-c', codigo, str(ROOT)], cwd=d, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_reparacion_respalda_e_invalida_solo_afectados(self):
        mod = importlib.import_module('src.mantenimiento.recuperar_multicampana')
        for modo in ('correcto', 'legacy', 'sin_registro', 'raster_invalido'):
            with self.subTest(modo=modo), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                base = Path(d); comp = base / 'comp'; dataset = base / 'dataset'
                comp.mkdir(); dataset.mkdir(); respaldo = base / 'respaldo'
                ruta = comp / '20JNL_202104_median.tif'; ruta.write_bytes(b'corrupto')
                escribir_json(comp / 'fechas_usadas.json', {'otro': {'valor': 1}})
                fuente = base / 'sentinel2_cordoba_2020_2021'; fuente.mkdir()
                escribir_json(fuente / 'seleccion_scl.json', {'20JNL_202104': {'fechas': ['20210401']}})
                for tile, refs in [('A', [str(ruta.resolve())]), ('B', ['otro.tif'])]:
                    escribir_json(dataset / f'dataset_{tile}_procedencia.json', {'tile': tile, 'composites': refs})
                    for tipo in ('X', 'Y'):
                        (dataset / f'dataset_{tile}_{tipo}.npy').write_bytes(b'array')
                prep = SimpleNamespace(Y3=base, CAMPANAS=[('2020_2021', comp, ['202104'], '2020' if modo == 'legacy' else '2021', '', dataset)], json_atomico=escribir_json)
                def componer(origen, salida, tile, mes, fechas, info):
                    (salida / ruta.name).write_bytes(b'reconstruido')
                    return tile, mes, 'OK', None if modo == 'sin_registro' else {'fechas': fechas}
                compositor = SimpleNamespace(BANDAS=['B02','B03','B04','B08'], cargar_info=lambda p: {}, componer_tile_mes=componer)
                def verificar(p):
                    if modo == 'raster_invalido' and 'reconstruidos' in str(p):
                        raise ValueError('ilegible')
                with patch.object(mod, 'verificar_raster', side_effect=verificar), patch.object(mod, 'cargar_dependencias', side_effect=AssertionError('carga implicita')):
                    if modo == 'correcto':
                        resultado = mod.reparar(ruta, respaldo, prep, compositor)
                        self.assertEqual(resultado['tiles_invalidados'], ['A'])
                        self.assertEqual(ruta.read_bytes(), b'reconstruido')
                        self.assertEqual((respaldo / 'composites_corruptos/2020_2021' / ruta.name).read_bytes(), b'corrupto')
                        self.assertTrue((respaldo / 'registros/2020_2021/fechas_usadas.json').exists())
                        self.assertEqual(json.loads((comp / 'fechas_usadas.json').read_text())['otro'], {'valor': 1})
                        for tipo in ('X', 'Y', 'procedencia'):
                            extension = 'json' if tipo == 'procedencia' else 'npy'
                            self.assertFalse((dataset / f'dataset_A_{tipo}.{extension}').exists())
                            self.assertTrue((respaldo / 'datasets_a_regenerar/2020_2021' / f'dataset_A_{tipo}.{extension}').exists())
                            self.assertTrue((dataset / f'dataset_B_{tipo}.{extension}').exists())
                    else:
                        with self.assertRaises((ValueError, RuntimeError)):
                            mod.reparar(ruta, respaldo, prep, compositor)
                        self.assertEqual(ruta.read_bytes(), b'corrupto')
                        self.assertTrue((dataset / 'dataset_A_X.npy').exists())

    def test_main_reanuda_solo_preparacion_y_registra_fallos(self):
        mod = importlib.import_module('src.mantenimiento.recuperar_multicampana')
        for modo in ('correcto', 'persisten', 'preparacion_falla'):
            with self.subTest(modo=modo), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                base = Path(d); logs = base / 'logs'
                prep = SimpleNamespace(Y3=base, CAMPANAS=[('campana', base / 'comp', ['202104'], '2021', '', base / 'datos')], TILES=['20JNL'], json_atomico=escribir_json)
                fallos = [['corrupto'], ['corrupto']] if modo == 'persisten' else [[]]
                with patch.object(mod, 'verificar_raster'), patch.object(mod, 'auditar_rasters', side_effect=fallos), patch.object(mod, 'reparar', return_value={}), patch.object(mod.subprocess, 'run', return_value=SimpleNamespace(returncode=3 if modo == 'preparacion_falla' else 0)) as ejecutar:
                    if modo == 'correcto':
                        mod.main(prep, object(), logs)
                        self.assertEqual(json.loads((logs / 'recuperacion.json').read_text())['estado'], 'preparacion_completa')
                        self.assertEqual(ejecutar.call_args.args[0], ['bash', 'src/preparar_multicampana.sh', '--ignorar-codigo20'])
                        self.assertEqual(ejecutar.call_args.kwargs['cwd'], ROOT)
                    else:
                        with self.assertRaises(RuntimeError):
                            mod.main(prep, object(), logs)
                        if modo == 'persisten': ejecutar.assert_not_called()

    def test_exitcode_de_entrada(self):
        mod = importlib.import_module('src.mantenimiento.recuperar_multicampana')
        with tempfile.TemporaryDirectory() as d, patch.object(mod, 'LOGS', Path(d)):
            with patch.object(mod, 'main'):
                mod.ejecutar()
            self.assertEqual((Path(d) / 'recuperacion.exitcode').read_text(), '0\n')
            with patch.object(mod, 'main', side_effect=RuntimeError('fallo')):
                with self.assertRaises(RuntimeError): mod.ejecutar()
            self.assertEqual((Path(d) / 'recuperacion.exitcode').read_text(), '1\n')

    def test_bloqueos_impiden_recuperacion_concurrente(self):
        import fcntl
        mod = importlib.import_module('src.mantenimiento.recuperar_multicampana')
        for nombre in ('preparacion.lock', 'recuperacion.lock'):
            with self.subTest(nombre=nombre), tempfile.TemporaryDirectory() as d:
                logs = Path(d)
                with open(logs / nombre, 'a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with patch.object(mod, 'reparar') as reparar, patch.object(mod, 'auditar_rasters') as auditar, patch.object(mod.subprocess, 'run') as ejecutar:
                        with self.assertRaises(BlockingIOError):
                            mod.main(SimpleNamespace(), object(), logs)
                        reparar.assert_not_called()
                        auditar.assert_not_called()
                        ejecutar.assert_not_called()
                    self.assertFalse((logs / 'recuperacion.pid').exists())
