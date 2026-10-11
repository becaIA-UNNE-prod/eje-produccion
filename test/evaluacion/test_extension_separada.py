"""Fixtures raster pequeños en /tmp; sin GPU ni inferencia de modelos."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.datos import preparar_extension as prep
from src.evaluacion import evaluar_extension as evaluation


class ExtensionSeparadaTest(unittest.TestCase):
    def test_import_preparacion_sin_torch_figuras_operaciones(self):
        code = '''import sys
from unittest.mock import patch
import rasterio
import utils.integridad_raster
sys.argv = ['prueba', '--invalido']
with patch('rasterio.open', side_effect=AssertionError('raster')), patch('utils.integridad_raster.auditar_rasters', side_effect=AssertionError('auditoria')):
    import src.datos.preparar_extension
assert 'torch' not in sys.modules
assert 'matplotlib' not in sys.modules
assert 'src.evaluacion.evaluar_extension' not in sys.modules
'''
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)

    def test_wrapper_exports_y_despacho_ambos_modos(self):
        sys.path.insert(0, str(ROOT/'src'))
        wrapper = importlib.import_module('preparar_evaluar_extension')
        for name in ['shapes', 'write', 'prepare', 'legend', 'check_codes', 'files']:
            self.assertIs(getattr(wrapper, name), getattr(prep, name))
        for name in ['figure', 'evaluate', 'confusion', 'inferir_tile', 'sha256']:
            self.assertIs(getattr(wrapper, name), getattr(evaluation, name))
        with tempfile.TemporaryDirectory() as tmp:
            for mode, function in [('preparar', 'prepare'), ('evaluar', 'evaluate')]:
                out = Path(tmp)/mode
                with patch.object(wrapper, function) as operation:
                    wrapper.main(['--modo', mode, '--salida', str(out)])
                operation.assert_called_once_with(out)
                self.assertTrue(out.is_dir())
            for module, function in [(prep, 'prepare'), (evaluation, 'evaluate')]:
                with patch.object(module, function) as operation:
                    module.main(['--salida', tmp])
                operation.assert_called_once_with(Path(tmp))

    def test_ayuda_wrapper_desde_otro_cwd_y_modulos(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = [([str(ROOT/'src/preparar_evaluar_extension.py')], tmp),
                        (['-m', 'src.datos.preparar_extension'], ROOT),
                        (['-m', 'src.evaluacion.evaluar_extension'], ROOT)]
            for args, cwd in commands:
                result = subprocess.run([sys.executable, '-B', *args, '--help'], cwd=cwd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--salida', result.stdout)
            self.assertIn('--modo', subprocess.check_output([sys.executable, '-B', str(ROOT/'src/preparar_evaluar_extension.py'), '--help'], cwd=tmp, text=True))

    def test_preparacion_cuatro_campanas_y_evaluacion_con_reuso(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base/'corrida'; out.mkdir()
            profile = dict(driver='GTiff', width=4, height=4, count=1, dtype='uint8', crs='EPSG:32720', transform=from_origin(500000,6200000,10,10))
            sources = {}; inputs = {}
            for year in prep.YEARS:
                lut, allowed, nodata = prep.legend(year)
                # Un código de Maíz acreditado y un píxel NoData por campaña.
                raw = np.full((4,4), 0 if year=='2018_2019' else 10, dtype=np.uint8)
                self.assertIn(int(raw[0,0]), allowed)
                raw[-1,-1] = nodata
                source = base/(year+'.tif'); sources[year] = source
                with rasterio.open(source, 'w', **profile, nodata=nodata) as dst:
                    dst.write(raw, 1)
                for tile in prep.TILES:
                    paths = []
                    for index in range(6):
                        path = base/f'{year}_{tile}_{index}.tif'; paths.append(path)
                        with rasterio.open(path, 'w', **{**profile, 'count':4, 'dtype':'uint16'}, nodata=0) as dst:
                            dst.write(np.full((4,4,4),1000,dtype=np.uint16))
                    inputs[year,tile] = paths
            geometry = {'type':'Polygon','coordinates':[[[500000,6199960],[500009,6199960],[500009,6200000],[500000,6200000],[500000,6199960]]]}
            get_files = lambda year,tile: inputs[year,tile]
            with patch.object(prep, 'SOURCES', sources), patch.object(prep, 'files', get_files), patch.object(prep, 'shapes', return_value=[(geometry,1)]):
                prep.prepare(out)
            labels = json.loads((out/'labels_verificados.json').read_text())
            self.assertEqual(len(labels), 8)
            for record in labels:
                self.assertEqual((record['pixeles_completos'],record['pixeles_independientes'],record['excluidos']), (15,11,4))
                year,tile = record['campana'],record['tile']
                with rasterio.open(out/'mascaras'/f'{year}_{tile}_completa.tif') as src:
                    full = src.read(1)
                    np.testing.assert_array_equal(full[:-1], np.full((3,4),2))
                    self.assertEqual(full[-1,-1],0)
                    self.assertEqual(src.tags()['fuente'],str(sources[year]))
                with rasterio.open(out/'mascaras'/f'{year}_{tile}_independiente.tif') as src:
                    strict = src.read(1)
                    np.testing.assert_array_equal(strict[:,0],np.zeros(4))
                    np.testing.assert_array_equal(strict[:,1:],full[:,1:])
            self.assertEqual(len(list((out/'mascaras').glob('*.tif'))),16)
            self.assertFalse(list(out.rglob('*.partial.tif')))
            (out/'modelo_congelado.pth').write_bytes(b'checkpoint fixture, no pesos reales')
            checksum = evaluation.sha256(out/'modelo_congelado.pth')
            (out/'protocolo.json').write_text(json.dumps({'checkpoint_sha256':checksum}))
            def synthetic_prediction(model, files, destination, device, tags):
                destination.parent.mkdir(parents=True,exist_ok=True)
                with rasterio.open(destination, 'w', **profile, nodata=0) as dst:
                    predicted = np.full((4,4),2,dtype=np.uint8)
                    predicted[0,1]=0  # Predicción NoData sobre referencia válida: error.
                    dst.write(predicted,1); dst.update_tags(**tags)
            model = Mock(); model.to.return_value=model
            with patch.object(evaluation, 'files', get_files), patch.object(evaluation.torch.cuda, 'is_available', return_value=True), patch.object(evaluation.torch, 'device', return_value=SimpleNamespace(type='cuda')), patch.object(evaluation.torch, 'load', return_value={}), patch.object(evaluation, 'SimpleUNet', return_value=model), patch.object(evaluation, 'inferir_tile', side_effect=synthetic_prediction) as infer, patch.object(evaluation, 'figure'), patch.object(evaluation, 'matriz_fig'):
                evaluation.evaluate(out)
                self.assertEqual(infer.call_count,8)
                evaluation.evaluate(out)
                self.assertEqual(infer.call_count,8)  # Firmas idénticas: reutiliza los TIFF.
                path=inputs[prep.YEARS[0],prep.TILES[0]][0]
                with rasterio.open(path,'r+') as dst: dst.update_tags(cambio='fixture')
                evaluation.evaluate(out)
                self.assertEqual(infer.call_count,9)  # Invalida solo la entrada modificada.
            metrics=json.loads((out/'metricas_tiles.json').read_text())
            self.assertEqual(len(metrics),16)
            for metric in metrics:
                count=15 if metric['criterio']=='completa' else 11
                self.assertEqual(metric['pixeles'],count)
                self.assertAlmostEqual(metric['accuracy'],(count-1)/count)
                self.assertEqual(metric['confusion'][2][0],1)
            report=json.loads((out/'evaluacion_nuevo_test.json').read_text())
            self.assertEqual(len(report['grupos']),8)
            self.assertEqual(report['checkpoint_sha256'],checksum)
            self.assertEqual(len((out/'metricas_resumen.csv').read_text().splitlines()),9)
            self.assertTrue((out/'figuras/resumen_nuevo_test.png').exists())
            self.assertEqual(json.loads((out/'progreso.json').read_text())['estado'],'finalizado')
            (out/'modelo_congelado.pth').write_bytes(b'modificado')
            with self.assertRaisesRegex(ValueError,'Checkpoint congelado modificado'):
                evaluation.evaluate(out)

    def test_evaluacion_rechaza_preparacion_faltante_y_sin_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'modelo_congelado.pth').write_bytes(b'fixture')
            (out/'protocolo.json').write_text(json.dumps({'checkpoint_sha256':evaluation.sha256(out/'modelo_congelado.pth')}))
            with self.assertRaisesRegex(ValueError,'Preparacion no verificada'):
                evaluation.evaluate(out)
            (out/'preparacion_verificada.json').write_text('{}')
            with patch.object(evaluation.torch.cuda,'is_available',return_value=False), patch.object(evaluation.torch,'load',side_effect=AssertionError('no cargar pesos')):
                with self.assertRaisesRegex(RuntimeError,'GPU no disponible'):
                    evaluation.evaluate(out)
