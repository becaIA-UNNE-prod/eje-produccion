"""Pruebas CPU con fixtures temporales; no ejecutan entrenamiento."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.multicampana import (SPLIT, normalizar_campanas, validar_campanas,
                                cargar_campanas, pesos_desde_conteos, metricas_confusion)
from utils.leyendas_mnc import REMAP_2021, verificar_codigos


def modulo(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, ROOT/'src'/archivo)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class MulticampanaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.split = {'train': ['A'], 'val': ['B'], 'test': ['C']}
        self.entradas = []
        for nombre in ('2018_2019', '2019_2020', '2020_2021'):
            d = self.base/nombre; d.mkdir()
            self.entradas.append((nombre, str(d)))
            for rol, tiles in self.split.items():
                for tile in tiles:
                    np.save(d/f'dataset_{tile}_X.npy', np.zeros((1,24,2,2), dtype=np.float16))
                    y = np.array([[[1,2],[3,4]]] if rol == 'train' else [[[4,4],[4,4]]], dtype=np.uint8)
                    np.save(d/f'dataset_{tile}_Y.npy', y)

    def validar(self):
        return validar_campanas(self.entradas, self.split, tamano=2)

    def test_split_carga_y_pesos(self):
        campanas, informe, conteos = self.validar()
        np.testing.assert_array_equal(conteos, [0,3,3,3,3])
        self.assertEqual(pesos_desde_conteos(conteos), [0.,1.,1.,1.,1.])
        conjuntos, por_campana = cargar_campanas(campanas, self.split)
        self.assertEqual([len(conjuntos[r]) for r in self.split], [3,3,3])
        for c in por_campana.values():
            self.assertTrue(c['train'].datasets[0].augment)
            self.assertFalse(c['val'].datasets[0].augment)
            self.assertFalse(c['test'].datasets[0].augment)
        self.assertEqual(len(informe), 3)
        roles = [set(t) for t in SPLIT.values()]
        self.assertFalse(roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])

    def test_duplicados(self):
        with self.assertRaises(ValueError):
            normalizar_campanas(self.entradas + [self.entradas[0]])
        with self.assertRaises(ValueError):
            normalizar_campanas([('2017_2019', '/tmp/a')])
        with self.assertRaises(ValueError):
            normalizar_campanas([self.entradas[0], ('2021_2022', self.entradas[0][1])])

    def test_faltante(self):
        (Path(self.entradas[0][1])/'dataset_C_Y.npy').unlink()
        with self.assertRaises(FileNotFoundError): self.validar()

    def test_canales(self):
        np.save(Path(self.entradas[0][1])/'dataset_A_X.npy', np.zeros((1,28,2,2), dtype=np.float16))
        with self.assertRaises(ValueError): self.validar()

    def test_clases(self):
        np.save(Path(self.entradas[0][1])/'dataset_B_Y.npy', np.full((1,2,2), 5, dtype=np.uint8))
        with self.assertRaises(ValueError): self.validar()

    def test_no_finitos(self):
        np.save(Path(self.entradas[0][1])/'dataset_A_X.npy', np.full((1,24,2,2), np.nan, dtype=np.float16))
        with self.assertRaises(ValueError): self.validar()

    def test_pares_incompatibles(self):
        np.save(Path(self.entradas[0][1])/'dataset_A_Y.npy', np.ones((2,2,2), dtype=np.uint8))
        with self.assertRaises(ValueError): self.validar()

    def test_leyenda_2021(self):
        np.testing.assert_array_equal(REMAP_2021[[10,11,16,26,28,255]], [2,3,4,1,1,0])
        verificar_codigos([10,11,16,26,28,255])
        with self.assertRaises(ValueError): verificar_codigos([0,27])
        with self.assertRaises(ValueError): verificar_codigos([20])
        verificar_codigos([20], ignorar_codigo20=True)
        self.assertEqual(REMAP_2021[20], 0)
        with self.assertRaises(ValueError): verificar_codigos([27], ignorar_codigo20=True)

    def test_metricas_cuentan_prediccion_nodata_como_error(self):
        cm = np.zeros((5,5), dtype=np.int64); cm[2,2] = 1; cm[2,0] = 1
        m = metricas_confusion(cm)
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['clases']['Maiz']['recall'], .5)

    def test_cli_y_verificacion_sin_salidas(self):
        salida = self.base/'experimento'
        comando = [sys.executable, str(ROOT/'src/1_train_cordoba.py'), '--verificar-datos', '--dir-exp', str(salida)]
        # Fixture completo de 51 pares con forma productiva, sin modelos/GPU.
        for nombre, d in self.entradas:
            for tile in sum(SPLIT.values(), []):
                np.save(Path(d)/f'dataset_{tile}_X.npy', np.zeros((1,24,256,256), dtype=np.float16))
                y = np.resize(np.array([1,2,3,4], dtype=np.uint8), (1,256,256))
                np.save(Path(d)/f'dataset_{tile}_Y.npy', y)
            comando += ['--campana', nombre, d]
        resultado = subprocess.run(comando, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        self.assertIn('VERIFICACION OK', resultado.stdout)
        self.assertFalse(salida.exists())
        resultado = subprocess.run(comando + ['--dir-train', '/tmp/legacy'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(resultado.returncode, 2)
        pesos = subprocess.run([sys.executable, '-m', 'src.entrenamiento.calcular_pesos', '--dirs',
                                 *[d for _,d in self.entradas]], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(pesos.returncode, 0, pesos.stderr)
        self.assertIn('PESOS_ARGS 0.0 1.0 1.0 1.0 1.0', pesos.stdout)
        ayuda = subprocess.run([sys.executable, str(ROOT/'src/1_train_cordoba.py'), '--help'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(ayuda.returncode, 0)
        self.assertIn('--dir-train-extra', ayuda.stdout)

    def test_generador_2021_normalizacion_y_procedencia(self):
        comp = self.base/'comp'; comp.mkdir()
        masks = self.base/'masks'; masks.mkdir()
        salida = self.base/'dataset_nuevo'
        meses = ['202012','202101','202102','202103','202104','202105']
        meta = {'driver':'GTiff', 'height':512, 'width':512, 'count':4,
                'dtype':'uint16', 'crs':'EPSG:32720', 'transform':from_origin(0,5120,10,10)}
        for i, mes in enumerate(meses):
            with rasterio.open(comp/f'A_{mes}_median.tif','w',**meta) as dst:
                dst.write(np.full((4,512,512), 1000*(i+1), dtype=np.uint16))
                dst.descriptions = ('B02','B03','B04','B08')
        meta.update(count=1,dtype='uint8',nodata=255)
        with rasterio.open(masks/'etiqueta_mnc2021_A_10m.tif','w',**meta) as dst:
            dst.write(np.resize(np.array([10,11,16,20,26,28,255],dtype=np.uint8),(512,512)),1)
        cmd = [sys.executable, str(ROOT/'src/03c_generar_dataset_cordoba_f16.py'),
               '--tiles','A','--meses',*meses,'--leyenda','2021','--dir-composites',str(comp),
               '--dir-mascaras',str(masks),'--dir-salida',str(salida),'--workers','1','--estricto']
        r = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(r.returncode, 0, 'El codigo 20 debe abortar sin politica explicita')
        cmd.append('--ignorar-codigo20')
        r = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        x = np.load(salida/'dataset_A_X.npy'); y = np.load(salida/'dataset_A_Y.npy')
        self.assertEqual(x.shape,(1,24,256,256)); self.assertEqual(y.shape,(1,256,256))
        np.testing.assert_allclose(x[0,:,0,0],np.repeat(np.arange(1,7)/10,4),rtol=.001)
        self.assertEqual(set(np.unique(y)),{0,1,2,3,4})
        meta_generada = json.loads((salida/'dataset_A_procedencia.json').read_text())
        self.assertEqual(meta_generada['leyenda'],'2021')
        self.assertEqual(meta_generada['codigos_ignorados'],[20])
        cmd[cmd.index('--dir-composites')+1] = str(self.base/'faltante')
        cmd[cmd.index('--dir-salida')+1] = str(self.base/'salida_error')
        r = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)

    def test_mascara_nearest_grilla_y_codigos(self):
        mod = modulo('mascaras', '01d_mascaras_mnc_2020_2021.py')
        comp = self.base/'composites'; comp.mkdir()
        origen = self.base/'mnc.tif'
        meta = {'driver': 'GTiff', 'height': 2, 'width': 2, 'count': 1, 'dtype': 'uint8',
                'crs': 'EPSG:4326', 'transform': from_origin(0,2,1,1), 'nodata': 255}
        with rasterio.open(origen, 'w', **meta) as dst:
            dst.write(np.array([[10,11],[16,255]], dtype=np.uint8),1)
        self.assertEqual(set(mod.verificar_fuente(origen)), {'10','11','16','255'})
        meta.update(height=4,width=4,count=4,dtype='uint16',transform=from_origin(0,2,.5,.5),nodata=0)
        with rasterio.open(comp/'A_202012_median.tif','w',**meta) as dst:
            dst.write(np.zeros((4,4,4),dtype=np.uint16))
        mod.generar('A', origen, comp, self.base/'mascaras')
        with rasterio.open(self.base/'mascaras/etiqueta_mnc2021_A_10m.tif') as m:
            np.testing.assert_array_equal(m.read(1), np.repeat(np.repeat([[10,11],[16,255]],2,axis=0),2,axis=1))
            self.assertEqual(m.transform, meta['transform'])
            self.assertEqual(m.nodata, 255)


if __name__ == '__main__':
    unittest.main()
