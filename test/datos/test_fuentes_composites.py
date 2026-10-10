"""Fuentes simuladas y composites diminutos en /tmp; nunca usa red real."""
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
from unittest.mock import Mock, patch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.datos import descargar_stac as stac
from src.datos import composites_mensuales as mensual
from src.datos import composites_2019_2020 as jp2
from src.datos import composites_cog as cog
from src.datos import composites_scl as scl

ROOT = Path(__file__).resolve().parents[2]
TILE = '20AAA'
BANDAS = ['B02', 'B03', 'B04', 'B08']


class FuentesCompositesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.out = self.base/'out'
        self.out.mkdir()

    def raster(self, path, data, resolution=10):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.asarray(data, dtype=np.uint16)
        with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
                           count=1, dtype='uint16', crs='EPSG:32720',
                           transform=from_origin(0, 240, resolution, resolution), nodata=0) as dst:
            dst.write(data, 1)

    def escena(self, fecha, valores, jp2_names=False, clasificacion=None):
        for i, banda in enumerate(BANDAS):
            nombre = f'{TILE}_{fecha}_{banda}.jp2' if jp2_names else banda+'.tif'
            self.raster(self.base/TILE/fecha/nombre, np.asarray(valores)+i*np.where(np.asarray(valores)>0,1,0))
        if clasificacion is not None:
            self.raster(self.base/TILE/fecha/'SCL.tif', clasificacion, 20)

    def leer(self):
        with rasterio.open(self.out/f'{TILE}_202001_median.tif') as src:
            return src.read(), src.profile, src.descriptions

    def test_mediana_original_y_omision(self):
        for fecha, valor in [('20200101',100),('20200102',300)]:
            self.escena(fecha, np.full((4,4),valor), jp2_names=True)
        args = ('202001',['20200101','20200102'],TILE,str(self.base/TILE),str(self.out),BANDAS)
        mensual.procesar_un_mes(args)
        data, meta, descriptions = self.leer()
        np.testing.assert_array_equal(data[:,0,0], [200,201,202,203])
        self.assertEqual(meta['dtype'],'uint16')
        self.assertEqual(meta['compress'],'lzw')
        self.assertEqual(descriptions,(None,)*4)
        with patch('rasterio.open',side_effect=AssertionError('No releer existente')):
            self.assertIn('Ya existe',mensual.procesar_un_mes(args))

    def test_jp2_primera_fecha_cobertura_y_float32(self):
        self.escena('20200101',np.zeros((100,100)),jp2_names=True)
        self.escena('20200102',np.full((100,100),300),jp2_names=True)
        self.escena('20200103',np.full((100,100),500),jp2_names=True)
        with patch.object(jp2,'BASE_S2',self.base), patch.object(jp2,'DIR_OUT',self.out):
            self.assertEqual(jp2.elegir_fecha(TILE,'202001')[0],'20200102')
            self.assertTrue(jp2.generar_composite(TILE,'202001'))
            self.assertFalse(jp2.generar_composite(TILE,'202002'))
        data, meta, descriptions = self.leer()
        np.testing.assert_array_equal(data[:,0,0],[300,301,302,303])
        self.assertEqual(meta['dtype'],'float32')
        self.assertEqual(descriptions,(None,)*4)
        self.assertFalse(list(self.out.glob('*.tmp')))

    def test_cog_offset_relleno_y_metadatos(self):
        self.escena('20200101',[[1500,0],[1500,0]])
        self.escena('20200102',[[700,700],[700,700]])
        info = {(TILE,'20200101'):dict(cloud=1,baseline='04.00',boa_offset_applied=False),
                (TILE,'20200102'):dict(cloud=10,baseline='04.00',boa_offset_applied=True)}
        result = cog.generar_composite(self.base,self.out,TILE,'202001',info,99,30)
        data, meta, descriptions = self.leer()
        np.testing.assert_array_equal(data[0],[[500,700],[500,700]])
        np.testing.assert_array_equal(data[:,0,1],[700,701,702,703])
        self.assertEqual(result[3],dict(fechas=['20200101','20200102'],cobertura=100.0))
        self.assertEqual(meta['dtype'],'uint16')
        self.assertEqual(meta['nodata'],0)
        self.assertEqual(descriptions,tuple(BANDAS))
        self.assertIsNone(cog.generar_composite(self.base,self.out,TILE,'202001',info,99,30)[3])
        self.assertEqual(cog.generar_composite(self.base,self.out,TILE,'202002',info,99,30)[2],'SIN FECHAS')

    def test_scl_dilatacion_y_clases(self):
        classes = np.full((24,24),4)
        classes[12,12] = 9
        for pos, code in enumerate([0,1,3,8,10]):
            classes[0,pos] = code
        ruta = self.base/'SCL.tif'
        self.raster(ruta,classes,20)
        mask = scl.despejado_20m(ruta)
        self.assertFalse(mask[12,16])
        self.assertTrue(mask[12,17])
        self.assertFalse(mask[0,:5].any())
        self.assertTrue(mask[23,23])

    def test_scl_seleccion_y_minimo_azul(self):
        azul1 = np.repeat(np.repeat([[500,1800],[500,1800]],2,axis=0),2,axis=1)
        azul2 = np.repeat(np.repeat([[700,1700],[700,1700]],2,axis=0),2,axis=1)
        self.escena('20200101',azul1,clasificacion=[[4,4],[0,0]])
        self.escena('20200102',azul2,clasificacion=[[0,0],[4,4]])
        info = {(TILE,'20200101'):dict(cloud=1), (TILE,'20200102'):dict(cloud=20)}
        selection = scl.seleccionar_tile_mes(self.base,TILE,'202001',info,99,8,.2)
        self.assertEqual(selection[2],dict(fechas=['20200101','20200102'],despejado_previsto=100.0,candidatas=2))
        result = scl.componer_tile_mes(self.base,self.out,TILE,'202001',['20200102','20200101'],info)
        data, meta, descriptions = self.leer()
        expected = np.repeat(np.repeat([[500,1700],[700,1700]],2,axis=0),2,axis=1)
        np.testing.assert_array_equal(data[0],expected)
        np.testing.assert_array_equal(data[:,0,3],[1700,1701,1702,1703])
        self.assertEqual(result[3],dict(fechas=['20200101','20200102'],despejado=50.0,cobertura=100.0))
        self.assertEqual(descriptions,tuple(BANDAS))
        self.assertEqual(meta['nodata'],0)
        self.assertIsNone(scl.componer_tile_mes(self.base,self.out,TILE,'202001',[],info)[3])
        self.assertTrue(scl.componer_tile_mes(self.base,self.out,TILE,'202002',['20200201'],info)[2].startswith('FALTAN'))

    def test_stac_busqueda_paginada_filtros_y_seleccion(self):
        tiles = self.base/'tiles.txt'
        tiles.write_text('T20AAA')
        def item(date,nodata=10,cloud=5,tile=TILE):
            return dict(properties={'grid:code':'MGRS-'+tile,'datetime':date+'T00:00:00Z',
                        'eo:cloud_cover':cloud,'s2:nodata_pixel_percentage':nodata,
                        's2:processing_baseline':'04.00','earthsearch:boa_offset_applied':False},
                        assets={a:{'href':'https://fixture.invalid/'+date+'/'+b} for a,b in {**stac.BANDS,**stac.SCL}.items()})
        response1 = Mock()
        response1.json.return_value = dict(features=[item('2020-01-01',30),item('2020-01-02',90),item('2020-01-03',cloud=99)],links=[dict(rel='next',body={'pagina':2})])
        response2 = Mock()
        response2.json.return_value = dict(features=[item('2020-01-01',5),item('2020-01-04',tile='OTRO')])
        args = SimpleNamespace(tiles_file=tiles,inicio='2020-01-01',fin='2020-01-31',out=self.out,max_nodata=50)
        with patch.object(stac.requests,'post',side_effect=[response1,response2]) as request:
            stac.buscar(args)
        self.assertEqual(request.call_count,2)
        products = json.loads((self.out/'productos.json').read_text())
        self.assertEqual(len(products),1)
        self.assertEqual(products[0]['nodata'],5)
        self.assertIs(products[0]['boa_offset_applied'],False)
        selection = self.base/'selection.json'
        selection.write_text(json.dumps({TILE+'_202001':{'fechas':['20200101']}}))
        args.bandas = ['SCL']; args.seleccion = selection
        self.assertEqual([b for _,b,_ in stac.archivos(args)],['SCL'])

    def test_descarga_reintento_y_fallo_preserva_original(self):
        destination = self.base/'descarga.tif'
        destination.write_bytes(b'original')
        bad = Mock()
        bad.headers = {'Content-Length':'9'}
        bad.iter_content.return_value = [b'x']
        bad.__enter__ = Mock(return_value=bad)
        bad.__exit__ = Mock(return_value=False)
        with patch.object(stac.requests,'get',return_value=bad), patch.object(stac.time,'sleep'):
            self.assertTrue(stac.bajar_archivo('https://fixture.invalid',destination).startswith('ERROR'))
        self.assertEqual(destination.read_bytes(),b'original')
        self.assertFalse(destination.with_suffix('.part').exists())
        good = Mock()
        good.headers = {'Content-Length':'2'}
        good.iter_content.return_value = [b'ok']
        good.__enter__ = Mock(return_value=good)
        good.__exit__ = Mock(return_value=False)
        with patch.object(stac.requests,'get',side_effect=[RuntimeError('red simulada'),good]), patch.object(stac.time,'sleep') as sleep:
            self.assertEqual(stac.bajar_archivo('https://fixture.invalid',destination),'ok')
        self.assertEqual(destination.read_bytes(),b'ok')
        sleep.assert_called_once_with(15)

    def test_imports_exports_y_ayuda_desde_otro_cwd(self):
        pairs = [('00_descargar_stac','descargar_stac'),('02_composites_mensuales','composites_mensuales'),('02b_composites_2019_2020','composites_2019_2020'),('02c_composites_cog','composites_cog'),('02d_composites_scl','composites_scl')]
        code = '''import sys, importlib
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import requests, rasterio, numpy, scipy.ndimage
sys.argv = ['comando','--inexistente']
with patch('requests.post',side_effect=AssertionError('red')), patch('requests.get',side_effect=AssertionError('red')), patch('rasterio.open',side_effect=AssertionError('raster')), patch('pathlib.Path.read_text',side_effect=AssertionError('lectura')), patch('pathlib.Path.mkdir',side_effect=AssertionError('escritura')), patch('os.listdir',side_effect=AssertionError('inventario')):
    for old,new in [('00_descargar_stac','descargar_stac'),('02_composites_mensuales','composites_mensuales'),('02b_composites_2019_2020','composites_2019_2020'),('02c_composites_cog','composites_cog'),('02d_composites_scl','composites_scl')]:
        a = importlib.import_module('src.'+old)
        b = importlib.import_module('src.datos.'+new)
        assert a.main is b.main
'''
        r = subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=self.base,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        for old,new in pairs:
            if hasattr(importlib.import_module('src.datos.'+new),'crear_parser'):
                for command,cwd in [([str(ROOT/'src'/f'{old}.py'),'--help'],self.base),(['-m','src.datos.'+new,'--help'],ROOT)]:
                    r = subprocess.run([sys.executable,'-B',*command],cwd=cwd,capture_output=True,text=True)
                    self.assertEqual(r.returncode,0,r.stderr)

    def test_cli_cog_scl_registros_y_errores(self):
        self.escena('20200101',np.full((4,4),500),clasificacion=[[4,4],[4,4]])
        (self.base/'productos.json').write_text(json.dumps([dict(tile=TILE,date='20200101',cloud=1)]))
        for prefix,cwd in [(['-m','src.datos.composites_cog'],ROOT),([str(ROOT/'src/02c_composites_cog.py')],self.base)]:
            r = subprocess.run([sys.executable,'-B',*prefix,'--base',str(self.base),'--out',str(self.out),'--tiles',TILE,'--meses','202001','--workers','1'],cwd=cwd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
        record = json.loads((self.out/'fechas_usadas.json').read_text())
        self.assertEqual(record[TILE+'_202001'],dict(fechas=['20200101'],cobertura=100.0))
        r = subprocess.run([sys.executable,'-B','-m','src.datos.composites_scl','seleccionar','--base',str(self.base),'--tiles',TILE,'--meses','202001','--workers','1'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        sclout = self.base/'sclout'
        r = subprocess.run([sys.executable,'-B',str(ROOT/'src/02d_composites_scl.py'),'componer','--base',str(self.base),'--out',str(sclout),'--tiles',TILE,'--workers','1'],cwd=self.base,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        record = json.loads((sclout/'fechas_usadas.json').read_text())
        self.assertEqual(record[TILE+'_202001'],dict(fechas=['20200101'],despejado=100.0,cobertura=100.0))
        r = subprocess.run([sys.executable,'-B','-m','src.datos.composites_scl','componer','--base',str(self.base),'--tiles',TILE],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,1)
        self.assertIn('componer necesita --out',r.stderr)
