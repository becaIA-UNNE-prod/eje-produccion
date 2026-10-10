"""Generadores con rasters diminutos y pares temporales; no prepara datos reales."""
import contextlib
import importlib
import io
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.datos import generar_arrays_originales as original
from src.datos import generar_dataset_cordoba_f16 as cordoba
from src.datos import generar_dataset_cordoba_2017_f16 as antiguo

ROOT=Path(__file__).resolve().parents[2]


class GeneradoresArraysTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base=Path(tmp.name)
        self.comp=self.base/'comp'; self.comp.mkdir()
        self.mask=self.base/'mask'; self.mask.mkdir()
        self.out=self.base/'out'
        self.meses=['202012','202101','202102','202103','202104','202105']
        self.meta=dict(driver='GTiff',height=4,width=4,count=4,dtype='uint16',
                       crs='EPSG:32720',transform=from_origin(0,40,10,10))
        for i,mes in enumerate(self.meses):
            with rasterio.open(self.comp/f'A_{mes}_median.tif','w',**self.meta) as dst:
                dst.write(np.full((4,4,4),1000*(i+1),dtype=np.uint16))
                dst.descriptions=('B02','B03','B04','B08')

    def mascara(self,path,codes):
        data=np.zeros((4,4),dtype=np.uint8)
        data[:2,:2]=np.array(codes,dtype=np.uint8)
        with rasterio.open(path,'w',**dict(self.meta,count=1,dtype='uint8')) as dst:
            dst.write(data,1)

    def comprobar(self,prefix,channels=24):
        x=np.load(self.out/f'{prefix}_X.npy')
        y=np.load(self.out/f'{prefix}_Y.npy')
        self.assertEqual(x.shape,(1,channels,2,2))
        self.assertEqual(y.shape,(1,2,2))
        self.assertEqual(x.dtype,np.float16)
        self.assertEqual(y.dtype,np.uint8)
        np.testing.assert_allclose(x[0,:,0,0],np.repeat(np.arange(1,channels//4+1)/10,4),rtol=.001)
        self.assertFalse(list(self.out.glob('*.tmp')))
        return x,y

    def test_original_npy_remapeo_y_ventanas(self):
        path=self.mask/'original.tif'
        self.mascara(path,[[16,17],[20,21]])
        original.generar_dataset_tile('A',path,self.comp,self.out,meses=self.meses,size=2,step=2,max_workers=1)
        _,y=self.comprobar('dataset_TA')
        np.testing.assert_array_equal(y[0],[[4,5],[8,8]])
        before={p.name:p.read_bytes() for p in self.out.iterdir()}
        with patch('rasterio.open',side_effect=AssertionError('No regenerar existente')):
            original.generar_dataset_tile('A',path,self.comp,self.out,meses=self.meses,size=2,step=2,max_workers=1)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.out.iterdir()})

    def test_cordoba_leyendas_procedencia_y_omision(self):
        for leyenda,codes,expected in [('nivel3',[[16,17],[20,21]],[[2,3],[2,3]]),
                                       ('1819',[[0,3],[4,16]],[[2,2],[4,0]]),
                                       ('2020',[[10,11],[16,255]],[[2,3],[4,0]]),
                                       ('2021',[[10,11],[16,255]],[[2,3],[4,0]])]:
            with self.subTest(leyenda=leyenda):
                self.out=self.base/leyenda
                ruta=self.mask/cordoba.LEYENDAS[leyenda][1].format('A')
                self.mascara(ruta,codes)
                with patch.multiple(cordoba,DIR_COMPOSITES=self.comp,DIR_MASCARAS=self.mask,
                                    DIR_SALIDA=self.out,MESES_FIJOS=self.meses,SIZE=2,STEP=2,
                                    LEYENDA=leyenda,MAX_WORKERS=1,ESTRICTO=True,IGNORAR_CODIGO20=False):
                    cordoba.generar_dataset_tile('A')
                    _,y=self.comprobar('dataset_A')
                    np.testing.assert_array_equal(y[0],expected)
                    meta=json.loads((self.out/'dataset_A_procedencia.json').read_text())
                    self.assertEqual(meta['leyenda'],leyenda)
                    self.assertEqual(meta['meses'],self.meses)
                    self.assertEqual(meta['shape_x'],[1,24,2,2])
                    self.assertEqual(meta['mascara'],str(ruta.resolve()))
                    self.assertEqual(meta['mascara_mtime_ns'],ruta.stat().st_mtime_ns)
                    self.assertEqual(meta['composites_mtime_ns'],[(self.comp/f'A_{m}_median.tif').stat().st_mtime_ns for m in self.meses])
                    self.assertEqual(meta['normalizacion'],10000)
                    self.assertEqual(meta['codigos_ignorados'],[])
                    before={p.name:p.read_bytes() for p in self.out.iterdir()}
                    with patch('rasterio.open',side_effect=AssertionError('No regenerar existente')):
                        cordoba.generar_dataset_tile('A')
                    self.assertEqual(before,{p.name:p.read_bytes() for p in self.out.iterdir()})

    def test_2017_octabr_sufijo_y_dobles_cultivos(self):
        meses=list(antiguo.MESES_2017)
        for i,mes in enumerate(meses):
            with rasterio.open(self.comp/f'A_{mes}_median.tif','w',**self.meta) as dst:
                dst.write(np.full((4,4,4),1000*(i+1),dtype=np.uint16))
        self.mascara(self.mask/'etiqueta_A_10m_test.tif',[[16,17],[20,21]])
        with patch.multiple(antiguo,DIR_COMPOSITES=self.comp,DIR_MASCARAS=self.mask,
                            DIR_SALIDA=self.out,SIZE=2,STEP=2):
            antiguo.generar_dataset_tile('A',max_workers=1)
            _,y=self.comprobar('dataset_A_2017',channels=28)
            np.testing.assert_array_equal(y[0],[[2,3],[1,1]])
            with patch('rasterio.open',side_effect=AssertionError('No regenerar existente')):
                antiguo.generar_dataset_tile('A',max_workers=1)
        self.assertEqual(meses,['201710','201711','201712','201801','201802','201803','201804'])

    def test_faltantes_vacio_y_grilla(self):
        with patch.multiple(cordoba,DIR_COMPOSITES=self.comp,DIR_MASCARAS=self.mask,
                            DIR_SALIDA=self.out,MESES_FIJOS=self.meses,SIZE=2,STEP=2,
                            LEYENDA='2020',ESTRICTO=True,MAX_WORKERS=1):
            with self.assertRaises(FileNotFoundError):
                cordoba.generar_dataset_tile('A')
            ruta=self.mask/cordoba.LEYENDAS['2020'][1].format('A')
            self.mascara(ruta,[[0,0],[0,0]])
            with self.assertRaisesRegex(ValueError,'sin parches validos'):
                cordoba.generar_dataset_tile('A')
            self.assertFalse(self.out.exists())
            with rasterio.open(self.comp/f'A_{self.meses[0]}_median.tif','w',**dict(self.meta,count=3)) as dst:
                dst.write(np.ones((3,4,4),dtype=np.uint16))
            with self.assertRaisesRegex(ValueError,'grilla/bandas'):
                cordoba.generar_dataset_tile('A')
        original.generar_dataset_tile('B',ruta,self.comp,self.out,meses=self.meses,size=2,step=2,max_workers=1)
        self.assertFalse(list(self.out.glob('*.npy')))
        with patch.multiple(antiguo,DIR_COMPOSITES=self.comp,DIR_MASCARAS=self.mask,DIR_SALIDA=self.out):
            antiguo.generar_dataset_tile('B',max_workers=1)
        self.assertFalse(list(self.out.glob('*.npy')))

    def test_imports_wrappers_y_cli(self):
        code='''import sys, importlib
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import numpy, rasterio, utils.leyendas_mnc
sys.argv=['prueba','--invalido']
with patch('numpy.load',side_effect=AssertionError('arrays')), patch('numpy.save',side_effect=AssertionError('arrays')), patch('rasterio.open',side_effect=AssertionError('raster')), patch('pathlib.Path.mkdir',side_effect=AssertionError('salida')), patch('argparse.ArgumentParser.parse_args',side_effect=AssertionError('parser')):
    for old,new in [('03_generar_datasets_npz','generar_arrays_originales'),('03c_generar_dataset_cordoba_f16','generar_dataset_cordoba_f16'),('03d_generar_dataset_cordoba_2017_f16','generar_dataset_cordoba_2017_f16')]:
        a=importlib.import_module('src.'+old)
        b=importlib.import_module('src.datos.'+new)
        assert a.main is b.main
        assert a._procesar_chunk_ventanas is b._procesar_chunk_ventanas
    import src.datos.mascaras_nivel3_2017_2018 as masks
    from src.datos.generar_dataset_cordoba_f16 import REMAP_NIVEL3
    assert masks.REMAP_NIVEL3 is REMAP_NIVEL3
'''
        r=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=self.base,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        for old,new in [('03c_generar_dataset_cordoba_f16','generar_dataset_cordoba_f16'),('03d_generar_dataset_cordoba_2017_f16','generar_dataset_cordoba_2017_f16')]:
            for prefix,cwd in [([str(ROOT/'src'/f'{old}.py')],self.base),(['-m','src.datos.'+new],ROOT)]:
                r=subprocess.run([sys.executable,'-B',*prefix,'--help'],cwd=cwd,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
        for args in [['--workers','0'],['--ignorar-codigo20','--leyenda','2020']]:
            r=subprocess.run([sys.executable,'-B','-m','src.datos.generar_dataset_cordoba_f16',*args],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,2,r.stderr)

    def test_main_configura_worker_y_parser(self):
        # CLI globals must remain in the implementation module read by workers.
        names=['IGNORAR_CODIGO20','ESTRICTO','MAX_WORKERS','LEYENDA','MESES_FIJOS','DIR_COMPOSITES','DIR_SALIDA','DIR_MASCARAS']
        before={name:getattr(cordoba,name) for name in names}
        try:
            with patch.object(cordoba,'generar_dataset_tile') as generate, contextlib.redirect_stdout(io.StringIO()):
                cordoba.main(['--tiles','A','--meses',*self.meses,'--leyenda','2021',
                              '--dir-composites',str(self.comp),'--dir-mascaras',str(self.mask),
                              '--dir-salida',str(self.out),'--workers','1','--estricto','--ignorar-codigo20'])
            generate.assert_called_once_with('A')
            self.assertEqual(cordoba.DIR_COMPOSITES,self.comp)
            self.assertEqual(cordoba.DIR_MASCARAS,self.mask)
            self.assertEqual(cordoba.DIR_SALIDA,self.out)
            self.assertEqual(cordoba.MESES_FIJOS,self.meses)
            self.assertEqual(cordoba.LEYENDA,'2021')
            self.assertEqual(cordoba.MAX_WORKERS,1)
            self.assertTrue(cordoba.ESTRICTO and cordoba.IGNORAR_CODIGO20)
        finally:
            for name,value in before.items():
                setattr(cordoba,name,value)
