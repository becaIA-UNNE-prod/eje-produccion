import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('post',ROOT/'src/postprocesar_multicampana.py')
post=importlib.util.module_from_spec(spec); spec.loader.exec_module(post)


class Toy(torch.nn.Module):
    def forward(self,x):
        result=torch.zeros((len(x),5,*x.shape[-2:]),device=x.device)
        result[:,1]=1
        result[:,2]=(x[:,0]>.5)*2
        return result


class PostprocesarTest(unittest.TestCase):
    def raster(self,path,array,transform=None):
        with rasterio.open(path,'w',driver='GTiff',height=array.shape[-2],width=array.shape[-1],
                           count=len(array),dtype=array.dtype,crs='EPSG:32720',
                           transform=transform or from_origin(100,200,10,10)) as dst:
            dst.write(array)

    def test_bordes_nodata_normalizacion_y_georreferencia(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            directory=Path(directory); files=[]
            x=np.full((4,35,29),1234,dtype=np.uint16)
            x[:,:5,:7]=0; x[:,10:15,15:20]=8000
            for month in range(6):
                path=directory/f'{month}.tif'; self.raster(path,x); files.append(path)
            output=directory/'pred.tif'
            post.inferir_tile(Toy(),files,output,torch.device('cpu'),batch=3,size=16,tags={'test':'ok'})
            with rasterio.open(output) as src:
                actual=src.read(1)
                self.assertEqual(src.shape,(35,29)); self.assertEqual(src.nodata,0)
                self.assertEqual(src.tags()['test'],'ok'); self.assertEqual(src.transform,from_origin(100,200,10,10))
            expected=np.ones((35,29),dtype=np.uint8); expected[:5,:7]=0; expected[10:15,15:20]=2
            self.assertTrue(np.array_equal(actual,expected))
            self.assertFalse(output.with_suffix('.partial.tif').exists())
            self.assertEqual(post.normalizar(x)[0,6,8],np.float32(np.float16(1234/10000)))

    def test_error_de_grilla_no_publica_tiff_parcial(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory); files=[]
            for month in range(6):
                path=directory/f'{month}.tif'
                self.raster(path,np.ones((4,16,16),dtype=np.uint16),from_origin(100+month,200,10,10))
                files.append(path)
            output=directory/'pred.tif'
            with self.assertRaises(ValueError): post.inferir_tile(Toy(),files,output,torch.device('cpu'),size=16)
            self.assertFalse(output.exists()); self.assertFalse(output.with_suffix('.partial.tif').exists())

    def test_mascara_comun_y_excepcion_20(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            self.raster(directory/'label.tif',np.array([[[20,10],[26,255]]],dtype=np.uint8))
            self.raster(directory/'pred.tif',np.array([[[1,2],[0,3]]],dtype=np.uint8))
            cm=post.evaluar(directory/'pred.tif',directory/'label.tif',post.REMAP_2021,'2020_2021')
            self.assertEqual(cm.sum(),2); self.assertEqual(cm[2,2],1); self.assertEqual(cm[1,0],1)

    def test_normalizacion_rechaza_nan(self):
        with self.assertRaises(ValueError): post.normalizar(np.array([np.nan],dtype=np.float32))

if __name__=='__main__': unittest.main()
