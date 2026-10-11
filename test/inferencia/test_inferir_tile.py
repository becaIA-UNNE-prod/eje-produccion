"""Rasters pequeños y modelo sintético: no checkpoints ni redes reales."""
import contextlib
import hashlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.inferencia import inferir_tile as command

MODELS={'unet':'SimpleUNet','transunet':'TransUNet','utae':'UTAE','fcrtransunet':'FCRTransUNet'}
DEFAULT_MONTHS=['201910','201911','201912','202001','202002','202003','202004']


def run_fixture(module, variant, count, operation=None):
    """Ejecuta únicamente un modelo determinista sobre un raster de 17x513."""
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp); inputs=base/'composites'; inputs.mkdir()
        exp=base/'exp'; exp.mkdir(); (exp/'best_model.pth').write_bytes(b'fixture, no pesos')
        out=base/'salida con espacios'
        months=DEFAULT_MONTHS if count==7 else ['201812','201901','201902','201903','201904','201905']
        arrays=[]
        profile=dict(driver='GTiff',width=513,height=17,count=4,dtype='uint16',crs='EPSG:32720',transform=from_origin(500000,6200000,10,10),nodata=0)
        for index,month in enumerate(months):
            data=np.stack([np.full((17,513),1234+index*100+band,dtype=np.uint16) for band in range(4)])
            data[:,0,0]=0
            if index==0:data[0,8,512]=2345
            arrays.append(data)
            with rasterio.open(inputs/f'TILE_{month}_median.tif','w',**profile) as dst:dst.write(data)
        # Un píxel con datos solo en el último mes también tiene que ser válido.
        for index in range(count):
            arrays[index][:,0,1]=0
        arrays[-1][-1,0,1]=1
        for month,data in zip(months,arrays):
            with rasterio.open(inputs/f'TILE_{month}_median.tif','r+') as dst:dst.write(data)
        tensors=[]
        class SyntheticModel:
            def to(self,device):
                assert device.type=='cpu'
                return self
            def load_state_dict(self,state):assert state=={'fixture':True}
            def eval(self):return self
            def __call__(self,tensor):
                tensors.append(tensor.detach().cpu().numpy().copy())
                logits=torch.zeros((1,5,*tensor.shape[-2:]),dtype=torch.float32)
                logits[:,2]=1
                logits[:,3]=(tensor[:,0]>.2).float()*2
                return logits
        model=SyntheticModel()
        constructor=Mock(return_value=model)
        argv=['TILE','--modelo',variant,'--dir-composites',str(inputs),'--dir-exp',str(exp),'--dir-salida',str(out)]
        if count==6:argv+=['--meses',*months]
        text=io.StringIO()
        with patch.object(module,MODELS[variant],constructor), patch.object(module.torch.cuda,'is_available',return_value=False), patch.object(module.torch,'load',return_value={'fixture':True}) as load, contextlib.redirect_stdout(text):
            if operation is None:module.main(argv)
            else:operation(argv)
        constructor.assert_called_once_with(count*4,5,**({'T':count} if variant=='utae' else {}))
        load.assert_called_once_with(str(exp/'best_model.pth'),map_location=torch.device('cpu'))
        assert len(tensors)==2
        normalized=np.concatenate(arrays,axis=0).astype(np.float32)/10000
        expected0=np.pad(normalized[:,:,:512],((0,0),(0,15),(0,0)),mode='edge')[None]
        expected1=np.pad(normalized[:,:,512:],((0,0),(0,15),(0,15)),mode='edge')[None]
        np.testing.assert_array_equal(tensors[0],expected0)
        np.testing.assert_array_equal(tensors[1],expected1)
        # Este valor detecta una conversión accidental a float16.
        assert tensors[0][0,0,1,1]==np.float32(1234)/np.float32(10000)
        assert tensors[0][0,0,1,1]!=np.float32(np.float16(np.float32(1234)/10000))
        suffix='' if variant=='unet' else '_'+variant
        destination=out/f'prediccion_TILE{suffix}.tif'
        with rasterio.open(destination) as dst:
            expected=np.full((17,513),2,dtype=np.uint8);expected[0,0]=0;expected[8,512]=3
            np.testing.assert_array_equal(dst.read(1),expected)
            assert (dst.shape,dst.count,dst.dtypes,dst.nodata)==((17,513),1,('uint8',),0)
            assert dst.transform==profile['transform'] and dst.crs==rasterio.crs.CRS.from_string(profile['crs'])
            assert dst.compression.value=='DEFLATE'
            result={'profile':dst.profile.copy(),'array':dst.read(1).tolist()}
        assert {p.name for p in out.iterdir()}=={destination.name}
        result.update(stdout=text.getvalue().replace(str(base),'FIXTURE'),
                      tensor_sha256=[hashlib.sha256(t.tobytes()).hexdigest() for t in tensors],
                      tiff_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      constructor=(count*4,5, count if variant=='utae' else None))
        return result


class InferirTileTest(unittest.TestCase):
    def test_cuatro_modelos_24_28_canales_bordes_nodata_y_float32(self):
        for variant in MODELS:
            for count in [6,7]:
                with self.subTest(modelo=variant,meses=count):
                    run_fixture(command,variant,count)

    def test_parser_defaults_y_nombre_compatible(self):
        args=command.crear_parser().parse_args([])
        self.assertEqual((args.tile,args.modelo,args.meses,args.dir_exp,args.dir_composites),('20HMK','unet',None,None,None))
        self.assertEqual(args.dir_salida,str(command.LEGACY_PREDICTIONS))
        wrapper=importlib.import_module('src.2_inferencia')
        self.assertIs(wrapper.generar_mapa_clasificacion,command.generar_mapa_clasificacion)
        self.assertIs(wrapper.ejecutar_inferencia,command.ejecutar_inferencia)
        with patch.object(command,'ejecutar_inferencia') as operation:
            wrapper.generar_mapa_clasificacion(['OTRO','--modelo','utae','--meses','202001','202002'])
        parsed=operation.call_args.args[0]
        self.assertEqual((parsed.tile,parsed.modelo,parsed.meses),('OTRO','utae',['202001','202002']))

    def test_import_no_parsea_ni_abre_pesos_rasters_ni_consulta_gpu(self):
        code='''import sys, importlib
from unittest.mock import patch
import torch, rasterio
sys.path.insert(0,sys.argv[1])
sys.argv=['prueba','--invalido']
with patch('torch.load',side_effect=AssertionError('pesos')), patch('rasterio.open',side_effect=AssertionError('raster')), patch('torch.cuda.is_available',side_effect=AssertionError('GPU')), patch('os.makedirs',side_effect=AssertionError('mkdir')):
    import src.inferencia.inferir_tile
    importlib.import_module('src.2_inferencia')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=tmp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_ayuda_desde_otro_cwd_y_salida_explicita_prevalece(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=dict(os.environ,CORDOBA_RUN_DIR=tmp)
            for args,cwd in [([str(ROOT/'src/2_inferencia.py')],tmp),(['-m','src.inferencia.inferir_tile'],ROOT)]:
                result=subprocess.run([sys.executable,'-B',*args,'--help'],cwd=cwd,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('--meses',result.stdout)
            code='''from src.inferencia.inferir_tile import crear_parser
from pathlib import Path
import os
parser=crear_parser()
assert Path(parser.parse_args([]).dir_salida)==Path(os.environ['CORDOBA_RUN_DIR'])/'predicciones'
assert parser.parse_args(['--dir-salida','/tmp/explicito']).dir_salida=='/tmp/explicito'
'''
            subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_faltan_composites_o_pesos_sin_inferencia(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            argv=['TILE','--dir-composites',tmp,'--dir-exp',tmp,'--dir-salida',str(base/'out'),'--meses','202001']
            with patch.object(command.torch.cuda,'is_available',return_value=False), patch.object(command,'SimpleUNet') as constructor, patch.object(command.torch,'load',side_effect=AssertionError('pesos')):
                with self.assertRaisesRegex(FileNotFoundError,'Faltan composites'):
                    command.main(argv)
                constructor.assert_not_called()
                (base/'TILE_202001_median.tif').write_bytes(b'solo presencia para comprobar falta de pesos')
                with self.assertRaisesRegex(FileNotFoundError,'No se encontraron los pesos'):
                    command.main(argv)
            self.assertEqual(list((base/'out').iterdir()),[])
