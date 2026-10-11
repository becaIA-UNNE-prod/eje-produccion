"""Contratos de despacho y protocolos anteriores con rasters/modelos simulados."""
import contextlib
import hashlib
import importlib
import io
import json
import os
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.evaluacion import postprocesar_multicampana as dispatcher
from src.historico import postprocesar_multicampana as post
from src.historico import evaluar_ano_no_visto as temporal
from src.figuras import historico as figures


def raster(path,data,nodata=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,'w',driver='GTiff',height=data.shape[-2],width=data.shape[-1],count=len(data),dtype=data.dtype,crs='EPSG:32720',transform=from_origin(500000,6200000,10,10),nodata=nodata) as dst:
        dst.write(data)


def snapshot(out):
    result={}
    for p in out.rglob('*'):
        if not p.is_file():continue
        name=str(p.relative_to(out))
        if p.suffix=='.json':
            result[name]=json.loads(p.read_text().replace(str(out),'SALIDA'))
        elif p.suffix=='.tif':
            result[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        elif p.suffix=='.png':
            assert p.read_bytes().startswith(b'\x89PNG')
            result[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        else:result[name]=p.read_text()
    return result


def setup_historical(base):
    repo=base/'repo'; exp=base/'exp'; exp.mkdir()
    (exp/'train.exitcode').write_text('0\n')
    (exp/'config_multicampana.json').write_text(json.dumps({'modelo':'unet','canales':24,'split':post.SPLIT}))
    for directory in [exp,base/'dataset/exp_cordoba_dicmay_unet_f16',base/'dataset/exp_cordoba_dicmay_fcrtransunet_f16']:
        directory.mkdir(parents=True,exist_ok=True);(directory/'best_model.pth').write_bytes(b'fixture, no pesos reales')
    (exp/'metricas_multicampana.json').write_text(json.dumps({'val':{},'test':{}}))
    directories={'2018_2019':'composites_2018_2019_scl','2019_2020':'composites_dicmay_2019_2020','2020_2021':'composites_2020_2021_scl'}
    for year in post.YEARS:
        first=int(year[:4]);months=[f'{first}12']+[f'{first+1}{i:02}' for i in range(1,6)]
        raw=np.array([[0,1],[4,16]],dtype=np.uint8) if year=='2018_2019' else np.array([[10,11],[16,255]],dtype=np.uint8)
        for tile in sum(post.SPLIT.values(),[]):
            raster(repo/'mascaras_procesadas'/post.LEGENDS[year][1].format(tile),raw[None])
            for month in months:
                raster(base/directories[year]/f'{tile}_{month}_median.tif',np.full((4,2,2),1234,dtype=np.uint16),0)
    return repo,exp


def synthetic_prediction(model,files,destination,device,tags):
    raster(destination,np.array([[[0,3],[4,1]]],dtype=np.uint8),0)
    with rasterio.open(destination,'r+') as dst:dst.update_tags(**tags)


def run_historical(module,base,out,operation=None,verify=False):
    repo=base/'repo';exp=base/'exp'
    model=Mock();model.to.return_value=model
    text=io.StringIO()
    argv=['--exp',str(exp),'--salida',str(out),'--base',str(base)]
    if verify:argv+=['--verificar']
    with patch.object(module,'ROOT',repo), patch.object(module,'SimpleUNet',return_value=model), patch.object(module,'FCRTransUNet',return_value=model), patch.object(module.torch,'load',return_value={}), patch.object(module.torch.cuda,'is_available',return_value=True), patch.object(module.torch,'device',return_value=SimpleNamespace(type='cuda')), patch.object(module.torch.cuda,'empty_cache'), patch.object(module,'inferir_tile',side_effect=synthetic_prediction) as inference, patch.object(module,'curvas'), patch.object(module,'matriz_fig'), patch.object(module,'figura_tile'), patch.object(module,'resumen'), patch.object(module.subprocess,'run'), patch.object(module.time,'monotonic',return_value=0), contextlib.redirect_stdout(text):
        if operation:operation(argv)
        else:module.main(argv)
    return snapshot(out) if out.exists() else {},text.getvalue().replace(str(out),'SALIDA'),inference.call_count


def setup_temporal(base):
    repo=base/'repo';out=base/'salida';out.mkdir(parents=True)
    (repo/'mascaras_procesadas').mkdir(parents=True)
    (repo/'config').mkdir()
    (repo/'config/leyenda_mnc_2021_2022.json').write_bytes((ROOT/'config/leyenda_mnc_2021_2022.json').read_bytes())
    (out/'modelo_congelado.pth').write_bytes(b'fixture checkpoint')
    (out/'protocolo.json').write_text(json.dumps({'checkpoint_sha256':post.sha256(out/'modelo_congelado.pth')}))
    source=base/'MNC/2021_2022/MNC_verano2022.tif'
    raster(source,np.array([[[10,11],[16,255]]],dtype=np.uint8),255)
    for directory in [base/'exp',base/'dataset/exp_cordoba_dicmay_unet_f16',base/'dataset/exp_cordoba_dicmay_fcrtransunet_f16']:
        directory.mkdir(parents=True,exist_ok=True);(directory/'best_model.pth').write_bytes(b'fixture checkpoint')
    for tile in ['20HLK','20HLJ','20HKJ','20HMH']:
        for month in ['202112','202201','202202','202203','202204','202205']:
            raster(base/'composites_2021_2022_scl'/f'{tile}_{month}_median.tif',np.full((4,2,2),1234,dtype=np.uint16),0)
    return repo,out,source


def run_temporal(module,base,out,operation=None):
    def mapped_path(value):
        if str(value)=='/mnt/yacy_3/prod_bis/ferreyra':return base
        if str(value)=='/mnt/yacy_1/prod/ferreyra/MNC_INTA/2021_2022/MNC_verano2022.tif':return base/'MNC/2021_2022/MNC_verano2022.tif'
        return Path(value)
    model=Mock();model.to.return_value=model
    text=io.StringIO();p=module.post
    with patch.object(module,'ROOT',base/'repo'), patch.object(module,'Path',mapped_path), patch.object(module.hashlib,'md5',return_value=SimpleNamespace(hexdigest=lambda:'f88ffac06865b36038916ec22c1ffe28')), patch.object(p,'SimpleUNet',return_value=model), patch.object(p,'FCRTransUNet',return_value=model), patch.object(module.torch,'load',return_value={}), patch.object(module.torch.cuda,'is_available',return_value=True), patch.object(module.torch,'device',return_value=SimpleNamespace(type='cuda')), patch.object(module.torch.cuda,'empty_cache'), patch.object(module,'inferir_tile',side_effect=synthetic_prediction) as inference, patch.object(p,'figura_tile'), patch.object(module,'matriz_fig'), patch.dict(p.LEGENDS,{},clear=False), contextlib.redirect_stdout(text):
        argv=['--exp',str(base/'exp'),'--salida',str(out)]
        if operation:operation(argv)
        else:module.main(argv)
    masks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'repo/mascaras_procesadas').glob('*.tif')}
    return snapshot(out),text.getvalue(),inference.call_count,masks


class PostprocesadoSeparadoTest(unittest.TestCase):
    def test_despacho_vigente_historico_y_lock_sin_mutar_argumentos(self):
        argv=['--exp','/tmp/exp','--salida','/tmp/out','--lock-adquirido']
        with patch('src.evaluacion.test_actual.main',return_value=17) as current, patch.object(post,'main') as history:
            self.assertEqual(dispatcher.main(argv),17)
            current.assert_called_once_with(argv);history.assert_not_called()
        historical=['--historico',*argv,'--verificar']
        copy=historical.copy()
        with patch.object(post,'main',return_value=18) as history, patch('src.evaluacion.test_actual.main') as current:
            self.assertEqual(dispatcher.main(historical),18)
            history.assert_called_once_with(['--exp','/tmp/exp','--salida','/tmp/out','--verificar'])
            current.assert_not_called()
        self.assertEqual(historical,copy)
        with patch.object(sys,'argv',['comando',*historical]), patch.object(post,'main') as history:
            dispatcher.main()
            history.assert_called_once()

    def test_exports_compatibles_y_leyenda_temporal_compartida(self):
        wrapper=importlib.import_module('src.postprocesar_multicampana')
        for name in ['main','inferir_tile','normalizar','evaluar','metricas_confusion','LEGENDS','SPLIT','guardar','sha256','SimpleUNet','FCRTransUNet','figura_tile','resumen']:
            self.assertIs(getattr(wrapper,name),getattr(post,name))
        self.assertIs(temporal.post,post)
        self.assertIs(post.LEGENDS,figures.LEGENDS)
        previous=importlib.import_module('src.evaluar_ano_no_visto')
        self.assertIs(previous.main,temporal.main)
        with patch.object(post,'_resumen') as summary:
            post.resumen([],{},Path('/tmp/out'))
        summary.assert_called_once_with([],{},Path('/tmp/out'),post.YEARS)
        self.assertEqual(post.SPLIT['test'],['20HLK','20HLJ'])

    def test_imports_seguros_y_dispatcher_ligero(self):
        code='''import sys,importlib
sys.path.insert(0,sys.argv[1])
sys.argv=['prueba','--invalido']
import src.evaluacion.postprocesar_multicampana
import src.postprocesar_multicampana
assert 'torch' not in sys.modules
assert 'matplotlib' not in sys.modules
assert 'src.historico.postprocesar_multicampana' not in sys.modules
from unittest.mock import patch
import torch,rasterio
with patch('rasterio.open',side_effect=AssertionError('raster')),patch('torch.load',side_effect=AssertionError('pesos')),patch('torch.cuda.is_available',side_effect=AssertionError('GPU')):
    import src.historico.postprocesar_multicampana
    import src.historico.evaluar_ano_no_visto
    importlib.import_module('src.evaluar_ano_no_visto')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=tmp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_ayudas_wrappers_y_modulos_desde_otro_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases=[([str(ROOT/'src/postprocesar_multicampana.py'),'--help'],tmp),([str(ROOT/'src/postprocesar_multicampana.py'),'--historico','--help'],tmp),([str(ROOT/'src/evaluar_ano_no_visto.py'),'--help'],tmp),(['-m','src.evaluacion.postprocesar_multicampana','--help'],ROOT),(['-m','src.historico.postprocesar_multicampana','--help'],ROOT),(['-m','src.historico.evaluar_ano_no_visto','--help'],ROOT)]
            for args,cwd in cases:
                result=subprocess.run([sys.executable,'-B',*args],cwd=cwd,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('--exp',result.stdout)
                if '--historico' in args:self.assertIn('--base',result.stdout)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_preflight_historico_sin_inferencia_y_rechaza_grilla(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);setup_historical(base);out=base/'salida'
            data,text,count=run_historical(post,base,out,verify=True)
            self.assertEqual((data,count),({},0));self.assertFalse(out.exists())
            self.assertIn('3 modelos compatibles',text)
            bad=base/'composites_2018_2019_scl/20HLK_201812_median.tif'
            raster(bad,np.ones((4,3,2),dtype=np.uint16))
            with self.assertRaisesRegex(ValueError,'Grilla o bandas incompatibles'):
                run_historical(post,base,out,verify=True)

    def test_orquestacion_historica_metricas_y_reutilizacion(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);setup_historical(base);out=base/'salida'
            data,text,count=run_historical(post,base,out)
            self.assertEqual(count,63)
            self.assertEqual(len(data['plan.json']['tareas']),63)
            records=data['metricas_tiles.json'];self.assertEqual(len(records),63)
            for record in records:
                self.assertEqual(record['pixeles'],3)
                self.assertAlmostEqual(record['accuracy'],2/3)
                self.assertEqual(record['confusion'][2][0],1)
            self.assertEqual(data['progreso.json']['estado'],'finalizado')
            again,_,count=run_historical(post,base,out)
            self.assertEqual(count,0);self.assertEqual(again,data)
            path=base/'composites_2018_2019_scl/20HLK_201812_median.tif'
            with rasterio.open(path,'r+') as dst:dst.update_tags(cambio='fixture')
            _,_,count=run_historical(post,base,out)
            self.assertEqual(count,3)

    def test_temporal_cuatro_tiles_mascaras_metricas_leyenda_y_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);repo,out,source=setup_temporal(base)
            data,text,count,masks=run_temporal(temporal,base,out)
            self.assertEqual(count,12);self.assertEqual(len(masks),4)
            report=data['evaluacion_temporal.json'];self.assertEqual(len(report['tiles']),12)
            self.assertEqual(len(report['grupos']),9)
            for record in report['tiles']:
                self.assertAlmostEqual(record['accuracy'],2/3)
                self.assertEqual(record['rol_espacial'],'test' if record['tile'] in ['20HLK','20HLJ'] else 'val')
            self.assertIn('figuras/comparacion_ano_no_visto.png',data)
            with patch.object(temporal,'ROOT',repo),patch.object(temporal,'Path',side_effect=lambda p:source if str(p).startswith('/mnt/yacy_1') else Path(p)),patch.object(temporal.hashlib,'md5',return_value=SimpleNamespace(hexdigest=lambda:'incorrecto')):
                with self.assertRaisesRegex(ValueError,'Checksum MNC2022 incorrecto'):
                    temporal.main(['--salida',str(out),'--exp',str(base/'exp')])

    def test_resumen_extraido_csv_metricas_y_png(self):
        cm=np.zeros((5,5),dtype=np.int64);cm[2,2]=2;cm[2,0]=1
        metric=post.metricas_confusion(cm)
        models={'multi':{'nombre':'Modelo fixture'}}
        records=[{'modelo':'multi','campana':year,'rol':'test',**metric} for year in post.YEARS]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'figuras').mkdir()
            figures.resumen_multicampana(records,models,out,post.YEARS)
            report=json.loads((out/'metricas_mapas.json').read_text())
            self.assertEqual(len(report['grupos']),3)
            for group in report['grupos']:self.assertAlmostEqual(group['accuracy'],2/3)
            self.assertIn('sin_prediccion',(out/'comparacion_metricas.csv').read_text())
            self.assertEqual(len(list((out/'figuras').glob('*.png'))),5)
            for png in (out/'figuras').glob('*.png'):self.assertTrue(png.read_bytes().startswith(b'\x89PNG'))
