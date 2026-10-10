#!/usr/bin/env python3
"""TEST vigente comun: checkpoints congelados, mapas y exclusion TRAIN/VAL."""
import argparse,json,shutil,csv,fcntl
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
from src.datos.campanas import files
from src.soporte.archivos import guardar, sha256
from utils.rutas import MULTICAMPANA_ROOT as Y3
from src.inferencia.raster import inferir_tile
from src.figuras.comunes import matriz_fig, curvas
from src.figuras.test_actual import figura_extension, comparar_modelos
from src.evaluacion.metricas import metricas_confusion, confusion_remapeada
from utils.test_actual import CONFIG,cargar_config,preflight
from utils.integridad_raster import auditar_rasters,firma
from model import SimpleUNet,FCRTransUNet,TransUNet,UTAE

def constructor(name):
 return {'unet':SimpleUNet,'fcrtransunet':FCRTransUNet,'transunet':TransUNet,'utae':lambda c,n:UTAE(c,n,T=c//4)}[name]


def comparison(out,year,tile,models,maskbase):
 return comparar_modelos(out,year,tile,models,maskbase,files(year,tile)[2])

def crear_parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exp',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);p.add_argument('--split-config',type=Path,default=CONFIG);p.add_argument('--modelo',choices=['unet','fcrtransunet','transunet','utae'],default=None);p.add_argument('--sin-comparar',action='store_true');p.add_argument('--verificar',action='store_true');p.add_argument('--lock-adquirido',action='store_true');return p


def main(argv=None):
 a=crear_parser().parse_args(argv)
 cfg=cargar_config(a.split_config);maskbase=preflight(cfg);years=cfg['campanas'];tiles=cfg['test'];out=a.salida
 expcfg=a.exp/'config_multicampana.json'
 traincfg=json.loads(expcfg.read_text()) if expcfg.exists() else {}
 if a.modelo is None:a.modelo=traincfg.get('modelo',next((kind for kind in ['fcrtransunet','transunet','utae'] if kind in a.exp.name),'unet'))
 exp=a.exp.resolve();models={'principal':{'nombre':'U-Net multicampana' if 'multicampana' in exp.name and a.modelo=='unet' else exp.name,'exp':exp,'arquitectura':a.modelo}}
 if not a.sin_comparar:
  for name,folder,arch in [('unet_2019_2020','exp_cordoba_dicmay_unet_f16','unet'),('fcr_2019_2020','exp_cordoba_dicmay_fcrtransunet_f16','fcrtransunet')]:
   path=Y3/'dataset'/folder
   if path.resolve()!=exp:models[name]={'nombre':'U-Net 2019/2020' if arch=='unet' else 'FCRTransUNet 2019/2020','exp':path,'arquitectura':arch}
 for name,record in models.items():
  checkpoint=record['exp']/'best_model.pth';model=constructor(record['arquitectura'])(24,5);model.load_state_dict(torch.load(checkpoint,map_location='cpu'));del model
  record['sha256']=sha256(checkpoint)
 for year in years:
  for tile in tiles:
   with rasterio.open(files(year,tile)[0]) as ref:
    for path in files(year,tile):
     with rasterio.open(path) as src:
      if (src.crs,src.transform,src.shape,src.count)!=(ref.crs,ref.transform,ref.shape,4):raise ValueError('Grilla incompatible: '+str(path))
 print(f'PREFLIGHT TEST ACTUAL OK: {tiles}, {len(years)} campanas, {len(models)} modelos',flush=True)
 if a.verificar:return
 out.mkdir(parents=True,exist_ok=True);(out/'figuras').mkdir(exist_ok=True)
 if all((r['exp']/'historial.npy').exists() for r in models.values()):curvas(models,out/'figuras/curvas_comparadas.png')
 lock=None
 if not a.lock_adquirido:
  lock=open(Y3/'logs_multicampana/inferencia.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
 failures=auditar_rasters([p for y in years for t in tiles for p in files(y,t)],maskbase/'integridad_composites.json',workers=2)
 failures.update(auditar_rasters([maskbase/'mascaras'/f'{y}_{t}_{s}.tif' for y in years for t in tiles for s in ['completa','independiente']],maskbase/'integridad_mascaras.json',workers=2))
 if failures:raise ValueError(str(failures))
 protocol={'split':cfg,'modelos':{n:{**r,'exp':str(r['exp'])} for n,r in models.items()},'preparacion_base':str(maskbase),'criterio_principal':'independiente','sin_reentrenar':True};guardar(out/'protocolo.json',protocol)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda':raise RuntimeError('GPU no disponible')
 torch.set_num_threads(4);records=[];task=0;total=len(models)*len(years)*len(tiles)
 for name,record in models.items():
  modelout=out/'modelos'/name;modelout.mkdir(parents=True,exist_ok=True);(modelout/'figuras').mkdir(exist_ok=True)
  weights=modelout/'modelo_congelado.pth'
  if not weights.exists():shutil.copy2(record['exp']/'best_model.pth',weights)
  if sha256(weights)!=record['sha256']:raise ValueError('Pesos congelados modificados')
  model=None
  for year in years:
   for tile in tiles:
    task+=1;guardar(out/'progreso.json',{'estado':'inferencia','modelo':name,'campana':year,'tile':tile,'tarea':task,'total':total})
    dest=modelout/'predicciones'/year/f'prediccion_{tile}.tif';tags={'checkpoint_sha256':record['sha256'],'entradas':json.dumps([firma(p) for p in files(year,tile)])};reuse=False
    old=maskbase/'predicciones/multicampana'/year/f'prediccion_{tile}.tif'
    if not dest.exists() and old.exists():
     with rasterio.open(old) as src:compatible=all(src.tags().get(k)==v for k,v in tags.items())
     if compatible:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dest)
    if dest.exists():
     with rasterio.open(dest) as src:reuse=all(src.tags().get(k)==v for k,v in tags.items())
    print(f'[{task}/{total}] {name} {year} {tile} reutilizar={reuse}',flush=True)
    if not reuse:
     if model is None:model=constructor(record['arquitectura'])(24,5).to(device);model.load_state_dict(torch.load(weights,map_location=device));model.eval()
     inferir_tile(model,files(year,tile),dest,device,tags=tags)
    for scope in ['completa','independiente']:
     metric=metricas_confusion(confusion_remapeada(dest,maskbase/'mascaras'/f'{year}_{tile}_{scope}.tif'));records.append({'modelo':name,'campana':year,'tile':tile,'criterio':scope,**metric})
    guardar(out/'metricas_tiles.json',records);figura_extension(modelout,year,tile,dest,files(year,tile)[2],maskbase=maskbase,title=record['nombre'])
  del model;torch.cuda.empty_cache()
 groups=[]
 for name,record in models.items():
  for year in years:
   for scope in ['completa','independiente']:
    cm=sum((np.array(r['confusion'],dtype=np.int64) for r in records if r['modelo']==name and r['campana']==year and r['criterio']==scope),np.zeros((5,5),dtype=np.int64));groups.append({'modelo':name,'campana':year,'criterio':scope,**metricas_confusion(cm)});matriz_fig(cm,f'{record["nombre"]} · {year} · {scope}',out/'figuras'/f'confusion_{name}_{year}_{scope}.png')
 for year in years:
  for tile in tiles:comparison(out,year,tile,models,maskbase)
 result={'tiles':records,'grupos':groups,'split':cfg,'criterio_principal':'independiente','protocolo':protocol};guardar(out/'evaluacion_test_actual.json',result)
 with (out/'comparacion_metricas.csv').open('w',newline='') as f:
  fields=['modelo','campana','criterio','accuracy','f1_macro','miou','pixeles'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in groups)
 fig,axes=plt.subplots(1,2,figsize=(15,5));x=np.arange(len(years));width=.8/len(models)
 for ax,metric in zip(axes,['accuracy','f1_macro']):
  for i,(name,record) in enumerate(models.items()):ax.bar(x+(i-(len(models)-1)/2)*width,[r[metric] for r in groups if r['modelo']==name and r['criterio']=='independiente'],width,label=record['nombre'])
  ax.set_xticks(x,[y.replace('_','/') for y in years]);ax.set_ylim(0,1);ax.set_title(metric);ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
 fig.suptitle('TEST vigente: 20HNJ/20HNH · mismos pixeles fuera de TRAIN/VAL');fig.tight_layout();fig.savefig(out/'figuras/comparacion_test_actual.png',dpi=160);plt.close(fig)
 for name,record in models.items():
  guardar(record['exp']/'test_actual.json',{'salida':str(out),'modelo':name,'split':cfg,'grupos':[r for r in groups if r['modelo']==name],'nota':'Este reporte sustituye el TEST historico para comparaciones actuales; pesos y resultados antiguos preservados.'})
 (out/'README.md').write_text('TEST vigente: 20HNJ y 20HNH. Criterio principal: fuera de grillas TRAIN y VAL.\nLos mismos composites y pixeles se usan para todos los modelos. Checkpoints congelados, sin reentrenar.\n2021/2022 no visto temporalmente; tres anios anteriores vistos en otras zonas.\nReferencias INTA, no campo independiente. Metricar mapas 512 no equivale a metricar parches 256.\nLa arquitectura y los anios de entrenamiento difieren entre modelos; no es una ablacion controlada.\n')
 guardar(out/'progreso.json',{'estado':'finalizado','tareas':total});print('EVALUACION TEST ACTUAL FINALIZADA',flush=True)
if __name__=='__main__':main()
