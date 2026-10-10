#!/usr/bin/env python3
"""TEST vigente comun: checkpoints congelados, mapas y exclusion TRAIN/VAL."""
import argparse,json,sys,shutil,csv,fcntl
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import preparar_evaluar_extension as ext
import postprocesar_multicampana as post
from utils.test_actual import CONFIG,cargar_config,preflight
from utils.integridad_raster import auditar_rasters,firma
from model import TransUNet,UTAE

def constructor(name):
 return {'unet':post.SimpleUNet,'fcrtransunet':post.FCRTransUNet,'transunet':TransUNet,'utae':lambda c,n:UTAE(c,n,T=c//4)}[name]

def comparison(out,year,tile,models,maskbase):
 size=(1024,1024)
 with rasterio.open(ext.files(year,tile)[2]) as src:rgb=src.read([3,2,1],out_shape=(3,*size)).astype(np.float32).transpose(1,2,0)
 for c in range(3):
  valid=rgb[:,:,c][rgb[:,:,c]>0];lo,hi=np.percentile(valid,[2,98]) if valid.size else (0,1);rgb[:,:,c]=np.clip((rgb[:,:,c]-lo)/max(hi-lo,1),0,1)
 with rasterio.open(maskbase/'mascaras'/f'{year}_{tile}_independiente.tif') as src:target=src.read(1,out_shape=size,resampling=Resampling.nearest)
 panels=[('RGB febrero',rgb),('INTA fuera de TRAIN/VAL',post.COLORS[target])]
 for name,record in models.items():
  with rasterio.open(out/'modelos'/name/'predicciones'/year/f'prediccion_{tile}.tif') as src:p=src.read(1,out_shape=size,resampling=Resampling.nearest)
  panels.append((record['nombre'],post.COLORS[p]))
 fig,axes=plt.subplots(2,3,figsize=(17,11))
 for ax,(title,array) in zip(axes.ravel(),panels):ax.imshow(array);ax.set_title(title);ax.axis('off')
 for ax in axes.ravel()[len(panels):]:ax.axis('off')
 fig.suptitle(f'TEST actual · {tile} · {year.replace("_","/")}');fig.tight_layout();fig.savefig(out/'figuras'/f'comparacion_modelos_{year}_{tile}.png',dpi=140);plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exp',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);p.add_argument('--split-config',type=Path,default=CONFIG);p.add_argument('--modelo',choices=['unet','fcrtransunet','transunet','utae'],default=None);p.add_argument('--sin-comparar',action='store_true');p.add_argument('--verificar',action='store_true');p.add_argument('--lock-adquirido',action='store_true');a=p.parse_args()
 cfg=cargar_config(a.split_config);maskbase=preflight(cfg);years=cfg['campanas'];tiles=cfg['test'];out=a.salida
 expcfg=a.exp/'config_multicampana.json'
 traincfg=json.loads(expcfg.read_text()) if expcfg.exists() else {}
 if a.modelo is None:a.modelo=traincfg.get('modelo',next((kind for kind in ['fcrtransunet','transunet','utae'] if kind in a.exp.name),'unet'))
 exp=a.exp.resolve();models={'principal':{'nombre':'U-Net multicampana' if 'multicampana' in exp.name and a.modelo=='unet' else exp.name,'exp':exp,'arquitectura':a.modelo}}
 if not a.sin_comparar:
  for name,folder,arch in [('unet_2019_2020','exp_cordoba_dicmay_unet_f16','unet'),('fcr_2019_2020','exp_cordoba_dicmay_fcrtransunet_f16','fcrtransunet')]:
   path=ext.Y3/'dataset'/folder
   if path.resolve()!=exp:models[name]={'nombre':'U-Net 2019/2020' if arch=='unet' else 'FCRTransUNet 2019/2020','exp':path,'arquitectura':arch}
 for name,record in models.items():
  checkpoint=record['exp']/'best_model.pth';model=constructor(record['arquitectura'])(24,5);model.load_state_dict(torch.load(checkpoint,map_location='cpu'));del model
  record['sha256']=post.sha256(checkpoint)
 for year in years:
  for tile in tiles:
   with rasterio.open(ext.files(year,tile)[0]) as ref:
    for path in ext.files(year,tile):
     with rasterio.open(path) as src:
      if (src.crs,src.transform,src.shape,src.count)!=(ref.crs,ref.transform,ref.shape,4):raise ValueError('Grilla incompatible: '+str(path))
 print(f'PREFLIGHT TEST ACTUAL OK: {tiles}, {len(years)} campanas, {len(models)} modelos',flush=True)
 if a.verificar:return
 out.mkdir(parents=True,exist_ok=True);(out/'figuras').mkdir(exist_ok=True)
 if all((r['exp']/'historial.npy').exists() for r in models.values()):post.curvas(models,out/'figuras/curvas_comparadas.png')
 lock=None
 if not a.lock_adquirido:
  lock=open(ext.Y3/'logs_multicampana/inferencia.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
 failures=auditar_rasters([p for y in years for t in tiles for p in ext.files(y,t)],maskbase/'integridad_composites.json',workers=2)
 failures.update(auditar_rasters([maskbase/'mascaras'/f'{y}_{t}_{s}.tif' for y in years for t in tiles for s in ['completa','independiente']],maskbase/'integridad_mascaras.json',workers=2))
 if failures:raise ValueError(str(failures))
 protocol={'split':cfg,'modelos':{n:{**r,'exp':str(r['exp'])} for n,r in models.items()},'preparacion_base':str(maskbase),'criterio_principal':'independiente','sin_reentrenar':True};post.guardar(out/'protocolo.json',protocol)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda':raise RuntimeError('GPU no disponible')
 torch.set_num_threads(4);records=[];task=0;total=len(models)*len(years)*len(tiles)
 for name,record in models.items():
  modelout=out/'modelos'/name;modelout.mkdir(parents=True,exist_ok=True);(modelout/'figuras').mkdir(exist_ok=True)
  weights=modelout/'modelo_congelado.pth'
  if not weights.exists():shutil.copy2(record['exp']/'best_model.pth',weights)
  if post.sha256(weights)!=record['sha256']:raise ValueError('Pesos congelados modificados')
  model=None
  for year in years:
   for tile in tiles:
    task+=1;post.guardar(out/'progreso.json',{'estado':'inferencia','modelo':name,'campana':year,'tile':tile,'tarea':task,'total':total})
    dest=modelout/'predicciones'/year/f'prediccion_{tile}.tif';tags={'checkpoint_sha256':record['sha256'],'entradas':json.dumps([firma(p) for p in ext.files(year,tile)])};reuse=False
    old=maskbase/'predicciones/multicampana'/year/f'prediccion_{tile}.tif'
    if not dest.exists() and old.exists():
     with rasterio.open(old) as src:compatible=all(src.tags().get(k)==v for k,v in tags.items())
     if compatible:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dest)
    if dest.exists():
     with rasterio.open(dest) as src:reuse=all(src.tags().get(k)==v for k,v in tags.items())
    print(f'[{task}/{total}] {name} {year} {tile} reutilizar={reuse}',flush=True)
    if not reuse:
     if model is None:model=constructor(record['arquitectura'])(24,5).to(device);model.load_state_dict(torch.load(weights,map_location=device));model.eval()
     post.inferir_tile(model,ext.files(year,tile),dest,device,tags=tags)
    for scope in ['completa','independiente']:
     metric=post.metricas_confusion(ext.confusion(dest,maskbase/'mascaras'/f'{year}_{tile}_{scope}.tif'));records.append({'modelo':name,'campana':year,'tile':tile,'criterio':scope,**metric})
    post.guardar(out/'metricas_tiles.json',records);ext.figure(modelout,year,tile,dest,maskbase=maskbase,title=record['nombre'])
  del model;torch.cuda.empty_cache()
 groups=[]
 for name,record in models.items():
  for year in years:
   for scope in ['completa','independiente']:
    cm=sum((np.array(r['confusion'],dtype=np.int64) for r in records if r['modelo']==name and r['campana']==year and r['criterio']==scope),np.zeros((5,5),dtype=np.int64));groups.append({'modelo':name,'campana':year,'criterio':scope,**post.metricas_confusion(cm)});post.matriz_fig(cm,f'{record["nombre"]} · {year} · {scope}',out/'figuras'/f'confusion_{name}_{year}_{scope}.png')
 for year in years:
  for tile in tiles:comparison(out,year,tile,models,maskbase)
 result={'tiles':records,'grupos':groups,'split':cfg,'criterio_principal':'independiente','protocolo':protocol};post.guardar(out/'evaluacion_test_actual.json',result)
 with (out/'comparacion_metricas.csv').open('w',newline='') as f:
  fields=['modelo','campana','criterio','accuracy','f1_macro','miou','pixeles'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in groups)
 fig,axes=plt.subplots(1,2,figsize=(15,5));x=np.arange(len(years));width=.8/len(models)
 for ax,metric in zip(axes,['accuracy','f1_macro']):
  for i,(name,record) in enumerate(models.items()):ax.bar(x+(i-(len(models)-1)/2)*width,[r[metric] for r in groups if r['modelo']==name and r['criterio']=='independiente'],width,label=record['nombre'])
  ax.set_xticks(x,[y.replace('_','/') for y in years]);ax.set_ylim(0,1);ax.set_title(metric);ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
 fig.suptitle('TEST vigente: 20HNJ/20HNH · mismos pixeles fuera de TRAIN/VAL');fig.tight_layout();fig.savefig(out/'figuras/comparacion_test_actual.png',dpi=160);plt.close(fig)
 for name,record in models.items():
  post.guardar(record['exp']/'test_actual.json',{'salida':str(out),'modelo':name,'split':cfg,'grupos':[r for r in groups if r['modelo']==name],'nota':'Este reporte sustituye el TEST historico para comparaciones actuales; pesos y resultados antiguos preservados.'})
 (out/'README.md').write_text('TEST vigente: 20HNJ y 20HNH. Criterio principal: fuera de grillas TRAIN y VAL.\nLos mismos composites y pixeles se usan para todos los modelos. Checkpoints congelados, sin reentrenar.\n2021/2022 no visto temporalmente; tres anios anteriores vistos en otras zonas.\nReferencias INTA, no campo independiente. Metricar mapas 512 no equivale a metricar parches 256.\nLa arquitectura y los anios de entrenamiento difieren entre modelos; no es una ablacion controlada.\n')
 post.guardar(out/'progreso.json',{'estado':'finalizado','tareas':total});print('EVALUACION TEST ACTUAL FINALIZADA',flush=True)
if __name__=='__main__':main()
