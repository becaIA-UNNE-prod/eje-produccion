#!/usr/bin/env python3
"""Mascaras y evaluacion de nuevo TEST, conservando resultados previos."""
import argparse,json,sys,csv
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject,transform_geom,Resampling
from rasterio.features import rasterize
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from model import SimpleUNet
from src.soporte.archivos import guardar, sha256
from src.datos.campanas import files, fuentes_mnc
from utils.rutas import EXTENSION_ROOT as BASE, MULTICAMPANA_ROOT as Y3, MNC_ROOT as MNC
from utils.integridad_raster import auditar_rasters,firma
from utils.multicampana import SPLIT
from src.datos.leyendas import legend, check_codes
from src.figuras.comunes import matriz_fig
from src.figuras.test_actual import figura_extension
from src.inferencia.raster import inferir_tile
from src.evaluacion.metricas import metricas_confusion, confusion_remapeada as confusion
YEARS=['2018_2019','2019_2020','2020_2021','2021_2022'];TILES=['20HNJ','20HNH']
SOURCES=fuentes_mnc(YEARS)

def shapes(crs):
 result=[]
 for directory,month in [('composites_2018_2019_scl','201812'),('composites_dicmay_2019_2020','201912'),('composites_2020_2021_scl','202012')]:
  for tile in SPLIT['train']+SPLIT['val']:
   with rasterio.open(Y3/directory/f'{tile}_{month}_median.tif') as src:
    b=src.bounds;poly={'type':'Polygon','coordinates':[[[b.left,b.bottom],[b.right,b.bottom],[b.right,b.top],[b.left,b.top],[b.left,b.bottom]]]}
    result.append((transform_geom(src.crs,crs,poly),1))
 return result

def write(path,data,ref,tags):
 profile=ref.profile.copy();profile.update(count=1,dtype='uint8',nodata=0,compress='deflate',predictor=1,tiled=True,blockxsize=256,blockysize=256)
 temp=path.with_suffix('.partial.tif')
 with rasterio.open(temp,'w',**profile) as dst:dst.write(data,1);dst.update_tags(**tags)
 temp.replace(path)

def prepare(out):
 failures=auditar_rasters([p for y in YEARS for t in TILES for p in files(y,t)],out/'integridad_composites.json',workers=2)
 if failures:raise ValueError(f'Composites corruptos: {failures}')
 records=[];(out/'mascaras').mkdir(exist_ok=True)
 for year in YEARS:
  lut,allowed,nodata=legend(year)
  for tile in TILES:
   with rasterio.open(files(year,tile)[0]) as ref,rasterio.open(SOURCES[year]) as src:
    if src.count!=1 or src.dtypes!=('uint8',) or src.nodata!=nodata:raise ValueError(f'{year}: fuente inesperada')
    for p in files(year,tile):
     with rasterio.open(p) as r:
      if (r.crs,r.transform,r.shape,r.count)!=(ref.crs,ref.transform,ref.shape,4):raise ValueError(f'Grilla incompatible: {p}')
    raw=np.full(ref.shape,nodata,dtype=np.uint8)
    reproject(source=rasterio.band(src,1),destination=raw,src_transform=src.transform,src_crs=src.crs,src_nodata=nodata,dst_transform=ref.transform,dst_crs=ref.crs,dst_nodata=nodata,resampling=Resampling.nearest,num_threads=4)
    codes,counts=np.unique(raw,return_counts=True);check_codes(codes,allowed);target=lut[raw];del raw
    excluded=rasterize(shapes(ref.crs),out_shape=ref.shape,transform=ref.transform,fill=0,all_touched=True,dtype='uint8')
    full=int((target>0).sum());strict=int(((target>0)&(excluded==0)).sum())
    if not full or not strict:raise ValueError(f'{tile}: mascara sin area valida o independiente')
    tags={'campana':year,'tile':tile,'fuente':str(SOURCES[year]),'fuente_firma':json.dumps(firma(SOURCES[year]))}
    write(out/'mascaras'/f'{year}_{tile}_completa.tif',target,ref,tags);target[excluded>0]=0
    write(out/'mascaras'/f'{year}_{tile}_independiente.tif',target,ref,{**tags,'exclusion':'grillas completas TRAIN y VAL; all_touched=True'})
    records.append({'campana':year,'tile':tile,'codigos':{str(int(c)):int(n) for c,n in zip(codes,counts)},'pixeles_completos':full,'pixeles_independientes':strict,'excluidos':full-strict,'codigos_ignorados_excepcionales':([0] if year=='2019_2020' else [20] if year=='2020_2021' else []),'fuente_firma':firma(SOURCES[year])})
    guardar(out/'labels_verificados.json',records);print(f'MASCARA {year} {tile}: {full} validos; {strict} independientes',flush=True)
 failures=auditar_rasters(list((out/'mascaras').glob('*.tif')),out/'integridad_mascaras.json',workers=2)
 if failures:raise ValueError(f'Mascaras corruptas: {failures}')
 guardar(out/'preparacion_verificada.json',{'campanas':YEARS,'tiles':TILES,'composites':48,'mascaras':16,'codigo20_2021':'NoData; excepcion autorizada, exclusiva de 2021'})


def figure(out,year,tile,pred,maskbase=None,title="U-Net multicampana"):
 return figura_extension(out,year,tile,pred,files(year,tile)[2],maskbase,title)

def evaluate(out):
 protocol=json.loads((out/'protocolo.json').read_text());weights=out/'modelo_congelado.pth'
 if sha256(weights)!=protocol['checkpoint_sha256']:raise ValueError('Checkpoint congelado modificado')
 if not (out/'preparacion_verificada.json').exists():raise ValueError('Preparacion no verificada')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda':raise RuntimeError('GPU no disponible')
 torch.set_num_threads(4);model=SimpleUNet(24,5).to(device);model.load_state_dict(torch.load(weights,map_location=device));model.eval();records=[];(out/'figuras').mkdir(exist_ok=True)
 for year in YEARS:
  for tile in TILES:
   guardar(out/'progreso.json',{'estado':'inferencia','campana':year,'tile':tile,'tarea':len(records)//2+1,'total':8})
   dest=out/'predicciones/multicampana'/year/f'prediccion_{tile}.tif';tags={'checkpoint_sha256':protocol['checkpoint_sha256'],'entradas':json.dumps([firma(p) for p in files(year,tile)])};reuse=False
   if dest.exists():
    with rasterio.open(dest) as src:reuse=all(src.tags().get(k)==v for k,v in tags.items())
   if not reuse:inferir_tile(model,files(year,tile),dest,device,tags=tags)
   for scope in ['completa','independiente']:
    metric=metricas_confusion(confusion(dest,out/'mascaras'/f'{year}_{tile}_{scope}.tif'));records.append({'campana':year,'tile':tile,'criterio':scope,**metric});print(f'{year} {tile} {scope}: accuracy={metric["accuracy"]:.4f} F1={metric["f1_macro"]:.4f}',flush=True)
   guardar(out/'metricas_tiles.json',records);figure(out,year,tile,dest)
 groups=[]
 for year in YEARS:
  for scope in ['completa','independiente']:
   cm=sum((np.array(r['confusion'],dtype=np.int64) for r in records if r['campana']==year and r['criterio']==scope),np.zeros((5,5),dtype=np.int64));groups.append({'campana':year,'criterio':scope,**metricas_confusion(cm)});matriz_fig(cm,f'Nuevo TEST {year} · {scope}',out/'figuras'/f'confusion_{year}_{scope}.png')
 guardar(out/'evaluacion_nuevo_test.json',{'checkpoint_sha256':protocol['checkpoint_sha256'],'tiles':records,'grupos':groups,'nota':'2021/2022 es prueba temporal y espacial; otros anos entrenados en otras zonas. Independiente excluye TRAIN y VAL. Referencia INTA, no campo independiente.'})
 with (out/'metricas_resumen.csv').open('w',newline='') as f:
  fields=['campana','criterio','accuracy','f1_macro','miou','pixeles'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in groups)
 fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(4)
 for ax,metric in zip(axes,['accuracy','f1_macro']):
  for i,scope in enumerate(['completa','independiente']):ax.bar(x+(i-.5)*.35,[r[metric] for r in groups if r['criterio']==scope],.35,label=scope)
  ax.set_xticks(x,[y.replace('_','/') for y in YEARS]);ax.set_ylim(0,1);ax.set_title(metric);ax.legend();ax.grid(axis='y',alpha=.2)
 fig.suptitle('20HNJ y 20HNH · U-Net congelada · completo vs fuera de TRAIN/VAL');fig.tight_layout();fig.savefig(out/'figuras/resumen_nuevo_test.png',dpi=160);plt.close(fig);guardar(out/'progreso.json',{'estado':'finalizado','tareas':8});print('NUEVO TEST FINALIZADO',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--salida',type=Path,required=True);p.add_argument('--modo',choices=['preparar','evaluar'],required=True);a=p.parse_args();a.salida.mkdir(parents=True,exist_ok=True);{'preparar':prepare,'evaluar':evaluate}[a.modo](a.salida)
