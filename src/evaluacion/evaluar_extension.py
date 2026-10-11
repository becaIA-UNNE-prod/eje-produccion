"""Evalúa la U-Net congelada del flujo específico de extensión."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import SimpleUNet
from src.soporte.archivos import guardar, sha256
from src.datos.campanas import files
from src.datos.preparar_extension import YEARS, TILES
from utils.integridad_raster import firma
from src.figuras.comunes import matriz_fig
from src.figuras.test_actual import figura_extension
from src.inferencia.raster import inferir_tile
from src.evaluacion.metricas import metricas_confusion, confusion_remapeada as confusion

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

def crear_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--salida', type=Path, required=True)
    return parser


def main(argv=None):
    args = crear_parser().parse_args(argv)
    args.salida.mkdir(parents=True, exist_ok=True)
    evaluate(args.salida)


if __name__ == '__main__':
    main()
