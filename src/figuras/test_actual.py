"""Figuras del TEST vigente a partir de rutas explicitas."""
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .comunes import NAMES, COLORS

def figura_extension(out,year,tile,pred,rgb_path,maskbase=None,title="U-Net multicampana"):
 maskbase=maskbase or out
 size=(1024,1024)
 with rasterio.open(rgb_path) as src:rgb=src.read([3,2,1],out_shape=(3,*size)).astype(np.float32).transpose(1,2,0)
 for c in range(3):
  vals=rgb[:,:,c][rgb[:,:,c]>0];lo,hi=np.percentile(vals,[2,98]) if vals.size else (0,1);rgb[:,:,c]=np.clip((rgb[:,:,c]-lo)/max(hi-lo,1),0,1)
 targets=[]
 for scope in ['completa','independiente']:
  with rasterio.open(maskbase/'mascaras'/f'{year}_{tile}_{scope}.tif') as src:targets.append(src.read(1,out_shape=size,resampling=Resampling.nearest))
 with rasterio.open(pred) as src:p=src.read(1,out_shape=size,resampling=Resampling.nearest)
 err=np.full((*size,3),55,dtype=np.uint8);err[(targets[1]>0)&(p==targets[1])]=[0,180,0];err[(targets[1]>0)&(p!=targets[1])]=[220,0,0]
 panels=[('RGB febrero',rgb),('INTA completa',COLORS[targets[0]]),('INTA fuera de TRAIN/VAL',COLORS[targets[1]]),(title,COLORS[p]),('Error area independiente',err),('Solapamiento excluido (blanco)',((targets[0]>0)&(targets[1]==0)).astype(np.uint8)*255)]
 fig,axes=plt.subplots(2,3,figsize=(17,11))
 for ax,(panel_title,array) in zip(axes.ravel(),panels):ax.imshow(array,cmap='gray',vmin=0,vmax=255);ax.set_title(panel_title);ax.axis('off')
 fig.suptitle(f'{year.replace("_","/")} · {tile} · TEST vigente · {title}');
 from matplotlib.patches import Patch
 fig.legend(handles=[Patch(color=COLORS[i]/255,label=NAMES[i]) for i in range(5)],loc='lower center',ncol=5)
 fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(out/'figuras'/f'comparacion_{year}_{tile}.png',dpi=140);plt.close(fig)

def comparar_modelos(out,year,tile,models,maskbase,rgb_path):
 size=(1024,1024)
 with rasterio.open(rgb_path) as src:rgb=src.read([3,2,1],out_shape=(3,*size)).astype(np.float32).transpose(1,2,0)
 for c in range(3):
  valid=rgb[:,:,c][rgb[:,:,c]>0];lo,hi=np.percentile(valid,[2,98]) if valid.size else (0,1);rgb[:,:,c]=np.clip((rgb[:,:,c]-lo)/max(hi-lo,1),0,1)
 with rasterio.open(maskbase/'mascaras'/f'{year}_{tile}_independiente.tif') as src:target=src.read(1,out_shape=size,resampling=Resampling.nearest)
 panels=[('RGB febrero',rgb),('INTA fuera de TRAIN/VAL',COLORS[target])]
 for name,record in models.items():
  with rasterio.open(out/'modelos'/name/'predicciones'/year/f'prediccion_{tile}.tif') as src:p=src.read(1,out_shape=size,resampling=Resampling.nearest)
  panels.append((record['nombre'],COLORS[p]))
 fig,axes=plt.subplots(2,3,figsize=(17,11))
 for ax,(title,array) in zip(axes.ravel(),panels):ax.imshow(array);ax.set_title(title);ax.axis('off')
 for ax in axes.ravel()[len(panels):]:ax.axis('off')
 fig.suptitle(f'TEST actual · {tile} · {year.replace("_","/")}');fig.tight_layout();fig.savefig(out/'figuras'/f'comparacion_modelos_{year}_{tile}.png',dpi=140);plt.close(fig)
