"""Comparacion de mapas del protocolo espacial historico."""
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .comunes import NAMES, COLORS
from src.datos.leyendas import LEGENDS

def figura_tile(year,tile,campaign,models,out,mascaras,split):
    shape = (1024,1024)
    remap,pattern = LEGENDS[year]
    mask = mascaras/pattern.format(tile)
    with rasterio.open(mask) as src:
        scale = min(1,1024/max(src.shape))
        shape = (max(1,round(src.height*scale)),max(1,round(src.width*scale)))
        target = remap[src.read(1,out_shape=shape,resampling=Resampling.nearest)]
    with rasterio.open(campaign['archivos'][tile][2]) as src:
        rgb = src.read([3,2,1],out_shape=(3,*shape)).astype(np.float32).transpose(1,2,0)
        for c in range(3):
            values = rgb[:,:,c][rgb[:,:,c]>0]
            lo,hi = np.percentile(values,[2,98]) if len(values) else (0,1)
            rgb[:,:,c] = np.clip((rgb[:,:,c]-lo)/max(hi-lo,1),0,1)
    panels = [('RGB febrero (contraste 2-98%)',rgb),('Referencia MNC',COLORS[target])]
    multi = None
    for name,record in models.items():
        path = out/'predicciones'/name/year/f'prediccion_{tile}.tif'
        if not path.exists(): continue
        with rasterio.open(path) as src:
            pred = src.read(1,out_shape=shape,resampling=Resampling.nearest)
        panels.append((record['nombre'],COLORS[pred]))
        if name=='multicampana': multi=pred
    error = np.full((*shape,3),55,dtype=np.uint8)
    error[(target>0)&(multi==target)]=[0,180,0]
    error[(target>0)&(multi!=target)]=[220,0,0]
    panels.append(('Error multicampana: verde acierto, rojo error',error))
    cols=3; rows=(len(panels)+cols-1)//cols
    fig,axes = plt.subplots(rows,cols,figsize=(17,5.5*rows))
    for ax,(name,array) in zip(axes.ravel(),panels):
        ax.imshow(array); ax.set_title(name,fontsize=10); ax.axis('off')
    for ax in axes.ravel()[len(panels):]: ax.axis('off')
    legend = [Patch(color=COLORS[i]/255,label=NAMES[i]) for i in range(5)]
    fig.legend(handles=legend,loc='lower center',ncol=5)
    fig.suptitle(f'{year.replace("_","/")} | {tile} | '+('TEST' if tile in split['test'] else 'VAL' if tile in split['val'] else 'TRAIN (descriptivo)'),fontsize=13)
    fig.tight_layout(rect=(0,.04,1,.96))
    destination = out/'figuras'/year/f'comparacion_{tile}.png'
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=140); plt.close(fig)
