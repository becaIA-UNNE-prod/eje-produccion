"""Comparacion de mapas del protocolo espacial historico."""
import csv
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .comunes import NAMES, COLORS, matriz_fig
from src.evaluacion.metricas import metricas_confusion
from src.soporte.archivos import guardar
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


def resumen_multicampana(records,models,out,years):
    aggregates = {}
    for record in records:
        key=(record['modelo'],record['campana'],record['rol'])
        aggregates.setdefault(key,np.zeros((5,5),dtype=np.int64))
        aggregates[key] += np.array(record['confusion'],dtype=np.int64)
    values = []
    for (model,year,role),cm in aggregates.items():
        metric = metricas_confusion(cm)
        values.append({'modelo':model,'campana':year,'rol':role,**metric})
        matriz_fig(cm,f'{models[model]["nombre"]} | {year} | {role.upper()}',out/'figuras'/f'confusion_{model}_{year}_{role}.png')
    guardar(out/'metricas_mapas.json',{'tiles':records,'grupos':values,
            'criterio':'MNC>0; prediccion=0 cuenta como error. Mapas completos, ventanas 512. No mezclar con metricas de parches 256.'})
    fields=['modelo','campana','rol','accuracy','f1_macro','miou','pixeles','sin_prediccion']+[f'f1_{c}' for c in NAMES[1:]]
    with (out/'comparacion_metricas.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields); writer.writeheader()
        for row in values:
            writer.writerow({**{key:row[key] for key in fields if key in row},
                             'sin_prediccion':sum(r[0] for r in row['confusion'][1:]),
                             **{f'f1_{name}':row['clases'][name]['f1'] for name in NAMES[1:]}})
    tests = {(row['modelo'],row['campana']):row for row in values if row['rol']=='test'}
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    model_names=list(models); width=.8/max(len(model_names),1); x=np.arange(3)
    for i,name in enumerate(model_names):
        for ax,metric in zip(axes,['accuracy','f1_macro']):
            vals=[tests.get((name,year),{}).get(metric,np.nan) for year in years]
            ax.bar(x+(i-(len(model_names)-1)/2)*width,vals,width,label=models[name]['nombre'])
            ax.set_xticks(x,[y.replace('_','/') for y in years]); ax.set_ylim(0,1)
            ax.set_title('TEST: '+metric); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Mismos tiles TEST, composites y mascara MNC para todos los modelos',fontsize=12)
    fig.tight_layout(); fig.savefig(out/'figuras/comparacion_test.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    x=np.arange(4)
    for ax,year in zip(axes,years):
        for i,name in enumerate(model_names):
            row=tests.get((name,year))
            if row:
                ax.bar(x+(i-(len(model_names)-1)/2)*width,[row['clases'][c]['f1'] for c in NAMES[1:]],width,label=models[name]['nombre'])
        ax.set_xticks(x,NAMES[1:]); ax.set_ylim(0,1); ax.set_title(year.replace('_','/')); ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('F1 por clase en TEST: mapas completos')
    fig.tight_layout(); fig.savefig(out/'figuras/comparacion_f1_clases.png',dpi=160); plt.close(fig)
