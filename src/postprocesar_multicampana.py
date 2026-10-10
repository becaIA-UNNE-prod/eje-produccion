#!/usr/bin/env python3
"""Inferencia completa, figuras y comparacion espacial de tres campanas."""
import argparse
from contextlib import ExitStack
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.model import SimpleUNet, FCRTransUNet
from utils.multicampana import SPLIT, metricas_confusion
from utils.leyendas_mnc import REMAP_2021, verificar_codigos

NAMES = ['NoData','Fondo','Maiz','Soja','Mani']
COLORS = np.array([[0,0,0],[70,130,180],[255,165,0],[0,100,0],[165,42,42]],dtype=np.uint8)
YEARS = ['2018_2019','2019_2020','2020_2021']
REMAP_1819 = np.zeros(256,dtype=np.uint8)
REMAP_1819[[0,3]], REMAP_1819[[1,2]], REMAP_1819[4] = 2,3,4
REMAP_1819[[5,6,7,8,9,11,12,13,14,15]] = 1
REMAP_2020 = np.zeros(256,dtype=np.uint8)
REMAP_2020[[10,11,16]] = [2,3,4]
REMAP_2020[[12,13,14,15,17,18,19,21,22]] = 1
LEGENDS = {'2018_2019':(REMAP_1819,'etiqueta_mnc1819_{}_10m.tif'),
           '2019_2020':(REMAP_2020,'etiqueta_mnc_{}_10m.tif'),
           '2020_2021':(REMAP_2021,'etiqueta_mnc2021_{}_10m.tif')}


def guardar(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for data in iter(lambda:file.read(8*1024**2),b''):
            digest.update(data)
    return digest.hexdigest()


def normalizar(array):
    # Igual al dataset: dividir /10000, almacenar f16 y convertir a f32.
    x = (array.astype(np.float32)/10000).astype(np.float16).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Composite con NaN/Inf o valores fuera de float16')
    return x


def inferir_tile(model, archivos, destino, device, batch=4, size=512, tags=None):
    destino.parent.mkdir(parents=True,exist_ok=True)
    partial = destino.with_suffix('.partial.tif')
    tags = tags or {}
    started = time.monotonic()
    try:
        with ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(path)) for path in archivos]
            ref = sources[0]
            grid = (ref.shape,ref.transform,ref.crs)
            if any((src.shape,src.transform,src.crs)!=grid or src.count!=4 for src in sources):
                raise ValueError('Composites incompatibles en grilla/bandas')
            profile = ref.profile.copy()
            profile.update(count=1,dtype='uint8',nodata=0,compress='deflate',predictor=1,
                           tiled=True,blockxsize=256,blockysize=256)
            windows = [Window(x,y,min(size,ref.width-x),min(size,ref.height-y))
                       for y in range(0,ref.height,size) for x in range(0,ref.width,size)]
            with rasterio.open(partial,'w',**profile) as dst, torch.inference_mode():
                dst.update_tags(**tags)
                for offset in range(0,len(windows),batch):
                    selected = windows[offset:offset+batch]
                    inputs, missing = [],[]
                    for window in selected:
                        x = normalizar(np.concatenate([src.read(window=window) for src in sources],axis=0))
                        missing.append((x==0).all(axis=0))
                        h,w = x.shape[-2:]
                        inputs.append(np.pad(x,((0,0),(0,size-h),(0,size-w)),mode='edge'))
                    tensor = torch.from_numpy(np.stack(inputs)).to(device)
                    with torch.autocast(device_type=device.type,enabled=(device.type=='cuda')):
                        logits = model(tensor)
                    if not torch.isfinite(logits).all():
                        raise ValueError('El modelo produjo NaN/Inf')
                    pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
                    for i,window in enumerate(selected):
                        result = pred[i,:int(window.height),:int(window.width)].copy()
                        result[missing[i]]=0
                        dst.write(result,1,window=window)
                    if offset==0 or (offset//batch)%20==0 or offset+batch>=len(windows):
                        print(f'  {destino.name}: {min(offset+batch,len(windows))}/{len(windows)} bloques, '
                              f'{time.monotonic()-started:.0f}s',flush=True)
        partial.replace(destino)
    except BaseException:
        if partial.exists(): partial.unlink()
        raise
    return time.monotonic()-started


def evaluar(pred_path,mask_path,remap,year):
    cm = np.zeros((5,5),dtype=np.int64)
    with rasterio.open(pred_path) as pred, rasterio.open(mask_path) as mask:
        if (pred.shape,pred.transform,pred.crs)!=(mask.shape,mask.transform,mask.crs):
            raise ValueError('Grilla distinta: '+str(pred_path))
        for _,window in pred.block_windows(1):
            raw = mask.read(1,window=window)
            if year=='2020_2021': verificar_codigos(np.unique(raw),ignorar_codigo20=True)
            y = remap[raw]
            p = pred.read(1,window=window)
            if p.max()>4: raise ValueError('Prediccion fuera de clases 0..4')
            valid = y>0  # misma mascara MNC para todos; pred=0 cuenta como error
            cm += np.bincount(y[valid].astype(np.int64)*5+p[valid],minlength=25).reshape(5,5)
    return cm


def matriz_fig(cm,title,path):
    norm = cm[1:,:]/np.maximum(cm[1:,:].sum(1,keepdims=True),1)
    fig,ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    for i in range(4):
        for j in range(5): ax.text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',color='white' if norm[i,j]>.5 else 'black')
    ax.set_xticks(range(5),NAMES); ax.set_yticks(range(4),NAMES[1:])
    ax.set_xlabel('Prediccion'); ax.set_ylabel('MNC'); ax.set_title(title,fontsize=10)
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def curvas(models,path):
    fig,axes = plt.subplots(2,2,figsize=(14,10))
    for name,record in models.items():
        h = np.load(record['exp']/'historial.npy',allow_pickle=False)
        for ax,col,title in zip(axes.ravel(),[1,3,2,4],['Train loss','Val loss','Train accuracy','Val accuracy']):
            ax.plot(h[:,0],h[:,col],label=record['nombre']); ax.set_title(title)
            ax.set_xlabel('Epoca'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Curvas de entrenamientos: VAL multicampana y VAL 2019/2020 son conjuntos distintos',fontsize=11)
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def figura_tile(year,tile,campaign,models,out):
    shape = (1024,1024)
    remap,pattern = LEGENDS[year]
    mask = ROOT/'mascaras_procesadas'/pattern.format(tile)
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
    fig.suptitle(f'{year.replace("_","/")} | {tile} | '+('TEST' if tile in SPLIT['test'] else 'VAL' if tile in SPLIT['val'] else 'TRAIN (descriptivo)'),fontsize=13)
    fig.tight_layout(rect=(0,.04,1,.96))
    destination = out/'figuras'/year/f'comparacion_{tile}.png'
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=140); plt.close(fig)


def resumen(records,models,out):
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
            vals=[tests.get((name,year),{}).get(metric,np.nan) for year in YEARS]
            ax.bar(x+(i-(len(model_names)-1)/2)*width,vals,width,label=models[name]['nombre'])
            ax.set_xticks(x,[y.replace('_','/') for y in YEARS]); ax.set_ylim(0,1)
            ax.set_title('TEST: '+metric); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Mismos tiles TEST, composites y mascara MNC para todos los modelos',fontsize=12)
    fig.tight_layout(); fig.savefig(out/'figuras/comparacion_test.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    x=np.arange(4)
    for ax,year in zip(axes,YEARS):
        for i,name in enumerate(model_names):
            row=tests.get((name,year))
            if row:
                ax.bar(x+(i-(len(model_names)-1)/2)*width,[row['clases'][c]['f1'] for c in NAMES[1:]],width,label=models[name]['nombre'])
        ax.set_xticks(x,NAMES[1:]); ax.set_ylim(0,1); ax.set_title(year.replace('_','/')); ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('F1 por clase en TEST: mapas completos')
    fig.tight_layout(); fig.savefig(out/'figuras/comparacion_f1_clases.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp',type=Path,required=True)
    parser.add_argument('--salida',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=Path('/mnt/yacy_3/prod_bis/ferreyra'))
    parser.add_argument('--verificar',action='store_true')
    args=parser.parse_args()
    exp=args.exp.resolve(); out=args.salida.resolve(); base=args.base
    if (exp/'train.exitcode').read_text().strip()!='0': raise ValueError('Entrenamiento no termino correctamente')
    config=json.loads((exp/'config_multicampana.json').read_text())
    if config['modelo']!='unet' or config['canales']!=24 or config['split']!=SPLIT:
        raise ValueError('Modelo/configuracion/split incompatibles')
    models={'multicampana':{'nombre':'U-Net multicampana','clase':SimpleUNet,'exp':exp},
            'unet_2019_2020':{'nombre':'U-Net 2019/2020','clase':SimpleUNet,'exp':base/'dataset/exp_cordoba_dicmay_unet_f16'},
            'fcr_2019_2020':{'nombre':'FCRTransUNet 2019/2020','clase':FCRTransUNet,'exp':base/'dataset/exp_cordoba_dicmay_fcrtransunet_f16'}}
    for name,record in list(models.items()):
        if not (record['exp']/'best_model.pth').exists():
            if name=='multicampana': raise FileNotFoundError('Sin best_model.pth')
            print(f'Comparacion omitida: {name}, sin checkpoint',flush=True); del models[name]; continue
        state=torch.load(record['exp']/'best_model.pth',map_location='cpu')
        model=record['clase'](24,5); model.load_state_dict(state)
        record['sha256']=sha256(record['exp']/'best_model.pth')
        del model,state
    dirs={'2018_2019':base/'composites_2018_2019_scl','2019_2020':base/'composites_dicmay_2019_2020','2020_2021':base/'composites_2020_2021_scl'}
    campaigns={}; tiles=SPLIT['test']+SPLIT['val']+SPLIT['train']
    for year in YEARS:
        first=int(year[:4]); months=[f'{first}12']+[f'{first+1}{month:02}' for month in range(1,6)]
        campaigns[year]={'archivos':{},'firmas':{}}
        remap,pattern=LEGENDS[year]
        for tile in tiles:
            files=[dirs[year]/f'{tile}_{month}_median.tif' for month in months]
            with ExitStack() as stack:
                sources=[stack.enter_context(rasterio.open(path)) for path in files]
                mask=stack.enter_context(rasterio.open(ROOT/'mascaras_procesadas'/pattern.format(tile)))
                grid=(mask.shape,mask.transform,mask.crs)
                if any((src.shape,src.transform,src.crs)!=grid or src.count!=4 for src in sources):
                    raise ValueError(f'Grilla o bandas incompatibles: {year}/{tile}')
            signature=[(str(path),path.stat().st_size,path.stat().st_mtime_ns) for path in files]
            campaigns[year]['archivos'][tile]=files
            campaigns[year]['firmas'][tile]=hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    print(f'PREFLIGHT OK: 51 tiles/campanas, {len(models)} modelos compatibles',flush=True)
    if args.verificar: return
    out.mkdir(parents=True,exist_ok=True); (out/'figuras').mkdir(exist_ok=True)
    curvas(models,out/'figuras/curvas_comparadas.png')
    subprocess.run([sys.executable,'src/4_generar_metricas.py','--exp',str(exp)],cwd=ROOT,check=True)
    metrics_train=json.loads((exp/'metricas_multicampana.json').read_text())
    for role in ['val','test']:
        for year,record in metrics_train[role].items():
            matriz_fig(np.array(record['confusion']),f'Parches 256 | {role.upper()} | {year}',out/'figuras'/f'confusion_parches_{role}_{year}.png')
    tasks=[(name,year,tile) for name in models for year in YEARS for tile in SPLIT['test']]
    tasks += [('multicampana',year,tile) for year in YEARS for tile in SPLIT['val']+SPLIT['train']]
    guardar(out/'plan.json',{'exp':str(exp),'ventana':'diciembre-mayo','bandas':['B02','B03','B04','B08'],
            'modelos':{name:{'nombre':r['nombre'],'exp':str(r['exp']),'sha256':r['sha256']} for name,r in models.items()},
            'tiles_multicampana':51,'tareas':tasks,'criterio':'TEST espacial comun. Multicampana vio TRAIN de tres anios; baselines solo 2019/2020. TRAIN es descriptivo.'})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('Sin GPU disponible; no lanzar 63 tiles en CPU silenciosamente')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    active=None; model=None; records=[]; started=time.monotonic()
    for index,(name,year,tile) in enumerate(tasks,1):
        destination=out/'predicciones'/name/year/f'prediccion_{tile}.tif'
        record=models[name]; tags={'modelo':name,'checkpoint_sha256':record['sha256'],'campana':year,
                                  'inputs_sha256':campaigns[year]['firmas'][tile],'ventana':'dic-may','size':'512','normalizacion':'float16_div10000'}
        reuse=False
        if destination.exists():
            with rasterio.open(destination) as src:
                reuse=all(src.tags().get(k)==v for k,v in tags.items())
        guardar(out/'progreso.json',{'estado':'inferencia','tarea':index,'total':len(tasks),
                                   'modelo':name,'campana':year,'tile':tile,'minutos':(time.monotonic()-started)/60})
        print(f'[{index}/{len(tasks)}] {name} {year} {tile}',flush=True)
        if not reuse:
            if active!=name:
                if model is not None: del model; torch.cuda.empty_cache()
                model=record['clase'](24,5).to(device)
                model.load_state_dict(torch.load(record['exp']/'best_model.pth',map_location=device)); model.eval(); active=name
            inferir_tile(model,campaigns[year]['archivos'][tile],destination,device,tags=tags)
        remap,pattern=LEGENDS[year]
        cm=evaluar(destination,ROOT/'mascaras_procesadas'/pattern.format(tile),remap,year)
        role=next(role for role,group in SPLIT.items() if tile in group)
        metric=metricas_confusion(cm); records.append({'modelo':name,'campana':year,'tile':tile,'rol':role,**metric})
        guardar(out/'metricas_tiles.json',records)
        print(f'  MAPA OK | accuracy={metric["accuracy"]:.4f} F1_macro={metric["f1_macro"]:.4f}',flush=True)
        if (out/'predicciones/multicampana'/year/f'prediccion_{tile}.tif').exists():
            figura_tile(year,tile,campaigns[year],models,out)
        # Las primeras 18 tareas producen toda la comparacion TEST; despues se completa TRAIN/VAL.
        if index==len(models)*len(YEARS)*len(SPLIT['test']): resumen(records,models,out)
    resumen(records,models,out)
    report=['# Inferencia multicampana y comparacion','',
            'Se usaron los mejores checkpoints de VAL, mismas seis posiciones mensuales y 5 clases.',
            'Comparacion sobre los mismos tiles TEST 20HLK/20HLJ y composites. Baselines entrenados en 2019/2020; multicampana en tres anios.',
            'No es una prueba de anio no visto para multicampana. TRAIN se muestra solo descriptivamente.',
            'Las curvas VAL de distintos conjuntos no sirven para ordenar directamente los modelos.',
            'Metricas de mapas completos (ventanas 512) y de parches (256) se reportan por separado.',
            'MNC>0 es la mascara fija; una prediccion NoData cuenta como error. 2021 codigo 20 se ignora como NoData por excepcion documentada.',
            'Figuras de mapas se reducen a 1024 px; las metricas se calculan a resolucion completa.',
            '', 'Ver comparacion_metricas.csv, metricas_mapas.json, figuras/ y predicciones/.']
    (out/'README.md').write_text('\n'.join(report)+'\n')
    guardar(out/'progreso.json',{'estado':'finalizado','tareas':len(tasks),'minutos':(time.monotonic()-started)/60})
    print('INFERENCIA Y FIGURAS FINALIZADAS',flush=True)


if __name__=='__main__':
    if '--historico' in sys.argv:
        sys.argv.remove('--historico')
        if '--lock-adquirido' in sys.argv: sys.argv.remove('--lock-adquirido')
        main()
    else:
        from evaluar_test_actual import main as test_actual_main
        test_actual_main()
