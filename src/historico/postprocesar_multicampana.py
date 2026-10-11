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
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
from model import SimpleUNet, FCRTransUNet
from utils.multicampana import SPLIT
from src.soporte.archivos import guardar, sha256
from utils.rutas import MULTICAMPANA_ROOT
from src.inferencia.raster import normalizar, inferir_tile
from src.evaluacion.metricas import metricas_confusion, confusion_mnc as evaluar
from src.datos.leyendas import REMAP_1819, REMAP_2020, REMAP_2021, LEGENDS
from src.figuras.comunes import NAMES, COLORS, matriz_fig, curvas
from src.figuras.historico import figura_tile as _figura_tile, resumen_multicampana as _resumen

YEARS = ['2018_2019','2019_2020','2020_2021']


def figura_tile(year,tile,campaign,models,out):
    return _figura_tile(year,tile,campaign,models,out,ROOT/'mascaras_procesadas',SPLIT)


def resumen(records,models,out):
    return _resumen(records,models,out,YEARS)


def crear_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp',type=Path,required=True)
    parser.add_argument('--salida',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=MULTICAMPANA_ROOT)
    parser.add_argument('--verificar',action='store_true')
    return parser


def ejecutar_postprocesado(args):
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
    subprocess.run([sys.executable,'-m','src.figuras.curvas_entrenamiento','--exp',str(exp)],cwd=ROOT,check=True)
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



def main(argv=None):
    args = crear_parser().parse_args(argv)
    return ejecutar_postprocesado(args)


if __name__ == '__main__':
    main()
