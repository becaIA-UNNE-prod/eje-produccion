#!/usr/bin/env python3
"""Evaluacion temporal de checkpoints congelados sobre MNC 2021/2022."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject,Resampling
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
from src.historico import postprocesar_multicampana as post
from src.inferencia.raster import inferir_tile
from src.figuras.comunes import matriz_fig
from src.evaluacion.metricas import metricas_confusion, confusion_mnc
from utils.integridad_raster import auditar_rasters


def crear_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida',type=Path,required=True)
    parser.add_argument('--exp',type=Path,required=True)
    return parser


def ejecutar_evaluacion(args):
    out=args.salida; base=Path('/mnt/yacy_3/prod_bis/ferreyra')
    tiles=['20HLK','20HLJ','20HKJ','20HMH']; months=['202112','202201','202202','202203','202204','202205']
    source=Path('/mnt/yacy_1/prod/ferreyra/MNC_INTA/2021_2022/MNC_verano2022.tif')
    legend=json.loads((ROOT/'config/leyenda_mnc_2021_2022.json').read_text())
    if post.sha256(out/'modelo_congelado.pth')!=json.loads((out/'protocolo.json').read_text())['checkpoint_sha256']:
        raise ValueError('El checkpoint congelado cambio')
    if hashlib.md5(source.read_bytes()).hexdigest()!='f88ffac06865b36038916ec22c1ffe28':
        raise ValueError('Checksum MNC2022 incorrecto')
    remap=np.array(legend['remap'],dtype=np.uint8); valid_codes=set(legend['codigos_documentados'])|{255}
    comps={tile:[base/'composites_2021_2022_scl'/f'{tile}_{month}_median.tif' for month in months] for tile in tiles}
    failures=auditar_rasters([path for files in comps.values() for path in files],out/'integridad_composites.json')
    if failures: raise ValueError(f'Composites ilegibles: {failures}')
    post.LEGENDS['2021_2022']=(remap,'etiqueta_mnc2022_{}_10m.tif')
    counts={}
    for tile in tiles:
        with rasterio.open(comps[tile][0]) as reference,rasterio.open(source) as src:
            if src.count!=1 or src.nodata!=255 or src.dtypes!=('uint8',):
                raise ValueError('Metadatos MNC2022 inesperados')
            data=np.full(reference.shape,255,dtype=np.uint8)
            reproject(source=rasterio.band(src,1),destination=data,src_transform=src.transform,src_crs=src.crs,
                      src_nodata=255,dst_transform=reference.transform,dst_crs=reference.crs,dst_nodata=255,
                      resampling=Resampling.nearest,num_threads=4)
            codes,values=np.unique(data,return_counts=True)
            unknown=set(map(int,codes))-valid_codes
            if unknown: raise ValueError(f'{tile}: codigos no documentados {unknown}; no remapear silenciosamente')
            counts[tile]={str(int(c)):int(n) for c,n in zip(codes,values)}
            if not np.any(remap[data]>0): raise ValueError(f'{tile}: mascara sin etiquetas')
            path=ROOT/'mascaras_procesadas'/f'etiqueta_mnc2022_{tile}_10m.tif'
            profile=reference.profile.copy(); profile.update(count=1,dtype='uint8',nodata=255,compress='deflate',predictor=1)
            temp=path.with_suffix('.partial.tif')
            with rasterio.open(temp,'w',**profile) as dst: dst.write(data,1)
            temp.replace(path)
    post.guardar(out/'labels_verificados.json',{'fuente':str(source),'md5':'f88ffac06865b36038916ec22c1ffe28','conteos_por_tile':counts,'leyenda':legend})
    models={'multicampana':{'nombre':'U-Net multicampana (2018-2021)','clase':post.SimpleUNet,'exp':args.exp,'pesos':out/'modelo_congelado.pth'},
            'unet_2019_2020':{'nombre':'U-Net 2019/2020','clase':post.SimpleUNet,'exp':base/'dataset/exp_cordoba_dicmay_unet_f16'},
            'fcr_2019_2020':{'nombre':'FCRTransUNet 2019/2020','clase':post.FCRTransUNet,'exp':base/'dataset/exp_cordoba_dicmay_fcrtransunet_f16'}}
    (out/'figuras').mkdir(exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('Esperar GPU disponible; no inferir en CPU silenciosamente')
    torch.set_num_threads(4); records=[]; total=len(models)*len(tiles)
    for name,record in models.items():
        weights=record.get('pesos',record['exp']/'best_model.pth')
        model=record['clase'](24,5).to(device); model.load_state_dict(torch.load(weights,map_location=device)); model.eval()
        for tile in tiles:
            post.guardar(out/'progreso.json',{'estado':'evaluacion_temporal','tarea':len(records)+1,'total':total,'modelo':name,'tile':tile})
            destination=out/'predicciones'/name/'2021_2022'/f'prediccion_{tile}.tif'
            inferir_tile(model,comps[tile],destination,device,tags={'campana':'2021_2022','modelo':name,'checkpoint_sha256':post.sha256(weights)})
            cm=confusion_mnc(destination,ROOT/'mascaras_procesadas'/f'etiqueta_mnc2022_{tile}_10m.tif',remap,'2021_2022')
            role='test' if tile in post.SPLIT['test'] else 'val'
            metric=metricas_confusion(cm); records.append({'modelo':name,'campana':'2021_2022','tile':tile,'rol_espacial':role,**metric})
            post.guardar(out/'metricas_tiles.json',records)
            post.figura_tile('2021_2022',tile,{'archivos':comps},models,out)
            print(f'{name} {tile}: accuracy={metric["accuracy"]:.4f} F1_macro={metric["f1_macro"]:.4f}',flush=True)
        del model; torch.cuda.empty_cache()
    groups=[]
    for name in models:
        for role in ['test','val','todos']:
            selected=[r for r in records if r['modelo']==name and (role=='todos' or r['rol_espacial']==role)]
            cm=sum((np.array(r['confusion'],dtype=np.int64) for r in selected),np.zeros((5,5),dtype=np.int64))
            groups.append({'modelo':name,'grupo_espacial':role,**metricas_confusion(cm)})
            matriz_fig(cm,f'Anio nunca visto: 2021/2022 | {models[name]["nombre"]} | {role}',out/'figuras'/f'confusion_{name}_{role}.png')
    post.guardar(out/'evaluacion_temporal.json',{'campana':'2021_2022','modelo_congelado':True,'sin_ajustes':True,'grupos':groups,'tiles':records})
    fig,axes=plt.subplots(1,2,figsize=(12,5)); labels=[m['nombre'] for m in models.values()]
    for ax,metric in zip(axes,['accuracy','f1_macro']):
        ax.bar(np.arange(len(models)),[next(r[metric] for r in groups if r['modelo']==name and r['grupo_espacial']=='test') for name in models])
        ax.set_xticks(np.arange(len(models)),labels,rotation=15,ha='right'); ax.set_ylim(0,1); ax.set_title(metric); ax.grid(axis='y',alpha=.2)
    fig.suptitle('2021/2022 nunca usado en entrenamiento: mismos tiles TEST y referencia MNC')
    fig.tight_layout(); fig.savefig(out/'figuras/comparacion_ano_no_visto.png',dpi=160); plt.close(fig)
    (out/'README.md').write_text('Evaluacion temporal 2021/2022. Checkpoint multicampana congelado de epoca 60.\nNo se entreno, ajusto ni selecciono el checkpoint con 2021/2022.\nMismos cuatro tiles espaciales VAL/TEST, ventana dic-may y preprocesamiento SCL.\nMetricas sobre mapas completos: MNC>0; pred=0 cuenta como error.\nLa referencia es un mapa clasificado INTA, no muestras de campo independientes.\n')
    post.guardar(out/'progreso.json',{'estado':'finalizado','tareas':total})
    print('EVALUACION ANIO NUNCA VISTO FINALIZADA',flush=True)


def main(argv=None):
    args = crear_parser().parse_args(argv)
    return ejecutar_evaluacion(args)


if __name__ == '__main__':
    main()
