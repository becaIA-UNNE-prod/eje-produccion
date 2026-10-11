"""Audita composites y prepara máscaras completas/independientes de extensión."""
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, transform_geom, Resampling
from rasterio.features import rasterize
from src.soporte.archivos import guardar
from src.datos.campanas import files, fuentes_mnc
from utils.rutas import ROOT, EXTENSION_ROOT as BASE, MULTICAMPANA_ROOT as Y3, MNC_ROOT as MNC
from utils.integridad_raster import auditar_rasters, firma
from utils.multicampana import SPLIT
from src.datos.leyendas import legend, check_codes

# Alcance original de extensión; no redefine el split ni el evaluador canónico.
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



def crear_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--salida', type=Path, required=True)
    return parser


def main(argv=None):
    args = crear_parser().parse_args(argv)
    args.salida.mkdir(parents=True, exist_ok=True)
    prepare(args.salida)


if __name__ == '__main__':
    main()
