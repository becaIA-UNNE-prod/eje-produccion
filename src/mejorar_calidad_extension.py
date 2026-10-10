#!/usr/bin/env python3
"""Ampliar escenas de dos meses y promover compuestos solo si no empeoran."""
import json,sys,subprocess,shutil,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.integridad_raster import auditar_rasters
BASE=Path('/mnt/yacy_1/prod/ferreyra/extension_este_multicampana')
out=Path(sys.argv[1]);reports=[]
def run(*args):subprocess.run([sys.executable,'-u',*map(str,args)],cwd=ROOT,check=True)
for year,key in [('2018_2019','20HNJ_201903'),('2020_2021','20HNH_202103')]:
 source=BASE/('sentinel2_'+year);comp=BASE/('composites_'+year+'_scl');tile,month=key.split('_');old=json.loads((comp/'fechas_usadas.json').read_text())[key]
 products=json.loads((source/'productos.json').read_text());dates=sorted({p['date'] for p in products if p['tile']==tile and p['date'].startswith(month) and (source/tile/p['date']/'SCL.tif').exists()})
 original=json.loads((source/'seleccion_scl.json').read_text());entry={**original[key],'fechas':dates,'nota':'Ampliacion por cobertura real inferior a prevista; todas las candidatas disponibles del mes'}
 selection=out/f'seleccion_mejora_{year}.json';selection.write_text(json.dumps({key:entry},indent=2)+'\n')
 print(f'MEJORA {key}: {len(original[key]["fechas"])} -> {len(dates)} escenas',flush=True)
 run('src/00_descargar_stac.py','descargar','--out',source,'--seleccion',selection)
 run('src/00_descargar_stac.py','verificar','--out',source,'--seleccion',selection)
 paths=[source/tile/date/(band+'.tif') for date in dates for band in ['B02','B03','B04','B08','SCL']]
 failures=auditar_rasters(paths,out/f'integridad_mejora_{year}.json',workers=2)
 if failures:raise ValueError(str(failures))
 tempdir=out/('compuestos_mejorados_'+year)
 run('src/02d_composites_scl.py','componer','--base',source,'--out',tempdir,'--seleccion',selection,'--tiles',tile,'--workers','1')
 improved=tempdir/(key+'_median.tif');failures=auditar_rasters([improved],out/f'integridad_compuesto_mejora_{year}.json',workers=1)
 if failures:raise ValueError(str(failures))
 new=json.loads((tempdir/'fechas_usadas.json').read_text())[key]
 if new['despejado']<old['despejado'] or new['cobertura']<old['cobertura']:raise ValueError(f'La mejora empeora cobertura: {old} -> {new}; original preservado')
 destination=comp/improved.name;backup=out/(key+'_antes_mejora.tif')
 if not backup.exists():shutil.copy2(destination,backup)
 shutil.copy2(improved,destination.with_suffix('.partial.tif'));destination.with_suffix('.partial.tif').replace(destination)
 regpath=comp/'fechas_usadas.json';reg=json.loads(regpath.read_text());reg[key]=new;tmp=regpath.with_suffix('.json.tmp');tmp.write_text(json.dumps(reg,indent=2)+'\n');tmp.replace(regpath)
 selpath=source/'seleccion_scl.json';backupsel=out/f'seleccion_antes_mejora_{year}.json'
 if not backupsel.exists():shutil.copy2(selpath,backupsel)
 original[key]=entry;tmpsel=selpath.with_suffix('.json.tmp');tmpsel.write_text(json.dumps(original,indent=2)+'\n');tmpsel.replace(selpath)
 reports.append({'key':key,'antes':old,'despues':new});(out/'mejoras_calidad.json').write_text(json.dumps(reports,indent=2)+'\n');print(f'MEJORA VERIFICADA {key}: {old} -> {new}',flush=True)
