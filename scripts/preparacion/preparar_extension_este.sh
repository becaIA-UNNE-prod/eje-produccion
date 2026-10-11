#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/s4a-venv/bin/python}
source scripts/lib/corridas.sh
OUT=${1:?Directorio de resultados requerido}
cordoba_run_init preparacion preparar_extension_este "$@"
BASE=/mnt/yacy_1/prod/ferreyra/extension_este_multicampana
cordoba_registrar --salida "base_operativa=$BASE"
cordoba_registrar --salida "resultados=$OUT"
mkdir -p "$OUT"
exec 8>"$BASE/preparacion.lock"
flock -n 8 || { echo 'Ya hay una preparacion activa'; exit 1; }
cordoba_lock extension_preparacion
echo "$$" > "$OUT/preparacion.pid"
trap 'rc=$?; echo "$rc" > "$OUT/preparacion.exitcode"; date -Is > "$OUT/fin.txt"' EXIT
date -Is > "$OUT/inicio.txt"
echo "$(date -Is) AUDITORIA COMPLETA DE BANDAS DESCARGADAS"
"$PY" -u - "$OUT" <<'PY'
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from utils.integridad_raster import auditar_rasters
base=Path('/mnt/yacy_1/prod/ferreyra/extension_este_multicampana');out=Path(sys.argv[1]);paths=[]
if shutil.disk_usage(base).free < 100*1024**3:raise SystemExit('Reserva de disco inferior a 100 GiB')
for year in ['2018_2019','2019_2020','2020_2021','2021_2022']:
 source=base/('sentinel2_'+year);sel=json.loads((source/'seleccion_scl.json').read_text())
 expected={f'{t}_{m}' for t in ['20HNJ','20HNH'] for m in [year[:4]+'12']+[year[5:]+f'{i:02}' for i in range(1,6)]}
 if set(sel)!=expected:raise SystemExit('Seleccion mensual incompleta')
 for key,record in sel.items():
  for date in record['fechas']:
   for band in ['B02','B03','B04','B08','SCL']:paths.append(source/key[:5]/date/(band+'.tif'))
failures=auditar_rasters(paths,out/'integridad_entradas.json',workers=2)
if failures:raise SystemExit('Rasters descargados corruptos: '+str(failures))
PY
for anio in 2018 2019 2020 2021; do
 siguiente=$((anio+1));year="${anio}_${siguiente}"
 echo "$(date -Is) COMPUESTOS $year"
 "$PY" -u -m src.datos.composites_scl componer --base "$BASE/sentinel2_$year" --out "$BASE/composites_${year}_scl" --tiles 20HNJ 20HNH --workers 2
done
echo "$(date -Is) AUDITORIA COMPUESTOS Y PREPARACION MASCARAS"
"$PY" -u -m src.datos.preparar_extension --salida "$OUT"
echo "$(date -Is) ESPERANDO TURNO DE GPU"
mkdir -p /mnt/yacy_3/prod_bis/ferreyra/logs_multicampana
exec 9>/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/inferencia.lock
flock 9
cordoba_lock inferencia esperar
echo "$(date -Is) EVALUACION DE U-NET CONGELADA"
"$PY" -u -m src.evaluacion.evaluar_extension --salida "$OUT"
echo 'PREPARACION Y EVALUACION EXTENSION ESTE FINALIZADAS'
