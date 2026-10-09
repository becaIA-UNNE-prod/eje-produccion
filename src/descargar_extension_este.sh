#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/s4a-venv/bin/python}
BASE=/mnt/yacy_1/prod/ferreyra/extension_este_multicampana
RUN=${1:?Directorio de logs requerido}
mkdir -p "$BASE" "$RUN"
exec 8>"$BASE/descarga.lock"
flock -n 8 || { echo 'Ya hay una descarga de extension este activa'; exit 1; }
echo "$$" > "$RUN/descarga.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/descarga.exitcode"; date -Is > "$RUN/fin.txt"' EXIT
date -Is > "$RUN/inicio.txt"
printf '%s\n' 20HNJ 20HNH > "$BASE/tiles.txt"
fallos=0
campana() {
 local inicio=$1 fin=$2 anio=$3 siguiente=$4
 local source="$BASE/sentinel2_${anio}_${siguiente}"
 mkdir -p "$source"
 echo "$(date -Is) CAMPANA ${anio}_${siguiente} INICIO"
 "$PY" -u - "$BASE" <<'PY'
import shutil,sys
free=shutil.disk_usage(sys.argv[1]).free
if free<100*1024**3:raise SystemExit('Espacio insuficiente: se conserva reserva de 100 GiB')
PY
 if [ ! -f "$source/productos.json" ]; then
  "$PY" -u src/00_descargar_stac.py buscar --out "$source" --inicio "$inicio" --fin "$fin" --tiles-file "$BASE/tiles.txt" --max-nodata 100
 fi
 "$PY" -u src/00_descargar_stac.py descargar --out "$source" --bandas SCL
 "$PY" -u src/00_descargar_stac.py verificar --out "$source" --bandas SCL
 "$PY" -u src/02d_composites_scl.py seleccionar --base "$source" --tiles 20HNJ 20HNH --meses "${anio}12" "${siguiente}01" "${siguiente}02" "${siguiente}03" "${siguiente}04" "${siguiente}05" --workers 4
 "$PY" -u - "$source" "$anio" "$siguiente" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1]);a,b=sys.argv[2:];sel=json.loads((base/'seleccion_scl.json').read_text())
expected={f'{t}_{m}' for t in ['20HNJ','20HNH'] for m in [a+'12']+[b+f'{i:02}' for i in range(1,6)]}
if set(sel)!=expected or any(not v['fechas'] for v in sel.values()):raise SystemExit('Seleccion mensual incompleta')
print('SELECCION VERIFICADA: 12 tile-mes')
PY
 "$PY" -u src/00_descargar_stac.py descargar --out "$source" --seleccion "$source/seleccion_scl.json"
 "$PY" -u src/00_descargar_stac.py verificar --out "$source" --seleccion "$source/seleccion_scl.json"
 "$PY" -u - "$source" <<'PY'
import json,sys,datetime
from pathlib import Path
base=Path(sys.argv[1]);sel=json.loads((base/'seleccion_scl.json').read_text());files=list(base.glob('*/*/*.tif'))
report={'fecha':datetime.datetime.now().isoformat(),'tile_meses':len(sel),'escenas_seleccionadas':sum(len(v['fechas']) for v in sel.values()),'archivos_tif':len(files),'bytes':sum(f.stat().st_size for f in files),'verificacion':'apertura y lectura de ventana 64x64 de cada archivo; no auditoria exhaustiva de todos los bloques','sin_cambios_split':True}
(base/'descarga_verificada.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
PY
 echo "$(date -Is) CAMPANA ${anio}_${siguiente} COMPLETA"
}
for anio in 2018 2019 2020 2021; do
 siguiente=$((anio + 1))
 # Un subshell mantiene errexit dentro de cada campaña y permite continuar las restantes.
 set +e
 (set -e; campana "${anio}-12-01" "${siguiente}-05-31" "$anio" "$siguiente") > "$RUN/${anio}_${siguiente}.log" 2>&1
 rc=$?
 set -e
 echo "$rc" > "$RUN/${anio}_${siguiente}.exitcode"
 if [ "$rc" -ne 0 ]; then echo "Campana ${anio}_${siguiente}: ERROR $rc; ver log"; fallos=$((fallos + 1)); else echo "Campana ${anio}_${siguiente}: OK"; fi
done
if [ "$fallos" -ne 0 ]; then echo "DESCARGA TERMINADA CON $fallos CAMPANAS CON ERROR"; exit 1; fi
echo 'DESCARGA EXTENSION ESTE FINALIZADA: cuatro campanas, 48 tile-mes'
