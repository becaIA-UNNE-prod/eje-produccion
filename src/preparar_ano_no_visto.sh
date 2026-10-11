#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/s4a-venv/bin/python}
Y3=/mnt/yacy_3/prod_bis/ferreyra
OUT=${1:?Directorio de evaluacion requerido}
EXP=${2:?Experimento congelado requerido}
S2=$Y3/sentinel2_cordoba_2021_2022
COMP=$Y3/composites_2021_2022_scl
TILES="20HLK 20HLJ 20HKJ 20HMH"
MESES="202112 202201 202202 202203 202204 202205"
exec 8>"$OUT/preparacion.lock"
flock -n 8 || { echo 'Ya hay una preparacion de este anio activa'; exit 1; }
printf '%s\n' "$$" > "$OUT/preparacion.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/preparacion.exitcode"; date -Is > "$OUT/fin.txt"' EXIT
mkdir -p "$S2" "$COMP"
paso() { echo "$(date -Is) $*"; }
paso 'Busqueda STAC diciembre 2021 - mayo 2022'
if [ ! -f "$S2/productos.json" ]; then
  "$PY" -u -m src.datos.descargar_stac buscar --out "$S2" --inicio 2021-12-01 --fin 2022-05-31 --tiles-file "$OUT/tiles.txt" --max-nodata 100
fi
paso 'Descarga SCL'
"$PY" -u -m src.datos.descargar_stac descargar --out "$S2" --bandas SCL
paso 'Seleccion de escenas despejadas'
"$PY" -u -m src.datos.composites_scl seleccionar --base "$S2" --tiles $TILES --meses $MESES --workers 4
paso 'Descarga bandas seleccionadas'
"$PY" -u -m src.datos.descargar_stac descargar --out "$S2" --seleccion "$S2/seleccion_scl.json"
"$PY" -u -m src.datos.descargar_stac verificar --out "$S2" --seleccion "$S2/seleccion_scl.json"
paso 'Composites mensuales SCL'
"$PY" -u -m src.datos.composites_scl componer --base "$S2" --out "$COMP" --tiles $TILES --workers 2
paso 'Esperando fuentes MNC verificadas y leyenda oficial'
for i in $(seq 1 240); do
  if [ -f /mnt/yacy_1/prod/ferreyra/MNC_INTA/2021_2022/MNC_verano2022.tif ] && [ -f config/leyenda_mnc_2021_2022.json ]; then break; fi
  sleep 30
done
test -f /mnt/yacy_1/prod/ferreyra/MNC_INTA/2021_2022/MNC_verano2022.tif
test -f config/leyenda_mnc_2021_2022.json
paso 'Esperando turno de GPU para inferencia'
exec 9>"$Y3/logs_multicampana/inferencia.lock"
flock 9
paso 'Inferencia y evaluacion temporal: ningun reentrenamiento'
"$PY" -u -m src.historico.evaluar_ano_no_visto --exp "$EXP" --salida "$OUT"
