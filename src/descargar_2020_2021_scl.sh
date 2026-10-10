#!/bin/bash
# 2020-2021 dic-may con el mismo flujo SCL que rehacer_2018_2019_scl.sh:
# busqueda (max-nodata 100) -> SCL -> seleccion -> bandas elegidas -> composites 02d.
# Cada paso saltea lo que ya existe.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/s4a-venv/bin/python}
Y3=/mnt/yacy_3/prod_bis/ferreyra
S2=$Y3/sentinel2_cordoba_2020_2021
COMP=$Y3/composites_2020_2021_scl
LOGS=$Y3/logs_2020_2021
MESES="202012 202101 202102 202103 202104 202105"
TILES=$(tr '\n' ' ' < $Y3/tiles.txt)
mkdir -p $LOGS $S2

paso() { echo; echo "===== $(date '+%F %T') $* ====="; }

paso "busqueda STAC dic-2020 a may-2021"
if [ -f $S2/productos.json ]; then
    echo "productos.json ya existe, no se busca de nuevo"
else
    $PY -m src.datos.descargar_stac buscar --out $S2 --inicio 2020-12-01 --fin 2021-05-31 \
        --tiles-file $Y3/tiles.txt --max-nodata 100 > $LOGS/busqueda.log 2>&1
    tail -3 $LOGS/busqueda.log
fi

paso "SCL de todos los candidatos 2020-2021"
$PY -m src.datos.descargar_stac descargar --out $S2 --bandas SCL > $LOGS/descarga_scl.log 2>&1 \
    || $PY -m src.datos.descargar_stac descargar --out $S2 --bandas SCL >> $LOGS/descarga_scl.log 2>&1
tail -1 $LOGS/descarga_scl.log

paso "seleccion de escenas por SCL"
$PY -m src.datos.composites_scl seleccionar --base $S2 --tiles $TILES --meses $MESES \
    > $LOGS/seleccion.log 2>&1
tail -2 $LOGS/seleccion.log

paso "bandas de las escenas elegidas"
$PY -m src.datos.descargar_stac descargar --out $S2 --seleccion $S2/seleccion_scl.json \
    > $LOGS/descarga_bandas.log 2>&1 \
    || $PY -m src.datos.descargar_stac descargar --out $S2 --seleccion $S2/seleccion_scl.json \
    >> $LOGS/descarga_bandas.log 2>&1
$PY -m src.datos.descargar_stac verificar --out $S2 --seleccion $S2/seleccion_scl.json | head -3

paso "composites 2020-2021 con SCL"
$PY -m src.datos.composites_scl componer --base $S2 --out $COMP --tiles $TILES \
    > $LOGS/composites.log 2>&1
tail -1 $LOGS/composites.log
grep "ATENCION" $LOGS/composites.log || echo "todos los composites >= 95% despejado"

paso "LISTO -- 2020-2021 descargado y compuesto"
