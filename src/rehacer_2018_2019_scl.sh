#!/bin/bash
# Rehace los composites 2018-2019 con mascara de nubes SCL (02d_composites_scl.py)
# y vuelve a inferir/evaluar los modelos dic-may base (pipeline_dicmay.sh) sobre
# ellos. Motivo: 02c no enmascara nubes y la busqueda original descartaba las
# escenas con >50% sin dato (bordes de orbita), que en meses nublados eran las
# unicas limpias (ej. 20HNK mar-2019). Ver resultados/README.md.
#
# La busqueda con --max-nodata 100 ya reemplazo productos.json (el original
# quedo en productos_v1_maxnodata50.json). Cada paso saltea lo que ya existe.
# Uso (desde la raiz del repo):
#   nohup bash src/rehacer_2018_2019_scl.sh > /mnt/yacy_3/prod_bis/ferreyra/logs_scl/rehacer_2018_2019.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/s4a-venv/bin/python}
Y3=/mnt/yacy_3/prod_bis/ferreyra
S2_1819=$Y3/sentinel2_cordoba_2018_2019
COMP=$Y3/composites_2018_2019_scl
PRED=$Y3/predicciones_2018_2019_scl
LOGS=$Y3/logs_scl
MODELOS=${MODELOS:-"unet fcrtransunet"}
MESES_1819="201812 201901 201902 201903 201904 201905"
TILES=$(tr '\n' ' ' < $Y3/tiles.txt)
mkdir -p $LOGS

paso() { echo; echo "===== $(date '+%F %T') $* ====="; }

paso "SCL de todos los candidatos 2018-2019"
$PY -m src.datos.descargar_stac descargar --out $S2_1819 --bandas SCL > $LOGS/descarga_scl_2018_2019.log 2>&1 \
    || $PY -m src.datos.descargar_stac descargar --out $S2_1819 --bandas SCL >> $LOGS/descarga_scl_2018_2019.log 2>&1
tail -1 $LOGS/descarga_scl_2018_2019.log

paso "seleccion de escenas por SCL"
$PY -m src.datos.composites_scl seleccionar --base $S2_1819 --tiles $TILES --meses $MESES_1819 \
    > $LOGS/seleccion_2018_2019.log 2>&1
tail -2 $LOGS/seleccion_2018_2019.log

paso "bandas de las escenas elegidas (log: $LOGS/descarga_bandas_2018_2019.log)"
$PY -m src.datos.descargar_stac descargar --out $S2_1819 --seleccion $S2_1819/seleccion_scl.json \
    > $LOGS/descarga_bandas_2018_2019.log 2>&1 \
    || $PY -m src.datos.descargar_stac descargar --out $S2_1819 --seleccion $S2_1819/seleccion_scl.json \
    >> $LOGS/descarga_bandas_2018_2019.log 2>&1
$PY -m src.datos.descargar_stac verificar --out $S2_1819 --seleccion $S2_1819/seleccion_scl.json | head -3

paso "composites 2018-2019 con SCL (log: $LOGS/composites_2018_2019.log)"
$PY -m src.datos.composites_scl componer --base $S2_1819 --out $COMP --tiles $TILES \
    > $LOGS/composites_2018_2019.log 2>&1
tail -1 $LOGS/composites_2018_2019.log
grep "ATENCION" $LOGS/composites_2018_2019.log || echo "todos los composites >= 95% despejado"

for M in $MODELOS; do
    paso "inferencia 2018-2019 (SCL) con modelo base $M"
    SUF=$([ $M = unet ] && echo "" || echo "_$M")
    for t in $TILES; do
        [ -f $PRED/prediccion_${t}${SUF}.tif ] && { echo "$t: ya existe"; continue; }
        $PY src/2_inferencia.py $t --modelo $M --dir-composites $COMP --meses $MESES_1819 \
            --dir-exp $Y3/dataset/exp_cordoba_dicmay_${M}_f16 --dir-salida $PRED 2>&1 | grep -v nested_tensor \
            || { rm -f $PRED/prediccion_${t}${SUF}.tif; exit 1; }
    done
done

paso "evaluacion modelos base sobre composites SCL (2018-2019 vs MNC 2018/2019)"
$PY src/11_evaluar_contra_mnc.py --leyenda 1819 --modelos $MODELOS \
    --dir-pred $PRED --dir-salida $Y3/evaluacion_2018_2019_scl

paso "graficos: composites 02c vs 02d (SCL)"
$PY src/12_comparar_con_1718.py --comparacion scl

paso "LISTO -- base sobre composites SCL"
