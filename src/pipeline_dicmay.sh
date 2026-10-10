#!/bin/bash
# Pipeline completo ventana dic-may: reentrena sobre 2019-2020 (dic 2019 -
# may 2020, etiquetas MNC verano 2020) e infiere sobre la campana 2018-2019
# (dic 2018 - may 2019, sentinel2_cordoba_2018_2019, sin etiquetas).
#
# Los modelos vigentes usan oct-abr y la descarga 2018-2019 es dic-may: como
# la posicion del canal es la unica senal de mes, hay que entrenar con la
# misma ventana que se va a inferir.
#
# Cada paso saltea lo que ya existe, se puede relanzar si se corta.
# Uso (desde la raiz del repo):
#   nohup bash src/pipeline_dicmay.sh > /mnt/yacy_3/prod_bis/ferreyra/logs_dicmay/pipeline.log 2>&1 &
# MODELOS="unet" bash src/pipeline_dicmay.sh   -> solo un modelo
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/s4a-venv/bin/python}
Y3=/mnt/yacy_3/prod_bis/ferreyra
S2_1819=$Y3/sentinel2_cordoba_2018_2019
S2_MAY20=$Y3/sentinel2_cordoba_2020_05
COMP_1819=$Y3/composites_2018_2019
COMP_DICMAY=$Y3/composites_dicmay_2019_2020
COMP_1920=/mnt/yacy_1/prod/ferreyra/cordoba_dataset_filtrado/composites_2019_2020
DIR_TRAIN=$Y3/dataset/train_cordoba_dicmay_f16
DIR_PRED=$Y3/predicciones_2018_2019
LOGS=$Y3/logs_dicmay
MODELOS=${MODELOS:-"unet fcrtransunet"}

MESES_1819="201812 201901 201902 201903 201904 201905"
MESES_1920="201912 202001 202002 202003 202004 202005"
TILES=$(tr '\n' ' ' < $Y3/tiles.txt)
mkdir -p $LOGS

paso() { echo; echo "===== $(date '+%F %T') $* ====="; }

# 1) Composites 2018-2019 (CPU/disco) en paralelo con la descarga de may-2020 (red)
paso "composites 2018-2019 (log: $LOGS/composites_2018_2019.log)"
$PY -m src.datos.composites_cog --base $S2_1819 --out $COMP_1819 --meses $MESES_1819 \
    > $LOGS/composites_2018_2019.log 2>&1 &
PID_COMP=$!

paso "descarga may-2020"
if [ ! -f $S2_MAY20/productos.json ]; then
    $PY -m src.datos.descargar_stac buscar --inicio 2020-05-01 --fin 2020-05-31 \
        --out $S2_MAY20 --tiles-file $Y3/tiles.txt
fi
$PY -m src.datos.descargar_stac descargar --out $S2_MAY20 > $LOGS/descarga_2020_05.log 2>&1 \
    || $PY -m src.datos.descargar_stac descargar --out $S2_MAY20 >> $LOGS/descarga_2020_05.log 2>&1
$PY -m src.datos.descargar_stac verificar --out $S2_MAY20

paso "composites may-2020 + links dic-abr 2019-2020"
mkdir -p $COMP_DICMAY
for t in $TILES; do
    for m in 201912 202001 202002 202003 202004; do
        ln -sfn $COMP_1920/${t}_${m}_median.tif $COMP_DICMAY/${t}_${m}_median.tif
    done
done
$PY -m src.datos.composites_cog --base $S2_MAY20 --out $COMP_DICMAY --meses 202005

wait $PID_COMP
tail -3 $LOGS/composites_2018_2019.log

# 2) Dataset de entrenamiento dic-may (mismos 17 tiles / mascaras MNC que train_cordoba_f16)
paso "dataset dic-may"
$PY -m src.datos.generar_dataset_cordoba_f16 --tiles $TILES --meses $MESES_1920 \
    --dir-composites $COMP_DICMAY --dir-salida $DIR_TRAIN

# 3) Entrenamiento + 4) inferencia, por modelo
for M in $MODELOS; do
    EXP=$Y3/dataset/exp_cordoba_dicmay_${M}_f16
    if [ ! -f $EXP/modelo_final.pth ]; then
        paso "entrenamiento $M (log: $EXP/train.log)"
        mkdir -p $EXP
        $PY -m src.entrenamiento.entrenar_cordoba --modelo $M --sin-2017 \
            --dir-train $DIR_TRAIN --dir-exp $EXP > $EXP/train.log 2>&1
        tail -4 $EXP/train.log
    fi
    paso "inferencia 2018-2019 con $M"
    for t in $TILES; do
        SUF=$([ $M = unet ] && echo "" || echo "_$M")
        [ -f $DIR_PRED/prediccion_${t}${SUF}.tif ] && { echo "$t: ya existe"; continue; }
        $PY src/2_inferencia.py $t --modelo $M --dir-composites $COMP_1819 \
            --meses $MESES_1819 --dir-exp $EXP --dir-salida $DIR_PRED \
            || { rm -f $DIR_PRED/prediccion_${t}${SUF}.tif; exit 1; }
    done
done

paso "LISTO -- predicciones en $DIR_PRED"
