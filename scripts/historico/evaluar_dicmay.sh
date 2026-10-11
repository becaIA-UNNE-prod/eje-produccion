#!/bin/bash
# Evaluacion de los modelos dic-may (pipeline_dicmay.sh) con el tile completo:
#  1) referencia mismo anio: inferencia sobre 2019-2020 dic-may en los tiles
#     de val/test, contra MNC verano 2020
#  2) generalizacion interanual: predicciones 2018-2019 (17 tiles) contra
#     MNC 2018/2019 (01b_mascaras_mnc_2018_2019.py)
# La diferencia entre (1) y (2) en val/test es la caida por cambiar de anio.
#
# Con ESPERAR_PID=<pid> espera a que termine ese proceso (pipeline_dicmay.sh)
# y solo sigue si el pipeline termino bien.
# Uso (desde la raiz del repo):
#   ESPERAR_PID=5421 nohup bash scripts/historico/evaluar_dicmay.sh > /dev/null 2>&1 &  # logs automáticos por corrida
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/s4a-venv/bin/python}
source scripts/lib/corridas.sh
cordoba_run_init historico evaluar_dicmay "$@"

Y3=/mnt/yacy_3/prod_bis/ferreyra
LOGS="$CORDOBA_LOG_DIR"
COMP_DICMAY=$Y3/composites_dicmay_2019_2020
cordoba_registrar --salida "composites_1920=$COMP_DICMAY"
MODELOS=${MODELOS:-"unet fcrtransunet"}
MESES_1920="201912 202001 202002 202003 202004 202005"
TILES_REF="20HLJ 20HLK 20HKJ 20HMH"   # test + val

if [ -n "${ESPERAR_PID:-}" ]; then
    echo "$(date '+%F %T') esperando a que termine el PID $ESPERAR_PID..."
    while kill -0 $ESPERAR_PID 2>/dev/null; do sleep 300; done
    grep -q "LISTO" "${PIPELINE_LOG:-$Y3/logs_dicmay/pipeline.log}" || { echo "pipeline_dicmay.sh no termino bien, ver "$LOGS/pipeline.log""; exit 1; }
fi

echo; echo "===== $(date '+%F %T') inferencia de referencia 2019-2020 dic-may ====="
for M in $MODELOS; do
    SUF=$([ $M = unet ] && echo "" || echo "_$M")
    for t in $TILES_REF; do
        OUT=$Y3/predicciones_2019_2020_dicmay/prediccion_${t}${SUF}.tif
        [ -f $OUT ] && { echo "$t $M: ya existe"; continue; }
        $PY -m src.inferencia.inferir_tile $t --modelo $M --dir-composites $COMP_DICMAY --meses $MESES_1920 \
            --dir-exp $Y3/dataset/exp_cordoba_dicmay_${M}_f16 --dir-salida $Y3/predicciones_2019_2020_dicmay \
            || { rm -f $OUT; exit 1; }
    done
done

echo; echo "===== $(date '+%F %T') evaluacion mismo anio (2019-2020 vs MNC 2020) ====="
$PY -m src.historico.evaluar_contra_mnc --leyenda 2020 --modelos $MODELOS --tiles $TILES_REF \
    --dir-pred $Y3/predicciones_2019_2020_dicmay --dir-salida $Y3/evaluacion_2019_2020_dicmay

echo; echo "===== $(date '+%F %T') evaluacion interanual (2018-2019 vs MNC 2018/2019) ====="
$PY -m src.historico.evaluar_contra_mnc --leyenda 1819 --modelos $MODELOS \
    --dir-pred $Y3/predicciones_2018_2019 --dir-salida $Y3/evaluacion_2018_2019

echo; echo "===== $(date '+%F %T') LISTO ====="
