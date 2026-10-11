#!/bin/bash
# Pipeline ventana dic-may sumando la campana 2017-2018 al entrenamiento:
# mismo esquema que pipeline_dicmay.sh + evaluar_dicmay.sh, pero TRAIN =
# 2019-2020 (MNC verano 2020) + 2017-2018 (capa Nivel3, dic 2017 - may 2018),
# para comparar contra los modelos entrenados solo con 2019-2020.
#
#  1) descarga dic 2017 - may 2018 (earth-search) y composites con mascara de
#     nubes SCL (02d_composites_scl.py, mismo metodo que composites_2018_2019_scl):
#     busqueda sin limite de % sin dato, SCL de todos los candidatos, seleccion
#     de escenas por pixeles despejados y bandas solo de las elegidas. Solo los
#     12 tiles de TRAIN con etiqueta Nivel3 (19HGC queda fuera de la capa).
#     Requiere haber corrido antes rehacer_2018_2019_scl.sh (la inferencia
#     2018-2019 usa composites_2018_2019_scl).
#  2) mascaras Nivel3 a 10 m (01c) y dataset 2017-2018 de los 13 tiles de
#     TRAIN (03c --leyenda nivel3). Los tiles de val/test de 2017-2018 NO se
#     usan, para que val/test sigan siendo lugares no vistos.
#  3) entrenamiento de unet y fcrtransunet (--dir-train-extra), mismo val/test
#     2019-2020 y mismos pesos por clase que los modelos base
#  4) inferencia 2018-2019 (17 tiles) y 2019-2020 (val/test), evaluacion
#     contra MNC 2018/2019 y MNC 2020 (11) y graficos comparativos (12)
#
# Cada paso saltea lo que ya existe, se puede relanzar si se corta.
# Uso (desde la raiz del repo):
#   nohup bash scripts/historico/pipeline_dicmay_1718.sh > /dev/null 2>&1 &  # logs automáticos por corrida
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/s4a-venv/bin/python}
source scripts/lib/corridas.sh
cordoba_run_init historico pipeline_dicmay_1718 "$@"

Y3=/mnt/yacy_3/prod_bis/ferreyra
S2_1718=$Y3/sentinel2_cordoba_2017_2018_dicmay
cordoba_registrar --salida "fuentes_1718=$S2_1718"
COMP_1718=$Y3/composites_dicmay_2017_2018
cordoba_registrar --salida "composites_1718=$COMP_1718"
COMP_1819=$Y3/composites_2018_2019_scl
cordoba_registrar --salida "composites_1819=$COMP_1819"
COMP_DICMAY=$Y3/composites_dicmay_2019_2020
cordoba_registrar --salida "composites_1920=$COMP_DICMAY"
DIR_TRAIN=$Y3/dataset/train_cordoba_dicmay_f16
cordoba_registrar --salida "dataset=$DIR_TRAIN"
DIR_TRAIN_1718=$Y3/dataset/train_cordoba_dicmay_1718_f16
cordoba_registrar --salida "dataset_1718=$DIR_TRAIN_1718"
PRED_1819=$Y3/predicciones_2018_2019_con1718
cordoba_registrar --salida "predicciones_1819=$PRED_1819"
PRED_1920=$Y3/predicciones_2019_2020_dicmay_con1718
cordoba_registrar --salida "predicciones_1920=$PRED_1920"
LOGS="$CORDOBA_LOG_DIR"
MODELOS=${MODELOS:-"unet fcrtransunet"}

MESES_1718="201712 201801 201802 201803 201804 201805"
MESES_1819="201812 201901 201902 201903 201904 201905"
MESES_1920="201912 202001 202002 202003 202004 202005"
TILES=$(tr '\n' ' ' < $Y3/tiles.txt)
TILES_TRAIN="19HGC 20HLG 20HLH 20HMJ 20HMK 20HNK 20JML 20HKH 20HMG 20HNG 20HPG 20HPH 20JNL"
TILES_REF="20HLJ 20HLK 20HKJ 20HMH"   # test + val
TILES_1718="20HLG 20HLH 20HMJ 20HMK 20HNK 20JML 20HKH 20HMG 20HNG 20HPG 20HPH 20JNL"   # TRAIN sin 19HGC
mkdir -p "$LOGS"
[ -d $COMP_1819 ] || { echo "falta $COMP_1819: correr antes scripts/preparacion/rehacer_2018_2019_scl.sh"; exit 1; }

paso() { echo; echo "===== $(date '+%F %T') $* ====="; }

# 1) Descarga + composites 2017-2018 con SCL
# Si queda corriendo una descarga anterior sobre el mismo directorio, esperar
# (bajarian los mismos archivos a la vez).
while pgrep -f "(00_descargar_stac[.]py|src[.]datos[.]descargar_stac) descargar --out $S2_1718" > /dev/null; do sleep 60; done
paso "busqueda dic 2017 - may 2018 sin limite de % sin dato"
echo "$TILES_1718" | tr ' ' '\n' > "$LOGS/tiles_1718.txt"
if [ ! -f $S2_1718/productos_v2_maxnodata100.json ]; then
    $PY -m src.datos.descargar_stac buscar --inicio 2017-12-01 --fin 2018-05-31 \
        --out $S2_1718 --tiles-file "$LOGS/tiles_1718.txt" --max-nodata 100 | tail -14
    cp $S2_1718/productos.json $S2_1718/productos_v2_maxnodata100.json
fi

paso "SCL de todos los candidatos 2017-2018"
$PY -m src.datos.descargar_stac descargar --out $S2_1718 --bandas SCL > "$LOGS/descarga_scl_2017_2018.log" 2>&1 \
    || $PY -m src.datos.descargar_stac descargar --out $S2_1718 --bandas SCL >> "$LOGS/descarga_scl_2017_2018.log" 2>&1
tail -1 "$LOGS/descarga_scl_2017_2018.log"

paso "seleccion de escenas por SCL"
$PY -m src.datos.composites_scl seleccionar --base $S2_1718 --tiles $TILES_1718 --meses $MESES_1718 \
    > "$LOGS/seleccion_2017_2018.log" 2>&1
tail -2 "$LOGS/seleccion_2017_2018.log"

paso "bandas de las escenas elegidas (log: $LOGS/descarga_2017_2018.log)"
$PY -m src.datos.descargar_stac descargar --out $S2_1718 --seleccion $S2_1718/seleccion_scl.json \
    > "$LOGS/descarga_2017_2018.log" 2>&1 \
    || $PY -m src.datos.descargar_stac descargar --out $S2_1718 --seleccion $S2_1718/seleccion_scl.json \
    >> "$LOGS/descarga_2017_2018.log" 2>&1
$PY -m src.datos.descargar_stac verificar --out $S2_1718 --seleccion $S2_1718/seleccion_scl.json | head -3

paso "composites 2017-2018 con SCL (log: $LOGS/composites_2017_2018.log)"
$PY -m src.datos.composites_scl componer --base $S2_1718 --out $COMP_1718 --tiles $TILES_1718 \
    > "$LOGS/composites_2017_2018.log" 2>&1
tail -1 "$LOGS/composites_2017_2018.log"
grep "ATENCION" "$LOGS/composites_2017_2018.log" || echo "todos los composites >= 95% despejado"

# 2) Mascaras Nivel3 + dataset 2017-2018 (solo tiles de TRAIN)
paso "mascaras Nivel3 2017-2018"
$PY -m src.datos.mascaras_nivel3_2017_2018 --ref $S2_1718 --tiles $TILES_1718 | tee "$LOGS/mascaras_nivel3.log"

paso "dataset dic-may 2017-2018"
$PY -m src.datos.generar_dataset_cordoba_f16 --leyenda nivel3 --tiles $TILES_TRAIN --meses $MESES_1718 \
    --dir-composites $COMP_1718 --dir-salida $DIR_TRAIN_1718 | grep -v "chunk terminado"

# 3) Pesos + entrenamiento
# Mismos pesos por clase que los modelos base (--sin-2017), para que la unica
# diferencia sea la campana sumada. Los recalculados quedan solo de referencia.
paso "pesos por clase (2019-2020 + 2017-2018, solo referencia)"
$PY -m src.entrenamiento.calcular_pesos --dirs $DIR_TRAIN $DIR_TRAIN_1718 | tee "$LOGS/pesos.log"

for M in $MODELOS; do
    EXP=$Y3/dataset/exp_cordoba_dicmay_con1718_${M}_f16
    cordoba_registrar --salida "experimento_${M}=$EXP"
    if [ ! -f $EXP/modelo_final.pth ]; then
        paso "entrenamiento $M (log: ${CORDOBA_LOG_DIR}/train_${M}.log)"
        mkdir -p $EXP
        $PY -m src.entrenamiento.entrenar_cordoba --modelo $M --sin-2017 \
            --dir-train $DIR_TRAIN --dir-train-extra $DIR_TRAIN_1718 --dir-exp $EXP > "${CORDOBA_LOG_DIR}/train_${M}.log" 2>&1
        tail -4 "${CORDOBA_LOG_DIR}/train_${M}.log"
    fi
done

# 4) Inferencias
inferir() {   # $1 modelo  $2 dir composites  $3 meses  $4 dir salida  $5.. tiles
    local M=$1 COMP=$2 MESES=$3 OUT=$4; shift 4
    local SUF=$([ $M = unet ] && echo "" || echo "_$M")
    for t in "$@"; do
        [ -f $OUT/prediccion_${t}${SUF}.tif ] && { echo "$t $M: ya existe"; continue; }
        $PY -m src.inferencia.inferir_tile $t --modelo $M --dir-composites $COMP --meses $MESES \
            --dir-exp $Y3/dataset/exp_cordoba_dicmay_con1718_${M}_f16 --dir-salida $OUT \
            2>&1 | grep -v nested_tensor \
            || { rm -f $OUT/prediccion_${t}${SUF}.tif; exit 1; }
    done
}
for M in $MODELOS; do
    paso "inferencia 2018-2019 con $M"
    inferir $M $COMP_1819 "$MESES_1819" $PRED_1819 $TILES
    paso "inferencia de referencia 2019-2020 con $M"
    inferir $M $COMP_DICMAY "$MESES_1920" $PRED_1920 $TILES_REF
done

# 5) Evaluacion + graficos
paso "evaluacion mismo anio (2019-2020 vs MNC 2020)"
$PY -m src.historico.evaluar_contra_mnc --leyenda 2020 --modelos $MODELOS --tiles $TILES_REF \
    --dir-pred $PRED_1920 --dir-salida $Y3/evaluacion_2019_2020_dicmay_con1718

paso "evaluacion interanual (2018-2019 vs MNC 2018/2019)"
$PY -m src.historico.evaluar_contra_mnc --leyenda 1819 --modelos $MODELOS \
    --dir-pred $PRED_1819 --dir-salida $Y3/evaluacion_2018_2019_con1718

paso "graficos comparativos"
$PY -m src.historico.comparar_con_1718 --comparacion 1718

paso "LISTO -- graficos en $Y3/comparacion_1718"
