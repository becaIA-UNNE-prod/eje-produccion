#!/usr/bin/env bash
# Una corrida U-Net multicampana. No lanza otros modelos.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/s4a-venv/bin/python}
Y3=/mnt/yacy_3/prod_bis/ferreyra
EXP=${1:?Uso: bash src/entrenar_multicampana_unet.sh DIRECTORIO_EXPERIMENTO}
exec 9>"$Y3/logs_multicampana/entrenamiento.lock"
flock -n 9 || { echo 'Ya existe un entrenamiento multicampana activo'; exit 1; }
printf '%s\n' "$$" > "$EXP/train.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$EXP/train.exitcode"; date -Is > "$EXP/fin.txt"' EXIT
date -Is > "$EXP/inicio.txt"
echo "$(date -Is) Iniciando solo U-Net multicampana: $EXP"
"$PY" -u src/1_train_cordoba.py --modelo unet \
  --campana 2018_2019 "$Y3/dataset/train_cordoba_dicmay_1819_scl_f16" \
  --campana 2019_2020 "$Y3/dataset/train_cordoba_dicmay_f16" \
  --campana 2020_2021 "$Y3/dataset/train_cordoba_dicmay_2021_f16" \
  --dir-exp "$EXP"
echo "$(date -Is) Entrenamiento y evaluacion finalizados"
