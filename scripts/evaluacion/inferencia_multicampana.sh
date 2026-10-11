#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/s4a-venv/bin/python}
source scripts/lib/corridas.sh
EXP=${1:?Experimento requerido}
OUT=${2:?Directorio de salida requerido}
cordoba_run_init evaluacion inferencia_multicampana "$@"
cordoba_registrar --salida "experimento=$EXP"
cordoba_registrar --salida "resultados=$OUT"
mkdir -p /mnt/yacy_3/prod_bis/ferreyra/logs_multicampana
exec 9>/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/inferencia.lock
flock -n 9 || { echo 'Ya existe una inferencia multicampana activa'; exit 1; }
cordoba_lock inferencia
printf '%s\n' "$$" > "$OUT/inferencia.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/inferencia.exitcode"; date -Is > "$OUT/fin.txt"' EXIT
date -Is > "$OUT/inicio.txt"
"$PY" -u -m src.evaluacion.postprocesar_multicampana --exp "$EXP" --salida "$OUT" --lock-adquirido
