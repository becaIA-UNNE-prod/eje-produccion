#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=${1:?Experimento requerido}
OUT=${2:?Directorio de salida requerido}
PY=${PY:-$HOME/s4a-venv/bin/python}
exec 9>/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/inferencia.lock
flock -n 9 || { echo 'Ya existe una inferencia multicampana activa'; exit 1; }
printf '%s\n' "$$" > "$OUT/inferencia.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/inferencia.exitcode"; date -Is > "$OUT/fin.txt"' EXIT
date -Is > "$OUT/inicio.txt"
"$PY" -u src/postprocesar_multicampana.py --exp "$EXP" --salida "$OUT"
