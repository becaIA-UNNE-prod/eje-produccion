#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=${1:?Directorio de experimento requerido}
PID=${2:?PID del entrenamiento actual requerido}
shift 2
PY=${PY:-$HOME/s4a-venv/bin/python}
exec 9>"$EXP/optimizacion_lectura.lock"
flock -n 9 || { echo 'Ya existe una optimizacion activa'; exit 1; }
printf '%s\n' "$$" > "$EXP/optimizacion_lectura.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$EXP/optimizacion_lectura.exitcode"' EXIT
"$PY" -u src/optimizar_lectura_multicampana.py --dir-exp "$EXP" --pid-train "$PID" "$@"
