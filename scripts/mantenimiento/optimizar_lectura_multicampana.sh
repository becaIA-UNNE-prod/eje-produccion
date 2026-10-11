#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/s4a-venv/bin/python}
source scripts/lib/corridas.sh
EXP=${1:?Directorio de experimento requerido}
PID=${2:?PID del entrenamiento actual requerido}
cordoba_run_init mantenimiento optimizar_lectura_multicampana "$@"
cordoba_registrar --salida "experimento=$EXP"
shift 2
exec 9>"$EXP/optimizacion_lectura.lock"
flock -n 9 || { echo 'Ya existe una optimizacion activa'; exit 1; }
cordoba_lock optimizacion_lectura
printf '%s\n' "$$" > "$EXP/optimizacion_lectura.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$EXP/optimizacion_lectura.exitcode"' EXIT
"$PY" -u -m src.mantenimiento.optimizar_lectura_multicampana --dir-exp "$EXP" --pid-train "$PID" "$@"
