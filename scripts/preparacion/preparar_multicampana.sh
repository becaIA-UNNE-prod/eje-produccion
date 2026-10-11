#!/usr/bin/env bash
# Solo datos; nunca entrenamiento ni inferencia.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/s4a-venv/bin/python}
LOGS=/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana
mkdir -p "$LOGS"
exec 9>"$LOGS/preparacion.lock"
flock -n 9 || { echo "Ya hay una preparacion activa"; exit 1; }
printf '%s\n' "$$" > "$LOGS/preparacion.pid"
"$PY" -c 'from pathlib import Path; Path("/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/preparacion.exitcode").unlink(missing_ok=True)'
trap 'rc=$?; printf "%s\n" "$rc" > "$LOGS/preparacion.exitcode"' EXIT
"$PY" -u -m src.datos.preparar_multicampana "$@"
