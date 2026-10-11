#!/usr/bin/env bash
# Solo datos; nunca entrenamiento ni inferencia.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/s4a-venv/bin/python}
source scripts/lib/corridas.sh
cordoba_run_init preparacion preparar_multicampana "$@"
LOGS="$CORDOBA_LOG_DIR"
mkdir -p "$LOGS"
mkdir -p /mnt/yacy_3/prod_bis/ferreyra/logs_multicampana
exec 9>/mnt/yacy_3/prod_bis/ferreyra/logs_multicampana/preparacion.lock
flock -n 9 || { echo "Ya hay una preparacion activa"; exit 1; }
cordoba_lock preparacion_multicampana
printf '%s\n' "$$" > "$CORDOBA_RUN_DIR/preparacion.pid"
rm -f -- "$CORDOBA_RUN_DIR/preparacion.exitcode"
trap 'rc=$?; printf "%s\n" "$rc" > "$CORDOBA_RUN_DIR/preparacion.exitcode"' EXIT
cordoba_registrar --salida "datasets=/mnt/yacy_3/prod_bis/ferreyra/dataset" --salida "mascaras=/mnt/yacy_1/prod/ferreyra/cordoba_ia_unne/data/mascaras"
"$PY" -u -m src.datos.preparar_multicampana "$@"
