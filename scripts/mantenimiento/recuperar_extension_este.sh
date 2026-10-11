#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?Directorio de resultados requerido}
BASE=/mnt/yacy_1/prod/ferreyra/extension_este_multicampana
exec 8>"$BASE/preparacion.lock"
flock -n 8 || { echo 'Hay otra preparacion activa'; exit 1; }
echo "$$" > "$OUT/preparacion.pid"
trap 'rc=$?; echo "$rc" > "$OUT/preparacion.exitcode"; date -Is > "$OUT/fin.txt"' EXIT
echo "$(date -Is) RECUPERACION: codigo 0 MNC2020 excluido como en remap anterior; mejora de dos compuestos"
"$HOME/s4a-venv/bin/python" -u -m src.mantenimiento.mejorar_calidad_extension "$OUT"
flock -u 8
exec 8>&-
trap - EXIT
exec bash scripts/preparacion/preparar_extension_este.sh "$OUT"
