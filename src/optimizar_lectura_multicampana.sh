#!/usr/bin/env bash
# Entrada compatible para preparar cachés y reanudar entrenamiento.
set -euo pipefail
exec bash "$(dirname "$0")/../scripts/mantenimiento/optimizar_lectura_multicampana.sh" "$@"
