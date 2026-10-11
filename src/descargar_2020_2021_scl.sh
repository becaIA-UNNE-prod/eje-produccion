#!/usr/bin/env bash
# Entrada compatible; flujo en scripts/preparacion/descargar_2020_2021_scl.sh.
set -euo pipefail
exec bash "$(dirname "$0")/../scripts/preparacion/descargar_2020_2021_scl.sh" "$@"
