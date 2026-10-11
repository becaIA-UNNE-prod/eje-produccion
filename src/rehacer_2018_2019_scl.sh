#!/usr/bin/env bash
# Entrada compatible; flujo en scripts/preparacion/rehacer_2018_2019_scl.sh.
set -euo pipefail
exec bash "$(dirname "$0")/../scripts/preparacion/rehacer_2018_2019_scl.sh" "$@"
