#!/usr/bin/env bash
# Entrada compatible; implementación en scripts/historico/preparar_ano_no_visto.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/historico/preparar_ano_no_visto.sh" "$@"
