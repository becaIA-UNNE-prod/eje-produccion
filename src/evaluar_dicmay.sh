#!/usr/bin/env bash
# Entrada compatible; implementación en scripts/historico/evaluar_dicmay.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/historico/evaluar_dicmay.sh" "$@"
