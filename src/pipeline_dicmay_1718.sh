#!/usr/bin/env bash
# Entrada compatible; implementación en scripts/historico/pipeline_dicmay_1718.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/historico/pipeline_dicmay_1718.sh" "$@"
