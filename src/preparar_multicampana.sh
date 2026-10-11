#!/usr/bin/env bash
# Entrada compatible para preparación multicampaña.
set -euo pipefail
exec bash "$(dirname "$0")/../scripts/preparacion/preparar_multicampana.sh" "$@"
