#!/usr/bin/env bash
# Entrada compatible; flujo en scripts/preparacion/preparar_extension_este.sh.
set -euo pipefail
exec bash "$(dirname "$0")/../scripts/preparacion/preparar_extension_este.sh" "$@"
