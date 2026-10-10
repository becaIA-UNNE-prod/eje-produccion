#!/usr/bin/env bash
# Entrada compatible para recomposición y continuación de preparación/evaluación.
set -euo pipefail
exec bash "$(dirname "$0")/../scripts/mantenimiento/recuperar_extension_este.sh" "$@"
