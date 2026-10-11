#!/usr/bin/env bash
# Entrada compatible: delega preservando argumentos y código de salida.
exec bash "$(dirname "$0")/../scripts/evaluacion/inferencia_multicampana.sh" "$@"
