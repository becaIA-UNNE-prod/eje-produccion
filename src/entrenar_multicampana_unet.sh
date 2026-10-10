#!/usr/bin/env bash
# Entrada compatible para una corrida U-Net multicampaña.
set -euo pipefail
exec bash "$(dirname "$0")/../scripts/entrenamiento/entrenar_multicampana_unet.sh" "$@"
