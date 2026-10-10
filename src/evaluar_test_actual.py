"""Entrada compatible del evaluador TEST vigente."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluacion.test_actual import (
    ROOT, Y3, CONFIG, files, guardar, sha256, inferir_tile, matriz_fig, curvas,
    figura_extension, comparar_modelos, metricas_confusion, confusion_remapeada,
    cargar_config, preflight, auditar_rasters, firma,
    SimpleUNet, FCRTransUNet, TransUNet, UTAE,
    constructor, comparison, crear_parser, main,
)


if __name__ == '__main__':
    main()
