"""Entrada compatible; implementación en src.entrenamiento.calcular_pesos."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.entrenamiento.calcular_pesos import (
    DIR,
    TILES_TRAIN,
    NOMBRES,
    NUM_CLASSES,
    SPLIT,
    crear_parser,
    seleccionar_rutas,
    contar_rutas,
    imprimir_pesos,
    main,
)


if __name__ == "__main__":
    main()
