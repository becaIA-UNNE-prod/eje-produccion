"""Entrada compatible para cachés y reanudación de entrenamiento."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mantenimiento.optimizar_lectura_multicampana import (
    ROOT, guardar, checksum, copiar_verificado, preparar, proceso_actual,
    detener, crear_parser, main, ejecutar,
)


if __name__ == '__main__':
    ejecutar()
