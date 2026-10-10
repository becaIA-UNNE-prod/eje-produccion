"""Entrada compatible para reconstruir imágenes y reanudar preparación."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mantenimiento.recuperar_multicampana import (
    ROOT, LOGS, cargar_dependencias, reparar, main, ejecutar,
)


if __name__ == '__main__':
    ejecutar()
