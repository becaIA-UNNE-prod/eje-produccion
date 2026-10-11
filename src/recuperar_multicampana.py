"""Entrada compatible para reconstruir imágenes y reanudar preparación."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Validar antes de cargar dependencias o iniciar operaciones.
if __name__ == "__main__":
    from src.soporte.cli_fija import validar_entrada_fija
    validar_entrada_fija('recuperar_multicampana')

from src.mantenimiento.recuperar_multicampana import (
    ROOT, LOGS, cargar_dependencias, reparar, main, ejecutar,
)


if __name__ == '__main__':
    ejecutar()
