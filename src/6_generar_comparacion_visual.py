"""Entrada compatible; implementación en src.figuras.comparacion_parches."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.figuras.comparacion_parches import (
    COLORES,
    clase_a_rgb,
    crear_parser,
    main,
)


if __name__ == "__main__":
    main()
