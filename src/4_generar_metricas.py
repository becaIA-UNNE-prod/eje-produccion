"""Entrada compatible; implementación en src.figuras.curvas_entrenamiento."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.figuras.curvas_entrenamiento import (
    crear_parser,
    main,
)


if __name__ == "__main__":
    main()
