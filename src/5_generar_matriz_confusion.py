"""Entrada compatible; implementación en src.evaluacion.matriz_confusion_parches."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluacion.matriz_confusion_parches import (
    crear_parser,
    main,
)


if __name__ == "__main__":
    main()
