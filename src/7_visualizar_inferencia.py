"""Entrada compatible; implementación en src.figuras.visualizar_inferencia."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.figuras.visualizar_inferencia import (
    COLORES,
    NOMBRES,
    LEGACY_FIGURES,
    crear_parser,
    main,
)


if __name__ == "__main__":
    main()
