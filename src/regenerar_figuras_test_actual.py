"""Entrada compatible para regenerar figuras del TEST vigente."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.figuras.regenerar_test_actual import (
    ROOT, files, figura_extension, crear_parser, main,
)


if __name__ == '__main__':
    main()
