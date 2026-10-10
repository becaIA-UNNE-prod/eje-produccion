"""Entrada compatible; implementación en src.datos.convertir_datasets_a_mmap."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datos.convertir_datasets_a_mmap import (
    crear_parser,
    convertir_npz,
    main,
)


if __name__ == "__main__":
    main()
