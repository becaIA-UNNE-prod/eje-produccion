"""Entrada compatible; implementación en src.datos.composites_2019_2020."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datos.composites_2019_2020 import *

if __name__ == "__main__":
    main()
