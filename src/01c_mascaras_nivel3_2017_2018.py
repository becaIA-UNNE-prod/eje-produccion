"""Entrada compatible; implementación en src.datos.mascaras_nivel3_2017_2018."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datos.mascaras_nivel3_2017_2018 import *

if __name__ == "__main__":
    main()
