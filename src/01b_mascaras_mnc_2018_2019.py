"""Entrada compatible; implementación en src.datos.mascaras_mnc_2018_2019."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datos.mascaras_mnc_2018_2019 import *

if __name__ == "__main__":
    main()
