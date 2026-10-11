"""Entrada compatible; implementación en src.historico.comparar_modelos."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.historico.comparar_modelos import *


if __name__ == "__main__":
    main()
