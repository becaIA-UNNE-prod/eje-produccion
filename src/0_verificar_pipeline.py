"""Entrada compatible; implementación en src.historico.verificar_pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.historico.verificar_pipeline import *


if __name__ == "__main__":
    main()
