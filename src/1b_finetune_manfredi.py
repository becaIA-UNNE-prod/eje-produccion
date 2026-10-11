"""Entrada compatible; implementación en src.entrenamiento.finetune_manfredi."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.entrenamiento.finetune_manfredi import *


if __name__ == "__main__":
    main()
