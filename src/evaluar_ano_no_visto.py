#!/usr/bin/env python3
"""Entrada compatible; implementación en src.historico.evaluar_ano_no_visto."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.historico.evaluar_ano_no_visto import *

if __name__ == '__main__':
    main()
