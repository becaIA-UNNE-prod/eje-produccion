"""Entrada compatible; implementación en src.datos.moldear_mascaras."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Validar antes de cargar dependencias o iniciar operaciones.
if __name__ == "__main__":
    from src.soporte.cli_fija import validar_entrada_fija
    validar_entrada_fija('moldear_mascaras')

from src.datos.moldear_mascaras import *

if __name__ == "__main__":
    main()
