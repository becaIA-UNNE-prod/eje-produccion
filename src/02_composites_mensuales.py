"""Entrada compatible; implementación en src.datos.composites_mensuales."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Validar antes de cargar dependencias o iniciar operaciones.
if __name__ == "__main__":
    from src.soporte.cli_fija import validar_entrada_fija
    validar_entrada_fija('composites_mensuales')

from src.datos.composites_mensuales import *

if __name__ == "__main__":
    main()
