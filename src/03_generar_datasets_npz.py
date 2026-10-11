"""Entrada compatible; implementación en src.datos.generar_arrays_originales."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Validar antes de cargar dependencias o iniciar operaciones.
if __name__ == "__main__":
    from src.soporte.cli_fija import validar_entrada_fija
    validar_entrada_fija('generar_arrays_originales')

from src.datos.generar_arrays_originales import *
from src.datos.generar_arrays_originales import _procesar_chunk_ventanas

if __name__ == "__main__":
    main()
