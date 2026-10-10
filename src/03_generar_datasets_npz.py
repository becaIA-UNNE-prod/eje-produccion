"""Entrada compatible; implementación en src.datos.generar_arrays_originales."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datos.generar_arrays_originales import *
from src.datos.generar_arrays_originales import _procesar_chunk_ventanas

if __name__ == "__main__":
    main()
