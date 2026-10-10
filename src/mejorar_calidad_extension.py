"""Entrada compatible para recomponer dos imágenes mensuales Sentinel."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mantenimiento.mejorar_calidad_extension import (
    ROOT, BASE, COMPUESTOS, crear_parser, ejecutar_comando,
    mejorar_compuesto, mejorar_calidad, main,
)


if __name__ == '__main__':
    main()
