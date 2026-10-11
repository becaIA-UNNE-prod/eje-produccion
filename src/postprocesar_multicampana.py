#!/usr/bin/env python3
"""Compatibilidad: CLI vigente por defecto; main importado conserva flujo anterior."""
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
__all__ = ['argparse', 'ExitStack', 'csv', 'datetime', 'hashlib', 'json', 'Path', 'sys', 'subprocess', 'time', 'np', 'rasterio', 'Resampling', 'torch', 'matplotlib', 'plt', 'Patch', 'ROOT', 'SimpleUNet', 'FCRTransUNet', 'SPLIT', 'guardar', 'sha256', 'MULTICAMPANA_ROOT', 'normalizar', 'inferir_tile', 'metricas_confusion', 'evaluar', 'REMAP_1819', 'REMAP_2020', 'REMAP_2021', 'LEGENDS', 'NAMES', 'COLORS', 'matriz_fig', 'curvas', 'YEARS', 'figura_tile', 'resumen', 'main']


def __getattr__(name):
    if name in __all__ or name == '_figura_tile':
        return getattr(import_module('src.historico.postprocesar_multicampana'), name)
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | set(__all__))


if __name__ == '__main__':
    from src.evaluacion.postprocesar_multicampana import main as despacho
    despacho()
