"""Entrada compatible; implementación en src.historico.comparar_con_1718."""
from pathlib import Path
import sys
from importlib import import_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_implementacion = import_module("src.historico.comparar_con_1718")
__all__ = [name for name in dir(_implementacion) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementacion, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))


if __name__ == "__main__":
    _implementacion.main()
