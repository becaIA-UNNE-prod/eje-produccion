#!/usr/bin/env python3
"""Registrar ubicaciones sin abrir ni copiar artefactos pesados."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.corridas import registrar_rutas, guardar_json, ruta_externa


def pares(values):
    result = []
    for value in values:
        role, separator, path = value.partition('=')
        if not separator or not role or not path:
            raise ValueError('Usar rol=ruta')
        result.append((role, path))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--entrada', action='append', default=[])
    parser.add_argument('--salida', action='append', default=[])
    parser.add_argument('--vinculo', type=Path)
    args = parser.parse_args()
    registrar_rutas(args.run, pares(args.entrada), pares(args.salida))
    if args.vinculo:
        guardar_json(ruta_externa(args.vinculo), {'run_dir': str(args.run.resolve())})
