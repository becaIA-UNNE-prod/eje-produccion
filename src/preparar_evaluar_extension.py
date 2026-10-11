#!/usr/bin/env python3
"""Mascaras y evaluacion de nuevo TEST, conservando resultados previos."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datos.preparar_extension import *
from src.evaluacion.evaluar_extension import *


def crear_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--salida', type=Path, required=True)
    parser.add_argument('--modo', choices=['preparar', 'evaluar'], required=True)
    return parser


def main(argv=None):
    args = crear_parser().parse_args(argv)
    args.salida.mkdir(parents=True, exist_ok=True)
    {'preparar': prepare, 'evaluar': evaluate}[args.modo](args.salida)


if __name__ == '__main__':
    main()
