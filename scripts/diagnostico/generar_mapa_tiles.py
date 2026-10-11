#!/usr/bin/env python3
"""Regenera el mapa desde el split y el catálogo de contornos acreditados."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.mapas_tiles import generar_mapa, actualizar_indice
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, default=ROOT/'config')
    parser.add_argument('--salida', type=Path)
    parser.add_argument('--titulo', default='Tiles del split vigente')
    args = parser.parse_args()
    destination = args.salida or ROOT/'artifacts/maps/vigente'
    print(generar_mapa(args.config_dir, destination, args.titulo))
    if destination.name == 'vigente':
        actualizar_indice(destination.parent)
